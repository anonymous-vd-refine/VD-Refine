from __future__ import annotations

import random
from contextlib import nullcontext
from os.path import join

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from nnunetv2.nets.LiteRBUNeXt3D import (
    LiteRBUNeXt3DFeatureRefiner,
    make_lite_rbunext_channels,
    model_summary_line,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipTBPTT2RandK8UNeXt3DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipTBPTT2RandK8UNeXt3DStableV2,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteSDRRAlternatingABCDec16Dec32UNeXt3DStableV2 import (
    AlternatingStageScheduler,
    _STAGE_TOTAL_EPOCHS,
    _stage_spec_scaled,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMultiStartK8TBPTT2ABCUNeXt3DStableV2(
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipTBPTT2RandK8UNeXt3DStableV2
):
    """Fresh Multi-Start D32 refinement with static upsampled D16 and per-step skip32."""

    clean_refiner_inputs = False
    terminal_depths = (2, 4, 8)
    tbptt_steps = 2
    monotonic_beta = 0.0
    stage_scale = 1

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = _STAGE_TOTAL_EPOCHS * self.stage_scale
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5
        self.grad_scaler = None
        self._active_stage = "A"

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 3:
            raise ValueError("D32 H128 Fresh Multi-Start only supports 3D configurations")
        label_manager = plans_manager.get_label_manager(dataset_json)
        channels = make_lite_rbunext_channels(
            len(configuration_manager.conv_kernel_sizes),
            base_features=configuration_manager.UNet_base_num_features,
            max_features=320,
        )
        model = LiteRBUNeXt3DFeatureRefiner(
            input_channels=num_input_channels,
            num_classes=label_manager.num_segmentation_heads,
            channels=channels,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            n_blocks_per_stage=configuration_manager.n_conv_per_stage_encoder,
            hidden_channels=128,
            recurrent_steps=8,
            alpha=0.5,
            randomize_training_steps=False,
            training_min_steps=1,
            tbptt_keep_steps=2,
            refine_scales="single_to_fine",
            single_refine_index=2,
            single_coarse_context_index=1,
            refiner_style="sdrr",
            normalize_finest_before_head=True,
            checkpoint_refiner=True,
            deep_supervision=False,
            clean_refiner_inputs=cls.clean_refiner_inputs,
        )
        print(
            "LiteFeatureRefinerD32H128D16UpSkipFreshMS: {}; clean_refiner_inputs={}".format(
                model_summary_line(model), cls.clean_refiner_inputs
            )
        )
        return model

    @staticmethod
    def _unwrap_network(network: nn.Module) -> nn.Module:
        return network.module if hasattr(network, "module") else network

    def configure_optimizers(self):
        groups: dict[str, list[nn.Parameter]] = {
            "refiner": [],
            "encoder": [],
            "decoder": [],
            "heads": [],
        }
        for name, parameter in self.network.named_parameters():
            if "single_refiner.shared_block." in name:
                group = "refiner"
            elif name.startswith("encoder_stages.") or name.startswith("mixers."):
                group = "encoder"
            elif name.startswith("mask_head.") or name.startswith("finest_head_norm."):
                group = "heads"
            else:
                group = "decoder"
            groups[group].append(parameter)
        if any(not parameters for parameters in groups.values()):
            raise RuntimeError(f"FreshMS optimizer received an empty parameter group: {groups.keys()}")
        optimizer = AdamW(
            [{"params": parameters, "lr": self.initial_lr, "name": name} for name, parameters in groups.items()],
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
        )
        return optimizer, AlternatingStageScheduler(optimizer, self.initial_lr, self.stage_scale)

    def _stage_spec_at(self, epoch: int) -> tuple[str, int | None, int]:
        return _stage_spec_scaled(epoch, self.stage_scale)

    def _set_stage_trainability(self, stage: str) -> None:
        network = self._unwrap_network(self.network)
        if network.single_refiner is None:
            raise RuntimeError("single_refiner is required")
        if stage == "A":
            for parameter in network.parameters():
                parameter.requires_grad_(True)
            return
        if stage.startswith("B"):
            for parameter in network.parameters():
                parameter.requires_grad_(False)
            for parameter in network.single_refiner.shared_block.parameters():
                parameter.requires_grad_(True)
            return
        if stage == "C1":
            for parameter in network.parameters():
                parameter.requires_grad_(True)
            for parameter in network.single_refiner.shared_block.parameters():
                parameter.requires_grad_(False)
            return
        if stage == "C2":
            for parameter in network.parameters():
                parameter.requires_grad_(False)
            for idx in range(network.single_refine_index + 1, len(network.proposal_stages)):
                for parameter in network.proposal_stages[idx].parameters():
                    parameter.requires_grad_(True)
            for parameter in network.finest_head_norm.parameters():
                parameter.requires_grad_(True)
            for parameter in network.mask_head.parameters():
                parameter.requires_grad_(True)
            return
        raise ValueError(f"Unsupported stage: {stage}")

    def on_train_epoch_start(self):
        stage, k_max, _ = self._stage_spec_at(self.current_epoch)
        self._active_stage = stage
        self._set_stage_trainability(stage)
        super().on_train_epoch_start()
        self.print_to_log_file(self._stage_log_message(stage, k_max))

    def _stage_log_message(self, stage, k_max) -> str:
        return (
            f"D32 FreshMS stage={stage}; K_max={k_max}; terminal_depths={self.terminal_depths}; "
            f"monotonic_beta={self.monotonic_beta}; grad_clip={self.grad_clip_max_norm}"
        )

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _per_case_seg_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_manager.has_regions:
            target_float = target.float()
            bce = F.binary_cross_entropy_with_logits(logits, target_float, reduction="none").flatten(1).mean(1)
            probabilities = torch.sigmoid(logits)
            intersection = (probabilities * target_float).flatten(2).sum(2)
            denominator = probabilities.flatten(2).sum(2) + target_float.flatten(2).sum(2)
            dice = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean(1)
            return bce + dice
        target_long = target[:, 0].long() if target.ndim == logits.ndim else target.long()
        ce = F.cross_entropy(logits, target_long, reduction="none").flatten(1).mean(1)
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target_long, num_classes=logits.shape[1]).movedim(-1, 1).to(probabilities.dtype)
        intersection = (probabilities[:, 1:] * one_hot[:, 1:]).flatten(2).sum(2)
        denominator = probabilities[:, 1:].flatten(2).sum(2) + one_hot[:, 1:].flatten(2).sum(2)
        dice = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean(1)
        return ce + dice

    @staticmethod
    def _target0(target):
        return target[0] if isinstance(target, list) else target

    def _proposal_stop_index(self, network) -> int | None:
        """Highest proposal-state index this trainer's B/C helpers actually read, or None for all.

        Returning None (the default) builds every proposal state, which is what every trainer in
        this lineage has always done. Subclasses whose helpers only read up to the refine index
        can return it to skip the two highest-resolution decoder stages, whose outputs are
        discarded -- see LiteRBUNeXt3DFeatureRefiner._build_proposal_states for why that is
        side-effect free and gradient-preserving.

        Must stay None for trainers that read further up, e.g.
        nnUNetTrainerLiteFeatureRefinerIndependentD16D32H64FreshMultiStartK8TBPTT2ABCUNeXt3D
        StableV2, which reads proposal_states[network.d32_refine_index].
        """
        return None

    def _proposal(self, data: torch.Tensor):
        network = self._unwrap_network(self.network)
        skips = network._encode(data)
        proposal_states = network._build_proposal_states(skips, stop_idx=self._proposal_stop_index(network))
        return proposal_states, skips, data.shape[2:]

    @staticmethod
    def _one_refine_step(network, state, proposal_states, skips, step_idx: int):
        idx = network.single_refine_index
        return network.single_refiner(
            state,
            proposal_states[idx],
            skips[-(idx + 2)],
            coarse_context=proposal_states[network.single_coarse_context_index],
            step_idx=step_idx,
        )

    def _behavior_rollout(self, network, proposal_states, skips, max_depth: int):
        idx = network.single_refine_index
        states = [proposal_states[idx]]
        state = states[0]
        with torch.no_grad():
            for step_idx in range(max_depth):
                state = self._one_refine_step(network, state, proposal_states, skips, step_idx)
                states.append(state)
        return states

    @staticmethod
    def _decode_from_refined_state(network, state, proposal_states, skips, input_shape):
        states = list(proposal_states)
        states[network.single_refine_index] = state
        states = network._replay_proposal_from_state(states, skips, network.single_refine_index)
        return network._decode_logits(states[-1], input_shape)

    def _fresh_multistart_backward(self, data: torch.Tensor, target) -> dict[str, torch.Tensor]:
        network = self._unwrap_network(self.network)
        target0 = self._target0(target)
        with torch.no_grad(), self._autocast_context():
            proposal_states, skips, input_shape = self._proposal(data)
            behavior = self._behavior_rollout(network, proposal_states, skips, max(self.terminal_depths))

        branch_losses: dict[str, torch.Tensor] = {}
        monotonic_losses = []
        previous_endpoint_loss = None
        for terminal_depth in self.terminal_depths:
            start_depth = terminal_depth - min(self.tbptt_steps, terminal_depth)
            with self._autocast_context():
                state = behavior[start_depth].detach()
                for step_idx in range(start_depth, terminal_depth):
                    state = self._one_refine_step(network, state, proposal_states, skips, step_idx)
                logits = self._decode_from_refined_state(network, state, proposal_states, skips, input_shape)
                per_case_loss = self._per_case_seg_loss(logits, target0)
                branch_loss = per_case_loss.mean()
                if previous_endpoint_loss is None or self.monotonic_beta <= 0:
                    monotonic_loss = torch.zeros((), device=data.device)
                else:
                    monotonic_loss = torch.relu(per_case_loss - previous_endpoint_loss).mean()
                objective = (branch_loss + self.monotonic_beta * monotonic_loss) / len(self.terminal_depths)
            objective.backward()
            branch_losses[f"loss_k{terminal_depth}"] = branch_loss.detach()
            monotonic_losses.append(monotonic_loss.detach())
            previous_endpoint_loss = per_case_loss.detach()
        branch_losses["monotonic_loss"] = torch.stack(monotonic_losses).mean()
        return branch_losses

    def _c1_backward(self, data: torch.Tensor, target) -> dict[str, torch.Tensor]:
        losses = {}
        target0 = self._target0(target)
        for depth in self.terminal_depths:
            with self._autocast_context():
                logits = self.network(data, recurrent_steps=depth, tbptt_keep_steps=depth)
                branch_loss = self._per_case_seg_loss(logits, target0).mean()
            (branch_loss / len(self.terminal_depths)).backward()
            losses[f"loss_k{depth}"] = branch_loss.detach()
        return losses

    def _c2_backward(self, data: torch.Tensor, target) -> dict[str, torch.Tensor]:
        network = self._unwrap_network(self.network)
        target0 = self._target0(target)
        with torch.no_grad(), self._autocast_context():
            proposal_states, skips, input_shape = self._proposal(data)
            behavior = self._behavior_rollout(network, proposal_states, skips, max(self.terminal_depths))
        losses = {}
        for depth in self.terminal_depths:
            with self._autocast_context():
                logits = self._decode_from_refined_state(
                    network, behavior[depth], proposal_states, skips, input_shape
                )
                branch_loss = self._per_case_seg_loss(logits, target0).mean()
            (branch_loss / len(self.terminal_depths)).backward()
            losses[f"loss_k{depth}"] = branch_loss.detach()
        return losses

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [value.to(self.device, non_blocking=True) for value in target]
        else:
            target = target.to(self.device, non_blocking=True)
        target0 = self._target0(target)

        stage, _, _ = self._stage_spec_at(self.current_epoch)
        self.optimizer.zero_grad(set_to_none=True)
        if stage == "A":
            depth = 1 if random.random() < 0.6 else 2
            with self._autocast_context():
                logits = self.network(data, recurrent_steps=depth, tbptt_keep_steps=depth)
                loss = self._per_case_seg_loss(logits, target0).mean()
            loss.backward()
            branch_losses = {"loss_kA": loss.detach()}
        elif stage.startswith("B"):
            branch_losses = self._fresh_multistart_backward(data, target)
        elif stage == "C1":
            branch_losses = self._c1_backward(data, target)
        elif stage == "C2":
            branch_losses = self._c2_backward(data, target)
        else:
            raise ValueError(f"Unsupported stage: {stage}")

        self._clip_gradients()
        self.optimizer.step()
        reported = [value for name, value in branch_losses.items() if name.startswith("loss_k")]
        result = {"loss": torch.stack(reported).mean().cpu().numpy()}
        result.update({name: value.cpu().numpy() for name, value in branch_losses.items()})
        return result

    def _stage_checkpoint_epochs(self) -> dict[int, str]:
        scale = self.stage_scale
        return {
            50 * scale - 1: "checkpoint_stage_A.pth",
            70 * scale - 1: "checkpoint_stage_B1.pth",
            80 * scale - 1: "checkpoint_stage_C1.pth",
            90 * scale - 1: "checkpoint_stage_B2.pth",
            _STAGE_TOTAL_EPOCHS * scale - 1: "checkpoint_stage_C2.pth",
        }

    def on_epoch_end(self):
        finished_epoch = self.current_epoch
        super().on_epoch_end()
        checkpoints = self._stage_checkpoint_epochs()
        if finished_epoch in checkpoints:
            # super().on_epoch_end() has already advanced current_epoch, while
            # save_checkpoint stores current_epoch + 1 as the resume epoch.
            # Temporarily restore the completed epoch so stage checkpoints obey
            # the same invariant as checkpoint_latest: resume at len(logging).
            self.current_epoch -= 1
            try:
                self.save_checkpoint(join(self.output_folder, checkpoints[finished_epoch]))
            finally:
                self.current_epoch += 1

    def perform_actual_validation(self, save_probabilities: bool = False):
        # run_training.py unconditionally calls this after run_training() finishes. It predicts
        # on the internal 59-case validation split at K=self.recurrent_steps (network.eval() ->
        # _get_recurrent_steps falls through to the constructor default, 8), duplicating the
        # per-epoch pseudo-dice signal, and writes to <output_folder>/validation/ -- nothing in
        # this project's eval pipeline (scripts/run.sh eval, eval_scaling_dice*.py,
        # build_all_experiments_summary.py) reads that directory. No-op to skip the several
        # minutes of extra inference this costs at the end of every `train` action.
        self.print_to_log_file(
            "Skipping perform_actual_validation (unused by this project's eval pipeline; "
            "see EXPERIMENTS.md FreshMultiStart Clean round notes)."
        )
