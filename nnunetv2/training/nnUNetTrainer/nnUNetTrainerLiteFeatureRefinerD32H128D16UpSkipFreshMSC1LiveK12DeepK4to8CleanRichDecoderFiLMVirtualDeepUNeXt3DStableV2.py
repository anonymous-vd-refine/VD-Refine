from __future__ import annotations

import os
import random

import torch
from torch import nn
from torch.optim import AdamW

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMUNeXt3DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMUNeXt3DStableV2,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteSDRRAlternatingABCDec16Dec32UNeXt3DStableV2 import (
    AlternatingStageScheduler,
    _stage_spec,
    _stage_spec_scaled,
)

_WARM_START_STAGE_A_ENV = "VIRTUALDEEP_WARM_START_A"


class FiLMAwareStageScheduler(AlternatingStageScheduler):
    """Same per-stage LR-scale table as AlternatingStageScheduler, except the "film" param
    group (single_refiner.film_head) rides along with "refiner" during B-stages instead of
    being forced to 0 -- the whole point of the Virtual Deep Step branch is to give film_head
    real deep-K gradient, which is useless if the optimizer never gives it a nonzero LR.

    With film_in_b=False the "film" group is forced to 0 during B instead, reproducing the plain
    FiLM trainer's freeze semantics (where film_head sits in the `decoder` group, scaled to 0.0
    in B) while keeping the virtual deep branch. In A and C the group's scales -- 1.0 and 0.10 --
    already equal what `decoder` would have given it, so only B differs either way."""

    def __init__(self, optimizer, base_lr: float, film_in_b: bool = True, stage_scale: int = 1):
        super().__init__(optimizer, base_lr, stage_scale)
        self.film_in_b = bool(film_in_b)

    def step(self, epoch: int | None = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = int(epoch)
        # Scale-aware, like the AlternatingStageScheduler this overrides: with the default
        # stage_scale=1 this is _stage_spec(epoch) exactly, so every existing trainer is
        # unaffected; a stage_scale>1 subclass gets correspondingly stretched stage boundaries
        # instead of silently falling through to C2 past epoch 100.
        stage, _, cycle = _stage_spec_scaled(self.last_epoch, self.stage_scale)
        cycle_scale = 1.0 if cycle == 1 else 0.5
        for group in self.optimizer.param_groups:
            name = group["name"]
            if stage == "A":
                scale = 1.0
            elif stage.startswith("B"):
                trains_in_b = name == "refiner" or (name == "film" and self.film_in_b)
                scale = (1.0 if cycle == 1 else 0.2) if trains_in_b else 0.0
            else:
                scale = {
                    "refiner": 0.0,
                    "film": 0.10,
                    "encoder": 0.05,
                    "decoder": 0.10,
                    "heads": 1.0,
                }[name] * cycle_scale
            group["lr"] = self.base_lr * scale


class nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepUNeXt3DStableV2(
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMUNeXt3DStableV2
):
    """Same architecture/curriculum as the FiLM RichDecoder parent, but adds a 4th "virtual deep
    step" branch to Stage B (B1/B2) training and unfreezes film_head during B so it finally gets
    deep-K gradient (see EXPERIMENTS.md Finding 6: film_head was frozen bit-for-bit in B1/B2
    under the parent trainer, so the parameter controlling the recurrent normalization never saw
    K16/K32 behavior during training).

    Mechanism: the FiLM refiner's state update is an empirically contractive fixed point
    (Delta_{t+1} ~= q * Delta_t). From the already-computed no-grad rollout d_0..d_8, estimate q
    per-sample from u6=d6-d4, u8=d8-d6, then use the closed-form geometric series to jump
    straight to a synthetic d_tilde at depth (target_depth-2) without running the intermediate
    real steps. Project it back through the refiner's own FiLM normalization (gamma/beta of d_0
    + raw_norm), stopgrad, then run exactly 2 real refiner steps to depth target_depth before
    decoding and computing a terminal segmentation loss. This gives film_head/shared_block real
    TBPTT-2 gradient at K16/K32 for the cost of one dot product + 2 refiner steps + 1 decode,
    instead of the ~22 extra no-grad steps a real rollout to K32 would need.

    Supports scratch training or optional historical Stage-A weight reuse with a fresh
    optimizer. The extra "film" group prevents loading the donor optimizer state unchanged.
    """

    _b1_start_epoch, _b1_end_epoch = 50, 70
    _b2_start_epoch, _b2_end_epoch = 80, 90
    _virtual_b1_weight_range = (0.05, 0.20)
    _virtual_b2_weight_range = (0.20, 0.25)
    # Whether Stage B trains single_refiner.film_head as well as single_refiner.shared_block.
    # True is this trainer's own design (see the class docstring). A subclass setting it False
    # keeps the virtual deep branch but restores the parent's B-stage freeze, so the branch's
    # gradient reaches shared_block alone.
    _train_film_in_b = True

    def _set_stage_trainability(self, stage: str) -> None:
        super()._set_stage_trainability(stage)
        if stage.startswith("B") and self._train_film_in_b:
            network = self._unwrap_network(self.network)
            for parameter in network.single_refiner.film_head.parameters():
                parameter.requires_grad_(True)

    def on_train_start(self):
        super().on_train_start()
        warm_start_path = os.environ.get(_WARM_START_STAGE_A_ENV)
        if not warm_start_path:
            return
        # Stage A's network/training code is bit-identical between this trainer and the plain
        # FiLM RichDecoder sibling (only the B-stage backward/optimizer differ), so its
        # checkpoint_stage_A.pth loads directly -- skips redoing 50 already-completed epochs.
        # This mirrors nnUNetTrainer.load_checkpoint() (network_weights, current_epoch, logger,
        # _best_ema, inference_allowed_mirroring_axes) but deliberately skips optimizer_state and
        # grad_scaler_state: this trainer's optimizer has an extra "film" param group, so
        # Optimizer.load_state_dict would raise "different number of parameter groups" (and a
        # fresh optimizer/scaler is exactly what a Stage-A-only warm start should have anyway).
        # Skipping the logger restore (not just the epoch number) is what actually matters here --
        # an earlier version only set self.current_epoch and left the logger empty, which crashed
        # one epoch later: nnUNetLogger.log() indexes its per-epoch lists by absolute epoch
        # number, so a current_epoch=50 with 0 logged epochs raised IndexError on the first
        # epoch-end EMA-dice update. Invoke this trainer's `train` action with neither CONTINUE=1
        # nor PRETRAINED_WEIGHTS set -- both trigger nnU-Net's own (conflicting) checkpoint logic
        # before this hook runs.
        checkpoint = torch.load(warm_start_path, map_location=self.device)
        network = self.network.module if hasattr(self.network, "module") else self.network
        new_state_dict = {}
        for key, value in checkpoint["network_weights"].items():
            if key not in network.state_dict() and key.startswith("module."):
                key = key[len("module."):]
            new_state_dict[key] = value
        network.load_state_dict(new_state_dict)
        self.logger.load_checkpoint(checkpoint["logging"])
        # nnUNetLogger.log(key, value, epoch) requires len(list) == epoch before logging that
        # epoch (it appends when len < epoch+1, then reads index epoch-1 for the EMA update) --
        # so current_epoch must equal the logger's own recorded length, not the checkpoint's
        # stored `current_epoch` scalar. Empirically checkpoint_stage_A.pth's stored value is 51
        # while its logger only has 50 recorded epochs (0..49); trusting the stored scalar skips
        # epoch 50 and desyncs the logger invariant, crashing one epoch later.
        self.current_epoch = len(self.logger.my_fantastic_logging["train_losses"])
        self._best_ema = checkpoint["_best_ema"]
        if "inference_allowed_mirroring_axes" in checkpoint:
            self.inference_allowed_mirroring_axes = checkpoint["inference_allowed_mirroring_axes"]
        self.print_to_log_file(
            f"{_WARM_START_STAGE_A_ENV} set: warm-started network weights + logger/epoch state "
            f"from {warm_start_path}; resuming at epoch {self.current_epoch} (Stage A skipped, "
            f"fresh optimizer)."
        )

    def configure_optimizers(self):
        groups: dict[str, list[nn.Parameter]] = {
            "refiner": [],
            "film": [],
            "encoder": [],
            "decoder": [],
            "heads": [],
        }
        for name, parameter in self.network.named_parameters():
            if "single_refiner.shared_block." in name:
                group = "refiner"
            elif "single_refiner.film_head." in name:
                group = "film"
            elif name.startswith("encoder_stages.") or name.startswith("mixers."):
                group = "encoder"
            elif name.startswith("mask_head.") or name.startswith("finest_head_norm."):
                group = "heads"
            else:
                group = "decoder"
            groups[group].append(parameter)
        if any(not parameters for parameters in groups.values()):
            raise RuntimeError(f"VirtualDeep optimizer received an empty parameter group: {groups.keys()}")
        optimizer = AdamW(
            [{"params": parameters, "lr": self.initial_lr, "name": name} for name, parameters in groups.items()],
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
        )
        return optimizer, FiLMAwareStageScheduler(
            optimizer, self.initial_lr, film_in_b=self._train_film_in_b, stage_scale=self.stage_scale
        )

    def _b_stage_bounds(self, cycle: int) -> tuple[int, int]:
        """(start, end) epoch of the requested B cycle, stretched by stage_scale.

        The raw class attributes are the stage_scale=1 boundaries (B1 spans 50-70, B2 80-90),
        matching _STAGE_BOUNDARIES. Multiplying by stage_scale keeps the w_v ramp spanning the
        whole B stage at any schedule length; at stage_scale=1 it is the original arithmetic.
        """
        if cycle == 1:
            return self._b1_start_epoch * self.stage_scale, self._b1_end_epoch * self.stage_scale
        return self._b2_start_epoch * self.stage_scale, self._b2_end_epoch * self.stage_scale

    def _virtual_ramp_weight(self, epoch: int, cycle: int) -> float:
        low, high = self._virtual_b1_weight_range if cycle == 1 else self._virtual_b2_weight_range
        start, end = self._b_stage_bounds(cycle)
        frac = (epoch - start) / (end - start)
        return low + (high - low) * min(max(frac, 0.0), 1.0)

    def _virtual_branch_schedule(self, epoch: int) -> tuple[int, float]:
        # self._stage_spec_at is scale-aware (the FreshMS base resolves it through
        # _stage_spec_scaled); at stage_scale=1 it equals _stage_spec(epoch).
        stage, _, cycle = self._stage_spec_at(epoch)
        if not stage.startswith("B"):
            raise ValueError(f"virtual branch schedule requested outside stage B: epoch={epoch}, stage={stage}")
        w_v = self._virtual_ramp_weight(epoch, cycle)
        if cycle == 1:
            return 16, w_v
        target_depth = random.choice((16, 32))
        return target_depth, w_v

    @staticmethod
    def _virtual_jump_n_steps(virtual_start: int) -> int | float:
        """How many 2-step blocks the geometric series jumps past the real depth-8 state.

        q is estimated from u6=d6-d4 and u8=d8-d6, i.e. it is the contraction ratio per TWO
        steps, so the series advances in 2-step blocks and only EVEN virtual_start values land
        on the requested depth. This trainer only ever asks for 14 and 30, both even. A subclass
        sampling odd depths must override this (see the RandDepth subclass, which returns a
        fractional block count -- torch.pow accepts it, and it is bitwise identical to this
        integer version whenever virtual_start is even).
        """
        return (virtual_start - 8) // 2

    def _sample_b_start_depths(self):
        # Yield lazily to preserve historical RNG ordering between branch forwards.
        for low, high in self.b_start_depth_bands:
            yield random.randint(low, high)

    def _fresh_multistart_backward(self, data, target):
        network = self._unwrap_network(self.network)
        target0 = self._target0(target)
        max_depth = max(high for _, high in self.b_start_depth_bands) + self.tbptt_steps
        with torch.no_grad(), self._autocast_context():
            proposal_states, skips, input_shape = self._proposal(data)
            behavior = self._behavior_rollout(network, proposal_states, skips, max_depth)

        target_depth, w_v = self._virtual_branch_schedule(self.current_epoch)

        branch_losses = {}
        monotonic_losses = []
        previous_endpoint_loss = None
        num_branches = len(self.b_start_depth_bands)
        for band_idx, start_depth in enumerate(self._sample_b_start_depths()):
            terminal_depth = start_depth + self.tbptt_steps
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
                objective = (1.0 - w_v) * (branch_loss + self.monotonic_beta * monotonic_loss) / num_branches
            objective.backward()
            branch_losses[f"loss_k_band{band_idx}"] = branch_loss.detach()
            monotonic_losses.append(monotonic_loss.detach())
            previous_endpoint_loss = per_case_loss.detach()
        branch_losses["monotonic_loss"] = torch.stack(monotonic_losses).mean()

        # Virtual deep step: geometric-series jump from the already-computed no-grad d_4/d_6/d_8
        # to a synthetic state near depth target_depth, then 2 real (grad-tracked) refiner steps.
        # film_head/raw_norm need the same autocast context as the rollout that produced d0/d8
        # (bf16 activations, fp32-weight modules) -- torch.no_grad() alone left it in fp32
        # autocast-off mode, mismatching bf16 pooled input against film_head's fp32 bias.
        with torch.no_grad(), self._autocast_context():
            d0, d4, d6, d8 = behavior[0], behavior[4], behavior[6], behavior[8]
            u6 = (d6 - d4).float()
            u8 = (d8 - d6).float()
            reduce_dims = tuple(range(1, u6.dim()))
            q = ((u8 * u6).sum(dim=reduce_dims) / ((u6 * u6).sum(dim=reduce_dims) + 1e-8)).clamp(0.0, 0.95)
            q = q.view((q.shape[0],) + (1,) * (u6.dim() - 1))
            virtual_start = target_depth - 2
            n_steps = self._virtual_jump_n_steps(virtual_start)
            a_scale = (q * (1.0 - q.pow(n_steps)) / (1.0 - q + 1e-8)).to(d8.dtype)
            d_tilde = d8 + a_scale * (d8 - d6)
            gamma, beta = network.single_refiner._film_params(d0)
            z0 = (gamma * network.single_refiner.raw_norm(d_tilde) + beta).detach()

        with self._autocast_context():
            z1 = self._one_refine_step(network, z0, proposal_states, skips, virtual_start)
            z2 = self._one_refine_step(network, z1, proposal_states, skips, virtual_start + 1)
            virtual_logits = self._decode_from_refined_state(network, z2, proposal_states, skips, input_shape)
            virtual_loss = self._per_case_seg_loss(virtual_logits, target0).mean()
        (w_v * virtual_loss).backward()
        branch_losses["loss_k_virtual"] = virtual_loss.detach()

        return branch_losses

    def _virtual_schedule_log_fragment(self, epoch: int) -> str:
        """The virtual-branch part of the per-epoch stage log line.

        Split out because it samples _virtual_branch_schedule(): fine here, where the only random
        element is B2's 16/32 coin flip, but a subclass that randomizes the depth per batch should
        report its POLICY instead, so the log line doesn't show one unrepresentative draw.
        """
        target_depth, w_v = self._virtual_branch_schedule(epoch)
        return f"; virtual_target_depth={target_depth}; virtual_w={w_v:.4f}"

    def _stage_log_message(self, stage, k_max) -> str:
        message = super()._stage_log_message(stage, k_max)
        if self.stage_scale != 1:
            message += f"; stage_scale={self.stage_scale}; num_epochs={self.num_epochs}"
        if stage.startswith("B"):
            message += self._virtual_schedule_log_fragment(self.current_epoch)
            # Log the realised state, not just the flag: on_train_epoch_start() has already run
            # _set_stage_trainability() and stepped the scheduler by this point, so these two
            # numbers are what Stage B will actually train with. film_trainable_params=0 and
            # film_lr=0 is the frozen-FiLM subclass's contract.
            network = self._unwrap_network(self.network)
            film_trainable = sum(
                parameter.numel()
                for parameter in network.single_refiner.film_head.parameters()
                if parameter.requires_grad
            )
            film_lr = next(
                (group["lr"] for group in self.optimizer.param_groups if group["name"] == "film"),
                float("nan"),
            )
            message += (
                f"; train_film_in_b={self._train_film_in_b}"
                f"; film_trainable_params={film_trainable}; film_lr={film_lr:.3e}"
            )
        return message
