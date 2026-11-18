from __future__ import annotations

from contextlib import nullcontext
from os.path import join
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW

from nnunetv2.nets.LiteRBUNeXt3D import (
    AlternatingDecoderRecurrentRefinement3D,
    LiteRBUNeXt3D,
    LiteRBUNeXt3DConfig,
    make_lite_rbunext_channels,
    model_summary_line,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteSDRRDec16Dec32RandK3UNeXt3DStableV2 import (
    nnUNetTrainerLiteSDRRDec16Dec32RandK3UNeXt3DStableV2,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


_STAGE_BOUNDARIES: tuple[tuple[int, str, int | None, int], ...] = (
    (50, "A", None, 1),
    (55, "B1-K4", 4, 1),
    (60, "B1-K8", 8, 1),
    (70, "B1-K16", 16, 1),
    (80, "C1", None, 1),
    (85, "B2-K8", 8, 2),
    (90, "B2-K16", 16, 2),
)
_STAGE_TERMINAL: tuple[str, int | None, int] = ("C2", None, 2)
_STAGE_TOTAL_EPOCHS = 100


def _stage_spec_scaled(epoch: int, stage_scale: int = 1) -> tuple[str, int | None, int]:
    if stage_scale < 1:
        raise ValueError("stage_scale must be >= 1")
    for boundary, stage, k_max, cycle in _STAGE_BOUNDARIES:
        if epoch < boundary * stage_scale:
            return stage, k_max, cycle
    return _STAGE_TERMINAL


def _stage_spec(epoch: int) -> tuple[str, int | None, int]:
    return _stage_spec_scaled(epoch, 1)


class AlternatingStageScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, stage_scale: int = 1):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.last_epoch = -1
        self.stage_scale = int(stage_scale)

    def step(self, epoch: int | None = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = int(epoch)
        stage, _, cycle = _stage_spec_scaled(self.last_epoch, self.stage_scale)
        cycle_scale = 1.0 if cycle == 1 else 0.5
        for group in self.optimizer.param_groups:
            name = group["name"]
            if stage == "A":
                scale = 1.0
            elif stage.startswith("B"):
                scale = (1.0 if cycle == 1 else 0.2) if name == "refiner" else 0.0
            else:
                scale = {
                    "refiner": 0.0,
                    "encoder": 0.05,
                    "decoder": 0.10,
                    "heads": 1.0,
                }[name] * cycle_scale
            group["lr"] = self.base_lr * scale

    def state_dict(self):
        return {"base_lr": self.base_lr, "last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict):
        self.base_lr = float(state_dict.get("base_lr", self.base_lr))
        self.last_epoch = int(state_dict.get("last_epoch", -1))
        if self.last_epoch >= 0:
            self.step(self.last_epoch)


class nnUNetTrainerLiteSDRRAlternatingABCDec16Dec32UNeXt3DStableV2(
    nnUNetTrainerLiteSDRRDec16Dec32RandK3UNeXt3DStableV2
):
    """A-(B<->C) block-coordinate training for anchored D32/D16 SDRR."""

    sdrr_aux16_weight = 0.05
    sdrr_aux32_weight = 0.05
    monotonic_beta = 0.25

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
        self.num_epochs = 100
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5
        self.grad_scaler = None
        self._active_stage = "A"

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 3:
            raise ValueError("Alternating SDRR only supports 3D configurations")

        label_manager = plans_manager.get_label_manager(dataset_json)
        channels = make_lite_rbunext_channels(
            len(configuration_manager.conv_kernel_sizes),
            base_features=configuration_manager.UNet_base_num_features,
            max_features=320,
        )
        model = LiteRBUNeXt3D(
            input_channels=num_input_channels,
            num_classes=label_manager.num_segmentation_heads,
            channels=channels,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            n_blocks_per_stage=configuration_manager.n_conv_per_stage_encoder,
            n_blocks_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
            config=LiteRBUNeXt3DConfig(
                recurrent_bottleneck=False,
                recurrent_encoder_mixer=False,
                sdrr_decoder_refinement=True,
                sdrr_decoder_stages=(1, 2),
                sdrr_hidden_channels=64,
                sdrr_recurrent_steps=1,
                sdrr_randomize_training_steps=False,
                sdrr_training_min_steps=1,
                sdrr_learnable_gamma=False,
                sdrr_tbptt_keep_steps=0,
                sdrr_update_mode="alternating_pull",
                sdrr_lambda=0.5,
                deep_supervision=enable_deep_supervision,
            ),
        )
        print("LiteSDRRAlternatingABCDec16Dec32: {}".format(model_summary_line(model)))
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
            if ".shared_block." in name or name.startswith("sdrr_shared_block."):
                group_name = "refiner"
            elif name.startswith("encoder_stages.") or name.startswith("mixers."):
                group_name = "encoder"
            elif name.startswith("seg_layers.") or ".aux_head." in name:
                group_name = "heads"
            else:
                group_name = "decoder"
            groups[group_name].append(parameter)

        if any(len(parameters) == 0 for parameters in groups.values()):
            sizes = {name: len(parameters) for name, parameters in groups.items()}
            raise RuntimeError(f"Alternating optimizer received an empty parameter group: {sizes}")

        optimizer = AdamW(
            [
                {"params": parameters, "lr": self.initial_lr, "name": name}
                for name, parameters in groups.items()
            ],
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            eps=1e-5,
        )
        scheduler = AlternatingStageScheduler(optimizer, self.initial_lr)
        return optimizer, scheduler

    def _alternating_modules(self) -> list[AlternatingDecoderRecurrentRefinement3D]:
        network = self._unwrap_network(self.network)
        return [
            module
            for module in network.modules()
            if isinstance(module, AlternatingDecoderRecurrentRefinement3D)
        ]

    def _set_stage_trainability(self, stage: str):
        network = self._unwrap_network(self.network)
        if stage == "A":
            for parameter in network.parameters():
                parameter.requires_grad_(True)
            return

        if stage.startswith("B"):
            for parameter in network.parameters():
                parameter.requires_grad_(False)
            for parameter in network.sdrr_shared_block.parameters():
                parameter.requires_grad_(True)
            return

        for parameter in network.parameters():
            parameter.requires_grad_(True)
        for parameter in network.sdrr_shared_block.parameters():
            parameter.requires_grad_(False)

    def on_train_epoch_start(self):
        stage, k_max, _ = _stage_spec(self.current_epoch)
        self._active_stage = stage
        self._set_stage_trainability(stage)
        super().on_train_epoch_start()
        self.print_to_log_file(f"Alternating stage: {stage}; K_max={k_max}; beta={self.monotonic_beta}")

    def _per_case_seg_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_manager.has_regions:
            target_float = target.float()
            bce = F.binary_cross_entropy_with_logits(logits, target_float, reduction="none").flatten(1).mean(1)
            probabilities = torch.sigmoid(logits)
            intersection = (probabilities * target_float).flatten(2).sum(2)
            denominator = probabilities.flatten(2).sum(2) + target_float.flatten(2).sum(2)
            dice_loss = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean(1)
            return bce + dice_loss

        target_long = target[:, 0].long() if target.ndim == logits.ndim else target.long()
        ce = F.cross_entropy(logits, target_long, reduction="none").flatten(1).mean(1)
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target_long, num_classes=logits.shape[1]).movedim(-1, 1).to(probabilities.dtype)
        intersection = (probabilities[:, 1:] * one_hot[:, 1:]).flatten(2).sum(2)
        denominator = probabilities[:, 1:].flatten(2).sum(2) + one_hot[:, 1:].flatten(2).sum(2)
        dice_loss = 1.0 - ((2.0 * intersection + 1e-5) / (denominator + 1e-5)).mean(1)
        return ce + dice_loss

    def _aux_all_steps(self, target) -> torch.Tensor:
        losses = []
        for module in self._alternating_modules():
            if not module.last_aux_logits:
                continue
            spatial_shape = tuple(int(size) for size in module.last_aux_logits[-1].shape[2:])
            aux_target = self._downsample_aux_target(target, spatial_shape)
            losses.append(torch.stack([self.sdrr_aux_loss(logits, aux_target) for logits in module.last_aux_logits]).mean())
        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).sum()

    def _aux_tail_and_monotonic(self, target) -> tuple[torch.Tensor, torch.Tensor]:
        aux_losses = []
        monotonic_losses = []
        for module in self._alternating_modules():
            logits_list = module.last_aux_logits
            if not logits_list:
                continue
            spatial_shape = tuple(int(size) for size in logits_list[-1].shape[2:])
            aux_target = self._downsample_aux_target(target, spatial_shape)
            last = self._per_case_seg_loss(logits_list[-1], aux_target)
            previous = self._per_case_seg_loss(logits_list[-2], aux_target) if len(logits_list) >= 2 else last.detach()
            aux_losses.append(0.5 * (previous.mean() + last.mean()))
            monotonic_losses.append(torch.relu(last - previous.detach()).mean())
        if not aux_losses:
            zero = torch.zeros((), device=self.device)
            return zero, zero
        return torch.stack(aux_losses).sum(), torch.stack(monotonic_losses).sum()

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _active_parameters(self):
        return [parameter for parameter in self.network.parameters() if parameter.requires_grad]

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [value.to(self.device, non_blocking=True) for value in target]
        else:
            target = target.to(self.device, non_blocking=True)

        stage, k_max, _ = _stage_spec(self.current_epoch)
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast_context():
            if stage == "A":
                recurrent_steps = 1 if random.random() < 0.6 else 2
                output = self.network(
                    data,
                    recurrent_steps=recurrent_steps,
                    sdrr_tbptt_keep_steps=0,
                )
                main_loss = self.loss(output, target)
                aux_loss = self._aux_all_steps(target)
                monotonic_loss = torch.zeros((), device=self.device)
                deep_loss = torch.zeros((), device=self.device)
                loss = main_loss + 0.05 * aux_loss
            elif stage.startswith("B"):
                recurrent_steps = random.randint(1, int(k_max))
                output = self.network(
                    data,
                    recurrent_steps=recurrent_steps,
                    sdrr_tbptt_keep_steps=2,
                )
                main_loss = self.loss(output, target)
                aux_loss, monotonic_loss = self._aux_tail_and_monotonic(target)
                deep_loss = torch.zeros((), device=self.device)
                loss = main_loss + 0.05 * aux_loss + self.monotonic_beta * monotonic_loss
            else:
                recurrent_steps = 1 if random.random() < 0.8 else 2
                live_output = self.network(
                    data,
                    recurrent_steps=recurrent_steps,
                    sdrr_tbptt_keep_steps=0,
                )
                live_main_loss = self.loss(live_output, target)
                aux_loss = self._aux_all_steps(target)
                live_loss = live_main_loss + 0.05 * aux_loss

                deep_steps = random.choice((4, 8, 16))
                deep_output = self.network(
                    data,
                    recurrent_steps=deep_steps,
                    sdrr_tbptt_keep_steps=0,
                    detach_after_last_sdrr=True,
                )
                deep_loss = self.loss(deep_output, target)
                main_loss = live_main_loss
                monotonic_loss = torch.zeros((), device=self.device)
                loss = 0.625 * live_loss + 0.375 * deep_loss

        loss.backward()
        self._clip_gradients()
        self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "main_loss": main_loss.detach().cpu().numpy(),
            "aux_loss": aux_loss.detach().cpu().numpy(),
            "monotonic_loss": monotonic_loss.detach().cpu().numpy(),
            "deep_loss": deep_loss.detach().cpu().numpy(),
            "recurrent_steps": recurrent_steps,
        }

    def on_epoch_end(self):
        finished_epoch = self.current_epoch
        super().on_epoch_end()
        checkpoints = {
            49: "checkpoint_stage_A.pth",
            69: "checkpoint_stage_B1.pth",
            79: "checkpoint_stage_C1.pth",
            89: "checkpoint_stage_B2.pth",
            99: "checkpoint_stage_C2.pth",
        }
        if finished_epoch in checkpoints:
            self.save_checkpoint(join(self.output_folder, checkpoints[finished_epoch]))
