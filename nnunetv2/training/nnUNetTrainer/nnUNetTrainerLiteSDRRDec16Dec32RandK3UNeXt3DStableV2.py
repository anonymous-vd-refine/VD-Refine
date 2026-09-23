import torch
from torch import nn
from torch.nn import functional as F

from nnunetv2.nets.LiteRBUNeXt3D import (
    LiteRBUNeXt3D,
    LiteRBUNeXt3DConfig,
    SharedDecoderRecurrentRefinement3D,
    make_lite_rbunext_channels,
    model_summary_line,
)
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteStableV2 import nnUNetTrainerLiteStableV2
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerLiteSDRRDec16Dec32RandK3UNeXt3DStableV2(nnUNetTrainerLiteStableV2):
    """Shared Decoder Recurrent Refinement at decoder 16^3 and 32^3 features."""

    sdrr_aux16_weight = 0.25
    sdrr_aux32_weight = 0.25

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 3:
            raise ValueError("nnUNetTrainerLiteSDRRDec16Dec32RandK3UNeXt3DStableV2 only supports 3D configurations")

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
                sdrr_recurrent_steps=3,
                sdrr_alpha=0.5,
                sdrr_randomize_training_steps=True,
                deep_supervision=enable_deep_supervision,
            ),
        )
        print("LiteSDRRDec16Dec32RandK3UNeXt3DStableV2: {}".format(model_summary_line(model)))
        return model

    def initialize(self):
        super().initialize()
        self.sdrr_aux_loss = self._build_sdrr_aux_loss()

    def _build_sdrr_aux_loss(self):
        if self.label_manager.has_regions:
            return DC_and_BCE_loss(
                {},
                {
                    "batch_dice": self.configuration_manager.batch_dice,
                    "do_bg": True,
                    "smooth": 1e-5,
                    "ddp": self.is_ddp,
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        return DC_and_CE_loss(
            {
                "batch_dice": self.configuration_manager.batch_dice,
                "smooth": 1e-5,
                "do_bg": False,
                "ddp": self.is_ddp,
            },
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

    @staticmethod
    def _unwrap_network(network: nn.Module) -> nn.Module:
        return network.module if hasattr(network, "module") else network

    def _get_sdrr_aux_logits_by_scale(self) -> dict[tuple[int, int, int], list[torch.Tensor]]:
        network = self._unwrap_network(self.network)
        by_scale: dict[tuple[int, int, int], list[torch.Tensor]] = {}
        for module in network.modules():
            if isinstance(module, SharedDecoderRecurrentRefinement3D) and module.last_aux_logits:
                scale = tuple(int(i) for i in module.last_aux_logits[-1].shape[2:])
                by_scale.setdefault(scale, []).extend(module.last_aux_logits)
        return by_scale

    def _downsample_aux_target(self, target, spatial_shape: tuple[int, int, int]):
        target0 = target[0] if isinstance(target, list) else target
        if target0.shape[2:] == spatial_shape:
            return target0
        return F.interpolate(target0.float(), size=spatial_shape, mode="nearest").to(dtype=target0.dtype)

    def _sdrr_aux_loss(self, target) -> tuple[torch.Tensor, torch.Tensor]:
        by_scale = self._get_sdrr_aux_logits_by_scale()
        zero = torch.zeros((), device=self.device)
        loss16 = zero
        loss32 = zero
        for scale, logits_list in by_scale.items():
            aux_target = self._downsample_aux_target(target, scale)
            loss = torch.stack([self.sdrr_aux_loss(logits, aux_target) for logits in logits_list]).mean()
            if scale == (16, 16, 16):
                loss16 = loss
            elif scale == (32, 32, 32):
                loss32 = loss
        return loss16, loss32

    def _sdrr_fixed_point_loss(self) -> torch.Tensor:
        network = self._unwrap_network(self.network)
        losses = []
        for module in network.modules():
            if isinstance(module, SharedDecoderRecurrentRefinement3D) and module.last_fixed_point_loss is not None:
                losses.append(module.last_fixed_point_loss)
        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        output = self.network(data)
        main_loss = self.loss(output, target)
        aux16_loss, aux32_loss = self._sdrr_aux_loss(target)
        loss = main_loss + self.sdrr_aux16_weight * aux16_loss + self.sdrr_aux32_weight * aux32_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        return {
            "loss": loss.detach().cpu().numpy(),
            "main_loss": main_loss.detach().cpu().numpy(),
            "sdrr_aux16_loss": aux16_loss.detach().cpu().numpy(),
            "sdrr_aux32_loss": aux32_loss.detach().cpu().numpy(),
        }
