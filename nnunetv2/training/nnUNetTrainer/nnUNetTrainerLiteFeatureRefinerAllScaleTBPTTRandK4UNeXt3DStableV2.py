import torch
from torch import nn

from nnunetv2.nets.LiteRBUNeXt3D import (
    LiteRBUNeXt3DFeatureRefiner,
    make_lite_rbunext_channels,
    model_summary_line,
)
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteStableV2 import nnUNetTrainerLiteStableV2
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerLiteFeatureRefinerAllScaleTBPTTRandK4UNeXt3DStableV2(nnUNetTrainerLiteStableV2):
    """
    idea1 trainer:
    - K ~ Uniform{1,2,3,4}
    - full forward recurrent trajectory
    - truncated BPTT that keeps only the last 2 refinement steps for gradients
    - step weights lambda_t = t + 1, normalized by sum_{t=0}^K (t + 1)
    """

    refiner_hidden_channels = 32
    refiner_recurrent_steps = 4
    refiner_alpha = 0.5
    refiner_training_min_steps = 1
    tbptt_keep_steps = 2

    def initialize(self):
        self.enable_deep_supervision = False
        super().initialize()

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 3:
            raise ValueError(
                "nnUNetTrainerLiteFeatureRefinerAllScaleTBPTTRandK4UNeXt3DStableV2 only supports 3D configurations"
            )

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
            hidden_channels=32,
            recurrent_steps=4,
            alpha=0.5,
            randomize_training_steps=True,
            training_min_steps=1,
            tbptt_keep_steps=2,
            deep_supervision=False,
        )
        print("LiteFeatureRefinerAllScaleTBPTTRandK4UNeXt3DStableV2: {}".format(model_summary_line(model)))
        return model

    @staticmethod
    def _unwrap_network(network: nn.Module) -> nn.Module:
        return network.module if hasattr(network, "module") else network

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            target0 = target[0]
        else:
            target = target.to(self.device, non_blocking=True)
            target0 = target

        self.optimizer.zero_grad(set_to_none=True)
        _ = self.network(data, tbptt_keep_steps=self.tbptt_keep_steps)
        network = self._unwrap_network(self.network)
        logits_by_step = network.last_all_step_logits
        step_indices = network.last_all_step_indices
        active_steps = set(int(i) for i in network.last_active_step_indices)
        metrics = network.last_step_metrics

        k_steps = int(metrics["K"])
        z = float(sum(range(1, k_steps + 2)))
        loss = torch.zeros((), device=data.device)
        step_losses = []
        for step_idx, logits in zip(step_indices, logits_by_step):
            step_loss = self.loss(logits, target0)
            step_losses.append(step_loss.detach())
            if step_idx in active_steps:
                loss = loss + (float(step_idx + 1) / z) * step_loss
            else:
                loss = loss + (float(step_idx + 1) / z) * step_loss.detach()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        r_d_values = metrics["r_d"]
        r_m_values = metrics["r_m"]
        return {
            "loss": loss.detach().cpu().numpy(),
            "step0_loss": step_losses[0].cpu().numpy(),
            "stepK_loss": step_losses[-1].cpu().numpy(),
            "mean_step_loss": torch.stack(step_losses).mean().cpu().numpy(),
            "K": float(k_steps),
            "T_cut": float(metrics["T_cut"]),
            "r_d": float(sum(r_d_values) / len(r_d_values)) if r_d_values else 0.0,
            "r_m": float(sum(r_m_values) / len(r_m_values)) if r_m_values else 0.0,
        }

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
            target0 = target[0]
        else:
            target = target.to(self.device, non_blocking=True)
            target0 = target

        output = self.network(data)
        loss = self.loss(output, target0)

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target0 != self.label_manager.ignore_label).float()
                target0[target0 == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target0[:, -1:]
                target0 = target0[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target0, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {"loss": loss.detach().cpu().numpy(), "tp_hard": tp_hard, "fp_hard": fp_hard, "fn_hard": fn_hard}
