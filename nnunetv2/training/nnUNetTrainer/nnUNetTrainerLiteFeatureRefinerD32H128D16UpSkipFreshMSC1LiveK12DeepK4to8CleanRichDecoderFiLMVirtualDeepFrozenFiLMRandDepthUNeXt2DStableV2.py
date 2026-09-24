from __future__ import annotations

import torch
from torch import nn

from nnunetv2.nets.LiteRBUNeXt3D import (
    LiteRBUNeXt3DFeatureRefiner,
    make_lite_rbunext_channels,
    model_summary_line,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt3DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt3DStableV2,
)


def _with_unit_depth(values):
    return [(1, *value) if isinstance(value, (list, tuple)) else (1, value, value) for value in values]


class LiteRBUNeXt3DFeatureRefiner2DAdapter(LiteRBUNeXt3DFeatureRefiner):
    """Run the established refiner on 2D tensors through a singleton spatial depth axis."""

    def forward(self, x, recurrent_steps=None, tbptt_keep_steps=None, decode_all_steps=None):
        if x.ndim != 4:
            raise ValueError(f"2D adapter expected BCHW input, got shape {tuple(x.shape)}")
        logits = super().forward(
            x.unsqueeze(2),
            recurrent_steps=recurrent_steps,
            tbptt_keep_steps=tbptt_keep_steps,
            decode_all_steps=decode_all_steps,
        )
        if isinstance(logits, list):
            return [value.squeeze(2) for value in logits]
        return logits.squeeze(2)


class nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt2DStableV2(
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt3DStableV2
):
    """Native nnU-Net 2D I/O with the freshms recurrent dynamics on a unit-depth axis."""

    # Decoder-state indices are class attributes so dataset-specific controlled
    # scale ablations can move the refiner without copying the entire builder.
    # Defaults preserve every historical ISIC/DRIVE experiment bit-for-bit.
    single_refine_index = 2
    single_coarse_context_index = 1

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager,
        dataset_json,
        configuration_manager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 2:
            raise ValueError("FreshMS ISIC adapter only supports 2D configurations")
        label_manager = plans_manager.get_label_manager(dataset_json)
        channels = make_lite_rbunext_channels(
            len(configuration_manager.conv_kernel_sizes),
            base_features=configuration_manager.UNet_base_num_features,
            max_features=320,
        )
        model = LiteRBUNeXt3DFeatureRefiner2DAdapter(
            input_channels=num_input_channels,
            num_classes=label_manager.num_segmentation_heads,
            channels=channels,
            kernel_sizes=_with_unit_depth(configuration_manager.conv_kernel_sizes),
            strides=_with_unit_depth(configuration_manager.pool_op_kernel_sizes),
            n_blocks_per_stage=configuration_manager.n_conv_per_stage_encoder,
            hidden_channels=128,
            recurrent_steps=8,
            alpha=0.5,
            randomize_training_steps=False,
            training_min_steps=1,
            tbptt_keep_steps=2,
            refine_scales="single_to_fine",
            single_refine_index=cls.single_refine_index,
            single_coarse_context_index=cls.single_coarse_context_index,
            refiner_style="sdrr_film",
            normalize_finest_before_head=False,
            checkpoint_refiner=cls.checkpoint_refiner,
            deep_supervision=False,
            clean_refiner_inputs=cls.clean_refiner_inputs,
            rich_proposal_stages=True,
            n_blocks_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
            store_all_step_logits=cls.store_all_step_logits,
        )
        print(
            "FreshMS-RichDecoder-FiLM-VirtualDeep-FrozenFiLM-RandDepth-2D: {}; "
            "unit_depth_adapter=True; clean_refiner_inputs={}; checkpoint_refiner={}; "
            "store_all_step_logits={}; refine_index={}; coarse_context_index={}".format(
                model_summary_line(model),
                cls.clean_refiner_inputs,
                cls.checkpoint_refiner,
                cls.store_all_step_logits,
                cls.single_refine_index,
                cls.single_coarse_context_index,
            )
        )
        return model

    def _proposal(self, data: torch.Tensor):
        if data.ndim != 4:
            raise ValueError(f"2D adapter expected BCHW input, got shape {tuple(data.shape)}")
        data_3d = data.unsqueeze(2)
        network = self._unwrap_network(self.network)
        skips = network._encode(data_3d)
        proposal_states = network._build_proposal_states(
            skips, stop_idx=self._proposal_stop_index(network)
        )
        return proposal_states, skips, data_3d.shape[2:]

    @staticmethod
    def _decode_from_refined_state(network, state, proposal_states, skips, input_shape):
        states = list(proposal_states)
        states[network.single_refine_index] = state
        states = network._replay_proposal_from_state(states, skips, network.single_refine_index)
        return network._decode_logits(states[-1], input_shape).squeeze(2)
