from torch import nn

from nnunetv2.nets.LiteRBUNeXt3D import (
    LiteRBUNeXt3DFeatureRefiner,
    make_lite_rbunext_channels,
    model_summary_line,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerAllScaleTBPTTRandK4UNeXt3DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerAllScaleTBPTTRandK4UNeXt3DStableV2,
)
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


class nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipTBPTT2RandK8UNeXt3DStableV2(
    nnUNetTrainerLiteFeatureRefinerAllScaleTBPTTRandK4UNeXt3DStableV2
):
    """D32 refiner with H128, per-step upsampled D16 context, and per-step skip32."""

    refiner_hidden_channels = 128
    refiner_recurrent_steps = 8
    refiner_training_min_steps = 1
    tbptt_keep_steps = 2

    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        if len(configuration_manager.patch_size) != 3:
            raise ValueError("D32 H128 D16-upsample+skip refiner only supports 3D configurations")
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
            randomize_training_steps=True,
            training_min_steps=1,
            tbptt_keep_steps=2,
            refine_scales="single_to_fine",
            single_refine_index=2,
            single_coarse_context_index=1,
            refiner_style="sdrr",
            normalize_finest_before_head=True,
            checkpoint_refiner=True,
            deep_supervision=False,
        )
        print(
            "LiteFeatureRefinerD32H128D16UpSkipTBPTT2RandK8: {}".format(
                model_summary_line(model)
            )
        )
        return model
