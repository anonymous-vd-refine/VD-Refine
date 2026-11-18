from torch import nn

from nnunetv2.nets.LiteRBUNeXt3D import (
    LiteRBUNeXt3DFeatureRefiner,
    make_lite_rbunext_channels,
    model_summary_line,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanUNeXt3DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanUNeXt3DStableV2,
)


class nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderUNeXt3DStableV2(
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanUNeXt3DStableV2
):
    """Same as the parent, but rich_proposal_stages=True: every decoder proposal-stage
    position runs LiteMBConv3D refine blocks (capped at max_decoder_blocks=1), matching the
    no-refiner LiteRBUNeXt3D baseline's decoder capacity exactly, and fixing d_0 to be the
    MBConv-processed proposal state instead of the bare fused one. Also turns off the parent's
    normalize_finest_before_head GroupNorm (True -> False) to match the no-refiner baseline,
    which has no norm before its mask head. Architecturally different from the parent ->
    trained from scratch, own RESULT_DIR, no warm-start."""

    @classmethod
    def build_network_architecture(
        cls,
        plans_manager,
        dataset_json,
        configuration_manager,
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
            normalize_finest_before_head=False,
            checkpoint_refiner=True,
            deep_supervision=False,
            clean_refiner_inputs=cls.clean_refiner_inputs,
            rich_proposal_stages=True,
            n_blocks_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
        )
        print(
            "LiteFeatureRefinerD32H128D16UpSkipFreshMS-RichDecoder: {}; clean_refiner_inputs={}".format(
                model_summary_line(model), cls.clean_refiner_inputs
            )
        )
        return model
