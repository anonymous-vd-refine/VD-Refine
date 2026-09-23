import os

import torch
from torch import nn

from nnunetv2.nets.LiteRBUNeXt3D import (
    LiteRBUNeXt3DFeatureRefiner,
    make_lite_rbunext_channels,
    model_summary_line,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderUNeXt3DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderUNeXt3DStableV2,
)

# Optional warm start from a sibling's checkpoint_stage_A.pth (see on_train_start below).
_WARM_START_STAGE_A_ENV = "FILM_WARM_START_A"


class nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMUNeXt3DStableV2(
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderUNeXt3DStableV2
):
    """Same as the RichDecoder parent, but refiner_style="sdrr_film": the recurrent update is
    re-normalized with a FiLM-style learned affine conditioned on d_0 (gamma/beta from a small
    learned head, zero-initialized to identity) instead of borrowing d_0's own per-channel
    mean/std the way AnchorAdaIN does -- more expressive, since gamma/beta aren't forced to
    equal any particular statistic of d_0. Combines with rich_proposal_stages=True (every
    decoder position gets LiteMBConv3D) and normalize_finest_before_head=False. Architecturally
    different from all siblings -> trained from scratch, own RESULT_DIR, no warm-start."""

    # Exposed as class attributes (like clean_refiner_inputs) so subclasses can flip them without
    # duplicating build_network_architecture. Both defaults reproduce this run exactly; neither
    # affects the model's mathematics, only its memory/compute profile.
    checkpoint_refiner = True
    store_all_step_logits = True
    # The historical RichDecoder uses the plans' decoder block list, capped at one block per
    # stage by LiteRBUNeXt3DFeatureRefiner. Subclasses can override these two attributes to run
    # targeted decoder-capacity ablations without copying this entire network builder.
    decoder_block_pattern = None
    max_decoder_blocks = 1

    def on_train_start(self):
        super().on_train_start()
        warm_start_path = os.environ.get(_WARM_START_STAGE_A_ENV)
        if not warm_start_path:
            return
        # Stage A is bit-identical across this trainer and its VirtualDeep descendants: the
        # network is the same (they inherit build_network_architecture below, and neither
        # overrides it), the virtual-deep branch lives only in the Stage-B backward, and
        # AlternatingStageScheduler gives *every* param group scale=1.0 while stage=="A" -- so
        # VirtualDeep's extra "film" optimizer group cannot change Stage A's LR for any
        # parameter. A sibling's checkpoint_stage_A.pth therefore loads directly here and skips
        # redoing 50 already-completed epochs.
        #
        # This mirrors the identical hook in the VirtualDeep subclass (env var
        # VIRTUALDEEP_WARM_START_A) and follows nnUNetTrainer.load_checkpoint()
        # (network_weights, current_epoch, logger, _best_ema, inference_allowed_mirroring_axes)
        # while deliberately skipping optimizer_state/grad_scaler_state: a VirtualDeep
        # checkpoint's optimizer has 5 param groups against this trainer's 4, so
        # Optimizer.load_state_dict would raise "different number of parameter groups" (and a
        # fresh optimizer is what a Stage-A-only warm start should have anyway).
        #
        # Invoke `train` with neither CONTINUE=1 nor PRETRAINED_WEIGHTS set -- both trigger
        # nnU-Net's own (conflicting) checkpoint logic before this hook runs.
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
        # epoch, so current_epoch must equal the logger's own recorded length, not the
        # checkpoint's stored `current_epoch` scalar (empirically 51 against 50 logged epochs --
        # trusting it skips epoch 50 and desyncs the logger, crashing one epoch later).
        self.current_epoch = len(self.logger.my_fantastic_logging["train_losses"])
        self._best_ema = checkpoint["_best_ema"]
        if "inference_allowed_mirroring_axes" in checkpoint:
            self.inference_allowed_mirroring_axes = checkpoint["inference_allowed_mirroring_axes"]
        self.print_to_log_file(
            f"{_WARM_START_STAGE_A_ENV} set: warm-started network weights + logger/epoch state "
            f"from {warm_start_path}; resuming at epoch {self.current_epoch} (Stage A skipped, "
            f"fresh optimizer)."
        )

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
        decoder_blocks = (
            configuration_manager.n_conv_per_stage_decoder
            if cls.decoder_block_pattern is None
            else list(cls.decoder_block_pattern)
        )
        if len(decoder_blocks) != len(channels) - 1:
            raise ValueError(
                f"decoder_block_pattern must have {len(channels) - 1} entries, got {decoder_blocks}"
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
            refiner_style="sdrr_film",
            normalize_finest_before_head=False,
            checkpoint_refiner=cls.checkpoint_refiner,
            deep_supervision=False,
            clean_refiner_inputs=cls.clean_refiner_inputs,
            rich_proposal_stages=True,
            n_blocks_per_stage_decoder=decoder_blocks,
            max_decoder_blocks=cls.max_decoder_blocks,
            store_all_step_logits=cls.store_all_step_logits,
        )
        print(
            "LiteFeatureRefinerD32H128D16UpSkipFreshMS-RichDecoder-FiLM: {}; clean_refiner_inputs={}; "
            "checkpoint_refiner={}; store_all_step_logits={}; decoder_blocks={}".format(
                model_summary_line(model),
                cls.clean_refiner_inputs,
                cls.checkpoint_refiner,
                cls.store_all_step_logits,
                [len(stage.refine) for stage in model.proposal_stages],
            )
        )
        return model
