import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt2DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt2DStableV2,
)


class nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthBatchDiceLossUNeXt2DStableV2(
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt2DStableV2
):
    """Same as the 2D ISIC RandDepth parent, but every backward pass uses nnU-Net's own standard
    batch-pooled DC_and_CE_loss (self.loss) instead of the FreshMultiStart base's per-case,
    unweighted-per-class _per_case_seg_loss -- port of the 3D BraTS/LiTS BatchDiceLoss siblings
    (nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLM
    {,VirtualDeepFrozenFiLMRandDepth}BatchDiceLossUNeXt3DStableV2) to this project's 2D ISIC
    adapter lineage, same fix, same one-line mechanism.

    Neither RandDepth nor its VirtualDeep/FrozenFiLM/2D-adapter ancestors override any loss/
    backward method beyond sampling schedules and the singleton-depth I/O adapter; the shared
    FreshMultiStart base's `_fresh_multistart_backward` routes BOTH the real multi-start branch
    and the synthetic/virtual-depth branch through `self._per_case_seg_loss(...)`, so overriding
    just this one method here transparently fixes both, without touching the 2D adapter's own
    `build_network_architecture`/`_proposal`/`_decode_from_refined_state` overrides or any shared
    base class other lineage members depend on.

    Motivation: unlike BraTS/LiTS (3D, batch_size=2, region-based or multi-class with a rare
    class), ISIC is 2D single-foreground-class lesion segmentation with batch_size=13 and
    batch_dice=True already in nnUNetPlans.json -- i.e. nnU-Net's own planner already judged
    batch-pooled dice appropriate for this dataset, same as it did for BraTS/LiTS's no-refiner
    baseline and this trainer's own validation_step/pseudo-dice. The FreshMultiStart base's
    _per_case_seg_loss instead computes an unweighted per-case dice every training step,
    inconsistent with both the plans-level default and what pseudo-dice actually tracks -- the
    same category of train/eval-metric mismatch the 3D siblings were built to isolate and test,
    applied here as a default rather than a confirmed post-hoc fix (no prior per-case-loss vs.
    batch-dice-loss comparison exists yet for ISIC).

    batch_size is left untouched at the plan's own value (13, unlike BraTS/LiTS's fixed 2) so this
    trainer changes only the loss formula relative to its RandDepth parent.

    Deliberately NOT warm-started, matching every other ISIC config in this project (the ISIC
    RandDepth parent config script has no VIRTUALDEEP_WARM_START_A of its own) -- Stage A is
    trained from scratch, so the loss fix already applies to it from epoch 0.
    """

    def _per_case_seg_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, target)
