import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt3DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt3DStableV2,
)


class nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthBatchDiceLossUNeXt3DStableV2(
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt3DStableV2
):
    """Same as the RandDepth parent, but every backward pass uses nnU-Net's own standard
    batch-pooled DC_and_CE_loss (self.loss) instead of the FreshMultiStart base's per-case,
    unweighted-per-class _per_case_seg_loss -- see
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLM
    BatchDiceLossUNeXt3DStableV2's docstring for the full investigation and motivation (same
    hypothesis, same fix, this is the RandDepth-lineage sibling).

    RandDepth itself doesn't override any loss/backward method (only the virtual-depth sampling
    schedule), and its VirtualDeep ancestor's own _fresh_multistart_backward override routes BOTH
    the real branch and the synthetic/virtual-depth branch through
    self._per_case_seg_loss(...) (see nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1
    LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepUNeXt3DStableV2.py:227,263) -- so overriding
    just this one method here transparently fixes both branches, same as it did for the plain
    FiLM trainer; no need to duplicate VirtualDeep's or RandDepth's own methods.

    batch_size stays at 2 (see the plain-FiLM BatchDiceLoss sibling's docstring for why) so the
    loss formula remains the only variable changed from this trainer's own per-case-loss parent.

    Deliberately NOT warm-started (no VIRTUALDEEP_WARM_START_A): the parent normally warm-starts
    Stage A from the plain FiLM run's checkpoint_stage_A.pth (itself trained under the old
    per-case loss), which would leave Stage A untouched by this fix. Training Stage A from scratch
    here keeps every BatchDiceLoss-family experiment self-contained and gives the loss fix full
    coverage from epoch 0, at the cost of the extra ~50 epochs / ~3h Stage A normally skips.
    """

    def _per_case_seg_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, target)
