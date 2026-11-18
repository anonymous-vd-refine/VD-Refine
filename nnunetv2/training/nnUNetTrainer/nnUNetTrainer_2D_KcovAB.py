"""2D (ISIC / DRIVE) port of the kcov_ab recipe validated on Dataset822 (LiTS).

kcov_ab = Stage-A K=0 auxiliary coverage (lambda=0.8) on top of proposal-preserving B
stages (encoder/decoder/heads at full 1e-4 in B1, 5e-5 in B2; FiLM pinned at 0). On
Dataset822's 26-case held-out set it scored 0.7936 (K=4) / 0.7978 (K=8) / 0.7983 (K=16)
against a 0.7727 no-refiner anchor, and beat both a fully frozen B (0.7617) and a
0.1x-backbone B (0.7766).

Why this module duplicates the Stage-A train_step instead of importing the mixin
--------------------------------------------------------------------------------
`_LiTS822KZeroAStageMixin` in nnUNetTrainer_LiTS822_KcovA is not a bare mixin: it inherits
`nnUNetTrainer_LiTS822_ProposalPreservingFiveStage`, which pins the 3D flagship
(...BatchDiceLossUNeXt3DStableV2) into every subclass's MRO. Deriving a 2D trainer from it
would either fail to linearize against the 2D flagship or silently resolve
`build_network_architecture` to the 3D one. Rebinding its `train_step` onto a fresh mixin
does not work either -- the zero-argument `super()` inside that method compiles with a
`__class__` cell bound to `_LiTS822KZeroAStageMixin`, so it raises TypeError on any `self`
that is not an instance of it.

Rewriting the 3D mixin to be base-free would fix this properly, but that file is imported by
the currently running Dataset802 job and defines the classes that produced the published
Dataset822 numbers. The ~25 duplicated lines below are the cheaper trade, and match how the
rest of this trainer directory is organised. `_KZeroAStageMixin` is kept base-free here so
the 2D side does not repeat the same trap.

Everything else is imported, not copied: `_ProposalPreservingBMixin` turned out to be fully
dimension-agnostic (it only reaches for `network.single_refiner.shared_block`, which the 2D
adapter also exposes), and the 2D flagship's five-stage schedule is identical to the 3D one
-- A 0-49, B1 50-69, C1 70-79, B2 80-89, C2 90-99 over 100 epochs -- so the hard-coded epoch
table in `_ProposalPreservingScheduler` lines up without modification. Its
`configure_optimizers` also yields the same five named groups
(refiner / film / encoder / decoder / heads) that the scheduler indexes by name.

Both datasets are 2-class and non-region-based, so `self.loss` builds as DC_and_CE_loss and
the BatchDiceLoss flagship's `_per_case_seg_loss` delegates to it unchanged.
"""

from __future__ import annotations

import random

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthBatchDiceLossUNeXt2DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthBatchDiceLossUNeXt2DStableV2 as _Flagship2D,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LiTS822_ProposalPreserving import (
    _ProposalPreservingBMixin,
)


class _KZeroAStageMixin:
    """Add a supervised K=0 proposal decode to Stage A (lambda = k0_weight).

    Base-free by design; see the module docstring. Behaviourally a verbatim port of
    `_LiTS822KZeroAStageMixin.train_step` -- keep the two in sync if either changes.
    """

    k0_weight = 0.8

    def train_step(self, batch: dict) -> dict:
        stage, _, _ = self._stage_spec_at(self.current_epoch)
        if stage != "A":
            return super().train_step(batch)

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [value.to(self.device, non_blocking=True) for value in target]
        else:
            target = target.to(self.device, non_blocking=True)
        target0 = self._target0(target)

        self.optimizer.zero_grad(set_to_none=True)
        depth = 1 if random.random() < 0.6 else 2
        with self._autocast_context():
            refined_logits = self.network(data, recurrent_steps=depth, tbptt_keep_steps=depth)
            refined_loss = self._per_case_seg_loss(refined_logits, target0).mean()
        refined_loss.backward()

        # Second objective: one decode of proposal_states[-1]. Its gradient reaches the
        # encoder/proposal stages but cannot update the refiner, exactly like k0aux.
        with self._autocast_context():
            proposal_logits = self.network(data, recurrent_steps=0, tbptt_keep_steps=1)
            proposal_loss = self._per_case_seg_loss(proposal_logits, target0).mean()
            weighted_proposal_loss = self.k0_weight * proposal_loss
        weighted_proposal_loss.backward()

        self._clip_gradients()
        self.optimizer.step()
        return {
            "loss": (refined_loss + self.k0_weight * proposal_loss).detach().cpu().numpy(),
            "loss_kA": refined_loss.detach().cpu().numpy(),
            "loss_k0": proposal_loss.detach().cpu().numpy(),
        }


class _KcovAB2D(_KZeroAStageMixin, _ProposalPreservingBMixin, _Flagship2D):
    """kcov_ab on the 2D flagship.

    C3 linearization is KZero -> _ProposalPreservingBMixin -> flagship: the Stage-A mixin
    intercepts A and delegates everything else upward, the B mixin intercepts B1/B2 and
    delegates C1/C2 to the flagship. Neither mixin knows about the other.
    """

    def _stage_log_message(self, stage, k_max) -> str:
        message = super()._stage_log_message(stage, k_max)
        if stage == "A":
            message += f"; k0_aux=True; lambda_k0={self.k0_weight}"
        return message


class nnUNetTrainer_ISIC803_KcovAB(_KcovAB2D):
    """kcov_ab on Dataset803_ISIC (2d, 3 input channels, 2 classes)."""


class nnUNetTrainer_DRIVE804_KcovAB(_KcovAB2D):
    """kcov_ab on Dataset804_DRIVE (2d, 3 input channels, 2 classes)."""
