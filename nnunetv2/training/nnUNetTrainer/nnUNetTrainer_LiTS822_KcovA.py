"""Dataset822 kcov lineage, part 1: Stage-A K=0 auxiliary coverage.

P0-1 `kcov_a`: the five-stage curriculum with Stage A extended to also supervise the
K=0 proposal decode (lambda=0.8), following the k0aux recipe already validated on the
A-only100 control (aonly100_k0aux 0.7925 vs plain aonly100 0.7653). Stage A keeps its
K=1(0.6)/K=2(0.4) refined branch; branch 2 is a separate `recurrent_steps=0` forward
(one decode of proposal_states[-1]) whose loss is scaled by `k0_weight` and backward()
ed onto the same parameter state. The K=0 gradient reaches the encoder/proposal stages
but cannot update the refiner, exactly like k0aux.

P0-2 `kcov_ab`: the same Stage-A recipe on top of the proposal-preserving B stages
(backbone at full 1e-4/5e-5), i.e. this module additionally provides
nnUNetTrainer_LiTS822_KcovAB.

MRO contract: _LiTS822KZeroAStageMixin inherits from
nnUNetTrainer_LiTS822_ProposalPreservingFiveStage (which itself is
(_ProposalPreservingBMixin, flagship)), so a trainer combining the two bases is simply
(_LiTS822KZeroAStageMixin, _ProposalPreservingFiveStage): the C3 linearization yields
KZero -> FiveStage -> _ProposalPreservingBMixin -> flagship, the mixin"s train_step
intercepts Stage A, and non-A stages delegate through super() to the B mixin (Stage B)
then the flagship (C1/C2). The B mixin itself delegates non-B stages upward, so the two
mixins compose without either knowing about the other.
"""

import random

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_LiTS822_ProposalPreserving import (
    nnUNetTrainer_LiTS822_ProposalPreservingFiveStage as _ProposalPreservingFiveStage,
)


class _LiTS822KZeroAStageMixin(_ProposalPreservingFiveStage):
    """Add a supervised K=0 proposal decode to Stage A (lambda = k0_weight)."""

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
            refined_logits = self.network(data, recurrent_steps=depth, tbptt_keep_steps=depth, decode_all_steps=False)
            refined_loss = self._per_case_seg_loss(refined_logits, target0).mean()
        refined_loss.backward()

        with self._autocast_context():
            proposal_logits = self.network(data, recurrent_steps=0, tbptt_keep_steps=1, decode_all_steps=False)
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


class nnUNetTrainer_LiTS822_KcovA(_LiTS822KZeroAStageMixin, _ProposalPreservingFiveStage):
    """P0-1: five-stage flagship, Stage A gains a K=0 aux branch at lambda=0.8."""

    def _stage_log_message(self, stage, k_max) -> str:
        message = super()._stage_log_message(stage, k_max)
        if stage == "A":
            message += f"; k0_aux=True; lambda_k0={self.k0_weight}"
        return message


class nnUNetTrainer_LiTS822_KcovAB(_LiTS822KZeroAStageMixin, _ProposalPreservingFiveStage):
    """P0-2: kcov_a Stage A + proposal-preserving B (backbone full 1e-4 / 5e-5)."""
