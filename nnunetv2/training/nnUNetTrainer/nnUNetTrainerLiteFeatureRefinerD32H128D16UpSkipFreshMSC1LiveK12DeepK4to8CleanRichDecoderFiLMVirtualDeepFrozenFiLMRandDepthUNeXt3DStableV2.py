from __future__ import annotations

import random

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMUNeXt3DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMUNeXt3DStableV2,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteSDRRAlternatingABCDec16Dec32UNeXt3DStableV2 import (
    _stage_spec,
)


class nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthUNeXt3DStableV2(
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMUNeXt3DStableV2
):
    """FrozenFiLM VirtualDeep, but the extrapolation depth is sampled per batch instead of fixed.

    The parents jump the geometric series to a single fixed synthetic depth: virtual_start=14 in
    B1 (target_depth 16), and 14 or 30 by coin flip in B2 (target_depth 16 or 32). Two real
    refiner steps then run from there, so the branch only ever supervises the refiner at two
    distinct depths. Here virtual_start is drawn per batch:

      B1: uniform over [8, 14]                            -> target_depth 10..16
      B2: 60% uniform over [8, 14], 40% uniform [15, 30]   -> target_depth 10..16 / 17..32

    Sampling from 8 upward means some batches get virtual_start=8 exactly, where the geometric
    jump contributes nothing (n_steps=0 -> a_scale=0 -> d_tilde == d8): that draw trains on the
    REAL depth-8 state rather than a synthetic one, which anchors the branch against
    extrapolation error. The upper ends, 14 and 30, are the parents' fixed values, so this widens
    the supervised depth range downward rather than moving it.

    The w_v ramps are inherited unchanged (0.05->0.20 across B1, 0.20->0.25 across B2), as is
    everything else: FiLM stays frozen through B, so the branch's gradient still reaches
    single_refiner.shared_block alone.

    ODD DEPTHS. q is estimated from u6=d6-d4 and u8=d8-d6, making it a per-TWO-step contraction
    ratio, so the parent's integer block count `(virtual_start - 8) // 2` only lands on the
    requested depth for even virtual_start -- 9, 11, 13 and 15 would silently land one step
    short. _virtual_jump_n_steps is therefore overridden to return a fractional block count, and
    torch.pow evaluates q**0.5 fine for the clamped q in [0, 0.95]. This is bitwise identical to
    the integer version at every even depth (verified: torch.equal(q.pow(3), q.pow(3.0)) is True),
    so the parents' 14/30 behaviour is untouched, and for odd depths it interpolates the same
    fixed-point model rather than misreporting the depth. Exactness matters little either way:
    d_tilde is detached and followed by two REAL refiner steps before any loss, so the synthetic
    state only has to land in the right neighbourhood.

    The historical model reused the plain FiLM Stage-A weights and reset the optimizer.
    The release runner trains seeds from scratch; see docs/PROTOCOL.md for provenance.
    """

    _virtual_b1_start_range = (8, 14)
    _virtual_b2_near_range = (8, 14)
    _virtual_b2_far_range = (15, 30)
    _virtual_b2_near_prob = 0.6

    @staticmethod
    def _virtual_jump_n_steps(virtual_start: int) -> float:
        return (virtual_start - 8) / 2.0

    def _record_virtual_start(self, virtual_start: int) -> None:
        if getattr(self, "_sampled_virtual_starts", None) is None:
            self._sampled_virtual_starts = []
        self._sampled_virtual_starts.append(int(virtual_start))

    def _virtual_branch_schedule(self, epoch: int) -> tuple[int, float]:
        # _stage_spec_at / _virtual_ramp_weight are scale-aware; at stage_scale=1 both reduce to
        # the original unscaled arithmetic, so this is inert for every existing trainer.
        stage, _, cycle = self._stage_spec_at(epoch)
        if not stage.startswith("B"):
            raise ValueError(f"virtual branch schedule requested outside stage B: epoch={epoch}, stage={stage}")
        w_v = self._virtual_ramp_weight(epoch, cycle)
        if cycle == 1:
            virtual_start = random.randint(*self._virtual_b1_start_range)
        else:
            near = random.random() < self._virtual_b2_near_prob
            span = self._virtual_b2_near_range if near else self._virtual_b2_far_range
            virtual_start = random.randint(*span)
        self._record_virtual_start(virtual_start)
        # The caller does virtual_start = target_depth - 2, so return the depth two real refiner
        # steps past the synthetic state.
        return virtual_start + 2, w_v

    def _virtual_schedule_log_fragment(self, epoch: int) -> str:
        # Report the policy, not a draw: virtual_start is resampled every batch, so printing one
        # sample here would be unrepresentative -- and would also pollute the per-epoch
        # sampled-depth statistics logged by on_train_epoch_end().
        stage, _, cycle = self._stage_spec_at(epoch)
        if cycle == 1:
            policy = f"uniform{self._virtual_b1_start_range}"
        else:
            policy = (
                f"{self._virtual_b2_near_prob:.2f}*uniform{self._virtual_b2_near_range}"
                f"+{1.0 - self._virtual_b2_near_prob:.2f}*uniform{self._virtual_b2_far_range}"
            )
        w_v = self._virtual_ramp_weight(epoch, cycle)
        return f"; virtual_start_policy={policy}; virtual_w={w_v:.4f}"

    def on_train_epoch_end(self, train_outputs):
        super().on_train_epoch_end(train_outputs)
        sampled = getattr(self, "_sampled_virtual_starts", None)
        if not sampled:
            return
        near_high = self._virtual_b2_near_range[1]
        near = sum(1 for depth in sampled if depth <= near_high)
        self.print_to_log_file(
            f"virtual_start sampled: n={len(sampled)} mean={sum(sampled) / len(sampled):.2f} "
            f"min={min(sampled)} max={max(sampled)} frac<={near_high}:{near / len(sampled):.3f}"
        )
        self._sampled_virtual_starts = []
