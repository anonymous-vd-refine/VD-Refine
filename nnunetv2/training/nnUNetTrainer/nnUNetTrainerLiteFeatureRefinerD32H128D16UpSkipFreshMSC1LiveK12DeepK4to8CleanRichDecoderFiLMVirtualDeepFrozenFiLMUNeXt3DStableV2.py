from __future__ import annotations

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepUNeXt3DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepUNeXt3DStableV2,
)


class nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMUNeXt3DStableV2(
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepUNeXt3DStableV2
):
    """VirtualDeep's Stage-B branch, but Stage B keeps the plain FiLM trainer's freeze semantics.

    The parent unfreezes single_refiner.film_head during B1/B2 and gives its optimizer group a
    nonzero LR, so the virtual deep branch's gradient reaches both film_head (33,024 params) and
    single_refiner.shared_block (86,272). Here film_head stays frozen bit-for-bit through B, as
    it is under the plain FiLM RichDecoder run, so the branch trains **shared_block alone**. That
    is the question this variant asks: does a cheap deep-K (16/32) gradient signal help the
    recurrent block itself, with the FiLM normalization held at its Stage-A values?

    Stage A and Stage C are untouched and identical to the plain FiLM run. In A every group is
    scaled 1.0; in C the `film` group's 0.10 already equals the `decoder` scale film_head would
    have received there under the parent trainer's 4-group optimizer. So B is the only stage that
    differs from the plain FiLM baseline, and it differs by the virtual branch alone -- a clean
    single-variable ablation.

    Three further flags are set here purely for memory/compute; none of them changes the model's
    mathematics. See the release verification report for the checks actually shipped:

    * checkpoint_refiner=False -- gradient checkpointing wraps only the 32^3 x 128ch recurrent
      block, the smallest part of the graph. Measured on BraTS 3d_fullres it saves 416 MB (4.4%
      of Stage B's 9.4 GB peak, 1.4% of Stage A's 29.9 GB) and costs +7.5% Stage-B wall time,
      because B recomputes 8 grad-tracked refiner steps per iteration against Stage A's 2. A bad
      trade on an 80 GB card. Toggling it is numerically neutral: initial_state_drop_prob is 0
      and _drop_initial_state() runs outside the checkpointed lambda, so no RNG lives inside it.
    * store_all_step_logits=False -- forward() otherwise pins every dead decode branch's
      activations on the module for the whole iteration and into the next (12.7 GB at Stage-A
      depth 2; back-to-back peak 42.2 GB instead of 29.4 GB). The historical run reused Stage A; this memory-saving setting also applies
      to the release seed runs, which start from scratch.
    * _proposal_stop_index -- the B/C helpers read only proposal_states[1] and [2] and rebuild
      everything above the refine index via _replay_proposal_from_state, so building the 64^3 and
      128^3 stages in _proposal() is wasted work. Wasted compute under no_grad in B/C2; in C1,
      whose _proposal() call is grad-enabled, it also pins those activations for the whole step.

    Optional historical Stage-A reuse uses VIRTUALDEEP_WARM_START_A; the release runner
    disables it for scratch runs. Stage A is bit-identical between the two trainers. Own RESULT_DIR, so the parent's completed
    unfrozen-FiLM run stays intact and comparable.
    """

    _train_film_in_b = False
    checkpoint_refiner = False
    store_all_step_logits = False

    def _proposal_stop_index(self, network) -> int | None:
        return network.single_refine_index
