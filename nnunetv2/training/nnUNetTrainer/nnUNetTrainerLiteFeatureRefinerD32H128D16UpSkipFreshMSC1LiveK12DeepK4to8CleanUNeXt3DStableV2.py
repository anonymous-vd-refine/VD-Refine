import random

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMultiStartK8TBPTT2ABCUNeXt3DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMultiStartK8TBPTT2ABCUNeXt3DStableV2,
)


class nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanUNeXt3DStableV2(
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMultiStartK8TBPTT2ABCUNeXt3DStableV2
):
    """Clean SDRR inputs with C1 live K=1/2 and deep K=4..8."""

    clean_refiner_inputs = True
    c1_live_depths = (1, 2)
    c1_deep_depths = (4, 5, 6, 7, 8)
    c1_live_weight = 0.625
    c1_deep_weight = 0.375
    b_start_depth_bands = ((0, 1), (2, 4), (4, 6))

    def _stage_log_message(self, stage, k_max) -> str:
        return (
            f"D32 FreshMS stage={stage}; K_max={k_max}; b_start_depth_bands={self.b_start_depth_bands}; "
            f"tbptt_steps={self.tbptt_steps}; c1_live_depths={self.c1_live_depths}; "
            f"c1_deep_depths={self.c1_deep_depths}; monotonic_beta={self.monotonic_beta}"
        )

    def _fresh_multistart_backward(self, data, target):
        network = self._unwrap_network(self.network)
        target0 = self._target0(target)
        max_depth = max(high for _, high in self.b_start_depth_bands) + self.tbptt_steps
        with torch.no_grad(), self._autocast_context():
            proposal_states, skips, input_shape = self._proposal(data)
            behavior = self._behavior_rollout(network, proposal_states, skips, max_depth)

        branch_losses = {}
        monotonic_losses = []
        previous_endpoint_loss = None
        num_branches = len(self.b_start_depth_bands)
        for band_idx, (low, high) in enumerate(self.b_start_depth_bands):
            start_depth = random.randint(low, high)
            terminal_depth = start_depth + self.tbptt_steps
            with self._autocast_context():
                state = behavior[start_depth].detach()
                for step_idx in range(start_depth, terminal_depth):
                    state = self._one_refine_step(network, state, proposal_states, skips, step_idx)
                logits = self._decode_from_refined_state(network, state, proposal_states, skips, input_shape)
                per_case_loss = self._per_case_seg_loss(logits, target0)
                branch_loss = per_case_loss.mean()
                if previous_endpoint_loss is None or self.monotonic_beta <= 0:
                    monotonic_loss = torch.zeros((), device=data.device)
                else:
                    monotonic_loss = torch.relu(per_case_loss - previous_endpoint_loss).mean()
                objective = (branch_loss + self.monotonic_beta * monotonic_loss) / num_branches
            objective.backward()
            # Keyed by band index, not the randomly-sampled terminal_depth: nnU-Net's
            # collate_outputs() requires every batch in an epoch to return dicts with identical
            # keys, but terminal_depth = start_depth + tbptt_steps varies batch-to-batch within a
            # band (e.g. band (0,1) yields terminal_depth 2 or 3) -- keying by the sampled depth
            # crashed with KeyError once a batch's key set didn't match another's.
            branch_losses[f"loss_k_band{band_idx}"] = branch_loss.detach()
            monotonic_losses.append(monotonic_loss.detach())
            previous_endpoint_loss = per_case_loss.detach()
        branch_losses["monotonic_loss"] = torch.stack(monotonic_losses).mean()
        return branch_losses

    def _c1_backward(self, data, target):
        network = self._unwrap_network(self.network)
        target0 = self._target0(target)
        live_depth = self.c1_live_depths[0] if random.random() < 0.8 else self.c1_live_depths[1]
        deep_depths = random.sample(self.c1_deep_depths, 3)

        with self._autocast_context():
            proposal_states, skips, input_shape = self._proposal(data)
            state = proposal_states[network.single_refine_index]
            for step_idx in range(live_depth):
                state = self._one_refine_step(network, state, proposal_states, skips, step_idx)
            live_logits = self._decode_from_refined_state(network, state, proposal_states, skips, input_shape)
            live_loss = self._per_case_seg_loss(live_logits, target0).mean()
        (self.c1_live_weight * live_loss).backward()

        # Reuse the live branch's encoder/proposal output (numerically identical, weights
        # unchanged until optimizer.step()) instead of re-running _proposal(data) from scratch.
        proposal_states_detached = [state.detach() for state in proposal_states]
        skips_detached = [skip.detach() for skip in skips]
        with self._autocast_context():
            behavior = self._behavior_rollout(network, proposal_states_detached, skips_detached, max(deep_depths))

        # Keyed by fixed labels, not the randomly-sampled live_depth/deep depths: collate_outputs()
        # requires identical dict keys across every batch in an epoch, but live_depth is randomly
        # 1 or 2 and deep_depths is a random 3-of-5 sample each batch -- keying by the sampled
        # value crashed with KeyError once a batch's key set didn't match another's (same class of
        # bug as _fresh_multistart_backward's band-index fix above).
        losses = {"loss_k_live": live_loss.detach()}
        num_deep = len(deep_depths)
        for deep_idx, depth in enumerate(deep_depths):
            with self._autocast_context():
                deep_logits = self._decode_from_refined_state(
                    network, behavior[depth], proposal_states_detached, skips_detached, input_shape
                )
                branch_loss = self._per_case_seg_loss(deep_logits, target0).mean()
            (self.c1_deep_weight * branch_loss / num_deep).backward()
            losses[f"loss_k_deep{deep_idx}"] = branch_loss.detach()
        return losses

    def _c2_backward(self, data, target):
        network = self._unwrap_network(self.network)
        target0 = self._target0(target)
        with torch.no_grad(), self._autocast_context():
            proposal_states, skips, input_shape = self._proposal(data)
            behavior = self._behavior_rollout(network, proposal_states, skips, max(self.c1_deep_depths))
        losses = {}
        num_deep = len(self.c1_deep_depths)
        for depth in self.c1_deep_depths:
            with self._autocast_context():
                logits = self._decode_from_refined_state(network, behavior[depth], proposal_states, skips, input_shape)
                branch_loss = self._per_case_seg_loss(logits, target0).mean()
            (branch_loss / num_deep).backward()
            losses[f"loss_k{depth}"] = branch_loss.detach()
        return losses
