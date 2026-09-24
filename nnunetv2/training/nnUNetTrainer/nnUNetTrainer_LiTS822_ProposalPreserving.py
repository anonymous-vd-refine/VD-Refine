from __future__ import annotations

import random
from os.path import join

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthBatchDiceLossUNeXt3DStableV2 import (
    nnUNetTrainerLiteFeatureRefinerD32H128D16UpSkipFreshMSC1LiveK12DeepK4to8CleanRichDecoderFiLMVirtualDeepFrozenFiLMRandDepthBatchDiceLossUNeXt3DStableV2 as _LiTS822Flagship,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiteSDRRAlternatingABCDec16Dec32UNeXt3DStableV2 import (
    _stage_spec_scaled,
)


class _AOnlyConstantScheduler:
    """Keep every optimizer group at Stage-A's LR for all 100 epochs."""

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.last_epoch = -1

    def step(self, epoch: int | None = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = int(epoch)
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr

    def state_dict(self):
        return {"base_lr": self.base_lr, "last_epoch": self.last_epoch}

    def load_state_dict(self, state_dict):
        self.base_lr = float(state_dict.get("base_lr", self.base_lr))
        self.last_epoch = int(state_dict.get("last_epoch", -1))
        if self.last_epoch >= 0:
            self.step(self.last_epoch)


class _ProposalPreservingScheduler:
    """Stage LR table with nonzero backbone LR during proposal-preserving B.

    This deliberately keeps the established constant/stage-scaled 1e-4 schedule. The only
    scheduler change from the flagship is that encoder/decoder/heads are no longer assigned
    zero LR in B. B1 uses the full Stage-A LR; cycle-2 B2 uses half of it. Frozen FiLM keeps a
    zero B-stage LR, and the recurrent shared block retains the flagship's 1.0/0.2 B scales.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, schedule: str):
        if schedule not in {"five_stage", "a80_b20"}:
            raise ValueError(f"unsupported proposal-preserving schedule: {schedule}")
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.schedule = schedule
        self.last_epoch = -1

    def _stage_spec(self, epoch: int) -> tuple[str, int | None, int]:
        if self.schedule == "a80_b20":
            return ("A", None, 1) if epoch < 80 else ("B1-K8", 8, 1)
        return _stage_spec_scaled(epoch, 1)

    def step(self, epoch: int | None = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = int(epoch)
        stage, _, cycle = self._stage_spec(self.last_epoch)
        cycle_scale = 1.0 if cycle == 1 else 0.5
        for group in self.optimizer.param_groups:
            name = group["name"]
            if stage == "A":
                scale = 1.0
            elif stage.startswith("B"):
                if name == "refiner":
                    scale = 1.0 if cycle == 1 else 0.2
                elif name == "film":
                    scale = 0.0
                else:
                    scale = cycle_scale
            else:
                scale = {
                    "refiner": 0.0,
                    "film": 0.10,
                    "encoder": 0.05,
                    "decoder": 0.10,
                    "heads": 1.0,
                }[name] * cycle_scale
            group["lr"] = self.base_lr * scale

    def state_dict(self):
        return {
            "base_lr": self.base_lr,
            "schedule": self.schedule,
            "last_epoch": self.last_epoch,
        }

    def load_state_dict(self, state_dict):
        self.base_lr = float(state_dict.get("base_lr", self.base_lr))
        saved_schedule = state_dict.get("schedule", self.schedule)
        if saved_schedule != self.schedule:
            raise RuntimeError(
                f"scheduler mismatch: checkpoint={saved_schedule}, trainer={self.schedule}"
            )
        self.last_epoch = int(state_dict.get("last_epoch", -1))
        if self.last_epoch >= 0:
            self.step(self.last_epoch)


class nnUNetTrainer_LiTS822_AOnly100(_LiTS822Flagship):
    """Strict E100 A-only control, trained from scratch with BatchDiceLoss.

    All 100 epochs jointly train the complete model at K=1 (60%) or K=2 (40%), exactly matching
    the flagship's Stage-A recipe except that Stage A does not end at epoch 50. This is distinct
    from the reusable adaptive-early-stop A checkpoint, whose selected snapshot is E109.
    """

    def _stage_spec_at(self, epoch: int) -> tuple[str, int | None, int]:
        del epoch
        return "A", None, 1

    def configure_optimizers(self):
        optimizer, _ = super().configure_optimizers()
        return optimizer, _AOnlyConstantScheduler(optimizer, self.initial_lr)

    def _stage_checkpoint_epochs(self) -> dict[int, str]:
        return {99: "checkpoint_stage_AOnly.pth"}

    def _stage_log_message(self, stage, k_max) -> str:
        return super()._stage_log_message(stage, k_max) + "; control=AOnly100; lr=constant-1e-4"


class _ProposalPreservingBMixin:
    """Give refiner and backbone separate, gradient-isolated objectives in Stage B."""

    proposal_aux_weight = 0.10
    proposal_preserving_depths = (1, 2, 4)
    proposal_schedule = "five_stage"

    def configure_optimizers(self):
        optimizer, _ = super().configure_optimizers()
        return optimizer, _ProposalPreservingScheduler(
            optimizer, self.initial_lr, self.proposal_schedule
        )

    def _set_refiner_objective_trainability(self) -> None:
        network = self._unwrap_network(self.network)
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        # Preserve FrozenFiLM semantics: only the recurrent shared transition receives the
        # multi-start/VirtualDeep objective, exactly as in the existing flagship.
        for parameter in network.single_refiner.shared_block.parameters():
            parameter.requires_grad_(True)

    def _set_backbone_objective_trainability(self) -> None:
        network = self._unwrap_network(self.network)
        for parameter in network.parameters():
            parameter.requires_grad_(True)
        # The K>=1 backbone loss must propagate THROUGH the fixed refiner to d0, but must not
        # update any refiner-owned parameter (including FiLM, adapters, and projections).
        for parameter in network.single_refiner.parameters():
            parameter.requires_grad_(False)

    def _proposal_preserving_backward(self, data: torch.Tensor, target) -> dict[str, torch.Tensor]:
        target0 = self._target0(target)
        depth = random.choice(self.proposal_preserving_depths)

        with self._autocast_context():
            refined_logits = self.network(
                data, recurrent_steps=depth, tbptt_keep_steps=depth, decode_all_steps=False
            )
            refined_loss = self._per_case_seg_loss(refined_logits, target0).mean()
        refined_loss.backward()

        with self._autocast_context():
            proposal_logits = self.network(
                data, recurrent_steps=0, tbptt_keep_steps=1, decode_all_steps=False
            )
            proposal_loss = self._per_case_seg_loss(proposal_logits, target0).mean()
            weighted_proposal_loss = self.proposal_aux_weight * proposal_loss
        weighted_proposal_loss.backward()
        return {
            f"loss_backbone_k{depth}": refined_loss.detach(),
            "loss_proposal_k0": proposal_loss.detach(),
        }

    def train_step(self, batch: dict) -> dict:
        stage, _, _ = self._stage_spec_at(self.current_epoch)
        if not stage.startswith("B"):
            return super().train_step(batch)

        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [value.to(self.device, non_blocking=True) for value in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        self._set_refiner_objective_trainability()
        refiner_losses = self._fresh_multistart_backward(data, target)

        # Switching requires_grad does not erase the gradients accumulated above. It only
        # isolates the second objective so it can update the proposal-producing backbone.
        self._set_backbone_objective_trainability()
        backbone_losses = self._proposal_preserving_backward(data, target)

        # At this point the refiner is intentionally requires_grad=False, so the flagship's
        # active-parameter clip would omit its already-accumulated gradients. Clip every tensor
        # that actually has a gradient instead.
        with_grad = [
            parameter for parameter in self.network.parameters() if parameter.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(with_grad, max_norm=self.grad_clip_max_norm)
        self.optimizer.step()

        # The sampled backbone depth changes from batch to batch. nnU-Net's collate_outputs
        # stacks every key across an epoch, so expose a fixed schema and use zero placeholders
        # for the two unselected depths. Without this, the first B epoch crashes with a KeyError
        # as soon as two batches sample different K values.
        fixed_backbone = {
            f"loss_backbone_k{depth}": backbone_losses.get(
                f"loss_backbone_k{depth}",
                torch.zeros((), device=self.device),
            )
            for depth in (1, 2, 4)
        }
        all_losses = {**refiner_losses, **fixed_backbone, "loss_proposal_k0": backbone_losses["loss_proposal_k0"]}
        refiner_reported = [
            value for name, value in refiner_losses.items() if name.startswith("loss_k")
        ]
        total = (
            torch.stack(refiner_reported).mean()
            + next(value for name, value in backbone_losses.items() if name.startswith("loss_backbone"))
            + self.proposal_aux_weight * backbone_losses["loss_proposal_k0"]
        )
        result = {"loss": total.cpu().numpy()}
        result.update({name: value.cpu().numpy() for name, value in all_losses.items()})
        return result

    def _stage_log_message(self, stage, k_max) -> str:
        message = super()._stage_log_message(stage, k_max)
        if stage.startswith("B"):
            message += (
                f"; proposal_preserving=True; backbone_depths={self.proposal_preserving_depths}"
                f"; lambda_k0={self.proposal_aux_weight}; lr_policy=constant-stage-scaled"
            )
        return message


class nnUNetTrainer_LiTS822_A80JointB20(
    _ProposalPreservingBMixin,
    _LiTS822Flagship,
):
    """E100 A80 -> proposal-preserving joint-B20 diagnostic."""

    proposal_schedule = "a80_b20"
    _b1_start_epoch, _b1_end_epoch = 80, 100

    def _stage_spec_at(self, epoch: int) -> tuple[str, int | None, int]:
        return ("A", None, 1) if epoch < 80 else ("B1-K8", 8, 1)

    def _stage_checkpoint_epochs(self) -> dict[int, str]:
        return {
            79: "checkpoint_stage_A.pth",
            99: "checkpoint_stage_B1.pth",
        }


class nnUNetTrainer_LiTS822_ProposalPreservingFiveStage(
    _ProposalPreservingBMixin,
    _LiTS822Flagship,
):
    """Flagship five-stage curriculum with proposal/backbone updates restored in B1/B2."""

    proposal_schedule = "five_stage"
