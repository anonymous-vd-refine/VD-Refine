from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from math import log, prod, sqrt
import os
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _as_3tuple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    value = tuple(int(i) for i in value)
    if len(value) != 3:
        raise ValueError(f"Expected a 3D tuple, got {value}")
    return value


def _same_padding(kernel_size: int | Sequence[int]) -> tuple[int, int, int]:
    return tuple(i // 2 for i in _as_3tuple(kernel_size))


def _roll_with_zeros(x: torch.Tensor, dim: int, amount: int) -> torch.Tensor:
    if amount == 0:
        return x
    shifted = torch.roll(x, shifts=amount, dims=dim)
    index = [slice(None)] * shifted.ndim
    if amount > 0:
        index[dim] = slice(0, amount)
    else:
        index[dim] = slice(amount, None)
    shifted[tuple(index)] = 0
    return shifted


class ConvNormAct3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 1,
        stride: int | Sequence[int] = 1,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=_as_3tuple(kernel_size),
                stride=_as_3tuple(stride),
                padding=_same_padding(kernel_size),
                groups=groups,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True),
            nn.LeakyReLU(inplace=True) if act else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ECA3D(nn.Module):
    """Small channel attention used only inside lightweight spatial blocks."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pool(x).flatten(2).transpose(1, 2)
        y = self.conv(y).transpose(1, 2).reshape(x.shape[0], x.shape[1], 1, 1, 1)
        return x * self.act(y)


class LiteMBConv3D(nn.Module):
    """
    3D inverted residual block.

    This keeps local spatial modeling cheap: 1x1 expansion, 3x3x3 depthwise
    convolution, light channel attention, then 1x1 projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        expansion: float = 2.0,
        use_eca: bool = True,
    ):
        super().__init__()
        stride = _as_3tuple(stride)
        hidden_channels = max(out_channels, int(round(in_channels * expansion)))
        self.use_residual = stride == (1, 1, 1) and in_channels == out_channels

        self.expand = ConvNormAct3D(in_channels, hidden_channels, kernel_size=1)
        self.depthwise = ConvNormAct3D(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=hidden_channels,
        )
        self.eca = ECA3D(hidden_channels) if use_eca else nn.Identity()
        self.project = ConvNormAct3D(hidden_channels, out_channels, kernel_size=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.eca(self.depthwise(self.expand(x))))
        if self.use_residual:
            y = y + x
        return y


class LiteEncoderStage3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int],
        n_blocks: int = 2,
        expansion: float = 2.0,
    ):
        super().__init__()
        blocks = [
            LiteMBConv3D(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                expansion=expansion,
            )
        ]
        for _ in range(max(0, n_blocks - 1)):
            blocks.append(
                LiteMBConv3D(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    expansion=expansion,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ShiftMLPDWConv3D(nn.Module):
    """
    UNeXt-style shifted MLP with a depthwise 3D refinement.

    It mixes channels by MLP and gives the block a cheap local spatial path,
    which is important for small tumors.
    """

    def __init__(self, channels: int, mlp_ratio: float = 1.0, shift_size: int = 1, use_dwconv: bool = True):
        super().__init__()
        hidden_channels = int(round(channels * mlp_ratio))
        self.channels = channels
        self.shift_size = shift_size
        self.norm = nn.LayerNorm(channels)
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.dwconv = (
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False)
            if use_dwconv
            else nn.Identity()
        )
        self.dw_norm = nn.InstanceNorm3d(hidden_channels, eps=1e-5, affine=True) if use_dwconv else nn.Identity()
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, channels)

    @staticmethod
    def _directions() -> tuple[tuple[int, int], ...]:
        return ((2, 1), (2, -1), (3, 1), (3, -1), (4, 1), (4, -1), (0, 0))

    def _spatial_shift(self, x: torch.Tensor) -> torch.Tensor:
        if self.shift_size == 0:
            return x
        chunks = torch.chunk(x, len(self._directions()), dim=1)
        shifted = []
        for chunk, (dim, direction) in zip(chunks, self._directions()):
            if direction == 0:
                shifted.append(chunk)
            else:
                shifted.append(_roll_with_zeros(chunk, dim, direction * self.shift_size))
        return torch.cat(shifted, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()
        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]
        n_tokens = int(prod(spatial_shape))

        x = self._spatial_shift(x)
        x = x.reshape(b, c, n_tokens).transpose(1, 2)
        x = self.fc1(self.norm(x))
        hidden_channels = x.shape[-1]
        x = x.transpose(1, 2).reshape(b, hidden_channels, *spatial_shape)
        x = self.dw_norm(self.dwconv(x))
        x = x.reshape(b, hidden_channels, n_tokens).transpose(1, 2)
        x = self.fc2(self.act(x))
        return x.transpose(1, 2).reshape(b, c, *spatial_shape)


class ResidualShiftMLPDW3D(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 1.0, shift_size: int = 1, use_dwconv: bool = True):
        super().__init__()
        self.mixer = ShiftMLPDWConv3D(channels, mlp_ratio=mlp_ratio, shift_size=shift_size, use_dwconv=use_dwconv)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip_scale * x + self.mixer(x)


class RecurrentBottleneck3D(nn.Module):
    """Shared-parameter recurrent feature refinement."""

    def __init__(self, block: nn.Module, recurrent_steps: int = 3, eta: float = 0.75):
        super().__init__()
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")
        if not 0 < eta <= 1:
            raise ValueError("eta must be in (0, 1]")
        self.block = block
        self.recurrent_steps = recurrent_steps
        self.eta = eta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x
        for _ in range(self.recurrent_steps):
            refined = self.block(state)
            state = (1.0 - self.eta) * state + self.eta * refined
        return state


class DirectResidualRecurrent3D(nn.Module):
    """Shared recurrent refinement with direct residual accumulation."""

    def __init__(self, block: nn.Module, recurrent_steps: int = 5, randomize_training_steps: bool = False):
        super().__init__()
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")
        self.block = block
        self.recurrent_steps = recurrent_steps
        self.randomize_training_steps = randomize_training_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x
        if self.training and self.randomize_training_steps and "LITE_RECURRENT_TEST_STEPS" not in os.environ:
            recurrent_steps = int(torch.randint(1, self.recurrent_steps + 1, (1,), device=x.device).item())
        else:
            recurrent_steps = int(os.environ.get("LITE_RECURRENT_TEST_STEPS", self.recurrent_steps))
        if recurrent_steps < 1:
            raise ValueError("LITE_RECURRENT_TEST_STEPS must be >= 1")
        residual_csv = os.environ.get("LITE_RECURRENT_RESIDUAL_CSV")
        residual_rows = []
        for step in range(recurrent_steps):
            prev = state
            state = state + self.block(state)
            if residual_csv:
                with torch.no_grad():
                    diff = (state.detach().float() - prev.detach().float()).norm()
                    base = prev.detach().float().norm().clamp_min(1e-8)
                    residual_rows.append(float((diff / base).cpu()))
        if residual_csv and residual_rows:
            parent = os.path.dirname(residual_csv)
            if parent:
                os.makedirs(parent, exist_ok=True)
            write_header = not os.path.exists(residual_csv)
            with open(residual_csv, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("pid,k_test,step,relative_l2\n")
                pid = os.getpid()
                for step, value in enumerate(residual_rows, start=1):
                    f.write(f"{pid},{recurrent_steps},{step},{value:.10f}\n")
        return state


class RelaxationRecurrent3D(nn.Module):
    """Shared recurrent refinement with initial-conditioned relaxation updates."""

    def __init__(
        self,
        block: nn.Module,
        recurrent_steps: int = 5,
        alpha: float = 0.5,
        randomize_training_steps: bool = False,
    ):
        super().__init__()
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.block = block
        self.recurrent_steps = recurrent_steps
        self.alpha = float(alpha)
        self.randomize_training_steps = randomize_training_steps

    def _get_recurrent_steps(self, x: torch.Tensor) -> int:
        if self.training and self.randomize_training_steps and "LITE_RECURRENT_TEST_STEPS" not in os.environ:
            recurrent_steps = int(torch.randint(1, self.recurrent_steps + 1, (1,), device=x.device).item())
        else:
            recurrent_steps = int(os.environ.get("LITE_RECURRENT_TEST_STEPS", self.recurrent_steps))
        if recurrent_steps < 1:
            raise ValueError("LITE_RECURRENT_TEST_STEPS must be >= 1")
        return recurrent_steps

    def forward_steps(self, x: torch.Tensor) -> list[torch.Tensor]:
        state = x
        initial_state = x
        states = []
        for _ in range(self._get_recurrent_steps(x)):
            refined = self.block(state + initial_state)
            state = state + self.alpha * (refined - state)
            states.append(state)
        return states

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_steps(x)[-1]


class MaskBeliefGuidedRecurrent3D(nn.Module):
    """Mask-belief guided recurrent refinement for high-level encoder embeddings."""

    def __init__(
        self,
        base_block: nn.Module,
        channels: int,
        num_classes: int,
        recurrent_steps: int = 5,
        alpha: float = 0.5,
        beta: float = 0.5,
        mlp_ratio: float = 1.0,
        randomize_training_steps: bool = False,
    ):
        super().__init__()
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 < beta <= 1:
            raise ValueError("beta must be in (0, 1]")
        self.base_block = base_block
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.recurrent_steps = int(recurrent_steps)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.randomize_training_steps = bool(randomize_training_steps)
        self.aux_head = nn.Conv3d(self.channels, self.num_classes, kernel_size=1, bias=True)
        self.input_proj = ConvNormAct3D(self.channels * 2 + self.num_classes * 2, self.channels, kernel_size=1)
        self.refine = ResidualShiftMLPDW3D(self.channels, mlp_ratio=mlp_ratio)
        self.last_step_logits: list[torch.Tensor] = []

    def _get_recurrent_steps(self, x: torch.Tensor) -> int:
        if self.training and self.randomize_training_steps and "LITE_RECURRENT_TEST_STEPS" not in os.environ:
            recurrent_steps = int(torch.randint(1, self.recurrent_steps + 1, (1,), device=x.device).item())
        else:
            recurrent_steps = int(os.environ.get("LITE_RECURRENT_TEST_STEPS", self.recurrent_steps))
        if recurrent_steps < 1:
            raise ValueError("LITE_RECURRENT_TEST_STEPS must be >= 1")
        return recurrent_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z0 = self.base_block(x)
        state = z0
        logits = self.aux_head(z0)
        step_logits = [logits]
        for _ in range(self._get_recurrent_steps(x)):
            prob = torch.sigmoid(logits)
            uncertainty = prob * (1.0 - prob)
            guided = torch.cat((state, z0, logits.to(dtype=state.dtype), uncertainty.to(dtype=state.dtype)), dim=1)
            z_hat = self.refine(self.input_proj(guided))
            state = state + self.alpha * (z_hat - state)
            logits_hat = self.aux_head(state)
            logits = logits + self.beta * (logits_hat - logits)
            step_logits.append(logits)
        self.last_step_logits = step_logits
        return state


class SharedDecoderRefinementBlock3D(nn.Module):
    """Shared hidden-space residual correction block for SDRR."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct3D(hidden_channels * 2, hidden_channels, kernel_size=1),
            ConvNormAct3D(hidden_channels, hidden_channels, kernel_size=3, groups=hidden_channels),
            ConvNormAct3D(hidden_channels, hidden_channels, kernel_size=1, act=False),
        )
        self.gate = nn.Sequential(
            nn.Conv3d(hidden_channels * 2, hidden_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        x = torch.cat((h, h0), dim=1)
        return self.gate(x) * self.body(x)


class AlternatingDecoderRefinementBlock3D(nn.Module):
    """Shared correction map used by the A-(B<->C) training schedule."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.pre = ConvNormAct3D(hidden_channels, hidden_channels, kernel_size=1)
        self.spatial = ConvNormAct3D(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            groups=hidden_channels,
        )
        self.output_conv = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.output_conv(self.spatial(self.pre(h)))


class AlternatingDecoderRecurrentRefinement3D(nn.Module):
    """Scale adapter for anchored relaxed fixed-point decoder refinement."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        num_classes: int,
        shared_block: AlternatingDecoderRefinementBlock3D,
        recurrent_steps: int = 1,
        lambda_value: float = 0.5,
        tbptt_keep_steps: int = 0,
    ):
        super().__init__()
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")
        if not 0 < lambda_value <= 1:
            raise ValueError("lambda_value must be in (0, 1]")
        if tbptt_keep_steps < 0:
            raise ValueError("tbptt_keep_steps must be >= 0")

        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.num_classes = int(num_classes)
        self.recurrent_steps = int(recurrent_steps)
        self.lambda_value = float(lambda_value)
        self.tbptt_keep_steps = int(tbptt_keep_steps)
        self.adapter_in = ConvNormAct3D(self.channels, self.hidden_channels, kernel_size=1)
        self.adapter_out = nn.Conv3d(self.hidden_channels, self.channels, kernel_size=1, bias=True)
        nn.init.zeros_(self.adapter_out.bias)
        self.shared_block = shared_block
        self.aux_head = nn.Conv3d(self.hidden_channels, self.num_classes, kernel_size=1, bias=True)
        self.last_aux_logits: list[torch.Tensor] = []
        self.last_hidden_states: list[torch.Tensor] = []
        self.last_h0: torch.Tensor | None = None
        self.last_h: torch.Tensor | None = None
        self.last_tbptt_cut = 0
        self.initial_hidden_noise_std = 0.0
        self.trajectory_hidden_noise_std = 0.0

    def _get_recurrent_steps(self) -> int:
        if "LITE_SDRR_TEST_STEPS" in os.environ:
            steps = int(os.environ["LITE_SDRR_TEST_STEPS"])
        elif "LITE_RECURRENT_TEST_STEPS" in os.environ:
            steps = int(os.environ["LITE_RECURRENT_TEST_STEPS"])
        else:
            steps = self.recurrent_steps
        if steps < 0:
            raise ValueError("recurrent steps must be >= 0")
        return steps

    def _add_relative_hidden_noise(self, h: torch.Tensor, noise_std: float) -> torch.Tensor:
        if noise_std <= 0 or not self.training:
            return h
        scale = h.detach().float().flatten(1).std(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1e-6)
        noise = torch.randn_like(h) * scale.to(dtype=h.dtype) * noise_std
        return h + noise

    def one_step(self, h: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        h = self._add_relative_hidden_noise(h, self.trajectory_hidden_noise_std)
        correction_target = self.shared_block(h)
        return (1.0 - self.lambda_value) * h + self.lambda_value * (h0 + correction_target)

    def forward(
        self,
        d: torch.Tensor,
        recurrent_steps: int | None = None,
        tbptt_keep_steps: int | None = None,
    ) -> torch.Tensor:
        steps = self._get_recurrent_steps() if recurrent_steps is None else int(recurrent_steps)
        if steps < 0:
            raise ValueError("recurrent_steps must be >= 0")

        self.last_aux_logits = []
        self.last_hidden_states = []
        self.last_h0 = None
        self.last_h = None
        self.last_tbptt_cut = 0
        if steps == 0:
            return d

        h0 = self.adapter_in(d)
        h = self._add_relative_hidden_noise(h0, self.initial_hidden_noise_std)
        keep = self.tbptt_keep_steps if tbptt_keep_steps is None else int(tbptt_keep_steps)
        cut = max(0, steps - keep) if self.training and keep > 0 else 0
        self.last_tbptt_cut = cut

        aux_logits: list[torch.Tensor] = []
        hidden_states: list[torch.Tensor] = []
        if cut > 0:
            with torch.no_grad():
                for _ in range(cut):
                    h = self.one_step(h, h0)
                    hidden_states.append(h)
                    aux_logits.append(self.aux_head(h))
            h = h.detach()

        for _ in range(cut, steps):
            h = self.one_step(h, h0)
            hidden_states.append(h)
            aux_logits.append(self.aux_head(h))

        self.last_aux_logits = aux_logits
        self.last_hidden_states = hidden_states
        self.last_h0 = h0
        self.last_h = h
        return d + self.adapter_out(h - h0)


class IndependentDecoderRecurrentRefinement3D(nn.Module):
    """Native-channel recurrent refiner with no shared cross-scale adapter."""

    def __init__(self, channels: int, num_classes: int, depth: int = 2,
                 recurrent_steps: int = 1, lambda_value: float = 0.5,
                 tbptt_keep_steps: int = 0):
        super().__init__()
        if recurrent_steps < 1 or depth < 1:
            raise ValueError("recurrent_steps and depth must be >= 1")
        self.channels = int(channels)
        self.num_classes = int(num_classes)
        self.recurrent_steps = int(recurrent_steps)
        self.lambda_value = float(lambda_value)
        self.tbptt_keep_steps = int(tbptt_keep_steps)
        self.refine = nn.Sequential(
            *[AlternatingDecoderRefinementBlock3D(self.channels) for _ in range(int(depth))]
        )
        self.aux_head = nn.Conv3d(self.channels, self.num_classes, kernel_size=1, bias=True)
        self.last_aux_logits: list[torch.Tensor] = []
        self.last_tbptt_cut = 0

    def _get_recurrent_steps(self) -> int:
        if "LITE_SDRR_TEST_STEPS" in os.environ:
            return int(os.environ["LITE_SDRR_TEST_STEPS"])
        if "LITE_RECURRENT_TEST_STEPS" in os.environ:
            return int(os.environ["LITE_RECURRENT_TEST_STEPS"])
        if self.training and os.environ.get("LITE_SDRR_RANDOMIZE_TRAINING_STEPS", "0") == "1":
            return int(torch.randint(1, self.recurrent_steps + 1, (1,)).item())
        return self.recurrent_steps

    def _one_step(self, h: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.lambda_value) * h + self.lambda_value * (h0 + self.refine(h))

    def forward(self, d: torch.Tensor, recurrent_steps: int | None = None,
                tbptt_keep_steps: int | None = None) -> torch.Tensor:
        steps = self._get_recurrent_steps() if recurrent_steps is None else int(recurrent_steps)
        if steps < 0:
            raise ValueError("recurrent steps must be >= 0")
        self.last_aux_logits = []
        self.last_tbptt_cut = 0
        if steps == 0:
            return d
        h0 = d
        h = h0
        keep = self.tbptt_keep_steps if tbptt_keep_steps is None else int(tbptt_keep_steps)
        cut = max(0, steps - keep) if self.training and keep > 0 else 0
        self.last_tbptt_cut = cut
        if cut > 0:
            with torch.no_grad():
                for _ in range(cut):
                    h = self._one_step(h, h0)
                    # Early states are detached for TBPTT, but their logits still
                    # supervise the shared auxiliary head.
                    self.last_aux_logits.append(self.aux_head(h.detach()))
            h = h.detach()
        for _ in range(cut, steps):
            h = self._one_step(h, h0)
            self.last_aux_logits.append(self.aux_head(h))
        return h


class SharedDecoderRecurrentRefinement3D(nn.Module):
    """Scale-specific SDRR adapter using a shared refinement block."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        num_classes: int,
        shared_block: SharedDecoderRefinementBlock3D,
        recurrent_steps: int = 3,
        alpha: float = 0.5,
        randomize_training_steps: bool = True,
        training_min_steps: int = 1,
        learnable_gamma: bool = False,
        gamma_init: float = 1.0,
        delta_clip_rho: float = 0.0,
        belief_guided: bool = False,
        tbptt_keep_steps: int = 0,
        randomize_tbptt_cut: bool = False,
        alpha_rho: float = 1.0,
    ):
        super().__init__()
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= training_min_steps <= recurrent_steps:
            raise ValueError("training_min_steps must be in [0, recurrent_steps]")
        if delta_clip_rho < 0:
            raise ValueError("delta_clip_rho must be >= 0")
        if tbptt_keep_steps < 0:
            raise ValueError("tbptt_keep_steps must be >= 0")
        if not 0 < alpha_rho <= 1:
            raise ValueError("alpha_rho must be in (0, 1]")
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.num_classes = int(num_classes)
        self.recurrent_steps = int(recurrent_steps)
        self.alpha = float(alpha)
        self.randomize_training_steps = bool(randomize_training_steps)
        self.training_min_steps = int(training_min_steps)
        self.learnable_gamma = bool(learnable_gamma)
        self.delta_clip_rho = float(delta_clip_rho)
        self.belief_guided = bool(belief_guided)
        self.tbptt_keep_steps = int(tbptt_keep_steps)
        self.randomize_tbptt_cut = bool(randomize_tbptt_cut)
        self.alpha_rho = float(alpha_rho)
        self.adapter_in = ConvNormAct3D(self.channels, self.hidden_channels, kernel_size=1)
        self.adapter_out = ConvNormAct3D(self.hidden_channels, self.channels, kernel_size=1, act=False)
        self.shared_block = shared_block
        self.aux_head = nn.Conv3d(self.hidden_channels, self.num_classes, kernel_size=1, bias=True)
        if self.belief_guided:
            self.belief_head = nn.Conv3d(self.hidden_channels, self.num_classes, kernel_size=1, bias=True)
            self.belief_gate = nn.Conv3d(self.num_classes, self.hidden_channels, kernel_size=1, bias=True)
        else:
            self.belief_head = None
            self.belief_gate = None
        if self.learnable_gamma:
            if gamma_init <= 0:
                raise ValueError("gamma_init must be > 0")
            self.correction_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.last_aux_logits: list[torch.Tensor] = []
        self.last_belief_logits: list[torch.Tensor] = []
        self.last_tbptt_cut = 0
        self.last_fixed_point_loss: torch.Tensor | None = None

    def _get_recurrent_steps(self, x: torch.Tensor) -> int:
        if "LITE_SDRR_TEST_STEPS" in os.environ:
            recurrent_steps = int(os.environ["LITE_SDRR_TEST_STEPS"])
        elif "LITE_RECURRENT_TEST_STEPS" in os.environ:
            recurrent_steps = int(os.environ["LITE_RECURRENT_TEST_STEPS"])
        elif self.training and self.randomize_training_steps:
            recurrent_steps = int(
                torch.randint(self.training_min_steps, self.recurrent_steps + 1, (1,), device=x.device).item()
            )
        else:
            recurrent_steps = self.recurrent_steps
        if recurrent_steps < 0:
            raise ValueError("LITE_RECURRENT_TEST_STEPS must be >= 0")
        return recurrent_steps

    def _clip_delta(self, delta: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        if self.delta_clip_rho <= 0:
            return delta
        eps = torch.finfo(delta.dtype).eps
        delta_norm = delta.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1, 1)
        anchor_norm = anchor.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1, 1)
        max_norm = self.delta_clip_rho * anchor_norm
        scale = torch.clamp(max_norm / (delta_norm + eps), max=1.0)
        return delta * scale

    def _get_step_alpha(self, step_idx: int) -> float:
        rho_raw = os.environ.get("LITE_SDRR_ALPHA_RHO")
        rho = self.alpha_rho if rho_raw is None else float(rho_raw)
        if not 0 < rho <= 1:
            raise ValueError("LITE_SDRR_ALPHA_RHO must be in (0, 1]")
        return self.alpha * (rho ** step_idx)

    def _get_tbptt_cut(self, recurrent_steps: int, x: torch.Tensor) -> int:
        if not self.training or recurrent_steps == 0:
            return 0
        if self.randomize_tbptt_cut:
            return int(torch.randint(0, recurrent_steps, (1,), device=x.device).item())
        if self.tbptt_keep_steps > 0:
            return max(0, recurrent_steps - self.tbptt_keep_steps)
        return 0

    @staticmethod
    def _relative_norm_delta(new_tensor: torch.Tensor, old_tensor: torch.Tensor) -> float:
        base = old_tensor.detach().float().norm().clamp_min(1e-8)
        delta = (new_tensor.detach().float() - old_tensor.detach().float()).norm()
        return float((delta / base).cpu())

    def _write_step_metrics(
        self,
        d: torch.Tensor,
        recurrent_steps: int,
        metrics_r_d: list[float],
        metrics_r_m: list[float],
    ) -> None:
        metrics_csv = os.environ.get("LITE_REFINER_METRICS_CSV")
        if not metrics_csv:
            return
        import csv

        metrics_dir = os.path.dirname(metrics_csv)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        row = {
            "module": "sdrr",
            "scale": "x".join(str(int(size)) for size in d.shape[2:]),
            "K": recurrent_steps,
            "n_steps": len(metrics_r_d),
            "r_d_mean": float(sum(metrics_r_d) / len(metrics_r_d)) if metrics_r_d else 0.0,
            "r_m_mean": float(sum(metrics_r_m) / len(metrics_r_m)) if metrics_r_m else 0.0,
            "r_d_steps": ";".join(f"{value:.8g}" for value in metrics_r_d),
            "r_m_steps": ";".join(f"{value:.8g}" for value in metrics_r_m),
        }
        write_header = not os.path.exists(metrics_csv)
        with open(metrics_csv, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        recurrent_steps = self._get_recurrent_steps(d)
        self.last_aux_logits = []
        self.last_belief_logits = []
        self.last_tbptt_cut = 0
        self.last_fixed_point_loss = None
        if recurrent_steps == 0:
            return d

        h0 = self.adapter_in(d)
        h = h0
        aux_logits = []
        metrics_r_d: list[float] = []
        metrics_r_m: list[float] = []
        previous_aux_logits = self.aux_head(h)
        t_cut = self._get_tbptt_cut(recurrent_steps, d)
        self.last_tbptt_cut = t_cut

        for step_idx in range(t_cut):
            with torch.no_grad():
                previous_h = h
                delta = self.shared_block(h, h0)
                if self.belief_guided:
                    if self.belief_head is None or self.belief_gate is None:
                        raise RuntimeError("belief_guided=True but belief modules are missing")
                    belief_logits = self.belief_head(h)
                    gate = torch.sigmoid(self.belief_gate(belief_logits))
                    delta = gate * delta
                    self.last_belief_logits.append(belief_logits)
                delta = self._clip_delta(delta, h0)
                h = h + self._get_step_alpha(step_idx) * delta
                current_aux_logits = self.aux_head(h)
                aux_logits.append(current_aux_logits)
                metrics_r_d.append(self._relative_norm_delta(h, previous_h))
                metrics_r_m.append(
                    self._relative_norm_delta(
                        torch.sigmoid(current_aux_logits),
                        torch.sigmoid(previous_aux_logits),
                    )
                )
                previous_aux_logits = current_aux_logits
        if t_cut > 0:
            h = h.detach()

        for step_idx in range(t_cut, recurrent_steps):
            previous_h = h
            delta = self.shared_block(h, h0)
            if self.belief_guided:
                if self.belief_head is None or self.belief_gate is None:
                    raise RuntimeError("belief_guided=True but belief modules are missing")
                belief_logits = self.belief_head(h)
                gate = torch.sigmoid(self.belief_gate(belief_logits))
                delta = gate * delta
                self.last_belief_logits.append(belief_logits)
            delta = self._clip_delta(delta, h0)
            h = h + self._get_step_alpha(step_idx) * delta
            current_aux_logits = self.aux_head(h)
            aux_logits.append(current_aux_logits)
            metrics_r_d.append(self._relative_norm_delta(h, previous_h))
            metrics_r_m.append(
                self._relative_norm_delta(
                    torch.sigmoid(current_aux_logits),
                    torch.sigmoid(previous_aux_logits),
                )
            )
            previous_aux_logits = current_aux_logits
        self.last_aux_logits = aux_logits
        self._write_step_metrics(d, recurrent_steps, metrics_r_d, metrics_r_m)
        if self.training:
            fp_delta = self.shared_block(h.detach(), h0.detach())
            fp_delta = self._clip_delta(fp_delta, h0.detach())
            fp_norm = fp_delta.float().flatten(1).norm(p=2, dim=1)
            h0_norm = h0.detach().float().flatten(1).norm(p=2, dim=1).clamp_min(1e-8)
            self.last_fixed_point_loss = (fp_norm / h0_norm).mean()
        correction = self.adapter_out(h - h0)
        if self.learnable_gamma:
            correction = self.correction_gamma.to(dtype=correction.dtype) * correction
        return d + correction


class SDRRFeatureRefinerStage3D(nn.Module):
    """
    Feature-refiner adapter with the same lightweight update core as SDRR:
    adapter_in -> gated shared 1x1/DW3x3/1x1 block -> adapter_out.
    """

    def __init__(
        self,
        state_channels: int,
        skip_channels: int,
        hidden_channels: int,
        shared_block: SharedDecoderRefinementBlock3D,
        alpha: float = 0.5,
        alpha_decay: bool = False,
        coarse_context_channels: int | None = None,
        use_checkpoint: bool = False,
        initial_state_drop_prob: float = 0.0,
        update_clip_ratio: float | None = None,
        num_classes: int | None = None,
        belief_guided: bool = False,
        clean_inputs: bool = False,
    ):
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= initial_state_drop_prob <= 1:
            raise ValueError("initial_state_drop_prob must be in [0, 1]")
        if update_clip_ratio is not None and update_clip_ratio <= 0:
            raise ValueError("update_clip_ratio must be > 0 when enabled")
        if belief_guided and num_classes is None:
            raise ValueError("num_classes is required when belief_guided=True")
        self.alpha = float(alpha)
        self.alpha_decay = bool(alpha_decay)
        self.use_checkpoint = bool(use_checkpoint)
        self.initial_state_drop_prob = float(initial_state_drop_prob)
        self.update_clip_ratio = None if update_clip_ratio is None else float(update_clip_ratio)
        self.belief_guided = bool(belief_guided)
        self.num_classes = None if num_classes is None else int(num_classes)
        self.clean_inputs = bool(clean_inputs)
        self.use_coarse_context = coarse_context_channels is not None
        self.skip_proj = ConvNormAct3D(skip_channels, state_channels, kernel_size=1)
        if self.use_coarse_context:
            self.coarse_proj = ConvNormAct3D(int(coarse_context_channels), state_channels, kernel_size=1)
        else:
            self.coarse_proj = None
        if self.clean_inputs:
            input_channels = state_channels * (3 if self.use_coarse_context else 2)
        else:
            input_channels = state_channels * (4 if self.use_coarse_context else 3)
        self.adapter_in = ConvNormAct3D(input_channels, hidden_channels, kernel_size=1)
        self.anchor_adapter_in = ConvNormAct3D(input_channels, hidden_channels, kernel_size=1)
        self.shared_block = shared_block
        self.adapter_out = ConvNormAct3D(hidden_channels, state_channels, kernel_size=1, act=False)
        if self.belief_guided:
            self.belief_head = nn.Conv3d(hidden_channels, self.num_classes, kernel_size=1, bias=True)
            self.belief_gate = nn.Conv3d(self.num_classes, hidden_channels, kernel_size=1, bias=True)
            nn.init.constant_(self.belief_gate.bias, 1.0)
        else:
            self.belief_head = None
            self.belief_gate = None
        self.last_belief_logits: list[torch.Tensor] = []
        # Opt-in for TBPTT policies that truncate only the evolving recurrent state while
        # deliberately keeping d0/skip/coarse anchors connected to the proposal graph. A
        # no-grad prefix normally fills this cache first; without a grad-mode discriminator the
        # grad-enabled tail then reuses those graph-free tensors and silently cuts the anchor
        # gradients too. Keep this False by default so every established trainer retains its
        # exact historical cache behaviour. New policies may set it True on the refiner instance.
        self.preserve_anchor_grad_after_tbptt: bool = False
        self._anchor_cache_key: tuple[int, int, int | None, bool | None] | None = None
        self._anchor_cache: tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor] | None = None

    def _drop_initial_state(self, initial_state: torch.Tensor) -> torch.Tensor:
        if not self.training or self.initial_state_drop_prob <= 0:
            return initial_state
        keep_prob = 1.0 - self.initial_state_drop_prob
        if keep_prob <= 0:
            return torch.zeros_like(initial_state)
        mask_shape = (initial_state.shape[0],) + (1,) * (initial_state.ndim - 1)
        mask = (torch.rand(mask_shape, device=initial_state.device) < keep_prob).to(dtype=initial_state.dtype)
        return initial_state * mask

    def _clip_update(self, update: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.update_clip_ratio is None:
            return update
        update_norm = update.detach().float().flatten(1).norm(dim=1).clamp_min(1e-8)
        state_norm = state.detach().float().flatten(1).norm(dim=1).clamp_min(1e-8)
        max_update_norm = self.update_clip_ratio * state_norm
        scale = (max_update_norm / update_norm).clamp(max=1.0).to(dtype=update.dtype)
        scale = scale.view((update.shape[0],) + (1,) * (update.ndim - 1))
        return update * scale

    def _anchor_features(
        self,
        initial_state: torch.Tensor,
        skip: torch.Tensor,
        coarse_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
        """
        skip_proj/coarse_proj/anchor_adapter_in depend only on the rollout's fixed
        proposal/skip/coarse tensors, never on the evolving recurrent state, so callers
        that loop refine steps (single_refine_index step loops, coarse_to_fine chains,
        etc.) invoke this with identical arguments across steps. Cache by input identity
        instead of redoing these convs on every step; recomputation is still exact
        whenever a caller's coarse_context genuinely changes between steps (e.g. the
        joint D16/D32 refiner feeds each D32 step a fresh D16 output as coarse_context),
        since the identity key then misses and falls through below.
        """
        cacheable = not (self.training and self.initial_state_drop_prob > 0)
        # The default None discriminator intentionally reproduces the old three-input cache
        # equivalence: no-grad and grad-enabled calls share an entry. The explicit opt-in adds
        # the ambient grad mode to the key, so a graph-free TBPTT prefix cannot poison the live
        # tail's d0/skip/coarse anchor paths.
        grad_mode_key = (
            torch.is_grad_enabled() if self.preserve_anchor_grad_after_tbptt else None
        )
        key = (
            (id(initial_state), id(skip), id(coarse_context), grad_mode_key)
            if cacheable
            else None
        )
        if cacheable and key == self._anchor_cache_key and self._anchor_cache is not None:
            return self._anchor_cache

        target_shape = initial_state.shape[2:]
        skip_feat = skip
        if skip_feat.shape[2:] != target_shape:
            skip_feat = F.interpolate(skip_feat, size=target_shape, mode="nearest")
        skip_feat = self.skip_proj(skip_feat)

        dropped_initial = self._drop_initial_state(initial_state)
        if self.clean_inputs:
            anchor_inputs = [dropped_initial, skip_feat]
        else:
            anchor_inputs = [dropped_initial, dropped_initial, skip_feat]

        coarse_feat = None
        if self.use_coarse_context:
            if coarse_context is None:
                raise ValueError("coarse_context is required for this SDRRFeatureRefinerStage3D")
            coarse_feat = coarse_context
            if coarse_feat.shape[2:] != target_shape:
                coarse_feat = F.interpolate(coarse_feat, size=target_shape, mode="nearest")
            coarse_feat = self.coarse_proj(coarse_feat)
            anchor_inputs.append(coarse_feat if self.clean_inputs else coarse_feat.detach())

        h0 = self.anchor_adapter_in(torch.cat(anchor_inputs, dim=1))

        result = (skip_feat, coarse_feat, dropped_initial, h0)
        if cacheable:
            self._anchor_cache_key = key
            self._anchor_cache = result
        return result

    def forward(
        self,
        state: torch.Tensor,
        initial_state: torch.Tensor,
        skip: torch.Tensor,
        coarse_context: torch.Tensor | None = None,
        step_idx: int = 0,
    ) -> torch.Tensor:
        self.last_belief_logits = []
        skip_feat, coarse_feat, dropped_initial, h0 = self._anchor_features(initial_state, skip, coarse_context)

        if self.clean_inputs:
            inputs = [state, skip_feat]
        else:
            inputs = [state, dropped_initial, skip_feat]
        if self.use_coarse_context:
            inputs.append(coarse_feat)
        q = torch.cat(inputs, dim=1)

        def run_refiner(current: torch.Tensor, anchor_hidden: torch.Tensor) -> torch.Tensor:
            h = self.adapter_in(current)
            delta_hidden = self.shared_block(h, anchor_hidden)
            if self.belief_guided:
                if self.belief_head is None or self.belief_gate is None:
                    raise RuntimeError("belief_guided=True but belief modules are missing")
                belief_logits = self.belief_head(h)
                gate = 0.5 + 0.5 * torch.sigmoid(self.belief_gate(belief_logits))
                delta_hidden = gate * delta_hidden
                self.last_belief_logits = [belief_logits]
            return self.adapter_out(delta_hidden)

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            delta = checkpoint(run_refiner, q, h0, use_reentrant=False)
        else:
            delta = run_refiner(q, h0)
        alpha = self.alpha
        if not self.training and "LITE_REFINER_TEST_ALPHA" in os.environ:
            alpha = float(os.environ["LITE_REFINER_TEST_ALPHA"])
        if self.alpha_decay:
            alpha = alpha / sqrt(float(step_idx + 1))
        update = self._clip_update(alpha * delta, state)
        return state + update


class SDRRNormStateFeatureRefinerStage3D(SDRRFeatureRefinerStage3D):
    """
    Same recurrent update as SDRRFeatureRefinerStage3D, but the accumulated state d_t
    is passed through InstanceNorm3d after the residual update:
        d_t = InstanceNorm3d(d_{t-1} + update)
    Opt-in via refiner_style="sdrr_normstate"; SDRRFeatureRefinerStage3D itself is
    untouched so all existing "sdrr"-style trainers are unaffected.
    """

    def __init__(
        self,
        state_channels: int,
        skip_channels: int,
        hidden_channels: int,
        shared_block: SharedDecoderRefinementBlock3D,
        alpha: float = 0.5,
        alpha_decay: bool = False,
        coarse_context_channels: int | None = None,
        use_checkpoint: bool = False,
        initial_state_drop_prob: float = 0.0,
        update_clip_ratio: float | None = None,
        num_classes: int | None = None,
        belief_guided: bool = False,
        clean_inputs: bool = False,
    ):
        super().__init__(
            state_channels=state_channels,
            skip_channels=skip_channels,
            hidden_channels=hidden_channels,
            shared_block=shared_block,
            alpha=alpha,
            alpha_decay=alpha_decay,
            coarse_context_channels=coarse_context_channels,
            use_checkpoint=use_checkpoint,
            initial_state_drop_prob=initial_state_drop_prob,
            update_clip_ratio=update_clip_ratio,
            num_classes=num_classes,
            belief_guided=belief_guided,
            clean_inputs=clean_inputs,
        )
        # affine=True matches every other InstanceNorm3d in this file (see ConvNormAct3D:
        # nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)).
        self.state_norm = nn.InstanceNorm3d(state_channels, eps=1e-5, affine=True)

    def forward(
        self,
        state: torch.Tensor,
        initial_state: torch.Tensor,
        skip: torch.Tensor,
        coarse_context: torch.Tensor | None = None,
        step_idx: int = 0,
    ) -> torch.Tensor:
        self.last_belief_logits = []
        skip_feat, coarse_feat, dropped_initial, h0 = self._anchor_features(initial_state, skip, coarse_context)

        if self.clean_inputs:
            inputs = [state, skip_feat]
        else:
            inputs = [state, dropped_initial, skip_feat]
        if self.use_coarse_context:
            inputs.append(coarse_feat)
        q = torch.cat(inputs, dim=1)

        def run_refiner(current: torch.Tensor, anchor_hidden: torch.Tensor) -> torch.Tensor:
            h = self.adapter_in(current)
            delta_hidden = self.shared_block(h, anchor_hidden)
            if self.belief_guided:
                if self.belief_head is None or self.belief_gate is None:
                    raise RuntimeError("belief_guided=True but belief modules are missing")
                belief_logits = self.belief_head(h)
                gate = 0.5 + 0.5 * torch.sigmoid(self.belief_gate(belief_logits))
                delta_hidden = gate * delta_hidden
                self.last_belief_logits = [belief_logits]
            return self.adapter_out(delta_hidden)

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            delta = checkpoint(run_refiner, q, h0, use_reentrant=False)
        else:
            delta = run_refiner(q, h0)
        alpha = self.alpha
        if not self.training and "LITE_REFINER_TEST_ALPHA" in os.environ:
            alpha = float(os.environ["LITE_REFINER_TEST_ALPHA"])
        if self.alpha_decay:
            alpha = alpha / sqrt(float(step_idx + 1))
        update = self._clip_update(alpha * delta, state)
        return self.state_norm(state + update)


class SDRRAnchorAdaINFeatureRefinerStage3D(SDRRFeatureRefinerStage3D):
    """
    Same recurrent update as SDRRFeatureRefinerStage3D, but the accumulated state borrows
    d_0's own per-channel mean/std (AdaIN-style) instead of accumulating unboundedly:
        m = InstanceNorm3d_learnable(d_0)
        raw = state + alpha * delta
        d_t = std(m) * (raw - mean(raw)) / std(raw) + mean(m)
    Bakes the norm-clamp ablation's empirically-confirmed fix (see EXPERIMENTS.md Mechanism
    Understanding #4) into the architecture as a learned step, conditioned on the anchor d_0,
    instead of relying on it happening for free several stages downstream (BaseProposalStage3D's
    InstanceNorm3d). Opt-in via refiner_style="sdrr_anchor_adain"; SDRRFeatureRefinerStage3D
    itself is untouched so all existing "sdrr"-style trainers are unaffected.
    """

    def __init__(
        self,
        state_channels: int,
        skip_channels: int,
        hidden_channels: int,
        shared_block: SharedDecoderRefinementBlock3D,
        alpha: float = 0.5,
        alpha_decay: bool = False,
        coarse_context_channels: int | None = None,
        use_checkpoint: bool = False,
        initial_state_drop_prob: float = 0.0,
        update_clip_ratio: float | None = None,
        num_classes: int | None = None,
        belief_guided: bool = False,
        clean_inputs: bool = False,
    ):
        super().__init__(
            state_channels=state_channels,
            skip_channels=skip_channels,
            hidden_channels=hidden_channels,
            shared_block=shared_block,
            alpha=alpha,
            alpha_decay=alpha_decay,
            coarse_context_channels=coarse_context_channels,
            use_checkpoint=use_checkpoint,
            initial_state_drop_prob=initial_state_drop_prob,
            update_clip_ratio=update_clip_ratio,
            num_classes=num_classes,
            belief_guided=belief_guided,
            clean_inputs=clean_inputs,
        )
        # affine=True matches every other InstanceNorm3d in this file (see ConvNormAct3D:
        # nn.InstanceNorm3d(out_channels, eps=1e-5, affine=True)).
        self.anchor_norm = nn.InstanceNorm3d(state_channels, eps=1e-5, affine=True)
        self._adain_eps = 1e-5

    def forward(
        self,
        state: torch.Tensor,
        initial_state: torch.Tensor,
        skip: torch.Tensor,
        coarse_context: torch.Tensor | None = None,
        step_idx: int = 0,
    ) -> torch.Tensor:
        self.last_belief_logits = []
        skip_feat, coarse_feat, dropped_initial, h0 = self._anchor_features(initial_state, skip, coarse_context)

        if self.clean_inputs:
            inputs = [state, skip_feat]
        else:
            inputs = [state, dropped_initial, skip_feat]
        if self.use_coarse_context:
            inputs.append(coarse_feat)
        q = torch.cat(inputs, dim=1)

        def run_refiner(current: torch.Tensor, anchor_hidden: torch.Tensor) -> torch.Tensor:
            h = self.adapter_in(current)
            delta_hidden = self.shared_block(h, anchor_hidden)
            if self.belief_guided:
                if self.belief_head is None or self.belief_gate is None:
                    raise RuntimeError("belief_guided=True but belief modules are missing")
                belief_logits = self.belief_head(h)
                gate = 0.5 + 0.5 * torch.sigmoid(self.belief_gate(belief_logits))
                delta_hidden = gate * delta_hidden
                self.last_belief_logits = [belief_logits]
            return self.adapter_out(delta_hidden)

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            delta = checkpoint(run_refiner, q, h0, use_reentrant=False)
        else:
            delta = run_refiner(q, h0)
        alpha = self.alpha
        if not self.training and "LITE_REFINER_TEST_ALPHA" in os.environ:
            alpha = float(os.environ["LITE_REFINER_TEST_ALPHA"])
        if self.alpha_decay:
            alpha = alpha / sqrt(float(step_idx + 1))
        update = self._clip_update(alpha * delta, state)
        raw = state + update

        m = self.anchor_norm(initial_state)
        spatial_dims = tuple(range(2, raw.ndim))
        raw_mean = raw.mean(dim=spatial_dims, keepdim=True)
        raw_std = raw.std(dim=spatial_dims, keepdim=True).clamp_min(self._adain_eps)
        m_mean = m.mean(dim=spatial_dims, keepdim=True)
        m_std = m.std(dim=spatial_dims, keepdim=True)
        return m_std * (raw - raw_mean) / raw_std + m_mean


class SDRRFiLMFeatureRefinerStage3D(SDRRFeatureRefinerStage3D):
    """
    Same recurrent update as SDRRFeatureRefinerStage3D, but the accumulated state is
    re-normalized with a FiLM-style learned affine conditioned on d_0, instead of literally
    borrowing d_0's own per-channel mean/std (see SDRRAnchorAdaINFeatureRefinerStage3D):
        raw = state + alpha * delta                                    (unchanged SDRR update)
        gamma, beta = film_head(global_avg_pool3d(d_0))                 (learned functions of
                                                                          d_0, NOT d_0's own
                                                                          borrowed statistics)
        d_t = gamma * InstanceNorm3d_no_affine(raw) + beta
    More expressive than the AdaIN variant: the target per-channel scale/shift are free
    parameters the network learns to condition on d_0, not forced to equal whatever d_0's own
    instance-norm statistics happen to be every step. film_head is zero-initialized so
    gamma=1, beta=0 at the start of training (identity transform, i.e. plain InstanceNorm3d
    with no affine) -- training then learns how much and in what direction to deviate from
    that, rather than starting from an arbitrary random affine. Opt-in via
    refiner_style="sdrr_film"; SDRRFeatureRefinerStage3D itself is untouched so all existing
    "sdrr"-style trainers are unaffected.
    """

    def __init__(
        self,
        state_channels: int,
        skip_channels: int,
        hidden_channels: int,
        shared_block: SharedDecoderRefinementBlock3D,
        alpha: float = 0.5,
        alpha_decay: bool = False,
        coarse_context_channels: int | None = None,
        use_checkpoint: bool = False,
        initial_state_drop_prob: float = 0.0,
        update_clip_ratio: float | None = None,
        num_classes: int | None = None,
        belief_guided: bool = False,
        clean_inputs: bool = False,
    ):
        super().__init__(
            state_channels=state_channels,
            skip_channels=skip_channels,
            hidden_channels=hidden_channels,
            shared_block=shared_block,
            alpha=alpha,
            alpha_decay=alpha_decay,
            coarse_context_channels=coarse_context_channels,
            use_checkpoint=use_checkpoint,
            initial_state_drop_prob=initial_state_drop_prob,
            update_clip_ratio=update_clip_ratio,
            num_classes=num_classes,
            belief_guided=belief_guided,
            clean_inputs=clean_inputs,
        )
        self.raw_norm = nn.InstanceNorm3d(state_channels, eps=1e-5, affine=False)
        self.film_pool = nn.AdaptiveAvgPool3d(1)
        self.film_head = nn.Conv3d(state_channels, state_channels * 2, kernel_size=1, bias=True)
        # Zero-init -> gamma=1+0=1, beta=0 at step 0: identity transform (plain InstanceNorm3d,
        # no affine), so training starts from the same behavior as an unconditioned norm and
        # only learns to deviate from it where useful.
        nn.init.zeros_(self.film_head.weight)
        nn.init.zeros_(self.film_head.bias)

    def _film_params(self, initial_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.film_pool(initial_state)
        gamma_raw, beta = self.film_head(pooled).chunk(2, dim=1)
        return 1.0 + gamma_raw, beta

    def forward(
        self,
        state: torch.Tensor,
        initial_state: torch.Tensor,
        skip: torch.Tensor,
        coarse_context: torch.Tensor | None = None,
        step_idx: int = 0,
    ) -> torch.Tensor:
        self.last_belief_logits = []
        skip_feat, coarse_feat, dropped_initial, h0 = self._anchor_features(initial_state, skip, coarse_context)

        if self.clean_inputs:
            inputs = [state, skip_feat]
        else:
            inputs = [state, dropped_initial, skip_feat]
        if self.use_coarse_context:
            inputs.append(coarse_feat)
        q = torch.cat(inputs, dim=1)

        def run_refiner(current: torch.Tensor, anchor_hidden: torch.Tensor) -> torch.Tensor:
            h = self.adapter_in(current)
            delta_hidden = self.shared_block(h, anchor_hidden)
            if self.belief_guided:
                if self.belief_head is None or self.belief_gate is None:
                    raise RuntimeError("belief_guided=True but belief modules are missing")
                belief_logits = self.belief_head(h)
                gate = 0.5 + 0.5 * torch.sigmoid(self.belief_gate(belief_logits))
                delta_hidden = gate * delta_hidden
                self.last_belief_logits = [belief_logits]
            return self.adapter_out(delta_hidden)

        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            delta = checkpoint(run_refiner, q, h0, use_reentrant=False)
        else:
            delta = run_refiner(q, h0)
        alpha = self.alpha
        if not self.training and "LITE_REFINER_TEST_ALPHA" in os.environ:
            alpha = float(os.environ["LITE_REFINER_TEST_ALPHA"])
        if self.alpha_decay:
            alpha = alpha / sqrt(float(step_idx + 1))
        update = self._clip_update(alpha * delta, state)
        raw = state + update

        gamma, beta = self._film_params(initial_state)
        return gamma * self.raw_norm(raw) + beta


class GatedResidualRecurrent3D(nn.Module):
    """Shared recurrent refinement with learnable residual update gates."""

    def __init__(
        self,
        block: nn.Module,
        recurrent_steps: int = 5,
        gate_init: float = 0.1,
        per_step_gate: bool = True,
    ):
        super().__init__()
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")
        if not 0 < gate_init < 1:
            raise ValueError("gate_init must be in (0, 1)")
        self.block = block
        self.recurrent_steps = recurrent_steps
        gate_shape = (recurrent_steps,) if per_step_gate else (1,)
        gate_logit = log(gate_init / (1.0 - gate_init))
        self.gate_logits = nn.Parameter(torch.full(gate_shape, gate_logit, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x
        recurrent_steps = int(os.environ.get("LITE_RECURRENT_TEST_STEPS", self.recurrent_steps))
        if recurrent_steps < 1:
            raise ValueError("LITE_RECURRENT_TEST_STEPS must be >= 1")
        for step in range(recurrent_steps):
            refined = self.block(state)
            gate_idx = min(step, self.gate_logits.numel() - 1) if self.gate_logits.numel() > 1 else 0
            alpha = torch.sigmoid(self.gate_logits[gate_idx]).to(dtype=state.dtype)
            state = state + alpha * (refined - state)
        return state


class ConditionalGatedResidualRecurrent3D(nn.Module):
    """Gated recurrent refinement with encoder bottleneck feature conditioning."""

    def __init__(
        self,
        block: nn.Module,
        recurrent_steps: int = 5,
        gate_init: float = 0.1,
        condition_init: float = 0.1,
        per_step_gate: bool = True,
        init_with_block: bool = False,
    ):
        super().__init__()
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")
        if not 0 < gate_init < 1:
            raise ValueError("gate_init must be in (0, 1)")
        self.block = block
        self.recurrent_steps = recurrent_steps
        gate_shape = (recurrent_steps,) if per_step_gate else (1,)
        gate_logit = log(gate_init / (1.0 - gate_init))
        self.gate_logits = nn.Parameter(torch.full(gate_shape, gate_logit, dtype=torch.float32))
        self.condition_scale = nn.Parameter(torch.tensor(float(condition_init), dtype=torch.float32))
        self.init_with_block = init_with_block

    def forward(self, x: torch.Tensor, condition: torch.Tensor | None = None) -> torch.Tensor:
        if condition is None:
            raise ValueError("Conditional recurrent refinement requires a condition tensor")
        state = self.block(x) if self.init_with_block else x
        condition = condition.to(dtype=state.dtype)
        beta = self.condition_scale.to(dtype=state.dtype)
        recurrent_steps = int(os.environ.get("LITE_RECURRENT_TEST_STEPS", self.recurrent_steps))
        if recurrent_steps < 1:
            raise ValueError("LITE_RECURRENT_TEST_STEPS must be >= 1")
        for step in range(recurrent_steps):
            refined = self.block(state + beta * condition)
            gate_idx = min(step, self.gate_logits.numel() - 1) if self.gate_logits.numel() > 1 else 0
            alpha = torch.sigmoid(self.gate_logits[gate_idx]).to(dtype=state.dtype)
            state = state + alpha * (refined - state)
        return state


class UnsharedRecurrentBottleneck3D(nn.Module):
    """Unshared recurrent feature refinement for depth-vs-sharing controls."""

    def __init__(self, blocks: Sequence[nn.Module], eta: float = 0.75):
        super().__init__()
        if not blocks:
            raise ValueError("blocks must not be empty")
        if not 0 < eta <= 1:
            raise ValueError("eta must be in (0, 1]")
        self.blocks = nn.ModuleList(blocks)
        self.eta = eta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = x
        for block in self.blocks:
            refined = block(state)
            state = (1.0 - self.eta) * state + self.eta * refined
        return state


class MambaBottleneck3D(nn.Module):
    """UMambaBot-style bottleneck Mamba mixer for low-resolution 3D features."""

    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        from mamba_ssm import Mamba

        self.channels = channels
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float16:
            x = x.float()
        b, c = x.shape[:2]
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}")
        spatial_shape = x.shape[2:]
        n_tokens = int(prod(spatial_shape))
        x_flat = x.reshape(b, c, n_tokens).transpose(-1, -2)
        x_mamba = self.mamba(self.norm(x_flat))
        return x_mamba.transpose(-1, -2).reshape(b, c, *spatial_shape)


class LiteDecoderStage3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        kernel_size: int | Sequence[int],
        n_blocks: int = 1,
        expansion: float = 2.0,
        refine_recurrent_steps: int = 1,
        randomize_refine_training_steps: bool = False,
        recurrent_steps: int = 0,
        eta: float = 0.75,
        mlp_ratio: float = 1.0,
    ):
        super().__init__()
        if refine_recurrent_steps < 1:
            raise ValueError("refine_recurrent_steps must be >= 1")
        self.up_proj = ConvNormAct3D(in_channels, skip_channels, kernel_size=1)
        self.fuse = ConvNormAct3D(skip_channels * 2, skip_channels, kernel_size=1)
        self.refine_recurrent_steps = refine_recurrent_steps
        self.eta = eta
        blocks = []
        for _ in range(max(1, n_blocks)):
            blocks.append(
                LiteMBConv3D(
                    skip_channels,
                    skip_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    expansion=expansion,
                )
            )
        self.refine = nn.Sequential(*blocks)
        self.randomize_refine_training_steps = randomize_refine_training_steps
        if recurrent_steps > 0:
            recurrent_block = ResidualShiftMLPDW3D(skip_channels, mlp_ratio=mlp_ratio)
            self.recurrent_refine = RecurrentBottleneck3D(recurrent_block, recurrent_steps, eta)
        else:
            self.recurrent_refine = nn.Identity()

    def fuse_skip(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = self.up_proj(x)
        x = torch.cat((x, skip), dim=1)
        return self.fuse(x)

    def refine_fused(self, x: torch.Tensor) -> torch.Tensor:
        x = self.refine(x)
        if (
            self.training
            and self.randomize_refine_training_steps
            and "LITE_RECURRENT_TEST_STEPS" not in os.environ
        ):
            refine_steps = int(torch.randint(1, self.refine_recurrent_steps + 1, (1,), device=x.device).item())
        else:
            refine_steps = int(os.environ.get("LITE_RECURRENT_TEST_STEPS", self.refine_recurrent_steps))
            if refine_steps < 1 and "LITE_RECURRENT_TEST_STEPS" in os.environ:
                refine_steps = 1
        if refine_steps < 1:
            raise ValueError("LITE_RECURRENT_TEST_STEPS must be >= 1")
        for _ in range(1, refine_steps):
            refined = self.refine(x)
            x = (1.0 - self.eta) * x + self.eta * refined
        return self.recurrent_refine(x)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.refine_fused(self.fuse_skip(x, skip))


class SimpleFuseStage3D(nn.Module):
    """Minimal decoder stage for proposal building and lightweight mask decoding."""

    def __init__(self, in_channels: int, skip_channels: int):
        super().__init__()
        self.up_proj = ConvNormAct3D(in_channels, skip_channels, kernel_size=1)
        self.fuse = ConvNormAct3D(skip_channels * 2, skip_channels, kernel_size=1)
        self.refine = ConvNormAct3D(skip_channels, skip_channels, kernel_size=3)

    def fuse_skip(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = self.up_proj(x)
        x = torch.cat((x, skip), dim=1)
        return self.fuse(x)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.refine(self.fuse_skip(x, skip))


class BaseProposalStage3D(nn.Module):
    """Strict proposal stage: nearest-neighbor upsample + skip concat + 1x1x1 fusion."""

    def __init__(self, in_channels: int, skip_channels: int):
        super().__init__()
        self.up_proj = ConvNormAct3D(in_channels, skip_channels, kernel_size=1)
        self.fuse = ConvNormAct3D(skip_channels * 2, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
        x = self.up_proj(x)
        x = torch.cat((x, skip), dim=1)
        return self.fuse(x)


class SharedFeatureRefinementBlock3D(nn.Module):
    """Shared f_theta block used by all decoder scales and all recurrent steps."""

    def __init__(self, hidden_channels: int, expansion: float = 2.0):
        super().__init__()
        self.block = LiteMBConv3D(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            expansion=expansion,
            use_eca=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FeatureRefinerStage3D(nn.Module):
    """
    Scale-specific adapter around a shared feature-space recurrent refiner.

    With coarse context enabled the input is:
    [d_f^t, M d_f^0, skip_f, Up(d_c - d_c^0)].
    """

    def __init__(
        self,
        state_channels: int,
        skip_channels: int,
        hidden_channels: int,
        shared_block: SharedFeatureRefinementBlock3D,
        alpha: float = 0.5,
        alpha_decay: bool = False,
        coarse_context_channels: int | None = None,
        use_checkpoint: bool = False,
        initial_state_drop_prob: float = 0.0,
        update_clip_ratio: float | None = None,
    ):
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if not 0 <= initial_state_drop_prob <= 1:
            raise ValueError("initial_state_drop_prob must be in [0, 1]")
        if update_clip_ratio is not None and update_clip_ratio <= 0:
            raise ValueError("update_clip_ratio must be > 0 when enabled")
        self.alpha = float(alpha)
        self.alpha_decay = bool(alpha_decay)
        self.use_checkpoint = bool(use_checkpoint)
        self.initial_state_drop_prob = float(initial_state_drop_prob)
        self.update_clip_ratio = None if update_clip_ratio is None else float(update_clip_ratio)
        self.use_coarse_context = coarse_context_channels is not None
        self.skip_proj = ConvNormAct3D(skip_channels, state_channels, kernel_size=1)
        if self.use_coarse_context:
            self.coarse_proj = ConvNormAct3D(int(coarse_context_channels), state_channels, kernel_size=1)
        else:
            self.coarse_proj = None
        input_channels = state_channels * (4 if self.use_coarse_context else 3)
        self.input_proj = ConvNormAct3D(input_channels, hidden_channels, kernel_size=1)
        self.shared_block = shared_block
        self.output_proj = ConvNormAct3D(hidden_channels, state_channels, kernel_size=1, act=False)

    def _drop_initial_state(self, initial_state: torch.Tensor) -> torch.Tensor:
        if not self.training or self.initial_state_drop_prob <= 0:
            return initial_state
        keep_prob = 1.0 - self.initial_state_drop_prob
        if keep_prob <= 0:
            return torch.zeros_like(initial_state)
        mask_shape = (initial_state.shape[0],) + (1,) * (initial_state.ndim - 1)
        mask = (torch.rand(mask_shape, device=initial_state.device) < keep_prob).to(dtype=initial_state.dtype)
        return initial_state * mask

    def _clip_update(self, update: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.update_clip_ratio is None:
            return update
        update_norm = update.detach().float().flatten(1).norm(dim=1).clamp_min(1e-8)
        state_norm = state.detach().float().flatten(1).norm(dim=1).clamp_min(1e-8)
        max_update_norm = self.update_clip_ratio * state_norm
        scale = (max_update_norm / update_norm).clamp(max=1.0).to(dtype=update.dtype)
        scale = scale.view((update.shape[0],) + (1,) * (update.ndim - 1))
        return update * scale

    def forward(
        self,
        state: torch.Tensor,
        initial_state: torch.Tensor,
        skip: torch.Tensor,
        coarse_context: torch.Tensor | None = None,
        step_idx: int = 0,
    ) -> torch.Tensor:
        if skip.shape[2:] != state.shape[2:]:
            skip = F.interpolate(skip, size=state.shape[2:], mode="nearest")
        skip = self.skip_proj(skip)
        initial_state = self._drop_initial_state(initial_state)
        inputs = [state, initial_state, skip]
        if self.use_coarse_context:
            if coarse_context is None:
                raise ValueError("coarse_context is required for this FeatureRefinerStage3D")
            if coarse_context.shape[2:] != state.shape[2:]:
                coarse_context = F.interpolate(coarse_context, size=state.shape[2:], mode="nearest")
            inputs.append(self.coarse_proj(coarse_context))
        q = torch.cat(inputs, dim=1)
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            delta = checkpoint(
                lambda y: self.output_proj(self.shared_block(self.input_proj(y))),
                q,
                use_reentrant=False,
            )
        else:
            delta = self.output_proj(self.shared_block(self.input_proj(q)))
        alpha = self.alpha
        if not self.training and "LITE_REFINER_TEST_ALPHA" in os.environ:
            alpha = float(os.environ["LITE_REFINER_TEST_ALPHA"])
        if self.alpha_decay:
            alpha = alpha / sqrt(float(step_idx + 1))
        update = self._clip_update(alpha * delta, state)
        return state + update


class SharedHierarchicalRefinerBlock3D(nn.Module):
    """Shared recurrent update block reused across 8^3, 16^3, and 32^3 states."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.body = nn.Sequential(
            ConvNormAct3D(hidden_channels, hidden_channels, kernel_size=1),
            ConvNormAct3D(hidden_channels, hidden_channels, kernel_size=3, groups=hidden_channels),
            ConvNormAct3D(hidden_channels, hidden_channels, kernel_size=1, act=False),
        )
        self.gate = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x) * self.body(x)


class HierarchicalRefinerStage3D(nn.Module):
    """Scale-specific adapter that feeds a shared recurrent update block."""

    def __init__(
        self,
        state_channels: int,
        num_classes: int,
        hidden_channels: int,
        shared_block: SharedHierarchicalRefinerBlock3D,
        alpha: float = 0.5,
    ):
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        in_channels = state_channels * 3 + num_classes * 2
        self.state_channels = int(state_channels)
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.input_proj = ConvNormAct3D(in_channels, hidden_channels, kernel_size=1)
        self.shared_block = shared_block
        self.output_proj = ConvNormAct3D(hidden_channels, state_channels, kernel_size=1, act=False)
        self.last_relative_delta: float = 0.0
        self.last_relative_residual: float = 0.0

    def forward(
        self,
        state: torch.Tensor,
        initial_state: torch.Tensor,
        guide: torch.Tensor,
        prob: torch.Tensor,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        prob = F.interpolate(prob.to(dtype=state.dtype), size=state.shape[2:], mode="trilinear", align_corners=False)
        uncertainty = F.interpolate(
            uncertainty.to(dtype=state.dtype), size=state.shape[2:], mode="trilinear", align_corners=False
        )
        hidden = self.input_proj(torch.cat((state, initial_state, guide, prob, uncertainty), dim=1))
        residual = self.output_proj(self.shared_block(hidden))
        updated = state + self.alpha * residual
        with torch.no_grad():
            base = state.detach().float().norm().clamp_min(1e-8)
            self.last_relative_delta = float(((updated.detach().float() - state.detach().float()).norm() / base).cpu())
            self.last_relative_residual = float((residual.detach().float().norm() / base).cpu())
        return updated


class ProposalMaskHead3D(nn.Module):
    """Shared head that upsamples the 32^3 recurrent state to full-resolution logits."""

    def __init__(self, channels32: int, channels64: int, channels128: int, num_classes: int):
        super().__init__()
        self.stage64 = SimpleFuseStage3D(channels32, channels64)
        self.stage128 = SimpleFuseStage3D(channels64, channels128)
        self.seg = nn.Conv3d(channels128, num_classes, kernel_size=1, bias=True)

    def forward(self, state32: torch.Tensor, skip64: torch.Tensor, skip128: torch.Tensor, output_size) -> torch.Tensor:
        x = self.stage64(state32, skip64)
        x = self.stage128(x, skip128)
        if x.shape[2:] != tuple(output_size):
            x = F.interpolate(x, size=output_size, mode="nearest")
        return self.seg(x)


class LiteRBUNeXt3DTwoBranchRefiner(nn.Module):
    """
    Encoder + simple proposal decoder + shared-step hierarchical recurrent refiner.

    Proposal states are built at 8^3, 16^3, and 32^3. The recurrent branch then
    refines them sequentially at each step, from coarse to fine.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int | Sequence[int]],
        strides: Sequence[int | Sequence[int]],
        n_blocks_per_stage: Sequence[int],
        *,
        hidden_channels: int = 96,
        recurrent_steps: int = 3,
        alpha: float = 0.5,
        randomize_training_steps: bool = True,
        training_min_steps: int = 0,
        expansion: float = 2.0,
        deep_supervision: bool = False,
    ):
        super().__init__()
        if len(channels) < 6:
            raise ValueError("LiteRBUNeXt3DTwoBranchRefiner expects at least 6 stages for 8/16/32 recurrent states")
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")
        if not 0 <= training_min_steps <= recurrent_steps:
            raise ValueError("training_min_steps must be in [0, recurrent_steps]")

        self.num_classes = int(num_classes)
        self.recurrent_steps = int(recurrent_steps)
        self.randomize_training_steps = bool(randomize_training_steps)
        self.training_min_steps = int(training_min_steps)
        self.deep_supervision = bool(deep_supervision)

        encoder_stages = []
        in_ch = input_channels
        for stage_idx, out_ch in enumerate(channels):
            encoder_stages.append(
                LiteEncoderStage3D(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_sizes[stage_idx],
                    stride=strides[stage_idx],
                    n_blocks=max(1, int(n_blocks_per_stage[stage_idx])),
                    expansion=expansion,
                )
            )
            in_ch = out_ch
        self.encoder_stages = nn.ModuleList(encoder_stages)
        self.mixers = nn.ModuleList(
            [nn.Identity() for _ in range(len(channels) - 1)] + [ResidualShiftMLPDW3D(channels[-1], mlp_ratio=1.0)]
        )

        c128, c64, c32, c16, c8, c4 = channels[0], channels[1], channels[2], channels[3], channels[4], channels[5]
        self.proposal8 = SimpleFuseStage3D(c4, c8)
        self.proposal16 = SimpleFuseStage3D(c8, c16)
        self.proposal32 = SimpleFuseStage3D(c16, c32)

        self.guide16 = SimpleFuseStage3D(c8, c16)
        self.guide32 = SimpleFuseStage3D(c16, c32)

        shared_refiner = SharedHierarchicalRefinerBlock3D(hidden_channels)
        self.refiner8 = HierarchicalRefinerStage3D(c8, num_classes, hidden_channels, shared_refiner, alpha=alpha)
        self.refiner16 = HierarchicalRefinerStage3D(c16, num_classes, hidden_channels, shared_refiner, alpha=alpha)
        self.refiner32 = HierarchicalRefinerStage3D(c32, num_classes, hidden_channels, shared_refiner, alpha=alpha)

        self.mask_head = ProposalMaskHead3D(c32, c64, c128, num_classes)
        self.step_head32 = nn.Conv3d(c32, num_classes, kernel_size=1, bias=True)
        self.last_step_logits: list[torch.Tensor] = []
        self.last_step_lowres_logits: list[torch.Tensor] = []
        self.last_step_metrics: dict[str, list[float]] = {"r_d8": [], "r_d16": [], "r_d32": [], "r_f8": [], "r_f16": [], "r_f32": []}
        self.decoder = _DeepSupervisionProxy(self)

    def _get_recurrent_steps(self, x: torch.Tensor) -> int:
        if "LITE_REFINER_TEST_STEPS" in os.environ:
            recurrent_steps = int(os.environ["LITE_REFINER_TEST_STEPS"])
        elif "LITE_SDRR_TEST_STEPS" in os.environ:
            recurrent_steps = int(os.environ["LITE_SDRR_TEST_STEPS"])
        elif "LITE_RECURRENT_TEST_STEPS" in os.environ:
            recurrent_steps = int(os.environ["LITE_RECURRENT_TEST_STEPS"])
        elif self.training and self.randomize_training_steps:
            recurrent_steps = int(
                torch.randint(self.training_min_steps, self.recurrent_steps + 1, (1,), device=x.device).item()
            )
        else:
            recurrent_steps = self.recurrent_steps
        if recurrent_steps < 0:
            raise ValueError("test recurrent steps must be >= 0")
        return recurrent_steps

    def forward(self, x: torch.Tensor):
        input_shape = x.shape[2:]
        skips = []
        for stage, mixer in zip(self.encoder_stages, self.mixers):
            raw = stage(x)
            x = mixer(raw)
            skips.append(x)

        skip128, skip64, skip32, skip16, skip8, bottleneck = skips
        d8_0 = self.proposal8(bottleneck, skip8)
        d16_0 = self.proposal16(d8_0, skip16)
        d32_0 = self.proposal32(d16_0, skip32)

        d8 = d8_0
        d16 = d16_0
        d32 = d32_0
        lowres_logits = self.step_head32(d32)
        step_lowres_logits = [lowres_logits]
        metrics = {"r_d8": [], "r_d16": [], "r_d32": [], "r_f8": [], "r_f16": [], "r_f32": []}

        for _ in range(self._get_recurrent_steps(x)):
            prob = torch.sigmoid(lowres_logits)
            uncertainty = prob * (1.0 - prob)

            d8 = self.refiner8(d8, d8_0, d8_0, prob, uncertainty)
            metrics["r_d8"].append(self.refiner8.last_relative_delta)
            metrics["r_f8"].append(self.refiner8.last_relative_residual)

            g16 = self.guide16(d8, skip16)
            d16 = self.refiner16(d16, d16_0, g16, prob, uncertainty)
            metrics["r_d16"].append(self.refiner16.last_relative_delta)
            metrics["r_f16"].append(self.refiner16.last_relative_residual)

            g32 = self.guide32(d16, skip32)
            d32 = self.refiner32(d32, d32_0, g32, prob, uncertainty)
            metrics["r_d32"].append(self.refiner32.last_relative_delta)
            metrics["r_f32"].append(self.refiner32.last_relative_residual)

            lowres_logits = self.step_head32(d32)
            step_lowres_logits.append(lowres_logits)

        logits = self.mask_head(d32, skip64, skip128, input_shape)
        self.last_step_logits = [logits]
        self.last_step_lowres_logits = step_lowres_logits
        self.last_step_metrics = metrics
        if self.deep_supervision:
            return [logits]
        return logits


class LiteRBUNeXt3DFeatureRefiner(nn.Module):
    """
    Teacher-specified idea1 model:
    encoder -> lightweight proposal decoder -> shared feature-space recurrent refiner -> mask head.

    The base decoder emits all decoder embeddings. Each recurrent step refines every decoder
    scale with the same shared f_theta through scale-specific input/output projections.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int | Sequence[int]],
        strides: Sequence[int | Sequence[int]],
        n_blocks_per_stage: Sequence[int],
        *,
        hidden_channels: int = 32,
        recurrent_steps: int = 4,
        alpha: float = 0.5,
        alpha_decay: bool = False,
        randomize_training_steps: bool = True,
        training_min_steps: int = 1,
        tbptt_keep_steps: int = 2,
        refine_scales: str = "all",
        single_refine_index: int = 1,
        single_coarse_context_index: int | None = None,
        c2f_start_index: int = 0,
        normalize_finest_before_head: bool = False,
        normalize_finest_anchor_to_proposal: bool = False,
        checkpoint_refiner: bool = False,
        initial_state_drop_prob: float = 0.0,
        update_clip_ratio: float | None = None,
        refiner_style: str = "mbconv",
        belief_guided: bool = False,
        logit_refine: bool = False,
        logit_refine_alpha: float = 0.25,
        logit_refine_rho: float = 0.8,
        expansion: float = 2.0,
        mlp_ratio: float = 1.0,
        mixer_start_stage: int = 3,
        deep_supervision: bool = False,
        initial_state_noise_std: float = 0.0,
        initial_state_noise_only_training: bool = True,
        clean_refiner_inputs: bool = False,
        rich_proposal_stages: bool = False,
        n_blocks_per_stage_decoder: Sequence[int] | None = None,
        max_decoder_blocks: int = 1,
        store_all_step_logits: bool = True,
    ):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("LiteRBUNeXt3DFeatureRefiner requires at least two encoder stages")
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")
        if not 1 <= training_min_steps <= recurrent_steps:
            raise ValueError("training_min_steps must be in [1, recurrent_steps]")
        if tbptt_keep_steps < 1:
            raise ValueError("tbptt_keep_steps must be >= 1")
        if not 0 <= initial_state_drop_prob <= 1:
            raise ValueError("initial_state_drop_prob must be in [0, 1]")
        if update_clip_ratio is not None and update_clip_ratio <= 0:
            raise ValueError("update_clip_ratio must be > 0 when enabled")
        if logit_refine_alpha < 0:
            raise ValueError("logit_refine_alpha must be >= 0")
        if not 0 <= logit_refine_rho <= 1:
            raise ValueError("logit_refine_rho must be in [0, 1]")
        if initial_state_noise_std < 0:
            raise ValueError("initial_state_noise_std must be >= 0")
        if normalize_finest_anchor_to_proposal and not normalize_finest_before_head:
            raise ValueError(
                "normalize_finest_anchor_to_proposal requires normalize_finest_before_head=True"
            )
        if refine_scales not in {"all", "finest", "coarse_to_fine", "coarse_to_fine_from", "single_to_fine"}:
            raise ValueError(
                "refine_scales must be one of: all, finest, coarse_to_fine, coarse_to_fine_from, single_to_fine"
            )
        if refiner_style not in {"mbconv", "sdrr", "sdrr_normstate", "sdrr_anchor_adain", "sdrr_film"}:
            raise ValueError(
                "refiner_style must be one of: mbconv, sdrr, sdrr_normstate, sdrr_anchor_adain, sdrr_film"
            )
        if belief_guided and refiner_style != "sdrr":
            raise ValueError("belief_guided=True currently requires refiner_style='sdrr'")
        if rich_proposal_stages and n_blocks_per_stage_decoder is None:
            raise ValueError("rich_proposal_stages=True requires n_blocks_per_stage_decoder")

        self.num_classes = int(num_classes)
        self.channels = tuple(int(i) for i in channels)
        num_decoder_states = len(self.channels) - 1
        if single_refine_index < 0:
            single_refine_index = num_decoder_states + int(single_refine_index)
        if not 0 <= int(single_refine_index) < num_decoder_states:
            raise ValueError(f"single_refine_index must be in [0, {num_decoder_states - 1}]")
        if c2f_start_index < 0:
            c2f_start_index = num_decoder_states + int(c2f_start_index)
        if not 0 <= int(c2f_start_index) < num_decoder_states:
            raise ValueError(f"c2f_start_index must be in [0, {num_decoder_states - 1}]")
        self.recurrent_steps = int(recurrent_steps)
        self.randomize_training_steps = bool(randomize_training_steps)
        self.training_min_steps = int(training_min_steps)
        self.tbptt_keep_steps = int(tbptt_keep_steps)
        self.refine_scales = str(refine_scales)
        self.single_refine_index = int(single_refine_index)
        if single_coarse_context_index is not None:
            if single_coarse_context_index < 0:
                single_coarse_context_index = num_decoder_states + int(single_coarse_context_index)
            if not 0 <= int(single_coarse_context_index) < num_decoder_states:
                raise ValueError(
                    f"single_coarse_context_index must be in [0, {num_decoder_states - 1}]"
                )
            if int(single_coarse_context_index) == self.single_refine_index:
                raise ValueError("single_coarse_context_index must differ from single_refine_index")
        self.single_coarse_context_index = (
            None if single_coarse_context_index is None else int(single_coarse_context_index)
        )
        self.c2f_start_index = int(c2f_start_index)
        self.checkpoint_refiner = bool(checkpoint_refiner)
        # When False, forward() stores DETACHED copies in last_all_step_logits. In training the
        # graph-carrying version pins every dead decode branch (base_logits, and each
        # intermediate step's logits) for the whole iteration and into the next one: backward
        # only frees saved tensors of nodes it actually traverses, and the loss uses
        # all_step_logits[-1] alone. Measured on BraTS 3d_fullres (batch 2 x 128^3, bf16):
        # 6.4 GB pinned at Stage-A depth 1, 12.7 GB at depth 2, taking the back-to-back
        # iteration peak from 29.4 GB to 42.2 GB. Only
        # nnUNetTrainerLiteFeatureRefinerAllScaleTBPTTRandK4UNeXt3DStableV2 needs the graph, so
        # it keeps the True default; detached copies still serve every read-only consumer.
        self.store_all_step_logits = bool(store_all_step_logits)
        self.alpha_decay = bool(alpha_decay)
        self.belief_guided = bool(belief_guided)
        self.logit_refine = bool(logit_refine)
        self.logit_refine_alpha = float(logit_refine_alpha)
        self.logit_refine_rho = float(logit_refine_rho)
        self.deep_supervision = bool(deep_supervision)
        self.initial_state_noise_std = float(initial_state_noise_std)
        self.initial_state_noise_only_training = bool(initial_state_noise_only_training)
        self.clean_refiner_inputs = bool(clean_refiner_inputs)
        self.normalize_finest_anchor_to_proposal = bool(normalize_finest_anchor_to_proposal)
        self.rich_proposal_stages = bool(rich_proposal_stages)
        self._proposal_norm_mean: torch.Tensor | None = None
        self._proposal_norm_var: torch.Tensor | None = None

        kernel_sizes = tuple(_as_3tuple(i) for i in kernel_sizes)
        strides = tuple(_as_3tuple(i) for i in strides)

        encoder_stages = []
        mixers = []
        in_ch = input_channels
        for stage_idx, out_ch in enumerate(self.channels):
            encoder_stages.append(
                LiteEncoderStage3D(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_sizes[stage_idx],
                    stride=strides[stage_idx],
                    n_blocks=max(1, int(n_blocks_per_stage[stage_idx])),
                    expansion=expansion,
                )
            )
            if stage_idx == len(self.channels) - 1:
                mixers.append(ResidualShiftMLPDW3D(out_ch, mlp_ratio=mlp_ratio))
            elif stage_idx >= int(mixer_start_stage):
                mixers.append(ResidualShiftMLPDW3D(out_ch, mlp_ratio=mlp_ratio))
            else:
                mixers.append(nn.Identity())
            in_ch = out_ch

        proposal_stages = []
        refiner_stages = []
        if refiner_style == "sdrr":
            shared_refiner = SharedDecoderRefinementBlock3D(hidden_channels)
            refiner_stage_cls = SDRRFeatureRefinerStage3D
            refiner_extra_kwargs = {
                "num_classes": self.num_classes,
                "belief_guided": self.belief_guided,
                "clean_inputs": self.clean_refiner_inputs,
            }
        elif refiner_style == "sdrr_normstate":
            shared_refiner = SharedDecoderRefinementBlock3D(hidden_channels)
            refiner_stage_cls = SDRRNormStateFeatureRefinerStage3D
            refiner_extra_kwargs = {
                "num_classes": self.num_classes,
                "belief_guided": self.belief_guided,
                "clean_inputs": self.clean_refiner_inputs,
            }
        elif refiner_style == "sdrr_anchor_adain":
            shared_refiner = SharedDecoderRefinementBlock3D(hidden_channels)
            refiner_stage_cls = SDRRAnchorAdaINFeatureRefinerStage3D
            refiner_extra_kwargs = {
                "num_classes": self.num_classes,
                "belief_guided": self.belief_guided,
                "clean_inputs": self.clean_refiner_inputs,
            }
        elif refiner_style == "sdrr_film":
            shared_refiner = SharedDecoderRefinementBlock3D(hidden_channels)
            refiner_stage_cls = SDRRFiLMFeatureRefinerStage3D
            refiner_extra_kwargs = {
                "num_classes": self.num_classes,
                "belief_guided": self.belief_guided,
                "clean_inputs": self.clean_refiner_inputs,
            }
        else:
            shared_refiner = SharedFeatureRefinementBlock3D(hidden_channels, expansion=1.0)
            refiner_stage_cls = FeatureRefinerStage3D
            refiner_extra_kwargs = {}
        for decoder_idx in range(len(self.channels) - 1):
            prev_ch = self.channels[-(decoder_idx + 1)]
            skip_ch = self.channels[-(decoder_idx + 2)]
            if self.rich_proposal_stages:
                n_blocks = min(int(n_blocks_per_stage_decoder[decoder_idx]), max_decoder_blocks)
                proposal_stages.append(
                    LiteDecoderStage3D(
                        prev_ch,
                        skip_ch,
                        kernel_size=kernel_sizes[-(decoder_idx + 2)],
                        n_blocks=n_blocks,
                        expansion=expansion,
                        mlp_ratio=mlp_ratio,
                    )
                )
            else:
                proposal_stages.append(BaseProposalStage3D(prev_ch, skip_ch))
            if refine_scales in {"all", "coarse_to_fine", "coarse_to_fine_from"}:
                if refine_scales == "coarse_to_fine_from" and decoder_idx < self.c2f_start_index:
                    refiner_stages.append(nn.Identity())
                    continue
                use_context = refine_scales in {"coarse_to_fine", "coarse_to_fine_from"} and decoder_idx > 0
                coarse_ch = prev_ch if use_context else None
                refiner_stages.append(
                    refiner_stage_cls(
                        state_channels=skip_ch,
                        skip_channels=skip_ch,
                        hidden_channels=hidden_channels,
                        shared_block=shared_refiner,
                        alpha=alpha,
                        alpha_decay=alpha_decay,
                        coarse_context_channels=coarse_ch,
                        use_checkpoint=checkpoint_refiner,
                        initial_state_drop_prob=initial_state_drop_prob,
                        update_clip_ratio=update_clip_ratio,
                        **refiner_extra_kwargs,
                    )
                )

        finest_refiner = None
        if refine_scales == "finest":
            finest_refiner = refiner_stage_cls(
                state_channels=self.channels[0],
                skip_channels=self.channels[0],
                hidden_channels=hidden_channels,
                shared_block=shared_refiner,
                alpha=alpha,
                alpha_decay=alpha_decay,
                use_checkpoint=checkpoint_refiner,
                initial_state_drop_prob=initial_state_drop_prob,
                update_clip_ratio=update_clip_ratio,
                **refiner_extra_kwargs,
            )

        single_refiner = None
        if refine_scales == "single_to_fine":
            single_ch = self.channels[-(self.single_refine_index + 2)]
            context_ch = (
                None
                if self.single_coarse_context_index is None
                else self.channels[-(self.single_coarse_context_index + 2)]
            )
            single_refiner = refiner_stage_cls(
                state_channels=single_ch,
                skip_channels=single_ch,
                hidden_channels=hidden_channels,
                shared_block=shared_refiner,
                alpha=alpha,
                alpha_decay=alpha_decay,
                coarse_context_channels=context_ch,
                use_checkpoint=checkpoint_refiner,
                initial_state_drop_prob=initial_state_drop_prob,
                update_clip_ratio=update_clip_ratio,
                **refiner_extra_kwargs,
            )

        self.encoder_stages = nn.ModuleList(encoder_stages)
        self.mixers = nn.ModuleList(mixers)
        self.proposal_stages = nn.ModuleList(proposal_stages)
        self.refiner_stages = nn.ModuleList(refiner_stages)
        self.finest_refiner = finest_refiner
        self.single_refiner = single_refiner
        if normalize_finest_before_head:
            num_groups = min(8, self.channels[0])
            while self.channels[0] % num_groups != 0:
                num_groups -= 1
            self.finest_head_norm = nn.GroupNorm(num_groups=num_groups, num_channels=self.channels[0])
        else:
            self.finest_head_norm = nn.Identity()
        self.mask_head = nn.Conv3d(self.channels[0], num_classes, kernel_size=1, bias=True)
        if self.logit_refine:
            num_groups = min(8, self.channels[0])
            while self.channels[0] % num_groups != 0:
                num_groups -= 1
            self.logit_refine_head = nn.Sequential(
                nn.GroupNorm(num_groups=num_groups, num_channels=self.channels[0]),
                nn.Conv3d(self.channels[0], self.channels[0], kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv3d(self.channels[0], num_classes, kernel_size=1, bias=True),
            )
        else:
            self.logit_refine_head = None
        self.decoder = _DeepSupervisionProxy(self)

        self.last_all_step_logits: list[torch.Tensor] = []
        self.last_all_step_indices: list[int] = []
        self.last_active_step_indices: list[int] = []
        self.last_belief_logits: list[torch.Tensor] = []
        self.last_belief_step_indices: list[int] = []
        self.last_step_metrics: dict[str, list[float] | float | int] = {
            "r_d": [],
            "r_m": [],
            "K": 0,
            "T_cut": 0,
        }

    def _get_recurrent_steps(self, x: torch.Tensor, recurrent_steps: int | None) -> int:
        if recurrent_steps is not None:
            steps = int(recurrent_steps)
        elif "LITE_REFINER_TEST_STEPS" in os.environ:
            steps = int(os.environ["LITE_REFINER_TEST_STEPS"])
        elif "LITE_SDRR_TEST_STEPS" in os.environ:
            steps = int(os.environ["LITE_SDRR_TEST_STEPS"])
        elif "LITE_RECURRENT_TEST_STEPS" in os.environ:
            steps = int(os.environ["LITE_RECURRENT_TEST_STEPS"])
        elif self.training and self.randomize_training_steps:
            steps = int(torch.randint(self.training_min_steps, self.recurrent_steps + 1, (1,), device=x.device).item())
        else:
            steps = self.recurrent_steps
        if steps < 0:
            raise ValueError("recurrent_steps must be >= 0")
        return steps

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips = []
        for stage, mixer in zip(self.encoder_stages, self.mixers):
            x = mixer(stage(x))
            skips.append(x)
        return skips

    def _build_proposal_states(
        self, skips: list[torch.Tensor], stop_idx: int | None = None
    ) -> list[torch.Tensor]:
        """Build proposal states d_0 for every decoder position, or only up to `stop_idx`.

        `stop_idx` exists because the multi-start B/C training helpers only ever read
        proposal_states[single_coarse_context_index] and [single_refine_index], and rebuild
        everything above the refine index through _replay_proposal_from_state() anyway. Running
        the full loop there computes the two highest-resolution decoder stages (64^3 and 128^3
        on BraTS 3d_fullres) whose outputs are then never read -- pure waste under no_grad in
        Stage B/C2, and waste that also pins grad-enabled activations for the whole iteration in
        Stage C1, whose _proposal() call is NOT under no_grad.

        Truncating is side-effect free: this file contains no BatchNorm (ConvNormAct3D uses
        InstanceNorm3d(affine=True), which keeps no running stats) and the decoder stages draw no
        RNG, so the skipped stages cannot influence model state or the trajectory. It removes no
        gradient path either -- proposal_stages above the refine index receive their gradient via
        the replay, not via this loop.

        Callers that read the finest state (forward()'s base_logits, i.e. the K=0 output) must
        leave stop_idx as None.
        """
        states = []
        x = skips[-1]
        for idx, stage in enumerate(self.proposal_stages):
            if stop_idx is not None and idx > stop_idx:
                break
            skip = skips[-(idx + 2)]
            x = stage(x, skip)
            states.append(x)
        return states

    def _replay_proposal_from_state(
        self,
        states: list[torch.Tensor],
        skips: list[torch.Tensor],
        start_idx: int,
    ) -> list[torch.Tensor]:
        updated_states = list(states[: start_idx + 1])
        x = updated_states[-1]
        for idx in range(start_idx + 1, len(self.proposal_stages)):
            skip = skips[-(idx + 2)]
            x = self.proposal_stages[idx](x, skip)
            updated_states.append(x)
        return updated_states

    def _apply_finest_norm(self, finest_state: torch.Tensor) -> torch.Tensor:
        """
        Apply the pre-mask_head normalization. In the default per-step mode
        (normalize_finest_anchor_to_proposal=False), this is a plain GroupNorm
        call that recomputes mean/var fresh from whatever finest_state is
        passed in -- at deep steps t on an OOD-drifted state, this
        re-standardizes to unit variance regardless of how far the state has
        wandered, which does nothing to damp that drift (it's a per-step
        "OOD amplifier": the GroupNorm output always looks statistically
        "normal" even when the underlying state is not).

        In anchor mode, GroupNorm's learnable affine (weight/bias) is still
        applied every step, but the mean/var used to standardize are computed
        ONCE from the step-0 proposal state and reused (detached, as a fixed
        reference point, not a differentiable pathway back into d_0) for
        every subsequent step. This decouples "does mask_head see roughly the
        same normalized statistics every step" from "how far has the raw
        feature state drifted" -- isolating whether GroupNorm's per-step
        re-standardization is itself a source of the K=16 divergence seen in
        finestonly_reg_bf16, independent of the feature-refine step size
        (alpha_decay) or the state trajectory (noise/randinit).
        """
        if isinstance(self.finest_head_norm, nn.Identity):
            return finest_state
        if not self.normalize_finest_anchor_to_proposal:
            return self.finest_head_norm(finest_state)

        gn = self.finest_head_norm
        num_groups = gn.num_groups
        b, c = finest_state.shape[0], finest_state.shape[1]
        reduce_dims = tuple(range(2, finest_state.dim() + 1))
        if self._proposal_norm_mean is None:
            grouped = finest_state.detach().float().view(b, num_groups, c // num_groups, *finest_state.shape[2:])
            self._proposal_norm_mean = grouped.mean(dim=reduce_dims, keepdim=True)
            self._proposal_norm_var = grouped.var(dim=reduce_dims, unbiased=False, keepdim=True)

        grouped = finest_state.view(b, num_groups, c // num_groups, *finest_state.shape[2:])
        mean = self._proposal_norm_mean.to(dtype=finest_state.dtype)
        var = self._proposal_norm_var.to(dtype=finest_state.dtype)
        normalized = (grouped - mean) / torch.sqrt(var + gn.eps)
        normalized = normalized.view(b, c, *finest_state.shape[2:])
        affine_shape = (1, c) + (1,) * (finest_state.dim() - 2)
        return normalized * gn.weight.view(affine_shape) + gn.bias.view(affine_shape)

    def _decode_logits(self, finest_state: torch.Tensor, output_size: Sequence[int]) -> torch.Tensor:
        if finest_state.shape[2:] != tuple(output_size):
            finest_state = F.interpolate(finest_state, size=output_size, mode="nearest")
        finest_state = self._apply_finest_norm(finest_state)
        return self.mask_head(finest_state)

    def _decode_logit_update(self, finest_state: torch.Tensor, output_size: Sequence[int]) -> torch.Tensor:
        if self.logit_refine_head is None:
            raise RuntimeError("logit_refine_head is only available when logit_refine=True")
        delta_logits = self.logit_refine_head(finest_state)
        if delta_logits.shape[2:] != tuple(output_size):
            delta_logits = F.interpolate(delta_logits, size=output_size, mode="nearest")
        return torch.tanh(delta_logits)

    def _update_logit_context(
        self,
        logit_context: torch.Tensor,
        finest_state: torch.Tensor,
        output_size: Sequence[int],
        step_idx: int,
    ) -> torch.Tensor:
        delta_logits = self._decode_logit_update(finest_state, output_size)
        alpha = self.logit_refine_alpha / sqrt(float(step_idx + 1))
        return self.logit_refine_rho * logit_context + alpha * delta_logits

    def _decode_step_logits(
        self,
        finest_state: torch.Tensor,
        output_size: Sequence[int],
        logit_context: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = self._decode_logits(finest_state, output_size)
        if logit_context is not None:
            logits = logits + logit_context
        return logits

    @staticmethod
    def _prob_from_logits(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def _add_initial_state_noise(self, feature: torch.Tensor) -> torch.Tensor:
        """
        EqR-style randomized state initialization: perturb the proposal state
        d^4_0 ONCE, before the recurrent refinement loop (and before it is used
        to compute the step-0 proposal logits m_0), rather than injecting noise
        at every step. This is a distinct mechanism from feature-trajectory
        noise (LiteRBUNeXt3DFeatureRefinerNoise adds noise at every refine
        step) -- here the goal is to expose the refiner to varied starting
        points during training so it learns a correction behavior that
        generalizes across the state-space neighborhood around the proposal,
        not noise-robustness at every intermediate step.
        """
        if self.initial_state_noise_std <= 0:
            return feature
        if self.initial_state_noise_only_training and not self.training:
            return feature
        scale = feature.detach().float().flatten(1).std(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1e-6)
        noise = torch.randn_like(feature) * scale.to(dtype=feature.dtype) * self.initial_state_noise_std
        return feature + noise

    @staticmethod
    def _relative_norm_delta(new_tensor: torch.Tensor, old_tensor: torch.Tensor) -> float:
        base = old_tensor.detach().float().norm().clamp_min(1e-8)
        delta = (new_tensor.detach().float() - old_tensor.detach().float()).norm()
        return float((delta / base).cpu())

    def _refine_once(
        self,
        states: list[torch.Tensor],
        proposal_states: list[torch.Tensor],
        skips: list[torch.Tensor],
        step_idx: int,
    ) -> list[torch.Tensor]:
        updated_states = []
        for idx, (state, initial_state, refiner) in enumerate(zip(states, proposal_states, self.refiner_stages)):
            skip = skips[-(idx + 2)]
            updated_states.append(refiner(state, initial_state, skip, step_idx=step_idx))
        return updated_states

    def _refine_coarse_to_fine_once(
        self,
        states: list[torch.Tensor],
        proposal_states: list[torch.Tensor],
        skips: list[torch.Tensor],
        step_idx: int,
    ) -> list[torch.Tensor]:
        updated_states = list(states)
        coarse_residual_context = None
        start_idx = self.c2f_start_index if self.refine_scales == "coarse_to_fine_from" else 0
        if start_idx > 0:
            coarse_residual_context = updated_states[start_idx - 1] - proposal_states[start_idx - 1]
        for idx in range(start_idx, len(states)):
            refiner = self.refiner_stages[idx]
            skip = skips[-(idx + 2)]
            updated = refiner(
                states[idx],
                proposal_states[idx],
                skip,
                coarse_context=coarse_residual_context,
                step_idx=step_idx,
            )
            updated_states[idx] = updated
            coarse_residual_context = updated - proposal_states[idx]
        return updated_states

    def _refine_finest_once(
        self,
        states: list[torch.Tensor],
        proposal_states: list[torch.Tensor],
        skips: list[torch.Tensor],
        step_idx: int,
    ) -> list[torch.Tensor]:
        if self.finest_refiner is None:
            raise RuntimeError("finest_refiner is only available when refine_scales='finest'")
        updated_states = list(states)
        finest_idx = len(states) - 1
        skip = skips[-(finest_idx + 2)]
        updated_states[finest_idx] = self.finest_refiner(
            states[finest_idx],
            proposal_states[finest_idx],
            skip,
            step_idx=step_idx,
        )
        return updated_states

    def _refine_single_to_fine_once(
        self,
        states: list[torch.Tensor],
        proposal_states: list[torch.Tensor],
        skips: list[torch.Tensor],
        step_idx: int,
        replay: bool = True,
    ) -> list[torch.Tensor]:
        """Update the refine-index state, then optionally replay it forward to the finest
        resolution (rebuilding every proposal_stage downstream of single_refine_index).

        `replay=False` skips that rebuild entirely. This is safe because the refiner recurrence
        itself never reads the replayed (fine) states -- refine_step only ever reads
        `proposal_states` (the fixed, un-refined build from _build_proposal_states) and `skips`,
        never `states[refine_idx+1:]`. Those fine states exist ONLY to be decoded; if the caller
        is not going to decode this step (do_decode=False in forward()), replaying them is pure
        discarded work -- confirmed empirically: replaying two decoder stages costs ~6.4x more
        than the refiner update itself (see EXPERIMENTS.md 2026-08-25 finding), so at K=32 this
        was ~84% of the model's total FLOPs, spent on 31 of 32 replays whose result was never
        read. With replay=False, states[refine_idx+1:] are left stale (whatever they were before
        this call -- typically the original, un-refined build); this is safe precisely because
        nothing downstream reads them unless the NEXT call passes replay=True, at which point
        they are freshly rebuilt from the just-updated refine_idx state before being read.
        """
        if self.single_refiner is None:
            raise RuntimeError("single_refiner is only available when refine_scales='single_to_fine'")
        refine_idx = self.single_refine_index
        updated_states = list(states)
        skip = skips[-(refine_idx + 2)]
        coarse_context = (
            None
            if self.single_coarse_context_index is None
            else proposal_states[self.single_coarse_context_index]
        )
        updated_states[refine_idx] = self.single_refiner(
            states[refine_idx],
            proposal_states[refine_idx],
            skip,
            coarse_context=coarse_context,
            step_idx=step_idx,
        )
        # len(self.proposal_stages), NOT len(updated_states): the caller may have passed a
        # states list truncated at single_refine_index (see forward()'s need_full_initial_build),
        # in which case len(updated_states)-1 == refine_idx and the old
        # `refine_idx < len(updated_states) - 1` check would incorrectly conclude there is
        # nothing coarser to replay and silently skip it forever, leaving states[-1] stuck at a
        # coarse, never-decoded resolution. self.proposal_stages is the network's fixed,
        # never-truncated module list, so this reflects the network's real topology regardless
        # of how long the states list passed in happens to be.
        if replay and refine_idx < len(self.proposal_stages) - 1:
            updated_states = self._replay_proposal_from_state(updated_states, skips, start_idx=refine_idx)
        return updated_states

    def _refine_selected_once(
        self,
        states: list[torch.Tensor],
        proposal_states: list[torch.Tensor],
        skips: list[torch.Tensor],
        step_idx: int,
        replay: bool = True,
    ) -> list[torch.Tensor]:
        if self.refine_scales == "finest":
            return self._refine_finest_once(states, proposal_states, skips, step_idx=step_idx)
        if self.refine_scales == "single_to_fine":
            return self._refine_single_to_fine_once(
                states, proposal_states, skips, step_idx=step_idx, replay=replay
            )
        if self.refine_scales in {"coarse_to_fine", "coarse_to_fine_from"}:
            return self._refine_coarse_to_fine_once(states, proposal_states, skips, step_idx=step_idx)
        return self._refine_once(states, proposal_states, skips, step_idx=step_idx)

    def _collect_recent_belief_logits(self, step_idx: int) -> None:
        if not self.belief_guided:
            return
        if self.refine_scales == "finest":
            modules = [self.finest_refiner]
        elif self.refine_scales == "single_to_fine":
            modules = [self.single_refiner]
        else:
            modules = [m for m in self.refiner_stages if not isinstance(m, nn.Identity)]
        for module in modules:
            if module is None:
                continue
            for belief_logits in getattr(module, "last_belief_logits", []):
                self.last_belief_logits.append(belief_logits)
                self.last_belief_step_indices.append(step_idx + 1)

    def forward(
        self,
        x: torch.Tensor,
        recurrent_steps: int | None = None,
        tbptt_keep_steps: int | None = None,
        decode_all_steps: bool | None = None,
    ):
        """decode_all_steps controls whether every intermediate step gets replayed to the finest
        resolution and decoded through mask_head, or only the one step whose result is actually
        returned. Historically every step was always decoded (K=0's base_logits included), even
        though only all_step_logits[-1] is ever returned when recurrent_steps >= 1 -- every
        earlier decode was pure discarded compute (see EXPERIMENTS.md 2026-08-25 finding). This
        is now opt-in per call:

          decode_all_steps=None (default): resolves to self.training. Every existing trainer
              that calls this module directly (nnUNetTrainer.train_step's `self.network(...)`)
              keeps its exact prior behavior unless it explicitly passes decode_all_steps=False
              below -- only two call sites (kcov_ab's Stage-A K0-aux train_step and Stage-B
              proposal-preserving backbone objective) have been updated to do so; every other
              trainer, including ones that genuinely consume last_all_step_logits/
              last_step_metrics for their own loss (e.g.
              nnUNetTrainerLiteFeatureRefinerAllScaleTBPTTRandK4UNeXt3DStableV2), is unaffected.
              Inference (self.training=False, i.e. every predict_from_raw_data.py call) newly
              defaults to the fast path -- this is what makes every K>=1 evaluation cheaper.
          decode_all_steps=True/False: explicit override, takes precedence over the default.

        Forced back to the full (all-steps) path regardless of decode_all_steps whenever:
          - LITE_REFINER_METRICS_CSV is set (the diagnostic scripts need every step's r_d/r_m),
          - self.logit_refine is True (the residual logit_context accumulates step-by-step;
            skipping steps would silently corrupt it -- no shipped trainer sets this True today,
            so this branch is defensive, not currently load-bearing).

        In every case, final_logits is bit-identical to what the pre-optimization code would
        have returned for the same inputs -- this only removes computation whose result was
        always going to be discarded, never changes what gets returned.
        """
        if os.environ.get("LITE_REFINER_INF_EXTRAPOLATE") == "1":
            return self._forward_inf_extrapolate(x)
        input_shape = x.shape[2:]
        recurrent_steps = self._get_recurrent_steps(x, recurrent_steps)
        keep_steps = self.tbptt_keep_steps if tbptt_keep_steps is None else int(tbptt_keep_steps)
        if keep_steps < 1:
            raise ValueError("tbptt_keep_steps must be >= 1")
        self._proposal_norm_mean = None
        self._proposal_norm_var = None
        self.last_belief_logits = []
        self.last_belief_step_indices = []

        if decode_all_steps is None:
            decode_all_steps = self.training
        track_all_steps = bool(decode_all_steps) or bool(os.environ.get("LITE_REFINER_METRICS_CSV")) or self.logit_refine

        # base_logits (the K=0 decode) is the actual answer only when recurrent_steps == 0; for
        # any K >= 1 it is unconditionally superseded by all_step_logits[-1] below. Skip it
        # whenever it is certain to be discarded.
        need_base_logits = recurrent_steps == 0 or track_all_steps

        skips = self._encode(x)
        # For single_to_fine, every proposal_stage past single_refine_index is read for the
        # FIRST time only by _decode_logits(states[-1], ...) (when need_base_logits) or by the
        # replay a real refine step performs on the step it decodes -- and replay always
        # recomputes those stages fresh from the just-updated refine_idx state (see
        # _replay_proposal_from_state), never from this initial build. So when
        # need_base_logits/initial_state_noise_std are both false, building past
        # single_refine_index here is pure discarded work: this project's own K-sweep evaluations
        # (K>=1) previously paid for it as a full extra replay-equivalent call on every single
        # inference, on top of the one genuinely-needed replay at the decode step (see
        # EXPERIMENTS.md 2026-08-25 finding). Other refine_scales modes ("finest",
        # "coarse_to_fine[_from]") have no truncated-build counterpart wired up below, so this
        # only applies to single_to_fine.
        need_full_initial_build = (
            need_base_logits
            or self.initial_state_noise_std > 0
            or self.refine_scales != "single_to_fine"
        )
        proposal_states = self._build_proposal_states(
            skips, stop_idx=None if need_full_initial_build else self.single_refine_index
        )
        states = [state for state in proposal_states]
        if self.initial_state_noise_std > 0:
            states[-1] = self._add_initial_state_noise(states[-1])

        base_logits = self._decode_logits(states[-1], input_shape) if need_base_logits else None
        logit_context = torch.zeros_like(base_logits) if (self.logit_refine and base_logits is not None) else None
        all_step_logits = [base_logits] if need_base_logits else []
        all_step_indices = [0] if need_base_logits else []
        metrics_r_d: list[float] = []
        metrics_r_m: list[float] = []
        prev_logits = base_logits

        if self.training:
            t_cut = max(0, recurrent_steps - keep_steps)
            for step_idx in range(t_cut):
                with torch.no_grad():
                    prev_finest = states[-1]
                    states = self._refine_selected_once(
                        states, proposal_states, skips, step_idx=step_idx, replay=track_all_steps
                    )
                    if track_all_steps:
                        if logit_context is not None:
                            logit_context = self._update_logit_context(
                                logit_context, states[-1], input_shape, step_idx=step_idx
                            )
                        current_logits = self._decode_step_logits(states[-1], input_shape, logit_context)
                        metrics_r_d.append(self._relative_norm_delta(states[-1], prev_finest))
                        metrics_r_m.append(self._relative_norm_delta(self._prob_from_logits(current_logits), self._prob_from_logits(prev_logits)))
                        prev_logits = current_logits
            if t_cut > 0:
                states = [state.detach() for state in states]
                if logit_context is not None:
                    logit_context = logit_context.detach()

            for step_idx in range(t_cut, recurrent_steps):
                is_last_step = step_idx == recurrent_steps - 1
                do_decode = track_all_steps or is_last_step
                prev_finest = states[-1]
                states = self._refine_selected_once(
                    states, proposal_states, skips, step_idx=step_idx, replay=do_decode
                )
                self._collect_recent_belief_logits(step_idx)
                if do_decode:
                    if logit_context is not None:
                        logit_context = self._update_logit_context(
                            logit_context, states[-1], input_shape, step_idx=step_idx
                        )
                    current_logits = self._decode_step_logits(states[-1], input_shape, logit_context)
                    if track_all_steps:
                        metrics_r_d.append(self._relative_norm_delta(states[-1], prev_finest))
                        metrics_r_m.append(self._relative_norm_delta(self._prob_from_logits(current_logits), self._prob_from_logits(prev_logits)))
                        prev_logits = current_logits
                    all_step_logits.append(current_logits)
                    all_step_indices.append(step_idx + 1)
            active_step_indices = [0] + list(range(t_cut + 1, recurrent_steps + 1))
        else:
            t_cut = 0
            for step_idx in range(recurrent_steps):
                is_last_step = step_idx == recurrent_steps - 1
                do_decode = track_all_steps or is_last_step
                prev_finest = states[-1]
                states = self._refine_selected_once(
                    states, proposal_states, skips, step_idx=step_idx, replay=do_decode
                )
                self._collect_recent_belief_logits(step_idx)
                if do_decode:
                    if logit_context is not None:
                        logit_context = self._update_logit_context(
                            logit_context, states[-1], input_shape, step_idx=step_idx
                        )
                    current_logits = self._decode_step_logits(states[-1], input_shape, logit_context)
                    if track_all_steps:
                        metrics_r_d.append(self._relative_norm_delta(states[-1], prev_finest))
                        metrics_r_m.append(self._relative_norm_delta(self._prob_from_logits(current_logits), self._prob_from_logits(prev_logits)))
                        prev_logits = current_logits
                    all_step_logits.append(current_logits)
                    all_step_indices.append(step_idx + 1)
            active_step_indices = list(range(recurrent_steps + 1))

        final_logits = all_step_logits[-1]
        self.last_all_step_logits = (
            all_step_logits
            if self.store_all_step_logits
            else [logits.detach() for logits in all_step_logits]
        )
        self.last_all_step_indices = all_step_indices
        self.last_active_step_indices = active_step_indices
        self.last_step_metrics = {
            "r_d": metrics_r_d,
            "r_m": metrics_r_m,
            "K": recurrent_steps,
            "T_cut": t_cut,
        }
        metrics_csv = os.environ.get("LITE_REFINER_METRICS_CSV")
        if metrics_csv:
            import csv

            os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
            row = {
                "K": recurrent_steps,
                "T_cut": t_cut,
                "n_steps": len(metrics_r_d),
                "r_d_mean": float(sum(metrics_r_d) / len(metrics_r_d)) if metrics_r_d else 0.0,
                "r_m_mean": float(sum(metrics_r_m) / len(metrics_r_m)) if metrics_r_m else 0.0,
                "r_d_steps": ";".join(f"{v:.8g}" for v in metrics_r_d),
                "r_m_steps": ";".join(f"{v:.8g}" for v in metrics_r_m),
            }
            write_header = not os.path.exists(metrics_csv)
            with open(metrics_csv, "a", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        if self.deep_supervision:
            return [final_logits]
        return final_logits

    def _forward_inf_extrapolate(self, x: torch.Tensor) -> torch.Tensor:
        """Decode the analytic K->infinity state estimated from d4, d6 and d8."""
        if self.refine_scales != "single_to_fine":
            raise RuntimeError(
                "LITE_REFINER_INF_EXTRAPOLATE currently requires refine_scales='single_to_fine'"
            )
        input_shape = x.shape[2:]
        with torch.no_grad():
            skips = self._encode(x)
            # Build only up to refine_idx: every proposal stage past it is rebuilt fresh by
            # the replay below, so building them here is pure discarded work. Mirrors
            # forward()'s need_full_initial_build path and RecurrentRefinerBackbone's
            # _forward_inf_extrapolate, which already passes stop_idx=refine_idx.
            proposal_states = self._build_proposal_states(
                skips, stop_idx=self.single_refine_index
            )
            refine_idx = self.single_refine_index
            states = list(proposal_states)
            d0 = proposal_states[refine_idx]
            checkpoints: dict[int, torch.Tensor] = {}
            for step_idx in range(8):
                # replay=False: the recurrence never reads the replayed (fine) states (see
                # _refine_single_to_fine_once) and this loop only reads states[refine_idx],
                # so replaying each step here is pure discarded work; the one replay that is
                # actually needed happens once, after the extrapolation, below.
                states = self._refine_selected_once(
                    states, proposal_states, skips, step_idx=step_idx, replay=False
                )
                if step_idx + 1 in (4, 6, 8):
                    checkpoints[step_idx + 1] = states[refine_idx]
            d4, d6, d8 = checkpoints[4], checkpoints[6], checkpoints[8]
            u6 = (d6 - d4).float()
            u8 = (d8 - d6).float()
            reduce_dims = tuple(range(1, u6.dim()))
            q = ((u8 * u6).sum(dim=reduce_dims) /
                 ((u6 * u6).sum(dim=reduce_dims) + 1e-8)).clamp(0.0, 0.95)
            q = q.view((q.shape[0],) + (1,) * (u6.dim() - 1))
            d_inf_raw = d8 + (q / (1.0 - q + 1e-8)).to(d8.dtype) * (d8 - d6)
            refiner = self.single_refiner
            if hasattr(refiner, "_film_params") and hasattr(refiner, "raw_norm"):
                gamma, beta = refiner._film_params(d0)
                d_inf = gamma * refiner.raw_norm(d_inf_raw) + beta
            else:
                d_inf = d_inf_raw
            states[refine_idx] = d_inf
            states = self._replay_proposal_from_state(states, skips, start_idx=refine_idx)
            logits = self._decode_logits(states[-1], input_shape)
        self.last_all_step_logits = [logits.detach()]
        self.last_all_step_indices = [0]
        self.last_active_step_indices = [0]
        self.last_step_metrics = {"r_d": [], "r_m": [], "K": float("inf"), "T_cut": 0}
        return [logits] if self.deep_supervision else logits


class LiteRBUNeXt3DFeatureRefinerIndependentD16D32(LiteRBUNeXt3DFeatureRefiner):
    """Independent coarse-to-fine D16 then D32 SDRR updates within each outer step."""

    def __init__(self, *args, hidden_channels: int = 64, **kwargs):
        kwargs.update(
            hidden_channels=hidden_channels,
            refine_scales="single_to_fine",
            single_refine_index=1,
            single_coarse_context_index=0,
            refiner_style="sdrr",
        )
        super().__init__(*args, **kwargs)

        d32_idx = 2
        d32_channels = self.channels[-(d32_idx + 2)]
        d16_channels = self.channels[-(self.single_refine_index + 2)]
        d32_shared_block = SharedDecoderRefinementBlock3D(hidden_channels)
        self.d32_refiner = SDRRFeatureRefinerStage3D(
            state_channels=d32_channels,
            skip_channels=d32_channels,
            hidden_channels=hidden_channels,
            shared_block=d32_shared_block,
            alpha=self.single_refiner.alpha,
            alpha_decay=self.single_refiner.alpha_decay,
            coarse_context_channels=d16_channels,
            use_checkpoint=self.checkpoint_refiner,
            num_classes=self.num_classes,
            belief_guided=False,
        )
        self.d32_refine_index = d32_idx

    def refine_joint_once(
        self,
        d16_state: torch.Tensor,
        d32_state: torch.Tensor,
        proposal_states: list[torch.Tensor],
        skips: list[torch.Tensor],
        step_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d16_idx = self.single_refine_index
        d32_idx = self.d32_refine_index
        d16_next = self.single_refiner(
            d16_state,
            proposal_states[d16_idx],
            skips[-(d16_idx + 2)],
            coarse_context=proposal_states[self.single_coarse_context_index],
            step_idx=step_idx,
        )
        d32_next = self.d32_refiner(
            d32_state,
            proposal_states[d32_idx],
            skips[-(d32_idx + 2)],
            coarse_context=d16_next,
            step_idx=step_idx,
        )
        return d16_next, d32_next

    def _refine_single_to_fine_once(
        self,
        states: list[torch.Tensor],
        proposal_states: list[torch.Tensor],
        skips: list[torch.Tensor],
        step_idx: int,
    ) -> list[torch.Tensor]:
        updated_states = list(states)
        d16_next, d32_next = self.refine_joint_once(
            states[self.single_refine_index],
            states[self.d32_refine_index],
            proposal_states,
            skips,
            step_idx,
        )
        updated_states[self.single_refine_index] = d16_next
        updated_states[self.d32_refine_index] = d32_next
        return self._replay_proposal_from_state(updated_states, skips, self.d32_refine_index)


class LiteRBUNeXt3DFeatureRefinerFinestAgg(LiteRBUNeXt3DFeatureRefiner):
    """
    Finest-state recurrent refiner with multi-scale proposal aggregation.

    The proposal decoder is kept unchanged. Before recurrent refinement, every
    proposal decoder state is projected to the finest-state channel count and
    resized to the finest spatial shape. Those aligned features are averaged and
    used as the finest d^0. Coarser proposal states are detached for the
    aggregation path, while the original finest proposal keeps its gradient.
    """

    def __init__(
        self,
        *args,
        aggregate_detach_nonfinest: bool = True,
        **kwargs,
    ):
        kwargs["refine_scales"] = "finest"
        super().__init__(*args, **kwargs)
        self.aggregate_detach_nonfinest = bool(aggregate_detach_nonfinest)
        finest_channels = self.channels[0]
        state_channels = [self.channels[-(idx + 2)] for idx in range(len(self.channels) - 1)]
        self.aggregate_projections = nn.ModuleList(
            [
                nn.Identity()
                if ch == finest_channels
                else ConvNormAct3D(ch, finest_channels, kernel_size=1, act=False)
                for ch in state_channels
            ]
        )

    def _aggregate_finest_proposal_state(self, proposal_states: list[torch.Tensor]) -> torch.Tensor:
        target_shape = proposal_states[-1].shape[2:]
        aggregated_state = None
        last_idx = len(proposal_states) - 1
        for idx, (state, projection) in enumerate(zip(proposal_states, self.aggregate_projections)):
            if self.aggregate_detach_nonfinest and idx != last_idx:
                state = state.detach()
            state = projection(state)
            if state.shape[2:] != target_shape:
                state = F.interpolate(state, size=target_shape, mode="trilinear", align_corners=False)
            aggregated_state = state if aggregated_state is None else aggregated_state + state
        if aggregated_state is None:
            raise RuntimeError("proposal_states must not be empty")
        return aggregated_state / float(len(proposal_states))

    def _build_proposal_states(
        self, skips: list[torch.Tensor], stop_idx: int | None = None
    ) -> list[torch.Tensor]:
        # Truncation is meaningless here: the aggregation below reads every proposal state and
        # overwrites the finest one, so a partial build would silently change d_0. Reject it
        # explicitly rather than letting a caller get a quietly different model.
        if stop_idx is not None:
            raise ValueError("FinestAgg aggregates all proposal states; stop_idx is unsupported")
        proposal_states = super()._build_proposal_states(skips)
        proposal_states = list(proposal_states)
        proposal_states[-1] = self._aggregate_finest_proposal_state(proposal_states)
        return proposal_states


class LiteRBUNeXt3DFeatureRefinerFinestLowResContext(LiteRBUNeXt3DFeatureRefiner):
    """
    Finest-state recurrent refiner with one detached low-resolution context.

    This keeps the successful finest-only recurrence path intact:
        d_f^{t+1} = R(d_f^t, d_f^0, skip_f, context_low)

    Unlike FinestAgg, it does not upsample and average all decoder states into
    d_f^0. The proposal d_f^0 and mask head stay unchanged. A single coarser
    proposal state is provided to the finest refiner as contextual guidance.
    """

    def __init__(
        self,
        *args,
        lowres_context_index: int = -2,
        lowres_context_detach: bool = True,
        hidden_channels: int = 32,
        alpha: float = 0.5,
        alpha_decay: bool = False,
        checkpoint_refiner: bool = False,
        initial_state_drop_prob: float = 0.0,
        update_clip_ratio: float | None = None,
        **kwargs,
    ):
        kwargs["refine_scales"] = "finest"
        super().__init__(
            *args,
            hidden_channels=hidden_channels,
            alpha=alpha,
            alpha_decay=alpha_decay,
            checkpoint_refiner=checkpoint_refiner,
            initial_state_drop_prob=initial_state_drop_prob,
            update_clip_ratio=update_clip_ratio,
            **kwargs,
        )
        num_decoder_states = len(self.channels) - 1
        if lowres_context_index < 0:
            lowres_context_index = num_decoder_states + int(lowres_context_index)
        if not 0 <= int(lowres_context_index) < num_decoder_states - 1:
            raise ValueError(
                f"lowres_context_index must select a non-finest decoder state in [0, {num_decoder_states - 2}]"
            )
        self.lowres_context_index = int(lowres_context_index)
        self.lowres_context_detach = bool(lowres_context_detach)

        state_channels = [self.channels[-(idx + 2)] for idx in range(num_decoder_states)]
        context_channels = state_channels[self.lowres_context_index]
        shared_refiner = self.finest_refiner.shared_block
        self.finest_refiner = FeatureRefinerStage3D(
            state_channels=self.channels[0],
            skip_channels=self.channels[0],
            hidden_channels=hidden_channels,
            shared_block=shared_refiner,
            alpha=alpha,
            alpha_decay=alpha_decay,
            coarse_context_channels=context_channels,
            use_checkpoint=checkpoint_refiner,
            initial_state_drop_prob=initial_state_drop_prob,
            update_clip_ratio=update_clip_ratio,
        )

    def _refine_finest_once(
        self,
        states: list[torch.Tensor],
        proposal_states: list[torch.Tensor],
        skips: list[torch.Tensor],
        step_idx: int,
    ) -> list[torch.Tensor]:
        if self.finest_refiner is None:
            raise RuntimeError("finest_refiner is only available when refine_scales='finest'")
        updated_states = list(states)
        finest_idx = len(states) - 1
        skip = skips[-(finest_idx + 2)]
        lowres_context = proposal_states[self.lowres_context_index]
        if self.lowres_context_detach:
            lowres_context = lowres_context.detach()
        updated_states[finest_idx] = self.finest_refiner(
            states[finest_idx],
            proposal_states[finest_idx],
            skip,
            coarse_context=lowres_context,
            step_idx=step_idx,
        )
        return updated_states

class LiteRBUNeXt3DFeatureRefinerDropInitClipLogit(LiteRBUNeXt3DFeatureRefiner):
    """
    Separate experiment variant:
    - randomly drops the d0 condition branch during training
    - clips each feature update relative to the current feature norm
    - adds a recurrent logit correction head after feature refinement
    """


class _DeepSupervisionProxy:
    """nnUNet trainer compatibility for networks without a Decoder module."""

    def __init__(self, network: "LiteRBUNeXt3D"):
        self.network = network

    @property
    def deep_supervision(self) -> bool:
        return self.network.deep_supervision

    @deep_supervision.setter
    def deep_supervision(self, enabled: bool):
        self.network.deep_supervision = bool(enabled)


@dataclass(frozen=True)
class LiteRBUNeXt3DConfig:
    expansion: float = 2.0
    mlp_ratio: float = 1.0
    mixer_start_stage: int = 3
    mamba_bottleneck: bool = False
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    recurrent_bottleneck: bool = False
    unshared_recurrent_bottleneck: bool = False
    direct_residual_recurrent: bool = False
    randomize_direct_residual_training_steps: bool = False
    relaxation_recurrent: bool = False
    relaxation_alpha: float = 0.5
    mask_belief_recurrent: bool = False
    mask_belief_recurrent_stage: int = 4
    mask_belief_decoder_fusion: bool = False
    mask_belief_decoder_stage: int = 0
    mask_belief_randomize_training_steps: bool = False
    mask_belief_alpha: float = 0.5
    mask_belief_beta: float = 0.5
    gated_recurrent: bool = False
    recurrent_gate_init: float = 0.1
    recurrent_gate_per_step: bool = True
    conditioned_bottleneck_recurrent: bool = False
    conditioned_initial_block: bool = False
    recurrent_condition_init: float = 0.1
    recurrent_encoder_mixer: bool = False
    encoder_recurrent_mixer_stages: tuple[int, ...] = (3, 4)
    recurrent_decoder: bool = False
    decoder_recurrent_stages: tuple[int, ...] = (0, 1, 2, 3)
    recurrent_decoder_refine: bool = False
    decoder_refine_recurrent_stages: tuple[int, ...] = (0, 1, 2, 3)
    randomize_decoder_refine_training_steps: bool = False
    sdrr_decoder_refinement: bool = False
    sdrr_decoder_stages: tuple[int, ...] = (1, 2)
    sdrr_hidden_channels: int = 64
    sdrr_recurrent_steps: int = 3
    sdrr_alpha: float = 0.5
    sdrr_randomize_training_steps: bool = True
    sdrr_training_min_steps: int = 1
    sdrr_learnable_gamma: bool = False
    sdrr_gamma_init: float = 1.0
    sdrr_delta_clip_rho: float = 0.0
    sdrr_tbptt_keep_steps: int = 0
    sdrr_randomize_tbptt_cut: bool = False
    sdrr_alpha_rho: float = 1.0
    sdrr_update_mode: str = "residual"
    sdrr_lambda: float = 0.5
    independent_sdrr_refinement: bool = False
    independent_sdrr_depth: int = 2
    recurrent_steps: int = 3
    eta: float = 0.75
    max_encoder_blocks: int = 2
    max_decoder_blocks: int = 1
    deep_supervision: bool = False


class LiteRBUNeXt3D(nn.Module):
    """
    Lightweight 3D U-Net for the main Lite-RBUNeXt line.

    High-resolution stages use MBConv-style local modeling. Middle/low
    resolution stages add ShiftMLP-DW token mixing. Recurrent refinement can be
    placed either in the bottleneck or after decoder skip fusion for ablations.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        n_blocks_per_stage: Sequence[int],
        n_blocks_per_stage_decoder: Sequence[int],
        config: LiteRBUNeXt3DConfig | None = None,
    ):
        super().__init__()
        self.config = config or LiteRBUNeXt3DConfig()
        cfg = self.config
        self.num_classes = num_classes
        self.deep_supervision = cfg.deep_supervision

        channels = tuple(int(i) for i in channels)
        kernel_sizes = tuple(_as_3tuple(i) for i in kernel_sizes)
        strides = tuple(_as_3tuple(i) for i in strides)
        if not (len(channels) == len(kernel_sizes) == len(strides) == len(n_blocks_per_stage)):
            raise ValueError("channels, kernel_sizes, strides and n_blocks_per_stage must have the same length")
        if len(channels) < 2:
            raise ValueError("LiteRBUNeXt3D requires at least two stages")

        self.channels = channels
        self.kernel_sizes = kernel_sizes
        self.strides = strides

        encoder_stages = []
        mixers = []
        encoder_recurrent_mixer_stages = {int(i) for i in cfg.encoder_recurrent_mixer_stages}
        in_ch = input_channels

        def make_shared_recurrent(block: nn.Module, conditioned: bool = False) -> nn.Module:
            if conditioned:
                return ConditionalGatedResidualRecurrent3D(
                    block,
                    cfg.recurrent_steps,
                    cfg.recurrent_gate_init,
                    cfg.recurrent_condition_init,
                    cfg.recurrent_gate_per_step,
                    cfg.conditioned_initial_block,
                )
            if cfg.relaxation_recurrent:
                return RelaxationRecurrent3D(
                    block,
                    cfg.recurrent_steps,
                    cfg.relaxation_alpha,
                    cfg.randomize_direct_residual_training_steps,
                )
            if cfg.direct_residual_recurrent:
                return DirectResidualRecurrent3D(
                    block,
                    cfg.recurrent_steps,
                    cfg.randomize_direct_residual_training_steps,
                )
            if cfg.gated_recurrent:
                return GatedResidualRecurrent3D(
                    block,
                    cfg.recurrent_steps,
                    cfg.recurrent_gate_init,
                    cfg.recurrent_gate_per_step,
                )
            return RecurrentBottleneck3D(block, cfg.recurrent_steps, cfg.eta)

        for stage_idx, out_ch in enumerate(channels):
            n_blocks = min(int(n_blocks_per_stage[stage_idx]), cfg.max_encoder_blocks)
            encoder_stages.append(
                LiteEncoderStage3D(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_sizes[stage_idx],
                    stride=strides[stage_idx],
                    n_blocks=max(1, n_blocks),
                    expansion=cfg.expansion,
                )
            )

            if stage_idx == len(channels) - 1:
                def make_bottleneck() -> nn.Module:
                    if cfg.mamba_bottleneck:
                        return MambaBottleneck3D(
                            out_ch,
                            d_state=cfg.mamba_d_state,
                            d_conv=cfg.mamba_d_conv,
                            expand=cfg.mamba_expand,
                        )
                    return ResidualShiftMLPDW3D(out_ch, mlp_ratio=cfg.mlp_ratio)

                if cfg.unshared_recurrent_bottleneck:
                    mixers.append(
                        UnsharedRecurrentBottleneck3D(
                            [make_bottleneck() for _ in range(cfg.recurrent_steps)],
                            cfg.eta,
                        )
                    )
                elif cfg.recurrent_bottleneck:
                    mixers.append(
                        make_shared_recurrent(
                            make_bottleneck(),
                            conditioned=cfg.conditioned_bottleneck_recurrent,
                        )
                    )
                else:
                    mixers.append(make_bottleneck())
            elif stage_idx >= cfg.mixer_start_stage:
                mixer = ResidualShiftMLPDW3D(out_ch, mlp_ratio=cfg.mlp_ratio)
                if cfg.mask_belief_recurrent and stage_idx == int(cfg.mask_belief_recurrent_stage):
                    mixer = MaskBeliefGuidedRecurrent3D(
                        mixer,
                        out_ch,
                        num_classes,
                        cfg.recurrent_steps,
                        cfg.mask_belief_alpha,
                        cfg.mask_belief_beta,
                        cfg.mlp_ratio,
                        cfg.mask_belief_randomize_training_steps,
                    )
                elif cfg.recurrent_encoder_mixer and stage_idx in encoder_recurrent_mixer_stages:
                    mixer = make_shared_recurrent(mixer)
                mixers.append(mixer)
            else:
                mixers.append(nn.Identity())
            in_ch = out_ch

        decoder_stages = []
        seg_layers = []
        decoder_mask_belief = []
        decoder_sdrr = []
        decoder_block_counts = list(n_blocks_per_stage_decoder)
        if len(decoder_block_counts) != len(channels) - 1:
            raise ValueError("n_blocks_per_stage_decoder must have one fewer entry than channels")
        decoder_recurrent_stages = {int(i) for i in cfg.decoder_recurrent_stages}
        decoder_refine_recurrent_stages = {int(i) for i in cfg.decoder_refine_recurrent_stages}
        sdrr_decoder_stages = {int(i) for i in cfg.sdrr_decoder_stages}
        sdrr_update_mode = str(cfg.sdrr_update_mode).strip().lower()
        if sdrr_update_mode not in {"residual", "alternating_pull"}:
            raise ValueError(f"Unsupported sdrr_update_mode: {cfg.sdrr_update_mode}")
        if cfg.sdrr_decoder_refinement:
            sdrr_shared_block = (
                AlternatingDecoderRefinementBlock3D(cfg.sdrr_hidden_channels)
                if sdrr_update_mode == "alternating_pull"
                else SharedDecoderRefinementBlock3D(cfg.sdrr_hidden_channels)
            )
        else:
            sdrr_shared_block = None
        for decoder_idx in range(len(channels) - 1):
            in_ch = channels[-(decoder_idx + 1)]
            skip_ch = channels[-(decoder_idx + 2)]
            kernel = kernel_sizes[-(decoder_idx + 2)]
            n_blocks = min(int(decoder_block_counts[decoder_idx]), cfg.max_decoder_blocks)
            refine_recurrent_steps = (
                cfg.recurrent_steps
                if cfg.recurrent_decoder_refine and decoder_idx in decoder_refine_recurrent_stages
                else 1
            )
            decoder_recurrent_steps = (
                cfg.recurrent_steps if cfg.recurrent_decoder and decoder_idx in decoder_recurrent_stages else 0
            )
            decoder_stages.append(
                LiteDecoderStage3D(
                    in_ch,
                    skip_ch,
                    kernel_size=kernel,
                    n_blocks=max(1, n_blocks),
                    expansion=cfg.expansion,
                    refine_recurrent_steps=refine_recurrent_steps,
                    randomize_refine_training_steps=cfg.randomize_decoder_refine_training_steps,
                    recurrent_steps=decoder_recurrent_steps,
                    eta=cfg.eta,
                    mlp_ratio=cfg.mlp_ratio,
                )
            )
            if cfg.mask_belief_decoder_fusion and decoder_idx == int(cfg.mask_belief_decoder_stage):
                decoder_mask_belief.append(
                    MaskBeliefGuidedRecurrent3D(
                        nn.Identity(),
                        skip_ch,
                        num_classes,
                        cfg.recurrent_steps,
                        cfg.mask_belief_alpha,
                        cfg.mask_belief_beta,
                        cfg.mlp_ratio,
                        cfg.mask_belief_randomize_training_steps,
                    )
                )
            else:
                decoder_mask_belief.append(nn.Identity())
            if cfg.independent_sdrr_refinement and decoder_idx in sdrr_decoder_stages:
                decoder_sdrr.append(
                    IndependentDecoderRecurrentRefinement3D(
                        skip_ch,
                        num_classes,
                        depth=cfg.independent_sdrr_depth,
                        recurrent_steps=cfg.sdrr_recurrent_steps,
                        lambda_value=cfg.sdrr_lambda,
                        tbptt_keep_steps=cfg.sdrr_tbptt_keep_steps,
                    )
                )
            elif cfg.sdrr_decoder_refinement and decoder_idx in sdrr_decoder_stages:
                if sdrr_shared_block is None:
                    raise RuntimeError("sdrr_shared_block was not initialized")
                if sdrr_update_mode == "alternating_pull":
                    decoder_sdrr.append(
                        AlternatingDecoderRecurrentRefinement3D(
                            skip_ch,
                            cfg.sdrr_hidden_channels,
                            num_classes,
                            sdrr_shared_block,
                            recurrent_steps=cfg.sdrr_recurrent_steps,
                            lambda_value=cfg.sdrr_lambda,
                            tbptt_keep_steps=cfg.sdrr_tbptt_keep_steps,
                        )
                    )
                else:
                    decoder_sdrr.append(
                        SharedDecoderRecurrentRefinement3D(
                            skip_ch,
                            cfg.sdrr_hidden_channels,
                            num_classes,
                            sdrr_shared_block,
                            cfg.sdrr_recurrent_steps,
                            cfg.sdrr_alpha,
                            cfg.sdrr_randomize_training_steps,
                            cfg.sdrr_training_min_steps,
                            cfg.sdrr_learnable_gamma,
                            cfg.sdrr_gamma_init,
                            cfg.sdrr_delta_clip_rho,
                            tbptt_keep_steps=cfg.sdrr_tbptt_keep_steps,
                            randomize_tbptt_cut=cfg.sdrr_randomize_tbptt_cut,
                            alpha_rho=cfg.sdrr_alpha_rho,
                        )
                    )
            else:
                decoder_sdrr.append(nn.Identity())
            seg_layers.append(nn.Conv3d(skip_ch, num_classes, kernel_size=1, bias=True))

        self.encoder_stages = nn.ModuleList(encoder_stages)
        self.mixers = nn.ModuleList(mixers)
        self.decoder_stages = nn.ModuleList(decoder_stages)
        self.decoder_mask_belief = nn.ModuleList(decoder_mask_belief)
        self.decoder_sdrr = nn.ModuleList(decoder_sdrr)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.sdrr_decoder_stages = sdrr_decoder_stages
        self.sdrr_update_mode = sdrr_update_mode
        self.conditioned_bottleneck_recurrent = cfg.conditioned_bottleneck_recurrent
        self.decoder = _DeepSupervisionProxy(self)

    @property
    def sdrr_shared_block(self) -> nn.Module | None:
        for module in self.decoder_sdrr:
            if hasattr(module, "shared_block"):
                return module.shared_block
        return None

    def _maybe_add_bottleneck_noise(self, x: torch.Tensor) -> torch.Tensor:
        ablate_mode = os.environ.get("LITE_BOTTLENECK_ABLATE", "").strip().lower()
        if ablate_mode:
            if ablate_mode == "zero":
                return torch.zeros_like(x)
            raise ValueError(f"Unknown LITE_BOTTLENECK_ABLATE mode: {ablate_mode}")

        noise_std = float(os.environ.get("LITE_BOTTLENECK_NOISE_STD", "0"))
        if noise_std <= 0:
            return x

        seed = os.environ.get("LITE_BOTTLENECK_NOISE_SEED")
        generator = None
        if seed is not None:
            if not hasattr(self, "_bottleneck_noise_calls"):
                self._bottleneck_noise_calls = 0
            generator = torch.Generator(device=x.device)
            generator.manual_seed(int(seed) + int(self._bottleneck_noise_calls))
            self._bottleneck_noise_calls += 1

        scale = x.detach().float().flatten(1).std(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1e-6)
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
        return x + noise * scale.to(dtype=x.dtype) * noise_std

    @staticmethod
    def _stage_selector_contains(selector: str, idx: int) -> bool:
        selector = selector.strip().lower()
        if not selector:
            return False
        if selector == "all":
            return True
        if selector == "top":
            return idx == 0
        return str(idx) in {item.strip() for item in selector.split(",") if item.strip()}

    def _maybe_perturb_skip(self, skip: torch.Tensor, decoder_idx: int) -> torch.Tensor:
        ablate = os.environ.get("LITE_SKIP_ABLATE", "")
        if self._stage_selector_contains(ablate, decoder_idx):
            return torch.zeros_like(skip)

        dropout_p = float(os.environ.get("LITE_SKIP_DROPOUT_P", "0"))
        dropout_stages = os.environ.get("LITE_SKIP_DROPOUT_STAGES", "top")
        dropout_mode = os.environ.get("LITE_SKIP_DROPOUT_MODE", "voxel").strip().lower()
        dropout_scale = os.environ.get("LITE_SKIP_DROPOUT_SCALE", "1").strip().lower() not in {"0", "false", "no"}
        if dropout_p <= 0 or not self._stage_selector_contains(dropout_stages, decoder_idx):
            return skip
        if not 0 <= dropout_p < 1:
            raise ValueError("LITE_SKIP_DROPOUT_P must be in [0, 1)")
        if dropout_mode not in {"voxel", "channel", "path"}:
            raise ValueError("LITE_SKIP_DROPOUT_MODE must be one of: voxel, channel, path")

        seed = os.environ.get("LITE_SKIP_DROPOUT_SEED")
        generator = None
        if seed is not None:
            if not hasattr(self, "_skip_dropout_calls"):
                self._skip_dropout_calls = 0
            generator = torch.Generator(device=skip.device)
            generator.manual_seed(int(seed) + int(self._skip_dropout_calls))
            self._skip_dropout_calls += 1

        if dropout_mode == "path":
            mask_shape = (skip.shape[0], 1, 1, 1, 1)
        elif dropout_mode == "channel":
            mask_shape = (skip.shape[0], skip.shape[1], 1, 1, 1)
        else:
            mask_shape = skip.shape

        mask = torch.rand(mask_shape, device=skip.device, dtype=skip.dtype, generator=generator) >= dropout_p
        skip = skip * mask.to(dtype=skip.dtype)
        if dropout_scale:
            skip = skip / (1.0 - dropout_p)
        return skip

    def forward(
        self,
        x: torch.Tensor,
        recurrent_steps: int | None = None,
        sdrr_tbptt_keep_steps: int | None = None,
        detach_after_last_sdrr: bool = False,
    ):
        input_shape = x.shape[2:]
        skips = []
        last_stage_idx = len(self.encoder_stages) - 1
        encoder_context = torch.no_grad() if detach_after_last_sdrr else nullcontext()
        with encoder_context:
            for stage_idx, (stage, mixer) in enumerate(zip(self.encoder_stages, self.mixers)):
                raw = stage(x)
                if (
                    stage_idx == last_stage_idx
                    and self.conditioned_bottleneck_recurrent
                    and isinstance(mixer, ConditionalGatedResidualRecurrent3D)
                ):
                    x = mixer(raw, raw)
                else:
                    x = mixer(raw)
                skips.append(x)

        x = skips[-1]
        x = self._maybe_add_bottleneck_noise(x)
        seg_outputs = []
        last_sdrr_idx = max(self.sdrr_decoder_stages) if self.sdrr_decoder_stages else -1
        for idx, decoder in enumerate(self.decoder_stages):
            detached_stage = detach_after_last_sdrr and idx <= last_sdrr_idx
            stage_context = torch.no_grad() if detached_stage else nullcontext()
            with stage_context:
                skip = skips[-(idx + 2)]
                skip = self._maybe_perturb_skip(skip, idx)
                if isinstance(self.decoder_mask_belief[idx], nn.Identity):
                    x = decoder(x, skip)
                else:
                    x = decoder.fuse_skip(x, skip)
                    x = self.decoder_mask_belief[idx](x)
                    x = decoder.refine_fused(x)
                sdrr_module = self.decoder_sdrr[idx]
                if isinstance(sdrr_module, (AlternatingDecoderRecurrentRefinement3D,
                                            IndependentDecoderRecurrentRefinement3D)):
                    x = sdrr_module(
                        x,
                        recurrent_steps=recurrent_steps,
                        tbptt_keep_steps=sdrr_tbptt_keep_steps,
                    )
                else:
                    x = sdrr_module(x)
                is_last = idx == len(self.decoder_stages) - 1
                if is_last and x.shape[2:] != input_shape:
                    x = F.interpolate(x, size=input_shape, mode="nearest")
                if self.deep_supervision or is_last:
                    seg_outputs.append(self.seg_layers[idx](x))
            if detach_after_last_sdrr and idx == last_sdrr_idx:
                x = x.detach()
                skips = [skip_tensor.detach() for skip_tensor in skips]

        seg_outputs = seg_outputs[::-1]
        if self.deep_supervision:
            return seg_outputs
        return seg_outputs[0]


class LiteRBUNeXt3DProposalD8D4Refiner(LiteRBUNeXt3D):
    """
    Lite-UNeXt3D proposal decoder followed by a two-scale recurrent refiner.

    The full Lite-UNeXt3D encoder/decoder first produces proposal decoder states.
    Only the two finest decoder states are recurrently updated:

    d8^{t+1} = R8(d8^t, d8^0, skip8, Up(d16^0))
    d4^{t+1} = R4(d4^t, d4^0, skip4, Up(d8^{t+1}))
    logits    = H(d4)

    K=0 is exactly the proposal decoder output. For K>0, R8 receives direct
    supervision through the current-step R4 path because d4 uses d8^{t+1}.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        n_blocks_per_stage: Sequence[int],
        n_blocks_per_stage_decoder: Sequence[int],
        config: LiteRBUNeXt3DConfig | None = None,
        *,
        hidden_channels: int = 32,
        recurrent_steps: int = 4,
        alpha: float = 0.5,
        alpha_decay: bool = False,
        randomize_training_steps: bool = True,
        training_min_steps: int = 1,
        tbptt_keep_steps: int = 2,
        checkpoint_refiner: bool = False,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            n_blocks_per_stage_decoder=n_blocks_per_stage_decoder,
            config=config,
        )
        if len(self.channels) < 4:
            raise ValueError("LiteRBUNeXt3DProposalD8D4Refiner requires at least four decoder states")
        if recurrent_steps < 1:
            raise ValueError("recurrent_steps must be >= 1")
        if not 1 <= training_min_steps <= recurrent_steps:
            raise ValueError("training_min_steps must be in [1, recurrent_steps]")
        if tbptt_keep_steps < 1:
            raise ValueError("tbptt_keep_steps must be >= 1")

        self.recurrent_steps = int(recurrent_steps)
        self.randomize_training_steps = bool(randomize_training_steps)
        self.training_min_steps = int(training_min_steps)
        self.tbptt_keep_steps = int(tbptt_keep_steps)
        self.alpha_decay = bool(alpha_decay)
        self.checkpoint_refiner = bool(checkpoint_refiner)

        state_channels = [self.channels[-(idx + 2)] for idx in range(len(self.channels) - 1)]
        self.d8_state_idx = len(state_channels) - 2
        self.d4_state_idx = len(state_channels) - 1
        self.d16_state_idx = len(state_channels) - 3
        d8_channels = state_channels[self.d8_state_idx]
        d4_channels = state_channels[self.d4_state_idx]
        d16_channels = state_channels[self.d16_state_idx]

        shared_refiner = SharedFeatureRefinementBlock3D(hidden_channels, expansion=1.0)
        self.refiner_d8 = FeatureRefinerStage3D(
            state_channels=d8_channels,
            skip_channels=d8_channels,
            hidden_channels=hidden_channels,
            shared_block=shared_refiner,
            alpha=alpha,
            alpha_decay=alpha_decay,
            coarse_context_channels=d16_channels,
            use_checkpoint=checkpoint_refiner,
        )
        self.refiner_d4 = FeatureRefinerStage3D(
            state_channels=d4_channels,
            skip_channels=d4_channels,
            hidden_channels=hidden_channels,
            shared_block=shared_refiner,
            alpha=alpha,
            alpha_decay=alpha_decay,
            coarse_context_channels=d8_channels,
            use_checkpoint=checkpoint_refiner,
        )

        self.last_all_step_logits: list[torch.Tensor] = []
        self.last_all_step_indices: list[int] = []
        self.last_active_step_indices: list[int] = []
        self.last_step_metrics: dict[str, list[float] | float | int] = {
            "r_d": [],
            "r_m": [],
            "r_d8": [],
            "r_d4": [],
            "K": 0,
            "T_cut": 0,
        }

    def _get_recurrent_steps(self, x: torch.Tensor, recurrent_steps: int | None) -> int:
        if recurrent_steps is not None:
            steps = int(recurrent_steps)
        elif "LITE_REFINER_TEST_STEPS" in os.environ:
            steps = int(os.environ["LITE_REFINER_TEST_STEPS"])
        elif "LITE_SDRR_TEST_STEPS" in os.environ:
            steps = int(os.environ["LITE_SDRR_TEST_STEPS"])
        elif "LITE_RECURRENT_TEST_STEPS" in os.environ:
            steps = int(os.environ["LITE_RECURRENT_TEST_STEPS"])
        elif self.training and self.randomize_training_steps:
            steps = int(torch.randint(self.training_min_steps, self.recurrent_steps + 1, (1,), device=x.device).item())
        else:
            steps = self.recurrent_steps
        if steps < 0:
            raise ValueError("recurrent_steps must be >= 0")
        return steps

    @staticmethod
    def _relative_norm_delta(new_tensor: torch.Tensor, old_tensor: torch.Tensor) -> float:
        base = old_tensor.detach().float().norm().clamp_min(1e-8)
        delta = (new_tensor.detach().float() - old_tensor.detach().float()).norm()
        return float((delta / base).cpu())

    @staticmethod
    def _prob_from_logits(logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        skips = []
        last_stage_idx = len(self.encoder_stages) - 1
        for stage_idx, (stage, mixer) in enumerate(zip(self.encoder_stages, self.mixers)):
            raw = stage(x)
            if (
                stage_idx == last_stage_idx
                and self.conditioned_bottleneck_recurrent
                and isinstance(mixer, ConditionalGatedResidualRecurrent3D)
            ):
                x = mixer(raw, raw)
            else:
                x = mixer(raw)
            skips.append(x)
        return skips

    def _build_lite_unext_proposal_states(self, skips: list[torch.Tensor]) -> list[torch.Tensor]:
        x = self._maybe_add_bottleneck_noise(skips[-1])
        states = []
        for idx, decoder in enumerate(self.decoder_stages):
            skip = skips[-(idx + 2)]
            skip = self._maybe_perturb_skip(skip, idx)
            if isinstance(self.decoder_mask_belief[idx], nn.Identity):
                x = decoder(x, skip)
            else:
                x = decoder.fuse_skip(x, skip)
                x = self.decoder_mask_belief[idx](x)
                x = decoder.refine_fused(x)
            x = self.decoder_sdrr[idx](x)
            states.append(x)
        return states

    def _decode_logits(self, d4_state: torch.Tensor, output_size: Sequence[int]) -> torch.Tensor:
        if d4_state.shape[2:] != tuple(output_size):
            d4_state = F.interpolate(d4_state, size=output_size, mode="nearest")
        return self.seg_layers[-1](d4_state)

    def _refine_d8_d4_once(
        self,
        states: list[torch.Tensor],
        proposal_states: list[torch.Tensor],
        skips: list[torch.Tensor],
        step_idx: int,
    ) -> list[torch.Tensor]:
        updated_states = list(states)

        d8_idx = self.d8_state_idx
        d4_idx = self.d4_state_idx
        d16_idx = self.d16_state_idx
        skip8 = skips[-(d8_idx + 2)]
        skip4 = skips[-(d4_idx + 2)]

        d8_next = self.refiner_d8(
            states[d8_idx],
            proposal_states[d8_idx],
            skip8,
            coarse_context=proposal_states[d16_idx],
            step_idx=step_idx,
        )
        d4_next = self.refiner_d4(
            states[d4_idx],
            proposal_states[d4_idx],
            skip4,
            coarse_context=d8_next,
            step_idx=step_idx,
        )
        updated_states[d8_idx] = d8_next
        updated_states[d4_idx] = d4_next
        return updated_states

    def forward(
        self,
        x: torch.Tensor,
        recurrent_steps: int | None = None,
        tbptt_keep_steps: int | None = None,
    ):
        input_shape = x.shape[2:]
        recurrent_steps = self._get_recurrent_steps(x, recurrent_steps)
        keep_steps = self.tbptt_keep_steps if tbptt_keep_steps is None else int(tbptt_keep_steps)
        if keep_steps < 1:
            raise ValueError("tbptt_keep_steps must be >= 1")

        skips = self._encode(x)
        proposal_states = self._build_lite_unext_proposal_states(skips)
        states = [state for state in proposal_states]

        base_logits = self._decode_logits(states[-1], input_shape)
        all_step_logits = [base_logits]
        all_step_indices = [0]
        metrics_r_d: list[float] = []
        metrics_r_m: list[float] = []
        metrics_r_d8: list[float] = []
        metrics_r_d4: list[float] = []
        prev_logits = base_logits

        if self.training:
            t_cut = max(0, recurrent_steps - keep_steps)
            for step_idx in range(t_cut):
                with torch.no_grad():
                    prev_d8 = states[self.d8_state_idx]
                    prev_d4 = states[self.d4_state_idx]
                    states = self._refine_d8_d4_once(states, proposal_states, skips, step_idx=step_idx)
                    current_logits = self._decode_logits(states[-1], input_shape)
                    r_d8 = self._relative_norm_delta(states[self.d8_state_idx], prev_d8)
                    r_d4 = self._relative_norm_delta(states[self.d4_state_idx], prev_d4)
                    metrics_r_d8.append(r_d8)
                    metrics_r_d4.append(r_d4)
                    metrics_r_d.append(r_d4)
                    metrics_r_m.append(
                        self._relative_norm_delta(self._prob_from_logits(current_logits), self._prob_from_logits(prev_logits))
                    )
                    prev_logits = current_logits
            if t_cut > 0:
                states = [state.detach() for state in states]

            for step_idx in range(t_cut, recurrent_steps):
                prev_d8 = states[self.d8_state_idx]
                prev_d4 = states[self.d4_state_idx]
                states = self._refine_d8_d4_once(states, proposal_states, skips, step_idx=step_idx)
                current_logits = self._decode_logits(states[-1], input_shape)
                r_d8 = self._relative_norm_delta(states[self.d8_state_idx], prev_d8)
                r_d4 = self._relative_norm_delta(states[self.d4_state_idx], prev_d4)
                metrics_r_d8.append(r_d8)
                metrics_r_d4.append(r_d4)
                metrics_r_d.append(r_d4)
                metrics_r_m.append(
                    self._relative_norm_delta(self._prob_from_logits(current_logits), self._prob_from_logits(prev_logits))
                )
                prev_logits = current_logits
                all_step_logits.append(current_logits)
                all_step_indices.append(step_idx + 1)
            active_step_indices = [0] + list(range(t_cut + 1, recurrent_steps + 1))
        else:
            t_cut = 0
            for step_idx in range(recurrent_steps):
                prev_d8 = states[self.d8_state_idx]
                prev_d4 = states[self.d4_state_idx]
                states = self._refine_d8_d4_once(states, proposal_states, skips, step_idx=step_idx)
                current_logits = self._decode_logits(states[-1], input_shape)
                r_d8 = self._relative_norm_delta(states[self.d8_state_idx], prev_d8)
                r_d4 = self._relative_norm_delta(states[self.d4_state_idx], prev_d4)
                metrics_r_d8.append(r_d8)
                metrics_r_d4.append(r_d4)
                metrics_r_d.append(r_d4)
                metrics_r_m.append(
                    self._relative_norm_delta(self._prob_from_logits(current_logits), self._prob_from_logits(prev_logits))
                )
                prev_logits = current_logits
                all_step_logits.append(current_logits)
                all_step_indices.append(step_idx + 1)
            active_step_indices = list(range(recurrent_steps + 1))

        final_logits = all_step_logits[-1]
        self.last_all_step_logits = all_step_logits
        self.last_all_step_indices = all_step_indices
        self.last_active_step_indices = active_step_indices
        self.last_step_metrics = {
            "r_d": metrics_r_d,
            "r_m": metrics_r_m,
            "r_d8": metrics_r_d8,
            "r_d4": metrics_r_d4,
            "K": recurrent_steps,
            "T_cut": t_cut,
        }
        if self.deep_supervision:
            return [final_logits]
        return final_logits


class LiteRBUNeXt3DSDRRProposalRecursiveLogit(LiteRBUNeXt3DProposalD8D4Refiner):
    """
    Proposal-mask-logit + recurrent logit update with an SDRR decoder proposal.

    The proposal pass is the regular Lite-UNeXt3D decoder with SDRR modules in
    the decoder when enabled in LiteRBUNeXt3DConfig. The recurrent part then
    refines the finest proposal feature with the SDRR-style feature refiner and
    accumulates bounded delta logits on top of the fixed proposal mask logit.
    Use LITE_REFINER_TEST_STEPS for test-time scaling; LITE_SDRR_TEST_STEPS can
    be left unset so the proposal SDRR keeps its configured inference depth.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[Sequence[int]],
        strides: Sequence[Sequence[int]],
        n_blocks_per_stage: Sequence[int],
        n_blocks_per_stage_decoder: Sequence[int],
        config: LiteRBUNeXt3DConfig | None = None,
        *,
        hidden_channels: int = 32,
        recurrent_steps: int = 4,
        alpha: float = 0.5,
        alpha_decay: bool = True,
        randomize_training_steps: bool = True,
        training_min_steps: int = 1,
        tbptt_keep_steps: int = 2,
        update_clip_ratio: float | None = 0.15,
        logit_refine_alpha: float = 0.20,
        checkpoint_refiner: bool = False,
        belief_guided: bool = False,
        belief_gate_floor: float = 0.5,
    ):
        super().__init__(
            input_channels=input_channels,
            num_classes=num_classes,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            n_blocks_per_stage_decoder=n_blocks_per_stage_decoder,
            config=config,
            hidden_channels=hidden_channels,
            recurrent_steps=recurrent_steps,
            alpha=alpha,
            alpha_decay=alpha_decay,
            randomize_training_steps=randomize_training_steps,
            training_min_steps=training_min_steps,
            tbptt_keep_steps=tbptt_keep_steps,
            checkpoint_refiner=checkpoint_refiner,
        )
        for module in (self.refiner_d8, self.refiner_d4):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        shared_refiner = SharedDecoderRefinementBlock3D(hidden_channels)
        self.recursive_finest_refiner = SDRRFeatureRefinerStage3D(
            state_channels=self.channels[0],
            skip_channels=self.channels[0],
            hidden_channels=hidden_channels,
            shared_block=shared_refiner,
            alpha=alpha,
            alpha_decay=alpha_decay,
            use_checkpoint=checkpoint_refiner,
            update_clip_ratio=update_clip_ratio,
        )
        num_groups = min(8, self.channels[0])
        while self.channels[0] % num_groups != 0:
            num_groups -= 1
        self.logit_refine_alpha = float(logit_refine_alpha)
        if not 0 <= belief_gate_floor <= 1:
            raise ValueError("belief_gate_floor must be in [0, 1]")
        self.belief_guided = bool(belief_guided)
        self.belief_gate_floor = float(belief_gate_floor)
        self.logit_refine_head = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=self.channels[0]),
            nn.Conv3d(self.channels[0], self.channels[0], kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(self.channels[0], num_classes, kernel_size=1, bias=True),
        )
        if self.belief_guided:
            self.belief_head = nn.Sequential(
                nn.GroupNorm(num_groups=num_groups, num_channels=self.channels[0]),
                nn.Conv3d(self.channels[0], self.channels[0], kernel_size=1, bias=False),
                nn.GELU(),
                nn.Conv3d(self.channels[0], num_classes, kernel_size=1, bias=True),
            )
        else:
            self.belief_head = None
        self.last_belief_logits: list[torch.Tensor] = []
        self.last_belief_step_indices: list[int] = []

    def _refine_finest_once(
        self,
        states: list[torch.Tensor],
        proposal_states: list[torch.Tensor],
        skips: list[torch.Tensor],
        step_idx: int,
    ) -> list[torch.Tensor]:
        updated_states = list(states)
        finest_idx = len(states) - 1
        skip = skips[-(finest_idx + 2)]
        updated_states[finest_idx] = self.recursive_finest_refiner(
            states[finest_idx],
            proposal_states[finest_idx],
            skip,
            step_idx=step_idx,
        )
        return updated_states

    def _decode_logit_update(self, finest_state: torch.Tensor, output_size: Sequence[int]) -> torch.Tensor:
        delta_logits = self.logit_refine_head(finest_state)
        if delta_logits.shape[2:] != tuple(output_size):
            delta_logits = F.interpolate(delta_logits, size=output_size, mode="nearest")
        return torch.tanh(delta_logits)

    def _decode_belief_logits(self, finest_state: torch.Tensor, output_size: Sequence[int]) -> torch.Tensor:
        if self.belief_head is None:
            raise RuntimeError("belief_head is only available when belief_guided=True")
        belief_logits = self.belief_head(finest_state)
        if belief_logits.shape[2:] != tuple(output_size):
            belief_logits = F.interpolate(belief_logits, size=output_size, mode="nearest")
        return belief_logits

    def _compute_single_step_correction(
        self,
        finest_state: torch.Tensor,
        output_size: Sequence[int],
        step_idx: int,
        record_belief: bool = True,
    ) -> torch.Tensor:
        alpha = self.logit_refine_alpha / sqrt(float(step_idx + 1))
        delta_logits = self._decode_logit_update(finest_state, output_size)
        if self.belief_guided:
            belief_logits = self._decode_belief_logits(finest_state, output_size)
            gate = self.belief_gate_floor + (1.0 - self.belief_gate_floor) * torch.sigmoid(belief_logits)
            delta_logits = gate * delta_logits
            if record_belief:
                self.last_belief_logits.append(belief_logits)
                self.last_belief_step_indices.append(step_idx + 1)
        return alpha * delta_logits

    def forward(
        self,
        x: torch.Tensor,
        recurrent_steps: int | None = None,
        tbptt_keep_steps: int | None = None,
    ):
        input_shape = x.shape[2:]
        recurrent_steps = self._get_recurrent_steps(x, recurrent_steps)
        keep_steps = self.tbptt_keep_steps if tbptt_keep_steps is None else int(tbptt_keep_steps)
        if keep_steps < 1:
            raise ValueError("tbptt_keep_steps must be >= 1")
        self.last_belief_logits = []
        self.last_belief_step_indices = []

        skips = self._encode(x)
        proposal_states = self._build_lite_unext_proposal_states(skips)
        states = [state for state in proposal_states]

        mask_logit = self._decode_logits(states[-1], input_shape)
        all_step_logits = [mask_logit]
        all_step_indices = [0]
        metrics_r_d: list[float] = []
        metrics_r_m: list[float] = []
        prev_logits = mask_logit

        if self.training:
            t_cut = max(0, recurrent_steps - keep_steps)
            for step_idx in range(t_cut):
                with torch.no_grad():
                    prev_finest = states[-1]
                    states = self._refine_finest_once(states, proposal_states, skips, step_idx=step_idx)
                    mask_logit = mask_logit + self._compute_single_step_correction(
                        states[-1], input_shape, step_idx, record_belief=False
                    )
                    metrics_r_d.append(self._relative_norm_delta(states[-1], prev_finest))
                    metrics_r_m.append(
                        self._relative_norm_delta(
                            self._prob_from_logits(mask_logit),
                            self._prob_from_logits(prev_logits),
                        )
                    )
                    prev_logits = mask_logit
            if t_cut > 0:
                states = [state.detach() for state in states]
                mask_logit = mask_logit.detach()

            for step_idx in range(t_cut, recurrent_steps):
                prev_finest = states[-1]
                states = self._refine_finest_once(states, proposal_states, skips, step_idx=step_idx)
                mask_logit = mask_logit + self._compute_single_step_correction(
                    states[-1], input_shape, step_idx, record_belief=True
                )
                metrics_r_d.append(self._relative_norm_delta(states[-1], prev_finest))
                metrics_r_m.append(
                    self._relative_norm_delta(
                        self._prob_from_logits(mask_logit),
                        self._prob_from_logits(prev_logits),
                    )
                )
                prev_logits = mask_logit
                all_step_logits.append(mask_logit)
                all_step_indices.append(step_idx + 1)
            active_step_indices = [0] + list(range(t_cut + 1, recurrent_steps + 1))
        else:
            t_cut = 0
            for step_idx in range(recurrent_steps):
                prev_finest = states[-1]
                states = self._refine_finest_once(states, proposal_states, skips, step_idx=step_idx)
                mask_logit = mask_logit + self._compute_single_step_correction(
                    states[-1], input_shape, step_idx, record_belief=True
                )
                metrics_r_d.append(self._relative_norm_delta(states[-1], prev_finest))
                metrics_r_m.append(
                    self._relative_norm_delta(
                        self._prob_from_logits(mask_logit),
                        self._prob_from_logits(prev_logits),
                    )
                )
                prev_logits = mask_logit
                all_step_logits.append(mask_logit)
                all_step_indices.append(step_idx + 1)
            active_step_indices = list(range(recurrent_steps + 1))

        self.last_all_step_logits = all_step_logits
        self.last_all_step_indices = all_step_indices
        self.last_active_step_indices = active_step_indices
        self.last_step_metrics = {
            "r_d": metrics_r_d,
            "r_m": metrics_r_m,
            "K": recurrent_steps,
            "T_cut": t_cut,
        }
        if self.deep_supervision:
            return [mask_logit]
        return mask_logit


def make_lite_rbunext_channels(n_stages: int, base_features: int = 32, max_features: int = 320) -> tuple[int, ...]:
    """Stage channels for the lightweight main line.

    For Dataset811's six-stage 3d_fullres plan this gives:
    [32, 64, 128, 256, 320, 320].
    """

    channels = []
    for i in range(n_stages):
        if i <= 2:
            value = base_features * (2**i)
        elif i == 3:
            value = base_features * 8
        else:
            value = base_features * 10
        channels.append(min(int(value), int(max_features)))
    return tuple(channels)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LiteRBUNeXt3DFeatureRefinerRecursive(LiteRBUNeXt3DFeatureRefiner):
    """
    RecursiveLogit: Fixed proposal base + recursive logit accumulation

    Architecture:
      mask_logit_0 = mask_head(d^4_0)  ← fixed proposal base
      for t in 1..K:
          d^4_t = refine(d^4_{t-1})
          ctx_t = α_t·tanh(logit_refine_head(d^4_t))
          mask_logit_t = mask_logit_{t-1} + ctx_t  ← recursive accumulation

      final = mask_logit_K = mask_head(d^4_0) + Σ ctx_t

    Key differences from RegLogit:
      1. Fixed proposal: mask_head always applies to d^4_0
      2. No EMA: direct sum instead of exponential moving average (rho=0)
      3. Simpler: recursive accumulation on mask_logit itself
      4. Clearer semantics: iterative improvement from fixed base

    Hyperparameters:
      - logit_refine_alpha: 0.15 (lower than RegLogit's 0.25 to compensate for no decay)
      - logit_refine_rho: ignored (always 0 for direct accumulation)
      - tbptt_keep_steps: 2 (same as RegLogit)
    """

    def __init__(self, *args, detach_base: bool = False, **kwargs):
        # Force logit_refine=True and rho=0 (no EMA)
        kwargs['logit_refine'] = True
        kwargs['logit_refine_rho'] = 0.0  # No EMA, direct accumulation
        super().__init__(*args, **kwargs)
        # detach_base=True: stop-gradient the running mask_logit accumulator
        # before adding each step's correction. m_0's own loss term (all_step_logits[0])
        # is unaffected -- it always carries gradient to mask_head/proposal via its
        # own weight w_0. What changes is that m_t (t>=1) no longer back-propagates
        # into m_{t-1} (and transitively into mask_head/proposal/earlier corrections);
        # each c_t is trained as a pure residual fit against a frozen snapshot of the
        # running logits, instead of also being able to "help" by nudging the base.
        self.detach_base = bool(detach_base)

    def _maybe_detach(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.detach() if self.detach_base else logits

    def forward(
        self,
        x: torch.Tensor,
        recurrent_steps: int | None = None,
        tbptt_keep_steps: int | None = None,
    ):
        input_shape = x.shape[2:]
        recurrent_steps = self._get_recurrent_steps(x, recurrent_steps)
        keep_steps = self.tbptt_keep_steps if tbptt_keep_steps is None else int(tbptt_keep_steps)
        if keep_steps < 1:
            raise ValueError("tbptt_keep_steps must be >= 1")
        self._proposal_norm_mean = None
        self._proposal_norm_var = None

        # Encode + Proposal
        skips = self._encode(x)
        proposal_states = self._build_proposal_states(skips)
        states = [state for state in proposal_states]

        # Fixed base from proposal (never changes)
        base_logits = self._decode_logits(states[-1], input_shape)
        mask_logit = base_logits  # ← recursive accumulation starting point

        all_step_logits = [mask_logit]
        all_step_indices = [0]
        metrics_r_d: list[float] = []
        metrics_r_m: list[float] = []
        prev_logits = mask_logit

        if self.training:
            t_cut = max(0, recurrent_steps - keep_steps)

            # Phase 1: No gradient
            for step_idx in range(t_cut):
                with torch.no_grad():
                    prev_finest = states[-1]
                    states = self._refine_selected_once(states, proposal_states, skips, step_idx=step_idx)

                    # Compute single-step correction (no EMA)
                    ctx_t = self._compute_single_step_correction(states[-1], input_shape, step_idx)
                    mask_logit = self._maybe_detach(mask_logit) + ctx_t  # ← recursive accumulation

                    metrics_r_d.append(self._relative_norm_delta(states[-1], prev_finest))
                    metrics_r_m.append(self._relative_norm_delta(self._prob_from_logits(mask_logit), self._prob_from_logits(prev_logits)))
                    prev_logits = mask_logit

            # Detach after no-gradient phase
            if t_cut > 0:
                states = [state.detach() for state in states]
                mask_logit = mask_logit.detach()  # ← detach accumulated logits

            # Phase 2: Gradient
            for step_idx in range(t_cut, recurrent_steps):
                prev_finest = states[-1]
                states = self._refine_selected_once(states, proposal_states, skips, step_idx=step_idx)

                # Compute single-step correction (with gradient)
                ctx_t = self._compute_single_step_correction(states[-1], input_shape, step_idx)
                mask_logit = self._maybe_detach(mask_logit) + ctx_t  # ← recursive accumulation (with gradient)

                metrics_r_d.append(self._relative_norm_delta(states[-1], prev_finest))
                metrics_r_m.append(self._relative_norm_delta(self._prob_from_logits(mask_logit), self._prob_from_logits(prev_logits)))
                prev_logits = mask_logit
                all_step_logits.append(mask_logit)
                all_step_indices.append(step_idx + 1)

            active_step_indices = [0] + list(range(t_cut + 1, recurrent_steps + 1))
        else:
            # Inference: all steps
            t_cut = 0
            for step_idx in range(recurrent_steps):
                prev_finest = states[-1]
                states = self._refine_selected_once(states, proposal_states, skips, step_idx=step_idx)

                ctx_t = self._compute_single_step_correction(states[-1], input_shape, step_idx)
                mask_logit = mask_logit + ctx_t

                metrics_r_d.append(self._relative_norm_delta(states[-1], prev_finest))
                metrics_r_m.append(self._relative_norm_delta(self._prob_from_logits(mask_logit), self._prob_from_logits(prev_logits)))
                prev_logits = mask_logit
                all_step_logits.append(mask_logit)
                all_step_indices.append(step_idx + 1)
            active_step_indices = list(range(recurrent_steps + 1))

        final_logits = mask_logit
        self.last_all_step_logits = all_step_logits
        self.last_all_step_indices = all_step_indices
        self.last_active_step_indices = active_step_indices
        self.last_step_metrics = {
            "r_d": metrics_r_d,
            "r_m": metrics_r_m,
            "K": recurrent_steps,
            "T_cut": t_cut,
        }
        metrics_csv = os.environ.get("LITE_REFINER_METRICS_CSV")
        if metrics_csv:
            import csv

            os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
            row = {
                "K": recurrent_steps,
                "T_cut": t_cut,
                "n_steps": len(metrics_r_d),
                "r_d_mean": float(sum(metrics_r_d) / len(metrics_r_d)) if metrics_r_d else 0.0,
                "r_m_mean": float(sum(metrics_r_m) / len(metrics_r_m)) if metrics_r_m else 0.0,
                "r_d_steps": ";".join(f"{v:.8g}" for v in metrics_r_d),
                "r_m_steps": ";".join(f"{v:.8g}" for v in metrics_r_m),
            }
            write_header = not os.path.exists(metrics_csv)
            with open(metrics_csv, "a", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        if self.deep_supervision:
            return [final_logits]
        return final_logits

    def _compute_single_step_correction(
        self,
        finest_state: torch.Tensor,
        output_size: tuple[int, ...],
        step_idx: int,
    ) -> torch.Tensor:
        """
        Compute single-step logit correction (no EMA accumulation).

        Returns:
            ctx_t = α_t·tanh(logit_refine_head(d^4_t))
            where α_t = logit_refine_alpha / sqrt(step_idx + 1)
        """
        delta_logits = self._decode_logit_update(finest_state, output_size)
        # delta_logits is already tanh-bounded to [-1, 1]

        alpha = self.logit_refine_alpha / sqrt(float(step_idx + 1))
        return alpha * delta_logits  # ← single-step correction only


class LiteRBUNeXt3DFeatureRefinerBoundedPull(LiteRBUNeXt3DFeatureRefiner):
    """
    BoundedPull: use the full mask head at every recurrent step, but update the
    running logits with a bounded pull instead of replacing them.

    This keeps FinestOnly's high-capacity target logits H(d_t), while borrowing
    RecursiveLogit's stable recurrent logit accumulation:

      m_0 = H(d_0)
      z_t = H(d_t)
      m_t = m_{t-1} + beta_t * tau * tanh((z_t - m_{t-1}) / tau)

    beta_t can optionally decay as beta / sqrt(t + 1). The tanh bound prevents
    a single late step from moving logits arbitrarily far when d_t drifts outside
    the training-depth distribution.
    """

    def __init__(
        self,
        *args,
        logit_pull_beta: float = 0.25,
        logit_pull_tau: float = 2.0,
        logit_pull_decay: bool = True,
        **kwargs,
    ):
        kwargs["logit_refine"] = False
        super().__init__(*args, **kwargs)
        if logit_pull_beta < 0:
            raise ValueError("logit_pull_beta must be >= 0")
        if logit_pull_tau <= 0:
            raise ValueError("logit_pull_tau must be > 0")
        self.logit_pull_beta = float(logit_pull_beta)
        self.logit_pull_tau = float(logit_pull_tau)
        self.logit_pull_decay = bool(logit_pull_decay)

    def _bounded_pull_update(
        self,
        running_logits: torch.Tensor,
        target_logits: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        beta = self.logit_pull_beta
        if self.logit_pull_decay:
            beta = beta / sqrt(float(step_idx + 1))
        tau = self.logit_pull_tau
        delta = tau * torch.tanh((target_logits - running_logits) / tau)
        return running_logits + beta * delta

    def forward(
        self,
        x: torch.Tensor,
        recurrent_steps: int | None = None,
        tbptt_keep_steps: int | None = None,
    ):
        input_shape = x.shape[2:]
        recurrent_steps = self._get_recurrent_steps(x, recurrent_steps)
        keep_steps = self.tbptt_keep_steps if tbptt_keep_steps is None else int(tbptt_keep_steps)
        if keep_steps < 1:
            raise ValueError("tbptt_keep_steps must be >= 1")
        self._proposal_norm_mean = None
        self._proposal_norm_var = None

        skips = self._encode(x)
        proposal_states = self._build_proposal_states(skips)
        states = [state for state in proposal_states]
        if self.initial_state_noise_std > 0:
            states[-1] = self._add_initial_state_noise(states[-1])

        running_logits = self._decode_logits(states[-1], input_shape)
        all_step_logits = [running_logits]
        all_step_indices = [0]
        metrics_r_d: list[float] = []
        metrics_r_m: list[float] = []
        prev_logits = running_logits

        if self.training:
            t_cut = max(0, recurrent_steps - keep_steps)

            for step_idx in range(t_cut):
                with torch.no_grad():
                    prev_finest = states[-1]
                    states = self._refine_selected_once(states, proposal_states, skips, step_idx=step_idx)
                    target_logits = self._decode_logits(states[-1], input_shape)
                    running_logits = self._bounded_pull_update(running_logits, target_logits, step_idx)
                    metrics_r_d.append(self._relative_norm_delta(states[-1], prev_finest))
                    metrics_r_m.append(
                        self._relative_norm_delta(
                            self._prob_from_logits(running_logits),
                            self._prob_from_logits(prev_logits),
                        )
                    )
                    prev_logits = running_logits

            if t_cut > 0:
                states = [state.detach() for state in states]
                running_logits = running_logits.detach()

            for step_idx in range(t_cut, recurrent_steps):
                prev_finest = states[-1]
                states = self._refine_selected_once(states, proposal_states, skips, step_idx=step_idx)
                target_logits = self._decode_logits(states[-1], input_shape)
                running_logits = self._bounded_pull_update(running_logits, target_logits, step_idx)
                metrics_r_d.append(self._relative_norm_delta(states[-1], prev_finest))
                metrics_r_m.append(
                    self._relative_norm_delta(
                        self._prob_from_logits(running_logits),
                        self._prob_from_logits(prev_logits),
                    )
                )
                prev_logits = running_logits
                all_step_logits.append(running_logits)
                all_step_indices.append(step_idx + 1)
            active_step_indices = [0] + list(range(t_cut + 1, recurrent_steps + 1))
        else:
            t_cut = 0
            for step_idx in range(recurrent_steps):
                prev_finest = states[-1]
                states = self._refine_selected_once(states, proposal_states, skips, step_idx=step_idx)
                target_logits = self._decode_logits(states[-1], input_shape)
                running_logits = self._bounded_pull_update(running_logits, target_logits, step_idx)
                metrics_r_d.append(self._relative_norm_delta(states[-1], prev_finest))
                metrics_r_m.append(
                    self._relative_norm_delta(
                        self._prob_from_logits(running_logits),
                        self._prob_from_logits(prev_logits),
                    )
                )
                prev_logits = running_logits
                all_step_logits.append(running_logits)
                all_step_indices.append(step_idx + 1)
            active_step_indices = list(range(recurrent_steps + 1))

        final_logits = running_logits
        self.last_all_step_logits = all_step_logits
        self.last_all_step_indices = all_step_indices
        self.last_active_step_indices = active_step_indices
        self.last_step_metrics = {
            "r_d": metrics_r_d,
            "r_m": metrics_r_m,
            "K": recurrent_steps,
            "T_cut": t_cut,
        }
        metrics_csv = os.environ.get("LITE_REFINER_METRICS_CSV")
        if metrics_csv:
            import csv

            os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
            row = {
                "K": recurrent_steps,
                "T_cut": t_cut,
                "n_steps": len(metrics_r_d),
                "r_d_mean": float(sum(metrics_r_d) / len(metrics_r_d)) if metrics_r_d else 0.0,
                "r_m_mean": float(sum(metrics_r_m) / len(metrics_r_m)) if metrics_r_m else 0.0,
                "r_d_steps": ";".join(f"{v:.8g}" for v in metrics_r_d),
                "r_m_steps": ";".join(f"{v:.8g}" for v in metrics_r_m),
            }
            write_header = not os.path.exists(metrics_csv)
            with open(metrics_csv, "a", newline="") as fp:
                writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        if self.deep_supervision:
            return [final_logits]
        return final_logits


class LiteRBUNeXt3DFeatureRefinerNoise(LiteRBUNeXt3DFeatureRefiner):
    """
    FinestOnly + Feature Noise for Denoising Training

    Architecture:
      Same as FinestOnly (LiteRBUNeXt3DFeatureRefiner with refine_scales='finest')
      but adds Gaussian noise to the finest feature d^4_t during refinement.

    Key idea:
      - Add noise to d^4_t before each refinement step
      - Refiner learns to denoise and optimize simultaneously
      - Acts as a regularizer and robustness enhancer

    Noise injection:
      d^4_t_noisy = d^4_t + ε, where ε ~ N(0, noise_std * std(d^4_t))

    Hyperparameters:
      - feature_noise_std: noise level (e.g., 0.1 = 10% of feature std)
      - feature_noise_prob: probability of adding noise (1.0 = always)
      - feature_noise_only_training: if True, only add noise during training
    """

    def __init__(
        self,
        *args,
        feature_noise_std: float = 0.1,
        feature_noise_prob: float = 1.0,
        feature_noise_only_training: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if feature_noise_std < 0:
            raise ValueError("feature_noise_std must be >= 0")
        if not 0 <= feature_noise_prob <= 1:
            raise ValueError("feature_noise_prob must be in [0, 1]")

        self.feature_noise_std = float(feature_noise_std)
        self.feature_noise_prob = float(feature_noise_prob)
        self.feature_noise_only_training = bool(feature_noise_only_training)

    def _add_feature_noise(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to feature tensor.

        Noise scale is adaptive: noise_std * std(feature) per sample in batch.
        """
        if self.feature_noise_std <= 0:
            return feature

        # Skip noise injection during inference if configured
        if self.feature_noise_only_training and not self.training:
            return feature

        # Probabilistic noise injection
        if self.feature_noise_prob < 1.0:
            if torch.rand(1, device=feature.device).item() > self.feature_noise_prob:
                return feature

        # Compute per-sample standard deviation
        # feature shape: (B, C, D, H, W)
        scale = feature.detach().float().flatten(1).std(dim=1).view(-1, 1, 1, 1, 1).clamp_min(1e-6)

        # Generate noise
        noise = torch.randn_like(feature) * scale.to(dtype=feature.dtype) * self.feature_noise_std

        return feature + noise

    def _refine_finest_once(
        self,
        states: list[torch.Tensor],
        proposal_states: list[torch.Tensor],
        skips: list[torch.Tensor],
        step_idx: int,
    ) -> list[torch.Tensor]:
        """
        Override to add noise to finest feature before refinement.
        """
        if self.finest_refiner is None:
            raise RuntimeError("finest_refiner is only available when refine_scales='finest'")

        updated_states = list(states)
        finest_idx = len(states) - 1
        skip = skips[-(finest_idx + 2)]

        # Add noise to current finest state
        noisy_finest = self._add_feature_noise(states[finest_idx])

        # Refine from noisy state
        updated_states[finest_idx] = self.finest_refiner(
            noisy_finest,
            proposal_states[finest_idx],
            skip,
            step_idx=step_idx,
        )
        return updated_states


def model_summary_line(model: nn.Module) -> str:
    return f"{model.__class__.__name__}(trainable_params={count_parameters(model):,})"
