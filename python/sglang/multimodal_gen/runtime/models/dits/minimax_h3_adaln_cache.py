# SPDX-License-Identifier: Apache-2.0
"""Precomputed AdaLN plan cache for the MiniMax H3 DiT."""

from __future__ import annotations

import os
import struct
from contextlib import ExitStack
from typing import Callable

import torch
import torch.nn as nn
from safetensors.torch import safe_open

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_world_size,
    tensor_model_parallel_all_gather,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_tp_rank
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_BF16_DTYPE = torch.bfloat16
_FP32_DTYPE = torch.float32


# A ref2va request carrying both a visual and an audio reference reaches four
# distinct timesteps in one step: video, audio, the imgvid condition and the
# audio reference. That is the widest case, so it is the default; a deployment
# serving only narrower tasks (t2va reaches 2, fl2va 3) can shrink the slab
# proportionally via --minimax-h3-adaln-plan-width.
MINIMAX_H3_ADALN_MAX_PLAN_WIDTH = 4


def _plan_key(timesteps: torch.Tensor) -> tuple[int, ...]:
    """One denoise step's unique timesteps as their exact fp32 bit patterns."""
    return tuple(
        struct.unpack("<I", struct.pack("<f", float(value)))[0]
        for value in timesteps.tolist()
    )


class MiniMaxH3AdalnCache(nn.Module):
    """Precomputed AdaLN outputs for fixed FP32 timestep plans."""

    _FORMAT_VERSION = "2"
    plan_timesteps: torch.Tensor
    plan_lengths: torch.Tensor
    block_params: torch.Tensor
    final_params: torch.Tensor

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        path: str | None = None,
        model_variant: str | None = None,
        weight_files: list[str] | None = None,
        max_plans: int = 64,
        max_plan_width: int = MINIMAX_H3_ADALN_MAX_PLAN_WIDTH,
    ) -> None:
        super().__init__()
        if (path is None) == (weight_files is None):
            raise ValueError(
                "MiniMax H3 AdaLN cache takes exactly one of path (prebuilt "
                "sidecar) or weight_files (rebuild from the checkpoint)"
            )
        if max_plans < 1:
            raise ValueError("MiniMax H3 AdaLN cache max_plans must be positive")
        if max_plan_width < 1:
            raise ValueError(
                "MiniMax H3 AdaLN cache max_plan_width must be positive; "
                "set --minimax-h3-adaln-plan-width to at least 1"
            )
        self.path = path
        self.model_variant = model_variant
        self.weight_files = weight_files
        self.max_plans = max_plans
        self.max_plan_width = max_plan_width
        self.num_layers = arch.num_layers
        self.hidden_size = arch.hidden_size
        self.block_width = 6 * MINIMAX_H3_ADALN_MODALITY_NUM * arch.hidden_size
        self.final_width = 2 * arch.hidden_size
        # Rebuild path only: plan bit pattern -> slot, tracked on the host.
        self._slots: dict[tuple[int, ...], int] = {}
        self.rebuilds = 0

    def load(self, device: torch.device) -> None:
        if self.path is None:
            self._allocate(device)
            return
        if not os.path.isfile(self.path):
            raise ValueError(f"MiniMax H3 AdaLN cache does not exist: {self.path}")

        with safe_open(self.path, framework="pt", device="cpu") as cache_file:
            metadata = cache_file.metadata() or {}
            if metadata.get("format_version") != self._FORMAT_VERSION:
                raise ValueError(
                    "MiniMax H3 AdaLN cache has an unsupported or missing format_version"
                )
            cache_variant = metadata.get("model_variant")
            if self.model_variant is not None and cache_variant != self.model_variant:
                raise ValueError(
                    "MiniMax H3 AdaLN cache model_variant does not match the loaded "
                    f"variant ({cache_variant!r} != {self.model_variant!r})"
                )
            plan_timesteps = cache_file.get_tensor("plan_timesteps")
            plan_lengths = cache_file.get_tensor("plan_lengths")
            block_params = cache_file.get_tensor("block_params")
            final_params = cache_file.get_tensor("final_params")

        expected_block_width = 6 * MINIMAX_H3_ADALN_MODALITY_NUM * self.hidden_size
        expected_final_width = 2 * self.hidden_size
        if (
            plan_timesteps.dtype != _FP32_DTYPE
            or plan_timesteps.ndim != 2
            or plan_lengths.dtype != torch.int64
            or plan_lengths.shape != (plan_timesteps.shape[0],)
            or (plan_lengths < 1).any()
            or (plan_lengths > plan_timesteps.shape[1]).any()
        ):
            raise ValueError("MiniMax H3 AdaLN cache has invalid timestep plans")
        if block_params.dtype != _BF16_DTYPE or block_params.shape != (
            plan_timesteps.shape[0],
            plan_timesteps.shape[1],
            self.num_layers,
            expected_block_width,
        ):
            raise ValueError("MiniMax H3 AdaLN cache has invalid block_params")
        if final_params.dtype != _BF16_DTYPE or final_params.shape != (
            plan_timesteps.shape[0],
            plan_timesteps.shape[1],
            expected_final_width,
        ):
            raise ValueError("MiniMax H3 AdaLN cache has invalid final_params")

        for slot in range(plan_timesteps.shape[0]):
            length = int(plan_lengths[slot])
            self._slots[_plan_key(plan_timesteps[slot, :length])] = slot

        # The 0.9-2.3 GiB slabs are derived data; keep them out of state_dict.
        self.register_buffer(
            "plan_timesteps", plan_timesteps.to(device), persistent=False
        )
        self.register_buffer("plan_lengths", plan_lengths.to(device), persistent=False)
        self.register_buffer("block_params", block_params.to(device), persistent=False)
        self.register_buffer("final_params", final_params.to(device), persistent=False)

    def _allocate(self, device: torch.device) -> None:
        """Empty slab for the rebuild path; its pointers must never move.

        ``plan_lengths`` starts at zero and that is what keeps unused slots out
        of ``lookup``: a real plan always has at least one timestep, so a zero
        length can never match. Breakable CUDA graph keys its replay signature
        on tensor pointers, so this is allocated once and only written in place.
        """
        width = self.max_plan_width
        self.register_buffer(
            "plan_timesteps",
            torch.zeros((self.max_plans, width), dtype=_FP32_DTYPE, device=device),
            persistent=False,
        )
        self.register_buffer(
            "plan_lengths",
            torch.zeros((self.max_plans,), dtype=torch.int64, device=device),
            persistent=False,
        )
        self.register_buffer(
            "block_params",
            torch.zeros(
                (self.max_plans, width, self.num_layers, self.block_width),
                dtype=_BF16_DTYPE,
                device=device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "final_params",
            torch.zeros(
                (self.max_plans, width, self.final_width),
                dtype=_BF16_DTYPE,
                device=device,
            ),
            persistent=False,
        )
        logger.info(
            "MiniMax H3 AdaLN rebuild slab: %d plans x %d timesteps = %.2f GiB",
            self.max_plans,
            width,
            self.block_params.numel() * 2 / 2**30,
        )

    def build(
        self,
        step_timesteps: list[torch.Tensor],
        *,
        embed: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Fill every plan this request will look up, in one streaming pass.

        Each plan keeps its own timestep count as the GEMM batch size, because
        cuBLAS selects kernels by shape and the selection is not monotonic in M:
        against the runtime's M == 2, results at M == 4/8/16/64/96 are
        bit-identical while M == 32 differs in 11760 of 96768 elements and
        M == 1 (the GEMV path the first denoise step takes) differs in 69.
        Rebuilding a plan at any other batch size silently perturbs the output.

        The pass reads all 50 adaln_proj layers regardless of how many plans are
        missing, so a request builds everything it needs before denoising rather
        than filling in step by step.
        """
        wanted: dict[tuple[int, ...], torch.Tensor] = {}
        for timesteps in step_timesteps:
            wanted.setdefault(_plan_key(timesteps), timesteps)
        missing = {k: v for k, v in wanted.items() if k not in self._slots}
        if not missing:
            return
        if len(wanted) > self.max_plans:
            raise ValueError(
                f"MiniMax H3 AdaLN rebuild needs {len(wanted)} plans but "
                f"max_plans is {self.max_plans}"
            )
        widest = max(timesteps.numel() for timesteps in wanted.values())
        if widest > self.max_plan_width:
            raise ValueError(
                f"MiniMax H3 AdaLN rebuild hit a {widest}-timestep plan but the "
                f"slab was allocated for {self.max_plan_width}; raise "
                "--minimax-h3-adaln-plan-width (t2va needs 2, fl2va 3, ref2va 4)"
            )

        reset = len(self._slots) + len(missing) > self.max_plans
        # A reset also evicts this request's cache hits, so rebuild its complete
        # plan set rather than only the plans that were initially missing.
        plans_to_build = wanted if reset else missing
        if reset:
            self._slots.clear()
            self.plan_lengths.zero_()

        device = self.block_params.device
        slots = []
        pending_slots: dict[tuple[int, ...], int] = {}
        for offset, (key, timesteps) in enumerate(plans_to_build.items()):
            slot = len(self._slots) + offset
            pending_slots[key] = slot
            slots.append((slot, timesteps.numel(), embed(timesteps.to(device))))
            self.plan_timesteps[slot, : timesteps.numel()] = timesteps.to(device)

        # adaln_proj is a ColumnParallelLinear: each rank owns a slice of the
        # output features and all-gathers afterwards. The rebuild has to do the
        # same rather than read the full width in one go -- a sharded GEMM has a
        # different N, so cuBLAS picks a different kernel and the outputs stop
        # matching. It also cuts per-rank checkpoint reads to 1/tp.
        tp_size = get_tp_world_size()
        tp_rank = get_tp_rank() if tp_size > 1 else 0

        with ExitStack() as stack:
            handles = [
                stack.enter_context(safe_open(f, framework="pt", device=str(device)))
                for f in self.weight_files
            ]
            index = {name: h for h in handles for name in h.keys()}

            def read_shard(name: str, out_features: int) -> torch.Tensor:
                if tp_size == 1:
                    return index[name].get_tensor(name)
                shard = out_features // tp_size
                start = tp_rank * shard
                return index[name].get_slice(name)[start : start + shard]

            def project(adaln_input: torch.Tensor, weight, bias) -> torch.Tensor:
                out = nn.functional.linear(adaln_input, weight, bias)
                return tensor_model_parallel_all_gather(out) if tp_size > 1 else out

            for layer in range(self.num_layers):
                prefix = f"blocks.{layer}.adaln_proj.linear"
                weight = read_shard(f"{prefix}.weight", self.block_width)
                bias = read_shard(f"{prefix}.bias", self.block_width)
                for slot, length, adaln_input in slots:
                    self.block_params[slot, :length, layer] = project(
                        adaln_input, weight, bias
                    )
                del weight, bias
            prefix = "final_layer.adaln_proj.linear"
            weight = read_shard(f"{prefix}.weight", self.final_width)
            bias = read_shard(f"{prefix}.bias", self.final_width)
            for slot, length, adaln_input in slots:
                self.final_params[slot, :length] = project(adaln_input, weight, bias)
            del weight, bias

        for slot, length, _ in slots:
            self.plan_lengths[slot] = length
        # Commit host metadata only after every layer has been written. If a
        # checkpoint read or projection raises, the zero-length slots remain
        # invisible and a later request can retry the rebuild.
        self._slots.update(pending_slots)
        self.rebuilds += 1
        logger.info(
            "MiniMax H3 AdaLN: rebuilt %d plan(s), %d/%d resident, pass #%d",
            len(plans_to_build),
            len(self._slots),
            self.max_plans,
            self.rebuilds,
        )

    def resolve_slots(self, step_timesteps: list[torch.Tensor]) -> torch.Tensor:
        """Per-step slab slots as one device tensor, resolved on the host.

        Forward receives one scalar view per step. Keeping it a device tensor
        matters twice over: a Python int would enter the breakable-CUDA-graph
        replay signature (one graph per slot value), and an int baked into a
        captured gather would read the wrong slab row after slot reuse.
        """
        slots = []
        for timesteps in step_timesteps:
            slot = self._slots.get(_plan_key(timesteps))
            if slot is None:
                raise ValueError(
                    "MiniMax H3 AdaLN cache does not cover the request timestep plan"
                )
            slots.append(slot)
        return torch.tensor(slots, dtype=torch.int64, device=self.block_params.device)

    def lookup(self, unique_timesteps: torch.Tensor) -> torch.Tensor:
        num_timesteps = unique_timesteps.shape[0]
        matches = self.plan_lengths.eq(num_timesteps) & self.plan_timesteps[
            :, :num_timesteps
        ].eq(unique_timesteps).all(dim=-1)
        if not bool(matches.any()):
            raise ValueError(
                "MiniMax H3 AdaLN cache does not cover the request timestep plan"
            )
        return matches.to(torch.int64).argmax()

    def block(
        self,
        index: int,
        cache_plan_index: torch.Tensor,
        num_timesteps: int,
    ) -> tuple[torch.Tensor, ...]:
        params = self.block_params[cache_plan_index, :num_timesteps, index]
        params = params.reshape(-1, 6, self.hidden_size)
        return tuple(params.unbind(dim=1))

    def block_all(
        self,
        cache_plan_index: torch.Tensor,
        num_timesteps: int,
    ) -> tuple[tuple[torch.Tensor, ...], ...]:
        """Every block's tuple via one slab gather instead of one per layer.

        Same elements and per-tensor strides as 50 block() calls; the single
        layer-major gather replaces 50 latency-bound indexing kernels.
        """
        stacked = self.block_params.permute(2, 0, 1, 3)[
            :, cache_plan_index, :num_timesteps
        ]
        stacked = stacked.reshape(self.num_layers, -1, 6, self.hidden_size)
        return tuple(tuple(layer.unbind(dim=1)) for layer in stacked)

    def final(
        self,
        cache_plan_index: torch.Tensor,
        num_timesteps: int,
    ) -> tuple[torch.Tensor, ...]:
        params = self.final_params[cache_plan_index, :num_timesteps]
        return tuple(params.reshape(-1, 2, self.hidden_size).unbind(dim=1))
