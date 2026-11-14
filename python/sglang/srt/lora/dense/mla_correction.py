"""Absorbed MLA: Q += Q @ B_k @ A; V += attention @ A.T @ B_v.T.

Plans name the shrink/expand stages, not the original adapter A/B order.
Only B has per-head slices. Decode overlaps the grouped shrink with the base BMM.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from functools import cache

import torch

from sglang.kernels.ops.lora.common.lora_a import grouped_lora_a
from sglang.kernels.ops.lora.common.lora_b import grouped_lora_b, slice_geometry
from sglang.kernels.ops.lora.common.route_view import RouteView
from sglang.srt.lora.dense.plan import DensePlan, Overlap
from sglang.srt.lora.utils import Phase


@cache
def _mla_plan(phase: Phase) -> DensePlan:
    return DensePlan(
        block_size=16 if phase is Phase.DECODE else 64,
        overlap=Overlap.A if phase is Phase.DECODE else Overlap.NONE,
    )


def _b_slice_per_head(b: torch.Tensor, heads: int, start: int, width: int):
    slots, output_dim, rank = b.shape
    return b.view(slots, heads, output_dim // heads, rank)[
        :, :, start : start + width, :
    ].reshape(slots * heads, width, rank)


@dataclass(slots=True)
class _Prepared:
    runner: object
    input: torch.Tensor  # Retain reshape copies until the side-stream join.
    weight: torch.Tensor
    route: RouteView
    plan: DensePlan
    heads: int
    side: str
    bridge: torch.Tensor
    done: torch.cuda.Event | None = None


def _prepare(attn, x, side):
    layer = attn.kv_b_proj
    runner = layer.lora_backend.runner
    a, b = layer.A_buffer, layer.B_buffer
    tokens, heads, input_dim = x.shape
    if side == "q":
        shrink_weight = _b_slice_per_head(b, heads, 0, input_dim).transpose(1, 2)
        expand_weight = a.transpose(1, 2)
    else:
        shrink_weight = a
        expand_weight = _b_slice_per_head(
            b, heads, attn.qk_nope_head_dim, attn.v_head_dim
        )
    plan = _mla_plan(runner.phase)
    workspace = runner.workspace
    # A reshape may copy; keep it before ready so side-stream reads are ordered.
    flat_input = x.reshape(tokens * heads, input_dim)
    done = None
    # Build the route before forking so the side stream can start with shrink.
    route = runner.pair_route(heads, plan.block_size, num_tokens=tokens)
    if plan.overlap is Overlap.A:
        current = torch.cuda.current_stream(x.device)
        stream = workspace.side_stream(x.device)
        ready = workspace.event(x.device, f"dense:mla_{side}:ready")
        done = workspace.event(x.device, f"dense:mla_{side}:done")
        ready.record(current)
        stream.wait_event(ready)
        context = torch.cuda.stream(stream)
    else:
        context = nullcontext()
    with context:
        bridge = workspace.tensor(
            f"dense:mla_{side}:bridge",
            (flat_input.shape[0], shrink_weight.shape[1]),
            dtype=x.dtype,
            device=x.device,
        )
        grouped_lora_a(
            flat_input,
            shrink_weight,
            bridge,
            route,
            config=plan.a_tiles,
            pair_input=True,
            weight_div=1 if side == "q" else heads,
            lora_ranks=runner.lora_ranks,
        )
        if done is not None:
            done.record(stream)
    return _Prepared(
        runner, flat_input, expand_weight, route, plan, heads, side, bridge, done
    )


def apply_correction(prepared, out):
    if prepared.done is not None:
        torch.cuda.current_stream(out.device).wait_event(prepared.done)
    weight, runner = prepared.weight, prepared.runner
    destination = (
        out.view(-1, prepared.heads, weight.shape[1]) if out.ndim == 2 else out
    )
    grouped_lora_b(
        prepared.bridge,
        weight,
        destination,
        prepared.route,
        pair_heads=prepared.heads,
        geometry=slice_geometry(
            (0, weight.shape[1]),
            int(prepared.plan.b_tiles["BLOCK_SIZE_N"]),
            weight.device,
        ),
        config=prepared.plan.b_tiles,
        add_inplace=True,
        zero_sentinel=False,
        lora_ranks=runner.lora_ranks,
        scalings=runner.scalings,
        weight_div=1 if prepared.side == "v" else prepared.heads,
    )
    return out


def prepare_q_correction(attn_module, q_nope):
    """Prepare before the base BMM; keep the result alive until apply."""
    return _prepare(attn_module, q_nope, "q")


def prepare_v_correction(attn_module, attn_output):
    return _prepare(attn_module, attn_output, "v")
