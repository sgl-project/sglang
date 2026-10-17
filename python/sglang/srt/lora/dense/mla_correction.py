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
    input: torch.Tensor
    shrink_weight: torch.Tensor
    weight: torch.Tensor
    route: RouteView
    plan: DensePlan
    heads: int
    side: str
    bridge: torch.Tensor | None = None
    done: torch.cuda.Event | None = None

    def tensor(self, name, shape, dtype=None):
        return self.runner.workspace.tensor(
            f"dense:mla_{self.side}:{name}",
            shape,
            dtype=dtype or self.input.dtype,
            device=self.input.device,
        )


def _shrink(p):
    x, weight = p.input, p.shrink_weight
    p.bridge = p.tensor("bridge", (x.shape[0], weight.shape[1]), x.dtype)
    grouped_lora_a(
        x,
        weight,
        p.bridge,
        p.route,
        config=p.plan.a_tiles,
        pair_input=True,
        weight_div=1 if p.side == "q" else p.heads,
        lora_ranks=p.runner.lora_ranks,
    )


def _expand(p, out):
    weight, runner = p.weight, p.runner
    grouped_lora_b(
        p.bridge,
        weight,
        out,
        p.route,
        pair_heads=p.heads,
        geometry=slice_geometry(
            (0, weight.shape[1]), int(p.plan.b_tiles["BLOCK_SIZE_N"]), weight.device
        ),
        config=p.plan.b_tiles,
        add_inplace=True,
        zero_sentinel=False,
        lora_ranks=runner.lora_ranks,
        scalings=runner.scalings,
        weight_div=1 if p.side == "v" else p.heads,
    )


def _prepare(attn, x, side, *, allow_overlap=True):
    layer = attn.kv_b_proj
    if (
        not getattr(layer, "set_lora", False)
        or not hasattr(layer, "A_buffer")
        or layer.lora_backend.batch_info is None
    ):
        return None
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
    runner.set_num_tokens(tokens)
    plan = _mla_plan(runner.phase)
    workspace = runner.workspace
    current = torch.cuda.current_stream(x.device) if x.is_cuda else None
    # A reshape may copy; keep it before ready so side-stream reads are ordered.
    flat_input = x.reshape(tokens * heads, input_dim)
    done = None
    overlap = allow_overlap and plan.overlap is Overlap.A and x.is_cuda
    # Build the route before forking so the side stream can start with shrink.
    route = runner.pair_route(heads, plan.block_size, num_tokens=tokens)
    if overlap:
        stream = workspace.side_stream(x.device)
        ready = workspace.event(x.device, f"dense:mla_{side}:ready")
        done = workspace.event(x.device, f"dense:mla_{side}:done")
        ready.record(current)
        stream.wait_event(ready)
        context = torch.cuda.stream(stream)
    else:
        context = nullcontext()
    with context:
        p = _Prepared(
            runner,
            flat_input,
            shrink_weight,
            expand_weight,
            route,
            plan,
            heads,
            side,
            done=done,
        )
        # Serial preparation preserves the existing shrink-before-BMM order.
        _shrink(p)
        if p.done is not None:
            p.done.record(stream)
    return p


def _apply(prepared, out):
    if prepared is None:
        return out
    if prepared.done is not None:
        torch.cuda.current_stream(out.device).wait_event(prepared.done)
    destination = (
        out.view(-1, prepared.heads, prepared.weight.shape[1]) if out.ndim == 2 else out
    )
    _expand(prepared, destination)
    return out


def prepare_q_correction(attn_module, q_nope):
    """Prepare before the base BMM; retain inputs and Q scratch until apply."""
    return _prepare(attn_module, q_nope, "q")


def prepare_v_correction(attn_module, attn_output):
    return _prepare(attn_module, attn_output, "v")


def apply_q_correction(attn_module, q_nope, q_nope_out, prepared=None):
    if prepared is None:
        prepared = _prepare(attn_module, q_nope, "q", allow_overlap=False)
    return _apply(prepared, q_nope_out)


def apply_v_correction(attn_module, attn_output, attn_bmm_flat, prepared=None):
    if prepared is None:
        prepared = _prepare(attn_module, attn_output, "v", allow_overlap=False)
    return _apply(prepared, attn_bmm_flat)
