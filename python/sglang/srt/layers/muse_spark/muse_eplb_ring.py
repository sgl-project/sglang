"""Non-SP forwards route the same batch on every TP rank and the pick is a pure function
of its logical ids, so each (token, slot) pair runs on exactly one rank; SP prefill
routes by logical id to the home slots."""

from __future__ import annotations

import logging
from typing import Optional

import msgspec
import torch

logger = logging.getLogger(__name__)

# Larger batches are prefill-like and stay on the home copies.
PICK_MAX_TOKENS = 256
_LOGGED = set()


class RingLayout(msgspec.Struct, frozen=True):
    num_ranks: int
    home: int
    rank: int

    @property
    def num_logical(self) -> int:
        return self.num_ranks * self.home

    @property
    def num_slots(self) -> int:
        return 2 * self.home

    @property
    def num_physical(self) -> int:
        return 2 * self.num_logical

    @property
    def phys_offset(self) -> int:
        return self.rank * self.num_slots


def resolve_layout(
    *, num_layers: int, num_experts: int, ep_size: int, ep_rank: int
) -> Optional[RingLayout]:
    from sglang.srt.eplb.expert_location import (
        RING_LOAD_FORMATS,
        get_global_expert_location_metadata,
        is_ring_placement,
    )
    from sglang.srt.runtime_context import get_exec, get_model

    if get_exec().moe.ep_dispatch_algorithm != "ring":
        return None
    # Load-format resolution runs after argument validation; check the final one.
    if get_model().load_format not in RING_LOAD_FORMATS:
        raise ValueError(
            "--ep-dispatch-algorithm ring needs a loader that calls load_weights, "
            f"got --load-format {get_model().load_format}."
        )
    if not is_ring_placement(
        get_global_expert_location_metadata(),
        num_layers=num_layers,
        num_logical_experts=num_experts,
        ep_size=ep_size,
    ):
        raise ValueError(
            "--ep-dispatch-algorithm ring needs the --init-expert-location ring "
            "placement on every MoE layer."
        )
    return RingLayout(num_ranks=ep_size, home=num_experts // ep_size, rank=ep_rank)


def log_expected_replica_bytes(
    layout: RingLayout,
    *,
    num_nvfp4_layers: int,
    num_bf16_layers: int,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
) -> None:
    elems = 3 * hidden_size * intermediate_size
    nvfp4 = elems // 2 + elems // group_size + 6 * 4
    total = layout.home * (num_nvfp4_layers * nvfp4 + num_bf16_layers * elems * 2)
    if layout.rank == 0:
        logger.info(
            "Muse ring: buddy expert copies need %.2f GiB per rank "
            "(%d NVFP4 + %d BF16 MoE layers).",
            total / 2**30,
            num_nvfp4_layers,
            num_bf16_layers,
        )


def log_replica_bytes(moes) -> None:
    layout = moes[0].eplb_ring
    total = sum(
        p.numel() * p.element_size()
        for moe in moes
        for p in moe.experts.parameters(recurse=False)
        if p.dim() and p.shape[0] == layout.num_slots
    )
    if layout.rank == 0:
        logger.info(
            "Muse ring: buddy expert copies hold %.2f GiB per rank.",
            total / 2 / 2**30,
        )


def copy_ring_slots(
    param: torch.Tensor, loaded_weight: torch.Tensor, layout: RingLayout
) -> None:
    if param.shape[0] != layout.num_slots or param.shape[1:] != loaded_weight.shape[1:]:
        raise RuntimeError(
            f"Ring expert slots {tuple(param.shape)} do not fit the checkpoint "
            f"stack {tuple(loaded_weight.shape)}."
        )
    L = layout.home
    for dst, src_rank in ((0, layout.rank), (L, (layout.rank + 1) % layout.num_ranks)):
        param.data.narrow(0, dst, L).copy_(loaded_weight.narrow(0, src_rank * L, L))


def home_packed(
    topk_ids: torch.Tensor, packed_topk_ids: torch.Tensor, layout: RingLayout
) -> torch.Tensor:
    ids = topk_ids.to(torch.int32)
    phys = ids + torch.div(ids, layout.home, rounding_mode="floor") * layout.home
    return torch.where(
        ids >= 0, (phys << 16) | (packed_topk_ids & 0xFFFF), packed_topk_ids
    )


def _fused_eligible(op, router_logits: torch.Tensor, top_k: int, layout) -> bool:
    from sglang.kernels.fused_op import get_fused_op_backend
    from sglang.kernels.ops.moe.muse_spark_route_warp_ring import launch_shape
    from sglang.kernels.spec import KernelBackend

    bias = op.expert_bias
    # The fused kernel reproduces only the JIT router's defaults.
    return (
        router_logits.dtype == torch.float32
        and bias.dtype == torch.float32
        and router_logits.shape[1] == layout.num_logical <= 512
        and router_logits.stride(1) == 1
        and router_logits.stride(0) % 8 == 0
        and router_logits.data_ptr() % 32 == 0
        and bias.is_contiguous()
        and bias.data_ptr() % 32 == 0
        and layout.home % 32 == 0
        and top_k <= 32
        and op.local_expert_mapping is None
        and op.sol_topk_out is None
        and not op.is_torch_compile
        and get_fused_op_backend() in (None, KernelBackend.JIT)
        and launch_shape(router_logits.shape[0]) is not None
        and torch.cuda.get_device_capability(router_logits.device) >= (9, 0)
    )


def _log_path_once(path: str, num_tokens: int) -> None:
    if path not in _LOGGED:
        _LOGGED.add(path)
        logger.info("Muse ring pick: %s path live (%d tokens)", path, num_tokens)


def route_and_pick(op, router_logits, top_k: int, layout: RingLayout, scratch):
    from sglang.kernels.ops.moe.muse_spark_route_warp_ring import (
        muse_spark_ring_pick,
        muse_spark_route_warp_ring,
    )

    num_tokens = router_logits.shape[0]
    if _fused_eligible(op, router_logits, top_k, layout):
        _log_path_once("fused", num_tokens)
        return muse_spark_route_warp_ring(
            router_logits,
            op.expert_bias,
            top_k,
            op.route_scale,
            num_ranks=layout.num_ranks,
            home=layout.home,
            router_output_multiplier=op.router_output_multiplier,
            division_eps=op.router_score_division_eps,
            num_valid_tokens=op.num_valid_tokens,
        )
    weights, ids, packed = op(None, router_logits, top_k, True, want_packed=True)
    if num_tokens <= PICK_MAX_TOKENS:
        _log_path_once("triton", num_tokens)
        packed = muse_spark_ring_pick(
            ids, packed, scratch, num_ranks=layout.num_ranks, home=layout.home
        )
    else:
        _log_path_once("home-only", num_tokens)
        packed = home_packed(ids, packed, layout)
    return weights, ids, packed


@torch.no_grad()
def startup_self_check(moe, tp_group) -> None:
    from sglang.kernels.ops.moe.muse_spark_route_warp_ring import (
        MAX_FUSED_TOKENS,
        muse_spark_ring_pick,
    )

    layout, op, top_k = moe.eplb_ring, moe._routing_op(), moe.top_k
    L, R = layout.home, layout.num_ranks
    device = op.expert_bias.device
    gen = torch.Generator(device=device)
    gen.manual_seed(0)
    for num_tokens in (
        1,
        8,
        MAX_FUSED_TOKENS,
        MAX_FUSED_TOKENS + 1,
        PICK_MAX_TOKENS + 1,
    ):
        # fp32 like the router; post-load runs under the checkpoint's default dtype.
        logits = 4 * torch.randn(
            num_tokens,
            layout.num_logical,
            generator=gen,
            device=device,
            dtype=torch.float32,
        )
        logits[:, :L] += 6  # crowd one home so that copies move
        weights, ids, packed = route_and_pick(
            op, logits, top_k, layout, moe._eplb_ring_scratch
        )
        ref_w, ref_ids, ref_packed = op(None, logits, top_k, True, want_packed=True)
        phys = packed >> 16
        holder, slot = phys // (2 * L), phys % (2 * L)
        logical = torch.where(slot < L, holder, (holder + 1) % R) * L + slot % L
        ok = (
            torch.equal(ids.int(), ref_ids.int())
            and torch.equal(weights.view(torch.int32), ref_w.view(torch.int32))
            and torch.equal(packed & 0xFFFF, ref_packed & 0xFFFF)
            and torch.equal(logical, ref_ids.int())
        )
        if num_tokens <= MAX_FUSED_TOKENS:
            # Fused and Triton picks are independent implementations of one rule.
            ok = ok and torch.equal(
                packed,
                muse_spark_ring_pick(
                    ref_ids, ref_packed, moe._eplb_ring_scratch, num_ranks=R, home=L
                ),
            )
        local = torch.cat([packed.flatten(), packed.new_tensor([int(ok)])])
        gathered = local.new_empty((tp_group.world_size, local.numel()))
        torch.distributed.all_gather_into_tensor(
            gathered, local, group=tp_group.device_group
        )
        if not bool((gathered == local).all() and (gathered[:, -1] == 1).all()):
            raise RuntimeError(
                f"Muse ring self-check failed at {num_tokens} tokens on some TP rank."
            )
    # Let serving log its own first use of each path.
    _LOGGED.clear()
    logger.info(
        "Muse ring self-check passed on %d TP ranks: %s",
        tp_group.world_size,
        layout,
    )
