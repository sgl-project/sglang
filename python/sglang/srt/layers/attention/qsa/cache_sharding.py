"""QSA cache sharding topology and CUDA-graph transport buffers."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class QSACacheShardingRuntime:
    """Resolved QSA cache-sharding contract shared by runtime components."""

    size: int
    rank: int
    group: object | None

    @property
    def enabled(self) -> bool:
        return self.size > 1


def resolve_qsa_cache_sharding_runtime(parallel) -> QSACacheShardingRuntime:
    """Resolve QSA cache sharding from the published parallel topology."""

    size = int(getattr(parallel, "qsa_cache_sharding_size", 1))
    return QSACacheShardingRuntime(
        size=size,
        rank=int(getattr(parallel, "qsa_cache_sharding_rank", 0)),
        group=(
            getattr(parallel, "qsa_cache_sharding_group", None) if size > 1 else None
        ),
    )


def get_qsa_cache_sharding_runtime(parallel=None) -> QSACacheShardingRuntime:
    """Resolve QSA cache sharding from the process-published topology."""

    from sglang.srt import runtime_context

    if parallel is None:
        parallel = runtime_context.get_parallel()
    try:
        return resolve_qsa_cache_sharding_runtime(parallel)
    except ValueError as exc:
        if "not published" not in str(exc):
            raise
        return QSACacheShardingRuntime(size=1, rank=0, group=None)


def assert_qsa_cache_sharding_runtime_match(
    expected: QSACacheShardingRuntime,
    actual: QSACacheShardingRuntime,
    *,
    component: str,
) -> None:
    """Reject partially enabled or differently sharded QSA components."""

    fields = ("size", "rank")
    mismatches = [
        name for name in fields if getattr(expected, name) != getattr(actual, name)
    ]
    if mismatches:
        raise ValueError(
            f"{component} QSA cache-sharding runtime mismatch: "
            + ", ".join(
                f"{name}={getattr(actual, name)!r} "
                f"(expected {getattr(expected, name)!r})"
                for name in mismatches
            )
        )
    if expected.enabled and expected.group is not actual.group:
        raise ValueError(
            f"{component} QSA cache-sharding collective group does not match"
        )


def validate_qsa_cache_sharding(
    *,
    device: str,
    is_qsa: bool,
    tp_size: int,
    sharding_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    pp_size: int,
    attn_cp_size: int,
    enable_prefill_cp: bool,
    ep_size: int,
    moe_dp_size: int,
    disaggregation_mode: str,
    dcp_size: int = 1,
) -> None:
    """Reject combinations without complete QSA cache-sharding semantics."""

    if sharding_size == 1:
        return
    if device != "cuda":
        raise ValueError("QSA cache sharding currently requires CUDA")
    if not is_qsa:
        raise ValueError("QSA cache sharding requires a QSA model")
    if sharding_size < 1 or tp_size % sharding_size:
        raise ValueError("QSA cache sharding requires its size to divide tp_size")
    if (
        min(num_attention_heads, num_key_value_heads) < 1
        or num_attention_heads % tp_size
        or num_attention_heads % num_key_value_heads
        or tp_size % num_key_value_heads
        or (tp_size // num_key_value_heads) % sharding_size
    ):
        raise ValueError(
            "QSA cache sharding must divide the KV-head replica group size: "
            f"attention_tp_size={tp_size}, num_attention_heads={num_attention_heads}, "
            f"num_key_value_heads={num_key_value_heads}, sharding_size={sharding_size}"
        )
    if dcp_size != 1:
        raise ValueError(
            "QSA cache sharding does not support decode context parallelism"
        )
    if pp_size != 1:
        raise ValueError("QSA cache sharding does not support pipeline parallelism")
    if attn_cp_size != 1:
        raise ValueError(
            "QSA cache sharding does not support attention context parallelism"
        )
    if enable_prefill_cp:
        raise ValueError("QSA cache sharding does not support prefill CP")
    if ep_size < 1 or moe_dp_size < 1 or tp_size % (ep_size * moe_dp_size):
        raise ValueError("QSA cache sharding requires a valid MoE EP topology")
    if disaggregation_mode != "null":
        raise ValueError("QSA cache sharding does not support PD disaggregation")


class QSACacheShardingStaticBuffers:
    """Fixed-address collective staging buffers suitable for CUDA Graph."""

    def __init__(
        self,
        *,
        max_rows: int,
        topk: int,
        owner_topk: int | None = None,
        num_heads: int,
        head_dim: int,
        world_size: int = 1,
        device: torch.device | str,
        dtype: torch.dtype,
        layer_ids: tuple[int, ...] | list[int] | None = None,
        candidate_group=None,
    ) -> None:
        owner_topk = topk if owner_topk is None else owner_topk
        if min(max_rows, topk, owner_topk, num_heads, head_dim, world_size) <= 0:
            raise ValueError("static buffer dimensions must be positive")
        if num_heads % world_size:
            raise ValueError("num_heads must be divisible by world_size")
        self.max_rows = max_rows
        self.world_size = world_size
        self.topk = topk
        self.owner_topk = owner_topk
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._device = torch.device(device)
        self._dtype = dtype
        self._layer_ids = None
        if layer_ids is not None:
            unique_layer_ids = tuple(
                dict.fromkeys(int(layer_id) for layer_id in layer_ids)
            )
            if not unique_layer_ids or len(unique_layer_ids) != len(layer_ids):
                raise ValueError("QSA layer ids must be non-empty and unique")
            self._layer_ids = unique_layer_ids
        local_heads = num_heads // world_size
        self.candidate_transport = torch.empty(
            (max_rows, topk, 2), dtype=torch.float32, device=device
        )
        self.candidate_recv = torch.empty(
            (world_size, max_rows, topk, 2), dtype=torch.float32, device=device
        )
        self.candidate_symm_state = self._create_candidate_symm_state(candidate_group)
        self.candidate_scores = torch.empty(
            (max_rows, topk), dtype=torch.float32, device=device
        )
        self.candidate_indices = torch.empty(
            (max_rows, topk), dtype=torch.int32, device=device
        )
        self.q_send = torch.empty(
            (local_heads, max_rows, head_dim), dtype=dtype, device=device
        )
        self.q_recv = torch.empty(
            (num_heads, max_rows, head_dim), dtype=dtype, device=device
        )
        self.q_gathered = torch.empty(
            (max_rows, num_heads, head_dim), dtype=dtype, device=device
        )
        self.local_slots = torch.empty(
            (max_rows, owner_topk), dtype=torch.int64, device=device
        )
        self.owner_lengths = torch.empty((max_rows,), dtype=torch.int32, device=device)
        self.owner_pack_max_rows = min(max_rows, 4)
        owner_pack_tokens = self.owner_pack_max_rows * owner_topk
        self.owner_packed_k = torch.empty(
            (owner_pack_tokens, 1, head_dim), dtype=dtype, device=device
        )
        self.owner_packed_v = torch.empty_like(self.owner_packed_k)
        self.owner_packed_slots = torch.empty(
            (self.owner_pack_max_rows, owner_topk),
            dtype=torch.int64,
            device=device,
        )
        self.owner_a2a_send = torch.empty(
            (world_size, max_rows, local_heads, head_dim + 1),
            dtype=torch.float32,
            device=device,
        )
        self.owner_a2a_recv = torch.empty_like(self.owner_a2a_send)

    def _create_candidate_symm_state(self, candidate_group):
        """Rendezvous one graph-only candidate buffer before graph capture."""

        if candidate_group is None:
            return None
        from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
            create_state,
        )

        device_group = getattr(candidate_group, "device_group", None)
        capability = (
            torch.cuda.get_device_capability(self._device)
            if self._device.type == "cuda"
            else (0, 0)
        )
        local_supported = (
            os.environ.get("SGLANG_QSA_CANDIDATE_SYMM_MEM", "1") == "1"
            and self.world_size == 4
            and self._device.type == "cuda"
            and capability[0] == 9
            and self.topk * 4 % 8 == 0
            and self.candidate_transport.data_ptr() % 16 == 0
            and type(device_group) is dist.ProcessGroup
        )
        if device_group is None:
            return None
        consensus = torch.tensor(
            int(local_supported), dtype=torch.int32, device=self._device
        )
        dist.all_reduce(consensus, op=dist.ReduceOp.MIN, group=device_group)
        if not bool(consensus.item()):
            return None
        state = create_state(
            group=device_group,
            rank_in_group=int(candidate_group.rank_in_group),
            max_tokens=self.max_rows,
            hidden_size=self.topk * 4 * self.world_size,
            device=self._device,
        )
        multicast = torch.tensor(
            int(state.symm_mem_hdl.multicast_ptr != 0),
            dtype=torch.int32,
            device=self._device,
        )
        dist.all_reduce(multicast, op=dist.ReduceOp.MIN, group=device_group)
        return state if bool(multicast.item()) else None

    def for_layer(self, layer_id: int) -> QSACacheShardingStaticBuffers:
        if self._layer_ids is None:
            return self
        if int(layer_id) not in self._layer_ids:
            raise KeyError(f"no static transport slice for QSA layer {layer_id}")
        return self

    def for_batch(self, rows: int) -> QSACacheShardingStaticBuffers:
        if rows < 0 or rows > self.max_rows:
            raise ValueError(
                f"batch rows {rows} exceed static buffer capacity {self.max_rows}"
            )
        return self


__all__ = [
    "QSACacheShardingRuntime",
    "QSACacheShardingStaticBuffers",
    "assert_qsa_cache_sharding_runtime_match",
    "get_qsa_cache_sharding_runtime",
    "resolve_qsa_cache_sharding_runtime",
    "validate_qsa_cache_sharding",
]
