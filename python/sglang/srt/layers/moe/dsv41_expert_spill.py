"""DSV4.1 routed-expert spill: LRU of MXFP4 rows onto pinned host.

Attention / routers / shared experts stay GPU. Generic ``--cpu-offload-gb``
wraps whole ``DeepseekV4DecoderLayer`` modules (CSA2 + indexer + MoE) and
``to(device)``-s them on every forward. This module spills only routed-expert
rows.

``maybe_spill_model_routed_experts`` attaches the plan. Shrink + host copies
happen only when ``SGLANG_DSV41_EXPERT_SPILL_APPLY`` is on. Prefill still
``ensure()`` + ``map_ids`` (LRU; not CUDA-graph safe). Decode uses a host
MXFP4 GEMV via a mapped mailbox when ``SGLANG_DSV41_HOST_GEMV`` is on,
else UVA page-in into a shared landing pool. Both remap with a
capturable kernel so Marlin/GEMV can stay in the decode graph.
"""

from __future__ import annotations

import gc
import json
import logging
import math
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.mem_cache.dsv41_host_placement import (
    EngramNumaError,
    engram_numa_node,
    gpu_numa_node,
    mmap_numa_thp,
    node_available_bytes,
    read_numa_nodes,
)
from sglang.srt.mem_cache.dsv41_v100_budget import (
    DEFAULT_SPILL_GIB,
    N_LAYERS,
    N_ROUTED_EXPERTS,
    expert_scale_bytes,
    mxfp4_expert_bytes,
)

logger = logging.getLogger(__name__)

GIB = 1024**3

# FusedMoE expert-dim parameters after MXFP4 Marlin pack. Shrink them together
# so w13/w2/scales stay row-aligned with the LRU slot map.
_EXPERT_PARAM_ATTRS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_inv",
    "w2_weight_scale_inv",
    "w13_weight_bias",
    "w2_weight_bias",
)


@dataclass(frozen=True)
class RoutedExpertSpillPlan:
    local_routed: int
    n_shared: int
    n_kept_routed: int
    n_spilled: int
    bytes_per_expert: int
    spill_bytes: int
    kept_bytes: int

    @property
    def spill_gib(self) -> float:
        return self.spill_bytes / GIB

    @property
    def kept_gib(self) -> float:
        return self.kept_bytes / GIB


def plan_routed_expert_spill(
    *,
    spill_gib: float,
    local_routed: int,
    bytes_per_expert: int,
    n_shared: int = 1,
    n_layers: int = N_LAYERS,
) -> RoutedExpertSpillPlan:
    """How many local routed experts leave HBM at this spill budget.

    Shared-expert slots are never spilled. ``spill_gib`` is per rank, across
    all layers (the dry-run number).
    """
    if local_routed < 0 or bytes_per_expert < 0:
        raise ValueError("local_routed and bytes_per_expert must be >= 0")
    spill_bytes = int(round(max(spill_gib, 0.0) * GIB))
    layer_budget = spill_bytes // max(n_layers, 1)
    n_spilled = 0 if bytes_per_expert == 0 else min(
        local_routed, layer_budget // bytes_per_expert
    )
    # If integer division under-spills the rank budget, spill extra experts
    # from the last layers conceptually — we still report rank-level bytes.
    n_kept_routed = local_routed - n_spilled
    kept_bytes = n_kept_routed * bytes_per_expert * n_layers
    actual_spill = n_spilled * bytes_per_expert * n_layers
    return RoutedExpertSpillPlan(
        local_routed=local_routed,
        n_shared=n_shared,
        n_kept_routed=n_kept_routed,
        n_spilled=n_spilled,
        bytes_per_expert=bytes_per_expert,
        spill_bytes=actual_spill,
        kept_bytes=kept_bytes,
    )


def v1_spill_plan(spill_gib: float = DEFAULT_SPILL_GIB, ep_size: int = 8) -> RoutedExpertSpillPlan:
    local_routed = N_ROUTED_EXPERTS // ep_size
    per_expert = mxfp4_expert_bytes() + expert_scale_bytes()
    return plan_routed_expert_spill(
        spill_gib=spill_gib,
        local_routed=local_routed,
        bytes_per_expert=per_expert,
        n_shared=1,
    )


def marlin_mxfp4_packed_expert_bytes(
    hidden_size: int, intermediate_size_per_partition: int
) -> int:
    """Bytes for one expert after MXFP4 Marlin pack (int8 + e8m0 scales)."""
    fp4_block_k = 32
    inter = (intermediate_size_per_partition + 127) // 128 * 128
    hidden = (hidden_size + 255) // 256 * 256
    w13 = 2 * inter * (hidden // 2)
    w2 = hidden * (inter // 2)
    s13 = 2 * inter * (hidden // fp4_block_k)
    s2 = hidden * (inter // fp4_block_k)
    return w13 + w2 + s13 + s2


def plan_gpu_expert_slots(
    *,
    num_local_experts: int,
    n_shared: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    n_layers: int = N_LAYERS,
) -> tuple[int, Optional[RoutedExpertSpillPlan]]:
    """GPU expert-dim length for create_weights. APPLY shrinks at alloc time."""
    spill_gib = float(envs.SGLANG_DSV41_EXPERT_SPILL_GB.get() or 0.0)
    apply = bool(envs.SGLANG_DSV41_EXPERT_SPILL_APPLY.get())
    if not apply or spill_gib <= 0 or num_local_experts <= 0:
        return num_local_experts, None
    local_routed = max(num_local_experts - n_shared, 0)
    # Spill GiB is sized for the 40-layer 384-expert target (48 local routed
    # at EP8). DSpark draft is 128/EP8=16 and must stay fully GPU-resident.
    target_local = N_ROUTED_EXPERTS // 8
    if 0 < local_routed < target_local:
        logger.info(
            "DSV4.1 expert spill skipped: local routed %d < target shard %d. "
            "Keeping %d GPU slots (DSpark draft / small MoE).",
            local_routed,
            target_local,
            num_local_experts,
        )
        return num_local_experts, None
    bytes_per = marlin_mxfp4_packed_expert_bytes(
        hidden_size, intermediate_size_per_partition
    )
    plan = plan_routed_expert_spill(
        spill_gib=spill_gib,
        local_routed=local_routed,
        bytes_per_expert=bytes_per,
        n_shared=n_shared,
        n_layers=n_layers,
    )
    if plan.n_spilled <= 0:
        return num_local_experts, plan
    gpu_n = plan.n_kept_routed + n_shared
    # The spill GiB knob is sized for the 40-layer 384-expert target. A 3-stage
    # DSpark draft is 128 experts (~0.8 GiB) and would compute gpu_n=0 if it
    # inherited that LRU. Keep the small MoE entirely on GPU.
    if gpu_n <= 0 or plan.n_spilled >= local_routed:
        logger.info(
            "DSV4.1 expert spill skipped: would host-spill all %d local routed "
            "experts (gpu slots %d). Keeping %d GPU slots. Draft/small MoE; "
            "the 384-expert target LRU is unchanged.",
            local_routed,
            gpu_n,
            num_local_experts,
        )
        return num_local_experts, None
    logger.info(
        "DSV4.1 expert create_weights: GPU slots %d (kept routed %d + shared %d), "
        "host-spill %d of %d local routed (%.1f GiB/rank packed)",
        gpu_n,
        plan.n_kept_routed,
        n_shared,
        plan.n_spilled,
        local_routed,
        plan.spill_gib,
    )
    return gpu_n, plan


def spill_host_is_marlin_packed(
    hosts: Dict[str, torch.Tensor], gpu_w13: torch.Tensor, gpu_scale: Optional[torch.Tensor]
) -> bool:
    """True only if host w13 *and* packed scales already match the GPU Marlin layout.

    Checkpoint host is ``w13_weight_scale_inv`` in ``[E, N, K/32]``. SM70 pack
    deletes that and writes ``w13_weight_scale`` in ``[E, K/32, N]``. Matching
    w13 trailing shape alone used to skip pack while leaving scales on the
    checkpoint name, so LRU swapped w13/w2 but Marlin kept the GPU slot's
    scales.
    """
    host_w13 = hosts.get("w13_weight")
    if host_w13 is None:
        return False
    if tuple(host_w13.shape[1:]) != tuple(gpu_w13.shape[1:]):
        return False
    if host_w13.dtype != gpu_w13.dtype:
        raise RuntimeError(
            "DSV4.1 spill host w13 trailing shape matches GPU "
            f"{tuple(gpu_w13.shape[1:])} but dtype {host_w13.dtype} != {gpu_w13.dtype}"
        )
    host_scale = hosts.get("w13_weight_scale")
    if host_scale is None or gpu_scale is None:
        return False
    return tuple(host_scale.shape[1:]) == tuple(gpu_scale.shape[1:]) and (
        host_scale.dtype == gpu_scale.dtype
    )


def alloc_spill_host_buffers(moe: nn.Module, plan: RoutedExpertSpillPlan) -> None:
    """Pinned host rows for spilled experts, matching each GPU expert-dim param."""
    if plan.n_spilled <= 0:
        return
    # Checkpoint-layout rows, filled by the weight loader; repacked and pinned
    # after Marlin pack (pin_spill_host_numa). Not registered here: pinning
    # 80 GiB before the load would only add to the loader's peak.
    #
    # Place them on the *same* NUMA node the pinned mirror of this
    # layer will use. With the default (preferred node 1) policy and node 1
    # full of Engram hugetlb, these 10 GiB/rank overflowed onto node 0 and
    # coexisted with the growing pinned node-0 half -> node-0 OOM
    # (CONSTRAINT_MEMORY_POLICY) during repack. Same node keeps the per-node
    # footprint flat through the transition.
    node: Optional[int] = None
    layer_id = int(getattr(moe, "layer_id", 0))
    use_numa = bool(envs.SGLANG_ENABLE_DSV41_EXPERT_SPILL_NUMA.get()) and torch.cuda.is_available()
    if use_numa:
        nodes = _spill_numa_nodes()
        node = nodes[layer_id % len(nodes)]
        if node < 0:
            node = None
    specs = _spill_row_specs(moe)
    if node is None:
        hosts, mms = _alloc_spill_rows(specs, n_rows=plan.n_spilled, node=None)
    else:
        hosts, mms, node = _alloc_spill_rows_with_failover(
            specs, n_rows=plan.n_spilled, preferred=node, layer_id=layer_id
        )
    moe._dsv41_spill_host = hosts  # type: ignore[attr-defined]
    moe._dsv41_spill_host_ctor_mms = mms  # type: ignore[attr-defined]
    moe._dsv41_spill_numa_node = node  # type: ignore[attr-defined]
    moe._dsv41_expert_spill_plan = plan  # type: ignore[attr-defined]


# (param attr, dtype, row shape) of one expert-dim param's host rows.
_SpillRowSpec = tuple[str, torch.dtype, tuple[int, ...]]


def _spill_row_specs(moe: nn.Module) -> List[_SpillRowSpec]:
    specs = []
    for attr in _EXPERT_PARAM_ATTRS:
        p = getattr(moe, attr, None)
        if p is None or not isinstance(p, torch.nn.Parameter) or p.ndim < 1:
            continue
        if int(p.shape[0]) == 0:
            continue
        specs.append((attr, p.dtype, tuple(p.shape[1:])))
    return specs


def _alloc_spill_rows(
    specs: List[_SpillRowSpec],
    *,
    n_rows: int,
    node: Optional[int],
) -> tuple[Dict[str, torch.Tensor], List]:
    hosts: Dict[str, torch.Tensor] = {}
    mms: List = []
    try:
        for attr, dtype, row_shape in specs:
            numel = n_rows * math.prod(row_shape)
            if node is not None and numel > 0:
                mms.append(mmap_numa_thp(numel * dtype.itemsize, node=node))
                hosts[attr] = torch.frombuffer(mms[-1], dtype=dtype, count=numel).view(
                    n_rows, *row_shape
                )
            else:
                hosts[attr] = torch.empty(n_rows, *row_shape, dtype=dtype, device="cpu")
    except BaseException:
        # The traceback keeps this frame alive; unmap the rows placed so far now.
        hosts.clear()
        mms.clear()
        raise
    return hosts, mms


def _alloc_spill_rows_with_failover(
    specs: List[_SpillRowSpec],
    *,
    n_rows: int,
    preferred: int,
    layer_id: int,
) -> tuple[Dict[str, torch.Tensor], List, int]:
    """Rows of one layer, all on one node: the stripe node if it has room."""
    nbytes = sum(
        n_rows * math.prod(shape) * dtype.itemsize for _, dtype, shape in specs
    )
    last_err: Optional[EngramNumaError] = None
    for node in _spill_numa_failover_order(preferred, nbytes=nbytes):
        try:
            hosts, mms = _alloc_spill_rows(specs, n_rows=n_rows, node=node)
        except EngramNumaError as e:
            last_err = e
            logger.warning(
                "DSV4.1 spill rows L%s: placing on node %d failed (%s); trying the next node",
                layer_id,
                node,
                e,
            )
            continue
        if node != preferred:
            logger.warning(
                "DSV4.1 spill rows L%s: on node %d instead of stripe node %d",
                layer_id,
                node,
                preferred,
            )
        return hosts, mms, node
    raise EngramNumaError(
        f"spill rows L{layer_id}: no NUMA node took {nbytes} bytes"
    ) from last_err


@contextmanager
def _gc_scans_only_new_objects():
    """Limit gc.collect() to objects created inside the block.

    Repack and pin call gc.collect() several times per layer; with the loaded
    model on the heap each full pass cost ~0.4 s (about half of the per-layer
    time, py-spy). Tensors are still freed by refcount.
    """
    gc.freeze()
    try:
        yield
    finally:
        gc.unfreeze()


def repack_spill_host_for_sm70_marlin(moe: nn.Module) -> None:
    """Pack create-time host rows the same way GPU weights were packed.

    Host buffers are checkpoint layout; SM70 Marlin rewrites GPU tensors in
    ``process_weights_after_loading``. LRU swaps require matching layouts.
    """
    hosts: Optional[Dict[str, torch.Tensor]] = getattr(moe, "_dsv41_spill_host", None)
    if not hosts or "w13_weight" not in hosts:
        return
    gpu = moe.w13_weight
    host_w13 = hosts["w13_weight"]
    host_trail = tuple(host_w13.shape[1:])
    gpu_trail = tuple(gpu.shape[1:])
    gpu_scale = getattr(moe, "w13_weight_scale", None)
    if spill_host_is_marlin_packed(hosts, gpu, gpu_scale):
        logger.info(
            "DSV4.1 spill host already Marlin-packed w13 %s %s scale %s %s",
            host_w13.dtype,
            host_trail,
            hosts["w13_weight_scale"].dtype,
            tuple(hosts["w13_weight_scale"].shape[1:]),
        )
        return
    logger.info(
        "DSV4.1 spill host packing w13 host=%s %s gpu=%s %s host_scale_keys=%s",
        host_w13.dtype,
        host_trail,
        gpu.dtype,
        gpu_trail,
        sorted(k for k in hosts if "scale" in k),
    )
    dummy = nn.Module()
    dummy.orig_dtype = torch.float16
    # Host rows stay on CPU. Do not park GPU w13: restoring it OOMed
    # (384 MiB) on a full card. SM70 pack streams one expert at a time and
    # writes the packed E-stack on CPU when the dummy is not CUDA.
    dummy.w13_weight = nn.Parameter(hosts["w13_weight"].contiguous(), requires_grad=False)
    dummy.w2_weight = nn.Parameter(hosts["w2_weight"].contiguous(), requires_grad=False)
    s13 = hosts.get("w13_weight_scale_inv", hosts.get("w13_weight_scale"))
    s2 = hosts.get("w2_weight_scale_inv", hosts.get("w2_weight_scale"))
    if s13 is None or s2 is None:
        raise RuntimeError(
            "DSV4.1 spill host pack needs w13/w2 scales "
            f"(keys={sorted(hosts)})"
        )
    dummy.w13_weight_scale_inv = nn.Parameter(s13.contiguous(), requires_grad=False)
    dummy.w2_weight_scale_inv = nn.Parameter(s2.contiguous(), requires_grad=False)
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        _prepare_moe_mxfp4_layer_for_sm70_marlin,
    )

    if torch.cuda.is_available():
        free_b, _total_b = torch.cuda.mem_get_info()
        logger.info(
            "DSV4.1 spill host pack GPU free=%.1f MiB before dummy pack",
            free_b / (1024**2),
        )
    try:
        _prepare_moe_mxfp4_layer_for_sm70_marlin(dummy)
        packed: Dict[str, torch.Tensor] = {}
        for attr in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_bias",
            "w2_weight_bias",
        ):
            t = getattr(dummy, attr, None)
            if t is None:
                continue
            packed[attr] = t.detach().to("cpu").contiguous()
            del t
        moe._dsv41_spill_host = packed  # type: ignore[attr-defined]
        logger.info(
            "DSV4.1 spill host Marlin-packed w13 %s %s -> %s %s",
            host_w13.dtype,
            host_trail,
            packed["w13_weight"].dtype,
            tuple(packed["w13_weight"].shape[1:]),
        )
    finally:
        if getattr(dummy, "workspace", None) is not None:
            del dummy.workspace
        del dummy
        gc.collect()
        torch.cuda.empty_cache()


# cudaHostRegisterMapped: pins for DMA *and* maps into the CUDA address space
# so a later in-graph UVA page-in can read the mirror directly.
_CUDA_HOST_REGISTER_MAPPED = 0x02


def _spill_numa_nodes() -> List[int]:
    nodes: List[int] = []
    for s in envs.SGLANG_DSV41_EXPERT_SPILL_NUMA_NODES.get():
        try:
            nodes.append(int(s))
        except ValueError:
            logger.warning("SGLANG_DSV41_EXPERT_SPILL_NUMA_NODES: ignoring %r", s)
    if nodes:
        return nodes
    # Unset: the GPU-local node first, then the other memory nodes.
    gpu = gpu_numa_node()
    return [gpu, *sorted(n for n in read_numa_nodes() if n != gpu)]


# Arbitrary margin: covers the reclaim watermarks and the other ranks
# placing the same layer at the same time.
_SPILL_NODE_HEADROOM_BYTES = 2 * GIB


def _spill_numa_failover_order(preferred: int, *, nbytes: int) -> List[int]:
    """The stripe node first, then the other configured nodes.

    Nodes without ``nbytes`` of free or page-cache memory go last: a bound
    allocation there reclaims and swaps on that node, and can end in an OOM
    kill, instead of failing.
    """
    ordered: List[int] = []
    for n in [preferred, *_spill_numa_nodes()]:
        if n not in ordered:
            ordered.append(n)

    def lacks_room(n: int) -> bool:
        node = n if n >= 0 else gpu_numa_node()
        return node_available_bytes(node) < nbytes + _SPILL_NODE_HEADROOM_BYTES

    return sorted(ordered, key=lacks_room)


def _pin_spill_hosts_on_node(
    hosts: Dict[str, torch.Tensor],
    node: int,
) -> tuple[Dict[str, torch.Tensor], List, int, int]:
    """THP-bind + cudaHostRegister every spill row onto ``node``.

    On ``EngramNumaError`` the already-registered pointers are unregistered
    and the mappings dropped so the caller can retry another node.
    """
    mms: List = []
    pinned: Dict[str, torch.Tensor] = {}
    registered_ptrs: List[int] = []
    total = 0
    registered = 0
    cudart = torch.cuda.cudart()
    bind_node = node if node >= 0 else gpu_numa_node()
    try:
        for attr, t in hosts.items():
            t = t.contiguous()
            nbytes = t.numel() * t.element_size()
            if nbytes == 0:
                pinned[attr] = t
                continue
            mm = mmap_numa_thp(nbytes, node=bind_node)
            mms.append(mm)
            p = torch.frombuffer(mm, dtype=t.dtype, count=t.numel()).view(t.shape)
            p.copy_(t)
            err = cudart.cudaHostRegister(
                p.data_ptr(), nbytes, _CUDA_HOST_REGISTER_MAPPED
            )
            if int(err) != 0:
                logger.warning(
                    "DSV4.1 spill mirror cudaHostRegister(%s, %d bytes) failed err=%s; "
                    "row stays NUMA-bound but pageable",
                    attr,
                    nbytes,
                    int(err),
                )
            else:
                registered += nbytes
                registered_ptrs.append(int(p.data_ptr()))
            total += nbytes
            pinned[attr] = p
    except Exception:
        for ptr in registered_ptrs:
            try:
                cudart.cudaHostUnregister(ptr)
            except Exception:
                pass
        del pinned
        del mms
        raise
    return pinned, mms, total, registered


def pin_spill_host_numa(moe: nn.Module, layer_ordinal: int) -> Optional[int]:
    """Move the packed host mirror into node-local THP mappings and
    ``cudaHostRegister`` them (mapped).

    ``repack_spill_host_for_sm70_marlin`` leaves the rows as pageable tensors
    from the default allocator; under memory pressure those were paged out
    and every LRU miss became a swap-in through the RAID. Layers are striped over the configured nodes so the mirror
    spreads across sockets. Spill=12 can exhaust the GPU-local node's THP
    remainder (1G hugepages already hold Engram); retry the other stripe
    node instead of aborting (explicit overflow, not silent UPI). Returns
    the node used, or None if skipped.
    """
    hosts: Optional[Dict[str, torch.Tensor]] = getattr(moe, "_dsv41_spill_host", None)
    if not hosts or not torch.cuda.is_available():
        return None
    if getattr(moe, "_dsv41_spill_host_pinned", False):
        return getattr(moe, "_dsv41_spill_host_node", None)
    preferred = getattr(moe, "_dsv41_spill_numa_node", None)
    if preferred is None:
        nodes = _spill_numa_nodes()
        preferred = nodes[layer_ordinal % len(nodes)]
    last_err: Optional[BaseException] = None
    pinned: Optional[Dict[str, torch.Tensor]] = None
    mms: List = []
    total = 0
    registered = 0
    node = preferred
    nbytes = sum(t.numel() * t.element_size() for t in hosts.values())
    for node in _spill_numa_failover_order(int(preferred), nbytes=nbytes):
        try:
            pinned, mms, total, registered = _pin_spill_hosts_on_node(hosts, node)
            if node != preferred:
                logger.warning(
                    "DSV4.1 spill mirror L%s: pinned on node %d instead of node %d",
                    layer_ordinal,
                    node,
                    preferred,
                )
            last_err = None
            break
        except EngramNumaError as e:
            last_err = e
            logger.warning(
                "DSV4.1 spill mirror L%s: placing on node %d failed (%s); trying the next node",
                layer_ordinal,
                node,
                e,
            )
    if last_err is not None or pinned is None:
        raise last_err or EngramNumaError(
            f"spill mirror L{layer_ordinal}: no NUMA node accepted the pin"
        )
    moe._dsv41_spill_host = pinned  # type: ignore[attr-defined]
    moe._dsv41_spill_host_mms = mms  # type: ignore[attr-defined]
    moe._dsv41_spill_host_pinned = registered == total  # type: ignore[attr-defined]
    moe._dsv41_spill_host_node = node  # type: ignore[attr-defined]
    del hosts
    # Construct-time checkpoint-layout mappings: the repack replaced their
    # tensors, so dropping the mmap handles lets them unmap now (same node
    # as the pinned rows -> footprint flat, not doubled).
    moe._dsv41_spill_host_ctor_mms = None  # type: ignore[attr-defined]
    gc.collect()
    logger.info(
        "DSV4.1 spill mirror L%s: %.0f MiB on NUMA node %d, pinned+mapped %.0f MiB",
        layer_ordinal,
        total / (1024**2),
        node,
        registered / (1024**2),
    )
    return node


_COLD_SET_CACHE: Dict[str, Optional[torch.Tensor]] = {}


def _load_cold_set_table(path: str) -> Optional[torch.Tensor]:
    """``cold_ids`` int64 [layers, ep, S], coldest first, from JSON or a torch file. Cached per path."""
    if path in _COLD_SET_CACHE:
        return _COLD_SET_CACHE[path]
    table: Optional[torch.Tensor] = None
    try:
        if path.endswith(".json"):
            # The shipped table is JSON: plain data, nothing to unpickle.
            with open(path) as f:
                obj = json.load(f)
        else:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        t = obj["cold_ids"] if isinstance(obj, dict) else obj
        table = torch.as_tensor(t, dtype=torch.int64)
        if table.ndim != 3:
            raise ValueError(f"cold_ids must be [layers, ep, S], got {tuple(table.shape)}")
        logger.info(
            "DSV4.1 spill cold set %s: layers=%d ep=%d S=%d source=%s",
            path,
            *table.shape,
            (obj.get("source") if isinstance(obj, dict) else None),
        )
    except Exception as e:  # noqa: BLE001 - fall back to tail placement, loudly
        logger.error("DSV4.1 spill cold set %s unusable (%s); using tail placement", path, e)
        table = None
    _COLD_SET_CACHE[path] = table
    return table


def spill_placement(moe: nn.Module) -> tuple[List[int], List[int]]:
    """(kept_ids, cold_ids) local routed ids for this (layer, ep rank).

    Host row of a cold expert = its index in ``cold_ids``; GPU slot of a kept
    expert = its index in ``kept_ids``; shared experts follow the kept rows.
    Without a cold-set table this is today's placement: cold = tail
    ``[n_kept_routed, n_routed)``, kept = ``[0, n_kept_routed)``.
    """
    cached = getattr(moe, "_dsv41_spill_placement", None)
    # Loader threads hit this concurrently (deepseek_v4 load_weights).
    # Placement is the publish flag: only return it once both slot maps exist.
    if (
        cached is not None
        and getattr(moe, "_dsv41_spill_kept_slot", None) is not None
        and getattr(moe, "_dsv41_spill_host_slot", None) is not None
    ):
        return cached
    plan: RoutedExpertSpillPlan = moe._dsv41_expert_spill_plan  # type: ignore[attr-defined]
    n_routed = int(getattr(moe, "_num_local_routed", 0)) or plan.local_routed
    cold: Optional[List[int]] = None
    path = envs.SGLANG_DSV41_EXPERT_SPILL_COLD_SET.get()
    if path and plan.n_spilled > 0:
        table = _load_cold_set_table(path)
        layer = int(getattr(moe, "layer_id", -1))
        rank = int(getattr(moe, "moe_ep_rank", 0))
        if table is not None and 0 <= layer < table.shape[0] and rank < table.shape[1]:
            ids: List[int] = []
            for x in table[layer, rank].tolist():
                if 0 <= x < n_routed and x not in ids:
                    ids.append(int(x))
                if len(ids) == plan.n_spilled:
                    break
            if len(ids) == plan.n_spilled:
                cold = sorted(ids)
            else:
                logger.warning(
                    "DSV4.1 spill cold set L%d r%d has %d usable ids < n_spilled %d; tail placement",
                    layer, rank, len(ids), plan.n_spilled,
                )
        elif table is not None:
            logger.warning(
                "DSV4.1 spill cold set has no entry for layer %d rank %d; tail placement",
                layer, rank,
            )
    moe._dsv41_spill_cold_source = "table" if cold is not None else "tail"  # type: ignore[attr-defined]
    if cold is None:
        cold = list(range(plan.n_kept_routed, n_routed))
    cold_set = set(cold)
    kept = [i for i in range(n_routed) if i not in cold_set]
    placement = (kept, cold)
    moe._dsv41_spill_kept_slot = {e: s for s, e in enumerate(kept)}  # type: ignore[attr-defined]
    moe._dsv41_spill_host_slot = {e: s for s, e in enumerate(cold)}  # type: ignore[attr-defined]
    moe._dsv41_spill_placement = placement  # type: ignore[attr-defined]
    return placement


def spilled_expert_host_row(
    moe: nn.Module, param: torch.Tensor, expert_id: int
) -> Optional[tuple[torch.Tensor, int]]:
    """If this local expert is spilled, return (host_tensor, host_row)."""
    plan: Optional[RoutedExpertSpillPlan] = getattr(moe, "_dsv41_expert_spill_plan", None)
    hosts: Optional[Dict[str, torch.Tensor]] = getattr(moe, "_dsv41_spill_host", None)
    if plan is None or not hosts or plan.n_spilled <= 0:
        return None
    n_routed = int(getattr(moe, "_num_local_routed", 0))
    if expert_id < 0 or expert_id >= n_routed:
        return None
    spill_placement(moe)
    host_row = moe._dsv41_spill_host_slot.get(expert_id)  # type: ignore[attr-defined]
    if host_row is None:
        return None
    src_ptr = param.data_ptr()
    for attr, host in hosts.items():
        gp = getattr(moe, attr, None)
        if gp is not None and gp.data_ptr() == src_ptr:
            return host, host_row
    return None


def remap_shared_expert_gpu_index(moe: nn.Module, expert_id: int) -> int:
    """GPU slot of a non-spilled local expert when GPU tensors are pre-shrunk.

    Kept routed experts sit at their index in ``kept_ids`` (identity for the
    tail placement); shared slots follow the kept rows.
    """
    plan: Optional[RoutedExpertSpillPlan] = getattr(moe, "_dsv41_expert_spill_plan", None)
    n_routed = int(getattr(moe, "_num_local_routed", 0))
    if plan is None or plan.n_spilled <= 0:
        return expert_id
    if expert_id >= n_routed:
        return plan.n_kept_routed + (expert_id - n_routed)
    spill_placement(moe)
    slot = moe._dsv41_spill_kept_slot.get(expert_id)  # type: ignore[attr-defined]
    if slot is None:
        raise KeyError(f"local expert {expert_id} is spilled; no GPU slot")
    return slot


class RoutedExpertLru:
    """Row-wise LRU over expert-dim tensors, spilling cold routed rows.

    Shared expert rows are the tail ``n_shared`` and never leave the device.
    ``ensure(ids)`` copies spilled rows into GPU slots, evicting the LRU hot
    routed expert onto the pinned host copy. If ``unique(ids)`` exceeds
    ``n_kept_routed``, last-resort eviction thrashes within the batch (the
    last ensured id is resident; earlier ones may have been evicted). Optional
    ``siblings`` (w2, scales) share the same slot map so Marlin w13/w2 stay
    aligned.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        *,
        n_shared: int = 1,
        n_spilled: int,
        pin_memory: bool = False,
        siblings: Optional[Sequence[torch.Tensor]] = None,
        hosts: Optional[Sequence[torch.Tensor]] = None,
        logical_n_experts: Optional[int] = None,
        already_shrunk: bool = False,
        kept_ids: Optional[Sequence[int]] = None,
        cold_ids: Optional[Sequence[int]] = None,
    ):
        if weight.ndim < 1:
            raise ValueError("expert weight must have an expert dimension")
        n_experts = int(logical_n_experts or weight.shape[0])
        if n_shared < 0 or n_spilled < 0:
            raise ValueError("n_shared and n_spilled must be >= 0")
        if n_shared + n_spilled > n_experts:
            raise ValueError(
                f"n_shared={n_shared} + n_spilled={n_spilled} > n_experts={n_experts}"
            )
        extra = list(siblings or ())
        gpu_rows = n_experts - n_spilled if already_shrunk else n_experts
        for t in extra:
            if int(t.shape[0]) != int(weight.shape[0]):
                raise ValueError(
                    f"sibling expert dim {t.shape[0]} != primary {weight.shape[0]}"
                )
        if already_shrunk and int(weight.shape[0]) != gpu_rows:
            raise ValueError(
                f"already-shrunk GPU dim {weight.shape[0]} != kept+shared {gpu_rows}"
            )
        self.n_experts = n_experts
        self.n_shared = n_shared
        self.n_routed = n_experts - n_shared
        self.n_spilled = n_spilled
        self.n_kept_routed = self.n_routed - n_spilled
        self._pin_memory = pin_memory
        # Which local routed ids start on host. Default = tail.
        if cold_ids is None:
            cold_list = list(range(self.n_kept_routed, self.n_routed))
        else:
            cold_list = sorted(int(i) for i in cold_ids)
        if len(cold_list) != n_spilled or any(
            i < 0 or i >= self.n_routed for i in cold_list
        ) or len(set(cold_list)) != len(cold_list):
            raise ValueError(f"cold_ids must be {n_spilled} distinct routed ids: {cold_list}")
        cold_set = set(cold_list)
        if kept_ids is None:
            kept_list = [i for i in range(self.n_routed) if i not in cold_set]
        else:
            kept_list = [int(i) for i in kept_ids]
            if len(kept_list) != self.n_kept_routed or cold_set.intersection(kept_list):
                raise ValueError("kept_ids must be the routed ids not in cold_ids")
        self._kept_ids = kept_list
        self._cold_ids = cold_list
        self._cold_set = cold_set  # membership: "its home is the host mirror"
        self._gpus: List[torch.Tensor] = [weight, *extra]
        if hosts is not None:
            host_list = list(hosts)
            if len(host_list) != len(self._gpus):
                raise ValueError(
                    f"hosts len {len(host_list)} != gpu tensors {len(self._gpus)}"
                )
            self._hosts = host_list
        else:
            self._hosts = [
                self._alloc_host(t, pin_memory=pin_memory) for t in self._gpus
            ]
        self._host_mm = getattr(self, "_host_mm", None)
        # gpu_slot -> expert id for routed slots [0, n_kept_routed)
        self._slot_to_id = list(kept_list)
        self._id_to_slot = {e: s for s, e in enumerate(kept_list)}
        self._host_slot_to_id = list(cold_list)
        self._id_to_host_slot = {e: s for s, e in enumerate(cold_list)}
        self._lru: OrderedDict[int, None] = OrderedDict(
            (i, None) for i in range(self.n_kept_routed)
        )
        self.applied = already_shrunk
        self._staging: Optional[List[torch.Tensor]] = None
        self._map_table: Optional[torch.Tensor] = None
        self._host_map_table: Optional[torch.Tensor] = None
        self.n_swaps = 0
        # Last-resort evictions when unique(batch) > n_kept_routed (intra-batch thrash).
        self.n_thrash = 0

    def _alloc_host(self, weight: torch.Tensor, *, pin_memory: bool) -> torch.Tensor:
        """Host holds only spilled routed rows (logical n_kept .. n_routed-1)."""
        device = weight.device
        n_host = self.n_spilled
        row_shape = weight.shape[1:]
        row_numel = int(weight.reshape(weight.shape[0], -1).shape[1]) if n_host else 0
        host_bytes = n_host * row_numel * weight.element_size()
        use_numa = (
            pin_memory
            and device.type == "cuda"
            and bool(envs.SGLANG_ENABLE_DSV41_EXPERT_SPILL_NUMA.get())
            and host_bytes > 0
        )
        if use_numa:
            node = engram_numa_node()
            if node < 0:
                node = gpu_numa_node()
            host_mm = mmap_numa_thp(host_bytes, node=node)
            if getattr(self, "_host_mms", None) is None:
                self._host_mms = []
            self._host_mms.append(host_mm)
            host = torch.frombuffer(
                host_mm, dtype=weight.dtype, count=n_host * row_numel
            ).view(n_host, *row_shape)
            err = torch.cuda.cudart().cudaHostRegister(
                host.data_ptr(), host_bytes, _CUDA_HOST_REGISTER_MAPPED
            )
            if int(err) != 0:
                logger.warning(
                    "cudaHostRegister expert-spill %s bytes failed err=%s; "
                    "NUMA bind still holds",
                    host_bytes,
                    int(err),
                )
        else:
            host = torch.empty(
                n_host,
                *row_shape,
                dtype=weight.dtype,
                device="cpu",
                pin_memory=pin_memory and device.type == "cuda",
            )
        if n_host and weight.shape[0] >= self.n_routed:
            host.copy_(weight[self._cold_ids].detach().to("cpu"))
        return host

    @property
    def _gpu(self) -> torch.Tensor:
        return self._gpus[0]

    @property
    def _host(self) -> torch.Tensor:
        return self._hosts[0]

    @property
    def gpu_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._gpus)

    def _shrink_one(self, tensor: torch.Tensor) -> torch.Tensor:
        kept = tensor[self._kept_ids].clone()
        if not self.n_shared:
            return kept
        shared = tensor[self.n_routed :].clone()
        return torch.cat([kept, shared], dim=0)

    def apply_shrink(self) -> torch.Tensor:
        """Drop spilled routed rows from every GPU tensor; shared tail is kept."""
        if self.applied:
            return self._gpus[0]
        self._gpus = [self._shrink_one(t) for t in self._gpus]
        self.applied = True
        return self._gpus[0]

    def shrunk_tensors(self) -> List[torch.Tensor]:
        if not self.applied:
            raise RuntimeError("apply_shrink() first")
        return list(self._gpus)

    def physical_ids(self, logical_ids: Iterable[int]) -> list[int]:
        """Map logical expert ids onto the shrunken GPU table (after apply_shrink)."""
        out = []
        for i in logical_ids:
            i = int(i)
            if i < 0 or i >= self.n_experts:
                raise IndexError(i)
            if i >= self.n_routed:
                # shared tail sits after kept routed rows
                out.append(self.n_kept_routed + (i - self.n_routed))
                continue
            slot = self._id_to_slot.get(i)
            if slot is None:
                raise KeyError(
                    f"expert {i} is spilled; call ensure() before the MoE runner"
                )
            out.append(slot)
        return out

    def _ensure_device_tables(self, like: torch.Tensor) -> None:
        """GPU map_table (logical -> slot) and host_map (logical -> host row)."""
        device = like.device
        dtype = torch.int32
        need = (
            self._map_table is None
            or self._host_map_table is None
            or self._map_table.device != device
            or self._map_table.dtype != dtype
            or int(self._map_table.numel()) != self.n_experts
        )
        if not need:
            return
        cpu_map = torch.full((self.n_experts,), -1, dtype=dtype)
        cpu_host = torch.full((self.n_experts,), -1, dtype=dtype)
        for logical, slot in self._id_to_slot.items():
            cpu_map[logical] = int(slot)
        for logical, hs in self._id_to_host_slot.items():
            cpu_host[logical] = int(hs)
        if self.n_shared:
            for i in range(self.n_shared):
                cpu_map[self.n_routed + i] = self.n_kept_routed + i
        self._map_table = cpu_map.to(device=device)
        self._host_map_table = cpu_host.to(device=device)

    def map_ids(self, logical_ids: torch.Tensor) -> torch.Tensor:
        """Vectorized ``physical_ids``; negative ids (EP remote) pass through."""
        self._ensure_device_tables(logical_ids)
        table = self._map_table
        assert table is not None
        n = int(table.shape[0])
        idx = logical_ids.clamp(min=0, max=max(n - 1, 0)).to(torch.int64)
        mapped = table[idx].to(dtype=logical_ids.dtype)
        return torch.where(logical_ids < 0, logical_ids, mapped)

    def _ensure_staging(self) -> List[torch.Tensor]:
        if self._staging is not None:
            return self._staging
        pin = bool(self._pin_memory) and any(t.is_cuda for t in self._gpus)
        self._staging = [
            torch.empty(
                host.shape[1:],
                dtype=host.dtype,
                device="cpu",
                pin_memory=pin,
            )
            for host in self._hosts
        ]
        return self._staging

    def _swap_row(
        self,
        gpu: torch.Tensor,
        host: torch.Tensor,
        staging: torch.Tensor,
        *,
        victim_slot: int,
        victim_id: int,
        logical: int,
    ) -> None:
        hs = self._id_to_host_slot[logical]
        incoming = host[hs]
        victim = gpu[victim_slot]
        if tuple(incoming.shape) != tuple(victim.shape) or incoming.dtype != victim.dtype:
            raise RuntimeError(
                "DSV4.1 spill LRU swap layout mismatch: "
                f"host{tuple(incoming.shape)} {incoming.dtype} vs "
                f"gpu{tuple(victim.shape)} {victim.dtype}. "
                "Host pack must run after GPU process_weights_after_loading."
            )
        # Pinned/host-registered copy_. Do not clone().to() — that is pageable
        # HtoD and stalls the CPU while NCCL waits (decode profile: 9k copies).
        staging.copy_(victim)
        victim.copy_(incoming)
        incoming.copy_(staging)
        if not self.applied and gpu.shape[0] > logical:
            gpu[logical].copy_(gpu[victim_slot])

    # Residents touched within this many most-recent slots are never chosen as
    # a cold-first victim (a cold expert that is hot *right now* should not
    # thrash against the next cold miss); plain LRU decides then.
    _MRU_PROTECT = 8

    def _pick_victim(self, needed: set, current: Optional[int] = None) -> int:
        """Slot to evict. Among residents outside the MRU window,
        prefer one whose home is the host mirror (a cold expert brought in
        earlier) over a kept expert, so the frequency-hot set stays on the
        GPU; otherwise the oldest resident. Never a slot needed by a
        currently-resident batch id while a non-needed occupant exists.

        Last resort (unique(batch) > n_slots): evict the oldest LRU occupant
        that is not ``current``. The evicted id maps to -1 afterwards, so the
        MoE path splits such batches into slot-sized passes first
        (``run_moe_with_expert_spill``).
        """
        n = len(self._lru)
        protect = min(self._MRU_PROTECT, n // 2)
        fallback = None
        for idx, slot in enumerate(self._lru):  # oldest first
            occupant = self._slot_to_id[slot]
            if occupant in needed:
                continue
            if idx < n - protect and occupant in self._cold_set:
                return slot
            if fallback is None:
                fallback = slot
        if fallback is not None:
            return fallback
        for slot in self._lru:
            occupant = self._slot_to_id[slot]
            if occupant == current:
                continue
            return slot
        raise RuntimeError("every GPU expert slot is needed by this batch")

    def ensure(self, logical_ids: Iterable[int]) -> None:
        """Make each routed id resident in a GPU slot, spilling LRU victims."""
        ids = [int(i) for i in logical_ids]
        batch = {i for i in ids if 0 <= i < self.n_routed}
        if not self.applied and self._kept_ids != list(range(self.n_kept_routed)):
            raise RuntimeError("non-tail spill placement requires apply_shrink() first")
        logged_thrash = False
        for logical in ids:
            if logical < 0:
                continue
            if logical >= self.n_experts:
                raise IndexError(logical)
            if logical >= self.n_routed:
                continue  # shared, always resident
            slot = self._id_to_slot.get(logical)
            if slot is not None:
                self._lru.move_to_end(slot)
                continue
            if self.n_kept_routed == 0:
                raise RuntimeError("no GPU slots to hold a routed expert")
            # Occupied slots, not future batch ids: an expert not yet loaded
            # does not hold a GPU row, so it must not pin a victim.
            needed = set(self._id_to_slot).intersection(batch)
            victim_slot = self._pick_victim(needed, current=logical)
            victim_id = self._slot_to_id[victim_slot]
            if victim_id in needed:
                self.n_thrash += 1
                if not logged_thrash:
                    logged_thrash = True
                    logger.info(
                        "DSV4.1 spill LRU intra-batch thrash: unique=%d slots=%d "
                        "(working set exceeds GPU expert slots)",
                        len(batch),
                        self.n_kept_routed,
                    )
            del self._lru[victim_slot]
            staging = self._ensure_staging()
            for gpu, host, st in zip(self._gpus, self._hosts, staging):
                self._swap_row(
                    gpu,
                    host,
                    st,
                    victim_slot=victim_slot,
                    victim_id=victim_id,
                    logical=logical,
                )
            hs = self._id_to_host_slot.pop(logical)
            self._id_to_host_slot[victim_id] = hs
            self._host_slot_to_id[hs] = victim_id
            del self._id_to_slot[victim_id]
            self._id_to_slot[logical] = victim_slot
            self._slot_to_id[victim_slot] = logical
            self._lru[victim_slot] = None
            self.n_swaps += 1
            table = self._map_table
            if table is not None:
                table[victim_id] = -1
                table[logical] = victim_slot
            host_table = self._host_map_table
            if host_table is not None:
                host_table[logical] = -1
                host_table[victim_id] = hs

    def gather_rows(self, logical_ids: Iterable[int]) -> torch.Tensor:
        self.ensure(logical_ids)
        ids = list(logical_ids)
        slots = self.physical_ids(ids) if self.applied else [
            self._id_to_slot[i] if i < self.n_routed else i for i in ids
        ]
        return self._gpus[0][slots]


def cpu_offload_gb_tax() -> str:
    return (
        "--cpu-offload-gb uses OffloaderV1, which walks decoder layers in order "
        "and moves whole module parameters (attention + MoE) to pinned host. "
        "Each forward does state_dict().to(device), so CSA2, the indexer, and "
        "shared experts pay a PCIe round trip even though they must stay GPU. "
        "It also races Engram host tables if combined "
        "(see handle_offload_compatibility for PLE). Prefer "
        "SGLANG_DSV41_EXPERT_SPILL_GB + this LRU."
    )


def _expert_dim_params(moe: nn.Module) -> Dict[str, torch.nn.Parameter]:
    n = int(getattr(moe, "num_local_experts", 0))
    gpu_n = int(getattr(moe, "_dsv41_gpu_expert_slots", n) or n)
    out: Dict[str, torch.nn.Parameter] = {}
    for attr in _EXPERT_PARAM_ATTRS:
        p = getattr(moe, attr, None)
        if p is None or not isinstance(p, torch.nn.Parameter) or p.ndim < 1:
            continue
        if int(p.shape[0]) in (n, gpu_n):
            out[attr] = p
    return out


# Per-rank swap-rate telemetry:
# (layer-calls, swaps, decode layer-calls, decode swaps, intra-batch thrash swaps).
_SWAP_STATS = [0, 0, 0, 0, 0]
_SWAP_LOG_EVERY = 40 * 200  # ~200 forward passes of 40 MoE layers


def _note_swaps(swaps: int, n_tokens: int, thrash: int = 0) -> None:
    s = _SWAP_STATS
    s[0] += 1
    s[1] += swaps
    s[4] += thrash
    if n_tokens <= 2:  # decode-shaped (bs 1-2 at np=1)
        s[2] += 1
        s[3] += swaps
    if s[0] >= _SWAP_LOG_EVERY:
        n_layers = 40
        logger.info(
            "DSV4.1 spill swaps: %d over %d layer-calls (%.2f/layer-call); decode-shaped "
            "%d over %d (%.1f swaps per %d-layer token); intra-batch thrash %d",
            s[1], s[0], s[1] / max(s[0], 1),
            s[3], s[2], n_layers * s[3] / max(s[2], 1), n_layers,
            s[4],
        )
        s[0] = s[1] = s[2] = s[3] = s[4] = 0


def spill_landing_slots() -> int:
    """D4-G landing slots (0 disables in-graph page-in)."""
    try:
        return max(int(envs.SGLANG_DSV41_SPILL_LANDING.get() or 0), 0)
    except Exception:
        return 0


def _decode_shaped_max_tokens() -> int:
    """Max T that still uses landing page-in instead of the prefill LRU.

    Greedy pads T to 2. DSpark target-verify is γ+1; D15-0 sizes landing to
    ``6 * (γ+1)``, so ``slots // 6`` recovers that width. Landing=6 stays T≤2.
    """
    slots = spill_landing_slots()
    verify_t = max(slots // 6, 1) if slots else 1
    return max(2, verify_t)


def _decode_shaped_topk(topk_ids: torch.Tensor) -> bool:
    return bool(topk_ids.numel()) and int(topk_ids.shape[0]) <= _decode_shaped_max_tokens()


def _cuda_graph_capturing() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


@dataclass
class SpillLandingPool:
    """Rank-global landing weight table reused every decode layer."""

    n_landing: int
    tensors: Dict[str, torch.Tensor]
    attrs: List[str]
    dst_ptrs: torch.Tensor
    row_bytes: torch.Tensor
    slot_host_row: torch.Tensor
    land_ids: Optional[torch.Tensor]
    quant_info: object


_LANDING_POOLS: Dict[int, SpillLandingPool] = {}


def _attach_landing_pool(moe: nn.Module, n_landing: int) -> None:
    if n_landing <= 0:
        return
    hosts: Optional[Dict[str, torch.Tensor]] = getattr(moe, "_dsv41_spill_host", None)
    params = _expert_dim_params(moe)
    if not hosts or not params:
        return
    attrs = [a for a in _EXPERT_PARAM_ATTRS if a in params and a in hosts]
    if not attrs:
        return
    device = params[attrs[0]].device
    if device.type != "cuda":
        return
    if "w13_weight" not in attrs or "w2_weight" not in attrs:
        logger.warning("DSV4.1 D4-G landing skipped: missing w13/w2")
        return
    if "w13_weight_scale" not in attrs or "w2_weight_scale" not in attrs:
        logger.warning("DSV4.1 D4-G landing skipped: missing Marlin scales")
        return
    key = int(device.index or 0)
    pool = _LANDING_POOLS.get(key)
    if pool is None:
        tensors: Dict[str, torch.Tensor] = {}
        for a in attrs:
            p = params[a]
            tensors[a] = torch.empty(
                (n_landing, *p.shape[1:]),
                dtype=p.dtype,
                device=device,
            )
        row_bytes = torch.tensor(
            [
                tensors[a].reshape(n_landing, -1).shape[1] * tensors[a].element_size()
                for a in attrs
            ],
            dtype=torch.int64,
            device=device,
        )
        dst_ptrs = torch.tensor(
            [tensors[a].data_ptr() for a in attrs],
            dtype=torch.int64,
            device=device,
        )
        slot_host_row = torch.empty(n_landing, dtype=torch.int32, device=device)
        from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo

        quant_info = MarlinMoeQuantInfo(
            w13_qweight=tensors["w13_weight"],
            w2_qweight=tensors["w2_weight"],
            w13_scales=tensors["w13_weight_scale"],
            w2_scales=tensors["w2_weight_scale"],
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            weight_bits=4,
            is_k_full=True,
            w13_bias=tensors.get("w13_weight_bias"),
            w2_bias=tensors.get("w2_weight_bias"),
            expert_map=None,
            global_num_experts=-1,
        )
        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        logger.info(
            "DSV4.1 D4-G landing pool: %d slots attrs=%s %.1f MiB (shared across layers)",
            n_landing,
            attrs,
            nbytes / (1024 * 1024),
        )
        pool = SpillLandingPool(
            n_landing=n_landing,
            tensors=tensors,
            attrs=attrs,
            dst_ptrs=dst_ptrs,
            row_bytes=row_bytes,
            slot_host_row=slot_host_row,
            land_ids=None,
            quant_info=quant_info,
        )
        _LANDING_POOLS[key] = pool
    moe._dsv41_landing_pool = pool  # type: ignore[attr-defined]
    src = [int(hosts[a].data_ptr()) for a in pool.attrs]
    moe._dsv41_uva_src_ptrs = torch.tensor(src, dtype=torch.int64, device=device)  # type: ignore[attr-defined]


def page_in_spill_experts(moe: nn.Module, topk_ids: torch.Tensor) -> torch.Tensor:
    """D4-G: UVA copy of spilled hits into the landing pool; remap topk in place."""
    lru: RoutedExpertLru = moe._dsv41_expert_lru  # type: ignore[attr-defined]
    pool: SpillLandingPool = moe._dsv41_landing_pool  # type: ignore[attr-defined]
    if topk_ids.dtype != torch.int32 or topk_ids.dim() != 2:
        raise RuntimeError(
            f"D4-G page-in requires int32 [T, K] topk_ids, got {topk_ids.dtype} "
            f"{tuple(topk_ids.shape)}"
        )
    if topk_ids.device.type != "cuda":
        raise RuntimeError("D4-G page-in requires CUDA topk_ids")
    topk_ids = topk_ids.contiguous()
    lru._ensure_device_tables(topk_ids)
    n_tok, k = int(topk_ids.shape[0]), int(topk_ids.shape[1])
    buf = pool.land_ids
    if (
        buf is None
        or buf.dtype != topk_ids.dtype
        or buf.device != topk_ids.device
        or buf.dim() != 2
        or int(buf.shape[1]) != k
        or int(buf.shape[0]) < n_tok
    ):
        if _cuda_graph_capturing():
            raise RuntimeError(
                "D4-G landing id buffer missing during CUDA graph capture; "
                "warmup must run page_in_spill_experts first"
            )
        # Pad T to decode-shaped max (2 greedy, γ+1 when landing is 6*(γ+1))
        # so T=1 capture and a later T=verify eager/capture share storage.
        pool.land_ids = torch.empty(
            (max(_decode_shaped_max_tokens(), n_tok), k),
            dtype=topk_ids.dtype,
            device=topk_ids.device,
        )
        buf = pool.land_ids
    land_ids = buf[:n_tok]
    from sglang.kernels.ops.moe.sm70_dsv41_spill_pagein import spill_page_in

    spill_page_in(
        topk_ids,
        land_ids,
        pool.slot_host_row,
        lru._map_table,
        lru._host_map_table,
        moe._dsv41_uva_src_ptrs,
        pool.dst_ptrs,
        pool.row_bytes,
    )
    moe._dsv41_land_ids = land_ids  # type: ignore[attr-defined]
    return topk_ids


def warmup_spill_landing_for_capture(
    model: nn.Module,
    n_tok: int,
    k: int,
    device: torch.device,
) -> None:
    """Allocate ``land_ids`` at verify width before CUDA-graph capture.

    Capture throws if the buffer is still the T=1 ``max(2, n_tok)`` allocation.
    """
    if spill_landing_slots() <= 0:
        return
    pad_t = max(int(n_tok), _decode_shaped_max_tokens(), 2)
    pad_k = max(int(k), 1)
    for moe in model.modules():
        pool = getattr(moe, "_dsv41_landing_pool", None)
        if pool is None:
            continue
        buf = pool.land_ids
        need = (
            buf is None
            or buf.device != device
            or buf.dtype != torch.int32
            or buf.dim() != 2
            or int(buf.shape[0]) < pad_t
            or int(buf.shape[1]) < pad_k
        )
        if need:
            pool.land_ids = torch.empty(
                (pad_t, pad_k), dtype=torch.int32, device=device
            )


def ensure_spill_experts(moe: nn.Module, topk_ids: torch.Tensor) -> torch.Tensor:
    """Host LRU swap + in-place ``map_ids``. Not CUDA-graph safe.

    Wrapped with ``eager_on_graph`` so breakable decode graphs split here.
    ``map_ids`` is ``aten::index`` (IndexKernel); keep it off the captured
    Marlin segment. Writes physical ids into ``topk_ids`` in place so the
    next segment keeps the same buffer address.
    """
    lru = getattr(moe, "_dsv41_expert_lru", None)
    if lru is None or not lru.applied:
        return topk_ids
    valid = topk_ids[topk_ids >= 0]
    if valid.numel():
        before = lru.n_swaps
        before_thrash = lru.n_thrash
        lru.ensure(torch.unique(valid).detach().cpu().tolist())
        _note_swaps(
            lru.n_swaps - before,
            int(topk_ids.shape[0]),
            lru.n_thrash - before_thrash,
        )
    if envs.SGLANG_DSV41_PREFILL_SYNC.get() and topk_ids.numel():
        table = lru._map_table
        logger.info(
            "DSV41 map_ids in min=%s max=%s shape=%s table_n=%s",
            int(topk_ids.min().item()),
            int(topk_ids.max().item()),
            tuple(topk_ids.shape),
            None if table is None else int(table.shape[0]),
        )
    physical = lru.map_ids(topk_ids)
    topk_ids.copy_(physical)
    return topk_ids


try:
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
        eager_on_graph,
    )

    ensure_spill_experts = eager_on_graph(True)(ensure_spill_experts)
except ImportError:
    pass


_DSV41_HIDDEN = 5120


def _host_gemv_hosts_ready(moe: nn.Module) -> bool:
    hosts: Optional[Dict[str, torch.Tensor]] = getattr(moe, "_dsv41_spill_host", None)
    if not hosts:
        return False
    return all(
        k in hosts
        for k in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    )


def _attach_host_gemv_bases(moe: nn.Module) -> bool:
    """Device int64[5] of host row bases + n_host. Stable for CUDA graph."""
    if not _host_gemv_hosts_ready(moe):
        return False
    device = getattr(moe, "w13_weight", None)
    if device is None or not isinstance(device, torch.Tensor) or device.device.type != "cuda":
        return False
    if getattr(moe, "_dsv41_host_gemv_bases", None) is not None:
        return True
    hosts: Dict[str, torch.Tensor] = moe._dsv41_spill_host  # type: ignore[attr-defined]
    n_host = int(hosts["w13_weight"].shape[0])
    moe._dsv41_host_gemv_bases = torch.tensor(  # type: ignore[attr-defined]
        [
            int(hosts["w13_weight"].data_ptr()),
            int(hosts["w13_weight_scale"].data_ptr()),
            int(hosts["w2_weight"].data_ptr()),
            int(hosts["w2_weight_scale"].data_ptr()),
            n_host,
        ],
        dtype=torch.int64,
        device=device.device,
    )
    moe._dsv41_host_gemv_pending = False  # type: ignore[attr-defined]
    return True


def _host_gemv_decode_ready(moe: nn.Module) -> bool:
    from sglang.kernels.ops.moe.sm70_dsv41_spill_host_gemv import (
        host_gemv_start,
        sm70_dsv41_host_gemv_available,
    )

    if not sm70_dsv41_host_gemv_available():
        return False
    if getattr(moe, "_dsv41_host_gemv_bases", None) is None:
        if _cuda_graph_capturing() or not _attach_host_gemv_bases(moe):
            return False
    return host_gemv_start()


def spill_request_host_gemv(
    moe: nn.Module,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    hidden: torch.Tensor,
) -> None:
    """D4-H: remap kept ids and post spilled hits; CPU overlaps the GPU GEMV."""
    from sglang.kernels.ops.moe.sm70_dsv41_spill_host_gemv import spill_request

    lru: RoutedExpertLru = moe._dsv41_expert_lru  # type: ignore[attr-defined]
    if topk_ids.dtype != torch.int32 or topk_ids.dim() != 2:
        raise RuntimeError(
            f"D4-H spill_request requires int32 [T, K] topk_ids, got {topk_ids.dtype} "
            f"{tuple(topk_ids.shape)}"
        )
    lru._ensure_device_tables(topk_ids)
    bases = moe._dsv41_host_gemv_bases  # type: ignore[attr-defined]
    if hidden.dim() != 2:
        hidden = hidden.view(-1, hidden.shape[-1])
    if int(hidden.shape[-1]) != _DSV41_HIDDEN:
        hidden = hidden[:, :_DSV41_HIDDEN]
    if not hidden.is_contiguous():
        hidden = hidden.contiguous()
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.float()
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()
    ids = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
    spill_request(
        ids,
        topk_weights,
        hidden,
        lru._map_table,
        lru._host_map_table,
        bases,
    )
    if ids is not topk_ids:
        topk_ids.copy_(ids)
    moe._dsv41_host_gemv_pending = True  # type: ignore[attr-defined]
    moe._dsv41_land_ids = None  # type: ignore[attr-defined]


def spill_join_host_gemv(moe: nn.Module, output: torch.Tensor) -> None:
    """D4-H: wait for the CPU GEMV and add y into the GPU MoE output."""
    from sglang.kernels.ops.moe.sm70_dsv41_spill_host_gemv import spill_join

    if not output.is_contiguous():
        raise RuntimeError("D4-H spill_join requires a contiguous output")
    spill_join(output)
    moe._dsv41_host_gemv_pending = False  # type: ignore[attr-defined]


def remap_dispatch_for_expert_spill(moe: nn.Module, dispatch_output):
    """Prefill: LRU ensure + map_ids. Decode: D4-H host GEMV or D4-G landing."""
    lru = getattr(moe, "_dsv41_expert_lru", None)
    if lru is None or not lru.applied:
        return dispatch_output
    topk_output = getattr(dispatch_output, "topk_output", None)
    if topk_output is None or not hasattr(topk_output, "topk_ids"):
        return dispatch_output
    ids = topk_output.topk_ids
    moe._dsv41_host_gemv_pending = False  # type: ignore[attr-defined]
    if _decode_shaped_topk(ids) and _host_gemv_decode_ready(moe):
        hidden = getattr(dispatch_output, "hidden_states", None)
        weights = getattr(topk_output, "topk_weights", None)
        if hidden is not None and weights is not None:
            spill_request_host_gemv(moe, ids, weights, hidden)
            return dispatch_output
    pool = getattr(moe, "_dsv41_landing_pool", None)
    if (
        pool is not None
        and spill_landing_slots() > 0
        and _decode_shaped_topk(ids)
    ):
        page_in_spill_experts(moe, ids)
    else:
        moe._dsv41_land_ids = None  # type: ignore[attr-defined]
        ensure_spill_experts(moe, ids)
    return dispatch_output


def _prefill_pass_groups(
    lru: RoutedExpertLru, topk_ids: torch.Tensor
) -> Optional[List[List[int]]]:
    """Routed-id groups that each fit the GPU slots, or None for one pass.

    A single ``ensure`` over more unique ids than slots evicts ids of the same
    batch, and ``map_ids`` then sends their tokens to -1 (expert skipped).
    Residents go in the first group so it needs the fewest swaps.
    """
    if _decode_shaped_topk(topk_ids):
        return None
    valid = topk_ids[topk_ids >= 0]
    if not valid.numel():
        return None
    routed = [i for i in torch.unique(valid).tolist() if i < lru.n_routed]
    slots = lru.n_kept_routed
    if len(routed) <= slots:
        return None
    resident = [i for i in routed if i in lru._id_to_slot]
    missing = [i for i in routed if i not in lru._id_to_slot]
    room = slots - len(resident)
    groups = [resident + missing[:room]]
    rest = missing[room:]
    groups += [rest[i : i + slots] for i in range(0, len(rest), slots)]
    return groups


def _map_pass(
    lru: RoutedExpertLru,
    logical: torch.Tensor,
    group: List[int],
    *,
    include_shared: bool,
) -> torch.Tensor:
    """Physical ids for ``group`` (plus shared once); every other choice -1."""
    before, before_thrash = lru.n_swaps, lru.n_thrash
    lru.ensure(group)
    _note_swaps(
        lru.n_swaps - before, int(logical.shape[0]), lru.n_thrash - before_thrash
    )
    physical = lru.map_ids(logical)
    member = torch.zeros(lru.n_experts, dtype=torch.bool, device=logical.device)
    member[torch.tensor(group, dtype=torch.int64, device=logical.device)] = True
    if include_shared:
        member[lru.n_routed :] = True
    keep = (logical >= 0) & member[logical.clamp(min=0).to(torch.int64)]
    return torch.where(keep, physical, torch.full_like(physical, -1))


def _prefill_reads_landing(moe: nn.Module, topk_ids: torch.Tensor) -> bool:
    return (
        bool(envs.SGLANG_DSV41_PREFILL_LANDING.get())
        and getattr(moe, "_dsv41_landing_pool", None) is not None
        and spill_landing_slots() > 0
        and bool(topk_ids.numel())
        and not _decode_shaped_topk(topk_ids)
    )


# Per-rank prefill landing telemetry: (layer-calls, host rows copied, tokens).
_LANDING_STATS = [0, 0, 0]
_LANDING_LOG_EVERY = 40 * 50  # ~50 prefill forwards of 40 MoE layers


def _note_landing_rows(rows: int, n_tokens: int) -> None:
    s = _LANDING_STATS
    s[0] += 1
    s[1] += rows
    s[2] += n_tokens
    if s[0] >= _LANDING_LOG_EVERY:
        logger.info(
            "DSV4.1 prefill landing: %.1f host rows copied per layer-call "
            "(%.0f tokens per call, %d calls)",
            s[1] / s[0], s[2] / s[0], s[0],
        )
        s[0] = s[1] = s[2] = 0


def _host_row_runs(rows: List[int]) -> List[tuple[int, int, int]]:
    """(first host row, first landing slot, length) for each run of
    consecutive host rows, so each run is one copy per tensor."""
    runs: List[tuple[int, int, int]] = []
    for slot, row in enumerate(rows):
        if runs and runs[-1][0] + runs[-1][2] == row:
            first, first_slot, length = runs[-1]
            runs[-1] = (first, first_slot, length + 1)
        else:
            runs.append((row, slot, 1))
    return runs


def _run_prefill_through_landing(
    moe: nn.Module, lru: RoutedExpertLru, dispatch_output, apply
):
    """Prefill without GPU-slot swaps: copy the host rows this batch routes to
    into the landing pool and run them as landing ids.

    No write-back and no LRU change, so decode keeps the placed GPU set.
    One host sync per layer picks the rows; kept ids run in pass 0 only.
    """
    pool: SpillLandingPool = moe._dsv41_landing_pool  # type: ignore[attr-defined]
    hosts: Dict[str, torch.Tensor] = moe._dsv41_spill_host  # type: ignore[attr-defined]
    topk_output = dispatch_output.topk_output
    logical = topk_output.topk_ids
    lru._ensure_device_tables(logical)
    index = logical.clamp(min=0).to(torch.int64)
    valid = logical >= 0
    kept = torch.where(valid, lru._map_table[index], -1).to(logical.dtype)
    host_row = torch.where(valid, lru._host_map_table[index], -1).to(logical.dtype)
    rows = torch.unique(host_row[host_row >= 0]).tolist()
    _note_landing_rows(len(rows), int(logical.shape[0]))
    n_host = int(hosts[pool.attrs[0]].shape[0])
    moe._dsv41_host_gemv_pending = False  # type: ignore[attr-defined]
    combined = None
    for start in range(0, max(len(rows), 1), pool.n_landing):
        chunk = rows[start : start + pool.n_landing]
        land_ids = None
        if chunk:
            for first, slot, length in _host_row_runs(chunk):
                for attr in pool.attrs:
                    pool.tensors[attr][slot : slot + length].copy_(
                        hosts[attr][first : first + length], non_blocking=True
                    )
            slot_of_row = torch.full(
                (n_host,), -1, dtype=logical.dtype, device=logical.device
            )
            slot_of_row[torch.tensor(chunk, device=logical.device)] = torch.arange(
                len(chunk), dtype=logical.dtype, device=logical.device
            )
            land_ids = torch.where(
                host_row >= 0, slot_of_row[host_row.clamp(min=0).to(torch.int64)], -1
            )
        moe._dsv41_land_ids = land_ids  # type: ignore[attr-defined]
        ids = kept if start == 0 else torch.full_like(kept, -1)
        out = apply(
            layer=moe,
            dispatch_output=dispatch_output._replace(
                topk_output=topk_output._replace(topk_ids=ids)
            ),
        )
        if combined is None:
            combined = out
        else:
            combined = combined._replace(
                hidden_states=combined.hidden_states + out.hidden_states
            )
    moe._dsv41_land_ids = None  # type: ignore[attr-defined]
    return combined


def run_moe_with_expert_spill(moe: nn.Module, dispatch_output, apply):
    """``apply`` once, or once per slot-sized expert group with summed outputs.

    The passes are exact: each (token, expert) choice runs in exactly one
    pass, and the MoE output is a sum over choices.
    """
    lru = getattr(moe, "_dsv41_expert_lru", None)
    topk_output = getattr(dispatch_output, "topk_output", None)
    groups = None
    if lru is not None and lru.applied and topk_output is not None:
        if _prefill_reads_landing(moe, topk_output.topk_ids):
            return _run_prefill_through_landing(moe, lru, dispatch_output, apply)
        groups = _prefill_pass_groups(lru, topk_output.topk_ids)
    if groups is None:
        dispatch_output = remap_dispatch_for_expert_spill(moe, dispatch_output)
        return apply(layer=moe, dispatch_output=dispatch_output)
    moe._dsv41_host_gemv_pending = False  # type: ignore[attr-defined]
    moe._dsv41_land_ids = None  # type: ignore[attr-defined]
    logical = topk_output.topk_ids.clone()
    combined = None
    for index, group in enumerate(groups):
        ids = _map_pass(lru, logical, group, include_shared=index == 0)
        out = apply(
            layer=moe,
            dispatch_output=dispatch_output._replace(
                topk_output=topk_output._replace(topk_ids=ids)
            ),
        )
        if combined is None:
            combined = out
        else:
            combined = combined._replace(
                hidden_states=combined.hidden_states + out.hidden_states
            )
    return combined


def maybe_spill_model_routed_experts(model: nn.Module) -> Optional[RoutedExpertSpillPlan]:
    """Attach a spill plan after Marlin pack, then pack host rows and build the LRU.

    Must run *after* ``process_weights_after_loading``. ``post_load_weights``
    is too early: host would be packed while GPU ``w13`` is still checkpoint
    layout. Plan-only mode does not pin host copies. APPLY requires
    ``remap_dispatch_for_expert_spill`` before Marlin indexes ``topk_ids``.
    """
    spill_gib = float(envs.SGLANG_DSV41_EXPERT_SPILL_GB.get() or 0.0)
    if spill_gib <= 0:
        return None
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    moes = [m for m in model.modules() if isinstance(m, FusedMoE)]
    if not moes:
        logger.warning("SGLANG_DSV41_EXPERT_SPILL_GB=%s but no FusedMoE found", spill_gib)
        return None
    moe0 = moes[0]
    existing = getattr(moe0, "_dsv41_expert_spill_plan", None)
    local_routed = int(getattr(moe0, "_num_local_routed", moe0.num_local_experts))
    n_shared = int(getattr(moe0, "num_fused_shared_experts", 0))
    if existing is not None:
        plan = existing
    else:
        bytes_per = 0
        gpu_n = int(getattr(moe0, "_dsv41_gpu_expert_slots", moe0.num_local_experts))
        for _name, p in moe0.named_parameters():
            if p.ndim >= 1 and int(p.shape[0]) in (moe0.num_local_experts, gpu_n):
                bytes_per += p.numel() // int(p.shape[0]) * p.element_size()
        if bytes_per == 0:
            bytes_per = mxfp4_expert_bytes() + expert_scale_bytes()
        plan = plan_routed_expert_spill(
            spill_gib=spill_gib,
            local_routed=local_routed,
            bytes_per_expert=bytes_per,
            n_shared=n_shared,
            n_layers=max(len(moes), 1),
        )
    apply = bool(envs.SGLANG_DSV41_EXPERT_SPILL_APPLY.get())
    logger.info(
        "DSV4.1 routed-expert spill: %.1f GiB/rank plan (%d spilled of %d "
        "local routed, shared=%d stay GPU). apply=%s. %s",
        plan.spill_gib,
        plan.n_spilled,
        plan.local_routed,
        plan.n_shared,
        apply,
        "Marlin remaps topk_ids through RoutedExpertLru after ensure()."
        if apply
        else "Plan only (SGLANG_DSV41_EXPERT_SPILL_APPLY=0); GPU tensors unchanged.",
    )
    pin_mirror = bool(envs.SGLANG_ENABLE_DSV41_EXPERT_SPILL_NUMA.get())
    for layer_ordinal, moe in enumerate(moes):
        moe._dsv41_expert_spill_plan = plan  # type: ignore[attr-defined]
        if not (apply and plan.n_spilled):
            continue
        if getattr(moe, "_dsv41_spill_host", None) is not None:
            with _gc_scans_only_new_objects():
                repack_spill_host_for_sm70_marlin(moe)
                if pin_mirror:
                    pin_spill_host_numa(moe, layer_ordinal)
        else:
            logger.warning(
                "DSV4.1 spill APPLY: no _dsv41_spill_host on %s; "
                "LRU will snapshot GPU rows after Marlin pack",
                type(moe).__name__,
            )
        params = _expert_dim_params(moe)
        w13 = params.get("w13_weight")
        if w13 is None:
            continue
        kept_ids, cold_ids = spill_placement(moe)
        pre_hosts = getattr(moe, "_dsv41_spill_host", None)
        if pre_hosts:
            sib_attrs = [
                attr
                for attr in params
                if attr != "w13_weight" and attr in pre_hosts
            ]
            siblings = [params[attr].data for attr in sib_attrs]
            host_w13 = pre_hosts.get("w13_weight")
            host_sibs = [pre_hosts[attr] for attr in sib_attrs]
            lru = RoutedExpertLru(
                w13.data,
                n_shared=int(getattr(moe, "num_fused_shared_experts", 0)),
                n_spilled=plan.n_spilled,
                # Pinned staging for the write-back leg once the mirror is
                # pinned; pageable staging would re-introduce a sync copy.
                pin_memory=bool(getattr(moe, "_dsv41_spill_host_pinned", False)),
                siblings=siblings,
                hosts=[host_w13, *host_sibs] if host_w13 is not None else None,
                logical_n_experts=int(moe.num_local_experts),
                already_shrunk=True,
                kept_ids=kept_ids,
                cold_ids=cold_ids,
            )
            moe._dsv41_expert_lru = lru  # type: ignore[attr-defined]
            if w13.is_cuda:
                lru._ensure_device_tables(
                    torch.empty(1, dtype=torch.int32, device=w13.device)
                )
            _attach_landing_pool(moe, spill_landing_slots())
            continue
        siblings = [p.data for attr, p in params.items() if attr != "w13_weight"]
        lru = RoutedExpertLru(
            w13.data,
            n_shared=int(getattr(moe, "num_fused_shared_experts", 0)),
            n_spilled=plan.n_spilled,
            pin_memory=True,
            siblings=siblings,
            kept_ids=kept_ids,
            cold_ids=cold_ids,
        )
        lru.apply_shrink()
        shrunk = lru.shrunk_tensors()
        w13.data = shrunk[0]
        i = 1
        for attr, p in params.items():
            if attr == "w13_weight":
                continue
            p.data = shrunk[i]
            i += 1
        moe._dsv41_expert_lru = lru  # type: ignore[attr-defined]
        if w13.is_cuda:
            lru._ensure_device_tables(
                torch.empty(1, dtype=torch.int32, device=w13.device)
            )
        _attach_landing_pool(moe, spill_landing_slots())
    if apply and plan.n_spilled:
        n_table = sum(
            1 for m in moes if getattr(m, "_dsv41_spill_cold_source", None) == "table"
        )
        logger.info(
            "DSV4.1 spill placement: cold set from table for %d/%d layers, tail for %d "
            "(SGLANG_DSV41_EXPERT_SPILL_COLD_SET=%s)",
            n_table,
            len(moes),
            len(moes) - n_table,
            envs.SGLANG_DSV41_EXPERT_SPILL_COLD_SET.get(),
        )
        n_h = 0
        for moe in moes:
            if _attach_host_gemv_bases(moe):
                n_h += 1
        if n_h:
            from sglang.kernels.ops.moe.sm70_dsv41_spill_host_gemv import (
                host_gemv_start,
                sm70_dsv41_host_gemv_available,
                sm70_dsv41_host_gemv_enabled,
            )

            if not sm70_dsv41_host_gemv_enabled():
                logger.info(
                    "DSV4.1 D4-H host GEMV off (SGLANG_DSV41_HOST_GEMV=0); "
                    "decode uses D4-G landing"
                )
            elif sm70_dsv41_host_gemv_available() and host_gemv_start():
                logger.info(
                    "DSV4.1 D4-H host GEMV: %d/%d layers, %d CPU threads "
                    "(SGLANG_DSV41_HOST_GEMV=0 restores D4-G landing)",
                    n_h,
                    len(moes),
                    int(envs.SGLANG_DSV41_HOST_GEMV_THREADS.get() or 4),
                )
            else:
                logger.warning(
                    "DSV4.1 D4-H host GEMV unavailable; decode stays on D4-G landing"
                )
    return plan
