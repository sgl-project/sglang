"""Kimi-K3 tensor-parallel KDA under attention DP (``--enable-kda-tp``).

Under attention DP with attention TP 1, every rank runs every KDA layer on its own
requests with all heads: in decode each rank streams the layer's full projection weights
every step (0.88 GB per layer, 61 GB per step over K3's 69 KDA layers). KDA TP keeps MLA
attention-DP and the MoE expert-parallel, and runs only the KDA layers tensor-parallel
over the whole TP group (W ranks):

    owner rows [r, H] --all-gather (rows padded to the DP max R)--> [W*R, H]
    in-proj / conv / KDA core / gated norm on this rank's heads/W for every DP rank's rows
    o_proj row-parallel partial [W*R, H] --reduce-scatter--> owner rows [r, H]

KDA state is head-sharded: every rank's state pool holds heads/W of every DP rank's
requests. Slot ids are global: DP rank j's allocator hands out [j*S + 1, (j+1)*S] of a
pool of W*S (+1 padding) slots (``slot_partition``), so a slot id names the same request on
every rank and every per-request index (forward metadata, speculative commit, deferred
copy-on-write / clear) only has to be gathered, never translated.

Per forward (``begin_forward``, called at the model forward start on every rank, idle ones
included) the KDA linear backend's local metadata becomes the metadata of the gathered
rows:
- fast path, every DP rank in decode or every one in target verify (always under decode
  CUDA graphs): the per-rank slot and track indices are padded to the DP max batch and
  all-gathered into persistent buffers (graph-safe), query_start_loc is the uniform global
  stride, and one kernel launch per layer covers all ranks;
- general path (eager, some rank in extend): each rank's whole metadata is all-gathered and
  every KDA layer runs the core once per source DP rank on that rank's row block.

On a PD decode worker each rank receives the prefill's full-width state of its own
requests into full-width staging slots (``PdStateStaging``) and every rank imports its
heads of every DP rank's staged states with one all-to-all before the forward that first
reads them.

The row collectives use K3's GB300 push collectives over the TP group when they apply
(k3_sp_collective.tp_group_state), else NCCL.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    _note_dp_gather_in_prefill_graph,
    get_dp_global_num_tokens,
)
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)


def _is_kimi_k3() -> bool:
    from sglang.srt.configs.model_config import is_kimi_k3
    from sglang.srt.runtime_context import process_model_config

    return is_kimi_k3(process_model_config().hf_config)


def _applies() -> bool:
    """Whether this server runs KDA TP (layout validated by handle_kda_tp). Safe to call
    before the model is built."""
    return bool(get_parallel().enable_kda_tp) and _is_kimi_k3()


def enabled() -> bool:
    """KDA TP for this model's KDA layers; raises on features it does not support."""
    if not _applies():
        return False
    _check_supported()
    return True


def _check_supported() -> None:
    """State operations that one DP scheduler issues on its own pool outside a forward
    have no cross-rank mirror yet, and every collective here must run on every DP rank."""
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        check_cuda_graph_backend,
    )
    from sglang.srt.runtime_context import get_disagg, get_exec, get_memory

    mamba = get_exec().mamba
    mem = get_memory()
    disagg = get_disagg()
    problems = []
    if mamba.enable_linear_replayssm_spec or mamba.enable_linear_replayssm:
        problems.append("ReplaySSM rings (per-slot cursors)")
    if mem.enable_hierarchical_cache:
        problems.append("HiCache (host copies of rank-local head shards)")
    if mem.enable_unified_memory:
        problems.append("--enable-unified-memory (state slots are global ids)")
    if disagg.disaggregation_mode != "decode":
        problems.append(
            f"disaggregation mode {disagg.disaggregation_mode!r}: KDA TP serves the PD "
            "decode role only (a PD prefill would send head shards)"
        )
    if (
        disagg.disaggregation_mode == "decode"
        and envs.SGLANG_K3_KDA_TP_LATE_STAGE.get()
        and disagg.disaggregation_transfer_backend != "nixl"
    ):
        problems.append(
            "SGLANG_K3_KDA_TP_LATE_STAGE with transfer backend "
            f"{disagg.disaggregation_transfer_backend} (the stage negotiation is "
            "implemented for nixl)"
        )
    if (
        disagg.disaggregation_mode == "decode"
        and disagg.disaggregation_decode_enable_radix_cache
    ):
        problems.append(
            "the PD decode radix cache (rank-local state copies on prefix match)"
        )
    if not mem.disable_radix_cache and not mamba.enable_mamba_extra_buffer:
        problems.append(
            "the no_buffer mamba radix strategy (eager slot copies on insert)"
        )
    if check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE):
        problems.append(
            "breakable prefill CUDA graphs (replay skips the per-forward KDA TP "
            "metadata exchange)"
        )
    from sglang.srt.runtime_context import get_spec

    algo = get_spec().speculative_algorithm
    if algo is not None and str(algo).upper().split(".")[-1] not in (
        "EAGLE",
        "EAGLE3",
        "NEXTN",
    ):
        problems.append(
            f"speculative algorithm {algo} (its verify state commit is not the "
            "collective one)"
        )
    if envs.SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH.get():
        problems.append(
            "SGLANG_ENABLE_EAGLE_POST_VERIFY_CUDA_GRAPH (the state commit is a "
            "collective that idle DP ranks must join)"
        )
    if problems:
        raise NotImplementedError(
            "--enable-kda-tp does not support: " + "; ".join(problems) + "."
        )


def state_shard_width() -> int:
    """W when KDA state is head-sharded over the TP group (KDA TP), else 1. Safe to call
    before the model is built."""
    return get_parallel().tp_size if _applies() else 1


def width() -> int:
    return get_parallel().tp_size


def group():
    return get_parallel().tp_group


def slot_partition(pool_slots: int) -> Tuple[int, int]:
    """(first slot, slot count) this DP rank allocates from a head-sharded pool of
    ``pool_slots`` slots (slot 0, the padding slot, excluded)."""
    W = width()
    per_rank = pool_slots // W
    if per_rank < 1:
        raise ValueError(
            f"KDA TP: {pool_slots} state slots cannot be split over {W} DP ranks"
        )
    return get_parallel().attn_dp_rank * per_rank + 1, per_rank


# ---- rows -----------------------------------------------------------------------------------


def padded_rows(local_rows: int, counts: Optional[List[int]] = None) -> int:
    """``counts``: every DP rank's rows this forward; default the DP context's."""
    if counts is None:
        counts = get_dp_global_num_tokens()
    if not counts:
        raise RuntimeError("KDA TP needs the DP global token counts of this forward")
    rows = max(counts)
    if local_rows > rows:
        raise RuntimeError(
            f"KDA TP: {local_rows} local rows exceed the DP maximum {rows}"
        )
    return rows


def _fast_collectives():
    """K3's tuned MNNVL push / pull / multicast kernels over the TP group, or None (NCCL)."""
    from sglang.srt.layers.communication import k3_sp_collective

    return k3_sp_collective.tp_group_state()


def all_gather_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """[local, D] -> [W * rows, D]; DP rank j's rows start at j * rows. May return a
    persistent symmetric buffer that the next KDA TP all-gather overwrites."""
    from sglang.srt.layers.communication import k3_sp_collective

    # A prefill graph that holds this exchange (captured or in an eager break) only replays
    # when every DP rank runs the bucket whose row count it recorded.
    _note_dp_gather_in_prefill_graph()
    g = group()
    if x.shape[0] != rows:
        x = torch.nn.functional.pad(x, (0, 0, 0, rows - x.shape[0]))
    x = x.contiguous()
    state = _fast_collectives()
    if state is not None and rows > 0:
        out = k3_sp_collective.all_gather(
            x, state=state, buffer_name="kda_tp_all_gather"
        )
        if out is not None:
            return out
    out = x.new_empty((g.world_size * rows, *x.shape[1:]))
    g.all_gather_into_tensor(out, x)
    return out


def o_proj_output(
    total_rows: int, width: int, dtype: torch.dtype
) -> Optional[torch.Tensor]:
    """Symmetric storage for the row-parallel o_proj partial when the reduce-scatter reads
    it through its multicast alias (pull strategy), else None."""
    from sglang.srt.layers.communication import k3_sp_collective

    state = _fast_collectives()
    if state is None or total_rows == 0:
        return None
    if not k3_sp_collective.requires_symmetric_rs(
        total_rows,
        torch.device("cuda"),
        torch.empty((), dtype=dtype).element_size(),
        state=state,
        fused=False,
    ):
        return None
    return k3_sp_collective.get_o_proj_output_buffer(
        total_rows, dtype, width, state=state
    )


def reduce_scatter_rows(y: torch.Tensor, local_rows: int) -> torch.Tensor:
    """[W * rows, D] partial sums -> this rank's [local_rows, D] full sums."""
    from sglang.srt.layers.communication import k3_sp_collective

    g = group()
    rows = y.shape[0] // g.world_size
    state = _fast_collectives()
    if state is not None and rows > 0:
        out = k3_sp_collective.reduce_scatter_res(y, None, state=state)
        if out is not None:
            return out[:local_rows]
    out = y.new_empty((rows, *y.shape[1:]))
    g.reduce_scatter_tensor(out, y.contiguous())
    return out[:local_rows]


def all_gather_padded(
    t: torch.Tensor,
    n: int,
    pad_value: int,
    name: Optional[str] = None,
    capacity: int = 0,
) -> torch.Tensor:
    """1-D [k <= n] per rank -> [W * n], each rank's block padded with ``pad_value``. With
    ``name``, the collective's input and output are views of persistent buffers (see
    _static), so a captured graph keeps reading and writing the addresses it recorded."""
    g = group()
    if name is None:
        if t.shape[0] < n:
            t = torch.cat([t, t.new_full((n - t.shape[0],), pad_value)])
        out = t.new_empty((g.world_size * n,))
        g.all_gather_into_tensor(out, t.contiguous())
        return out
    src = _static(f"{name}.in", capacity, t.dtype)[:n]
    src[t.shape[0] :].fill_(pad_value)
    src[: t.shape[0]].copy_(t)
    out = _static(f"{name}.out", g.world_size * capacity, t.dtype)[: g.world_size * n]
    g.all_gather_into_tensor(out, src)
    return out


# Persistent per-process buffers for the fast path's gathered metadata, never freed or
# rebound: CUDA-graph captures record their addresses. Sized to the per-rank request
# capacity, so they never grow after the first forward.
_STATIC: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}


def _static(name: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
    buf = _STATIC.get((name, dtype))
    if buf is not None and buf.numel() >= numel:
        return buf
    if buf is not None and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"KDA TP: static buffer {name} must grow to {numel} during a CUDA graph capture"
        )
    buf = torch.empty(numel, dtype=dtype, device=torch.cuda.current_device())
    _STATIC[(name, dtype)] = buf
    return buf


# ---- per-forward context ----------------------------------------------------------------------


@dataclasses.dataclass
class Source:
    """One DP rank's block of the gathered rows (general path)."""

    rank: int
    rows: int  # real rows in the block, at [rank * R, rank * R + rows)
    metadata: Any  # that rank's ForwardMetadata (slot ids already global)
    batch: SimpleNamespace  # the ForwardBatch fields the KDA backend reads


@dataclasses.dataclass
class Context:
    local_rows: int
    R: int  # rows per DP rank block in the gathered tensor
    mode: Any  # ForwardMode shared by every rank (fast path) or None (general path)
    batch: Any  # the forward batch the KDA backend sees on the fast path
    sources: Optional[List[Source]] = None  # general path


_CTX: Optional[Context] = None
# (gathered metadata the fast path installed on the linear backend, the per-rank metadata
# it was built from).
_GATHERED: Optional[Tuple[Any, Any]] = None


def current() -> Context:
    if _CTX is None:
        raise RuntimeError("KDA TP: no forward context (begin_forward was not called)")
    return _CTX


def source_tokens(src: Source) -> int:
    """The rows of ``src``'s block the KDA kernels consume: its real tokens. A breakable
    prefill graph pads every rank's block to the replayed size."""
    b = src.batch
    if b.forward_mode.is_target_verify():
        return b.batch_size * b.spec_info.draft_token_num
    if b.extend_seq_lens_cpu is not None:
        return sum(b.extend_seq_lens_cpu)
    return src.rows


def _mode_kind(mode) -> str:
    if mode.is_idle():
        return "idle"
    if mode.is_target_verify():
        return "verify"
    if mode.is_decode():
        return "decode"
    if mode.is_extend():
        return "extend"
    raise NotImplementedError(f"KDA TP: forward mode {mode} is not supported")


def begin_forward(
    forward_batch, linear_backend, local_rows: int, counts: Optional[List[int]] = None
) -> None:
    """Build this forward's KDA TP context on every rank (a collective: idle DP ranks call it
    too). On the fast path the linear backend's forward_metadata becomes the gathered rows'
    metadata; ``local_rows`` is this rank's row count entering the layers, ``counts`` every
    rank's (default: the DP context's, which breakable-graph replay does not refresh)."""
    global _CTX
    R = padded_rows(local_rows, counts)
    kinds = _global_kinds(forward_batch)
    active = {k for k in kinds if k != "idle"}
    if len(active) <= 1 and active <= {"decode", "verify"}:
        _CTX = _fast_context(forward_batch, linear_backend, local_rows, R, active)
    else:
        _CTX = _general_context(forward_batch, linear_backend, local_rows, R)


def _speculative() -> bool:
    from sglang.srt.runtime_context import get_spec

    return bool(get_spec().speculative_algorithm)


def _global_kinds(forward_batch) -> List[str]:
    """Every DP rank's forward kind: from the MLP-sync gather the scheduler already did
    (ForwardBatch.dp_forward_modes); inside a CUDA graph capture every rank captures the
    same graph, so the local kind is everyone's; otherwise gathered over the CPU group."""
    from sglang.srt.model_executor.forward_mode import ForwardMode

    # The rows' own mode: a decode batch relabeled as a 1-token extend for the MLP sync is decode.
    kind = _mode_kind(forward_batch.logical_forward_mode)
    if torch.cuda.is_current_stream_capturing():
        return [kind] * width()
    modes = getattr(forward_batch, "dp_forward_modes", None)
    if modes is not None and len(modes) == width():
        kinds = [
            "idle" if m == ForwardMode.PREBUILT.value else _mode_kind(ForwardMode(m))
            for m in modes
        ]
        if _speculative():
            # A speculative decode step is DECODE in the scheduler and TARGET_VERIFY in
            # the target forward, on every DP rank that has requests (idle ranks too
            # must agree on the verify row layout).
            if kind == "decode":
                raise RuntimeError(
                    "KDA TP: plain decode target forward under speculative decoding"
                )
            kinds = ["verify" if k == "decode" else k for k in kinds]
        return kinds
    out: List[Optional[str]] = [None] * width()
    torch.distributed.all_gather_object(out, kind, group=group().cpu_group)
    return out


def _fast_context(forward_batch, backend, local_rows, R, active) -> Context:
    """Every non-idle DP rank is in decode, or every one in target verify: one launch per
    layer over the gathered rows, with the per-rank slot / track indices padded to the DP max
    batch and all-gathered (graph-safe; idle ranks contribute padding)."""
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.runtime_context import get_spec

    global _GATHERED
    fm = backend.forward_metadata
    if fm is None:
        raise RuntimeError(
            "KDA TP: the linear attention backend has no forward metadata"
        )
    if _GATHERED is not None and fm is _GATHERED[0]:
        # A forward re-run on unchanged metadata (CUDA-graph capture right after its warmup
        # forward) must gather the backend's own per-rank metadata, not the last gather.
        fm = _GATHERED[1]
    W = width()
    kind = next(iter(active)) if active else "decode"
    if kind == "verify":
        D = get_spec().speculative_num_draft_tokens
        mode = ForwardMode.TARGET_VERIFY
        if forward_batch.ragged_verify_layout is not None:
            raise NotImplementedError("KDA TP: ragged verify layouts are not supported")
        spec = SimpleNamespace(draft_token_num=D, ragged_verify_layout=None)
    else:
        D, mode, spec = 1, ForwardMode.DECODE, None
    if R % D:
        raise RuntimeError(
            f"KDA TP: {R} rows per rank is not a multiple of {D} verify tokens"
        )
    bs = R // D
    dev = torch.device("cuda", torch.cuda.current_device())
    if forward_batch.forward_mode.is_idle():
        # An idle rank's backend metadata may be a previous batch's: contribute none.
        local_slots = torch.zeros(0, dtype=torch.int32, device=dev)
        local_track = None
    else:
        # Graph metadata buffers are sized to the captured max batch: take this forward's
        # requests (local rows / D).
        local_bs = local_rows // D
        local_slots = fm.mamba_cache_indices[:local_bs]
        local_track = (
            fm.mamba_track_indices[:local_bs]
            if fm.mamba_track_indices is not None
            else None
        )
    if local_slots.shape[0] > bs:
        raise RuntimeError(
            f"KDA TP: local batch {local_slots.shape[0]} exceeds the DP max batch {bs}"
        )
    # Padding requests point at the reserved padding slot 0. With the mamba extra buffer
    # (a server-wide setting) every rank gathers track indices, zeros where it has none,
    # so the collectives and the tracking writes agree on every rank.
    from sglang.srt.runtime_context import get_exec

    # Every tensor the layers read is a view of a persistent buffer (_static): a captured
    # graph reads them through recorded addresses, so they must outlive every capture.
    cap = _rank_capacity(backend)
    if bs > cap:
        raise RuntimeError(
            f"KDA TP: DP max batch {bs} exceeds the per-rank capacity {cap}"
        )
    # int64 on the wire: every rank must gather one dtype.
    slots = _static("slots", W * cap, fm.mamba_cache_indices.dtype)[: W * bs]
    slots.copy_(
        all_gather_padded(
            local_slots.to(torch.int64), bs, 0, name="slots", capacity=cap
        )
    )
    track = None
    if get_exec().mamba.enable_mamba_extra_buffer:
        track = all_gather_padded(
            (local_track if local_track is not None else local_slots.new_zeros(0)).to(
                torch.int64
            ),
            bs,
            0,
            name="track",
            capacity=cap,
        )
        if fm.mamba_track_indices is not None:
            gathered = track
            track = _static("track", W * cap, fm.mamba_track_indices.dtype)[: W * bs]
            track.copy_(gathered)
    track_mask = None
    if kind == "decode" and get_exec().mamba.enable_mamba_extra_buffer:
        # Plain decode tracks in-layer (_track_mamba_state_decode) by this mask.
        local_mask = forward_batch.mamba_track_mask
        if local_mask is None or forward_batch.forward_mode.is_idle():
            local_mask = torch.zeros(0, dtype=torch.int64, device=dev)
        local_mask = local_mask[: local_slots.shape[0]]
        track_mask = _static("track_mask", W * cap, torch.bool)[: W * bs]
        track_mask.copy_(
            all_gather_padded(
                local_mask.to(torch.int64), bs, 0, name="track_mask", capacity=cap
            )
        )
    qsl = _static("qsl", W * cap + 1, torch.int32)[: W * bs + 1]
    qsl.copy_(
        torch.arange(0, W * bs * D + 1, D, dtype=torch.int32, device=slots.device)
    )
    if kind == "verify":
        # Request i of DP rank j uses verify scratch row j * n + i (n: per-rank request
        # capacity), the row commit_after_verify reads back.
        pos = torch.arange(W * bs, dtype=torch.int64, device=slots.device)
        rows = _static("verify_rows", W * cap, torch.int32)[: W * bs]
        rows.copy_(_scratch_rows(backend, (pos // bs) * cap + pos % bs))
        backend.verify_intermediate_state_indices = rows
    gathered = dataclasses.replace(
        fm,
        query_start_loc=qsl,
        mamba_cache_indices=slots,
        mamba_track_indices=track,
        logical_num_tokens=None,
        # Rebuilt from the gathered slots by the fused-accept path.
        fused_accept_state_indices=None,
        fused_accept_num_accepted=None,
    )
    backend.forward_metadata = gathered
    _GATHERED = (gathered, fm)
    batch = SimpleNamespace(
        forward_mode=mode,
        logical_forward_mode=mode,
        batch_size=W * bs,
        spec_info=spec,
        # The KDA verify path reads the layout from the forward batch (None: uniform draft_token_num rows).
        ragged_verify_layout=None,
        req_pool_indices=None,
        input_ids=None,
        mamba_track_mask=track_mask,
    )
    return Context(local_rows=local_rows, R=R, mode=mode, batch=batch)


# Fields of a local ForwardBatch that the KDA backend reads on its extend / verify paths.
def _batch_fields(forward_batch) -> Dict[str, Any]:
    spec = forward_batch.spec_info
    return dict(
        forward_mode=int(forward_batch.logical_forward_mode.value),
        batch_size=forward_batch.batch_size,
        extend_seq_lens_cpu=list(forward_batch.extend_seq_lens_cpu)
        if forward_batch.extend_seq_lens_cpu is not None
        else None,
        draft_token_num=getattr(spec, "draft_token_num", None)
        if spec is not None
        else None,
    )


def _general_context(forward_batch, backend, local_rows, R) -> Context:
    """Eager: gather every DP rank's whole KDA metadata (tensor fields through one padded
    int64 all-gather, the rest over the CPU group)."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("KDA TP: mixed-mode forwards cannot run inside a CUDA graph")
    from sglang.srt.model_executor.forward_mode import ForwardMode

    fm = backend.forward_metadata if not forward_batch.forward_mode.is_idle() else None
    if _GATHERED is not None and fm is _GATHERED[0]:
        fm = _GATHERED[1]
    tensors: Dict[str, torch.Tensor] = {}
    scalars: Dict[str, Any] = {}
    if fm is not None:
        for f in dataclasses.fields(fm):
            v = getattr(fm, f.name)
            if isinstance(v, torch.Tensor):
                tensors[f.name] = v
            else:
                scalars[f.name] = v
        if forward_batch.extend_prefix_lens is not None:
            tensors["__extend_prefix_lens"] = forward_batch.extend_prefix_lens
        track_mask = getattr(forward_batch, "mamba_track_mask", None)
        if forward_batch.logical_forward_mode.is_decode() and track_mask is not None:
            # Plain decode snapshots prefix-cache states in-layer by this mask.
            tensors["__mamba_track_mask"] = track_mask[: forward_batch.batch_size]
    desc = {
        "cls": type(fm) if fm is not None else None,
        "tensors": {k: (v.dtype, tuple(v.shape)) for k, v in tensors.items()},
        "scalars": scalars,
        "batch": _batch_fields(forward_batch),
        "rows": local_rows,
    }
    W = width()
    descs: List[Optional[dict]] = [None] * W
    torch.distributed.all_gather_object(descs, desc, group=group().cpu_group)
    dev = torch.device("cuda", torch.cuda.current_device())
    flat = (
        torch.cat([v.reshape(-1).to(torch.int64) for v in tensors.values()])
        if tensors
        else torch.zeros(0, dtype=torch.int64, device=dev)
    )
    sizes = [
        sum(int(torch.Size(shape).numel()) for _, shape in d["tensors"].values())
        for d in descs
    ]
    n = max(max(sizes), 1)
    gathered = all_gather_padded(flat.to(dev), n, 0).view(W, n)
    sources: List[Source] = []
    for j, d in enumerate(descs):
        if d["cls"] is None or d["rows"] == 0:
            continue
        off, vals = 0, {}
        for name, (dtype, shape) in d["tensors"].items():
            k = int(torch.Size(shape).numel())
            vals[name] = gathered[j, off : off + k].view(shape).to(dtype)
            off += k
        prefix = vals.pop("__extend_prefix_lens", None)
        track_mask = vals.pop("__mamba_track_mask", None)
        meta = d["cls"](**vals, **d["scalars"])
        b = d["batch"]
        mode = ForwardMode(b["forward_mode"])
        batch = SimpleNamespace(
            forward_mode=mode,
            logical_forward_mode=mode,
            batch_size=b["batch_size"],
            extend_seq_lens_cpu=b["extend_seq_lens_cpu"],
            extend_prefix_lens=prefix,
            spec_info=SimpleNamespace(
                draft_token_num=b["draft_token_num"], ragged_verify_layout=None
            )
            if b["draft_token_num"] is not None
            else None,
            ragged_verify_layout=None,
            req_pool_indices=None,
            mamba_track_indices=vals.get("mamba_track_indices"),
            mamba_track_mask=track_mask,
        )
        sources.append(Source(rank=j, rows=d["rows"], metadata=meta, batch=batch))
    return Context(local_rows=local_rows, R=R, mode=None, batch=None, sources=sources)


def _rank_capacity(backend) -> int:
    """Per-DP-rank request capacity n: DP rank j owns verify scratch rows [j*n, (j+1)*n)."""
    return backend.req_to_token_pool.size


def source_scratch_rows(backend, src: Source) -> torch.Tensor:
    """Verify scratch rows of a general-path source's requests: src.rank * n + i."""
    dev = torch.device("cuda", torch.cuda.current_device())
    rows = src.rank * _rank_capacity(backend) + torch.arange(
        src.batch.batch_size, dtype=torch.int64, device=dev
    )
    return _scratch_rows(backend, rows)


def _scratch_rows(backend, rows: torch.Tensor) -> torch.Tensor:
    """int32 verify scratch rows, out-of-range ones on the scratch's padding row (W * n)."""
    pad = width() * _rank_capacity(backend)
    return torch.where(rows < pad, rows, torch.full_like(rows, pad)).to(torch.int32)


def commit_after_verify(
    model_runner,
    batch,
    last_correct_step_indices,
    mamba_steps_to_track,
    state_slots=None,
):
    """Commit every DP rank's accepted verify states into this rank's head shard (a
    collective: idle DP ranks call it with no requests). Each rank packs its requests'
    (slot, last accepted step, track slot, track step, scratch row), padded to the per-rank
    request capacity, and all-gathers them; the scatter then writes all ranks' requests.
    ``state_slots``: the verify forward's resolved local slots (default: the request
    table's current slots)."""
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        scatter_mamba_states_after_mtp_verify,
    )
    from sglang.srt.mem_cache.kv.decl import STATE_PART

    req_pool = model_runner.req_to_token_pool
    W = width()
    n = req_pool.size  # per-DP-rank request capacity
    dev = torch.device("cuda", torch.cuda.current_device())
    bs = 0 if last_correct_step_indices is None else last_correct_step_indices.shape[0]
    if bs > n:
        raise RuntimeError(
            f"KDA TP commit: {bs} requests exceed the per-rank capacity {n}"
        )
    pack = torch.zeros((5, n), dtype=torch.int64, device=dev)
    # Padding rows commit nothing (step -1); a step 0 would copy into slot 0 every step.
    pack[1].fill_(-1)
    pack[3].fill_(-1)  # no track step
    if bs:
        if state_slots is not None:
            slots = state_slots.working[:bs]
            track = state_slots.track
        else:
            slots = req_pool.get_mamba_indices(batch.req_pool_indices[:bs])
            track = batch.mamba_track_indices
        pack[0, :bs] = slots.to(torch.int64)
        pack[1, :bs] = last_correct_step_indices.to(torch.int64)
        if (
            batch.mamba_track_indices is not None
            and track is not None
            and mamba_steps_to_track is not None
        ):
            pack[2, :bs] = track[:bs].to(torch.int64)
            pack[3, :bs] = mamba_steps_to_track.to(torch.int64)
    rank = get_parallel().attn_dp_rank
    pack[4] = rank * n + torch.arange(n, device=dev)  # see _fast_context's scratch rows
    mamba_pool = model_runner.token_to_kv_pool.layer_router.part_view(STATE_PART)
    caches = mamba_pool.get_speculative_mamba2_params_all_layers()
    out = pack.new_empty((W, 5, n))
    group().all_gather_into_tensor(out.view(W * 5, n), pack)
    slots, last, track, steps, rows = out.permute(1, 0, 2).reshape(5, W * n)
    scatter_mamba_states_after_mtp_verify(
        caches,
        slots,
        last,
        track,
        steps,
        src_indices_raw=rows,
    )


def mirror_deferred_state_ops(model_runner, forward_batch) -> None:
    """Run every DP rank's deferred state clears and copy-on-writes (prefix matches of
    extend forwards) on this rank's head shard. Called on every rank at the start of every
    target forward; a no-op unless some DP rank is in extend."""
    modes = getattr(forward_batch, "dp_forward_modes", None)
    if modes is not None:
        from sglang.srt.model_executor.forward_mode import ForwardMode

        if not any(
            ForwardMode(m).is_extend() and not ForwardMode(m).is_target_verify()
            for m in modes
        ):
            return
    state = model_runner.token_to_kv_pool_allocator.state
    if state.ckpt_pool is not None:
        raise NotImplementedError("KDA TP: int8 mamba checkpoints are not supported")

    def ids(t):
        return [] if t is None else [int(x) for x in t.tolist()]

    local = (
        ids(forward_batch.mamba_clear_indices),
        ids(forward_batch.mamba_cow_src_indices),
        ids(forward_batch.mamba_cow_dst_indices),
    )
    forward_batch.mamba_clear_indices = None
    forward_batch.mamba_cow_src_indices = None
    forward_batch.mamba_cow_dst_indices = None
    out: List[Optional[tuple]] = [None] * width()
    torch.distributed.all_gather_object(out, local, group=group().cpu_group)
    clear = [i for o in out for i in o[0]]
    src = [i for o in out for i in o[1]]
    dst = [i for o in out for i in o[2]]
    dev = state.pool.device
    # Global slot ids index physical storage directly (no compaction under KDA TP).
    if clear:
        state.pool.clear_slots(torch.tensor(clear, dtype=torch.int64, device=dev))
    if src:
        from sglang.srt.mem_cache.kv.decl import STATE_PART

        # The copy reads every state layer of the source slot.
        model_runner.token_to_kv_pool.layer_router.wait_for_part(STATE_PART)
        state.copy(
            torch.tensor(src, dtype=torch.int64, device=dev),
            torch.tensor(dst, dtype=torch.int64, device=dev),
        )


def end_forward() -> None:
    global _CTX
    _CTX = None


# ---- PD decode: receiving prefill state into the head shards ----------------------------------


def lazy_mamba_slot(enable_mamba_extra_buffer: bool) -> bool:
    """Whether a PD decode request defers its head-sharded state slot to admission (it
    holds only a staging slot through its handoff). Not with ping-pong tracking slots,
    which are allocated with the main slot."""
    return (
        state_shard_width() > 1
        and not enable_mamba_extra_buffer
        and envs.SGLANG_K3_KDA_TP_LAZY_MAMBA_SLOT.get()
    )


def late_stage() -> bool:
    """Whether a PD decode request binds its staging slot when the prefill is about to
    send the state (KDA_STAGE_REQ with its last chunk) rather than at prealloc, so handoff
    slots can outnumber staging slots (SGLANG_K3_KDA_TP_LATE_STAGE)."""
    return state_shard_width() > 1 and envs.SGLANG_K3_KDA_TP_LATE_STAGE.get()


# State destination a late-staged request advertises until its stage is bound.
LATE_STAGE_UNBOUND = -1


def lazy_stage_key(req_pool_idx: int) -> int:
    """Staging key of a request with no state slot yet: its request row, negated, so it
    never collides with a state slot id (>= 1)."""
    return -int(req_pool_idx)


class FullStateStaging:
    """Full-width (all heads) KDA state slots on this rank, and the all-to-alls over the TP
    group that move such states between their owner DP rank and every rank's head shard of
    the requests' global slots (``scatter``: owner -> shards, ``gather``: shards -> owner)."""

    def __init__(self, pool, cache_params, num_slots: int, device):
        from sglang.srt.configs.mamba_utils import KimiLinearStateShape

        shape = cache_params.shape
        if not isinstance(shape, KimiLinearStateShape):
            raise TypeError(
                f"KDA TP staging needs Kimi linear state, got {type(shape)}"
            )
        if num_slots < 1:
            raise ValueError(f"KDA TP staging needs at least one slot, got {num_slots}")
        full = KimiLinearStateShape.create(
            tp_world_size=1,
            num_heads=shape.num_heads,
            head_dim=shape.head_dim,
            num_k_heads=shape.num_k_heads,
            head_k_dim=shape.head_k_dim,
            conv_kernel_size=shape.conv_kernel,
        )
        W = width()
        if full.temporal[0] % W:
            raise ValueError(
                f"KDA TP: {full.temporal[0]} heads do not split over {W} ranks"
            )
        self.layer_ids = list(pool.mamba_layer_ids)
        L = len(self.layer_ids)
        with pool.state_allocation_region():
            self.conv = [
                torch.zeros(
                    (L, num_slots) + tuple(c),
                    dtype=cache_params.dtype.conv,
                    device=device,
                )
                for c in full.conv
            ]
            self.temporal = torch.zeros(
                (L, num_slots) + tuple(full.temporal),
                dtype=cache_params.dtype.temporal,
                device=device,
            )
        self.conv_slice_axis = full.conv_slice_axis
        # Full conv channels [g0 | g1 | ...] with every group head-sharded W ways, reordered
        # rank-major so rank r's block is its local [g0_r | g1_r | ...] layout.
        groups = full.conv_shard_groups
        cols, off = [[] for _ in range(W)], 0
        for g in groups:
            if g % W:
                raise ValueError(
                    f"KDA TP: conv group {g} does not split over {W} ranks"
                )
            for r in range(W):
                cols[r].extend(range(off + r * g // W, off + (r + 1) * g // W))
            off += g
        self._conv_rank_major = torch.tensor(
            [c for r in cols for c in r], dtype=torch.int64, device=device
        )
        self.num_slots = num_slots

    # Transfer registration, in MambaPool._iter_transfer_state_entries order and format.
    def iter_transfer_state_entries(self):
        for conv in self.conv:
            for i, layer_id in enumerate(self.layer_ids):
                yield "conv", conv[i], self.conv_slice_axis, layer_id
        for i, layer_id in enumerate(self.layer_ids):
            yield "temporal", self.temporal[i], 0, layer_id

    def _rank_major(self, t: torch.Tensor, kind: str) -> torch.Tensor:
        """[Lc, c, *full slot] -> [W, c, Lc, E]: rank r's heads of every (state, layer).
        Sizes are explicit: a rank with nothing to send has c == 0."""
        W = width()
        Lc, c = t.shape[:2]
        per = math.prod(t.shape[2:]) // W
        if kind == "temporal":
            t = t.reshape(Lc, c, W, per)
            return t.permute(2, 1, 0, 3).contiguous()
        # conv: [Lc, c, K-1, D] (channels last) -> [Lc, c, K-1, W, D/W]
        k1, d = t.shape[2], t.shape[3]
        t = t.index_select(3, self._conv_rank_major).reshape(Lc, c, k1, W, d // W)
        return t.permute(3, 1, 0, 2, 4).reshape(W, c, Lc, per).contiguous()

    def _from_rank_major(self, t: torch.Tensor, kind: str, full_shape) -> torch.Tensor:
        """Inverse of ``_rank_major``: [W, c, Lc, E] -> [Lc, c, *full slot]."""
        W, c, Lc = t.shape[:3]
        if kind == "temporal":
            return t.permute(2, 1, 0, 3).reshape(Lc, c, *full_shape)
        k1, d = full_shape
        t = t.view(W, c, Lc, k1, d // W).permute(2, 1, 3, 0, 4).reshape(Lc, c, k1, d)
        return torch.empty_like(t).index_copy_(3, self._conv_rank_major, t)

    def _state_pairs(self, mamba_pool):
        cache = mamba_pool.mamba_cache
        pairs = [(stage, shard, "conv") for stage, shard in zip(self.conv, cache.conv)]
        pairs.append((self.temporal, cache.temporal, "temporal"))
        return pairs

    # Bounds the all-to-all buffers of one chunk of layers.
    chunk_bytes = 256 << 20

    def _layer_chunk(self, L: int, nbytes_per_layer: int) -> int:
        """Layers per all-to-all; every rank must derive it from the same global sizes."""
        return max(1, min(L, self.chunk_bytes // max(1, nbytes_per_layer)))

    def scatter(
        self, mamba_pool, stages: List[int], counts: List[int], dst: torch.Tensor
    ) -> None:
        """Collective over the TP group (every rank, ``counts`` agreed): write rank r's heads
        of every DP rank's staged states (this rank's: ``stages``) into their global slots
        ``dst`` (every rank's, in rank order)."""
        g = group()
        W = g.world_size
        dev = self.temporal.device
        n_max, n_total = max(counts), sum(counts)
        c = len(stages)
        stage_idx = torch.tensor(stages, dtype=torch.int64, device=dev)
        L = len(self.layer_ids)
        for src, dst_t, kind in self._state_pairs(mamba_pool):
            Lc = self._layer_chunk(L, n_max * src[0, 0].nbytes)
            for l0 in range(0, L, Lc):
                l1 = min(L, l0 + Lc)
                send = self._rank_major(src[l0:l1].index_select(1, stage_idx), kind)
                per = send.shape[-1]
                recv = torch.empty((n_total, l1 - l0, per), dtype=src.dtype, device=dev)
                torch.distributed.all_to_all_single(
                    recv.view(-1),
                    send.view(-1),
                    output_split_sizes=[n * (l1 - l0) * per for n in counts],
                    input_split_sizes=[c * (l1 - l0) * per] * W,
                    group=g.device_group,
                )
                slot_shape = dst_t.shape[2:]
                dst_t[l0:l1, dst] = recv.view(n_total, l1 - l0, *slot_shape).transpose(
                    0, 1
                )

    def gather(
        self, mamba_pool, stages: List[int], counts: List[int], src: torch.Tensor
    ) -> None:
        """Collective over the TP group, the inverse of ``scatter``: every rank sends its
        heads of each DP rank's global slots ``src`` (every rank's, in rank order) to that
        rank, which assembles the full states into its staging slots ``stages``."""
        g = group()
        W = g.world_size
        dev = self.temporal.device
        n_total = sum(counts)
        c = len(stages)
        stage_idx = torch.tensor(stages, dtype=torch.int64, device=dev)
        L = len(self.layer_ids)
        for stage_t, shard_t, kind in self._state_pairs(mamba_pool):
            full_shape = stage_t.shape[2:]
            per = shard_t[0, 0].numel()
            Lc = self._layer_chunk(
                L, max(n_total, W * max(counts)) * shard_t[0, 0].nbytes
            )
            for l0 in range(0, L, Lc):
                l1 = min(L, l0 + Lc)
                send = shard_t[l0:l1].index_select(1, src).transpose(0, 1).contiguous()
                recv = torch.empty(
                    (W, c, l1 - l0, per), dtype=shard_t.dtype, device=dev
                )
                torch.distributed.all_to_all_single(
                    recv.view(-1),
                    send.view(-1),
                    output_split_sizes=[c * (l1 - l0) * per] * W,
                    input_split_sizes=[n * (l1 - l0) * per for n in counts],
                    group=g.device_group,
                )
                if c:
                    stage_t[l0:l1].index_copy_(
                        1, stage_idx, self._from_rank_major(recv, kind, full_shape)
                    )


class PdStateStaging(FullStateStaging):
    """KDA TP on a PD decode worker: receive buffers for the prefill's full KDA state.

    The prefill (attention TP 1, KDA weight-sharded or replicated) sends every request's full
    state (all heads) to the one decode DP rank that owns the request, but under KDA TP each rank
    keeps heads/W of every slot. So the decode rank advertises these full-width staging
    slots as its transfer destination (the prefill sees matching layouts and sends as usual),
    and once the request joins the running batch every rank runs ``import_states`` before
    the forward that first reads the slot: an all-to-all hands rank r its heads of every
    staged state, written into the request's global slot.

    Lifecycle of a stage: ``assign`` (prealloc, keyed by the request's main slot) ->
    ``queue_import`` (the request joins the running batch) -> ``take_pending`` (the batch's
    MLP sync advertises the count) -> ``finish_import`` (the forward enqueued the copies; the
    stage is reused once they complete). ``release_slot`` drops a request that leaves first.
    A scheduler that publishes destinations after admitting several requests ``reserve``s a
    stage at admission so ``has_room`` counts it, and passes ``reserved=True`` to ``assign``.
    With SGLANG_K3_KDA_TP_LAZY_MAMBA_SLOT the request has no state slot until it joins the
    running batch, so its stage is keyed by ``lazy_stage_key`` (its request row) and moved
    onto the slot it is given there with ``rekey``.
    """

    def __init__(self, pool, cache_params, num_slots: int, device):
        super().__init__(pool, cache_params, num_slots, device)
        self._free: List[int] = list(range(num_slots))
        self._stage_of: Dict[int, int] = {}  # request main slot -> stage
        self._pending: List[int] = []  # slots whose request joined the running batch
        self._reserved = 0  # admitted requests whose assign is still to come
        self._draining: List[
            Tuple[Any, List[int]]
        ] = []  # (event, stages) being imported
        nbytes = sum(t.nbytes for t in self.conv) + self.temporal.nbytes
        logger.info(
            "KDA TP PD decode: %d staging slots for prefill state (%.2f GB)",
            num_slots,
            nbytes / 1e9,
        )

    def _reclaim(self) -> None:
        still = []
        for event, stages in self._draining:
            if event is None or event.query():
                self._free.extend(stages)
            else:
                still.append((event, stages))
        self._draining = still

    def has_room(self, allocator, req_pool=None) -> bool:
        self._reclaim()
        if len(self._free) <= self._reserved and self._stage_of:
            # Stages left behind by a request freed outside free_mamba_cache: reclaim those
            # whose slot the allocator holds as free (one sync, only when stages run out),
            # and those keyed by a request row (lazy_stage_key) the request pool holds as free.
            free_slots = set(allocator.free_slots.tolist())
            free_rows = set(req_pool.free_slots) if req_pool is not None else set()
            stale = [
                s
                for s in self._stage_of
                if s not in self._pending
                and ((s > 0 and s in free_slots) or (s < 0 and -s in free_rows))
            ]
            for slot in stale:
                self.release_slot(slot)
        return len(self._free) > self._reserved

    def reserve(self) -> None:
        """Hold a stage for an admitted request (after ``has_room``) until its ``assign``."""
        if len(self._free) <= self._reserved:
            raise RuntimeError(
                "KDA TP PD staging: no stage to reserve (check has_room first)"
            )
        self._reserved += 1

    def unreserve(self) -> None:
        """Drop the reservation of a request that left before its ``assign``."""
        if self._reserved <= 0:
            raise RuntimeError("KDA TP PD staging: unreserve without a reservation")
        self._reserved -= 1

    def rekey(self, key: int, slot: int) -> None:
        """Move a lazily keyed stage (lazy_stage_key) to the state slot its request was
        given at admission; the import then writes into that slot."""
        if key not in self._stage_of:
            raise RuntimeError(f"KDA TP PD staging: key {key} has no staged state")
        if slot in self._stage_of:
            # The slot was freed and reallocated without release_slot (as in assign).
            self.release_slot(slot)
        if key in self._pending:
            raise RuntimeError(
                f"KDA TP PD staging: key {key} was queued before admission"
            )
        self._stage_of[slot] = self._stage_of.pop(key)

    def assign(self, slot: int, reserved: bool = False) -> int:
        if slot in self._stage_of:
            # The slot was freed and reallocated without release_slot.
            self.release_slot(slot)
        if reserved:
            self.unreserve()
        if len(self._free) <= self._reserved:
            raise RuntimeError(
                "KDA TP PD staging: no free stage (check has_room first)"
            )
        stage = self._free.pop()
        self._stage_of[slot] = stage
        return stage

    def bind(self, slot: int) -> Optional[int]:
        """Late staging: the stage of ``slot``, binding a free one if it has none yet;
        None when every stage is taken (the prefill keeps the state and asks again)."""
        stage = self._stage_of.get(slot)
        if stage is not None:
            return stage
        self._reclaim()
        if len(self._free) <= self._reserved:
            return None
        return self.assign(slot)

    def stage_of(self, slot: int) -> Optional[int]:
        return self._stage_of.get(slot)

    def num_bound(self) -> int:
        return len(self._stage_of)

    def queue_import(self, slot: int) -> None:
        if slot not in self._stage_of:
            raise RuntimeError(f"KDA TP PD staging: slot {slot} has no staged state")
        if slot not in self._pending:
            self._pending.append(slot)

    def take_pending(self) -> List[Tuple[int, int]]:
        """(stage, slot) of every request queued since the last call."""
        out = [(self._stage_of[s], s) for s in self._pending]
        self._pending = []
        return out

    def finish_import(self, pairs: List[Tuple[int, int]], event) -> None:
        stages = []
        for stage, slot in pairs:
            if self._stage_of.get(slot) == stage:
                del self._stage_of[slot]
                stages.append(stage)
        if stages:
            self._draining.append((event, stages))

    def release_slot(self, slot: int) -> None:
        stage = self._stage_of.pop(slot, None)
        if stage is None:
            return
        if slot in self._pending:
            self._pending.remove(slot)
        self._free.append(stage)

    def import_states(
        self, mamba_pool, local: List[Tuple[int, int]], counts: List[int]
    ) -> None:
        """Collective over the TP group (every rank, counts from the MLP sync): write rank r's
        heads of every DP rank's staged states into the requests' global slots."""
        dev = self.temporal.device
        slots = torch.tensor([s for _, s in local], dtype=torch.int64, device=dev)
        gathered = all_gather_padded(slots, max(counts), -1).view(
            group().world_size, max(counts)
        )
        dst = torch.cat(
            [gathered[j, : counts[j]] for j in range(len(counts)) if counts[j]]
        )
        self.scatter(mamba_pool, [st for st, _ in local], counts, dst)
        event = None
        if dev.type == "cuda":
            event = torch.cuda.Event()
            event.record()
        self.finish_import(local, event)


def run_pd_imports(pool, forward_batch) -> None:
    """Import the PD-received states the MLP sync advertised for this forward (every rank,
    before the KDA layers or a graph replay read them)."""
    counts = forward_batch.dp_kda_tp_state_imports
    local = forward_batch.kda_tp_state_imports or []
    forward_batch.kda_tp_state_imports = None
    if not counts or not any(counts):
        return
    staging = pool.mamba_pool.pd_staging
    if staging is None:
        raise RuntimeError("KDA TP: state imports advertised without PD staging")
    rank = get_parallel().attn_dp_rank
    if len(local) != counts[rank]:
        raise RuntimeError(
            f"KDA TP: {len(local)} local state imports, MLP sync advertised {counts[rank]}"
        )
    staging.import_states(pool.mamba_pool, local, counts)
