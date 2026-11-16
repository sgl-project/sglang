"""Kimi-K3 tensor-parallel KDA under attention DP (``--enable-kda-tp``).

MLA stays attention-DP; each KDA layer all-gathers every DP rank's rows, runs on this
rank's heads/W with a head-sharded state pool (global slot ids), and reduce-scatters o_proj.
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
    return bool(get_parallel().enable_kda_tp) and _is_kimi_k3()


def enabled() -> bool:
    if not _applies():
        return False
    _check_supported()
    return True


def _check_supported() -> None:
    # Every collective here must run on every DP rank.
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
    topk = get_spec().speculative_eagle_topk
    if topk not in (None, 1):
        problems.append(
            f"--speculative-eagle-topk {topk} (the gathered verify metadata keeps each "
            "rank's own draft-tree fields)"
        )
    if envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get():
        problems.append(
            "SGLANG_SCHEDULER_SKIP_ALL_GATHER (the state imports and forward kinds "
            "come from the scheduler's DP all-gather)"
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
    return get_parallel().tp_size if _applies() else 1


def width() -> int:
    return get_parallel().tp_size


def group():
    return get_parallel().tp_group


def slot_partition(pool_slots: int) -> Tuple[int, int]:
    W = width()
    per_rank = pool_slots // W
    if per_rank < 1:
        raise ValueError(
            f"KDA TP: {pool_slots} state slots cannot be split over {W} DP ranks"
        )
    return get_parallel().attn_dp_rank * per_rank + 1, per_rank


def padded_rows(local_rows: int, counts: Optional[List[int]] = None) -> int:
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
    from sglang.srt.layers.communication import k3_sp_collective

    return k3_sp_collective.tp_group_state()


def all_gather_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    # May return a persistent buffer that the next KDA TP all-gather overwrites.
    from sglang.srt.layers.communication import k3_sp_collective

    # A prefill graph holding this replays only when every DP rank runs its recorded bucket.
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


# Never freed or rebound: CUDA-graph captures record their addresses.
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
# (gathered metadata installed on the linear backend, the per-rank metadata it came from)
_GATHERED: Optional[Tuple[Any, Any]] = None


def current() -> Context:
    if _CTX is None:
        raise RuntimeError("KDA TP: no forward context (begin_forward was not called)")
    return _CTX


def source_tokens(src: Source) -> int:
    # A breakable prefill graph pads every rank's block to the replayed size.
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
    # A collective: idle DP ranks call it too.
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
    from sglang.srt.model_executor.forward_mode import ForwardMode

    # The rows' own mode: a decode batch relabeled as a 1-token extend for the MLP sync is decode.
    kind = _mode_kind(forward_batch.logical_forward_mode)
    # Every rank captures the same graph, so the local kind is everyone's.
    if torch.cuda.is_current_stream_capturing():
        return [kind] * width()
    modes = getattr(forward_batch, "dp_forward_modes", None)
    if modes is not None and len(modes) == width():
        kinds = [
            "idle" if m == ForwardMode.PREBUILT.value else _mode_kind(ForwardMode(m))
            for m in modes
        ]
        if _speculative():
            # Spec decode runs TARGET_VERIFY; idle ranks must agree on the verify row layout.
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
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.runtime_context import get_spec

    global _GATHERED
    fm = backend.forward_metadata
    if fm is None:
        raise RuntimeError(
            "KDA TP: the linear attention backend has no forward metadata"
        )
    if _GATHERED is not None and fm is _GATHERED[0]:
        # A capture right after its warmup forward must re-gather the per-rank metadata.
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
        # Graph metadata buffers are sized to the captured max batch.
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
    # Padding requests use slot 0; every rank gathers track indices so collectives agree.
    from sglang.srt.runtime_context import get_exec

    # Persistent buffers (_static): captured graphs read them through recorded addresses.
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
        # Request i of DP rank j uses verify scratch row j * n + i (n: per-rank capacity).
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
    return backend.req_to_token_pool.size


def source_scratch_rows(backend, src: Source) -> torch.Tensor:
    dev = torch.device("cuda", torch.cuda.current_device())
    rows = src.rank * _rank_capacity(backend) + torch.arange(
        src.batch.batch_size, dtype=torch.int64, device=dev
    )
    return _scratch_rows(backend, rows)


def _scratch_rows(backend, rows: torch.Tensor) -> torch.Tensor:
    pad = width() * _rank_capacity(backend)
    return torch.where(rows < pad, rows, torch.full_like(rows, pad)).to(torch.int32)


def _local_state_slots(model_runner, bs: int):
    # Not the request table, which the scheduler may have rewritten since the forward.
    from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardStateSlots

    fm = model_runner.attn_backend.linear_attn_backend.forward_metadata
    if fm is None:
        raise RuntimeError("KDA TP commit: no verify forward metadata")
    if _GATHERED is not None and fm is _GATHERED[0]:
        fm = _GATHERED[1]
    track = fm.mamba_track_indices
    return ForwardStateSlots(
        fm.mamba_cache_indices[:bs], None if track is None else track[:bs]
    )


def commit_after_verify(
    model_runner,
    batch,
    last_correct_step_indices,
    mamba_steps_to_track,
    state_slots=None,
):
    # A collective: idle DP ranks call it with no requests.
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
        if state_slots is None:
            state_slots = _local_state_slots(model_runner, bs)
        slots = state_slots.working[:bs]
        track = state_slots.track
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
        rows_loop=envs.SGLANG_MAMBA_SSM_COMMIT_ROW_LOOP.get(),
    )


def mirror_deferred_state_ops(model_runner, forward_batch) -> None:
    # Runs on every rank at every target forward start; no-op unless a DP rank extends.
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


def lazy_mamba_slot(enable_mamba_extra_buffer: bool) -> bool:
    # Not with ping-pong tracking slots, which are allocated with the main slot.
    return (
        state_shard_width() > 1
        and not enable_mamba_extra_buffer
        and envs.SGLANG_K3_KDA_TP_LAZY_MAMBA_SLOT.get()
    )


def late_stage() -> bool:
    return state_shard_width() > 1 and envs.SGLANG_K3_KDA_TP_LATE_STAGE.get()


# State destination a late-staged request advertises until its stage is bound.
LATE_STAGE_UNBOUND = -1


def lazy_stage_key(req_pool_idx: int) -> int:
    # Negated, so it never collides with a state slot id (>= 1).
    return -int(req_pool_idx)


class FullStateStaging:
    """Full-width KDA state slots and the all-to-all into every rank's head shard."""

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
        # Conv channels reordered rank-major so rank r's block is its local layout.
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
        # Sizes are explicit: a rank with nothing to send has c == 0.
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

    def _state_pairs(self, mamba_pool):
        cache = mamba_pool.mamba_cache
        pairs = [(stage, shard, "conv") for stage, shard in zip(self.conv, cache.conv)]
        pairs.append((self.temporal, cache.temporal, "temporal"))
        return pairs

    # Bounds the all-to-all buffers of one chunk of layers.
    chunk_bytes = 256 << 20

    def _layer_chunk(self, L: int, nbytes_per_layer: int) -> int:
        # Every rank must derive it from the same global sizes.
        return max(1, min(L, self.chunk_bytes // max(1, nbytes_per_layer)))

    def scatter(
        self, mamba_pool, stages: List[int], counts: List[int], dst: torch.Tensor
    ) -> None:
        # Collective over the TP group; every rank passes the same ``counts``.
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


class PdStateStaging(FullStateStaging):
    """KDA TP on a PD decode worker: full-width receive slots for the prefill's KDA state."""

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
            # Reclaim stages leaked by requests freed outside free_mamba_cache (one sync).
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
        if len(self._free) <= self._reserved:
            raise RuntimeError(
                "KDA TP PD staging: no stage to reserve (check has_room first)"
            )
        self._reserved += 1

    def unreserve(self) -> None:
        if self._reserved <= 0:
            raise RuntimeError("KDA TP PD staging: unreserve without a reservation")
        self._reserved -= 1

    def rekey(self, key: int, slot: int) -> None:
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

    def release_request(self, req) -> None:
        # Both keys are host ints recorded at assign / admission, so this does not sync.
        for attr in ("kda_tp_staged_slot", "kda_tp_state_slot"):
            key = getattr(req, attr, None)
            if key is not None:
                self.release_slot(key)
                setattr(req, attr, None)

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
        # Collective over the TP group (every rank, counts from the MLP sync).
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
    # Every rank, before the KDA layers or a graph replay read the states.
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
