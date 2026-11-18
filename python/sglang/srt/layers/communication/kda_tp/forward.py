"""Per-forward KDA metadata of every DP rank, gathered for the TP-sharded KDA layers."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.layers.communication import k3_sp_collective
from sglang.srt.layers.communication.kda_tp import (
    all_gather_padded,
    group,
    width,
)
from sglang.srt.layers.dp_attention import _note_dp_gather_in_prefill_graph


class Source(msgspec.Struct):
    """One DP rank's block of the gathered rows (general path)."""

    rank: int
    rows: int  # real rows in the block, at [rank * R, rank * R + rows)
    metadata: Any  # that rank's ForwardMetadata (slot ids already global)
    batch: SimpleNamespace  # the ForwardBatch fields the KDA backend reads


class Context(msgspec.Struct):
    local_rows: int
    R: int  # rows per DP rank block in the gathered tensor
    mode: Any  # ForwardMode shared by every rank (fast path) or None (general path)
    batch: Any  # the forward batch the KDA backend sees on the fast path
    metadata: Any = None  # fast path: every rank's metadata, installed per layer
    sources: Optional[List[Source]] = None  # general path


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


class KdaTp:
    """One Kimi-K3 model's tensor-parallel KDA: every DP rank's rows and metadata of this forward."""

    def __init__(self):
        self.ctx: Optional[Context] = None
        # Never freed or rebound: CUDA-graph captures record their addresses.
        self._buffers: Dict[Tuple[str, torch.dtype], torch.Tensor] = {}
        # Collective workspace request: every rank, before graph capture.
        k3_sp_collective.tp_group_state()

    def context(self) -> Context:
        if self.ctx is None:
            raise RuntimeError(
                "KDA TP: no forward context (begin_forward was not called)"
            )
        return self.ctx

    def all_gather_rows(self, x: torch.Tensor, rows: int) -> torch.Tensor:
        # May return a persistent buffer that the next KDA TP all-gather overwrites.
        # A prefill graph holding this replays only when every DP rank runs its recorded bucket.
        _note_dp_gather_in_prefill_graph()
        g = group()
        return k3_sp_collective.all_gather_rows(
            x,
            rows,
            g,
            k3_sp_collective.tp_rows_state(g.world_size * rows),
            "kda_tp_all_gather",
        )

    def o_proj_output(
        self, total_rows: int, width: int, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        state = k3_sp_collective.tp_rows_state(total_rows)
        if state is None or not k3_sp_collective.requires_symmetric_rs(
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

    def reduce_scatter_rows(self, y: torch.Tensor, local_rows: int) -> torch.Tensor:
        return k3_sp_collective.reduce_scatter_rows(
            y, local_rows, group(), k3_sp_collective.tp_rows_state(y.shape[0])
        )

    def _buffer(self, name: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
        buf = self._buffers.get((name, dtype))
        if buf is not None and buf.numel() >= numel:
            return buf
        if buf is not None and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"KDA TP: static buffer {name} must grow to {numel} during a CUDA graph capture"
            )
        buf = torch.empty(numel, dtype=dtype, device=torch.cuda.current_device())
        self._buffers[(name, dtype)] = buf
        return buf

    def _all_gather_padded(
        self, t: torch.Tensor, n: int, pad_value: int, name: str, capacity: int
    ) -> torch.Tensor:
        g = group()
        src = self._buffer(f"{name}.in", capacity, t.dtype)[:n]
        src[t.shape[0] :].fill_(pad_value)
        src[: t.shape[0]].copy_(t)
        out = self._buffer(f"{name}.out", g.world_size * capacity, t.dtype)[
            : g.world_size * n
        ]
        g.all_gather_into_tensor(out, src)
        return out

    def begin_forward(
        self,
        forward_batch,
        linear_backend,
        local_rows: int,
        counts: Optional[List[int]] = None,
    ) -> None:
        # A collective: idle DP ranks call it too.
        R = k3_sp_collective.dp_max_rows(local_rows, counts)
        kinds = _global_kinds(forward_batch)
        active = {k for k in kinds if k != "idle"}
        if len(active) <= 1 and active <= {"decode", "verify"}:
            self.ctx = self._fast_context(
                forward_batch, linear_backend, local_rows, R, active
            )
        else:
            self.ctx = self._general_context(
                forward_batch, linear_backend, local_rows, R
            )

    def _fast_context(self, forward_batch, backend, local_rows, R, active) -> Context:
        from sglang.srt.model_executor.forward_mode import ForwardMode
        from sglang.srt.runtime_context import get_spec

        fm = backend.forward_metadata
        if fm is None:
            raise RuntimeError(
                "KDA TP: the linear attention backend has no forward metadata"
            )
        W = width()
        kind = next(iter(active)) if active else "decode"
        if kind == "verify":
            D = get_spec().speculative_num_draft_tokens
            mode = ForwardMode.TARGET_VERIFY
            if forward_batch.ragged_verify_layout is not None:
                raise NotImplementedError(
                    "KDA TP: ragged verify layouts are not supported"
                )
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

        # Persistent buffers (_buffer): captured graphs read them through recorded addresses.
        cap = _rank_capacity(backend)
        if bs > cap:
            raise RuntimeError(
                f"KDA TP: DP max batch {bs} exceeds the per-rank capacity {cap}"
            )
        # int64 on the wire: every rank must gather one dtype.
        slots = self._buffer("slots", W * cap, fm.mamba_cache_indices.dtype)[: W * bs]
        slots.copy_(
            self._all_gather_padded(
                local_slots.to(torch.int64), bs, 0, name="slots", capacity=cap
            )
        )
        track = None
        if get_exec().mamba.enable_mamba_extra_buffer:
            track = self._all_gather_padded(
                (
                    local_track if local_track is not None else local_slots.new_zeros(0)
                ).to(torch.int64),
                bs,
                0,
                name="track",
                capacity=cap,
            )
            if fm.mamba_track_indices is not None:
                gathered = track
                track = self._buffer("track", W * cap, fm.mamba_track_indices.dtype)[
                    : W * bs
                ]
                track.copy_(gathered)
        track_mask = None
        if kind == "decode" and get_exec().mamba.enable_mamba_extra_buffer:
            # Plain decode tracks in-layer (_track_mamba_state_decode) by this mask.
            local_mask = forward_batch.mamba_track_mask
            if local_mask is None or forward_batch.forward_mode.is_idle():
                local_mask = torch.zeros(0, dtype=torch.int64, device=dev)
            local_mask = local_mask[: local_slots.shape[0]]
            track_mask = self._buffer("track_mask", W * cap, torch.bool)[: W * bs]
            track_mask.copy_(
                self._all_gather_padded(
                    local_mask.to(torch.int64), bs, 0, name="track_mask", capacity=cap
                )
            )
        qsl = self._buffer("qsl", W * cap + 1, torch.int32)[: W * bs + 1]
        qsl.copy_(
            torch.arange(0, W * bs * D + 1, D, dtype=torch.int32, device=slots.device)
        )
        if kind == "verify":
            # Request i of DP rank j uses verify scratch row j * n + i (n: per-rank capacity).
            pos = torch.arange(W * bs, dtype=torch.int64, device=slots.device)
            rows = self._buffer("verify_rows", W * cap, torch.int32)[: W * bs]
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
        return Context(
            local_rows=local_rows, R=R, mode=mode, batch=batch, metadata=gathered
        )

    def _general_context(self, forward_batch, backend, local_rows, R) -> Context:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "KDA TP: mixed-mode forwards cannot run inside a CUDA graph"
            )
        from sglang.srt.model_executor.forward_mode import ForwardMode

        fm = (
            backend.forward_metadata
            if not forward_batch.forward_mode.is_idle()
            else None
        )
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
            if (
                forward_batch.logical_forward_mode.is_decode()
                and track_mask is not None
            ):
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
        return Context(
            local_rows=local_rows, R=R, mode=None, batch=None, sources=sources
        )
