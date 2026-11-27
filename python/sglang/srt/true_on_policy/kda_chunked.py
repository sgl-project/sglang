"""TOP KDA (GLM-5.3-Flash): rollout decode follows the pinned KDA chunk-64 prefill bit for bit.

Under ``SGLANG_TOP_KDA_MODE=chunked`` every KDA output of a rollout equals row t of ``chunk_kda`` (the safe gate,
the fused intra kernel and the configs ``kda_top`` pins) over the request's whole sequence from zero state, so the
trainer runs that same prefill per sample (``top_kda_kernel().extend``).

Serving state per running request:

- the mamba slot's FP32 state holds E, the state at the start of the request's open 64-token chunk;
- a per-layer row cache indexed by the request's req-pool row holds the open chunk's rows (``kda_rows``).

Decode runs one row per step through the row engine (``kda_rows.decode``, closing the chunk at row 63); an extend
runs the chunk prefill up to each request's last 64-token boundary from E, then its tail rows one row-engine step at
a time. An extend must start on the 64-token grid (fresh requests, chunked-prefill segments of a 64-aligned size,
retractions, radix hits); mixed batches are refused, since no other path writes the row cache.

Radix reuse keeps SGLang's mamba track (``extra_buffer``): every state handed to the radix cache is the prefill's state
at a 64-token boundary, so a hit resumes E there and prefills the rest on the trainer's grid. An extend's track is the
prefill call's snapshot at a chunk start or its final state (E at the last boundary); a decode track fires on row 63,
where decode closes the chunk in place, and under speculation, where the close is deferred, ``close_to`` writes that
chunk's close to the track slot.

Under speculation (EAGLE chains, top-k 1) a verify runs the root and its drafts as consecutive one-row steps of the
row engine, each exactly the decode step of its position, with the window cut at the root's chunk end (the sampler
never accepts a draft past it, ``eagle_utils._top_chunked_cut_drafts``). The chunk close is deferred to the first step
past the chunk (``kda_rows.close_pending``), so E never holds a draft's row; rejected rows in the row cache are
overwritten by the next step, which starts right after the last accepted row. Verify keeps no per-draft states: the
commit moves only the convolution windows.
"""

import dataclasses
from types import SimpleNamespace
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.attention.fla import kda_rows, kda_top
from sglang.kernels.ops.attention.fla.kda_top import (
    TOP_KDA_CONFIG,
    top_kda_chunked,
)
from sglang.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.true_on_policy.gdn_chunked_engine import strided_mixed

CHUNK = 64

# Admitted text architectures (GLM-5.3-Flash: 64 KDA heads of 128, width-4 convolution, gate lower bound -5).
KDA_TOP_ARCHITECTURES = frozenset({"glm5_next_text"})


def top_kda_chunked_active() -> bool:
    """TOP is on for the running engine and its KDA layers run the chunked program."""
    from sglang.srt.runtime_context import get_server_args

    return top_kda_chunked() and get_server_args().true_on_policy_contract is not None


def validate_kda_contract(args, config) -> None:
    """What the TOP KDA program supports; raises on anything else (a refusal, never a fallback)."""
    if config.model_type not in KDA_TOP_ARCHITECTURES:
        raise ValueError(
            f"KDA TOP admits {sorted(KDA_TOP_ARCHITECTURES)}, got {config.model_type!r}"
        )
    lin = config.linear_attn_config
    lower = lin.get("gate_lower_bound")
    if (
        lin["head_dim"] != 128
        or lin["short_conv_kernel_size"] != 4
        or lower is None
        or not -5.0 <= lower < 0
    ):
        raise ValueError(
            "KDA TOP requires 128-wide KDA heads, a width-4 convolution and the safe gate with a lower bound "
            "in [-5, 0) (it bounds the gate's in-chunk exponent)"
        )
    if not top_kda_chunked():
        raise ValueError(
            "KDA TOP runs the chunked KDA program: the engine selects it at launch (SGLANG_TOP_KDA_MODE=chunked); "
            "a process that starts schedulers itself sets it"
        )
    if not kda_top.CHUNKED_AT_IMPORT:
        raise ValueError(
            "KDA TOP: the FLA KDA kernels were imported before the chunked KDA program was selected"
        )
    _validate_kda_kernel_envs()
    _validate_kda_speculation(args)
    _validate_kda_radix(args)
    if args.enable_mixed_chunk:
        raise ValueError("KDA TOP forbids mixed prefill/decode batches")
    if args.enable_two_batch_overlap:
        raise ValueError(
            "KDA TOP plans each extend over the whole batch: two-batch overlap is not supported"
        )
    if args.pp_size > 1:
        raise ValueError("KDA TOP does not support pipeline parallelism")
    if args.chunked_prefill_size is not None and args.chunked_prefill_size > 0:
        if args.chunked_prefill_size % CHUNK:
            raise ValueError("KDA TOP requires 64-aligned chunked prefill")
    if args.linear_attn_backend != "triton" or args.mamba_ssm_dtype != "float32":
        raise ValueError("KDA TOP requires Triton KDA kernels and FP32 recurrent state")
    for name in (
        "linear_attn_prefill_backend",
        "linear_attn_decode_backend",
        "linear_attn_verify_backend",
    ):
        if getattr(args, name) not in (None, "triton"):
            raise ValueError(f"KDA TOP requires Triton KDA kernels; got {name}")
    if (
        args.enable_streaming_session
        or args.disaggregation_mode != "null"
        or args.enable_hierarchical_cache
        or args.enable_lmcache
        or args.enable_flexkv
        or args.radix_cache_backend
    ):
        raise ValueError(
            "KDA TOP builds every state itself: streaming sessions, disaggregation, host state offload and "
            "external caches are not supported"
        )
    if args.enable_int8_mamba_checkpoint:
        raise ValueError("KDA TOP requires lossless FP32 states")
    if args.enable_linear_replayssm:
        raise ValueError("KDA TOP decodes through its own row engine, not ReplaySSM")


def _validate_kda_radix(args) -> None:
    if args.disable_radix_cache:
        return
    # no_buffer caches a request's state where it stops, off the grid; extra_buffer tracks on page and interval grids.
    if args.mamba_radix_cache_strategy != "extra_buffer":
        raise ValueError(
            "KDA TOP radix reuse requires --mamba-radix-cache-strategy extra_buffer"
        )
    if args.page_size % CHUNK or args.mamba_track_interval % CHUNK:
        raise ValueError(
            "KDA TOP radix reuse requires 64-aligned pages and Mamba tracking (hits resume on the 64-token grid)"
        )
    if args.enable_unified_cache_external_linker:
        raise ValueError(
            "KDA TOP builds every state itself: external cache linkers are not supported"
        )


def _validate_kda_speculation(args) -> None:
    if args.speculative_algorithm is None:
        return
    if args.speculative_algorithm != "EAGLE" or args.speculative_eagle_topk != 1:
        raise ValueError(
            "KDA TOP verifies EAGLE chains (top-k 1): the row engine steps through one draft per position"
        )
    drafts = args.speculative_num_draft_tokens
    if drafts is None or not 1 < drafts <= CHUNK:
        raise ValueError(
            f"KDA TOP verifies 2 to {CHUNK} tokens per request; got --speculative-num-draft-tokens {drafts}"
        )
    if args.enable_linear_replayssm_spec:
        raise ValueError("KDA TOP verifies through its own row engine, not ReplaySSM")


def _validate_kda_kernel_envs() -> None:
    # The row engine replays the state kernel at its default tile.
    from sglang.kernels.ops.attention.fla import chunk_delta_h
    from sglang.srt.environ import envs

    if (
        chunk_delta_h.GDN_CHUNK_H_BV,
        chunk_delta_h.GDN_CHUNK_H_NUM_WARPS,
        chunk_delta_h.GDN_CHUNK_H_NUM_STAGES,
    ) != (32, 4, 2):
        raise ValueError(
            "KDA TOP replays the state kernel at its default config: unset SGLANG_GDN_CHUNK_H_{BV,NUM_WARPS,NUM_STAGES}"
        )
    # The fused verify runs a T-step recurrence; TOP verifies with kda_rows on the chunk program.
    if envs.SGLANG_OPT_FUSED_KDA_VERIFY.get():
        raise ValueError("KDA TOP does not run the fused KDA verify kernel")


class _TopKDAKernel(TritonKDAKernel):
    """The dispatcher's decode and extend kernel under TOP KDA; the layer and the forward's coordinates come from the
    backend. Its ``extend`` with no backend (``top_kda_kernel()``) is the plain pinned prefill the trainer runs."""

    supports_packed_decode: bool = False
    supports_packed_decode_safe_gate: bool = False
    supports_fused_chain_verify: bool = False
    supports_track_state_snapshot: bool = True
    # Verify writes neither E nor per-draft states: the pool allocates no intermediate_ssm.
    verify_needs_intermediate_states: bool = False

    def __init__(self, backend: Optional["_TopKDAAttnBackend"] = None):
        self.backend = backend

    def decode(
        self,
        q,
        k,
        v,
        a,
        b,
        *,
        A_log,
        dt_bias,
        ssm_states,
        cache_indices,
        query_start_loc,
        lower_bound=None,
        **kwargs,
    ):
        be = self.backend
        mixed = torch.cat(
            (
                q.reshape(q.shape[1], -1),
                k.reshape(k.shape[1], -1),
                v.reshape(v.shape[1], -1),
            ),
            dim=-1,
        )
        out = be.engine.decode(
            layer=be.layer,
            step=be.step,
            mixed_qkv=mixed,
            f=a.reshape(mixed.shape[0], -1),
            b=b.reshape(mixed.shape[0], -1),
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            state=ssm_states,
        )
        return out.unsqueeze(0)

    def extend(
        self, q, k, v, g, beta, *, ssm_states, cache_indices, query_start_loc, **kwargs
    ):
        if self.backend is None:
            return super().extend(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                **kwargs,
            )
        be = self.backend
        out = be.engine.extend(
            layer=be.layer,
            plan=be.plan,
            q=q,
            k=k,
            v=v,
            f=g,
            b=beta,
            A_log=kwargs["A_log"],
            dt_bias=kwargs["dt_bias"],
            lower_bound=kwargs["lower_bound"],
            state=ssm_states,
            cache_indices=cache_indices,
            track_state=kwargs.get("track_state"),
        )
        # (output, per-chunk h): the radix track reads track_state or the final state, never h.
        return (out, None) if kwargs.get("return_intermediate_states") else out

    def target_verify(
        self,
        A_log,
        dt_bias,
        q,
        k,
        v,
        a,
        b,
        *,
        ssm_states,
        retrieve_parent_token=None,
        lower_bound=None,
        **kwargs,
    ):
        if retrieve_parent_token is not None:
            raise ValueError("KDA TOP verifies chain drafts only")
        be = self.backend
        T = q.shape[1]
        out = be.engine.verify(
            layer=be.layer,
            step=be.step,
            mixed_qkv=strided_mixed(q[0], k[0], v[0]),
            f=a.reshape(T, -1),
            b=b.reshape(T, -1),
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            state=ssm_states,
        )
        return out.unsqueeze(0)


_KERNEL = None


def top_kda_kernel():
    """The KDA chunk prefill serving runs under TOP; the trainer's prefill calls this same kernel object."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _TopKDAKernel()
    return _KERNEL


def _track_position(*, prefix, end, track_len) -> int:
    """The position whose state a prefill track names (``_init_track_ssm_indices``), on the 64-token grid since the
    prefix is."""
    length = track_len - prefix
    at = track_len if length % CHUNK == 0 else prefix + length // CHUNK * CHUNK
    if not prefix < at <= end:
        raise RuntimeError(
            f"[top] KDA: radix track at {at} outside the extend over ({prefix}, {end}]"
        )
    # The native track copies an aligned length from the slot's final state, E at the last boundary: only the end.
    if length % CHUNK == 0 and at != end:
        raise RuntimeError(
            f"[top] KDA: radix track at {at} of an extend over ({prefix}, {end}] is not its final state"
        )
    return at


class KDAStep(msgspec.Struct):
    """Row-engine coordinates of one decode or chain-verify forward, one entry per batch row (int32 device tensors);
    a verify's window entries are [nmax, batch], one row per draft depth."""

    ring: torch.Tensor  # row-cache slots (req-pool rows; the pad slot for padding)
    state: torch.Tensor  # mamba slots; -1 for padding
    rows: torch.Tensor  # the first token's row in its open chunk
    chunks: torch.Tensor  # the first token's chunk index
    # Window row j: ring / state / row of depth j, past the root's chunk end the pad slot, -1 and row 0.
    win_ring: Optional[torch.Tensor]
    win_state: Optional[torch.Tensor]
    win_rows: Optional[torch.Tensor]
    # bool [batch, nmax]: depth j is inside the root's chunk.
    inside: Optional[torch.Tensor]
    nmax: int


class KDAExtendPlan(msgspec.Struct):
    """One extend: a chunk prefill over each request's rows up to its last 64-token boundary, then tail steps."""

    fla_batch: Optional[
        torch.Tensor
    ]  # int64 batch rows of the requests with rows to prefill
    fla_cu: Optional[torch.Tensor]  # int32 cu_seqlens of the prefill call
    fla_src: Optional[
        torch.Tensor
    ]  # int64 packed extend rows prefilled (None: all rows, in order)
    fla_track_idx: Optional[
        torch.Tensor
    ]  # int32 tracked chunk inside the prefill call, -1 for none
    final_track: list  # (prefill-call row, batch row) whose tracked state is the prefill's final state
    tails: list  # (ring, state, row, src) per tail step: int32, int32, int32, int64 [n]


class ChunkedKDAEngine:
    def __init__(self, *, num_layers, num_slots, H, K, device, max_bs, spec=False):
        self.H, self.K, self.device = H, K, device
        self.num_slots = num_slots
        self.num_layers = num_layers
        # Under speculation every step defers its close (decode too), since any step may be a verify.
        self.spec = spec
        self.caches = [
            kda_rows.KDARowCache(
                num_slots=num_slots, num_heads=H, head_dim=K, device=device
            )
            for _ in range(num_layers)
        ]
        self.scratch = kda_rows.KDAScratch(
            max_bs=max_bs, num_slots=num_slots, H=H, K=K, device=device
        )
        self.graph_bufs = None

    def nbytes(self) -> int:
        return sum(c.bytes_per_slot() for c in self.caches) * (self.num_slots + 1)

    def alloc_graph_bufs(self, max_bs: int, nmax: int = 1):
        def buf(*shape, dtype=torch.int32):
            return torch.zeros(*shape, max_bs, dtype=dtype, device=self.device)

        self.graph_bufs = SimpleNamespace(
            ring=buf(),
            state=buf(),
            rows=buf(),
            chunks=buf(),
            win_ring=buf(nmax),
            win_state=buf(nmax),
            win_rows=buf(nmax),
            inside=buf(nmax, dtype=torch.bool),
        )

    def build_step(self, *, positions, ring, state, bs, static, nmax=1) -> KDAStep:
        """Coordinates of a decode (nmax 1, positions [bs]) or chain verify (positions [bs * nmax]) forward. With
        ``static``, written into the CUDA-graph buffers (device ops only, pointer-stable)."""
        first = positions.view(bs, nmax)[:, 0] if nmax > 1 else positions[:bs]
        first = first.clamp(min=0)
        values = dict(
            ring=torch.where(ring[:bs] < 0, self.num_slots, ring[:bs]),
            state=state[:bs],
            rows=first % CHUNK,
            chunks=first // CHUNK,
        )
        if nmax > 1:
            depth_rows = values["rows"][None, :] + torch.arange(
                nmax, device=first.device
            ).view(nmax, 1)
            inside = depth_rows < CHUNK
            values.update(
                win_ring=torch.where(inside, values["ring"][None, :], self.num_slots),
                win_state=torch.where(inside, values["state"][None, :], -1),
                win_rows=torch.where(inside, depth_rows, 0),
                inside=inside,
            )
        if not static:
            out = {
                k: v if v.dtype == torch.bool else v.to(torch.int32)
                for k, v in values.items()
            }
        else:
            out = {}
            for name, src in values.items():
                dst = getattr(self.graph_bufs, name)[..., :bs]
                dst.copy_(src)
                out[name] = dst
        if nmax > 1:
            out["inside"] = out["inside"].t()
        else:
            out.update(win_ring=None, win_state=None, win_rows=None, inside=None)
        return KDAStep(**out, nmax=nmax)

    def build_extend_plan(
        self, *, prefix, lens, ring, state, track_mask=None, track_lens=None
    ) -> Optional[KDAExtendPlan]:
        """Host plan of an extend. prefix / lens / ring / state: host ints per request; track_mask / track_lens
        (``mamba_prefill_track_mask_cpu`` / ``mamba_track_seqlens_cpu``): a track length L names the final state when
        (L - prefix) % 64 == 0, else the state at prefix + 64 * ((L - prefix) // 64)."""
        bs = len(lens)
        if bs == 0:
            return None
        starts = [0]
        for n in lens:
            starts.append(starts[-1] + n)
        fla_batch, fla_cu, fla_src = [], [0], []
        fla_track_idx, final_track = [], []
        tail_rows, e_vals = [], []
        for i in range(bs):
            p0, t1 = prefix[i], prefix[i] + lens[i]
            if p0 % CHUNK:
                raise RuntimeError(
                    f"[top] KDA: an extend from prefix {p0} is off the 64-token grid (a radix hit off the grid or a "
                    "mixed batch; the row cache holds no rows before it)"
                )
            c1 = t1 // CHUNK * CHUNK
            e_vals.append(max(c1, p0) // CHUNK)
            track_at = None
            if track_mask is not None and track_mask[i]:
                track_at = _track_position(prefix=p0, end=t1, track_len=track_lens[i])
            if c1 > p0:
                fla_batch.append(i)
                fla_src.extend(range(starts[i], starts[i] + c1 - p0))
                fla_cu.append(fla_cu[-1] + c1 - p0)
                if track_at == c1:
                    final_track.append((len(fla_batch) - 1, i))
                fla_track_idx.append(
                    -1 if track_at in (None, c1) else (track_at - p0) // CHUNK
                )
            for pos in range(max(c1, p0), t1):
                tail_rows.append((pos - max(c1, p0), i, pos, starts[i] + pos - p0))
        dev = self.device

        def d(values, dtype):
            return torch.tensor(values, dtype=dtype).to(dev, non_blocking=True)

        # E stands at each request's last boundary once its prefill ran (else at its open chunk's start).
        ring_t, e_t = d(ring, torch.int64), d(e_vals, torch.int32)
        for cache in self.caches:
            cache.e_chunk[ring_t] = e_t
        tails = []
        for j in range(CHUNK):
            sel = [(i, pos, src) for jj, i, pos, src in tail_rows if jj == j]
            if not sel:
                break
            tails.append(
                (
                    d([ring[i] for i, _, _ in sel], torch.int32),
                    d([state[i] for i, _, _ in sel], torch.int32),
                    d([pos % CHUNK for _, pos, _ in sel], torch.int32),
                    d([src for _, _, src in sel], torch.int64),
                )
            )
        has_fla = bool(fla_batch)
        identity = has_fla and len(fla_src) == starts[-1]
        return KDAExtendPlan(
            fla_batch=d(fla_batch, torch.int64) if has_fla else None,
            fla_cu=d(fla_cu, torch.int32) if has_fla else None,
            fla_src=None if identity or not has_fla else d(fla_src, torch.int64),
            fla_track_idx=d(fla_track_idx, torch.int32) if has_fla else None,
            final_track=final_track,
            tails=tails,
        )

    def decode(
        self, *, layer, step, mixed_qkv, f, b, A_log, dt_bias, lower_bound, state
    ):
        B = mixed_qkv.shape[0]
        out = mixed_qkv.new_empty(B, self.H, self.K)
        if step is None or B == 0:
            return out
        cache = self.caches[layer]
        gates = dict(
            A_log=A_log.reshape(-1),
            dt_bias=dt_bias.reshape(-1),
            lower_bound=lower_bound,
        )
        if self.spec:
            self._close_pending(cache=cache, step=step, state=state, gates=gates)
        kda_rows.decode(
            cache=cache,
            scratch=self.scratch,
            state=state,
            slots=(step.ring, step.state),
            rows=step.rows,
            mixed_qkv=mixed_qkv.contiguous(),
            f=f.contiguous(),
            b=b.contiguous(),
            out=out,
            config=TOP_KDA_CONFIG,
            close=not self.spec,
            **gates,
        )
        return out

    def verify(
        self, *, layer, step, mixed_qkv, f, b, A_log, dt_bias, lower_bound, state
    ):
        """``step.nmax`` rows per request (root + drafts, request-major), each the decode step of its position; rows
        past the root's chunk end are zeros (their drafts are never accepted). E advances only over chunks before
        the root's. Returns [T, H, V]."""
        T = mixed_qkv.shape[0]
        out = mixed_qkv.new_zeros(T, self.H, self.K)
        if step is None or T == 0:
            return out
        nmax = step.nmax
        cache = self.caches[layer]
        gates = dict(
            A_log=A_log.reshape(-1),
            dt_bias=dt_bias.reshape(-1),
            lower_bound=lower_bound,
        )
        self._close_pending(cache=cache, step=step, state=state, gates=gates)
        by_depth = out.view(-1, nmax, self.H, self.K)
        for j in range(nmax):
            kda_rows.decode(
                cache=cache,
                scratch=self.scratch,
                state=state,
                slots=(step.win_ring[j], step.win_state[j]),
                rows=step.win_rows[j],
                mixed_qkv=mixed_qkv[j::nmax],
                f=f[j::nmax],
                b=b[j::nmax],
                out=by_depth[:, j],
                config=TOP_KDA_CONFIG,
                close=False,
                **gates,
            )
        by_depth.masked_fill_(~step.inside[:, :, None, None], 0)
        return out

    def _close_pending(self, *, cache, step, state, gates):
        kda_rows.close_pending(
            cache=cache,
            scratch=self.scratch,
            state=state,
            slots=(step.ring, step.state),
            chunks=step.chunks,
            config=TOP_KDA_CONFIG,
            **gates,
        )

    def close_to(self, *, state, ring, slots, dst, gates):
        """For rows with dst >= 0, whose open chunk has all 64 rows cached and its close still deferred: the chunk's
        end state into state[:, dst] for every layer (E unchanged). ``state`` [layers, slots, H, V, K]; ``gates``
        per layer: (A_log, dt_bias, lower_bound) as the layer's forward passes them."""
        ring_slots = torch.where(ring < 0, self.num_slots, ring).to(torch.int32)
        state_slots, dst_slots = slots.to(torch.int32), dst.to(torch.int32)
        for layer, cache in enumerate(self.caches):
            A_log, dt_bias, lower_bound = gates[layer]
            kda_rows.close_to(
                cache=cache,
                scratch=self.scratch,
                state=state[layer],
                slots=(ring_slots, state_slots),
                dst_slots=dst_slots,
                A_log=A_log.reshape(-1),
                dt_bias=dt_bias.reshape(-1),
                lower_bound=lower_bound,
                config=TOP_KDA_CONFIG,
            )

    def extend(
        self,
        *,
        layer,
        plan,
        q,
        k,
        v,
        f,
        b,
        A_log,
        dt_bias,
        lower_bound,
        state,
        cache_indices,
        track_state=None,
    ):
        """q, k, v [1, T, H, K] post-conv rows; f [1, T, H, K] raw gate; b [1, T, H] raw beta; ``state`` the FP32
        pool read and updated at ``cache_indices`` [batch]. Returns [1, T, H, V]. ``track_state`` [batch, H, V, K]
        receives the radix track states the plan names."""
        T = q.shape[1]
        out = v.new_empty(1, T, self.H, self.K)
        if plan is None or T == 0:
            return out
        if plan.fla_batch is not None:
            src = plan.fla_src

            def pick(x):
                return x if src is None else x[:, src]

            fla_track = None
            if track_state is not None:
                fla_track = track_state.new_zeros(
                    plan.fla_batch.numel(), *track_state.shape[1:]
                )
            slots = cache_indices.index_select(0, plan.fla_batch)
            o = top_kda_kernel().extend(
                pick(q),
                pick(k),
                pick(v),
                pick(f),
                pick(b),
                ssm_states=state,
                cache_indices=slots,
                query_start_loc=plan.fla_cu,
                A_log=A_log,
                dt_bias=dt_bias,
                lower_bound=lower_bound,
                beta_is_raw=True,
                track_state=fla_track,
                track_chunk_idx=None if fla_track is None else plan.fla_track_idx,
            )
            if src is None:
                out = o
            else:
                out[:, src] = o
            if track_state is not None:
                # Before the tails: they never reach row 63, so E stays the prefill's final state.
                for row, batch_row in plan.final_track:
                    fla_track[row] = state[slots[row]]
                track_state[plan.fla_batch] = fla_track
        if plan.tails:
            cache = self.caches[layer]
            mixed = torch.cat(
                (q[0].reshape(T, -1), k[0].reshape(T, -1), v[0].reshape(T, -1)), dim=-1
            )
            f2, b2 = f[0].reshape(T, -1), b[0].reshape(T, -1)
            for ring, slots, rows, src in plan.tails:
                o = mixed.new_empty(src.numel(), self.H, self.K)
                kda_rows.decode(
                    cache=cache,
                    scratch=self.scratch,
                    state=state,
                    slots=(ring, slots),
                    rows=rows,
                    mixed_qkv=mixed.index_select(0, src),
                    f=f2.index_select(0, src),
                    b=b2.index_select(0, src),
                    A_log=A_log.reshape(-1),
                    dt_bias=dt_bias.reshape(-1),
                    lower_bound=lower_bound,
                    out=o,
                    config=TOP_KDA_CONFIG,
                )
                out[0, src] = o
        return out


class _TopKDAAttnBackend(KDAAttnBackend):
    """The native KDA backend with its convolution and gating, the kernel object its dispatcher calls swapped for
    the TOP one and the prefill CUDA-graph extend off."""

    def __init__(self, model_runner):
        from sglang.srt.arg_groups.overrides import resolved_view
        from sglang.srt.utils import rank0_log

        super().__init__(model_runner)
        args = resolved_view(model_runner.server_args)
        config = model_runner.model_config.hf_text_config
        temporal = self.req_to_token_pool.mamba_pool.mamba_cache.temporal
        H, V, K = temporal.shape[-3:]
        validate_kda_contract(args, config)
        if H != config.linear_attn_config["num_heads"]:
            raise ValueError(
                f"KDA TOP runs every KDA head on each rank (attention TP1), got {H} of {config.linear_attn_config['num_heads']}"
            )
        num_slots = self.req_to_token_pool.size + 1
        spec = args.speculative_algorithm is not None
        self.nmax = args.speculative_num_draft_tokens if spec else 1
        self.engine = ChunkedKDAEngine(
            num_layers=temporal.shape[0],
            num_slots=num_slots,
            H=H,
            K=K,
            device=self.device,
            max_bs=num_slots,
            spec=spec,
        )
        self.step = self.plan = self.layer = None
        # Per KDA layer, (A_log, dt_bias, lower_bound) as its forward last passed them: a track's close after the
        # forward recomputes each layer's gate cumsum.
        self.layer_gates = [None] * temporal.shape[0]
        kernel = _TopKDAKernel(self)
        dispatcher = self.kernel_dispatcher
        dispatcher.extend_kernel = dispatcher.decode_kernel = (
            dispatcher.verify_kernel
        ) = kernel
        # The packed recurrent decode advances the state one token at a time, unlike the chunk prefill kda_rows follows.
        dispatcher.supports_packed_decode = False
        dispatcher.supports_packed_decode_safe_gate = False
        rank0_log(
            f"[top] chunked KDA: row cache {self.engine.nbytes() / (1 << 30):.2f} GiB over {temporal.shape[0]} layers "
            f"x {num_slots} request rows; config {TOP_KDA_CONFIG}"
        )

    def supports_prefill_graph_extend(self) -> bool:
        return False

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        super().init_cuda_graph_state(max_bs, max_num_tokens)
        self.engine.alloc_graph_bufs(max_bs, self.nmax)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture: bool = False):
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self.plan = None
        self.step = self._build_step(forward_batch, static=True)

    def init_forward_metadata(self, forward_batch):
        super().init_forward_metadata(forward_batch)
        self.step = self.plan = None
        mode = forward_batch.forward_mode
        if mode.is_idle() or forward_batch.batch_size == 0:
            return
        if mode.is_decode() or mode.is_target_verify():
            self.step = self._build_step(forward_batch, static=False)
        elif mode.is_extend_without_speculative():
            prefix = forward_batch.extend_prefix_lens_cpu
            lens = forward_batch.extend_seq_lens_cpu
            if prefix is None or lens is None:
                raise ValueError("TOP KDA extends need host prefix and extend lengths")
            track_mask = forward_batch.mamba_prefill_track_mask_cpu
            track_lens = forward_batch.mamba_track_seqlens_cpu
            if forward_batch.mamba_track_mask is not None and (
                not self._has_cpu_prefill_track_metadata(forward_batch)
            ):
                # Prefill graph capture runs a dummy batch,
                # with an all-zero mask and no host lists.
                if bool(forward_batch.mamba_track_mask.any()):
                    raise ValueError("TOP KDA radix tracks need host track metadata")
                track_mask = track_lens = None
            self.plan = self.engine.build_extend_plan(
                prefix=[int(x) for x in prefix],
                lens=[int(x) for x in lens],
                ring=forward_batch.req_pool_indices.tolist(),
                state=self.forward_metadata.mamba_cache_indices.tolist(),
                track_mask=track_mask,
                track_lens=track_lens,
            )
        else:
            raise ValueError(f"TOP KDA does not run {mode}")

    def _build_step(self, forward_batch, *, static: bool):
        nmax = 1
        if forward_batch.forward_mode.is_target_verify():
            spec = forward_batch.spec_info
            if spec.ragged_verify_layout is not None:
                raise ValueError(
                    "TOP KDA verifies a dense [requests, draft tokens] window"
                )
            nmax = spec.draft_token_num
            if nmax != self.nmax:
                raise ValueError(
                    f"TOP KDA verify got {nmax} rows per request, sized for {self.nmax}"
                )
        return self.engine.build_step(
            positions=forward_batch.positions,
            ring=forward_batch.req_pool_indices,
            state=self.forward_metadata.mamba_cache_indices,
            bs=forward_batch.batch_size,
            static=static,
            nmax=nmax,
        )

    def forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        self._enter_layer(layer)
        return super().forward_decode(layer, forward_batch, mixed_qkv, a, b, **kwargs)

    def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        self._enter_layer(layer)
        return super().forward_extend(layer, forward_batch, mixed_qkv, a, b, **kwargs)

    def _enter_layer(self, layer):
        self.layer = self.req_to_token_pool.mamba2_layer_index(layer.layer_id)
        self.layer_gates[self.layer] = (layer.A_log, layer.dt_bias, layer.lower_bound)

    def _track_mamba_state_decode(
        self, forward_batch, conv_states, ssm_states, cache_indices, layer_id=None
    ):
        # A decode track fires when a request's token lands on row 63. Without speculation decode closes the chunk in
        # place there, so the slot holds the boundary state the copy takes; under speculation the close is deferred,
        # so the copied E is replaced by the chunk's close, for every layer at the last one.
        super()._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer_id
        )
        engine = self.engine
        if (
            not engine.spec
            or forward_batch.mamba_track_mask is None
            or self.step is None
            or self.layer != engine.num_layers - 1
        ):
            return
        step = self.step
        bs = step.ring.numel()
        engine.close_to(
            state=self.req_to_token_pool.mamba_pool.mamba_cache.temporal,
            ring=step.ring,
            slots=step.state,
            dst=torch.where(
                forward_batch.mamba_track_mask[:bs],
                self.forward_metadata.mamba_track_indices[:bs].to(torch.int64),
                -1,
            ),
            gates=self.layer_gates,
        )

    def commit_after_verify(
        self,
        *,
        state_indices,
        req_pool_indices,
        last_correct_step_indices,
        mamba_track_indices,
        mamba_steps_to_track,
        mamba_caches,
    ):
        """The verify commit: E and the row cache already hold the committed rows (E never takes a draft's row,
        rejected rows are overwritten by the next step), so only the convolution windows move; a track slot crossed
        this step (its chunk ended on a committed row 63) gets the chunk's close."""
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            scatter_mamba_states_after_mtp_verify,
        )

        # An empty temporal view: the scatter skips SSM states and moves the conv windows only.
        scatter_mamba_states_after_mtp_verify(
            dataclasses.replace(mamba_caches, temporal=mamba_caches.temporal[:0]),
            state_indices,
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
        )
        if mamba_track_indices is None:
            return
        self.engine.close_to(
            state=mamba_caches.temporal,
            ring=req_pool_indices,
            slots=state_indices,
            dst=torch.where(
                mamba_steps_to_track >= 0, mamba_track_indices.to(torch.int64), -1
            ),
            gates=self.layer_gates,
        )


def top_kda_attn_backend(runner):
    """The KDA backend for TOP runs; None when TOP is off."""
    from sglang.srt.true_on_policy.contracts import (
        resolve_true_on_policy_runtime_policy,
    )

    if not resolve_true_on_policy_runtime_policy(runner.server_args).enabled:
        return None
    return _TopKDAAttnBackend(runner)
