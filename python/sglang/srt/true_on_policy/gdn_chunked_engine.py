"""The TOP chunked GDN row engine as serving drives it (see gdn_chunked.py), free of model and scheduler objects.

``ChunkedGDNEngine`` owns the per-request row caches of every GDN layer and runs one layer's GDN core for a
decode, a chain verify or an extend, given the post-convolution rows and the FP32 state pool. Its inputs are
the coordinates a forward carries: per request row, the req-pool row (row-cache index), the mamba slot (E) and
the token positions. The backend (gdn_chunked_backend.py) only adapts SGLang's forward metadata to them.

Exactness: every output row equals row t of ``chunk_gated_delta_rule(fixed_output_tile=True)`` over the request's
whole sequence from zero state, and every FP32 state handed to the radix cache equals that prefill's state at the
same 64-token boundary (``test_gdn_chunked_rows.py``).
"""

from types import SimpleNamespace
from typing import Optional

import msgspec
import torch

from sglang.kernels.ops.attention.fla import chunk_rows, chunk_rows2
from sglang.kernels.ops.attention.fla.chunk_fwd import TOP_CHUNKED_KKT_CONFIG
from sglang.srt.true_on_policy.gdn_chunked import CHUNK, MAX_STEP_ROWS


class RowPool:
    """Row caches of every GDN layer (``chunk_rows2.RowCache`` layout) stacked on a leading layer dim, one entry
    per req-pool row; row 0, SGLang's padding row, absorbs padded batch rows."""

    def __init__(self, *, num_layers, num_slots, Hg, H, K, V, device):
        def zeros(*shape, dtype):
            return torch.zeros(
                num_layers, num_slots, *shape, dtype=dtype, device=device
            )

        self.k = zeros(CHUNK, Hg, K, dtype=torch.bfloat16)
        self.v = zeros(CHUNK, H, V, dtype=torch.bfloat16)
        self.v_new = zeros(CHUNK, H, V, dtype=torch.bfloat16)
        self.g = zeros(CHUNK, H, dtype=torch.float32)
        self.beta = zeros(CHUNK, H, dtype=torch.float32)
        self.solved = zeros(H, CHUNK, CHUNK, dtype=torch.float32)
        self.e_chunk = zeros(H, dtype=torch.int32)
        self.layers = [
            SimpleNamespace(
                **{
                    name: getattr(self, name)[i]
                    for name in ("k", "v", "v_new", "g", "beta", "solved", "e_chunk")
                }
            )
            for i in range(num_layers)
        ]

    def nbytes(self) -> int:
        return sum(
            getattr(self, n).numel() * getattr(self, n).element_size()
            for n in ("k", "v", "v_new", "g", "beta", "solved", "e_chunk")
        )


class Step(msgspec.Struct):
    """Row-engine coordinates of one decode or verify forward, one entry per batch row (int32 device tensors)."""

    ring: torch.Tensor  # req-pool rows
    state: torch.Tensor  # mamba slots; -1 for padding
    rows: torch.Tensor  # first row of the step in the open chunk
    counts: (
        torch.Tensor
    )  # rows this step: 1 for decode, the verify window cut at the chunk end
    chunks: torch.Tensor  # chunk index of the first row
    nmax: int


class TailStep(msgspec.Struct):
    """One row-engine step over extend tail rows: up to MAX_STEP_ROWS rows of each listed request."""

    ring: torch.Tensor
    state: torch.Tensor
    rows: torch.Tensor
    counts: torch.Tensor
    chunks: torch.Tensor
    src: torch.Tensor  # int64 [n * MAX_STEP_ROWS] packed extend rows of the step's rows (padding repeats one)
    out_src: torch.Tensor  # int64 step-output rows that are real
    out_dst: torch.Tensor  # int64 their packed extend rows


class ExtendPlan(msgspec.Struct):
    """One extend forward: chunk prefill over each request's rows up to its last 64-token boundary (from E, after
    the open chunk's cached rows when the prefix is off the grid), then the tail rows through row-engine steps."""

    fla_batch: Optional[
        torch.Tensor
    ]  # int64 batch rows of the requests with rows to prefill
    fla_cu: Optional[torch.Tensor]  # int32 cu_seqlens of the prefill call
    fla_identity: bool  # the prefill covers every new row of every request: no gather
    new_src: Optional[torch.Tensor]  # int64 packed extend rows prefilled
    new_dst: Optional[torch.Tensor]  # int64 their rows in the prefill call
    open_ring: Optional[
        torch.Tensor
    ]  # int64 flat cached rows (req row * 64 + row) of off-grid open chunks
    open_dst: Optional[torch.Tensor]  # int64 their rows in the prefill call
    fla_track_idx: Optional[
        torch.Tensor
    ]  # int32 tracked chunk inside the prefill call, -1 for none
    final_track: list  # (prefill-call row, batch row) whose tracked state is the prefill's final state
    tails: list  # TailStep


def _to_device(values, dtype, device):
    from sglang.srt.utils import is_pin_memory_available

    host = torch.tensor(values, dtype=dtype)
    if torch.device(device).type != "cuda":
        return host
    if is_pin_memory_available(device):
        host = host.pin_memory()
    return host.to(device, non_blocking=True)


def strided_mixed(q, k, v):
    """The packed [T, q | k | v] rows ``q``, ``k`` and ``v`` were split from when they are strided views of it
    (SGLang's strided target-verify split), else a concatenated copy."""
    t = q.shape[-3]
    width = sum(x.shape[-2] * x.shape[-1] for x in (q, k, v))
    row = q.stride(-3)
    if (
        q.stride(-1) == 1
        and row == k.stride(-3) == v.stride(-3)
        and row >= width
        and k.data_ptr() == q.data_ptr() + q.shape[-2] * q.shape[-1] * q.element_size()
        and v.data_ptr() == k.data_ptr() + k.shape[-2] * k.shape[-1] * k.element_size()
    ):
        return torch.as_strided(q, (t, width), (row, 1))
    return torch.cat((q.reshape(t, -1), k.reshape(t, -1), v.reshape(t, -1)), dim=-1)


class ChunkedGDNEngine:
    def __init__(
        self, *, num_layers, num_slots, Hg, H, K, V, device, spec, nmax, max_bs
    ):
        self.num_layers, self.num_slots = num_layers, num_slots
        self.Hg, self.H, self.K, self.V = Hg, H, K, V
        self.device = device
        # Under speculation every step may be a verify, whose close is deferred: decode then runs chunk_rows2 too.
        self.spec = spec
        self.nmax = nmax
        self.kkt = TOP_CHUNKED_KKT_CONFIG
        self.rows = RowPool(
            num_layers=num_layers,
            num_slots=num_slots,
            Hg=Hg,
            H=H,
            K=K,
            V=V,
            device=device,
        )
        self._scratch_bs = 0
        self.ensure_scratch(max_bs)
        self.graph_bufs = None

    def ensure_scratch(self, bs: int):
        if bs <= self._scratch_bs:
            return
        self._scratch_bs = bs
        nmax = max(self.nmax, MAX_STEP_ROWS)
        self.scratch2 = chunk_rows2.Scratch(
            bs, nmax, self.Hg, self.H, self.K, self.V, self.device
        )
        self.scratch1 = chunk_rows.Scratch(bs, self.Hg, self.H, self.K, self.device)

    def layer(self, i):
        return self.rows.layers[i]

    # ------------------------------------------------------------------ coordinates

    def alloc_graph_bufs(self, max_bs: int):
        self.ensure_scratch(max_bs)

        def buf():
            return torch.zeros(max_bs, dtype=torch.int32, device=self.device)

        self.graph_bufs = SimpleNamespace(
            ring=buf(), state=buf(), rows=buf(), counts=buf(), chunks=buf()
        )

    def build_step(self, *, positions, ring, state, bs, nmax, static=False) -> Step:
        """Coordinates of a decode (nmax 1, positions [bs]) or chain verify (positions [bs * nmax]) forward. With
        ``static``, written into the CUDA-graph buffers (device ops only, pointer-stable)."""
        if nmax > 1:
            first = positions.view(bs, nmax)[:, 0]
        else:
            first = positions[:bs]
        first = first.clamp(min=0)
        rows = first % CHUNK
        values = dict(
            ring=ring[:bs],
            state=state[:bs],
            rows=rows,
            counts=torch.clamp(CHUNK - rows, max=nmax),
            chunks=first // CHUNK,
        )
        if not static:
            return Step(**{k: v.to(torch.int32) for k, v in values.items()}, nmax=nmax)
        out = {}
        for name, src in values.items():
            dst = getattr(self.graph_bufs, name)[:bs]
            dst.copy_(src)
            out[name] = dst
        return Step(**out, nmax=nmax)

    def build_extend_plan(
        self, *, prefix, lens, ring, state, track_mask=None, track_lens=None
    ) -> Optional[ExtendPlan]:
        """Host plan of an extend (host ints per request); also records, for every request, the chunk E will stand at.

        A track length L (``mamba_track_seqlens_cpu``) names the state at L when (L - prefix) % 64 == 0, else the
        state at prefix + 64 * ((L - prefix) // 64)."""
        bs = len(lens)
        if bs == 0:
            return None
        starts = [0]
        for n in lens:
            starts.append(starts[-1] + n)
        fla_batch, fla_cu = [], [0]
        new_src, new_dst, open_ring, open_dst = [], [], [], []
        fla_track_idx, final_track = [], []
        identity = True
        tails, e_vals = [], []
        for i in range(bs):
            p0, t1 = prefix[i], prefix[i] + lens[i]
            m = p0 % CHUNK
            c0, c1 = p0 - m, t1 // CHUNK * CHUNK
            e_vals.append(max(c0, c1) // CHUNK)
            track_at = None
            if track_mask is not None and track_mask[i]:
                length = track_lens[i] - p0
                track_at = (
                    track_lens[i]
                    if length % CHUNK == 0
                    else p0 + length // CHUNK * CHUNK
                )
                if m or track_at % CHUNK or track_at <= p0:
                    raise RuntimeError(
                        f"[top] chunked GDN: radix track at {track_at} from prefix {p0} is off the 64-token grid"
                    )
            if c1 > p0:
                row = len(fla_batch)
                fla_batch.append(i)
                base = fla_cu[-1]
                if m:
                    open_ring.extend(ring[i] * CHUNK + r for r in range(m))
                    open_dst.extend(range(base, base + m))
                n_new = c1 - p0
                new_src.extend(range(starts[i], starts[i] + n_new))
                new_dst.extend(range(base + m, base + m + n_new))
                fla_cu.append(base + m + n_new)
                identity &= m == 0 and n_new == lens[i]
                idx = -1
                if track_at is not None:
                    if track_at == c1:
                        final_track.append((row, i))
                    else:
                        idx = (track_at - c0) // CHUNK
                fla_track_idx.append(idx)
            else:
                identity = False
                if track_at is not None:
                    raise RuntimeError(
                        f"[top] chunked GDN: radix track at {track_at} inside an extend that prefills no chunk"
                    )
            if t1 > max(c1, p0):
                tails.append((i, max(c1, p0), t1))
        identity &= len(fla_batch) == bs

        dev = self.device

        def d(values, dtype=torch.int64):
            return _to_device(values, dtype, dev)

        # E stands at each request's last boundary once its prefill ran (else at its open chunk's start); rows
        # cached for that chunk by an earlier request on this req-pool row are dead.
        self.rows.e_chunk[:, d(ring)] = d(e_vals, torch.int32)[None, :, None]

        steps = []
        i32 = torch.int32
        for j in range(CHUNK // MAX_STEP_ROWS):
            sel, src, out_src, out_dst = [], [], [], []
            for i, start, end in tails:
                p = start + j * MAX_STEP_ROWS
                if p >= end:
                    continue
                n = min(MAX_STEP_ROWS, end - p)
                k = len(sel)
                sel.append((i, p, n))
                first = starts[i] + (p - prefix[i])
                src.extend(first + r if r < n else first for r in range(MAX_STEP_ROWS))
                out_src.extend(k * MAX_STEP_ROWS + r for r in range(n))
                out_dst.extend(range(first, first + n))
            if not sel:
                break
            steps.append(
                TailStep(
                    ring=d([ring[i] for i, _, _ in sel], i32),
                    state=d([state[i] for i, _, _ in sel], i32),
                    rows=d([p % CHUNK for _, p, _ in sel], i32),
                    counts=d([n for _, _, n in sel], i32),
                    chunks=d([p // CHUNK for _, p, _ in sel], i32),
                    src=d(src),
                    out_src=d(out_src),
                    out_dst=d(out_dst),
                )
            )
            self.ensure_scratch(len(sel))
        has_fla = bool(fla_batch)
        gathered = has_fla and not identity
        return ExtendPlan(
            fla_batch=d(fla_batch) if has_fla else None,
            fla_cu=d(fla_cu, torch.int32) if has_fla else None,
            fla_identity=identity,
            new_src=d(new_src) if gathered else None,
            new_dst=d(new_dst) if gathered else None,
            open_ring=d(open_ring) if open_ring else None,
            open_dst=d(open_dst) if open_ring else None,
            fla_track_idx=d(fla_track_idx, torch.int32) if has_fla else None,
            final_track=final_track,
            tails=steps,
        )

    # ------------------------------------------------------------------ one layer

    def decode(self, layer, step, mixed_qkv, a, b, A_log, dt_bias, state):
        """One row per request; returns [B, H, V]."""
        B = mixed_qkv.shape[0]
        out = mixed_qkv.new_empty(B, self.H, self.V)
        if step is None or B == 0:
            return out
        a, b = a.contiguous(), b.contiguous()
        common = dict(
            cache=self.layer(layer),
            state=state,
            slots=(step.ring, step.state),
            rows=step.rows,
            chunks=step.chunks,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            out=out,
            num_k_heads=self.Hg,
            head_k_dim=self.K,
            kkt=self.kkt,
        )
        if self.spec:
            chunk_rows2.step(
                scratch=self.scratch2, counts=step.counts, nmax=1, **common
            )
        else:
            chunk_rows.decode(scratch=self.scratch1, **common)
        return out

    def verify(self, layer, step, mixed_qkv, a, b, A_log, dt_bias, state):
        """``step.nmax`` rows per request (root + drafts), the window cut at the chunk end; returns [T, H, V] with
        zeros on rows past a window (their drafts are never accepted). E is not written."""
        T = mixed_qkv.shape[0]
        out = mixed_qkv.new_zeros(T, self.H, self.V)
        if step is None or T == 0:
            return out
        chunk_rows2.step(
            cache=self.layer(layer),
            scratch=self.scratch2,
            state=state,
            slots=(step.ring, step.state),
            rows=step.rows,
            counts=step.counts,
            chunks=step.chunks,
            mixed_qkv=mixed_qkv,
            a=a.reshape(T, -1).contiguous(),
            b=b.reshape(T, -1).contiguous(),
            A_log=A_log,
            dt_bias=dt_bias,
            out=out,
            nmax=step.nmax,
            num_k_heads=self.Hg,
            head_k_dim=self.K,
            kkt=self.kkt,
        )
        return out

    def extend(
        self,
        layer,
        plan,
        q,
        k,
        v,
        g,
        beta,
        a,
        b,
        A_log,
        dt_bias,
        state,
        cache_indices,
        track_state=None,
    ):
        """q, k, v [1, T, heads, dim] (post-conv, not normalized), g / beta [1, T, H] (fused_gdn_gating), a / b
        [T, H] raw gates; ``state`` the FP32 pool read and updated at ``cache_indices`` [batch]. Returns
        [1, T, H, V]. ``track_state`` [batch, H, V, K] receives the requested radix track states."""
        from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd

        T = q.shape[1]
        out = v.new_empty(1, T, self.H, self.V)
        if plan is None or T == 0:
            return out
        cache = self.layer(layer)
        if plan.fla_batch is not None:
            prenorm = plan.open_ring is not None
            fq, fk = q, k
            if prenorm:
                # Cached open-chunk keys are stored l2-normed: normalize the new rows the same way (l2norm_fwd is the
                # kernel's in-kernel normalization) and turn it off in the kernel.
                fq, fk = l2norm_fwd(q.contiguous()), l2norm_fwd(k.contiguous())
            if plan.fla_identity:
                fv, fg, fb = v, g, beta
            else:
                tf = plan.new_dst.numel() + (
                    0 if plan.open_dst is None else plan.open_dst.numel()
                )

                def gather(x, ring=None):
                    y = x.new_zeros(1, tf, *x.shape[2:])
                    y[0, plan.new_dst] = x[0, plan.new_src]
                    if ring is not None and plan.open_ring is not None:
                        y[0, plan.open_dst] = ring.flatten(0, 1)[plan.open_ring].to(
                            x.dtype
                        )
                    return y

                fq, fk = gather(fq), gather(fk, cache.k)
                fv, fg, fb = (
                    gather(v, cache.v),
                    gather(g, cache.g),
                    gather(beta, cache.beta),
                )
            fla_track = None
            if track_state is not None:
                fla_track = track_state.new_zeros(
                    plan.fla_batch.numel(), *track_state.shape[1:]
                )
            slots = cache_indices.index_select(0, plan.fla_batch)
            o = chunk_gated_delta_rule(
                q=fq,
                k=fk,
                v=fv,
                g=fg,
                beta=fb,
                initial_state=state,
                initial_state_indices=slots,
                cu_seqlens=plan.fla_cu,
                head_first=False,
                use_qk_l2norm_in_kernel=not prenorm,
                inplace_update=True,
                fixed_output_tile=True,
                track_state=fla_track,
                track_chunk_idx=None if fla_track is None else plan.fla_track_idx,
            )[0]
            if plan.fla_identity:
                out = o
            else:
                out[0, plan.new_src] = o[0, plan.new_dst]
            if track_state is not None:
                for row, batch_row in plan.final_track:
                    fla_track[row] = state[cache_indices[batch_row]]
                track_state[plan.fla_batch] = fla_track
        if plan.tails:
            # The row engine normalizes q and k itself: it takes the raw rows.
            mixed = torch.cat(
                (q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)), dim=-1
            )
            for t in plan.tails:
                o = mixed.new_empty(t.ring.numel() * MAX_STEP_ROWS, self.H, self.V)
                chunk_rows2.step(
                    cache=cache,
                    scratch=self.scratch2,
                    state=state,
                    slots=(t.ring, t.state),
                    rows=t.rows,
                    counts=t.counts,
                    chunks=t.chunks,
                    mixed_qkv=mixed.index_select(0, t.src),
                    a=a.index_select(0, t.src).contiguous(),
                    b=b.index_select(0, t.src).contiguous(),
                    A_log=A_log,
                    dt_bias=dt_bias,
                    out=o,
                    nmax=MAX_STEP_ROWS,
                    num_k_heads=self.Hg,
                    head_k_dim=self.K,
                    kkt=self.kkt,
                )
                out[0, t.out_dst] = o[t.out_src]
        return out

    # ------------------------------------------------------------------ all layers

    def close_to(self, *, state, ring, slots, dst):
        """For rows with dst >= 0, whose open chunk has all 64 rows cached and its close still deferred: the
        chunk's end state into state[:, dst] for every layer (E unchanged). ``state`` [layers, slots, H, V, K]."""
        chunk_rows2.close_to(
            cache=self.rows,
            state=state,
            slots=(ring.to(torch.int32), slots.to(torch.int32)),
            dst_slots=dst,
            num_layers=self.num_layers,
        )
