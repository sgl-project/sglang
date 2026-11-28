"""Token-parallel prefill for DeepSeek-V4.1 on ROCm gfx950 (opt-in).

Under pure TP every rank repeats the per-token work of a layer for all T prefill
rows: the Mega-mHC seams, the input projections (wqkv_a, the compressors' wkv /
wkv_gate, the indexer weights_proj), the router, top-k and the MoE-input quant,
and the Engram wkv GEMM. Token-parallel prefill keeps 1/tp of the
[T, hc_mult, hidden] residual on each rank and runs that work on the local rows.
Attention heads and MoE experts stay TP-sharded and run on all rows:

- attention input: wqkv_a on the local rows; the q latent is normed and put on
  the fp8 grid exactly as the off path does, then the fp8 q latent (+ ue8m0
  scales) and the bf16 kv latent are all-gathered bit-exactly. The compressors'
  projections and the indexer head weights run on the local rows and their
  narrow outputs are all-gathered; the full [T, hidden] attention input is never
  gathered.
- wo_b output: INT4 reduce-scatter (AITER #7444 at 3c77ea12e, side-loaded).
- MoE input: the attention-to-FFN Mega-mHC seam emits MXFP8 on the local rows
  (the bits fused_moe would compute); router and top-k run on the local
  rows; the MXFP8 rows, scales and top-k are all-gathered bit-exactly; experts
  and the fused shared expert run on all rows; INT4 reduce-scatter after.
- Engram: the table lookup runs on all rows (as off), the wkv GEMM and the gate
  on the local rows.
- exit: under decoder SWA bounded replay the region ends at late_layer_start.
  The source layer's compressed KV is written from a gathered projection, and
  only the tail rows of the mHC state are gathered.

Only the two reduce-scatters per layer are lossy (INT4); every gather is exact.
Scope: eager EXTEND batches of at least SGLANG_DSV4_TOKEN_PARALLEL_MIN_TOKENS
rows on the plain HIP layer loop with Mega-mHC prefill. Decode, verify, draft,
monokernel, asm layers and prefill CUDA graphs never enter.

Torch and the standard library are the only module-level imports, so the layout
and collective logic can be tested on CPU without SGLang.
"""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

LOG_PREFIX = "[dsv4-token-parallel]"
# #7444's reduce-scatter takes whole 32 KiB tiles per rank: 16 rows of 5120 bf16.
ROW_ALIGN = 16
# Byte alignment of every piece in a packed gather row (AITER's custom all-gather
# copies 16-byte vectors).
PACK_ALIGN = 16

_ACTIVE: Optional[TokenParallelState] = None
_DISABLED_REASON: Optional[str] = None
_LOGGED: set = set()


def _log_once(key: str, msg: str, *args, rank0: bool = True) -> None:
    if key in _LOGGED:
        return
    _LOGGED.add(key)
    if rank0 and _rank() != 0:
        return
    text = msg % args if args else msg
    logger.info("%s %s", LOG_PREFIX, text)


def _rank() -> int:
    try:
        import torch.distributed as dist

        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    except Exception:  # noqa: BLE001
        return 0


# ---------------------------------------------------------------------------
# Row layout (pure)
# ---------------------------------------------------------------------------


class RowLayout:
    """Which rows of a [T, ...] extend this rank keeps.

    T is padded to a multiple of world * row_align; rank r keeps rows
    [r * local, (r + 1) * local). Rows past T exist only on the last rank(s)
    and are zero on entry; every consumer ignores them.
    """

    def __init__(
        self, num_tokens: int, world: int, rank: int, row_align: int = ROW_ALIGN
    ):
        if num_tokens <= 0 or world <= 0 or not 0 <= rank < world:
            raise ValueError((num_tokens, world, rank))
        unit = world * row_align
        self.num_tokens = num_tokens
        self.world = world
        self.rank = rank
        self.padded = -(-num_tokens // unit) * unit
        self.local = self.padded // world
        self.lo = rank * self.local
        self.hi = min(self.lo + self.local, num_tokens)

    def rank_range(self, r: int) -> Tuple[int, int]:
        lo = r * self.local
        return lo, min(lo + self.local, self.num_tokens)

    def real_rows(self) -> int:
        return max(0, self.hi - self.lo)

    def local_rows(self, x: torch.Tensor) -> torch.Tensor:
        """Replicated [T or padded, ...] -> this rank's [local, ...] rows (zero padding)."""
        n = x.shape[0]
        if self.lo + self.local <= n:
            return x[self.lo : self.lo + self.local]
        out = x.new_zeros((self.local, *x.shape[1:]))
        real = max(0, min(n, self.lo + self.local) - self.lo)
        if real:
            out[:real].copy_(x[self.lo : self.lo + real])
        return out

    def selection(self, global_rows: Sequence[int]) -> Tuple[List[List[int]], int]:
        """Split sorted global row indices by owner: per-rank local indices and the
        largest per-rank count."""
        per_rank: List[List[int]] = [[] for _ in range(self.world)]
        prev = -1
        for g in global_rows:
            if g <= prev or not 0 <= g < self.num_tokens:
                raise ValueError(f"rows must be sorted, unique and < {self.num_tokens}")
            prev = g
            r = g // self.local
            per_rank[r].append(g - r * self.local)
        return per_rank, max((len(p) for p in per_rank), default=0)


def tail_global_rows(extend_lens: Sequence[int], tail_lens: Sequence[int]) -> List[int]:
    """Row indices (extend order) of each request's last tail_lens[b] rows."""
    if len(extend_lens) != len(tail_lens):
        raise ValueError((len(extend_lens), len(tail_lens)))
    rows: List[int] = []
    start = 0
    for n, t in zip(extend_lens, tail_lens):
        if not 0 <= t <= n:
            raise ValueError((n, t))
        rows.extend(range(start + n - t, start + n))
        start += n
    return rows


# ---------------------------------------------------------------------------
# Packed rows (pure): several per-row tensors in one byte row
# ---------------------------------------------------------------------------


class RowPacker:
    """Lays out per-row pieces (name, per-row shape, dtype) in one uint8 row with
    PACK_ALIGN-aligned offsets, so one exact all-gather moves all of them."""

    def __init__(self, specs: Sequence[Tuple[str, Tuple[int, ...], torch.dtype]]):
        self.specs = list(specs)
        self.offsets: Dict[str, Tuple[int, int]] = {}
        off = 0
        for name, shape, dtype in self.specs:
            nbytes = (
                int(torch.Size(shape).numel())
                * torch.empty((), dtype=dtype).element_size()
            )
            off = -(-off // PACK_ALIGN) * PACK_ALIGN
            self.offsets[name] = (off, nbytes)
            off += nbytes
        self.row_bytes = -(-off // PACK_ALIGN) * PACK_ALIGN

    def pack(
        self, pieces: Dict[str, torch.Tensor], rows: Optional[int] = None
    ) -> torch.Tensor:
        first = next(iter(pieces.values()))
        n = first.shape[0] if rows is None else rows
        buf = torch.zeros((n, self.row_bytes), dtype=torch.uint8, device=first.device)
        for name, shape, dtype in self.specs:
            t = pieces[name]
            if t.dtype != dtype or tuple(t.shape[1:]) != tuple(shape):
                raise ValueError(
                    f"{name}: {tuple(t.shape)} {t.dtype}, want (*, {shape}) {dtype}"
                )
            off, nbytes = self.offsets[name]
            m = t.shape[0]
            if m:
                buf[:m, off : off + nbytes].copy_(_bytes_view(t.reshape(m, -1)))
        return buf

    def unpack(self, buf: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Views into buf (no copy); a piece is contiguous only when it fills the row."""
        out = {}
        n = buf.shape[0]
        for name, shape, dtype in self.specs:
            off, nbytes = self.offsets[name]
            flat = buf[:, off : off + nbytes].view(dtype)
            out[name] = (
                flat.view(n, *shape) if len(shape) == 1 else flat.unflatten(1, shape)
            )
        return out


def _bytes_view(t: torch.Tensor) -> torch.Tensor:
    """[n, k] -> its raw bytes [n, k * itemsize] (a view when the last dim is dense)."""
    if t.stride(-1) != 1:
        t = t.contiguous()
    return t.view(torch.uint8)


def gather_view(t: torch.Tensor) -> torch.Tensor:
    """A dtype the exact all-gather moves (fp32 / fp16 / bf16): other dtypes travel
    as bf16 pairs of their bytes."""
    if t.dtype in (torch.float32, torch.float16, torch.bfloat16):
        return t
    b = _bytes_view(t.reshape(t.shape[0], -1))
    if b.shape[1] % 2:
        raise ValueError(f"odd row bytes {b.shape[1]}")
    return b.view(torch.bfloat16)


# ---------------------------------------------------------------------------
# Collectives
# ---------------------------------------------------------------------------


class Collectives:
    """The two primitives token-parallel prefill needs. all_gather must be exact;
    reduce_scatter takes [world * local, H] bf16 partials and returns this rank's
    [local, H] block of the sum."""

    world: int
    rank: int
    name: str = "abstract"

    def all_gather(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        raise NotImplementedError

    def reduce_scatter(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        raise NotImplementedError


class SglangCollectives(Collectives):
    """SGLang's TP group: GroupCoordinator.all_gather_into_tensor (AITER custom IPC
    all-gather when it fits, else RCCL; exact either way) and, for the
    reduce-scatter, AITER #7444's INT4 FlyDSL engine."""

    name = "fly7444"

    def __init__(self, group):
        self.group = group
        self.world = group.world_size
        self.rank = group.rank_in_group
        self._engine = _fly7444_engine(group)

    def all_gather(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        self.group.all_gather_into_tensor(
            gather_view(out), gather_view(inp.contiguous())
        )

    def reduce_scatter(self, out: torch.Tensor, inp: torch.Tensor) -> None:
        self._engine.reduce_scatter(inp, out)


_FLY7444 = None


def _fly7444_engine(group):
    """One #7444 QuickAllReduceInt4 per process for the reduce-scatters, separate
    from the INT4 all-reduce engine. Built on first use (a collective over the
    TP CPU group); the first reduce_scatter runs compile_sp (also a collective)."""
    global _FLY7444
    if _FLY7444 is None:
        import inspect

        from sglang.srt.models.deepseek_common.amd.fly7444.quick_allreduce_int4 import (
            QuickAllReduceInt4,
        )

        kw = {}
        if "inbox_memory" in inspect.signature(QuickAllReduceInt4).parameters:
            kw["inbox_memory"] = "uncached"
        t0 = time.time()
        _FLY7444 = QuickAllReduceInt4(
            group=group.cpu_group,
            device=torch.device("cuda", torch.cuda.current_device()),
            rank=group.rank_in_group,
            world_size=group.world_size,
            **kw,
        )
        _log_once(
            "fly7444",
            "INT4 reduce-scatter: AITER #7444 3c77ea12e engine built in %.1f s (inbox %s)",
            time.time() - t0,
            kw.get("inbox_memory", "default"),
        )
    return _FLY7444


# ---------------------------------------------------------------------------
# Per-forward state
# ---------------------------------------------------------------------------


class TokenParallelState:
    """The layout, the collectives and the per-layer hand-offs of one forward."""

    def __init__(
        self,
        num_tokens: int,
        input_ids: Optional[torch.Tensor],
        comm: Collectives,
        *,
        end_layer: int,
    ):
        self.comm = comm
        self.layout = RowLayout(num_tokens, comm.world, comm.rank)
        self.num_tokens = num_tokens
        self.padded = self.layout.padded
        self.local = self.layout.local
        self.lo, self.hi = self.layout.lo, self.layout.hi
        self.end_layer = end_layer
        self.forward_batch = None
        self.input_ids = input_ids
        self._ids_local = None
        self._pending: Dict[Tuple[int, str], Any] = {}
        self.bytes_gathered = 0
        self.rs_calls = 0
        self.rs_padded_calls = 0

    # -- rows ------------------------------------------------------------------

    def scatter(self, x: torch.Tensor) -> torch.Tensor:
        return self.layout.local_rows(x)

    def local_input_ids(self) -> Optional[torch.Tensor]:
        if self.input_ids is None:
            return None
        if self._ids_local is None:
            self._ids_local = self.layout.local_rows(self.input_ids)
        return self._ids_local

    # -- collectives -------------------------------------------------------------

    def gather(self, local: torch.Tensor) -> torch.Tensor:
        """[local, ...] on every rank -> [padded, ...], bit-exact."""
        if local.shape[0] != self.local:
            raise ValueError(f"gather expects {self.local} rows, got {local.shape[0]}")
        out = local.new_empty((self.padded, *local.shape[1:]))
        self.comm.all_gather(out, local.contiguous())
        self.bytes_gathered += out.numel() * out.element_size()
        return out

    def gather_packed(
        self, packer: RowPacker, pieces: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return packer.unpack(self.gather(packer.pack(pieces, rows=self.local)))

    def reduce_scatter(self, x: torch.Tensor) -> torch.Tensor:
        """Per-rank partial sums [T or padded, H] -> this rank's summed [local, H]."""
        if x.shape[0] != self.padded:
            if x.shape[0] != self.num_tokens:
                raise ValueError(f"reduce_scatter rows {x.shape[0]}")
            pad = x.new_zeros((self.padded, *x.shape[1:]))
            pad[: self.num_tokens].copy_(x)
            x = pad
            self.rs_padded_calls += 1
        x = x.contiguous()
        out = x.new_empty((self.local, *x.shape[1:]))
        self.comm.reduce_scatter(out, x)
        self.rs_calls += 1
        return out

    def gather_selected(
        self, pieces: Dict[str, torch.Tensor], global_rows: Sequence[int]
    ) -> Dict[str, torch.Tensor]:
        """The listed global rows (sorted) of per-rank [local, ...] pieces, on every
        rank, in row order; bit-exact. Moves max-per-rank-count rows per rank."""
        per_rank, cmax = self.layout.selection(global_rows)
        if cmax == 0:
            return {k: v.new_empty((0, *v.shape[1:])) for k, v in pieces.items()}
        packer = RowPacker(
            [(k, tuple(v.shape[1:]), v.dtype) for k, v in pieces.items()]
        )
        mine = per_rank[self.comm.rank]
        dev = next(iter(pieces.values())).device
        idx = torch.tensor(mine, dtype=torch.long, device=dev)
        local = packer.pack(
            {k: v.index_select(0, idx) for k, v in pieces.items()}, rows=cmax
        )
        full = local.new_empty((self.comm.world * cmax, packer.row_bytes))
        self.comm.all_gather(full, local)
        self.bytes_gathered += full.numel()
        keep = [
            r * cmax + j
            for r in range(self.comm.world)
            for j in range(len(per_rank[r]))
        ]
        sel = full.index_select(0, torch.tensor(keep, dtype=torch.long, device=dev))
        return {k: v.contiguous() for k, v in packer.unpack(sel).items()}

    # -- hand-offs between the attention entry and the backend ------------------

    def put(self, module, key: str, value) -> None:
        self._pending[(id(module), key)] = value

    def take(self, module, key: str):
        return self._pending.pop((id(module), key), None)

    def drop(self, module) -> None:
        for k in [k for k in self._pending if k[0] == id(module)]:
            del self._pending[k]


def token_parallel_state(forward_batch) -> Optional[TokenParallelState]:
    state = _ACTIVE
    return state if state is not None and state.forward_batch is forward_batch else None


def active_state() -> Optional[TokenParallelState]:
    """The state of the forward in flight (for callees without the forward batch)."""
    return _ACTIVE


def exit_token_parallel() -> None:
    global _ACTIVE
    _ACTIVE = None


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


class TokenParallelConfig(NamedTuple):
    """The switches, read once per model."""

    enabled: bool
    min_tokens: int
    dspark: bool


def _config(model) -> TokenParallelConfig:
    config = model.__dict__.get("_dsv4_tp_config")
    if config is None:
        from sglang.srt.environ import envs

        config = TokenParallelConfig(
            enabled=envs.SGLANG_DSV4_TOKEN_PARALLEL_PREFILL.get(),
            min_tokens=envs.SGLANG_DSV4_TOKEN_PARALLEL_MIN_TOKENS.get(),
            dspark=envs.SGLANG_DSV4_TOKEN_PARALLEL_DSPARK.get(),
        )
        model.__dict__["_dsv4_tp_config"] = config
    return config


def _static_ineligible(model) -> Optional[str]:
    """Model-level reasons token-parallel prefill cannot run (checked once per model)."""
    cached = model.__dict__.get("_dsv4_tp_static")
    if cached is not None:
        return cached or None
    reason = ""
    try:
        reason = _static_check(model) or ""
    except Exception as e:  # noqa: BLE001
        reason = f"static check raised {type(e).__name__}: {e}"
    model.__dict__["_dsv4_tp_static"] = reason
    return reason or None


def _static_check(model) -> Optional[str]:
    from sglang.srt.utils import is_gfx95_supported, is_hip

    if not (is_hip() and is_gfx95_supported()):
        return "needs ROCm gfx950"
    if model.config.model_type not in ("deepseek_v41", "deepseek_v41_text"):
        return f"model_type {model.config.model_type}"
    if not model.hc_pre_from_prev_sublayer:
        return "needs the hc_pre_from_prev layer loop"
    if model.late_layer_start is None or not model.tail_writes_source_kv:
        return "needs decoder SWA bounded replay (--enable-decoder-swa-bounded-replay)"
    if model.pp_group.world_size != 1:
        return "pipeline parallel"
    if model.late_layer_start <= model.start_layer:
        return "empty token-parallel region"
    for i in range(model.start_layer, model.late_layer_start + 1):
        layer = model.layers[i]
        attn = layer.self_attn
        if not attn.fuse_wqa_wkv:
            return f"layer {i}: no fused wqkv_a"
        if attn.use_mega_attention:
            return f"layer {i}: mega attention"
        if attn.use_fused_qk_norm_rope:
            return f"layer {i}: fused qk norm rope (q_head_norm)"
        ratio = attn.compress_ratio
        if i < model.late_layer_start and ratio not in (0, 2):
            return f"layer {i}: compress ratio {ratio}"
        if i == model.late_layer_start:
            break
        mlp = layer.mlp
        if mlp.shared_experts is not None:
            return (
                f"layer {i}: shared expert not fused (--enforce-shared-experts-fusion)"
            )
        if mlp._shared_expert_tp1 or mlp._fuse_shared_experts_inside_sbo:
            return f"layer {i}: shared expert layout"
        if mlp.experts.reduce_results:
            return f"layer {i}: experts reduce their own output"
        if layer.attn_hc is None or layer.ffn_hc is None:
            return f"layer {i}: no mHC sublayers"
    src = model.layers[model.late_layer_start].self_attn
    if src.compressor is None:
        return "the tail source layer has no compressor"
    return None


def _dynamic_ineligible(
    model,
    forward_batch,
    num_tokens: int,
    *,
    tail,
    tail_in_graph,
    use_tail_graph,
    capture_dspark,
    config: TokenParallelConfig,
) -> Optional[str]:
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
    from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.model_executor.runner import get_is_capture_mode
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
        is_in_breakable_cuda_graph,
    )
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_hip as _hip
    from sglang.srt.runtime_context import get_forward, get_parallel

    if forward_batch.forward_mode != ForwardMode.EXTEND:
        return "not a plain extend"
    par = get_parallel()
    if par.tp_size < 2:
        return "tp 1"
    if num_tokens < max(config.min_tokens, par.tp_size**2):
        return "short"
    if (
        par.attn_tp_size != par.tp_size
        or par.attn_dp_size != 1
        or par.attn_cp_size != 1
    ):
        return "DP / CP attention"
    if not get_moe_a2a_backend().is_none():
        return "a2a MoE"
    if is_cp_active(forward_batch):
        return "CP"
    if (
        get_is_capture_mode()
        or is_in_breakable_cuda_graph()
        or torch.cuda.is_current_stream_capturing()
    ):
        return "graph"
    if get_forward().sp_active:
        return "layernorm SP"
    if tail is None or tail_in_graph:
        return "no eager decoder SWA tail"
    if tail.pad_rows or tail.cp_metadata is not None:
        return "CP tail"
    if capture_dspark and not config.dspark:
        return "DSpark capture without SGLANG_DSV4_TOKEN_PARALLEL_DSPARK"
    if _hip.fused_boundary_this_step(forward_batch):
        return "fused HIP boundary step (needs SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=1)"
    if not envs.SGLANG_OPT_DSV41_MEGA_MHC_PREFILL.get():
        return "needs Mega-mHC prefill"
    if is_unified_kv_triton() or envs.SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE.get():
        return "unified / bf16-source KV store"
    if envs.SGLANG_OPT_MOE_QUANT_ONCE.get():
        return "SGLANG_OPT_MOE_QUANT_ONCE (the MoE takes a pre-quantized bf16 input)"
    if forward_batch.extend_seq_lens_cpu is None or forward_batch.seq_lens_cpu is None:
        return "no CPU lengths"
    if forward_batch.encoder_swa_replay:
        return "encoder SWA replay"
    if forward_batch.contains_image_inputs():
        return "image inputs (vision top-k)"
    from sglang.srt.layers.moe.utils import get_moe_runner_backend

    if get_moe_runner_backend().is_gluon():
        return "Gluon MoE runner"
    from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
        DeepseekV4HipRadixBackend,
    )

    backend = get_attn_backend()
    if not isinstance(backend, DeepseekV4HipRadixBackend):
        return f"attention backend {type(backend).__name__}"
    for i in range(model.start_layer, model.late_layer_start):
        idx = model.layers[i].self_attn.indexer
        if (
            isinstance(idx, DeepseekV41Indexer)
            and 0 < num_tokens <= idx.weights_proj_hip_max_tokens
        ):
            return "indexer GEMV row count"
    return None


def enter_token_parallel(
    model,
    forward_batch,
    hidden_states: torch.Tensor,
    input_ids: Optional[torch.Tensor],
    *,
    tail,
    tail_in_graph: bool,
    use_tail_graph: bool,
    capture_dspark: bool,
) -> Optional[TokenParallelState]:
    """Called at the start of every hc_pre_from_prev layer loop. Clears any stale
    state, then attaches a TokenParallelState when this prefill can run
    token-parallel."""
    global _ACTIVE
    _ACTIVE = None
    config = _config(model)
    if not config.enabled or _DISABLED_REASON is not None:
        return None
    num_tokens = hidden_states.shape[0]
    reason = _static_ineligible(model)
    if reason is not None:
        _log_once("static:" + reason, "off for this model: %s", reason)
        return None
    reason = _dynamic_ineligible(
        model,
        forward_batch,
        num_tokens,
        tail=tail,
        tail_in_graph=tail_in_graph,
        use_tail_graph=use_tail_graph,
        capture_dspark=capture_dspark,
        config=config,
    )
    if reason is not None:
        _log_once(
            "skip:" + reason, "this forward stays replicated (%s); logged once", reason
        )
        return None
    comm = _collectives()
    if comm is None:
        return None
    state = TokenParallelState(
        num_tokens, input_ids, comm, end_layer=model.late_layer_start
    )
    state.forward_batch = forward_batch
    _ACTIVE = state
    _log_once(
        "entered",
        "token-parallel prefill entered: tokens=%d padded=%d local=%d layers=[%d,%d) rs=%s",
        num_tokens,
        state.padded,
        state.local,
        model.start_layer,
        model.late_layer_start,
        comm.name,
    )
    return state


_COMM: Optional[Collectives] = None


def _collectives() -> Optional[Collectives]:
    """The process's collectives, built on the first eligible forward. The ranks
    agree on the outcome once, over the TP CPU group, so no rank waits in a
    collective the others skip."""
    global _COMM, _DISABLED_REASON
    if _COMM is not None:
        return _COMM
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_parallel

    group = get_parallel().tp_group
    comm, ok = None, 1.0
    try:
        comm = SglangCollectives(group)
    except Exception:  # noqa: BLE001
        ok = 0.0
        logger.exception("%s reduce-scatter engine failed to build", LOG_PREFIX)
    flag = torch.tensor([ok])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group.cpu_group)
    if flag.item() < 1.0:
        _DISABLED_REASON = "reduce-scatter engine unavailable on some rank"
        logger.warning("%s DISABLED for this process: %s", LOG_PREFIX, _DISABLED_REASON)
        return None
    _COMM = comm
    return _COMM


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def _compressor_runs(attn, forward_batch, backend) -> bool:
    if attn.compressor is None or forward_batch.encoder_swa_replay:
        return False
    return attn.layer_id != backend.tail_source_kv_layer_id


def _compressor_projection_local(comp, x: torch.Tensor):
    """The off path's projection call for this compressor, on the local rows."""
    if comp.compress_ratio == 1:
        return ("wkv", comp.wkv(x))
    if comp.use_fused_gate:
        return ("fused", comp.project_fused(x))
    kv, score = comp.project(x)
    return ("pair", (kv, score))


def _gather_projection(state: TokenParallelState, proj):
    kind, value = proj
    if kind == "pair":
        kv, score = value
        packer = RowPacker(
            [
                ("kv", tuple(kv.shape[1:]), kv.dtype),
                ("score", tuple(score.shape[1:]), score.dtype),
            ]
        )
        got = state.gather_packed(packer, {"kv": kv, "score": score})
        n = state.num_tokens
        return kind, (got["kv"][:n].contiguous(), got["score"][:n].contiguous())
    return kind, state.gather(value)[: state.num_tokens]


def _indexer_head_weights_local(indexer, x_local: torch.Tensor) -> torch.Tensor:
    """What low_ratio_backend_hip._indexer_head_weights returns for a T-row batch
    (T is above the GEMV's row cap), on the local rows."""
    return indexer.head_weights(x_local).contiguous()


def attention_inputs(
    attn, x, x_quant, positions, forward_batch, state: TokenParallelState
):
    """MQALayer.forward's entry under token-parallel prefill. Runs every projection
    of the attention input on the local rows, gathers their narrow outputs, and
    returns a zero-stride [T, hidden] stand-in for x: after this point every reader
    of x takes its rows from the hand-offs, so the stand-in only carries T, the
    dtype and the device."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    if x.shape[0] != state.local:
        raise RuntimeError(
            f"attention input has {x.shape[0]} rows, expected {state.local}"
        )
    x_linear = x_quant if x_quant is not None else x
    qkv_a, _ = attn.wqkv_a(x_linear)
    q_lora = qkv_a[..., : attn.q_lora_rank]
    kv = qkv_a[..., attn.q_lora_rank :]
    # The off path's own q norm (gfx950 32-block route: rmsnorm_fake_quant_fp8, one
    # program per row), so the gathered operand is the one wq_b and the indexer read.
    _, q_for_wqb = attn._normalize_q_lora(q_lora)
    state.put(attn, "latent", _gather_latent(state, q_for_wqb, kv))
    del qkv_a, q_lora

    backend = get_attn_backend()
    if attn.compress_ratio in (1, 2):
        comp = attn.compressor
        if comp is not None and _compressor_runs(attn, forward_batch, backend):
            state.put(
                comp,
                "proj",
                _gather_projection(state, _compressor_projection_local(comp, x)),
            )
        if attn.indexer is not None:
            w = _indexer_head_weights_local(attn.indexer, x)
            state.put(attn.indexer, "weights", state.gather(w)[: state.num_tokens])
    return x[:1].expand(state.num_tokens, -1)


def _gather_latent(state: TokenParallelState, q_for_wqb, kv: torch.Tensor):
    """Gather the wq_b operand and the kv latent, bit-exactly. The q operand travels
    as fp8 codes + ue8m0 scales when the off path built it that way (Mxfp8Activation),
    else in its bf16 form."""
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
        Fp8GridActivation,
        Mxfp8Activation,
    )

    n = state.num_tokens
    if isinstance(q_for_wqb, Mxfp8Activation):
        q, scale = q_for_wqb.q, q_for_wqb.scale
        packer = RowPacker(
            [
                ("q", tuple(q.shape[1:]), q.dtype),
                ("kv", tuple(kv.shape[1:]), kv.dtype),
                ("scale", tuple(scale.shape[1:]), scale.dtype),
            ]
        )
        got = state.gather_packed(packer, {"q": q, "kv": kv, "scale": scale})
        _log_once(
            "latent:fp8",
            "latent gather: fp8 q + ue8m0 scales + bf16 kv, %d B a row",
            packer.row_bytes,
        )
        operand = Mxfp8Activation(
            got["q"][:n].contiguous(), got["scale"][:n].contiguous()
        )
        return operand, got["kv"][:n]
    wrap = None
    q = q_for_wqb
    if isinstance(q_for_wqb, Fp8GridActivation):
        wrap, q = Fp8GridActivation, q_for_wqb.x
    packer = RowPacker(
        [("q", tuple(q.shape[1:]), q.dtype), ("kv", tuple(kv.shape[1:]), kv.dtype)]
    )
    got = state.gather_packed(packer, {"q": q, "kv": kv})
    _log_once(
        "latent:bf16",
        "latent gather: bf16 q operand + bf16 kv, %d B a row",
        packer.row_bytes,
    )
    q_full = got["q"][:n].contiguous()
    return (wrap(q_full) if wrap is not None else q_full), got["kv"][:n]


class _Served(torch.nn.Module):
    """Stands in for a compressor's wkv Linear for one call: returns the gathered rows."""

    def __init__(self, value: torch.Tensor, used: List[int]):
        super().__init__()
        self.value = value
        self.used = used

    def forward(self, x):
        self.used[0] += 1
        return self.value[: x.shape[0]]


@contextlib.contextmanager
def serve_projections(attn, state: TokenParallelState):
    """For one backend call, the compressor projection and the indexer head weights
    of attn come from the gathered hand-offs instead of the (stand-in) x."""
    from sglang.srt.layers.attention.dsv4 import low_ratio_backend_hip as lrb

    restore: List[Callable[[], None]] = []
    used = [0]
    comp = attn.compressor
    proj = state.take(comp, "proj") if comp is not None else None
    if proj is not None:
        kind, value = proj
        if kind == "wkv":
            original = comp.wkv
            comp.wkv = _Served(value, used)
            restore.append(lambda: setattr(comp, "wkv", original))
        elif kind == "fused":

            def _project_fused(x, _v=value):
                used[0] += 1
                return _v[: x.shape[0]]

            def _project(x, _v=value):
                used[0] += 1
                fused = _v[: x.shape[0]]
                d = fused.shape[-1] // 2
                return fused[..., :d], fused[..., d:]

            comp.project_fused = _project_fused
            comp.project = _project
            restore.append(
                lambda: (
                    comp.__dict__.pop("project_fused", None),
                    comp.__dict__.pop("project", None),
                )
            )
        else:
            kv, score = value

            def _project_pair(x, _k=kv, _s=score):
                used[0] += 1
                return _k[: x.shape[0]], _s[: x.shape[0]]

            comp.project = _project_pair
            restore.append(lambda: comp.__dict__.pop("project", None))
    indexer = attn.indexer
    weights = state.take(indexer, "weights") if indexer is not None else None
    if weights is not None:
        original_hw = lrb._indexer_head_weights

        def _head_weights(idx, x, _w=weights, _i=indexer, _orig=original_hw):
            if idx is not _i:
                return _orig(idx, x)
            if lrb._gemv_head_weight_rows(idx, x):
                raise RuntimeError(
                    "token-parallel indexer weights asked for a GEMV row count"
                )
            return _w[: x.shape[0]]

        lrb._indexer_head_weights = _head_weights
        restore.append(lambda: setattr(lrb, "_indexer_head_weights", original_hw))
    try:
        yield
    finally:
        for fn in reversed(restore):
            fn()
        if comp is not None:
            state.drop(comp)
        if indexer is not None:
            state.drop(indexer)
    # The backend reads x only through the shadowed calls; if it stopped using them it would
    # read the stand-in, so a projection that was served but never taken is a hard error.
    if proj is not None and used[0] == 0:
        raise RuntimeError(
            f"layer {attn.layer_id}: the backend did not take the gathered compressor projection"
        )


def forward_prepare(
    attn, x, positions, forward_batch, attn_backend, q_out, state: TokenParallelState
):
    """MQALayer._forward_prepare under token-parallel prefill: the HIP prefill branch
    (not unified, not CP) with the gathered latent in place of wqkv_a(x)."""
    from sglang.kernels.ops.attention.dsv4.opus_sparse_prefill_hip import OpusPackedQ
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
    from sglang.srt.models.deepseek_common.amd import deepseek_v4_hip as _hip

    latent = state.take(attn, "latent")
    if latent is None:
        raise RuntimeError(f"layer {attn.layer_id}: no gathered latent")
    q_for_wqb, kv = latent
    q, q_lora, fuse_q_rope = _hip.compute_q_b(
        attn,
        None,
        q_for_wqb,
        positions,
        q_out,
        False,
        False,
        forward_batch=forward_batch,
        attn_backend=attn_backend,
    )
    q_rope = (q.q_rope if isinstance(q, OpusPackedQ) else q) if fuse_q_rope else None
    get_token_to_kv_pool().set_swa_key_buffer_radix_fused_norm_rope(
        layer_id=attn.layer_id,
        swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch),
        kv=kv,
        kv_weight=attn.kv_norm.weight.data,
        eps=attn.eps,
        freqs_cis=attn.freqs_cis,
        positions=positions,
        **({} if q_rope is None else {"q": q_rope}),
    )
    if attn.compress_ratio in (1, 2) and (
        attn.compressor is not None or attn.indexer is not None
    ):
        with serve_projections(attn, state):
            attn_backend.forward_low_ratio_sources(
                layer=attn,
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
            )
    return q, None


def project_wo_b(attn, o, state: TokenParallelState) -> torch.Tensor:
    """wo_b's partial sums over all T rows, reduce-scattered back to the local rows."""
    o, _ = attn.wo_b(
        o.flatten(1) if isinstance(o, torch.Tensor) else o, skip_all_reduce=True
    )
    return state.reduce_scatter(o)


# ---------------------------------------------------------------------------
# One layer
# ---------------------------------------------------------------------------


def _mega_post_mxfp8(y, residual, pre, post_mix, comb, nxt):
    """deepseek_v4_mhc._hip_flydsl_mega_mhc with the next sublayer's input also
    written as MXFP8 (e4m3 codes + row-major ue8m0 [rows, hidden / 32])."""
    from aiter.ops.flydsl import flydsl_mega_mhc

    from sglang.srt.models.deepseek_v4_mhc import HcQuantized, HcState

    post_in = post_mix.unsqueeze(-1) if post_mix.ndim == 2 else post_mix
    residual_out = torch.empty_like(residual)
    updated, next_post, next_comb, normalized, next_pre = flydsl_mega_mhc(
        residual,
        nxt.fn,
        nxt.scale,
        nxt.base,
        nxt.cfg.rms_eps,
        nxt.cfg.eps,
        nxt.cfg.eps,
        2.0,
        nxt.cfg.sinkhorn_iters,
        pre,
        sublayer_out=y,
        post_layer_mix=post_in,
        comb_res_mix=comb,
        norm_weight=nxt.norm.weight,
        norm_eps=nxt.norm.variance_epsilon,
        residual_out=residual_out,
        out_dtype="mxfp8",
    )
    norm, q, e8 = normalized
    if next_post.ndim == 3:
        next_post = next_post.squeeze(-1)
    if next_pre.ndim == 3:
        next_pre = next_pre.squeeze(-1)
    return HcState(
        updated, pre, HcQuantized(norm, (q, e8)), (next_pre, next_post, next_comb)
    )


def _attn_post(hc, y, residual, *, next_boundary, precomputed):
    """deepseek_v4_mhc.run_attn_post for a plain tensor y; where it can, the
    chained Mega-mHC launch also writes the MoE input as MXFP8."""
    from sglang.srt.models import deepseek_v4_mhc as mhc

    nxt = next_boundary
    if (
        nxt is not None
        and nxt.norm_fusable
        and nxt.hc is not None
        and mhc._can_use_mega_mhc(hc.cfg)
        and y.shape[0] > 0
    ):
        pre, post_mix, comb = mhc._compute_triplet(hc, residual, None, precomputed)
        return _mega_post_mxfp8(y, residual, pre, post_mix, comb, nxt.hc)
    _log_once(
        "mxfp8:declined",
        "the attention-to-FFN seam cannot emit MXFP8 here; the MoE input is gathered in bf16",
    )
    return mhc.run_attn_post(
        hc,
        y,
        residual,
        stats_stream=None,
        next=next_boundary,
        world_size=1,
        precomputed=precomputed,
    )


def forward_layer(
    layer,
    *,
    positions: torch.Tensor,
    state,
    input_ids: Optional[torch.Tensor],
    forward_batch,
    input_ids_global: Optional[torch.Tensor],
    tp_state: TokenParallelState,
):
    """NonCudaDecoderMixin.forward_hc_pre_from_prev on the local rows (Mega-mHC
    prefill; no stats stream on extend)."""
    from sglang.srt.models import deepseek_v4_mhc as mhc

    layer._init_boundaries()
    local_boundary, next_boundary = (
        layer._extend_boundaries
        if mhc.is_local_extend(forward_batch)
        else (layer.local_boundary, layer.next_boundary)
    )

    state = mhc.mega_closed_seam(layer.attn_hc, state)
    residual = state.residual
    x = mhc.combine(layer.attn_hc, state)
    precomputed = state.stats
    state.release()
    del state
    with layer.self_attn.maybe_use_decode_attn_tp(forward_batch):
        y = layer.self_attn(
            x=x,
            positions=positions,
            forward_batch=forward_batch,
            x_quant=None,
            defer_all_reduce=False,
        )
    del x
    state = _attn_post(
        layer.attn_hc,
        y,
        residual,
        next_boundary=local_boundary,
        precomputed=precomputed,
    )
    del y

    residual = state.residual
    pair = state.input.pair if isinstance(state.input, mhc.HcQuantized) else None
    x = mhc.combine(layer.ffn_hc, state)
    precomputed = state.stats
    state.release()
    del state
    y = run_moe(layer.mlp, x, pair, forward_batch, tp_state)
    del x, pair
    return mhc.run_moe_post(
        layer.ffn_hc,
        y,
        residual,
        stats_stream=None,
        next=next_boundary,
        world_size=layer.mlp.tp_size,
        precomputed=precomputed,
    )


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------

_STAGED_MOE: Optional[Tuple[int, int, torch.Tensor]] = None
_STAGED_TAKEN = [False]


@contextlib.contextmanager
def staged_moe_input(rows: torch.Tensor, scale: torch.Tensor):
    """Offer the ue8m0 scales of the MXFP8 rows ``rows`` to the AITER runner."""
    global _STAGED_MOE
    _STAGED_MOE = (rows.data_ptr(), rows.shape[0], scale)
    _STAGED_TAKEN[0] = False
    try:
        yield
    finally:
        _STAGED_MOE = None


def take_staged_moe_input(hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
    """The staged scales for hidden_states, else None. Called by the AITER runner's
    standard pre-permute; returns None (and costs one global read) when nothing is
    staged, which is always the case outside token-parallel prefill."""
    s = _STAGED_MOE
    if s is None:
        return None
    ptr, rows, scale = s
    if hidden_states.data_ptr() != ptr or hidden_states.shape[0] != rows:
        raise RuntimeError(
            "token-parallel MXFP8 MoE input was copied before the AITER runner"
        )
    _STAGED_TAKEN[0] = True
    if scale.dtype == torch.uint8:
        scale = scale.view(torch.float8_e8m0fnu)
    return scale


def _routed_scale_needed(moe) -> bool:
    """forward_normal's condition for final_hidden_states *= routed_scaling_factor."""
    from sglang.srt.models import deepseek_v2 as dv2

    need = (
        not dv2._is_cuda
        and not dv2._is_musa
        and not dv2._is_xpu
        and not (dv2._use_aiter and moe.routed_scaling_factor_fused_in_topk)
    )
    return need or isinstance(moe.experts.quant_method, dv2.KTEPWrapperMethod)


def run_moe(
    moe, x_local: torch.Tensor, pair, forward_batch, state: TokenParallelState
) -> torch.Tensor:
    """The MoE block of this rank's rows: route the local rows, gather the expert
    input and the top-k bit-exactly, run the experts (and the fused shared expert)
    on all rows without their all-reduce, and reduce-scatter the partial sums."""
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
    from sglang.srt.models.deepseek_common.amd import deepseek_v2_hip_moe as _hip_moe
    from sglang.srt.runtime_context import get_forward

    if x_local.shape[0] != state.local:
        raise RuntimeError(
            f"MoE input has {x_local.shape[0]} rows, expected {state.local}"
        )
    router_logits, partials = _hip_moe.forward_gate(moe, x_local, None, fused_gate=True)
    kw = {"input_ids": state.local_input_ids()} if moe.is_hash else {}
    if partials is not None:
        kw["router_logits_partials"] = partials
    topk = moe.topk(x_local, router_logits, **kw)
    if not TopKOutputChecker.format_is_standard(topk):
        raise RuntimeError(
            f"token-parallel MoE needs a standard top-k output, got {topk.format}"
        )
    w, ids = topk.topk_weights, topk.topk_ids
    side_specs = [
        ("w", tuple(w.shape[1:]), w.dtype),
        ("ids", tuple(ids.shape[1:]), ids.dtype),
    ]
    if pair is not None:
        q_local, e8_local = pair
        hidden = state.gather(q_local)
        side = RowPacker(
            [("e8", tuple(e8_local.shape[1:]), e8_local.dtype), *side_specs]
        )
        got = state.gather_packed(side, {"e8": e8_local, "w": w, "ids": ids})
        scale = got["e8"].contiguous()
        _log_once(
            "moe:mxfp8",
            "MoE input gathered as MXFP8 (%d + %d B a row); fused_moe skips its input quant",
            q_local.shape[1],
            side.row_bytes,
        )
    else:
        hidden = state.gather(x_local)
        side = RowPacker(side_specs)
        got = state.gather_packed(side, {"w": w, "ids": ids})
        scale = None
        _log_once(
            "moe:bf16",
            "MoE input gathered in bf16 (%d B a row)",
            x_local.shape[1] * x_local.element_size(),
        )
    topk_full = StandardTopKOutput(
        got["w"].contiguous(),
        got["ids"].contiguous(),
        router_logits[:1].expand(state.padded, -1),
    )
    staged = (
        staged_moe_input(hidden, scale)
        if scale is not None
        else contextlib.nullcontext()
    )
    with (
        get_forward().scoped(mlp_reduce_scatter=True, defer_moe_finalize=False),
        staged,
    ):
        out = moe.experts(hidden, topk_full)
    if scale is not None and not _STAGED_TAKEN[0]:
        raise RuntimeError(
            "the AITER runner did not take the token-parallel MXFP8 MoE input"
        )
    if _routed_scale_needed(moe):
        out *= moe.routed_scaling_factor
    return state.reduce_scatter(out)


# ---------------------------------------------------------------------------
# Engram, DSpark capture, exit
# ---------------------------------------------------------------------------


def run_engram(
    engram,
    residual_local,
    hash_col,
    forward_batch,
    input_ids,
    state: TokenParallelState,
):
    """Engram.forward for the local rows: the lookup covers every row, as the off
    path (same indices, same collectives); the wkv GEMM and the gate run on the
    local rows only."""
    from sglang.srt.layers.engram import engram_gate

    emb = engram.embed_rows(hash_col, forward_batch, False)
    if residual_local.shape[0] == 0:
        return residual_local
    kv, _ = engram.wkv(state.layout.local_rows(emb).flatten(-2))
    image_ids = None
    if engram.image_token_id is not None and input_ids is not None:
        image_ids = state.layout.local_rows(input_ids)
    return engram_gate(
        residual_local,
        kv,
        engram.q_weight,
        engram.k_weight,
        engram.eps,
        engram.clamp_value,
        image_ids,
        engram.image_token_id,
    )


def stream_mean_rows(residual_local: torch.Tensor, mean_fn=None) -> torch.Tensor:
    """mhc_stream_mean of the local rows into a fresh [local, hidden] buffer."""
    if mean_fn is None:
        from sglang.kernels.ops.layernorm.mhc import mhc_stream_mean as mean_fn
    out = residual_local.new_empty(residual_local[:, 0].shape)
    mean_fn(residual_local, out=out)
    return out


def capture_dspark_aux(
    state: TokenParallelState, hc_state, tail, forward_batch, packer, mean_fn=None
) -> None:
    """A DSpark capture layer inside the token-parallel region: the stream mean on
    the local rows, then an exact gather of only the rows the draft reads (the tail
    rows under bounded replay, else all rows) into the packer's slot."""
    local = stream_mean_rows(hc_state.residual, mean_fn)
    if tail is not None:
        rows = tail_global_rows(
            forward_batch.extend_seq_lens_cpu, tail.extend_seq_lens_cpu
        )
        full = state.gather_selected({"aux": local}, rows)["aux"]
    else:
        full = state.gather(local)[: state.num_tokens]
    packer.slot(full).copy_(full)


def gather_tail_state(state: TokenParallelState, hc_state, rows: Sequence[int]):
    """HcState.take_rows of the full-row state, built from the local rows: the
    residual, pre, a normed input and the stats of the listed rows, bit-exactly."""
    from sglang.srt.models.deepseek_v4_mhc import HcNormed, HcPending, HcState

    if isinstance(hc_state.streams, HcPending):
        raise RuntimeError("token-parallel exit on a pending post")
    pieces = {"residual": hc_state.residual}
    if hc_state.pre is not None:
        pieces["pre"] = hc_state.pre
    if isinstance(hc_state.input, HcNormed):
        pieces["input"] = hc_state.input.rows
    if hc_state.stats is not None:
        for k, t in zip(("s0", "s1", "s2"), hc_state.stats):
            pieces[k] = t
    got = state.gather_selected(pieces, rows)
    stats = (got["s0"], got["s1"], got["s2"]) if "s0" in got else None
    normed = HcNormed(got["input"]) if "input" in got else None
    return HcState(got["residual"], got.get("pre"), normed, stats)


def write_source_kv_and_gather_tail(
    model, state: TokenParallelState, hc_state, tail, positions, forward_batch
):
    """The token-parallel region's exit at late_layer_start: write the source layer's
    compressed KV for every row (its compressor projection runs on the local rows and
    is gathered), then return the tail rows of the mHC state, gathered exactly, which
    is what state.take_rows(tail.rows) returns on the off path."""
    from sglang.srt.model_executor.forward_context import get_attn_backend
    from sglang.srt.models import deepseek_v4_mhc as mhc

    attn_backend = get_attn_backend()
    try:
        if model.tail_writes_source_kv:
            source = model.layers[model.late_layer_start]
            attn = source.self_attn
            x_local = mhc.combine(source.attn_hc, hc_state)
            proj = _compressor_projection_local(attn.compressor, x_local)
            state.put(attn.compressor, "proj", _gather_projection(state, proj))
            with serve_projections(attn, state):
                attn_backend.write_tail_source_kv(
                    layer=attn,
                    x=x_local[:1].expand(state.num_tokens, -1),
                    positions=positions,
                    forward_batch=forward_batch,
                )
            del x_local
        rows = tail_global_rows(
            forward_batch.extend_seq_lens_cpu, tail.extend_seq_lens_cpu
        )
        if len(rows) != tail.token_indices.shape[0]:
            raise RuntimeError(
                f"tail rows {len(rows)} != {tail.token_indices.shape[0]}"
            )
        if "tail_rows_checked" not in _LOGGED:
            # Once per process (one device sync): the CPU-built rows must be the tail's own.
            _LOGGED.add("tail_rows_checked")
            if tail.token_indices.tolist() != rows:
                raise RuntimeError(
                    "token-parallel tail rows differ from tail.token_indices"
                )
        narrowed = gather_tail_state(state, hc_state, rows)
        _log_once(
            "exit",
            "exit at layer %d: gathered %d tail rows of %d (%d gathered MiB this forward, %d reduce-scatters, %d padded)",
            model.late_layer_start,
            len(rows),
            state.num_tokens,
            state.bytes_gathered >> 20,
            state.rs_calls,
            state.rs_padded_calls,
        )
        return narrowed
    finally:
        exit_token_parallel()
