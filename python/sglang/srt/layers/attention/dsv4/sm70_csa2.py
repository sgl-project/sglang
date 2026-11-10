"""SM70 CSA2 orchestration: compress, pack, index, sparse decode vs the torch oracle.

Used by ``DeepseekV4AttnBackend`` for ratio-0/1/2 layers on SM70. Hopper/DeepGEMM stay off.
Replay flags are off: every query sees its exact 128-token window.

All per-sequence state lives in static, position-indexed device
buffers so the bs=1 decode path has no host sync, no Python-side sequence
state and a fixed launch shape (CUDA-graph capturable):

* ``swa_ring[lid]``   uint8 [W, 528]        slot = pos % W
* ``kv_rows[src]``    uint8 [cap_c, 288]    row  = pos // ratio
* ``index_rows[src]`` uint8 [cap_c, 68]     row  = pos // ratio
* ``pending_kv/score[lid]`` fp32 [512]      last token's ratio-2 projections
* ``topk[lid]``       int32 [1, index_topk] this step's selection (decode)
* ``cand_ids``        int32 [1, cand_blocks] this step's candidate blocks

Visibility is derived from the position on the device (``vis = (pos+1)//ratio``,
ring validity from ``(pos - slot) % W <= pos``), so a new sequence needs no
reset: prefill from position 0 overwrites everything it will read.

Ratio-2 decode is branch-free: every token pools ``(pending, current)`` into
row ``pos // 2``. At even ``pos`` that row is not yet visible and is rewritten
by the next token; at odd ``pos`` it is the completed pair. ``pending`` always
holds the previous token.

Prefill (chunked extend) keeps the torch compress/index math and Python ints;
sparse attn uses the packed SM70 kernel (same dequant + online softmax as
decode). ``SGLANG_DSV41_TORCH_PREFILL_SPARSE=1`` restores unpack + einsum.

TARGET_VERIFY (T=γ+1) uses the packed decode kernels, one token at a time:
compress+index in order (B.4c ratio-2 prefix unrolled at static T so the
graph keeps fixed shapes), then the prefill sparse oracle over a scratch
window. The live ring, ratio-2 ``pending_*``, and compressed rows stay on the
committed prefix until ``sm70_commit_target_verify`` (after accept). No
``positions[0].item()``. Chunked prefill stays on the oracle path; draft
DECODE T>1 also uses that path but skips the host sync while capturing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from sglang.kernels.ops.attention.sm70_dsv41_csa2 import (
    HEAD_DIM,
    INDEX_ROW_BYTES,
    KV_ROW_BYTES,
    SWA_ROW_BYTES,
    freqs_cis_interleaved_table,
    index_logits as cuda_index_logits,
    index_logits_decode_dense,
    index_logits_decode_gather,
    pack_index_k,
    pack_index_k_at,
    pack_kv_fp4,
    pack_kv_fp4_at,
    pack_swa_fp8,
    pack_swa_fp8_at,
    sparse_decode as cuda_sparse_decode,
    sparse_decode_indexed,
    sparse_prefill as cuda_sparse_prefill,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.sm70_csa2_reference import (
    expand_candidate_block_ids,
    select_candidate_block_ids,
    sparse_attention_rows,
    topk_positions,
    unpack_kv_fp4_e4m3,
    unpack_swa_fp8_ue8m0,
)

_FLASH_KV_SOURCES = (2, 8, 14, 20)
_FLASH_INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
_FLASH_CANDIDATE_SOURCE = 20
# Prefill unpacks SWA to fp16 [T, 128, 512]. T=2048 is 256 MiB plus a fp32
# dequant, which OOMs 32 GiB V100 after the 2048-token MoE working set.
# Packed sparse keeps the rows quantized, so the tile can be larger.
_PREFILL_SPARSE_Q_TILE = 64
_PREFILL_SPARSE_Q_TILE_PACKED = 256
# CUDA indexer logits are [T, N_compressed] fp32. Cap the product so a 250k
# prefill chunk cannot materialize a 1 GiB score matrix.
_INDEX_LOGITS_MAX_ELEMS = 8 * 1024 * 1024
_DEFAULT_CAPACITY = 4096
_FAST_TOPK_K = (512, 2048)

logger = logging.getLogger(__name__)


_MM_ALIGN = 32
_CAPTURE_T_MAX = 8


def _capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _pad_leading(x: torch.Tensor, align: int = _MM_ALIGN) -> Tuple[torch.Tensor, int]:
    """Pad dim 0 so T=6 verify and T=25 extend share one GEMM tile."""
    m = int(x.shape[0])
    if m == 0:
        return x, 0
    mp = (m + align - 1) // align * align
    if mp == m:
        return x, m
    padded = x.new_zeros((mp,) + tuple(x.shape[1:]))
    padded[:m].copy_(x)
    return padded, m


def _matmul_aligned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [M,K] @ b [K,N] with M padded to ``_MM_ALIGN``."""
    a_pad, m = _pad_leading(a)
    if m == 0 or a_pad.shape[0] == m:
        return a @ b
    return (a_pad @ b)[:m]


def _linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _matmul_aligned(x.float(), weight.float().t()).to(x.dtype)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * xf).to(x.dtype)


def _weight(module, name: str) -> torch.Tensor:
    obj = getattr(module, name)
    return obj.weight if hasattr(obj, "weight") else obj


def _freqs_for(layer, positions: torch.Tensor, check: bool = True) -> torch.Tensor:
    freqs = layer.freqs_cis
    if check and not _capturing():
        n = int(freqs.shape[0])
        pmin = int(positions.amin().item())
        pmax = int(positions.amax().item())
        if pmin < 0 or pmax >= n:
            raise RuntimeError(
                f"SM70 CSA2 freqs_cis OOB: pos=[{pmin},{pmax}] "
                f"n={n} positions={tuple(positions.shape)}"
            )
    return freqs[positions]


def _attn_glue_on() -> bool:
    return bool(envs.SGLANG_DSV41_ATTN_GLUE.get())


def _freqs_table(layer) -> torch.Tensor:
    cached = getattr(layer, "_sm70_freqs_real", None)
    if cached is None:
        cached = freqs_cis_interleaved_table(layer.freqs_cis)
        try:
            layer._sm70_freqs_real = cached
        except Exception:
            pass
    return cached


def _topology(layer, backend) -> dict:
    hf = None
    runner = getattr(backend, "model_runner", None)
    if runner is not None:
        hf = getattr(getattr(runner, "model_config", None), "hf_text_config", None)

    def seq(attr, default):
        val = getattr(hf, attr, None) if hf is not None else None
        if not val:
            val = getattr(layer, attr, None)
        t = tuple(val) if val else ()
        return t if t else default

    return {
        "kv_sources": seq("kv_source_layer_ids", _FLASH_KV_SOURCES),
        "index_sources": seq("index_source_layer_ids", _FLASH_INDEX_SOURCES),
        "candidate_source": (
            getattr(hf, "candidate_source_layer_id", None)
            if hf is not None
            else getattr(layer, "candidate_source_layer_id", None)
        )
        or _FLASH_CANDIDATE_SOURCE,
        "candidate_topk_blocks": int(
            getattr(getattr(layer, "indexer", None), "candidate_topk_blocks", 2048)
            or 2048
        ),
        "candidate_block_size": int(
            getattr(getattr(layer, "indexer", None), "candidate_block_size", 8) or 8
        ),
        "index_topk": int(getattr(getattr(layer, "indexer", None), "index_topk", 512)),
        "sliding_window": int(getattr(layer, "sliding_window", 128) or 128),
    }


def kv_source_for(layer_id: int, ratio: int, kv_sources) -> int:
    src = [s for s in kv_sources if s <= layer_id]
    # Same-ratio owner: Flash shares a KV source among layers of that ratio.
    # Layers 2-19 are ratio 2 (sources 2,8,14); 20+ are ratio 1 (source 20).
    if ratio == 1:
        src = [s for s in src if s >= 20] or src
    else:
        src = [s for s in src if s < 20] or src
    if not src:
        raise RuntimeError(f"SM70 CSA2: layer {layer_id} ratio {ratio} has no kv source")
    return max(src)


def index_source_for(layer_id: int, ratio: int, index_sources) -> int:
    src = [s for s in index_sources if s <= layer_id]
    if ratio == 1:
        src = [s for s in src if s >= 20] or src
    else:
        src = [s for s in src if s < 20] or src
    if not src:
        raise RuntimeError(
            f"SM70 CSA2: layer {layer_id} ratio {ratio} has no index source"
        )
    return max(src)


def compressed_capacity(capacity: int, ratio: int, block_size: int = 8) -> int:
    """Rows for ``capacity`` tokens at ``ratio``, padded to whole candidate blocks."""
    rows = (int(capacity) + ratio - 1) // ratio + 1
    return (rows + block_size - 1) // block_size * block_size


@dataclass
class Sm70Csa2State:
    capacity: int = _DEFAULT_CAPACITY
    sliding_window: int = 128
    candidate_block_size: int = 8
    candidate_topk_blocks: int = 2048
    # Static device buffers (see module docstring).
    swa_ring: Dict[int, torch.Tensor] = field(default_factory=dict)
    kv_rows: Dict[int, torch.Tensor] = field(default_factory=dict)
    index_rows: Dict[int, torch.Tensor] = field(default_factory=dict)
    pending_kv: Dict[int, torch.Tensor] = field(default_factory=dict)
    pending_score: Dict[int, torch.Tensor] = field(default_factory=dict)
    topk: Dict[int, torch.Tensor] = field(default_factory=dict)
    cand_ids: Optional[torch.Tensor] = None
    cand_key_ids: Optional[torch.Tensor] = None
    cand_offs: Optional[torch.Tensor] = None
    _ring_slots: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    logits_c: Optional[torch.Tensor] = None
    # Prefill-only Python state for the chunk in flight.
    prefill_swa: Dict[int, Tuple[torch.Tensor, object]] = field(default_factory=dict)
    prefill_topk: Dict[int, torch.Tensor] = field(default_factory=dict)
    prefill_cand_ids: Optional[torch.Tensor] = None
    verify_kv: Dict[int, torch.Tensor] = field(default_factory=dict)
    verify_kv_buf: Dict[int, torch.Tensor] = field(default_factory=dict)
    verify_topk: Dict[int, torch.Tensor] = field(default_factory=dict)
    verify_cand: Optional[torch.Tensor] = None
    # Target-verify scratch. The forward reads these; commit publishes the
    # accepted prefix into the live ring / pending / compressed rows.
    verify_swa_packed: Dict[int, torch.Tensor] = field(default_factory=dict)
    verify_swa_stage: Optional[torch.Tensor] = None
    verify_positions: Optional[torch.Tensor] = None
    verify_open: Optional[torch.Tensor] = None
    verify_t: Optional[torch.Tensor] = None
    verify_pending_kv_traj: Dict[int, torch.Tensor] = field(default_factory=dict)
    verify_pending_score_traj: Dict[int, torch.Tensor] = field(default_factory=dict)
    verify_pending_kv_cur: Dict[int, torch.Tensor] = field(default_factory=dict)
    verify_pending_score_cur: Dict[int, torch.Tensor] = field(default_factory=dict)
    verify_row: Dict[int, torch.Tensor] = field(default_factory=dict)
    verify_kv_orig: Dict[int, torch.Tensor] = field(default_factory=dict)
    verify_index_orig: Dict[int, torch.Tensor] = field(default_factory=dict)
    swa_scratch: Dict[int, torch.Tensor] = field(default_factory=dict)
    gather_n: Optional[torch.Tensor] = None
    capture_t: int = 1
    last_positions: Optional[torch.Tensor] = None
    layers: Dict[int, object] = field(default_factory=dict)

    def clear_sequence(self) -> None:
        """Drop prefill leftovers. Device buffers need no reset (position-indexed)."""
        self.prefill_swa.clear()
        self.prefill_topk.clear()
        self.prefill_cand_ids = None
        self.verify_kv.clear()
        self.last_positions = None

    def ring_valid(self, positions: torch.Tensor) -> torch.Tensor:
        """[T, W] bool: ring slot holds a position in ``(pos - W, pos]`` and ``>= 0``."""
        w = self.sliding_window
        slots = getattr(self, "_ring_slots", None)
        if (
            slots is None
            or slots.device != positions.device
            or slots.dtype != positions.dtype
            or int(slots.numel()) != w
        ):
            slots = torch.arange(w, device=positions.device, dtype=positions.dtype)
            self._ring_slots = slots
        d = torch.remainder(positions[:, None] - slots[None, :], w)
        return d <= positions[:, None]


def _capture_t(backend, st: Sm70Csa2State) -> int:
    runner = getattr(backend, "model_runner", None)
    n = 1
    if runner is not None and hasattr(runner, "decode_num_tokens_per_req"):
        try:
            n = int(runner.decode_num_tokens_per_req())
        except Exception:
            n = 1
    n = max(1, n, int(getattr(st, "capture_t", 1) or 1))
    return min(max(n, 1), _CAPTURE_T_MAX)


def get_state(backend) -> Sm70Csa2State:
    st = getattr(backend, "_sm70_csa2", None)
    if st is None:
        cap = getattr(backend, "max_context_len", None)
        st = Sm70Csa2State(capacity=int(cap) if cap else _DEFAULT_CAPACITY)
        backend._sm70_csa2 = st
    return st


def _graph_buffer(factory, *args, **kwargs):
    """Tensor CUDA-graph capture may update in place.

    Warmup forwards run under ``torch.inference_mode``. Capture does not, and
    PyTorch rejects an inplace write to a tensor created there. Opt out for
    every buffer the captured verify forward writes with ``copy_`` / ``fill_``.
    """
    with torch.inference_mode(False):
        return factory(*args, **kwargs)


def _ensure_buffers(st: Sm70Csa2State, layer, topo: dict, device: torch.device, backend=None) -> None:
    """Allocate this layer's static buffers once (never inside a graph capture)."""
    lid = int(layer.layer_id)
    if lid in st.swa_ring:
        return
    if _capturing():
        raise RuntimeError(
            f"SM70 CSA2 layer {lid} buffers requested during CUDA-graph capture; "
            "warm up eagerly first"
        )
    st.sliding_window = int(topo["sliding_window"])
    st.candidate_block_size = int(topo["candidate_block_size"])
    st.candidate_topk_blocks = int(topo["candidate_topk_blocks"])
    st.capture_t = _capture_t(backend, st) if backend is not None else max(st.capture_t, 1)
    tmax = st.capture_t
    bs = st.candidate_block_size
    u8 = dict(dtype=torch.uint8, device=device)
    st.swa_ring[lid] = _graph_buffer(torch.zeros, (st.sliding_window, SWA_ROW_BYTES), **u8)
    st.swa_scratch[lid] = _graph_buffer(
        torch.zeros, (st.sliding_window + tmax, SWA_ROW_BYTES), **u8
    )
    ratio = int(getattr(layer, "compress_ratio", 0) or 0)
    compressor = getattr(layer, "compressor", None)
    if compressor is not None:
        cap_c = compressed_capacity(st.capacity, ratio, bs)
        st.kv_rows[lid] = _graph_buffer(torch.zeros, (cap_c, KV_ROW_BYTES), **u8)
        idxer = getattr(layer, "indexer", None)
        if idxer is not None and getattr(idxer, "owns_k", False):
            st.index_rows[lid] = _graph_buffer(torch.zeros, (cap_c, INDEX_ROW_BYTES), **u8)
        if ratio == 2:
            st.pending_kv[lid] = _graph_buffer(
                torch.zeros, layer.head_dim, dtype=torch.float32, device=device
            )
            st.pending_score[lid] = _graph_buffer(
                torch.zeros, layer.head_dim, dtype=torch.float32, device=device
            )
    if getattr(layer, "indexer", None) is not None:
        k = int(topo["index_topk"])
        st.topk[lid] = _graph_buffer(
            torch.full, (1, k), -1, dtype=torch.int32, device=device
        )
        st.verify_topk[lid] = _graph_buffer(
            torch.full, (tmax, k), -1, dtype=torch.int32, device=device
        )
    if st.logits is None:
        cap_max = compressed_capacity(st.capacity, 1, bs)
        st.logits = _graph_buffer(torch.zeros, (1, cap_max), dtype=torch.float32, device=device)
        nc = st.candidate_topk_blocks * bs
        st.logits_c = _graph_buffer(torch.zeros, (1, nc), dtype=torch.float32, device=device)
        st.cand_ids = _graph_buffer(
            torch.full, (1, st.candidate_topk_blocks), -1, dtype=torch.int32, device=device
        )
        st.cand_key_ids = _graph_buffer(
            torch.full, (1, nc), -1, dtype=torch.int32, device=device
        )
        st.cand_offs = _graph_buffer(torch.arange, bs, device=device, dtype=torch.int32)
        st.gather_n = _graph_buffer(torch.full, (1,), nc, dtype=torch.int32, device=device)
        st.verify_cand = _graph_buffer(
            torch.full, (tmax, st.candidate_topk_blocks), -1, dtype=torch.int32, device=device
        )


# --------------------------------------------------------------------------- #
# Prefill (chunked extend): torch oracle math over the static buffers
# --------------------------------------------------------------------------- #


def _prefill_swa_view(
    st: Sm70Csa2State, lid: int, pos0, packed_chunk: torch.Tensor
) -> Tuple[torch.Tensor, object]:
    """Rows for absolute positions ``[pos0 - W, pos0 + T)``; negatives are garbage
    and masked by ``_gather_swa`` (``idx >= 0``).

    ``pos0`` may be a Python int (eager prefill) or a 0-dim/1-elem tensor
    (CUDA-graph capture / replay).
    """
    w = st.sliding_window
    ring = st.swa_ring[lid]
    t = int(packed_chunk.shape[0])
    scratch = st.swa_scratch.get(lid)
    use_scratch = (
        scratch is not None
        and int(scratch.shape[0]) >= w + t
        and scratch.dtype == packed_chunk.dtype
    )
    if isinstance(pos0, torch.Tensor):
        p0 = pos0.reshape(()).to(dtype=torch.int64)
        hist_pos = p0 - w + torch.arange(w, device=ring.device, dtype=torch.int64)
        prev = ring[torch.remainder(hist_pos.clamp(min=0), w)]
        origin = p0 - w
        if use_scratch:
            scratch[:w].copy_(prev)
            scratch[w : w + t].copy_(packed_chunk)
            return scratch[: w + t], origin
        return torch.cat([prev, packed_chunk], dim=0), origin
    prev_pos = torch.arange(pos0 - w, pos0, device=ring.device, dtype=torch.int64)
    prev = ring[torch.remainder(prev_pos, w)]
    return torch.cat([prev, packed_chunk], dim=0), pos0 - w


def _store_prefill_packed_swa(
    st: Sm70Csa2State,
    lid: int,
    pos0,
    positions: torch.Tensor,
    packed_swa: torch.Tensor,
    *,
    commit_ring: bool,
) -> None:
    """Cat ring prefix + chunk into ``prefill_swa``. Skip ring write for DSpark MASK."""
    st.prefill_swa[lid] = _prefill_swa_view(st, lid, pos0, packed_swa)
    if not commit_ring:
        return
    w = st.sliding_window
    k = min(int(packed_swa.shape[0]), w)
    st.swa_ring[lid][torch.remainder(positions[-k:], w)] = packed_swa[-k:]


def pack_swa_window_from_kv(
    backend,
    layer,
    kv: torch.Tensor,
    positions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """Write ``wkv(main_x)`` into CSA2 ``swa_ring``. ``mask`` False leaves the slot."""
    if kv is None or kv.numel() == 0 or positions.numel() == 0:
        return
    st = get_state(backend)
    topo = _topology(layer, backend)
    _ensure_buffers(st, layer, topo, kv.device, backend)
    lid = int(layer.layer_id)
    w = st.sliding_window
    t = int(kv.shape[0])
    k = min(t, w)
    kv = kv[-k:]
    positions = positions[-k:].to(dtype=torch.int64)
    if mask is not None:
        mask = mask[-k:].to(dtype=torch.bool)
    kv = _rmsnorm(kv, layer.kv_norm.weight, layer.eps)
    packed = pack_swa_fp8(
        kv, _freqs_for(layer, positions, check=False), int(layer.qk_rope_head_dim)
    )
    slots = torch.remainder(positions, w)
    ring = st.swa_ring[lid]
    if mask is None:
        ring.index_copy_(0, slots, packed)
        return
    src = torch.where(mask.unsqueeze(-1), packed, ring[slots])
    ring.index_copy_(0, slots, src)


def _gather_swa(
    st: Sm70Csa2State, buf: torch.Tensor, origin, positions: torch.Tensor
):
    """For each query position, the W-token window ending at that position."""
    w = st.sliding_window
    offs = torch.arange(w, device=positions.device, dtype=positions.dtype)
    idx = positions[:, None] - (w - 1) + offs[None, :]
    physical = idx - origin
    n = int(buf.shape[0])
    valid = (idx >= 0) & (physical >= 0) & (physical < n)
    gathered = buf[physical.clamp(0, n - 1)]
    return gathered, valid


def _gather_swa_from_ring(st: Sm70Csa2State, lid: int, positions: torch.Tensor):
    """Window ending at each query position, read from the live decode ring."""
    w = st.sliding_window
    ring = st.swa_ring[lid]
    offs = torch.arange(w, device=positions.device, dtype=positions.dtype)
    idx = positions[:, None] - (w - 1) + offs[None, :]
    valid = idx >= 0
    physical = torch.remainder(idx.clamp(min=0), w)
    return ring[physical], valid


def _sparse_attention_rows_aligned(
    q: torch.Tensor,
    keys: torch.Tensor,
    valid: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Prefill/verify oracle with T padded so T=6 and T=25 share one einsum tile."""
    t = int(q.shape[0])
    q_pad, _ = _pad_leading(q)
    keys_pad, _ = _pad_leading(keys)
    valid_pad, _ = _pad_leading(valid)
    out = sparse_attention_rows(q_pad, keys_pad, valid_pad, attn_sink, softmax_scale)
    return out[:t]


def _pack_swa_prefill_one(
    layer, kv: torch.Tensor, positions: torch.Tensor, st: Sm70Csa2State, lid: int
) -> None:
    """Same packer as chunked prefill (``pack_swa_fp8``), one token into the ring."""
    packed = pack_swa_fp8(
        kv,
        _freqs_for(layer, positions, check=False),
        int(layer.qk_rope_head_dim),
    )
    st.swa_ring[lid].index_copy_(
        0,
        torch.remainder(positions.to(torch.int64), st.sliding_window),
        packed,
    )


def dspark_shared_block_attn(layer, st: Optional[Sm70Csa2State], lid: int) -> bool:
    """Inject flag may sit on RadixAttention or the MQA module in ``st.layers``."""
    if bool(getattr(layer, "sm70_swa_ring_from_inject", False)):
        return True
    if st is None:
        return False
    return bool(getattr(st.layers.get(lid), "sm70_swa_ring_from_inject", False))


def dspark_shared_block_valid(buf: torch.Tensor, origin) -> torch.Tensor:
    """Keys with abs pos >= 0 in ``cat(window, draft_kv)`` (shared across the block)."""
    n = int(buf.shape[0])
    abs_pos = origin + torch.arange(n, device=buf.device, dtype=torch.int64)
    return abs_pos >= 0


def _prefill_compress(layer, x: torch.Tensor, pos0: int, st: Sm70Csa2State):
    """kv_source: complete compressed latents (pre-RoPE) for this chunk."""
    comp = layer.compressor
    ratio = int(layer.compress_ratio)
    lid = int(layer.layer_id)
    eps = float(layer.eps)
    positions = torch.arange(
        pos0, pos0 + x.shape[0], device=x.device, dtype=torch.int64
    )
    if ratio == 1:
        latent = _rmsnorm(_linear(x, _weight(comp, "wkv")), _weight(comp, "norm"), eps)
        return latent, positions

    xf = x.float()
    kv_cur = _matmul_aligned(xf, _weight(comp, "wkv").float().t())
    score_cur = _matmul_aligned(xf, _weight(comp, "wgate").float().t())
    if pos0 % 2 == 1:
        kv = torch.cat([st.pending_kv[lid][None], kv_cur], dim=0)
        score = torch.cat([st.pending_score[lid][None], score_cur], dim=0)
        first_pos = pos0 - 1
    else:
        kv, score, first_pos = kv_cur, score_cur, pos0
    # Always carry the last token: the next chunk/decode token pairs with it.
    st.pending_kv[lid].copy_(kv_cur[-1])
    st.pending_score[lid].copy_(score_cur[-1])
    n = kv.shape[0]
    complete = n - n % 2
    if complete == 0:
        return (
            x.new_empty((0, layer.head_dim)),
            positions.new_empty((0,), dtype=torch.int64),
        )
    kv2 = kv[:complete].unflatten(0, (-1, 2))
    score2 = score[:complete].unflatten(0, (-1, 2))
    pooled = (kv2 * score2.softmax(dim=1)).sum(dim=1)
    latent = _rmsnorm(pooled.to(x.dtype), _weight(comp, "norm"), eps)
    group_pos = first_pos + 2 * torch.arange(
        pooled.shape[0], device=x.device, dtype=torch.int64
    )
    return latent, group_pos


def _prefill_index_topk(
    q, w, index_rows, query_pos, compress_lens, ratio, topo, layer, st, n
):
    """CUDA indexer logits, tiled so [T, N] fp32 cannot grow to a gigabyte."""
    t = q.shape[0]
    tile = max(1, _INDEX_LOGITS_MAX_ELEMS // max(int(n), 1))
    is_src = bool(layer.indexer.is_candidate_source)
    uses_cand = bool(layer.indexer.uses_candidates)
    topk_parts = []
    id_parts = []
    first_logits = None
    block_size = int(topo["candidate_block_size"])
    for s in range(0, t, tile):
        e = min(s + tile, t)
        logits = cuda_index_logits(q[s:e], w[s:e], index_rows, query_pos[s:e], ratio)
        if is_src:
            id_parts.append(
                select_candidate_block_ids(
                    logits,
                    compress_lens[s:e, None],
                    topo["candidate_topk_blocks"],
                    block_size,
                )
            )
        elif uses_cand:
            ids = st.prefill_cand_ids
            if ids is None or int(ids.shape[0]) < e:
                raise RuntimeError(
                    "SM70 CSA2 prefill candidate ids missing for rows "
                    f"{s}:{e} (have {None if ids is None else ids.shape[0]})"
                )
            mask = expand_candidate_block_ids(ids[s:e], n, block_size)
            logits = logits.masked_fill(~mask, -torch.inf)
        topk_parts.append(
            topk_positions(logits, compress_lens[s:e, None], topo["index_topk"])
        )
        if first_logits is None:
            first_logits = logits
    topk = topk_parts[0] if len(topk_parts) == 1 else torch.cat(topk_parts, dim=0)
    if is_src:
        st.prefill_cand_ids = (
            id_parts[0] if len(id_parts) == 1 else torch.cat(id_parts, dim=0)
        )
    return topk, first_logits


def _prefill_low_ratio_sources(backend, layer, x, q_lora, positions, forward_batch, topo, st):
    lid = int(layer.layer_id)
    ratio = int(layer.compress_ratio)
    t = int(x.shape[0])
    # Host sync. Skip under CUDA-graph capture (draft DECODE T>1); TARGET_VERIFY
    # uses the packed path and never enters here.
    pos0 = None if _capturing() else int(positions[0].item())
    if pos0 is not None and pos0 + t > st.capacity:
        raise RuntimeError(
            f"SM70 CSA2: chunk [{pos0},{pos0 + t}) exceeds context capacity {st.capacity}"
        )
    pos0_view = positions[0] if pos0 is None else pos0

    if getattr(layer, "fuse_wqa_wkv", False):
        qkv, _ = layer.wqkv_a(x)
        kv = qkv[..., layer.q_lora_rank :]
    else:
        kv, _ = layer.wkv(x)
    kv = _rmsnorm(kv, layer.kv_norm.weight, layer.eps)
    packed_swa = pack_swa_fp8(kv, _freqs_for(layer, positions), layer.qk_rope_head_dim)
    _store_prefill_packed_swa(
        st,
        lid,
        pos0_view,
        positions,
        packed_swa,
        commit_ring=not dspark_shared_block_attn(layer, st, lid),
    )
    if envs.SGLANG_DEBUG_DSV41_PROBE_STATS.get():
        from sglang.srt.debug.dsv41_probe_stats import record, record_rope_vs_official

        record(f"L{lid}.csa2.kv_prepack", kv)
        record(f"L{lid}.csa2.packed_swa_u8", packed_swa)
        if lid in (0, 1, 2) and kv.numel():
            rd = int(layer.qk_rope_head_dim)
            payload, scales = packed_swa[..., :HEAD_DIM], packed_swa[..., HEAD_DIM:]
            k_u = unpack_swa_fp8_ue8m0(payload, scales, kv.dtype)
            record(f"L{lid}.csa2.kv_unpacked", k_u)
            record_rope_vs_official(
                f"L{lid}.csa2.k_rope_vs_official",
                kv[..., -rd:],
                k_u[..., -rd:],
                layer.freqs_cis,
                positions,
            )

    if getattr(layer, "compressor", None) is not None:
        if pos0 is None:
            raise RuntimeError(
                "SM70 CSA2: prefill compress cannot run under CUDA-graph capture"
            )
        latent, group_pos = _prefill_compress(layer, x, pos0, st)
        if latent.shape[0] > 0:
            gfreq = _freqs_for(layer, group_pos)
            rows = group_pos // ratio
            st.kv_rows[lid][rows] = pack_kv_fp4(latent, gfreq, layer.qk_rope_head_dim)
            idxer = layer.indexer
            if idxer is not None and getattr(idxer, "owns_k", False):
                ik = _rmsnorm(_linear(latent, idxer.wk.weight), idxer.k_norm.weight, layer.eps)
                st.index_rows[lid][rows] = pack_index_k(ik, gfreq, layer.qk_rope_head_dim)

    if getattr(layer, "indexer", None) is not None:
        if pos0 is None:
            raise RuntimeError(
                "SM70 CSA2: prefill indexer cannot run under CUDA-graph capture"
            )
        kv_src = kv_source_for(lid, ratio, topo["kv_sources"])
        n = (pos0 + t) // ratio  # rows visible to the last query
        index_rows = st.index_rows[kv_src][:n]
        compress_lens = (positions + 1) // ratio
        q = layer.indexer.queries(q_lora, _freqs_for(layer, positions))
        wgt = layer.indexer.head_weights(x)
        if envs.SGLANG_DEBUG_DSV41_PROBE_STATS.get():
            from sglang.srt.debug.dsv41_probe_stats import record, record_ids

            record(f"L{lid}.csa2.index_q", q)
            record(f"L{lid}.csa2.index_w", wgt)
        if n == 0:
            topk = torch.full(
                (t, topo["index_topk"]), -1, dtype=torch.int32, device=x.device
            )
            logits = torch.zeros((t, 0), dtype=torch.float32, device=x.device)
        else:
            use_cuda = t == 1 or not envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get()
            query_pos = positions.to(torch.int32)
            if use_cuda:
                topk, logits = _prefill_index_topk(
                    q, wgt, index_rows, query_pos, compress_lens, ratio, topo, layer, st, n
                )
            else:
                from sglang.srt.layers.attention.dsv4.sm70_csa2_reference import (
                    index_scores,
                    unpack_index_fp4_ue8m0,
                )

                payload = index_rows[:, :64]
                exps = index_rows[:, 64:]
                ik = unpack_index_fp4_ue8m0(payload, exps, q.dtype)
                logits = index_scores(q, ik, wgt, torch.float32)
                reach = torch.arange(n, device=logits.device)[None, :] < compress_lens[:, None]
                logits = logits.masked_fill(~reach, -torch.inf)
                if layer.indexer.is_candidate_source:
                    st.prefill_cand_ids = select_candidate_block_ids(
                        logits,
                        compress_lens[:, None],
                        topo["candidate_topk_blocks"],
                        topo["candidate_block_size"],
                    )
                elif layer.indexer.uses_candidates:
                    ids = st.prefill_cand_ids
                    if ids is None:
                        raise RuntimeError("SM70 CSA2 candidate mask missing")
                    mask = expand_candidate_block_ids(
                        ids, n, int(topo["candidate_block_size"])
                    )
                    logits = logits.masked_fill(~mask, -torch.inf)
                topk = topk_positions(logits, compress_lens[:, None], topo["index_topk"])
        st.prefill_topk[lid] = topk
        if envs.SGLANG_DEBUG_DSV41_PROBE_STATS.get():
            from sglang.srt.debug.dsv41_probe_stats import record, record_ids

            if n != 0:
                record(f"L{lid}.csa2.index_logits", logits)
            record_ids(f"L{lid}.csa2.topk", topk)


# --------------------------------------------------------------------------- #
# Decode (bs=1, T=1): no host sync, fixed launch shapes
# --------------------------------------------------------------------------- #


def _select_topk(scores: torch.Tensor, lens32: torch.Tensor, k: int) -> torch.Tensor:
    """Row 0 top-k indices over ``scores[:, :lens]``; -1 where absent or -inf."""
    if k in _FAST_TOPK_K:
        from sglang.kernels.ops.attention.fast_topk import fast_topk

        idx = fast_topk(scores, lens32, k)
    else:
        # Test-size fallback (k not compiled): mask beyond lens, torch.topk.
        width = scores.shape[1]
        reach = torch.arange(width, device=scores.device)[None, :] < lens32[:, None]
        idx = scores.masked_fill(~reach, -torch.inf).topk(min(k, width), dim=-1).indices
        idx = idx.to(torch.int32)
        if idx.shape[1] < k:
            idx = torch.nn.functional.pad(idx, (0, k - idx.shape[1]), value=-1)
    vals = scores.gather(1, idx.clamp_min(0).to(torch.int64))
    ok = (idx >= 0) & (idx < lens32[:, None]) & (vals > -torch.inf)
    return torch.where(ok, idx, torch.full_like(idx, -1))


def _project_swa_kv(layer, x: torch.Tensor) -> torch.Tensor:
    if getattr(layer, "fuse_wqa_wkv", False):
        qkv, _ = layer.wqkv_a(x)
        kv = qkv[..., layer.q_lora_rank :]
    else:
        kv, _ = layer.wkv(x)
    return _rmsnorm(kv, layer.kv_norm.weight, layer.eps)


def _decode_pack_swa(layer, kv: torch.Tensor, positions: torch.Tensor, st: Sm70Csa2State, lid: int) -> None:
    """Write this token's SWA KV into the live ring (T=1 decode kernel)."""
    w = st.sliding_window
    rope_dim = int(layer.qk_rope_head_dim)
    if _attn_glue_on():
        pack_swa_fp8_at(
            st.swa_ring[lid],
            kv,
            positions,
            freqs_table=_freqs_table(layer),
            rope_dim=rope_dim,
            dst_mod=w,
        )
        return
    freqs = _freqs_for(layer, positions, check=False)
    packed_swa = pack_swa_fp8(kv, freqs, rope_dim)
    st.swa_ring[lid].index_copy_(0, torch.remainder(positions, w), packed_swa)


def _decode_compress_and_index(
    backend, layer, x, q_lora, positions, topo, st, verify_step: Optional[int] = None
):
    """T=1 compressor + hierarchical indexer. Does not touch the SWA ring.

    ``verify_step`` runs the ratio-2 recurrence on a scratch copy of
    ``pending_*`` and records each compressed row so commit can drop the
    unaccepted tail. The live pending slot is left unchanged.
    """
    if getattr(layer, "compressor", None) is None and getattr(layer, "indexer", None) is None:
        return
    lid = int(layer.layer_id)
    ratio = int(layer.compress_ratio)
    bs = st.candidate_block_size
    glue = _attn_glue_on()
    freqs = _freqs_for(layer, positions, check=False)
    table = _freqs_table(layer) if glue else None
    rope_dim = int(layer.qk_rope_head_dim)

    if getattr(layer, "compressor", None) is not None:
        comp = layer.compressor
        eps = float(layer.eps)
        if ratio == 1:
            latent = _rmsnorm(_linear(x, _weight(comp, "wkv")), _weight(comp, "norm"), eps)
            row = positions
            gfreq = freqs
            row_div, freq_mul = 1, 1
        else:
            xf = x.float()
            kv_cur = _matmul_aligned(xf, _weight(comp, "wkv").float().t())
            score_cur = _matmul_aligned(xf, _weight(comp, "wgate").float().t())
            if verify_step is None:
                prev_kv = st.pending_kv[lid]
                prev_score = st.pending_score[lid]
            else:
                prev_kv, prev_score = _verify_pending_prev(st, lid, verify_step)
            kv2 = torch.stack([prev_kv[None], kv_cur], dim=1)
            score2 = torch.stack([prev_score[None], score_cur], dim=1)
            pooled = (kv2 * score2.softmax(dim=1)).sum(dim=1)
            latent = _rmsnorm(pooled.to(x.dtype), _weight(comp, "norm"), eps)
            if verify_step is None:
                prev_kv.copy_(kv_cur[0])
                prev_score.copy_(score_cur[0])
            else:
                _verify_pending_save(st, lid, verify_step, kv_cur, score_cur)
            row = positions // 2
            gfreq = _freqs_for(layer, row * 2, check=False)
            row_div, freq_mul = 2, 2
        row_idx = None
        if verify_step is not None:
            row_idx = row.reshape(-1)[:1].to(dtype=torch.int64)
            _snap_verify_row(
                st, lid, verify_step, row_idx, st.kv_rows[lid], st.verify_kv_orig, KV_ROW_BYTES
            )
        if glue:
            pack_kv_fp4_at(
                st.kv_rows[lid],
                latent,
                positions,
                freqs_table=table,
                rope_dim=rope_dim,
                row_div=row_div,
                freq_mul=freq_mul,
            )
        else:
            st.kv_rows[lid].index_copy_(
                0, row, pack_kv_fp4(latent, gfreq, rope_dim)
            )
        idxer = layer.indexer
        if idxer is not None and getattr(idxer, "owns_k", False):
            ik = _rmsnorm(_linear(latent, idxer.wk.weight), idxer.k_norm.weight, eps)
            if verify_step is not None:
                if row_idx is None:
                    row_idx = row.reshape(-1)[:1].to(dtype=torch.int64)
                _snap_verify_row(
                    st,
                    lid,
                    verify_step,
                    row_idx,
                    st.index_rows[lid],
                    st.verify_index_orig,
                    INDEX_ROW_BYTES,
                    store_index=False,
                )
            if glue:
                pack_index_k_at(
                    st.index_rows[lid],
                    ik,
                    positions,
                    freqs_table=table,
                    rope_dim=rope_dim,
                    row_div=row_div,
                    freq_mul=freq_mul,
                )
            else:
                st.index_rows[lid].index_copy_(
                    0, row, pack_index_k(ik, gfreq, rope_dim)
                )

    idxer = getattr(layer, "indexer", None)
    if idxer is None:
        return
    kv_src = kv_source_for(lid, ratio, topo["kv_sources"])
    rows = st.index_rows[kv_src]
    q = idxer.queries(q_lora, freqs)
    wgt = idxer.head_weights(x)
    qp32 = positions.to(torch.int32)
    vis32 = ((positions + 1) // ratio).to(torch.int32)
    k = int(topo["index_topk"])
    if bool(idxer.is_candidate_source) or not bool(idxer.uses_candidates):
        logits = st.logits[:, : rows.shape[0]]
        index_logits_decode_dense(logits, q, wgt, rows, qp32, ratio, bs)
        if bool(idxer.is_candidate_source):
            blk = logits.view(1, -1, bs).amax(dim=-1)
            last = ((vis32 - 1) // bs).clamp_min(0).to(torch.int64)
            blk.scatter_(1, last[:, None], torch.inf)
            nblk32 = (vis32 + bs - 1) // bs
            st.cand_ids.copy_(_select_topk(blk, nblk32, st.candidate_topk_blocks))
        topk = _select_topk(logits, vis32, k)
    else:
        cand = st.cand_ids
        offs = st.cand_offs
        if offs is None or int(offs.numel()) != bs:
            offs = torch.arange(bs, device=x.device, dtype=torch.int32)
            st.cand_offs = offs
        key_ids = torch.where(
            cand[:, :, None] >= 0,
            cand[:, :, None] * bs + offs[None, None, :],
            torch.full((), -1, dtype=torch.int32, device=x.device),
        ).view(1, -1)
        st.cand_key_ids.copy_(key_ids)
        logits_c = st.logits_c
        index_logits_decode_gather(logits_c, q, wgt, rows, qp32, st.cand_key_ids, ratio)
        full32 = st.gather_n
        if full32 is None or int(full32.numel()) != 1:
            full32 = torch.full((1,), logits_c.shape[1], dtype=torch.int32, device=x.device)
            st.gather_n = full32
        local = _select_topk(logits_c, full32, k)
        glob = st.cand_key_ids.gather(1, local.clamp_min(0).to(torch.int64))
        topk = torch.where(local >= 0, glob, torch.full_like(glob, -1))
    st.topk[lid].copy_(topk)


def _decode_low_ratio_sources(backend, layer, x, q_lora, positions, topo, st):
    lid = int(layer.layer_id)
    kv = _project_swa_kv(layer, x)
    _decode_pack_swa(layer, kv, positions, st, lid)
    _decode_compress_and_index(backend, layer, x, q_lora, positions, topo, st)


def _ensure_verify_kv(st: Sm70Csa2State, lid: int, kv: torch.Tensor) -> torch.Tensor:
    """Copy projected SWA KV into a static buffer; return a T-row view."""
    t = int(kv.shape[0])
    buf = st.verify_kv_buf.get(lid)
    if (
        buf is None
        or buf.dtype != kv.dtype
        or buf.device != kv.device
        or int(buf.shape[0]) < t
        or tuple(buf.shape[1:]) != tuple(kv.shape[1:])
    ):
        if _capturing():
            raise RuntimeError(
                f"SM70 CSA2 verify KV buffer missing during capture (L{lid})"
            )
        st.verify_kv_buf[lid] = _graph_buffer(
            torch.empty,
            (max(t, st.capture_t),) + tuple(kv.shape[1:]),
            dtype=kv.dtype,
            device=kv.device,
        )
        buf = st.verify_kv_buf[lid]
    buf[:t].copy_(kv)
    return buf[:t]


def _alloc_verify_buf(existing, t: int, shape, dtype, device, capture_t: int, what: str):
    if (
        existing is not None
        and existing.device == device
        and existing.dtype == dtype
        and int(existing.shape[0]) >= t
        and tuple(existing.shape[1:]) == tuple(shape)
    ):
        return existing
    if _capturing():
        raise RuntimeError(f"SM70 CSA2 verify {what} missing during capture")
    return _graph_buffer(
        torch.empty, (max(t, capture_t),) + tuple(shape), dtype=dtype, device=device
    )


def _note_verify_block(st: Sm70Csa2State, positions: torch.Tensor) -> None:
    """Record this verify block. Replay fills the same static buffers."""
    t = int(positions.shape[0])
    if t <= 0:
        return
    device = positions.device
    if positions.dtype != torch.int64:
        positions = positions.to(dtype=torch.int64)
    st.verify_positions = _alloc_verify_buf(
        st.verify_positions, t, (), torch.int64, device, st.capture_t, "positions"
    )
    st.verify_positions[:t].copy_(positions[:t])
    if (
        st.verify_open is None
        or st.verify_open.device != device
        or st.verify_t is None
        or st.verify_t.device != device
    ):
        if _capturing():
            raise RuntimeError("SM70 CSA2 verify commit flag missing during capture")
        st.verify_open = _graph_buffer(torch.zeros, 1, dtype=torch.int32, device=device)
        st.verify_t = _graph_buffer(torch.zeros, 1, dtype=torch.int32, device=device)
    st.verify_t.fill_(t)
    st.verify_open.fill_(1)


def _verify_pending_prev(st: Sm70Csa2State, lid: int, step: int):
    """Scratch previous-token projection. Step 0 copies the live slot."""
    live_kv = st.pending_kv[lid]
    device = live_kv.device
    dim = int(live_kv.shape[0])
    need = max(step + 1, _CAPTURE_T_MAX)
    st.verify_pending_kv_traj[lid] = _alloc_verify_buf(
        st.verify_pending_kv_traj.get(lid),
        need,
        (dim,),
        torch.float32,
        device,
        st.capture_t,
        "pending kv",
    )
    st.verify_pending_score_traj[lid] = _alloc_verify_buf(
        st.verify_pending_score_traj.get(lid),
        need,
        (dim,),
        torch.float32,
        device,
        st.capture_t,
        "pending score",
    )
    cur_kv = st.verify_pending_kv_cur.get(lid)
    if cur_kv is None or cur_kv.device != device or int(cur_kv.shape[0]) != dim:
        if _capturing():
            raise RuntimeError("SM70 CSA2 verify pending cursor missing during capture")
        cur_kv = _graph_buffer(torch.empty, dim, dtype=torch.float32, device=device)
        cur_score = _graph_buffer(torch.empty, dim, dtype=torch.float32, device=device)
        st.verify_pending_kv_cur[lid] = cur_kv
        st.verify_pending_score_cur[lid] = cur_score
    cur_score = st.verify_pending_score_cur[lid]
    if step == 0:
        cur_kv.copy_(live_kv)
        cur_score.copy_(st.pending_score[lid])
    return cur_kv, cur_score


def _verify_pending_save(st: Sm70Csa2State, lid: int, step: int, kv_cur, score_cur) -> None:
    """Remember this token's projection and advance the scratch cursor."""
    st.verify_pending_kv_traj[lid][step].copy_(kv_cur[0])
    st.verify_pending_score_traj[lid][step].copy_(score_cur[0])
    st.verify_pending_kv_cur[lid].copy_(kv_cur[0])
    st.verify_pending_score_cur[lid].copy_(score_cur[0])


def _snap_verify_row(
    st: Sm70Csa2State,
    lid: int,
    step: int,
    row_idx: torch.Tensor,
    table: torch.Tensor,
    orig_store: Dict[int, torch.Tensor],
    nbytes: int,
    store_index: bool = True,
) -> None:
    """Bytes in ``table[row]`` before this step's write, so commit can peel it."""
    device = table.device
    need = max(step + 1, _CAPTURE_T_MAX)
    orig = _alloc_verify_buf(
        orig_store.get(lid),
        need,
        (nbytes,),
        torch.uint8,
        device,
        st.capture_t,
        "row snapshot",
    )
    orig_store[lid] = orig
    if store_index:
        rows = _alloc_verify_buf(
            st.verify_row.get(lid),
            need,
            (),
            torch.int64,
            device,
            st.capture_t,
            "row index",
        )
        st.verify_row[lid] = rows
        rows[step].copy_(row_idx.reshape(()).to(dtype=torch.int64))
    orig[step].copy_(table.index_select(0, row_idx.reshape(1)).reshape(-1))


def _verify_low_ratio_sources(backend, layer, x, q_lora, positions, topo, st):
    """Packed decode kernels at T=γ+1. The live SWA ring is not written here.

    Ratio-2 pairing is the T=1 prefix op unrolled over static T (B.4c), against
    a scratch copy of ``pending_*``. A masked unique-row scatter would be a
    dynamic shape and cannot live in a CUDA graph. Commit peels the unaccepted
    tail after accept.
    """
    lid = int(layer.layer_id)
    t = int(x.shape[0])
    _note_verify_block(st, positions[:t])
    st.verify_kv[lid] = _ensure_verify_kv(st, lid, _project_swa_kv(layer, x))
    idxer = getattr(layer, "indexer", None)
    is_src = bool(getattr(idxer, "is_candidate_source", False))
    uses_cand = bool(getattr(idxer, "uses_candidates", False))
    if idxer is not None:
        topk_buf = st.verify_topk.get(lid)
        if topk_buf is None or int(topk_buf.shape[0]) < t:
            if _capturing():
                raise RuntimeError(
                    f"SM70 CSA2 verify topk buffer missing during capture (L{lid})"
                )
            k = int(topo["index_topk"])
            topk_buf = _graph_buffer(
                torch.full, (max(t, st.capture_t), k), -1, dtype=torch.int32, device=x.device
            )
            st.verify_topk[lid] = topk_buf
        st.prefill_topk[lid] = topk_buf[:t]
    if is_src:
        cand = st.verify_cand
        nblk = int(st.candidate_topk_blocks)
        if cand is None or int(cand.shape[0]) < t:
            if _capturing():
                raise RuntimeError("SM70 CSA2 verify cand buffer missing during capture")
            cand = _graph_buffer(
                torch.full,
                (max(t, st.capture_t), nblk),
                -1,
                dtype=torch.int32,
                device=x.device,
            )
            st.verify_cand = cand
        st.prefill_cand_ids = cand[:t]
    for i in range(t):
        if uses_cand:
            ids = st.prefill_cand_ids
            if ids is None or int(ids.shape[0]) <= i:
                raise RuntimeError(
                    f"SM70 CSA2 verify candidate ids missing at token {i} "
                    f"(have {None if ids is None else ids.shape[0]})"
                )
            st.cand_ids.copy_(ids[i : i + 1])
        q_i = None if q_lora is None else q_lora[i : i + 1]
        _decode_compress_and_index(
            backend,
            layer,
            x[i : i + 1],
            q_i,
            positions[i : i + 1],
            topo,
            st,
            verify_step=i,
        )
        if idxer is not None:
            st.prefill_topk[lid][i].copy_(st.topk[lid][0])
        if is_src:
            st.prefill_cand_ids[i].copy_(st.cand_ids[0])


def _decode_sparse_one(
    q_i: torch.Tensor,
    positions_i: torch.Tensor,
    topk_i: Optional[torch.Tensor],
    st: Sm70Csa2State,
    lid: int,
    kv_src: Optional[int],
    sink: torch.Tensor,
    scale: float,
    h: int,
) -> torch.Tensor:
    """One-query packed sparse attn. ``q_i`` is [H, D]; ``topk_i`` is [K] or None."""
    ring = st.swa_ring[lid]
    if _attn_glue_on():
        if kv_src is None or topk_i is None:
            kv_table = q_i.new_empty((0, KV_ROW_BYTES), dtype=torch.uint8)
            kv_idx = q_i.new_empty((0,), dtype=torch.int32)
        else:
            kv_table = st.kv_rows[kv_src]
            kv_idx = topk_i
        sink32 = sink if sink.dtype == torch.float32 else sink.float()
        return sparse_decode_indexed(
            q_i, ring, kv_table, kv_idx, positions_i, sink32[:h], scale
        )
    valid_s = st.ring_valid(positions_i)[0].to(torch.uint8)
    if kv_src is None or topk_i is None:
        kv_q = q_i.new_empty((0, KV_ROW_BYTES), dtype=torch.uint8)
        valid_k = q_i.new_empty((0,), dtype=torch.uint8)
    else:
        ci = topk_i
        kv_q = st.kv_rows[kv_src][ci.clamp_min(0).to(torch.int64)]
        valid_k = (ci >= 0).to(torch.uint8)
    return cuda_sparse_decode(q_i, ring, kv_q, valid_s, valid_k, sink[:h], scale)


def _pack_verify_swa(
    layer, kv: torch.Tensor, positions: torch.Tensor, st: Sm70Csa2State, lid: int
) -> torch.Tensor:
    """Pack this block with the decode packer. Does not write the live ring.

    Glue packs into a window-sized stage (the block is shorter than the window,
    so slots do not collide) and the dense rows are copied out for commit.
    """
    t = int(kv.shape[0])
    rope_dim = int(layer.qk_rope_head_dim)
    w = st.sliding_window
    device = kv.device
    st.verify_swa_packed[lid] = _alloc_verify_buf(
        st.verify_swa_packed.get(lid),
        max(t, _CAPTURE_T_MAX),
        (SWA_ROW_BYTES,),
        torch.uint8,
        device,
        st.capture_t,
        "packed swa",
    )
    if _attn_glue_on():
        stage = st.verify_swa_stage
        if (
            stage is None
            or stage.device != device
            or int(stage.shape[0]) != w
            or int(stage.shape[1]) != SWA_ROW_BYTES
        ):
            if _capturing():
                raise RuntimeError("SM70 CSA2 verify SWA stage missing during capture")
            stage = _graph_buffer(
                torch.empty, (w, SWA_ROW_BYTES), dtype=torch.uint8, device=device
            )
            st.verify_swa_stage = stage
        pack_swa_fp8_at(
            stage,
            kv,
            positions,
            freqs_table=_freqs_table(layer),
            rope_dim=rope_dim,
            dst_mod=w,
        )
        slots = torch.remainder(positions.to(dtype=torch.int64), w)
        packed = stage.index_select(0, slots)
    else:
        packed = pack_swa_fp8(
            kv, _freqs_for(layer, positions, check=False), rope_dim
        )
    st.verify_swa_packed[lid][:t].copy_(packed)
    return st.verify_swa_packed[lid][:t]


def _gather_verify_swa(
    st: Sm70Csa2State, lid: int, packed: torch.Tensor, positions: torch.Tensor
):
    """Causal window: committed ring history, then this block. Query i stops at i."""
    buf, origin = _prefill_swa_view(st, lid, positions[:1], packed)
    return _gather_swa(st, buf, origin, positions)


def _verify_sparse(
    q: torch.Tensor,
    layer,
    st: Sm70Csa2State,
    lid: int,
    positions: torch.Tensor,
    kv_src: Optional[int],
    idx_src: Optional[int],
    sink: torch.Tensor,
    scale: float,
    h: int,
) -> torch.Tensor:
    """Score the block against a scratch SWA window, then the prefill oracle.

    The live ring is not updated here. A direct ring gather aliases the next
    few draft positions onto slots that are still inside the 128-token window.
    Commit writes only the accepted prefix. The scored bonus token must match
    EXTEND last-row, which uses ``sparse_attention_rows``.
    """
    kv = st.verify_kv.pop(lid)
    attn = st.layers.get(lid)
    if attn is None:
        raise RuntimeError(f"SM70 CSA2 verify sparse ran before low-ratio sources (L{lid})")
    t = int(q.shape[0])
    _note_verify_block(st, positions[:t])
    topk = st.prefill_topk.get(idx_src) if idx_src is not None else None
    packed = _pack_verify_swa(attn, kv, positions, st, lid)
    tile_rows, valid = _gather_verify_swa(st, lid, packed, positions)
    payload, scales = tile_rows[..., :HEAD_DIM], tile_rows[..., HEAD_DIM:]
    keys = unpack_swa_fp8_ue8m0(payload, scales, q.dtype)
    if kv_src is not None and topk is not None and topk.shape[1] > 0:
        kv_all = st.kv_rows.get(kv_src)
        if kv_all is not None:
            ci = topk.to(torch.int64)
            packed = kv_all[ci.clamp_min(0)]
            flat = packed.reshape(-1, packed.shape[-1])
            kv_u = unpack_kv_fp4_e4m3(flat[:, :256], flat[:, 256:], q.dtype)
            kv_u = kv_u.view(t, ci.shape[1], -1)
            keys = torch.cat([keys, kv_u], dim=1)
            valid = torch.cat([valid, ci >= 0], dim=1)
    return _sparse_attention_rows_aligned(q, keys, valid, sink[:h], scale)


def _peel_uncommitted_rows(
    table: torch.Tensor,
    orig: torch.Tensor,
    rows: torch.Tensor,
    n: torch.Tensor,
    t: int,
    open_flag: torch.Tensor,
    limit: torch.Tensor,
) -> None:
    """Undo verify writes from the end. ``orig[i]`` is the row before step i.

    Peeling an unaccepted odd step restores the incomplete pair the accepted
    even step wrote. Peeling both steps restores the pre-block row.
    """
    n = n.reshape(())
    limit = limit.reshape(())
    open_flag = open_flag.reshape(())
    for i in range(t - 1, -1, -1):
        row = rows[i].to(dtype=torch.int64).reshape(1)
        current = table.index_select(0, row)
        peel = (i >= n) & (i < limit) & (open_flag != 0)
        src = torch.where(peel.reshape(1, 1), orig[i].reshape(1, -1), current)
        table.index_copy_(0, row, src)


def _commit_pending_vec(
    live: torch.Tensor,
    traj: torch.Tensor,
    n: torch.Tensor,
    t: int,
    open_flag: torch.Tensor,
    limit: torch.Tensor,
) -> None:
    """Pending becomes the last committed input. n == 0 leaves the live slot."""
    if t <= 0:
        return
    last = (n.reshape(()) - 1).clamp(min=0, max=t - 1).to(dtype=torch.int64)
    chosen = traj.index_select(0, last.reshape(1)).reshape(-1)
    use = (n.reshape(()) > 0) & (open_flag.reshape(()) != 0) & (limit.reshape(()) > 0)
    live.copy_(torch.where(use, chosen, live))


def _commit_swa_prefix(
    ring: torch.Tensor,
    packed: torch.Tensor,
    positions: torch.Tensor,
    n: torch.Tensor,
    t: int,
    w: int,
    open_flag: torch.Tensor,
    limit: torch.Tensor,
) -> None:
    """Write packed rows for the accepted inputs. The block is shorter than W."""
    if t <= 0:
        return
    pos = positions[:t].to(dtype=torch.int64)
    slots = torch.remainder(pos, w)
    idx = torch.arange(t, device=ring.device)
    pred = (idx < n.reshape(())) & (idx < limit.reshape(())) & (open_flag.reshape(()) != 0)
    current = ring.index_select(0, slots)
    src = torch.where(pred.unsqueeze(-1), packed[:t], current)
    # Consecutive positions in a window do not share a slot. A longer block
    # writes one row at a time so a later accepted token wins over an earlier one.
    if t <= w:
        ring.index_copy_(0, slots, src)
        return
    for i in range(t):
        slot = slots[i].reshape(1)
        current_i = ring.index_select(0, slot)
        src_i = torch.where(pred[i].reshape(1, 1), packed[i].reshape(1, -1), current_i)
        ring.index_copy_(0, slot, src_i)


def sm70_commit_target_verify(backend, commit_lens: torch.Tensor, num_positions: int) -> None:
    """Publish the accepted verify prefix into the live CSA2 state.

    ``commit_lens`` counts the anchor plus accepted drafts (the bonus has no
    key yet). Inputs at ``i >= commit_lens`` are dropped. A second call is a
    no-op until the next verify forward.
    """
    st = getattr(backend, "_sm70_csa2", None)
    if st is None or st.verify_open is None or st.verify_positions is None or st.verify_t is None:
        return
    t = int(num_positions)
    if t <= 0 or commit_lens is None or commit_lens.numel() == 0:
        return
    n = commit_lens.reshape(-1)[:1]
    open_flag = st.verify_open
    limit = st.verify_t
    positions = st.verify_positions
    for lid, orig in st.verify_kv_orig.items():
        rows = st.verify_row.get(lid)
        table = st.kv_rows.get(lid)
        if rows is None or table is None:
            continue
        _peel_uncommitted_rows(table, orig, rows, n, t, open_flag, limit)
    for lid, orig in st.verify_index_orig.items():
        rows = st.verify_row.get(lid)
        table = st.index_rows.get(lid)
        if rows is None or table is None:
            continue
        _peel_uncommitted_rows(table, orig, rows, n, t, open_flag, limit)
    for lid, traj in st.verify_pending_kv_traj.items():
        live = st.pending_kv.get(lid)
        if live is not None:
            _commit_pending_vec(live, traj, n, t, open_flag, limit)
    for lid, traj in st.verify_pending_score_traj.items():
        live = st.pending_score.get(lid)
        if live is not None:
            _commit_pending_vec(live, traj, n, t, open_flag, limit)
    w = int(st.sliding_window)
    for lid, ring in st.swa_ring.items():
        packed = st.verify_swa_packed.get(lid)
        if packed is None:
            continue
        _commit_swa_prefix(ring, packed, positions, n, t, w, open_flag, limit)
    st.verify_open.fill_(0)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #


def _is_decode(forward_batch, t: int) -> bool:
    mode = getattr(forward_batch, "forward_mode", None)
    return (
        t == 1
        and callable(getattr(mode, "is_decode", None))
        and bool(mode.is_decode())
    )


def _use_packed_csa2(forward_batch, t: int) -> bool:
    """T=1 decode, or TARGET_VERIFY at any T (packed decode kernels, no host sync)."""
    if _is_decode(forward_batch, t):
        return True
    mode = getattr(forward_batch, "forward_mode", None)
    return callable(getattr(mode, "is_target_verify", None)) and bool(
        mode.is_target_verify()
    )


def sm70_forward_low_ratio_sources(backend, layer, x, q_lora, positions, forward_batch):
    """Compress + pack + hierarchical indexer. Decode bs=1 and chunked prefill."""
    if x.shape[0] == 0:
        return
    bs = int(getattr(forward_batch, "batch_size", 1) or 1)
    if bs != 1:
        raise RuntimeError(
            f"SM70 CSA2 v1 supports decode/prefill batch size 1, got {bs}"
        )
    topo = _topology(layer, backend)
    st = get_state(backend)
    _ensure_buffers(st, layer, topo, x.device, backend)
    lid = int(layer.layer_id)
    st.last_positions = positions
    st.layers[lid] = layer
    t = int(x.shape[0])
    mode = getattr(forward_batch, "forward_mode", None)
    if _is_decode(forward_batch, t):
        _decode_low_ratio_sources(backend, layer, x, q_lora, positions, topo, st)
    elif callable(getattr(mode, "is_target_verify", None)) and bool(
        mode.is_target_verify()
    ):
        _verify_low_ratio_sources(backend, layer, x, q_lora, positions, topo, st)
    else:
        _prefill_low_ratio_sources(
            backend, layer, x, q_lora, positions, forward_batch, topo, st
        )


def sm70_forward_sparse(
    backend,
    q: torch.Tensor,
    layer,
    forward_batch,
    compress_ratio: int,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    """Attend Q to the SWA window plus selected compressed KV. Inverse RoPE stays in the model."""
    st = get_state(backend)
    lid = int(layer.layer_id)
    topo = _topology(layer, backend)
    positions = st.last_positions
    if positions is None or lid not in st.swa_ring:
        raise RuntimeError("SM70 CSA2 sparse forward ran before low-ratio sources")

    if q.ndim == 4:
        q = q.squeeze(1)
    # q is [T, H, D]
    t, h, _d = q.shape
    kv_src = idx_src = None
    if compress_ratio in (1, 2):
        kv_src = kv_source_for(lid, int(compress_ratio), topo["kv_sources"])
        idx_src = index_source_for(lid, int(compress_ratio), topo["index_sources"])

    sink = attn_sink.float() if attn_sink is not None else layer.attn_sink.float()
    scale = getattr(layer, "softmax_scale", None)
    scale = float(scale if scale is not None else layer.scaling)

    mode = getattr(forward_batch, "forward_mode", None)
    if _is_decode(forward_batch, int(t)):
        topk_i = None if idx_src is None else st.topk[idx_src][0]
        out = _decode_sparse_one(
            q[0], positions, topk_i, st, lid, kv_src, sink, scale, h
        )
        if envs.SGLANG_DEBUG_DSV41_PROBE_STATS.get():
            from sglang.srt.debug.dsv41_probe_stats import record

            record(f"L{lid}.csa2.sparse_o", out)
        return out.unsqueeze(0)
    if callable(getattr(mode, "is_target_verify", None)) and bool(
        mode.is_target_verify()
    ):
        return _verify_sparse(
            q, layer, st, lid, positions, kv_src, idx_src, sink, scale, h
        )

    # Prefill: packed dequant+softmax (same kernel math as decode). Tile so
    # gathered SWA/KV stay tens of MiB. Torch unpack+einsum is the fallback.
    buf, origin = st.prefill_swa.pop(lid)
    topk = st.prefill_topk.get(idx_src) if idx_src is not None else None
    kv_all = st.kv_rows.get(kv_src) if kv_src is not None else None
    have_kv = topk is not None and kv_all is not None and topk.shape[1] > 0
    dspark_shared = dspark_shared_block_attn(layer, st, lid)
    use_torch = dspark_shared or bool(envs.SGLANG_DSV41_TORCH_PREFILL_SPARSE.get())
    shared_k = shared_valid = None
    if dspark_shared:
        payload, scales = buf[..., :HEAD_DIM], buf[..., HEAD_DIM:]
        shared_k = unpack_swa_fp8_ue8m0(payload, scales, q.dtype)
        shared_valid = dspark_shared_block_valid(buf, origin)
    tile = min(
        _PREFILL_SPARSE_Q_TILE if use_torch else _PREFILL_SPARSE_Q_TILE_PACKED,
        t,
    )
    outs = []
    swa_k_probe = None
    for s in range(0, t, tile):
        e = min(s + tile, t)
        nq = e - s
        if dspark_shared:
            keys = shared_k.unsqueeze(0).expand(nq, -1, -1)
            valid = shared_valid.unsqueeze(0).expand(nq, -1)
            if swa_k_probe is None:
                swa_k_probe = shared_k
        else:
            tile_rows, tile_valid = _gather_swa(st, buf, origin, positions[s:e])
            if use_torch:
                payload, scales = tile_rows[..., :HEAD_DIM], tile_rows[..., HEAD_DIM:]
                swa_k = unpack_swa_fp8_ue8m0(payload, scales, q.dtype)
                if swa_k_probe is None:
                    swa_k_probe = swa_k
                keys = swa_k
                valid = tile_valid
            else:
                kv_rows = q.new_empty((nq, 0, KV_ROW_BYTES), dtype=torch.uint8)
                kv_valid = q.new_empty((nq, 0), dtype=torch.uint8)
                if have_kv:
                    ci = topk[s:e].to(torch.int64)
                    kv_rows = kv_all[ci.clamp_min(0)]
                    kv_valid = (ci >= 0).to(torch.uint8)
                outs.append(
                    cuda_sparse_prefill(
                        q[s:e],
                        tile_rows,
                        kv_rows,
                        tile_valid.to(torch.uint8),
                        kv_valid,
                        sink[:h],
                        scale,
                    )
                )
                continue
        if have_kv:
            ci = topk[s:e].to(torch.int64)
            packed = kv_all[ci.clamp_min(0)]
            flat = packed.reshape(-1, packed.shape[-1])
            kv_u = unpack_kv_fp4_e4m3(flat[:, :256], flat[:, 256:], q.dtype)
            kv_u = kv_u.view(nq, ci.shape[1], -1)
            keys = torch.cat([keys, kv_u], dim=1)
            valid = torch.cat([valid, ci >= 0], dim=1)
        outs.append(_sparse_attention_rows_aligned(q[s:e], keys, valid, sink[:h], scale))
    o = outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)
    if envs.SGLANG_DEBUG_DSV41_PROBE_STATS.get():
        from sglang.srt.debug.dsv41_probe_stats import record, record_q_split

        n_local = int(
            getattr(layer, "tp_q_head_num", getattr(layer, "n_local_heads", h))
        )
        rope_dim = int(getattr(layer, "qk_rope_head_dim", 64) or 64)
        record_q_split(f"L{lid}.csa2.q", q, n_local, rope_dim)
        if swa_k_probe is not None and t <= _PREFILL_SPARSE_Q_TILE:
            record(f"L{lid}.csa2.swa_k", swa_k_probe)
            if swa_k_probe.ndim >= 2 and swa_k_probe.shape[-1] > rope_dim:
                record(f"L{lid}.csa2.swa_k_nope", swa_k_probe[..., :-rope_dim])
                record(f"L{lid}.csa2.swa_k_rope", swa_k_probe[..., -rope_dim:])
        record(f"L{lid}.csa2.sparse_o", o)
    return o
