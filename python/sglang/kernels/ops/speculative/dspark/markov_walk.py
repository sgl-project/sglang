# SPDX-License-Identifier: Apache-2.0
"""Fused int8 DSpark markov walk for a vanilla markov head (rank 256) on Hopper.

One persistent cooperative launch per round replaces the default walk's gamma x
(W2 GEMV + add + sample) launches (``jit/csrc/speculative/dspark_markov_walk_*``)::

    prev = anchor
    for k < gamma:
        logit_k = bf16(base_k + W2 W1[prev])
        tok_k = argmax(logit_k) (greedy) or a sample of softmax(logit_k / T)
        prev = tok_k

W2 is int8 with one fp32 scale per row and W1[prev] two int8 planes
u ~= s_hi q_hi + s_lo q_lo with s_lo = s_hi / 256, so near-ties can differ from
the default walk on the bf16 weights. The logits are rounded to bf16 once; argmax
and sampler read exactly those values, and ``corrected_out`` receives exactly
those bits, which is what the verifier rebuilds q = softmax(corrected / T) from.

Layout modes, from V and the SM count (tiles = ceil(V / (#SMs x 64)) wgmma tiles
of 64 rows per CTA):

* standard (tiles <= 18): bs 1 -> single, bs 2..4 -> small_batch, bs 5..64 -> wgmma;
* big vocabulary (18 < tiles <= 32): wgmma for every bs (its W2 partly streams
  from L2; single / small_batch keep all of W2 on chip and cannot hold it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import msgspec
import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MARKOV_RANK = 256  # hard-wired in every kernel: 528-B W1 rows, 8 k32 chunks
MAX_STEPS = 256  # kMaxSteps: the Philox counters pack the step into 8 bits
MAX_BS = 64  # kMaxB of small_batch / wgmma
SMALL_BATCH_MAX_BS = (
    4  # serving dispatch; the small_batch launcher itself takes up to 8
)
ROWS_CTA = (
    1152  # standard layouts: 72 x 16-row (single / small_batch), 18 x 64-row (wgmma)
)
WGMMA_TILE_ROWS = 64  # one wgmma N = 64 B operand, 16 KiB
W1_ROW_BYTES = 528  # 256 q_hi + 256 q_lo + s_hi f32 + s_lo f32 + 8 pad
WGMMA_RES, WGMMA_RING = 11, 2  # wgmma template parameters kRes, kRing
WGMMA_TILES = ROWS_CTA // WGMMA_TILE_ROWS
WGMMA_MAX_TILES = 32  # the stream mask is a 32-bit word
# Measured on H100 SXM (132 SMs): the 18-tile layout streams these tiles.
WGMMA_MASK = sum(1 << t for t in (1, 2, 4, 7, 9, 11, 14))

# wgmma: in every 64-row tile, wgmma column n = 8 i + 2 tig + e holds tile row
# rho(n), so a thread's 16 columns are the 8-row runs 8 tig.. and 32 + 8 tig..
WGMMA_RHO = [
    (
        8 * ((n % 8) // 2) + 2 * (n // 8) + n % 2
        if n // 8 < 4
        else 32 + 8 * ((n % 8) // 2) + 2 * (n // 8 - 4) + n % 2
    )
    for n in range(64)
]


def _require_sm90() -> None:
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("The DSpark markov-walk kernels need sm_90 (Hopper).")


@cache_once
def _jit_single_module() -> Module:
    _require_sm90()
    return load_jit(
        "dspark_markov_walk_single",
        cuda_files=["speculative/dspark_markov_walk_single.cuh"],
        cuda_wrappers=[("walk", "dspark_markov_walk::single::walk")],
    )


@cache_once
def _jit_small_batch_module() -> Module:
    _require_sm90()
    return load_jit(
        "dspark_markov_walk_small_batch",
        cuda_files=["speculative/dspark_markov_walk_small_batch.cuh"],
        cuda_wrappers=[("walk", "dspark_markov_walk::small_batch::walk")],
    )


@cache_once
def _jit_wgmma_module(tiles: int, res: int, stream_mask: int) -> Module:
    _require_sm90()
    if not res < tiles <= WGMMA_MAX_TILES or bin(stream_mask).count("1") != tiles - res:
        raise ValueError(f"bad wgmma layout: {tiles=}, {res=}, mask={stream_mask:#x}")
    args = make_cpp_args(tiles, res, WGMMA_RING, stream_mask)
    return load_jit(
        "dspark_markov_walk_wgmma",
        *args,
        cuda_files=["speculative/dspark_markov_walk_wgmma.cuh"],
        cuda_wrappers=[("walk", f"dspark_markov_walk::wgmma::walk<{args}>")],
    )


def markov_walk_single(
    *,
    frag: torch.Tensor,
    row_scale: torch.Tensor,
    w1q: torch.Tensor,
    base_logits: torch.Tensor,
    anchor: torch.Tensor,
    tokens_out: torch.Tensor,
    corrected_out: Optional[torch.Tensor],
    state: torch.Tensor,
    temps: torch.Tensor,
    num_steps: int,
    valid_rows: int,
    seed: int,
) -> None:
    """bs = 1 walk with W2 held in registers + SMEM.

    base_logits bf16 [1, kb >= num_steps, V]; anchor int64 [1]; temps fp32 [1]
    (<= 0 or NaN: greedy); tokens_out int64 [num_steps]; corrected_out None or
    bf16 shaped like base_logits, written for T > 0 only. The weights are those
    of MarkovWalkWeights; state is int64 [state_words(kind, S)] for S >= num_steps
    (MarkovWalker.states holds one per kind, S = gamma rounded up to 16).
    """
    _jit_single_module().walk(
        frag,
        row_scale,
        w1q,
        base_logits,
        anchor,
        tokens_out,
        corrected_out,
        state,
        temps,
        num_steps,
        valid_rows,
        seed,
    )


def markov_walk_small_batch(
    *,
    frag: torch.Tensor,
    row_scale: torch.Tensor,
    w1q: torch.Tensor,
    base_logits: torch.Tensor,
    anchor: torch.Tensor,
    tokens_out: torch.Tensor,
    corrected_out: Optional[torch.Tensor],
    state: torch.Tensor,
    temps: torch.Tensor,
    num_steps: int,
    valid_rows: int,
    seed: int,
) -> None:
    """bs 1..8 walk sharing one on-chip W2.

    base_logits bf16 [bs, kb, V]; anchor int64 [bs]; temps fp32 [bs]; tokens_out
    int64 [bs * num_steps], row-major; otherwise as markov_walk_single.
    """
    _jit_small_batch_module().walk(
        frag,
        row_scale,
        w1q,
        base_logits,
        anchor,
        tokens_out,
        corrected_out,
        state,
        temps,
        num_steps,
        valid_rows,
        seed,
    )


def markov_walk_wgmma(
    *,
    w2_res: torch.Tensor,
    w2_str: torch.Tensor,
    row_scale: torch.Tensor,
    w1f: torch.Tensor,
    base_logits: torch.Tensor,
    anchor: torch.Tensor,
    tokens_out: torch.Tensor,
    corrected_out: Optional[torch.Tensor],
    state: torch.Tensor,
    temps: torch.Tensor,
    num_steps: int,
    valid_rows: int,
    seed: int,
    stream_mask: int,
) -> None:
    """bs 1..64 wgmma walk; W2 tiles SMEM-resident or streamed from L2.

    w2_res / w2_str int8 [grid, res | tiles - res, 16384]; bit t of stream_mask
    = tile t streams. Batch tensors as markov_walk_small_batch.
    """
    module = _jit_wgmma_module(
        w2_res.shape[1] + w2_str.shape[1], w2_res.shape[1], stream_mask
    )
    module.walk(
        w2_res,
        w2_str,
        row_scale,
        w1f,
        base_logits,
        anchor,
        tokens_out,
        corrected_out,
        state,
        temps,
        num_steps,
        valid_rows,
        seed,
    )


# ===== Weight layouts =====


def _wgmma_tiles_for(vocab: int, num_sms: int) -> int:
    """wgmma tiles per CTA for `vocab` rows on `num_sms` co-resident CTAs: the
    standard 18 whenever that fits (then single / small_batch fit too), else the minimum."""
    need = -(-vocab // (num_sms * WGMMA_TILE_ROWS))
    if need > WGMMA_MAX_TILES:
        raise ValueError(
            f"V = {vocab} needs {need} wgmma tiles per CTA on {num_sms} SMs; at most "
            f"{WGMMA_MAX_TILES} are supported"
        )
    return max(need, WGMMA_TILES)


def _wgmma_stream_mask(tiles: int, res: int = WGMMA_RES, ring: int = WGMMA_RING) -> int:
    """A stream mask (bit t = tile t streams) with tiles - res streamed tiles.

    Warpgroup p takes tiles p, p + 4, ... (with 64 requests: even / odd tiles
    per pair). Residents per warpgroup are proportional to its tile count
    (largest remainder; ties go to the parity class with fewer residents, which
    balances the bs > 32 split); the first `ring` warpgroups of (1, 2, 3, 0)
    open on a streamed tile, as they consume the ring prefill, the others on a
    resident one; streamed tiles are spread evenly over each warpgroup.
    """
    if not 1 <= res < tiles <= WGMMA_MAX_TILES:
        raise ValueError(
            f"need 1 <= res < tiles <= {WGMMA_MAX_TILES}: {res=}, {tiles=}"
        )
    seqs = [list(range(p, tiles, 4)) for p in range(4)]
    quota = [res * len(sq) / tiles for sq in seqs]
    resident = [int(q) for q in quota]
    for _ in range(res - sum(resident)):
        parity = [resident[0] + resident[2], resident[1] + resident[3]]
        best = max(
            (p for p in range(4) if resident[p] < len(seqs[p])),
            key=lambda p: (round(quota[p] - resident[p], 9), -parity[p % 2], -p),
        )
        resident[best] += 1
    opens_streamed = {(1, 2, 3, 0)[i] for i in range(min(ring, 4))}
    mask = 0
    for p, sq in enumerate(seqs):
        n, n_res = len(sq), resident[p]
        n_str = n - n_res
        if p in opens_streamed:
            pos = {i * n // n_str for i in range(n_str)} if n_str else set()
        else:
            pos = set(range(n)) - {i * n // n_res for i in range(n_res)}
        for j in pos:
            mask |= 1 << sq[j]
    assert bin(mask).count("1") == tiles - res and mask < (1 << tiles), hex(mask)
    return mask


def _wgmma_mask_for(tiles: int) -> int:
    return WGMMA_MASK if tiles == WGMMA_TILES else _wgmma_stream_mask(tiles)


def quantize_markov(
    w1: torch.Tensor, w2: torch.Tensor, *, chunk_rows: int = 16384
) -> dict[str, torch.Tensor]:
    """The int8 planes every layout is cut from, chunk_rows rows at a time.

    W2 [V, 256]: per-row scale max|w| / 127 (at least 1e-30), q_w2 =
    round(w / scale) clamped to +-127. W1 [V, 256]: u ~= s_hi q_hi + s_lo q_lo,
    s_hi = max|u| / 127, and q_lo quantizes the residual with the FIXED ratio
    s_lo = s_hi / 256 (wgmma combines the planes as s_lo (256 d_hi + d_lo)).
    """
    vocab, dev = w2.shape[0], w2.device
    out = {
        "q_w2": torch.empty(vocab, MARKOV_RANK, dtype=torch.int8, device=dev),
        "row_scale": torch.empty(vocab, dtype=torch.float32, device=dev),
        "q_hi": torch.empty(vocab, MARKOV_RANK, dtype=torch.int8, device=dev),
        "q_lo": torch.empty(vocab, MARKOV_RANK, dtype=torch.int8, device=dev),
        "s_hi": torch.empty(vocab, dtype=torch.float32, device=dev),
        "s_lo": torch.empty(vocab, dtype=torch.float32, device=dev),
    }
    for a in range(0, vocab, chunk_rows):
        b = min(a + chunk_rows, vocab)
        w = w2[a:b].float()
        scale = (w.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-30)
        out["q_w2"][a:b] = (w / scale).round().clamp(-127, 127)
        out["row_scale"][a:b] = scale.view(-1)
        u = w1[a:b].float()
        s_hi = (u.abs().amax(1, keepdim=True) / 127.0).clamp_min(1e-30)
        q_hi = (u / s_hi).round().clamp(-127, 127)
        s_lo = s_hi / 256.0
        out["q_lo"][a:b] = ((u - s_hi * q_hi) / s_lo).round().clamp(-127, 127)
        out["q_hi"][a:b] = q_hi
        out["s_hi"][a:b] = s_hi.view(-1)
        out["s_lo"][a:b] = s_lo.view(-1)
    return out


def _build_frag(qp: torch.Tensor) -> torch.Tensor:
    """single / small_batch: padded int8 rows -> m16n8k32 A fragments [tiles, 8 chunks,
    32 lanes, 16 B]; lane bytes 4q..4q+3 = row g + 8 (q & 1), cols
    32 j + 4 tig + 16 (q >> 1) + i."""
    lane = torch.arange(32, device=qp.device)
    g, tig = lane // 4, lane % 4
    byte = torch.arange(16, device=qp.device)
    quad, i = byte // 4, byte % 4
    rows = g[:, None] + 8 * (quad[None] & 1)
    cols = 4 * tig[:, None] + 16 * (quad[None] >> 1) + i[None]
    n_tiles = qp.shape[0] // 16
    frag = qp.view(n_tiles, 16, 8, 32).permute(0, 2, 1, 3)[:, :, rows, cols]
    return frag.contiguous().view(-1)


def _scales_as_bytes(planes: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([planes["s_hi"], planes["s_lo"]], 1).contiguous()


def _build_w1q(planes: dict[str, torch.Tensor]) -> torch.Tensor:
    """single / small_batch W1 rows: q_hi[256] | q_lo[256] | s_hi f32 | s_lo f32 | pad."""
    vocab, dev = planes["q_hi"].shape[0], planes["q_hi"].device
    row = torch.zeros(vocab, W1_ROW_BYTES, dtype=torch.uint8, device=dev)
    row[:, :512] = torch.cat([planes["q_hi"], planes["q_lo"]], 1).view(torch.uint8)
    row[:, 512:520] = _scales_as_bytes(planes).view(torch.uint8)
    return row


def _build_wgmma_tiles(
    qp: torch.Tensor, *, tiles: int, stream_mask: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """wgmma: padded int8 rows -> per CTA 64-row tiles as canonical K-major wgmma
    B operands [j][8 col groups][2 k16 cores][8 cols][16 B], columns permuted by
    rho; split into resident / streamed [grid, RES | tiles - RES, 16384]."""
    n_tiles = qp.shape[0] // WGMMA_TILE_ROWS
    t = qp.view(n_tiles, WGMMA_TILE_ROWS, MARKOV_RANK)[:, WGMMA_RHO]
    v = t.view(n_tiles, 8, 8, 8, 2, 16).permute(0, 3, 1, 4, 2, 5)
    v = v.contiguous().view(n_tiles // tiles, tiles, 16384)
    streamed = torch.tensor(
        [(stream_mask >> i) & 1 == 1 for i in range(tiles)], device=qp.device
    )
    return v[:, ~streamed].contiguous(), v[:, streamed].contiguous()


def _build_w1f(planes: dict[str, torch.Tensor]) -> torch.Tensor:
    """wgmma W1 rows in A-fragment order: per chunk j and tig, q_hi(32j+4tig..) |
    q_lo(..) | q_hi(32j+16+4tig..) | q_lo(..), 4 B each, so the per-step gather
    is 8 x LDG.128 straight into the fragment registers; then s_hi, s_lo."""
    vocab, dev = planes["q_hi"].shape[0], planes["q_hi"].device
    hi = planes["q_hi"].view(vocab, 8, 2, 4, 4)
    lo = planes["q_lo"].view(vocab, 8, 2, 4, 4)
    fr = torch.stack([hi[:, :, 0], lo[:, :, 0], hi[:, :, 1], lo[:, :, 1]], dim=3)
    row = torch.zeros(vocab, W1_ROW_BYTES, dtype=torch.uint8, device=dev)
    row[:, :512] = fr.reshape(vocab, 512).view(torch.uint8)
    row[:, 512:520] = _scales_as_bytes(planes).view(torch.uint8)
    return row


class MarkovWalkWeights(msgspec.Struct, frozen=True):
    """The int8 layouts the kernels read; single / small_batch ones in standard mode only."""

    vocab: int
    tiles: int
    stream_mask: int
    row_scale: torch.Tensor  # fp32 [rows_pad], natural order, 1e-30 on padding
    w2_res: torch.Tensor  # int8 [grid, WGMMA_RES, 16384]
    w2_str: torch.Tensor  # int8 [grid, tiles - WGMMA_RES, 16384]
    w1f: torch.Tensor  # uint8 [V, 528]
    frag: Optional[torch.Tensor]  # int8 [rows_pad * 256]
    w1q: Optional[torch.Tensor]  # uint8 [V, 528]

    @property
    def big_vocab(self) -> bool:
        return self.tiles != WGMMA_TILES


def _build_weights(
    w1: torch.Tensor, w2: torch.Tensor, *, num_sms: int
) -> MarkovWalkWeights:
    vocab = w2.shape[0]
    tiles = _wgmma_tiles_for(vocab, num_sms)
    stream_mask = _wgmma_mask_for(tiles)
    rows_cta = tiles * WGMMA_TILE_ROWS
    rows_pad = -(-vocab // rows_cta) * rows_cta
    planes = quantize_markov(w1, w2)
    qp = torch.zeros(rows_pad, MARKOV_RANK, dtype=torch.int8, device=w2.device)
    qp[:vocab] = planes.pop("q_w2")
    big_vocab = tiles != WGMMA_TILES
    w2_res, w2_str = _build_wgmma_tiles(qp, tiles=tiles, stream_mask=stream_mask)
    frag = None if big_vocab else _build_frag(qp)
    del qp
    row_scale = torch.full((rows_pad,), 1e-30, dtype=torch.float32, device=w2.device)
    row_scale[:vocab] = planes["row_scale"]
    return MarkovWalkWeights(
        vocab=vocab,
        tiles=tiles,
        stream_mask=stream_mask,
        row_scale=row_scale,
        w2_res=w2_res,
        w2_str=w2_str,
        w1f=_build_w1f(planes),
        frag=frag,
        w1q=None if big_vocab else _build_w1q(planes),
    )


def _kernel_for(bs: int, *, big_vocab: bool) -> str:
    if not 1 <= bs <= MAX_BS:
        raise ValueError(f"markov walk kernels take 1 <= bs <= {MAX_BS}, got {bs}")
    if big_vocab or bs > SMALL_BATCH_MAX_BS:
        return "wgmma"
    return "single" if bs == 1 else "small_batch"


def state_words(kind: str, num_steps: int) -> int:
    """int64 words of a kernel's state for up to num_steps steps: round | pad | two
    sets (round parity) of {key, count} pairs, one 16-B pair per step for single,
    one 128-B pair per (step, request) for small_batch / wgmma."""
    if kind == "single":
        return 2 + 4 * num_steps
    return 16 + 2 * num_steps * MAX_BS * 16


def _aligned_zeros(n: int, *, device: torch.device) -> torch.Tensor:
    t = torch.zeros(n, dtype=torch.int64, device=device)
    assert t.data_ptr() % 128 == 0, "state buffer not 128-B aligned"
    return t


class MarkovWalker:
    """Weight layouts + persistent state of the int8 markov walk for one
    drafter (one gamma, one GPU).

    Serving contract of walk():

    * base_logits: bf16 [bs, gamma, V], contiguous (the unpadded lm_head output).
    * anchor: int64 [bs], any stride; clamped into [0, V) in the kernels.
    * temps: fp32 [bs] or None (all greedy); T <= 0 or NaN = greedy, T > 0 is
      clamped to [1e-5, 1e4]. Read in-kernel: one graph serves every mix.
    * tokens_out: int64 [bs * gamma], contiguous, row-major b * gamma + k.
    * corrected_out: None or bf16 [bs * gamma, V], written ONLY for requests with
      T > 0; zero-initialise it once (stale rows reach the verifier's softmax).
    * One zeroed state buffer per kernel family; its round counter advances
      in-kernel, so every replay of a captured graph draws fresh noise from the
      fixed seed. Calls must be serialized on one stream.
    * No host sync and no allocation: capturable into a CUDA graph after
      warmup().
    * A request whose logits are all NaN / -inf (a padded graph row) still gets
      tokens in [0, V).
    """

    def __init__(
        self,
        w1: torch.Tensor,
        w2: torch.Tensor,
        *,
        gamma: int,
        max_bs: int = MAX_BS,
        device: Optional[torch.device] = None,
        seed: int = 0x5EED,
    ):
        device = torch.device(device if device is not None else w2.device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if not 1 <= gamma <= MAX_STEPS:
            raise ValueError(f"gamma must be in [1, {MAX_STEPS}], got {gamma}")
        if not 1 <= max_bs <= MAX_BS:
            raise ValueError(f"max_bs must be in [1, {MAX_BS}], got {max_bs}")
        vocab, rank = w2.shape
        if rank != MARKOV_RANK or tuple(w1.shape) != (vocab, MARKOV_RANK):
            raise ValueError(
                f"need rank-{MARKOV_RANK} w1 / w2, got {w1.shape} {w2.shape}"
            )
        if vocab % 8 != 0:
            raise ValueError(f"vocab {vocab} is not a multiple of 8")
        self.gamma = gamma
        self.max_bs = max_bs
        self.vocab = vocab
        self.device = device
        # One Philox key per kernel family: their counters overlap, and a request
        # moving between families (bs 1 -> a 2..4 batch) must not replay noise.
        base_seed = seed & ((1 << 63) - 1)
        salts = {
            "single": 0,
            "small_batch": 0x2545F4914F6CDD1D,
            "wgmma": 0x5851F42D4C957F2D,
        }
        self.seeds = {k: (base_seed ^ s) & ((1 << 63) - 1) for k, s in salts.items()}
        with torch.cuda.device(device), torch.no_grad():
            self._load_modules_and_weights(w1, w2)
        self._warm = False

    def _load_modules_and_weights(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        _require_sm90()
        num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.weights = _build_weights(
            w1.to(self.device), w2.to(self.device), num_sms=num_sms
        )
        kinds = (
            ("wgmma",) if self.weights.big_vocab else ("single", "small_batch", "wgmma")
        )
        # Capacity in whole 16-step units: on H100 a 7-step set stride made wgmma
        # 1.2-1.7% slower at bs 5-16 than the 16-step one, with identical code.
        steps = -(-self.gamma // 16) * 16
        self.states = {
            k: _aligned_zeros(state_words(k, steps), device=self.device) for k in kinds
        }
        self.anchor_buf = torch.zeros(MAX_BS, dtype=torch.int64, device=self.device)
        # temps_buf is caller-writable: stage temperatures into it in-graph.
        self.temps_buf = torch.zeros(MAX_BS, dtype=torch.float32, device=self.device)
        self._greedy_temps = torch.zeros_like(self.temps_buf)
        if not self.weights.big_vocab:
            _jit_single_module()
            _jit_small_batch_module()
        _jit_wgmma_module(self.weights.tiles, WGMMA_RES, self.weights.stream_mask)

    def supports(self, bs: int) -> bool:
        return 1 <= bs <= self.max_bs

    def kernel_for(self, bs: int) -> str:
        return _kernel_for(bs, big_vocab=self.weights.big_vocab)

    def walk(
        self,
        base_logits: torch.Tensor,
        anchor: torch.Tensor,
        temps: Optional[torch.Tensor],
        tokens_out: torch.Tensor,
        corrected_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One round; returns tokens_out viewed [bs, gamma]. bs is a Python int,
        so the kernel choice is fixed per captured graph bucket."""
        bs, steps, vocab = base_logits.shape
        if not self.supports(bs):
            raise ValueError(
                f"bs {bs} outside [1, {self.max_bs}]: use the default walk"
            )
        if steps != self.gamma or vocab != self.vocab:
            raise ValueError(
                f"base_logits {tuple(base_logits.shape)} != "
                f"[bs, {self.gamma}, {self.vocab}]"
            )
        # A [1] anchor / temps would broadcast in copy_.
        if anchor.numel() != bs or (temps is not None and temps.numel() != bs):
            raise ValueError(f"anchor / temps must have bs = {bs} elements")
        if not self._warm and torch.cuda.is_current_stream_capturing():
            # A cooperative kernel's first launch loads its module, which
            # deadlocks while another cooperative grid spins on its exchange.
            raise RuntimeError("MarkovWalker.warmup() must run before graph capture")
        anchors = self.anchor_buf[:bs]
        anchors.copy_(anchor.reshape(-1))
        if temps is None:
            temps = self._greedy_temps[:bs]
        elif not (
            temps.dtype == torch.float32
            and temps.dim() == 1
            and temps.is_contiguous()
            and temps.device == self.device
        ):
            self.temps_buf[:bs].copy_(temps.reshape(-1))
            temps = self.temps_buf[:bs]
        corrected = (
            None if corrected_out is None else corrected_out.view(bs, steps, vocab)
        )
        self._launch(
            kind=self.kernel_for(bs),
            base_logits=base_logits,
            anchor=anchors,
            temps=temps,
            tokens_out=tokens_out,
            corrected=corrected,
        )
        return tokens_out.view(bs, steps)

    def _launch(
        self,
        *,
        kind: str,
        base_logits: torch.Tensor,
        anchor: torch.Tensor,
        temps: torch.Tensor,
        tokens_out: torch.Tensor,
        corrected: Optional[torch.Tensor],
    ) -> None:
        w = self.weights
        common = dict(
            row_scale=w.row_scale,
            base_logits=base_logits,
            anchor=anchor,
            tokens_out=tokens_out,
            corrected_out=corrected,
            state=self.states[kind],
            temps=temps,
            num_steps=self.gamma,
            valid_rows=self.vocab,
            seed=self.seeds[kind],
        )
        if kind == "wgmma":
            markov_walk_wgmma(
                w2_res=w.w2_res,
                w2_str=w.w2_str,
                w1f=w.w1f,
                stream_mask=w.stream_mask,
                **common,
            )
        elif kind == "small_batch":
            markov_walk_small_batch(frag=w.frag, w1q=w.w1q, **common)
        else:
            markov_walk_single(frag=w.frag, w1q=w.w1q, **common)

    def warmup(
        self,
        bs_list: Optional[Sequence[int]] = None,
        corrected_out: Optional[torch.Tensor] = None,
    ) -> None:
        """Run every kernel greedy and sampling once; required before any graph
        capture (see walk()). corrected_out, bf16 [>= max(bs_list) * gamma, V],
        replaces a scratch buffer. Every walk's tokens are range-checked."""
        if bs_list is None:
            cand = (1, 2, min(4, self.max_bs), 5, 33, self.max_bs)
            bs_list = sorted({b for b in cand if self.supports(b)})
        n, k, vocab = max(bs_list), self.gamma, self.vocab
        with torch.cuda.device(self.device), torch.no_grad():
            gen = torch.Generator(device=self.device).manual_seed(0)
            base = torch.randn(
                n, k, vocab, device=self.device, generator=gen, dtype=torch.bfloat16
            )
            base.mul_(3)
            anchor = torch.randint(0, vocab, (n,), device=self.device, generator=gen)
            tokens = torch.empty(n * k, dtype=torch.int64, device=self.device)
            if corrected_out is None:
                corr = torch.zeros(
                    n * k, vocab, dtype=torch.bfloat16, device=self.device
                )
            else:
                corr = corrected_out.view(-1, vocab)[: n * k]
            ones = torch.ones(n, dtype=torch.float32, device=self.device)
            bad = torch.zeros((), dtype=torch.int64, device=self.device)
            for b in bs_list:
                for t in (None, ones[:b]):
                    out = self.walk(
                        base_logits=base[:b],
                        anchor=anchor[:b],
                        temps=t,
                        tokens_out=tokens[: b * k],
                        corrected_out=corr[: b * k],
                    )
                    bad += ((out < 0) | (out >= vocab)).sum()
            n_bad = int(bad)
            del base, anchor, tokens, corr, ones, bad
        torch.cuda.empty_cache()
        if n_bad:
            raise RuntimeError(f"markov walk warmup: {n_bad} out-of-range tokens")
        self._warm = True

    def memory_bytes(self) -> int:
        w = self.weights
        tensors = [w.row_scale, w.w2_res, w.w2_str, w.w1f, w.frag, w.w1q]
        tensors += [self.anchor_buf, self.temps_buf, self._greedy_temps]
        tensors += list(self.states.values())
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)
