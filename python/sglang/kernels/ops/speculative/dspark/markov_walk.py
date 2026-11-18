# SPDX-License-Identifier: Apache-2.0
"""Fused int8 DSpark markov walk for a vanilla markov head (rank 256) on Hopper.

One persistent cooperative launch per round replaces the default walk's gamma x
(W2 GEMV + add + sample) launches (``jit/csrc/speculative/dspark_markov_walk.cuh``)::

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

W2 is cut into 64-row wgmma tiles, max(18, ceil(V / (#SMs x 64))) <= 32 per CTA;
each CTA keeps WGMMA_RES of them in shared memory and streams the others from L2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import msgspec
import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

MARKOV_RANK = 256  # hard-wired in the kernel: 528-B W1 rows, 8 k32 chunks
MAX_STEPS = 256  # kMaxSteps: the Philox counters pack the step into 8 bits
MAX_BS = 64  # kMaxBatch
W1_ROW_BYTES = 528  # 256 q_hi + 256 q_lo + s_hi f32 + s_lo f32 + 8 pad
WGMMA_TILE_ROWS = 64  # one wgmma N = 64 B operand, 16 KiB
WGMMA_RES, WGMMA_RING = 11, 2  # wgmma template parameters kRes, kRing
WGMMA_TILES = 18  # per CTA when V fits: 132 SMs x 18 x 64 = 152,064 rows
WGMMA_MAX_TILES = 32  # the stream mask is a 32-bit word
# Measured on H100 SXM (132 SMs): the 18-tile layout streams these tiles.
WGMMA_MASK = sum(1 << t for t in (1, 2, 4, 7, 9, 11, 14))

# In every 64-row wgmma tile, column n = 8 i + 2 tig + e holds tile row rho(n),
# so a thread's 16 columns are the 8-row runs 8 tig.. and 32 + 8 tig..
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
        raise RuntimeError("The DSpark markov-walk kernel needs sm_90 (Hopper).")


@cache_once
def _jit_walk_module(tiles: int, stream_mask: int) -> Module:
    _require_sm90()
    args = make_cpp_args(tiles, WGMMA_RES, WGMMA_RING, stream_mask)
    return load_jit(
        "dspark_markov_walk",
        *args,
        cuda_files=["speculative/dspark_markov_walk.cuh"],
        cuda_wrappers=[("walk", f"dspark_markov_walk::walk<{args}>")],
    )


def _wgmma_tiles_for(vocab: int, num_sms: int) -> int:
    """wgmma tiles per CTA: the standard 18 whenever V fits, else the minimum."""
    need = -(-vocab // (num_sms * WGMMA_TILE_ROWS))
    if need > WGMMA_MAX_TILES:
        raise ValueError(
            f"vocab {vocab} needs {need} wgmma tiles per CTA on {num_sms} SMs; at "
            f"most {WGMMA_MAX_TILES} are supported"
        )
    return max(need, WGMMA_TILES)


def _wgmma_stream_mask(tiles: int) -> int:
    """A stream mask (bit t = tile t streams) with tiles - WGMMA_RES streamed tiles.

    Warpgroup p takes tiles p, p + 4, ... (with 64 requests: even / odd tiles
    per pair). Residents per warpgroup are proportional to its tile count
    (largest remainder; ties go to the parity class with fewer residents, which
    balances the bs > 32 split); the first WGMMA_RING warpgroups of (1, 2, 3, 0)
    open on a streamed tile, as they consume the ring prefill, the others on a
    resident one; streamed tiles are spread evenly over each warpgroup.
    """
    res = WGMMA_RES
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
    opens_streamed = {(1, 2, 3, 0)[i] for i in range(min(WGMMA_RING, 4))}
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
    return mask


def quantize_markov(
    w1: torch.Tensor, w2: torch.Tensor, *, chunk_rows: int = 16384
) -> dict[str, torch.Tensor]:
    """The int8 planes the layouts are cut from, chunk_rows rows at a time.

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


def _build_w1f(planes: dict[str, torch.Tensor]) -> torch.Tensor:
    """W1 rows [V, 528]: the 512 q bytes in A-fragment order (per chunk j and tig,
    q_hi(32j+4tig..) | q_lo(..) | q_hi(32j+16+4tig..) | q_lo(..), 4 B each, so the
    per-step gather is 8 LDG.128 straight into the fragment registers) | s_hi f32 |
    s_lo f32 | pad."""
    vocab, dev = planes["q_hi"].shape[0], planes["q_hi"].device
    hi = planes["q_hi"].view(vocab, 8, 2, 4, 4)
    lo = planes["q_lo"].view(vocab, 8, 2, 4, 4)
    q = torch.stack([hi[:, :, 0], lo[:, :, 0], hi[:, :, 1], lo[:, :, 1]], dim=3)
    row = torch.zeros(vocab, W1_ROW_BYTES, dtype=torch.uint8, device=dev)
    row[:, :512] = q.reshape(vocab, 512).view(torch.uint8)
    scales = torch.stack([planes["s_hi"], planes["s_lo"]], 1).contiguous()
    row[:, 512:520] = scales.view(torch.uint8)
    return row


def _build_wgmma_tiles(
    qp: torch.Tensor, *, tiles: int, stream_mask: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Padded int8 rows -> per CTA 64-row tiles as canonical K-major wgmma B
    operands [j][8 col groups][2 k16 cores][8 cols][16 B], columns permuted by
    rho; split into resident / streamed [grid, RES | tiles - RES, 16384]."""
    n_tiles = qp.shape[0] // WGMMA_TILE_ROWS
    t = qp.view(n_tiles, WGMMA_TILE_ROWS, MARKOV_RANK)[:, WGMMA_RHO]
    v = t.view(n_tiles, 8, 8, 8, 2, 16).permute(0, 3, 1, 4, 2, 5)
    v = v.contiguous().view(n_tiles // tiles, tiles, 16384)
    streamed = torch.tensor(
        [(stream_mask >> i) & 1 == 1 for i in range(tiles)], device=qp.device
    )
    return v[:, ~streamed].contiguous(), v[:, streamed].contiguous()


class MarkovWalkWeights(msgspec.Struct, frozen=True):
    """The int8 layouts the kernel reads."""

    vocab: int
    tiles: int  # wgmma tiles per CTA
    stream_mask: int
    row_scale: torch.Tensor  # fp32 [rows_pad], natural order, 1e-30 on padding
    w2_res: torch.Tensor  # int8 [grid, WGMMA_RES, 16384]
    w2_str: torch.Tensor  # int8 [grid, tiles - WGMMA_RES, 16384]
    w1f: torch.Tensor  # uint8 [V, 528]


def _build_weights(
    w1: torch.Tensor, w2: torch.Tensor, *, num_sms: int
) -> MarkovWalkWeights:
    vocab = w2.shape[0]
    tiles = _wgmma_tiles_for(vocab, num_sms)
    stream_mask = WGMMA_MASK if tiles == WGMMA_TILES else _wgmma_stream_mask(tiles)
    rows_cta = tiles * WGMMA_TILE_ROWS
    rows_pad = -(-vocab // rows_cta) * rows_cta
    planes = quantize_markov(w1, w2)
    qp = torch.zeros(rows_pad, MARKOV_RANK, dtype=torch.int8, device=w2.device)
    qp[:vocab] = planes.pop("q_w2")
    w2_res, w2_str = _build_wgmma_tiles(qp, tiles=tiles, stream_mask=stream_mask)
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
    )


class MarkovWalkState(msgspec.Struct, frozen=True):
    """The kernel's exchange state, zeroed once and never reset: its round
    counter advances in-kernel."""

    steps: int  # capacity: serves any num_steps <= steps
    buffer: torch.Tensor

    @staticmethod
    def words(steps: int) -> int:
        """int64 words: round | pad to 128 B | two sets (round parity) of one
        128-B {key, count} pair per (step, request)."""
        return 16 + 2 * steps * MAX_BS * 16

    @classmethod
    def zeros(cls, *, steps: int, device: torch.device) -> MarkovWalkState:
        buffer = torch.zeros(cls.words(steps), dtype=torch.int64, device=device)
        return cls(steps=steps, buffer=buffer)


_SEED_MASK = (1 << 63) - 1


def markov_walk(
    *,
    weights: MarkovWalkWeights,
    state: MarkovWalkState,
    base_logits: torch.Tensor,
    anchor: torch.Tensor,
    temps: torch.Tensor,
    tokens_out: torch.Tensor,
    corrected_out: Optional[torch.Tensor],
    seed: int,
) -> None:
    """One round of the int8 markov walk.

    base_logits bf16 [bs, num_steps, V], V = weights.vocab; anchor int64 [bs];
    temps fp32 [bs] (<= 0 or NaN: greedy, T > 0 clamped to [1e-5, 1e4]);
    tokens_out int64 [bs * num_steps], row-major; corrected_out None or bf16
    shaped like base_logits, written for T > 0 only; num_steps <= state.steps.
    The launcher checks every tensor. No host sync and no allocation: capturable
    into a CUDA graph once the kernel has run (MarkovWalker.warmup).
    """
    num_steps = base_logits.shape[1]
    _jit_walk_module(weights.tiles, weights.stream_mask).walk(
        weights.w2_res,
        weights.w2_str,
        weights.row_scale,
        weights.w1f,
        base_logits,
        anchor,
        tokens_out,
        corrected_out,
        state.buffer,
        temps,
        num_steps,
        weights.vocab,
        seed & _SEED_MASK,
    )


class MarkovWalker:
    """Weight layouts + persistent state of the int8 markov walk for one
    drafter (one gamma, one GPU).

    Serving contract of walk():

    * base_logits: bf16 [bs, gamma, V], contiguous (the unpadded lm_head output).
    * anchor: int64 [bs], any stride (staged into anchor_buf); clamped into
      [0, V) in the kernel.
    * temps: contiguous fp32 [bs] or None (all greedy); T <= 0 or NaN = greedy,
      T > 0 is clamped to [1e-5, 1e4]. Read in-kernel: one graph serves every
      temperature.
    * tokens_out: int64 [bs * gamma], contiguous, row-major b * gamma + k.
    * corrected_out: None or bf16 [bs * gamma, V], written ONLY for requests with
      T > 0; other rows keep their contents.
    * One zeroed state buffer; its round counter advances in-kernel, so every
      replay of a captured graph draws fresh noise from the fixed seed. Calls
      must be serialized on one stream.
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
        if not 1 <= max_bs <= MAX_BS:
            raise ValueError(f"max_bs must be in [1, {MAX_BS}], got {max_bs}")
        vocab, rank = w2.shape
        if rank != MARKOV_RANK or tuple(w1.shape) != (vocab, MARKOV_RANK):
            raise ValueError(
                f"need rank-{MARKOV_RANK} w1 / w2, got {w1.shape} {w2.shape}"
            )
        self.gamma = gamma
        self.max_bs = max_bs
        self.vocab = vocab
        self.device = device
        self.seed = seed & _SEED_MASK
        with torch.cuda.device(device), torch.no_grad():
            self._load_modules_and_weights(w1, w2)
        self._warm = False

    def _load_modules_and_weights(self, w1: torch.Tensor, w2: torch.Tensor) -> None:
        _require_sm90()
        num_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.weights = _build_weights(
            w1.to(self.device), w2.to(self.device), num_sms=num_sms
        )
        # Capacity in whole 16-step units: on H100 the exchange-line placement of
        # an exact-gamma state cost up to 1.7% against the 16-step one.
        self.state = MarkovWalkState.zeros(
            steps=-(-self.gamma // 16) * 16, device=self.device
        )
        self.anchor_buf = torch.zeros(MAX_BS, dtype=torch.int64, device=self.device)
        self._greedy_temps = torch.zeros(
            MAX_BS, dtype=torch.float32, device=self.device
        )
        _jit_walk_module(self.weights.tiles, self.weights.stream_mask)

    def supports(self, bs: int) -> bool:
        return 1 <= bs <= self.max_bs

    def walk(
        self,
        base_logits: torch.Tensor,
        anchor: torch.Tensor,
        temps: Optional[torch.Tensor],
        tokens_out: torch.Tensor,
        corrected_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """One round; returns tokens_out viewed [bs, steps]."""
        bs, steps, vocab = base_logits.shape
        if not self._warm and torch.cuda.is_current_stream_capturing():
            # A cooperative kernel's first launch loads its module, which
            # deadlocks while another cooperative grid spins on its exchange.
            raise RuntimeError("MarkovWalker.warmup() must run before graph capture")
        if anchor.shape != (bs,):  # copy_ would broadcast a [1] anchor
            raise ValueError(f"anchor must be [{bs}], got {tuple(anchor.shape)}")
        anchors = self.anchor_buf[:bs]
        anchors.copy_(anchor)
        markov_walk(
            weights=self.weights,
            state=self.state,
            base_logits=base_logits,
            anchor=anchors,
            temps=self._greedy_temps[:bs] if temps is None else temps,
            tokens_out=tokens_out,
            corrected_out=(
                None if corrected_out is None else corrected_out.view(bs, steps, vocab)
            ),
            seed=self.seed,
        )
        return tokens_out.view(bs, steps)

    def warmup(self, bs_list: Optional[Sequence[int]] = None) -> None:
        """Run the kernel greedy and sampling once; required before any graph
        capture (see walk()). Every walk's tokens are range-checked."""
        if bs_list is None:
            candidates = (1, 33, self.max_bs)
            bs_list = sorted({b for b in candidates if self.supports(b)})
        n, k, vocab = max(bs_list), self.gamma, self.vocab
        with torch.cuda.device(self.device), torch.no_grad():
            gen = torch.Generator(device=self.device).manual_seed(0)
            base = torch.randn(
                n, k, vocab, device=self.device, generator=gen, dtype=torch.bfloat16
            )
            base.mul_(3)
            anchor = torch.randint(0, vocab, (n,), device=self.device, generator=gen)
            tokens = torch.empty(n * k, dtype=torch.int64, device=self.device)
            corr = torch.zeros(n * k, vocab, dtype=torch.bfloat16, device=self.device)
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
        tensors = [w.row_scale, w.w2_res, w.w2_str, w.w1f]
        tensors += [self.anchor_buf, self._greedy_temps, self.state.buffer]
        return sum(t.numel() * t.element_size() for t in tensors)
