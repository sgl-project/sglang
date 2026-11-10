"""CPU/torch reference for DeepSeek-V4.1 Compressed Sparse Attention 2 on SM70.

This module is the *correctness oracle* for the V100 (SM70) DSV4.1 attention
path: sliding-window attention plus ratio-1/2 compressed KV selected by a
hierarchical FP4 indexer, with the main compressed KV held as E2M1 with one
E4M3 scale per 16 channels. Every formula is a direct port of one of

* the model's own reference implementation, ``inference/model.py`` and
  ``inference/kernel.py`` of ``deepseek-ai/DeepSeek-V4.1-Flash`` (ground truth), and
* the torch paths of upstream SGLang PR #38798 (git ref ``sgl/dsv41``):
  ``layers/attention/dsv4/dsv41_sparse.py``, ``torch_quant.py``,
  ``candidate_torch.py``, ``indexer.select_candidate_blocks`` and the
  ``_low_ratio_*`` torch paths of ``deepseek_v4_backend.py``.

It is deliberately slow and explicit. It imports nothing from the SGLang runtime
so it runs on a CPU-only interpreter. Later SM70 CUDA/JIT kernels are tested
against it; anything a fused kernel wants to drop (an intermediate rounding, a
pinned candidate block, the sink term) must first be shown harmless here.

Precision model
---------------
``act_dtype`` is the activation/storage dtype: ``torch.float16`` on SM70 (no
BF16), ``torch.bfloat16`` for parity runs against the upstream torch path.
Every GEMM is emulated as fp32-accumulate with a single rounding to the output
dtype (what cuBLAS does for fp16/bf16 inputs), so rounding points match the
serving path up to reduction order.

No replay/approximation flags exist here: encoder/decoder SWA bounded replay is
off by construction, every query sees its exact causal window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

FP4_MAX = 6.0  # largest E2M1 magnitude
FP8_MAX = 448.0  # largest E4M3FN magnitude
E4M3_MIN_NORMAL_SCALE = 2.0**-9  # smallest positive E4M3 value (subnormal)
FP4_UE8M0_AMAX_FLOOR = 6 * 2.0**-126  # HF fp4_quant_kernel, e8m0 branch
FP4_E4M3_AMAX_FLOOR = 6 * 2.0**-9  # HF fp4_quant_kernel, fp8-scale branch
FP8_ACT_AMAX_FLOOR = 1.0e-4  # HF act_quant_kernel
INDEX_FP4_BLOCK = 32  # indexer q/k: one UE8M0 scale per 32
KV_FP4_BLOCK = 16  # compressed KV: one E4M3 scale per 16
SWA_FP8_BLOCK = 32  # HF window K: act_quant(kv, fp8_block_size=32, "ue8m0")

# E2M1 magnitudes by 3-bit code (sign is bit 3).
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_DEV: Dict[str, torch.Tensor] = {}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def _hf_flash_compress_ratios() -> Tuple[int, ...]:
    # config.json: 0,0 | 2 x 18 (layers 2..19) | 1 x 20 (layers 20..39) | 0,0,0 (MTP)
    return (0, 0) + (2,) * 18 + (1,) * 20 + (0, 0, 0)


@dataclass(frozen=True)
class CSA2Config:
    """The attention-relevant slice of DeepSeek-V4.1-Flash ``config.json``.

    Dimensions can be shrunk for tests; the *topology* fields default to the
    real model and are what the tests pin down.
    """

    num_layers: int = 40
    hidden_size: int = 5120
    n_heads: int = 64
    head_dim: int = 512
    rope_head_dim: int = 64
    q_lora_rank: int = 1280
    rms_norm_eps: float = 1e-20
    sliding_window: int = 128
    compress_ratios: Tuple[int, ...] = field(default_factory=_hf_flash_compress_ratios)
    kv_source_layer_ids: Tuple[int, ...] = (2, 8, 14, 20)
    index_source_layer_ids: Tuple[int, ...] = (2, 8, 14, 20, 24, 28, 32, 36)
    candidate_source_layer_id: int = 20
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    index_topk: int = 512
    index_n_heads: int = 32
    index_head_dim: int = 128
    # RoPE: pure-SWA layers use rope_theta without YaRN; compressed layers use
    # compress_rope_theta with YaRN (HF Attention.__init__).
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_factor: float = 16.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    original_seq_len: int = 65536
    max_seq_len: int = 4096  # size of the freqs tables built by the reference

    def __post_init__(self):
        assert self.head_dim % KV_FP4_BLOCK == 0
        assert self.index_head_dim % INDEX_FP4_BLOCK == 0
        assert self.rope_head_dim % 2 == 0 and self.rope_head_dim <= self.head_dim
        assert len(self.compress_ratios) >= self.num_layers
        for lid in range(self.num_layers):
            assert self.compress_ratios[lid] in (0, 1, 2), (
                f"layer {lid}: CSA2 reference covers ratios 0/1/2 only"
            )
        for lid in self.kv_source_layer_ids:
            assert self.compress_ratios[lid] > 0
        for lid in self.index_source_layer_ids:
            assert self.compress_ratios[lid] > 0

    # ---- topology -------------------------------------------------------- #

    def ratio(self, layer_id: int) -> int:
        return self.compress_ratios[layer_id]

    def kv_source_for(self, layer_id: int) -> Optional[int]:
        """The layer whose compressed latents ``layer_id`` attends through: the
        nearest kv_source at or before it with the same ratio (HF: "layers
        sharing a ratio also share one compressed KV ... produced by the first").
        """
        r = self.ratio(layer_id)
        if r == 0:
            return None
        src = [
            s for s in self.kv_source_layer_ids if s <= layer_id and self.ratio(s) == r
        ]
        assert src, f"layer {layer_id} (ratio {r}) has no kv source"
        return max(src)

    def index_source_for(self, layer_id: int) -> Optional[int]:
        """The layer whose indexer top-k ``layer_id`` reuses."""
        r = self.ratio(layer_id)
        if r == 0:
            return None
        src = [
            s
            for s in self.index_source_layer_ids
            if s <= layer_id and self.ratio(s) == r
        ]
        assert src, f"layer {layer_id} (ratio {r}) has no index source"
        return max(src)

    def is_kv_source(self, layer_id: int) -> bool:
        return layer_id in self.kv_source_layer_ids

    def is_index_source(self, layer_id: int) -> bool:
        return layer_id in self.index_source_layer_ids

    def is_candidate_source(self, layer_id: int) -> bool:
        return layer_id == self.candidate_source_layer_id

    def uses_candidates(self, layer_id: int) -> bool:
        # HF Indexer.__init__: 0 <= candidate_source_layer < layer_id
        return (
            self.is_index_source(layer_id)
            and 0 <= self.candidate_source_layer_id < layer_id
        )

    @property
    def index_head_weight_scale(self) -> float:
        # HF: weights_proj(x) * (softmax_scale * n_heads ** -0.5)
        return self.index_head_dim**-0.5 * self.index_n_heads**-0.5

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5


# --------------------------------------------------------------------------- #
# Emulated GEMM / norm
# --------------------------------------------------------------------------- #


def linear(x: torch.Tensor, w: torch.Tensor, out_dtype: Optional[torch.dtype] = None):
    """``x @ w.T`` with fp32 accumulation and one rounding to ``out_dtype``
    (default: x.dtype). Weights are used at their stored values."""
    out_dtype = x.dtype if out_dtype is None else out_dtype
    return (x.float() @ w.float().t()).to(out_dtype)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF/upstream RMSNorm: fp32 statistics, fp32 weight multiply, cast at the end."""
    dtype = x.dtype
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * xf).to(dtype)


# --------------------------------------------------------------------------- #
# FP4 / FP8 fake quantization (exact ports)
# --------------------------------------------------------------------------- #


def ceil_pow2(x: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(x)) for positive finite fp32 ``x`` on the IEEE bits
    (HF ``fast_round_scale``; upstream ``torch_quant.ceil_pow2``)."""
    x = x.float().contiguous()
    bits = x.view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) - 127
    exponent = exponent + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((exponent + 127) << 23).view(torch.float32)


def round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 values in [-6, 6] to the E2M1 grid, ties to even
    (``cvt.rn.satfinite.e2m1x2`` semantics; upstream ``round_fp4``)."""
    magnitude = x.abs()
    step = torch.where(magnitude < 2.0, 0.5, torch.where(magnitude < 4.0, 1.0, 2.0))
    return torch.round(magnitude / step) * step * torch.sign(x)


def fake_quant_fp4_ue8m0(x: torch.Tensor, block_size: int = INDEX_FP4_BLOCK):
    """Indexer q/k: E2M1 with one UE8M0 (power-of-two) scale per ``block_size``.

    HF ``fp4_act_quant(x, 32, inplace=True)`` / upstream ``fake_quant_fp4``:
    amax floored at 6 * 2**-126, scale = ceil_pow2(amax / 6).
    """
    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(FP4_UE8M0_AMAX_FLOOR)
    scale = ceil_pow2(amax * (1.0 / FP4_MAX))
    deq = round_e2m1((blocks / scale).clamp(-FP4_MAX, FP4_MAX)) * scale
    return deq.flatten(-2).to(x.dtype)


def fake_quant_fp4_e4m3(x: torch.Tensor, block_size: int = KV_FP4_BLOCK):
    """Compressed KV: E2M1 with one E4M3FN scale per ``block_size`` (16).

    HF ``fp4_act_quant(latent, 16, inplace=True, scale_dtype=float8_e4m3fn)`` /
    upstream ``fake_quant_compressed_kv``: amax floored at 6 * 2**-9 so an all-zero
    block keeps a nonzero scale, scale = e4m3(amax / 6) clamped to [2**-9, 448].
    """
    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale = (amax * (1.0 / FP4_MAX)).clamp(min=E4M3_MIN_NORMAL_SCALE, max=FP8_MAX)
    scale = scale.to(torch.float8_e4m3fn).float()
    deq = round_e2m1((blocks / scale).clamp(-FP4_MAX, FP4_MAX)) * scale
    return deq.flatten(-2).to(x.dtype)


def fake_quant_fp8_ue8m0(x: torch.Tensor, block_size: int = SWA_FP8_BLOCK):
    """Sliding-window K as the model keeps it: E4M3FN with one UE8M0 scale per
    32 over the whole post-RoPE vector (HF ``act_quant(kv, 32, "ue8m0",
    float8_e8m0fnu, inplace=True)``): amax floored at 1e-4, scale =
    ceil_pow2(amax / 448)."""
    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(FP8_ACT_AMAX_FLOOR)
    scale = ceil_pow2(amax * (1.0 / FP8_MAX))
    q = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return (q * scale).flatten(-2).to(x.dtype)


def fake_quant_fp8_flashmla(
    x: torch.Tensor, nope_dim: int = 448, tile: int = 64, eps: float = 1e-8
):
    """Upstream SGLang's FlashMLA cache format, for gap measurement only: the
    first ``nope_dim`` channels as E4M3FN with one UE8M0 scale per 64-tile
    (``quant_k_cache._quant_k_cache_fused_kernel``, amax floored at 1e-8), the
    RoPE tail stored unquantized. Not what the model was trained with."""
    nope, rope = x[..., :nope_dim], x[..., nope_dim:]
    blocks = nope.float().unflatten(-1, (-1, tile))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    scale = ceil_pow2(amax * (1.0 / FP8_MAX))
    q = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()
    return torch.cat([(q * scale).flatten(-2).to(x.dtype), rope], dim=-1)


# --------------------------------------------------------------------------- #
# Byte layouts for the SM70 caches (bit-exact with the fake quant above)
# --------------------------------------------------------------------------- #


def _e2m1_codes(scaled: torch.Tensor) -> torch.Tensor:
    """fp32 values already on the E2M1 grid (|v| in E2M1_VALUES) -> uint8 codes
    ``idx | sign << 3``; -0 is encoded as +0 (upstream ``_fp4_e2m1_code_rne``)."""
    mag = scaled.abs()
    idx = torch.zeros_like(mag, dtype=torch.uint8)
    for code, v in enumerate(E2M1_VALUES.tolist()):
        idx = torch.where(mag == v, torch.tensor(code, dtype=torch.uint8), idx)
    sign = ((scaled < 0) & (idx != 0)).to(torch.uint8)
    return idx | (sign << 3)


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """[..., N] codes -> [..., N // 2] bytes, even element in the low nibble
    (upstream ``_quantize_fp4_indexer_kernel``)."""
    pairs = codes.unflatten(-1, (-1, 2))
    return (pairs[..., 0] & 0x0F) | ((pairs[..., 1] & 0x0F) << 4)


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).flatten(-2)


def _decode_e2m1(codes: torch.Tensor) -> torch.Tensor:
    key = str(codes.device)
    table = _E2M1_DEV.get(key)
    if table is None or table.device != codes.device:
        table = E2M1_VALUES.to(device=codes.device)
        _E2M1_DEV[key] = table
    mag = table[(codes & 0x7).to(torch.int64)]
    return torch.where((codes & 0x8) != 0, -mag, mag)


def pack_kv_fp4_e4m3(x: torch.Tensor, block_size: int = KV_FP4_BLOCK):
    """Compressed KV row(s) [..., D] -> (payload uint8 [..., D // 2],
    scales uint8 (E4M3FN bits) [..., D // block_size]).

    For D = 512: 256 + 32 = 288 bytes per compressed position."""
    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    scale8 = (amax * (1.0 / FP4_MAX)).clamp(min=E4M3_MIN_NORMAL_SCALE, max=FP8_MAX)
    scale8 = scale8.to(torch.float8_e4m3fn)
    scaled = round_e2m1((blocks / scale8.float()).clamp(-FP4_MAX, FP4_MAX))
    codes = _e2m1_codes(scaled).flatten(-2)
    return _pack_nibbles(codes), scale8.squeeze(-1).view(torch.uint8)


def unpack_kv_fp4_e4m3(
    payload: torch.Tensor, scales: torch.Tensor, out_dtype: torch.dtype, block_size=16
):
    codes = _unpack_nibbles(payload)
    vals = _decode_e2m1(codes).unflatten(-1, (-1, block_size))
    scale = scales.view(torch.float8_e4m3fn).float().unsqueeze(-1)
    return (vals * scale).flatten(-2).to(out_dtype)


def pack_index_fp4_ue8m0(x: torch.Tensor, block_size: int = INDEX_FP4_BLOCK):
    """Index key row(s) [..., 128] -> (payload uint8 [..., 64], scale exponents
    uint8 [..., 4]) = the 68-byte index-pool row of upstream (``64 payload + 4
    scale``), scale byte = biased exponent of the power-of-two scale."""
    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(FP4_UE8M0_AMAX_FLOOR)
    scale = ceil_pow2(amax * (1.0 / FP4_MAX))
    scaled = round_e2m1((blocks / scale).clamp(-FP4_MAX, FP4_MAX))
    codes = _e2m1_codes(scaled).flatten(-2)
    exp_bits = ((scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8).squeeze(-1)
    return _pack_nibbles(codes), exp_bits


def unpack_index_fp4_ue8m0(
    payload: torch.Tensor, exps: torch.Tensor, out_dtype: torch.dtype, block_size=32
):
    codes = _unpack_nibbles(payload)
    vals = _decode_e2m1(codes).unflatten(-1, (-1, block_size))
    scale = (exps.to(torch.int32) << 23).view(torch.float32).unsqueeze(-1)
    return (vals * scale).flatten(-2).to(out_dtype)


def pack_swa_fp8_ue8m0(x: torch.Tensor, block_size: int = SWA_FP8_BLOCK):
    """Window K row(s) [..., D] -> (E4M3FN bytes [..., D], scale exponents
    uint8 [..., D // 32]). For D = 512: 512 + 16 = 528 bytes per token."""
    blocks = x.float().unflatten(-1, (-1, block_size))
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(FP8_ACT_AMAX_FLOOR)
    scale = ceil_pow2(amax * (1.0 / FP8_MAX))
    q = (blocks / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    exp_bits = ((scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8).squeeze(-1)
    return q.flatten(-2).view(torch.uint8), exp_bits


def unpack_swa_fp8_ue8m0(
    payload: torch.Tensor, exps: torch.Tensor, out_dtype: torch.dtype, block_size=32
):
    vals = payload.view(torch.float8_e4m3fn).float().unflatten(-1, (-1, block_size))
    scale = (exps.to(torch.int32) << 23).view(torch.float32).unsqueeze(-1)
    return (vals * scale).flatten(-2).to(out_dtype)


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> torch.Tensor:
    """HF ``precompute_freqs_cis``: complex64 [seqlen, dim // 2], YaRN when
    ``original_seq_len > 0``."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return (
                dim
                * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen, dtype=torch.float32), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def layer_freqs_cis(cfg: CSA2Config, layer_id: int) -> torch.Tensor:
    """Per-layer table (HF ``Attention.__init__``): compressed layers rotate with
    ``compress_rope_theta`` + YaRN, pure-SWA layers with ``rope_theta`` and no YaRN."""
    if cfg.ratio(layer_id):
        return precompute_freqs_cis(
            cfg.rope_head_dim,
            cfg.max_seq_len,
            cfg.original_seq_len,
            cfg.compress_rope_theta,
            cfg.rope_factor,
            cfg.beta_fast,
            cfg.beta_slow,
        )
    return precompute_freqs_cis(
        cfg.rope_head_dim,
        cfg.max_seq_len,
        0,
        cfg.rope_theta,
        cfg.rope_factor,
        cfg.beta_fast,
        cfg.beta_slow,
    )


def rope_tail(
    x: torch.Tensor, freqs: torch.Tensor, rope_dim: int, inverse: bool = False
) -> torch.Tensor:
    """Rotate the last ``rope_dim`` features of ``x`` [T, ..., D] with complex
    ``freqs`` [T, rope_dim // 2]; adjacent pairs form one complex number. Computed
    in fp32 and rounded back to ``x.dtype`` (HF ``apply_rotary_emb``, upstream
    ``rope_tail``): this rounding sits *before* every FP4/FP8 quant below."""
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    tc = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs.conj() if inverse else freqs
    f = f.view(x.shape[0], *([1] * (x.ndim - 2)), rope_dim // 2)
    rotated = torch.view_as_real(tc * f).flatten(-2).to(x.dtype)
    return torch.cat([head, rotated], dim=-1)


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #


@dataclass
class CSA2LayerWeights:
    """One attention layer's weights, as the serving path holds them.

    ``act_dtype`` for everything the bf16 checkpoint stores, except the ratio-2
    compressor projections which HF promotes to fp32 (values still bf16-exact).
    """

    wq_a: torch.Tensor  # [q_lora_rank, hidden]
    q_norm: torch.Tensor  # [q_lora_rank]
    wq_b: torch.Tensor  # [n_heads * head_dim, q_lora_rank]
    wkv: torch.Tensor  # [head_dim, hidden]
    kv_norm: torch.Tensor  # [head_dim]
    attn_sink: torch.Tensor  # [n_heads] fp32
    # kv source only
    c_wkv: Optional[torch.Tensor] = None  # [head_dim, hidden]
    c_wgate: Optional[torch.Tensor] = None  # [head_dim, hidden], ratio 2 only
    c_norm: Optional[torch.Tensor] = None  # [head_dim]
    # index source only
    i_wq_b: Optional[torch.Tensor] = (
        None  # [index_n_heads * index_head_dim, q_lora_rank]
    )
    i_weights_proj: Optional[torch.Tensor] = None  # [index_n_heads, hidden]
    # index-key owner (kv source that is also an index source)
    i_wk: Optional[torch.Tensor] = None  # [index_head_dim, head_dim]
    i_k_norm: Optional[torch.Tensor] = None  # [index_head_dim]


def random_layer_weights(
    cfg: CSA2Config,
    layer_id: int,
    act_dtype: torch.dtype,
    generator: torch.Generator,
    std: float = 0.02,
) -> CSA2LayerWeights:
    """Deterministic bf16-exact random weights (checkpoint values are bf16)."""

    def w(*shape, dtype=act_dtype, scale=std):
        t = torch.randn(*shape, generator=generator, dtype=torch.float32) * scale
        return t.to(torch.bfloat16).to(dtype)  # bf16-exact, then stored as act dtype

    def norm_w(n):
        return (
            (1.0 + 0.1 * torch.randn(n, generator=generator)).to(torch.bfloat16).float()
        )

    ratio = cfg.ratio(layer_id)
    lw = CSA2LayerWeights(
        wq_a=w(cfg.q_lora_rank, cfg.hidden_size),
        q_norm=norm_w(cfg.q_lora_rank),
        wq_b=w(cfg.n_heads * cfg.head_dim, cfg.q_lora_rank, scale=std * 2),
        wkv=w(cfg.head_dim, cfg.hidden_size),
        kv_norm=norm_w(cfg.head_dim),
        attn_sink=torch.randn(cfg.n_heads, generator=generator) * 0.5,
    )
    if cfg.is_kv_source(layer_id):
        c_dtype = torch.float32 if ratio > 1 else act_dtype
        lw.c_wkv = w(cfg.head_dim, cfg.hidden_size, dtype=c_dtype)
        if ratio > 1:
            lw.c_wgate = w(cfg.head_dim, cfg.hidden_size, dtype=torch.float32)
        lw.c_norm = norm_w(cfg.head_dim)
    if cfg.is_index_source(layer_id):
        lw.i_wq_b = w(
            cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank, scale=std * 2
        )
        lw.i_weights_proj = w(cfg.index_n_heads, cfg.hidden_size, scale=std * 4)
        if cfg.is_kv_source(layer_id):
            lw.i_wk = w(cfg.index_head_dim, cfg.head_dim, scale=std * 2)
            lw.i_k_norm = norm_w(cfg.index_head_dim)
    return lw


# --------------------------------------------------------------------------- #
# Compressor (HF Compressor / upstream DeepseekV41Compressor)
# --------------------------------------------------------------------------- #


def compress_ratio1(x: torch.Tensor, lw: CSA2LayerWeights, eps: float) -> torch.Tensor:
    """Ratio 1: ``norm(wkv(x))`` in the activation dtype, one latent per token."""
    return rmsnorm(linear(x, lw.c_wkv), lw.c_norm, eps)


def compress_project_ratio2(x: torch.Tensor, lw: CSA2LayerWeights):
    """Ratio 2 projections in fp32 (HF promotes wkv/wgate to fp32; upstream
    ``linear_bf16_fp32``): returns (kv fp32 [T, D], score fp32 [T, D])."""
    xf = x.float()
    return xf @ lw.c_wkv.float().t(), xf @ lw.c_wgate.float().t()


def pool_pairs(kv2: torch.Tensor, score2: torch.Tensor) -> torch.Tensor:
    """kv2, score2 fp32 [n, ratio, D] -> fp32 [n, D]: softmax gate over the
    group, per channel (HF ``(kv * score.softmax(dim=2)).sum(dim=2)``)."""
    return (kv2 * score2.softmax(dim=1)).sum(dim=1)


def finish_latent(
    pooled_fp32: torch.Tensor, lw: CSA2LayerWeights, eps: float, act_dtype: torch.dtype
) -> torch.Tensor:
    """fp32 pooled -> act dtype -> RMSNorm (HF ``self.norm(kv.to(dtype))``)."""
    return rmsnorm(pooled_fp32.to(act_dtype), lw.c_norm, eps)


# --------------------------------------------------------------------------- #
# Indexer (HF Indexer / upstream DeepseekV41Indexer)
# --------------------------------------------------------------------------- #


def index_keys(
    latent: torch.Tensor,
    group_freqs: torch.Tensor,
    lw: CSA2LayerWeights,
    cfg: CSA2Config,
) -> torch.Tensor:
    """Pre-RoPE latents [n, D] -> FP4-grid index keys [n, index_head_dim]:
    ``k_norm(wk(latent))`` -> RoPE at the group's first position -> per-32 UE8M0 FP4."""
    k = rmsnorm(linear(latent, lw.i_wk), lw.i_k_norm, cfg.rms_norm_eps)
    return fake_quant_fp4_ue8m0(rope_tail(k, group_freqs, cfg.rope_head_dim))


def index_queries(
    q_lora: torch.Tensor, freqs: torch.Tensor, lw: CSA2LayerWeights, cfg: CSA2Config
) -> torch.Tensor:
    """[T, q_lora_rank] -> FP4-grid queries [T, index_n_heads, index_head_dim]."""
    q = linear(q_lora, lw.i_wq_b).view(-1, cfg.index_n_heads, cfg.index_head_dim)
    return fake_quant_fp4_ue8m0(rope_tail(q, freqs, cfg.rope_head_dim))


def index_head_weights(x: torch.Tensor, lw: CSA2LayerWeights, cfg: CSA2Config):
    """``weights_proj(x) * (index_head_dim**-0.5 * index_n_heads**-0.5)``, in the
    activation dtype (HF: bf16 Linear times a python float)."""
    return linear(x, lw.i_weights_proj) * cfg.index_head_weight_scale


def index_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    score_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """q [T, H, d], k [n, d], weights [T, H] -> fp32 [T, n]:
    ``sum_h relu(q_h . k) * w_h``.

    ``score_dtype=float32`` accumulates every step in fp32, which is what the fp4
    logits kernels do (DeepGEMM/Triton) and what an SM70 kernel should do.
    ``score_dtype=act dtype`` reproduces the HF/upstream *torch* path, where the
    einsum output, the product and the head sum are rounded to bf16.
    """
    if score_dtype == torch.float32:
        s = torch.einsum("bhd,nd->bhn", q.float(), k.float())
        return (s.relu() * weights.float().unsqueeze(-1)).sum(dim=1)
    s = torch.einsum("bhd,nd->bhn", q.to(score_dtype), k.to(score_dtype))
    s = (s.relu() * weights.to(score_dtype).unsqueeze(-1)).sum(dim=1)
    return s.float()


def select_candidate_block_ids(
    logits: torch.Tensor, compress_lens: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Block ids kept by ``select_candidate_blocks``, shape [T, k], ``-1`` padded."""
    width = logits.size(-1)
    scores = torch.nn.functional.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )
    k = min(topk_blocks, num_blocks)
    top = scores.topk(k, dim=-1)
    idx = top.indices.to(torch.int32)
    return idx.masked_fill(top.values <= -torch.inf, -1)


def expand_candidate_block_ids(
    block_ids: torch.Tensor, width: int, block_size: int
) -> torch.Tensor:
    """[T, k] block ids -> bool mask [T, width] (same as ``select_candidate_blocks``)."""
    t = block_ids.shape[0]
    num_blocks = (width + block_size - 1) // block_size
    keep = torch.zeros(
        (t, num_blocks), dtype=torch.bool, device=block_ids.device
    )
    valid = block_ids >= 0
    if bool(valid.any()):
        rows, slots = torch.where(valid)
        keep[rows, block_ids[rows, slots].to(torch.int64)] = True
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def select_candidate_blocks(
    logits: torch.Tensor, compress_lens: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Level one of the two-level top-k (HF / upstream ``select_candidate_blocks``):
    keep the ``topk_blocks`` best blocks per row by block-max score. Unreachable
    positions are already -inf so an all -inf block is "not reachable yet"; the
    block holding the row's newest position is pinned with +inf. Returns a bool
    mask shaped like ``logits``."""
    ids = select_candidate_block_ids(logits, compress_lens, topk_blocks, block_size)
    return expand_candidate_block_ids(ids, logits.size(-1), block_size)


def topk_positions(
    logits: torch.Tensor, compress_lens: torch.Tensor, topk: int
) -> torch.Tensor:
    """Per row: the ``topk`` best reachable positions, ascending, ``-1`` padded
    (HF: ``topk(sorted=False).indices.sort()``, ``where(idx < compress_lens, idx, -1)``;
    upstream additionally drops selections whose score is -inf, which is the same
    set here because unreachable *and* candidate-masked positions are -inf).

    Rows are compared by score only; ties fall to torch.topk's order. Callers
    that need exact index equality with another implementation must avoid ties."""
    width = logits.size(-1)
    k = min(topk, width)
    if k == 0:
        return torch.full((logits.shape[0], 0), -1, dtype=torch.int32)
    top = logits.topk(k, dim=-1, sorted=False)
    idx = top.indices.masked_fill(top.values == -torch.inf, width)
    idx = idx.sort(dim=-1).values
    reach = (idx < compress_lens) & (idx < width)
    return torch.where(reach, idx, -1).to(torch.int32)


# --------------------------------------------------------------------------- #
# Sparse attention with sink (HF sparse_attn / FlashMLA with attn_sink)
# --------------------------------------------------------------------------- #


@dataclass
class SparseSoftmaxTerms:
    """fp32 online-softmax intermediates shared by the output and the mass split.

    ``p`` / ``sink_p`` are unnormalized ``exp(· - m)``. Mass of a key is
    ``p / l``; mass of the sink is ``sink_p / l``. A row with no valid key has
    ``has_key`` False and must not be read as a mass (``l`` overflows).
    """

    scores: torch.Tensor  # [T, H, K]
    m: torch.Tensor  # [T, H, 1]
    p: torch.Tensor  # [T, H, K]
    sink_p: torch.Tensor  # [T, H, 1]
    l: torch.Tensor  # [T, H, 1]
    has_key: torch.Tensor  # [T, 1, 1] bool


def sparse_softmax_terms(
    q: torch.Tensor,
    keys: torch.Tensor,
    valid: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> SparseSoftmaxTerms:
    """The sink-aware softmax used by ``sparse_attention_rows``.

    Isolated so a testbench can read sink / window / sparse mass without
    reimplementing the serving formula.
    """
    s = torch.einsum("thd,tkd->thk", q.float(), keys.float()) * softmax_scale
    s = s.masked_fill(~valid[:, None, :], -torch.inf)
    m = s.amax(dim=-1, keepdim=True)
    has_key = valid.any(dim=-1)[:, None, None]
    m = torch.where(has_key, m, torch.full_like(m, -1e30))
    p = torch.exp(s - m)
    sink_p = torch.exp(attn_sink.float()[None, :, None] - m)
    l = p.sum(dim=-1, keepdim=True) + sink_p
    return SparseSoftmaxTerms(s, m, p, sink_p, l, has_key)


def sparse_attention_rows(
    q: torch.Tensor,
    keys: torch.Tensor,
    valid: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    p_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """q [T, H, D], keys [T, K, D] (V = K, MQA), valid [T, K] bool,
    attn_sink [H] fp32 -> [T, H, D] in q.dtype.

    Per row and head: s = (q . k) * scale over valid keys (fp32 accumulate),
    m = max s, p = exp(s - m), l = sum p + exp(sink - m), o = (p @ v) / l.
    ``p`` is rounded to ``p_dtype`` (default q.dtype) before the PV product, as the
    HF kernel (bf16 ``acc_s_cast``) and FlashMLA do. A row with no valid key
    yields zeros (HF: finite -1e30 lower bound)."""
    p_dtype = q.dtype if p_dtype is None else p_dtype
    sm = sparse_softmax_terms(q, keys, valid, attn_sink, softmax_scale)
    o = torch.einsum("thk,tkd->thd", sm.p.to(p_dtype).float(), keys.float())
    return (o / sm.l).to(q.dtype)


@dataclass
class AttnMassSplit:
    """Per-row, per-head attention mass.

    ``sink`` + ``window`` + ``sparse`` sum to 1 on a row with keys. ``needle``
    is a subset of window ∪ sparse, not a fourth bucket. Empty rows (no valid
    key) are all zeros, not NaN. Fields other than ``sink_logit`` / counts
    are [T, H].
    """

    sink: torch.Tensor
    window: torch.Tensor
    sparse: torch.Tensor
    needle: torch.Tensor
    max_score: torch.Tensor
    sink_logit: torch.Tensor  # [H]
    n_window: torch.Tensor  # [T]
    n_sparse: torch.Tensor  # [T]
    n_needle: torch.Tensor  # [T]

    def mean_over_heads(self) -> Dict[str, torch.Tensor]:
        return {
            "sink": self.sink.mean(dim=-1),
            "window": self.window.mean(dim=-1),
            "sparse": self.sparse.mean(dim=-1),
            "needle": self.needle.mean(dim=-1),
            "max_score": self.max_score.mean(dim=-1),
        }


def sparse_attention_mass_split(
    q: torch.Tensor,
    keys: torch.Tensor,
    valid: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    *,
    window_width: int,
    needle_mask: Optional[torch.Tensor] = None,
) -> AttnMassSplit:
    """Split softmax mass into sink / first ``window_width`` keys / the rest.

    ``needle_mask`` is [T, K] bool over the same key axis as ``keys`` (window
    then sparse). Mass on those keys is reported as ``needle``; it is a subset,
    not an extra partition.
    """
    sm = sparse_softmax_terms(q, keys, valid, attn_sink, softmax_scale)
    t, k = valid.shape
    w = max(0, min(int(window_width), k))
    has = sm.has_key.squeeze(-1).squeeze(-1)  # [T]
    denom = sm.l.clamp_min(torch.finfo(sm.l.dtype).tiny)
    key_mass = sm.p / denom
    sink = (sm.sink_p / denom).squeeze(-1)
    window = key_mass[..., :w].sum(dim=-1) if w else torch.zeros_like(sink)
    sparse = key_mass[..., w:].sum(dim=-1) if w < k else torch.zeros_like(sink)
    if needle_mask is None:
        needle = torch.zeros_like(sink)
        n_needle = torch.zeros(t, dtype=torch.int64)
    else:
        needle = (key_mass * needle_mask[:, None, :].to(key_mass.dtype)).sum(dim=-1)
        n_needle = needle_mask.sum(dim=-1).to(torch.int64)
    finite = has[:, None]
    z = torch.zeros_like(sink)
    max_score = sm.m.squeeze(-1)
    max_score = torch.where(finite, max_score, z)
    return AttnMassSplit(
        sink=torch.where(finite, sink, z),
        window=torch.where(finite, window, z),
        sparse=torch.where(finite, sparse, z),
        needle=torch.where(finite, needle, z),
        max_score=max_score,
        sink_logit=attn_sink.float(),
        n_window=(
            valid[:, :w].sum(dim=-1).to(torch.int64)
            if w
            else torch.zeros(t, dtype=torch.int64)
        ),
        n_sparse=(
            valid[:, w:].sum(dim=-1).to(torch.int64)
            if w < k
            else torch.zeros(t, dtype=torch.int64)
        ),
        n_needle=n_needle,
    )


# --------------------------------------------------------------------------- #
# Stateful reference model: caches + chunked forward
# --------------------------------------------------------------------------- #


@dataclass
class KvSourceState:
    """Caches owned by one kv_source layer."""

    ratio: int
    latents: List[torch.Tensor]  # rotated + FP4(E4M3/16) fake-quantized rows [D]
    index_k: List[
        torch.Tensor
    ]  # rotated + FP4(UE8M0/32) fake-quantized rows [index_head_dim]
    pending_kv: Optional[torch.Tensor] = (
        None  # fp32 [D], ratio 2: first token of an open pair
    )
    pending_score: Optional[torch.Tensor] = None  # fp32 [D]

    @property
    def num_compressed(self) -> int:
        return len(self.latents)


@dataclass
class ChunkOutputs:
    """What one chunk produced, per layer, for tests to inspect."""

    attn_out: Dict[int, torch.Tensor]  # layer -> [T, H, D] (after inverse RoPE)
    topk_idx: Dict[
        int, torch.Tensor
    ]  # index-source layer -> [T, topk] int32 (-1 padded)
    candidate_mask: Optional[torch.Tensor]  # [T, n_compressed] bool from layer 20
    compress_lens: Dict[int, torch.Tensor]  # ratio -> [T] visible compressed count
    swa_valid_count: torch.Tensor  # [T] number of window keys per query
    # index-source layer -> [T, n_compressed] fp32 logits the top-k ran on
    # (-inf where unreachable or outside the candidate blocks); for tie-aware tests
    index_logits: Dict[int, torch.Tensor] = field(default_factory=dict)


class CSA2Reference:
    """Runs the attention sub-layers of every layer over token chunks with
    explicit caches. ``forward_chunk`` accepts any chunk size at any position
    (prefill, chunked prefill, decode) and must give identical selections and
    tolerance-equal outputs regardless of chunking: that is the invariant the
    SM70 kernels are later held to.

    ``x`` is the post-norm attention input [T, hidden] per layer. This reference
    has no MoE/residual stream: every layer receives the *same* ``x`` (the tests
    care about attention internals, not a language model)."""

    def __init__(
        self,
        cfg: CSA2Config,
        weights: Sequence[CSA2LayerWeights],
        act_dtype: torch.dtype = torch.float16,
        index_score_dtype: torch.dtype = torch.float32,
        swa_kv_quant: str = "fp8_ue8m0",
        inverse_rope_output: bool = True,
        attn_query_chunk: int = 64,
    ):
        assert swa_kv_quant in ("fp8_ue8m0", "fp8_flashmla", "none")
        self.cfg = cfg
        self.weights = list(weights)
        self.act_dtype = act_dtype
        self.index_score_dtype = index_score_dtype
        self.swa_kv_quant = swa_kv_quant
        self.inverse_rope_output = inverse_rope_output
        self.attn_query_chunk = attn_query_chunk
        self.freqs = {lid: layer_freqs_cis(cfg, lid) for lid in range(cfg.num_layers)}
        self.reset()

    # ---- state ------------------------------------------------------------ #

    def reset(self):
        self.pos = 0
        self.swa_k: Dict[int, List[torch.Tensor]] = {
            lid: [] for lid in range(self.cfg.num_layers)
        }
        self.kv_state: Dict[int, KvSourceState] = {
            lid: KvSourceState(self.cfg.ratio(lid), [], [])
            for lid in self.cfg.kv_source_layer_ids
        }

    # ---- per-layer pieces ------------------------------------------------- #

    def _swa_quant(self, k: torch.Tensor) -> torch.Tensor:
        if self.swa_kv_quant == "fp8_ue8m0":
            return fake_quant_fp8_ue8m0(k)
        if self.swa_kv_quant == "fp8_flashmla":
            return fake_quant_fp8_flashmla(
                k, nope_dim=self.cfg.head_dim - self.cfg.rope_head_dim
            )
        return k

    def _window_keys(self, layer_id: int, x: torch.Tensor, positions: torch.Tensor):
        """Write this chunk's window K, return (all_k [pos_end, D], idx [T, W],
        valid [T, W]) where row t indexes positions max(0, p_t - W + 1) .. p_t
        (HF get_window_topk_idxs: the window includes the query's own token)."""
        cfg, lw = self.cfg, self.weights[layer_id]
        k = rmsnorm(linear(x, lw.wkv), lw.kv_norm, cfg.rms_norm_eps)
        k = rope_tail(k, self.freqs[layer_id][positions], cfg.rope_head_dim)
        k = self._swa_quant(k)
        self.swa_k[layer_id].extend(k.unbind(0))
        all_k = torch.stack(self.swa_k[layer_id])  # [pos_end, D]
        W = cfg.sliding_window
        offs = torch.arange(W)
        idx = positions[:, None] - (W - 1) + offs[None, :]  # ascending, ends at p_t
        valid = idx >= 0
        return all_k, idx.clamp_min(0), valid

    def _compress(self, layer_id: int, x: torch.Tensor, positions: torch.Tensor):
        """kv_source: pool this chunk's complete groups into latents, publish index
        keys (owner) and the rotated FP4 latents. Returns the new pre-RoPE latents
        [n_new, D] and their group positions [n_new]."""
        cfg, lw = self.cfg, self.weights[layer_id]
        st = self.kv_state[layer_id]
        eps = cfg.rms_norm_eps
        T = x.shape[0]
        if st.ratio == 1:
            latent = compress_ratio1(x, lw, eps)
            group_pos = positions.clone()
        else:
            kv, score = compress_project_ratio2(x, lw)
            # Prepend the pending first-of-pair token from the previous chunk.
            if st.pending_kv is not None:
                kv = torch.cat([st.pending_kv[None], kv])
                score = torch.cat([st.pending_score[None], score])
                first_pos = positions[0].item() - 1
            else:
                first_pos = positions[0].item()
            assert first_pos % 2 == 0, "pair state out of step"
            n = kv.shape[0]
            complete = n - n % 2
            if complete < n:
                st.pending_kv, st.pending_score = kv[-1], score[-1]
            else:
                st.pending_kv = st.pending_score = None
            kv2 = kv[:complete].unflatten(0, (-1, 2))
            score2 = score[:complete].unflatten(0, (-1, 2))
            pooled = pool_pairs(kv2, score2)
            latent = finish_latent(pooled, lw, eps, self.act_dtype)
            group_pos = first_pos + 2 * torch.arange(pooled.shape[0])
        if latent.shape[0] == 0:
            return latent, group_pos
        gfreqs = self.freqs[layer_id][group_pos]
        if cfg.is_index_source(layer_id):
            st.index_k.extend(index_keys(latent, gfreqs, lw, cfg).unbind(0))
        rotated = rope_tail(latent, gfreqs, cfg.rope_head_dim)
        st.latents.extend(fake_quant_fp4_e4m3(rotated).unbind(0))
        return latent, group_pos

    def _index_topk(
        self,
        layer_id: int,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        candidate_mask: Optional[torch.Tensor],
    ):
        """index_source: score every visible compressed position, apply the
        two-level candidate scheme, return (topk idx [T, topk], logits [T, n],
        published mask or None, compress_lens [T])."""
        cfg, lw = self.cfg, self.weights[layer_id]
        ratio = cfg.ratio(layer_id)
        src = self.kv_state[cfg.kv_source_for(layer_id)]
        n = src.num_compressed
        T = x.shape[0]
        compress_lens = (positions + 1) // ratio  # visible once its last token passed
        assert int(compress_lens.max()) <= n, "indexer ran before its kv source"
        q = index_queries(q_lora, self.freqs[layer_id][positions], lw, cfg)
        w = index_head_weights(x, lw, cfg)
        if n == 0:
            empty = torch.full((T, 0), -1, dtype=torch.int32)
            return empty, torch.zeros(T, 0), None, compress_lens
        k = torch.stack(src.index_k)  # [n, index_head_dim]
        logits = index_scores(q, k, w, self.index_score_dtype)
        reach = torch.arange(n)[None, :] < compress_lens[:, None]
        logits = logits.masked_fill(~reach, -torch.inf)
        published = None
        if cfg.is_candidate_source(layer_id):
            published = select_candidate_blocks(
                logits,
                compress_lens[:, None],
                cfg.candidate_topk_blocks,
                cfg.candidate_block_size,
            )
        elif cfg.uses_candidates(layer_id):
            assert candidate_mask is not None, "candidate mask missing"
            logits = logits.masked_fill(~candidate_mask[:, :n], -torch.inf)
        idx = topk_positions(logits, compress_lens[:, None], cfg.index_topk)
        return idx, logits, published, compress_lens

    def _attend(
        self,
        layer_id: int,
        q: torch.Tensor,
        swa_all_k: torch.Tensor,
        swa_idx: torch.Tensor,
        swa_valid: torch.Tensor,
        comp_idx: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Union of window keys and selected compressed latents (both branches
        keep their own K; a token in both is counted twice, as HF concatenates)."""
        cfg = self.cfg
        lw = self.weights[layer_id]
        T = q.shape[0]
        lat = None
        if comp_idx is not None and comp_idx.shape[1] > 0:
            lat = torch.stack(self.kv_state[cfg.kv_source_for(layer_id)].latents)
        outs = []
        for s in range(0, T, self.attn_query_chunk):
            e = min(T, s + self.attn_query_chunk)
            keys, valid = swa_all_k[swa_idx[s:e]], swa_valid[s:e]
            if lat is not None:
                ci = comp_idx[s:e].to(torch.int64)
                keys = torch.cat([keys, lat[ci.clamp_min(0)]], dim=1)
                valid = torch.cat([valid, ci >= 0], dim=1)
            outs.append(
                sparse_attention_rows(
                    q[s:e], keys, valid, lw.attn_sink, cfg.softmax_scale
                )
            )
        return torch.cat(outs, dim=0)

    # ---- chunk forward ----------------------------------------------------- #

    def forward_chunk(self, x: torch.Tensor, layer_inputs=None) -> ChunkOutputs:
        """Process ``T`` new tokens at positions ``self.pos .. self.pos + T - 1``.

        ``layer_inputs`` optionally maps layer_id -> its own [T, hidden] input;
        otherwise every layer sees ``x``."""
        cfg = self.cfg
        T = x.shape[0]
        positions = torch.arange(self.pos, self.pos + T)
        attn_out: Dict[int, torch.Tensor] = {}
        topk_idx: Dict[int, torch.Tensor] = {}
        index_logits: Dict[int, torch.Tensor] = {}
        compress_lens: Dict[int, torch.Tensor] = {}
        candidate_mask: Optional[torch.Tensor] = None
        current_topk: Dict[int, torch.Tensor] = {}  # index source -> idx
        swa_count = None

        for lid in range(cfg.num_layers):
            lw = self.weights[lid]
            xl = x if layer_inputs is None else layer_inputs[lid]
            xl = xl.to(self.act_dtype)
            freqs = self.freqs[lid][positions]
            ratio = cfg.ratio(lid)

            q_lora = rmsnorm(linear(xl, lw.wq_a), lw.q_norm, cfg.rms_norm_eps)
            q = linear(q_lora, lw.wq_b).view(T, cfg.n_heads, cfg.head_dim)
            q = rope_tail(q, freqs, cfg.rope_head_dim)

            swa_all_k, swa_idx, swa_valid = self._window_keys(lid, xl, positions)
            if swa_count is None:
                swa_count = swa_valid.sum(dim=-1)

            comp_idx = None
            if ratio:
                if cfg.is_kv_source(lid):
                    self._compress(lid, xl, positions)
                if cfg.is_index_source(lid):
                    idx, logits, published, lens = self._index_topk(
                        lid, xl, q_lora, positions, candidate_mask
                    )
                    if published is not None:
                        candidate_mask = published
                    current_topk[lid] = idx
                    topk_idx[lid] = idx
                    index_logits[lid] = logits
                    compress_lens[ratio] = lens
                comp_idx = current_topk[cfg.index_source_for(lid)]

            o = self._attend(lid, q, swa_all_k, swa_idx, swa_valid, comp_idx)
            if self.inverse_rope_output:
                o = rope_tail(o, freqs, cfg.rope_head_dim, inverse=True)
            attn_out[lid] = o

        self.pos += T
        return ChunkOutputs(
            attn_out, topk_idx, candidate_mask, compress_lens, swa_count, index_logits
        )


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


def tiny_config(**overrides) -> CSA2Config:
    """The real 40-layer topology with small dimensions for CPU tests.
    ``head_dim``/``index_head_dim``/``rope_head_dim`` keep the real values so the
    byte layouts (288 / 68 / 528 bytes per row) are the real ones."""
    # 8 index heads: an index score is exactly 0 when every head's q.k is negative
    # (relu), so fewer heads make exact top-k ties common and selections chunking-
    # dependent. The real model has 32 heads (2**-32 per position).
    base = dict(
        hidden_size=96,
        n_heads=4,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=64,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=64,
        max_seq_len=2304,
    )
    base.update(overrides)
    return CSA2Config(**base)


def build_reference(
    cfg: CSA2Config, act_dtype: torch.dtype = torch.float16, seed: int = 0, **kw
) -> CSA2Reference:
    g = torch.Generator().manual_seed(seed)
    weights = [
        random_layer_weights(cfg, lid, act_dtype, g) for lid in range(cfg.num_layers)
    ]
    return CSA2Reference(cfg, weights, act_dtype=act_dtype, **kw)
