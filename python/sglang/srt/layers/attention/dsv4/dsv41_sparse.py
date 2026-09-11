"""DeepSeek V4.1 ratio-1/2 compressors and indexers.

Only kv_source layers own compressed latents; later layers with the same ratio
share that storage. Index sources score the latents and publish top-k slots
for subsequent attention layers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from sglang.kernels.ops.attention.dsv4 import linear_bf16_fp32
from sglang.kernels.ops.attention.dsv4.rmsnorm_fp32 import rmsnorm_fp32
from sglang.srt.layers.attention.dsv4.indexer import CandidateRole
from sglang.srt.layers.attention.dsv4.torch_quant import (
    fake_quant_compressed_kv,
    fake_quant_fp4,
)
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.utils import add_prefix


def _rope_fq4(x, freqs, rope_dim, *, compressed_kv=False):
    """RoPE plus fake FP4 quantization, fused for CUDA BF16 inputs."""
    if x.is_cuda and torch.version.cuda is not None and x.dtype == torch.bfloat16:
        from sglang.kernels.ops.attention.dsv4.rope_fake_quant_fp4 import (
            rope_tail_fake_quant_fp4,
        )

        return rope_tail_fake_quant_fp4(x, freqs, rope_dim, compressed_kv=compressed_kv)
    quant = fake_quant_compressed_kv if compressed_kv else fake_quant_fp4
    return quant(rope_tail(x, freqs, rope_dim))


class RMSNorm(nn.Module):
    """fp32 statistics and fp32 weight multiply, cast back at the very end."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and torch.version.cuda is not None
            and x.dtype in (torch.bfloat16, torch.float32)
            and self.weight.dtype in (torch.bfloat16, torch.float32)
            and x.shape[-1] in (128, 512)
            and x.is_contiguous()
            and self.weight.is_contiguous()
        ):
            return rmsnorm_fp32(x, self.weight, self.eps)
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def token_req_indices(forward_batch, *, num_tokens=None) -> torch.Tensor:
    """req_pool_indices repeated once per token of the batch."""
    req = forward_batch.req_pool_indices.to(torch.int64)
    if forward_batch.forward_mode.is_decode():
        return req
    if forward_batch.forward_mode.is_target_verify():
        return torch.repeat_interleave(
            req, int(forward_batch.spec_info.draft_token_num), output_size=num_tokens
        )
    assert forward_batch.forward_mode.is_extend(), (
        "the V4.1 torch attention path serves extend, target-verify and decode"
    )
    return torch.repeat_interleave(
        req, forward_batch.extend_seq_lens.to(torch.int64), output_size=num_tokens
    )


def rope_tail(
    x: torch.Tensor, freqs: torch.Tensor, rope_dim: int, inverse: bool = False
) -> torch.Tensor:
    """Rotate the last rope_dim features of x [T, ..., D] with complex freqs [T, rope_dim // 2]."""
    head, tail = x[..., :-rope_dim], x[..., -rope_dim:]
    tc = torch.view_as_complex(tail.float().unflatten(-1, (-1, 2)).contiguous())
    f = freqs.conj() if inverse else freqs
    f = f.view(x.shape[0], *([1] * (x.ndim - 2)), rope_dim // 2)
    rotated = torch.view_as_real(tc * f).flatten(-2).to(x.dtype)
    return torch.cat([head, rotated], dim=-1)


def fused_low_ratio_compress_supported() -> bool:
    """Whether the fused c1 / c2 / index-K decode kernels can serve this process.

    They pack fp4 with `cvt.rn.satfinite.e2m1x2`, a Blackwell (sm100+) CUDA
    instruction, so HIP and pre-Blackwell parts keep the split projection and the
    unfused write. Decided once at load time: the choice also fixes the weight
    layout of the ratio-2 projection (one `wkv_gate` or `wkv` plus `wgate`)."""
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return False
    return torch.cuda.get_device_capability()[0] >= 10


class DeepseekV41Compressor(nn.Module):
    """Pool consecutive tokens into one pre-RoPE KV latent.

    Ratio 1 uses a bf16 projection. Ratio 2 keeps checkpoint weights in bf16 but
    accumulates projections and softmax pooling in fp32; finish rounds to bf16
    before RMSNorm. The fp32 reference can differ in GEMM reduction order.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        compress_ratio: int,
        eps: float,
        *,
        fused_compress: Optional[bool] = None,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.norm = RMSNorm(head_dim, eps)
        # The loader concatenates ratio-2 wkv/wgate when it finds wkv_gate.weight;
        # ratio 1 must retain wkv.weight because there is no gate half to load.
        self.use_fused_compress = (
            fused_low_ratio_compress_supported()
            if fused_compress is None
            else bool(fused_compress)
        )
        self.use_fused_gate = compress_ratio > 1 and self.use_fused_compress
        if self.use_fused_gate:
            self.wkv_gate = nn.Linear(
                hidden_size, 2 * head_dim, bias=False, dtype=torch.bfloat16
            )
        else:
            self.wkv = nn.Linear(
                hidden_size, head_dim, bias=False, dtype=torch.bfloat16
            )
            if compress_ratio > 1:
                self.wgate = nn.Linear(
                    hidden_size, head_dim, bias=False, dtype=torch.bfloat16
                )

    def project(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.compress_ratio == 1:
            return self.wkv(x), None
        if self.use_fused_gate:
            # Extend accepts strided column views; pair_pool_decode requires contiguous
            # halves and is reachable only when the fused projection is disabled.
            fused = self.project_fused(x)
            head_dim = fused.shape[-1] // 2
            return fused[..., :head_dim], fused[..., head_dim:]
        # Two GEMMs rather than one fused [2D, K] projection: the decode epilogue
        # kernel (pair_pool_decode) reads kv and score as contiguous [n, D] fp32
        # rows, which column slices of a fused output are not.
        kv = linear_bf16_fp32(x, self.wkv.weight)
        score = linear_bf16_fp32(x, self.wgate.weight)
        return kv, score

    def project_fused(self, x: torch.Tensor) -> torch.Tensor:
        """`[n, 2D]` fp32, `| kv | score |`, for the fused decode kernel."""
        return linear_bf16_fp32(x, self.wkv_gate.weight)

    def finish(self, kv: torch.Tensor) -> torch.Tensor:
        return self.norm(kv.to(torch.bfloat16))

    @staticmethod
    def pool_pairs(kv2: torch.Tensor, score2: torch.Tensor) -> torch.Tensor:
        """kv2, score2 [n, 2, D] fp32 -> [n, D]"""
        return (kv2 * score2.softmax(dim=1)).sum(dim=1)


def _small_weights_proj_max_m(n_heads: int, hidden_size: int) -> int:
    """Maximum decode rows for the small head-weight GEMM;
    -1 means the device or checkpoint shape requires the linear fallback.
    """
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return -1
    from sglang.kernels.ops.gemm.n32k5120 import MAX_M, can_use_n32k5120_gemm

    return MAX_M if can_use_n32k5120_gemm(n_heads, hidden_size, 1) else -1


class DeepseekV41Indexer(nn.Module):
    """Scores compressed positions with a small fp4 side attention. Only a
    kv_source layer owns index keys; the other index sources read the source's.

    The projections are replicated across TP, as in the c4 indexer: every rank
    scores with all heads, so the decode kernel path needs no cross-rank
    reduction and every rank selects the same top-k."""

    def __init__(
        self,
        config,
        layer_id: int,
        head_dim: int,
        quant_config: Optional[QuantizationConfig],
        prefix: str,
    ):
        super().__init__()
        self.n_heads = config.index_n_heads
        self.n_local_heads = self.n_heads
        self.index_head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.owns_k = layer_id in config.kv_source_layer_ids
        self.candidate_role = CandidateRole.for_layer(
            layer_id, config.candidate_source_layer_id
        )
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            self.n_heads * self.index_head_dim,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            self.n_heads,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )
        # The decode GEMM matches tiny_gemm's reduction order, not cuBLAS's;
        # wider batches use the linear path.
        self.weights_proj_small_max_m = _small_weights_proj_max_m(
            self.n_heads, config.hidden_size
        )
        if self.owns_k:
            self.wk = nn.Linear(
                head_dim, self.index_head_dim, bias=False, dtype=torch.bfloat16
            )
            self.k_norm = RMSNorm(self.index_head_dim, config.rms_norm_eps)

    def forward_wk(self, latent: torch.Tensor) -> torch.Tensor:
        from sglang.kernels.ops.gemm.n128k512 import (
            can_use_n128k512_gemm,
            n128k512_gemm_bf16,
        )

        # Use the shared admission check: the JIT kernel requires supported shapes and
        # hardware, and raises instead of falling back when either condition is violated.
        if can_use_n128k512_gemm(
            self.index_head_dim, latent.shape[-1], latent.shape[0]
        ):
            return n128k512_gemm_bf16(latent, self.wk.weight)
        return self.wk(latent)

    def index_keys(self, latent: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Pre-RoPE latents [n, D] -> fp4-rounded index keys [n, index_head_dim]."""
        k = self.k_norm(self.forward_wk(latent))
        return _rope_fq4(k, freqs, self.rope_head_dim)

    def queries(self, q_lora: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        q, _ = self.wq_b(q_lora)
        q = q.view(q.shape[0], self.n_local_heads, self.index_head_dim)
        return _rope_fq4(q, freqs, self.rope_head_dim)

    def head_weights_raw(self, x: torch.Tensor) -> torch.Tensor:
        """`weights_proj(x)` before the scale, [tokens, n_heads] bf16."""
        if 0 < x.shape[0] <= self.weights_proj_small_max_m and x.is_cuda:
            from sglang.kernels.ops.gemm.n32k5120 import n32k5120_gemm_bf16

            return n32k5120_gemm_bf16(x, self.weights_proj.weight)
        w, _ = self.weights_proj(x)
        return w

    @property
    def head_weight_scale(self) -> float:
        return self.softmax_scale * self.n_heads**-0.5

    def head_weights(self, x: torch.Tensor) -> torch.Tensor:
        return self.head_weights_raw(x) * self.head_weight_scale

    def scores(
        self, q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """q [t, H, d], k [n, d], weights [t, H] -> [t, n], summed over all heads;
        bf16 up to the reduction, as the reference does."""
        s = torch.einsum("bhd,nd->bhn", q, k)
        s = (s.relu() * weights.unsqueeze(-1)).sum(dim=1)
        return s.float()
