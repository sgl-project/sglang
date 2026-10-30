# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

import sglang.multimodal_gen.envs as envs

try:
    from torch.nn.attention.varlen import varlen_attn as torch_varlen_attn
except ImportError:
    torch_varlen_attn = None

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (  # FlashAttentionMetadata,
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Ordered by preference, which only takes effect with ``set_priority=True``;
# without it this is the set PyTorch already picks from, so the flag was inert.
_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

_MPS_VARLEN_QUERY_CHUNK_SIZE = 128

# Dense Triton prefill path (ROCm). The kernel tiles the query length and the
# key/value length by 64 and keeps head_dim as the tile width.
_DENSE_ATTN_HEAD_DIM = 128
_DENSE_ATTN_BLOCK = 64
# Below this length the three head-major copies cost more than the kernel saves:
# at S=1024 the copies are 69% of the call and the path is 0.65x of AOTriton,
# at S=2048 it is break-even, at S=4096 it is 1.29x and at S=16384 it is 11.9x.
_DENSE_ATTN_MIN_SEQ = 4096
_dense_attn_kernel = None


def triton_dense_attn_kernel():
    """The kernel module, imported on first use so non-ROCm runs never pay for it."""
    global _dense_attn_kernel
    if _dense_attn_kernel is None:
        from sglang.kernels.ops.diffusion.attention import flash_attn_dense_triton

        _dense_attn_kernel = flash_attn_dense_triton
    return _dense_attn_kernel


def torch_default_sdpa_backends():
    """Reference SDPA with a stable cuDNN-first preference and fallback set.

    The CUDA platform disables cuDNN SDPA process-wide (diffusers#9704). Modules
    opt back in locally. Priority must be explicit: restoring another SDPA
    context can move disabled backends to the end of the ambient priority list,
    otherwise changing reference outputs after the first DiT forward.
    """
    return sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS, set_priority=True)


class SDPABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.TORCH_SDPA

    @staticmethod
    def get_impl_cls() -> type["SDPAImpl"]:
        return SDPAImpl

    # @staticmethod
    # def get_metadata_cls() -> Type["AttentionMetadata"]:
    #     return FlashAttentionMetadata


class SDPAImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)
        self.allow_cudnn_sdp = bool(extra_impl_args.get("allow_cudnn_sdp", False))

    def _sdpa_context(self, query: torch.Tensor):
        if self.allow_cudnn_sdp and query.device.type == "cuda":
            return sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS, set_priority=True)
        return nullcontext()

    def _triton_dense_output(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ):
        """Run the dense Triton prefill kernel, or return None to fall back to SDPA.

        Gated on ``SGLANG_USE_ROCM_TRITON_DENSE_ATTN``. At the shapes diffusion
        prefill hits on gfx1151 (128 head dim, 16K tokens) AOTriton reaches
        18 TF where the Triton kernel reaches 31.8 TF; the kernel needs head-major
        dense operands, which this backend otherwise only has as strided views,
        so they are materialised here. Everything the kernel does not model --
        causality, dropout, GQA, other dtypes, other head dims, short sequences
        -- returns None.
        """
        if not envs.SGLANG_USE_ROCM_TRITON_DENSE_ATTN:
            return None
        if torch.version.hip is None or torch.compiler.is_compiling():
            return None
        if self.causal or self.dropout != 0.0:
            return None
        if query.dtype is not torch.bfloat16:
            return None
        if query.shape != key.shape or key.shape != value.shape:
            return None
        batch, seq_len, _, head_dim = query.shape
        if (
            batch != 1
            or head_dim != _DENSE_ATTN_HEAD_DIM
            or seq_len < _DENSE_ATTN_MIN_SEQ
            or seq_len % _DENSE_ATTN_BLOCK
        ):
            return None

        kernel = triton_dense_attn_kernel()
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        operands = (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        if not kernel.can_use_flash_attn_dense(*operands):
            return None
        return kernel.flash_attn_dense(*operands, scale=self.softmax_scale)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        dense_output = self._triton_dense_output(query, key, value)
        if dense_output is not None:
            return dense_output

        # transpose to bs, heads, seq_len, head_dim
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attn_mask = None
        is_causal = self.causal
        if self.causal and query.shape[-2] != key.shape[-2]:
            is_causal = False
            if query.shape[-2] > 1:
                query_length = query.shape[-2]
                key_length = key.shape[-2]
                attn_mask = torch.ones(
                    query_length,
                    key_length,
                    dtype=torch.bool,
                    device=query.device,
                ).tril(diagonal=key_length - query_length)

        attn_kwargs = {
            "attn_mask": attn_mask,
            "dropout_p": self.dropout,
            "is_causal": is_causal,
            "scale": self.softmax_scale,
        }
        if query.shape[1] != key.shape[1]:
            attn_kwargs["enable_gqa"] = True
        with self._sdpa_context(query):
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, **attn_kwargs
            )
        output = output.transpose(1, 2)
        return output

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        if (
            type(self) is SDPAImpl
            and torch_varlen_attn is not None
            and not torch.compiler.is_compiling()
            and not torch.is_grad_enabled()
            and query.is_cuda
            and torch.version.hip is None
            and query.ndim == 3
            and query.shape == key.shape == value.shape
            and query.dtype in (torch.float16, torch.bfloat16)
            and query.dtype == key.dtype == value.dtype
            and query.device == key.device == value.device == cu_seqlens.device
            and query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
            and query.numel() > 0
            and query.shape[-1] <= 256
            and query.shape[-1] % 8 == 0
            and cu_seqlens.dtype == torch.int32
            and cu_seqlens.ndim == 1
            and cu_seqlens.is_contiguous()
            and cu_seqlens.numel() > 2
            and max_seqlen > 0
            and self.dropout == 0.0
            and not self.allow_cudnn_sdp
            and torch.backends.cuda.flash_sdp_enabled()
            and not torch.backends.cuda.cudnn_sdp_enabled()
            and torch.cuda.get_device_capability(query.device)[0] == 9
        ):
            # Keep the existing Flash SDPA arithmetic while consuming all
            # packed windows in one call, including ragged and empty windows.
            # Call the Flash op directly: from torch 2.14, varlen_attn silently
            # routes sm90/sm100 to cuDNN, which changes outputs versus the Flash path.
            output, *_ = torch.ops.aten._flash_attention_forward(
                query,
                key,
                value,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen,
                0.0,
                self.causal,
                return_debug_mask=False,
                scale=self.softmax_scale,
                window_size_left=-1,
                window_size_right=0 if self.causal else -1,
            )
            return output
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        lengths = {stop - start for start, stop in zip(bounds[:-1], bounds[1:])}
        covers_query = bounds[0] == 0 and bounds[-1] == query.shape[0]
        if (
            type(self) is not SDPAImpl
            and query.device.type != "mps"
            and covers_query
            and len(lengths) == 1
            and 0 not in lengths
        ):
            # Subclasses (cuDNN SDPA) never take the native varlen op above, so
            # equal segments (e.g. packed CFG pairs) run as one batch instead.
            segments = len(bounds) - 1
            output = self.forward(
                query.reshape(segments, -1, *query.shape[1:]),
                key.reshape(segments, -1, *key.shape[1:]),
                value.reshape(segments, -1, *value.shape[1:]),
                None,
            )
            return output.reshape(query.shape)
        output = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if start == stop:
                continue
            if query.device.type != "mps":
                segment = self.forward(
                    query[start:stop].unsqueeze(0),
                    key[start:stop].unsqueeze(0),
                    value[start:stop].unsqueeze(0),
                    None,
                )
                output[start:stop].copy_(segment[0])
                continue

            # mps SDPA materializes a quadratic temporary for a varlen segment
            # chunking query rows keeps every row's complete K/V context intact
            keys = key[start:stop].unsqueeze(0)
            values = value[start:stop].unsqueeze(0)
            for query_start in range(start, stop, _MPS_VARLEN_QUERY_CHUNK_SIZE):
                query_stop = min(query_start + _MPS_VARLEN_QUERY_CHUNK_SIZE, stop)
                segment = self.forward(
                    query[query_start:query_stop].unsqueeze(0),
                    keys,
                    values,
                    None,
                )
                output[query_start:query_stop].copy_(segment[0])
                torch.mps.synchronize()
                torch.mps.empty_cache()
        return output


class CudnnSDPABackend(SDPABackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.TORCH_CUDNN_SDPA

    @staticmethod
    def get_impl_cls() -> type["CudnnSDPAImpl"]:
        return CudnnSDPAImpl


class CudnnSDPAImpl(SDPAImpl):
    def _sdpa_context(self, query: torch.Tensor):
        if query.device.type == "cuda":
            return sdpa_kernel(SDPBackend.CUDNN_ATTENTION)
        return nullcontext()


# B300/sm_103 self-attention at head_dim 128 from this many tokens runs faster on
# FA4 (hardware exp2) than on cuDNN under a sustained, power-capped load: 1.5-2%
# from 16K to 75.6K tokens; at 8K and below the two are even
_SM103_FA4_MIN_SELF_ATTN_TOKENS = 16384


def _prefers_fa4_on_sm103(query: torch.Tensor, key: torch.Tensor) -> bool:
    return (
        query.shape[-1] == 128
        and query.shape[1] == key.shape[1]
        and query.shape[1] >= _SM103_FA4_MIN_SELF_ATTN_TOKENS
    )


class DynamicCudnnSDPABackend(SDPABackend):
    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.DYNAMIC_CUDNN_SDPA

    @staticmethod
    def get_impl_cls() -> type["DynamicCudnnSDPAImpl"]:
        return DynamicCudnnSDPAImpl


class DynamicCudnnSDPAImpl(SDPAImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        causal: bool,
        softmax_scale: float,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
            FlashAttentionImpl,
            set_fa_ver,
        )

        self.causal = causal
        self.head_size = head_size
        capability = (
            torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        )
        self._is_sm100 = capability is not None and capability[0] == 10
        self._is_sm103 = capability == (10, 3)
        # Set once cuDNN SDPA raised for this layer; permanently pins the
        # fail-safe FA path so we do not retry a failing kernel every step.
        self._cudnn_failed = False
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
            set_fa_ver(4)
        self.cudnn_impl = CudnnSDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.cudnn",
            **extra_impl_args,
        )
        self.fa_impl = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.fa",
            **extra_impl_args,
        )

    def _use_cudnn_sdpa(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> bool:
        if self.causal or self._cudnn_failed:
            return False
        if query.device.type != "cuda":
            return False
        if query.dtype not in (torch.float16, torch.bfloat16):
            return False
        if query.shape[2] != key.shape[2]:
            return False
        if self._is_sm103 and _prefers_fa4_on_sm103(query, key):
            return False
        if self._is_sm100:
            # B200/sm_100: cuDNN SDPA measured 1.25-1.5x faster than FA4 CuTe
            # for dense non-causal diffusion attention, both self-attn
            # (Sq == Skv, up to S=506K) and cross-attn (Skv = text len). A side
            # shorter than one 128-row tile while the other is not drops cuDNN to
            # an sm80 kernel that FA4 beats 2-3x (B300, 126 x 1536 tokens:
            # 24.9 vs 12.1 us)
            short_side, long_side = sorted((query.shape[1], key.shape[1]))
            return short_side >= 128 or long_side < 128
        if query.shape[1] != key.shape[1]:
            return False
        return query.shape[-1] == 64 and query.shape[1] == 1024 and query.shape[0] >= 4

    def forward_into(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor | None,
        attn_metadata: AttentionMetadata,
    ) -> bool:
        if self._use_cudnn_sdpa(query, key, value):
            return False
        return self.fa_impl.forward_into(query, key, value, out, attn_metadata)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        # LSE is only produced by the FA impl (e.g. ring attention).
        if not kwargs.get("return_softmax_lse") and self._use_cudnn_sdpa(
            query, key, value
        ):
            try:
                return self.cudnn_impl.forward(query, key, value, attn_metadata)
            except RuntimeError as e:
                # cuDNN raises "No available kernel" for some shapes; pin the
                # FA fail-safe path for this layer and keep going.
                logger.warning_once(
                    "cuDNN SDPA failed (%s); falling back to FlashAttention for %s.",
                    e,
                    type(self).__name__,
                )
                self._cudnn_failed = True
        return self.fa_impl.forward(query, key, value, attn_metadata, **kwargs)
