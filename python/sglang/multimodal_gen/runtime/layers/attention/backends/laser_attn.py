import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Import to use torch.ops.attentions, install package with sgl_kernel_npu
try:
    import attentions  # noqa: F401
except ImportError as e:
    logger.warning_once(
        "The 'attentions' library is not installed. Laser Attention is unavailable. "
        "Installing this library may improve performance on NPU. "
        "See: sgl-project/sgl-kernel-npu"
    )
    raise ImportError(
        (
            "The required 'attentions' package is not installed. "
            "Install it from sgl-project/sgl-kernel-npu."
        )
    ) from e

# The current NPU kernel stores QK scores and V in FP16 even for BF16 inputs.
_BF16_LASER_SCALE = 256.0


class LaserAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.LASER_ATTN

    @staticmethod
    def get_impl_cls() -> type["LaserAttentionImpl"]:
        return LaserAttentionImpl


class LaserAttentionImpl(AttentionImpl):
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
        self.softmax_scale = softmax_scale

        # After preprocess input layout should be BNSD.
        self.seqlen_base = 256
        self.seqlen_index = 2
        self.dim_index = 3
        self.dim_base = 128
        self.max_token = 2**31 - 1
        self.seq_len_pad_base = 256

        # the laser attention operator has issues with small seq_len
        self.min_seqlen = 2048
        self.sdpa_impl = SDPABackend.get_impl_cls()(
            num_heads,
            head_size,
            causal,
            softmax_scale,
            num_kv_heads,
            prefix,
            **extra_impl_args,
        )

    def _pad(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Pad the input tensor along the sequence length and head dimension.
        to multiples of base values. self.seqlen_index and self.dim_index should be positive integers.
        """

        seq_len = input_tensor.size(self.seqlen_index)
        head_dim = input_tensor.size(self.dim_index)

        pad_seq = 0
        if seq_len % self.seqlen_base != 0:
            pad_seq = ((seq_len // self.seqlen_base) + 1) * self.seqlen_base - seq_len

        pad_dim = 0
        if head_dim % self.dim_base != 0:
            pad_dim = ((head_dim // self.dim_base) + 1) * self.dim_base - head_dim

        if pad_seq == 0 and pad_dim == 0:
            return input_tensor

        pad_list = [0] * (2 * input_tensor.ndim)

        pad_list[len(pad_list) - 2 * self.seqlen_index - 1] = pad_seq
        pad_list[len(pad_list) - 2 * self.dim_index - 1] = pad_dim

        return torch.nn.functional.pad(input_tensor, pad_list)

    def _la_preprocess_input(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        preserve_bf16_range: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        # Currently BSND input layout is not supported
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        q_scale = 1.0
        k_scale = 1.0
        value_scale = 1.0
        if preserve_bf16_range:
            q_scale = _BF16_LASER_SCALE if q.dtype == torch.bfloat16 else 1.0
            k_scale = _BF16_LASER_SCALE if k.dtype == torch.bfloat16 else 1.0
            value_scale = _BF16_LASER_SCALE if v.dtype == torch.bfloat16 else 1.0
            if q.dtype != torch.float16:
                q = q.mul(1.0 / q_scale).to(torch.float16)
            if k.dtype != torch.float16:
                k = k.mul(1.0 / k_scale).to(torch.float16)
            if v.dtype != torch.float16:
                v = v.mul(1.0 / value_scale).to(torch.float16)
        elif q.dtype != torch.float16:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        q = self._pad(q)
        k = self._pad(k)
        v = self._pad(v)

        return q, k, v, q_scale * k_scale, value_scale

    def _la_postprocess_output(
        self,
        attention_out: torch.Tensor,
        dtype: torch.dtype,
        qseqlen: int,
        head_dim: int,
    ) -> torch.Tensor:
        if dtype != attention_out.dtype:
            attention_out = attention_out.to(dtype)

        attention_out = attention_out[:, :, :qseqlen, :head_dim]
        attention_out = attention_out.transpose(1, 2).contiguous()
        return attention_out

    def _laser_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        head_num: int,
        pre_tokens: int,
        scale_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.attentions.la(
            query=query,
            key=key,
            value=value,
            atten_mask=None,
            alibi_mask=None,
            drop_mask=None,
            scale_value=scale_value,
            head_num=head_num,
            input_layout="BNSD",
            keep_prob=1.0,
            pre_tokens=pre_tokens,
            next_tokens=1,
            is_highPrecision=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return self._forward_dense(query, key, value, attn_metadata)

    def _forward_dense(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        *,
        preserve_bf16_range: bool = False,
    ) -> torch.Tensor:
        q_seqlen, head_dim = query.shape[1], query.shape[3]
        kv_seqlen = key.shape[1]

        if q_seqlen < self.min_seqlen or kv_seqlen != q_seqlen:
            output = self.sdpa_impl.forward(
                query,
                key,
                value,
                attn_metadata,
            )
        else:
            pre_tokens = self.max_token
            if kv_seqlen % self.seq_len_pad_base != 0:
                pre_tokens = (
                    kv_seqlen // self.seq_len_pad_base + 1
                ) * self.seq_len_pad_base - kv_seqlen

            q, k, v, qk_scale, value_scale = self._la_preprocess_input(
                query,
                key,
                value,
                preserve_bf16_range=preserve_bf16_range,
            )
            _, la_output = self._laser_attention(
                q,
                k,
                v,
                q.shape[1],
                pre_tokens,
                self.softmax_scale * qk_scale,
            )
            if value_scale != 1.0:
                la_output.mul_(value_scale)
            output = self._la_postprocess_output(
                la_output, query.dtype, q_seqlen, head_dim
            )

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
        del max_seqlen
        bounds = (
            cu_seqlens_host
            if cu_seqlens_host is not None
            else tuple(int(item) for item in cu_seqlens.tolist())
        )
        # MiniMax-H3 is the current Laser varlen caller and encodes one real
        # segment followed by alignment padding as [0, used, padded].
        padding_start = (
            bounds[1]
            if len(bounds) == 3 and bounds[0] == 0 and bounds[-1] == query.shape[0]
            else None
        )
        # Packed segments are independent; a single dense call would let real
        # tokens attend alignment padding in MiniMax-H3.
        # MiniMax-H3 packs one real segment followed by alignment padding.
        # Delay allocation until Laser releases its temporary tensors and
        # avoid allocating/copying the result when no padding was added.
        if (
            padding_start is not None
            and padding_start > 0
            and bounds[0] == 0
            and bounds[1] == padding_start
        ):
            segment = self._forward_dense(
                query[:padding_start].unsqueeze(0),
                key[:padding_start].unsqueeze(0),
                value[:padding_start].unsqueeze(0),
                None,
                preserve_bf16_range=True,
            )[0]
            if padding_start == query.shape[0]:
                return segment

            output = torch.empty_like(query)
            output[:padding_start].copy_(segment)
            output[padding_start:].zero_()
            return output

        output = torch.empty_like(query)
        for start, stop in zip(bounds[:-1], bounds[1:]):
            if padding_start is not None and start >= padding_start:
                break
            if start == stop:
                continue
            segment = self._forward_dense(
                query[start:stop].unsqueeze(0),
                key[start:stop].unsqueeze(0),
                value[start:stop].unsqueeze(0),
                None,
                preserve_bf16_range=padding_start is not None,
            )
            output[start:stop].copy_(segment[0])
        if padding_start is not None:
            output[padding_start:].zero_()
        return output
