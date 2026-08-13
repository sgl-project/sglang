import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# Import to use torch.ops.attentions, install package with sgl_kernel_npu
try:
    import attentions  # noqa: F401
except ImportError as e:
    raise ImportError(
        (
            "The required 'attentions' package is not installed."
            "The package can be installed with sgl_kernel_npu"
        )
    ) from e

logger = init_logger(__name__)


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
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Currently BSND input layout is not supported
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        if q.dtype != torch.float16:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        q = self._pad(q)
        k = self._pad(k)
        v = self._pad(v)

        return q, k, v

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.attentions.la(
            query=query,
            key=key,
            value=value,
            atten_mask=None,
            alibi_mask=None,
            drop_mask=None,
            scale_value=self.softmax_scale,
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

            q, k, v = self._la_preprocess_input(query, key, value)
            _, la_output = self._laser_attention(q, k, v, q.shape[1], pre_tokens)
            output = self._la_postprocess_output(
                la_output, query.dtype, q_seqlen, head_dim
            )

        return output
