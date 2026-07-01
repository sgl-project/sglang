import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPAImpl
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd,
)


class SM120TritonAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.SM120_TRITON_ATTN

    @staticmethod
    def get_impl_cls() -> type["SM120TritonAttentionImpl"]:
        return SM120TritonAttentionImpl


class SM120TritonAttentionImpl(AttentionImpl):
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
        self.sdpa_impl = SDPAImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if (
            query.device.type != "cuda"
            or query.shape[1] != key.shape[1]
            or key.shape[1] != value.shape[1]
        ):
            return self.sdpa_impl.forward(query, key, value, attn_metadata)

        batch_size, seq_len, _, _ = query.shape
        q = query.reshape(batch_size * seq_len, query.shape[2], query.shape[3])
        k = key.reshape(batch_size * seq_len, key.shape[2], key.shape[3])
        v = value.reshape(batch_size * seq_len, value.shape[2], value.shape[3])
        out = torch.empty_like(q)
        qo_indptr = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            dtype=torch.int32,
            device=query.device,
        )
        kv_indptr = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device=query.device
        )
        kv_indices = torch.empty((0,), dtype=torch.int64, device=query.device)
        extend_attention_fwd(
            q,
            k,
            v,
            out,
            k,
            v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            custom_mask=None,
            is_causal=self.causal,
            mask_indptr=None,
            max_len_extend=seq_len,
            k_scale=1.0,
            v_scale=1.0,
            sm_scale=self.softmax_scale,
            skip_prefix=True,
        )
        return out.reshape_as(query)
