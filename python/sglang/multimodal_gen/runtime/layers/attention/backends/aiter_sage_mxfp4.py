# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

# Single shared Hadamard matrix per device, reused by all AITERSageMxfp4Impl instances.
_shared_hadamard: dict[torch.device, torch.Tensor] = {}


def _get_or_create_hadamard() -> torch.Tensor:
    """Return the shared Hadamard rotation matrix for the local device."""
    device = (
        torch.device(torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device not in _shared_hadamard:
        from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
            create_hadamard_matrix,
        )

        block_r = 128
        hadamard = create_hadamard_matrix(block_r, dtype=torch.bfloat16) / (
            block_r**0.5
        )
        _shared_hadamard[device] = hadamard.to(device)
    return _shared_hadamard[device]


class AITERSageMxfp4Backend(AttentionBackend):

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.AITER_SAGE_MXFP4

    @staticmethod
    def get_impl_cls() -> type["AITERSageMxfp4Impl"]:
        return AITERSageMxfp4Impl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return AttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        raise NotImplementedError(
            "AITER Sage MXFP4 backend does not have a metadata builder."
        )


class AITERSageMxfp4Impl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        dropout_p: float = 0.0,
        **extra_impl_args,
    ) -> None:
        self.causal = causal

        try:
            from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
                fav3_sage_mxfp4_wrapper,
            )
        except ImportError:
            raise ImportError(
                "AITER Sage MXFP4 attention is not available, please update AITER version."
            )

        self.attn_fn = fav3_sage_mxfp4_wrapper
        self._hadamard_R = _get_or_create_hadamard()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        """
        Performs attention using AITER Sage MXFP4 (MXFP4 + Hadamard rotation).

        Args:
            query: Query tensor of shape [batch_size, seq_len, head_num, head_dim]
            key: Key tensor of shape [batch_size, seq_len, head_num, head_dim]
            value: Value tensor of shape [batch_size, seq_len, head_num, head_dim]
            attn_metadata: Metadata for the attention operation (unused).

        Returns:
            Output tensor of shape [batch_size, seq_len, head_num, head_dim]
        """
        # Contiguous is needed for Sage MXFP4 in older AITER versions.
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        output = self.attn_fn(
            query,
            key,
            value,
            hadamard_rotation=True,
            R=self._hadamard_R,
            causal=self.causal,
        )
        return output
