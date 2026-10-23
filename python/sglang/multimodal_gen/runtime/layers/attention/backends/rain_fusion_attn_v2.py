# SPDX-License-Identifier: Apache-2.0
"""SGLang entry point for MindIE-SD sparse and quantized dense attention.

MindIE-SD owns operator selection and native input constraints. This adapter
supplies denoising schedules and executes batch/packed documents independently.
"""

from dataclasses import dataclass, field, replace

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class RainFusionAttentionV2Backend(AttentionBackend):
    @staticmethod
    def get_enum():
        return AttentionBackendEnum.RAIN_FUSION_ATTN_V2

    @staticmethod
    def get_impl_cls():
        return RainFusionAttentionV2Impl

    @staticmethod
    def get_metadata_cls():
        return RainFusionAttentionV2Metadata

    @staticmethod
    def get_builder_cls():
        return RainFusionAttentionV2MetadataBuilder


@dataclass
class RainFusionAttentionV2Metadata(AttentionMetadata):
    skip_first_steps: int = 10
    sparsity: float = 0.2
    latent_shape: list[int] | None = None
    txt_len: int = 0
    video_spans: list[dict] | None = None
    sparse_type: str | None = "rf_v3"
    precision: str = "mix"
    block_size: int | None = None
    inner_precise: int = 4
    dense_last_steps: int = 0
    num_steps: int | None = None
    dense_precision: str = "fp8"
    dense_kwargs: dict = field(default_factory=dict)
    sparse_kwargs: dict = field(default_factory=dict)


class RainFusionAttentionV2MetadataBuilder(AttentionMetadataBuilder):
    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(
        self,
        current_timestep,
        skip_first_steps=10,
        sparsity=0.2,
        raw_latent_shape=None,
        patch_size=None,
        txt_len=0,
        video_spans=None,
        sparse_type="rf_v3",
        precision="mix",
        block_size=None,
        inner_precise=4,
        dense_last_steps=0,
        num_steps=None,
        dense_precision="fp8",
        dense_kwargs=None,
        **kwargs,
    ):
        latent_shape = None
        if raw_latent_shape is not None:
            latent_shape = [
                dim // size for dim, size in zip(raw_latent_shape[-3:], patch_size)
            ]
        return RainFusionAttentionV2Metadata(
            current_timestep=current_timestep,
            skip_first_steps=skip_first_steps,
            sparsity=sparsity,
            latent_shape=latent_shape,
            txt_len=txt_len,
            video_spans=video_spans,
            sparse_type=sparse_type,
            precision=precision,
            block_size=block_size,
            inner_precise=inner_precise,
            dense_last_steps=dense_last_steps,
            num_steps=num_steps,
            dense_precision=dense_precision,
            dense_kwargs=dict(dense_kwargs or {}),
            sparse_kwargs=dict(kwargs),
        )


class RainFusionAttentionV2Impl(AttentionImpl):
    def __init__(
        self,
        num_heads,
        head_size,
        causal,
        softmax_scale,
        num_kv_heads=None,
        prefix="",
        **extra_impl_args,
    ):
        from mindiesd import quant_attention, sparse_attention

        self.quant_attention = quant_attention
        self.sparse_attention = sparse_attention
        self.softmax_scale = softmax_scale
        self.causal = causal

    def _dense_attention(self, query, key, value, metadata):
        options = dict(metadata.dense_kwargs)
        options.update(
            precision=metadata.dense_precision,
            layout="BSND",
            scale=self.softmax_scale,
        )
        if metadata.dense_precision == "fp8":
            options.setdefault("fp8_fa_mode", "HIGH_PRECISION")
        elif metadata.dense_precision == "mxfp4":
            options.update(layout_kv="BSND", layout_out="BSND")
        logger.info_once(
            "MindIE-SD dense attention: quant_attention, precision=%s.",
            metadata.dense_precision,
        )
        output = torch.empty_like(query)
        for index in range(query.shape[0]):
            result = self.quant_attention(
                query[index : index + 1].contiguous(),
                key[index : index + 1].contiguous(),
                value[index : index + 1].contiguous(),
                **options,
            )
            output[index : index + 1].copy_(
                result[0] if isinstance(result, tuple) else result
            )
        return output

    def _sparse_attention(self, query, key, value, metadata):
        block_size = metadata.block_size
        if block_size is None:
            block_size = (
                256
                if metadata.sparse_type == "rf_v3"
                and metadata.precision in ("fp8", "mxfp4")
                else 128
            )
        options = dict(metadata.sparse_kwargs)
        options.update(
            sparse_type=metadata.sparse_type,
            precision=metadata.precision,
            block_size=block_size,
            inner_precise=metadata.inner_precise,
            sparsity=metadata.sparsity,
            input_layout="BSND",
            head_num=query.shape[2],
            scale=self.softmax_scale,
        )
        options.setdefault("is_causal", self.causal)
        if metadata.sparse_type in ("rf_v2", "rf_v3"):
            spans = metadata.video_spans
            if spans is None and metadata.txt_len:
                spans = [
                    {"start": metadata.txt_len, "latent_shape": metadata.latent_shape}
                ]
            if spans is not None:
                options.update(video_spans=spans)
            else:
                options.update(
                    txt_len=metadata.txt_len,
                    latent_shape_q=metadata.latent_shape,
                    latent_shape_k=metadata.latent_shape,
                )
        logger.info_once(
            "MindIE-SD sparse attention: sparse_type=%s, precision=%s.",
            metadata.sparse_type,
            metadata.precision,
        )
        output = torch.empty_like(query)
        for index in range(query.shape[0]):
            output[index : index + 1].copy_(
                self.sparse_attention(
                    query[index : index + 1].contiguous(),
                    key[index : index + 1].contiguous(),
                    value[index : index + 1].contiguous(),
                    **options,
                )
            )
        return output

    def forward(self, query, key, value, attn_metadata):
        if (
            attn_metadata.current_timestep < attn_metadata.skip_first_steps
            or (
                attn_metadata.dense_last_steps
                and attn_metadata.current_timestep
                >= attn_metadata.num_steps - attn_metadata.dense_last_steps
            )
            or attn_metadata.sparsity == 0
            or attn_metadata.video_spans == []
        ):
            return self._dense_attention(query, key, value, attn_metadata)
        return self._sparse_attention(query, key, value, attn_metadata)

    def forward_varlen(
        self, query, key, value, *, cu_seqlens, max_seqlen, cu_seqlens_host=None
    ):
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        bounds = cu_seqlens_host if cu_seqlens_host is not None else cu_seqlens.tolist()
        metadata = get_forward_context().attn_metadata
        output = torch.empty_like(query)
        for start, end in zip(bounds[:-1], bounds[1:]):
            if start == end:
                continue
            spans = metadata.video_spans
            if spans is not None:
                spans = [
                    {
                        "start": span["start"] - start,
                        "latent_shape": span["latent_shape"],
                    }
                    for span in spans
                    if start <= span["start"] < end
                ]
            local_metadata = replace(metadata, video_spans=spans)
            output[start:end] = self.forward(
                query[start:end].unsqueeze(0),
                key[start:end].unsqueeze(0),
                value[start:end].unsqueeze(0),
                local_metadata,
            )[0]
        return output
