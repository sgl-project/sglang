# SPDX-License-Identifier: Apache-2.0
"""Ascend sparse attention with a shared AscendFA dense executor."""

from dataclasses import dataclass, field, replace
from math import isfinite, prod

import torch
from mindiesd import sparse_attention

from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
    AscendFAImpl,
    validate_attention_batch_config,
    validate_attention_config,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    trailing_padding_used_len,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class AscendSparseAttentionBackend(AttentionBackend):
    @staticmethod
    def get_enum():
        return AttentionBackendEnum.ASCEND_SPARSE_ATTN

    @staticmethod
    def get_impl_cls():
        return AscendSparseAttentionImpl

    @staticmethod
    def get_metadata_cls():
        return AscendSparseAttentionMetadata

    @staticmethod
    def get_builder_cls():
        return AscendSparseAttentionMetadataBuilder


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name, minimum, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be a finite number in {interval}")


def _grid(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three positive integers")
    return [_integer(dim, name, 1) for dim in value]


def _video_spans(spans):
    if not isinstance(spans, (list, tuple)):
        raise ValueError(
            "video_spans must be a sequence of start/latent_shape mappings"
        )
    copied = []
    previous_end = 0
    for span in spans:
        if not isinstance(span, dict) or set(span) != {"start", "latent_shape"}:
            raise ValueError("Each video span requires start and latent_shape")
        start = _integer(span["start"], "video span start")
        shape = _grid(span["latent_shape"], "video span latent_shape")
        if start < previous_end:
            raise ValueError("video_spans must be ordered and nonoverlapping")
        previous_end = start + prod(shape)
        copied.append({"start": start, "latent_shape": shape})
    return copied


@dataclass
class AscendSparseAttentionMetadata(AttentionMetadata):
    skip_first_steps: int = 10
    skip_last_steps: int = 0
    num_steps: int | None = None
    sparsity: float = 0.6
    latent_shape: list[int] | None = None
    txt_len: int = 0
    video_spans: list[dict] | None = None
    sparse_type: str = "rf_v3"
    precision: str = "bf16"
    block_size: int = 128
    keep_sink: bool = True
    keep_recent: bool = True
    cdf_threshold: float = 1.0
    mxfp4_dst_type_max: float = 0.0
    mxfp4_scale_alg: int | None = None
    dense_kwargs: dict = field(default_factory=lambda: {"precision": "bf16"})


class AscendSparseAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(
        self,
        current_timestep,
        skip_first_steps=10,
        skip_last_steps=None,
        num_steps=None,
        sparsity=0.6,
        raw_latent_shape=None,
        patch_size=None,
        txt_len=0,
        video_spans=None,
        sparse_type="rf_v3",
        precision="bf16",
        block_size=None,
        keep_sink=True,
        keep_recent=True,
        cdf_threshold=1.0,
        mxfp4_dst_type_max=0.0,
        mxfp4_scale_alg=None,
        dense_kwargs=None,
        # Migration aliases for the previous PR configuration.
        dense_last_steps=None,
        dense_precision=None,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(
                f"Unknown Ascend sparse attention options: {', '.join(sorted(kwargs))}"
            )
        if skip_last_steps is not None:
            _integer(skip_last_steps, "skip_last_steps")
        if dense_last_steps is not None:
            _integer(dense_last_steps, "dense_last_steps")
            if skip_last_steps is not None and skip_last_steps != dense_last_steps:
                raise ValueError("skip_last_steps and dense_last_steps disagree")
            skip_last_steps = dense_last_steps
        if skip_last_steps is None:
            skip_last_steps = 0
        _integer(current_timestep, "current_timestep")
        _integer(skip_first_steps, "skip_first_steps")
        _integer(skip_last_steps, "skip_last_steps")
        if num_steps is not None:
            _integer(num_steps, "num_steps", 1)
            if current_timestep >= num_steps:
                raise ValueError("current_timestep must be less than num_steps")
        elif skip_last_steps:
            raise ValueError("num_steps is required when skip_last_steps > 0")
        if sparse_type not in ("rf_v2", "rf_v3", "ada_bsa"):
            raise ValueError(
                "sparse_type must be rf_v2, rf_v3, or ada_bsa; use fa for dense attention"
            )
        if precision not in ("bf16", "mix", "fp8", "mxfp4"):
            raise ValueError("precision must be bf16, mix, fp8, or mxfp4")
        _number(sparsity, "sparsity", 0, 1)
        if block_size is None:
            block_size = (
                256 if sparse_type == "rf_v3" and precision in ("fp8", "mxfp4") else 128
            )
        _integer(block_size, "block_size", 1)
        _integer(txt_len, "txt_len")
        for name, value in (("keep_sink", keep_sink), ("keep_recent", keep_recent)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        _number(cdf_threshold, "cdf_threshold", 0, 1)
        _number(mxfp4_dst_type_max, "mxfp4_dst_type_max", 0)
        if mxfp4_scale_alg is not None:
            _integer(mxfp4_scale_alg, "mxfp4_scale_alg")
        if dense_kwargs is not None and not isinstance(dense_kwargs, dict):
            raise ValueError("dense_kwargs must be a dictionary")
        dense_config = dict(dense_kwargs or {})
        if dense_precision is not None:
            if (
                "precision" in dense_config
                and dense_config["precision"] != dense_precision
            ):
                raise ValueError("dense_precision and dense_kwargs.precision disagree")
            dense_config["precision"] = dense_precision
        dense_config = validate_attention_config(dense_config)

        latent_shape = None
        if video_spans is not None:
            if raw_latent_shape is not None or txt_len:
                raise ValueError(
                    "video_spans cannot be combined with raw_latent_shape or txt_len"
                )
            video_spans = _video_spans(video_spans)
        elif raw_latent_shape is not None:
            if (
                not isinstance(raw_latent_shape, (list, tuple))
                or len(raw_latent_shape) < 3
            ):
                raise ValueError(
                    "raw_latent_shape must include three spatial/temporal dimensions"
                )
            raw = _grid(raw_latent_shape[-3:], "raw_latent_shape")
            patch = _grid(patch_size, "patch_size")
            if any(dim % size for dim, size in zip(raw, patch)):
                raise ValueError("raw_latent_shape must be divisible by patch_size")
            latent_shape = [dim // size for dim, size in zip(raw, patch)]
        elif sparse_type in ("rf_v2", "rf_v3"):
            raise ValueError(
                "Rain Fusion requires video_spans or raw_latent_shape and patch_size"
            )
        return AscendSparseAttentionMetadata(
            current_timestep=current_timestep,
            skip_first_steps=skip_first_steps,
            skip_last_steps=skip_last_steps,
            num_steps=num_steps,
            sparsity=sparsity,
            latent_shape=latent_shape,
            txt_len=txt_len,
            video_spans=video_spans,
            sparse_type=sparse_type,
            precision=precision,
            block_size=block_size,
            keep_sink=keep_sink,
            keep_recent=keep_recent,
            cdf_threshold=cdf_threshold,
            mxfp4_dst_type_max=mxfp4_dst_type_max,
            mxfp4_scale_alg=mxfp4_scale_alg,
            dense_kwargs=dense_config,
        )


class AscendSparseAttentionImpl(AttentionImpl):
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
        self.softmax_scale = softmax_scale
        self.causal = causal
        self.inner_precise = 4 if torch.npu.get_soc_version() >= 260 else 0
        self.packed_trailing_padding = extra_impl_args.get(
            "packed_trailing_padding", False
        )
        if not isinstance(self.packed_trailing_padding, bool):
            raise ValueError("packed_trailing_padding must be a boolean")
        self._dense_impl_args = dict(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
            **extra_impl_args,
        )
        self._dense_config_key = None
        self._dense_impl = None

    def _dense_attention(self, query, key, value, metadata):
        # Per-request metadata can change independently of global server settings.
        config = metadata.dense_kwargs
        config_key = tuple(
            (name, id(option) if isinstance(option, torch.Tensor) else option)
            for name, option in sorted(config.items())
        )
        if self._dense_impl is None or config_key != self._dense_config_key:
            self._dense_impl = AscendFAImpl(
                **self._dense_impl_args, attention_config=config
            )
            self._dense_config_key = config_key
        return self._dense_impl.forward(query, key, value, None)

    def _sparse_attention(self, query, key, value, metadata):
        if self.causal and metadata.sparse_type in ("rf_v2", "rf_v3"):
            raise ValueError("Rain Fusion does not support causal attention")
        options = dict(
            sparse_type=metadata.sparse_type,
            precision=metadata.precision,
            block_size=metadata.block_size,
            inner_precise=self.inner_precise,
            sparsity=metadata.sparsity,
            input_layout="BSND",
            head_num=query.shape[2],
            scale=self.softmax_scale,
            is_causal=self.causal,
            keep_sink=metadata.keep_sink,
            keep_recent=metadata.keep_recent,
            cdf_threshold=metadata.cdf_threshold,
            mxfp4_dst_type_max=metadata.mxfp4_dst_type_max,
            mxfp4_scale_alg=metadata.mxfp4_scale_alg,
        )
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
            "Ascend sparse attention: sparse_type=%s, precision=%s.",
            metadata.sparse_type,
            metadata.precision,
        )
        return torch.cat(
            [
                sparse_attention(
                    query[index : index + 1].contiguous(),
                    key[index : index + 1].contiguous(),
                    value[index : index + 1].contiguous(),
                    **options,
                )
                for index in range(query.shape[0])
            ],
            dim=0,
        )

    def forward(self, query, key, value, attn_metadata):
        if not isinstance(attn_metadata, AscendSparseAttentionMetadata):
            raise RuntimeError(
                "Ascend sparse attention requires AscendSparseAttentionMetadata"
            )
        if query.numel() == 0:
            return torch.empty_like(query)
        if (
            attn_metadata.current_timestep < attn_metadata.skip_first_steps
            or (
                attn_metadata.skip_last_steps
                and attn_metadata.current_timestep
                >= attn_metadata.num_steps - attn_metadata.skip_last_steps
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
            get_forward_context_or_none,
        )

        context = get_forward_context_or_none()
        metadata = context.attn_metadata if context is not None else None
        if not isinstance(metadata, AscendSparseAttentionMetadata):
            raise RuntimeError(
                "Packed Ascend sparse attention requires attention metadata in the forward context"
            )
        _integer(max_seqlen, "max_seqlen")
        if any(tensor.ndim != 3 for tensor in (query, key, value)):
            raise ValueError("Packed Ascend sparse attention requires TND tensors")
        if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
            raise ValueError("Packed Q/K/V must have matching token counts")
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype not in (torch.int32, torch.int64):
            raise ValueError("cu_seqlens must be a 1D integer tensor")
        if cu_seqlens_host is not None and len(cu_seqlens_host) != cu_seqlens.numel():
            raise ValueError("cu_seqlens and its host copy must have the same length")
        bounds = tuple(
            cu_seqlens_host if cu_seqlens_host is not None else cu_seqlens.tolist()
        )
        if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != query.shape[0]:
            raise ValueError(
                "cu_seqlens must cover the packed tokens from 0 to the token count"
            )
        for start, end in zip(bounds[:-1], bounds[1:]):
            _integer(start, "cu_seqlens boundary")
            _integer(end, "cu_seqlens boundary")
            if start > end:
                raise ValueError("cu_seqlens must be nondecreasing")
        validate_attention_batch_config(
            metadata.dense_kwargs, bounds=bounds, total_tokens=query.shape[0]
        )
        used = (
            trailing_padding_used_len(query.shape[0], max_seqlen, bounds)
            if self.packed_trailing_padding
            else None
        )
        if any(
            end - start > max_seqlen
            for start, end in zip(bounds[:-1], bounds[1:])
            if used is None or start < used
        ):
            raise ValueError("A packed document exceeds max_seqlen")
        spans = metadata.video_spans
        if (
            spans is None
            and metadata.sparse_type in ("rf_v2", "rf_v3")
            and sum(end > start for start, end in zip(bounds[:-1], bounds[1:])) > 1
            and used is None
        ):
            raise ValueError(
                "Packed Rain Fusion documents require explicit video_spans"
            )
        if spans:
            live_end = query.shape[0] if used is None else used
            if spans[-1]["start"] + prod(spans[-1]["latent_shape"]) > live_end:
                raise ValueError("video_spans must lie within the live packed tokens")
        for span in spans or ():
            span_start = span["start"]
            span_end = span_start + prod(span["latent_shape"])
            if not any(
                start <= span_start and span_end <= end
                for start, end in zip(bounds[:-1], bounds[1:])
            ):
                raise ValueError("A video span cannot cross packed document boundaries")
        output = torch.empty_like(query) if used is None else torch.zeros_like(query)
        for start, end in zip(bounds[:-1], bounds[1:]):
            if start == end or (used is not None and start >= used):
                continue
            local_spans = spans
            if spans is not None:
                local_spans = []
                for span in spans:
                    span_start = span["start"]
                    span_end = span_start + prod(span["latent_shape"])
                    if span_start < end and span_end > start:
                        local_spans.append(
                            {
                                "start": span_start - start,
                                "latent_shape": span["latent_shape"],
                            }
                        )
            local_metadata = replace(metadata, video_spans=local_spans)
            output[start:end] = self.forward(
                query[start:end].unsqueeze(0),
                key[start:end].unsqueeze(0),
                value[start:end].unsqueeze(0),
                local_metadata,
            )[0]
        return output
