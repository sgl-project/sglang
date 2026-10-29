# SPDX-License-Identifier: Apache-2.0
"""MindIE-SD EQBSA MIX self-attention for BF16 packed multimodal inputs.

Sparse steps use EagleQBSA (INT8 Q/K and FP8 V), not the other RFv3 kernels.
Packed sequences and non-video context retain their original token layout.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from inspect import Parameter, signature
from math import prod

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
_EQBSA_BLOCK_SIZE = 128


class EQBSAAttentionBackend(AttentionBackend):
    @staticmethod
    def get_supported_head_sizes():
        return [128]

    @staticmethod
    def get_enum():
        return AttentionBackendEnum.EQBSA_ATTN

    @staticmethod
    def get_impl_cls():
        return EQBSAAttentionImpl

    @staticmethod
    def get_metadata_cls():
        return EQBSAAttentionMetadata

    @staticmethod
    def get_builder_cls():
        return EQBSAAttentionMetadataBuilder


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_precision(precision):
    if precision != "mix":
        raise ValueError(
            "EQBSA Attention only supports precision='mix' "
            "(INT8 Q/K and FP8 V); BF16, FP8, and MXFP4 BSA modes are unsupported"
        )


def _dense_schedule_reason(
    current_timestep, skip_first_steps, dense_last_steps, num_steps
):
    _nonnegative_int(current_timestep, "current_timestep")
    _nonnegative_int(skip_first_steps, "skip_first_steps")
    _nonnegative_int(dense_last_steps, "dense_last_steps")
    if num_steps is not None:
        if _nonnegative_int(num_steps, "num_steps") == 0:
            raise ValueError("num_steps must be positive")
        if current_timestep >= num_steps:
            raise ValueError("current_timestep must be smaller than num_steps")
    elif dense_last_steps:
        raise ValueError("dense_last_steps requires num_steps")
    if current_timestep < skip_first_steps:
        return "dense warmup steps"
    if dense_last_steps and current_timestep >= max(0, num_steps - dense_last_steps):
        return "dense tail steps"
    return None


def _shape(value, name):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three positive token-grid dimensions")
    result = tuple(_nonnegative_int(v, name) for v in value)
    if not all(result):
        raise ValueError(f"{name} dimensions must be positive")
    return result


def validate_video_spans(video_spans, sequence_length=None):
    """Copy spans without changing physical offsets; [] means no video rows."""
    if not isinstance(video_spans, (list, tuple)):
        raise ValueError("video_spans must be a sequence")
    result = []
    previous_end = 0
    for span in video_spans:
        if not isinstance(span, dict) or set(span) != {"start", "latent_shape"}:
            raise ValueError("Each video span requires start and latent_shape")
        start = _nonnegative_int(span["start"], "video span start")
        shape = _shape(span["latent_shape"], "video span latent_shape")
        end = start + prod(shape)
        if start < previous_end or (
            sequence_length is not None and end > sequence_length
        ):
            raise ValueError(
                "video_spans must be ordered, nonoverlapping, and within the sequence"
            )
        result.append({"start": start, "latent_shape": list(shape)})
        previous_end = end
    return result


@dataclass
class EQBSAAttentionMetadata(AttentionMetadata):
    skip_first_steps: int = 10
    sparsity: float = 0.2
    latent_shape: tuple[int, int, int] | None = None
    txt_len: int = 0
    video_spans: list[dict] | None = None
    precision: str = "mix"
    dense_last_steps: int = 0
    num_steps: int | None = None

    def __post_init__(self):
        _validate_precision(self.precision)
        _dense_schedule_reason(
            self.current_timestep,
            self.skip_first_steps,
            self.dense_last_steps,
            self.num_steps,
        )


class EQBSAAttentionMetadataBuilder(AttentionMetadataBuilder):
    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(
        self,
        current_timestep: int,
        skip_first_steps: int = 10,
        sparsity: float = 0.2,
        raw_latent_shape=None,
        patch_size=None,
        txt_len: int = 0,
        video_spans=None,
        precision: str = "mix",
        dense_last_steps: int = 0,
        num_steps: int | None = None,
        **kwargs,
    ) -> EQBSAAttentionMetadata:
        _nonnegative_int(current_timestep, "current_timestep")
        _nonnegative_int(skip_first_steps, "skip_first_steps")
        _dense_schedule_reason(
            current_timestep, skip_first_steps, dense_last_steps, num_steps
        )
        _nonnegative_int(txt_len, "txt_len")
        if (
            isinstance(sparsity, bool)
            or not isinstance(sparsity, (int, float))
            or not 0 <= sparsity < 1
        ):
            raise ValueError("sparsity must be in [0, 1)")
        _validate_precision(precision)
        shape = None
        if video_spans is not None:
            if txt_len or raw_latent_shape is not None:
                raise ValueError(
                    "video_spans cannot be combined with txt_len or raw_latent_shape"
                )
            video_spans = validate_video_spans(video_spans)
        elif raw_latent_shape is not None:
            if not isinstance(raw_latent_shape, (list, tuple)):
                raise ValueError("raw_latent_shape must contain token-grid dimensions")
            raw = _shape(raw_latent_shape[-3:], "raw_latent_shape")
            patch = _shape(patch_size, "patch_size")
            if any(dim % size for dim, size in zip(raw, patch)):
                raise ValueError("raw_latent_shape must be divisible by patch_size")
            shape = tuple(dim // size for dim, size in zip(raw, patch))
        else:
            raise ValueError(
                "EQBSA requires explicit video_spans or a video latent shape"
            )
        return EQBSAAttentionMetadata(
            current_timestep=current_timestep,
            skip_first_steps=skip_first_steps,
            sparsity=sparsity,
            latent_shape=shape,
            txt_len=txt_len,
            video_spans=video_spans,
            precision=precision,
            dense_last_steps=dense_last_steps,
            num_steps=num_steps,
        )


class EQBSAAttentionImpl(AttentionImpl):
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
        if causal or extra_impl_args.get("dropout_p", 0.0):
            raise ValueError(
                "EQBSA Attention does not support causal attention or dropout"
            )
        if extra_impl_args.get("is_cross_attention", False):
            raise ValueError("EQBSA Attention only supports self-attention")
        if _nonnegative_int(num_heads, "num_heads") == 0:
            raise ValueError("EQBSA Attention requires a positive head count")
        if num_kv_heads is not None and num_kv_heads != num_heads:
            raise ValueError("EQBSA Attention requires equal Q/K/V head counts")
        if head_size not in EQBSAAttentionBackend.get_supported_head_sizes():
            raise ValueError("EQBSA MIX Attention supports head size 128 only")
        # Metadata and backend discovery remain usable without MindIE-SD.
        try:
            from mindiesd import sparse_attention
        except ImportError as error:
            raise ImportError(
                "EQBSA Attention requires MindIE-SD sparse_attention with explicit "
                "rf_v3 MIX precision and video_spans support."
            ) from error
        try:
            parameters = signature(sparse_attention).parameters
        except (TypeError, ValueError) as error:
            raise ImportError(
                "EQBSA Attention requires a MindIE-SD sparse_attention API with "
                "explicit precision and video_spans parameters; upgrade MindIE-SD."
            ) from error
        if any(
            name not in parameters
            or parameters[name].kind
            not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
            for name in ("precision", "video_spans")
        ):
            # Older APIs accept **kwargs but silently ignore the requested MIX
            # operator. Such versions cannot satisfy the EQBSA contract.
            raise ImportError(
                "EQBSA Attention requires a MindIE-SD sparse_attention API with "
                "explicit precision and video_spans parameters; upgrade MindIE-SD."
            )
        self.sparse_attention = sparse_attention
        self.quant_attention = None
        self.softmax_scale = softmax_scale
        self.head_size = head_size
        self.num_heads = num_heads
        self.packed_trailing_padding = extra_impl_args.get(
            "packed_trailing_padding", False
        )
        if not isinstance(self.packed_trailing_padding, bool):
            raise ValueError("packed_trailing_padding must be a boolean")

    def _validate_qkv(self, query, key, value, ndim, layout):
        if (
            any(x.ndim != ndim for x in (query, key, value))
            or query.shape != key.shape
            or key.shape != value.shape
        ):
            raise ValueError(
                f"EQBSA requires equal Q/K/V shapes in {layout} self-attention layout"
            )
        if any(x.dtype != torch.bfloat16 for x in (query, key, value)):
            raise ValueError("EQBSA Attention requires BF16 Q/K/V inputs")
        if any(x.device != query.device for x in (key, value)):
            raise ValueError("EQBSA Attention requires Q/K/V on the same device")
        if query.shape[-1] != self.head_size:
            raise ValueError("EQBSA Attention requires head size 128")
        # Ulysses partitions the constructor's heads before backend dispatch.
        if query.shape[-2] <= 0 or self.num_heads % query.shape[-2]:
            raise ValueError(
                "EQBSA Attention requires a positive runtime head count "
                "dividing the configured head count"
            )

    def _validate_metadata(self, metadata, sequence_length):
        if not isinstance(metadata, EQBSAAttentionMetadata):
            raise ValueError(
                "EQBSA Attention requires EQBSAAttentionMetadata with an explicit token layout"
            )
        _validate_precision(metadata.precision)
        reason = _dense_schedule_reason(
            metadata.current_timestep,
            metadata.skip_first_steps,
            metadata.dense_last_steps,
            metadata.num_steps,
        )
        _nonnegative_int(metadata.txt_len, "txt_len")
        if (
            isinstance(metadata.sparsity, bool)
            or not isinstance(metadata.sparsity, (int, float))
            or not 0 <= metadata.sparsity < 1
        ):
            raise ValueError("sparsity must be in [0, 1)")
        if metadata.video_spans is not None:
            if metadata.txt_len or metadata.latent_shape is not None:
                raise ValueError(
                    "video_spans cannot be combined with txt_len or latent_shape"
                )
            validate_video_spans(metadata.video_spans, sequence_length)
        else:
            shape = _shape(metadata.latent_shape, "latent_shape")
            if sequence_length != metadata.txt_len + prod(shape):
                raise ValueError(
                    "Legacy EQBSA layout must be [txt_len prefix, T*H*W video tokens]; use video_spans for mixed sequences"
                )
        return reason

    def _dense_attention(self, query, key, value, *, reason, fallback=False, **options):
        log = logger.warning_once if fallback else logger.info_once
        log(
            "EQBSA dense dispatch: operator=mindiesd.quant_attention; "
            f"precision=fp8; fp8_fa_mode=HIGH_PRECISION; reason={reason}."
        )
        if self.quant_attention is None:
            try:
                from mindiesd import quant_attention
            except (ImportError, OSError) as error:
                raise ImportError(
                    "EQBSA dense steps require MindIE-SD quant_attention with "
                    "FP8 FIA HIGH_PRECISION support."
                ) from error
            self.quant_attention = quant_attention
        output = torch.empty_like(query)
        for index in range(query.shape[0]):
            q, k, v = (
                tensor[index : index + 1].transpose(1, 2).contiguous()
                for tensor in (query, key, value)
            )
            result = self.quant_attention(
                q,
                k,
                v,
                precision="fp8",
                layout="BNSD",
                scale=self.softmax_scale,
                fp8_fa_mode="HIGH_PRECISION",
                pre_tokens=2147483647,
                next_tokens=2147483647,
            )
            output[index : index + 1].copy_(result.transpose(1, 2))
        return output

    def _eqbsa_sparse_attention(self, query, key, value, metadata, **options):
        """Select only the RFv3 MIX path, which dispatches to EagleQBSA."""
        spans = metadata.video_spans
        if spans is None:
            shape, txt_len = metadata.latent_shape, metadata.txt_len
            if shape is None or query.shape[1] != txt_len + prod(shape):
                raise ValueError(
                    "Legacy EQBSA layout must be [txt_len prefix, T*H*W video tokens]; use video_spans for mixed sequences"
                )
            # Protect every text-containing block at MIX's shared Q/KV size.
            video_length = prod(shape)
            b = _EQBSA_BLOCK_SIZE
            if txt_len and (
                (video_length + txt_len + b - 1) // b - video_length // b
                > (txt_len + b - 1) // b
            ):
                spans = [{"start": txt_len, "latent_shape": list(shape)}]
        else:
            spans = validate_video_spans(spans, query.shape[1])

        shapes = [s["latent_shape"] for s in spans] if spans is not None else [shape]
        lengths = [prod(shape) for shape in shapes]
        fillers = sum(-length % _EQBSA_BLOCK_SIZE for length in lengths[:-1])
        if any(h < 8 or w < 8 or (t == 1 and (h % 8 or w % 8)) for t, h, w in shapes):
            return self._dense_attention(
                query,
                key,
                value,
                reason="video grid is unsupported by MindIE-SD rearrangement",
                fallback=True,
                **options,
            )
        if query.shape[1] - sum(lengths) < fillers:
            return self._dense_attention(
                query,
                key,
                value,
                reason="insufficient dense context tokens for video block alignment",
                fallback=True,
                **options,
            )
        if spans is None:
            options.update(
                txt_len=metadata.txt_len,
                latent_shape_q=list(metadata.latent_shape),
                latent_shape_k=list(metadata.latent_shape),
            )
        else:
            options.update(video_spans=spans)
        logger.info_once(
            "EQBSA sparse dispatch: operator=mindiesd.sparse_attention(rf_v3) "
            "-> mindiesd.eagle_quant_block_sparse_attention; "
            "precision=mix (INT8 Q/K, FP8 V); Q/KV block_size=128; reason=sparse step."
        )
        output = torch.empty_like(query)
        for index in range(query.shape[0]):
            result = self.sparse_attention(
                query[index : index + 1].contiguous(),
                key[index : index + 1].contiguous(),
                value[index : index + 1].contiguous(),
                sparse_type="rf_v3",
                inner_precise=4,
                block_size=_EQBSA_BLOCK_SIZE,
                sparsity=metadata.sparsity,
                precision="mix",
                **options,
            )
            output[index : index + 1].copy_(result)
        return output

    def forward(self, query, key, value, attn_metadata):
        self._validate_qkv(query, key, value, 4, "BSND")
        reason = self._validate_metadata(attn_metadata, query.shape[1])
        if query.shape[0] == 0 or query.shape[1] == 0:
            return torch.empty_like(query)
        options = dict(
            input_layout="BSND", head_num=query.shape[2], scale=self.softmax_scale
        )
        if reason is not None:
            return self._dense_attention(query, key, value, reason=reason, **options)
        if attn_metadata.sparsity == 0:
            return self._dense_attention(
                query, key, value, reason="zero sparsity", **options
            )
        if attn_metadata.video_spans == []:
            return self._dense_attention(
                query, key, value, reason="sequence has no video spans", **options
            )
        return self._eqbsa_sparse_attention(query, key, value, attn_metadata, **options)

    def forward_varlen(
        self, query, key, value, *, cu_seqlens, max_seqlen, cu_seqlens_host=None
    ):
        self._validate_qkv(query, key, value, 3, "TND")
        _nonnegative_int(max_seqlen, "max_seqlen")
        if (
            not isinstance(cu_seqlens, torch.Tensor)
            or cu_seqlens.ndim != 1
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("cu_seqlens must be a one-dimensional integer tensor")
        if cu_seqlens_host is not None and not isinstance(cu_seqlens_host, Sequence):
            raise ValueError("cu_seqlens_host must be a sequence of integers")
        bounds = tuple(
            cu_seqlens_host if cu_seqlens_host is not None else cu_seqlens.tolist()
        )
        if (
            len(bounds) < 2
            or len(bounds) != cu_seqlens.numel()
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in bounds
            )
            or bounds[0] != 0
            or bounds[-1] != query.shape[0]
            or any(a > b for a, b in zip(bounds, bounds[1:]))
        ):
            raise ValueError("cu_seqlens must monotonically cover all packed tokens")
        lengths = [b - a for a, b in zip(bounds, bounds[1:])]
        if (
            self.packed_trailing_padding
            and len(bounds) == 3
            and bounds[1] == max_seqlen
            and bounds[1] < bounds[2]
        ):
            # The physical alignment segment remains an independent sequence;
            # H3's max_seqlen describes only its real prefix.
            lengths = lengths[:-1]
        if any(length > max_seqlen for length in lengths):
            raise ValueError("max_seqlen is smaller than a packed sequence")
        from sglang.multimodal_gen.runtime.managers.forward_context import (
            get_forward_context,
        )

        metadata = get_forward_context().attn_metadata
        if (
            not isinstance(metadata, EQBSAAttentionMetadata)
            or metadata.video_spans is None
        ):
            raise ValueError(
                "Packed EQBSA requires explicit global video_spans metadata"
            )
        self._validate_metadata(metadata, query.shape[0])
        spans = validate_video_spans(metadata.video_spans, query.shape[0])
        for span in spans:
            end = span["start"] + prod(span["latent_shape"])
            if not any(
                a <= span["start"] and end <= b for a, b in zip(bounds, bounds[1:])
            ):
                raise ValueError("A video span cannot cross packed sequence boundaries")
        output = torch.empty_like(query)
        # All segments are independent, including H3's alignment segment. It
        # must never become part of the real sequence's keys or values.
        for start, end in zip(bounds, bounds[1:]):
            if start == end:
                continue
            local = replace(
                metadata,
                video_spans=[
                    {
                        "start": span["start"] - start,
                        "latent_shape": span["latent_shape"],
                    }
                    for span in spans
                    if start <= span["start"] < end
                ],
            )
            output[start:end] = self.forward(
                query[start:end].unsqueeze(0),
                key[start:end].unsqueeze(0),
                value[start:end].unsqueeze(0),
                local,
            )[0]
        return output
