# ruff: noqa: BLE001, RUF012, SIM102
"""Torch-native Muse 1.2 / Muse Spark text model for SGLang.

BF16 checkpoints run a plain ``torch.nn.functional.linear`` expert loop on CUDA
and a fused grouped GEMM on ROCm (``ops/muse_moe_triton.py``, or
``ops/muse_moe_flydsl.py`` when ``MUSE_MOE_BACKEND=flydsl``).
Serialized ModelOpt NVFP4 expert weights stay packed and are executed by the
public FlashInfer TRT-LLM MoE kernel via the public ModelOpt quant method. Muse
applies its post-expert norm between GEMM2 and the top-k combine, which that
kernel fuses together, so the combine is finished separately by the CuTeDSL op
in ``sglang_meta_common/ops`` (with a torch fallback).
"""

from __future__ import annotations

import csv
import functools
import logging
import os
import re
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.attention.verify_reduce_norm_gate import (
    DEFERRAL as VERIFY_REDUCE_DEFERRAL,
)
from sglang.kernels.ops.gemm.bf16_tiled_small_m_gemm import LAYOUT as TILED_GEMM_LAYOUT
from sglang.kernels.ops.gemm.bf16_tiled_small_m_gemm import (
    prepare_tiled_weight,
    rowmajor_gemm,
    rowmajor_gemm_applies,
    tiled_gemm,
    tiled_gemm_applies,
)
from sglang.kernels.ops.gemm.fp8_prefill_gemm import (
    fp8_prefill_gemm,
    fp8_prefill_gemm_applies,
    quantize_rows_fp8,
)
from sglang.kernels.ops.gemm.fp8w_small_m_gemm import (
    fp8_gemm,
    fp8_gemm_applies,
)
from sglang.kernels.ops.gemm.fp8w_small_m_gemm import (
    quantize_weight as quantize_fp8w_weight,
)
from sglang.kernels.ops.gemm.router_gemm_fp32 import (
    router_gemm_tuned,
    router_gemv_fp32,
    router_gemv_fp32_supported,
)
from sglang.kernels.ops.gemm.unpool_gemm_norm_residual import (
    unpool_gemm_norm_residual,
)
from sglang.srt.configs.muse_spark import (
    MuseSparkTextConfig,
    should_skip_weight_loading,
)
from sglang.srt.distributed import get_tp_group, tensor_model_parallel_all_reduce
from sglang.srt.environ import envs
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor as OSSLogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import (
    register_gfx950_unquantized_gemm_shapes,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.rocm_muse_spark import finalize_allreduce, mxfp4_moe, router_pooler
from sglang.srt.rocm_muse_spark.attention_kv_write import (
    FUSED_QKV_CACHE,
    QK_ROPE_CACHE,
    qk_norm_rope_kv_write,
    qkv_proj_norm_rope_kv_write,
)
from sglang.srt.rocm_muse_spark.fp4_moe import MuseSpark1xFp4Experts
from sglang.srt.rocm_muse_spark.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.srt.rocm_muse_spark.fused_norm_gate import FusedRMSNormGate
from sglang.srt.rocm_muse_spark.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.srt.rocm_muse_spark.layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
    QKVOGateParallelLinear,
)
from sglang.srt.rocm_muse_spark.muse_qk_norm import fused_qk_norm, qk_norm_supported
from sglang.srt.rocm_muse_spark.sequence_parallel import (
    SPShape,
    prepare_ulysses,
    sp_gather,
    sp_reduce_scatter,
    sp_shape,
    sp_shard,
    ulysses_attention,
    ulysses_enabled,
    ulysses_prepared,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class MuseSpark1xLogitsProcessor(OSSLogitsProcessor):
    """Public logits pipeline with the checkpoint-requested accumulator dtype."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        # Same as models/muse_spark.py (#146): the lm_head output and the final-logit
        # softcap run in fp32; the released checkpoints carry no use_fp32_for_output flag.
        self.use_fp32_output = True

    def _compute_lm_head(self, hidden_states, lm_head, embedding_bias=None):
        if not hasattr(lm_head, "weight"):
            return super()._compute_lm_head(hidden_states, lm_head, embedding_bias)
        output_dtype = torch.float32 if self.use_fp32_output else lm_head.weight.dtype
        logits = torch.mm(
            hidden_states.to(lm_head.weight.dtype),
            lm_head.weight.T,
            out_dtype=output_dtype,
        )
        if embedding_bias is not None:
            logits = logits + embedding_bias
        return logits


_FINALIZE_AR_PUSH = envs.SGLANG_MUSE_FINALIZE_AR_PUSH.get()


def _is_rocm() -> bool:
    return getattr(torch.version, "hip", None) is not None


def _moe_backend() -> str:
    backend = os.environ.get("MUSE_MOE_BACKEND", "auto").lower()
    if backend not in {"auto", "native", "triton", "flydsl"}:
        raise ValueError(
            "MUSE_MOE_BACKEND must be one of auto, native, triton, or flydsl; "
            f"got {backend!r}"
        )
    if backend == "auto":
        return "triton" if _is_rocm() else "native"
    return backend


def _want_triton_moe() -> bool:
    """ROCm defaults to the fused Triton expert path; CUDA stays on the PyTorch loop.

    Explicit ``native``, ``triton``, and ``flydsl`` values select that backend.
    Triton and FlyDSL are graph-capturable, so they are used for both eager and
    captured forwards.
    """
    return _moe_backend() == "triton"


def _want_fused_moe() -> bool:
    """Grouped GEMM expert path that keeps Triton weight layouts (Triton or FlyDSL)."""
    return _moe_backend() in {"triton", "flydsl"}


def _run_fused_moe(
    hidden_states: torch.Tensor,
    topk_scores: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    local_expert_start: int,
    post_expert_norm: nn.Module | None,
    w13_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    if _moe_backend() == "flydsl":
        from sglang.srt.rocm_muse_spark.muse_moe_flydsl import (
            muse_moe_flydsl as fused_moe,
        )
    else:
        from sglang.srt.rocm_muse_spark.muse_moe_triton import (
            muse_moe_triton as fused_moe,
        )

    weight, gain, has_norm = _post_expert_norm_args(post_expert_norm)
    eps = float(post_expert_norm.eps) if post_expert_norm is not None else 0.0
    kwargs = {}
    if w13_scale is not None or w2_scale is not None:
        kwargs["w13_scale"] = w13_scale
        kwargs["w2_scale"] = w2_scale
    return fused_moe(
        hidden_states,
        topk_scores,
        topk_ids,
        w13,
        w2,
        int(local_expert_start),
        weight,
        eps,
        post_norm_gain=gain,
        has_norm=has_norm,
        **kwargs,
    )


def _post_expert_norm_args(
    post_expert_norm: nn.Module | None,
) -> tuple[torch.Tensor | None, float, bool]:
    """Weight / gain / has_norm without allocating tensors (unsafe under capture)."""
    if post_expert_norm is None:
        return None, 0.0, False
    weight = getattr(post_expert_norm, "weight", None)
    gain = 0.0
    if weight is not None and getattr(post_expert_norm, "zero_centered_gamma", False):
        gain = float(post_expert_norm.gain_center)
    return weight, gain, True


def _quantization_name(quant_config: QuantizationConfig | None) -> str | None:
    if quant_config is None:
        return None
    get_name = getattr(quant_config, "get_name", None)
    return get_name() if get_name is not None else type(quant_config).__name__


def _is_quark_mxfp4(quant_config: QuantizationConfig | None) -> bool:
    """Return whether this is the AMD Quark OCP-MX FP4 expert format."""
    if _quantization_name(quant_config) != "quark":
        return False
    config = getattr(quant_config, "quant_config", None) or {}
    global_config = (
        config.get("global_quant_config", {}) if isinstance(config, dict) else {}
    )
    weight = (global_config or {}).get("weight") or {}
    return (
        weight.get("dtype") == "fp4"
        and weight.get("scale_format") == "e8m0"
        and weight.get("group_size") == 32
    )


def _uses_quark_mxfp4_experts(
    quant_config: QuantizationConfig | None, moe_prefix: str
) -> bool:
    """
    Whether this layer's experts are packed MXFP4 rather than left in BF16.

    Quark exports can be partially quantized: Muse Spark 1.2 MXFP4 keeps two of
    the 62 MoE layers in BF16. The exclude list names them the way its own
    checkpoint generation spelled them, so both spellings are tried below.
    """
    if not _is_quark_mxfp4(quant_config):
        return False
    layer_prefix = moe_prefix.rpartition(".")[0]
    if _is_nvfp4_source(quant_config):
        # Requantized ModelOpt export: Quark turns the producer's exclude list
        # (substrings such as "layers.0." / "layers.61.", ModelOpt semantics) into
        # anchored fnmatch regexes, which the generic module-path matcher would
        # never match against the full module name. Search them unanchored.
        target = add_prefix("mlp.experts", layer_prefix)
        for pattern in getattr(quant_config, "exclude_layers", None) or []:
            if pattern.startswith("re:"):
                body = pattern[3:]
                if body.endswith("\\Z"):
                    body = body[:-2]
                if re.search(body, target):
                    return False
            elif pattern in target:
                return False
        return True
    excluded = getattr(quant_config, "exclude_layers", None) or []
    return not is_layer_skipped(add_prefix("mlp.experts", layer_prefix), excluded)


def _is_nvfp4_source(quant_config: QuantizationConfig | None) -> bool:
    """Whether a Quark MXFP4 config requantizes from an NVFP4 export."""
    source = getattr(quant_config, "dequantization_config", None)
    if source is None:
        return False
    try:
        from sglang.srt.layers.quantization.quark.utils import Nvfp4SourceConfig
    except ImportError:  # pragma: no cover
        return False
    return isinstance(source, Nvfp4SourceConfig)


def _uses_nvfp4_experts(
    quant_config: QuantizationConfig | None, layer_prefix: str
) -> bool:
    if _quantization_name(quant_config) != "modelopt_fp4":
        return False
    excluded = getattr(quant_config, "exclude_modules", None) or []
    return not is_layer_skipped(add_prefix("mlp.experts", layer_prefix), excluded)


# The released config states the per-layer schedule outright: a zero rope base marks a
# NoPE layer and layer_types marks the sliding ones. Reading the lists keeps this in step
# with the checkpoint instead of re-deriving the pattern from every_n_layers_nope.
def _compute_use_rope(config, layer_idx: int) -> bool:
    """A zero rope base marks a NoPE layer."""
    return bool(config.layer_rope_theta[layer_idx])


def _compute_sliding_window(config, layer_idx: int) -> int | None:
    """The config states the span; the kernels want the causal offset."""
    if config.layer_types[layer_idx] != "sliding_attention":
        return None
    return config.sliding_window - 1


# One-launch per-head q/k RMSNorm (+ query scale) on ROCm; MUSE_FUSE_QK_NORM=0 restores the eager path.
_USE_FUSED_QK_NORM = (
    os.environ.get("MUSE_FUSE_QK_NORM", "1") == "1" and torch.version.hip is not None
)
# Unquantized linear layers on the row-major small-M GEMM: the TP4 attention O projection. (The QKV projection
# shares the joint router-pooler shape [4608, 8192] and stays on hipBLASLt.)
_ROWMAJOR_LINEAR_SHAPES = ((8192, 2048), (8192, 1024))  # O-proj at TP4 / TP8
# Unquantized linear layers with an FP8 weight copy: the TP4 QKV (+ gate) projection, fed the per-row FP8 copy of
# the attention input that the previous FFN boundary kernel writes (decode / verify up to 128 rows).
_FP8W_LINEAR_SHAPES = ((4608, 8192), (2304, 8192))  # QKV at TP4 / TP8
_DENSE_FP8 = envs.SGLANG_MUSE_DENSE_FP8.get()
_FP8_MAX_ROWS = 128
# Plain extend batches above _FP8_MAX_ROWS run the QKV / O projections and the pooler / unpooler on the FP8
# weight copies through aiter's CK GEMM from each weight's own minimum width (kernels/ops/gemm/fp8_prefill_gemm);
# the O projection's copy and the pooler / unpooler copies exist for this path only.
_PREFILL_DENSE_FP8 = _DENSE_FP8 and envs.SGLANG_MUSE_PREFILL_DENSE_FP8.get()
# Target verify: the attention partials' split-KV merge runs inside the output norm-gate kernel.
_VERIFY_REDUCE_NORM_GATE = envs.SGLANG_MUSE_VERIFY_REDUCE_NORM_GATE.get()
# Decode-sized fp32 router as one split-K Triton GEMV (+reduce); MUSE_ROUTER_GEMV=0 restores torch.mm.
_USE_ROUTER_GEMV = (
    os.environ.get("MUSE_ROUTER_GEMV", "1") == "1" and torch.version.hip is not None
)

# Replicated pre_expert_proj/post_expert_proj GEMMs at decode-sized M stay on nn.Linear (hipBLASLt). aiter's tuned
# GEMM is opt-in via MUSE_AITER_DENSE=1: with the pinned aiter its table resolves these shapes to kernels that are
# slower than hipBLASLt on MI355X (pre_expert_proj M=1..32: 21 us vs 15-19 us; post_expert_proj M<=64: 42 us vs
# 14-18 us).
_USE_AITER_DENSE = (
    os.environ.get("MUSE_AITER_DENSE", "0") == "1" and torch.version.hip is not None
)
_tgemm = None

# MI355X AgentX fast paths (switches in sglang.srt.environ, all on by default).
# Pooler / unpooler at the verify widths aiter patch 06 tunes go through aiter's
# tuned BF16 GEMM (searched hipBLASLt solutions); the FP32 router runs a tuned
# split-K GEMM (configs in kernels/ops/gemm/configs).
_DENSE_TUNED = envs.SGLANG_MUSE_DENSE_TUNED.get()
# Tiled 17..64-row GEMMs: the pooler here, unquantized linear layers through
# register_gfx950_unquantized_gemm_shapes (the O projection, the DFlash draft MLP).
_DENSE_TILED = envs.SGLANG_MUSE_DENSE_TILED.get()
# The DFlash draft's gate/up and down GEMMs at the measured M = 24 (TP8) on the
# aiter Triton a16w16 kernel instead of the tuned table's padded-M pick.
_DRAFT_TRITON_A16W16_GEMMS = ((24, 7168, 8192), (24, 8192, 3584))
# Unpooler GEMM fused with the next FFN boundary's norms at decode / verify widths.
_FUSED_UNPOOL_BOUNDARY = envs.SGLANG_MUSE_FUSED_UNPOOL_BOUNDARY.get()
# Device memory left for the KV pool, CUDA graphs and activations beside both MXFP4 expert layouts; below it
# the experts keep only the a4w4 preshuffled copy. Arbitrary, not tuned.
_A4W4_RESERVE_BYTES = 64 << 30
_dense_tuned_logged: set = set()


def _log_dense_tuned(kind: str, key) -> None:
    if key not in _dense_tuned_logged:
        _dense_tuned_logged.add(key)
        logger.info("Muse dense tuned: %s %s", kind, key)


def _get_tgemm():
    global _tgemm
    if _tgemm is None:
        try:
            from aiter.tuned_gemm import tgemm as _t

            _tgemm = _t
        except Exception:  # aiter without tuned_gemm: stay on torch
            _tgemm = False
    return _tgemm


@functools.lru_cache(maxsize=1)
def _aiter_tuned_moe_projections() -> frozenset:
    """(N, K, M) of the pooler / unpooler GEMMs that aiter patch 06 tunes for this GPU.

    Read from aiter's own table (model_configs/muse_spark_bf16_tuned_gemm.csv), so it
    is the single source of the solutions; only these exact M leave nn.Linear (any
    other M would hit aiter's padded-M lookup of other models' rows).
    """
    try:
        import aiter
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    except Exception:
        return frozenset()
    path = os.path.join(
        os.path.dirname(aiter.__file__),
        "configs",
        "model_configs",
        "muse_spark_bf16_tuned_gemm.csv",
    )
    if not os.path.exists(path):
        return frozenset()
    gfx, cu_num = get_gfx(), get_cu_num()
    shapes = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            n, k = int(row["N"]), int(row["K"])
            if (
                (n, k) in ((4096, 8192), (8192, 4096))
                and row["gfx"] == gfx
                and int(row["cu_num"]) == cu_num
                and row["bias"] == "False"
                and row["outdtype"] == "torch.bfloat16"
            ):
                shapes.add((n, k, int(row["M"])))
    return frozenset(shapes)


def _prefill_fp8_rows(forward_batch: ForwardBatch, rows: int) -> bool:
    """Whether this forward's dense GEMMs over ``rows`` activation rows run on the FP8 weight copies."""
    return (
        _PREFILL_DENSE_FP8
        and rows > _FP8_MAX_ROWS
        and forward_batch.forward_mode.is_extend_without_speculative()
    )


def _dense_linear(
    x: torch.Tensor,
    linear: nn.Linear,
    max_m: int,
    *,
    prefill_fp8: bool = False,
    x_fp8: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """``linear(x)`` on the fastest route for this width; with ``prefill_fp8`` the FP8 weight copy, fed ``x_fp8`` =
    (per-row FP8 copy of ``x``, scales) when a boundary kernel produced it."""
    if prefill_fp8 and fp8_prefill_gemm_applies(x, linear.small_m_fp8_weight):
        xq, xs = x_fp8 if x_fp8 is not None else quantize_rows_fp8(x)
        return fp8_prefill_gemm(xq, xs, linear.small_m_fp8_weight)
    if (
        _DENSE_TILED
        and linear.bias is None
        and rowmajor_gemm_applies(x, linear.weight)
    ):
        return rowmajor_gemm(x, linear.weight)
    tiled = getattr(linear, "small_m_tiled_weight", None)
    if tiled is not None and linear.bias is None and tiled_gemm_applies(x, tiled):
        return tiled_gemm(x, tiled, *linear.weight.shape)
    if (
        _DENSE_TUNED
        and linear.bias is None
        and x.dim() == 2
        and x.dtype == torch.bfloat16
        and linear.weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and linear.weight.is_contiguous()
        and (*linear.weight.shape, x.shape[0]) in _aiter_tuned_moe_projections()
        and _get_tgemm()
    ):
        _log_dense_tuned("gemm", (*linear.weight.shape, x.shape[0]))
        return _tgemm.mm(x, linear.weight, None, otype=torch.bfloat16)
    if (
        _USE_AITER_DENSE
        and linear.bias is None
        and x.dtype == torch.bfloat16
        and linear.weight.dtype == torch.bfloat16
        and 0 < x.shape[0] <= max_m
        and _get_tgemm()
    ):
        return _tgemm.mm(x, linear.weight, None, otype=torch.bfloat16)
    return linear(x)


class MuseSpark1xRouter(nn.Module):
    """Router with the checkpoint-native ``[hidden, experts]`` layout."""

    def __init__(self, hidden_size: int, num_experts: int, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(hidden_size, num_experts, dtype=dtype), requires_grad=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.mm(hidden_states, self.weight)


class MuseSpark1xAttention(nn.Module):
    def __init__(self, config: MuseSparkTextConfig, layer_idx: int, prefix: str):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.use_o_gate = True
        self.use_o_norm = True
        self.use_qk_norm = True
        self.use_rope = _compute_use_rope(config, layer_idx)
        sliding_window = _compute_sliding_window(config, layer_idx)
        # Attention temperature tuning is not part of the released architecture.
        self.use_attn_temperature_tuning = False
        self.temperature_floor = None
        self.temperature_q_scale = None

        parallel = get_parallel()
        tp_size = parallel.attn_tp_size
        if config.num_attention_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by attention TP")
        if config.num_key_value_heads >= tp_size:
            if config.num_key_value_heads % tp_size:
                raise ValueError(
                    "num_key_value_heads must be divisible by attention TP"
                )
        elif tp_size % config.num_key_value_heads:
            raise ValueError("attention TP must divide replicated KV heads")
        self.tp_size = tp_size
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = max(1, config.num_key_value_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.scale_query_by = config.qk_scale_factor
        # Gathered Ulysses projections (sequence_parallel.prepare_ulysses): BF16, or FP8 copies + channel scales.
        self.ulysses_qkv_weight = None
        self.ulysses_o_weight = None
        self.ulysses_qkv_fp8 = None
        self.ulysses_o_fp8 = None

        if self.use_o_gate:
            self.qkv_proj = QKVOGateParallelLinear(
                hidden_size=config.hidden_size,
                head_size=self.head_dim,
                total_num_heads=config.num_attention_heads,
                total_num_kv_heads=config.num_key_value_heads,
                bias=False,
                prefix=add_prefix("qkv_proj", prefix),
                parallel_group="attn_tp",
            )
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=config.hidden_size,
                head_size=self.head_dim,
                total_num_heads=config.num_attention_heads,
                total_num_kv_heads=config.num_key_value_heads,
                bias=False,
                quant_config=None,
                prefix=add_prefix("qkv_proj", prefix),
                parallel_group="attn_tp",
            )
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * self.head_dim,
            output_size=config.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=None,
            prefix=add_prefix("o_proj", prefix),
            parallel_group="attn_tp",
        )
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=getattr(
                    config, "context_len", config.max_position_embeddings
                ),
                base=int(config.layer_rope_theta[layer_idx]),
                is_neox_style=False,
                dtype=torch.float32,
            )
            if self.use_rope
            else None
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            prefix=add_prefix("attn", prefix),
            use_irope=self.use_rope,
            sliding_window_size=sliding_window or -1,
        )
        self.norm_eps = config.rms_norm_eps
        # The output norm and the sigmoid gate are one fused kernel. The norm is
        # weightless here, so this module carries no parameters -- it exists to
        # hand the op its eps.
        self._o_norm_module = (
            MuseSpark1xRMSNorm(self.head_dim, eps=self.norm_eps, has_weight=False)
            if self.use_o_norm
            else None
        )
        self._fused_o_norm_gate = FusedRMSNormGate(
            self._o_norm_module, self.use_o_gate, self.head_dim
        )

    def _per_head_norm(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        return (
            F.rms_norm(
                value.reshape(-1, self.head_dim),
                (self.head_dim,),
                None,
                self.norm_eps,
            )
            .reshape_as(value)
            .to(dtype)
        )

    def qk_norm_rope(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_qk_norm:
            if _USE_FUSED_QK_NORM and qk_norm_supported(q, k, self.head_dim):
                q, k = fused_qk_norm(
                    q, k, self.head_dim, self.norm_eps, self.scale_query_by
                )
            else:
                q = self._per_head_norm(q) * self.scale_query_by
                k = self._per_head_norm(k)
        if self.use_attn_temperature_tuning:
            temperature = (
                torch.log((positions.float() + 1.0) / self.temperature_floor)
                * self.temperature_q_scale
                + 1.0
            ).clamp_min(1.0)
            q = (q * temperature.unsqueeze(-1)).to(q.dtype)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        reduce: bool = True,
        x_fp8: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """``x_fp8`` = (per-row FP8 copy, scales) of ``hidden_states`` from the boundary kernel: the QKV projection
        runs on the FP8 weight copy when both exist for this width."""
        if hidden_states.shape[0] == 0:
            return hidden_states
        prefill_fp8 = _prefill_fp8_rows(forward_batch, hidden_states.shape[0])
        # Decode / verify: the fused QK-norm / RoPE kernels also store K/V into
        # the FP8 cache, so the attention layer must not store them again.
        projected = (
            qkv_proj_norm_rope_kv_write(self, positions, hidden_states, forward_batch)
            if FUSED_QKV_CACHE and QK_ROPE_CACHE and _USE_FUSED_QK_NORM
            else None
        )
        if projected is not None:
            q, k, v, o_gate = projected
            kv_written = True
        else:
            qkv_fp8 = (
                self.qkv_proj.small_m_fp8_weight
                if x_fp8 is not None or prefill_fp8
                else None
            )
            if (
                qkv_fp8 is not None
                and x_fp8 is not None
                and fp8_gemm_applies(x_fp8[0], qkv_fp8)
            ):
                qkv = fp8_gemm(x_fp8[0], x_fp8[1], qkv_fp8)
            elif prefill_fp8 and fp8_prefill_gemm_applies(hidden_states, qkv_fp8):
                xq, xs = (
                    x_fp8 if x_fp8 is not None else quantize_rows_fp8(hidden_states)
                )
                qkv = fp8_prefill_gemm(xq, xs, qkv_fp8)
            else:
                qkv, _ = self.qkv_proj(hidden_states)
            split_sizes = [self.q_size, self.kv_size, self.kv_size]
            if self.use_o_gate:
                q, k, v, o_gate = qkv.split(split_sizes + [self.q_size], dim=-1)
            else:
                q, k, v = qkv.split(split_sizes, dim=-1)
                o_gate = None
            fused_qk = (
                qk_norm_rope_kv_write(self, positions, q, k, v, forward_batch)
                if QK_ROPE_CACHE and _USE_FUSED_QK_NORM
                else None
            )
            kv_written = fused_qk is not None
            q, k = fused_qk if kv_written else self.qk_norm_rope(positions, q, k)
        defer = (
            _VERIFY_REDUCE_NORM_GATE
            and forward_batch.forward_mode.is_target_verify()
            and self._fused_o_norm_gate.merges_verify_partials()
        )
        VERIFY_REDUCE_DEFERRAL.requested = defer
        try:
            output = self.attn(q, k, v, forward_batch, save_kv_cache=not kv_written)
        finally:
            VERIFY_REDUCE_DEFERRAL.requested = False
        state = VERIFY_REDUCE_DEFERRAL.take() if defer else None
        if state is not None:
            output = self._fused_o_norm_gate.forward_verify_partials(state, output, o_gate)
        else:
            output = self._fused_o_norm_gate.forward(output, o_gate)
        o_fp8 = self.o_proj.small_m_fp8_weight if prefill_fp8 else None
        if prefill_fp8 and fp8_prefill_gemm_applies(output, o_fp8):
            xq, xs = quantize_rows_fp8(output)
            output = fp8_prefill_gemm(xq, xs, o_fp8)
            # RowParallelLinear's reduction for this attention-TP layer (no symmetric memory, no DP attention here).
            if reduce and self.tp_size > 1:
                output = tensor_model_parallel_all_reduce(output)
            return output
        output, _ = self.o_proj(output, skip_all_reduce=not reduce)
        return output


class MuseSpark1xMLP(nn.Module):
    def __init__(self, config: MuseSparkTextConfig, prefix: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=config.hidden_size,
            output_sizes=[config.intermediate_size] * 2,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            input_size=config.intermediate_size,
            output_size=config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        output, _ = self.down_proj(F.silu(gate) * up)
        return output


class MuseSpark1xNativeExperts(nn.Module):
    """Local expert weights: PyTorch loop on CUDA, fused Triton or FlyDSL MoE on ROCm."""

    def __init__(
        self,
        num_local_experts: int,
        expert_hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.expert_hidden_size = expert_hidden_size
        self.intermediate_size = intermediate_size
        self.w13_weight = nn.Parameter(
            torch.empty(
                num_local_experts,
                2 * intermediate_size,
                expert_hidden_size,
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                num_local_experts,
                expert_hidden_size,
                intermediate_size,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        local_expert_start: int,
        post_expert_norm: nn.Module | None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        if _want_fused_moe():
            return _run_fused_moe(
                hidden_states,
                topk_scores,
                topk_ids,
                self.w13_weight,
                self.w2_weight,
                int(local_expert_start),
                post_expert_norm,
            )
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return self._forward_capturable(
                hidden_states,
                topk_scores,
                topk_ids,
                local_expert_start,
                post_expert_norm,
            )
        output = torch.zeros_like(hidden_states)
        for local_id in range(self.num_local_experts):
            global_id = local_expert_start + local_id
            token_slot = (topk_ids == global_id).nonzero(as_tuple=False)
            if token_slot.numel() == 0:
                continue
            token_ids = token_slot[:, 0]
            slots = token_slot[:, 1]
            selected = hidden_states.index_select(0, token_ids)
            gate_up = F.linear(selected, self.w13_weight[local_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, self.w2_weight[local_id])
            if post_expert_norm is not None:
                expert_output = post_expert_norm(expert_output)
            expert_output = expert_output * topk_scores[token_ids, slots].unsqueeze(-1)
            output.index_add_(0, token_ids, expert_output.to(output.dtype))
        return output

    def _forward_capturable(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        local_expert_start: int,
        post_expert_norm: nn.Module | None,
    ) -> torch.Tensor:
        """Fixed-shape equivalent of ``forward`` for CUDA graph capture.

        ``forward`` picks out each expert's tokens with ``nonzero`` and skips
        the experts that drew none. Both read device data on the host, which
        stream capture rejects. Here every local expert instead runs over all
        tokens and is scaled by its routing weight, which is zero for a token
        that did not select it -- the same result with no host synchronisation,
        paid for by running each local expert densely.
        """
        output = torch.zeros_like(hidden_states)
        for local_id in range(self.num_local_experts):
            global_id = local_expert_start + local_id
            # A token's top-k ids are distinct, so at most one slot matches and
            # this sum is exactly that slot's score (zero when unrouted).
            weight = (topk_scores * (topk_ids == global_id)).sum(dim=-1, keepdim=True)
            gate_up = F.linear(hidden_states, self.w13_weight[local_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(F.silu(gate) * up, self.w2_weight[local_id])
            if post_expert_norm is not None:
                expert_output = post_expert_norm(expert_output)
            output = output + (expert_output * weight).to(output.dtype)
        return output


class MuseSpark1xMxfp4Experts(nn.Module):
    """AMD Quark MXFP4 experts with packed E2M1 weights and E8M0 scales."""

    def __init__(
        self,
        num_local_experts: int,
        expert_hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.expert_hidden_size = expert_hidden_size
        self.intermediate_size = intermediate_size
        # Set by mxfp4_moe.prepare_a4w4_weights; native_dropped means only the preshuffled copy is left.
        self.a4w4_weights = None
        self.native_dropped = False
        self.w13_weight = nn.Parameter(
            torch.empty(
                num_local_experts,
                2 * intermediate_size,
                expert_hidden_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.w13_weight_scale = nn.Parameter(
            torch.empty(
                num_local_experts,
                2 * intermediate_size,
                expert_hidden_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                num_local_experts,
                expert_hidden_size,
                intermediate_size // 2,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.w2_weight_scale = nn.Parameter(
            torch.empty(
                num_local_experts,
                expert_hidden_size,
                intermediate_size // 32,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_scores: torch.Tensor,
        topk_ids: torch.Tensor,
        local_expert_start: int,
        post_expert_norm: nn.Module | None,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        if not _want_fused_moe():
            raise RuntimeError(
                "Quark MXFP4 experts require MUSE_MOE_BACKEND=triton or flydsl"
            )
        return _run_fused_moe(
            hidden_states,
            topk_scores,
            topk_ids,
            self.w13_weight,
            self.w2_weight,
            int(local_expert_start),
            post_expert_norm,
            w13_scale=self.w13_weight_scale,
            w2_scale=self.w2_weight_scale,
        )

    _forward_capturable = forward


class MuseSpark1xMoE(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        layer_idx: int,
        nvfp4_checkpoint: bool,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        if config.num_shared_experts:
            raise ValueError("Muse Spark OSS supports routed experts only")
        parallel = get_parallel()
        if parallel.moe_tp_size != 1:
            raise ValueError(
                "The torch-native Muse MoE requires expert TP1; use EP equal to TP"
            )
        if config.num_local_experts % parallel.moe_ep_size:
            raise ValueError("num_experts must be divisible by expert parallel size")
        self.layer_idx = layer_idx
        self._compact_a4_reported = False
        self.tp_size = parallel.tp_size
        self.ep_size = parallel.moe_ep_size
        self.ep_rank = parallel.moe_ep_rank
        self.num_experts = config.num_local_experts
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.route_eps = 1e-15
        self.router_multiplier = config.output_multiplier
        self.router_in_fp32 = True
        self.hidden_size = config.hidden_size
        self.expert_hidden_size = config.moe_hidden_size
        self.intermediate_size = config.moe_intermediate_size
        router_dtype = (
            torch.float32 if self.router_in_fp32 else torch.get_default_dtype()
        )
        self.router = MuseSpark1xRouter(
            self.hidden_size, self.num_experts, dtype=router_dtype
        )
        self.register_buffer(
            "expert_bias", torch.zeros(self.num_experts, dtype=torch.float32)
        )
        self.pre_expert_proj = (
            nn.Linear(self.hidden_size, self.expert_hidden_size, bias=False)
            if self.expert_hidden_size != self.hidden_size
            else None
        )
        self.post_expert_proj = (
            nn.Linear(self.expert_hidden_size, self.hidden_size, bias=False)
            if self.expert_hidden_size != self.hidden_size
            else None
        )
        # (FP8 copy, channel scales) of the pooler / unpooler for the prefill GEMMs, made after loading.
        for proj in (self.pre_expert_proj, self.post_expert_proj):
            if proj is not None:
                proj.small_m_fp8_weight = None
        self.pre_expert_norm = MuseSpark1xRMSNorm(
            self.expert_hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            residual_id=2 * layer_idx + 2,
        )
        self.post_expert_norm = MuseSpark1xRMSNormWithInputScale(
            self.expert_hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
        )
        self.nvfp4_checkpoint = nvfp4_checkpoint
        self.mxfp4_checkpoint = _uses_quark_mxfp4_experts(quant_config, prefix)
        if nvfp4_checkpoint:
            # Packed NVFP4: the FlashInfer TRT-LLM kernel runs the experts and we
            # finalize in torch so the post-expert norm lands before the top-k sum.
            self.experts = MuseSpark1xFp4Experts(
                num_experts=self.num_experts,
                num_local_experts=self.num_local_experts,
                local_expert_offset=self.local_expert_start,
                hidden_size=self.expert_hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.top_k,
                quant_config=quant_config,
            )
        elif self.mxfp4_checkpoint:
            self.experts = MuseSpark1xMxfp4Experts(
                self.num_local_experts,
                self.expert_hidden_size,
                self.intermediate_size,
            )
        else:
            self.experts = MuseSpark1xNativeExperts(
                self.num_local_experts,
                self.expert_hidden_size,
                self.intermediate_size,
            )
        # Collective and allocating: built at model construction, never under graph capture.
        self._finalize_ar = (
            finalize_allreduce.ensure_finalize_allreduce(
                get_tp_group().cpu_group,
                torch.device("cuda", torch.cuda.current_device()),
            )
            if _FINALIZE_AR_PUSH
            and _is_rocm()
            and mxfp4_moe.COMPACT_A4_ENABLED
            and self.mxfp4_checkpoint
            and self.tp_size == self.ep_size == 4
            and self.num_local_experts == 64
            and self.top_k == 8
            and self.expert_hidden_size == 4096
            else None
        )

    def _routing_op(self) -> FusedMuseSparkRouting:
        """The fused router: sigmoid, biased top-k, gather, renormalise, scale.

        Built lazily and rebound if ``expert_bias`` is replaced, because
        ``Module._apply`` (``.to()``, ``.cuda()``) swaps the buffer for a new
        tensor and the op holds a direct reference to it.
        """
        op = getattr(self, "_fused_routing_op", None)
        if op is None or op.expert_bias is not self.expert_bias:
            op = FusedMuseSparkRouting(
                expert_bias=self.expert_bias,
                route_scale=self.route_scale,
                router_output_multiplier=self.router_multiplier,
                router_score_division_eps=self.route_eps,
            )
            object.__setattr__(self, "_fused_routing_op", op)
        return op

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        skip_post_expert_proj: bool = False,
        target_verify: bool = False,
        sp: SPShape | None = None,
        x_fp8: tuple[torch.Tensor, torch.Tensor] | None = None,
        prefill_fp8: bool = False,
    ) -> torch.Tensor:
        """MoE on ``hidden_states``; under sequence parallelism (``sp``) they are this
        rank's token shard and so is the result. ``skip_post_expert_proj`` returns the
        reduced expert output before the unpooler (fused into the next boundary).
        ``x_fp8`` = (per-row FP8 copy, scales) of ``hidden_states`` for the FP8 pooler;
        ``prefill_fp8`` runs the pooler / unpooler on their FP8 copies (prefill widths)."""
        quantize_a4 = sp is None and self.uses_compact_a4(
            hidden_states.shape[0], target_verify
        )
        topk_scores, topk_ids, expert_input, quantized = self.route(
            hidden_states, quantize_a4=quantize_a4, x_fp8=x_fp8, prefill_fp8=prefill_fp8
        )
        rows = topk_ids.shape[0] if sp is None else sp.padded
        use_a4w4 = self.uses_a4w4(rows, target_verify)
        if sp is not None:
            if use_a4w4:
                # Row-local MXFP4: gather the shards' FP4 data and scales, not BF16 rows.
                a1, a1s = mxfp4_moe.quantize_rows(expert_input)
                quantized = (
                    sp_gather(a1.view(torch.uint8), rows).view(a1.dtype),
                    sp_gather(a1s.view(torch.uint8), rows).view(a1s.dtype),
                )
                expert_input = None
            else:
                expert_input = sp_gather(expert_input, rows)
            topk_ids = sp_gather(topk_ids, rows)
            topk_scores = sp_gather(topk_scores, rows)
        if (
            sp is None
            and self._finalize_ar is not None
            and not use_a4w4
            and self.uses_compact_a4(rows, target_verify)
            and self._finalize_ar.applies(rows)
        ):
            norm_weight, gain, _ = _post_expert_norm_args(self.post_expert_norm)
            output = finalize_allreduce.run_compact_a4_allreduce(
                self._finalize_ar,
                expert_input,
                topk_scores,
                topk_ids,
                self.experts,
                self.local_expert_start,
                norm_weight,
                self.post_expert_norm.eps,
                gain,
                quantized=quantized,
            )
            if skip_post_expert_proj:
                return output
            return self.unpool(output, prefill_fp8=prefill_fp8)
        output = self.run_experts(
            expert_input, topk_scores, topk_ids, target_verify, use_a4w4, quantized
        )
        if sp is not None:
            output = sp_reduce_scatter(output, sp.padded)
        elif self.tp_size > 1:
            output = tensor_model_parallel_all_reduce(output)
        if skip_post_expert_proj:
            return output
        return self.unpool(output, prefill_fp8=prefill_fp8)

    def route(
        self,
        hidden_states: torch.Tensor,
        quantize_a4: bool = False,
        x_fp8: tuple[torch.Tensor, torch.Tensor] | None = None,
        prefill_fp8: bool = False,
    ):
        """Router top-k, the expert input (pooler + pre-expert norm) of these rows and, when ``quantize_a4`` and
        the shared epilogue runs, that input's compact A4 MXFP4 form (else None). ``x_fp8`` / ``prefill_fp8`` as in
        ``forward``."""
        if (
            router_pooler.SHARED_ROUTER_POOLER
            and _USE_ROUTER_GEMV
            and (
                router_pooler.can_share_router_pooler(self, hidden_states)
                or router_pooler.can_route_fp8_joint(self, hidden_states, x_fp8)
            )
        ):
            m = hidden_states.shape[0]
            quant_out = (
                (
                    hidden_states.new_empty((m, 2048), dtype=torch.uint8),
                    hidden_states.new_empty((m, 128), dtype=torch.uint8),
                )
                if quantize_a4
                else None
            )
            return router_pooler.route_with_shared_pooler(
                self, hidden_states, quant_out=quant_out, x_fp8=x_fp8
            )
        if (
            _USE_ROUTER_GEMV
            and self.router_in_fp32
            and router_gemv_fp32_supported(hidden_states, self.router.weight)
        ):
            # bf16 -> fp32 conversion happens inside the kernel (exact), IEEE fp32 accumulate
            router_logits = router_gemv_fp32(hidden_states, self.router.weight)
        else:
            router_logits = None
            if _DENSE_TUNED and self.router_in_fp32:
                router_logits = router_gemm_tuned(hidden_states, self.router.weight)
                if router_logits is not None:
                    _log_dense_tuned("router", hidden_states.shape[0])
            if router_logits is None:
                router_input = (
                    hidden_states.float() if self.router_in_fp32 else hidden_states
                )
                router_logits = self.router(router_input)
        topk_scores, topk_ids = self._routing_op().forward(
            None, router_logits, self.top_k, True
        )
        expert_input = (
            _dense_linear(
                hidden_states,
                self.pre_expert_proj,
                max_m=128,
                prefill_fp8=prefill_fp8,
                x_fp8=x_fp8,
            )
            if self.pre_expert_proj is not None
            else hidden_states
        )
        if self.pre_expert_norm is not None:
            expert_input = self.pre_expert_norm(expert_input)
        return topk_scores, topk_ids, expert_input, None

    def uses_compact_a4(self, rows: int, target_verify: bool) -> bool:
        return (
            mxfp4_moe.COMPACT_A4_ENABLED
            and target_verify
            and self.mxfp4_checkpoint
            and self.tp_size == self.ep_size
            and self.num_local_experts in (32, 64)
            and self.top_k == 8
            and rows % 8 == 0
            and 0 < rows <= mxfp4_moe.COMPACT_A4_MAX_ROWS
        )

    def uses_a4w4(self, rows: int, target_verify: bool) -> bool:
        if not (
            mxfp4_moe.A4W4_ENABLED
            and self.mxfp4_checkpoint
            and self.experts.a4w4_weights is not None
        ):
            return False
        if self.experts.native_dropped:
            # Only the preshuffled copy is left: a4w4 runs everything compact A4 does not.
            return not self.uses_compact_a4(rows, target_verify)
        return not target_verify and rows >= mxfp4_moe.A4W4_MIN_ROWS

    def run_experts(
        self,
        expert_input,
        topk_scores,
        topk_ids,
        target_verify: bool,
        use_a4w4: bool,
        quantized=None,
    ):
        """This rank's experts over all given rows; per-token partial sums before the TP reduction."""
        if use_a4w4:
            norm_weight, gain, has_norm = _post_expert_norm_args(self.post_expert_norm)
            assert has_norm and norm_weight is not None
            return mxfp4_moe.run_a4w4(
                expert_input,
                topk_scores,
                topk_ids,
                self.experts,
                self.local_expert_start,
                norm_weight,
                self.post_expert_norm.eps,
                gain,
                quantized=quantized,
            )
        if self.uses_compact_a4(expert_input.shape[0], target_verify):
            norm_weight, gain, has_norm = _post_expert_norm_args(self.post_expert_norm)
            assert has_norm and norm_weight is not None
            if not self._compact_a4_reported:
                logger.info(
                    "Muse compact A4 target-verify: layer=%d rows=%d local_start=%d",
                    self.layer_idx,
                    expert_input.shape[0],
                    self.local_expert_start,
                )
                self._compact_a4_reported = True
            return mxfp4_moe.run_compact_a4(
                expert_input,
                topk_scores,
                topk_ids,
                self.experts,
                self.local_expert_start,
                norm_weight,
                self.post_expert_norm.eps,
                gain,
                quantized=quantized,
            )
        if self.nvfp4_checkpoint:
            return self.experts(
                expert_input, topk_scores, topk_ids, self.post_expert_norm
            )
        return self.experts(
            expert_input,
            topk_scores,
            topk_ids,
            self.local_expert_start,
            self.post_expert_norm,
        )

    def unpool(self, output: torch.Tensor, prefill_fp8: bool = False) -> torch.Tensor:
        if self.post_expert_proj is not None:
            output = _dense_linear(
                output, self.post_expert_proj, max_m=64, prefill_fp8=prefill_fp8
            )
        return output


class MuseSpark1xDecoderLayer(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        layer_idx: int,
        nvfp4_checkpoint: bool,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = MuseSpark1xAttention(
            config, layer_idx, add_prefix("self_attn", prefix)
        )
        self.post_attn_norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.post_norm_eps,
            has_weight=False,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 1,
        )
        self.pre_feedforward_layernorm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            residual_id=2 * layer_idx + 2,
        )
        self.post_feedforward_layernorm = MuseSpark1xRMSNormWithInputScale(
            config.hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            is_post_norm=True,
            zero_centered_gamma=True,
            post_norm_gain_center_type="one",
            residual_id=2 * layer_idx + 2,
        )
        self.is_moe_layer = self._is_moe_layer(config, layer_idx)
        self.mlp = (
            MuseSpark1xMoE(
                config,
                layer_idx,
                nvfp4_checkpoint,
                add_prefix("mlp", prefix),
                quant_config,
            )
            if self.is_moe_layer
            else MuseSpark1xMLP(config, add_prefix("mlp", prefix))
        )
        self.post_attention_residual_gate = MuseSpark1xNormPreservingResidualConnection(
            config.hidden_size, temperature=config.residual_gate_temperature
        )
        self.post_feedforward_residual_gate = (
            MuseSpark1xNormPreservingResidualConnection(
                config.hidden_size, temperature=config.residual_gate_temperature
            )
        )
        # The attention boundary -- post-norm, residual add, then the FFN
        # pre-norm -- is one fused kernel. The FFN boundary is not fused: its
        # pre-norm belongs to the next layer, so it cannot be expressed here.
        self._fused_attn_boundary = FusedPostNormResidualPreNorm(
            self.post_attn_norm,
            self.post_attention_residual_gate,
            self.pre_feedforward_layernorm,
        )

    @staticmethod
    def _is_moe_layer(config: MuseSparkTextConfig, layer_idx: int) -> bool:
        return config.mlp_layer_types[layer_idx] == "sparse"

    def forward(
        self,
        positions: torch.Tensor,
        stream: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_input: torch.Tensor | None = None,
        ffn_boundary: FusedPostNormResidualPreNorm | None = None,
        pre_norm_dtype: torch.dtype | None = None,
        sp: SPShape | None = None,
        attn_input_fp8: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        """Under sequence parallelism (``sp``) ``stream`` / ``attn_input`` are this rank's
        token shard; attention and the experts still see every token. With ``ffn_boundary`` returns
        (stream, next attention input, its FP8 copy or None)."""
        attn_input = (
            self.self_attn_input_norm(stream) if attn_input is None else attn_input
        )
        prefill_fp8 = _prefill_fp8_rows(forward_batch, attn_input.shape[0])
        if sp is not None and ulysses_prepared(self.self_attn):
            attn_output = ulysses_attention(
                self.self_attn,
                attn_input,
                positions,
                forward_batch,
                sp,
                x_fp8=attn_input_fp8,
            )
        elif sp is not None:
            partial = self.self_attn(
                positions,
                sp_gather(attn_input, sp.num_tokens),
                forward_batch,
                reduce=False,
            )
            attn_output = sp_reduce_scatter(partial, sp.padded)
        else:
            attn_output = self.self_attn(
                positions, attn_input, forward_batch, x_fp8=attn_input_fp8
            )
        target_verify = forward_batch.forward_mode.is_target_verify()
        ffn_input_fp8 = None
        if (
            _DENSE_FP8
            and self.is_moe_layer
            and (
                (
                    sp is None
                    and 0 < attn_output.shape[0] <= _FP8_MAX_ROWS
                    and (forward_batch.forward_mode.is_decode() or target_verify)
                )
                or (
                    prefill_fp8
                    and fp8_prefill_gemm_applies(attn_output, self._pooler_fp8_weight())
                )
            )
        ):
            # The FFN input's per-row FP8 copy feeds the joint router / pooler's FP8 pooler tiles (decode /
            # verify) or the FP8 pooler GEMM (prefill).
            ffn_input, stream, _, fp8, scale = self._fused_attn_boundary.forward_fp8(
                stream, attn_output, branch_dtype=torch.bfloat16
            )
            ffn_input_fp8 = (fp8, scale)
        else:
            ffn_input, stream, _ = self._fused_attn_boundary.forward(
                stream, attn_output, branch_dtype=torch.bfloat16
            )
        if (
            sp is None
            and _FUSED_UNPOOL_BOUNDARY
            and self._can_fuse_unpool_boundary(ffn_input, ffn_boundary)
        ):
            # The MoE keeps its per-expert norm / combine and TP reduction; only the
            # unpooler moves into the next boundary's norm-residual kernel.
            expert_output = self.mlp(
                ffn_input,
                skip_post_expert_proj=True,
                target_verify=target_verify,
                x_fp8=ffn_input_fp8,
            )
            ffn_boundary._ensure_cached_weights(torch.bfloat16)
            next_attn_input, stream = unpool_gemm_norm_residual(
                expert_output,
                self.mlp.post_expert_proj.weight,
                stream,
                ffn_boundary._cc_post_weight,
                ffn_boundary._cc_pre_weight,
                ffn_boundary._cc_alpha,
                ffn_boundary._cc_beta,
                ffn_boundary._post_norm.eps,
                ffn_boundary._pre_norm.eps,
                block_n=64,
                block_k=128,
                split_k=4,
                pre_bf16=pre_norm_dtype is torch.bfloat16,
                num_warps=16,
            )
            return stream, next_attn_input, None
        ffn_output = (
            self.mlp(
                ffn_input,
                target_verify=target_verify,
                sp=sp,
                x_fp8=ffn_input_fp8,
                prefill_fp8=prefill_fp8,
            )
            if self.is_moe_layer
            else self.mlp(ffn_input)
        )
        if ffn_boundary is None:
            return self.post_feedforward_residual_gate(
                stream, self.post_feedforward_layernorm(ffn_output)
            )
        # Cross-layer fusion: post_feedforward_layernorm + gated residual + the NEXT layer's pre_attn_norm
        # (or the final norm) in one kernel; returns the new stream and the next attention input, with its
        # per-row FP8 copy when the next layer's QKV GEMM runs on the FP8 weight (decode / verify widths, and
        # the prefill widths that weight admits).
        if (
            _DENSE_FP8
            and pre_norm_dtype is None
            and (
                (
                    sp is None
                    and 0 < ffn_output.shape[0] <= _FP8_MAX_ROWS
                    and (
                        forward_batch.forward_mode.is_decode()
                        or forward_batch.forward_mode.is_target_verify()
                    )
                )
                or (
                    prefill_fp8
                    and fp8_prefill_gemm_applies(
                        ffn_output, self._prefill_qkv_fp8_weight(sp)
                    )
                )
            )
        ):
            next_attn_input, stream, _, fp8, scale = ffn_boundary.forward_fp8(
                stream, ffn_output, branch_dtype=torch.bfloat16
            )
            return stream, next_attn_input, (fp8, scale)
        next_attn_input, stream, _ = ffn_boundary.forward(
            stream,
            ffn_output,
            branch_dtype=torch.bfloat16,
            pre_norm_dtype=pre_norm_dtype,
        )
        return stream, next_attn_input, None

    def self_attn_input_norm(self, stream: torch.Tensor) -> torch.Tensor:
        return self.self_attn.pre_attn_norm(stream).to(torch.bfloat16)

    def _pooler_fp8_weight(self):
        proj = self.mlp.pre_expert_proj
        return None if proj is None else proj.small_m_fp8_weight

    def _prefill_qkv_fp8_weight(self, sp: SPShape | None):
        """The FP8 weight a prefill QKV projection of the attention input reads (every layer's attention is
        prepared alike), None when it stays BF16; the gathered-shard attention projects the gather, not the shard."""
        attn = self.self_attn
        if sp is None:
            return attn.qkv_proj.small_m_fp8_weight
        return attn.ulysses_qkv_fp8 if ulysses_prepared(attn) else None

    def _can_fuse_unpool_boundary(
        self, x: torch.Tensor, boundary: FusedPostNormResidualPreNorm | None
    ) -> bool:
        """Unpooler + FFN boundary fusion: MoE layers at up to 16 rows (replicated unpooler, any TP)."""
        if not self.is_moe_layer or boundary is None:
            return False
        proj = self.mlp.post_expert_proj
        return (
            proj is not None
            and proj.bias is None
            and 0 < x.shape[0] <= 16
            and x.dtype == torch.bfloat16
            and x.shape[1] == 8192
            and x.is_contiguous()
            and proj.weight.shape == (8192, 4096)
            and proj.weight.dtype == torch.bfloat16
            and proj.weight.is_contiguous()
            and boundary._post_norm_is_input_scale
            and boundary._use_residual_gate
            and boundary._post_norm.has_weight
            and boundary._pre_norm.has_weight
            and boundary._residual_add.gate.numel() == 8192
        )


# Keep the pre-attention norm under self_attn to match Muse checkpoint names.
def _install_pre_attn_norm(
    config: MuseSparkTextConfig, layer: MuseSpark1xDecoderLayer
) -> None:
    layer.self_attn.pre_attn_norm = MuseSpark1xRMSNorm(
        config.hidden_size,
        eps=config.rms_norm_eps,
        has_weight=True,
        zero_centered_gamma=True,
        residual_id=2 * layer.layer_idx + 1,
    )


class MuseSpark1xModel(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        super().__init__()
        self.config = config
        self.residual_dtype = torch.float32
        # Layer indices whose output residual stream feeds the DFLASH draft
        # context (set through MuseSparkTextForCausalLM.set_dflash_layers_to_capture).
        self.layers_to_capture: list[int] = []
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.embed_norm = MuseSpark1xRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_hidden_layers):
            layer_prefix = add_prefix(f"layers.{layer_idx}", prefix)
            layer = MuseSpark1xDecoderLayer(
                config,
                layer_idx,
                _uses_nvfp4_experts(quant_config, layer_prefix),
                layer_prefix,
                quant_config,
            )
            _install_pre_attn_norm(config, layer)
            self.layers.append(layer)
        self.norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            zero_centered_gamma=True,
            output_norm_gain_center_type="zero",
            residual_id=-1,
        )
        # Cross-layer FFN boundaries: layer i's post_feedforward_layernorm + post_feedforward_residual_gate + layer i+1's
        # pre_attn_norm (the final norm for the last layer) as one fused kernel each.
        # MUSE_FUSE_FFN_BOUNDARY=0 restores the per-module eager path.
        self._ffn_boundaries = None
        if (
            os.environ.get("MUSE_FUSE_FFN_BOUNDARY", "1") == "1"
            and len(self.layers) > 0
        ):
            pre_norms = [layer.self_attn.pre_attn_norm for layer in self.layers[1:]] + [
                self.norm
            ]
            self._ffn_boundaries = [
                FusedPostNormResidualPreNorm(
                    layer.post_feedforward_layernorm,
                    layer.post_feedforward_residual_gate,
                    pre_norm,
                )
                for layer, pre_norm in zip(self.layers, pre_norms)
            ]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        if input_embeds is None and self.embed_norm is not None:
            hidden_states = self.embed_norm(hidden_states)
        stream = hidden_states.to(self.residual_dtype)
        # DFLASH draft context: the residual stream after each listed layer, read
        # in operator dtype (bf16) like the CUDA model does.
        aux_hidden_states: list[torch.Tensor] = []
        if self._ffn_boundaries is not None and stream.shape[0] > 0:
            # Long plain prefills: every rank keeps a 1/TP token shard of the stream.
            sp = sp_shape(forward_batch, stream.shape[0])
            if sp is not None:
                stream = sp_shard(stream, sp)
            final_pre_norm_dtype = torch.bfloat16
            attn_input = self.layers[0].self_attn_input_norm(stream)
            attn_input_fp8 = None
            last = len(self.layers) - 1
            for i, layer in enumerate(self.layers):
                stream, attn_input, attn_input_fp8 = layer(
                    positions,
                    stream,
                    forward_batch,
                    attn_input=attn_input,
                    ffn_boundary=self._ffn_boundaries[i],
                    pre_norm_dtype=final_pre_norm_dtype if i == last else None,
                    sp=sp,
                    attn_input_fp8=attn_input_fp8,
                )
                if i in self.layers_to_capture:
                    aux_hidden_states.append(stream.to(torch.bfloat16))
            # == self.norm(norm_input).to(bf16) from the last boundary
            hidden_states = attn_input
            if sp is not None:
                hidden_states = sp_gather(hidden_states, sp.num_tokens)
                aux_hidden_states = [
                    sp_gather(a, sp.num_tokens) for a in aux_hidden_states
                ]
        else:
            for i, layer in enumerate(self.layers):
                stream = layer(positions, stream, forward_batch)
                if i in self.layers_to_capture:
                    aux_hidden_states.append(stream.to(torch.bfloat16))
            norm_input = stream.to(torch.bfloat16)
            hidden_states = self.norm(norm_input).to(torch.bfloat16)
        if self.layers_to_capture:
            return hidden_states, aux_hidden_states
        return hidden_states


class MuseSparkModel(nn.Module):
    """Holds the decoder under the released ``model.language_model.*`` parameter names.

    The release introduced this level so the text stack sits beside the vision and audio
    towers. Attribute reads fall through to the decoder, so the rest of this file keeps
    addressing ``self.model.layers`` and ``self.model.embed_tokens`` as before.
    """

    def __init__(
        self,
        config: MuseSparkTextConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ):
        super().__init__()
        self.language_model = MuseSpark1xModel(
            config, quant_config, add_prefix("language_model", prefix)
        )

    def __getattr__(self, name):
        # nn.Module resolves real submodules/parameters first; anything else is the
        # decoder's, which keeps `self.model.layers` and friends working.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.language_model, name)

    def forward(self, *args, **kwargs):
        return self.language_model(*args, **kwargs)


@torch.no_grad()
def _prepare_prefill_fp8_projections(moe: MuseSpark1xMoE) -> None:
    """FP8 copies + channel scales of the pooler / unpooler for the prefill GEMMs; the pooler's is the joint
    router / pooler's copy when that exists (same values, no second copy)."""
    if moe.pre_expert_proj is not None:
        moe.pre_expert_proj.small_m_fp8_weight = (
            (moe.router_pooler_fp8_weight, moe.router_pooler_fp8_scale)
            if router_pooler.JOINT_POOLER_FP8
            and moe.router_pooler_fp8_weight is not None
            else quantize_fp8w_weight(moe.pre_expert_proj.weight.data)
        )
    if moe.post_expert_proj is not None:
        moe.post_expert_proj.small_m_fp8_weight = quantize_fp8w_weight(
            moe.post_expert_proj.weight.data
        )


def _a4w4_keeps_native(experts: list) -> bool:
    """Whether every MoE layer can hold the a4w4 copy beside the stored weights above the reserve on every TP rank;
    otherwise each layer keeps only the preshuffled copy (TP4: about 103 GB per rank for the second copy)."""
    free, _ = torch.cuda.mem_get_info()
    free += torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    first = experts[0]
    per_layer = sum(
        t.numel() * t.element_size()
        for t in (
            first.w13_weight,
            first.w2_weight,
            first.w13_weight_scale,
            first.w2_weight_scale,
        )
    )
    fits = torch.tensor(
        [int(free - per_layer * len(experts) >= _A4W4_RESERVE_BYTES)], dtype=torch.int64
    )
    torch.distributed.all_reduce(
        fits, op=torch.distributed.ReduceOp.MIN, group=get_tp_group().cpu_group
    )
    keep = bool(fits.item())
    logger.info(
        "Muse MXFP4 experts: %s (%d layers, %.1f GB each, %.1f GB free)",
        "stored + a4w4 copies" if keep else "a4w4 preshuffled copy only",
        len(experts),
        per_layer / 1e9,
        free / 1e9,
    )
    return keep


class MuseSparkTextForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: MuseSparkTextConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        # Unquantized linear shapes tuned on MI355X at TP8, registered before any
        # weights load: tiled small-M GEMMs for the attention O projection and the
        # DFlash draft's MLP (the draft loads after the target, in this process),
        # and the draft MLP's M = 24 GEMMs on the aiter Triton a16w16 kernel.
        # The O projection's FP8 copy serves the prefill GEMMs only.
        fp8_shapes = _FP8W_LINEAR_SHAPES + (
            _ROWMAJOR_LINEAR_SHAPES if _PREFILL_DENSE_FP8 else ()
        )
        register_gfx950_unquantized_gemm_shapes(
            tiled_weights=tuple(TILED_GEMM_LAYOUT),
            rowmajor_weights=_ROWMAJOR_LINEAR_SHAPES,
            fp8_weights=fp8_shapes if _DENSE_FP8 else (),
            triton_a16w16=_DRAFT_TRITON_A16W16_GEMMS,
        )
        if getattr(config, "inference_moe_w13_interleaved", False):
            raise ValueError("Interleaved expert checkpoints are not supported")
        quant_name = _quantization_name(quant_config)
        if quant_name not in (None, "modelopt_fp4", "quark"):
            raise ValueError(
                "Only BF16, serialized ModelOpt NVFP4, and Quark MXFP4 are "
                f"supported, got {quant_name}"
            )
        if quant_name == "quark" and not _is_quark_mxfp4(quant_config):
            raise ValueError(
                "Unsupported Quark format; expected FP4/E8M0/group-size 32"
            )
        self.config = config
        self.quant_config = quant_config
        self.nvfp4_checkpoint = quant_name == "modelopt_fp4"
        self.mxfp4_checkpoint = _is_quark_mxfp4(quant_config)
        # ``--quantization quark_mxfp4`` on a ModelOpt NVFP4 checkpoint: Quark builds
        # an online MXFP4 config with the NVFP4 export as its dequantization source.
        # The generic requantizer only drives FusedMoE, so this model converts its
        # stacked NVFP4 expert tensors itself while loading (see
        # _stash_nvfp4_expert_tensor / _requantize_nvfp4_stack).
        self.requant_nvfp4_experts = self.mxfp4_checkpoint and _is_nvfp4_source(
            quant_config
        )
        self._nvfp4_pending: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
        self.model = MuseSparkModel(config, quant_config, add_prefix("model", prefix))
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=None,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=False,
        )
        if config.tie_word_embeddings:
            self.lm_head.tie_weights(self.model.embed_tokens)
        self.logits_processor = MuseSpark1xLogitsProcessor(
            config, logit_scale=config.output_multiplier
        )
        self.capture_aux_hidden_states = False

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ):
        hidden_states = self.model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        """Capture the residual stream after each listed decoder layer.

        DFLASH drafts consume the target hidden state produced by layer
        ``layer_ids[k]`` (HF ``hidden_states[layer_ids[k] + 1]``). Mirrors the
        CUDA model's hook so the generic DFLASH worker can drive this copy.
        """
        if layer_ids is None:
            raise ValueError(
                "DFLASH requires explicit layer_ids for aux hidden capture."
            )
        num_layers = len(self.model.language_model.layers)
        bad = [i for i in layer_ids if not 0 <= i < num_layers]
        if bad:
            raise ValueError(
                f"DFLASH target layer ids {bad} are out of range for a "
                f"{num_layers}-layer Muse Spark target."
            )
        self.capture_aux_hidden_states = True
        # MuseSparkModel.__getattr__ only reads through to the decoder; write the
        # list on the decoder itself.
        self.model.language_model.layers_to_capture = list(layer_ids)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_layers(self):
        return self.model.layers

    def get_attention_sliding_window_size(self) -> int | None:
        if "sliding_attention" in self.config.layer_types:
            return self.config.sliding_window - 1
        return None

    # The release stacks the experts into one tensor per projection and renames the
    # router; shapes are unchanged, so these are pure name moves onto the same params.
    # Checkpoint spellings this model maps onto its own module names.
    _WEIGHT_REMAPS = (
        (".mlp.gate.e_score_correction_bias", ".mlp.expert_bias"),
        (".mlp.gate.weight", ".mlp.router.weight"),
        # this model folds the pre-attention norm into the attention module
        (".input_layernorm.", ".self_attn.pre_attn_norm."),
        (".mlp.experts.post_expert_norm", ".mlp.post_expert_norm"),
        (".mlp.experts.gate_up_proj", ".mlp.experts.w13_weight"),
        (".mlp.experts.down_proj", ".mlp.experts.w2_weight"),
    )

    @staticmethod
    def _deinterleave_rope_halves(loaded_weight, heads: int):
        """Undo the released q/k rope-pair interleave.

        The checkpoint stores each head's rope pairs interleaved, the layout stock HF
        rope expects; this model's rotary embedding wants the two halves contiguous.
        """
        shape = loaded_weight.shape
        return (
            loaded_weight.reshape(heads, 2, shape[0] // heads // 2, *shape[1:])
            .transpose(1, 2)
            .reshape(shape)
            .contiguous()
        )

    _NVFP4_STACKED_EXPERT_RE = re.compile(
        r"^(.*\.mlp\.experts)\.(gate_up_proj|down_proj)"
        r"(_weight_scale_2|_weight_scale|_input_scale)?$"
    )

    def _stash_nvfp4_expert_tensor(
        self,
        params: dict[str, torch.Tensor],
        name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        """Collect a layer's stacked NVFP4 expert tensors and requantize them to
        MXFP4 once weight, block scale and global scale have all arrived.

        ModelOpt exports a layer's routed experts as ``gate_up_proj`` (uint8 packed
        e2m1 [E, 2I, H/2]), ``gate_up_proj_weight_scale`` (fp8 e4m3 [E, 2I, H/16]),
        ``gate_up_proj_weight_scale_2`` (fp32 [E, 2] = gate/up global scales) and
        ``gate_up_proj_input_scale`` (unused here), likewise for ``down_proj`` with
        per-expert scalars. Excluded layers ship BF16 experts without scales; those
        keep the regular path. Returns True when the tensor was consumed.
        """
        match = self._NVFP4_STACKED_EXPERT_RE.match(name)
        if match is None:
            return False
        prefix, proj, suffix = match.groups()
        if suffix is None and loaded_weight.dtype != torch.uint8:
            return False  # BF16 experts of an excluded layer
        if suffix == "_input_scale":
            return True
        if loaded_weight.shape[0] == self.config.num_local_experts:
            parallel = get_parallel()
            local = loaded_weight.shape[0] // parallel.moe_ep_size
            loaded_weight = loaded_weight.narrow(0, parallel.moe_ep_rank * local, local)
        key = (prefix, proj)
        pieces = self._nvfp4_pending.setdefault(key, {})
        pieces["weight" if suffix is None else suffix.lstrip("_")] = (
            loaded_weight.contiguous()
        )
        if len(pieces) == 3:
            self._requantize_nvfp4_stack(params, prefix, proj, pieces)
            del self._nvfp4_pending[key]
        return True

    @torch.no_grad()
    def _requantize_nvfp4_stack(
        self,
        params: dict[str, torch.Tensor],
        prefix: str,
        proj: str,
        pieces: dict[str, torch.Tensor],
    ) -> None:
        """NVFP4 (e2m1 x fp8 block-16 scale x fp32 global scale) -> MXFP4 (e2m1 x
        e8m0 block-32 scale), one expert at a time on the GPU, written straight into
        the Quark-layout parameters the MXFP4 kernels read."""
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4

        tag = "w13" if proj == "gate_up_proj" else "w2"
        weight = params[f"{prefix}.{tag}_weight"]
        scale = params[f"{prefix}.{tag}_weight_scale"]
        device = weight.device
        w_q, w_s, s2 = (
            pieces["weight"],
            pieces["weight_scale"],
            pieces["weight_scale_2"],
        )
        if w_q.shape[0] != weight.shape[0]:
            raise RuntimeError(
                f"Expert count mismatch for {prefix}.{proj}: "
                f"parameter={weight.shape[0]}, checkpoint={w_q.shape[0]}"
            )
        for expert in range(w_q.shape[0]):
            if tag == "w13":
                # [gate rows; up rows] share one global scale per half.
                half = w_q.shape[1] // 2
                s2_rows = torch.cat(
                    [s2[expert, 0].repeat(half), s2[expert, 1].repeat(half)]
                ).view(-1, 1)
            else:
                s2_rows = s2[expert].reshape(())
            dequant = dequantize_nvfp4(
                w_q[expert].to(device, non_blocking=True),
                w_s[expert].to(device, non_blocking=True),
                s2_rows.to(device=device, dtype=torch.float32),
                out_dtype=torch.float32,
            )
            q_weight, q_scale = dynamic_mxfp4_quant(dequant)
            if q_weight.shape != weight.shape[1:] or q_scale.shape != scale.shape[1:]:
                raise RuntimeError(
                    f"Requantized {prefix}.{proj} shape mismatch: "
                    f"{tuple(q_weight.shape)}/{tuple(q_scale.shape)} vs "
                    f"{tuple(weight.shape[1:])}/{tuple(scale.shape[1:])}"
                )
            weight.data[expert].copy_(q_weight)
            scale.data[expert].copy_(q_scale.to(scale.dtype))

    def _copy_expert_stack(
        self, param: torch.Tensor, loaded_weight: torch.Tensor, name: str
    ) -> None:
        if param.shape == loaded_weight.shape:
            param.data.copy_(loaded_weight)
            return
        parallel = get_parallel()
        if loaded_weight.shape[0] == self.config.num_local_experts:
            local = loaded_weight.shape[0] // parallel.moe_ep_size
            start = parallel.moe_ep_rank * local
            loaded_weight = loaded_weight.narrow(0, start, local)
        if param.shape != loaded_weight.shape:
            raise RuntimeError(
                f"Expert weight shape mismatch for {name}: "
                f"parameter={tuple(param.shape)}, checkpoint={tuple(loaded_weight.shape)}"
            )
        param.data.copy_(loaded_weight)

    def _load_single_expert(
        self,
        params: dict[str, torch.Tensor],
        name: str,
        loaded_weight: torch.Tensor,
    ) -> bool:
        match = re.match(
            r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\."
            r"(weight|weight_scale|weight_scale_2|input_scale)$",
            name,
        )
        if match is None:
            return False
        prefix, expert_raw, projection, suffix = match.groups()
        if suffix == "input_scale":
            return True
        expert_id = int(expert_raw)
        parallel = get_parallel()
        local_count = self.config.num_local_experts // parallel.moe_ep_size
        start = parallel.moe_ep_rank * local_count
        if not start <= expert_id < start + local_count:
            return True
        local_id = expert_id - start
        is_w13 = projection in ("gate_proj", "up_proj")
        param_name = f"{prefix}.{'w13' if is_w13 else 'w2'}_{suffix}"
        param = params.get(param_name)
        if param is None:
            raise ValueError(
                f"Checkpoint weight '{name}' has no parameter '{param_name}'"
            )
        if suffix == "weight_scale_2":
            if is_w13 and param.ndim == 2:
                shard = 0 if projection == "gate_proj" else 1
                param.data[local_id, shard].copy_(loaded_weight.reshape(()))
            else:
                param.data[local_id].copy_(loaded_weight.reshape(()))
            return True
        target = param.data[local_id]
        if is_w13:
            half = target.shape[0] // 2
            offset = 0 if projection == "gate_proj" else half
            target = target.narrow(0, offset, half)
        if target.shape != loaded_weight.shape:
            raise RuntimeError(
                f"Expert weight shape mismatch for {name}: "
                f"parameter={tuple(target.shape)}, checkpoint={tuple(loaded_weight.shape)}"
            )
        target.copy_(loaded_weight)
        return True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        params = dict(self.named_parameters())
        params.update(
            {name: value for name, value in self.named_buffers() if name not in params}
        )
        stacked = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        # The o-gate ships in every Muse Spark 1.2 checkpoint and the release spells
        # it self_attn.gate_proj. The dense .gate_up_proj entries above are tried first,
        # so an MLP gate_proj never reaches this mapping.
        stacked.append((".qkv_proj", ".gate_proj", "g"))

        skipped = 0
        for raw_name, loaded_weight in weights:
            name = raw_name
            if self.requant_nvfp4_experts and self._stash_nvfp4_expert_tensor(
                params, name, loaded_weight
            ):
                continue
            if name.endswith((".self_attn.q_proj.weight", ".self_attn.k_proj.weight")):
                loaded_weight = self._deinterleave_rope_halves(
                    loaded_weight,
                    self.config.num_attention_heads
                    if name.endswith(".q_proj.weight")
                    else self.config.num_key_value_heads,
                )
            for old, new in self._WEIGHT_REMAPS:
                if old in name:
                    name = name.replace(old, new)
                    break
            if should_skip_weight_loading(name, self.config):
                skipped += 1
                continue
            if name.endswith(
                (".self_attn.q_scale", ".self_attn.k_scale", ".self_attn.v_scale")
            ):
                skipped += 1
                continue
            # Expert activation scales are only meaningful when the experts stay
            # packed: the FP4 quant method turns them into w13/w2_input_scale_quant
            # and g1_scale_c. A BF16 expert loop has no use for them.
            if name.endswith(
                (
                    ".w13_input_scale",
                    ".w2_input_scale",
                    ".gate_up_proj_input_scale",
                    ".down_proj_input_scale",
                )
            ):
                if not self.nvfp4_checkpoint:
                    skipped += 1
                    continue
            if self._load_single_expert(params, name, loaded_weight):
                continue

            handled = False
            original_name = name
            for param_pattern, weight_pattern, shard_id in stacked:
                if weight_pattern not in name:
                    continue
                candidate = name.replace(weight_pattern, param_pattern)
                param = params.get(candidate)
                if param is None:
                    continue
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight, shard_id)
                handled = True
                break
            if handled:
                continue
            name = original_name
            param = params.get(name)
            if param is None:
                raise ValueError(
                    f"Checkpoint weight '{raw_name}' is outside the Muse Spark text model"
                )
            if "router.weight" in name and param.shape != loaded_weight.shape:
                loaded_weight = loaded_weight.t().contiguous()
            if ".experts.w" in name:
                self._copy_expert_stack(param, loaded_weight, name)
            else:
                loader = getattr(param, "weight_loader", default_weight_loader)
                loader(param, loaded_weight)
        if skipped:
            logger.info("Skipped %d non-text or unused checkpoint tensors", skipped)
        if self._nvfp4_pending:
            raise RuntimeError(
                "Incomplete NVFP4 expert tensors for "
                + ", ".join(f"{p}.{proj}" for p, proj in sorted(self._nvfp4_pending))
            )
        self.post_load_weights()

    @torch.no_grad()
    def post_load_weights(self) -> None:
        moe_layers = [
            layer.mlp
            for layer in self.model.layers
            if isinstance(layer.mlp, MuseSpark1xMoE)
        ]
        if router_pooler.JOINT_ROUTER_POOLER:
            for moe in moe_layers:
                router_pooler.prepare_joint_router_pooler(moe)
        if mxfp4_moe.A4W4_ENABLED:
            experts = [
                module
                for module in self.model.modules()
                if isinstance(module, MuseSpark1xMxfp4Experts)
            ]
            keep_native = bool(experts) and _a4w4_keeps_native(experts)
            for module in experts:
                mxfp4_moe.prepare_a4w4_weights(module, keep_native=keep_native)
        if _DENSE_TILED:
            for moe in moe_layers:
                for proj in (moe.pre_expert_proj, moe.post_expert_proj):
                    if proj is not None:
                        proj.small_m_tiled_weight = prepare_tiled_weight(
                            proj.weight.data
                        )
        if ulysses_enabled():
            for layer in self.model.layers:
                prepare_ulysses(layer.self_attn, fp8=_PREFILL_DENSE_FP8)
        if _PREFILL_DENSE_FP8:
            for moe in moe_layers:
                _prepare_prefill_fp8_projections(moe)
        for layer in self.model.layers:
            for residual in (
                layer.post_attention_residual_gate,
                layer.post_feedforward_residual_gate,
            ):
                if isinstance(residual, MuseSpark1xNormPreservingResidualConnection):
                    residual.precompute_coefficients()
        for module in self.model.modules():
            if isinstance(module, MuseSpark1xRMSNorm):
                module.cache_effective_weight()
        boundaries = [layer._fused_attn_boundary for layer in self.model.layers]
        boundaries += list(getattr(self.model, "_ffn_boundaries", None) or [])
        for op in boundaries:
            ensure = getattr(op, "_ensure_cached_weights", None)
            if ensure is not None:
                ensure(torch.bfloat16)


__all__ = [
    "MuseSparkTextForCausalLM",
    "MuseSparkTextConfig",
]
