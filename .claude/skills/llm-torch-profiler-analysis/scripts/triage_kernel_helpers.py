"""Internal kernel attribution helpers for triage-only torch-profiler analysis."""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

from profile_common import (
    coerce_optional_int,
    contains_any_keyword,
    extract_trace_events,
    has_stream_marker,
    is_annotation_event,
    is_complete_duration_event,
    is_non_kernel_trace_category,
    is_trace_metadata_name,
    looks_like_python_scope_name,
    normalize_repo_relative_path,
    normalize_text,
    select_heaviest_pid,
)

CATEGORY_PATTERNS: List[Tuple[str, Tuple[str, ...]]] = [
    (
        "hybrid_linear",
        (
            "gdn",
            "gated_delta",
            "mamba",
            "selective_scan",
            "ssd",
            "causal_conv",
            "ssm",
        ),
    ),
    (
        "attention",
        (
            "flash_attn",
            "flashattention",
            "flash_attention",
            "fmha",
            "attention",
            "mla",
            "paged_attention",
            "decode_attention",
        ),
    ),
    (
        "moe",
        (
            "fused_moe",
            "grouped_mm",
            "groupgemm",
            "group_gemm",
            "moe",
            "expert",
            "groupproblemshape",
        ),
    ),
    (
        "gemm",
        (
            "gemm",
            "gemv",
            "matmul",
            "cublas",
            "cutlass",
            "wgmma",
            "mma",
            "bmm",
            "nvjet",
        ),
    ),
    (
        "norm",
        (
            "rmsnorm",
            "layernorm",
            "_norm_",
            " norm",
            "normkernel",
        ),
    ),
    ("rope", ("rotary", "rope", "mrope")),
    ("softmax", ("softmax",)),
    ("activation", ("silu", "gelu", "relu", "act_and_mul", "sigmoid")),
    ("quantize", ("quant", "fp8", "mxfp", "nvfp4", "dequant", "cvt")),
    (
        "reduce_topk",
        ("topk", "reduce", "argmax", "argtopk", "sampling", "multinomial"),
    ),
    (
        "sampling_io",
        (
            "prepare_inputs",
            "write_req_to",
            "catarraybatched",
            "prepare_next",
            "copy_next",
        ),
    ),
    (
        "elementwise",
        (
            "elementwise",
            "vectorized_elementwise_kernel",
            "unrolled_elementwise_kernel",
            "gpu_kernel_impl",
            "binary_internal",
            "unaryfunctor",
            "add_kernel",
            "sub_kernel",
            "mul_kernel",
            "div_",
            "floor_kernel",
            "log_kernel",
            "neg_kernel",
        ),
    ),
]

COMMUNICATION_STRONG_KEYWORDS = (
    "nccl",
    "allreduce",
    "all_reduce",
    "reduce_scatter",
    "allgather",
    "all_gather",
    "alltoall",
    "all_to_all",
    "cross_device_reduce",
    "deepep",
    "mooncake",
)

COMMUNICATION_WEAK_KEYWORDS = (
    "broadcast",
    "dispatch",
    "combine",
)

MEMORY_STRONG_KEYWORDS = (
    "memcpy",
    "memset",
    "dma",
    "prefetch",
)

MEMORY_WEAK_KEYWORDS = (
    "copy",
    "fill",
)

COMPUTE_HINT_KEYWORDS = (
    "gemm",
    "gemv",
    "matmul",
    "cublas",
    "cutlass",
    "wgmma",
    "mma",
    "bmm",
    "nvjet",
    "fmha",
    "attention",
    "flash_attn",
    "flashattention",
    "flash_attention",
    "grouped_mm",
    "groupgemm",
    "moe",
    "expert",
)

NOISE_FRAME_PREFIXES = (
    "threading.py(",
    "multiprocessing/",
    "contextlib.py(",
    "torch/utils/_contextlib.py(",
    "runpy.py(",
    "asyncio/",
    "selectors.py(",
    "queue.py(",
    "socket.py(",
    "tqdm/_monitor.py(",
    "<string>(",
    "<built-in method ",
)

LOW_LEVEL_FRAME_PREFIXES = (
    "triton/runtime/",
    "triton/backends/",
    "torch/_ops.py",
    "torch/nn/modules/module.py",
)

LOW_SIGNAL_FUNCTION_TOKENS = (
    "__torch_function__",
    "__torch_dispatch__",
    "__call__",
    "_call_impl",
    "_wrapped_call_impl",
)

LOW_SIGNAL_PATH_TOKENS = (
    "model_executor/parameter.py:",
    "model_executor/cuda_graph_runner.py:",
    "compilation/cuda_graph.py:",
    "pyexecutor/cuda_graph_runner.py:",
    "pyexecutor/py_executor.py:",
    "_torch/utils.py:",
    "torch/fx/graph_module.py:",
)


@dataclass
class KernelEvent:
    name: str
    canonical_name: str
    category: str
    stage: str
    pid: str
    tid: str
    ts: float
    dur: float
    external_id: Optional[int]
    correlation: Optional[int] = None


@dataclass
class CpuOpEvent:
    name: str
    pid: str
    tid: str
    ts: float
    dur: float
    external_id: int


@dataclass
class LaunchEvent:
    name: str
    pid: str
    tid: str
    ts: float
    dur: float
    correlation: int


@dataclass
class PythonFrame:
    name: str
    normalized_name: str
    pid: str
    tid: str
    ts: float
    dur: float
    python_id: Optional[int]
    parent_id: Optional[int]
    end_ts: float
    priority: int


@dataclass
class TimedEventIndex:
    events: List[object]
    start_ts: List[float]


@dataclass
class FrameResolution:
    location: str
    stack: str


@dataclass(frozen=True)
class StageAnnotation:
    stage: str
    ts: float
    end_ts: float
    external_id: Optional[int]
    is_gpu: bool


@dataclass(frozen=True)
class StageWindow:
    stage: str
    ts: float
    end_ts: float


@dataclass
class Aggregate:
    total_us: float = 0.0
    count: int = 0
    max_us: float = 0.0

    @property
    def avg_us(self) -> float:
        return self.total_us / self.count if self.count else 0.0


@dataclass
class MappingSiteAggregate:
    total_us: float = 0.0
    count: int = 0
    cpu_ops: Counter = field(default_factory=Counter)
    stacks: Counter = field(default_factory=Counter)


@dataclass
class KernelRow:
    name: str
    category: str
    aggregate: Aggregate
    location: str
    cpu_op: str
    entry: Optional[dict]

    @property
    def total_us(self) -> float:
        return self.aggregate.total_us


@dataclass
class FusionOpportunity:
    pattern: str
    status: str
    confidence: str
    related_us: float
    evidence: str
    current_locations: str
    candidate_path: str
    rationale: str
    covered_row_keys: Tuple[Tuple[str, str, str], ...] = field(
        default_factory=tuple, repr=False
    )
    pattern_span: int = field(default=1, repr=False)
    has_active_match: bool = field(default=False, repr=False)
    priority: int = field(default=0, repr=False)
    subsumes: Tuple[str, ...] = field(default_factory=tuple, repr=False)


@dataclass(frozen=True)
class FusionPatternSpec:
    pattern: str
    candidate_path: str
    active_keywords: Tuple[str, ...] = ()
    split_groups: Tuple[Tuple[str, ...], ...] = ()
    rationale_hint: str = ""
    origin: str = "mainline"
    model_include: Tuple[str, ...] = ()
    model_exclude: Tuple[str, ...] = ()
    min_tp_size: int = 1
    require_tp: bool = False
    min_share: float = 0.25
    likely_share: float = 3.0
    priority: int = 0
    subsumes: Tuple[str, ...] = ()


FUSION_PATTERN_REGISTRY: Tuple[FusionPatternSpec, ...] = (
    FusionPatternSpec(
        pattern="Fused residual add + RMSNorm",
        candidate_path=(
            "python/sglang/srt/layers/layernorm.py"
            "<br>python/sglang/srt/layers/quantization/modelslim/modelslim.py"
        ),
        active_keywords=(
            "fused_add_rmsnorm",
            "gemma_fused_add_rmsnorm",
            "npu_add_rms_norm",
            "add_rmsnorm_bias",
        ),
        rationale_hint=(
            "Residual add plus RMSNorm already has fused implementations across"
            " several backends."
        ),
        min_share=0.1,
        likely_share=1.0,
    ),
    FusionPatternSpec(
        pattern="FlashInfer unified allreduce_fusion",
        candidate_path=(
            "python/sglang/srt/layers/flashinfer_comm_fusion.py"
            "<br>python/sglang/srt/layers/layernorm.py"
            "<br>python/sglang/srt/layers/communicator.py"
        ),
        active_keywords=(
            "allreduce_fusion",
            "fusedaddrmsnormkernel",
            "flashinfer_comm_fusion.py",
        ),
        split_groups=(
            (
                "cross_device_reduce",
                "allreduce",
                "all_reduce",
                "custom_all_reduce_ops.py",
            ),
            ("rmsnorm", "layernorm", "fused_add_rmsnorm", "layernorm.py"),
        ),
        rationale_hint=(
            "FlashInfer has a TP all-reduce plus residual/RMSNorm fusion path."
        ),
        require_tp=True,
        min_tp_size=2,
        min_share=0.5,
        likely_share=4.0,
    ),
    FusionPatternSpec(
        pattern="AITER allreduce fusion",
        candidate_path=(
            "python/sglang/srt/distributed/communication_op.py"
            "<br>python/sglang/srt/layers/communicator.py"
            "<br>python/sglang/srt/layers/layernorm.py"
        ),
        active_keywords=(
            "tensor_model_parallel_fused_allreduce_rmsnorm",
            "apply_aiter_all_reduce_fusion",
            "custom_fused_ar_rms",
        ),
        split_groups=(
            ("allreduce", "all_reduce", "cross_device_reduce"),
            ("rmsnorm", "layernorm"),
        ),
        rationale_hint=(
            "ROCm already has an AITER fused all-reduce plus RMSNorm family."
        ),
        require_tp=True,
        min_tp_size=2,
        min_share=0.5,
        likely_share=4.0,
    ),
    FusionPatternSpec(
        pattern="Fused activation-and-mul (SwiGLU / GeGLU)",
        candidate_path="python/sglang/srt/layers/activation.py",
        active_keywords=("silu_and_mul", "gelu_and_mul", "npu_swiglu"),
        rationale_hint=(
            "Packed MLP activation and multiply already has dedicated fused ops."
        ),
        min_share=0.1,
        likely_share=1.0,
    ),
    FusionPatternSpec(
        pattern="In-place QK RMSNorm",
        candidate_path=(
            "python/sglang/srt/models/utils.py" "<br>python/sglang/jit_kernel/norm.py"
        ),
        active_keywords=("fused_inplace_qknorm", "minimaxm2rmsnormtp"),
        split_groups=(("apply_qk_norm", "q_norm", "k_norm", "qknorm"),),
        rationale_hint=(
            "Q/K normalization already has in-place or model-specific fused"
            " implementations."
        ),
        min_share=0.3,
        likely_share=2.0,
    ),
    FusionPatternSpec(
        pattern="Fused QK RMSNorm + RoPE",
        candidate_path=(
            "python/sglang/jit_kernel/fused_qknorm_rope.py"
            "<br>python/sglang/srt/models/qwen3_moe.py"
        ),
        active_keywords=("fused_qknorm_rope", "fused_qk_norm_rope"),
        split_groups=(
            ("apply_qk_norm", "q_norm", "k_norm", "qknorm"),
            ("apply_rope", "rotary", "rope", "mrope"),
        ),
        rationale_hint=("SGLang has a fused QK-norm plus RoPE kernel family."),
        min_share=0.3,
        likely_share=2.0,
        priority=30,
    ),
    FusionPatternSpec(
        pattern="Fused QK RoPE reshape + KV cache write",
        candidate_path="python/sglang/srt/layers/attention/utils.py",
        active_keywords=("fused_qk_rope_reshape_and_cache",),
        split_groups=(
            ("rotary", "rope", "mrope"),
            ("reshape", "set_kv", "kv_cache", "cache write", "paged kv"),
        ),
        rationale_hint=(
            "Attention prep already has a fused RoPE plus reshape plus cache"
            " write path."
        ),
        min_share=0.4,
        likely_share=2.0,
        priority=40,
        subsumes=("Fused RoPE + KV cache store",),
    ),
    FusionPatternSpec(
        pattern="Fused RoPE + KV cache store",
        candidate_path=(
            "python/sglang/jit_kernel/rope.py" "<br>python/sglang/srt/models/utils.py"
        ),
        active_keywords=("fused_set_kv_buffer",),
        split_groups=(
            ("rotary", "rope", "mrope"),
            ("set_kv_buffer", "kv cache write", "paged kv", "cache write"),
        ),
        rationale_hint=(
            "RoPE application and KV cache storage already have fused fast"
            " paths in several models."
        ),
        min_share=0.3,
        likely_share=1.5,
        priority=20,
    ),
    FusionPatternSpec(
        pattern="Fused decode metadata setup",
        candidate_path=("python/sglang/srt/layers/attention/flashattention_backend.py"),
        active_keywords=(
            "normal_decode_set_metadata",
            "cache_seqlens_int32",
            "cu_seqlens_k",
            "swa_page_table",
        ),
        rationale_hint=(
            "Decode metadata setup already has a fused Triton preparation path."
        ),
        min_share=0.05,
        likely_share=0.5,
    ),
    FusionPatternSpec(
        pattern="NSA fused metadata copy for graph replay",
        candidate_path="python/sglang/jit_kernel/fused_metadata_copy.py",
        active_keywords=(
            "fused_metadata_copy",
            "fused_metadata_copy_multi",
            "fused_nsa_cache_seqlens",
            "fused_flashmla_metadata",
        ),
        rationale_hint=(
            "NSA replay metadata copies are already fused into one-kernel" " families."
        ),
        min_share=0.02,
        likely_share=0.2,
    ),
    FusionPatternSpec(
        pattern="DeepSeek MLA fused projection + norm + RoPE",
        candidate_path=(
            "python/sglang/srt/models/deepseek_common/attention_forward_methods/"
            "forward_mla_fused_rope_cpu.py"
            "<br>python/sglang/srt/models/deepseek_common/attention_forward_methods/"
            "forward_mla_fused_rope_rocm.py"
        ),
        active_keywords=(
            "qkv_proj_with_rope_fused_weight",
            "fused_qkv_a_proj_with_mqa",
            "forward_absorb_fused_mla_rope",
        ),
        split_groups=(
            ("mla", "qkv_a_proj", "q_a_proj"),
            ("qknorm", "rmsnorm", "apply_qk_norm"),
            ("rope", "rotary"),
        ),
        rationale_hint=(
            "DeepSeek MLA has backend-specific fused projection, norm, and"
            " RoPE prep paths."
        ),
        model_include=("deepseek", "glm"),
        min_share=0.4,
        likely_share=2.0,
        priority=80,
        subsumes=("Fused QK RMSNorm + RoPE",),
    ),
    FusionPatternSpec(
        pattern="Fused QK RoPE concat + MLA cache write",
        candidate_path=(
            "python/sglang/srt/layers/rocm_linear_utils.py"
            "<br>python/sglang/srt/models/deepseek_common/attention_forward_methods/"
            "forward_mla.py"
        ),
        active_keywords=("fused_qk_rope_cat_and_cache_mla", "set_mla_kv_buffer"),
        split_groups=(
            ("mla", "rope", "rotary"),
            ("cache", "kv_buffer", "concat"),
        ),
        rationale_hint=(
            "MLA RoPE packing and cache write already have fused backend paths."
        ),
        model_include=("deepseek", "glm"),
        min_share=0.3,
        likely_share=1.5,
        priority=85,
        subsumes=("Fused RoPE + KV cache store",),
    ),
    FusionPatternSpec(
        pattern="Qwen3 decode fused QK norm + 3D mRoPE + KV cache write",
        candidate_path="python/sglang/srt/models/qwen3.py",
        active_keywords=("fused_qk_norm_mrope_3d_cache_pts_quant_shuffle",),
        split_groups=(
            ("apply_qk_norm", "q_norm", "k_norm", "qknorm"),
            ("mrope", "3d rope", "rotary"),
            ("cache", "kv_buffer", "paged kv", "cache write"),
        ),
        rationale_hint=(
            "Qwen3-style decode already has a fused QK-norm plus 3D mRoPE plus"
            " cache-write path."
        ),
        model_include=("qwen3",),
        model_exclude=("qwen3.5", "qwen3_5"),
        min_share=0.4,
        likely_share=2.0,
        priority=90,
        subsumes=(
            "Fused QK RMSNorm + RoPE",
            "Fused QK RoPE reshape + KV cache write",
            "Fused RoPE + KV cache store",
        ),
    ),
    FusionPatternSpec(
        pattern="Fused MoE router / top-k / softcapping",
        candidate_path="python/sglang/srt/layers/moe/router.py",
        active_keywords=("fusedmoerouter", "fused_moe_router"),
        split_groups=(
            ("router", "gate", "router logits"),
            ("topk", "softmax", "softcap", "tanh"),
        ),
        rationale_hint=(
            "MoE routing already has fused router, softcap, and top-k kernels."
        ),
        min_share=0.3,
        likely_share=1.5,
        priority=30,
    ),
    FusionPatternSpec(
        pattern="Fused MoE grouped-topk / gate kernels",
        candidate_path="python/sglang/srt/layers/moe/topk.py",
        active_keywords=(
            "fused_topk_deepseek",
            "moe_fused_gate",
            "aiter_fused_topk",
            "kimi_k2_moe_fused_gate",
        ),
        split_groups=(
            ("grouped_topk", "topk", "biased_grouped_topk"),
            ("gate", "router", "renorm", "routed scaling"),
        ),
        rationale_hint=(
            "Grouped-topk, bias handling, and routed scaling already have fused"
            " gate kernels."
        ),
        min_share=0.3,
        likely_share=1.5,
        priority=50,
        subsumes=("Fused MoE router / top-k / softcapping",),
    ),
    FusionPatternSpec(
        pattern="Qwen-style shared-expert append into routed top-k output",
        candidate_path=(
            "python/sglang/srt/models/qwen2_moe.py"
            "<br>python/sglang/srt/layers/moe/moe_runner/triton_utils/"
            "fused_moe_triton_kernels.py"
        ),
        active_keywords=(
            "_append_shared_to_topk_output",
            "fused_append_shared_experts_with_weights",
            "_fused_append_shared_experts_with_weights_kernel",
        ),
        split_groups=(
            ("_append_shared_to_topk_output", "topk", "grouped_topk"),
            ("shared_expert", "shared_expert_gate", "sigmoid"),
        ),
        rationale_hint=(
            "Qwen-style shared experts can already be appended into routed top-k"
            " output in one Triton prep kernel before fused MoE execution."
        ),
        min_share=0.05,
        likely_share=0.5,
        priority=55,
    ),
    FusionPatternSpec(
        pattern="Fused MoE sum + all-reduce",
        candidate_path=("python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py"),
        active_keywords=("fuse_sum_all_reduce", "enable_fused_moe_sum_all_reduce"),
        split_groups=(
            ("fused_moe", "expert", "moe"),
            ("allreduce", "all_reduce", "cross_device_reduce"),
        ),
        rationale_hint=(
            "The second MoE GEMM already has a fused sum-plus-all-reduce path."
        ),
        require_tp=True,
        min_tp_size=2,
        min_share=0.4,
        likely_share=2.0,
    ),
    FusionPatternSpec(
        pattern="Fused MoE activation + quant / re-quant",
        candidate_path=(
            "python/sglang/srt/layers/moe/ep_moe/kernels.py"
            "<br>python/sglang/jit_kernel/nvfp4.py"
            "<br>python/sglang/srt/layers/moe/cutlass_w4a8_moe.py"
        ),
        active_keywords=(
            "silu_and_mul_scaled_fp4",
            "npu_dequant_swiglu_quant",
            "swiglu_quant",
        ),
        split_groups=(
            ("silu", "gelu", "act_and_mul"),
            ("quant", "fp8", "mxfp", "nvfp4", "dequant"),
        ),
        rationale_hint=(
            "Quantized MoE backends already fuse activation with re-quantization."
        ),
        min_share=0.3,
        likely_share=1.5,
    ),
    FusionPatternSpec(
        pattern="DeepSeek comm-prep fused RMSNorm + quant / flatten-quant",
        candidate_path=(
            "python/sglang/srt/layers/communicator.py"
            "<br>python/sglang/srt/models/deepseek_common/attention_forward_methods/"
            "forward_mla.py"
            "<br>python/sglang/srt/models/deepseek_common/attention_forward_methods/"
            "forward_mha.py"
        ),
        active_keywords=(
            "fused_rms_fp8_group_quant",
            "fused_rms_mxfp4_quant",
            "fused_flatten_fp8_group_quant",
            "fused_flatten_mxfp4_quant",
        ),
        split_groups=(
            ("rmsnorm", "layernorm", "flatten"),
            ("fp8", "mxfp4", "quant"),
        ),
        rationale_hint=(
            "DeepSeek comm preparation already fuses norm or flatten work with"
            " quantization."
        ),
        model_include=("deepseek", "glm"),
        min_share=0.3,
        likely_share=1.5,
    ),
    FusionPatternSpec(
        pattern="NSA fused top-k transform / page-table build",
        candidate_path="python/sglang/srt/layers/attention/nsa_backend.py",
        active_keywords=(
            "fast_topk_transform_fused",
            "fast_topk_transform_ragged_fused",
        ),
        rationale_hint=(
            "NSA top-k metadata preparation already has fused transform kernels."
        ),
        min_share=0.05,
        likely_share=0.3,
    ),
    FusionPatternSpec(
        pattern="NSA fused quantize + indexed K-cache store",
        candidate_path=(
            "python/sglang/jit_kernel/fused_store_index_cache.py"
            "<br>python/sglang/srt/layers/attention/nsa/nsa_indexer.py"
        ),
        active_keywords=("fused_store_index_k_cache",),
        split_groups=(
            ("act_quant", "quant", "scale_buffer"),
            ("index_k", "cache", "store"),
        ),
        rationale_hint=(
            "NSA already has a fused quantize-and-indexed-store kernel family."
        ),
        min_share=0.2,
        likely_share=1.0,
    ),
    FusionPatternSpec(
        pattern="Fused sampling temperature + softmax",
        candidate_path=(
            "python/sglang/srt/layers/fused_sampling.py"
            "<br>python/sglang/srt/layers/sampler.py"
        ),
        active_keywords=("fused_temperature_softmax",),
        split_groups=(
            ("temperature", "temp_scale"),
            ("softmax", "sampling"),
        ),
        rationale_hint=(
            "Decode-time sampling already has fused temperature and softmax" " kernels."
        ),
        min_share=0.05,
        likely_share=0.5,
    ),
    FusionPatternSpec(
        pattern="Fused logit softcap",
        candidate_path=(
            "python/sglang/srt/layers/elementwise.py"
            "<br>python/sglang/srt/layers/logits_processor.py"
        ),
        active_keywords=("fused_softcap", "final_logit_softcapping"),
        rationale_hint=(
            "Logit softcap math already has dedicated fused elementwise kernels."
        ),
        min_share=0.02,
        likely_share=0.2,
    ),
    FusionPatternSpec(
        pattern="PR #20667 Qwen3.5 fused QK norm + RoPE + KV cache write",
        candidate_path=(
            "PR #20667"
            "<br>python/sglang/srt/models/qwen3_5.py"
            "<br>python/sglang/srt/models/utils.py"
        ),
        active_keywords=(
            "fused_qk_norm_rope_cache_pts_quant_shuffle",
            "fused_qk_norm_mrope_3d_cache_pts_quant_shuffle",
        ),
        split_groups=(
            ("apply_qk_norm", "qknorm", "q_norm", "k_norm"),
            ("rotary", "rope", "mrope"),
            ("cache", "kv_buffer", "cache write"),
        ),
        rationale_hint=(
            "Open SGLang ROCm PR wires a fused QK-norm plus RoPE plus KV-cache"
            " family for Qwen3.5."
        ),
        origin="inflight",
        model_include=("qwen3.5", "qwen3_5"),
        min_share=0.4,
        likely_share=2.0,
        priority=100,
        subsumes=(
            "Fused QK RMSNorm + RoPE",
            "Fused QK RoPE reshape + KV cache write",
            "Fused RoPE + KV cache store",
        ),
    ),
    FusionPatternSpec(
        pattern="PR #22392 CUTLASS FP8 scaled MM replacing nvjet",
        candidate_path=(
            "PR #22392"
            "<br>sgl-kernel/python/sgl_kernel/gemm.py"
            "<br>python/sglang/srt/layers/quantization/fp8_utils.py"
        ),
        active_keywords=("cutlass_scaled_mm", "fp8_scaled_mm"),
        split_groups=(
            ("nvjet", "_scaled_mm"),
            ("memset", "memcpy128"),
        ),
        rationale_hint=(
            "Open SGLang PR replaces nvjet FP8 GEMM with CUTLASS to remove"
            " memset bubbles and extra copies."
        ),
        origin="inflight",
        min_share=0.2,
        likely_share=1.0,
        priority=90,
    ),
    FusionPatternSpec(
        pattern="vLLM-origin Attention + Quantization",
        candidate_path=(
            "vllm/compilation/passes/fusion/attn_quant_fusion.py"
            "<br>vllm/v1/attention/ops/merge_attn_states.py"
            "<br>vllm/csrc/attention/merge_attn_states.cu"
            "<br>vllm/docs/design/fusions.md"
        ),
        active_keywords=(
            "merge_attn_states",
            "attn_quant_fusion",
            "output_scale",
            "output_group_scale",
        ),
        split_groups=(
            ("attention", "flash_attn", "flashattention", "mla"),
            ("quant", "fp8", "nvfp4", "group_scale"),
        ),
        rationale_hint=(
            "vLLM combines attention merge with attention-epilogue quantization."
        ),
        origin="upstream",
        min_share=0.3,
        likely_share=1.5,
    ),
    FusionPatternSpec(
        pattern="vLLM-origin DSV3.2 fused indexer projections",
        candidate_path=(
            "vllm/model_executor/models/deepseek_v2.py"
            "<br>vllm/model_executor/models/deepseek_mtp.py"
        ),
        active_keywords=("wk_weights_proj",),
        split_groups=(
            ("wk_weights_proj", "wk", "weights_proj"),
            ("mergedcolumnparallellinear", "gemm", "matmul"),
        ),
        rationale_hint=(
            "vLLM already fuses the paired `wk` and `weights_proj` indexer"
            " projections into one DSV3.2 linear family."
        ),
        origin="upstream",
        min_share=0.2,
        likely_share=1.0,
    ),
    FusionPatternSpec(
        pattern="vLLM-origin RMSNorm + Quantization",
        candidate_path=(
            "vllm/compilation/passes/fusion/rms_quant_fusion.py"
            "<br>vllm/docs/design/fusions.md"
        ),
        active_keywords=(
            "fused_add_rms_norm_static_fp8_quant",
            "rms_quant_fusion",
            "norm_quant",
        ),
        split_groups=(
            ("rmsnorm", "layernorm", "fused_add_rms_norm"),
            ("quant", "fp8", "fp4", "per-group"),
        ),
        rationale_hint=(
            "vLLM already has a compile-time norm-plus-quant fusion family."
        ),
        origin="upstream",
        min_share=0.3,
        likely_share=1.5,
    ),
    FusionPatternSpec(
        pattern="vLLM-origin SiLU+Mul + Quantization",
        candidate_path=(
            "vllm/compilation/passes/fusion/act_quant_fusion.py"
            "<br>vllm/docs/design/fusions.md"
        ),
        active_keywords=(
            "silu_mul_quant_fp4",
            "fused_silu_mul_block_quant",
            "act_quant_fusion",
        ),
        split_groups=(
            ("silu", "gelu", "act_and_mul"),
            ("quant", "fp8", "fp4", "block_quant"),
        ),
        rationale_hint=("vLLM has an activation-plus-quant fusion family."),
        origin="upstream",
        min_share=0.3,
        likely_share=1.5,
    ),
    FusionPatternSpec(
        pattern="vLLM-origin DSV3 router GEMM",
        candidate_path=(
            "vllm/model_executor/layers/fused_moe/router/gate_linear.py"
            "<br>vllm/csrc/moe/dsv3_router_gemm_entry.cu"
        ),
        active_keywords=("dsv3_router_gemm", "fp32_router_gemm"),
        split_groups=(
            ("router", "gate", "router logits"),
            ("gemm", "matmul", "cublas", "cutlass"),
        ),
        rationale_hint=(
            "vLLM has a specialized DeepSeek router GEMM family for small"
            " decode batches."
        ),
        origin="upstream",
        min_share=0.3,
        likely_share=1.5,
    ),
    FusionPatternSpec(
        pattern="vLLM-origin GPT-OSS router GEMM",
        candidate_path=(
            "vllm/_custom_ops.py"
            "<br>vllm/model_executor/layers/fused_moe/router/gate_linear.py"
            "<br>vllm/csrc/moe/gpt_oss_router_gemm.cu"
        ),
        active_keywords=("gpt_oss_router_gemm",),
        split_groups=(
            ("router", "gate", "router logits", "gpt_oss"),
            ("gemm", "matmul", "cublas", "cutlass"),
        ),
        rationale_hint=("vLLM has a GPT-OSS-specific router GEMM path."),
        origin="upstream",
        model_include=("gpt-oss", "gpt_oss"),
        min_share=0.3,
        likely_share=1.5,
    ),
    FusionPatternSpec(
        pattern="vLLM-origin DeepSeek min-latency fused QKV-A projection",
        candidate_path=(
            "vllm/model_executor/models/deepseek_v2.py"
            "<br>vllm/csrc/dsv3_fused_a_gemm.cu"
        ),
        active_keywords=("dsv3_fused_a_gemm", "fused_qkv_a_proj"),
        split_groups=(
            ("q_a_proj", "kv_a_proj", "weights_proj"),
            ("gemm", "matmul", "cutlass", "cublas"),
        ),
        rationale_hint=(
            "vLLM has a fused DeepSeek QKV-A projection family for decode"
            " latency reduction."
        ),
        origin="upstream",
        model_include=("deepseek", "glm"),
        min_share=0.3,
        likely_share=1.5,
    ),
    FusionPatternSpec(
        pattern="PR #38621 fused QK norm + RoPE + cache + quant",
        candidate_path=(
            "PR #38621"
            "<br>vllm/csrc/fused_qk_norm_rope_cache_quant.cu"
            "<br>vllm/compilation/passes/fusion/qk_norm_rope_cache_quant_fusion.py"
        ),
        active_keywords=("fused_qk_norm_rope_cache_quant",),
        split_groups=(
            ("qknorm", "q_norm", "k_norm"),
            ("rope", "rotary", "mrope"),
            ("cache", "kv_buffer", "cache write"),
            ("quant", "fp8", "nvfp4"),
        ),
        rationale_hint=(
            "Open vLLM PR covers QK-norm plus RoPE plus cache plus quant as"
            " one fusion family."
        ),
        origin="inflight",
        min_share=0.4,
        likely_share=2.0,
        priority=100,
        subsumes=("vLLM-origin Attention + Quantization",),
    ),
    FusionPatternSpec(
        pattern="vLLM-origin MiniMax allreduce_rms kernels",
        candidate_path="vllm/model_executor/models/minimax_m2.py",
        active_keywords=("minimax_allreduce_rms", "minimax_allreduce_rmsnorm"),
        split_groups=(
            ("q_norm", "k_norm", "rmsnorm", "minimax"),
            ("allreduce", "all_reduce", "cross_device_reduce"),
        ),
        rationale_hint=(
            "vLLM includes the TRTLLM-derived MiniMax allreduce-plus-RMSNorm"
            " kernel family."
        ),
        origin="upstream",
        model_include=("minimax",),
        min_share=0.3,
        likely_share=1.5,
    ),
    FusionPatternSpec(
        pattern="vLLM fused residual add + RMSNorm",
        candidate_path=(
            "vllm/_custom_ops.py"
            "<br>vllm/compilation/passes/fusion/rms_quant_fusion.py"
        ),
        active_keywords=(
            "fused_add_rms_norm",
            "fused_add_rms_norm_static_fp8_quant",
        ),
        rationale_hint=(
            "vLLM exposes fused residual-add-plus-RMSNorm kernels and matching"
            " compile-time hooks."
        ),
        origin="upstream",
        min_share=0.1,
        likely_share=1.0,
    ),
    FusionPatternSpec(
        pattern="vLLM fused activation-and-mul",
        candidate_path=(
            "vllm/_custom_ops.py"
            "<br>vllm/compilation/passes/fusion/act_quant_fusion.py"
        ),
        active_keywords=(
            "silu_and_mul",
            "silu_and_mul_quant",
            "silu_and_mul_per_block_quant",
            "act_and_mul",
        ),
        rationale_hint=(
            "vLLM ships fused activation-and-multiply kernels plus quantized"
            " variants for the MLP epilogue."
        ),
        origin="upstream",
        min_share=0.1,
        likely_share=1.0,
    ),
    FusionPatternSpec(
        pattern="TensorRT-LLM FlashInfer residual add + RMSNorm",
        candidate_path=(
            "tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py"
            "<br>tensorrt_llm/_torch/modules/rms_norm.py"
            "<br>tensorrt_llm/_torch/auto_deploy/transform/library/fused_add_rms_norm.py"
        ),
        active_keywords=(
            "flashinfer_fused_add_rmsnorm",
            "flashinfer_gemma_fused_add_rmsnorm",
            "flashinfer::norm::FusedAddRMSNormKernel",
            "FusedAddRMSNormKernel",
            "auto_deploy::flashinfer_fused_add_rms_norm_inplace",
        ),
        rationale_hint=(
            "TensorRT-LLM exposes a FlashInfer fused residual-add plus RMSNorm"
            " family, including AutoDeploy rewrites."
        ),
        origin="upstream",
        min_share=0.1,
        likely_share=1.0,
    ),
    FusionPatternSpec(
        pattern="TensorRT-LLM Triton fused residual add + RMSNorm + FP8 quant",
        candidate_path=(
            "tensorrt_llm/_torch/auto_deploy/custom_ops/normalization/"
            "triton_fused_add_rms_norm_quant_fp8.py"
            "<br>tensorrt_llm/_torch/auto_deploy/transform/library/"
            "fuse_rmsnorm_quant_fp8.py"
        ),
        active_keywords=(
            "triton_fused_add_rms_norm_quant_fp8",
            "fuse_rmsnorm_quant_fp8",
        ),
        rationale_hint=(
            "TensorRT-LLM mainline has a Triton residual-add plus RMSNorm plus"
            " FP8-quant family in AutoDeploy."
        ),
        origin="upstream",
        min_share=0.2,
        likely_share=1.0,
        priority=20,
    ),
    FusionPatternSpec(
        pattern="TensorRT-LLM FlashInfer RMSNorm family",
        candidate_path=(
            "tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py"
            "<br>tensorrt_llm/_torch/modules/rms_norm.py"
            "<br>tensorrt_llm/_torch/auto_deploy/custom_ops/normalization/rms_norm.py"
        ),
        active_keywords=(
            "flashinfer_rmsnorm",
            "flashinfer_gemma_rmsnorm",
            "auto_deploy::flashinfer_rms_norm",
        ),
        rationale_hint=(
            "TensorRT-LLM lowers RMSNorm-style ladders to FlashInfer kernels"
            " and AutoDeploy custom ops."
        ),
        origin="upstream",
        min_share=0.1,
        likely_share=1.0,
    ),
    FusionPatternSpec(
        pattern="TensorRT-LLM FlashInfer activation / gate epilogues",
        candidate_path=(
            "tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py"
            "<br>tensorrt_llm/_torch/auto_deploy/transform/library/fuse_silu_mul.py"
            "<br>tensorrt_llm/_torch/models/modeling_gemma3.py"
        ),
        active_keywords=(
            "flashinfer_silu_and_mul",
            "flashinfer_gelu_tanh_and_mul",
            "auto_deploy::silu_and_mul",
        ),
        rationale_hint=(
            "TensorRT-LLM already rewrites gate activation plus multiply"
            " ladders into FlashInfer epilogue kernels."
        ),
        origin="upstream",
        min_share=0.1,
        likely_share=1.0,
    ),
)


def short_name(name: str, max_len: int = 96) -> str:
    text = normalize_text(name)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


@lru_cache(maxsize=65536)
def canonicalize_name(name: str) -> str:
    text = normalize_text(name)
    text = re.sub(r"0x[0-9a-fA-F]+", "0xADDR", text)
    if text.startswith("void ") and text.endswith(")"):
        depth = 0
        split_idx: Optional[int] = None
        for idx in range(len(text) - 1, -1, -1):
            char = text[idx]
            if char == ")":
                depth += 1
            elif char == "(":
                depth -= 1
                if depth == 0:
                    split_idx = idx
                    break
        if split_idx is not None:
            text = text[:split_idx]
    return text


@lru_cache(maxsize=65536)
def classify_kernel(name: str) -> str:
    # Keep the matching order explicit: strong communication/memory signals win
    # first, then we fall back to weaker category hints.
    lowered = name.lower()
    if contains_any_keyword(lowered, COMMUNICATION_STRONG_KEYWORDS):
        return "communication"
    if contains_any_keyword(lowered, MEMORY_STRONG_KEYWORDS):
        return "memory"
    looks_compute_like = contains_any_keyword(lowered, COMPUTE_HINT_KEYWORDS)
    if contains_any_keyword(lowered, MEMORY_WEAK_KEYWORDS) and not looks_compute_like:
        return "memory"
    for category, keywords in CATEGORY_PATTERNS:
        if contains_any_keyword(lowered, keywords):
            return category
    if (
        contains_any_keyword(lowered, COMMUNICATION_WEAK_KEYWORDS)
        and not looks_compute_like
    ):
        return "communication"
    return "other"


@lru_cache(maxsize=65536)
def normalize_source_location(name: str) -> str:
    text = normalize_text(name)
    match = re.match(r"(?P<path>.+?)\((?P<line>\d+)\): (?P<func>.+)$", text)
    if not match:
        return text
    path = normalize_repo_relative_path(match.group("path"))
    return f"{path}:{match.group('line')} {match.group('func')}"


def source_location_priority(location: str) -> int:
    text = str(location).strip()
    if not text or text == "unresolved":
        return -100
    penalty = 80 if is_low_signal_source_location(text) else 0
    if text.startswith("python/sglang/"):
        return 300 - penalty
    if text.startswith("sglang/"):
        return 290 - penalty
    if text.startswith("vllm/"):
        return 285 - penalty
    if text.startswith("tensorrt_llm/"):
        return 280 - penalty
    if text.startswith("sgl_kernel/"):
        return 260 - penalty
    if text.startswith("python/"):
        return 180 - penalty
    if text.startswith("torch/") or "/torch/" in text:
        return 20
    if ".py:" in text:
        return 120 - penalty
    return 0


def is_preferred_source_location(location: str) -> bool:
    text = str(location).strip()
    return (
        text.startswith("python/sglang/")
        or text.startswith("sglang/")
        or text.startswith("vllm/")
        or text.startswith("tensorrt_llm/")
        or text.startswith("sgl_kernel/")
    )


def extract_preferred_stack_location(stack: Optional[str]) -> Optional[str]:
    if not stack:
        return None
    parts = [str(part).strip() for part in str(stack).split("->")]
    ranked: List[Tuple[int, int, str]] = []
    for index, part in enumerate(parts):
        normalized = normalize_source_location(part)
        priority = source_location_priority(normalized)
        if priority <= 0:
            continue
        ranked.append((priority, index, normalized))
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


def site_display_location(site: dict) -> str:
    location = str(site.get("location") or "unresolved").strip()
    if is_preferred_source_location(location) and not is_low_signal_source_location(
        location
    ):
        return location
    stack_location = extract_preferred_stack_location(site.get("stack"))
    if stack_location:
        return stack_location
    return location


def choose_best_location(locations: Dict[str, MappingSiteAggregate]) -> str:
    if not locations:
        return "unresolved"
    ranked = sorted(
        locations.items(),
        key=lambda pair: (
            source_location_priority(pair[0]),
            pair[1].total_us,
            pair[1].count,
        ),
        reverse=True,
    )
    return ranked[0][0]


@lru_cache(maxsize=65536)
def frame_priority(frame_name: str) -> int:
    raw_text = str(frame_name).strip()
    normalized_text = normalize_source_location(raw_text)
    penalty = 80 if is_low_signal_source_location(normalized_text) else 0
    if raw_text.startswith(NOISE_FRAME_PREFIXES):
        return -20
    if normalized_text.startswith("python/sglang/"):
        return 300 - penalty
    if normalized_text.startswith("sglang/"):
        return 290 - penalty
    if normalized_text.startswith("vllm/"):
        return 285 - penalty
    if normalized_text.startswith("tensorrt_llm/"):
        return 280 - penalty
    if normalized_text.startswith("sgl_kernel/"):
        return 260 - penalty
    if normalized_text.startswith("triton_kernels/"):
        return 220 - penalty
    if normalized_text.startswith(LOW_LEVEL_FRAME_PREFIXES):
        return 0
    if raw_text.startswith("/data/") or raw_text.startswith("/Users/"):
        if "/sglang/" in raw_text:
            return 120
        if "/vllm/" in raw_text:
            return 118
        if "/TensorRT-LLM/" in raw_text or "/tensorrt_llm/" in raw_text:
            return 116
        return 100
    if ".py(" in raw_text and "/sglang/" in raw_text:
        return 110
    if ".py(" in raw_text and "/vllm/" in raw_text:
        return 108
    if ".py(" in raw_text and (
        "/TensorRT-LLM/" in raw_text or "/tensorrt_llm/" in raw_text
    ):
        return 106
    if ".py:" in normalized_text and (
        "site-packages" in raw_text or normalized_text.startswith("torch/")
    ):
        return 45
    if ".py:" in normalized_text:
        return 35
    if raw_text.startswith("<built-in method "):
        return -10
    return 0


@lru_cache(maxsize=65536)
def is_low_signal_source_location(location: str) -> bool:
    lowered = str(location).strip().lower()
    if not lowered:
        return False
    return any(token in lowered for token in LOW_SIGNAL_FUNCTION_TOKENS) or any(
        token in lowered for token in LOW_SIGNAL_PATH_TOKENS
    )


def stage_label(stage: str) -> str:
    if stage == "extend":
        return "extend/prefill"
    return stage


def stage_aliases(stage: str) -> List[str]:
    if stage == "extend":
        return ["extend", "prefill", "all"]
    if stage == "prefill":
        return ["prefill", "extend", "all"]
    if stage == "decode":
        return ["decode", "all"]
    return [stage, "all"]


def escape_md_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole else 0.0


def format_ms(value_us: float) -> str:
    return f"{value_us / 1000.0:.2f} ms"


@lru_cache(maxsize=16384)
def _is_cuda_launch_event_cached(name: str, cat: str) -> bool:
    lowered_name = normalize_text(name).lower()
    lowered_cat = normalize_text(cat).lower()
    if lowered_cat not in {"cuda_runtime", "cuda_driver"}:
        return False
    return "launch" in lowered_name


def is_cuda_launch_event(name: str, cat: str) -> bool:
    return _is_cuda_launch_event_cached(str(name), str(cat))


def is_gpu_kernel_event(event: dict) -> bool:
    # Be conservative here: first drop trace metadata / Python scopes /
    # annotations, then only accept entries with clear GPU-kernel markers.
    if not is_complete_duration_event(event):
        return False
    name = normalize_text(event.get("name", ""))
    if is_trace_metadata_name(name):
        return False
    cat = normalize_text(event.get("cat", "")).lower()
    args = event.get("args") or {}
    if is_non_kernel_trace_category(cat):
        return False
    if is_annotation_event(name, cat):
        return False
    if "kernel" in cat or cat.startswith("gpu_"):
        return True
    if looks_like_python_scope_name(name):
        return False
    return has_stream_marker(args)


def infer_stage_from_annotation_name(name: str) -> Optional[str]:
    lowered = normalize_text(name).lower()
    if not lowered:
        return None
    if "generation_1" in lowered or "decode" in lowered:
        return "decode"
    if "generation_0" in lowered or "prefill" in lowered:
        return "extend"
    return None


def build_stage_annotations(
    raw_events: Sequence[dict],
) -> Tuple[
    Dict[int, StageAnnotation],
    List[StageWindow],
    List[StageWindow],
]:
    by_external_id: Dict[int, StageAnnotation] = {}
    gpu_annotations: List[StageAnnotation] = []
    cpu_annotations: List[StageAnnotation] = []

    def should_replace(current: StageAnnotation, candidate: StageAnnotation) -> bool:
        if candidate.is_gpu != current.is_gpu:
            return candidate.is_gpu
        return (candidate.end_ts - candidate.ts) > (current.end_ts - current.ts)

    for event in raw_events:
        if not is_complete_duration_event(event):
            continue
        category = normalize_text(event.get("cat", "")).lower()
        if category not in {"user_annotation", "gpu_user_annotation"}:
            continue
        stage = infer_stage_from_annotation_name(str(event.get("name", "")))
        if not stage:
            continue
        annotation = StageAnnotation(
            stage=stage,
            ts=float(event.get("ts", 0.0)),
            end_ts=float(event.get("ts", 0.0)) + float(event.get("dur", 0.0)),
            external_id=coerce_optional_int(
                (event.get("args") or {}).get("External id")
            ),
            is_gpu=(category == "gpu_user_annotation"),
        )
        if annotation.external_id is not None:
            existing = by_external_id.get(annotation.external_id)
            if existing is None or should_replace(existing, annotation):
                by_external_id[annotation.external_id] = annotation
        if annotation.is_gpu:
            gpu_annotations.append(annotation)
        else:
            cpu_annotations.append(annotation)

    gpu_annotations.sort(key=lambda item: (item.ts, item.end_ts))
    cpu_annotations.sort(key=lambda item: (item.ts, item.end_ts))
    return (
        by_external_id,
        merge_stage_windows(gpu_annotations),
        merge_stage_windows(cpu_annotations),
    )


def merge_stage_windows(annotations: Sequence[StageAnnotation]) -> List[StageWindow]:
    merged: List[StageWindow] = []
    for annotation in annotations:
        if (
            merged
            and merged[-1].stage == annotation.stage
            and annotation.ts <= merged[-1].end_ts + 1e-3
        ):
            merged[-1] = StageWindow(
                stage=merged[-1].stage,
                ts=merged[-1].ts,
                end_ts=max(merged[-1].end_ts, annotation.end_ts),
            )
            continue
        merged.append(
            StageWindow(
                stage=annotation.stage,
                ts=annotation.ts,
                end_ts=annotation.end_ts,
            )
        )
    return merged


def resolve_stage_from_windows(
    probe_ts: float,
    windows: Sequence[StageWindow],
) -> Tuple[Optional[str], Optional[float]]:
    nearest_stage: Optional[str] = None
    nearest_gap: Optional[float] = None
    for window in windows:
        if window.ts <= probe_ts <= window.end_ts + 1e-3:
            return window.stage, 0.0
        gap = min(abs(probe_ts - window.ts), abs(probe_ts - window.end_ts))
        if nearest_gap is None or gap < nearest_gap:
            nearest_gap = gap
            nearest_stage = window.stage
    return nearest_stage, nearest_gap


def resolve_kernel_stage(
    *,
    kernel_ts: float,
    external_id: Optional[int],
    annotations_by_external_id: Dict[int, StageAnnotation],
    gpu_annotations: Sequence[StageWindow],
    cpu_annotations: Sequence[StageWindow],
) -> str:
    if external_id is not None:
        annotation = annotations_by_external_id.get(external_id)
        if annotation is not None:
            return annotation.stage
    probe_ts = kernel_ts + 1e-3
    nearest_stage: Optional[str] = None
    nearest_gap: Optional[float] = None
    for windows in (gpu_annotations, cpu_annotations):
        stage, gap = resolve_stage_from_windows(probe_ts, windows)
        if gap == 0.0 and stage is not None:
            return stage
        if stage is not None and (
            nearest_gap is None or (gap is not None and gap < nearest_gap)
        ):
            nearest_stage = stage
            nearest_gap = gap
    if (
        nearest_stage is not None
        and nearest_gap is not None
        and nearest_gap <= 20_000.0
    ):
        return nearest_stage
    return "all"


def extract_trace_data(
    trace: dict,
) -> Tuple[
    List[KernelEvent],
    List[CpuOpEvent],
    Dict[Tuple[str, str], List[PythonFrame]],
    List[LaunchEvent],
    Optional[str],
    float,
]:
    # Build the basic trace views in one pass so later stages can stay simple:
    # GPU kernels for ranking, CPU ops for External-id mapping, Python frames for
    # source attribution, and CUDA launch calls for correlation-based fallback.
    raw_events = extract_trace_events(trace)
    correlation_external = build_correlation_external_lookup(raw_events)
    (
        annotations_by_external_id,
        gpu_stage_annotations,
        cpu_stage_annotations,
    ) = build_stage_annotations(raw_events)
    chosen_pid = select_heaviest_pid(
        raw_events,
        is_gpu_kernel_event,
        preferred_substrings=("TP00", "TP-0"),
    )

    kernels: List[KernelEvent] = []
    cpu_ops: List[CpuOpEvent] = []
    launches: List[LaunchEvent] = []
    python_frames: DefaultDict[Tuple[str, str], List[PythonFrame]] = defaultdict(list)
    min_ts = None
    max_end = None

    for event in raw_events:
        if event.get("ph") != "X":
            continue

        pid = str(event.get("pid"))
        tid = str(event.get("tid"))
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        cat = str(event.get("cat", ""))
        args = event.get("args") or {}
        name = str(event.get("name", ""))

        if cat == "python_function":
            python_frames[(pid, tid)].append(
                PythonFrame(
                    name=name,
                    normalized_name=normalize_source_location(name),
                    pid=pid,
                    tid=tid,
                    ts=ts,
                    dur=dur,
                    python_id=coerce_optional_int(args.get("Python id")),
                    parent_id=coerce_optional_int(args.get("Python parent id")),
                    end_ts=ts + dur,
                    priority=frame_priority(name),
                )
            )

        correlation = coerce_optional_int(args.get("correlation"))
        external_id = coerce_optional_int(args.get("External id"))
        if external_id is None and correlation is not None:
            external_id = correlation_external.get(correlation)
        if cat == "cpu_op" and external_id is not None:
            cpu_ops.append(
                CpuOpEvent(
                    name=name,
                    pid=pid,
                    tid=tid,
                    ts=ts,
                    dur=dur,
                    external_id=external_id,
                )
            )
        if is_cuda_launch_event(name, cat) and correlation is not None:
            launches.append(
                LaunchEvent(
                    name=name,
                    pid=pid,
                    tid=tid,
                    ts=ts,
                    dur=dur,
                    correlation=correlation,
                )
            )

        if chosen_pid is None or not is_gpu_kernel_event(event) or pid != chosen_pid:
            continue

        min_ts = ts if min_ts is None else min(min_ts, ts)
        max_end = ts + dur if max_end is None else max(max_end, ts + dur)
        kernels.append(
            KernelEvent(
                name=name,
                canonical_name=canonicalize_name(name),
                category=classify_kernel(name),
                stage=resolve_kernel_stage(
                    kernel_ts=ts,
                    external_id=external_id,
                    annotations_by_external_id=annotations_by_external_id,
                    gpu_annotations=gpu_stage_annotations,
                    cpu_annotations=cpu_stage_annotations,
                ),
                pid=pid,
                tid=tid,
                ts=ts,
                dur=dur,
                external_id=external_id,
                correlation=correlation,
            )
        )

    for frames in python_frames.values():
        frames.sort(key=lambda item: (item.ts, item.end_ts))

    window_us = 0.0 if min_ts is None or max_end is None else max_end - min_ts
    return kernels, cpu_ops, dict(python_frames), launches, chosen_pid, window_us


def build_correlation_external_lookup(raw_events: Sequence[dict]) -> Dict[int, int]:
    lookup: Dict[int, int] = {}
    for event in raw_events:
        args = event.get("args", {}) or {}
        correlation = coerce_optional_int(args.get("correlation"))
        external_id = coerce_optional_int(args.get("External id"))
        if correlation is not None and external_id is not None:
            lookup[correlation] = external_id
    return lookup


def build_timed_event_index(events: Sequence[object]) -> TimedEventIndex:
    ordered = list(events)
    ordered.sort(key=lambda item: item.ts)
    return TimedEventIndex(
        events=ordered,
        start_ts=[float(item.ts) for item in ordered],
    )


def build_cpu_op_index(cpu_ops: Sequence[CpuOpEvent]) -> Dict[int, TimedEventIndex]:
    output: DefaultDict[int, List[CpuOpEvent]] = defaultdict(list)
    for cpu_op in cpu_ops:
        output[cpu_op.external_id].append(cpu_op)
    return {
        external_id: build_timed_event_index(items)
        for external_id, items in output.items()
    }


def match_cpu_op(
    kernel: KernelEvent, cpu_ops_by_external_id: Dict[int, TimedEventIndex]
) -> Optional[CpuOpEvent]:
    if kernel.external_id is None:
        return None
    return match_timed_event(
        cpu_ops_by_external_id.get(kernel.external_id, []), kernel.ts
    )


def build_launch_index(
    launch_events: Sequence[LaunchEvent],
) -> Dict[int, TimedEventIndex]:
    output: DefaultDict[int, List[LaunchEvent]] = defaultdict(list)
    for launch in launch_events:
        output[launch.correlation].append(launch)
    return {
        correlation: build_timed_event_index(items)
        for correlation, items in output.items()
    }


def match_launch_event(
    kernel: KernelEvent, launches_by_correlation: Dict[int, TimedEventIndex]
) -> Optional[LaunchEvent]:
    if kernel.correlation is None:
        return None
    return match_timed_event(
        launches_by_correlation.get(kernel.correlation, []), kernel.ts
    )


def match_timed_event(index: object, probe_ts: float):
    if not index:
        return None
    if isinstance(index, TimedEventIndex):
        events = index.events
        if not events:
            return None
        right = bisect_right(index.start_ts, probe_ts + 1e-3)
        candidates: List[object] = []
        if right > 0:
            candidates.extend(events[max(0, right - 4) : right])
        if right < len(events):
            candidates.extend(events[right : min(len(events), right + 2)])
        if not candidates:
            return None
        earlier = [item for item in candidates if item.ts <= probe_ts + 1e-3]
        if earlier:
            return min(earlier, key=lambda item: abs((item.ts + item.dur) - probe_ts))
        return min(candidates, key=lambda item: abs(item.ts - probe_ts))
    events = list(index)
    if not events:
        return None
    earlier = [item for item in events if item.ts <= probe_ts + 1e-3]
    if earlier:
        return min(earlier, key=lambda item: abs((item.ts + item.dur) - probe_ts))
    return min(events, key=lambda item: abs(item.ts - probe_ts))


def resolve_active_frames_linear(
    frames: Sequence[PythonFrame], probe_ts: float
) -> List[PythonFrame]:
    active = [item for item in frames if item.ts <= probe_ts <= item.end_ts]
    active.sort(key=lambda item: (item.ts, item.end_ts))
    return active


def thread_has_crossing_frames(frames: Sequence[PythonFrame]) -> bool:
    ordered_frames = sorted(frames, key=lambda item: (item.ts, -item.end_ts))
    stack: List[PythonFrame] = []
    for frame in ordered_frames:
        while stack and stack[-1].end_ts < frame.ts:
            stack.pop()
        if stack and frame.end_ts > stack[-1].end_ts + 1e-3:
            return True
        stack.append(frame)
    return False


def render_frame_resolution(
    active_frames: Sequence[PythonFrame],
) -> Optional[FrameResolution]:
    if not active_frames:
        return None
    chosen_frame = choose_mapping_frame(active_frames)
    if chosen_frame is None:
        return None
    return FrameResolution(
        location=chosen_frame.normalized_name,
        stack=build_stack_display(active_frames),
    )


def resolve_thread_query_times(
    frames: Sequence[PythonFrame], query_times: Sequence[float]
) -> Dict[float, Optional[FrameResolution]]:
    if not frames or not query_times:
        return {}
    ordered_frames = sorted(frames, key=lambda item: (item.ts, -item.end_ts))
    ordered_queries = sorted(set(float(ts) for ts in query_times))
    results: Dict[float, Optional[FrameResolution]] = {}
    active_frames: List[PythonFrame] = []
    frame_idx = 0
    total_frames = len(ordered_frames)

    for ts in ordered_queries:
        while frame_idx < total_frames and ordered_frames[frame_idx].ts <= ts:
            active_frames.append(ordered_frames[frame_idx])
            frame_idx += 1
        if active_frames:
            active_frames = [
                frame for frame in active_frames if frame.end_ts >= ts - 1e-3
            ]
        results[ts] = render_frame_resolution(active_frames)
    return results


def build_frame_resolution_index(
    python_frames: Dict[Tuple[str, str], List[PythonFrame]],
    query_times_by_thread: Dict[Tuple[str, str], Sequence[float]],
) -> Dict[Tuple[str, str], Dict[float, Optional[FrameResolution]]]:
    output: Dict[Tuple[str, str], Dict[float, Optional[FrameResolution]]] = {}
    for thread_key, query_times in query_times_by_thread.items():
        frames = python_frames.get(thread_key, [])
        output[thread_key] = resolve_thread_query_times(frames, query_times)
    return output


def find_active_python_frames(
    cpu_op: CpuOpEvent,
    python_frames: Dict[Tuple[str, str], List[PythonFrame]],
) -> List[PythonFrame]:
    frames = python_frames.get((cpu_op.pid, cpu_op.tid), [])
    if not frames:
        return []
    probe_ts = cpu_op.ts + min(cpu_op.dur * 0.5, 1.0)
    return resolve_active_frames_linear(frames, probe_ts)


def find_active_python_frames_at_ts(
    *,
    pid: str,
    tid: str,
    ts: float,
    python_frames: Dict[Tuple[str, str], List[PythonFrame]],
) -> List[PythonFrame]:
    frames = python_frames.get((pid, tid), [])
    if not frames:
        return []
    return resolve_active_frames_linear(frames, ts)


def render_kernel_site(
    active_frames: Sequence[PythonFrame], cpu_op_name: str
) -> Tuple[str, str, str]:
    chosen_frame = choose_mapping_frame(active_frames)
    if chosen_frame is None:
        return "unresolved", "", cpu_op_name
    return chosen_frame.normalized_name, build_stack_display(active_frames), cpu_op_name


def resolve_kernel_site_context(
    kernel: KernelEvent,
    cpu_ops_by_external_id: Dict[int, TimedEventIndex],
    python_frames: Dict[Tuple[str, str], List[PythonFrame]],
    launches_by_correlation: Dict[int, TimedEventIndex],
    frame_resolution_index: Optional[
        Dict[Tuple[str, str], Dict[float, Optional[FrameResolution]]]
    ] = None,
) -> Tuple[str, str, str]:
    # Prefer the normal External-id path first. If the kernel dropped that link,
    # fall back to the correlated CUDA launch and reuse the Python frames that
    # were active when the launch happened.
    cpu_op = match_cpu_op(kernel, cpu_ops_by_external_id)
    if cpu_op is not None:
        probe_ts = cpu_op.ts + min(cpu_op.dur * 0.5, 1.0)
        if frame_resolution_index is not None:
            resolved = frame_resolution_index.get((cpu_op.pid, cpu_op.tid), {}).get(
                probe_ts
            )
            if resolved is not None:
                return resolved.location, resolved.stack, cpu_op.name
        active_frames = find_active_python_frames(cpu_op, python_frames)
        if active_frames:
            return render_kernel_site(active_frames, cpu_op.name)

    launch_event = match_launch_event(kernel, launches_by_correlation)
    if launch_event is not None:
        if frame_resolution_index is not None:
            resolved = frame_resolution_index.get(
                (launch_event.pid, launch_event.tid), {}
            ).get(launch_event.ts)
            if resolved is not None:
                cpu_op_name = cpu_op.name if cpu_op is not None else launch_event.name
                return resolved.location, resolved.stack, cpu_op_name
        active_frames = find_active_python_frames_at_ts(
            pid=launch_event.pid,
            tid=launch_event.tid,
            ts=launch_event.ts,
            python_frames=python_frames,
        )
        if active_frames:
            cpu_op_name = cpu_op.name if cpu_op is not None else launch_event.name
            return render_kernel_site(active_frames, cpu_op_name)
        return "unresolved", "", launch_event.name

    cpu_op_name = cpu_op.name if cpu_op is not None else ""
    return "unresolved", "", cpu_op_name


def choose_mapping_frame(active_frames: Sequence[PythonFrame]) -> Optional[PythonFrame]:
    if not active_frames:
        return None
    best = active_frames[0]
    best_key = (best.priority, best.ts, -best.dur)
    for item in active_frames[1:]:
        key = (item.priority, item.ts, -item.dur)
        if key > best_key:
            best = item
            best_key = key
    return best


def build_stack_display(active_frames: Sequence[PythonFrame]) -> str:
    if not active_frames:
        return ""
    filtered = [item.normalized_name for item in active_frames if item.priority > 0]
    if not filtered:
        filtered = [active_frames[-1].normalized_name]
    return " -> ".join(filtered[-4:])


def aggregate(events: Iterable[KernelEvent], key_fn) -> Dict[str, Aggregate]:
    output: Dict[str, Aggregate] = defaultdict(Aggregate)
    for event in events:
        key = key_fn(event)
        item = output[key]
        item.total_us += event.dur
        item.count += 1
        item.max_us = max(item.max_us, event.dur)
    return output


def group_kernels_by_stage(
    kernels: Sequence[KernelEvent], default_stage: str
) -> Dict[str, List[KernelEvent]]:
    grouped: DefaultDict[str, List[KernelEvent]] = defaultdict(list)
    for kernel in kernels:
        stage = default_stage if default_stage != "all" else (kernel.stage or "all")
        grouped[stage].append(kernel)
    return dict(grouped)


def aggregate_kernel_sites(
    kernels: Sequence[KernelEvent],
    cpu_ops_by_external_id: Dict[int, TimedEventIndex],
    python_frames: Dict[Tuple[str, str], List[PythonFrame]],
    launches_by_correlation: Optional[Dict[int, TimedEventIndex]] = None,
    site_context_cache: Optional[
        Dict[Tuple[str, str, float, Optional[int], Optional[int]], Tuple[str, str, str]]
    ] = None,
) -> Dict[str, Dict[str, MappingSiteAggregate]]:
    # Each kernel is mapped independently so the fallback behavior stays easy to
    # reason about and easy to regression-test.
    output: DefaultDict[str, DefaultDict[str, MappingSiteAggregate]] = defaultdict(
        lambda: defaultdict(MappingSiteAggregate)
    )
    launch_index = launches_by_correlation or {}
    query_times_by_thread: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)
    for kernel in kernels:
        cpu_op = match_cpu_op(kernel, cpu_ops_by_external_id)
        if cpu_op is not None:
            query_times_by_thread[(cpu_op.pid, cpu_op.tid)].append(
                cpu_op.ts + min(cpu_op.dur * 0.5, 1.0)
            )
        launch_event = match_launch_event(kernel, launch_index)
        if launch_event is not None:
            query_times_by_thread[(launch_event.pid, launch_event.tid)].append(
                launch_event.ts
            )
    frame_resolution_index = build_frame_resolution_index(
        python_frames, query_times_by_thread
    )
    resolved_cache = site_context_cache if site_context_cache is not None else {}
    for kernel in kernels:
        cache_key = (
            kernel.pid,
            kernel.tid,
            kernel.ts,
            kernel.external_id,
            kernel.correlation,
        )
        cached = resolved_cache.get(cache_key)
        if cached is None:
            cached = resolve_kernel_site_context(
                kernel,
                cpu_ops_by_external_id,
                python_frames,
                launch_index,
                frame_resolution_index=frame_resolution_index,
            )
            resolved_cache[cache_key] = cached
        location, stack, cpu_op_name = cached

        item = output[kernel.canonical_name][location]
        item.total_us += kernel.dur
        item.count += 1
        if cpu_op_name:
            item.cpu_ops[cpu_op_name] += 1
        if stack:
            item.stacks[stack] += 1
    return {kernel_name: dict(locations) for kernel_name, locations in output.items()}


def merge_site_stats(
    destination: DefaultDict[str, DefaultDict[str, MappingSiteAggregate]],
    source: Dict[str, Dict[str, MappingSiteAggregate]],
) -> None:
    for kernel_name, locations in source.items():
        for location, aggregate_item in locations.items():
            target = destination[kernel_name][location]
            target.total_us += aggregate_item.total_us
            target.count += aggregate_item.count
            target.cpu_ops.update(aggregate_item.cpu_ops)
            target.stacks.update(aggregate_item.stacks)


def build_stage_payload(
    site_stats: Dict[str, Dict[str, MappingSiteAggregate]],
    kernel_categories: Dict[str, str],
) -> Dict[str, dict]:
    kernels_payload: Dict[str, dict] = {}
    for kernel_name, locations in sorted(site_stats.items()):
        total_us = sum(item.total_us for item in locations.values())
        sites = []
        for location, aggregate_item in sorted(
            locations.items(),
            key=lambda pair: pair[1].total_us,
            reverse=True,
        ):
            sites.append(
                {
                    "location": location,
                    "display_location": extract_preferred_stack_location(
                        aggregate_item.stacks.most_common(1)[0][0]
                        if aggregate_item.stacks
                        else None
                    )
                    or location,
                    "launches": aggregate_item.count,
                    "total_us": round(aggregate_item.total_us, 3),
                    "share_pct_within_kernel": round(
                        pct(aggregate_item.total_us, total_us), 3
                    ),
                    "top_cpu_op": (
                        aggregate_item.cpu_ops.most_common(1)[0][0]
                        if aggregate_item.cpu_ops
                        else None
                    ),
                    "stack": (
                        aggregate_item.stacks.most_common(1)[0][0]
                        if aggregate_item.stacks
                        else None
                    ),
                }
            )
        sites.sort(
            key=lambda site: (
                source_location_priority(site_display_location(site)),
                float(site.get("total_us", 0.0)),
                int(site.get("launches", 0)),
            ),
            reverse=True,
        )
        kernels_payload[kernel_name] = {
            "category": kernel_categories.get(kernel_name, "other"),
            "sites": sites,
            "best_location": (
                site_display_location(sites[0])
                if sites
                else choose_best_location(locations)
            ),
        }
    return {"kernels": kernels_payload}


def load_kernel_map(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def relaxed_kernel_entry_lookup(
    kernels: Dict[str, dict], kernel_name: str
) -> Optional[dict]:
    if kernel_name in kernels:
        return kernels[kernel_name]
    lowered = kernel_name.lower()
    best_key = None
    best_score = -1
    for candidate_key in kernels:
        candidate_lowered = candidate_key.lower()
        if candidate_lowered.startswith(lowered) or lowered.startswith(
            candidate_lowered
        ):
            score = min(len(candidate_lowered), len(lowered))
        elif candidate_lowered in lowered or lowered in candidate_lowered:
            score = min(len(candidate_lowered), len(lowered)) // 2
        else:
            continue
        if score > best_score:
            best_key = candidate_key
            best_score = score
    if best_key:
        return kernels.get(best_key)

    # Long auto-generated kernels such as CUTLASS / FlashAttention templates can
    # differ in the middle of the symbol while still sharing the same high-level
    # family. Fall back to a conservative common-prefix match so we can still
    # recover the higher-level Python callsite from the mapping trace.
    lowered_compact = normalize_match_text(kernel_name)
    if len(lowered_compact) < 96:
        return alias_kernel_entry_lookup(kernels, kernel_name)

    def common_prefix_len(left: str, right: str) -> int:
        count = 0
        for left_ch, right_ch in zip(left, right):
            if left_ch != right_ch:
                break
            count += 1
        return count

    best_key = None
    best_score = -1
    for candidate_key in kernels:
        candidate_compact = normalize_match_text(candidate_key)
        if len(candidate_compact) < 96:
            continue
        prefix_len = common_prefix_len(lowered_compact, candidate_compact)
        shorter_len = min(len(lowered_compact), len(candidate_compact))
        if prefix_len < 64 or prefix_len < int(shorter_len * 0.4):
            continue
        score = prefix_len
        if lowered_compact.startswith(
            "voidcutlassdevicekernelflash"
        ) and candidate_compact.startswith("voidcutlassdevicekernelflash"):
            score += 32
        if score > best_score:
            best_key = candidate_key
            best_score = score
    if best_key:
        return kernels.get(best_key)
    return alias_kernel_entry_lookup(kernels, kernel_name)


def lookup_kernel_map_entry(
    kernel_map: dict, stage: str, kernel_name: str
) -> Optional[dict]:
    stage_map = kernel_map.get("stages", {})
    for candidate_stage in stage_aliases(stage):
        entry = relaxed_kernel_entry_lookup(
            stage_map.get(candidate_stage, {}).get("kernels", {}),
            kernel_name,
        )
        if entry:
            return entry
    return relaxed_kernel_entry_lookup(
        kernel_map.get("global", {}).get("kernels", {}), kernel_name
    )


def best_site_summary(kernel_entry: Optional[dict]) -> Tuple[str, str]:
    if not kernel_entry:
        return "unresolved", "-"
    sites = kernel_entry.get("sites") or []
    if not sites:
        return kernel_entry.get("best_location", "unresolved"), "-"
    preferred_sites = [
        site
        for site in sites
        if is_preferred_source_location(site_display_location(site))
    ]
    candidate_sites = preferred_sites or sites
    rendered_locations = []
    rendered_cpu_ops = []
    for site in candidate_sites[:2]:
        location = site_display_location(site)
        share = site.get("share_pct_within_kernel")
        if len(candidate_sites) > 1 and share is not None:
            rendered_locations.append(f"{location} (site share {share:.0f}%)")
        else:
            rendered_locations.append(location)
        cpu_op = site.get("top_cpu_op")
        if cpu_op:
            rendered_cpu_ops.append(cpu_op)
    return "<br>".join(rendered_locations), (
        "<br>".join(rendered_cpu_ops) if rendered_cpu_ops else "-"
    )


def resolve_kernel_entry(
    stage: str,
    kernel_name: str,
    local_stage_payload: dict,
    external_kernel_map: Optional[dict],
) -> Optional[dict]:
    if external_kernel_map:
        kernel_entry = lookup_kernel_map_entry(external_kernel_map, stage, kernel_name)
        if kernel_entry:
            return kernel_entry
    return relaxed_kernel_entry_lookup(
        local_stage_payload.get("kernels", {}), kernel_name
    )


def build_kernel_rows(
    stage: str,
    kernel_stats: Dict[str, Aggregate],
    kernel_categories: Dict[str, str],
    local_stage_payload: dict,
    external_kernel_map: Optional[dict],
) -> List[KernelRow]:
    rows: List[KernelRow] = []
    for kernel_name, aggregate_item in sorted(
        kernel_stats.items(),
        key=lambda pair: pair[1].total_us,
        reverse=True,
    ):
        kernel_entry = resolve_kernel_entry(
            stage, kernel_name, local_stage_payload, external_kernel_map
        )
        location, cpu_op = best_site_summary(kernel_entry)
        rows.append(
            KernelRow(
                name=kernel_name,
                category=kernel_categories.get(kernel_name, "other"),
                aggregate=aggregate_item,
                location=location,
                cpu_op=cpu_op,
                entry=kernel_entry,
            )
        )
    return rows


def limit_kernel_rows(rows: Sequence[KernelRow], table_limit: int) -> List[KernelRow]:
    if table_limit <= 0:
        return list(rows)
    return list(rows[:table_limit])


def entry_sites(kernel_entry: Optional[dict]) -> List[dict]:
    if not kernel_entry:
        return []
    sites = kernel_entry.get("sites") or []
    return [site for site in sites if site.get("location")]


def ordered_unique(values: Iterable[str], limit: int = 4) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def kernel_row_locations(row: KernelRow, limit: int = 4) -> List[str]:
    values = [site_display_location(site) for site in entry_sites(row.entry)]
    if not values and row.location and row.location != "unresolved":
        values = [fragment.strip() for fragment in row.location.split("<br>")]
    return ordered_unique(values, limit=limit)


def format_location_for_fusion_display(location: str) -> str:
    text = normalize_text(location)
    match = re.match(r"(?P<path>.+?):(?P<line>\d+)\s+(?P<func>.+)$", text)
    if not match:
        return text
    return f"{match.group('func')} @ {match.group('path')}:{match.group('line')}"


def normalize_match_text(text: object) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", normalize_text(text)).lower()


def kernel_entry_total_us(entry: Optional[dict]) -> float:
    if not entry:
        return 0.0
    return sum(float(site.get("total_us", 0.0)) for site in entry.get("sites", []))


def kernel_entry_lookup_text(kernel_name: str, entry: Optional[dict]) -> str:
    parts = [kernel_name]
    if entry:
        parts.append(str(entry.get("best_location") or ""))
        for site in entry.get("sites", [])[:4]:
            parts.append(str(site.get("location") or ""))
            parts.append(str(site.get("display_location") or ""))
            parts.append(str(site.get("top_cpu_op") or ""))
            parts.append(str(site.get("stack") or ""))
    return normalize_match_text(" ".join(parts))


def kernel_alias_token_groups(kernel_name: str) -> List[Tuple[str, ...]]:
    lowered = normalize_match_text(kernel_name)
    groups: List[Tuple[str, ...]] = []
    if "flashattnfwdcombine" in lowered:
        groups.append(
            (
                "flashattnfwdsm90",
                "flashattnvarlenfunc",
                "vllmflashattnflashattninterface",
                "vllmfa3cfwd",
            )
        )
    if "kernelmha" in lowered:
        groups.append(
            (
                "maskedmultiheadattentionkernel",
                "attentioninplace",
                "attentionbackendtrtllm",
            )
        )
    if "applybiasropeupdatekvcachev2" in lowered:
        groups.append(
            (
                "fusedqknormropekernel",
                "applyqknormrope",
                "modelingqwen3py98applyqknormrope",
            )
        )
    if lowered.startswith("memset"):
        groups.append(("memset",))
    return groups


def alias_kernel_entry_lookup(
    kernels: Dict[str, dict], kernel_name: str
) -> Optional[dict]:
    alias_groups = kernel_alias_token_groups(kernel_name)
    if not alias_groups:
        return None

    best_key = None
    best_score = -1
    for candidate_key, entry in kernels.items():
        candidate_text = kernel_entry_lookup_text(candidate_key, entry)
        score = 0
        for group_index, group in enumerate(alias_groups):
            group_score = max(
                (len(token) for token in group if token in candidate_text),
                default=0,
            )
            if group_score:
                score += 1000 * (group_index + 1) + group_score
        if score <= 0:
            continue
        score += max(
            source_location_priority(str(entry.get("best_location") or "")),
            source_location_priority(best_site_summary(entry)[0]),
        )
        score += int(kernel_entry_total_us(entry) // 10)
        if score > best_score:
            best_key = candidate_key
            best_score = score
    return kernels.get(best_key) if best_key else None


def row_matches(row: KernelRow, *needles: str) -> bool:
    lowered = " ".join([row.name, row.location, row.cpu_op]).lower()
    lowered_compact = normalize_match_text(lowered)
    for needle in needles:
        needle_lowered = needle.lower()
        if needle_lowered in lowered:
            return True
        needle_compact = normalize_match_text(needle)
        if needle_compact and needle_compact in lowered_compact:
            return True
    return False


def summarize_text(values: Iterable[str], limit: int = 4) -> str:
    items = ordered_unique(values, limit=limit)
    return "<br>".join(items) if items else "-"


def summarize_locations(values: Iterable[str], limit: int = 4) -> str:
    items = ordered_unique(
        (format_location_for_fusion_display(value) for value in values),
        limit=limit,
    )
    return "<br>".join(items) if items else "-"


def summarize_evidence(
    rows: Sequence[KernelRow],
    total_us: float,
    limit: int = 3,
    min_share_pct: float = 1.0,
) -> str:
    items = []
    for row in rows:
        share = pct(row.total_us, total_us)
        if share < min_share_pct:
            continue
        items.append(f"{row.name} ({share:.1f}%)")
        if len(items) >= limit:
            break
    return "<br>".join(items) if items else "-"


def model_path_from_server_args(server_args: Optional[dict]) -> str:
    if not isinstance(server_args, dict):
        return ""
    return str(server_args.get("model_path") or server_args.get("model") or "")


def fusion_framework_hints(spec: FusionPatternSpec) -> set[str]:
    text = normalize_text(spec.candidate_path).lower()
    hints: set[str] = set()
    if "vllm/" in text:
        hints.add("vllm")
    if "tensorrt_llm/" in text:
        hints.add("trtllm")
    if any(token in text for token in ("python/sglang/", "sgl-kernel/", "sgl_kernel/")):
        hints.add("sglang")
    return hints


def pattern_supports_framework(
    spec: FusionPatternSpec, framework: Optional[str]
) -> bool:
    normalized = normalize_text(framework).lower()
    if not normalized or normalized == "auto":
        return True
    hints = fusion_framework_hints(spec)
    if not hints:
        return True
    return normalized in hints


def matching_rows_for_keywords(
    kernel_rows: Sequence[KernelRow],
    keywords: Sequence[str],
) -> List[KernelRow]:
    if not keywords:
        return []
    return [row for row in kernel_rows if row_matches(row, *keywords)]


def row_identity(row: KernelRow) -> Tuple[str, str, str]:
    return (row.name, row.location, row.cpu_op)


def merge_kernel_rows(*groups: Sequence[KernelRow]) -> List[KernelRow]:
    output: List[KernelRow] = []
    seen = set()
    for group in groups:
        for row in group:
            row_key = row_identity(row)
            if row_key in seen:
                continue
            seen.add(row_key)
            output.append(row)
    return output


def pattern_model_matches(spec: FusionPatternSpec, model_path: str) -> bool:
    if spec.model_include and not any(
        token in model_path for token in spec.model_include
    ):
        return False
    if spec.model_exclude and any(token in model_path for token in spec.model_exclude):
        return False
    return True


def pattern_status(spec: FusionPatternSpec, has_active_match: bool) -> str:
    if spec.origin == "mainline":
        return "mainline direct" if has_active_match else "mainline split"
    if spec.origin == "upstream":
        return "upstream direct" if has_active_match else "upstream split"
    return "pending direct" if has_active_match else "pending split"


def build_pattern_rationale(
    spec: FusionPatternSpec,
    has_active_match: bool,
    related_us: float,
    total_us: float,
) -> str:
    share = pct(related_us, total_us)
    if spec.origin == "mainline":
        if has_active_match:
            return (
                f"`{spec.pattern}` is present in this trace ({share:.1f}% related GPU time). "
                f"{spec.rationale_hint}"
            )
        return (
            f"Split kernels in this family take {share:.1f}% of GPU time. "
            f"This tree already has a matching path. {spec.rationale_hint}"
        )
    if spec.origin == "upstream":
        return (
            f"Matches an upstream path ({share:.1f}% related GPU time). "
            f"{spec.rationale_hint}"
        )
    return (
        f"Matches an open upstream path ({share:.1f}% related GPU time). "
        f"{spec.rationale_hint}"
    )


def pattern_span(spec: FusionPatternSpec) -> int:
    return max(len(spec.split_groups), 1 if spec.active_keywords else 0)


def fusion_priority_key(item: FusionOpportunity) -> Tuple[int, int, int, float]:
    return (
        item.priority,
        item.pattern_span,
        len(item.covered_row_keys),
        item.related_us,
    )


def detect_pattern_match(
    spec: FusionPatternSpec,
    kernel_rows: Sequence[KernelRow],
    total_us: float,
    model_path: str,
    tp_size: int,
    framework: Optional[str],
) -> Optional[FusionOpportunity]:
    if total_us <= 0:
        return None
    if not pattern_supports_framework(spec, framework):
        return None
    if spec.require_tp and tp_size < spec.min_tp_size:
        return None
    if not pattern_model_matches(spec, model_path):
        return None

    active_rows = matching_rows_for_keywords(kernel_rows, spec.active_keywords)
    split_groups = [
        matching_rows_for_keywords(kernel_rows, keywords)
        for keywords in spec.split_groups
    ]
    has_active_match = bool(active_rows)
    has_split_match = bool(split_groups) and all(split_groups)
    if not has_active_match and not has_split_match:
        return None

    related_rows = merge_kernel_rows(active_rows, *split_groups)
    related_us = sum(row.total_us for row in related_rows)
    if related_us <= 0:
        return None
    if not has_active_match and pct(related_us, total_us) < spec.min_share:
        return None

    return FusionOpportunity(
        pattern=spec.pattern,
        status=pattern_status(spec, has_active_match),
        confidence=(
            "Confirmed"
            if has_active_match or pct(related_us, total_us) >= spec.likely_share
            else "Candidate"
        ),
        related_us=related_us,
        evidence=summarize_evidence(related_rows, total_us),
        current_locations=summarize_locations(
            location for row in related_rows for location in kernel_row_locations(row)
        ),
        candidate_path=spec.candidate_path,
        rationale=build_pattern_rationale(
            spec=spec,
            has_active_match=has_active_match,
            related_us=related_us,
            total_us=total_us,
        ),
        covered_row_keys=tuple(row_identity(row) for row in related_rows),
        pattern_span=pattern_span(spec),
        has_active_match=has_active_match,
        priority=spec.priority,
        subsumes=spec.subsumes,
    )


def detect_fusion_opportunities(
    kernel_rows: Sequence[KernelRow],
    total_us: float,
    server_args: Optional[dict],
    framework: Optional[str] = None,
) -> List[FusionOpportunity]:
    opportunities: List[FusionOpportunity] = []
    if total_us <= 0:
        return opportunities

    model_path = model_path_from_server_args(server_args).lower()
    tp_size = 1
    if isinstance(server_args, dict):
        tp_size = int(server_args.get("tp_size") or 1)

    raw_matches: List[FusionOpportunity] = []
    for spec in FUSION_PATTERN_REGISTRY:
        opportunity = detect_pattern_match(
            spec=spec,
            kernel_rows=kernel_rows,
            total_us=total_us,
            model_path=model_path,
            tp_size=tp_size,
            framework=framework,
        )
        if opportunity is not None:
            raw_matches.append(opportunity)

    raw_matches.sort(key=fusion_priority_key, reverse=True)
    consumed_row_keys = set()
    blocked_patterns = set()
    for opportunity in raw_matches:
        if opportunity.pattern in blocked_patterns:
            continue
        if any(
            row_key in consumed_row_keys for row_key in opportunity.covered_row_keys
        ):
            continue
        opportunities.append(opportunity)
        consumed_row_keys.update(opportunity.covered_row_keys)
        blocked_patterns.update(opportunity.subsumes)
    return opportunities
