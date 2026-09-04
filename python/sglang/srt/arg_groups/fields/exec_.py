"""Config fields of the ``exec`` namespace.

One class per namespace. The class *is* the namespace: a field declared here
lands in the ``exec`` bag, which is what ``get_exec()`` returns, so a reader
spells it exactly as before. ``ServerArgs`` composes these classes, so the
record stays one flat object -- the split moves where declarations live, not
how config is shaped at runtime.
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import (
    List,
    Literal,
    Optional,
)

from sglang.srt.arg_groups.arg_utils import (
    A,
    Arg,
)
from sglang.srt.arg_groups.choices import (
    ATTENTION_BACKEND_CHOICES,
    FP4_GEMM_RUNNER_BACKEND_CHOICES,
    FP8_GEMM_RUNNER_BACKEND_CHOICES,
    GRAMMAR_BACKEND_CHOICES,
    LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
    MOE_RUNNER_BACKEND_CHOICES,
    RL_ON_POLICY_TARGET_CHOICES,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    parse_cuda_graph_config_arg,
)


@dataclasses.dataclass
class ExecFeatures:
    """Namespace ``exec.features``."""

    _NS_PATH = "exec.features"
    enable_fp32_lm_head: A[
        bool,
        "If set, the LM head outputs (logits) are in FP32.",
    ] = False
    enable_tf32_matmul: A[
        bool,
        Arg(
            help="Enable float32 matmuls to use TensorFloat32 precision for better performance (via torch.set_float32_matmul_precision). CUDA only.",
            resolvable=True,
        ),
    ] = False

    # -------------------------------------------------------------------------
    # Misc runtime features
    # -------------------------------------------------------------------------
    enable_memory_saver: A[
        bool,
        "Allow saving memory using release_memory_occupation and resume_memory_occupation",
    ] = False
    enable_weights_cpu_backup: A[
        bool,
        "Save model weights (both main model and draft model, if any) to CPU memory during release_weights_occupation and resume_weights_occupation",
    ] = False
    enable_draft_weights_cpu_backup: A[
        bool,
        "Save draft model weights to CPU memory during release_weights_occupation and resume_weights_occupation",
    ] = False
    enable_custom_logit_processor: A[
        bool,
        "Enable users to pass custom logit processors to the server (disabled by default for security)",
    ] = False
    enable_return_hidden_states: A[
        bool,
        "Enable returning full hidden states with responses. Equivalent to "
        "`--return-hidden-states-mode full`.",
    ] = False
    return_hidden_states_mode: A[
        Optional[str],
        Arg(
            help="Set the maximum hidden-state return mode supported by the "
            "server. `last` allows requests with return_hidden_states=False or "
            "`last`; `full` also allows return_hidden_states=True.",
            choices=["last", "full"],
        ),
    ] = None
    enable_return_routed_experts: A[
        bool, "Enable returning routed experts of each layer with responses."
    ] = False
    enable_return_indexer_topk: A[
        bool,
        "Enable returning indexer topk indices of layers with indexer with responses.",
    ] = False
    disable_outlines_disk_cache: A[
        bool,
        "Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency.",
    ] = False
    enable_mis: A[
        bool,
        "Enable Multi-Item Scoring optimization. Combines query and multiple items into a single sequence for efficient batch processing. Requires --attention-backend flashinfer; auto-disables CUDA graph, radix cache, and chunked prefill.",
    ] = False


@dataclasses.dataclass
class ExecKernel:
    """Namespace ``exec.kernel``."""

    _NS_PATH = "exec.kernel"

    # -------------------------------------------------------------------------
    # Kernel backend
    # -------------------------------------------------------------------------
    attention_backend: A[
        Optional[str],
        Arg(
            help="Choose the kernels for attention layers.",
            choices=ATTENTION_BACKEND_CHOICES,
            resolvable=True,
        ),
    ] = None
    decode_attention_backend: A[
        Optional[str],
        Arg(
            help="Choose the kernels for decode attention layers (have priority over --attention-backend).",
            choices=ATTENTION_BACKEND_CHOICES,
            resolvable=True,
        ),
    ] = None
    enable_lean_attention: A[
        Optional[bool],
        "Enable Lean (Work-Centric) Attention decode kernel for long-context serving. When None (default), uses auto-gate that activates Lean for long contexts and falls back to standard kernel for short contexts. Set to True to force enable, False to force disable.",
    ] = None
    prefill_attention_backend: A[
        Optional[str],
        Arg(
            help="Choose the kernels for prefill attention layers (have priority over --attention-backend).",
            choices=ATTENTION_BACKEND_CHOICES,
            resolvable=True,
        ),
    ] = None
    sampling_backend: A[
        Optional[str],
        Arg(
            help="Choose the kernels for sampling layers.",
            no_cli=True,
            resolvable=True,
        ),
    ] = None
    grammar_backend: A[
        Optional[str],
        Arg(
            help="Choose the backend for grammar-guided decoding.",
            choices=GRAMMAR_BACKEND_CHOICES,
        ),
    ] = None
    fp8_gemm_runner_backend: A[
        str,
        Arg(
            help="Choose the runner backend for Blockwise FP8 GEMM operations. Options: 'auto' (default, auto-selects based on hardware; MXFP8 dense picks flashinfer_cutedsl on SM100/SM103 and FlashInfer CUTLASS on other supported Blackwell GPUs), 'deep_gemm' (JIT-compiled; enabled by default on NVIDIA Hopper (SM90) and Blackwell (SM100) when DeepGEMM is installed), 'flashinfer_trtllm' (optimal for Blackwell and low-latency), 'flashinfer_cutlass' (FlashInfer CUTLASS groupwise FP8 GEMM), 'flashinfer_cutedsl' (FlashInfer CuTe DSL MXFP8 GEMM on SM100/SM103), 'flashinfer_deepgemm' (Hopper SM90 only; uses swapAB optimization for small M dimensions in decoding), 'cutlass' (optimal for SM120 GPUs), 'triton' (fallback, widely compatible), 'aiter' (ROCm only). ",
            cli_name="--fp8-gemm-backend",
            choices=FP8_GEMM_RUNNER_BACKEND_CHOICES,
            resolvable=True,
        ),
    ] = "auto"
    fp4_gemm_runner_backend: A[
        str,
        Arg(
            help="Choose the runner backend for NVFP4 GEMM operations. Options: 'auto' (default; selects flashinfer_cutedsl on SM100, marlin on SM80-SM90, flashinfer_cutlass otherwise (including SM120)), 'flashinfer_cutlass' (FlashInfer CUTLASS backend), 'flashinfer_cudnn' (FlashInfer cuDNN backend, optimal on CUDA 13+ with cuDNN 9.15+), 'flashinfer_cutedsl' (FlashInfer CuTe DSL backend), 'flashinfer_trtllm' (FlashInfer TensorRT-LLM backend, requires different weight preparation with shuffling), 'marlin' (weight-only W4A16 fallback for SM80+). ",
            cli_name="--fp4-gemm-backend",
            choices=FP4_GEMM_RUNNER_BACKEND_CHOICES,
            resolvable=True,
        ),
    ] = "auto"
    bf16_gemm_backend: A[
        str,
        Arg(
            help="Choose the backend for unquantized BF16 GEMM operations. Options: 'auto' (default; selects 'cutedsl' on SM10x GPUs, except deterministic inference selects 'torch'; otherwise uses cuBLAS via torch.nn.functional.linear), 'cutedsl' (SGLang JIT CuTe DSL TGV BF16 GEMM on SM10x; dispatches between the allowlisted low-M Split-K kernel, the CuTe DSL kernel, and cuBLAS; set SGLANG_ENABLE_BF16_SPLITK_GEMM=0 to disable Split-K), 'flashinfer_pr4266' (legacy compatibility alias for the optimized CuTe DSL path), 'gemv', 'torch' (always uses cuBLAS via torch.nn.functional.linear).",
            cli_name="--bf16-gemm-backend",
            choices=["auto", "cutedsl", "flashinfer_pr4266", "gemv", "torch"],
        ),
    ] = "auto"
    dsa_prefill_backend: A[
        Optional[str],
        Arg(
            help="DSA (DeepSeek Sparse Attention) prefill backend. If not specified, auto-detects based on hardware and kv_cache_dtype.",
            choices=[
                "flashmla_sparse",
                "flashmla_sparse_q8",
                "flashmla_kv",
                "flashmla_auto",
                "flashinfer_sparse_mla",
                "fa3",
                "tilelang",
                "aiter",
                "trtllm",
            ],
            resolvable=True,
        ),
    ] = None
    dsv4_prefill_backend: A[
        str,
        Arg(
            help=(
                "DeepSeek-V4 sparse prefill backend. 'auto' and "
                "'flashmla_sparse' use the existing BF16 sparse prefill path; "
                "'flashmla_sparse_q8' enables the Q8KV8 sparse prefill path."
            ),
            choices=["auto", "flashmla_sparse", "flashmla_sparse_q8"],
        ),
    ] = "auto"
    dsa_decode_backend: A[
        Optional[str],
        Arg(
            help="DSA (DeepSeek Sparse Attention) decode backend. If not specified, auto-detects based on hardware and kv_cache_dtype.",
            choices=[
                "flashmla_sparse",
                "flashmla_sparse_q8",
                "flashmla_kv",
                "flashmla_auto",
                "flashinfer_sparse_mla",
                "fa3",
                "tilelang",
                "aiter",
                "trtllm",
            ],
            resolvable=True,
        ),
    ] = None
    dsa_paged_mqa_logits_backend: A[
        str,
        Arg(
            help="DSA indexer paged MQA logits kernel backend. Options: 'auto' (default; DeepGEMM on CUDA, aiter on ROCm), 'deepgemm', 'cutedsl' (CuTe DSL kernel, SM 100 (Blackwell) only; wins at low batch size and long context), 'aiter' (ROCm only).",
            choices=["auto", "deepgemm", "cutedsl", "aiter"],
        ),
    ] = "auto"
    dsa_topk_backend: A[
        str,
        Arg(
            help="DSA indexer top-k backend for the target model. Options: 'sgl-kernel', 'torch', 'flashinfer'. The 'torch' backend currently requires SGLANG_DSA_FUSE_TOPK=false.",
            choices=["sgl-kernel", "torch", "flashinfer"],
        ),
    ] = "sgl-kernel"
    disable_flashinfer_autotune: A[
        bool,
        "Disable FlashInfer autotuning.",
    ] = False
    flashinfer_autotune_skip_ops: A[
        Optional[List[str]],
        Arg(
            help=(
                "FlashInfer custom-op identifiers to skip during autotuning. "
                "Skipped ops use FlashInfer's heuristic fallback. SGLang "
                "temporarily skips mxfp8_gemm by default due to an IMA."
            ),
            nargs="+",
        ),
    ] = None
    triton_attention_reduce_in_fp32: A[
        bool,
        "Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16."
        "This only affects Triton attention kernels.",
    ] = False
    triton_attention_num_kv_splits: A[
        int,
        "The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8.",
    ] = 8
    triton_attention_split_tile_size: A[
        Optional[int],
        "The size of split KV tile in flash decoding Triton kernel. Used for deterministic inference.",
    ] = None
    flashinfer_mla_disable_ragged: A[
        bool, "Not using ragged prefill wrapper when running flashinfer mla"
    ] = False
    enable_fused_qk_norm_rope: A[
        bool, "Enable fused qk normalization and rope rotary embedding."
    ] = False
    enable_precise_embedding_interpolation: A[
        bool,
        "Enable corner alignment for resize of embeddings grid to ensure more accurate(but slower) evaluation of interpolated embedding values.",
    ] = False
    enable_deepseek_v4_fp4_indexer: A[
        bool,
        "Enable the experimental FP4 C4 indexer path for DeepSeek V4. Default keeps the existing indexer implementation.",
    ] = False


@dataclasses.dataclass
class ExecMamba:
    """Namespace ``exec.mamba``."""

    _NS_PATH = "exec.mamba"
    mamba_backend: A[
        str,
        Arg(
            help="Choose the kernel backend for Mamba SSM operations. Default is 'triton'. Options: 'triton' (default), 'flashinfer' (requires FlashInfer with Mamba support).",
            choices=["triton", "flashinfer"],
        ),
    ] = "triton"
    mamba_ssm_dtype: A[
        Optional[str],
        Arg(
            help="The data type of the SSM states in mamba cache. If not set, will be read from model config (mamba_ssm_dtype).",
            choices=["float32", "bfloat16", "float16"],
        ),
    ] = None
    mamba_max_states_per_path: A[
        int,
        "Maximum number of cached Mamba states retained per root-to-tail path "
        "(-1 means unlimited). When enabled, after each insert the shallowest eligible "
        "interior states beyond the cap are removed while their full KV remains. "
        "Tail, fork, and locked nodes are preserved. Must be -1 or a positive integer.",
    ] = -1
    enable_mamba_cache_stochastic_rounding: A[
        bool,
        "Enable stochastic rounding when writing FP16 Mamba SSM cache states. Requires --mamba-ssm-dtype float16 and CUDA. With --mamba-backend triton, requires SM100.",
    ] = False
    mamba_cache_philox_rounds: A[
        int,
        "Number of Philox rounds to use for stochastic rounding of FP16 Mamba SSM cache writes. Triton uses the Triton default when set to 0; FlashInfer uses 10 rounds when set to 0.",
    ] = 0
    mamba_radix_cache_strategy: A[
        str,
        Arg(
            help="The strategy to use for mamba radix cache.",
            choices=["auto", "no_buffer", "extra_buffer", "extra_buffer_lazy"],
            resolvable=True,
        ),
    ] = "auto"
    uses_mamba_radix_cache: A[
        bool,
        Arg(
            help="(Derived) whether the model routes through the hybrid-mamba "
            "radix cache handling; resolved from the model architecture, no "
            "CLI surface.",
            no_cli=True,
            resolvable=True,
        ),
    ] = False
    mamba_track_interval: A[
        int,
        "The interval to track the mamba state during decode.",
    ] = 256
    enable_int8_mamba_checkpoint: A[
        bool,
        "Store radix-cached linear-attn (mamba) states in int8 (separate checkpoint pool) for ~2x cached-prefix capacity at fixed memory.",
    ] = False
    int8_mamba_ckpt_size: A[
        Optional[int],
        "Number of int8 mamba checkpoint slots (default: 2x the active mamba pool size).",
    ] = None
    linear_attn_backend: A[
        str,
        Arg(
            help="The default kernel backend for linear attention (GDN/KDA). Can be overridden per-mode by --linear-attn-decode-backend and --linear-attn-prefill-backend. The Helion backend is KDA-only.",
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
        ),
    ] = "triton"
    linear_attn_decode_backend: A[
        Optional[str],
        Arg(
            help="Override the kernel backend for linear attention decode. If not set, uses --linear-attn-backend.",
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
        ),
    ] = None
    linear_attn_prefill_backend: A[
        Optional[str],
        Arg(
            help="Override the kernel backend for linear attention prefill/extend. If not set, uses --linear-attn-backend; compatible SM100 GDN models may automatically select FlashInfer.",
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES,
        ),
    ] = None
    linear_attn_verify_backend: A[
        Optional[str],
        Arg(
            help="Override the kernel backend for linear attention speculative target-verify. If not set, follows the decode backend (flashinfer decode -> flashinfer verify, otherwise triton). KDA supports triton, nv_cutedsl, and flashinfer verify backends.",
            choices=LINEAR_ATTN_KERNEL_BACKEND_CHOICES + ["nv_cutedsl"],
        ),
    ] = None
    # ReplaySSM buffered output-only linear-attn decode (GDN + KDA): per-slot
    # ring + periodic flush to cut per-step HBM state traffic.
    enable_linear_replayssm: A[
        bool,
        "Enable the ReplaySSM buffered output-only linear-attn decode kernel. "
        "Primarily a GDN (scalar-gate) decode-bandwidth optimization (~1.2-1.5x "
        "at batch >= 64). KDA uses its selected Triton or Helion implementation, "
        "but its per-K gate ring is larger and ReplaySSM is typically slower "
        "than packed KDA decode; benchmark before enabling it. Requires the "
        "Triton linear-attn decode backend, or Helion for KDA, and "
        "--mamba-radix-cache-strategy no_buffer (the default).",
    ] = False
    linear_replayssm_cache_len: A[
        int,
        "Ring-buffer length L for ReplaySSM linear-attn decode. The full recurrent state is flushed to HBM every L decode steps.",
    ] = 16
    # ReplaySSM spec-verify (Part B of RFC #28511): linear-attn target-verify via
    # fold-every-commit instead of per-draft full-state snapshots -- the verify
    # stores each draft step's raw inputs into a per-slot window and the commit
    # replays the accepted prefix into the fp32 checkpoint. GDN sizes the window
    # to the draft maximum; KDA folds a (raw v, pre-norm k, gate, beta) ring of
    # length --linear-replayssm-cache-len. Linear-chain (topk <= 1) only.
    enable_linear_replayssm_spec: A[
        bool,
        "Enable the ReplaySSM spec-verify: fold-every-commit -- a per-slot raw-input window replaces the recurrent verify's per-draft full-state snapshots. GDN or KDA hybrid linear-attn models, linear-chain (--speculative-eagle-topk in {None, 1}) only.",
    ] = False


@dataclasses.dataclass
class ExecGraph:
    """Namespace ``exec.graph``."""

    _NS_PATH = "exec.graph"

    # -------------------------------------------------------------------------
    # Cuda graphs
    # -------------------------------------------------------------------------
    cuda_graph_config: A[
        Optional[CudaGraphConfig],
        Arg(
            help='Per-phase CUDA graph settings as JSON, e.g. \'{"decode":{"backend":"full","max_bs":256},"prefill":{"backend":"tc_piecewise","tc_compiler":"eager"}}\'. Allowed backends per phase: full, breakable, tc_piecewise, disabled (full is decode-only). JSON wins over the per-phase --cuda-graph-* convenience flags and over legacy flags.',
            type_parser=parse_cuda_graph_config_arg,
        ),
    ] = None
    cuda_graph_backend_decode: A[
        Optional[Literal["full", "breakable", "tc_piecewise", "disabled"]],
        Arg(
            help="Backend for the decode phase. Folds into cuda_graph_config[decode].backend.",
            choices=Backend.ALL,
        ),
    ] = None
    cuda_graph_backend_prefill: A[
        Optional[Literal["full", "breakable", "tc_piecewise", "disabled"]],
        Arg(
            help="Backend for the prefill phase. Folds into cuda_graph_config[prefill].backend.",
            choices=Backend.ALL,
        ),
    ] = None
    cuda_graph_max_bs_decode: A[
        Optional[int], "Maximum batch size captured for the decode cuda graph."
    ] = None
    cuda_graph_max_bs_prefill: A[
        Optional[int], "Maximum batch size captured for the prefill cuda graph."
    ] = None
    cuda_graph_bs_decode: A[
        Optional[List[int]],
        "Explicit list of batch sizes to capture for the decode cuda graph.",
    ] = None
    cuda_graph_bs_prefill: A[
        Optional[List[int]],
        "Explicit list of batch sizes to capture for the prefill cuda graph.",
    ] = None
    cuda_graph_tc_compiler: A[
        Optional[Literal["eager", "inductor"]],
        "Compiler used by the tc_piecewise backend (currently only the prefill phase consumes it).",
    ] = None
    disable_prefill_cuda_graph: A[
        bool,
        "Disable the prefill-phase CUDA graph. Convenience for --cuda-graph-backend-prefill=disabled.",
    ] = False
    disable_decode_cuda_graph: A[
        bool,
        "Disable the decode-phase CUDA graph. Convenience for --cuda-graph-backend-decode=disabled.",
    ] = False
    disable_cuda_graph: A[bool, Arg(no_cli=True)] = False
    disable_cuda_graph_padding: A[
        bool,
        "Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
    ] = False
    enable_profile_cuda_graph: A[
        bool,
        "Enable profiling of cuda graph capture.",
    ] = False
    enable_cudagraph_gc: A[
        bool,
        "Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process.",
    ] = False
    debug_cuda_graph: A[
        bool,
        "Enable debug/eager mode for CUDA graph using breakable CUDA graph. When enabled, graph breaks are inserted so every operation runs eagerly while still going through the CUDA graph capture / replay path. Useful for debugging CUDA graph capture / replay issues.",
    ] = False

    # -------------------------------------------------------------------------
    # Torch compile
    # -------------------------------------------------------------------------
    enable_torch_compile: A[
        bool, "Optimize the model with torch.compile. Experimental feature."
    ] = False
    enable_torch_compile_debug_mode: A[
        bool,
        "Enable debug mode for torch compile",
    ] = False
    torch_compile_max_bs: A[
        int,
        "Set the maximum batch size when using torch compile.",
    ] = 32


@dataclasses.dataclass
class ExecComm:
    """Namespace ``exec.comm``."""

    _NS_PATH = "exec.comm"

    # -------------------------------------------------------------------------
    # Communication and kernels
    # -------------------------------------------------------------------------
    enable_layerwise_nvtx_marker: A[
        bool, "Enable layerwise NVTX profiling annotations for the model."
    ] = False
    enable_nccl_nvls: A[
        bool, "Enable NCCL NVLS for prefill heavy requests when available."
    ] = False
    enable_symm_mem: A[
        bool,
        Arg(
            help="Enable NCCL symmetric memory for fast collectives.",
            resolvable=True,
        ),
    ] = False
    disable_custom_all_reduce: A[
        bool,
        Arg(
            help="Disable the custom all-reduce kernel and fall back to NCCL.",
            resolvable=True,
        ),
    ] = False
    enable_mscclpp: A[
        bool,
        "Enable using mscclpp for small messages for all-reduce kernel and fall back to NCCL.",
    ] = False
    enable_torch_symm_mem: A[
        bool,
        "Enable using torch symm mem for all-reduce kernel and fall back to NCCL. Only supports CUDA device SM90 and above. SM90 supports world size 4, 6, 8. SM100 supports world size 6, 8.",
    ] = False
    enable_scattered_sconv: A[
        bool,
        "Inkling: replace the attention/MLP output all-reduce with a hidden-dimension reduce-scatter, run the channelwise output short convolution on the [T, H/P] shard, then all-gather before the residual add. This shards the convolution cache across tensor-parallel ranks without changing communication volume.",
    ] = False
    pre_warm_nccl: A[
        bool,
        "Pre-warm NCCL/RCCL communicators during startup to reduce P99 TTFT cold-start latency. Default: enabled for AMD/HIP (RCCL), disabled for NVIDIA/CUDA (NCCL).",
    ] = False
    enable_quant_communications: A[
        Optional[bool],
        "Enable INT8 quantization of TP communications (limited support).",
    ] = False
    enable_flashinfer_allreduce_fusion: A[bool, Arg(no_cli=True)] = False
    enforce_disable_flashinfer_allreduce_fusion: A[
        bool,
        "Enforce disable FlashInfer allreduce fusion.",
    ] = False
    flashinfer_allreduce_fusion_backend: A[
        Optional[Literal["auto", "trtllm", "mnnvl"]],
        Arg(
            help=(
                "Enable FlashInfer allreduce fusion and choose backend. "
                "Requires SM90 or SM10X NVIDIA GPUs. "
                "Defaults to auto. "
                "'auto': choose mnnvl on Blackwell (SM100/SM103) systems "
                "(single- and multi-node) and trtllm on SM90 single-node systems. "
                "'trtllm': available on single-node systems only. "
                "'mnnvl': available on SM90 single-node systems and SM100/SM103 "
                "single-node or multi-node systems via MNNVL fabric. "
                "Fuses allreduce with Residual + RMSNorm for supported MoE models."
            ),
            resolvable=True,
        ),
    ] = None
    enable_aiter_allreduce_fusion: A[
        bool, Arg(help="Enable Aiter AllReduce Fusion.", resolvable=True)
    ] = False


@dataclasses.dataclass
class ExecMoe:
    """Namespace ``exec.moe``."""

    _NS_PATH = "exec.moe"
    enable_fused_moe_sum_all_reduce: A[
        bool,
        "Enable fused moe triton and sum all reduce.",
    ] = False
    moe_a2a_backend: A[
        Literal[
            "none",
            "deepep",
            "mooncake",
            "nixl",
            "mori",
            "ascend_fuseep",
            "flashinfer",
            "megamoe",
            "deepep_v2",
            "ascend_tp",
            "pplx",
        ],
        Arg(
            help="Choose the backend for MoE A2A.",
            choices=[
                "none",
                "deepep",
                "mooncake",
                "nixl",
                "mori",
                "ascend_fuseep",
                "flashinfer",
                "megamoe",
                "deepep_v2",
                "pplx",
                "ascend_tp",
            ],
            resolvable=True,
        ),
    ] = "none"
    enable_w4a4_mxfp4_megamoe: A[
        bool,
        "Enable the W4A4 MXFP4 MegaMoE path with DeepGEMM's "
        "mxf4xmxf4 MMA type. Use with "
        "--moe-a2a-backend megamoe.",
    ] = False
    deepep_v2_mode: A[
        Literal["direct", "hybrid"],
        "DeepEP v2 ElasticBuffer communication topology, fixed at server init: "
        "`direct` (single-node NVLink) or `hybrid` (multi-node scale-out). "
        "Layout/grouped-GEMM and the decode CUDA graph are chosen per batch by "
        "inference phase, independent of this knob; not equivalent to DeepEP v1 "
        "normal/low_latency.",
    ] = "direct"
    moe_runner_backend: A[
        str,
        Arg(
            help="Choose the runner backend for MoE.",
            choices=MOE_RUNNER_BACKEND_CHOICES,
            resolvable=True,
        ),
    ] = "auto"
    flashinfer_mxfp4_moe_precision: A[
        Literal["default", "bf16", "fp8"],
        "Choose the computation precision of flashinfer mxfp4 moe. "
        "On SM90, `fp8` selects the Humming-style MXFP4-weight x FP8-activation "
        "path introduced by FlashInfer #3738 and requires FlashInfer >= 0.6.18.",
    ] = "default"
    deepep_mode: A[
        Literal["auto", "normal", "low_latency"],
        "Select the mode when enable DeepEP or MoriEP MoE, could be `normal`, `low_latency` or `auto`. Default is `auto`, which means `low_latency` for decode batch and `normal` for prefill batch.",
    ] = "auto"
    fuseep_mode: A[
        Literal[1, 2],
        "Select the mode when enable Ascend FuseEP MoE, 1 -> dispatch_gmm_combine_decode is executed；2 -> dispatch_ffn_combine is executed (support hybrid deployment when 2).",
    ] = 2
    deepep_dispatcher_output_dtype: A[
        Literal["auto", "bf16", "fp8", "int8", "nvfp4"],
        "Select DeepEP dispatcher output dtype",
    ] = "auto"
    ep_num_redundant_experts: A[
        int, "Allocate this number of redundant experts in expert parallel."
    ] = 0
    ep_dispatch_algorithm: A[
        Optional[Literal["static", "dynamic", "fake", "lp"]],
        "The algorithm to choose ranks for redundant experts in expert parallel.",
    ] = None
    init_expert_location: A[str, "Initial location of EP experts."] = "trivial"
    enable_eplb: A[bool, "Enable EPLB algorithm"] = False
    eplb_algorithm: A[str, "Chosen EPLB algorithm"] = "auto"
    eplb_rebalance_num_iterations: A[
        int, "Number of iterations to automatically trigger a EPLB re-balance."
    ] = 1000
    eplb_rebalance_layers_per_chunk: A[
        Optional[int],
        "Number of layers to rebalance per forward pass.",
    ] = None
    eplb_min_rebalancing_utilization_threshold: A[
        float,
        "Minimum threshold for GPU average utilization to trigger EPLB rebalancing. Must be in the range [0.0, 1.0].",
    ] = 1.0
    expert_distribution_recorder_mode: A[
        Optional[Literal["stat", "stat_approx", "per_pass", "per_token"]],
        "Mode of expert distribution recorder.",
    ] = None
    expert_distribution_recorder_buffer_size: A[
        Optional[int],
        "Circular buffer size of expert distribution recorder. Set to -1 to denote infinite buffer.",
    ] = None
    expert_balancedness_report_mode: A[
        Literal["off", "server_log", "prometheus", "both"],
        "Where to report expert balancedness. Options: off, server_log, prometheus, both.",
    ] = "off"
    deepep_config: A[
        Optional[str],
        "Tuned DeepEP config suitable for your own cluster. It can be either a string with JSON content or a file path.",
    ] = None
    elastic_ep_backend: A[
        Literal[None, "mooncake", "nixl"],
        Arg(
            help="Specify the collective communication backend for elastic EP. Supports 'mooncake' and 'nixl'.",
            choices=["none", "mooncake", "nixl"],
        ),
    ] = None
    enable_elastic_expert_backup: A[
        bool,
        "Enable elastic expert backup feature.",
    ] = False
    mooncake_ib_device: A[
        Optional[str],
        "The InfiniBand devices for Mooncake Backend transfer, accepts multiple comma-separated devices (e.g., --mooncake-ib-device mlx5_0,mlx5_1). Default is None, which triggers automatic device detection when Mooncake Backend is enabled.",
    ] = None
    enable_waterfill: A[
        bool,
        "Enable Waterfill: dispatch the fused shared expert as an extra routed expert slot to the least-loaded EP rank. Supports DeepEP and MegaMOE MoE A2A backends, implicitly enables shared-expert fusion, and supports --deepep-mode auto, normal, or low_latency when used with DeepEP. Use auto or low_latency for production DeepEP decode so CUDA graph remains enabled. Supported on DeepSeek-V3/R1 with EP >= 2.",
    ] = False
    ep_join_mode: A[
        Optional[Literal["scale", "recover"]],
        Arg(
            help="Join mode for elastic EP. 'recover' rejoins an existing slot after a fault. 'scale' joins as a new rank beyond the original group size and requires --node-rank 1.",
            cli_name="--elastic-ep-join-mode",
            choices=["scale", "recover"],
        ),
    ] = None
    elastic_ep_scale_timeout: A[
        float, "Timeout in seconds for a pending elastic EP scale operation."
    ] = 600
    elastic_ep_rejoin: A[
        bool,
        "[Deprecated] Alias for --elastic-ep-join-mode recover.",
    ] = False
    disable_flashinfer_cutlass_moe_fp4_allgather: A[
        bool, "Disables quantize before all-gather for flashinfer cutlass moe."
    ] = False
    disable_shared_experts_fusion: A[
        bool,
        Arg(
            help="Disable the built-in shared experts fusion optimization for DeepSeek V3/R1. Note: Waterfill (--enable-waterfill) routes the shared expert as an extra MoE slot, so the shared expert is not separated from the MoE path when Waterfill is enabled.",
            resolvable=True,
        ),
    ] = False
    enforce_shared_experts_fusion: A[
        bool,
        "Enforce shared experts fusion even when it would normally be disabled (e.g. under DeepEP). Mutually exclusive with --disable-shared-experts-fusion.",
    ] = False

    # -------------------------------------------------------------------------
    # Ktransformers/AMX expert parallelism
    # -------------------------------------------------------------------------
    kt_weight_path: A[
        Optional[str],
        "[ktransformers parameter] The path of the quantized expert weights for amx kernel. A local folder.",
    ] = None
    kt_method: A[
        str, "[ktransformers parameter] Quantization formats for CPU execution."
    ] = "AMXINT4"
    kt_cpuinfer: A[
        Optional[int], "[ktransformers parameter] The number of CPUInfer threads."
    ] = None
    kt_threadpool_count: A[
        int,
        "[ktransformers parameter] One-to-one with the number of NUMA nodes (one thread pool per NUMA).",
    ] = 2
    kt_num_gpu_experts: A[
        Optional[int], "[ktransformers parameter] The number of GPU experts."
    ] = None
    kt_max_deferred_experts_per_token: A[
        Optional[int],
        "[ktransformers parameter] Maximum number of experts deferred to CPU per token. All MoE layers except the final one use this value; the final layer always uses 0.",
    ] = None


@dataclasses.dataclass
class ExecOverlap:
    """Namespace ``exec.overlap``."""

    _NS_PATH = "exec.overlap"

    # -------------------------------------------------------------------------
    # Two batch overlap
    # -------------------------------------------------------------------------
    enable_two_batch_overlap: A[
        bool,
        "Enabling two micro batches to overlap.",
    ] = False
    enable_single_batch_overlap: A[
        bool, "Let computation and communication overlap within one micro batch."
    ] = False
    tbo_token_distribution_threshold: A[
        float,
        "The threshold of token distribution between two batches in micro-batch-overlap, determines whether to two-batch-overlap or two-chunk-overlap. Set to 0 denote disable two-chunk-overlap.",
    ] = 0.48


@dataclasses.dataclass
class ExecOffload:
    """Namespace ``exec.offload``."""

    _NS_PATH = "exec.offload"

    # -------------------------------------------------------------------------
    # Offloading
    # -------------------------------------------------------------------------
    cpu_offload_gb: A[
        int,
        "How many GBs of RAM to reserve for CPU offloading.",
    ] = 0
    offload_group_size: A[
        int,
        "Number of layers per group in offloading.",
    ] = -1
    offload_num_in_group: A[
        int,
        "Number of layers to be offloaded within a group.",
    ] = 1
    offload_prefetch_step: A[
        int,
        "Steps to prefetch in offloading.",
    ] = 1
    offload_mode: A[str, "Mode of offloading."] = "cpu"


@dataclasses.dataclass
class ExecDllm:
    """Namespace ``exec.dllm``."""

    _NS_PATH = "exec.dllm"

    # -------------------------------------------------------------------------
    # Diffusion LLM
    # -------------------------------------------------------------------------
    dllm_algorithm: A[
        Optional[str], "The diffusion LLM algorithm, such as LowConfidence."
    ] = None
    dllm_algorithm_config: A[
        Optional[str],
        "The diffusion LLM algorithm configurations. Must be a YAML file.",
    ] = None
    dllm_fdfo: A[
        bool,
        Arg(
            help="Enable First-Done-First-Out (FDFO) scheduling for diffusion LLM inference. Enabled by default; use --no-dllm-fdfo to fall back to synchronous block scheduling.",
            action=argparse.BooleanOptionalAction,
        ),
    ] = True


@dataclasses.dataclass
class ExecDeterministic:
    """Namespace ``exec.deterministic``."""

    _NS_PATH = "exec.deterministic"

    # -------------------------------------------------------------------------
    # Deterministic inference
    # -------------------------------------------------------------------------
    enable_deterministic_inference: A[
        bool, "Enable deterministic inference mode with batch invariant ops."
    ] = False
    rl_on_policy_target: A[
        Optional[str],
        Arg(
            help="The training system that SGLang needs to match for true on-policy.",
            choices=RL_ON_POLICY_TARGET_CHOICES,
        ),
    ] = None
