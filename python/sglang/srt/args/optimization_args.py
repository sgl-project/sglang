# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CLI argument definitions for optimization and debug options."""

import argparse


class OptimizationArgs:
    """CLI argument definitions for optimization/debug options, dynamic batch tokenizer, and debug tensor dumps."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.args.actions import DeprecatedAction
        from sglang.srt.args.constants import RL_ON_POLICY_TARGET_CHOICES

        # Optimization/debug options
        parser.add_argument(
            "--disable-radix-cache",
            action="store_true",
            help="Disable RadixAttention for prefix caching.",
        )
        parser.add_argument(
            "--cuda-graph-max-bs",
            type=int,
            default=ServerArgs.cuda_graph_max_bs,
            help="Set the maximum batch size for cuda graph. It will extend the cuda graph capture batch size to this value.",
        )
        parser.add_argument(
            "--cuda-graph-bs",
            type=int,
            nargs="+",
            help="Set the list of batch sizes for cuda graph.",
        )
        parser.add_argument(
            "--disable-cuda-graph",
            action="store_true",
            help="Disable cuda graph.",
        )
        parser.add_argument(
            "--disable-cuda-graph-padding",
            action="store_true",
            help="Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
        )
        parser.add_argument(
            "--enable-profile-cuda-graph",
            action="store_true",
            help="Enable profiling of cuda graph capture.",
        )
        parser.add_argument(
            "--enable-cudagraph-gc",
            action="store_true",
            help="Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process.",
        )
        parser.add_argument(
            "--enable-layerwise-nvtx-marker",
            action="store_true",
            help="Enable layerwise NVTX profiling annotations for the model.",
        )
        parser.add_argument(
            "--enable-nccl-nvls",
            action="store_true",
            help="Enable NCCL NVLS for prefill heavy requests when available.",
        )
        parser.add_argument(
            "--enable-symm-mem",
            action="store_true",
            help="Enable NCCL symmetric memory for fast collectives.",
        )
        parser.add_argument(
            "--disable-flashinfer-cutlass-moe-fp4-allgather",
            action="store_true",
            help="Disables quantize before all-gather for flashinfer cutlass moe.",
        )
        parser.add_argument(
            "--enable-tokenizer-batch-encode",
            action="store_true",
            help="Enable batch tokenization for improved performance when processing multiple text inputs. Do not use with image inputs, pre-tokenized input_ids, or input_embeds.",
        )
        parser.add_argument(
            "--disable-tokenizer-batch-decode",
            action="store_true",
            help="Disable batch decoding when decoding multiple completions.",
        )
        parser.add_argument(
            "--disable-outlines-disk-cache",
            action="store_true",
            help="Disable disk cache of outlines to avoid possible crashes related to file system or high concurrency.",
        )
        parser.add_argument(
            "--disable-custom-all-reduce",
            action="store_true",
            help="Disable the custom all-reduce kernel and fall back to NCCL.",
        )
        parser.add_argument(
            "--enable-mscclpp",
            action="store_true",
            help="Enable using mscclpp for small messages for all-reduce kernel and fall back to NCCL.",
        )
        parser.add_argument(
            "--enable-torch-symm-mem",
            action="store_true",
            help="Enable using torch symm mem for all-reduce kernel and fall back to NCCL. Only supports CUDA device SM90 and above. SM90 supports world size 4, 6, 8. SM100 supports world size 6, 8.",
        )
        parser.add_argument(
            "--pre-warm-nccl",
            action="store_true",
            help="Pre-warm NCCL/RCCL communicators during startup to reduce P99 TTFT cold-start latency. Default: enabled for AMD/HIP (RCCL), disabled for NVIDIA/CUDA (NCCL).",
        )
        parser.add_argument(
            "--disable-overlap-schedule",
            action="store_true",
            help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
        )
        parser.add_argument(
            "--enable-mixed-chunk",
            action="store_true",
            help="Enabling mixing prefill and decode in a batch when using chunked prefill.",
        )
        parser.add_argument(
            "--enable-dp-attention",
            action="store_true",
            help="Enabling data parallelism for attention and tensor parallelism for FFN. The dp size should be equal to the tp size. Currently DeepSeek-V2 and Qwen 2/3 MoE models are supported.",
        )
        parser.add_argument(
            "--enable-dp-lm-head",
            action="store_true",
            help="Enable vocabulary parallel across the attention TP group to avoid all-gather across DP groups, optimizing performance under DP attention.",
        )
        parser.add_argument(
            "--enable-two-batch-overlap",
            action="store_true",
            help="Enabling two micro batches to overlap.",
        )
        parser.add_argument(
            "--enable-single-batch-overlap",
            action="store_true",
            help="Let computation and communication overlap within one micro batch.",
        )
        parser.add_argument(
            "--tbo-token-distribution-threshold",
            type=float,
            default=ServerArgs.tbo_token_distribution_threshold,
            help="The threshold of token distribution between two batches in micro-batch-overlap, determines whether to two-batch-overlap or two-chunk-overlap. Set to 0 denote disable two-chunk-overlap.",
        )
        parser.add_argument(
            "--enable-torch-compile",
            action="store_true",
            help="Optimize the model with torch.compile. Experimental feature.",
        )
        parser.add_argument(
            "--enable-torch-compile-debug-mode",
            action="store_true",
            help="Enable debug mode for torch compile",
        )
        parser.add_argument(
            "--disable-piecewise-cuda-graph",
            action="store_true",
            help="Disable piecewise cuda graph for extend/prefill.",
        )
        parser.add_argument(
            "--enable-piecewise-cuda-graph",
            action=DeprecatedAction,
            help="Deprecated: Piecewise cuda graph is enabled by default. Use --enforce-piecewise-cuda-graph to skip auto-disable conditions.",
        )
        parser.add_argument(
            "--enforce-piecewise-cuda-graph",
            action="store_true",
            help="Enforce piecewise cuda graph, skipping all auto-disable conditions. Used for testing.",
        )
        parser.add_argument(
            "--piecewise-cuda-graph-tokens",
            type=int,
            nargs="+",
            help="Set the list of token lengths for piecewise cuda graph capture.",
        )
        parser.add_argument(
            "--piecewise-cuda-graph-compiler",
            type=str,
            default=ServerArgs.piecewise_cuda_graph_compiler,
            help="Set the compiler for piecewise cuda graph. Choices are: eager, inductor.",
            choices=["eager", "inductor"],
        )
        parser.add_argument(
            "--torch-compile-max-bs",
            type=int,
            default=ServerArgs.torch_compile_max_bs,
            help="Set the maximum batch size when using torch compile.",
        )
        parser.add_argument(
            "--piecewise-cuda-graph-max-tokens",
            type=int,
            default=ServerArgs.piecewise_cuda_graph_max_tokens,
            help="Set the maximum tokens when using piecewise cuda graph.",
        )
        parser.add_argument(
            "--torchao-config",
            type=str,
            default=ServerArgs.torchao_config,
            help="Optimize the model with torchao. Experimental feature. Current choices are: int8dq, int8wo, int4wo-<group_size>, fp8wo, fp8dq-per_tensor, fp8dq-per_row",
        )
        parser.add_argument(
            "--enable-nan-detection",
            action="store_true",
            help="[Deprecated] Use SGLANG_SPEC_NAN_DETECTION=1 and SGLANG_SPEC_OOB_DETECTION=1 instead.",
        )
        parser.add_argument(
            "--enable-p2p-check",
            action="store_true",
            help="Enable P2P check for GPU access, otherwise the p2p access is allowed by default.",
        )
        parser.add_argument(
            "--triton-attention-reduce-in-fp32",
            action="store_true",
            help="Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16."
            "This only affects Triton attention kernels.",
        )
        parser.add_argument(
            "--triton-attention-num-kv-splits",
            type=int,
            default=ServerArgs.triton_attention_num_kv_splits,
            help="The number of KV splits in flash decoding Triton kernel. Larger value is better in longer context scenarios. The default value is 8.",
        )
        parser.add_argument(
            "--triton-attention-split-tile-size",
            type=int,
            default=ServerArgs.triton_attention_split_tile_size,
            help="The size of split KV tile in flash decoding Triton kernel. Used for deterministic inference.",
        )
        parser.add_argument(
            "--num-continuous-decode-steps",
            type=int,
            default=ServerArgs.num_continuous_decode_steps,
            help="Run multiple continuous decoding steps to reduce scheduling overhead. "
            "This can potentially increase throughput but may also increase time-to-first-token latency. "
            "The default value is 1, meaning only run one decoding step at a time.",
        )
        parser.add_argument(
            "--delete-ckpt-after-loading",
            action="store_true",
            help="Delete the model checkpoint after loading the model.",
        )
        parser.add_argument(
            "--enable-memory-saver",
            action="store_true",
            help="Allow saving memory using release_memory_occupation and resume_memory_occupation",
        )
        parser.add_argument(
            "--enable-weights-cpu-backup",
            action="store_true",
            help="Save model weights (both main model and draft model, if any) to CPU memory during release_weights_occupation and resume_weights_occupation",
        )
        parser.add_argument(
            "--enable-draft-weights-cpu-backup",
            action="store_true",
            help="Save draft model weights to CPU memory during release_weights_occupation and resume_weights_occupation",
        )
        parser.add_argument(
            "--allow-auto-truncate",
            action="store_true",
            help="Allow automatically truncating requests that exceed the maximum input length instead of returning an error.",
        )
        parser.add_argument(
            "--enable-custom-logit-processor",
            action="store_true",
            help="Enable users to pass custom logit processors to the server (disabled by default for security)",
        )
        parser.add_argument(
            "--flashinfer-mla-disable-ragged",
            action="store_true",
            help="Not using ragged prefill wrapper when running flashinfer mla",
        )
        parser.add_argument(
            "--disable-shared-experts-fusion",
            action="store_true",
            help="Disable shared experts fusion optimization for deepseek v3/r1.",
        )
        parser.add_argument(
            "--disable-chunked-prefix-cache",
            action="store_true",
            help="Disable chunked prefix cache feature for deepseek, which should save overhead for short sequences.",
        )
        parser.add_argument(
            "--disable-fast-image-processor",
            action="store_true",
            help="Adopt base image processor instead of fast image processor.",
        )
        parser.add_argument(
            "--keep-mm-feature-on-device",
            action="store_true",
            help="Keep multimodal feature tensors on device after processing to save D2H copy.",
        )
        parser.add_argument(
            "--enable-return-hidden-states",
            action="store_true",
            help="Enable returning hidden states with responses.",
        )
        parser.add_argument(
            "--enable-return-routed-experts",
            action="store_true",
            help="Enable returning routed experts of each layer with responses.",
        )
        parser.add_argument(
            "--scheduler-recv-interval",
            type=int,
            default=ServerArgs.scheduler_recv_interval,
            help="The interval to poll requests in scheduler. Can be set to >1 to reduce the overhead of this.",
        )
        parser.add_argument(
            "--numa-node",
            type=int,
            nargs="+",
            help="Sets the numa node for the subprocesses. i-th element corresponds to i-th subprocess. If unset, will be automatically detected on NUMA systems.",
        )
        parser.add_argument(
            "--enable-deterministic-inference",
            action="store_true",
            help="Enable deterministic inference mode with batch invariant ops.",
        )
        parser.add_argument(
            "--rl-on-policy-target",
            type=str,
            default=ServerArgs.rl_on_policy_target,
            choices=RL_ON_POLICY_TARGET_CHOICES,
            help="The training system that SGLang needs to match for true on-policy.",
        )
        parser.add_argument(
            "--enable-attn-tp-input-scattered",
            action="store_true",
            help="Allow input of attention to be scattered when only using tensor parallelism, to reduce the computational load of operations such as qkv latent.",
        )
        parser.add_argument(
            "--enable-fused-qk-norm-rope",
            action="store_true",
            help="Enable fused qk normalization and rope rotary embedding.",
        )
        parser.add_argument(
            "--enable-precise-embedding-interpolation",
            action="store_true",
            help="Enable corner alignment for resize of embeddings grid to ensure more accurate(but slower) evaluation of interpolated embedding values.",
        )
        parser.add_argument(
            "--enable-fused-moe-sum-all-reduce",
            action="store_true",
            help="Enable fused moe triton and sum all reduce.",
        )
        parser.add_argument(
            "--gc-threshold",
            type=int,
            nargs="+",
            help="Set the garbage collection thresholds (the collection frequency). Accepts 1 to 3 integers.",
        )

        # Dynamic batch tokenizer
        parser.add_argument(
            "--enable-dynamic-batch-tokenizer",
            action="store_true",
            help="Enable async dynamic batch tokenizer for improved performance when multiple requests arrive concurrently.",
        )
        parser.add_argument(
            "--dynamic-batch-tokenizer-batch-size",
            type=int,
            default=ServerArgs.dynamic_batch_tokenizer_batch_size,
            help="[Only used if --enable-dynamic-batch-tokenizer is set] Maximum batch size for dynamic batch tokenizer.",
        )
        parser.add_argument(
            "--dynamic-batch-tokenizer-batch-timeout",
            type=float,
            default=ServerArgs.dynamic_batch_tokenizer_batch_timeout,
            help="[Only used if --enable-dynamic-batch-tokenizer is set] Timeout in seconds for batching tokenization requests.",
        )

        # Debug tensor dumps
        parser.add_argument(
            "--debug-tensor-dump-output-folder",
            type=str,
            default=ServerArgs.debug_tensor_dump_output_folder,
            help="The output folder for dumping tensors.",
        )
        parser.add_argument(
            "--debug-tensor-dump-layers",
            type=int,
            nargs="+",
            help="The layer ids to dump. Dump all layers if not specified.",
        )
        parser.add_argument(
            "--debug-tensor-dump-input-file",
            type=str,
            default=ServerArgs.debug_tensor_dump_input_file,
            help="The input filename for dumping tensors",
        )
        parser.add_argument(
            "--debug-tensor-dump-inject",
            type=str,
            default=ServerArgs.debug_tensor_dump_inject,
            help="Inject the outputs from jax as the input of every layer.",
        )
