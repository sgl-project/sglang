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
"""CLI argument definitions for kernel backend."""

import argparse


class BackendArgs:
    """CLI argument definitions for kernel backend."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.args.constants import (
            ATTENTION_BACKEND_CHOICES,
            SAMPLING_BACKEND_CHOICES,
            GRAMMAR_BACKEND_CHOICES,
            NSA_CHOICES,
            FP8_GEMM_RUNNER_BACKEND_CHOICES,
            FP4_GEMM_RUNNER_BACKEND_CHOICES,
        )

        # Kernel backend
        parser.add_argument(
            "--attention-backend",
            type=str,
            choices=ATTENTION_BACKEND_CHOICES,
            default=ServerArgs.attention_backend,
            help="Choose the kernels for attention layers.",
        )
        parser.add_argument(
            "--prefill-attention-backend",
            type=str,
            choices=ATTENTION_BACKEND_CHOICES,
            default=ServerArgs.prefill_attention_backend,
            help="Choose the kernels for prefill attention layers (have priority over --attention-backend).",
        )
        parser.add_argument(
            "--decode-attention-backend",
            type=str,
            choices=ATTENTION_BACKEND_CHOICES,
            default=ServerArgs.decode_attention_backend,
            help="Choose the kernels for decode attention layers (have priority over --attention-backend).",
        )
        parser.add_argument(
            "--sampling-backend",
            type=str,
            choices=SAMPLING_BACKEND_CHOICES,
            default=ServerArgs.sampling_backend,
            help="Choose the kernels for sampling layers.",
        )
        parser.add_argument(
            "--grammar-backend",
            type=str,
            choices=GRAMMAR_BACKEND_CHOICES,
            default=ServerArgs.grammar_backend,
            help="Choose the backend for grammar-guided decoding.",
        )
        parser.add_argument(
            "--mm-attention-backend",
            type=str,
            choices=[
                "sdpa",
                "fa3",
                "fa4",
                "triton_attn",
                "ascend_attn",
                "aiter_attn",
                "flashinfer_cudnn",
            ],
            default=ServerArgs.mm_attention_backend,
            help="Set multimodal attention backend.",
        )
        parser.add_argument(
            "--nsa-prefill-backend",
            default=ServerArgs.nsa_prefill_backend,
            type=str,
            choices=NSA_CHOICES,
            help="NSA prefill backend. If not specified, auto-detects based on hardware and kv_cache_dtype.",
        )
        parser.add_argument(
            "--nsa-decode-backend",
            default=ServerArgs.nsa_decode_backend,
            type=str,
            choices=NSA_CHOICES,
            help="NSA decode backend. If not specified, auto-detects based on hardware and kv_cache_dtype.",
        )
        parser.add_argument(
            "--fp8-gemm-backend",
            type=str,
            choices=FP8_GEMM_RUNNER_BACKEND_CHOICES,
            default=ServerArgs.fp8_gemm_runner_backend,
            dest="fp8_gemm_runner_backend",
            help="Choose the runner backend for Blockwise FP8 GEMM operations. "
            "Options: 'auto' (default, auto-selects based on hardware), "
            "'deep_gemm' (JIT-compiled; enabled by default on NVIDIA Hopper (SM90) and Blackwell (SM100) when DeepGEMM is installed), "
            "'flashinfer_trtllm' (optimal for Blackwell and low-latency), "
            "'flashinfer_cutlass' (FlashInfer CUTLASS groupwise FP8 GEMM), "
            "'flashinfer_deepgemm' (Hopper SM90 only; uses swapAB optimization for small M dimensions in decoding), "
            "'cutlass' (optimal for Hopper/Blackwell GPUs and high-throughput), "
            "'triton' (fallback, widely compatible), "
            "'aiter' (ROCm only). ",
        )
        parser.add_argument(
            "--fp4-gemm-backend",
            type=str,
            choices=FP4_GEMM_RUNNER_BACKEND_CHOICES,
            default=ServerArgs.fp4_gemm_runner_backend,
            dest="fp4_gemm_runner_backend",
            help="Choose the runner backend for NVFP4 GEMM operations. "
            "Options: 'auto' (default; selects flashinfer_cudnn on SM120, flashinfer_cutlass otherwise), "
            "'cutlass' (SGLang CUTLASS kernel), "
            "'flashinfer_cutlass' (FlashInfer CUTLASS backend), "
            "'flashinfer_cudnn' (FlashInfer cuDNN backend, optimal on CUDA 13+ with cuDNN 9.15+), "
            "'flashinfer_trtllm' (FlashInfer TensorRT-LLM backend, requires different weight preparation with shuffling). ",
        )
        parser.add_argument(
            "--disable-flashinfer-autotune",
            default=ServerArgs.disable_flashinfer_autotune,
            action="store_true",
            help="Disable FlashInfer autotuning.",
        )
