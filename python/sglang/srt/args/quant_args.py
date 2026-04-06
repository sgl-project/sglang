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
"""CLI argument definitions for quantization and data type."""

import argparse


class QuantArgs:
    """CLI argument definitions for quantization and data type."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.args.constants import QUANTIZATION_CHOICES
        from sglang.srt.utils.common import nullable_str

        # Quantization and data type
        parser.add_argument(
            "--dtype",
            type=str,
            default=ServerArgs.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="Data type for model weights and activations.\n\n"
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            "BF16 precision for BF16 models.\n"
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.',
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=ServerArgs.quantization,
            choices=QUANTIZATION_CHOICES,
            help="The quantization method.",
        )
        parser.add_argument(
            "--quantization-param-path",
            type=nullable_str,
            default=None,
            help="Path to the JSON file containing the KV cache "
            "scaling factors. This should generally be supplied, when "
            "KV cache dtype is FP8. Otherwise, KV cache scaling factors "
            "default to 1.0, which may cause accuracy issues. ",
        )
        parser.add_argument(
            "--kv-cache-dtype",
            type=str,
            default=ServerArgs.kv_cache_dtype,
            choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1"],
            help='Data type for kv cache storage. "auto" will use model data type. "bf16" or "bfloat16" for BF16 KV cache. "fp8_e5m2" and "fp8_e4m3" are supported for CUDA 11.8+. "fp4_e2m1" (only mxfp4) is supported for CUDA 12.8+ and PyTorch 2.8.0+',
        )
        parser.add_argument(
            "--enable-fp32-lm-head",
            action="store_true",
            help="If set, the LM head outputs (logits) are in FP32.",
        )
        parser.add_argument(
            "--modelopt-quant",
            type=str,
            default=ServerArgs.modelopt_quant,
            help="The ModelOpt quantization configuration. "
            "Supported values: 'fp8', 'int4_awq', 'w4a8_awq', 'nvfp4', 'nvfp4_awq'. "
            "This requires the NVIDIA Model Optimizer library to be installed: pip install nvidia-modelopt",
        )
        parser.add_argument(
            "--modelopt-checkpoint-restore-path",
            type=str,
            default=ServerArgs.modelopt_checkpoint_restore_path,
            help="Path to restore a previously saved ModelOpt quantized checkpoint. "
            "If provided, the quantization process will be skipped and the model "
            "will be loaded from this checkpoint.",
        )
        parser.add_argument(
            "--modelopt-checkpoint-save-path",
            type=str,
            default=ServerArgs.modelopt_checkpoint_save_path,
            help="Path to save the ModelOpt quantized checkpoint after quantization. "
            "This allows reusing the quantized model in future runs.",
        )
        parser.add_argument(
            "--modelopt-export-path",
            type=str,
            default=ServerArgs.modelopt_export_path,
            help="Path to export the quantized model in HuggingFace format after ModelOpt quantization. "
            "The exported model can then be used directly with SGLang for inference. "
            "If not provided, the model will not be exported.",
        )
        parser.add_argument(
            "--quantize-and-serve",
            action="store_true",
            default=ServerArgs.quantize_and_serve,
            help="Quantize the model with ModelOpt and immediately serve it without exporting. "
            "This is useful for development and prototyping. For production, it's recommended "
            "to use separate quantization and deployment steps.",
        )
        parser.add_argument(
            "--rl-quant-profile",
            type=str,
            default=ServerArgs.rl_quant_profile,
            help="Path to the FlashRL quantization profile. Required when using --load-format flash_rl.",
        )
