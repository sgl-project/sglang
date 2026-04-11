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
"""Model-architecture-specific argument handlers for ServerArgs.

All methods in this mixin are mixed into ``ServerArgs`` via inheritance.
They should never be instantiated directly.
"""

from __future__ import annotations

import logging

from sglang.srt.environ import envs
from sglang.srt.utils.common import (
    get_bool_env_var,
    get_device_sm,
    get_nvidia_driver_version,
    get_quantization_config,
    is_blackwell_supported,
    is_cuda,
    is_hip,
    is_npu,
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
    is_triton_kernels_available,
)

logger = logging.getLogger(__name__)


class ServerArgsModelHandlersMixin:
    """Mixin that provides per-model-architecture init handlers for ServerArgs.

    Methods follow the signature ``_handle_<family>_family(self, hf_config, model_arch)``.
    Two registry helpers (``_get_model_arch_handlers`` and
    ``_get_model_arch_substring_handlers``) map architecture strings to the
    correct handler, and ``_handle_model_specific_adjustments`` acts as the
    single entry-point that dispatches to the right handler.
    """

    # ------------------------------------------------------------------
    # Per-family handlers
    # ------------------------------------------------------------------

    def _handle_mistral_large_family(self, hf_config, model_arch):
        self.dtype = "bfloat16"
        self._handle_deepseek_family(hf_config, model_arch)

    def _handle_deepseek_family(self, hf_config, model_arch):
        from sglang.srt.configs.model_config import is_deepseek_nsa

        # Set attention backend for DeepSeek
        if is_deepseek_nsa(hf_config):  # DeepSeek 3.2/GLM 5
            if model_arch == "GlmMoeDsaForCausalLM" and is_blackwell_supported():
                envs.SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.set(0)
                logger.warning(
                    "Force NSA prefill to use sparse MLA (i.e. disable MHA_ONE_SHOT) for GlmMoeDsaForCausalLM on Blackwell."
                )
            else:
                if envs.SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.is_set():
                    logger.warning(
                        f"Dense attention kv len threshold is manually set to {envs.SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()} for DSA. Caution: This may cause performance regression if the threshold is larger than the index topk of model."
                    )
                else:
                    # When threshold is not manually set, set it to the index topk of model
                    from sglang.srt.configs.model_config import get_nsa_index_topk

                    envs.SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.set(
                        get_nsa_index_topk(hf_config)
                    )
                    logger.warning(
                        f"Set dense attention kv len threshold to model index_topk={envs.SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.get()} for DeepSeek with DSA."
                    )
            if self.is_attention_backend_not_set():
                self.attention_backend = "nsa"
                logger.info("Use nsa attention backend for DeepSeek with DSA.")

            if not is_npu():  # CUDA or ROCm GPU
                if self.enable_nsa_prefill_context_parallel:
                    logger.warning(
                        "Context parallel feature is still under experiment. It has only been verified on Hopper platform."
                    )
                    if self.nsa_prefill_cp_mode == "in-seq-split":
                        # TODO Supports moe_dense_tp_size != 1, kv cache dtype = "fp8",moe_a2a_backend non-deepep and cross-machine operation .
                        self.enable_dp_attention = True
                        self.moe_dense_tp_size = 1
                        self.moe_a2a_backend = "deepep"
                        self.ep_size = self.tp_size
                        logger.warning(
                            "For in-seq split mode, we have the following restrictions: moe_dense_tp_size == 1, moe_a2a_backend == deepep, ep_size == tp_size, batch_size == 1"
                        )
                    else:
                        self.enable_dp_attention = True
                        self.moe_dense_tp_size = 1
                        assert (
                            self.dp_size == 1
                        ), "For round-robin split mode, dp attention is not supported."
                    assert (
                        self.tp_size == 8
                    ), "Current multi-machine CP support suffers from precision issues. So context parallel only support Single machine(tp_size == 8)"
                    self.attn_cp_size = self.tp_size // self.dp_size

                    logger.warning(
                        f"Enable Context Parallel opt for deeeseekv3.2-DSA, Setting dp_size == {self.dp_size} and moe_dense_tp_size == {self.moe_dense_tp_size}, ep_size == {self.ep_size}, tp_size == {self.tp_size}, kv_cache_dtype == {self.kv_cache_dtype}, moe_a2a_backend {self.moe_a2a_backend} "
                    )
                else:
                    # Pure TP and partial DP Attention mode is active for NSA, logging a warning
                    if self.dp_size < self.tp_size:
                        logger.warning(
                            f"DSA with TP mode is active, dp_size={self.dp_size}, tp_size={self.tp_size}, "
                            f"attn_tp_size={self.tp_size}, attention weights will be sharded across {self.tp_size} ranks."
                        )

                if is_hip():
                    self.page_size = 1
                    logger.warning("Setting page size to 1 for DeepSeek DSA on ROCm.")
                else:
                    # For CUDA GPU
                    self.page_size = 64
                    logger.warning("Setting page size to 64 for DeepSeek DSA.")

                import torch

                major, _ = torch.cuda.get_device_capability()
                self._set_default_nsa_kv_cache_dtype(major, self.quantization)
                self._set_default_nsa_backends(self.kv_cache_dtype, major)

            if self.enable_nsa_prefill_context_parallel:
                assert (
                    self.disaggregation_mode != "decode"
                ), "CP is only supported for prefill when PD disaggregation, please remove --enable-nsa-prefill-context-parallel."

        else:
            # DeepSeek V3/R1/V3.1
            if not self.disable_piecewise_cuda_graph:
                logger.info("Piecewise CUDA graph is enabled, use MLA for prefill.")

            if is_sm100_supported():
                if (
                    self.attention_backend is None
                    and self.prefill_attention_backend is None
                    and self.decode_attention_backend is None
                ):
                    self.attention_backend = "trtllm_mla"
                    logger.info(
                        "Use trtllm_mla as attention backend on sm100 for DeepseekV3ForCausalLM"
                    )

        # Set moe backend for DeepSeek
        if is_sm100_supported():
            quant_method = get_quantization_config(hf_config)
            quant_cfg = getattr(hf_config, "quantization_config", None) or {}
            config_groups = quant_cfg.get("config_groups", {})
            group0 = config_groups.get("group_0", {})
            weights_cfg = group0.get("weights", {})
            # this also apply to kimi k2.5
            # since it follow the compressed tensor int4 recipe
            # but not kimi k2 instruct or 0905 instruct.
            is_kimi_k2_k25_thinking_int4 = (
                quant_method == "compressed-tensors"
                and weights_cfg.get("num_bits") == 4
                and weights_cfg.get("group_size") == 32
                and weights_cfg.get("strategy") == "group"
                and weights_cfg.get("type") == "int"
            )
            if self.quantization is None:
                # Default DeepSeek V3/R1 native FP8 when not explicitly set,
                # Because we need this condition for an assertion in
                # flashinfer_trtllm MoE runner backend.
                if quant_method is None and model_arch in ["DeepseekV3ForCausalLM"]:
                    self.quantization = "fp8"
                    logger.info(
                        "Quantization not specified, default to fp8 for DeepSeek on sm100"
                    )
                else:
                    self.quantization = quant_method
            if (
                self.moe_a2a_backend == "none"
                and self.moe_runner_backend == "auto"
                and (
                    self.quantization
                    in ["fp8", "modelopt_fp8", "modelopt_fp4", "modelopt_mixed"]
                    or is_kimi_k2_k25_thinking_int4
                )
            ):
                self.moe_runner_backend = "flashinfer_trtllm"
                if is_kimi_k2_k25_thinking_int4:
                    logger.info(
                        "Use flashinfer_trtllm as MoE runner backend on Blackwell for Kimi K2 / K2.5 thinking int4"
                    )
                else:
                    logger.info(
                        "Use flashinfer_trtllm as MoE runner backend on sm100 for DeepseekV3ForCausalLM"
                    )
        elif is_hip():
            if not self.enable_dp_attention and self.nnodes == 1:
                logger.info("Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM")

            if (
                self.quantization == "modelopt_fp4"
                and self.speculative_algorithm == "EAGLE"
                and (
                    self.speculative_moe_runner_backend is None
                    or self.speculative_moe_a2a_backend is None
                )
            ):
                if envs.SGLANG_NVFP4_CKPT_FP8_NEXTN_MOE.get():
                    self.speculative_moe_runner_backend = "deep_gemm"
                    self.speculative_moe_a2a_backend = "deepep"
                    logger.info(
                        "Use deep_gemm moe runner and deepep a2a backend for bf16 nextn layer in deepseek fp4 checkpoint."
                    )
                    # Validate usage of ep
                    if self.ep_size == 1:
                        raise ValueError(
                            "Invalid configuration: 'deep_gemm' speculative MoE runner backend with "
                            "'deepep' a2a backend requires expert parallelism (ep_size > 1). "
                            f"Current ep_size is {self.ep_size}. "
                            "Please set --ep-size > 1 (e.g., --ep-size 8) to use this configuration, "
                            "or change --speculative-moe-a2a-backend to 'none' if expert parallelism is not available."
                        )
                else:
                    self.speculative_moe_runner_backend = "triton"
                    self.speculative_moe_a2a_backend = "none"
                    logger.info(
                        "Use triton fused moe by default for bf16 nextn layer in deepseek fp4 checkpoint."
                    )

    def _handle_gpt_oss_family(self, hf_config, model_arch):
        # Set attention backend for GPT-OSS
        if self.is_attention_backend_not_set():
            if is_sm100_supported():
                self.attention_backend = "trtllm_mha"
            elif is_sm90_supported():
                self.attention_backend = "fa3"
            elif is_hip():
                self.attention_backend = "aiter"
            else:
                self.attention_backend = "triton"

        supported_backends = [
            "triton",
            "trtllm_mha",
            "fa3",
            "fa4",
            "ascend",
            "aiter",
        ]
        prefill_attn_backend, decode_attn_backend = self.get_attention_backends()
        assert (
            prefill_attn_backend in supported_backends
            and decode_attn_backend in supported_backends
        ), (
            f"GptOssForCausalLM requires one of {supported_backends} attention backend, but got the following backends\n"
            f"- Prefill: {prefill_attn_backend}\n"
            f"- Decode: {decode_attn_backend}\n"
        )

        quant_method = get_quantization_config(hf_config)
        is_mxfp4_quant_format = quant_method == "mxfp4"
        if not self.enable_dp_attention and self.nnodes == 1 and is_hip():
            logger.info("Enable Aiter AllReduce Fusion for GptOssForCausalLM")
        quantization_config = getattr(hf_config, "quantization_config", None)
        is_mxfp4_quant_format = (
            quantization_config is not None
            and quantization_config.get("quant_method") == "mxfp4"
        )
        if is_mxfp4_quant_format:
            # use bf16 for mxfp4 triton kernels
            self.dtype = "bfloat16"

        if self.moe_runner_backend == "auto":
            if is_sm100_supported() and is_mxfp4_quant_format:
                self.moe_runner_backend = "flashinfer_mxfp4"
                logger.warning(
                    "Detected SM100 and MXFP4 quantization format for GPT-OSS model, enabling FlashInfer MXFP4 MOE kernel."
                )
            elif is_sm120_supported() and is_mxfp4_quant_format:
                # trtllm-gen only supports SM100
                self.moe_runner_backend = "triton_kernel"
                logger.warning(
                    "Detected SM120 and MXFP4 quantization format for GPT-OSS model, enabling triton_kernel MOE kernel."
                )
            elif (
                is_hip() and get_bool_env_var("SGLANG_USE_AITER")
            ) and is_mxfp4_quant_format:
                self.moe_runner_backend = "auto"
                logger.warning(
                    "Detected ROCm and MXFP4 quantization format for GPT-OSS model, enabling aiter MXFP4 MOE kernel."
                )
            elif is_hip() and get_bool_env_var("SGLANG_USE_AITER"):
                # For GPT-OSS bf16 on ROCm with aiter, use triton backend
                # because aiter CK kernel doesn't support all GEMM dimensions
                self.moe_runner_backend = "triton"
                logger.warning(
                    "Detected ROCm with SGLANG_USE_AITER for GPT-OSS bf16 model, using triton MOE kernel."
                )
            elif (
                self.ep_size == 1
                and is_triton_kernels_available()
                and self.quantization is None
            ):
                # The triton_kernels package segfaults on Blackwell (B200)
                # with NVIDIA driver >= 595. Fall back to triton backend.
                if is_blackwell_supported() and get_nvidia_driver_version() >= (595,):
                    self.moe_runner_backend = "triton"
                    logger.warning(
                        "Detected GPT-OSS model on Blackwell with driver >= 595, "
                        "using triton MOE kernel to avoid triton_kernels SIGSEGV."
                    )
                else:
                    self.moe_runner_backend = "triton_kernel"
                    logger.warning(
                        "Detected GPT-OSS model, enabling triton_kernels MOE kernel."
                    )

        if self.moe_runner_backend == "triton_kernel":
            assert (
                self.ep_size == 1
            ), "Triton kernel MoE is only supported when ep_size == 1"

    def _handle_mimo_family(self, hf_config, model_arch):
        if self.speculative_algorithm == "EAGLE":
            self.enable_multi_layer_eagle = True
            logger.info(
                f"Enable multi-layer EAGLE speculative decoding for {model_arch} model."
            )
            if not envs.SGLANG_ENABLE_SPEC_V2.get():
                envs.SGLANG_ENABLE_SPEC_V2.set(True)
                logger.warning(
                    "Spec v2 is enabled for multi-layer EAGLE speculative decoding."
                )

        if self.enable_hierarchical_cache:
            self.swa_full_tokens_ratio = 1.0
            logger.warning(
                f"Reset swa_full_tokens_ratio to 1.0 for {model_arch} model with hierarchical cache"
            )
            self.disable_hybrid_swa_memory = True
            logger.warning(
                f"Disable hybrid SWA memory for {model_arch} model with hierarchical cache"
            )

    def _handle_llama4_family(self, hf_config, model_arch):
        if self.device == "cpu":
            return
        # Auto-select attention backend for Llama4 if not specified
        if self.attention_backend is None:
            if is_sm100_supported():
                self.attention_backend, platform = "trtllm_mha", "sm100"
            elif is_sm90_supported():
                self.attention_backend, platform = "fa3", "sm90"
            elif is_hip():
                self.attention_backend, platform = "aiter", "hip"
            elif self.device == "xpu":
                self.attention_backend, platform = "intel_xpu", "xpu"
            else:
                self.attention_backend, platform = "triton", "other platforms"
            logger.warning(
                f"Use {self.attention_backend} as attention backend on {platform} for Llama4 model"
            )
        assert self.attention_backend in {
            "fa3",
            "aiter",
            "triton",
            "ascend",
            "trtllm_mha",
            "intel_xpu",
        }, f"fa3, aiter, triton, ascend, trtllm_mha or intel_xpu is required for Llama4 model but got {self.attention_backend}"
        if is_sm100_supported() and self.moe_runner_backend == "auto":
            if self.quantization in {"fp8", "modelopt_fp8"}:
                self.moe_runner_backend = "flashinfer_trtllm"
                logger.info(
                    "Use flashinfer_trtllm as MoE runner backend on SM100 for Llama4"
                )

    def _handle_gemma_family(self, hf_config, model_arch):
        # FIXME: https://github.com/sgl-project/sglang/pull/7367 is not compatible with gemma2 model.
        logger.warning(
            f"Disable hybrid SWA memory for {model_arch} as it is not yet supported."
        )
        self.disable_hybrid_swa_memory = True

    def _handle_gemma4_family(self, hf_config, model_arch):
        if self.is_attention_backend_not_set():
            self.attention_backend = "triton"
            logger.info("Use triton as default attention backend for Gemma4")

    def _handle_exaone_family(self, hf_config, model_arch):
        if hf_config.sliding_window_pattern is not None:
            logger.warning(
                f"Disabling hybrid SWA memory for {model_arch} as it is not yet supported."
            )
            self.disable_hybrid_swa_memory = True
            # https://docs.sglang.ai/advanced_features/attention_backend.html
            accepted_backends = ["fa3", "triton", "trtllm_mha"]
            assert (
                self.attention_backend in accepted_backends
            ), f"One of the attention backends in {accepted_backends} is required for {model_arch}, but got {self.attention_backend}"

    def _handle_olmo_family(self, hf_config, model_arch):
        # FIXME: https://github.com/sgl-project/sglang/pull/7367 is not compatible with Olmo3 model.
        logger.warning(
            f"Disabling hybrid SWA memory for {model_arch} as it is not yet supported."
        )
        self.disable_hybrid_swa_memory = True

        if self.attention_backend is None:
            if is_cuda() and is_sm100_supported():
                self.attention_backend = "trtllm_mha"
            elif is_cuda() and get_device_sm() >= 80:
                self.attention_backend = "fa3"
            else:
                self.attention_backend = "triton"

        # Flashinfer appears to degrade performance when sliding window attention
        # is used for the Olmo2 architecture. Olmo2 does not use sliding window attention
        # but Olmo3 does.
        assert (
            self.attention_backend != "flashinfer"
        ), "FlashInfer backend can significantly degrade the performance of Olmo3 models."

        logger.info(
            f"Using {self.attention_backend} as attention backend for {model_arch}."
        )

    def _handle_kimi_linear_family(self, hf_config, model_arch):
        self._handle_mamba_radix_cache(
            model_arch=model_arch,
            support_mamba_cache=False,
        )

    def _handle_nemotron_family(self, hf_config, model_arch):
        model_config = self.get_model_config()
        if model_config.quantization in [
            "modelopt",
            "modelopt_fp8",
            "modelopt_fp4",
            "modelopt_mixed",
        ]:
            assert model_config.hf_config.mlp_hidden_act == "relu2"
            if model_config.quantization == "modelopt":
                quant_algo = model_config.hf_config.quantization_config["quant_algo"]
                if quant_algo == "MIXED_PRECISION":
                    self.quantization = "modelopt_mixed"
                else:
                    self.quantization = (
                        "modelopt_fp4" if quant_algo == "NVFP4" else "modelopt_fp8"
                    )
            else:
                self.quantization = model_config.quantization
            self.moe_runner_backend = "flashinfer_cutlass"

        self._handle_mamba_radix_cache(
            model_arch=model_arch,
            support_mamba_cache=True,
            support_mamba_cache_extra_buffer=False,
            sm100_default_attention_backend="flashinfer",
        )
        assert self.attention_backend != "triton", (
            "NemotronHForCausalLM does not support triton attention backend,"
            "as the first layer might not be an attention layer"
        )

    def _handle_qwen_family(self, hf_config, model_arch):
        if is_sm100_supported():
            quant_method = get_quantization_config(hf_config)
            if self.quantization is None and quant_method is not None:
                self.quantization = quant_method
            if (
                (
                    self.quantization in ("fp8", "modelopt_fp4")
                    or self.quantization is None
                )
                and self.moe_a2a_backend == "none"
                and self.moe_runner_backend == "auto"
            ):
                self.moe_runner_backend = "flashinfer_trtllm"
                logger.info(
                    "Use flashinfer_trtllm as MoE runner backend on sm100 for "
                    f"{model_arch}"
                )

        if model_arch in [
            "Qwen3NextForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
        ]:
            sm100_default_attn_backend = "triton"
            if is_sm100_supported():
                # trtllm_mha requires speculative_eagle_topk == 1 and page_size > 1.
                # _get_default_attn_backend handles the eagle_topk check.
                # There is only one case where page_size=1 is required,
                # which is when radix cache is enabled and both extra_buffer
                # and spec decoding are disabled.
                default_attn_backend = self._get_default_attn_backend(
                    use_mla_backend=self.use_mla_backend(),
                    model_config=self.get_model_config(),
                )
                if default_attn_backend == "trtllm_mha" and not (
                    not self.enable_mamba_extra_buffer()
                    and not self.disable_radix_cache
                    and self.speculative_algorithm is None
                ):
                    sm100_default_attn_backend = "trtllm_mha"

            self._handle_mamba_radix_cache(
                model_arch=model_arch,
                support_mamba_cache=True,
                support_mamba_cache_extra_buffer=True,
                sm100_default_attention_backend=sm100_default_attn_backend,
            )

    def _handle_glm4_family(self, hf_config, model_arch):
        if is_sm100_supported():
            quantization_config = getattr(hf_config, "quantization_config", None)
            quant_method = (
                quantization_config.get("quant_method")
                if quantization_config is not None
                else None
            )
            if self.quantization is None and quant_method is not None:
                self.quantization = quant_method
            if (
                self.quantization == "modelopt_fp4"
                and self.moe_a2a_backend == "none"
                and self.moe_runner_backend == "auto"
            ):
                self.moe_runner_backend = "flashinfer_trtllm"
                logger.info(
                    "Use flashinfer_trtllm as MoE runner backend on sm100 for Glm4MoeForCausalLM"
                )

    def _handle_falcon_jet_family(self, hf_config, model_arch):
        self._handle_mamba_radix_cache(
            model_arch=model_arch,
            support_mamba_cache=True,
            support_mamba_cache_extra_buffer=False,
            sm100_default_attention_backend="triton",
        )

    def _handle_granite_moe_family(self, hf_config, model_arch):
        hf_config = self.get_model_config().hf_config
        has_mamba = any(
            layer_type == "mamba"
            for layer_type in getattr(hf_config, "layer_types", [])
        )
        if has_mamba:
            self._handle_mamba_radix_cache(
                model_arch=model_arch,
                support_mamba_cache_extra_buffer=False,
                sm100_default_attention_backend="triton",
            )

    def _handle_lfm2_family(self, hf_config, model_arch):
        self._handle_mamba_radix_cache(
            model_arch=model_arch,
            support_mamba_cache=True,
            support_mamba_cache_extra_buffer=False,
            sm100_default_attention_backend="flashinfer",
        )
        assert self.attention_backend != "triton", (
            f"{model_arch} does not support triton attention backend, "
            "as the first layer might not be an attention layer"
        )

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _get_model_arch_handlers(self):
        """Return a dict mapping exact architecture strings to their handler."""
        return {
            "MistralLarge3ForCausalLM": self._handle_mistral_large_family,
            "PixtralForConditionalGeneration": self._handle_mistral_large_family,
            "DeepseekV3ForCausalLM": self._handle_deepseek_family,
            "KimiK25ForConditionalGeneration": self._handle_deepseek_family,
            "GlmMoeDsaForCausalLM": self._handle_deepseek_family,
            "GptOssForCausalLM": self._handle_gpt_oss_family,
            "Gemma2ForCausalLM": self._handle_gemma_family,
            "Gemma3ForCausalLM": self._handle_gemma_family,
            "Gemma3ForConditionalGeneration": self._handle_gemma_family,
            "Gemma3nForCausalLM": self._handle_gemma_family,
            "Gemma3nForConditionalGeneration": self._handle_gemma_family,
            "Gemma4ForConditionalGeneration": self._handle_gemma4_family,
            "Exaone4ForCausalLM": self._handle_exaone_family,
            "ExaoneMoEForCausalLM": self._handle_exaone_family,
            "Olmo2ForCausalLM": self._handle_olmo_family,
            "KimiLinearForCausalLM": self._handle_kimi_linear_family,
            "BailingMoeV2_5ForCausalLM": self._handle_kimi_linear_family,
            "NemotronHForCausalLM": self._handle_nemotron_family,
            "Qwen3MoeForCausalLM": self._handle_qwen_family,
            "Qwen3VLMoeForConditionalGeneration": self._handle_qwen_family,
            "Qwen3NextForCausalLM": self._handle_qwen_family,
            "Qwen3_5MoeForConditionalGeneration": self._handle_qwen_family,
            "Qwen3_5ForConditionalGeneration": self._handle_qwen_family,
            "Glm4MoeForCausalLM": self._handle_glm4_family,
            "FalconH1ForCausalLM": self._handle_falcon_jet_family,
            "JetNemotronForCausalLM": self._handle_falcon_jet_family,
            "JetVLMForConditionalGeneration": self._handle_falcon_jet_family,
            "GraniteMoeHybridForCausalLM": self._handle_granite_moe_family,
            "Lfm2ForCausalLM": self._handle_lfm2_family,
        }

    def _get_model_arch_substring_handlers(self):
        """Return a dict for architectures matched by substring (order matters)."""
        return {
            "Llama4": self._handle_llama4_family,
            "MiMoV2FlashForCausalLM": self._handle_mimo_family,
            "Step3p5ForCausalLM": self._handle_mimo_family,
        }
