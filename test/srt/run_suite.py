import argparse
import glob
from pathlib import Path

from sglang.test.ci.ci_utils import TestFile, run_unittest_files

# NOTE: please sort the test cases alphabetically by the test file name
suites = {
    "per-commit-1-gpu": [
        TestFile("debug_utils/test_tensor_dump_forward_hook.py", 15),
        TestFile("hicache/test_hicache_storage.py", 127),
        TestFile("hicache/test_hicache_variants.py", 393),
        TestFile("layers/attention/mamba/test_causal_conv1d.py", 25),
        TestFile("layers/attention/mamba/test_mamba_ssm.py", 50),
        TestFile("layers/attention/mamba/test_mamba_ssm_ssd.py", 20),
        TestFile("lora/test_lora.py", 150),
        TestFile("lora/test_lora_eviction.py", 240),
        TestFile("lora/test_lora_update.py", 600),
        TestFile("lora/test_lora_backend.py", 99),
        TestFile("lora/test_lora_spec_decoding.py", 150),
        TestFile("lora/test_multi_lora_backend.py", 60),
        TestFile("models/test_compressed_tensors_models.py", 42),
        TestFile("models/test_cross_encoder_models.py", 100),
        TestFile("models/test_embedding_models.py", 73),
        TestFile("models/test_encoder_embedding_models.py", 460),
        TestFile("models/test_generation_models.py", 103),
        TestFile("models/test_nvidia_nemotron_nano_v2.py", 160),
        TestFile("models/test_qwen_models.py", 150),
        TestFile("models/test_reward_models.py", 132),
        TestFile("models/test_transformers_models.py", 320),
        TestFile("models/test_vlm_models.py", 741),
        TestFile("openai_server/basic/test_openai_embedding.py", 79),
        TestFile("openai_server/basic/test_openai_server.py", 270),
        TestFile("openai_server/basic/test_protocol.py", 10),
        TestFile("openai_server/basic/test_serving_chat.py", 10),
        TestFile("openai_server/basic/test_serving_completions.py", 10),
        TestFile("openai_server/basic/test_serving_embedding.py", 10),
        TestFile("openai_server/features/test_enable_thinking.py", 70),
        TestFile("openai_server/features/test_json_mode.py", 120),
        TestFile("openai_server/features/test_openai_server_ebnf.py", 20),
        TestFile("openai_server/features/test_openai_server_hidden_states.py", 240),
        TestFile("openai_server/features/test_reasoning_content.py", 89),
        TestFile("openai_server/function_call/test_openai_function_calling.py", 60),
        TestFile("openai_server/function_call/test_tool_choice.py", 120),
        TestFile("openai_server/validation/test_large_max_new_tokens.py", 41),
        TestFile("openai_server/validation/test_matched_stop.py", 60),
        TestFile("openai_server/validation/test_openai_server_ignore_eos.py", 85),
        TestFile("openai_server/validation/test_request_length_validation.py", 31),
        TestFile("quant/test_block_int8.py", 22),
        TestFile("quant/test_fp8_kernel.py", 8),
        TestFile("quant/test_int8_kernel.py", 8),
        TestFile("quant/test_triton_scaled_mm.py", 8),
        TestFile("quant/test_w8a8_quantization.py", 160),
        TestFile("quant/test_autoround.py", 60),
        TestFile("rl/test_fp32_lm_head.py", 30),
        TestFile("rl/test_update_weights_from_disk.py", 210),
        TestFile("rl/test_update_weights_from_tensor.py", 80),
        TestFile("test_abort.py", 190),
        TestFile("test_build_eagle_tree.py", 8),
        TestFile("test_chunked_prefill.py", 410),
        TestFile("test_create_kvindices.py", 2),
        TestFile("test_deterministic.py", 400),
        TestFile("test_eagle_infer_a.py", 750),
        TestFile("test_eagle_infer_b.py", 750),
        TestFile("test_eagle_infer_beta.py", 90),
        TestFile("test_constrained_decoding.py", 150),
        TestFile("test_eval_fp8_accuracy.py", 303),
        TestFile("test_fa3.py", 420),
        TestFile("test_flashmla.py", 230),
        TestFile("test_fp8_utils.py", 5),
        TestFile("rotary_embedding/test_mrope.py", 10),
        TestFile("test_fused_moe.py", 80),
        TestFile("test_gpt_oss_1gpu.py", 750),
        TestFile("test_harmony_parser.py", 20),
        TestFile("test_hidden_states.py", 55),
        TestFile("test_hybrid_attn_backend.py", 379),
        TestFile("test_input_embeddings.py", 38),
        TestFile("test_io_struct.py", 8),
        TestFile("test_jinja_template_utils.py", 1),
        TestFile("test_mamba_unittest.py", 4),
        TestFile("test_metrics.py", 32),
        TestFile("test_metrics_utils.py", 1),
        TestFile("test_mla.py", 180),
        TestFile("test_mla_deepseek_v3.py", 500),
        TestFile("test_mla_flashinfer.py", 302),
        TestFile("test_mla_fp8.py", 93),
        TestFile("test_mla_int8_deepseek_v3.py", 300),
        TestFile("test_model_hooks.py", 1),
        TestFile("test_modelopt_loader.py", 30),
        TestFile("test_multi_tokenizer.py", 230),
        TestFile("test_ngram_speculative_decoding.py", 290),
        TestFile("test_no_chunked_prefill.py", 108),
        TestFile("test_no_overlap_scheduler.py", 234),
        TestFile("test_original_logprobs.py", 41),
        TestFile("test_page_size.py", 60),
        TestFile("test_penalty.py", 82),
        TestFile("test_piecewise_cuda_graph.py", 600),
        TestFile("test_priority_scheduling.py", 130),
        TestFile("test_pytorch_sampling_backend.py", 66),
        TestFile("test_radix_attention.py", 105),
        TestFile("test_radix_cache_unit.py", 5),
        TestFile("test_reasoning_parser.py", 5),
        TestFile("test_request_queue_validation.py", 30),
        TestFile("test_retract_decode.py", 450),
        TestFile("test_score_api.py", 310),
        TestFile("test_server_args.py", 1),
        TestFile("test_speculative_registry.py", 1),
        TestFile("test_skip_tokenizer_init.py", 117),
        TestFile("test_srt_endpoint.py", 130),
        TestFile("test_srt_engine.py", 450),
        TestFile("test_standalone_speculative_decoding.py", 150),
        TestFile("test_start_profile.py", 180),
        TestFile("test_profile_merger.py", 60),
        TestFile("test_profile_merger_http_api.py", 15),
        TestFile("test_swa_unittest.py", 1),
        TestFile("test_torch_compile.py", 76),
        TestFile("test_torch_compile_moe.py", 210),
        TestFile("test_torch_native_attention_backend.py", 123),
        TestFile("test_torchao.py", 70),
        TestFile("test_triton_attention_kernels.py", 4),
        TestFile("test_triton_attention_backend.py", 150),
        TestFile("test_triton_attention_kernels.py", 4),
        TestFile("test_triton_moe_channel_fp8_kernel.py", 25),
        TestFile("test_triton_sliding_window.py", 100),
        TestFile("test_utils_update_weights.py", 48),
        TestFile("test_vision_chunked_prefill.py", 170),
        TestFile("test_vision_openai_server_a.py", 900),
        TestFile("test_vlm_input_format.py", 300),
        TestFile("test_modelopt_loader.py", 30),
        TestFile("test_modelopt_export.py", 30),
    ],
    "per-commit-2-gpu": [
        TestFile("ep/test_moe_ep.py", 140),
        TestFile("hicache/test_hicache_storage_3fs_backend.py", 200),
        TestFile("hicache/test_hicache_storage_file_backend.py", 200),
        TestFile("hicache/test_hicache_storage_mooncake_backend.py", 300),
        TestFile("layers/attention/mamba/test_mamba2_mixer.py", 50),
        TestFile("lora/test_lora_tp.py", 116),
        TestFile("models/test_glm4_moe_models.py", 100),
        TestFile("models/test_kimi_linear_models.py", 90),
        TestFile("rl/test_update_weights_from_distributed.py", 103),
        TestFile("test_data_parallelism.py", 73),
        TestFile("test_disaggregation_basic.py", 400),
        TestFile("test_dp_attention.py", 350),
        TestFile("test_load_weights_from_remote_instance.py", 72),
        TestFile("test_patch_torch.py", 19),
        TestFile("test_release_memory_occupation.py", 200),
        TestFile("test_eagle_dp_attention.py", 200),
    ],
    "per-commit-4-gpu": [
        TestFile("models/test_qwen3_next_models.py", 291),
        TestFile("test_gpt_oss_4gpu.py", 300),
        TestFile("test_local_attn.py", 411),
        TestFile("test_multi_instance_release_memory_occupation.py", 64),
        TestFile("test_pp_single_node.py", 481),
    ],
    "per-commit-8-gpu-h200": [
        TestFile("test_deepseek_v3_basic.py", 275),
        TestFile("test_deepseek_v3_mtp.py", 275),
        TestFile("test_disaggregation_hybrid_attention.py", 200),
        TestFile("models/test_kimi_k2_models.py", 200),
        TestFile("test_deepseek_v32_basic.py", 275),
        TestFile("test_deepseek_v32_mtp.py", 275),
    ],
    "per-commit-8-gpu-h20": [
        TestFile("quant/test_w4a8_deepseek_v3.py", 520),
        TestFile("test_disaggregation_different_tp.py", 600),
        TestFile("test_disaggregation_pp.py", 140),
        TestFile("test_disaggregation_dp_attention.py", 155),
    ],
    "per-commit-4-gpu-b200": [
        TestFile("test_deepseek_v3_fp4_4gpu.py", 1800),
        TestFile("test_flash_attention_4.py", 300),
        TestFile("test_gpt_oss_4gpu.py", 600),
        TestFile("test_llama31_fp4.py", 300),
        # TODO: Add it back after the bug is fixed
        # TestFile("test_eagle_infer_beta_dp_attention.py", 200),
    ],
    "per-commit-8-gpu-b200": [],
    "per-commit-4-gpu-gb200": [
        TestFile("test_cutedsl_moe.py", 300),
        TestFile("test_deepseek_v3_fp4_4gpu.py", 1800),
        # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/12533
        # TestFile("test_deepseek_v3_cutedsl_4gpu.py", 3600),
    ],
    "per-commit-4-gpu-deepep": [
        TestFile("ep/test_deepep_small.py", 531),
        TestFile("ep/test_mooncake_ep_small.py", 450),
    ],
    "per-commit-8-gpu-h200-deepep": [
        TestFile("ep/test_deepep_large.py", 338),
    ],
    "quantization_test": [
        TestFile("quant/test_awq.py", 163),
        TestFile("test_bnb.py", 5),
        TestFile("test_gptqmodel_dynamic.py", 102),
        TestFile("test_quantization.py", 185),
        TestFile("test_gguf.py", 96),
    ],
    # If the test cases take too long, considering adding them to nightly tests instead of per-commit tests
    "nightly-1-gpu": [
        TestFile("layers/attention/nsa/test_nsa_indexer.py", 2),
        TestFile("lora/test_lora_qwen3.py", 97),
        TestFile("lora/test_lora_radix_cache.py", 200),
        TestFile("lora/test_lora_eviction_policy.py", 200),
        TestFile("lora/test_lora_openai_api.py", 30),
        TestFile("openai_server/features/test_lora_openai_compatible.py", 150),
        TestFile("batch_invariant/test_batch_invariant_ops.py", 10),
        TestFile("test_cpp_radix_cache.py", 60),
        TestFile("test_deepseek_v3_deterministic.py", 240),
    ],
    "nightly-4-gpu-b200": [
        TestFile("nightly/test_flashinfer_trtllm_gen_moe_backend.py", 300),
        TestFile("nightly/test_gpt_oss_4gpu_perf.py", 600),
        TestFile("nightly/test_flashinfer_trtllm_gen_attn_backend.py", 300),
        TestFile("test_deepseek_v3_fp4_cutlass_moe.py", 900),
        TestFile("test_fp4_moe.py", 300),
    ],
    "nightly-8-gpu-b200": [
        TestFile("test_deepseek_r1_fp8_trtllm_backend.py", 3600),
    ],
    "nightly-4-gpu": [
        TestFile("nightly/test_encoder_dp.py", 500),
        TestFile("test_qwen3_next_deterministic.py", 200),
    ],
    "nightly-8-gpu": [],
    "nightly-8-gpu-h200": [
        TestFile("test_deepseek_v32_nsabackend.py", 600),
    ],
    "nightly-8-gpu-h20": [],
    "__not_in_ci__": [
        TestFile("ascend/test_ascend_w8a8_quantization.py"),
        TestFile("cpu/test_comm.py"),
        TestFile("debug_utils/test_log_parser.py", 5),
        TestFile("test_deepseek_v3_cutedsl_4gpu.py"),
        TestFile("entrypoints/http_server/test_abort_request.py"),
        TestFile("ep/test_deepep_internode.py"),
        TestFile("ep/test_deepep_intranode.py"),
        TestFile("ep/test_deepep_low_latency.py"),
        TestFile("ep/test_eplb.py"),
        TestFile("ep/test_hybrid_dp_ep_tp_mtp.py"),
        TestFile("ep/test_moe_deepep.py"),
        TestFile("ep/test_moe_deepep_eval_accuracy_large.py"),
        TestFile("hicache/test_disaggregation_hicache.py"),
        TestFile("hicache/test_hicache_storage_benchmark.py"),
        TestFile("hicache/test_hicache_storage_e2e.py"),
        TestFile("layers/attention/nsa/test_act_quant_triton.py"),
        TestFile("layers/moe/test_moe_runners.py"),
        TestFile("lora/test_chunked_sgmv_backend.py"),
        TestFile("lora/test_lora_llama4.py"),
        TestFile("lora/test_lora_cuda_graph.py"),
        TestFile("lora/test_lora_qwen3_vl.py"),
        TestFile("models/test_clip_models.py"),
        TestFile("models/test_dummy_grok_models.py"),
        TestFile("models/test_falcon_h1_models.py"),
        TestFile("models/test_gme_qwen_models.py"),
        TestFile("models/test_grok_models.py"),
        TestFile("models/test_llama4_models.py"),
        TestFile("models/test_mtp_models.py"),
        TestFile("models/test_unsloth_models.py"),
        TestFile("openai/test_server.py"),
        TestFile("openai_server/features/test_cache_report.py"),
        TestFile("openai_server/features/test_continuous_usage_stats.py"),
        TestFile("openai_server/features/test_structural_tag.py"),
        TestFile("quant/test_fp8_kvcache.py"),
        TestFile("rl/test_verl_engine_2_gpu.py"),
        TestFile("rl/test_verl_engine_4_gpu.py"),
        TestFile("test_ascend_attention_backend.py"),
        TestFile("test_ascend_mla_backend.py"),
        TestFile("test_ascend_mla_w8a8int8.py"),
        TestFile("test_ascend_tp1_bf16.py"),
        TestFile("test_ascend_tp2_bf16.py"),
        TestFile("test_ascend_w8a8_quantization.py"),
        TestFile("test_async_dynamic_batch_tokenizer.py"),
        TestFile("test_async_mm_data_processor.py"),
        TestFile("test_awq.py"),
        TestFile("test_awq_dequant.py"),
        TestFile("test_bench_one_batch.py"),
        TestFile("test_bench_serving.py"),
        TestFile("test_block_int8.py"),
        TestFile("test_cache_report.py"),
        TestFile("test_config_integration.py"),
        TestFile("test_cpp_radix_cache.py"),
        TestFile("test_cpu_graph.py"),
        TestFile("test_custom_allreduce.py"),
        TestFile("test_cutedsl_flashinfer_8gpu.py"),
        TestFile("test_deepep_internode.py"),
        TestFile("test_deepep_intranode.py"),
        TestFile("test_deepep_large.py"),
        TestFile("test_deepep_low_latency.py"),
        TestFile("test_deepep_small.py"),
        TestFile("test_deepseek_chat_templates.py"),
        TestFile("test_disaggregation.py"),
        TestFile("test_double_sparsity.py"),
        TestFile("test_eagle_infer_beta_dp_attention.py"),
        TestFile("test_embedding_openai_server.py"),
        TestFile("test_enable_thinking.py"),
        TestFile("test_eplb.py"),
        TestFile("test_eval_accuracy_large.py"),
        TestFile("test_expert_distribution.py"),
        TestFile("test_expert_location_updater.py"),
        TestFile("test_fim_completion.py"),
        TestFile("test_forward_split_prefill.py"),
        TestFile("test_fp8_kernel.py"),
        TestFile("test_fp8_kvcache.py"),
        TestFile("test_full_deepseek_v3.py"),
        TestFile("test_get_weights_by_name.py"),
        TestFile("test_gpt_oss_common.py"),
        TestFile("test_health_check.py"),
        TestFile("test_hicache_storage.py"),
        TestFile("test_hicache_variants.py"),
        TestFile("test_hybrid_dp_ep_tp_mtp.py"),
        TestFile("test_int4_kernel.py"),
        TestFile("test_int8_kernel.py"),
        TestFile("test_intel_amx_attention_backend.py"),
        TestFile("test_constrained_decoding.py"),
        TestFile("test_json_mode.py"),
        TestFile("test_kv_events.py"),
        TestFile("test_large_max_new_tokens.py"),
        TestFile("test_logprobs.py"),
        TestFile("test_lookahead_speculative_decoding.py"),
        TestFile("test_matched_stop.py"),
        TestFile("test_mla_tp.py"),
        TestFile("test_modelopt.py"),
        TestFile("test_modelopt_fp8kvcache.py"),
        TestFile("test_models_from_modelscope.py"),
        TestFile("test_moe_deepep.py"),
        TestFile("test_moe_deepep_eval_accuracy_large.py"),
        TestFile("test_moe_ep.py"),
        TestFile("test_moe_eval_accuracy_large.py"),
        TestFile("test_mscclpp.py"),
        TestFile("nightly/test_deepseek_v31_perf.py"),
        TestFile("nightly/test_deepseek_v32_perf.py"),
        TestFile("nightly/test_gpt_oss_4gpu_perf.py"),
        TestFile("nightly/test_gsm8k_eval_amd.py"),
        TestFile("nightly/test_text_models_gsm8k_eval.py"),
        TestFile("nightly/test_text_models_perf.py"),
        TestFile("nightly/test_vlms_mmmu_eval.py"),
        TestFile("nightly/test_vlms_perf.py"),
        TestFile("test_openai_adapter.py"),
        TestFile("test_openai_function_calling.py"),
        TestFile("test_openai_server.py"),
        TestFile("test_openai_server_hidden_states.py"),
        TestFile("test_piecewise_cuda_graph.py"),
        TestFile("test_quick_allreduce.py"),
        TestFile("test_reasoning_content.py"),
        TestFile("test_request_length_validation.py"),
        TestFile("test_sagemaker_server.py"),
        TestFile("test_schedule_policy.py"),
        TestFile("test_session_control.py"),
        TestFile("test_srt_engine_with_quant_args.py"),
        TestFile("test_tokenizer_batch_encode.py"),
        TestFile("test_tokenizer_manager.py"),
        TestFile("test_tool_choice.py"),
        TestFile("test_torch_flex_attention_backend.py"),
        TestFile("test_torch_tp.py"),
        TestFile("test_tracing.py"),
        TestFile("test_triton_attention_rocm_mla.py"),
        TestFile("test_triton_fused_moe.py"),
        TestFile("test_triton_moe_wna16.py"),
        TestFile("test_two_batch_overlap.py"),
        TestFile("test_update_weights_from_disk.py"),
        TestFile("test_update_weights_from_distributed.py"),
        TestFile("test_update_weights_from_tensor.py"),
        TestFile("test_verl_engine_2_gpu.py"),
        TestFile("test_verl_engine_4_gpu.py"),
        TestFile("test_verl_engine_server.py"),
        TestFile("test_vertex_endpoint.py"),
        # TestFile("test_vision_openai_server_a.py"),  # TODO: Fix timeout
        TestFile("test_vision_openai_server_b.py"),
        TestFile("test_vision_openai_server_common.py"),
        TestFile("test_vlm_accuracy.py"),
        TestFile("test_w4a8.py"),
        TestFile("test_w8a8_quantization.py"),
        TestFile("test_wave_attention_backend.py"),
        TestFile("test_weight_version.py"),
        TestFile("test_deepseek_v32_cp_single_node.py", 275),
    ],
}

# Add AMD tests
# NOTE: please sort the test cases alphabetically by the test file name
suite_amd = {
    "per-commit-amd": [
        # TestFile("hicache/test_hicache.py", 116), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/12575
        # TestFile("hicache/test_hicache_mla.py", 127), # Disabled temporarily,  # Temporarily disabled, see https://github.com/sgl-project/sglang/issues/12574
        # TestFile("hicache/test_hicache_storage.py", 127), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/12575
        TestFile("lora/test_lora.py", 665),
        # TestFile("lora/test_lora_backend.py", 99), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/13107
        # TestFile("lora/test_lora_cuda_graph.py", 250), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/13107
        TestFile("lora/test_lora_eviction.py", 240),
        # TestFile("lora/test_lora_qwen3.py", 97), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/13107
        TestFile("lora/test_multi_lora_backend.py", 60),
        TestFile("models/test_compressed_tensors_models.py", 42),
        TestFile("models/test_qwen_models.py", 82),
        TestFile("models/test_reward_models.py", 132),
        TestFile("models/test_transformers_models.py", 320),
        TestFile("models/test_vlm_models.py", 437),
        TestFile("openai_server/basic/test_openai_embedding.py", 141),
        TestFile("openai_server/basic/test_openai_server.py", 149),
        TestFile("openai_server/basic/test_protocol.py", 10),
        TestFile("openai_server/basic/test_serving_chat.py", 10),
        TestFile("openai_server/basic/test_serving_completions.py", 10),
        TestFile("openai_server/basic/test_serving_embedding.py", 10),
        TestFile("openai_server/features/test_enable_thinking.py", 70),
        TestFile("openai_server/features/test_json_mode.py", 120),
        TestFile("openai_server/features/test_openai_server_ebnf.py", 20),
        TestFile("openai_server/features/test_reasoning_content.py", 89),
        TestFile("openai_server/function_call/test_openai_function_calling.py", 60),
        TestFile("openai_server/function_call/test_tool_choice.py", 120),
        TestFile("openai_server/validation/test_large_max_new_tokens.py", 41),
        TestFile("openai_server/validation/test_matched_stop.py", 60),
        TestFile("openai_server/validation/test_openai_server_ignore_eos.py", 85),
        TestFile("openai_server/validation/test_request_length_validation.py", 31),
        TestFile("quant/test_awq_dequant.py", 2),
        TestFile("quant/test_block_int8.py", 22),
        TestFile("quant/test_fused_rms_fp8_group_quant.py", 10),
        TestFile("rl/test_update_weights_from_disk.py", 210),
        TestFile("test_abort.py", 51),
        TestFile("test_bench_typebaseddispatcher.py", 10),
        TestFile("test_chunked_prefill.py", 410),
        TestFile("test_create_kvindices.py", 2),
        TestFile("test_eval_fp8_accuracy.py", 303),
        TestFile("test_fused_moe.py", 30),
        TestFile("test_harmony_parser.py", 20),
        TestFile("test_input_embeddings.py", 38),
        TestFile("test_io_struct.py", 8),
        TestFile("test_jinja_template_utils.py", 1),
        TestFile("test_metrics.py", 32),
        TestFile("test_metrics_utils.py", 1),
        # TestFile("test_mla.py", 242), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/13107
        # TestFile("test_mla_deepseek_v3.py", 221), # Temporarily disabled, see https://github.com/sgl-project/sglang/issues/12574
        TestFile("test_no_chunked_prefill.py", 108),
        TestFile("test_page_size.py", 60),
        TestFile("test_penalty.py", 180),
        TestFile("test_pytorch_sampling_backend.py", 66),
        TestFile("test_radix_attention.py", 105),
        TestFile("test_reasoning_parser.py", 5),
        TestFile("test_constrained_decoding.py", 120),
        TestFile("test_retract_decode.py", 450),
        TestFile("test_rope_rocm.py", 3),
        TestFile("test_server_args.py", 1),
        TestFile("test_skip_tokenizer_init.py", 117),
        TestFile("test_srt_endpoint.py", 130),
        TestFile("test_srt_engine.py", 261),
        TestFile("test_torch_compile.py", 169),
        # TestFile("test_torch_compile_moe.py", 210), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/13107
        TestFile("test_torch_native_attention_backend.py", 123),
        # TestFile("test_triton_attention_kernels.py", 4),
        TestFile("test_triton_attention_backend.py", 150),
        TestFile("test_triton_sliding_window.py", 250),
        TestFile("test_type_based_dispatcher.py", 10),
        TestFile("test_wave_attention_kernels.py", 2),
        # Disabled temporarily
        # TestFile("test_vlm_input_format.py", 300),
        # TestFile("models/test_embedding_models.py", 73), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/11127
        # TestFile("openai_server/features/test_openai_server_hidden_states.py", 240),
        # TestFile("rl/test_update_weights_from_tensor.py", 48),
        # TestFile("test_no_overlap_scheduler.py", 234), # Disabled temporarily and track in #7703
        # TestFile("test_vision_chunked_prefill.py", 175), # Disabled temporarily and track in #7701
        # TestFile("test_wave_attention_backend.py", 150), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/11127
    ],
    "per-commit-amd-mi35x": [
        TestFile("test_gpt_oss_1gpu.py", 750),
        TestFile("test_mla.py", 242),
    ],
    "per-commit-2-gpu-amd": [
        # TestFile("lora/test_lora_tp.py", 116), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/13107
        TestFile("rl/test_update_weights_from_distributed.py", 103),
        TestFile("test_data_parallelism.py", 73),
        TestFile("test_load_weights_from_remote_instance.py", 72),
        # TestFile("test_patch_torch.py", 19), # Disabled temporarily, see https://github.com/sgl-project/sglang/issues/11127
    ],
    "per-commit-4-gpu-amd": [
        TestFile("test_pp_single_node.py", 150),
    ],
    "per-commit-8-gpu-amd": [
        TestFile("test_deepseek_v3_basic.py", 275),
        TestFile("test_deepseek_v3_mtp.py", 275),
    ],
    "nightly-amd": [
        TestFile("nightly/test_gsm8k_eval_amd.py"),
    ],
}

# Add Intel Xeon tests
# NOTE: please sort the test cases alphabetically by the test file name
suite_xeon = {
    "per-commit-cpu": [
        TestFile("cpu/test_activation.py"),
        TestFile("cpu/test_binding.py"),
        TestFile("cpu/test_decode.py"),
        TestFile("cpu/test_extend.py"),
        TestFile("cpu/test_gemm.py"),
        TestFile("cpu/test_mla.py"),
        TestFile("cpu/test_moe.py"),
        TestFile("cpu/test_norm.py"),
        TestFile("cpu/test_qkv_proj_with_rope.py"),
        TestFile("cpu/test_rope.py"),
        TestFile("cpu/test_shared_expert.py"),
        TestFile("cpu/test_topk.py"),
        TestFile("cpu/test_cpu_graph.py"),
        TestFile("cpu/test_intel_amx_attention_backend_a.py"),
        TestFile("cpu/test_intel_amx_attention_backend_b.py"),
        TestFile("cpu/test_intel_amx_attention_backend_c.py"),
    ],
}

# Add Intel XPU tests
suite_xpu = {
    "per-commit-xpu": [
        TestFile("xpu/test_intel_xpu_backend.py"),
    ],
}

# Add Ascend NPU tests
# NOTE: please sort the test cases alphabetically by the test file name
suite_ascend = {
    "per-commit-1-ascend-npu": [
        TestFile("ascend/test_ascend_graph_tp1_bf16.py", 400),
        TestFile("ascend/test_ascend_tp1_bf16.py", 400),
        TestFile("ascend/test_ascend_hicache_mha.py", 400),
        TestFile("ascend/test_ascend_sampling_backend.py", 400),
    ],
    "per-commit-2-ascend-npu": [
        TestFile("ascend/test_ascend_graph_tp2_bf16.py", 400),
        TestFile("ascend/test_ascend_mla_fia_w8a8int8.py", 400),
        TestFile("ascend/test_ascend_tp2_bf16.py", 400),
        TestFile("ascend/test_ascend_tp2_fia_bf16.py", 400),
    ],
    "per-commit-4-ascend-npu": [
        TestFile("ascend/test_ascend_mla_w8a8int8.py", 400),
        TestFile("ascend/test_ascend_tp4_bf16.py", 400),
    ],
    "per-commit-16-ascend-a3": [
        TestFile("ascend/test_ascend_deepep.py", 400),
        TestFile("ascend/test_ascend_deepseek_mtp.py", 400),
    ],
    "__not_in_ascend_ci__": [
        TestFile("ascend/test_mindspore_models.py"),
    ],
}

suites.update(suite_amd)
suites.update(suite_xeon)
suites.update(suite_ascend)
suites.update(suite_xpu)


def auto_partition(files, rank, size):
    """
    Partition files into size sublists with approximately equal sums of estimated times
    using stable sorting, and return the partition for the specified rank.

    Args:
        files (list): List of file objects with estimated_time attribute
        rank (int): Index of the partition to return (0 to size-1)
        size (int): Number of partitions

    Returns:
        list: List of file objects in the specified rank's partition
    """
    weights = [f.estimated_time for f in files]

    if not weights or size <= 0 or size > len(weights):
        return []

    # Create list of (weight, original_index) tuples
    # Using negative index as secondary key to maintain original order for equal weights
    indexed_weights = [(w, -i) for i, w in enumerate(weights)]
    # Stable sort in descending order by weight
    # If weights are equal, larger (negative) index comes first (i.e., earlier original position)
    indexed_weights = sorted(indexed_weights, reverse=True)

    # Extract original indices (negate back to positive)
    indexed_weights = [(w, -i) for w, i in indexed_weights]

    # Initialize partitions and their sums
    partitions = [[] for _ in range(size)]
    sums = [0.0] * size

    # Greedy approach: assign each weight to partition with smallest current sum
    for weight, idx in indexed_weights:
        # Find partition with minimum sum
        min_sum_idx = sums.index(min(sums))
        partitions[min_sum_idx].append(idx)
        sums[min_sum_idx] += weight

    # Return the files corresponding to the indices in the specified rank's partition
    indices = partitions[rank]
    return [files[i] for i in indices]


def _sanity_check_suites(suites):
    dir_base = Path(__file__).parent
    disk_files = set(
        [
            str(x.relative_to(dir_base))
            for x in dir_base.glob("**/*.py")
            if x.name.startswith("test_")
        ]
    )

    suite_files = set(
        [test_file.name for _, suite in suites.items() for test_file in suite]
    )

    missing_files = sorted(list(disk_files - suite_files))
    missing_text = "\n".join(f'TestFile("{x}"),' for x in missing_files)
    assert len(missing_files) == 0, (
        f"Some test files are not in test suite. "
        f"If this is intentional, please add the following to `not_in_ci` section:\n"
        f"{missing_text}"
    )


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--timeout-per-file",
        type=int,
        default=1200,
        help="The time limit for running one file in seconds.",
    )
    arg_parser.add_argument(
        "--suite",
        type=str,
        default=list(suites.keys())[0],
        choices=list(suites.keys()) + ["all"],
        help="The suite to run",
    )
    arg_parser.add_argument(
        "--auto-partition-id",
        type=int,
        help="Use auto load balancing. The part id.",
    )
    arg_parser.add_argument(
        "--auto-partition-size",
        type=int,
        help="Use auto load balancing. The number of parts.",
    )
    arg_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=False,
        help="Continue running remaining tests even if one fails (useful for nightly tests)",
    )
    args = arg_parser.parse_args()
    print(f"{args=}")

    _sanity_check_suites(suites)

    if args.suite == "all":
        files = glob.glob("**/test_*.py", recursive=True)
    else:
        files = suites[args.suite]

    if args.auto_partition_size:
        files = auto_partition(files, args.auto_partition_id, args.auto_partition_size)

    print("The running tests are ", [f.name for f in files])

    exit_code = run_unittest_files(files, args.timeout_per_file, args.continue_on_error)
    exit(exit_code)


if __name__ == "__main__":
    main()
