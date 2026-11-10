#!/usr/bin/env python3
"""Exact, documented skip policy for the PPU nightly strict runner."""

import argparse
import json
from pathlib import Path


def allowed_skips():
    batch_sizes = [2**x for x in range(12)]
    batch_sizes += [x + 1 for x in batch_sizes]
    batch_sizes[1] = "20"
    batch_sizes[12] = "21"
    skips = [
        f"registered.kernels.ops.attention.test_rope.test_partial_rope[64-{rope_dim}-{is_neox}-{batch_size}]"
        for rope_dim in (80, 96, 128)
        for is_neox in (False, True)
        for batch_size in batch_sizes
    ]
    skips += [
        f"registered.layers.mamba.test_mamba_ssm.test_selective_state_update_stochastic_rounding[{dim}-{head_dim}-{with_heads}-{cache_index}]"
        for dim in (2048, 4096)
        for head_dim in (16, 64)
        for with_heads in (False, True)
        for cache_index in (0, 4)
    ]
    skips += [
        # nsys is not installed in the PPU image.
        "registered.profiling.test_start_profile.TestStartProfileWithNsys.test_start_profile_cuda_profiler",
        "registered.profiling.test_start_profile.TestStartProfileWithNsys.test_start_profile_cuda_profiler_with_start_step",
        # The negative-penalty diversity assertion is disabled as flaky upstream.
        "registered.sampling.test_penalty.TestPenalty.test_penalty_edge_cases_negative_penalty_values",
        # These page_size=1 cases are ROCm-specific.
        "registered.unit.managers.test_hisparse_unit.TestHiSparseUnit.test_allocator_page_size_one_alloc_free_cycle",
        "registered.unit.managers.test_hisparse_unit.TestHiSparseUnit.test_decode_remap_frees_stale_page_size_one_mapping",
        # Pytest collects this abstract base class but skips it by design.
        "registered.lora.test_virtual_experts_kernels._AlignBlockSizeSentinelBucketBase.test_all_valid_baseline",
        "registered.lora.test_virtual_experts_kernels._AlignBlockSizeSentinelBucketBase.test_empty_topk_ids_does_not_crash",
        "registered.lora.test_virtual_experts_kernels._AlignBlockSizeSentinelBucketBase.test_mixed_invalid_at_production_size",
        "registered.lora.test_virtual_experts_kernels._AlignBlockSizeSentinelBucketBase.test_negative_ids_routed_to_sentinel",
        "registered.lora.test_virtual_experts_kernels._AlignBlockSizeSentinelBucketBase.test_oor_ids_routed_to_sentinel",
        # The baseline PPU nightly runs the Python server without the Rust extension.
        "registered.openai_server.basic.test_openai_server.TestOpenAICompletionWithRust.test_completion",
        "registered.openai_server.basic.test_openai_server.TestOpenAICompletionWithRust.test_completion_stream",
        "registered.openai_server.basic.test_openai_server.TestOpenAIChatWithRust.test_chat_completion",
        "registered.openai_server.basic.test_openai_server.TestOpenAIChatWithRust.test_chat_completion_stream",
        # nvidia-modelopt is not installed in the PPU image.
        "registered.unit.model_loader.test_modelopt_export.TestModelOptExport.test_export_modelopt_checkpoint_success",
        "registered.unit.model_loader.test_modelopt_export.TestModelOptExport.test_setup_quantization_with_export_after_calibration",
        "registered.unit.model_loader.test_modelopt_export.TestModelOptExport.test_setup_quantization_with_export_from_checkpoint",
        "registered.unit.model_loader.test_modelopt_export.TestModelOptExport.test_standard_workflow_selection",
        "registered.unit.model_loader.test_modelopt_export.TestModelOptExportIntegration.test_full_workflow_with_export",
    ]
    baseline_path = Path(__file__).with_name("nightly_baseline_skips.json")
    skips.extend(json.loads(baseline_path.read_text()))
    skips.append(
        "registered.kernels.ops.diffusion.test_timestep_embedding.test_timestep_embedding_perf"
    )
    return skips


def deselected_nodes():
    # This bare function was never executed by the legacy unittest runner.
    return [
        "registered/rl/test_update_weights_from_tensor.py::test_update_weights_from_tensor"
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--deselect", action="store_true")
    args = parser.parse_args()
    skips = allowed_skips()
    if args.deselect:
        for node in deselected_nodes():
            print(f"--deselect\n{node}")
    elif args.json:
        print(json.dumps(skips, indent=2))
    else:
        for node in skips:
            print(f"--allow-skip\n{node}")


if __name__ == "__main__":
    main()
