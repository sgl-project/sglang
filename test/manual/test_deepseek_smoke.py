"""
Quick smoke test for DeepSeek refactoring changes.
Run: python test/manual/test_deepseek_smoke.py
"""

import importlib.util

import torch


def main():
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected (tests will still run)")

    # Test module availability using importlib
    required_modules = [
        "sglang.srt.models.deepseek_v2",
        "sglang.srt.models.deepseek_common.utils",
        "sglang.srt.models.deepseek_nextn",
    ]

    for module_name in required_modules:
        if importlib.util.find_spec(module_name) is None:
            print(f"Module {module_name} not found")
            return 1

    print("All required modules are available")

    try:
        from sglang.srt.models.deepseek_common.utils import (
            FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
            NVFP4_CKPT_FP8_ATTN_QUANT_MODULES,
            _get_llama_4_scaling,
            _is_cublas_ge_129,
            enable_nextn_moe_bf16_cast_to_fp8,
            yarn_get_mscale,
        )

    except ImportError as e:
        print(f"Import failed: {e}")
        return 1

    print("\n--- Testing Utility Functions ---")
    try:
        # Test yarn_get_mscale
        result = yarn_get_mscale(2.0, 1.0)
        print(f"yarn_get_mscale(2.0, 1.0) = {result:.4f}")

        # Test _get_llama_4_scaling
        positions = torch.tensor([0, 8192])
        scaling = _get_llama_4_scaling(8192, 0.5, positions)
        print(f"_get_llama_4_scaling shape: {scaling.shape}")

        # Test constants
        print(f"NVFP4 modules: {NVFP4_CKPT_FP8_ATTN_QUANT_MODULES}")
        print(
            f"Attention backends count: {len(FORWARD_ABSORB_CORE_ATTENTION_BACKENDS)}"
        )
        print(f"CuBLAS >= 12.9: {_is_cublas_ge_129}")

        # Test enable_nextn_moe_bf16_cast_to_fp8
        result = enable_nextn_moe_bf16_cast_to_fp8(None)
        print(f"enable_nextn_moe_bf16_cast_to_fp8(None) = {result}")

    except Exception as e:
        print(f"Function test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
