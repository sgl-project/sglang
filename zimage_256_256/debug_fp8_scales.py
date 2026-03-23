#!/usr/bin/env python3
"""
FP8 Scale Loading Diagnostic Script
====================================
诊断 FP8 weight_scale_inv 是否被正确加载到模型中。

运行方式（在 GPU 机器上）：
    python zimage_256_256/debug_fp8_scales.py

输出：
  Part A — checkpoint 中的 scale key 及其统计信息
  Part B — 模型加载后 scale 参数状态（是否仍为默认初始值）
  Part C — 加载过程中 skipped / missing keys 的详细日志
"""

import glob
import logging
import sys
from pathlib import Path

import torch

# ============================================================
# Configuration
# ============================================================
MODEL_DIR = "/mnt/geminihzceph/rhyshen/models/Z-Image-Turbo"
FP8_DIR = f"{MODEL_DIR}/transformer-FP8-block128"
FP8_NOFFN_DIR = f"{MODEL_DIR}/transformer-FP8-block128-no-ffn"

# Default init value used in fp8.py:274
DEFAULT_INIT_VALUE = torch.finfo(torch.float32).min  # -3.40282e+38


# ============================================================
# Part A: 检查 checkpoint 中的 scale 键名
# ============================================================
def check_checkpoint_scales():
    print("=" * 72)
    print("Part A: Checkpoint scale keys in safetensors files")
    print("=" * 72)

    from safetensors import safe_open

    for label, d in [("FP8+FFN", FP8_DIR), ("FP8-noFFN", FP8_NOFFN_DIR)]:
        print(f"\n--- {label}: {d} ---")
        safetensors_files = sorted(glob.glob(f"{d}/*.safetensors"))
        if not safetensors_files:
            print(f"  ⚠️  No safetensors files found in {d}")
            continue

        scale_count = 0
        weight_count = 0
        for f in safetensors_files:
            with safe_open(f, framework="pt") as sf:
                for key in sorted(sf.keys()):
                    if "scale" in key:
                        t = sf.get_tensor(key)
                        print(f"  SCALE: {key}")
                        print(
                            f"         shape={t.shape}, dtype={t.dtype}, "
                            f"min={t.min().item():.6g}, max={t.max().item():.6g}, "
                            f"mean={t.mean().item():.6g}"
                        )
                        scale_count += 1
                    elif key.endswith(".weight"):
                        t = sf.get_tensor(key)
                        weight_count += 1
                        if t.dtype == torch.float8_e4m3fn:
                            print(f"  FP8_W: {key}  shape={t.shape} dtype={t.dtype}")

        print(f"\n  Summary: {scale_count} scale keys, {weight_count} weight keys")

        # Check for naming patterns
        print(f"\n  Scale key patterns:")
        scale_keys = []
        for f in safetensors_files:
            with safe_open(f, framework="pt") as sf:
                for key in sf.keys():
                    if "scale" in key:
                        scale_keys.append(key)

        # Analyze: are scales named w1/w3 or w13?
        w1_scales = [k for k in scale_keys if ".w1." in k]
        w3_scales = [k for k in scale_keys if ".w3." in k]
        w13_scales = [k for k in scale_keys if ".w13." in k]
        print(f"    w1 scales: {len(w1_scales)}")
        print(f"    w3 scales: {len(w3_scales)}")
        print(f"    w13 scales: {len(w13_scales)}")
        if w1_scales:
            print(f"    Example w1 scale: {w1_scales[0]}")
        if w13_scales:
            print(f"    Example w13 scale: {w13_scales[0]}")


# ============================================================
# Part B: 检查模型加载后的 scale 状态
# ============================================================
def check_model_loaded_scales():
    print("\n" + "=" * 72)
    print("Part B: Model loaded scale state (after sglang loading)")
    print("=" * 72)

    # Enable verbose logging to catch skipped keys
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    for name in ["sglang", "sglang.multimodal_gen"]:
        logging.getLogger(name).setLevel(logging.DEBUG)

    print("\n--- Loading transformer directly via TransformerLoader ---")
    try:
        from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
            TransformerLoader,
        )
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        # Build ServerArgs with FP8 config
        server_args = ServerArgs.from_kwargs(
            model_path=MODEL_DIR,
            transformer_weights_path=FP8_DIR,
        )

        # Use TransformerLoader to load the transformer model directly
        loader = TransformerLoader()
        component_model_path = f"{MODEL_DIR}/transformer"
        transformer = loader.load_customized(
            component_model_path=component_model_path,
            server_args=server_args,
            component_name="transformer",
        )

        # Check all scale parameters
        total_scales = 0
        default_scales = 0
        correct_scales = 0

        print("\nScale parameters in loaded model:")
        for name, param in transformer.named_parameters():
            if "scale" in name:
                total_scales += 1
                data = param.data.float()
                is_default = (data == DEFAULT_INIT_VALUE).all().item()
                is_all_negative = (data < 0).all().item()

                status = ""
                if is_default:
                    status = "❌ DEFAULT_INIT (scale NOT loaded!)"
                    default_scales += 1
                elif is_all_negative:
                    status = "⚠️  ALL NEGATIVE (suspicious)"
                else:
                    status = "✅ OK"
                    correct_scales += 1

                print(f"  {name}:")
                print(f"    shape={data.shape}, dtype={param.dtype}")
                print(
                    f"    min={data.min().item():.6g}, max={data.max().item():.6g}, "
                    f"mean={data.mean().item():.6g}"
                )
                print(f"    status: {status}")

        print(
            f"\n  Summary: {total_scales} scale params, "
            f"{correct_scales} OK, {default_scales} DEFAULT (broken)"
        )

        if total_scales == 0:
            print("\n  ⚠️⚠️⚠️  NO SCALE PARAMS FOUND!")
            print(
                "  This means quant_config was not resolved → FP8 layers not initialized"
            )
            print(
                "  Check: does transformer-FP8-block128/config.json have quantization_config?"
            )
        elif default_scales > 0:
            print("\n  ⚠️⚠️⚠️  Some FP8 scales are NOT being loaded!")
            print(
                f"  {default_scales}/{total_scales} scales still at default init value"
            )
        else:
            print("\n  ✅ All FP8 scales loaded correctly!")

        # Cleanup
        del transformer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n  ❌ Failed to load model: {e}")
        import traceback

        traceback.print_exc(file=sys.stdout)


# ============================================================
# Part C: 检查 hf_to_custom_state_dict 的 mapping 结果
# ============================================================
def check_mapping_simulation():
    print("\n" + "=" * 72)
    print("Part C: Simulate param_names_mapping on checkpoint keys")
    print("=" * 72)

    from safetensors import safe_open

    # Get all checkpoint keys from FP8 dir
    all_keys = []
    for f in sorted(glob.glob(f"{FP8_DIR}/*.safetensors")):
        with safe_open(f, framework="pt") as sf:
            all_keys.extend(sf.keys())

    # Import and build the mapping function
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
    from sglang.multimodal_gen.configs.models.dits.zimage import ZImageArchConfig
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

    config = ZImageArchConfig()
    mapping_fn = get_param_names_mapping(config.param_names_mapping)

    print("\nMapping results for scale-related keys:")
    unmapped_scales = []
    mapped_scales = []
    passthrough_scales = []
    for key in sorted(all_keys):
        if "scale" not in key:
            continue
        target_name, merge_idx, num_merge = mapping_fn(key)
        was_mapped = target_name != key

        # Classify: keys with w1/w3 NEED mapping; others (to_k, to_q, to_v, to_out, w2)
        # are pass-through and should match model params directly
        needs_mapping = (".w1." in key or ".w3." in key) and "w13" not in key

        if was_mapped:
            status = "✅ MAPPED"
            mapped_scales.append(key)
        elif needs_mapping:
            status = "❌ NEEDS MAPPING but NOT MAPPED!"
            unmapped_scales.append(key)
        else:
            status = "➡️  PASS-THROUGH (no rename needed)"
            passthrough_scales.append(key)

        print(f"  {key}")
        print(
            f"    -> target={target_name}, merge_idx={merge_idx}, num_merge={num_merge}"
        )
        print(f"    {status}")

    print(f"\n  Mapped (w1/w3→w13): {len(mapped_scales)}")
    print(f"  Pass-through (to_k/to_q/to_v/to_out/w2): {len(passthrough_scales)}")
    print(f"  Missing mapping: {len(unmapped_scales)}")
    if unmapped_scales:
        print(f"\n  ⚠️  {len(unmapped_scales)} scale keys NEED mapping but have none!")
        for k in unmapped_scales[:5]:
            print(f"    - {k}")
    else:
        print("\n  ✅ All scale keys that need mapping are correctly mapped.")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("FP8 Scale Loading Diagnostic")
    print(f"Model dir: {MODEL_DIR}")
    print(f"FP8 dir: {FP8_DIR}")
    print(f"FP8-noFFN dir: {FP8_NOFFN_DIR}")
    print(f"Default init value: {DEFAULT_INIT_VALUE}")

    # Part A: Always runs (no GPU needed for checkpoint inspection)
    check_checkpoint_scales()

    # Part C: Simulate mapping (no GPU needed)
    check_mapping_simulation()

    # Part B: Requires GPU + sglang
    if "--skip-model-load" not in sys.argv:
        check_model_loaded_scales()
    else:
        print("\n[Skipping Part B: model loading (--skip-model-load flag set)]")

    print("\n" + "=" * 72)
    print("Diagnostic complete.")
    print("=" * 72)
