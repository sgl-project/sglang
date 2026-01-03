"""
test_lora_format_adapter.py

Small regression test for the LoRA format adapter.

It downloads several public LoRA checkpoints from Hugging Face, runs
format detection and normalization, and prints a compact summary table.
"""

import logging
import os
import tempfile
from typing import Dict, List

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from sglang.multimodal_gen.runtime.pipelines_core.lora_format_adapter import (
    LoRAFormat,
    detect_lora_format_from_state_dict,
    normalize_lora_state_dict,
)

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("lora_test")

ROOT_DIR = os.path.join(tempfile.gettempdir(), "sglang_lora_tests")
os.makedirs(ROOT_DIR, exist_ok=True)


def download_lora(
    repo_id: str,
    filename: str,
    local_name: str,
) -> str:
    """
    Download a LoRA safetensors file into ROOT_DIR and return its local path.
    """
    print(f"=== Downloading LoRA from {repo_id} ({filename}) ===")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=ROOT_DIR,
        local_dir_use_symlinks=False,
    )
    dst = os.path.join(ROOT_DIR, local_name)
    if os.path.abspath(path) != os.path.abspath(dst):
        try:
            import shutil

            shutil.copy2(path, dst)
        except Exception:
            dst = path
    print(f"Saved to: {dst}")
    return dst


def is_diffusers_style_keys(
    sd: Dict[str, torch.Tensor],
    debug_name: str = "",
) -> bool:
    """
    Relaxed structural check that a state_dict looks like diffusers-style LoRA.

    The check verifies:
    1) No known non-diffusers prefixes.
    2) No non-diffusers suffixes such as alpha / dora_scale / magnitude vectors.
    3) Most top-level roots match common diffusers module namespaces.
    """
    if not sd:
        print(f"[{debug_name}] diffusers-style check: EMPTY state_dict")
        return False

    keys: List[str] = list(sd.keys())
    total = len(keys)

    banned_prefixes = (
        "lora_unet_",
        "lora_te_",
        "lora_te1_",
        "lora_te2_",
        "lora_unet_double_blocks_",
        "lora_unet_single_blocks_",
    )
    bad_prefix_keys = [k for k in keys if k.startswith(banned_prefixes)]
    cond1 = len(bad_prefix_keys) == 0

    banned_suffixes = (
        ".alpha",
        ".dora_scale",
        ".lora_magnitude_vector",
    )
    bad_suffix_keys = [k for k in keys if k.endswith(banned_suffixes)]
    cond2 = len(bad_suffix_keys) == 0

    allowed_roots = {
        "unet",
        "text_encoder",
        "text_encoder_2",
        "transformer",
        "prior",
        "image_encoder",
        "vae",
        "diffusion_model",
    }
    root_names = [k.split(".", 1)[0] for k in keys]
    root_ok_count = sum(r in allowed_roots for r in root_names)
    cond3 = root_ok_count >= 0.6 * total

    ok = cond1 and cond2 and cond3

    if not ok:
        print(f"[{debug_name}] diffusers-style check FAILED (relaxed):")
        print(f"  total keys = {total}")
        print(
            f"  cond1(no banned prefixes)  = {cond1}, bad_prefix_keys={len(bad_prefix_keys)}"
        )
        if not cond1 and bad_prefix_keys:
            print("    example bad prefix key:", bad_prefix_keys[0])
        print(
            f"  cond2(no banned suffixes)  = {cond2}, bad_suffix_keys={len(bad_suffix_keys)}"
        )
        if not cond2 and bad_suffix_keys:
            print("    example bad suffix key:", bad_suffix_keys[0])
        print(f"  cond3(allowed roots>=60%)  = {cond3}, root_ok_count={root_ok_count}")
    return ok


def run_single_test(
    name: str,
    repo_id: str,
    filename: str,
    local_name: str,
    expected_before: LoRAFormat,
    expected_after: LoRAFormat = LoRAFormat.STANDARD,
):
    """
    Run a single end-to-end test for one LoRA checkpoint.

    Steps:
    1) Download.
    2) Detect format on raw keys.
    3) Normalize via lora_format_adapter.
    4) Detect again on the normalized dict.
    5) Optionally check for diffusers-style key structure.
    """
    logger.info(f"=== Running test: {name} ===")
    local_path = download_lora(repo_id, filename, local_name)
    raw_state = load_file(local_path)

    detected_before = detect_lora_format_from_state_dict(raw_state)
    norm_state = normalize_lora_state_dict(raw_state, logger=logger)
    detected_after = detect_lora_format_from_state_dict(norm_state)
    standard_like = is_diffusers_style_keys(norm_state, debug_name=name)

    passed = detected_before == expected_before and detected_after == expected_after

    return {
        "name": name,
        "expected_before": expected_before.value,
        "detected_before": detected_before.value,
        "expected_after": expected_after.value,
        "detected_after": detected_after.value,
        "standard_like_keys": standard_like,
        "pass": passed,
        "num_keys_raw": len(raw_state),
        "num_keys_norm": len(norm_state),
    }


def _run_all_tests() -> List[Dict]:
    results: List[Dict] = []

    # SDXL LoRA that is already in diffusers/PEFT format.
    results.append(
        run_single_test(
            name="HF standard SDXL LoRA",
            repo_id="jbilcke-hf/sdxl-cinematic-1",
            filename="pytorch_lora_weights.safetensors",
            local_name="sdxl_cinematic1_pytorch_lora_weights.safetensors",
            expected_before=LoRAFormat.STANDARD,
            expected_after=LoRAFormat.STANDARD,
        )
    )

    # XLabs FLUX LoRA (non-diffusers → diffusers).
    results.append(
        run_single_test(
            name="XLabs FLUX Realism LoRA",
            repo_id="XLabs-AI/flux-RealismLora",
            filename="lora.safetensors",
            local_name="flux_realism_lora.safetensors",
            expected_before=LoRAFormat.XLABS_FLUX,
            expected_after=LoRAFormat.STANDARD,
        )
    )

    # Kohya-style FLUX LoRA (sd-scripts flux_lora.py → diffusers).
    results.append(
        run_single_test(
            name="Kohya-style Flux LoRA",
            repo_id="kohya-ss/misc-models",
            filename="flux-hasui-lora-d4-sigmoid-raw-gs1.0.safetensors",
            local_name="flux_hasui_lora_d4_sigmoid_raw_gs1_0.safetensors",
            expected_before=LoRAFormat.KOHYA_FLUX,
            expected_after=LoRAFormat.STANDARD,
        )
    )

    # Classic Kohya/A1111 SD LoRA (non-diffusers SD → diffusers).
    results.append(
        run_single_test(
            name="Kohya-style SD LoRA",
            repo_id="kohya-ss/misc-models",
            filename="fp-1f-chibi-1024.safetensors",
            local_name="fp_1f_chibi_1024.safetensors",
            expected_before=LoRAFormat.NON_DIFFUSERS_SD,
            expected_after=LoRAFormat.STANDARD,
        )
    )

    # Wan2.1 Fun Reward LoRA (ComfyUI format → diffusers).
    results.append(
        run_single_test(
            name="Wan2.1 Fun Reward LoRA (Comfy)",
            repo_id="alibaba-pai/Wan2.1-Fun-Reward-LoRAs",
            filename="Wan2.1-Fun-1.3B-InP-MPS.safetensors",
            local_name="wan21_fun_1_3b_inp_mps.safetensors",
            expected_before=LoRAFormat.NON_DIFFUSERS_SD,
            expected_after=LoRAFormat.STANDARD,
        )
    )

    # Qwen-Image EVA LoRA (already diffusers/PEFT-style).
    results.append(
        run_single_test(
            name="Qwen-Image EVA LoRA",
            repo_id="starsfriday/Qwen-Image-EVA-LoRA",
            filename="qwen_image_eva.safetensors",
            local_name="qwen_image_eva.safetensors",
            expected_before=LoRAFormat.STANDARD,
            expected_after=LoRAFormat.STANDARD,
        )
    )

    # Qwen-Image Lightning LoRA (non-diffusers Qwen → diffusers).
    results.append(
        run_single_test(
            name="Qwen-Image Lightning LoRA",
            repo_id="lightx2v/Qwen-Image-Lightning",
            filename="Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors",
            local_name="qwen_image_lightning_4steps_v1_bf16.safetensors",
            expected_before=LoRAFormat.NON_DIFFUSERS_SD,
            expected_after=LoRAFormat.STANDARD,
        )
    )

    # Classic Painting Z-Image Turbo LoRA (Z-Image family).
    results.append(
        run_single_test(
            name="Classic Painting Z-Image LoRA",
            repo_id="renderartist/Classic-Painting-Z-Image-Turbo-LoRA",
            filename="Classic_Painting_Z_Image_Turbo_v1_renderartist_1750.safetensors",
            local_name="classic_painting_z_image_turbo_v1_renderartist_1750.safetensors",
            expected_before=LoRAFormat.STANDARD,
            expected_after=LoRAFormat.STANDARD,
        )
    )

    return results


def _print_summary(results: List[Dict]) -> None:
    print("\n================ LoRA format adapter test ================")

    header = (
        f"{'Test Name':30} "
        f"{'Exp(b)':12} "
        f"{'Act(b)':12} "
        f"{'Exp(a)':12} "
        f"{'Act(a)':12} "
        f"{'StdLike':8} "
        f"{'#Raw':7} "
        f"{'#Norm':7} "
        f"{'PASS':5}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['name'][:30]:30} "
            f"{r['expected_before'][:12]:12} "
            f"{r['detected_before'][:12]:12} "
            f"{r['expected_after'][:12]:12} "
            f"{r['detected_after'][:12]:12} "
            f"{str(r['standard_like_keys']):8} "
            f"{r['num_keys_raw']:7d} "
            f"{r['num_keys_norm']:7d} "
            f"{str(r['pass']):5}"
        )

    print("=========================================================\n")


def main() -> None:
    results = _run_all_tests()
    _print_summary(results)

    if not all(r["pass"] for r in results):
        raise SystemExit(1)


class TestLoRAFormatAdapter:
    def test_lora_format_adapter_all_formats(self):
        results = _run_all_tests()
        assert all(
            r["pass"] for r in results
        ), "At least one LoRA format adapter case failed"


if __name__ == "__main__":
    main()
