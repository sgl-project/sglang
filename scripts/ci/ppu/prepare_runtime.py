import hashlib
import os
from pathlib import Path

MMLU_SHA256 = "2182319e74add5855b87c4abf4136310b900f8d2772460054681feadb7f2c52f"
MODEL_PATH_ENVS = {
    "meta-llama/Llama-3.1-8B-Instruct": "PPU_CI_TARGET_MODEL_PATH",
    "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B": "PPU_CI_EAGLE3_MODEL_PATH",
    "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat": "PPU_CI_DFLASH_MODEL_PATH",
}


def prepare_model_cache(hub, checkpoints):
    for model, checkpoint in checkpoints.items():
        checkpoint = Path(checkpoint).resolve(strict=True)
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(
                f"Missing model config: {checkpoint / 'config.json'}"
            )
        root = hub / ("models--" + model.replace("/", "--"))
        (root / "refs").mkdir(parents=True)
        (root / "snapshots").mkdir()
        # Hub reads the ref verbatim, including any trailing newline.
        (root / "refs/main").write_text("main")
        (root / "snapshots/main").symlink_to(checkpoint, target_is_directory=True)


def prepare_mmlu_cache(source, destination):
    data = (source / "mmlu/test.jsonl").read_bytes()
    marker = (source / "mmlu/test.jsonl.sha256").read_text()
    if hashlib.sha256(data).hexdigest() != MMLU_SHA256 or marker != MMLU_SHA256 + "\n":
        raise ValueError(f"MMLU cache checksum mismatch: {source}")
    (destination / "mmlu").mkdir(parents=True)
    (destination / "mmlu/test.jsonl").write_bytes(data)
    (destination / "mmlu/test.jsonl.sha256").write_text(marker)


def main():
    from huggingface_hub import hf_hub_download
    from sgl_eval.evals._loader import load_via_prepare
    from sgl_eval.registry import get

    home = Path.home()
    hub = Path(os.environ["HF_HUB_CACHE"])
    checkpoints = {model: os.environ[key] for model, key in MODEL_PATH_ENVS.items()}
    prepare_model_cache(hub, checkpoints)
    for model in checkpoints:
        hf_hub_download(model, "config.json", cache_dir=hub, local_files_only=True)
    prepare_mmlu_cache(Path(os.environ["PPU_CI_EVAL_CACHE"]), home / ".cache/sgl_eval")
    assert get("mmlu").name == "mmlu", "MMLU evaluator must be registered"
    examples = load_via_prepare("mmlu", ("test",), sample_seed=0)(200)
    assert len(examples) == 200, "MMLU must load all 200 evaluation examples"
    print(
        "Offline model references and independent MMLU cache are ready (200 examples)."
    )


if __name__ == "__main__":
    main()
