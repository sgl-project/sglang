"""Pinned, offline, real-weight Engine smoke for a Bazel GPU devbox."""

import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from huggingface_hub import snapshot_download  # noqa: E402

import sglang as sgl  # noqa: E402

MODEL_ID = "Qwen/Qwen2-0.5B"
MODEL_REVISION = "91d2aff3f957f99e4c74c962f2f408dcc88a18d8"
MODEL_WEIGHT_BYTES = 988097824
MODEL_WEIGHT_SHA256 = "9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad"

PROMPT = "The capital of the United Kingdom is"
EXPECTED_OUTPUT_IDS = [7148, 13, 1084, 374, 279, 7772, 3283, 304]
EXPECTED_TEXT = " London. It is the largest city in"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_cached_snapshot() -> Path:
    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    if snapshot.name != MODEL_REVISION:
        raise AssertionError(f"expected snapshot {MODEL_REVISION}, resolved {snapshot}")

    required_files = ("config.json", "model.safetensors", "tokenizer.json")
    missing = [name for name in required_files if not (snapshot / name).is_file()]
    if missing:
        raise AssertionError(f"incomplete cached snapshot: missing {missing}")

    config = json.loads((snapshot / "config.json").read_text())
    if config.get("model_type") != "qwen2":
        raise AssertionError(f"unexpected model config: {config!r}")

    weight = snapshot / "model.safetensors"
    if weight.stat().st_size != MODEL_WEIGHT_BYTES:
        raise AssertionError(
            f"unexpected model.safetensors size: {weight.stat().st_size}"
        )
    actual_sha256 = sha256_file(weight)
    if actual_sha256 != MODEL_WEIGHT_SHA256:
        raise AssertionError(f"model.safetensors sha256 mismatch: {actual_sha256}")

    return snapshot


def main() -> None:
    snapshot = resolve_cached_snapshot()
    engine = sgl.Engine(
        model_path=str(snapshot),
        model_impl="sglang",
        load_format="safetensors",
        dtype="float16",
        tp_size=1,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        random_seed=0,
        log_level="error",
        max_running_requests=1,
        max_total_tokens=256,
        mem_fraction_static=0.2,
    )
    try:
        output = engine.generate(
            PROMPT,
            sampling_params={
                "temperature": 0.0,
                "max_new_tokens": len(EXPECTED_OUTPUT_IDS),
                "ignore_eos": True,
            },
        )

        output_ids = list(output["output_ids"])
        if output_ids != EXPECTED_OUTPUT_IDS:
            raise AssertionError(
                f"expected output IDs {EXPECTED_OUTPUT_IDS}, got {output_ids}"
            )
        if output["text"] != EXPECTED_TEXT:
            raise AssertionError(
                f"expected text {EXPECTED_TEXT!r}, got {output['text']!r}"
            )

        meta = output["meta_info"]
        if meta["completion_tokens"] != len(EXPECTED_OUTPUT_IDS):
            raise AssertionError(f"unexpected completion metadata: {meta!r}")
        if meta["finish_reason"]["type"] != "length":
            raise AssertionError(f"unexpected finish reason: {meta!r}")

        print(
            json.dumps(
                {
                    "model_revision": MODEL_REVISION,
                    "output_ids": output_ids,
                    "text": output["text"],
                },
                sort_keys=True,
            )
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
