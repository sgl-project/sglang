"""Compare the original and refactored Qwen async processor boundaries bit for bit.

    pinned benchmark rows + rendered prompts + SHA-256 checked local media
                     |                         |
        legacy route subprocess        candidate route subprocess
        full input_text every turn     full input_ids every turn
                     |                         |
        legacy Qwen class          serving factory -> Qwen token-space processor
                     |                         |
        process_mm_data_async      process_mm_data_async
                     |                         |
           final IDs + every output/item field + tensor shape/dtype/raw bytes
                     +---------- exact comparison ----------+

The same Python environment and checkpoint processor files serve both captures.
Every turn supplies its full ordered media URL lists without session recipes.
All output fields are compared unchanged, including item grouping, offsets and
mRoPE; cache identity (item hash / pad value, padded IDs) only when both set it.
No model weights or generation are needed. CUDA CI runs a small pinned matrix
through both real routes: Qwen3.5 and Qwen2.5-VL with two images, Qwen3-VL with
an image then a video, Qwen3-Omni with an image then one audio clip; one backend
per model covers PIL and the GPU path.
Turns are joined at an explicit assistant end token.
Both routes run from the same clean checkout; only the candidate enables the
startup strategy.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import struct
import subprocess
import sys
from enum import Enum
from pathlib import Path

import msgspec
import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "type": "torch.Tensor",
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "bytes": tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": value.shape,
            "dtype": value.dtype.str,
            "bytes": value.tobytes(order="C"),
        }
    if isinstance(value, msgspec.Struct):
        return _snapshot(msgspec.structs.asdict(value))
    if isinstance(value, Enum):
        return (type(value).__name__, value.name)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    if isinstance(value, np.generic):
        return ("numpy.scalar", value.dtype.str, value.tobytes())
    if isinstance(value, float):
        return ("float64", struct.pack("!d", value))
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    raise TypeError(f"Uncaptured processor output type: {type(value).__qualname__}")


# Cache identity: legacy may key a hash by its artifact, token-space by tensor
# content. Both are valid, so these are compared only when both sides set them.
_CACHE_IDENTITY_ITEM_FIELDS = ("hash", "pad_value", "pad_values", "identity")


def _drop_unshared_cache_identity(reference_cases, candidate_cases):
    for case_id in reference_cases.keys() & candidate_cases.keys():
        sides = [
            cases[case_id].get("model_inputs")
            for cases in (reference_cases, candidate_cases)
            if isinstance(cases[case_id], dict)
        ]
        if len(sides) != 2 or None in sides:
            continue
        if any(side.get("padded_input_ids") is None for side in sides):
            for side in sides:
                side.pop("padded_input_ids", None)
        for items in zip(*(side.get("mm_items") or [] for side in sides)):
            for name in _CACHE_IDENTITY_ITEM_FIELDS:
                if any(item.get(name) is None for item in items):
                    for item in items:
                        item.pop(name, None)


def _assert_identical(reference, candidate, field="output"):
    assert type(reference) is type(candidate), f"{field}: types differ"
    if isinstance(reference, dict):
        assert reference.keys() == candidate.keys(), (
            f"{field}: fields differ: "
            f"reference-only={reference.keys() - candidate.keys()}, "
            f"candidate-only={candidate.keys() - reference.keys()}"
        )
        for key in reference:
            _assert_identical(reference[key], candidate[key], f"{field}.{key}")
    elif isinstance(reference, (list, tuple)):
        assert len(reference) == len(candidate), f"{field}: lengths differ"
        for index, (old, new) in enumerate(zip(reference, candidate)):
            _assert_identical(old, new, f"{field}[{index}]")
    elif isinstance(reference, bytes):
        assert reference == candidate, f"{field}: raw bytes differ"
    else:
        assert reference == candidate, f"{field}: {reference!r} != {candidate!r}"


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _checkpoint_files(checkpoint):
    weight_suffixes = {".safetensors", ".bin", ".pt", ".pth", ".onnx", ".gguf"}
    return {
        str(path.relative_to(checkpoint)): _sha256(path)
        for path in sorted(checkpoint.rglob("*"))
        if path.is_file()
        and path.suffix not in weight_suffixes
        and ".cache" not in path.parts
        and ".git" not in path.parts
    }


def _checkout_identity(checkout):
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
        ).strip(),
        "diff": subprocess.check_output(
            ["git", "diff", "HEAD", "--", "python"], cwd=checkout, text=True
        ),
    }


async def _capture(checkout, manifest_path, route):
    import sglang
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
    from sglang.srt.runtime_context import publish, reset_context
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils.hf_transformers import get_config, get_processor

    assert Path(sglang.__file__).resolve().is_relative_to(checkout / "python")
    source = _checkout_identity(checkout)
    if route == "legacy":
        assert not source["diff"], "Legacy capture requires clean source"
    manifest = json.loads(manifest_path.read_text())
    checkpoint = (manifest_path.parent / manifest["checkpoint"]["path"]).resolve()
    seed = manifest["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    backend = manifest["image_processor_backend"]
    server_args = ServerArgs(
        model_path=str(checkpoint),
        model_impl="sglang",
        trust_remote_code=manifest.get("trust_remote_code", False),
        image_processor_backend=backend,
        disable_fast_image_processor=backend == "pil",
        mm_feature_transport="cpu",
        mm_preprocess_cache_size_mb=0,
        mm_io_worker_num=1,
        mm_processor_worker_num=1,
        mm_process_config=manifest["mm_process_config"],
        **({"enable_token_space_processor": True} if route == "candidate" else {}),
    )
    publish(server_args, role="tokenizer")
    hf_config = get_config(
        str(checkpoint),
        trust_remote_code=server_args.trust_remote_code,
        local_files_only=True,
    )
    hf_processor = get_processor(
        str(checkpoint),
        trust_remote_code=server_args.trust_remote_code,
        image_processor_backend=backend,
        local_files_only=True,
    )
    if route == "candidate":
        from sglang.srt.managers.multimodal_processor import get_mm_processor
        from sglang.srt.multimodal.token_space_mm_process.models.qwen_vl import (
            QwenTokenSpaceProcessStrategy,
        )
        from sglang.srt.multimodal.token_space_mm_process.sglang import (
            TokenSpaceSGLangProcessor,
        )

        processor = get_mm_processor(
            hf_config, server_args, hf_processor, None, skip_mm_pool=True
        )
        assert type(processor) is TokenSpaceSGLangProcessor.for_strategy(
            QwenTokenSpaceProcessStrategy
        )
    else:
        processor = QwenVLImageProcessor(
            hf_config, server_args, hf_processor, None, skip_mm_pool=True
        )
    captures = {}
    try:
        for sample in manifest["samples"]:
            previous_ids, previous_expanded_ids = [], []
            previous_media = {modality: [] for modality in ("image", "video", "audio")}
            for turn_index, turn in enumerate(sample["turns"]):
                case_id = f"{sample['id']}/turn-{turn_index}"
                assert case_id not in captures, f"Duplicate fixture: {case_id}"
                sources = {}
                for modality in previous_media:
                    files = turn["media"][modality]
                    assert (
                        files[: len(previous_media[modality])]
                        == previous_media[modality]
                    )
                    sources[modality] = []
                    for entry in files:
                        path = (manifest_path.parent / entry["path"]).resolve()
                        assert _sha256(path) == entry["sha256"], (
                            f"Changed media: {path}"
                        )
                        sources[modality].append(str(path))
                prompt = turn["prompt"]
                add_special_tokens = not (
                    hf_processor.tokenizer.bos_token
                    and prompt.startswith(hf_processor.tokenizer.bos_token)
                )
                input_ids = hf_processor.tokenizer.encode(
                    prompt, add_special_tokens=add_special_tokens
                )
                assert input_ids[: len(previous_ids)] == previous_ids, (
                    f"{case_id}: cumulative prompt does not preserve the prior token prefix"
                )
                if route == "legacy":
                    request = GenerateReqInput(
                        text=prompt,
                        image_data=sources["image"],
                        video_data=sources["video"],
                        audio_data=sources["audio"],
                    )
                    output = await processor.process_mm_data_async(
                        image_data=request.image_data,
                        input_text=prompt,
                        request_obj=request,
                    )
                else:
                    request = GenerateReqInput(
                        input_ids=input_ids,
                        image_data=sources["image"],
                        video_data=sources["video"],
                        audio_data=sources["audio"],
                    )
                    output = await processor.process_mm_data_async(
                        image_data=request.image_data,
                        audio_data=request.audio_data,
                        input_text="",
                        input_ids=input_ids,
                        request_obj=request,
                    )
                    boundary = len(previous_expanded_ids)
                    assert output.input_ids[:boundary] == previous_expanded_ids
                model_inputs = msgspec.structs.asdict(output)
                captures[case_id] = {
                    "unexpanded_input_ids": input_ids,
                    "model_inputs": _snapshot(model_inputs),
                }
                previous_ids = input_ids
                previous_expanded_ids = list(output.input_ids)
                previous_media = turn["media"]
    finally:
        processor.shutdown()
        reset_context()
    return {
        "route": route,
        "source": source,
        "manifest": manifest,
        "checkpoint_files": _checkpoint_files(checkpoint),
        "dependencies": {
            distribution.metadata["Name"]: distribution.version
            for distribution in importlib.metadata.distributions()
        },
        "python": sys.version,
        "cases": captures,
    }


def _compare_routes(checkout, manifest_path, artifacts):
    artifacts.mkdir(parents=True, exist_ok=True)
    snapshots = []
    seed = json.loads(manifest_path.read_text())["seed"]
    for route in ("legacy", "candidate"):
        output_path = artifacts / f"{route}.pickle"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--capture",
                route,
                "--checkout",
                str(checkout),
                "--manifest",
                str(manifest_path),
                "--output",
                str(output_path),
            ],
            cwd=checkout,
            env={
                **os.environ,
                "PYTHONPATH": str(checkout / "python"),
                "PYTHONHASHSEED": str(seed),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            check=True,
        )
        with output_path.open("rb") as stream:
            snapshots.append(pickle.load(stream))
    _assert_identical(snapshots[0]["source"], snapshots[1]["source"], "source")
    assert not snapshots[0]["source"]["diff"], "Compared source must be clean"
    _drop_unshared_cache_identity(snapshots[0]["cases"], snapshots[1]["cases"])
    for field in ("manifest", "checkpoint_files", "dependencies", "python", "cases"):
        _assert_identical(snapshots[0][field], snapshots[1][field], field)
    print(f"Bitwise identical: {len(snapshots[0]['cases'])} benchmark turns")


DATASET_ID = "xai-org/RealworldQA"
DATASET_REVISION = "17e7f75e092e47169732462ea3cdfebe911105dd"
SEED = 20260918
# Drawn once with random.Random(SEED).sample(range(765), 16)[:2].
ROW_IDS = (315, 157)
# One backend per model keeps CI short while covering both PIL and the GPU path.
DEFAULT_CASES = {
    "qwen3_5": (
        "Qwen/Qwen3.5-27B",
        "fc05daec18b0a78c049392ed2e771dde82bdf654",
        "pil",
        ("image", "image"),
    ),
    "qwen2_5_vl": (
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        "torchvision",
        ("image", "image"),
    ),
    "qwen3_vl": (
        "Qwen/Qwen3-VL-8B-Instruct",
        "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "torchvision",
        ("image", "video"),
    ),
    "qwen3_omni": (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "26291f793822fb6be9555850f06dfe95f2d7e695",
        "pil",
        ("image", "audio"),
    ),
}
QUESTIONS = {
    "video": "Describe what moves in this video.",
    "audio": "What do you hear in this clip?",
}


def _benchmark_media(directory):
    from datasets import Image, load_dataset

    rows = load_dataset(
        DATASET_ID, "default", revision=DATASET_REVISION, split="test", streaming=True
    ).cast_column("image", Image(decode=False))
    images = []
    for index, row in enumerate(rows.take(max(ROW_IDS) + 1)):
        if index in ROW_IDS:
            path = directory / f"realworldqa-{index}.image"
            path.write_bytes(row["image"]["bytes"])
            images.append((path, row["question"]))
    video = directory / "testsrc.mp4"
    audio = directory / "tone.wav"
    for source, target in (
        ("testsrc2=size=640x360:rate=30:duration=6", video),
        ("sine=frequency=440:duration=3:sample_rate=16000", audio),
    ):
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", source, str(target)],
            check=True,
        )
    return {
        "image": images,
        "video": [(video, QUESTIONS["video"])],
        "audio": [(audio, QUESTIONS["audio"])],
        "row_ids": list(ROW_IDS),
    }


def _default_benchmark_manifest(directory, case):
    from huggingface_hub import snapshot_download

    from sglang.srt.utils.hf_transformers import get_processor

    checkpoint_id, checkpoint_revision, backend, modalities = DEFAULT_CASES[case]
    checkpoint = snapshot_download(
        checkpoint_id,
        revision=checkpoint_revision,
        allow_patterns=["*.json", "*.txt", "*.model", "*.jinja", "*.py"],
    )
    media = _benchmark_media(directory)
    processor = get_processor(
        checkpoint, image_processor_backend="pil", local_files_only=True
    )
    prompt, turns = "", []
    selected = {"image": [], "video": [], "audio": []}
    for modality in modalities:
        path, question = media[modality][len(selected[modality])]
        if prompt:
            prompt += processor.tokenizer.eos_token + "\n"
        prompt += processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [{"type": modality}, {"type": "text", "text": question}],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        selected[modality].append({"path": path.name, "sha256": _sha256(path)})
        turns.append(
            {
                "prompt": prompt,
                "media": {key: list(value) for key, value in selected.items()},
            }
        )
    manifest = {
        "checkpoint": {
            "path": checkpoint,
            "repo_id": checkpoint_id,
            "revision": checkpoint_revision,
        },
        "seed": SEED,
        "image_processor_backend": backend,
        "mm_process_config": {},
        "samples": [
            {
                "id": f"{case}-multiturn",
                "dataset": {
                    "repo_id": DATASET_ID,
                    "revision": DATASET_REVISION,
                    "config": "default",
                    "split": "test",
                    "row_ids": media["row_ids"],
                },
                "turns": turns,
            }
        ],
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


@pytest.mark.parametrize("case", list(DEFAULT_CASES))
def test_benchmark_prompts_match_original_async_boundary(tmp_path, case):
    if not torch.cuda.is_available():
        pytest.skip("Real async benchmark parity runs on the registered CUDA CI runner")
    checkout = Path(__file__).resolve().parents[4]
    manifest = _default_benchmark_manifest(tmp_path, case)
    _compare_routes(checkout, manifest, tmp_path / "captures")


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", choices=("legacy", "candidate"), required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    captured = asyncio.run(
        _capture(args.checkout.resolve(), args.manifest.resolve(), args.capture)
    )
    with args.output.open("wb") as stream:
        pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    if "--manifest" in sys.argv:
        _main()
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))
