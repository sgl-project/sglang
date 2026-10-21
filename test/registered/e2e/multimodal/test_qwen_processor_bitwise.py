"""Compare the original and refactored Qwen async processor boundaries bit for bit.

    pinned benchmark rows + rendered prompts + SHA-256 checked local media
                     |                         |
        legacy route subprocess        candidate route subprocess
        full input_text every turn     input_ids; expanded prefix + new suffix
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
No model weights or generation are needed. CPU tests exercise the strict
comparator; CUDA CI runs a small pinned matrix through both real routes: Qwen3.5 and
Qwen2.5-VL with two images, Qwen3-VL with an image then a video, Qwen3-Omni with
an image then one audio clip; one backend per model covers PIL and the GPU path.
Turns are joined at an explicit assistant end token.
A manifest override supplies the complete model/benchmark/turn matrix.
The default reference is the same clean revision with the startup strategy
disabled (legacy route); SGLANG_MM_BITWISE_REFERENCE=pinned compares against
the exported historical baseline, or names another reference checkout.
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
import tarfile
from enum import Enum
from pathlib import Path

import msgspec
import numpy as np
import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")

BASELINE_REVISION = "54b551df8e5721ecc356d50287e68e638b38f5e5"


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
_CACHE_IDENTITY_ITEM_FIELDS = ("hash", "pad_value")


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
    exported_revision = checkout / ".bitwise-reference-revision"
    if exported_revision.exists():
        return {"revision": exported_revision.read_text().strip(), "diff": ""}
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
    if route == "baseline":
        assert source == {"revision": BASELINE_REVISION, "diff": ""}
    elif route == "legacy":
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
        from sglang.srt.managers.multimodal_processor import (
            TOKEN_SPACE_PROCESSOR_MAPPING,
            get_mm_processor,
        )
        from sglang.srt.multimodal.processors.qwen_vl_token_space import (
            QwenVLTokenSpaceProcessor,
        )

        TOKEN_SPACE_PROCESSOR_MAPPING.update(
            {
                model: QwenVLTokenSpaceProcessor
                for model in QwenVLTokenSpaceProcessor.models
            }
        )
        processor = get_mm_processor(
            hf_config, server_args, hf_processor, None, skip_mm_pool=True
        )
        assert type(processor) is QwenVLTokenSpaceProcessor
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
                if route in ("baseline", "legacy"):
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
                    boundary = len(previous_expanded_ids)
                    partial_ids = previous_expanded_ids + input_ids[len(previous_ids) :]
                    request = GenerateReqInput(
                        input_ids=partial_ids,
                        mm_token_expansion_start_len=boundary,
                        image_data=sources["image"],
                        video_data=sources["video"],
                        audio_data=sources["audio"],
                    )
                    output = await processor.process_mm_data_async(
                        image_data=request.image_data,
                        audio_data=request.audio_data,
                        input_text="",
                        input_ids=partial_ids,
                        request_obj=request,
                    )
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


def compare_checkouts(
    reference, candidate, manifest_path, artifacts, *, reference_route="baseline"
):
    artifacts.mkdir(parents=True, exist_ok=True)
    snapshots = []
    seed = json.loads(manifest_path.read_text())["seed"]
    for name, route, checkout in (
        ("baseline", reference_route, reference),
        ("candidate", "candidate", candidate),
    ):
        output_path = artifacts / f"{name}.pickle"
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
    assert snapshots[0]["route"] == reference_route
    assert snapshots[1]["route"] == "candidate"
    if reference_route == "legacy":
        _assert_identical(snapshots[0]["source"], snapshots[1]["source"], "source")
        assert not snapshots[0]["source"]["diff"], "Compared source must be clean"
    _drop_unshared_cache_identity(snapshots[0]["cases"], snapshots[1]["cases"])
    for field in ("manifest", "checkpoint_files", "dependencies", "python", "cases"):
        _assert_identical(snapshots[0][field], snapshots[1][field], field)
    print(f"Bitwise identical: {len(snapshots[0]['cases'])} benchmark turns")


def _export_baseline(candidate, reference):
    if subprocess.run(
        ["git", "cat-file", "-e", BASELINE_REVISION], cwd=candidate
    ).returncode:
        subprocess.run(
            ["git", "fetch", "origin", BASELINE_REVISION], cwd=candidate, check=True
        )
    archive_path = reference.parent / "baseline.tar"
    with archive_path.open("wb") as archive:
        subprocess.run(
            ["git", "archive", BASELINE_REVISION, "python"],
            cwd=candidate,
            stdout=archive,
            check=True,
        )
    reference.mkdir()
    with tarfile.open(archive_path) as archive:
        archive.extractall(reference, filter="data")
    (reference / ".bitwise-reference-revision").write_text(BASELINE_REVISION)


DATASET_ID = "xai-org/RealworldQA"
DATASET_REVISION = "17e7f75e092e47169732462ea3cdfebe911105dd"
SEED = 20260918
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

    row_indices = random.Random(SEED).sample(range(765), 16)[:2]
    rows = load_dataset(
        DATASET_ID, "default", revision=DATASET_REVISION, split="test", streaming=True
    ).cast_column("image", Image(decode=False))
    images = []
    for index, row in enumerate(rows.take(max(row_indices) + 1)):
        if index in row_indices:
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
        "row_ids": row_indices,
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
    used = {"image": 0, "video": 0, "audio": 0}
    prompt, turns = "", []
    selected = {"image": [], "video": [], "audio": []}
    for modality in modalities:
        path, question = media[modality][used[modality]]
        used[modality] += 1
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


def test_cache_identity_is_compared_only_when_both_routes_set_it():
    def cases(hash_value, padded_input_ids):
        item = {"hash": hash_value, "pad_value": hash_value, "offsets": [(1, 2)]}
        inputs = {"padded_input_ids": padded_input_ids, "mm_items": [item]}
        return {"turn-0": {"model_inputs": inputs}}

    reference, candidate = cases(7, [1, 7]), cases(None, None)
    _drop_unshared_cache_identity(reference, candidate)
    _assert_identical(reference, candidate)

    reference, candidate = cases(7, [1, 7]), cases(8, [1, 8])
    _drop_unshared_cache_identity(reference, candidate)
    with pytest.raises(AssertionError):
        _assert_identical(reference, candidate)


@pytest.mark.parametrize(
    "change", ["signed_zero", "dtype", "shape", "metadata", "numpy_scalar_dtype"]
)
def test_bitwise_comparison_rejects_numerically_equal_or_metadata_changed_outputs(
    change,
):
    reference = {"feature": torch.tensor([0.0, 1.0]), "offsets": [(1, 2)]}
    candidate = dict(reference)
    if change == "signed_zero":
        candidate["feature"] = torch.tensor([-0.0, 1.0])
        assert torch.equal(reference["feature"], candidate["feature"])
    elif change == "dtype":
        candidate["feature"] = reference["feature"].to(torch.float64)
    elif change == "shape":
        candidate["feature"] = reference["feature"].reshape(1, 2)
    elif change == "metadata":
        candidate["offsets"] = [(1, 3)]
    else:
        reference["scalar"] = np.float32(1)
        candidate["scalar"] = np.float64(1)
    with pytest.raises(AssertionError, match="output\\."):
        _assert_identical(_snapshot(reference), _snapshot(candidate))
    _assert_identical(_snapshot(reference), _snapshot(reference))


@pytest.mark.parametrize(
    "reference_route,change",
    [
        ("baseline", None),
        ("legacy", None),
        ("legacy", "revision"),
        ("legacy", "dirty"),
        ("legacy", "route"),
        ("legacy", "feature"),
    ],
)
def test_checkout_comparison_requires_matching_legacy_source(
    tmp_path, monkeypatch, reference_route, change
):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"seed": 42}))
    requested_routes = []

    def capture(command, **kwargs):
        route = command[command.index("--capture") + 1]
        requested_routes.append(route)
        source = {
            "revision": BASELINE_REVISION if route == "baseline" else "current-commit",
            "diff": "changed" if change == "dirty" else "",
        }
        feature = torch.tensor([0.0])
        if route == "candidate":
            if change == "revision":
                source["revision"] = "another-commit"
            elif change == "route":
                route = "legacy"
            elif change == "feature":
                feature = torch.tensor([-0.0])
        snapshot = {
            "route": route,
            "source": source,
            "manifest": {"seed": 42},
            "checkpoint_files": {},
            "dependencies": {},
            "python": sys.version,
            "cases": {"turn-0": _snapshot(feature)},
        }
        Path(command[command.index("--output") + 1]).write_bytes(pickle.dumps(snapshot))

    monkeypatch.setattr(subprocess, "run", capture)
    options = (
        {"reference_route": reference_route} if reference_route == "legacy" else {}
    )
    arguments = (tmp_path, tmp_path, manifest_path, tmp_path / "captures")
    if change:
        with pytest.raises(AssertionError):
            compare_checkouts(*arguments, **options)
    else:
        compare_checkouts(*arguments, **options)
    assert requested_routes == [reference_route, "candidate"]
    assert {path.name for path in (tmp_path / "captures").iterdir()} == {
        "baseline.pickle",
        "candidate.pickle",
    }


@pytest.mark.parametrize("case", list(DEFAULT_CASES))
def test_benchmark_prompts_match_original_async_boundary(tmp_path, case):
    if not torch.cuda.is_available():
        pytest.skip("Real async benchmark parity runs on the registered CUDA CI runner")
    candidate = Path(__file__).resolve().parents[4]
    manifest_override = os.environ.get("SGLANG_MM_BITWISE_MANIFEST")
    if manifest_override and case != next(iter(DEFAULT_CASES)):
        pytest.skip("A manifest override replaces the default model matrix")
    manifest = (
        Path(manifest_override).resolve()
        if manifest_override
        else _default_benchmark_manifest(tmp_path, case)
    )
    # Default: the same clean checkout with token-space disabled. "pinned" exports
    # BASELINE_REVISION; any other value is a reference checkout path.
    reference_override = os.environ.get("SGLANG_MM_BITWISE_REFERENCE")
    if not reference_override:
        compare_checkouts(
            candidate,
            candidate,
            manifest,
            tmp_path / "captures",
            reference_route="legacy",
        )
        return
    if reference_override == "pinned":
        reference = tmp_path / "baseline"
        _export_baseline(candidate, reference)
    else:
        reference = Path(reference_override).resolve()
    compare_checkouts(reference, candidate, manifest, tmp_path / "captures")


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", choices=("baseline", "legacy", "candidate"))
    parser.add_argument(
        "--reference-route",
        choices=("baseline", "legacy"),
        default="baseline",
    )
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.capture:
        captured = asyncio.run(
            _capture(args.checkout.resolve(), args.manifest.resolve(), args.capture)
        )
        with args.output.open("wb") as stream:
            pickle.dump(captured, stream, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        compare_checkouts(
            args.reference.resolve(),
            args.candidate.resolve(),
            args.manifest.resolve(),
            args.output.resolve(),
            reference_route=args.reference_route,
        )


if __name__ == "__main__":
    if "--manifest" in sys.argv:
        _main()
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))
