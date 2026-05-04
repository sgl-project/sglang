# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sglang.srt.ug.parity import (
    UGParityArtifact,
    UGParityCase,
    UGTensorSummary,
    compare_ug_parity_artifacts,
    summarize_ug_image,
    write_ug_parity_bundle,
)


class TestU1OfficialParityHarness(unittest.TestCase):
    def test_u1_official_parity_harness(self):
        run_u1_official_parity_from_env(os.environ)


def run_u1_official_parity_from_env(env) -> Path:
    if env.get("SGLANG_TEST_U1_PARITY_DRY_RUN") == "1":
        output_dir = Path(
            env.get("SGLANG_TEST_U1_PARITY_OUTPUT")
            or tempfile.mkdtemp(prefix="u1-parity-")
        )
        return _write_dry_run_bundle(output_dir)

    if env.get("SGLANG_TEST_U1_PARITY_MODE") == "vlm_official_reference":
        return _run_vlm_official_reference_mode(env)

    reference_path = env.get("SGLANG_TEST_U1_PARITY_REFERENCE_ARTIFACT")
    candidate_path = env.get("SGLANG_TEST_U1_PARITY_CANDIDATE_ARTIFACT")
    output_dir = env.get("SGLANG_TEST_U1_PARITY_OUTPUT")
    missing = [
        name
        for name, value in (
            ("SGLANG_TEST_U1_PARITY_REFERENCE_ARTIFACT", reference_path),
            ("SGLANG_TEST_U1_PARITY_CANDIDATE_ARTIFACT", candidate_path),
            ("SGLANG_TEST_U1_PARITY_OUTPUT", output_dir),
        )
        if not value
    ]
    if missing:
        raise unittest.SkipTest(
            "U1 official parity harness is opt-in; missing env: "
            + ", ".join(missing)
        )

    reference = UGParityArtifact.from_json(Path(reference_path).read_text())
    candidate = UGParityArtifact.from_json(Path(candidate_path).read_text())
    case = _case_from_artifact(reference)
    report = compare_ug_parity_artifacts(reference, candidate)
    bundle = write_ug_parity_bundle(
        output_dir=output_dir,
        case=case,
        reference=reference,
        candidate=candidate,
        report=report,
    )
    if not report.passed:
        raise AssertionError(f"U1 parity failed; report: {bundle / 'report.json'}")
    return bundle


def _run_vlm_official_reference_mode(env) -> Path:
    output_dir = Path(
        env.get("SGLANG_TEST_U1_PARITY_OUTPUT")
        or tempfile.mkdtemp(prefix="u1-vlm-parity-")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    official_py = Path(
        env.get("SGLANG_TEST_U1_OFFICIAL_PY")
        or "/data/venvs/sensenova_u1_official/bin/python"
    )
    official_repo = Path(
        env.get("SGLANG_TEST_U1_OFFICIAL_REPO") or "/data/repos/SenseNova-U1"
    )
    model_path = env.get("SGLANG_TEST_U1_MODEL_PATH") or (
        "/data/models/SenseNova-U1-8B-MoT"
    )
    image_path = Path(
        env.get("SGLANG_TEST_U1_VLM_IMAGE")
        or official_repo / "examples/vqa/data/images/image1.jpg"
    )
    question = env.get("SGLANG_TEST_U1_VLM_QUESTION") or "What is in this image?"
    max_new_tokens = int(env.get("SGLANG_TEST_U1_VLM_MAX_NEW_TOKENS") or "4")
    case_id = env.get("SGLANG_TEST_U1_VLM_CASE_ID") or "u1-vlm-official-reference"
    device = env.get("SGLANG_TEST_U1_VLM_DEVICE") or "cuda"
    dtype = env.get("SGLANG_TEST_U1_VLM_DTYPE") or "bfloat16"
    attn_backend = env.get("SGLANG_TEST_U1_VLM_ATTN_BACKEND") or "sdpa"
    timeout = int(env.get("SGLANG_TEST_U1_PARITY_TIMEOUT") or "600")

    case = UGParityCase(
        case_id=case_id,
        model="sensenova-u1",
        task="vlm",
        prompt=question,
        image_path=str(image_path),
        sampling_params={
            "max_new_tokens": max_new_tokens,
            "device": device,
            "dtype": dtype,
            "attn_backend": attn_backend,
        },
        dump_points=("text", "input_image"),
        metadata={
            "mode": "vlm_official_reference",
            "official_python": str(official_py),
            "official_repo": str(official_repo),
            "model_path": model_path,
        },
    )

    reference = _run_official_vlm_reference(
        case=case,
        official_py=official_py,
        official_repo=official_repo,
        model_path=model_path,
        image_path=image_path,
        question=question,
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=dtype,
        attn_backend=attn_backend,
        timeout=timeout,
        output_dir=output_dir,
        env=env,
    )
    candidate_path = env.get("SGLANG_TEST_U1_PARITY_CANDIDATE_ARTIFACT")
    if candidate_path:
        candidate = UGParityArtifact.from_json(Path(candidate_path).read_text())
    else:
        candidate = _candidate_unavailable_artifact(case, image_path=image_path)

    report = compare_ug_parity_artifacts(reference, candidate)
    bundle = write_ug_parity_bundle(
        output_dir=output_dir,
        case=case,
        reference=reference,
        candidate=candidate,
        report=report,
    )
    if reference.error:
        raise AssertionError(
            f"U1 official VLM reference failed; report: {bundle / 'report.json'}"
        )
    if candidate_path and not report.passed:
        raise AssertionError(f"U1 VLM parity failed; report: {bundle / 'report.json'}")
    return bundle


def _run_official_vlm_reference(
    *,
    case: UGParityCase,
    official_py: Path,
    official_repo: Path,
    model_path: str,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    device: str,
    dtype: str,
    attn_backend: str,
    timeout: int,
    output_dir: Path,
    env,
) -> UGParityArtifact:
    script = official_repo / "examples/vqa/inference.py"
    raw_output = output_dir / f"{case.case_id}.official.txt"
    cmd = [
        str(official_py),
        str(script),
        "--model_path",
        model_path,
        "--image",
        str(image_path),
        "--question",
        question,
        "--output",
        str(raw_output),
        "--max_new_tokens",
        str(max_new_tokens),
        "--device",
        device,
        "--dtype",
        dtype,
        "--attn_backend",
        attn_backend,
    ]
    run_env = os.environ.copy()
    cuda_visible_devices = env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    try:
        completed = subprocess.run(
            cmd,
            cwd=official_repo,
            env=run_env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return _official_vlm_artifact_from_completed(
            case=case,
            image_path=image_path,
            raw_output=raw_output,
            cmd=cmd,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except subprocess.TimeoutExpired as exc:
        return UGParityArtifact(
            case_id=case.case_id,
            model=case.model,
            task=case.task,
            runner="official",
            image=summarize_ug_image(image_path),
            metadata={
                "command": cmd,
                "timeout_seconds": timeout,
                "stdout_tail": _tail(exc.stdout),
                "stderr_tail": _tail(exc.stderr),
            },
            error=f"official_vlm_timeout: timeout={timeout}s",
        )


def _official_vlm_artifact_from_completed(
    *,
    case: UGParityCase,
    image_path: Path,
    raw_output: Path,
    cmd: list[str],
    returncode: int,
    stdout: str | None,
    stderr: str | None,
) -> UGParityArtifact:
    text = raw_output.read_text() if raw_output.exists() else None
    error = None
    if returncode != 0:
        error = f"official_vlm_failed: returncode={returncode}"
    elif text is None:
        error = "official_vlm_failed: missing_output"

    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="official",
        text=text,
        image=summarize_ug_image(image_path),
        metadata={
            "command": cmd,
            "returncode": returncode,
            "stdout_tail": _tail(stdout),
            "stderr_tail": _tail(stderr),
        },
        error=error,
    )


def _candidate_unavailable_artifact(
    case: UGParityCase,
    *,
    image_path: Path,
) -> UGParityArtifact:
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="sglang",
        image=summarize_ug_image(image_path),
        metadata={
            "mode": "vlm_official_reference",
            "stop_signal": "native_u1_vlm_not_wired",
        },
        error="SGLang U1 VLM native path is not wired yet",
    )


def _tail(text: str | bytes | None, *, limit: int = 2000) -> str | None:
    if text is None:
        return None
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    if len(text) <= limit:
        return text
    return text[-limit:]


def _write_dry_run_bundle(output_dir: Path) -> Path:
    case = UGParityCase(
        case_id="u1-dry-run",
        model="sensenova-u1",
        task="vlm",
        prompt="describe this image",
        seed=1,
        sampling_params={"max_new_tokens": 4},
        dump_points=("text", "u_logits"),
    )
    reference = _dry_run_artifact(case, runner="official")
    candidate = _dry_run_artifact(case, runner="sglang")
    report = compare_ug_parity_artifacts(reference, candidate)
    return write_ug_parity_bundle(
        output_dir=output_dir,
        case=case,
        reference=reference,
        candidate=candidate,
        report=report,
    )


def _dry_run_artifact(case: UGParityCase, *, runner: str) -> UGParityArtifact:
    image = Image.fromarray(np.full((4, 4, 3), 7, dtype=np.uint8), "RGB")
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner=runner,
        text="dry run",
        image=summarize_ug_image(image),
        tensors={"u_logits": UGTensorSummary.from_tensor(torch.ones(2, 2))},
        metadata={"dry_run": True},
    )


def _case_from_artifact(artifact: UGParityArtifact) -> UGParityCase:
    return UGParityCase(
        case_id=artifact.case_id,
        model=artifact.model,
        task=artifact.task,
        metadata={"source": "artifact_env"},
    )


if __name__ == "__main__":
    unittest.main()
