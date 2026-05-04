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
from sglang.srt.ug.runtime import UGInterleavedMessage
from sglang.srt.ug.u1 import (
    U1SubprocessVLMBackend,
    U1UGModelAdapter,
    U1SRTBackedUGMiddleBridge,
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
    elif env.get("SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE") == "1":
        candidate = _run_sglang_native_vlm_candidate(
            case=case,
            model_path=env.get("SGLANG_TEST_U1_CANDIDATE_MODEL_PATH") or model_path,
            image_path=image_path,
            question=question,
            max_new_tokens=max_new_tokens,
            cuda_visible_devices=env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES"),
        )
    elif env.get("SGLANG_TEST_U1_PARITY_RUN_SGLANG_CANDIDATE") == "1":
        candidate = _run_sglang_vlm_candidate(
            case=case,
            official_py=Path(
                env.get("SGLANG_TEST_U1_CANDIDATE_PY") or str(official_py)
            ),
            official_repo=Path(
                env.get("SGLANG_TEST_U1_CANDIDATE_REPO") or str(official_repo)
            ),
            model_path=env.get("SGLANG_TEST_U1_CANDIDATE_MODEL_PATH") or model_path,
            image_path=image_path,
            question=question,
            max_new_tokens=max_new_tokens,
            device=env.get("SGLANG_TEST_U1_CANDIDATE_DEVICE") or device,
            dtype=env.get("SGLANG_TEST_U1_CANDIDATE_DTYPE") or dtype,
            attn_backend=(
                env.get("SGLANG_TEST_U1_CANDIDATE_ATTN_BACKEND") or attn_backend
            ),
            timeout=timeout,
            cuda_visible_devices=env.get("SGLANG_TEST_U1_CUDA_VISIBLE_DEVICES"),
            output_dir=output_dir / "candidate-work",
        )
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


def _run_sglang_native_vlm_candidate(
    *,
    case: UGParityCase,
    model_path: str,
    image_path: Path,
    question: str,
    max_new_tokens: int,
    cuda_visible_devices: str | None,
) -> UGParityArtifact:
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    from transformers import AutoTokenizer

    from sensenova_u1.models.neo_unify.utils import load_image_native
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.managers.mm_utils import init_mm_embedding_cache
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import PortArgs, ServerArgs

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pixel_values, grid_hw = load_image_native(image_path)
    input_ids, image_offsets = _build_u1_vlm_input_ids_and_offsets(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        grid_hw=grid_hw,
        question=question,
    )

    server_args = ServerArgs(
        model_path=model_path,
        tokenizer_path=model_path,
        trust_remote_code=False,
        disable_cuda_graph=True,
        disable_hybrid_swa_memory=True,
        mem_fraction_static=0.80,
        chunked_prefill_size=-1,
    )
    port_args = PortArgs.init_new(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
        moe_ep_rank=0,
        moe_ep_size=1,
    )
    model = runner.model
    init_mm_embedding_cache()

    generated = []
    with torch.no_grad():
        batch = None
        try:
            if max_new_tokens > 0:
                batch, token = _run_native_u1_vlm_prefill(
                    runner=runner,
                    model=model,
                    model_config=model_config,
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    grid_hw=grid_hw,
                    image_offsets=image_offsets,
                )
                generated.append(token)
            for _ in range(1, max_new_tokens):
                token = _run_native_u1_vlm_decode_step(
                    runner=runner,
                    model_config=model_config,
                    batch=batch,
                    input_token=generated[-1],
                )
                generated.append(token)
        finally:
            runner.req_to_token_pool.clear()
            runner.token_to_kv_pool_allocator.clear()

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="sglang",
        text=text,
        image=summarize_ug_image(image_path),
        metadata={
            "candidate_backend": "u1_native_srt_vlm_kv_decode",
            "native_srt_model_runner": True,
            "model_cls": type(model).__module__ + "." + type(model).__name__,
            "language_cls": (
                type(model.language_model).__module__
                + "."
                + type(model.language_model).__name__
            ),
            "token_ids": generated,
            "kv_decode": True,
            "prefill_forwards": 1 if max_new_tokens > 0 else 0,
            "decode_forwards": max(0, max_new_tokens - 1),
        },
    )


def _build_u1_vlm_input_ids_and_offsets(
    *,
    tokenizer,
    pixel_values: torch.Tensor,
    grid_hw: torch.Tensor,
    question: str,
) -> tuple[list[int], list[tuple[int, int]]]:
    from sensenova_u1.models.neo_unify.conversation import get_conv_template

    del pixel_values
    img_start_token = "<img>"
    img_end_token = "</img>"
    img_context_token = "<IMG_CONTEXT>"
    img_context_token_id = tokenizer.convert_tokens_to_ids(img_context_token)

    template = get_conv_template("neo1_0")
    template.append_message(template.roles[0], "<image>\n" + question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    for i in range(grid_hw.shape[0]):
        num_patch_token = int(grid_hw[i, 0] * grid_hw[i, 1] * 0.5**2)
        image_tokens = (
            img_start_token + img_context_token * num_patch_token + img_end_token
        )
        query = query.replace("<image>", image_tokens, 1)

    input_ids = tokenizer(query, return_tensors="pt")["input_ids"][0].tolist()
    selected = [i for i, token in enumerate(input_ids) if token == img_context_token_id]
    if not selected:
        raise RuntimeError("U1 native VLM prompt did not contain image context tokens")
    return input_ids, [(selected[0], selected[-1])]


def _run_native_u1_vlm_prefill(
    *,
    runner,
    model,
    model_config,
    input_ids: list[int],
    pixel_values: torch.Tensor,
    grid_hw: torch.Tensor,
    image_offsets: list[tuple[int, int]],
):
    from sglang.bench_one_batch import TreeCacheNamespace
    from sglang.srt.managers.schedule_batch import (
        Modality,
        MultimodalDataItem,
        MultimodalInputs,
        Req,
        ScheduleBatch,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    runner.req_to_token_pool.clear()
    runner.token_to_kv_pool_allocator.clear()
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        feature=pixel_values.clone(),
        model_specific_data={"image_grid_hws": grid_hw.clone()},
        offsets=image_offsets,
    )
    item.set_pad_value()
    mm_inputs = MultimodalInputs(mm_items=[item])
    padded_ids = model.pad_input_ids(list(input_ids), mm_inputs)
    req = Req(
        rid="u1-native-vlm-kv-decode",
        origin_input_text="",
        origin_input_ids=padded_ids,
        sampling_params=SamplingParams(temperature=0.0, max_new_tokens=1),
    )
    req.fill_ids = list(padded_ids)
    req.multimodal_inputs = mm_inputs
    req.logprob_start_len = -1
    req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
    tree_cache = TreeCacheNamespace(
        page_size=1,
        device=runner.device,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
    )
    batch = ScheduleBatch.init_new(
        reqs=[req],
        req_to_token_pool=runner.req_to_token_pool,
        token_to_kv_pool_allocator=runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )
    batch.prepare_for_extend()
    worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(worker_batch, runner)
    output, _ = runner.forward_extend(forward_batch)
    return batch, _greedy_next_token(output.next_token_logits)


def _run_native_u1_vlm_decode_step(
    *,
    runner,
    model_config,
    batch,
    input_token: int,
) -> int:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    del model_config
    batch.output_ids = torch.tensor(
        [input_token],
        dtype=torch.int64,
        device=runner.device,
    )
    batch.prepare_for_decode()
    worker_batch = batch.get_model_worker_batch()
    forward_batch = ForwardBatch.init_new(worker_batch, runner)
    output = runner.forward_decode(forward_batch)
    return _greedy_next_token(output.next_token_logits)


def _greedy_next_token(next_token_logits: torch.Tensor) -> int:
    return int(torch.argmax(next_token_logits[0]).item())



def _run_sglang_vlm_candidate(
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
    cuda_visible_devices: str | None,
    output_dir: Path,
) -> UGParityArtifact:
    from sglang.srt.session.session_controller import SessionController
    from sglang.srt.ug.adapter import UGModelRunnerAdapter
    from sglang.srt.ug.runtime import UGSessionRuntime

    backend = U1SubprocessVLMBackend(
        python=official_py,
        repo=official_repo,
        model_path=model_path,
        device=device,
        dtype=dtype,
        attn_backend=attn_backend,
        timeout=timeout,
        cuda_visible_devices=cuda_visible_devices,
        output_dir=output_dir,
    )
    adapter = U1UGModelAdapter(vlm_backend=backend)
    runtime = UGSessionRuntime(
        model_runner=UGModelRunnerAdapter(adapter),
        session_controller=SessionController(_HarnessTreeCache()),
        srt_image_tokenization="text_placeholder",
    )
    bridge = U1SRTBackedUGMiddleBridge(runtime)
    result = bridge.generate_vlm_text(
        messages=[
            UGInterleavedMessage(type="image", content=str(image_path)),
            UGInterleavedMessage(type="text", content=question),
        ],
        max_new_tokens=max_new_tokens,
    )
    try:
        debug_counters = runtime.get_debug_counters(result.session)
    finally:
        runtime.close_session(result.session)
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner="sglang",
        text=result.text,
        image=summarize_ug_image(image_path),
        metadata={
            "candidate_backend": "u1_external_vlm_backend",
            "native_srt_model_runner": False,
        },
        debug_counters=debug_counters,
    )


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


class _HarnessTreeCache:
    def __init__(self) -> None:
        self.released_sessions: list[str] = []

    def release_session(self, session_id: str) -> None:
        self.released_sessions.append(session_id)


if __name__ == "__main__":
    unittest.main()
