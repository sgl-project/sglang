# SPDX-License-Identifier: Apache-2.0
"""Opt-in BAGEL official-vs-SGLang G parity smoke.

Usage:
CUDA_VISIBLE_DEVICES=6 \
SGLANG_TEST_BAGEL_OFFICIAL_REPO=/data/BAGEL \
SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL=/data/models/BAGEL-7B-MoT \
SGLANG_TEST_BAGEL_G_PARITY_OUTPUT=/tmp/ug-g-parity \
SGLANG_TEST_BAGEL_G_TASKS=text_to_image,image_edit,interleave \
python3 test/registered/scheduler/test_bagel_g_official_parity.py
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from sglang.srt.ug.parity import (
    UGParityArtifact,
    UGParityCase,
    UGParityTolerance,
    compare_ug_parity_artifacts,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=900,
    suite="stage-b-test-1-gpu-large",
    disabled=(
        "Manual BAGEL G official parity smoke; requires "
        "SGLANG_TEST_BAGEL_OFFICIAL_REPO and "
        "SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL"
    ),
)

_OFFICIAL_REPO_ENV = "SGLANG_TEST_BAGEL_OFFICIAL_REPO"
_MODEL_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL"
_OUTPUT_ENV = "SGLANG_TEST_BAGEL_G_PARITY_OUTPUT"
_IMAGE_ENV = "SGLANG_TEST_BAGEL_G_IMAGE"
_PROMPT_ENV = "SGLANG_TEST_BAGEL_G_PROMPT"
_STEPS_ENV = "SGLANG_TEST_BAGEL_G_NUM_STEPS"
_HEIGHT_ENV = "SGLANG_TEST_BAGEL_G_HEIGHT"
_WIDTH_ENV = "SGLANG_TEST_BAGEL_G_WIDTH"
_CFG_TEXT_ENV = "SGLANG_TEST_BAGEL_G_CFG_TEXT_SCALE"
_CFG_IMG_ENV = "SGLANG_TEST_BAGEL_G_CFG_IMG_SCALE"
_TASKS_ENV = "SGLANG_TEST_BAGEL_G_TASKS"
_SEED_ENV = "SGLANG_TEST_BAGEL_G_SEED"
_POST_TEXT_TOKENS_ENV = "SGLANG_TEST_BAGEL_INTERLEAVED_TEXT_TOKENS"
_TENSOR_STAT_ATOL_ENV = "SGLANG_TEST_BAGEL_G_TENSOR_STAT_ATOL"
_TENSOR_STAT_RTOL_ENV = "SGLANG_TEST_BAGEL_G_TENSOR_STAT_RTOL"
_GPU_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_GPU_ID"
_ATTENTION_BACKEND_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_ATTENTION_BACKEND"


def _has_live_env() -> bool:
    return bool(os.getenv(_OFFICIAL_REPO_ENV) and os.getenv(_MODEL_ENV))


@unittest.skipUnless(
    _has_live_env(),
    f"Set {_OFFICIAL_REPO_ENV} and {_MODEL_ENV} for BAGEL G parity smoke",
)
class TestBAGELGOfficialParity(CustomTestCase):
    def test_selected_g_tasks_match_official_denoise(self):
        official_repo = Path(os.environ[_OFFICIAL_REPO_ENV]).expanduser()
        checkpoint_dir = Path(os.environ[_MODEL_ENV]).expanduser()
        self.assertTrue(official_repo.exists(), official_repo)
        self.assertTrue(checkpoint_dir.exists(), checkpoint_dir)

        output_dir = Path(
            os.getenv(_OUTPUT_ENV) or tempfile.mkdtemp(prefix="ug-g-parity-")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        cases = _build_g_parity_cases(
            official_repo=official_repo,
            checkpoint_dir=checkpoint_dir,
        )

        tolerance = UGParityTolerance(
            image_sha256_exact=False,
            tensor_stat_atol=float(os.getenv(_TENSOR_STAT_ATOL_ENV, "5e-2")),
            tensor_stat_rtol=float(os.getenv(_TENSOR_STAT_RTOL_ENV, "5e-2")),
        )
        for case in cases:
            case_dir = output_dir / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            case_path = case_dir / "case.json"
            official_path = case_dir / "reference.official.json"
            sglang_path = case_dir / "candidate.sglang.json"
            report_path = case_dir / "report.json"
            case.write_json(case_path)

            _run_g_subprocess(
                _OFFICIAL_G_RUNNER,
                case_path=case_path,
                output_path=official_path,
                runner="official",
            )
            _run_g_subprocess(
                _SGLANG_G_RUNNER,
                case_path=case_path,
                output_path=sglang_path,
                runner="sglang",
            )

            reference = UGParityArtifact.read_json(official_path)
            candidate = UGParityArtifact.read_json(sglang_path)
            report = compare_ug_parity_artifacts(
                reference,
                candidate,
                tolerance=tolerance,
            )
            report.write_json(report_path)

            self.assertTrue(case_path.exists())
            self.assertTrue(official_path.exists())
            self.assertTrue(sglang_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue(report.passed, report.to_json())
            if case.task == "text_to_image":
                _assert_t2i_sglang_artifact(self, candidate, case)
            elif case.task == "image_edit":
                _assert_edit_sglang_artifact(self, candidate, case)
            elif case.task == "interleave":
                _assert_interleave_sglang_artifact(self, candidate, case)


def _build_g_parity_cases(
    *,
    official_repo: Path,
    checkpoint_dir: Path,
) -> list[UGParityCase]:
    tasks = _selected_tasks()
    sampling_params = _sampling_params_from_env()
    seed = int(os.getenv(_SEED_ENV, "123"))
    metadata = {
        "official_repo": str(official_repo),
        "checkpoint": str(checkpoint_dir),
    }

    cases: list[UGParityCase] = []
    if "text_to_image" in tasks:
        cases.append(
            UGParityCase(
                case_id="bagel-g-text-to-image-official-parity-smoke",
                task="text_to_image",
                prompt=os.getenv(
                    _PROMPT_ENV,
                    "A small red teapot on a wooden table, studio lighting.",
                ),
                seed=seed,
                sampling_params=sampling_params,
                dump_points=("init_noise", "velocity_00", "final_latents", "image"),
                metadata={**metadata, "mode": "t2i"},
            )
        )
    if "image_edit" in tasks:
        image_path = Path(
            os.getenv(_IMAGE_ENV) or official_repo / "test_images" / "women.jpg"
        ).expanduser()
        if not image_path.exists():
            raise AssertionError(image_path)
        cases.append(
            UGParityCase(
                case_id="bagel-g-image-edit-official-parity-smoke",
                task="image_edit",
                prompt=os.getenv(
                    _PROMPT_ENV,
                    "Turn the scene into a warm cinematic portrait.",
                ),
                image_path=str(image_path),
                seed=seed,
                sampling_params={
                    **sampling_params,
                    # Let both runners use BAGEL's resized edit image shape.
                    "height": None,
                    "width": None,
                },
                dump_points=("init_noise", "velocity_00", "final_latents", "image"),
                metadata={**metadata, "mode": "edit"},
            )
        )
    if "interleave" in tasks:
        image_path = Path(
            os.getenv(_IMAGE_ENV) or official_repo / "test_images" / "women.jpg"
        ).expanduser()
        if not image_path.exists():
            raise AssertionError(image_path)
        cases.append(
            UGParityCase(
                case_id="bagel-interleave-official-parity-smoke",
                task="interleave",
                prompt=os.getenv(
                    _PROMPT_ENV,
                    "Turn the scene into a warm cinematic portrait.",
                ),
                image_path=str(image_path),
                seed=seed,
                sampling_params={
                    **sampling_params,
                    # Let both runners use BAGEL's resized edit image shape.
                    "height": None,
                    "width": None,
                },
                dump_points=(
                    "init_noise",
                    "velocity_00",
                    "final_latents",
                    "image",
                    "post_image_text",
                    "token_ids",
                ),
                metadata={
                    **metadata,
                    "mode": "interleave",
                    "post_image_max_new_tokens": int(
                        os.getenv(_POST_TEXT_TOKENS_ENV, "4")
                    ),
                },
            )
        )
    return cases


def _selected_tasks() -> tuple[str, ...]:
    raw_tasks = os.getenv(_TASKS_ENV, "text_to_image,image_edit")
    tasks = tuple(
        _normalize_selected_task(task) for task in raw_tasks.split(",") if task.strip()
    )
    if not tasks:
        raise ValueError(f"{_TASKS_ENV} must select at least one task")
    allowed = {"text_to_image", "image_edit", "interleave"}
    unsupported = [task for task in tasks if task not in allowed]
    if unsupported:
        raise ValueError(f"{_TASKS_ENV} has unsupported tasks: {unsupported}")
    return tasks


def _normalize_selected_task(task: str) -> str:
    normalized = task.strip().lower().replace("-", "_")
    if normalized == "interleaved":
        return "interleave"
    return normalized


def _sampling_params_from_env() -> dict:
    return {
        "height": int(os.getenv(_HEIGHT_ENV, "512")),
        "width": int(os.getenv(_WIDTH_ENV, "512")),
        "num_inference_steps": int(os.getenv(_STEPS_ENV, "3")),
        "cfg_text_scale": float(os.getenv(_CFG_TEXT_ENV, "4.0")),
        "cfg_img_scale": float(os.getenv(_CFG_IMG_ENV, "1.5")),
        "cfg_interval": [0.4, 1.0],
        "cfg_renorm_min": 0.0,
        "cfg_renorm_type": "global",
        "timestep_shift": 3.0,
    }


def _assert_t2i_sglang_artifact(
    test_case: CustomTestCase,
    candidate: UGParityArtifact,
    case: UGParityCase,
) -> None:
    test_case.assertIsNone(candidate.error)
    test_case.assertEqual(candidate.task, "text_to_image")
    for tensor_name in (
        "init_noise",
        "velocity_00",
        "final_latents",
        "generated_image_pixels",
    ):
        test_case.assertIn(tensor_name, candidate.tensors)
    test_case.assertIn("generated_image", candidate.images)

    counters = candidate.debug_counters
    num_steps = int(case.sampling_params["num_inference_steps"])
    expected_velocity_count = max(num_steps - 1, 0)
    test_case.assertEqual(counters.get("prefill_count"), 1)
    test_case.assertEqual(counters.get("velocity_count"), expected_velocity_count)
    test_case.assertEqual(counters.get("append_image_count"), 0)
    test_case.assertEqual(counters.get("srt_sidecar_request_count"), 0)
    test_case.assertEqual(counters.get("state"), "g_denoise")
    test_case.assertGreaterEqual(
        int(counters.get("temp_g_forward_count", 0)),
        expected_velocity_count,
    )


def _assert_edit_sglang_artifact(
    test_case: CustomTestCase,
    candidate: UGParityArtifact,
    case: UGParityCase,
) -> None:
    test_case.assertIsNone(candidate.error)
    test_case.assertEqual(candidate.task, "image_edit")
    test_case.assertIsNotNone(case.image_path)
    for tensor_name in (
        "init_noise",
        "velocity_00",
        "final_latents",
        "generated_image_pixels",
    ):
        test_case.assertIn(tensor_name, candidate.tensors)
    test_case.assertIn("generated_image", candidate.images)

    image_shape = candidate.metadata.get("image_shape")
    test_case.assertIsInstance(image_shape, list)
    test_case.assertEqual(len(image_shape), 2)
    test_case.assertGreater(int(image_shape[0]), 0)
    test_case.assertGreater(int(image_shape[1]), 0)

    counters = candidate.debug_counters
    session_id = counters.get("session_id")
    num_steps = int(case.sampling_params["num_inference_steps"])
    expected_velocity_count = max(num_steps - 1, 0)
    test_case.assertEqual(counters.get("prefill_count"), 1)
    test_case.assertEqual(counters.get("velocity_count"), expected_velocity_count)
    test_case.assertEqual(counters.get("append_image_count"), 0)
    test_case.assertEqual(counters.get("srt_sidecar_request_count"), 1)
    test_case.assertIn(
        f"{session_id}:cfg_img", counters.get("srt_sidecar_session_ids", [])
    )
    test_case.assertEqual(counters.get("state"), "g_denoise")
    test_case.assertGreater(int(counters.get("context_length", 0)), 0)
    test_case.assertGreaterEqual(
        int(counters.get("temp_g_forward_count", 0)),
        expected_velocity_count,
    )


def _assert_interleave_sglang_artifact(
    test_case: CustomTestCase,
    candidate: UGParityArtifact,
    case: UGParityCase,
) -> None:
    test_case.assertIsNone(candidate.error)
    test_case.assertEqual(candidate.task, "interleave")
    test_case.assertIsNotNone(case.image_path)
    for tensor_name in (
        "init_noise",
        "velocity_00",
        "final_latents",
        "generated_image_pixels",
    ):
        test_case.assertIn(tensor_name, candidate.tensors)
    test_case.assertIn("generated_image", candidate.images)
    test_case.assertIsNotNone(candidate.text)
    test_case.assertIn("post_image_text", candidate.token_ids)
    test_case.assertGreater(len(candidate.token_ids["post_image_text"]), 0)

    counters = candidate.debug_counters
    session_id = counters.get("session_id")
    num_steps = int(case.sampling_params["num_inference_steps"])
    expected_velocity_count = max(num_steps - 1, 0)
    test_case.assertEqual(counters.get("prefill_count"), 1)
    test_case.assertEqual(counters.get("velocity_count"), expected_velocity_count)
    test_case.assertEqual(counters.get("append_image_count"), 1)
    test_case.assertEqual(counters.get("srt_sidecar_request_count"), 1)
    test_case.assertIn(
        f"{session_id}:cfg_img", counters.get("srt_sidecar_session_ids", [])
    )
    test_case.assertEqual(counters.get("state"), "u_decode")
    test_case.assertGreater(int(counters.get("context_length", 0)), 0)
    test_case.assertGreater(int(counters.get("srt_u_decode_request_count", 0)), 0)
    test_case.assertEqual(
        counters.get("session_id"),
        f"bagel-g-parity-{case.task}",
    )
    test_case.assertGreaterEqual(
        int(counters.get("temp_g_forward_count", 0)),
        expected_velocity_count,
    )


def _run_g_subprocess(
    code: str,
    *,
    case_path: Path,
    output_path: Path,
    runner: str,
) -> None:
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[3]
    python_path = str(repo_root / "python")
    env["PYTHONPATH"] = (
        python_path
        if not env.get("PYTHONPATH")
        else f"{python_path}{os.pathsep}{env['PYTHONPATH']}"
    )
    if os.getenv(_GPU_ENV) and not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = os.environ[_GPU_ENV]

    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(code),
            str(case_path),
            str(output_path),
            runner,
        ],
        check=True,
        cwd=repo_root,
        env=env,
    )


_COMMON_G_RUNNER = r"""
import json
import os
import random
import sys
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.sample.ug import UGSamplingParams
from sglang.srt.ug.parity import (
    UGImageSummary,
    UGParityArtifact,
    UGParityCase,
    UGTensorSummary,
)
from sglang.srt.ug.sampling import build_bagel_denoise_schedule


def _set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _record_tensor_summary(tensor_summaries, key, tensor, output_path, runner):
    tensor_summaries[key] = UGTensorSummary.from_tensor(tensor)
    if os.getenv("SGLANG_TEST_BAGEL_PARITY_DUMP_TENSORS") == "1":
        torch.save(tensor.detach().cpu(), output_path.parent / f"{runner}.{key}.pt")


def _record_image(images, tensor_summaries, key, image, output_path, runner):
    image_path = output_path.parent / f"{runner}.{key}.png"
    image.save(image_path)
    images[key] = UGImageSummary.from_path(image_path)
    array = np.asarray(image.convert("RGB"))
    _record_tensor_summary(
        tensor_summaries,
        f"{key}_pixels",
        torch.from_numpy(array.copy()),
        output_path,
        runner,
    )


def _sampling_params(case):
    params = dict(case.sampling_params)
    return UGSamplingParams(
        height=params.get("height"),
        width=params.get("width"),
        num_inference_steps=int(params.get("num_inference_steps", 3)),
        cfg_text_scale=float(params.get("cfg_text_scale", 1.0)),
        cfg_img_scale=float(params.get("cfg_img_scale", 1.0)),
        cfg_interval=list(params.get("cfg_interval", [0.4, 1.0])),
        cfg_renorm_min=float(params.get("cfg_renorm_min", 0.0)),
        cfg_renorm_type=str(params.get("cfg_renorm_type", "global")),
        timestep_shift=float(params.get("timestep_shift", 3.0)),
    )


def _denoise_loop(predict_velocity, x_t, sampling_params, record_velocity):
    schedule = build_bagel_denoise_schedule(
        num_inference_steps=int(sampling_params.num_inference_steps),
        timestep_shift=float(sampling_params.timestep_shift),
        device=x_t.device,
    )
    for step, timestep in enumerate(schedule.timesteps):
        velocity = predict_velocity(x_t, timestep)
        if step == 0:
            record_velocity(velocity)
        x_t = x_t - velocity.to(x_t) * schedule.dts[step].to(x_t)
    return x_t


def _effective_cfg_scales(sampling_params, timestep):
    t = float(timestep.flatten()[0].detach().cpu())
    start, end = sampling_params.cfg_interval
    if t > float(start) and t <= float(end):
        return sampling_params.cfg_text_scale, sampling_params.cfg_img_scale
    return 1.0, 1.0


def _post_image_max_new_tokens(case):
    return int(case.metadata.get("post_image_max_new_tokens", 4))


def _decode_bagel_token_ids(tokenizer, token_ids):
    text = str(tokenizer.decode(list(token_ids)))
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    if "<|im_start|>" in text:
        text = text.split("<|im_start|>", 1)[1]
    return text
"""


_OFFICIAL_G_RUNNER = _COMMON_G_RUNNER + r"""


def _load_official_inferencer(official_repo, checkpoint_dir):
    from sglang.srt.ug.bagel import _ensure_bagel_transformers_compat

    _ensure_bagel_transformers_compat()
    sys.path.insert(0, str(official_repo))
    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate import load_checkpoint_and_dispatch
    from data.data_utils import add_special_tokens, pil_img2rgb
    from data.transforms import ImageTransform
    from inferencer import InterleaveInferencer
    from modeling.autoencoder import load_ae
    from modeling.bagel import Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM
    from modeling.bagel import SiglipVisionConfig, SiglipVisionModel
    from modeling.qwen2 import Qwen2Tokenizer

    llm_config = Qwen2Config.from_json_file(str(checkpoint_dir / "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    if not hasattr(llm_config, "pad_token_id"):
        llm_config.pad_token_id = 0

    vit_config = SiglipVisionConfig.from_json_file(
        str(checkpoint_dir / "vit_config.json")
    )
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(
        local_path=str(checkpoint_dir / "ae.safetensors")
    )
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            vit_config, meta=True
        )

    tokenizer = Qwen2Tokenizer.from_pretrained(str(checkpoint_dir))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    max_memory = {i: "140GiB" for i in range(max(1, torch.cuda.device_count()))}
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for name in same_device_modules:
        device_map[name] = device_map.get(name, first_device)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(checkpoint_dir / "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        offload_folder="offload",
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer, pil_img2rgb


def _build_official_context(case, inferencer, pil_img2rgb):
    gen_context = inferencer.init_gen_context()
    cfg_text_context = deepcopy(gen_context)
    cfg_img_context = deepcopy(gen_context)
    image_shape = (
        case.sampling_params.get("height") or 1024,
        case.sampling_params.get("width") or 1024,
    )
    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        if case.task in ("image_edit", "interleave"):
            image = Image.open(case.image_path)
            image = inferencer.vae_transform.resize_transform(pil_img2rgb(image))
            gen_context = inferencer.update_context_image(
                image,
                gen_context,
                vae=True,
            )
            image_shape = image.size[::-1]
            cfg_text_context = deepcopy(gen_context)
        elif case.task != "text_to_image":
            raise ValueError(f"Unsupported BAGEL G parity task: {case.task}")

        cfg_text_context = deepcopy(gen_context)
        gen_context = inferencer.update_context_text(case.prompt, gen_context)
        cfg_img_context = inferencer.update_context_text(case.prompt, cfg_img_context)
    return gen_context, cfg_text_context, cfg_img_context, tuple(image_shape)


def _generate_official_text_from_context(inferencer, gen_context, max_new_tokens):
    generation_input = inferencer.model.prepare_start_tokens(
        gen_context["kv_lens"],
        gen_context["ropes"],
        inferencer.new_token_ids,
    )
    output = inferencer.model.generate_text(
        past_key_values=gen_context["past_key_values"],
        max_length=int(max_new_tokens),
        do_sample=False,
        temperature=1.0,
        end_token_id=inferencer.new_token_ids["eos_token_id"],
        **generation_input,
    )
    token_ids = tuple(int(token_id) for token_id in output[:, 0].detach().cpu().tolist())
    return _decode_bagel_token_ids(inferencer.tokenizer, token_ids), token_ids


case = UGParityCase.read_json(sys.argv[1])
output_path = Path(sys.argv[2])
runner = sys.argv[3]

try:
    _set_seed(case.seed)
    official_repo = Path(case.metadata["official_repo"])
    checkpoint_dir = Path(case.metadata["checkpoint"])
    sampling_params = _sampling_params(case)
    inferencer, pil_img2rgb = _load_official_inferencer(
        official_repo,
        checkpoint_dir,
    )
    tensor_summaries = {}
    images = {}
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        enabled=torch.cuda.is_available(),
        dtype=torch.bfloat16,
    ):
        inferencer.model.language_model.model.enable_taylorseer = False
        gen_context, cfg_text_context, cfg_img_context, image_shape = (
            _build_official_context(case, inferencer, pil_img2rgb)
        )
        _set_seed(case.seed)
        generation_input = inferencer.model.prepare_vae_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            image_sizes=[image_shape],
            new_token_ids=inferencer.new_token_ids,
        )
        generation_input_cfg_text = inferencer.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=cfg_text_context["ropes"],
            image_sizes=[image_shape],
        )
        generation_input_cfg_img = inferencer.model.prepare_vae_latent_cfg(
            curr_kvlens=cfg_img_context["kv_lens"],
            curr_rope=cfg_img_context["ropes"],
            image_sizes=[image_shape],
        )
        x_t = generation_input["packed_init_noises"]
        _record_tensor_summary(tensor_summaries, "init_noise", x_t, output_path, runner)

        def _predict_velocity(latents, timestep):
            timestep = torch.full(
                (latents.shape[0],),
                float(timestep.item()),
                device=latents.device,
                dtype=latents.dtype,
            )
            cfg_text_scale, cfg_img_scale = _effective_cfg_scales(
                sampling_params,
                timestep,
            )
            return inferencer.model._forward_flow(
                x_t=latents,
                timestep=timestep,
                packed_vae_token_indexes=generation_input["packed_vae_token_indexes"],
                packed_vae_position_ids=generation_input["packed_vae_position_ids"],
                packed_text_ids=generation_input["packed_text_ids"],
                packed_text_indexes=generation_input["packed_text_indexes"],
                packed_position_ids=generation_input["packed_position_ids"],
                packed_indexes=generation_input["packed_indexes"],
                packed_seqlens=generation_input["packed_seqlens"],
                key_values_lens=generation_input["key_values_lens"],
                past_key_values=gen_context["past_key_values"],
                packed_key_value_indexes=generation_input["packed_key_value_indexes"],
                cfg_renorm_min=sampling_params.cfg_renorm_min,
                cfg_renorm_type=sampling_params.cfg_renorm_type,
                cfg_text_scale=cfg_text_scale,
                cfg_text_packed_position_ids=(
                    generation_input_cfg_text["cfg_packed_position_ids"]
                ),
                cfg_text_packed_query_indexes=(
                    generation_input_cfg_text["cfg_packed_query_indexes"]
                ),
                cfg_text_key_values_lens=generation_input_cfg_text[
                    "cfg_key_values_lens"
                ],
                cfg_text_past_key_values=cfg_text_context["past_key_values"],
                cfg_text_packed_key_value_indexes=(
                    generation_input_cfg_text["cfg_packed_key_value_indexes"]
                ),
                cfg_img_scale=cfg_img_scale,
                cfg_img_packed_position_ids=(
                    generation_input_cfg_img["cfg_packed_position_ids"]
                ),
                cfg_img_packed_query_indexes=(
                    generation_input_cfg_img["cfg_packed_query_indexes"]
                ),
                cfg_img_key_values_lens=generation_input_cfg_img[
                    "cfg_key_values_lens"
                ],
                cfg_img_past_key_values=cfg_img_context["past_key_values"],
                cfg_img_packed_key_value_indexes=(
                    generation_input_cfg_img["cfg_packed_key_value_indexes"]
                ),
            )

        final_latents = _denoise_loop(
            _predict_velocity,
            x_t,
            sampling_params,
            lambda velocity: _record_tensor_summary(
                tensor_summaries,
                "velocity_00",
                velocity,
                output_path,
                runner,
            ),
        )
        _record_tensor_summary(
            tensor_summaries,
            "final_latents",
            final_latents,
            output_path,
            runner,
        )
        image = inferencer.decode_image(final_latents, image_shape)
        _record_image(images, tensor_summaries, "generated_image", image, output_path, runner)
        text = None
        token_ids = {}
        if case.task == "interleave":
            gen_context = inferencer.update_context_image(
                image,
                gen_context,
                vae=True,
                vit=True,
            )
            text, post_image_token_ids = _generate_official_text_from_context(
                inferencer,
                gen_context,
                _post_image_max_new_tokens(case),
            )
            token_ids["post_image_text"] = post_image_token_ids

    artifact = UGParityArtifact(
        case_id=case.case_id,
        runner=runner,
        task=case.task,
        text=text,
        token_ids=token_ids,
        images=images,
        tensors=tensor_summaries,
        metadata={
            "checkpoint": str(checkpoint_dir),
            "implementation": "official",
            "mode": case.metadata.get("mode"),
            "image_shape": list(image_shape),
            "cfg": {
                "text": sampling_params.cfg_text_scale,
                "image": sampling_params.cfg_img_scale,
            },
        },
    )
except Exception:
    artifact = UGParityArtifact(
        case_id=case.case_id,
        runner=runner,
        task=case.task,
        error=traceback.format_exc(),
    )
artifact.write_json(output_path)
"""


_SGLANG_G_RUNNER = _COMMON_G_RUNNER + r"""

_ATTENTION_BACKEND_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_ATTENTION_BACKEND"


class _NoopSender:
    def send_output(self, *args, **kwargs):
        del args, kwargs


def _replace_sender_with_noop(scheduler, name):
    sender = getattr(scheduler, name, None)
    socket = getattr(sender, "socket", None)
    if socket is not None:
        socket.close(linger=0)
    setattr(scheduler, name, _NoopSender())


def _write_language_model_view(checkpoint_dir, output_dir):
    config_path = checkpoint_dir / "llm_config.json"
    weight_path = checkpoint_dir / "ema.safetensors"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config.update(
        {
            "architectures": ["BAGELQwen2MoTForCausalLM"],
            "bagel_checkpoint_dir": str(checkpoint_dir),
            "bagel_enable_visual_feature_extractors": True,
            "bagel_connector_act": "gelu_pytorch_tanh",
            "bagel_latent_patch_size": 2,
            "bagel_max_latent_size": 64,
            "bagel_max_latent_tokens": 64 * 64,
            "bagel_vit_max_num_patch_per_side": 70,
            "layer_module": "Qwen2MoTDecoderLayer",
            "qk_norm": True,
            "tie_word_embeddings": False,
        }
    )
    (output_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    os.symlink(weight_path, output_dir / "model.safetensors")
    return output_dir


def _bagel_logical_rope(adapter, session_id):
    state = adapter.backend._state_for(session_id)
    return int((state.gen_context.get("ropes") or [0])[0])


def _decode_sglang_text_like_official(runtime, adapter, handle, max_new_tokens):
    start_token_id = int(adapter.backend.inferencer.new_token_ids["bos_token_id"])
    end_token_id = int(adapter.backend.inferencer.new_token_ids["eos_token_id"])
    generated = []
    next_token_ids = []
    position_ids = []
    current_token = start_token_id
    current_handle = handle
    for step in range(int(max_new_tokens)):
        rope = _bagel_logical_rope(adapter, current_handle.session_id)
        decoded = runtime.decode_text(
            current_handle,
            max_new_tokens=1,
            start_token_id=current_token,
            position_ids=[rope],
            drop_previous_output=step > 0,
            greedy=True,
        )
        generated.append(int(current_token))
        position_ids.append(int(rope))
        current_handle = decoded.session
        if not decoded.output_ids:
            break
        next_token = int(decoded.output_ids[0])
        next_token_ids.append(next_token)
        if next_token == end_token_id:
            break
        current_token = next_token
    return tuple(generated), tuple(next_token_ids), tuple(position_ids), current_handle


case = UGParityCase.read_json(sys.argv[1])
output_path = Path(sys.argv[2])
runner = sys.argv[3]
scheduler = None
runtime = None

try:
    import torch.distributed as dist

    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.server_args import (
        PortArgs,
        ServerArgs,
        set_global_server_args_for_scheduler,
    )
    from sglang.srt.ug.adapter import UGModelRunnerAdapter
    from sglang.srt.ug.bagel import BAGELUGModelAdapter
    from sglang.srt.ug.runtime import (
        UGInterleavedMessage,
        UGLatentDecodeRequest,
        UGLatentPrepareRequest,
        UGSessionRuntime,
        UGVelocityRequest,
    )
    from sglang.srt.ug.srt_executor import UGSRTSchedulerExecutor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SGLang BAGEL G parity")

    _set_seed(case.seed)
    checkpoint_dir = Path(case.metadata["checkpoint"])
    sampling_params = _sampling_params(case)
    tensor_summaries = {}
    images = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = _write_language_model_view(checkpoint_dir, Path(tmpdir))
        server_args = ServerArgs(
            model_path=str(model_path),
            tokenizer_path=str(checkpoint_dir),
            trust_remote_code=True,
            dtype="bfloat16",
            tp_size=1,
            pp_size=1,
            dp_size=1,
            disable_cuda_graph=True,
            disable_piecewise_cuda_graph=True,
            disable_overlap_schedule=True,
            skip_server_warmup=True,
            attention_backend=os.getenv(_ATTENTION_BACKEND_ENV) or None,
            mem_fraction_static=float(
                os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_MEM_FRACTION", "0.35")
            ),
            chunked_prefill_size=int(
                os.getenv("SGLANG_TEST_BAGEL_QWEN2_MOT_CHUNKED_PREFILL", "256")
            ),
            log_level="error",
        )
        server_args.check_server_args()
        set_global_server_args_for_scheduler(server_args)

        scheduler = Scheduler(
            server_args,
            PortArgs.init_new(server_args),
            gpu_id=0,
            tp_rank=0,
            moe_ep_rank=0,
            pp_rank=0,
            attn_cp_rank=0,
            moe_dp_rank=0,
            dp_rank=None,
        )
        _replace_sender_with_noop(scheduler, "send_to_tokenizer")
        _replace_sender_with_noop(scheduler, "send_to_detokenizer")
        executor = UGSRTSchedulerExecutor(scheduler, max_sync_steps=64)
        adapter = BAGELUGModelAdapter(
            str(checkpoint_dir),
            native_srt_denoise_executor=(
                executor.create_bagel_native_srt_denoise_executor()
            ),
            native_srt_u_context=True,
        )
        runtime = UGSessionRuntime(
            model_runner=UGModelRunnerAdapter(adapter),
            session_controller=scheduler.session_controller,
            srt_request_executor=executor,
            tokenizer=scheduler.tokenizer,
            vocab_size=scheduler.model_config.vocab_size,
        )

        messages = []
        if case.task in ("image_edit", "interleave"):
            image = Image.open(case.image_path)
            messages.append(UGInterleavedMessage(type="image", content=image))
        elif case.task != "text_to_image":
            raise ValueError(f"Unsupported BAGEL G parity task: {case.task}")
        messages.append(UGInterleavedMessage(type="text", content=case.prompt))

        handle = runtime.prefill_interleaved(
            messages,
            session_id=f"bagel-g-parity-{case.task}",
        )
        marker = runtime.decode_next_segment(handle)
        if marker.type != "image_marker":
            raise RuntimeError(f"Expected image marker before G, got {marker.type}")

        latent_prepare = runtime.prepare_latents(
            UGLatentPrepareRequest(
                session=handle,
                sampling_params=sampling_params,
                seed=case.seed,
            )
        )
        x_t = latent_prepare.latent_tokens
        _record_tensor_summary(tensor_summaries, "init_noise", x_t, output_path, runner)

        def _predict_velocity(latents, timestep):
            response = runtime.predict_velocity(
                UGVelocityRequest(
                    session=handle,
                    latent_tokens=latents,
                    timestep=timestep.reshape(1),
                    latent_position_ids=latent_prepare.latent_position_ids,
                    sampling_params=sampling_params,
                )
            )
            return response.velocity

        final_latents = _denoise_loop(
            _predict_velocity,
            x_t,
            sampling_params,
            lambda velocity: _record_tensor_summary(
                tensor_summaries,
                "velocity_00",
                velocity,
                output_path,
                runner,
            ),
        )
        _record_tensor_summary(
            tensor_summaries,
            "final_latents",
            final_latents,
            output_path,
            runner,
        )
        image = runtime.decode_latents_to_image(
            UGLatentDecodeRequest(
                session=handle,
                latent_tokens=final_latents,
                sampling_params=sampling_params,
            )
        )
        _record_image(images, tensor_summaries, "generated_image", image, output_path, runner)
        text = None
        token_ids = {}
        decoded_handle = handle
        post_image_next_ids = ()
        post_image_position_ids = ()
        if case.task == "interleave":
            decoded_handle = runtime.append_generated_image(handle, image=image)
            (
                post_image_token_ids,
                post_image_next_ids,
                post_image_position_ids,
                decoded_handle,
            ) = _decode_sglang_text_like_official(
                runtime,
                adapter,
                decoded_handle,
                _post_image_max_new_tokens(case),
            )
            text = _decode_bagel_token_ids(
                adapter.backend.inferencer.tokenizer,
                post_image_token_ids,
            )
            token_ids["post_image_text"] = post_image_token_ids

        debug_counters = runtime.get_debug_counters(decoded_handle)
        debug_counters.update(
            {
                "temp_g_forward_count": executor.temp_g_forward_count,
                "temp_g_allocated_token_count": executor.temp_g_allocated_token_count,
                "bagel_post_image_next_ids": list(post_image_next_ids),
                "bagel_post_image_position_ids": list(post_image_position_ids),
            }
        )
        artifact = UGParityArtifact(
            case_id=case.case_id,
            runner=runner,
            task=case.task,
            text=text,
            token_ids=token_ids,
            images=images,
            tensors=tensor_summaries,
            debug_counters=debug_counters,
            metadata={
                "checkpoint": str(checkpoint_dir),
                "implementation": "sglang",
                "mode": case.metadata.get("mode"),
                "attention_backend": os.getenv(_ATTENTION_BACKEND_ENV) or "default",
                "image_shape": list(
                    adapter.backend._image_shape_from_params(
                        sampling_params,
                        adapter.backend._state_for(handle.session_id).image_shape,
                    )
                ),
                "cfg": {
                    "text": sampling_params.cfg_text_scale,
                    "image": sampling_params.cfg_img_scale,
                },
            },
        )
        runtime.close_session(decoded_handle)
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    torch.cuda.empty_cache()
except Exception:
    artifact = UGParityArtifact(
        case_id=case.case_id,
        runner=runner,
        task=case.task,
        error=traceback.format_exc(),
    )
    try:
        if runtime is not None:
            runtime.close_session(f"bagel-g-parity-{case.task}")
    except Exception:
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
artifact.write_json(output_path)
"""


if __name__ == "__main__":
    unittest.main(verbosity=3)
