# SPDX-License-Identifier: Apache-2.0
"""Opt-in BAGEL official-vs-SGLang VLM parity smoke.

Usage:
CUDA_VISIBLE_DEVICES=6 \
SGLANG_TEST_BAGEL_OFFICIAL_REPO=/data/BAGEL \
SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL=/data/models/BAGEL-7B-MoT \
SGLANG_TEST_BAGEL_PARITY_OUTPUT=/tmp/ug-vlm-parity \
python3 test/registered/scheduler/test_bagel_vlm_official_parity.py
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
    est_time=600,
    suite="stage-b-test-1-gpu-large",
    disabled=(
        "Manual BAGEL VLM official parity smoke; requires "
        "SGLANG_TEST_BAGEL_OFFICIAL_REPO and "
        "SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL"
    ),
)

_OFFICIAL_REPO_ENV = "SGLANG_TEST_BAGEL_OFFICIAL_REPO"
_MODEL_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL"
_OUTPUT_ENV = "SGLANG_TEST_BAGEL_PARITY_OUTPUT"
_IMAGE_ENV = "SGLANG_TEST_BAGEL_VLM_IMAGE"
_PROMPT_ENV = "SGLANG_TEST_BAGEL_VLM_PROMPT"
_MAX_TOKENS_ENV = "SGLANG_TEST_BAGEL_VLM_MAX_NEW_TOKENS"
_GPU_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_GPU_ID"
_ATTENTION_BACKEND_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_ATTENTION_BACKEND"
_CAPTURE_TENSORS_ENV = "SGLANG_TEST_BAGEL_VLM_CAPTURE_TENSORS"
_WARM_VIT_ENV = "SGLANG_TEST_BAGEL_VLM_WARM_VIT"
_SYNC_AFTER_SRT_STEP_ENV = "SGLANG_TEST_BAGEL_VLM_SYNC_AFTER_SRT_STEP"


def _has_live_env() -> bool:
    return bool(os.getenv(_OFFICIAL_REPO_ENV) and os.getenv(_MODEL_ENV))


@unittest.skipUnless(
    _has_live_env(),
    f"Set {_OFFICIAL_REPO_ENV} and {_MODEL_ENV} for BAGEL VLM parity smoke",
)
class TestBAGELVLMOfficialParity(CustomTestCase):
    def test_vlm_text_matches_official_short_greedy_decode(self):
        official_repo = Path(os.environ[_OFFICIAL_REPO_ENV]).expanduser()
        checkpoint_dir = Path(os.environ[_MODEL_ENV]).expanduser()
        image_env = os.getenv(_IMAGE_ENV)
        image_path = (
            None
            if image_env == "none"
            else Path(
                image_env or official_repo / "test_images" / "women.jpg"
            ).expanduser()
        )
        self.assertTrue(official_repo.exists(), official_repo)
        self.assertTrue(checkpoint_dir.exists(), checkpoint_dir)
        if image_path is not None:
            self.assertTrue(image_path.exists(), image_path)

        output_dir = Path(
            os.getenv(_OUTPUT_ENV) or tempfile.mkdtemp(prefix="ug-vlm-parity-")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        max_length = int(os.getenv(_MAX_TOKENS_ENV, "8"))
        case = UGParityCase(
            case_id="bagel-vlm-official-parity-smoke",
            task="vlm",
            prompt=os.getenv(_PROMPT_ENV, "Describe this image briefly."),
            image_path=str(image_path) if image_path is not None else None,
            seed=123,
            sampling_params={
                "max_length": max_length,
                "do_sample": False,
                "temperature": 1.0,
            },
            dump_points=("text", "token_ids"),
            metadata={
                "official_repo": str(official_repo),
                "checkpoint": str(checkpoint_dir),
                "vlm_image_mode": "text_only" if image_path is None else "vit_only",
            },
        )
        case_path = output_dir / "case.json"
        official_path = output_dir / "reference.official.json"
        sglang_path = output_dir / "candidate.sglang.json"
        report_path = output_dir / "report.json"
        case.write_json(case_path)

        _run_vlm_subprocess(
            _OFFICIAL_VLM_RUNNER,
            case_path=case_path,
            output_path=official_path,
            runner="official",
        )
        _run_vlm_subprocess(
            _SGLANG_VLM_RUNNER,
            case_path=case_path,
            output_path=sglang_path,
            runner="sglang",
        )

        reference = UGParityArtifact.read_json(official_path)
        candidate = UGParityArtifact.read_json(sglang_path)
        compare_tensors = (
            os.getenv(_CAPTURE_TENSORS_ENV) == "1" or "tensors" in case.dump_points
        )
        report = compare_ug_parity_artifacts(
            reference,
            candidate,
            tolerance=UGParityTolerance(compare_tensors=compare_tensors),
        )
        report.write_json(report_path)

        self.assertTrue(case_path.exists())
        self.assertTrue(official_path.exists())
        self.assertTrue(sglang_path.exists())
        self.assertTrue(report_path.exists())
        self.assertTrue(report.passed, report.to_json())


def _run_vlm_subprocess(
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


_OFFICIAL_VLM_RUNNER = r"""
import json
import os
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sglang.srt.ug.parity import UGParityArtifact, UGParityCase, UGTensorSummary


def _set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _decode_bagel_text(tokenizer, token_ids):
    text = tokenizer.decode(list(token_ids))
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    if "<|im_start|>" in text:
        text = text.split("<|im_start|>", 1)[1]
    return text


def _bagel_logical_rope(adapter, session_id):
    state = adapter.backend._state_for(session_id)
    return int((state.gen_context.get("ropes") or [0])[0])


def _decode_bagel_text_like_official(runtime, adapter, handle, max_length):
    start_token_id = int(adapter.backend.inferencer.new_token_ids["bos_token_id"])
    end_token_id = int(adapter.backend.inferencer.new_token_ids["eos_token_id"])
    generated = []
    unused_next_ids = []
    position_ids = []
    current_token = start_token_id
    current_handle = handle
    for step in range(max_length):
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
        unused_next_ids.append(next_token)
        if next_token == end_token_id:
            break
        current_token = next_token
    return tuple(generated), tuple(unused_next_ids), tuple(position_ids), current_handle


def _summarize_official_vit_packed_sequence(inferencer, generation_input):
    model = inferencer.model
    packed_text_embedding = model.language_model.model.embed_tokens(
        generation_input["packed_text_ids"]
    )
    seq_len = int(generation_input["packed_seqlens"].sum().item())
    packed_sequence = packed_text_embedding.new_zeros((seq_len, model.hidden_size))
    packed_sequence[generation_input["packed_text_indexes"]] = packed_text_embedding

    vit_token_seqlens = generation_input["vit_token_seqlens"]
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
    cu_seqlens = cu_seqlens.to(torch.int32)
    max_seqlen = torch.max(vit_token_seqlens).item()
    packed_vit_token_embed = model.vit_model(
        packed_pixel_values=generation_input["packed_vit_tokens"],
        packed_flattened_position_ids=generation_input["packed_vit_position_ids"],
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
    )
    packed_vit_token_embed = model.connector(packed_vit_token_embed)
    packed_vit_token_embed = packed_vit_token_embed + model.vit_pos_embed(
        generation_input["packed_vit_position_ids"]
    )
    if packed_vit_token_embed.dtype != packed_sequence.dtype:
        packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
    packed_sequence[generation_input["packed_vit_token_indexes"]] = (
        packed_vit_token_embed
    )
    return UGTensorSummary.from_tensor(packed_sequence)


def _load_official_inferencer(official_repo, checkpoint_dir):
    from sglang.srt.ug.bagel import _ensure_bagel_transformers_compat

    _ensure_bagel_transformers_compat()
    sys.path.insert(0, str(official_repo))
    from accelerate import infer_auto_device_map, init_empty_weights
    from accelerate import load_checkpoint_and_dispatch
    from data.data_utils import add_special_tokens
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
    return InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )


case = UGParityCase.read_json(sys.argv[1])
output_path = Path(sys.argv[2])
runner = sys.argv[3]
_DUMP_DEBUG_TENSORS = os.getenv("SGLANG_TEST_BAGEL_PARITY_DUMP_TENSORS") == "1"
_CAPTURE_TENSORS = (
    os.getenv("SGLANG_TEST_BAGEL_VLM_CAPTURE_TENSORS") == "1"
    or "tensors" in case.dump_points
)


def _record_tensor_summary(tensor_summaries, key, tensor):
    tensor_summaries[key] = UGTensorSummary.from_tensor(tensor)
    if _DUMP_DEBUG_TENSORS:
        torch.save(tensor.detach().cpu(), output_path.parent / f"{runner}.{key}.pt")


try:
    _set_seed(case.seed)
    official_repo = Path(case.metadata["official_repo"])
    checkpoint_dir = Path(case.metadata["checkpoint"])
    inferencer = _load_official_inferencer(official_repo, checkpoint_dir)
    captured = {}
    tensor_summaries = {}
    original_generate_text = inferencer.model.generate_text
    original_forward_cache_update_vit = inferencer.model.forward_cache_update_vit
    original_forward_inference = inferencer.model.language_model.forward_inference

    def _capture_generate_text(*args, **kwargs):
        output = original_generate_text(*args, **kwargs)
        captured["generated"] = [
            int(token_id) for token_id in output[:, 0].detach().cpu().tolist()
        ]
        return output

    inferencer.model.generate_text = _capture_generate_text

    def _official_scope_from_forward_kwargs(kwargs):
        is_causal = kwargs.get("is_causal")
        if is_causal is False:
            return "image"
        if is_causal is not True:
            return None
        query_lens = kwargs.get("query_lens")
        if query_lens is not None:
            query_tokens = int(query_lens.sum().item())
        else:
            query_tokens = int(kwargs["packed_query_sequence"].shape[0])
        if query_tokens == 1:
            if "decode_00_hidden" not in tensor_summaries:
                return "decode_00"
            return None
        if "text_prefill_hidden" not in tensor_summaries:
            return "text_prefill"
        return None

    def _capture_forward_inference(*args, **kwargs):
        scope = _official_scope_from_forward_kwargs(kwargs)
        output = original_forward_inference(*args, **kwargs)
        if scope == "image" and "image_block_hidden" not in tensor_summaries:
            tensor_summaries["image_block_hidden"] = UGTensorSummary.from_tensor(
                output.packed_query_sequence
            )
        elif scope and f"{scope}_hidden" not in tensor_summaries:
            tensor_summaries[f"{scope}_hidden"] = UGTensorSummary.from_tensor(
                output.packed_query_sequence
            )
        return output

    if _CAPTURE_TENSORS:
        inferencer.model.language_model.forward_inference = _capture_forward_inference

    def _install_official_image_layer_capture():
        layers = inferencer.model.language_model.model.layers
        active_layer = {"idx": None, "scope": None}

        def _tensor_key(scope, suffix):
            return f"image_layer_00_{suffix}" if scope == "image" else f"{scope}_layer_00_{suffix}"

        def _capture_layer0_module(name, module, output_selector=lambda x: x):
            original_forward = module.forward

            def _capture_forward(*args, **kwargs):
                output = original_forward(*args, **kwargs)
                if active_layer["idx"] == 0 and active_layer["scope"]:
                    key = _tensor_key(active_layer["scope"], name)
                    if key not in tensor_summaries:
                        selected_output = output_selector(output)
                        if isinstance(selected_output, tuple):
                            selected_output = selected_output[0]
                        _record_tensor_summary(
                            tensor_summaries, key, selected_output
                        )
                return output

            module.forward = _capture_forward

        def _capture_layer0_attention(module):
            original_forward = module.forward_inference

            def _capture_forward(*args, **kwargs):
                if active_layer["idx"] == 0 and active_layer["scope"]:
                    scope = active_layer["scope"]
                    for suffix, tensor in (
                        ("q_norm_weight", module.q_norm.weight),
                        ("k_norm_weight", module.k_norm.weight),
                    ):
                        key = _tensor_key(scope, suffix)
                        if key not in tensor_summaries:
                            _record_tensor_summary(tensor_summaries, key, tensor)
                    sequence = kwargs["packed_query_sequence"]
                    cos, sin = kwargs["packed_query_position_embeddings"]
                    q = module.q_proj(sequence).view(
                        -1, module.num_heads, module.head_dim
                    )
                    k = module.k_proj(sequence).view(
                        -1, module.num_key_value_heads, module.head_dim
                    )
                    v = module.v_proj(sequence).view(
                        -1, module.num_key_value_heads, module.head_dim
                    )
                    q_norm = module.q_norm(q)
                    k_norm = module.k_norm(k)
                    from modeling.qwen2.modeling_qwen2 import apply_rotary_pos_emb

                    q_rope, k_rope = apply_rotary_pos_emb(
                        q_norm, k_norm, cos, sin, unsqueeze_dim=1
                    )
                    for suffix, tensor in (
                        ("q_raw", q),
                        ("k_raw", k),
                        ("q_norm", q_norm),
                        ("k_norm", k_norm),
                        ("v", v),
                        ("q_rope", q_rope),
                        ("k_rope", k_rope),
                    ):
                        key = _tensor_key(scope, suffix)
                        if key not in tensor_summaries:
                            _record_tensor_summary(tensor_summaries, key, tensor)
                output = original_forward(*args, **kwargs)
                if active_layer["idx"] == 0 and active_layer["scope"]:
                    key = _tensor_key(active_layer["scope"], "attn")
                    if key not in tensor_summaries:
                        _record_tensor_summary(tensor_summaries, key, output[0])
                return output

            module.forward_inference = _capture_forward

        layer0 = layers[0]
        _capture_layer0_module("input_norm", layer0.input_layernorm)
        _capture_layer0_attention(layer0.self_attn)
        _capture_layer0_module("post_attn_norm", layer0.post_attention_layernorm)
        _capture_layer0_module("mlp", layer0.mlp)

        for layer_idx, layer in enumerate(layers):
            original_layer_forward = layer.forward_inference

            def _capture_layer_forward(*args, _idx=layer_idx, _orig=original_layer_forward, **kwargs):
                scope = _official_scope_from_forward_kwargs(kwargs)
                if scope and _idx == 0:
                    key = (
                        "image_layer_00_input"
                        if scope == "image"
                        else f"{scope}_layer_00_input"
                    )
                    if key not in tensor_summaries:
                        _record_tensor_summary(
                            tensor_summaries, key, kwargs["packed_query_sequence"]
                        )
                active_layer["idx"] = _idx if scope else None
                active_layer["scope"] = scope
                output = _orig(*args, **kwargs)
                active_layer["idx"] = None
                active_layer["scope"] = None
                if scope:
                    key = (
                        f"image_layer_{_idx:02d}"
                        if scope == "image"
                        else f"{scope}_layer_{_idx:02d}"
                    )
                    if key not in tensor_summaries:
                        _record_tensor_summary(tensor_summaries, key, output[0])
                return output

            layer.forward_inference = _capture_layer_forward

    if _CAPTURE_TENSORS:
        _install_official_image_layer_capture()

    def _capture_forward_cache_update_vit(*args, **kwargs):
        if "vit_packed_sequence" not in tensor_summaries:
            with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                tensor_summaries["vit_packed_sequence"] = (
                    _summarize_official_vit_packed_sequence(inferencer, kwargs)
                )
        return original_forward_cache_update_vit(*args, **kwargs)

    if _CAPTURE_TENSORS:
        inferencer.model.forward_cache_update_vit = _capture_forward_cache_update_vit
    image = Image.open(case.image_path) if case.image_path is not None else None
    result = inferencer(
        image=image,
        text=case.prompt,
        understanding_output=True,
        think=False,
        do_sample=bool(case.sampling_params.get("do_sample", False)),
        text_temperature=float(case.sampling_params.get("temperature", 1.0)),
        max_think_token_n=int(case.sampling_params.get("max_length", 8)),
    )
    token_ids = tuple(captured.get("generated", ()))
    artifact = UGParityArtifact(
        case_id=case.case_id,
        runner=runner,
        task=case.task,
        text=result.get("text"),
        token_ids={"generated": token_ids},
        tensors=tensor_summaries,
        metadata={"checkpoint": str(checkpoint_dir), "implementation": "official"},
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


_SGLANG_VLM_RUNNER = r"""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from sglang.srt.ug.parity import UGParityArtifact, UGParityCase, UGTensorSummary


_ATTENTION_BACKEND_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_ATTENTION_BACKEND"
_WARM_VIT_ENV = "SGLANG_TEST_BAGEL_VLM_WARM_VIT"
_SYNC_AFTER_SRT_STEP_ENV = "SGLANG_TEST_BAGEL_VLM_SYNC_AFTER_SRT_STEP"


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


def _decode_bagel_text(tokenizer, token_ids):
    text = tokenizer.decode(list(token_ids))
    if "<|im_end|>" in text:
        text = text.split("<|im_end|>", 1)[0]
    if "<|im_start|>" in text:
        text = text.split("<|im_start|>", 1)[1]
    return text


def _bagel_logical_rope(adapter, session_id):
    state = adapter.backend._state_for(session_id)
    return int((state.gen_context.get("ropes") or [0])[0])


def _decode_bagel_text_like_official(runtime, adapter, handle, max_length):
    start_token_id = int(adapter.backend.inferencer.new_token_ids["bos_token_id"])
    end_token_id = int(adapter.backend.inferencer.new_token_ids["eos_token_id"])
    generated = []
    unused_next_ids = []
    position_ids = []
    current_token = start_token_id
    current_handle = handle
    for step in range(max_length):
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
        unused_next_ids.append(next_token)
        if next_token == end_token_id:
            break
        current_token = next_token
    return tuple(generated), tuple(unused_next_ids), tuple(position_ids), current_handle


def _summarize_sglang_vit_packed_sequence(adapter, image):
    backend = adapter.backend
    model = backend._native_srt_model()
    image = backend._prepare_image(image)
    generation_input, _, _ = model.prepare_vit_images(
        curr_kvlens=[0],
        curr_rope=[0],
        images=[image],
        transforms=backend.inferencer.vit_transform,
        new_token_ids=backend.inferencer.new_token_ids,
    )
    with torch.no_grad():
        return UGTensorSummary.from_tensor(
            model.embed_bagel_vit_image(generation_input)
        )


def _install_sync_after_srt_step(executor):
    original_run_scheduler_step = executor._run_scheduler_step

    def _run_scheduler_step_with_sync(*args, **kwargs):
        batch = original_run_scheduler_step(*args, **kwargs)
        torch.cuda.synchronize()
        return batch

    executor._run_scheduler_step = _run_scheduler_step_with_sync


def _install_sglang_image_block_hidden_capture(adapter, tensor_summaries, capture_debug):
    model = adapter.backend._native_srt_model()
    original_model_forward = model.model.forward
    active_layer = {"idx": None, "scope": None}

    def _sglang_scope_from_forward_batch(forward_batch, token_count):
        if getattr(forward_batch, "ug_g_non_causal_query_attention", False):
            return "image"
        if token_count == 1:
            if "decode_00_hidden" not in tensor_summaries:
                return "decode_00"
            return None
        if token_count > 1 and "text_prefill_hidden" not in tensor_summaries:
            return "text_prefill"
        return None

    def _tensor_key(scope, suffix):
        return f"image_layer_00_{suffix}" if scope == "image" else f"{scope}_layer_00_{suffix}"

    def _capture_model_forward(*args, **kwargs):
        output = original_model_forward(*args, **kwargs)
        forward_batch = args[2] if len(args) > 2 else kwargs.get("forward_batch")
        input_embeds = args[3] if len(args) > 3 else kwargs.get("input_embeds")
        scope = _sglang_scope_from_forward_batch(forward_batch, int(output.shape[0]))
        if scope == "image" and input_embeds is not None and "image_block_hidden" not in tensor_summaries:
            tensor_summaries["image_block_hidden"] = UGTensorSummary.from_tensor(
                output
            )
        elif scope and f"{scope}_hidden" not in tensor_summaries:
            tensor_summaries[f"{scope}_hidden"] = UGTensorSummary.from_tensor(output)
        return output

    model.model.forward = _capture_model_forward

    def _capture_layer0_module(name, module, output_selector=lambda x: x):
        original_forward = module.forward

        def _capture_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)
            if active_layer["idx"] == 0 and active_layer["scope"]:
                key = _tensor_key(active_layer["scope"], name)
                if key not in tensor_summaries:
                    selected_output = output_selector(output)
                    if isinstance(selected_output, tuple):
                        selected_output = selected_output[0]
                    _record_tensor_summary(tensor_summaries, key, selected_output)
            return output

        module.forward = _capture_forward

    def _capture_layer0_attention(module):
        original_forward = module.forward

        def _capture_forward(*args, **kwargs):
            if active_layer["idx"] == 0 and active_layer["scope"]:
                scope = active_layer["scope"]
                for suffix, tensor in (
                    ("q_norm_weight", module.q_norm.weight),
                    ("k_norm_weight", module.k_norm.weight),
                ):
                    key = _tensor_key(scope, suffix)
                    if key not in tensor_summaries:
                        _record_tensor_summary(tensor_summaries, key, tensor)
                positions = args[0] if len(args) > 0 else kwargs["positions"]
                hidden_states = args[1] if len(args) > 1 else kwargs["hidden_states"]
                forward_batch = args[2] if len(args) > 2 else kwargs["forward_batch"]
                q, _ = module.q_proj(hidden_states)
                k, _ = module.k_proj(hidden_states)
                v, _ = module.v_proj(hidden_states)
                q_view = q.view(-1, module.num_heads, module.head_dim)
                k_view = k.view(-1, module.num_kv_heads, module.head_dim)
                q_norm, k_norm = module._apply_official_qk_norm(
                    q=q,
                    k=k,
                    q_norm=module.q_norm,
                    k_norm=module.k_norm,
                )
                q_norm_view = q_norm.view(
                    -1, module.num_heads, module.head_dim
                ).clone()
                k_norm_view = k_norm.view(
                    -1, module.num_kv_heads, module.head_dim
                ).clone()
                use_official_rope = (
                    getattr(forward_batch, "ug_g_non_causal_query_attention", False)
                    or getattr(forward_batch, "ug_u_forward_metadata", None)
                    or getattr(forward_batch, "out_cache_loc", None) is not None
                )
                debug_prefix = f"capture_{scope}_layer_00"
                capture_debug.setdefault(
                    f"{debug_prefix}_use_official_rope", bool(use_official_rope)
                )
                capture_debug.setdefault(
                    f"{debug_prefix}_out_cache_loc_is_none",
                    getattr(forward_batch, "out_cache_loc", None) is None,
                )
                capture_debug.setdefault(
                    f"{debug_prefix}_ug_metadata_is_none",
                    getattr(forward_batch, "ug_u_forward_metadata", None) is None,
                )
                capture_debug.setdefault(
                    f"{debug_prefix}_g_non_causal",
                    bool(
                        getattr(
                            forward_batch,
                            "ug_g_non_causal_query_attention",
                            False,
                        )
                    ),
                )
                capture_debug.setdefault(
                    f"{debug_prefix}_positions", positions.detach().cpu().tolist()
                )
                capture_debug.setdefault(
                    f"{debug_prefix}_forward_mode",
                    str(getattr(forward_batch, "forward_mode", None)),
                )
                if use_official_rope:
                    q_rope, k_rope = module._apply_official_rotary_pos_emb(
                        positions, q_norm, k_norm
                    )
                else:
                    q_rope, k_rope = module.rotary_emb(positions, q_norm, k_norm)
                for suffix, tensor in (
                    ("q_raw", q_view),
                    ("k_raw", k_view),
                    ("q_norm", q_norm_view),
                    ("k_norm", k_norm_view),
                    ("v", v.view(-1, module.num_kv_heads, module.head_dim)),
                    ("q_rope", q_rope.view(-1, module.num_heads, module.head_dim)),
                    ("k_rope", k_rope.view(-1, module.num_kv_heads, module.head_dim)),
                ):
                    key = _tensor_key(scope, suffix)
                    if key not in tensor_summaries:
                        _record_tensor_summary(tensor_summaries, key, tensor)
            output = original_forward(*args, **kwargs)
            if active_layer["idx"] == 0 and active_layer["scope"]:
                key = _tensor_key(active_layer["scope"], "attn")
                if key not in tensor_summaries:
                    _record_tensor_summary(tensor_summaries, key, output)
            return output

        module.forward = _capture_forward

    layer0 = model.model.layers[0]
    _capture_layer0_module("input_norm", layer0.input_layernorm)
    _capture_layer0_attention(layer0.self_attn)
    _capture_layer0_module("post_attn_norm", layer0.post_attention_layernorm)
    _capture_layer0_module("mlp", layer0.mlp)

    for layer_idx, layer in enumerate(model.model.layers):
        original_layer_forward = layer.forward

        def _capture_layer_forward(*args, _idx=layer_idx, _orig=original_layer_forward, **kwargs):
            forward_batch = args[2] if len(args) > 2 else kwargs.get("forward_batch")
            token_count = int(args[1].shape[0])
            scope = _sglang_scope_from_forward_batch(forward_batch, token_count)
            if scope and _idx == 0:
                key = (
                    "image_layer_00_input"
                    if scope == "image"
                    else f"{scope}_layer_00_input"
                )
                if key not in tensor_summaries:
                    _record_tensor_summary(tensor_summaries, key, args[1])
            active_layer["idx"] = _idx if scope else None
            active_layer["scope"] = scope
            output = _orig(*args, **kwargs)
            active_layer["idx"] = None
            active_layer["scope"] = None
            if scope:
                key = (
                    f"image_layer_{_idx:02d}"
                    if scope == "image"
                    else f"{scope}_layer_{_idx:02d}"
                )
                if key not in tensor_summaries:
                    hidden_states, residual = output
                    layer_output = (
                        hidden_states + residual
                        if residual is not None
                        else hidden_states
                    )
                    _record_tensor_summary(tensor_summaries, key, layer_output)
            return output

        layer.forward = _capture_layer_forward


case = UGParityCase.read_json(sys.argv[1])
output_path = Path(sys.argv[2])
runner = sys.argv[3]
_DUMP_DEBUG_TENSORS = os.getenv("SGLANG_TEST_BAGEL_PARITY_DUMP_TENSORS") == "1"
_CAPTURE_TENSORS = (
    os.getenv("SGLANG_TEST_BAGEL_VLM_CAPTURE_TENSORS") == "1"
    or "tensors" in case.dump_points
)


def _record_tensor_summary(tensor_summaries, key, tensor):
    tensor_summaries[key] = UGTensorSummary.from_tensor(tensor)
    if _DUMP_DEBUG_TENSORS:
        torch.save(tensor.detach().cpu(), output_path.parent / f"{runner}.{key}.pt")


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
    from sglang.srt.ug.runtime import UGInterleavedMessage, UGSessionRuntime
    from sglang.srt.ug.srt_executor import UGSRTSchedulerExecutor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SGLang BAGEL VLM parity")

    torch.manual_seed(case.seed or 0)
    torch.cuda.manual_seed_all(case.seed or 0)
    checkpoint_dir = Path(case.metadata["checkpoint"])
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
        executor = UGSRTSchedulerExecutor(scheduler, max_sync_steps=32)
        if os.getenv(_SYNC_AFTER_SRT_STEP_ENV) == "1":
            _install_sync_after_srt_step(executor)
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
        tensor_summaries = {}
        capture_debug = {}
        if _CAPTURE_TENSORS:
            _install_sglang_image_block_hidden_capture(
                adapter, tensor_summaries, capture_debug
            )
        messages = []
        if case.image_path is not None:
            image = Image.open(case.image_path)
            if _CAPTURE_TENSORS:
                tensor_summaries["vit_packed_sequence"] = (
                    _summarize_sglang_vit_packed_sequence(adapter, image)
                )
            elif os.getenv(_WARM_VIT_ENV) == "1":
                _summarize_sglang_vit_packed_sequence(adapter, image)
            messages.append(
                UGInterleavedMessage(
                    type="image",
                    content={"image": image, "vae": False, "vit": True},
                )
            )
        messages.append(UGInterleavedMessage(type="text", content=case.prompt))
        handle = runtime.prefill_interleaved(
            messages,
            session_id="bagel-vlm-parity",
        )
        max_length = int(case.sampling_params.get("max_length", 8))
        token_ids, unused_next_ids, position_ids, decoded_handle = (
            _decode_bagel_text_like_official(
                runtime,
                adapter,
                handle,
                max_length,
            )
        )
        text = _decode_bagel_text(adapter.backend.inferencer.tokenizer, token_ids)
        debug_counters = runtime.get_debug_counters(decoded_handle)
        debug_counters.update(
            {
                "bagel_unused_next_ids": list(unused_next_ids),
                "bagel_decode_position_ids": list(position_ids),
            }
        )
        debug_counters.update(capture_debug)
        artifact = UGParityArtifact(
            case_id=case.case_id,
            runner=runner,
            task=case.task,
            text=text,
            token_ids={"generated": token_ids},
            tensors=tensor_summaries,
            debug_counters=debug_counters,
            metadata={
                "checkpoint": str(checkpoint_dir),
                "implementation": "sglang",
                "attention_backend": os.getenv(_ATTENTION_BACKEND_ENV) or "default",
                "bagel_decode_mode": "official_iterative",
                "bagel_vlm_warm_vit": os.getenv(_WARM_VIT_ENV) == "1",
                "bagel_vlm_sync_after_srt_step": (
                    os.getenv(_SYNC_AFTER_SRT_STEP_ENV) == "1"
                ),
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
            runtime.close_session("bagel-vlm-parity")
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
