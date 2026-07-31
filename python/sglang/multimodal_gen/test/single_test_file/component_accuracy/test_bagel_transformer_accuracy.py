# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Pinned official-vs-SGLang BAGEL Transformer accuracy test.

Run this test explicitly with ``SGLANG_BAGEL_OFFICIAL_REPO`` pointing to a
clean checkout of the pinned official BAGEL source and
``SGLANG_BAGEL_MODEL_PATH`` pointing to the pinned checkpoint snapshot. It is
kept separate from the generic Diffusers component harness because BAGEL uses
one monolithic non-Diffusers checkpoint and request-owned prefix KV caches.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.machinery
import json
import os
import subprocess
import sys
import types
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch
from safetensors import safe_open
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.configs.models.dits.bagel import BagelDiTConfig
from sglang.multimodal_gen.runtime.models.dits.bagel_transformer import (
    BagelTransformer,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

OFFICIAL_REPO_ENV = "SGLANG_BAGEL_OFFICIAL_REPO"
MODEL_PATH_ENV = "SGLANG_BAGEL_MODEL_PATH"
OFFICIAL_REPO_URL = "https://github.com/ByteDance-Seed/Bagel.git"
OFFICIAL_REPO_COMMIT = "a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f"
MODEL_REVISION = "5019f57d168e5816e8f3f701b17cc816bb7cf24b"
EMA_SIZE = 29_214_685_336
EMA_SHA256 = "0b41c43835fd737b8c948e604870da522c091dcf151f3e8d55f84781765ee1a3"

PROMPT = "Doraemon is eating dorayaki"
PROMPT_TOKEN_IDS = (151644, 35, 6215, 7291, 374, 12182, 52303, 352, 14624, 151645)
START_OF_IMAGE_TOKEN_ID = 151652
END_OF_IMAGE_TOKEN_ID = 151653
IMAGE_SIZE = 256
LATENT_SIDE = 16
LATENT_TOKEN_COUNT = LATENT_SIDE * LATENT_SIDE
PATCH_WIDTH = 64
QUERY_TOKEN_COUNT = LATENT_TOKEN_COUNT + 2
TRANSFORMER_PARAMETER_COUNT = 796
TRANSFORMER_PARAMETER_ELEMENTS = 13_625_167_424
COSINE_THRESHOLD = 0.995
GUIDANCE_SCALE = 4.0
SEED = 20260708


def _run_git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _normalize_git_url(url: str) -> str:
    normalized = url.strip().removesuffix("/").removesuffix(".git")
    if normalized.startswith("git@github.com:"):
        normalized = "https://github.com/" + normalized.removeprefix("git@github.com:")
    return normalized.lower()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_inputs() -> tuple[Path, Path]:
    official_value = os.environ.get(OFFICIAL_REPO_ENV)
    model_value = os.environ.get(MODEL_PATH_ENV)
    if not official_value and not model_value:
        pytest.skip(f"set {OFFICIAL_REPO_ENV} and {MODEL_PATH_ENV} for BAGEL parity")
    if not official_value or not model_value:
        pytest.fail(f"{OFFICIAL_REPO_ENV} and {MODEL_PATH_ENV} must be set together")
    official_repo = Path(official_value).resolve()
    model_path = Path(model_value).resolve()
    if not official_repo.is_dir() or not model_path.is_dir():
        pytest.fail("BAGEL parity source and model paths must be directories")
    return official_repo, model_path


def _verify_pins(official_repo: Path, model_path: Path) -> Path:
    commit = _run_git(official_repo, "rev-parse", "HEAD")
    if commit != OFFICIAL_REPO_COMMIT:
        pytest.fail(
            f"official BAGEL commit mismatch: {commit} != {OFFICIAL_REPO_COMMIT}"
        )
    if _run_git(official_repo, "status", "--porcelain"):
        pytest.fail("official BAGEL checkout must be clean")
    origin = _run_git(official_repo, "remote", "get-url", "origin")
    if _normalize_git_url(origin) != _normalize_git_url(OFFICIAL_REPO_URL):
        pytest.fail(f"official BAGEL origin mismatch: {origin}")

    ema_path = model_path / "ema.safetensors"
    if not ema_path.is_file() or ema_path.stat().st_size != EMA_SIZE:
        pytest.fail("pinned BAGEL ema.safetensors size mismatch")
    if _sha256(ema_path) != EMA_SHA256:
        pytest.fail("pinned BAGEL ema.safetensors SHA256 mismatch")
    return ema_path


@contextmanager
def _default_dtype(dtype: torch.dtype) -> Iterator[None]:
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@contextmanager
def _math_sdpa() -> Iterator[None]:
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        yield


def _legacy_default_rope(
    config: Any,
    device: torch.device | str | None,
    seq_len: int | None = None,
    **_: Any,
) -> tuple[torch.Tensor, float]:
    """Reproduce the unscaled Qwen2 RoPE initializer from Transformers 4.49."""
    del seq_len
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    rope_device = torch.device("cpu") if device is None else torch.device(device)
    inv_freq = 1.0 / (
        float(config.rope_theta)
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=rope_device)
            / head_dim
        )
    )
    return inv_freq, 1.0


def _official_varlen_math_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **_: Any,
) -> torch.Tensor:
    """Implement official varlen attention with the same math SDPA as SGLang."""
    del max_seqlen_q, max_seqlen_k
    if dropout_p != 0.0:
        raise ValueError("BAGEL parity requires dropout_p=0")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3 or k.shape != v.shape:
        raise ValueError("BAGEL attention expects matching [tokens, heads, dim] KV")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError("BAGEL Q heads must be divisible by KV heads")

    outputs: list[torch.Tensor] = []
    repeat_factor = q.shape[1] // k.shape[1]
    batch_size = int(cu_seqlens_q.numel()) - 1
    if int(cu_seqlens_k.numel()) - 1 != batch_size:
        raise ValueError("BAGEL query and KV batches must match")
    for index in range(batch_size):
        q_start = int(cu_seqlens_q[index].item())
        q_end = int(cu_seqlens_q[index + 1].item())
        k_start = int(cu_seqlens_k[index].item())
        k_end = int(cu_seqlens_k[index + 1].item())
        query = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        key = k[k_start:k_end]
        value = v[k_start:k_end]
        if repeat_factor != 1:
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        if causal and query.shape[-2] != key.shape[-2]:
            raise ValueError("parity only permits square causal prefill attention")
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


@contextmanager
def _official_import_compatibility(official_repo: Path) -> Iterator[None]:
    """Load pinned BAGEL under Transformers 5 without editing its checkout."""
    from transformers import modeling_rope_utils

    previous_flash_attn = sys.modules.get("flash_attn")
    flash_attn_module = types.ModuleType("flash_attn")
    flash_attn_module.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn", loader=None
    )
    setattr(flash_attn_module, "__version__", "2.5.8")
    setattr(
        flash_attn_module,
        "flash_attn_varlen_func",
        _official_varlen_math_attention,
    )
    previous_default_rope = modeling_rope_utils.ROPE_INIT_FUNCTIONS.get("default")
    previous_path = list(sys.path)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.modules["flash_attn"] = flash_attn_module
    modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _legacy_default_rope
    sys.path.insert(0, str(official_repo))
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        sys.path[:] = previous_path
        if previous_flash_attn is None:
            sys.modules.pop("flash_attn", None)
        else:
            sys.modules["flash_attn"] = previous_flash_attn
        if previous_default_rope is None:
            modeling_rope_utils.ROPE_INIT_FUNCTIONS.pop("default", None)
        else:
            modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = previous_default_rope


def _copy_exact_parameters(
    model: nn.Module,
    ema_path: Path,
    device: torch.device,
) -> tuple[int, int]:
    parameters = dict(model.named_parameters())
    loaded: set[str] = set()
    with torch.no_grad(), safe_open(ema_path, framework="pt", device="cpu") as source:
        source_keys = set(source.keys())
        missing = sorted(parameters.keys() - source_keys)
        if missing:
            raise ValueError(
                f"official BAGEL parameters missing from checkpoint: {missing}"
            )
        for name, parameter in parameters.items():
            tensor = source.get_tensor(name)
            if tuple(parameter.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"official BAGEL shape mismatch for {name}: "
                    f"{tuple(parameter.shape)} != {tuple(tensor.shape)}"
                )
            parameter.copy_(tensor.to(device=device, dtype=parameter.dtype))
            loaded.add(name)
    return len(loaded), sum(parameter.numel() for parameter in parameters.values())


def _load_official_model(
    official_repo: Path,
    model_path: Path,
    ema_path: Path,
    device: torch.device,
) -> tuple[nn.Module, type[Any], int, int]:
    with _official_import_compatibility(official_repo):
        from modeling.bagel import (  # type: ignore[import-not-found]
            Bagel,
            BagelConfig,
            Qwen2Config,
            Qwen2ForCausalLM,
        )
        from modeling.bagel.qwen2_navit import (  # type: ignore[import-not-found]
            NaiveCache,
        )

        llm_config = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        llm_config.pad_token_id = None
        bagel_config = BagelConfig(
            visual_gen=True,
            visual_und=False,
            llm_config=llm_config,
            vit_config=None,
            vae_config=types.SimpleNamespace(downsample=8, z_channels=16),
            latent_patch_size=2,
            max_latent_size=64,
        )
        with _default_dtype(torch.bfloat16), torch.device("meta"):
            language_model = Qwen2ForCausalLM(llm_config)
            language_model.lm_head = None
            model = Bagel(language_model, None, bagel_config)

    model.to_empty(device=device)
    model.language_model.model.enable_taylorseer = False
    rotary = model.language_model.model.rotary_emb
    inv_freq, attention_scaling = _legacy_default_rope(llm_config, device)
    rotary.inv_freq = inv_freq
    rotary.original_inv_freq = inv_freq.clone()
    rotary.attention_scaling = attention_scaling
    loaded_count, parameter_elements = _copy_exact_parameters(model, ema_path, device)
    return model.eval(), NaiveCache, loaded_count, parameter_elements


def _stream_sglang_weights(
    model: BagelTransformer,
    ema_path: Path,
) -> set[str]:
    with safe_open(ema_path, framework="pt", device="cpu") as source:
        return model.load_weights(
            (
                (name, source.get_tensor(name))
                for name in source.keys()
                if model.accepts_checkpoint_weight(name)
            )
        )


def _load_sglang_model(
    ema_path: Path,
    device: torch.device,
) -> tuple[BagelTransformer, int, int]:
    with _default_dtype(torch.bfloat16), torch.device("meta"):
        model = BagelTransformer(
            BagelDiTConfig(),
            attention_backend=AttentionBackendEnum.TORCH_SDPA,
        )
    model.to_empty(device=device)
    arch = model.config.arch_config
    model.rotary_emb.inv_freq = 1.0 / (
        float(arch.rope_theta)
        ** (
            torch.arange(
                0,
                arch.attention_head_dim,
                2,
                dtype=torch.float32,
                device=device,
            )
            / arch.attention_head_dim
        )
    )
    loaded = _stream_sglang_weights(model, ema_path)
    parameter_elements = sum(parameter.numel() for parameter in model.parameters())
    return model.eval(), len(loaded), parameter_elements


def _cache_snapshot(cache: Any) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    if isinstance(cache.key_cache, Mapping):
        if not isinstance(cache.value_cache, Mapping):
            raise ValueError("BAGEL cache key/value containers must match")
        layers = sorted(set(cache.key_cache) | set(cache.value_cache))
        entries = (
            (cache.key_cache.get(layer), cache.value_cache.get(layer))
            for layer in layers
        )
    else:
        entries = zip(cache.key_cache, cache.value_cache)
    for key, value in entries:
        if (key is None) != (value is None):
            raise ValueError("BAGEL cache keys and values must be paired")
        if key is not None and value is not None:
            tensors.extend(
                (
                    key.detach().float().cpu().reshape(-1),
                    value.detach().float().cpu().reshape(-1),
                )
            )
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(tensors)


def _tensor_hash(tensor: torch.Tensor) -> str:
    data = tensor.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    if reference.shape != candidate.shape:
        raise ValueError(
            f"BAGEL output shape mismatch: {tuple(reference.shape)} != "
            f"{tuple(candidate.shape)}"
        )
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("BAGEL parity outputs must be finite")
    delta = candidate - reference
    cosine = torch.nn.functional.cosine_similarity(
        reference.reshape(1, -1), candidate.reshape(1, -1)
    ).item()
    return {
        "cosine_similarity": float(cosine),
        "mean_absolute_error": float(delta.abs().mean().item()),
        "max_absolute_error": float(delta.abs().max().item()),
    }


def _to_device(
    inputs: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in inputs.items()}


def _official_forward(
    model: nn.Module,
    generation: dict[str, torch.Tensor],
    prefix_cache: Any,
    layout: dict[str, torch.Tensor],
    timestep: torch.Tensor,
    *,
    cfg_text_scale: float = 1.0,
    cfg_cache: Any | None = None,
    cfg_layout: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {
        "x_t": generation["packed_init_noises"],
        "timestep": timestep,
        "packed_vae_token_indexes": generation["packed_vae_token_indexes"],
        "packed_vae_position_ids": generation["packed_vae_position_ids"],
        "packed_text_ids": generation["packed_text_ids"],
        "packed_text_indexes": generation["packed_text_indexes"],
        "packed_seqlens": generation["packed_seqlens"],
        "packed_indexes": layout["packed_indexes"],
        "packed_position_ids": layout["packed_position_ids"],
        "past_key_values": prefix_cache,
        "key_values_lens": layout["key_values_lens"],
        "packed_key_value_indexes": layout["packed_key_value_indexes"],
        "cfg_text_scale": cfg_text_scale,
        "cfg_img_scale": 1.0,
        "cfg_renorm_min": 0.0,
        "cfg_renorm_type": "global",
    }
    if cfg_text_scale > 1.0:
        if cfg_cache is None or cfg_layout is None:
            raise ValueError("guided BAGEL forward requires an unconditional prefix")
        kwargs.update(
            cfg_text_past_key_values=cfg_cache,
            cfg_text_packed_position_ids=cfg_layout["packed_position_ids"],
            cfg_text_packed_query_indexes=cfg_layout["packed_indexes"],
            cfg_text_key_values_lens=cfg_layout["key_values_lens"],
            cfg_text_packed_key_value_indexes=cfg_layout["packed_key_value_indexes"],
        )
    return model._forward_flow(**kwargs).detach().float().cpu()


def _run_official(
    model: nn.Module,
    cache_type: type[Any],
    device: torch.device,
    noise: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    prompt_ids = torch.tensor(PROMPT_TOKEN_IDS, dtype=torch.long, device=device)
    prompt_length = prompt_ids.numel()
    main_cache = cache_type(28)
    empty_cache = cache_type(28)
    prefill = {
        "packed_text_ids": prompt_ids,
        "packed_text_position_ids": torch.arange(prompt_length, device=device),
        "text_token_lens": torch.tensor(
            [prompt_length], dtype=torch.int32, device=device
        ),
        "packed_text_indexes": torch.arange(prompt_length, device=device),
        "packed_key_value_indexes": torch.empty(0, dtype=torch.long, device=device),
        "key_values_lens": torch.zeros(1, dtype=torch.int32, device=device),
    }
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        _math_sdpa(),
    ):
        main_cache = model.forward_cache_update_text(main_cache, **prefill)

    generation_cpu = model.prepare_vae_latent(
        curr_kvlens=[prompt_length],
        curr_rope=[prompt_length],
        image_sizes=[(IMAGE_SIZE, IMAGE_SIZE)],
        new_token_ids={
            "start_of_image": START_OF_IMAGE_TOKEN_ID,
            "end_of_image": END_OF_IMAGE_TOKEN_ID,
        },
    )
    cfg_cpu = model.prepare_vae_latent_cfg(
        curr_kvlens=[0],
        curr_rope=[0],
        image_sizes=[(IMAGE_SIZE, IMAGE_SIZE)],
    )
    generation_cpu["packed_init_noises"] = noise.clone()

    expected_positions = torch.tensor(
        [
            row * 64 + column
            for row in range(LATENT_SIDE)
            for column in range(LATENT_SIDE)
        ]
    )
    torch.testing.assert_close(
        generation_cpu["packed_vae_position_ids"], expected_positions
    )
    torch.testing.assert_close(
        generation_cpu["packed_text_indexes"], torch.tensor([0, QUERY_TOKEN_COUNT - 1])
    )
    torch.testing.assert_close(
        generation_cpu["packed_vae_token_indexes"],
        torch.arange(1, QUERY_TOKEN_COUNT - 1),
    )
    torch.testing.assert_close(
        generation_cpu["packed_indexes"],
        torch.arange(prompt_length, prompt_length + QUERY_TOKEN_COUNT),
    )
    torch.testing.assert_close(
        generation_cpu["packed_key_value_indexes"], torch.arange(prompt_length)
    )
    torch.testing.assert_close(
        cfg_cpu["cfg_packed_query_indexes"], torch.arange(QUERY_TOKEN_COUNT)
    )
    assert cfg_cpu["cfg_packed_key_value_indexes"].numel() == 0

    generation = _to_device(generation_cpu, device)
    main_layout = {
        "packed_indexes": generation["packed_indexes"],
        "packed_position_ids": generation["packed_position_ids"],
        "key_values_lens": generation["key_values_lens"],
        "packed_key_value_indexes": generation["packed_key_value_indexes"],
    }
    cfg = _to_device(cfg_cpu, device)
    cfg_layout = {
        "packed_indexes": cfg["cfg_packed_query_indexes"],
        "packed_position_ids": cfg["cfg_packed_position_ids"],
        "key_values_lens": cfg["cfg_key_values_lens"],
        "packed_key_value_indexes": cfg["cfg_packed_key_value_indexes"],
    }
    prefix_before = _cache_snapshot(main_cache)
    assert prefix_before.numel() == 28 * 2 * prompt_length * 4 * 128
    timestep = torch.full(
        (LATENT_TOKEN_COUNT,), 1.0, dtype=torch.float32, device=device
    )
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        _math_sdpa(),
    ):
        conditional = _official_forward(
            model, generation, main_cache, main_layout, timestep
        )
        unconditional = _official_forward(
            model, generation, empty_cache, cfg_layout, timestep
        )
        guided = _official_forward(
            model,
            generation,
            main_cache,
            main_layout,
            timestep,
            cfg_text_scale=GUIDANCE_SCALE,
            cfg_cache=empty_cache,
            cfg_layout=cfg_layout,
        )
    prefix_after = _cache_snapshot(main_cache)
    assert _tensor_hash(prefix_before) == _tensor_hash(prefix_after)
    return (
        {
            "conditional": conditional,
            "unconditional": unconditional,
            "guided": guided,
        },
        {
            "prefix": prefix_before,
            "prefix_length": prompt_length,
            "rope_offset": prompt_length,
            "position_ids": generation_cpu["packed_vae_position_ids"].clone(),
        },
    )


def _run_sglang(
    model: BagelTransformer,
    device: torch.device,
    noise: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    prompt_ids = torch.tensor(PROMPT_TOKEN_IDS, dtype=torch.long, device=device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        _math_sdpa(),
    ):
        context = model.build_context(
            prompt_ids,
            None,
            height=IMAGE_SIZE,
            width=IMAGE_SIZE,
            start_of_image_token_id=START_OF_IMAGE_TOKEN_ID,
            end_of_image_token_id=END_OF_IMAGE_TOKEN_ID,
        )

    assert context.conditional_kv_lens.tolist() == [len(PROMPT_TOKEN_IDS)]
    assert context.unconditional_kv_lens.tolist() == [0]
    assert context.conditional_rope_offset == len(PROMPT_TOKEN_IDS)
    assert context.unconditional_rope_offset == 0
    expected_positions = torch.tensor(
        [
            row * 64 + column
            for row in range(LATENT_SIDE)
            for column in range(LATENT_SIDE)
        ],
        device=device,
    )
    torch.testing.assert_close(
        model._latent_position_ids(IMAGE_SIZE, IMAGE_SIZE, device), expected_positions
    )
    prefix_before = _cache_snapshot(context.conditional_kv)
    assert prefix_before.numel() == 28 * 2 * len(PROMPT_TOKEN_IDS) * 4 * 128

    latents = noise.to(device)
    timestep = torch.full(
        (LATENT_TOKEN_COUNT,), 1.0, dtype=torch.float32, device=device
    )
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        _math_sdpa(),
    ):
        conditional = model._generation_step_single(
            latents,
            timestep,
            context.conditional_kv,
            context.conditional_kv_lens,
            context.conditional_rope_offset,
            context,
            None,
        ).detach()
        unconditional = model._generation_step_single(
            latents,
            timestep,
            context.unconditional_kv,
            context.unconditional_kv_lens,
            context.unconditional_rope_offset,
            context,
            None,
        ).detach()
        guided = model(
            latents,
            torch.tensor([1.0], dtype=torch.float32, device=device),
            bagel_context=context,
            guidance_scale=GUIDANCE_SCALE,
            cfg_interval=(0.4, 1.0),
            cfg_renorm_min=0.0,
            cfg_renorm_type="global",
        ).detach()
        expected_guided = model._apply_cfg(
            conditional,
            unconditional,
            GUIDANCE_SCALE,
            renorm_min=0.0,
            renorm_type="global",
        )
    assert conditional.dtype == torch.bfloat16
    assert unconditional.dtype == torch.bfloat16
    assert guided.dtype == torch.bfloat16
    torch.testing.assert_close(guided, expected_guided, rtol=0.0, atol=0.0)
    prefix_after = _cache_snapshot(context.conditional_kv)
    assert _tensor_hash(prefix_before) == _tensor_hash(prefix_after)
    return (
        {
            "conditional": conditional.float().cpu(),
            "unconditional": unconditional.float().cpu(),
            "guided": guided.float().cpu(),
        },
        {
            "prefix": prefix_before,
            "prefix_length": int(context.conditional_kv_lens.item()),
            "rope_offset": int(context.conditional_rope_offset),
            "position_ids": expected_positions.cpu(),
        },
    )


def _release_cuda_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def test_bagel_transformer_single_step_matches_official() -> None:
    """Compare pinned official and SGLang one-step Transformer outputs.

    The comparison uses one fixed 256x256 latent so the non-contiguous 64-wide
    position-table indexing, request-owned prefix caches, conditional branch,
    empty-prefix branch, and global CFG combine are all exercised.
    """
    official_repo, model_path = _required_inputs()
    if not torch.cuda.is_available():
        pytest.skip("BAGEL Transformer parity requires CUDA")
    if torch.cuda.device_count() != 1:
        pytest.fail("BAGEL Transformer parity requires one visible CUDA GPU")
    if "h100" not in torch.cuda.get_device_name(0).lower():
        pytest.skip("BAGEL Transformer parity is pinned to H100")
    ema_path = _verify_pins(official_repo, model_path)
    device = torch.device("cuda:0")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    noise = torch.randn(
        (LATENT_TOKEN_COUNT, PATCH_WIDTH),
        generator=generator,
        dtype=torch.float32,
    )

    torch.cuda.reset_peak_memory_stats(device)
    official_model, cache_type, official_loaded, official_elements = (
        _load_official_model(official_repo, model_path, ema_path, device)
    )
    assert official_loaded == TRANSFORMER_PARAMETER_COUNT
    assert official_elements == TRANSFORMER_PARAMETER_ELEMENTS
    official_outputs, official_contract = _run_official(
        official_model, cache_type, device, noise
    )
    official_peak = torch.cuda.max_memory_allocated(device)
    del official_model
    _release_cuda_memory()

    torch.cuda.reset_peak_memory_stats(device)
    sglang_model, sglang_loaded, sglang_elements = _load_sglang_model(ema_path, device)
    assert sglang_loaded == TRANSFORMER_PARAMETER_COUNT
    assert sglang_elements == TRANSFORMER_PARAMETER_ELEMENTS
    sglang_outputs, sglang_contract = _run_sglang(sglang_model, device, noise)
    sglang_peak = torch.cuda.max_memory_allocated(device)
    del sglang_model
    _release_cuda_memory()

    assert official_contract["prefix_length"] == sglang_contract["prefix_length"]
    assert official_contract["rope_offset"] == sglang_contract["rope_offset"]
    torch.testing.assert_close(
        official_contract["position_ids"], sglang_contract["position_ids"]
    )
    prefix_metrics = _metrics(official_contract["prefix"], sglang_contract["prefix"])
    assert prefix_metrics["cosine_similarity"] >= COSINE_THRESHOLD, (
        "BAGEL prefix-cache cosine "
        f"{prefix_metrics['cosine_similarity']:.8f} is below {COSINE_THRESHOLD}"
    )
    branch_metrics = {
        name: _metrics(official_outputs[name], sglang_outputs[name])
        for name in ("conditional", "unconditional", "guided")
    }
    for name, metrics in branch_metrics.items():
        assert metrics["cosine_similarity"] >= COSINE_THRESHOLD, (
            f"BAGEL {name} cosine {metrics['cosine_similarity']:.8f} is below "
            f"{COSINE_THRESHOLD}"
        )

    print(
        json.dumps(
            {
                "official_commit": OFFICIAL_REPO_COMMIT,
                "model_revision": MODEL_REVISION,
                "prompt": PROMPT,
                "prompt_token_ids": list(PROMPT_TOKEN_IDS),
                "seed": SEED,
                "image_size": [IMAGE_SIZE, IMAGE_SIZE],
                "noise_sha256_float32": _tensor_hash(noise),
                "threshold": {"cosine_similarity_min": COSINE_THRESHOLD},
                "prefix_metrics": prefix_metrics,
                "branch_metrics": branch_metrics,
                "loaded_parameters": {
                    "official": official_loaded,
                    "sglang": sglang_loaded,
                },
                "peak_cuda_memory_bytes": {
                    "official": official_peak,
                    "sglang": sglang_peak,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
