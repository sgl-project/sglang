#!/usr/bin/env python
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Parity test: SGLang π0.5 vs LeRobot π0.5.

This test verifies that SGLang's ``Pi05ForActionPrediction`` produces
matching action chunks with LeRobot's ``PI05Policy`` when both are fed:

- the same weights
- the same LeRobot-preprocessed inputs (images, masks, language tokens)
- the same fixed initial noise tensor

Unlike π0, π0.5 serializes state into the prompt during preprocessing, so
the parity oracle compares both implementations *after* LeRobot preprocessing.
That keeps the comparison focused on model behavior rather than request/server
plumbing.

Usage (local, ~13 GB download the first time)::

    # From repo root
    PYTHONPATH=lerobot/src:sglang/python \
        python -m pytest sglang/test/srt/models/test_pi05_parity.py -v -s

    # Or just run directly
    PYTHONPATH=lerobot/src:sglang/python \
        python sglang/test/srt/models/test_pi05_parity.py

Environment variables:
  PI05_PARITY_DEVICE            cpu | cuda  (default: cuda if available else cpu)
  PI05_PARITY_DTYPE             float32 | bfloat16 | float16 (default: float32)
  PI05_PARITY_ATOL              absolute tolerance (default: 1e-5)
  PI05_PARITY_FROM_PRETRAINED   "1" to use real weights, "0" for random (default: 1)
  PI05_PARITY_BATCH_SIZE        default: 1
  PI05_PARITY_NUM_STEPS         default: 10
  PI05_PARITY_MODEL_PATH        Local path (or HF repo id) of the pi05_base checkpoint.
                                If set and it's an existing directory, we load from
                                there and skip the HuggingFace download. Defaults to
                                "lerobot/pi05_base" (downloads from HF).
"""

from __future__ import annotations

import copy
import os

import pytest
import torch

pytest.importorskip("lerobot")
pytest.importorskip("transformers")

DEVICE = os.environ.get(
    "PI05_PARITY_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)
DTYPE_STR = os.environ.get("PI05_PARITY_DTYPE", "float32")
ATOL = float(os.environ.get("PI05_PARITY_ATOL", "1e-5"))
FROM_PRETRAINED = os.environ.get("PI05_PARITY_FROM_PRETRAINED", "1") == "1"
BATCH_SIZE = int(os.environ.get("PI05_PARITY_BATCH_SIZE", "1"))
NUM_STEPS = int(os.environ.get("PI05_PARITY_NUM_STEPS", "10"))
MODEL_PATH = os.environ.get("PI05_PARITY_MODEL_PATH", "lerobot/pi05_base")

_DTYPE = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[DTYPE_STR]

ACTION_DIM = 32
STATE_DIM = 32
ACTION_HORIZON = 50
MAX_TOKEN_LEN = 200


def _resolve_checkpoint_dir() -> str:
    """Return a local directory containing the pi05_base checkpoint.

    If ``MODEL_PATH`` is an existing directory, use it as-is (no network).
    Otherwise treat it as an HF repo id and ``snapshot_download`` it.
    """
    if os.path.isdir(MODEL_PATH):
        return MODEL_PATH
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=MODEL_PATH, repo_type="model")


def _dummy_dataset_stats() -> dict:
    image_stats = {
        "mean": torch.zeros(3, 224, 224),
        "std": torch.ones(3, 224, 224),
        "q01": torch.zeros(3, 224, 224),
        "q99": torch.ones(3, 224, 224),
    }
    return {
        "observation.state": {
            "mean": torch.zeros(STATE_DIM),
            "std": torch.ones(STATE_DIM),
            "q01": -torch.ones(STATE_DIM),
            "q99": torch.ones(STATE_DIM),
        },
        "action": {
            "mean": torch.zeros(ACTION_DIM),
            "std": torch.ones(ACTION_DIM),
            "q01": -torch.ones(ACTION_DIM),
            "q99": torch.ones(ACTION_DIM),
        },
        "images": {
            "base_0_rgb": image_stats,
            "left_wrist_0_rgb": image_stats,
            "right_wrist_0_rgb": image_stats,
        },
    }


def _create_dummy_batch(batch_size: int = BATCH_SIZE, device: str = DEVICE) -> dict:
    """Reproducible dummy inputs."""
    g = torch.Generator(device="cpu").manual_seed(0)
    prompt = "Pick up the red block and place it in the bin"
    return {
        "observation.state": torch.randn(
            batch_size, STATE_DIM, generator=g, dtype=torch.float32
        ).to(device),
        "action": torch.randn(
            batch_size, ACTION_HORIZON, ACTION_DIM, generator=g, dtype=torch.float32
        ).to(device),
        "observation.images.base_0_rgb": torch.rand(
            batch_size, 3, 224, 224, generator=g, dtype=torch.float32
        ).to(device),
        "observation.images.left_wrist_0_rgb": torch.rand(
            batch_size, 3, 224, 224, generator=g, dtype=torch.float32
        ).to(device),
        "observation.images.right_wrist_0_rgb": torch.rand(
            batch_size, 3, 224, 224, generator=g, dtype=torch.float32
        ).to(device),
        "task": [prompt for _ in range(batch_size)],
    }


def _make_fixed_noise(batch_size: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(1234)
    return torch.randn(
        batch_size,
        ACTION_HORIZON,
        ACTION_DIM,
        generator=g,
        dtype=torch.float32,
    ).to(device)


def _instantiate_lerobot():
    from lerobot.policies.pi05 import PI05Config, PI05Policy, make_pi05_pre_post_processors

    if FROM_PRETRAINED:
        policy = PI05Policy.from_pretrained(MODEL_PATH, strict=True)
    else:
        config = PI05Config(
            max_action_dim=ACTION_DIM,
            max_state_dim=STATE_DIM,
            chunk_size=ACTION_HORIZON,
            num_inference_steps=NUM_STEPS,
            dtype=DTYPE_STR,
            tokenizer_max_length=MAX_TOKEN_LEN,
        )
        policy = PI05Policy(config)

    policy.to(DEVICE)
    policy.config.device = DEVICE
    policy.eval()

    pre, post = make_pi05_pre_post_processors(
        config=policy.config,
        dataset_stats=_dummy_dataset_stats(),
    )
    return policy, pre, post


def _instantiate_sglang():
    from sglang.srt.configs.pi05 import Pi05Config
    from sglang.srt.models.pi05 import Pi05ForActionPrediction

    cfg = Pi05Config(
        max_action_dim=ACTION_DIM,
        max_state_dim=STATE_DIM,
        chunk_size=ACTION_HORIZON,
        num_inference_steps=NUM_STEPS,
        tokenizer_max_length=MAX_TOKEN_LEN,
        dtype=DTYPE_STR,
    )
    model = Pi05ForActionPrediction(cfg)
    model.to(DEVICE).eval()

    if FROM_PRETRAINED:
        _load_lerobot_weights_into_sglang(model)
    return model


def _load_lerobot_weights_into_sglang(sglang_model):
    import safetensors.torch

    cache_dir = _resolve_checkpoint_dir()
    path = os.path.join(cache_dir, "model.safetensors")
    state = safetensors.torch.load_file(path)
    sglang_model.load_weights(list(state.items()))


def _extract_lerobot_model_inputs(lerobot_policy, processed_batch):
    """Mimic what ``PI05Policy.predict_action_chunk`` feeds into ``model.sample_actions``."""
    images, img_masks = lerobot_policy._preprocess_images(processed_batch)

    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    lang_tokens = processed_batch[OBS_LANGUAGE_TOKENS]
    lang_masks = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
    return images, img_masks, lang_tokens, lang_masks


def _diagnose_divergence(
    lerobot_model,
    sglang_model,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    noise,
):
    with torch.no_grad():
        lerobot_actions = lerobot_model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            noise=noise,
            num_steps=NUM_STEPS,
        )
        sglang_actions = sglang_model.sample_actions(
            images=images,
            image_masks=img_masks,
            tokens=lang_tokens,
            masks=lang_masks,
            noise=noise,
            num_steps=NUM_STEPS,
        )

    diff = (lerobot_actions.float() - sglang_actions.float()).abs()
    print("[parity-pi05] diagnose:")
    print(f"  lerobot.shape={tuple(lerobot_actions.shape)}")
    print(f"  sglang.shape={tuple(sglang_actions.shape)}")
    print(f"  max_diff={diff.max().item():.6e}")
    print(f"  mean_diff={diff.mean().item():.6e}")


@pytest.mark.skipif(
    DEVICE.startswith("cuda") and not torch.cuda.is_available(),
    reason="CUDA device requested but not available",
)
def test_pi05_sglang_vs_lerobot():
    print("[parity-pi05] Instantiating LeRobot…")
    lerobot_policy, lerobot_pre, _ = _instantiate_lerobot()

    print("[parity-pi05] Instantiating SGLang…")
    sglang_model = _instantiate_sglang()

    print("[parity-pi05] Preparing shared inputs…")
    raw_batch = _create_dummy_batch()
    processed_batch = lerobot_pre(copy.deepcopy(raw_batch))
    images, img_masks, lang_tokens, lang_masks = _extract_lerobot_model_inputs(
        lerobot_policy, processed_batch
    )
    noise = _make_fixed_noise(raw_batch["observation.state"].shape[0], DEVICE)

    print(f"[parity-pi05] lang_tokens.shape={tuple(lang_tokens.shape)}")
    print(f"[parity-pi05] images[0].shape={tuple(images[0].shape)} (num_cams={len(images)})")
    print(f"[parity-pi05] noise.shape={tuple(noise.shape)} dtype={noise.dtype}")

    print("[parity-pi05] Running LeRobot sample_actions…")
    with torch.no_grad():
        lerobot_actions = lerobot_policy.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            noise=noise,
            num_steps=NUM_STEPS,
        )
    print(
        f"[parity-pi05] LeRobot actions: shape={tuple(lerobot_actions.shape)} "
        f"mean={lerobot_actions.float().mean().item():.6f} "
        f"std={lerobot_actions.float().std().item():.6f}"
    )

    print("[parity-pi05] Running SGLang sample_actions…")
    with torch.no_grad():
        sglang_actions = sglang_model.sample_actions(
            images=images,
            image_masks=img_masks,
            tokens=lang_tokens,
            masks=lang_masks,
            noise=noise,
            num_steps=NUM_STEPS,
        )
    print(
        f"[parity-pi05] SGLang  actions: shape={tuple(sglang_actions.shape)} "
        f"mean={sglang_actions.float().mean().item():.6f} "
        f"std={sglang_actions.float().std().item():.6f}"
    )

    diff = (lerobot_actions.float() - sglang_actions.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[parity-pi05] |Δ| max={max_diff:.2e}  mean={mean_diff:.2e}  atol={ATOL:.1e}")

    close = torch.allclose(
        lerobot_actions.float(),
        sglang_actions.float(),
        atol=ATOL,
    )
    print(f"[parity-pi05] torch.allclose(atol={ATOL}): {close}")

    if not close:
        print("\n[parity-pi05] Outputs diverge — running diagnostics…")
        _diagnose_divergence(
            lerobot_policy.model,
            sglang_model,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            noise,
        )

    assert close, (
        f"SGLang vs LeRobot pi05 actions differ beyond atol={ATOL}. "
        f"max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}"
    )
