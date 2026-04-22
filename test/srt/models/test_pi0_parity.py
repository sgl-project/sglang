#!/usr/bin/env python
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Parity test: SGLang π0 vs LeRobot π0 (and/or OpenPI π0).

Port of::
    lerobot/tests/policies/pi0_pi05/test_pi0_original_vs_lerobot.py

This test verifies that SGLang's :class:`sglang.srt.models.pi0.Pi0ForActionPrediction`
produces bit-for-bit matching action chunks with LeRobot's ``PI0Policy`` (and,
optionally, with OpenPI's reference ``PI0Pytorch``) when fed:
  * the same weights (from ``lerobot/pi0_base``)
  * the same pre-processed inputs (images, masks, tokens, state)
  * the same initial noise tensor

Because π0 is a flow-matching model (Euler-integrated ODE from t=1 → t=0 with
a fixed ``num_steps``), the output becomes deterministic once the noise is
fixed, so ``torch.allclose`` on the final action chunk is a valid oracle.

Usage (local, ~13 GB download the first time)::

    # From repo root
    PYTHONPATH=openpi/src:lerobot/src:sglang/python \
        python -m pytest sglang/test/srt/models/test_pi0_parity.py -v -s

    # Or just run directly
    PYTHONPATH=openpi/src:lerobot/src:sglang/python \
        python sglang/test/srt/models/test_pi0_parity.py

Environment variables:
  PI0_PARITY_DEVICE       cpu | cuda  (default: cpu)
  PI0_PARITY_DTYPE        float32 | bfloat16  (default: float32)
  PI0_PARITY_ATOL         absolute tolerance (default: 1e-4)
  PI0_PARITY_FROM_PRETRAINED  "1" to use real weights, "0" for random (default: 1)
  PI0_PARITY_BATCH_SIZE   default: 2
  PI0_PARITY_NUM_STEPS    number of flow-matching steps (default: 10)
  PI0_PARITY_CHECK_OPENPI "1" to also compare vs OpenPI (default: 0)
  PI0_PARITY_MODEL_PATH   Local path (or HF repo id) of the pi0_base checkpoint.
                          If set and it's an existing directory, we load from
                          there and skip the HuggingFace download. Defaults to
                          "lerobot/pi0_base" (downloads from HF).

This test is skipped by default in CI because it needs LeRobot + a model
download.  It prints rich diagnostics on mismatch: per-stage intermediate
comparisons (prefix embeddings, prefix KV cache, first denoise v_t).
"""

from __future__ import annotations

import os
import sys
import copy
from typing import Any, List, Tuple

import pytest
import torch

# ─── Skip rules ───────────────────────────────────────────────────────
pytest.importorskip("lerobot")
pytest.importorskip("transformers")

pytestmark = pytest.mark.skipif(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Parity test requires real weights + LeRobot; not meant for CI",
)


# ─── Config ───────────────────────────────────────────────────────────
DEVICE = os.environ.get("PI0_PARITY_DEVICE", "cpu")
DTYPE_STR = os.environ.get("PI0_PARITY_DTYPE", "float32")
ATOL = float(os.environ.get("PI0_PARITY_ATOL", "1e-4"))
FROM_PRETRAINED = os.environ.get("PI0_PARITY_FROM_PRETRAINED", "1") == "1"
BATCH_SIZE = int(os.environ.get("PI0_PARITY_BATCH_SIZE", "2"))
NUM_STEPS = int(os.environ.get("PI0_PARITY_NUM_STEPS", "10"))
CHECK_OPENPI = os.environ.get("PI0_PARITY_CHECK_OPENPI", "0") == "1"
# Either a local directory containing ``model.safetensors`` (+ config etc.) or
# an HF repo id. Defaults to the public lerobot/pi0_base repo (~13 GB download).
MODEL_PATH = os.environ.get("PI0_PARITY_MODEL_PATH", "lerobot/pi0_base")

_DTYPE = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[DTYPE_STR]


def _resolve_checkpoint_dir() -> str:
    """Return a local directory containing the pi0_base checkpoint.

    If ``MODEL_PATH`` is an existing directory, use it as-is (no network).
    Otherwise treat it as an HF repo id and ``snapshot_download`` it.
    """
    if os.path.isdir(MODEL_PATH):
        return MODEL_PATH
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=MODEL_PATH, repo_type="model")

# Must match LeRobot defaults for `lerobot/pi0_base`
ACTION_DIM = 32
STATE_DIM = 32
ACTION_HORIZON = 50
MAX_TOKEN_LEN = 48


# ─── Dummy dataset stats (identity) ──────────────────────────────────
def _dummy_dataset_stats() -> dict:
    return {
        "observation.state": {
            "mean": torch.zeros(STATE_DIM),
            "std": torch.ones(STATE_DIM),
            "q01": torch.zeros(STATE_DIM),
            "q99": torch.ones(STATE_DIM),
        },
        "action": {
            "mean": torch.zeros(ACTION_DIM),
            "std": torch.ones(ACTION_DIM),
            "q01": torch.zeros(ACTION_DIM),
            "q99": torch.ones(ACTION_DIM),
        },
        "images": {
            cam: {
                "mean": torch.zeros(3, 224, 224),
                "std": torch.ones(3, 224, 224),
                "q01": torch.zeros(3, 224, 224),
                "q99": torch.ones(3, 224, 224),
            }
            for cam in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
    }


def _create_dummy_batch(batch_size: int = BATCH_SIZE, device: str = DEVICE) -> dict:
    """Reproducible dummy inputs — identical across all three implementations."""
    g = torch.Generator(device="cpu").manual_seed(0)
    prompt = "Pick up the red block and place it in the bin"
    batch = {
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
    return batch


# ─── LeRobot instantiation ────────────────────────────────────────────
def _instantiate_lerobot():
    from lerobot.policies.pi0 import PI0Config, PI0Policy  # noqa: E402
    from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors  # noqa: E402

    if FROM_PRETRAINED:
        # ``from_pretrained`` accepts both a local directory and an HF repo id,
        # so we just forward whatever the user set (default: lerobot/pi0_base).
        policy = PI0Policy.from_pretrained(MODEL_PATH, strict=True)
    else:
        config = PI0Config(
            max_action_dim=ACTION_DIM, max_state_dim=STATE_DIM, dtype=DTYPE_STR
        )
        policy = PI0Policy(config)
    policy.to(DEVICE)
    policy.config.device = DEVICE
    policy.eval()

    pre, post = make_pi0_pre_post_processors(
        config=policy.config, dataset_stats=_dummy_dataset_stats()
    )
    return policy, pre, post


# ─── SGLang instantiation ─────────────────────────────────────────────
def _instantiate_sglang():
    """Build the SGLang π0 model in isolation (no server, no scheduler)."""
    from sglang.srt.configs.pi0 import Pi0Config
    from sglang.srt.models.pi0 import Pi0ForActionPrediction

    cfg = Pi0Config(
        max_action_dim=ACTION_DIM,
        max_state_dim=STATE_DIM,
        chunk_size=ACTION_HORIZON,
        num_inference_steps=NUM_STEPS,
        dtype=DTYPE_STR,
    )
    model = Pi0ForActionPrediction(cfg)
    model.to(DEVICE).eval()

    if FROM_PRETRAINED:
        _load_lerobot_weights_into_sglang(model)
    return model


def _load_lerobot_weights_into_sglang(sglang_model):
    """Feed the ``lerobot/pi0_base`` safetensors into SGLang.

    Respects ``PI0_PARITY_MODEL_PATH``: if set to a local dir we load from disk;
    otherwise we ``snapshot_download`` it (default: ``lerobot/pi0_base``).
    SGLang's ``Pi0ForActionPrediction.load_weights`` handles the leading
    ``model.`` prefix and other key remaps (including the
    ``paligemma.lm_head.weight`` → ``embed_tokens.weight`` rewrite), so we
    can pass the raw checkpoint dict.
    """
    import safetensors.torch

    cache_dir = _resolve_checkpoint_dir()
    path = os.path.join(cache_dir, "model.safetensors")
    state = safetensors.torch.load_file(path)
    sglang_model.load_weights(list(state.items()))


# ─── OpenPI instantiation (optional) ──────────────────────────────────
def _instantiate_openpi():
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    class _Cfg:
        action_dim = ACTION_DIM
        action_horizon = ACTION_HORIZON
        paligemma_variant = "gemma_2b"
        action_expert_variant = "gemma_300m"
        precision = "float32"
        pi05 = False
        dtype = "float32"

    model = PI0Pytorch(_Cfg())
    if FROM_PRETRAINED:
        import safetensors.torch

        cache_dir = _resolve_checkpoint_dir()
        state = safetensors.torch.load_file(os.path.join(cache_dir, "model.safetensors"))
        model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


# ─── Helpers to extract LeRobot's pre-processed inputs ────────────────
def _extract_lerobot_model_inputs(lerobot_policy, processed_batch):
    """Mimic what ``PI0Policy.predict_action_chunk`` feeds into
    ``self.model.sample_actions``.  We use these *exact* tensors for SGLang
    so that any divergence must come from the core model, not preprocessing.
    """
    images, img_masks = lerobot_policy._preprocess_images(processed_batch)
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )
    lang_tokens = processed_batch[OBS_LANGUAGE_TOKENS]
    lang_masks = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
    state = lerobot_policy.prepare_state(processed_batch)
    return images, img_masks, lang_tokens, lang_masks, state


# ─── Shared fixed-noise sampler ───────────────────────────────────────
def _make_fixed_noise(batch_size: int, device: str) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(42)
    return torch.randn(
        batch_size, ACTION_HORIZON, ACTION_DIM, generator=g, dtype=torch.float32
    ).to(device)


# ─── Main test ────────────────────────────────────────────────────────
def test_pi0_sglang_vs_lerobot():
    print("\n[parity] Instantiating LeRobot…")
    lerobot_policy, lerobot_pre, _ = _instantiate_lerobot()

    print("[parity] Instantiating SGLang…")
    sglang_model = _instantiate_sglang()

    print("[parity] Preparing shared inputs…")
    raw_batch = _create_dummy_batch()
    processed_batch = lerobot_pre(copy.deepcopy(raw_batch))
    images, img_masks, lang_tokens, lang_masks, state = _extract_lerobot_model_inputs(
        lerobot_policy, processed_batch
    )
    noise = _make_fixed_noise(raw_batch["observation.state"].shape[0], DEVICE)

    print(f"[parity] state.shape={state.shape}  lang_tokens.shape={lang_tokens.shape}")
    print(f"[parity] images[0].shape={images[0].shape} (num_cams={len(images)})")
    print(f"[parity] noise.shape={noise.shape}  dtype={noise.dtype}")

    # ── LeRobot forward ──
    print("[parity] Running LeRobot sample_actions…")
    with torch.no_grad():
        lerobot_actions = lerobot_policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=noise, num_steps=NUM_STEPS,
        )
    print(
        f"[parity] LeRobot actions: shape={lerobot_actions.shape} "
        f"mean={lerobot_actions.mean().item():.6f} std={lerobot_actions.std().item():.6f}"
    )

    # ── SGLang forward ──
    print("[parity] Running SGLang sample_actions…")
    with torch.no_grad():
        sglang_actions = sglang_model.sample_actions(
            images=images,
            image_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            state=state,
            noise=noise,
            num_steps=NUM_STEPS,
        )
    print(
        f"[parity] SGLang  actions: shape={sglang_actions.shape} "
        f"mean={sglang_actions.mean().item():.6f} std={sglang_actions.std().item():.6f}"
    )

    # ── Compare ──
    diff = (lerobot_actions.float() - sglang_actions.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"[parity] |Δ| max={max_diff:.2e}  mean={mean_diff:.2e}  atol={ATOL:.1e}")
    close = torch.allclose(lerobot_actions.float(), sglang_actions.float(), atol=ATOL)
    print(f"[parity] torch.allclose(atol={ATOL}): {close}")

    # On mismatch run intermediate diagnostics before failing
    if not close:
        print("\n[parity] ⚠️  Outputs diverge — running per-stage diagnostics…")
        _diagnose_divergence(
            lerobot_policy.model, sglang_model,
            images, img_masks, lang_tokens, lang_masks, state, noise,
        )

    assert close, (
        f"SGLang vs LeRobot actions differ beyond atol={ATOL}. "
        f"max_diff={max_diff:.2e}  mean_diff={mean_diff:.2e}"
    )


@pytest.mark.skipif(not CHECK_OPENPI, reason="Set PI0_PARITY_CHECK_OPENPI=1 to enable")
def test_pi0_sglang_vs_openpi():
    pytest.importorskip("openpi")
    from openpi.models_pytorch import preprocessing_pytorch as _pre  # noqa: F401

    print("\n[parity-openpi] Instantiating LeRobot (for preprocessing)…")
    lerobot_policy, lerobot_pre, _ = _instantiate_lerobot()

    print("[parity-openpi] Instantiating SGLang…")
    sglang_model = _instantiate_sglang()

    print("[parity-openpi] Instantiating OpenPI…")
    openpi_model = _instantiate_openpi()

    raw_batch = _create_dummy_batch()
    processed_batch = lerobot_pre(copy.deepcopy(raw_batch))
    images, img_masks, lang_tokens, lang_masks, state = _extract_lerobot_model_inputs(
        lerobot_policy, processed_batch
    )
    noise = _make_fixed_noise(raw_batch["observation.state"].shape[0], DEVICE)

    # Build OpenPI observation object from LeRobot's pre-processed tensors
    class _Obs:
        pass
    obs = _Obs()
    obs.state = state
    obs.images = {
        "base_0_rgb": images[0],
        "left_wrist_0_rgb": images[1],
        "right_wrist_0_rgb": images[2],
    }
    obs.image_masks = {
        "base_0_rgb": img_masks[0],
        "left_wrist_0_rgb": img_masks[1],
        "right_wrist_0_rgb": img_masks[2],
    }
    obs.tokenized_prompt = lang_tokens
    obs.tokenized_prompt_mask = lang_masks
    obs.token_ar_mask = torch.zeros_like(lang_tokens, dtype=torch.int32)
    obs.token_loss_mask = torch.ones_like(lang_masks, dtype=torch.bool)

    with torch.no_grad():
        openpi_actions = openpi_model.sample_actions(
            device=DEVICE, observation=obs, noise=noise, num_steps=NUM_STEPS,
        )
        sglang_actions = sglang_model.sample_actions(
            images=images, image_masks=img_masks,
            lang_tokens=lang_tokens, lang_masks=lang_masks,
            state=state, noise=noise, num_steps=NUM_STEPS,
        )
    max_diff = (openpi_actions.float() - sglang_actions.float()).abs().max().item()
    print(f"[parity-openpi] max |Δ| = {max_diff:.2e}")
    assert max_diff < ATOL, f"SGLang vs OpenPI differ: max |Δ|={max_diff:.2e}"


# ─── Diagnostic helpers ───────────────────────────────────────────────
@torch.no_grad()
def _diagnose_divergence(
    lerobot_flow_model, sglang_model,
    images, img_masks, lang_tokens, lang_masks, state, noise,
):
    """Localize a numerical mismatch to a specific pipeline stage.

    Runs three comparisons, each narrowing the scope:
      1. Prefix embeddings — isolates the input-side (SigLIP / embed_tokens /
         projector). The image and language slices are diffed separately so
         we can tell at a glance which modality is off.
      2. Prefix KV cache (layer 0) — isolates PaliGemma language-model
         attention.
      3. A single ``denoise_step`` velocity at t=1.0 — isolates the action
         expert forward.

    Only called from ``test_pi0_sglang_vs_lerobot`` after the action chunks
    fail ``torch.allclose``; prints rich context then lets the assert fail.
    """
    from sglang.srt.models.pi0 import make_att_2d_masks, prepare_attention_masks_4d

    # ── Stage 1: prefix embeddings ──
    lr_prefix_embs, lr_prefix_pad, lr_prefix_att = lerobot_flow_model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    sg_prefix_embs, sg_prefix_pad, sg_prefix_att = sglang_model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks
    )
    total_diff = (lr_prefix_embs.float() - sg_prefix_embs.float()).abs().max().item()
    print(
        f"[diag] prefix_embs max |Δ| = {total_diff:.2e}   "
        f"(shape={tuple(sg_prefix_embs.shape)})"
    )
    print(f"[diag] prefix_pad_masks equal: {torch.equal(lr_prefix_pad, sg_prefix_pad)}")
    print(f"[diag] prefix_att_masks equal: {torch.equal(lr_prefix_att.bool(), sg_prefix_att.bool())}")

    # Image vs language slice breakdown. Prefix layout is
    #     [image_tokens × 256] × num_cameras + [lang_tokens]
    num_cams = len(images)
    img_len = 256 * num_cams
    lang_len = lr_prefix_embs.shape[1] - img_len
    img_diff = (
        lr_prefix_embs[:, :img_len].float() - sg_prefix_embs[:, :img_len].float()
    ).abs().max().item()
    lang_diff = (
        lr_prefix_embs[:, img_len:].float() - sg_prefix_embs[:, img_len:].float()
    ).abs().max().item()
    print(f"[diag]   image slice [:{img_len}] max|Δ| = {img_diff:.2e}   (num_cams={num_cams})")
    print(f"[diag]   lang slice  [{img_len}:] max|Δ| = {lang_diff:.2e}   (len={lang_len})")

    def _stats(name, t):
        t = t.float()
        print(
            f"[diag] {name} prefix_embs: mean={t.mean().item():+.4f} "
            f"std={t.std().item():.4f} min={t.min().item():+.2f} "
            f"max={t.max().item():+.2f}"
        )
    _stats("LeRobot", lr_prefix_embs)
    _stats("SGLang ", sg_prefix_embs)

    # ── Stage 2: prefix KV cache ──
    prefix_att_2d = make_att_2d_masks(sg_prefix_pad, sg_prefix_att)
    prefix_pos = torch.cumsum(sg_prefix_pad, dim=1) - 1
    prefix_att_4d = prepare_attention_masks_4d(prefix_att_2d)

    # LeRobot uses HF's Gemma forward via OpenPI's patched transformers, which
    # still honours ``_attn_implementation = "eager"`` — flip it so its KV
    # cache is directly comparable with ours.
    lerobot_flow_model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, lr_kv = lerobot_flow_model.paligemma_with_expert.forward(
        attention_mask=prefix_att_4d, position_ids=prefix_pos,
        past_key_values=None, inputs_embeds=[lr_prefix_embs, None], use_cache=True,
    )
    _, sg_kv = sglang_model.paligemma_with_expert.forward(
        attention_mask=prefix_att_4d, position_ids=prefix_pos,
        past_key_values=None, inputs_embeds=[sg_prefix_embs, None], use_cache=True,
    )

    # Extract layer-0 K/V for a direct comparison. SGLang returns a plain
    # ``list[(k, v)]``; LeRobot returns a transformers ``DynamicCache`` with
    # one ``.layers[i]`` entry per decoder layer.
    def _layer0_kv(cache):
        if isinstance(cache, list):
            return cache[0]
        layer0 = cache.layers[0]
        return layer0.keys, layer0.values

    try:
        lr_k, lr_v = _layer0_kv(lr_kv)
        sg_k, sg_v = _layer0_kv(sg_kv)
        dk = (lr_k.float() - sg_k.float()).abs().max().item()
        dv = (lr_v.float() - sg_v.float()).abs().max().item()
        print(f"[diag] prefix KV layer0  K max|Δ|={dk:.2e}  V max|Δ|={dv:.2e}")
    except Exception as e:
        print(f"[diag] could not extract prefix KV for comparison: {e}")

    # ── Stage 3: a single denoise_step at t=1.0 ──
    bsize = state.shape[0]
    t = torch.ones(bsize, dtype=torch.float32, device=state.device)
    lr_vt = lerobot_flow_model.denoise_step(state, sg_prefix_pad, lr_kv, noise, t)
    sg_vt = sglang_model.denoise_step(state, sg_prefix_pad, sg_kv, noise, t)
    print(
        f"[diag] denoise_step(t=1) v_t max|Δ| = "
        f"{(lr_vt.float() - sg_vt.float()).abs().max().item():.2e}"
    )


# ─── Direct-run entry point ───────────────────────────────────────────
if __name__ == "__main__":
    # Mimic pytest -s behavior
    test_pi0_sglang_vs_lerobot()
    if CHECK_OPENPI:
        test_pi0_sglang_vs_openpi()
    print("\n[parity] ✅ All parity checks passed.")
