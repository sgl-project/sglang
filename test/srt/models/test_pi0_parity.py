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
  PI0_PARITY_MODEL_PATH   Local path (or HF repo id) of the pi0_base checkpoint
                          IN LEROBOT FORMAT. This is consumed by
                          ``PI0Policy.from_pretrained`` and ``draccus`` rejects
                          any SGLang-only keys (``model_type``, ``architectures``,
                          ``auto_map``) it finds in ``config.json``. Defaults to
                          "lerobot/pi0_base" (auto-downloaded from HuggingFace).
                          DO NOT point this at a SGLang-config directory like
                          ``/data08/models/pi0`` — see PI0_PARITY_SERVER_MODEL_PATH
                          for that path.

  Server-parity test only (gated on PI0_PARITY_RUN_SERVER=1):
  PI0_PARITY_RUN_SERVER       "1" to run the /generate end-to-end test
  PI0_PARITY_SERVER_MODEL_PATH Local path of a *SGLang-compatible* π0 dir
                               (config.json contains ``model_type: pi0`` plus
                               the architecture / chunk-size / dtype keys
                               ``Pi0Config`` consumes). The LeRobot HF snapshot
                               does NOT qualify; you need a separate dir.
                               Required when PI0_PARITY_RUN_SERVER=1.
  PI0_PARITY_SERVER_DTYPE      float32 | bfloat16  (default: float32)
  PI0_PARITY_PORT              Server port (default: 30888)
  PI0_PARITY_HOST              Server host (default: 127.0.0.1)
  PI0_PARITY_SERVER_BOOT_TIMEOUT_S  Boot wait (default: 300)

Why two model paths?
    LeRobot's ``PI0Policy.from_pretrained`` parses ``config.json`` through
    draccus and treats unknown top-level keys as a hard error::

        DecodingError: The fields `model_type`, `architectures`, `auto_map`
        are not valid for PI0Config

    while SGLang's ``AutoConfig.from_pretrained`` requires exactly those
    keys. The two formats are mutually incompatible, so the test takes
    each path from its own env var:

      * ``PI0_PARITY_MODEL_PATH`` → LeRobot reference (default: HF download)
      * ``PI0_PARITY_SERVER_MODEL_PATH`` → SGLang server dir (no default)

    Setting both to the same SGLang-config directory will fail in
    ``_instantiate_lerobot``; this is by design.

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

    # Friendly check: if MODEL_PATH points at a *local* dir whose
    # config.json is the SGLang flavor (top-level ``model_type: pi0``),
    # ``PI0Policy.from_pretrained`` blows up deep inside draccus with a
    # confusing ``DecodingError: The fields `model_type`, `architectures`,
    # `auto_map` are not valid for PI0Config``. Catch that here so the
    # user gets a clear message pointing at the right env var.
    if FROM_PRETRAINED and os.path.isdir(MODEL_PATH):
        cfg_path = os.path.join(MODEL_PATH, "config.json")
        if os.path.exists(cfg_path):
            try:
                import json as _json
                with open(cfg_path) as _f:
                    _cfg = _json.load(_f)
                if _cfg.get("model_type") == "pi0":
                    raise RuntimeError(
                        f"PI0_PARITY_MODEL_PATH={MODEL_PATH!r} appears to "
                        "be a SGLang-config π0 directory (config.json has "
                        "'model_type': 'pi0'), but this env var is consumed "
                        "by LeRobot's PI0Policy.from_pretrained which "
                        "rejects SGLang-only keys. Either:\n"
                        "  • unset PI0_PARITY_MODEL_PATH (defaults to the "
                        "    HF repo lerobot/pi0_base, auto-downloaded), or\n"
                        "  • point it at a *LeRobot-format* dir (no "
                        "    'model_type'/'architectures'/'auto_map' keys), "
                        "    keeping PI0_PARITY_SERVER_MODEL_PATH for the "
                        "    SGLang server. See module docstring for details."
                    )
            except (OSError, ValueError):
                # File missing or unparseable; fall through and let
                # PI0Policy.from_pretrained surface its own error.
                pass

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


# ─── /generate server parity ──────────────────────────────────────────
#
# The model-level test above directly calls ``Pi0ForActionPrediction.
# sample_actions(...)``, which bypasses the scheduler / sampler / IPC
# pipeline entirely. That left a regression class uncovered: π0's
# ``forward()`` packs the continuous action chunk into
# ``logits_output.next_token_logits``, but ``Sampler.forward`` mutates
# that tensor in place via ``logits[:] = torch.softmax(...)`` — so
# without an explicit bypass the scheduler reads softmax probabilities
# as actions and /generate returns near-zero values
# (``server_mean ≈ 6.25e-4`` instead of ``-0.0535``).
#
# The test below boots a real SGLang server, hits ``/generate`` with the
# same fixed-noise prefix, and compares the returned ``actions`` chunk
# against a fresh LeRobot ``sample_actions(noise=…)`` call. It is gated
# on ``PI0_PARITY_RUN_SERVER=1`` because it requires:
#   * GPU memory for the full π0_base model
#   * a free port for the server (configurable via PI0_PARITY_PORT)
#   * preprocessed images / state passed via ``extra_body`` —
#     exercising the dataclass-field + __getitem__ plumbing fix.
#
# IMPORTANT — model-path requirements for the server test:
#   The raw ``lerobot/pi0_base`` HF snapshot ships LeRobot's own config
#   layout, NOT the ``model_type: pi0`` config that SGLang's loader
#   recognises. Booting ``sglang serve`` against the unmodified snapshot
#   crashes in ``AutoConfig.from_pretrained`` with::
#
#       ValueError: Unrecognized model in <snapshot_dir>. Should have
#       a `model_type` key in its config.json.
#
#   You need a SGLang-compatible directory: a ``config.json`` whose
#   top-level fields include ``"model_type": "pi0"`` plus the
#   architecture / chunk-size / dtype fields ``Pi0Config`` consumes,
#   pointing at the same ``model.safetensors``. Set
#   ``PI0_PARITY_SERVER_MODEL_PATH`` to that directory (typical layout
#   is what the user already has at ``/data08/models/pi0``); the test
#   does NOT auto-download a server-mode model. If unset, falls back to
#   ``PI0_PARITY_MODEL_PATH`` and prints a hint pointing at this comment.
RUN_SERVER = os.environ.get("PI0_PARITY_RUN_SERVER", "0") == "1"
SERVER_PORT = int(os.environ.get("PI0_PARITY_PORT", "30888"))
SERVER_HOST = os.environ.get("PI0_PARITY_HOST", "127.0.0.1")
SERVER_MODEL_PATH = os.environ.get("PI0_PARITY_SERVER_MODEL_PATH")
SERVER_DTYPE = os.environ.get("PI0_PARITY_SERVER_DTYPE", "float32")
SERVER_BOOT_TIMEOUT_S = int(os.environ.get("PI0_PARITY_SERVER_BOOT_TIMEOUT_S", "300"))



@pytest.mark.skipif(
    not RUN_SERVER,
    reason="Set PI0_PARITY_RUN_SERVER=1 to enable end-to-end /generate test",
)
def test_pi0_sglang_server_vs_lerobot():
    """End-to-end: hit /generate, compare actions to LeRobot.

    This is the test that catches the sampler-bypass bug: it would have
    flagged the original PR (server_mean ≈ 6e-4 vs reference ≈ -0.054)
    immediately, since the model-level path doesn't run the sampler at
    all.
    """
    import json
    import subprocess
    import time

    import requests

    # ── 1. Build the LeRobot reference ──
    print("\n[server-parity] Instantiating LeRobot…")
    lerobot_policy, lerobot_pre, _ = _instantiate_lerobot()
    raw_batch = _create_dummy_batch(batch_size=1)
    processed_batch = lerobot_pre(copy.deepcopy(raw_batch))
    images, img_masks, lang_tokens, lang_masks, state = (
        _extract_lerobot_model_inputs(lerobot_policy, processed_batch)
    )
    noise = _make_fixed_noise(1, DEVICE)
    with torch.no_grad():
        ref_actions = lerobot_policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state,
            noise=noise, num_steps=NUM_STEPS,
        )
    ref_mean = ref_actions.mean().item()
    ref_std = ref_actions.std().item()
    print(f"[server-parity] reference: mean={ref_mean:+.4f}  std={ref_std:.4f}")

    # ── 2. Boot the server in a subprocess ──
    # Resolve the server model path. The raw lerobot/pi0_base HF snapshot
    # does NOT have ``model_type: pi0`` in its config.json, so booting
    # ``sglang serve`` against it crashes with "Unrecognized model …
    # Should have a `model_type` key". We therefore require an explicit
    # SGLang-compatible directory via PI0_PARITY_SERVER_MODEL_PATH (e.g.
    # /data08/models/pi0). If unset, fall back to PI0_PARITY_MODEL_PATH
    # but only if it's a local dir (HF repo ids will fail at boot).
    if SERVER_MODEL_PATH:
        server_model_path = SERVER_MODEL_PATH
    else:
        server_model_path = MODEL_PATH if os.path.isdir(MODEL_PATH) else None
    if not server_model_path or not os.path.isdir(server_model_path):
        pytest.skip(
            "Server-mode test needs a SGLang-compatible π0 directory "
            "(config.json must have model_type='pi0'). The raw "
            "lerobot/pi0_base HF snapshot does NOT qualify. Set "
            "PI0_PARITY_SERVER_MODEL_PATH=/path/to/sglang/pi0 to enable."
        )

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", server_model_path,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--dtype", SERVER_DTYPE,
        "--mem-fraction-static", "0.6",
        # CUDA-graph capture is auto-disabled for VLA models in
        # ServerArgs._handle_attention_backend_compatibility, but pass
        # the flag explicitly so the test still works on older SGLang
        # builds that don't have that auto-disable yet.
        "--disable-cuda-graph",
    ]
    print(f"[server-parity] launching: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    base_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
    deadline = time.time() + SERVER_BOOT_TIMEOUT_S
    try:
        boot_ok = False
        while time.time() < deadline:
            # Surface server crashes (e.g. config-mismatch ValueError)
            # immediately instead of waiting for the full timeout.
            if proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {proc.returncode} during boot. "
                    "Check the server stdout/stderr above for the real error "
                    "(common: model_type mismatch, FA fp32-KV crash, "
                    "OOM during weight load)."
                )
            try:
                r = requests.get(f"{base_url}/health", timeout=2)
                if r.status_code == 200:
                    boot_ok = True
                    break
            except Exception:
                pass
            time.sleep(2)
        if not boot_ok:
            raise RuntimeError(
                f"server failed to come up in {SERVER_BOOT_TIMEOUT_S}s"
            )


        # ── 3. Submit a JSON payload through the normal image-loading
        #       path. We can't use the precomputed-feature fast path
        #       because it requires a real ``torch.Tensor`` (not JSON
        #       serializable) and a single pre-stacked
        #       ``(num_cams, 3, 224, 224)`` dict; per-camera dicts and
        #       nested-list features are rejected by the strict
        #       ``Pi0Processor._process_precomputed`` contract.
        #
        #       So we round-trip the camera tensors through PIL → JPEG
        #       → base64 (the same wire format the docs advertise) and
        #       hand the server raw image bytes. The π0 multimodal
        #       processor decodes them through ``load_image``,
        #       resize_with_pad, and SigLIP exactly like a real client
        #       would. Numerical results no longer match LeRobot
        #       bit-for-bit (JPEG loss + a second float→float SigLIP
        #       pass introduce noise on the order of 1e-3 in action
        #       space), but the std/mean checks are still tight enough
        #       to catch the softmax-of-actions bug — that one
        #       collapses ``srv_std`` to ~1e-4 while real actions land
        #       at ``ref_std ≈ 0.30``.
        #
        #       ``extra_body`` carries the per-request state +
        #       num_inference_steps — without the dataclass plumbing
        #       fix this field would be silently dropped before the
        #       processor reads ``request_obj.extra_body``.
        import base64
        import io

        from PIL import Image

        def _images_to_b64(t: torch.Tensor) -> str:
            """Convert a (1, 3, 224, 224) tensor in [-1, 1] to a base64
            JPEG data URL. We undo the LeRobot normalization (x*0.5+0.5)
            so PIL sees a plausible 0-1 image; the server will re-apply
            its own preprocessing."""
            arr = (t[0].clamp(-1, 1) * 0.5 + 0.5).clamp(0, 1)
            arr = (arr * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=95)
            return "data:image/jpeg;base64," + base64.b64encode(
                buf.getvalue()
            ).decode("ascii")

        payload = {
            "text": raw_batch["task"][0],
            "image_data": [_images_to_b64(images[c]) for c in range(len(images))],
            "sampling_params": {"max_new_tokens": 1},
            "extra_body": {
                # Plumbing-test signal: π0's processor reads ``state``
                # off ``request_obj.extra_body``. Without the dataclass
                # field + ``__getitem__`` thread-through, this dict is
                # silently dropped on the wire and the server falls
                # back to a zero state (which still yields a valid
                # action chunk but a different one).
                "state": state[0].cpu().tolist(),
                "num_inference_steps": NUM_STEPS,
            },
        }
        r = requests.post(f"{base_url}/generate", json=payload, timeout=120)
        if r.status_code != 200:
            raise AssertionError(
                f"/generate returned {r.status_code}: {r.text[:500]}"
            )
        body = r.json()

        # ── 4. Compare ──
        assert "actions" in body, (
            f"/generate response is missing 'actions' field; got keys "
            f"{list(body.keys())}. The VLA output promotion path is broken."
        )
        # Verify the FINISH_VLA marker survived the tokenizer-manager
        # promotion path — this is the documented client-facing contract.
        finish = body.get("meta_info", {}).get("finish_reason") or {}
        assert finish.get("matched") == "vla_done", (
            f"finish_reason.matched should be 'vla_done' for VLA; got {finish!r}"
        )

        srv_actions = torch.tensor(body["actions"], dtype=torch.float32)
        if srv_actions.dim() == 2:
            srv_actions = srv_actions.unsqueeze(0)
        srv_mean = srv_actions.mean().item()
        srv_std = srv_actions.std().item()
        print(
            f"[server-parity] server   : mean={srv_mean:+.4f}  std={srv_std:.4f}"
        )

        # The strongest signal that the sampler-bypass is in place: the
        # server's std should match the reference order of magnitude,
        # NOT collapse toward ``1/(horizon*dim)``. Pre-fix this check
        # was the smoking gun (server_std ≈ 1.8e-4 vs reference 0.30).
        # We use a fairly loose tolerance (0.15) because the server
        # goes through JPEG round-trip + a second SigLIP pass, both of
        # which add float noise. The bypass-still-broken case has a
        # ~1000× std gap, so this bound easily separates the two.
        assert abs(srv_std - ref_std) < 0.15, (
            f"server std {srv_std:.4f} differs from reference {ref_std:.4f} — "
            "this is the signature of the in-place softmax mutation in "
            "Sampler.forward. The VLA branch must call "
            "forward_batch_generation(skip_sample=True)."
        )
        # Mean check is even looser because the server samples its own
        # noise (we don't plumb a fixed-noise override through
        # extra_body) and JPEG/SigLIP shifts the mean a bit.
        assert abs(srv_mean - ref_mean) < 0.15, (
            f"server mean {srv_mean:+.4f} differs from reference "
            f"{ref_mean:+.4f}; bypass is likely broken."
        )

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ─── Direct-run entry point ───────────────────────────────────────────
if __name__ == "__main__":
    # Mimic pytest -s behavior
    test_pi0_sglang_vs_lerobot()
    if CHECK_OPENPI:
        test_pi0_sglang_vs_openpi()
    if RUN_SERVER:
        test_pi0_sglang_server_vs_lerobot()
    print("\n[parity] ✅ All parity checks passed.")

