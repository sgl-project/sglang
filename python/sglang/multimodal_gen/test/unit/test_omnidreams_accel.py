# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the OmniDreams acceleration code.

* ``CUDAGraphWrapper`` input-staging logic + ``set_or_copy`` pointer
  stability (the CPU-checkable half; capture/replay needs CUDA).
* LightTAE (TAEHV) checkpoint-key coverage, ``frames_to_trim`` math, and a
  decode-shape smoke (real ``lighttaew2_1.pth`` -- skipped if absent).
* LightVAE (pruned Wan) checkpoint-key coverage + encode-shape smoke
  (real ``lightvaew2_1.pth`` -- skipped if absent).

The checkpoint-dependent tests resolve the weights from
``SGLANG_OMNIDREAMS_{LIGHTTAE,LIGHTVAE}_CKPT`` or a few known local paths, and
``pytest.skip`` when none exist (so CI without the weights stays green).
"""

from __future__ import annotations

import os

import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits.omnidreams_cuda_graph import (
    CUDAGraphWrapper,
    set_or_copy,
)


def _find_ckpt(env_key: str, *names: str) -> str | None:
    if os.environ.get(env_key) and os.path.isfile(os.environ[env_key]):
        return os.environ[env_key]
    roots = ["/Users/cerdore/gitRepo/models", "/root/blockdata", os.getcwd()]
    for root in roots:
        for name in names:
            cand = os.path.join(root, name)
            if os.path.isfile(cand):
                return cand
    return None


# --------------------------------------------------------------------------- #
# CUDAGraphWrapper staging logic                                              #
# --------------------------------------------------------------------------- #
def test_set_or_copy_pointer_stable_on_same_shape():
    state: dict = {}
    set_or_copy(state, 0, torch.ones(2, 3))
    ptr = state[0].data_ptr()
    set_or_copy(state, 0, torch.full((2, 3), 5.0))  # same shape -> in-place copy_
    assert state[0].data_ptr() == ptr, "same-shape write must preserve storage"
    assert torch.equal(state[0], torch.full((2, 3), 5.0))
    set_or_copy(state, 0, torch.ones(4, 5))  # new shape -> fresh buffer
    assert state[0].shape == (4, 5)


def test_cuda_graph_wrapper_warmup_runs_eager_through_static_buffers():
    # High warmup so we stay on the eager path (the capture path needs CUDA).
    seen = []

    def fn(a, *, b):
        seen.append(a)  # the staged static buffer, not the caller's tensor
        return a + b

    w = CUDAGraphWrapper(fn, warmup_iters=8)
    out = w(torch.tensor([1.0]), b=torch.tensor([2.0]))
    assert torch.equal(out, torch.tensor([3.0]))
    # The arg fn sees IS the wrapper's static buffer (copy-in staging).
    assert seen[0].data_ptr() == w._static_args[0].data_ptr()
    assert not w.is_capturing_or_captured  # still warming up, not captured


def test_cuda_graph_wrapper_signature_change_resets():
    def fn(a):
        return a * 2

    # High warmup so neither shape captures (CPU has no CUDAGraph).
    w = CUDAGraphWrapper(fn, warmup_iters=8)
    w(torch.zeros(2, 2))
    first_buf = w._static_args[0]
    w(torch.zeros(2, 2))  # same shape -> reuse the same static buffer
    assert w._static_args[0].data_ptr() == first_buf.data_ptr()
    rem_before = w._warmup_remaining
    w(torch.zeros(3, 3))  # shape change -> reset + realloc + warmup restart
    assert w._static_args[0].shape == (3, 3)
    # Reset restored warmup_remaining to the full count, then consumed one.
    assert w._warmup_remaining == w.warmup_iters - 1 > rem_before - 1


def test_cuda_graph_wrapper_non_tensor_args_passthrough():
    seen = {}

    def fn(t, cache, flag):
        seen["cache_is_same"] = cache is sentinel
        seen["flag"] = flag
        return t

    w = CUDAGraphWrapper(fn, warmup_iters=1)
    sentinel = ["kv-cache-list"]  # non-tensor container passes through verbatim
    w(torch.ones(1), sentinel, 7)
    assert seen["cache_is_same"] and seen["flag"] == 7


# --------------------------------------------------------------------------- #
# LightTAE                                                                     #
# --------------------------------------------------------------------------- #
def test_lighttae_frames_to_trim_math():
    from sglang.multimodal_gen.runtime.models.vaes.taehv import TAEHV

    m = TAEHV(checkpoint_path=None)
    # 2 ** sum((True, True)) - 1 == 3
    assert m.frames_to_trim == 3
    assert m.TEMPORAL_COMPRESSION_RATIO == 4 and m.SPATIAL_COMPRESSION_RATIO == 8


def test_lighttae_checkpoint_key_coverage():
    from sglang.multimodal_gen.runtime.models.vaes.taehv import (
        TAEHV,
        lighttae_state_dict_transform,
    )

    ckpt = _find_ckpt("SGLANG_OMNIDREAMS_LIGHTTAE_CKPT", "lighttaew2_1.pth")
    if ckpt is None:
        pytest.skip("lighttaew2_1.pth not found")
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    remapped = lighttae_state_dict_transform(raw)
    model_keys = set(TAEHV(checkpoint_path=None).state_dict().keys())
    missing = [k for k in model_keys if k not in remapped]
    assert not missing, f"checkpoint missing {len(missing)} model keys: {missing[:5]}"
    # Full load leaves nothing on meta.
    loaded = TAEHV(checkpoint_path=ckpt)
    assert not [k for k, v in loaded.state_dict().items() if v.is_meta]


def test_lighttae_decode_shape():
    from sglang.multimodal_gen.runtime.models.vaes.taehv import LightTAEDecoder

    ckpt = _find_ckpt("SGLANG_OMNIDREAMS_LIGHTTAE_CKPT", "lighttaew2_1.pth")
    if ckpt is None:
        pytest.skip("lighttaew2_1.pth not found")
    dec = LightTAEDecoder(ckpt, dtype=torch.float32)
    z = torch.randn(1, 16, 3, 8, 8)  # [B, C, F, H, W], 3 latent frames
    out = dec.decode(z)
    # F_out = 1 + (3-1)*4 = 9 ; spatial 8 * 8 = 64 ; 3 image channels ; [-1,1].
    assert out.shape == (1, 3, 9, 64, 64)
    assert torch.isfinite(out).all()
    assert out.min() >= -1.0 - 1e-4 and out.max() <= 1.0 + 1e-4


# --------------------------------------------------------------------------- #
# LightVAE                                                                     #
# --------------------------------------------------------------------------- #
def test_lightvae_checkpoint_key_coverage_and_encode_shape():
    from sglang.multimodal_gen.runtime.models.vaes.omnidreams_light_vae import (
        LightVAEEncoder,
    )

    ckpt = _find_ckpt("SGLANG_OMNIDREAMS_LIGHTVAE_CKPT", "lightvaew2_1.pth")
    if ckpt is None:
        pytest.skip("lightvaew2_1.pth not found")
    enc = LightVAEEncoder(
        ckpt, latents_mean=[0.0] * 16, latents_std=[1.0] * 16, dtype=torch.float32
    )
    # No encoder/quant params left on meta after the (strict=False) load.
    assert not [k for k, v in enc.state_dict().items() if v.is_meta]
    # First-frame image: [B,3,1,H,W] -> [B,16,1,H/8,W/8].
    z1 = enc.encode(torch.randn(1, 3, 1, 64, 64)).mode()
    assert z1.shape == (1, 16, 1, 8, 8) and torch.isfinite(z1).all()
    # HD-map clip: 5 pixel frames -> 2 causal latent frames.
    z2 = enc.encode(torch.randn(1, 3, 5, 64, 64)).mode()
    assert z2.shape == (1, 16, 2, 8, 8) and torch.isfinite(z2).all()
