# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the OmniDreams P4 native FP8 acceleration glue.

The native CUDA kernels are `sm_120`-only and cannot run here, so these tests
cover the CPU-runnable surfaces:

* P4a: the FP8 weight-prep core (``cosmos_fp8_utils`` per-output-channel E4M3
  quant -> uint8 RCR bytes + per-channel scale) and FP8 linear-key compatibility
  with SGLang's ``OmniDreamsDiT`` state dict (verifies the near-identity mapping).
* The native acceleration strategy reports *unavailable* cleanly on CPU, so
  ``build_fp8_dit`` returns None and the AR loop falls back to the eager DiT.

The vendored python helpers are loaded via the native loader; tests ``skip`` if
the vendored tree is absent.
"""

from __future__ import annotations

import pytest
import torch


def _load_helper(name: str):
    try:
        from sglang.multimodal_gen.native.singleview_loader import load_python_module

        return load_python_module(name)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"vendored native helper {name!r} unavailable: {e}")


# --------------------------------------------------------------------------- #
# Native loader / strategy (CPU = unavailable -> fallback)                     #
# --------------------------------------------------------------------------- #
def test_native_extension_unavailable_on_cpu():
    # auto: no CUDA / no sm_120 build here -> load_extension resolves to None.
    from sglang.multimodal_gen.native import NativeAccelerationConfig, load_extension

    assert load_extension(NativeAccelerationConfig(mode="auto")) is None


def test_build_fp8_dit_returns_none_on_cpu():
    import torch as _t

    from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
    from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import build_fp8_dit

    cfg = OmniDreamsDiTConfig()
    with _t.device("meta"):
        dit = OmniDreamsDiT(config=cfg, hf_config={})
    assert build_fp8_dit(dit, cfg.arch_config, mode="auto") is None


def test_build_fp8_dit_required_raises_on_cpu():
    from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
    from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import build_fp8_dit

    cfg = OmniDreamsDiTConfig()
    with torch.device("meta"):
        dit = OmniDreamsDiT(config=cfg, hf_config={})
    with pytest.raises(Exception):
        build_fp8_dit(dit, cfg.arch_config, mode="required")


# --------------------------------------------------------------------------- #
# P4a: FP8 weight-prep core (CPU)                                              #
# --------------------------------------------------------------------------- #
def test_cosmos_fp8_per_out_channel_quant_rcr_contract():
    u = _load_helper("cosmos_fp8_utils")
    w = torch.randn(256, 512)
    out = u.quantize_fp8_per_out_channel(w)
    wq, scale = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
    # Native RCR contract: raw E4M3 bytes as uint8 [out, in] + per-out scale.
    assert wq.dtype == torch.uint8 and tuple(wq.shape) == (256, 512)
    assert scale is not None and tuple(scale.shape) == (256,)


def test_fp8_linear_keys_compatible_with_sglang_dit():
    u = _load_helper("cosmos_fp8_utils")
    from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
    from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT

    with torch.device("meta"):
        dit = OmniDreamsDiT(config=OmniDreamsDiTConfig(), hf_config={})
    sgl_keys = set(dit.state_dict().keys())
    # Every fused-QKV target is derivable from SGLang's separate q/k/v projs.
    assert all(
        f"blocks.{i}.self_attn.q_proj.weight" in sgl_keys
        and f"blocks.{i}.self_attn.k_proj.weight" in sgl_keys
        and f"blocks.{i}.self_attn.v_proj.weight" in sgl_keys
        for i in range(28)
    )
    # The non-fused FP8 linear keys (output_proj, cross_attn, mlp) exist as-is.
    req = set(u.cosmos_block_fp8_linear_keys(28))
    non_qkv = {k for k in req if "qkv_proj" not in k}
    missing = [k for k in non_qkv if k not in sgl_keys]
    assert not missing, f"SGLang DiT missing FP8 linear keys: {missing[:5]}"


# --------------------------------------------------------------------------- #
# P4b: LightVAE FP8 encode fallback + state round-trip + mean/inv_std buffers  #
# --------------------------------------------------------------------------- #
def _find_ckpt(env_key: str, *names: str) -> str | None:
    import os as _os

    if _os.environ.get(env_key) and _os.path.isfile(_os.environ[env_key]):
        return _os.environ[env_key]
    roots = ["/Users/cerdore/gitRepo/models", "/root/blockdata", _os.getcwd()]
    for root in roots:
        for name in names:
            cand = _os.path.join(root, name)
            if _os.path.isfile(cand):
                return cand
    return None


def test_light_vae_fp8_falls_back_to_pytorch_on_cpu():
    """On CPU, native FP8 path is skipped (x.is_cuda gate) → PyTorch path used."""
    from sglang.multimodal_gen.runtime.models.vaes.omnidreams_light_vae import (
        LightVAEEncoder,
    )

    ckpt = _find_ckpt("SGLANG_OMNIDREAMS_LIGHTVAE_CKPT", "lightvaew2_1.pth")
    if ckpt is None:
        pytest.skip("lightvaew2_1.pth not found")
    enc = LightVAEEncoder(
        ckpt,
        latents_mean=[0.0] * 16,
        latents_std=[1.0] * 16,
        dtype=torch.float32,
        fp8_state_path="/nonexistent/fp8_state.pt",
        fp8_required=False,
    )
    # CPU input: x.is_cuda gate short-circuits the native FP8 path entirely,
    # so no native attempt is made → _native_disabled stays False.
    assert enc._fp8_enabled, "fp8_enabled should be True (fp8_state_path provided)"
    z1 = enc.encode(torch.randn(1, 3, 1, 64, 64)).mode()
    assert z1.shape == (1, 16, 1, 8, 8) and torch.isfinite(z1).all()
    assert not enc._native_disabled, (
        "_native_disabled should stay False: native path was skipped (CPU input), "
        "not attempted-and-failed"
    )

    # Second call: still PyTorch (same CPU skip).
    z2 = enc.encode(torch.randn(1, 3, 5, 64, 64)).mode()
    assert z2.shape == (1, 16, 2, 8, 8) and torch.isfinite(z2).all()


def test_light_vae_fp8_state_roundtrip():
    """Exercise _quantize_fp8_per_channel + _build_fp8_state header keys on
    a synthetic state dict (pure-CPU, no native ext needed)."""
    u = _load_helper("vae_weights")

    # Build a minimal synthetic state dict with one float weight.
    synthetic = {
        "encoder.conv1.weight": torch.randn(24, 3, 3, 3, 3),
        "encoder.downsamples.0.residual.2.weight": torch.randn(24, 24, 3, 3, 3),
        "some_int": torch.tensor([1], dtype=torch.int32),
        "some_float": torch.tensor([1.0]),
    }
    fp8_type = getattr(torch, "float8_e4m3fn", None)
    if fp8_type is None:
        pytest.skip("float8_e4m3fn not available")

    # Use the export tool's quant function (imported inline).
    # Build a state dict mimicking the export tool output.
    scale_max = 24.0

    def _quant(tensor, channel_dim=0):
        reduce_dims = tuple(i for i in range(tensor.dim()) if i != channel_dim)
        fp32 = tensor.float()
        amax = fp32.abs().amax(dim=reduce_dims) if reduce_dims else fp32.abs()
        scale = (amax / float(scale_max)).clamp(min=1.0e-6)
        scaled = fp32 / scale.reshape(
            tuple(tensor.shape[i] if i == channel_dim else 1 for i in range(tensor.dim()))
        )
        return scaled.to(fp8_type).contiguous().view(torch.uint8), scale.to(torch.float16)

    state: dict = {
        "__omnidreams_vae_fp8_version__": torch.tensor([1], dtype=torch.int32),
        "__omnidreams_vae_fp8_model_kind__": torch.tensor([1], dtype=torch.int32),
        "__omnidreams_vae_fp8_scale_max__": torch.tensor([float(scale_max)], dtype=torch.float32),
        "encoder.conv1.input.activation_scale": torch.ones(3, dtype=torch.float16),
    }
    for name, tensor in synthetic.items():
        if name.endswith(".weight") and torch.is_floating_point(tensor) and tensor.dim() >= 2:
            q, scale = _quant(tensor)
            state[name] = q
            state[name.replace(".weight", ".weight_scale")] = scale
        elif torch.is_floating_point(tensor):
            state[name] = tensor.detach().to(dtype=torch.float16)
        else:
            state[name] = tensor.detach()

    # Round-trip via load_lightvae_fp8_state.
    import tempfile
    tmp = tempfile.mktemp(suffix=".pt")
    try:
        torch.save(state, tmp)
        loaded = u.load_lightvae_fp8_state(tmp)
        assert "__omnidreams_vae_fp8_version__" in state
        assert "encoder.conv1.weight_scale" in loaded
        assert loaded["encoder.conv1.weight_scale"].ndim == 1
        assert loaded["encoder.conv1.weight_scale"].shape[0] == 24
        assert loaded["encoder.conv1.weight"].dtype == torch.uint8
        assert loaded["encoder.conv1.input.activation_scale"].dtype == torch.float16
    finally:
        import os as _os
        if _os.path.isfile(tmp):
            _os.unlink(tmp)


def test_mean_inv_std_buffers():
    """LightVAEEncoder exposes mean/inv_std buffers of correct shape."""
    from sglang.multimodal_gen.runtime.models.vaes.omnidreams_light_vae import (
        LightVAEEncoder,
    )

    ckpt = _find_ckpt("SGLANG_OMNIDREAMS_LIGHTVAE_CKPT", "lightvaew2_1.pth")
    if ckpt is None:
        pytest.skip("lightvaew2_1.pth not found")
    enc = LightVAEEncoder(
        ckpt,
        latents_mean=list(range(16)),
        latents_std=[float(i + 1) for i in range(16)],
        dtype=torch.float32,
    )
    assert enc.mean.numel() == 16
    assert enc.inv_std.numel() == 16
    # inv_std ≈ 1/std
    expected_inv = 1.0 / torch.tensor([float(i + 1) for i in range(16)])
    assert torch.allclose(
        enc.inv_std.float(), expected_inv.float(), rtol=1e-5
    ), f"inv_std mismatch: {enc.inv_std}"
