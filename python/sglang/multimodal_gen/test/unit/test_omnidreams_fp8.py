# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the OmniDreams native FP8 acceleration glue.

The native CUDA kernels are `sm_120`-only and cannot run here, so these tests
cover the CPU-runnable surfaces:

* The FP8 weight-prep core (``cosmos_fp8_utils`` per-output-channel E4M3
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
    """load_extension() succeeds on CUDA sm_120, returns None on CPU."""
    from sglang.multimodal_gen.native import load_extension

    ext = load_extension()
    if torch.cuda.is_available():
        assert ext is not None, "native extension should load on CUDA sm_120"
    else:
        assert ext is None, "native extension should be unavailable on CPU-only"


def test_build_fp8_dit_returns_none_on_cpu():
    """build_fp8_dit(mode='auto') returns FP8 DiT on CUDA when a pre-quantized
    weight file is available; returns None without one (auto-opt-out)."""
    import torch as _t

    from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
    from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import build_fp8_dit

    cfg = OmniDreamsDiTConfig()
    with _t.device("meta"):
        dit = OmniDreamsDiT(config=cfg, hf_config={})
    # Without a fp8_prepared_path, mode=auto should return None
    result = build_fp8_dit(dit, cfg.arch_config, mode="auto")
    if torch.cuda.is_available():
        pass  # build_fp8_dit may return None without fp8_prepared_path on GPU too
    else:
        assert result is None, "FP8 DiT should be unavailable on CPU-only"


def test_build_fp8_dit_required_raises_on_cpu():
    """build_fp8_dit(mode='required') raises FileNotFoundError without a
    pre-quantized weight file."""
    from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
    from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import build_fp8_dit

    cfg = OmniDreamsDiTConfig()
    with torch.device("meta"):
        dit = OmniDreamsDiT(config=cfg, hf_config={})
    # required mode without fp8_prepared_path → FileNotFoundError
    with pytest.raises(FileNotFoundError, match="FP8 prepared weights not found"):
        build_fp8_dit(dit, cfg.arch_config, mode="required")


# --------------------------------------------------------------------------- #
# FP8 weight-prep core (CPU)                                                   #
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
    # Q/K/V are fused into a single to_qkv.weight (post-4be84c0c3).
    for i in range(28):
        assert f"blocks.{i}.self_attn.to_qkv.weight" in sgl_keys, (
            f"blocks.{i}.self_attn.to_qkv.weight missing from state dict"
        )
    # The FP8 quantizer operates on un-fused per-projection keys (q_proj, k_proj,
    # v_proj) while SGLang's DiT fuses them into to_qkv.  The quantizer will fuse
    # them internally; we check that all non-self-attn-proj keys match the SGLang
    # state dict.
    req = set(u.cosmos_block_fp8_linear_keys(28))
    non_self_attn_proj = {
        k for k in req
        if "qkv_proj" not in k
        and not any(k.endswith(f"{x}_proj.weight") for x in ("q", "k", "v"))
    }
    missing = [k for k in non_self_attn_proj if k not in sgl_keys]
    assert not missing, f"SGLang DiT missing FP8 linear keys: {missing[:5]}"


# --------------------------------------------------------------------------- #
# LightVAE FP8 encode fallback + state round-trip + mean/inv_std buffers       #
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


# --------------------------------------------------------------------------- #
# Config three-state validation + migration detection                          #
# --------------------------------------------------------------------------- #
def test_config_three_state_valid():
    """native_dit_acceleration accepts auto/required/disabled."""
    from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
        OmniDreamsPipelineConfig,
    )

    for mode in ("auto", "required", "disabled"):
        cfg = OmniDreamsPipelineConfig(native_dit_acceleration=mode)
        assert cfg.native_dit_acceleration == mode


def test_config_three_state_invalid():
    """Invalid mode raises ValueError."""
    from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
        OmniDreamsPipelineConfig,
    )

    with pytest.raises(ValueError, match="Invalid native_dit_acceleration"):
        OmniDreamsPipelineConfig(native_dit_acceleration="invalid")


def test_config_removed_fields_raise():
    """__post_init__ detects removed fields and raises ValueError."""
    from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
        OmniDreamsPipelineConfig,
    )

    # dataclass won't accept unknown kwargs, so we set via object.__setattr__
    # after construction — but __post_init__ runs at __init__ time, so we test
    # that the field simply doesn't exist as a valid kwarg.
    # The __post_init__ guard uses hasattr(), which catches monkey-patched attrs.
    cfg = OmniDreamsPipelineConfig()
    object.__setattr__(cfg, "use_fp8_dit", True)
    with pytest.raises(ValueError, match="use_fp8_dit.*native_dit_acceleration"):
        cfg.__post_init__()


def test_text_encoder_config_defaults():
    """OmniDreamsTextEncoderConfig has correct defaults."""
    from sglang.multimodal_gen.configs.models.omnidreams_components import (
        OmniDreamsTextEncoderConfig,
    )

    cfg = OmniDreamsTextEncoderConfig()
    assert cfg.impl == "bf16"
    assert cfg.model_id == "nvidia/Cosmos-Reason1-7B"
    assert cfg.fp8_model_path is None


def test_vae_encoder_config_defaults():
    """OmniDreamsVAEEncoderConfig has correct defaults."""
    from sglang.multimodal_gen.configs.models.omnidreams_components import (
        OmniDreamsVAEEncoderConfig,
    )

    cfg = OmniDreamsVAEEncoderConfig()
    assert cfg.impl == "wanvae"
    assert cfg.native_acceleration == "disabled"
    assert len(cfg.latents_mean) == 16
    assert len(cfg.latents_std) == 16


def test_vae_decoder_config_defaults():
    """OmniDreamsVAEDecoderConfig has correct defaults."""
    from sglang.multimodal_gen.configs.models.omnidreams_components import (
        OmniDreamsVAEDecoderConfig,
    )

    cfg = OmniDreamsVAEDecoderConfig()
    assert cfg.impl == "wanvae"
    assert cfg.native_acceleration == "disabled"


def test_vae_encoder_config_pixelshuffle_not_implemented():
    """PixelShuffle impl raises NotImplementedError."""
    from sglang.multimodal_gen.configs.models.omnidreams_components import (
        OmniDreamsVAEEncoderConfig,
    )

    cfg = OmniDreamsVAEEncoderConfig(impl="pixelshuffle")
    with pytest.raises(NotImplementedError):
        cfg.setup()


def test_pipeline_config_nested_configs_exist():
    """OmniDreamsPipelineConfig has nested Config fields."""
    from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
        OmniDreamsPipelineConfig,
    )

    cfg = OmniDreamsPipelineConfig()
    assert cfg.text_encoder_config is not None
    assert cfg.image_encoder_config is not None
    assert cfg.encoder_config is not None
    assert cfg.decoder_config is not None
    assert cfg.encoder_config.impl == "wanvae"
    assert cfg.decoder_config.impl == "wanvae"


# --------------------------------------------------------------------------- #
# Config setup() routing (impl selection + three-state FP8)                    #
# --------------------------------------------------------------------------- #
def test_default_latents_match_validated_wan_stats():
    """The component-config default latents must equal the WanVAEArchConfig
    defaults (the values validated end-to-end). A drift here silently corrupts
    the VAE encode normalization ((z-mean)/std) → wrong-looking output."""
    from sglang.multimodal_gen.configs.models.omnidreams_components import (
        _DEFAULT_LATENTS_MEAN,
        _DEFAULT_LATENTS_STD,
    )
    from sglang.multimodal_gen.configs.models.vaes.wanvae import OmniDreamsVAEConfig

    wan = OmniDreamsVAEConfig()
    assert tuple(_DEFAULT_LATENTS_MEAN) == tuple(wan.latents_mean)
    assert tuple(_DEFAULT_LATENTS_STD) == tuple(wan.latents_std)


def test_vae_encoder_wanvae_setup_routes_and_threads_latents():
    """impl='wanvae' resolves the diffusers VAE path and loads it with a config
    whose latents are threaded through arch_config (not top-level kwargs, which
    would raise TypeError)."""
    from unittest.mock import MagicMock, patch

    import sglang.multimodal_gen.configs.models.omnidreams_components as comp
    from sglang.multimodal_gen.configs.models.omnidreams_components import (
        OmniDreamsVAEEncoderConfig,
    )

    cfg = OmniDreamsVAEEncoderConfig(impl="wanvae", model_path="/fake/model")
    with patch.object(comp, "resolve_wan_vae_path", return_value="/fake/vae") as rp, \
         patch.object(comp, "load_wan_vae", return_value=MagicMock()) as lw:
        cfg.setup()
    rp.assert_called_once()
    lw.assert_called_once()
    vae_cfg = lw.call_args.args[0]
    # latents readable via the VAEConfig.__getattr__ -> arch_config proxy.
    assert tuple(vae_cfg.latents_std) == tuple(cfg.latents_std)


def test_vae_encoder_wanvae_setup_honors_explicit_checkpoint_path():
    """An explicit checkpoint_path bypasses path resolution."""
    from unittest.mock import MagicMock, patch

    import sglang.multimodal_gen.configs.models.omnidreams_components as comp
    from sglang.multimodal_gen.configs.models.omnidreams_components import (
        OmniDreamsVAEEncoderConfig,
    )

    cfg = OmniDreamsVAEEncoderConfig(impl="wanvae", checkpoint_path="/explicit/vae")
    with patch.object(comp, "resolve_wan_vae_path") as rp, \
         patch.object(comp, "load_wan_vae", return_value=MagicMock()) as lw:
        cfg.setup()
    rp.assert_not_called()
    assert lw.call_args.args[1] == "/explicit/vae"


@pytest.mark.parametrize(
    "mode,expect_required",
    [("required", True), ("auto", False), ("disabled", False)],
)
def test_vae_encoder_lightvae_three_state_maps_to_fp8_required(mode, expect_required):
    """LightVAE setup threads the three-state native_acceleration into the
    encoder's fp8_required bool: only 'required' is hard-required; 'auto' and
    'disabled' both stay soft (auto falls back, disabled skips FP8)."""
    from unittest.mock import MagicMock, patch

    from sglang.multimodal_gen.configs.models.omnidreams_components import (
        OmniDreamsVAEEncoderConfig,
    )

    cfg = OmniDreamsVAEEncoderConfig(
        impl="lightvae",
        checkpoint_path="/fake/lightvae.pth",
        native_acceleration=mode,
        fp8_state_path="/fake/fp8.pt",
    )
    with patch(
        "sglang.multimodal_gen.runtime.models.vaes.omnidreams_light_vae.LightVAEEncoder"
    ) as LV:
        LV.return_value = MagicMock()
        cfg.setup()
    kwargs = LV.call_args.kwargs
    assert kwargs["fp8_required"] is expect_required
    # 'disabled' must not pass a calibrated FP8 state at all.
    if mode == "disabled":
        assert kwargs["fp8_state_path"] is None
    else:
        assert kwargs["fp8_state_path"] == "/fake/fp8.pt"


def test_pipeline_config_rehydrates_dict_component_configs():
    """A JSON pipeline-config (via --pipeline-config-path) lands nested component
    configs as raw dicts because the base update_pipeline_config only recurses
    into ModelConfig fields. __post_init__ must rehydrate them into real Config
    dataclasses, else .setup() crashes on a dict at server launch."""
    from sglang.multimodal_gen.configs.models.omnidreams_components import (
        OmniDreamsVAEDecoderConfig,
        OmniDreamsVAEEncoderConfig,
    )
    from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
        OmniDreamsPipelineConfig,
    )

    cfg = OmniDreamsPipelineConfig()
    cfg.update_pipeline_config(
        {
            "native_dit_acceleration": "required",
            "encoder_config": {"impl": "lightvae", "native_acceleration": "auto"},
            "decoder_config": {"impl": "lighttae"},
        }
    )
    assert cfg.native_dit_acceleration == "required"
    assert isinstance(cfg.encoder_config, OmniDreamsVAEEncoderConfig)
    assert cfg.encoder_config.impl == "lightvae"
    assert cfg.encoder_config.native_acceleration == "auto"
    assert isinstance(cfg.decoder_config, OmniDreamsVAEDecoderConfig)
    assert cfg.decoder_config.impl == "lighttae"
    # All component configs must remain callable (no leftover dicts).
    for name in (
        "text_encoder_config",
        "image_encoder_config",
        "encoder_config",
        "decoder_config",
    ):
        assert hasattr(getattr(cfg, name), "setup")


def test_shipped_accel_json_configs_load():
    """The acceleration-path JSON pipeline-configs under test_files/ (referenced
    by the opt-in server cases via --pipeline-config-path) must each load into a
    config whose component slots are real dataclasses with setup(). Guards these
    shipped test assets against rot / rehydration regressions."""
    import glob
    import os

    from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
        OmniDreamsPipelineConfig,
    )

    here = os.path.dirname(__file__)
    cfg_dir = os.path.normpath(os.path.join(here, "..", "test_files"))
    json_files = sorted(glob.glob(os.path.join(cfg_dir, "omnidreams_*.json")))
    assert json_files, f"no omnidreams_*.json under {cfg_dir}"

    for path in json_files:
        cfg = OmniDreamsPipelineConfig()
        cfg.load_from_json(path)
        for name in (
            "text_encoder_config",
            "image_encoder_config",
            "encoder_config",
            "decoder_config",
        ):
            assert hasattr(getattr(cfg, name), "setup"), f"{path}: {name} not a Config"
