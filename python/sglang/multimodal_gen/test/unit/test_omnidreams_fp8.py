# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the OmniDreams FP8 weight utilities (Phase 1).

Phase 1 removed the vendored native CUDA FP8 DiT tree. These tests cover the
CPU-runnable surfaces that remain:

* The FP8 weight-prep core (relocated ``omnidreams_cosmos_fp8_utils``
  per-output-channel E4M3 quant -> uint8 RCR bytes + per-channel scale) and FP8
  linear-key compatibility with SGLang's ``OmniDreamsDiT`` state dict.
* ``prepare_fp8_dit_weights`` unfuses the fused ``to_qkv`` into q/k/v before
  quantization (byte-identical to the pre-refactor split path).
* Config three-state validation + ``auto``/``required`` back-compat alias
  mapping + component-config rehydration.

The native-ext / CUDA-source-text / LightVAE-FP8-state tests were deleted with
the native tree.
"""

from __future__ import annotations

import pytest
import torch


# --------------------------------------------------------------------------- #
# FP8 weight-prep core (CPU)                                                   #
# --------------------------------------------------------------------------- #
def test_cosmos_fp8_per_out_channel_quant_rcr_contract():
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_cosmos_fp8_utils import (
        quantize_fp8_per_out_channel,
    )

    w = torch.randn(256, 512)
    out = quantize_fp8_per_out_channel(w)
    wq, scale = (out[0], out[1]) if isinstance(out, tuple) else (out, None)
    # Native RCR contract: raw E4M3 bytes as uint8 [out, in] + per-out scale.
    assert wq.dtype == torch.uint8 and tuple(wq.shape) == (256, 512)
    assert scale is not None and tuple(scale.shape) == (256,)


def test_fp8_linear_keys_compatible_with_sglang_dit():
    from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
    from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_cosmos_fp8_utils import (
        cosmos_block_fp8_linear_keys,
    )

    with torch.device("meta"):
        dit = OmniDreamsDiT(config=OmniDreamsDiTConfig(), hf_config={})
    sgl_keys = set(dit.state_dict().keys())
    # Q/K/V are fused into a single to_qkv.weight (post-4be84c0c3).
    for i in range(28):
        assert (
            f"blocks.{i}.self_attn.to_qkv.weight" in sgl_keys
        ), f"blocks.{i}.self_attn.to_qkv.weight missing from state dict"
    # The FP8 quantizer operates on un-fused per-projection keys (q_proj, k_proj,
    # v_proj) while SGLang's DiT fuses them into to_qkv.  The quantizer will fuse
    # them internally; we check that all non-self-attn-proj keys match the SGLang
    # state dict.
    req = set(cosmos_block_fp8_linear_keys(28))
    non_self_attn_proj = {
        k
        for k in req
        if "qkv_proj" not in k
        and not any(k.endswith(f"{x}_proj.weight") for x in ("q", "k", "v"))
    }
    missing = [k for k in non_self_attn_proj if k not in sgl_keys]
    assert not missing, f"SGLang DiT missing FP8 linear keys: {missing[:5]}"


def _fake_fused_dit_state_dict(num_blocks: int, *, qdim: int = 64, inner: int = 48):
    """Minimal bf16 state dict mimicking the post-to_qkv-refactor OmniDreamsDiT.

    Self-attn Q/K/V is fused into ``to_qkv`` (q,k,v row order); cross-attn K/V
    is fused into ``to_kv`` (unused by Cosmos FP8 prep, must pass through).
    """
    torch.manual_seed(0)
    sd: dict[str, torch.Tensor] = {}
    for i in range(num_blocks):
        p = f"blocks.{i}."
        q = torch.randn(inner, qdim, dtype=torch.bfloat16)
        k = torch.randn(inner, qdim, dtype=torch.bfloat16)
        v = torch.randn(inner, qdim, dtype=torch.bfloat16)
        sd[p + "self_attn.to_qkv.weight"] = torch.cat([q, k, v], dim=0).contiguous()
        sd[p + "self_attn.output_proj.weight"] = torch.randn(
            qdim, inner, dtype=torch.bfloat16
        )
        sd[p + "cross_attn.q_proj.weight"] = torch.randn(
            inner, qdim, dtype=torch.bfloat16
        )
        sd[p + "cross_attn.to_kv.weight"] = torch.randn(
            2 * inner, qdim, dtype=torch.bfloat16
        )
        sd[p + "cross_attn.output_proj.weight"] = torch.randn(
            qdim, inner, dtype=torch.bfloat16
        )
        sd[p + "mlp.layer1.weight"] = torch.randn(inner * 2, qdim, dtype=torch.bfloat16)
        sd[p + "mlp.layer2.weight"] = torch.randn(qdim, inner * 2, dtype=torch.bfloat16)
        sd[p + "self_attn.q_norm.weight"] = torch.ones(inner, dtype=torch.bfloat16)
        sd[p + "self_attn.k_norm.weight"] = torch.ones(inner, dtype=torch.bfloat16)
        sd[p + "cross_attn.k_norm.weight"] = torch.ones(inner, dtype=torch.bfloat16)
    return sd


def test_prepare_fp8_dit_weights_unfuses_to_qkv_into_qkv_proj():
    """The offline exporter must accept the post-refactor fused ``to_qkv`` DiT
    state dict (commit 5dff6576c merged self-attn q/k/v into MergedColumnParallelLinear).

    Before the fix ``prepare_fp8_dit_weights`` raised KeyError on the missing
    split ``q_proj``; now it unfuses ``to_qkv`` -> q/k/v, lets the Cosmos prep
    rebuild ``qkv_proj`` (fp8), and drops the dead ``to_qkv``.
    """
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
        prepare_fp8_dit_weights,
    )

    nb = 2
    inner = 48  # matches _fake_fused_dit_state_dict default
    sd = _fake_fused_dit_state_dict(nb)
    fused_snapshot = {
        i: sd[f"blocks.{i}.self_attn.to_qkv.weight"].clone() for i in range(nb)
    }

    out = prepare_fp8_dit_weights(sd, num_blocks=nb, linear_policy="all")

    for i in range(nb):
        # to_qkv dropped (dead bf16 must not ship in the fp8 artifact).
        assert f"blocks.{i}.self_attn.to_qkv.weight" not in out
        # rebuilt fused qkv_proj is fp8 with a per-out-channel scale.
        qk = f"blocks.{i}.self_attn.qkv_proj.weight"
        sk = qk + "_scale"
        assert out[qk].dtype == torch.uint8, f"{qk} not fp8"
        assert tuple(out[sk].shape) == (out[qk].shape[0],)
        # split q/k/v are NOT retained (default drops them in favor of qkv_proj).
        for rel in ("q_proj", "k_proj", "v_proj"):
            assert f"blocks.{i}.self_attn.{rel}.weight" not in out
        # dequant recovers the original to_qkv within e4m3 precision, with the
        # q/k/v shard boundaries intact (q rows 0..inner, k inner..2*inner, ...).
        deq = out[qk].view(torch.float8_e4m3fn).to(torch.float32) * out[sk].to(
            torch.float32
        ).unsqueeze(1)
        orig = fused_snapshot[i].to(torch.float32)
        assert (deq - orig).abs().max().item() < 1.0, f"block {i} dequant drift"
        assert torch.allclose(deq[:inner], orig[:inner], atol=1.0)


def test_prepare_fp8_dit_weights_unfused_matches_split_path_bytes():
    """Per-output-channel FP8 scales are row-independent, so unfusing to_qkv
    then quantizing must be byte-identical to the pre-refactor split-q/k/v path
    (the artifact sglang currently ships). Guards against a regression that
    changes the unfuse shard order or re-introduces a stale-weight mismatch."""
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
        _unfuse_self_attn_qkv_for_cosmos,
        prepare_fp8_dit_weights,
    )

    nb = 2
    fused = _fake_fused_dit_state_dict(nb)
    # Pre-refactor equivalent: split to_qkv back into q/k/v BEFORE prep (bypass
    # the unfuse inside prepare_fp8_dit_weights by feeding already-split input
    # that has no to_qkv key).
    split = _unfuse_self_attn_qkv_for_cosmos(dict(fused))

    got_fused = prepare_fp8_dit_weights(dict(fused), num_blocks=nb, linear_policy="all")
    got_split = prepare_fp8_dit_weights(dict(split), num_blocks=nb, linear_policy="all")

    assert set(got_fused) == set(got_split)
    for key in got_fused:
        a, b = got_fused[key], got_split[key]
        if isinstance(a, torch.Tensor) and a.dtype == torch.uint8:
            assert torch.equal(a, b), f"fp8 byte mismatch: {key}"
            sk = key + "_scale"
            if sk in got_fused:
                assert torch.equal(
                    got_fused[sk], got_split[sk]
                ), f"scale mismatch: {sk}"


# --------------------------------------------------------------------------- #
# Phase 2: FP8-compute linears + CPU fallback                                  #
# --------------------------------------------------------------------------- #
def test_fp8_compute_method_matmul_matches_bf16_within_fp8_tolerance():
    """OmniDreamsFP8ComputeLinearMethod.apply (rowwise _scaled_mm) must produce
    output within FP8 e4m3 tolerance of the bf16 reference GEMM. GPU-only: the
    rowwise torch._scaled_mm path needs CUDA + float8_e4m3fn support."""
    if not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("FP8 _scaled_mm requires CUDA + float8_e4m3fn")
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
        OmniDreamsFP8ComputeLinearMethod,
    )

    torch.manual_seed(0)
    N, K, M = 64, 128, 32
    weight = torch.randn(N, K, dtype=torch.bfloat16, device="cuda") * 0.3
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.3

    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    method = OmniDreamsFP8ComputeLinearMethod()
    out_fp8 = method.apply(layer, x, bias=None)
    ref = (x.float() @ weight.float().t()).to(torch.bfloat16)
    # FP8 e4m3 has 3 mantissa bits; per-token/per-channel scales keep drift small.
    max_abs_err = (out_fp8.float() - ref.float()).abs().max().item()
    assert max_abs_err < 0.5, f"fp8_compute drift too large: {max_abs_err}"
    assert out_fp8.shape == (M, N)


def test_install_fp8_compute_on_dit_is_noop_on_cpu():
    """fp8_compute must gracefully fall back to bf16 on CPU (no _scaled_mm):
    install_fp8_compute_on_dit returns False and leaves the DiT linears intact.

    Only meaningful on CPU-only hosts; on a CUDA host the install proceeds (and
    is exercised by the GPU E2E test instead), so skip here.
    """
    if torch.cuda.is_available():
        pytest.skip("CPU fallback assertion only runs on CPU-only hosts")
    from sglang.multimodal_gen.configs.models.dits.omnidreams import OmniDreamsDiTConfig
    from sglang.multimodal_gen.runtime.models.dits.omnidreams import OmniDreamsDiT
    from sglang.multimodal_gen.runtime.models.dits.omnidreams_fp8 import (
        install_fp8_compute_on_dit,
    )

    with torch.device("meta"):
        dit = OmniDreamsDiT(config=OmniDreamsDiTConfig(), hf_config={})
    installed = install_fp8_compute_on_dit(dit)
    assert installed is False, "install must be a no-op on CPU"
    assert not getattr(dit, "_fp8_compute_applied", False)


def test_sage3_self_attn_falls_back_to_sdpa_on_cpu():
    """The sage3 resolver must return None (-> sdpa fallback) on CPU and for
    unsupported head dims, so the DiT self-attn stays correct off-GPU."""
    from sglang.multimodal_gen.runtime.models.dits.omnidreams import (
        _sage3_self_attn,
    )

    q = torch.randn(1, 4, 8, 128)
    k = torch.randn(1, 4, 8, 128)
    v = torch.randn(1, 4, 8, 128)
    # CPU input -> None (caller falls back to F.sdpa).
    assert _sage3_self_attn(q, k, v, "sage3") is None
    # backend="sdpa" -> None regardless.
    assert _sage3_self_attn(q, k, v, "sdpa") is None
    # Unsupported head_dim (33) -> None even if cuda path would otherwise run.
    if not torch.cuda.is_available():
        q2 = torch.randn(1, 4, 8, 33)
        k2 = torch.randn(1, 4, 8, 33)
        v2 = torch.randn(1, 4, 8, 33)
        assert _sage3_self_attn(q2, k2, v2, "sage3") is None


# --------------------------------------------------------------------------- #
# LightVAE mean/inv_std buffers                                                #
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
    """Phase 1+2: native_dit_acceleration accepts disabled/weight_only_fp8/fp8_compute."""
    from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
        OmniDreamsPipelineConfig,
    )

    for mode in ("disabled", "weight_only_fp8", "fp8_compute"):
        cfg = OmniDreamsPipelineConfig(native_dit_acceleration=mode)
        assert cfg.native_dit_acceleration == mode


def test_config_back_compat_aliases_mapped():
    """auto/required are accepted as inert back-compat aliases and mapped:
    auto -> disabled, required -> weight_only_fp8."""
    from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
        OmniDreamsPipelineConfig,
    )

    cfg = OmniDreamsPipelineConfig(native_dit_acceleration="auto")
    assert cfg.native_dit_acceleration == "disabled"
    cfg = OmniDreamsPipelineConfig(native_dit_acceleration="required")
    assert cfg.native_dit_acceleration == "weight_only_fp8"


def test_config_three_state_invalid():
    """Invalid mode raises ValueError."""
    from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
        OmniDreamsPipelineConfig,
    )

    with pytest.raises(ValueError, match="native_acceleration mode must be"):
        OmniDreamsPipelineConfig(native_dit_acceleration="invalid")


def test_config_removed_fields_raise():
    """__post_init__ detects removed fields and raises ValueError."""
    from sglang.multimodal_gen.configs.pipeline_configs.omnidreams import (
        OmniDreamsPipelineConfig,
    )

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
# Config setup() routing (impl selection)                                      #
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
    with patch.object(
        comp, "resolve_wan_vae_path", return_value="/fake/vae"
    ) as rp, patch.object(comp, "load_wan_vae", return_value=MagicMock()) as lw:
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
    with patch.object(comp, "resolve_wan_vae_path") as rp, patch.object(
        comp, "load_wan_vae", return_value=MagicMock()
    ) as lw:
        cfg.setup()
    rp.assert_not_called()
    assert lw.call_args.args[1] == "/explicit/vae"


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
            "native_dit_acceleration": "weight_only_fp8",
            "encoder_config": {"impl": "lightvae", "native_acceleration": "auto"},
            "decoder_config": {"impl": "lighttae"},
        }
    )
    assert cfg.native_dit_acceleration == "weight_only_fp8"
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
