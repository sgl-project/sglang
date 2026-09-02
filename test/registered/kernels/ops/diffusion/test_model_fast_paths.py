"""Per-model fast paths: each model wrapper must reproduce its own reference.

A diffusion kernel is only as good as the wrapper that decides when to use it,
and that decision is model-specific: FLUX.1 and GLM-Image feed different adaLN
layouts to the same LayerNorm+modulate kernel, Sana only engages on non-default
streams, ERNIE runs a bit-exact chain unconditionally.  Kernel-level numerics
live in ``test_norm.py`` / ``test_modulate.py`` / ``test_rope.py``; this file
covers the wiring: right kernel, right reference, gate ends in the right state.

Two assertion styles appear, and the difference is the point:

- ``torch.equal`` for the bit-exact default-on paths.  These self-verify at
  runtime, so a tolerance here would let a real regression through -- the gate
  would silently fall back to eager and the fusion would simply stop running.
- a tolerance for the quality-gated paths, which are *documented* as differing
  from eager at half-precision rounding-order level.

Each section also asserts the gate ended up ``verified`` / not ``disabled``:
without it a test still passes when the fast path never engaged at all.
"""

import sys
import unittest
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.upsampling import Upsample2D

import sglang.kernels.ops.diffusion.sites.hunyuan_qknorm_site as hunyuan_qknorm
import sglang.multimodal_gen.runtime.models.dits.ernie_image as ernie_image
import sglang.multimodal_gen.runtime.models.dits.flux as flux
import sglang.multimodal_gen.runtime.models.dits.flux_2 as flux2
import sglang.multimodal_gen.runtime.models.dits.glm_image as glm_image
import sglang.multimodal_gen.runtime.models.dits.longcat_image as longcat_image
import sglang.multimodal_gen.runtime.models.dits.ltx_2 as ltx2_module
import sglang.multimodal_gen.runtime.models.dits.qwen_image as qwen_image
import sglang.multimodal_gen.runtime.models.dits.sana as sana
from sglang.kernels.ops.diffusion import (
    can_use_fused_layernorm_modulate,
    can_use_fused_qk_head_layernorm,
    can_use_fused_rmsnorm_scale_shift,
    can_use_wan_rmsnorm_silu,
    fused_ltx2_rms_norm_modulate,
    hunyuan_qkv_rope_pack,
    mark_fused_ln_modulate_site,
    mark_hunyuan_qknorm_site,
    mark_ltx2_rms_norm_modulate_site,
    mark_qwen_image_added_qkv_site,
    mount_fused_ln_modulate,
    mount_hunyuan_qknorm,
    mount_ltx2_rms_norm_modulate,
    mount_qwen_image_added_qkv,
    try_flux2_token_cat_nvfp4,
    unmount_hunyuan_qknorm,
    unmount_ltx2_rms_norm_modulate,
    unmount_qwen_image_added_qkv,
    wan_rmsnorm_silu,
)
from sglang.kernels.ops.diffusion.common.platform import is_cuda
from sglang.multimodal_gen.configs.models.vaes.stablediffusion3 import (
    StableDiffusion3VAEConfig,
)
from sglang.multimodal_gen.runtime.layers.layernorm import (
    RMSNorm,
    RMSNormNoWeight,
    apply_qk_norm,
)
from sglang.multimodal_gen.runtime.layers.rotary_embedding.utils import (
    _apply_rotary_emb,
)
from sglang.multimodal_gen.runtime.models.dits.ernie_image import (
    _ernie_gated_norm_scale_shift,
    _ernie_norm_scale_shift,
    _ernie_qknorm_rope,
    _ernie_qknorm_rope_reference,
)
from sglang.multimodal_gen.runtime.models.dits.flux import (
    _flux_fused_ln_modulate,
    _flux_norm_modulate,
)
from sglang.multimodal_gen.runtime.models.dits.flux_2 import (
    _can_use_nvfp4_swiglu_quant_fusion,
    _flux2_norm_modulate,
    _flux2_swiglu,
)
from sglang.multimodal_gen.runtime.models.dits.glm_image import (
    _eager_ln_modulate as _glm_eager_ln_modulate,
)
from sglang.multimodal_gen.runtime.models.dits.glm_image import (
    _glm_ln_modulate,
    _glm_qk_layernorm,
)
from sglang.multimodal_gen.runtime.models.dits.hunyuanvideo import (
    _hunyuan_pack_qkv,
    _hunyuan_qknorm,
)
from sglang.multimodal_gen.runtime.models.dits.longcat_image import (
    _apply_longcat_qknorm_rope,
)
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import _ltx2_rms_norm_modulate
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageCrossAttention,
    QwenImageTransformerBlock,
    _qwen_modulation_cache_key,
    _qwen_norm_out,
    _split_unquantized_merged_linear,
)
from sglang.multimodal_gen.runtime.models.dits.sana import (
    _eager_ln_modulate as _sana_eager_ln_modulate,
)
from sglang.multimodal_gen.runtime.models.dits.sana import (
    sana_ln_modulate,
)
from sglang.multimodal_gen.runtime.models.vaes import flux2_vae_cuda_opt as vae_opt
from sglang.multimodal_gen.runtime.models.vaes.autoencoder import AutoencoderKL
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
from sglang.multimodal_gen.runtime.models.vaes.wan_vae_cuda_opt import (
    FusedWanRMSNormSiLU,
    VaeFastPathGate,
)
from sglang.multimodal_gen.runtime.models.vaes.wanvae import WanRMS_norm
from sglang.multimodal_gen.runtime.platforms.interface import DeviceCapability
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=95, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=8, suite="nightly-amd-kernel-1-gpu", nightly=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# The bit-exact LayerNorm/RMSNorm fusions are NVIDIA inline PTX, so their
# guards reject on ROCm and these sites serve eager there; only the subtests
# asserting a fused outcome are CUDA-only.
requires_inline_ptx = pytest.mark.skipif(
    not is_cuda(), reason="bit-exact norm fusions are NVIDIA PTX"
)


@pytest.fixture(autouse=True)
def _seed_cuda():
    """Every wrapper below asserts against a reference computed from the same
    random draw, so the seed must be fixed per test, not per module."""
    torch.cuda.manual_seed(0)


def test_bitexact_norm_guards_follow_platform():
    # Runs on both lanes, with shapes inside every guard's contract so only the
    # platform decides: engaged on CUDA, rejected on ROCm.  A fatal LLVM error
    # there kills the process, so the sites' own try/except cannot be what
    # catches it -- the guards have to.
    x = torch.randn(1, 256, 4096, device="cuda", dtype=torch.bfloat16)
    row = torch.randn(1, 4096, device="cuda", dtype=torch.bfloat16)
    vec = torch.randn(1, 1, 4096, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(1, 256, 32, 128, device="cuda", dtype=torch.bfloat16)
    assert can_use_fused_layernorm_modulate(x, row, row) is is_cuda()
    assert can_use_fused_qk_head_layernorm(q, q) is is_cuda()
    assert can_use_fused_rmsnorm_scale_shift(x, weight, vec, vec) is is_cuda()


# -------------------------------------------------------------------------
# Qwen-Image -- final LayerNorm + adaLN scale/shift
# -------------------------------------------------------------------------


@requires_inline_ptx
def test_qwen_norm_out_matches_adaln_reference():
    qwen_image._QWEN_NORM_OUT.disabled = False
    qwen_image._QWEN_NORM_OUT.verified = False
    qwen_image._QWEN_NORM_OUT_SIGS.clear()
    torch.manual_seed(0)
    norm_out = (
        qwen_image.AdaLayerNormContinuous(
            3072, 3072, elementwise_affine=False, eps=1e-6
        )
        .cuda()
        .bfloat16()
    )
    hidden_states = torch.randn(1, 257, 3072, device="cuda", dtype=torch.bfloat16)
    conditioning = torch.randn(1, 3072, device="cuda", dtype=torch.bfloat16)

    expected = norm_out(hidden_states, conditioning)
    actual = _qwen_norm_out(norm_out, hidden_states, conditioning)

    assert torch.equal(actual, expected)
    assert qwen_image._QWEN_NORM_OUT.verified
    assert not qwen_image._QWEN_NORM_OUT.disabled


def test_qwen_norm_out_preserves_compile_path(monkeypatch):
    norm_out = (
        qwen_image.AdaLayerNormContinuous(16, 16, elementwise_affine=False, eps=1e-6)
        .cuda()
        .bfloat16()
    )
    hidden_states = torch.randn(1, 3, 16, device="cuda", dtype=torch.bfloat16)
    conditioning = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
    expected = norm_out(hidden_states, conditioning)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        qwen_image,
        "fused_layernorm_modulate_raw",
        lambda *args, **kwargs: pytest.fail("compile path must not dispatch kernel"),
    )

    assert torch.equal(_qwen_norm_out(norm_out, hidden_states, conditioning), expected)


def test_qwen_norm_out_does_not_verify_during_graph_capture(monkeypatch):
    qwen_image._QWEN_NORM_OUT.disabled = False
    qwen_image._QWEN_NORM_OUT.verified = False
    qwen_image._QWEN_NORM_OUT_SIGS.clear()
    norm_out = (
        qwen_image.AdaLayerNormContinuous(
            3072, 3072, elementwise_affine=False, eps=1e-6
        )
        .cuda()
        .bfloat16()
    )
    hidden_states = torch.randn(1, 17, 3072, device="cuda", dtype=torch.bfloat16)
    conditioning = torch.randn(1, 3072, device="cuda", dtype=torch.bfloat16)
    expected = norm_out(hidden_states, conditioning)

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        qwen_image,
        "fused_layernorm_modulate_raw",
        lambda *args, **kwargs: pytest.fail("capture must not verify a new layout"),
    )

    assert torch.equal(_qwen_norm_out(norm_out, hidden_states, conditioning), expected)
    assert not qwen_image._QWEN_NORM_OUT_SIGS


# -------------------------------------------------------------------------
# FLUX.1 -- LayerNorm + adaLN modulate, and the shared-FF GELU site
# -------------------------------------------------------------------------


def _flux_eager(norm, x, scale, shift):
    return norm(x) * (1 + scale[:, None]) + shift[:, None]


def _flux_site_inputs(shape, chunks, seed):
    torch.manual_seed(seed)
    batch, seq, hidden = shape
    norm = torch.nn.LayerNorm(hidden, eps=1e-6, elementwise_affine=False).cuda()
    x = (torch.randn(batch, seq, hidden, device="cuda") * 8).bfloat16()
    emb = torch.randn(batch, chunks * hidden, device="cuda").bfloat16()
    parts = emb.chunk(chunks, dim=1)  # strided adaLN projection views
    return norm, x, parts[0], parts[1]


@requires_inline_ptx
@pytest.mark.parametrize(
    "shape,chunks",
    [
        ((1, 4096, 3072), 6),  # dual-stream image tokens (1024^2), chunk(6)
        ((1, 512, 3072), 6),  # dual-stream text tokens
        ((1, 4608, 3072), 3),  # single-stream concat, chunk(3)
        ((2, 300, 3072), 6),  # CFG batch, odd seq
    ],
)
def test_flux_fused_ln_modulate_is_bit_exact(shape, chunks):
    # Every distinct (shape, stride, eps) signature the FLUX.1 sites emit
    # must verify torch.equal on first sight and stay enabled.
    norm, x, shift, scale = _flux_site_inputs(shape, chunks, seed=0)
    out = _flux_fused_ln_modulate(norm, x, scale, shift)
    assert out is not None
    assert torch.equal(out, _flux_eager(norm, x, scale, shift))
    assert not flux._FLUX_LN_MOD.disabled
    assert flux._FLUX_LN_MOD.verified


@requires_inline_ptx
def test_flux_norm_modulate_bitexact_supersedes_high_fold():
    # With the quality="high" affine fold mounted, the bit-exact kernel
    # still takes priority, so the site output stays lossless.
    site = torch.nn.Module()
    mark_fused_ln_modulate_site(site)
    assert mount_fused_ln_modulate(site)
    norm, x, shift, scale = _flux_site_inputs((1, 128, 3072), 6, seed=1)
    out = _flux_norm_modulate(site, norm, x, scale, shift)
    assert torch.equal(out, _flux_eager(norm, x, scale, shift))


def test_flux_fused_ln_modulate_rejects_unsupported_hidden():
    # hidden % 4 != 0 is outside the kernel contract and must bail out.
    norm, x, shift, scale = _flux_site_inputs((1, 64, 3070), 6, seed=2)
    assert _flux_fused_ln_modulate(norm, x, scale, shift) is None


# -------------------------------------------------------------------------
# FLUX.2 -- packed norm+modulate and packed SwiGLU views
# -------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestFlux2EagerFusions(CustomTestCase):
    def setUp(self):
        flux2._FLUX2_LN_MOD.disabled = False
        flux2._FLUX2_LN_MOD.verified = False
        flux2._FLUX2_LN_MOD_SIGS.clear()
        flux2._FLUX2_SWIGLU.disabled = False
        flux2._FLUX2_SWIGLU.verified = False
        flux2._FLUX2_SWIGLU_SIGS.clear()

    @requires_inline_ptx
    def test_norm_modulate_is_bit_exact_across_sequence_lengths(self):
        torch.manual_seed(0)
        hidden = 256
        norm = torch.nn.LayerNorm(
            hidden, eps=1e-6, elementwise_affine=False, device="cuda"
        )
        # FLUX.2 modulation values are views of one packed projection.
        params = torch.randn(1, 1, 6 * hidden, device="cuda").bfloat16()
        shift, scale = params.chunk(6, dim=-1)[:2]

        for seq in (17, 65):
            x = torch.randn(1, seq, hidden, device="cuda").bfloat16()
            expected = norm(x) * (1 + scale) + shift
            actual = _flux2_norm_modulate(norm, x, scale, shift)
            self.assertTrue(torch.equal(actual, expected))

        self.assertFalse(flux2._FLUX2_LN_MOD.disabled)
        self.assertEqual(len(flux2._FLUX2_LN_MOD_SIGS), 1)

    def test_packed_swiglu_is_bit_exact_for_contiguous_and_strided_views(self):
        torch.manual_seed(1)
        hidden = 384
        inputs = [
            torch.randn(1, 19, 2 * hidden, device="cuda").bfloat16(),
            torch.randn(1, 19, 3 * hidden, device="cuda").bfloat16()[..., : 2 * hidden],
        ]
        for x in inputs:
            expected = F.silu(x[..., :hidden]) * x[..., hidden:]
            actual = _flux2_swiglu(x)
            self.assertTrue(torch.equal(actual, expected))

        self.assertFalse(flux2._FLUX2_SWIGLU.disabled)
        self.assertEqual(len(flux2._FLUX2_SWIGLU_SIGS), 2)

    def test_nvfp4_swiglu_quant_fusion_is_sm103_only(self):
        self.assertFalse(_can_use_nvfp4_swiglu_quant_fusion(DeviceCapability(10, 0)))
        self.assertTrue(_can_use_nvfp4_swiglu_quant_fusion(DeviceCapability(10, 3)))
        self.assertFalse(_can_use_nvfp4_swiglu_quant_fusion(DeviceCapability(12, 0)))

    def test_fp16_preserves_reference_path(self):
        x = torch.randn(1, 17, 512, device="cuda", dtype=torch.float16)
        expected = F.silu(x[..., :256]) * x[..., 256:]
        actual = _flux2_swiglu(x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(flux2._FLUX2_SWIGLU.disabled)

    def test_packed_swiglu_rejects_non_dense_outer_stride(self):
        base = torch.randn(2, 23, 512, device="cuda", dtype=torch.bfloat16)
        x = base[:, :19]
        self.assertNotEqual(x.stride(0), x.shape[1] * x.stride(1))

        expected = F.silu(x[..., :256]) * x[..., 256:]
        actual = _flux2_swiglu(x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(flux2._FLUX2_SWIGLU_SIGS), 0)

    def test_new_swiglu_signature_is_not_verified_during_graph_capture(self):
        first = torch.randn(1, 17, 512, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(
            torch.equal(
                _flux2_swiglu(first),
                F.silu(first[..., :256]) * first[..., 256:],
            )
        )
        self.assertEqual(len(flux2._FLUX2_SWIGLU_SIGS), 1)

        second = torch.randn(1, 19, 768, device="cuda", dtype=torch.bfloat16)
        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            actual = _flux2_swiglu(second)

        expected = F.silu(second[..., :384]) * second[..., 384:]
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(flux2._FLUX2_SWIGLU_SIGS), 1)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
        reason="FLUX.2 token-cat NVFP4 requires Blackwell SM103",
    )
    def test_token_cat_nvfp4_matches_flashinfer(self):
        import flashinfer

        torch.manual_seed(20260830)
        attention = torch.randn(1, 17, 6144, device="cuda", dtype=torch.bfloat16)
        mlp = torch.randn(1, 17, 18432, device="cuda", dtype=torch.bfloat16)
        global_scale = torch.tensor(0.625, device="cuda", dtype=torch.float32)

        expected_fp4, expected_scales = flashinfer.fp4_quantize(
            torch.cat([attention, mlp], dim=-1).view(-1, 24576), global_scale
        )
        actual = try_flux2_token_cat_nvfp4(attention, mlp, global_scale)

        self.assertIsNotNone(actual)
        actual_fp4, actual_scales = actual
        self.assertTrue(torch.equal(actual_fp4, expected_fp4))
        self.assertTrue(
            torch.equal(
                actual_scales.view(torch.uint8), expected_scales.view(torch.uint8)
            )
        )

    def test_token_cat_nvfp4_falls_back_while_compiling(self):
        attention = torch.empty(1, 1, 6144, device="cuda", dtype=torch.bfloat16)
        mlp = torch.empty(1, 1, 18432, device="cuda", dtype=torch.bfloat16)
        global_scale = torch.ones(1, device="cuda", dtype=torch.float32)

        with patch("torch.compiler.is_compiling", return_value=True):
            self.assertIsNone(try_flux2_token_cat_nvfp4(attention, mlp, global_scale))


# -------------------------------------------------------------------------
# Qwen-Image -- reuse timestep-only modulation across serial CFG branches
# -------------------------------------------------------------------------


class _PackedAddedQKV(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.output_partition_sizes = [dim, dim, dim]
        self.quant_config = None
        self.weight = nn.Parameter(
            torch.randn(3 * dim, dim, device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.bias = nn.Parameter(
            torch.randn(3 * dim, device="cuda", dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return F.linear(x, self.weight, self.bias), None


def test_qwen_added_qkv_lossless_uses_three_reference_gemms():
    torch.manual_seed(20260831)
    dim = 64
    x = torch.randn(1, 17, dim, device="cuda", dtype=torch.bfloat16)
    packed = _PackedAddedQKV(dim)

    attention = QwenImageCrossAttention.__new__(QwenImageCrossAttention)
    nn.Module.__init__(attention)
    attention.use_fused_added_qkv = True
    attention._unquantized_added_qkv_is_packed = True
    attention.to_added_qkv = packed
    mark_qwen_image_added_qkv_site(attention)

    expected_lossless = _split_unquantized_merged_linear(packed, x)
    actual_lossless = attention._get_added_qkv_projections(x)
    assert packed.calls == 0
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(actual_lossless, expected_lossless)
    )

    assert mount_qwen_image_added_qkv(attention)
    actual_high = attention._get_added_qkv_projections(x)
    expected_high = tuple(
        tensor.contiguous()
        for tensor in F.linear(x, packed.weight, packed.bias).chunk(3, dim=-1)
    )
    assert packed.calls == 1
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(actual_high, expected_high)
    )

    unmount_qwen_image_added_qkv(attention)
    attention._get_added_qkv_projections(x)
    assert packed.calls == 1


class _CountingProjection(nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.offset = offset
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + self.offset, None


class TestQwenImageModulationCache(CustomTestCase):
    def _block(self):
        block = QwenImageTransformerBlock.__new__(QwenImageTransformerBlock)
        nn.Module.__init__(block)
        block.img_mod = nn.ModuleList([nn.Identity(), _CountingProjection(1.0)])
        block.txt_mod = nn.ModuleList([nn.Identity(), _CountingProjection(2.0)])
        block._modulation_cache = None
        return block

    def _key(self, timestep, hidden, additional_t_cond=None):
        with torch.no_grad():
            return _qwen_modulation_cache_key(
                timestep,
                additional_t_cond,
                hidden,
            )

    def test_matching_cfg_key_reuses_both_modulation_projections(self):
        block = self._block()
        timestep = torch.tensor([500.0], device="cuda")
        hidden = torch.empty(1, 17, 32, device="cuda", dtype=torch.bfloat16)
        img_temb = torch.randn(1, 32, device="cuda", dtype=torch.bfloat16)
        txt_temb = torch.randn_like(img_temb)
        key = self._key(timestep, hidden)

        first = block._get_modulation_params(img_temb, txt_temb, key)
        second = block._get_modulation_params(img_temb, txt_temb, key)

        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertIsNone(block._modulation_cache)
        self.assertEqual(block.img_mod[1].calls, 1)
        self.assertEqual(block.txt_mod[1].calls, 1)

    def test_tensor_identity_version_and_condition_invalidate_cache(self):
        block = self._block()
        timestep = torch.tensor([500.0], device="cuda")
        hidden = torch.empty(1, 17, 32, device="cuda", dtype=torch.bfloat16)
        temb = torch.randn(1, 32, device="cuda", dtype=torch.bfloat16)
        key = self._key(timestep, hidden)
        block._get_modulation_params(temb, temb, key)

        with torch.no_grad():
            timestep.add_(1)
        mutated = self._key(timestep, hidden)
        block._get_modulation_params(temb, temb, mutated)
        self.assertEqual(block.img_mod[1].calls, 2)

        same_value_new_tensor = self._key(timestep.clone(), hidden)
        block._get_modulation_params(temb, temb, same_value_new_tensor)
        self.assertEqual(block.img_mod[1].calls, 3)

        condition = torch.tensor([1], device="cuda")
        conditioned = self._key(timestep, hidden, condition)
        block._get_modulation_params(temb, temb, conditioned)
        self.assertEqual(block.img_mod[1].calls, 4)

    def test_grad_enabled_path_disables_and_clears_cache(self):
        block = self._block()
        timestep = torch.tensor([500.0], device="cuda")
        hidden = torch.empty(1, 17, 32, device="cuda", dtype=torch.bfloat16)
        temb = torch.randn(1, 32, device="cuda", dtype=torch.bfloat16)
        key = self._key(timestep, hidden)
        block._get_modulation_params(temb, temb, key)

        self.assertIsNone(_qwen_modulation_cache_key(timestep, None, hidden))
        block._get_modulation_params(temb, temb, None)

        self.assertIsNone(block._modulation_cache)
        self.assertEqual(block.img_mod[1].calls, 2)

    def test_inference_tensors_cache_and_graph_path_falls_back(self):
        block = self._block()
        with torch.inference_mode():
            timestep = torch.tensor([500.0], device="cuda")
            hidden = torch.empty(1, 17, 32, device="cuda", dtype=torch.bfloat16)
            temb = torch.randn(1, 32, device="cuda", dtype=torch.bfloat16)
            key = _qwen_modulation_cache_key(timestep, None, hidden)

            first = block._get_modulation_params(temb, temb, key)
            second = block._get_modulation_params(temb, temb, key)

        self.assertIs(first[0], second[0])
        self.assertEqual(block.img_mod[1].calls, 1)

        with (
            patch(
                "sglang.multimodal_gen.runtime.models.dits.qwen_image.is_in_breakable_cuda_graph",
                return_value=True,
            ),
            torch.no_grad(),
        ):
            self.assertIsNone(_qwen_modulation_cache_key(timestep, None, hidden))


# -------------------------------------------------------------------------
# GLM-Image -- LayerNorm + modulate and per-head qk LayerNorm
# -------------------------------------------------------------------------


@requires_inline_ptx
@pytest.mark.parametrize("shape", [(1, 4096, 4096), (2, 301, 4096), (1, 1, 2560)])
def test_glm_ln_modulate_is_bit_exact(shape):
    # (1, 4096, 4096) is the real GLM-Image image-stream shape (1024^2,
    # hidden 4096); the others cover the text stream and another hidden.
    torch.manual_seed(0)
    batch, seq, hidden = shape
    norm = torch.nn.LayerNorm(hidden, eps=1e-5, elementwise_affine=False).cuda()
    x = (torch.randn(batch, seq, hidden, device="cuda") * 8).bfloat16()
    emb = torch.randn(batch, 12 * hidden, device="cuda").bfloat16()
    chunks = emb.chunk(12, dim=1)  # strided adaLN projection views
    shift, scale = chunks[0], chunks[2]
    out = _glm_ln_modulate(norm, x, scale, shift, x.dtype)
    assert torch.equal(out, _glm_eager_ln_modulate(norm, x, scale, shift, x.dtype))
    assert glm_image._GLM_LN_MOD.verified
    assert not glm_image._GLM_LN_MOD.disabled


@requires_inline_ptx
@pytest.mark.parametrize("shape", [(1, 4360, 32, 128), (2, 37, 3, 40), (1, 129, 5, 64)])
def test_glm_qk_head_layernorm_is_bit_exact(shape):
    # (1, 4360, 32, 128) is the real GLM-Image q/k shape (text + image
    # tokens, 32 heads of dim 128); the others cover partially-filled warps.
    torch.manual_seed(1)
    batch, seq, heads, head_dim = shape
    norm_q = torch.nn.LayerNorm(head_dim, eps=1e-5, elementwise_affine=False).cuda()
    norm_k = torch.nn.LayerNorm(head_dim, eps=1e-5, elementwise_affine=False).cuda()
    q = (torch.randn(batch, seq, heads, head_dim, device="cuda") * 5).bfloat16()
    k = (torch.randn(batch, seq, heads, head_dim, device="cuda") * 5).bfloat16()
    q_out, k_out = _glm_qk_layernorm(norm_q, norm_k, q, k, q.dtype)
    assert torch.equal(q_out, norm_q(q).to(q.dtype))
    assert torch.equal(k_out, norm_k(k).to(k.dtype))
    assert glm_image._GLM_QK_LN.verified
    assert not glm_image._GLM_QK_LN.disabled


# -------------------------------------------------------------------------
# Sana -- stream-conditional LayerNorm + modulate
# -------------------------------------------------------------------------


@requires_inline_ptx
@pytest.mark.parametrize(
    "shape,nmod,transposed",
    [
        ((2, 1024, 2240), 6, False),
        ((2, 1024, 2240), 2, False),
        ((1, 1024, 2240), 6, True),
        ((1, 37, 2240), 6, False),
    ],
)
def test_sana_fused_ln_modulate_is_bit_exact(shape, nmod, transposed):
    # (., 1024, 2240) is the real Sana 1024px shape; hidden 2240 % 512 != 0
    # exercises the kernel's partial tail chunk.  nmod mirrors the two adaLN
    # chunk layouts, transposed the permuted layout the Sana DiT serves.
    torch.manual_seed(0)
    batch, seq, hidden = shape
    norm = torch.nn.LayerNorm(hidden, eps=1e-6, elementwise_affine=False).cuda()
    x = (torch.randn(batch, seq, hidden, device="cuda") * 4).bfloat16()
    if transposed:
        x = x.permute(0, 2, 1).contiguous().permute(0, 2, 1)
    emb = torch.randn(batch, nmod, hidden, device="cuda").bfloat16()
    shift, scale = emb.chunk(nmod, dim=1)[0], emb.chunk(nmod, dim=1)[-1]
    # default-stream eager serving must stay on the untouched eager chain
    n_sigs = len(sana._SANA_LN_MOD.verified_sigs)
    sana_ln_modulate(norm, x, scale, shift)
    assert len(sana._SANA_LN_MOD.verified_sigs) == n_sigs
    # The fusion engages on non-default streams (the BCG warmup/capture path).
    # x/scale/shift were filled on the default stream, so the side stream must
    # wait for that work before reading them -- without this the fused kernel
    # can read a half-written tensor, the first-sight torch.equal check fails,
    # and the gate disables itself *permanently*, which then breaks every later
    # parametrization too. It only loses the race when the GPU is contended,
    # which is why it shows up on shared CI runners and not on an idle box.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        out = sana_ln_modulate(norm, x, scale, shift)
        assert len(sana._SANA_LN_MOD.verified_sigs) == n_sigs + 1  # verified
        out2 = sana_ln_modulate(norm, x, scale, shift)  # verified-sig lane
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    assert torch.equal(out, _sana_eager_ln_modulate(norm, x, scale, shift))
    assert torch.equal(out2, out) and not sana._SANA_LN_MOD.disabled


# -------------------------------------------------------------------------
# ERNIE-Image -- bit-exact RMSNorm scale/shift and rotate-half RoPE
# -------------------------------------------------------------------------


@requires_inline_ptx
@pytest.mark.parametrize("shape", [(1, 4216, 4096), (2, 1140, 4096), (1, 128, 2048)])
def test_ernie_norm_scale_shift_is_bit_exact(shape):
    # (1, 4216, 4096) is the real ERNIE-Image shape (1024^2 image + text
    # tokens, hidden 4096); 2048 covers the threads_per_row=32 regime.
    torch.manual_seed(0)
    batch, seq, hidden = shape
    norm = RMSNorm(hidden, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden))
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    update = torch.randn_like(x)
    scale = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    shift = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    gate = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out = _ernie_norm_scale_shift(norm, x, scale, shift)
        ref = norm(x) * (1 + scale) + shift
        assert torch.equal(out, ref)

        out2, res = _ernie_gated_norm_scale_shift(
            norm, residual, update, gate, scale, shift
        )
        res_ref = residual + gate * update
        ref2 = norm(res_ref) * (1 + scale) + shift
        assert torch.equal(res, res_ref)
        assert torch.equal(out2, ref2)

    # the fast paths must actually be in use (not silently disabled)
    assert ernie_image._ERNIE_NORM.verified
    assert ernie_image._ERNIE_GATED_NORM.verified
    assert not ernie_image._ERNIE_NORM.disabled
    assert not ernie_image._ERNIE_GATED_NORM.disabled


def test_ernie_qknorm_rope_is_bit_exact():
    torch.manual_seed(1)
    ernie_image._ERNIE_QKNORM_ROPE.disabled = False
    ernie_image._ERNIE_QKNORM_ROPE.verified = False
    batch, seq, heads, head_dim = 1, 257, 32, 128
    q = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    q_norm = RMSNorm(head_dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    k_norm = RMSNorm(head_dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(seq, head_dim, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    cache = torch.cat((cos, sin), dim=-1).contiguous()
    positions = torch.arange(seq, device="cuda", dtype=torch.long)

    q_ref, k_ref = _ernie_qknorm_rope_reference(
        q.clone(), k.clone(), q_norm, k_norm, head_dim, cos, sin
    )
    q_out, k_out = _ernie_qknorm_rope(
        q,
        k,
        q_norm,
        k_norm,
        head_dim,
        cos,
        sin,
        cache,
        positions,
    )

    assert torch.equal(q_out, q_ref)
    assert torch.equal(k_out, k_ref)
    assert ernie_image._ERNIE_QKNORM_ROPE.verified
    assert not ernie_image._ERNIE_QKNORM_ROPE.disabled


def test_ernie_qknorm_rope_first_attempt_exception_uses_pristine_inputs():
    torch.manual_seed(2)
    ernie_image._ERNIE_QKNORM_ROPE.disabled = False
    ernie_image._ERNIE_QKNORM_ROPE.verified = False
    batch, seq, heads, head_dim = 1, 17, 4, 128
    q = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    q_norm = RMSNorm(head_dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    k_norm = RMSNorm(head_dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(seq, head_dim, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    cache = torch.cat((cos, sin), dim=-1).contiguous()
    positions = torch.arange(seq, device="cuda", dtype=torch.long)
    q_ref, k_ref = _ernie_qknorm_rope_reference(
        q.clone(), k.clone(), q_norm, k_norm, head_dim, cos, sin
    )

    def mutate_then_raise(**kwargs):
        kwargs["q"].zero_()
        kwargs["k"].zero_()
        raise RuntimeError("synthetic kernel failure")

    with patch.object(ernie_image, "apply_qk_norm_rope", mutate_then_raise):
        q_out, k_out = _ernie_qknorm_rope(
            q,
            k,
            q_norm,
            k_norm,
            head_dim,
            cos,
            sin,
            cache,
            positions,
        )

    assert torch.equal(q_out, q_ref)
    assert torch.equal(k_out, k_ref)
    assert ernie_image._ERNIE_QKNORM_ROPE.disabled


# -------------------------------------------------------------------------
# LongCat-Image -- full-width interleaved QKNorm + RoPE
# -------------------------------------------------------------------------


@requires_inline_ptx
def test_longcat_qknorm_rope_is_bit_exact():
    torch.manual_seed(3)
    batch, seq, heads, head_dim = 2, 17, 24, 128
    offset = 11
    q = torch.randn(batch, seq, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    q_norm = RMSNorm(head_dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    k_norm = RMSNorm(head_dim, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        q_norm.weight.copy_(torch.randn_like(q_norm.weight))
        k_norm.weight.copy_(torch.randn_like(k_norm.weight))

    cos = torch.randn(offset + seq, head_dim, device="cuda")
    sin = torch.randn_like(cos)
    image_rotary_emb = (cos[offset:], sin[offset:])
    cache = torch.cat((cos, sin), dim=-1).contiguous()
    positions = torch.arange(offset, offset + seq, device="cuda", dtype=torch.int64)

    q_ref, k_ref = apply_qk_norm(q.clone(), k.clone(), q_norm, k_norm, head_dim)
    q_ref = longcat_image.apply_rotary_emb(q_ref, image_rotary_emb, sequence_dim=1)
    k_ref = longcat_image.apply_rotary_emb(k_ref, image_rotary_emb, sequence_dim=1)

    q_fused, k_fused = q.clone(), k.clone()
    q_out, k_out = _apply_longcat_qknorm_rope(
        q_fused,
        k_fused,
        q_norm,
        k_norm,
        head_dim,
        image_rotary_emb,
        cache,
        positions,
    )

    assert q_out.data_ptr() == q_fused.data_ptr()
    assert k_out.data_ptr() == k_fused.data_ptr()
    assert torch.equal(q_out, q_ref)
    assert torch.equal(k_out, k_ref)
    assert longcat_image._LONGCAT_QKNORM_ROPE.verified
    assert not longcat_image._LONGCAT_QKNORM_ROPE.disabled


# -------------------------------------------------------------------------
# LTX-2 -- weightless RMSNorm + modulate (quality-gated)
# -------------------------------------------------------------------------


def _ltx2_eager(rms, x, scale, shift, eps):
    return rms(x, eps) * (1 + scale) + shift


def _ltx2_inputs(hidden, batch=1, seq=4096):
    rms = RMSNormNoWeight()
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.bfloat16)
    scale = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    shift = torch.randn(batch, 1, hidden, device="cuda", dtype=torch.bfloat16) * 0.1
    return rms, x, scale, shift


# hidden 4096 = LTX-2 video stream, 2048 = audio stream.
@pytest.mark.parametrize("hidden", [4096, 2048])
def test_ltx2_lossless_default_is_bitexact(hidden):
    # A marked-but-unmounted site uses only the self-verified bit-exact
    # modulate fast path after the reference aten RMSNorm.
    block = nn.Module()
    mark_ltx2_rms_norm_modulate_site(block)
    rms, x, scale, shift = _ltx2_inputs(hidden)
    out = _ltx2_rms_norm_modulate(block, rms, x, scale, shift, 1e-6)
    assert torch.equal(out, _ltx2_eager(rms, x, scale, shift, 1e-6))


def test_ltx2_lossless_compile_keeps_expression_visible_to_inductor(monkeypatch):
    block = nn.Module()
    mark_ltx2_rms_norm_modulate_site(block)
    rms, x, scale, shift = _ltx2_inputs(2048, seq=126)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(
        ltx2_module,
        "_ltx2_modulate",
        lambda *_args: pytest.fail("compiled path must not call the opaque custom op"),
    )

    out = _ltx2_rms_norm_modulate(block, rms, x, scale, shift, 1e-6)
    assert torch.equal(out, _ltx2_eager(rms, x, scale, shift, 1e-6))


@requires_inline_ptx
@pytest.mark.parametrize("hidden", [4096, 2048])
def test_ltx2_mounted_high_uses_fused_kernel(hidden):
    block = nn.Module()
    mark_ltx2_rms_norm_modulate_site(block)
    assert mount_ltx2_rms_norm_modulate(block)
    try:
        rms, x, scale, shift = _ltx2_inputs(hidden)
        out = _ltx2_rms_norm_modulate(block, rms, x, scale, shift, 1e-6)
        # The mounted path routes through the fused kernel exactly.
        assert torch.equal(out, fused_ltx2_rms_norm_modulate(x, scale, shift, 1e-6))
        # And stays within half-precision rounding of the eager reference.
        ref = _ltx2_eager(rms, x, scale, shift, 1e-6)
        assert torch.allclose(out.float(), ref.float(), atol=3e-2, rtol=1e-2)
    finally:
        unmount_ltx2_rms_norm_modulate(block)


# -------------------------------------------------------------------------
# HunyuanVideo -- QKV/RoPE pack and quality-gated QK RMSNorm
# -------------------------------------------------------------------------


@pytest.mark.parametrize("img_tokens,txt_tokens", [(257, 31), (4096, 256)])
def test_hunyuan_qkv_rope_pack_is_bit_exact(img_tokens, txt_tokens):
    torch.manual_seed(0)
    shape_img = (1, img_tokens, 24, 128)
    shape_txt = (1, txt_tokens, 24, 128)
    img_q, img_k, img_v = (
        torch.randn(shape_img, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    txt_q, txt_k, txt_v = (
        torch.randn(shape_txt, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    cos = torch.randn(img_tokens, 64, device="cuda")
    sin = torch.randn_like(cos)

    q, k, v = _hunyuan_pack_qkv(img_q, img_k, img_v, txt_q, txt_k, txt_v, cos, sin)
    q_ref = torch.cat(
        (_apply_rotary_emb(img_q, cos, sin, is_neox_style=False), txt_q), dim=1
    )
    k_ref = torch.cat(
        (_apply_rotary_emb(img_k, cos, sin, is_neox_style=False), txt_k), dim=1
    )
    v_ref = torch.cat((img_v, txt_v), dim=1)

    assert torch.equal(q, q_ref)
    assert torch.equal(k, k_ref)
    assert torch.equal(v, v_ref)


def test_hunyuan_qkv_rope_pack_uses_int64_row_offsets():
    if torch.cuda.get_device_properties(0).total_memory < 16 * 2**30:
        pytest.skip("needs >= 16 GB GPU memory")

    img_tokens, txt_tokens = 115200, 8
    num_heads, head_dim = 24, 128
    total_tokens = img_tokens + txt_tokens
    projection_width = 21504

    projection = torch.zeros(
        (1, total_tokens, projection_width),
        device="cuda",
        dtype=torch.bfloat16,
    )
    qkv = projection[..., : 3 * num_heads * head_dim].view(
        1, total_tokens, 3, num_heads, head_dim
    )
    q = qkv[:, :, 0].contiguous()
    k = qkv[:, :, 1].contiguous()
    v = qkv[:, :, 2]
    assert (img_tokens - 1) * v.stride(1) > torch.iinfo(torch.int32).max

    cos = torch.ones((img_tokens, head_dim // 2), device="cuda")
    sin = torch.zeros_like(cos)
    packed = hunyuan_qkv_rope_pack(
        q[:, :img_tokens],
        k[:, :img_tokens],
        v[:, :img_tokens],
        q[:, img_tokens:],
        k[:, img_tokens:],
        v[:, img_tokens:],
        cos,
        sin,
    )
    torch.cuda.synchronize()

    expected_shape = (1, total_tokens, num_heads, head_dim)
    assert all(x.shape == expected_shape for x in packed)
    assert all(x[0, img_tokens - 1, -1, -1].item() == 0 for x in packed)


def test_hunyuan_quality_qknorm_matches_rmsnorm():
    torch.manual_seed(1)
    site = torch.nn.Module()
    mark_hunyuan_qknorm_site(site)
    q_norm = RMSNorm(128, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    k_norm = RMSNorm(128, eps=1e-6).to(device="cuda", dtype=torch.bfloat16)
    packed = torch.randn(1, 257, 3, 24, 128, device="cuda", dtype=torch.bfloat16)
    q, k = packed[:, :, 0], packed[:, :, 1]
    q_ref = q_norm(q.contiguous()).to(q)
    k_ref = k_norm(k.contiguous()).to(k)

    q_unmounted, k_unmounted = _hunyuan_qknorm(site, q, k, q_norm, k_norm)
    assert torch.equal(q_unmounted, q_ref)
    assert torch.equal(k_unmounted, k_ref)

    assert mount_hunyuan_qknorm(site)
    q_out, k_out = _hunyuan_qknorm(site, q, k, q_norm, k_norm)
    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_out, k_ref, atol=2e-2, rtol=2e-2)

    unmount_hunyuan_qknorm(site)
    q_unmounted, k_unmounted = _hunyuan_qknorm(site, q, k, q_norm, k_norm)
    assert torch.equal(q_unmounted, q_ref)
    assert torch.equal(k_unmounted, k_ref)


def test_hunyuan_quality_qknorm_stays_unmounted_without_cute_kernel():
    site = torch.nn.Module()
    mark_hunyuan_qknorm_site(site)

    with patch.object(hunyuan_qknorm, "_get_qk_rmsnorm_cute", return_value=None):
        assert not mount_hunyuan_qknorm(site)

    assert not hunyuan_qknorm._FUSION.is_enabled(site)


# -------------------------------------------------------------------------
# Wan VAE -- fused RMSNorm+SiLU module gate
# -------------------------------------------------------------------------


def _wan_cl3d(shape, dtype):
    return torch.randn(shape, device="cuda", dtype=dtype).contiguous(
        memory_format=torch.channels_last_3d
    )


@torch.no_grad()
def test_wan_vae_gate_dispatch() -> None:
    # Gate off must stay bit-exact; gate on must route to the fused kernel.
    torch.cuda.manual_seed(0)
    norm = WanRMS_norm(96, images=False).to(device="cuda", dtype=torch.bfloat16)
    norm.gamma.add_(torch.randn_like(norm.gamma))
    gate = VaeFastPathGate()
    fused = FusedWanRMSNormSiLU(norm, gate)
    # Parameter names must not change (weight transfer matches by name).
    assert [n for n, _ in fused.named_parameters()] == ["gamma"]
    x = _wan_cl3d((1, 96, 3, 10, 14), torch.bfloat16)
    assert torch.equal(fused(x), nn.SiLU()(norm(x)))
    gate.enabled = True
    expected = wan_rmsnorm_silu(x, norm.gamma, rms_scale=float(norm.scale))
    assert torch.equal(fused(x), expected)


@torch.no_grad()
def test_wan_vae_rejects_empty_input() -> None:
    x = torch.empty(1, 96, 0, 2, 2, device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last_3d
    )
    gamma = torch.ones(96, 1, 1, 1, device="cuda", dtype=torch.bfloat16)
    assert not can_use_wan_rmsnorm_silu(x, gamma, None)


# -------------------------------------------------------------------------
# FLUX.2 VAE -- fused GroupNorm+SiLU and folded 2x upsample conv
# -------------------------------------------------------------------------


@torch.no_grad()
def test_flux2_vae_fast_path():
    torch.manual_seed(0)
    gate = vae_opt.VaeFastPathGate()
    gn = nn.GroupNorm(32, 128, eps=1e-6).to("cuda", torch.bfloat16)
    x = torch.randn(1, 128, 64, 64, device="cuda", dtype=torch.bfloat16).to(
        memory_format=torch.channels_last
    )
    ref = F.silu(gn(x))
    fused_gn = vae_opt.FusedGroupNormSiLU(gn, gate)
    assert set(fused_gn.state_dict()) == {"weight", "bias"}
    assert torch.equal(fused_gn(x), ref)  # gate off: bit-exact reference

    gate.enabled = True
    fast = fused_gn(x)
    assert fast.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(fast.float(), ref.float(), atol=0.06, rtol=0)

    gate.enabled = False
    up = Upsample2D(channels=32, use_conv=True).to("cuda", torch.bfloat16)
    fused_up = vae_opt.FusedUpsample2xConv2d(up, gate)
    assert set(fused_up.state_dict()) == {"conv.weight", "conv.bias"}
    x = torch.randn(2, 32, 33, 29, device="cuda", dtype=torch.bfloat16)
    ref = up(x)
    assert torch.equal(fused_up(x), ref)
    assert fused_up._fused_weight is None

    gate.enabled = True
    fast = fused_up(x)
    assert fused_up._fused_weight is not None
    ref_range = ref.float().max() - ref.float().min()
    relative_mse = F.mse_loss(fast.float(), ref.float()) / ref_range.square()
    assert relative_mse < 3.2e-5


# ---------------------------------------------------------------------------
# AutoencoderKL (generic) -- fast-path install must not disturb the checkpoint
# ---------------------------------------------------------------------------


def _small_config():
    config = StableDiffusion3VAEConfig()
    config.arch_config.latent_channels = 2
    config.arch_config.block_out_channels = (4, 4)
    config.arch_config.down_block_types = ("DownEncoderBlock2D",) * 2
    config.arch_config.up_block_types = ("UpDecoderBlock2D",) * 2
    config.arch_config.layers_per_block = 1
    config.arch_config.norm_num_groups = 1
    config.arch_config.sample_size = 8
    return config


@torch.no_grad()
def test_autoencoder_kl_fastpath_install():
    torch.manual_seed(0)
    vae = AutoencoderKL(_small_config()).to("cuda", torch.bfloat16).eval()
    ref_names = {n for n, _ in vae.named_parameters()}
    ref_sd = {k: v.clone() for k, v in vae.state_dict().items()}
    z = torch.randn(1, 2, 8, 8, device="cuda", dtype=torch.bfloat16)
    ref = vae.decode(z)

    opt = vae_opt.maybe_optimize_autoencoder_kl(vae)
    # Wrappers must not change parameter FQNs; strict load must round-trip.
    assert {n for n, _ in opt.named_parameters()} == ref_names
    opt.load_state_dict(ref_sd, strict=True)
    # Gate off: bit-for-bit the original path.
    assert torch.equal(opt.decode(z), ref)
    # use_vae_fast_path() is a no-op when nothing registered a gate, so check
    # the wrappers went in before relying on it to switch paths.
    assert any(
        isinstance(m, (vae_opt.FusedGroupNormSiLU, vae_opt.FusedUpsample2xConv2d))
        for m in opt.modules()
    )
    # Gate on: fast path runs and stays close; leaving the scope restores exact.
    with use_vae_fast_path(opt, True):
        torch.testing.assert_close(opt.decode(z).float(), ref.float(), atol=0.1, rtol=0)
    assert torch.equal(opt.decode(z), ref)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
