"""Checkpoint-driven mHC validation for GLM-5.3-Flash.

This test loads only the 270 mHC tensors, never the model. Set:

  GLM53_FLASH_FP8_PATH=/path/to/zai-org/GLM-5.3-Flash
  GLM53_FLASH_MXFP4_PATH=/path/to/GLM-5.3-Flash-Quark-MXFP4-AttnFP8
"""

import json
import os
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open
from sglang.kernels.ops.layernorm import mhc
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.test_utils import CustomTestCase


class TestGLM53FlashMHCCheckpoint(CustomTestCase):
    num_layers = 45
    hidden_size = 4096
    hc_mult = 4
    mix_size = 24
    rms_eps = 1e-6
    hc_eps = 1e-6
    sinkhorn_iters = 20

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        source = os.environ.get("GLM53_FLASH_FP8_PATH")
        quantized = os.environ.get("GLM53_FLASH_MXFP4_PATH")
        if not source or not quantized:
            raise unittest.SkipTest(
                "set GLM53_FLASH_FP8_PATH and GLM53_FLASH_MXFP4_PATH"
            )
        cls.source = Path(source)
        cls.quantized = Path(quantized)
        cls.source_map = cls._load_weight_map(cls.source)
        cls.quantized_map = cls._load_weight_map(cls.quantized)

    @staticmethod
    def _load_weight_map(root):
        index_path = root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise RuntimeError(f"missing checkpoint index: {index_path}")
        return json.loads(index_path.read_text())["weight_map"]

    @classmethod
    def _mhc_names(cls, weight_map):
        return {
            name for name in weight_map if ".hc_attn_" in name or ".hc_ffn_" in name
        }

    @classmethod
    def _load_tensor(cls, root, weight_map, name, device="cpu"):
        with safe_open(root / weight_map[name], framework="pt", device=device) as shard:
            return shard.get_tensor(name)

    @classmethod
    def _load_triplet(cls, layer, stage, root=None, weight_map=None):
        root = cls.quantized if root is None else root
        weight_map = cls.quantized_map if weight_map is None else weight_map
        prefix = f"model.language_model.layers.{layer}.hc_{stage}_"
        base = cls._load_tensor(root, weight_map, prefix + "base")
        fn = cls._load_tensor(root, weight_map, prefix + "fn")
        scale = cls._load_tensor(root, weight_map, prefix + "scale")

        # The checkpoint stores fn in BF16. Glm5NextDecoderLayer declares the
        # parameter as FP32, so default_weight_loader promotes it during copy.
        return (
            fn.to(device="cuda", dtype=torch.float32),
            scale.to(device="cuda", dtype=torch.float32),
            base.to(device="cuda", dtype=torch.float32),
        )

    def setUp(self):
        mhc._AITER_MHC_RUNTIME_DISABLED = False
        mhc._AITER_MHC_IMPORT_WARNED = False
        mhc._AITER_MHC_ACTIVE_LOGGED = False

    def test_checkpoint_mhc_tensors_are_identical_and_complete(self):
        source_names = self._mhc_names(self.source_map)
        quantized_names = self._mhc_names(self.quantized_map)
        self.assertEqual(len(source_names), self.num_layers * 6)
        self.assertEqual(source_names, quantized_names)

        for name in sorted(source_names):
            with self.subTest(name=name):
                source = self._load_tensor(self.source, self.source_map, name)
                quantized = self._load_tensor(self.quantized, self.quantized_map, name)
                self.assertEqual(source.dtype, quantized.dtype)
                self.assertEqual(source.shape, quantized.shape)
                self.assertTrue(torch.equal(source, quantized))

                if name.endswith("_fn"):
                    self.assertEqual(source.dtype, torch.bfloat16)
                    self.assertEqual(
                        tuple(source.shape),
                        (self.mix_size, self.hc_mult * self.hidden_size),
                    )
                elif name.endswith("_base"):
                    self.assertEqual(source.dtype, torch.float32)
                    self.assertEqual(tuple(source.shape), (self.mix_size,))
                else:
                    self.assertTrue(name.endswith("_scale"))
                    self.assertEqual(source.dtype, torch.float32)
                    self.assertEqual(tuple(source.shape), (3,))
                self.assertTrue(torch.isfinite(source).all())

    def _residual(self, tokens, seed):
        torch.manual_seed(seed)
        return (
            torch.randn(
                tokens,
                self.hc_mult,
                self.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
            )
            * 0.1
        )

    def _torch_pre(self, residual, fn, scale, base):
        return mhc._mhc_pre_torch(
            residual,
            fn,
            scale,
            base,
            self.rms_eps,
            self.hc_eps,
            self.hc_eps,
            2.0,
            self.sinkhorn_iters,
        )

    def _aiter_pre(self, residual, fn, scale, base, norm_weight=None):
        result = mhc._try_aiter_mhc_pre(
            residual,
            fn,
            scale,
            base,
            self.rms_eps,
            self.hc_eps,
            self.hc_eps,
            2.0,
            self.sinkhorn_iters,
            norm_weight,
            self.rms_eps if norm_weight is not None else None,
        )
        self.assertIsNotNone(result, "AITER mHC pre unexpectedly fell back")
        return result

    def _assert_pre_close(self, actual, reference):
        post, comb, layer_input = actual
        post_ref, comb_ref, layer_ref = reference
        torch.testing.assert_close(post, post_ref, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(comb, comb_ref, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(layer_input, layer_ref, atol=2e-2, rtol=2e-2)
        for tensor in actual:
            self.assertTrue(torch.isfinite(tensor).all())

    @unittest.skipUnless(
        torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
        "requires one gfx950 GPU",
    )
    def test_every_real_layer_matches_torch_oracle(self):
        token_counts = (1, 8, 17, 32, 64, 128)
        with (
            patch.dict(os.environ, {"SGLANG_USE_AITER": "1"}),
            patch.object(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext()),
            patch.object(mhc, "is_allocation_symmetric", return_value=False),
            patch.object(mhc, "get_tp_group", return_value=None),
        ):
            for layer in range(self.num_layers):
                for stage_index, stage in enumerate(("attn", "ffn")):
                    tokens = token_counts[(2 * layer + stage_index) % len(token_counts)]
                    with self.subTest(layer=layer, stage=stage, tokens=tokens):
                        fn, scale, base = self._load_triplet(layer, stage)
                        residual = self._residual(tokens, seed=2 * layer + stage_index)
                        reference = self._torch_pre(residual, fn, scale, base)
                        actual = self._aiter_pre(residual, fn, scale, base)
                        self._assert_pre_close(actual, reference)

                        x = (reference[2].float() * 0.75 + 0.125).bfloat16()
                        post_ref = mhc._mhc_post_torch(
                            x, residual, reference[0], reference[1]
                        )
                        post_actual = mhc._try_aiter_mhc_post(
                            x, residual, actual[0], actual[1]
                        )
                        self.assertIsNotNone(post_actual)
                        torch.testing.assert_close(
                            post_actual,
                            post_ref,
                            atol=2e-2,
                            rtol=2e-2,
                        )
                        self.assertTrue(torch.isfinite(post_actual).all())

    @unittest.skipUnless(
        torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
        "requires one gfx950 GPU",
    )
    def test_real_parameter_attention_to_ffn_flow_all_layers(self):
        with (
            patch.dict(os.environ, {"SGLANG_USE_AITER": "1"}),
            patch.object(mhc, "use_symmetric_memory", lambda *a, **kw: nullcontext()),
            patch.object(mhc, "is_allocation_symmetric", return_value=False),
            patch.object(mhc, "get_tp_group", return_value=None),
        ):
            for layer in range(self.num_layers):
                with self.subTest(layer=layer):
                    attn_fn, attn_scale, attn_base = self._load_triplet(layer, "attn")
                    ffn_fn, ffn_scale, ffn_base = self._load_triplet(layer, "ffn")
                    residual = self._residual(4, seed=layer)

                    attn_ref = self._torch_pre(residual, attn_fn, attn_scale, attn_base)
                    attn_actual = self._aiter_pre(
                        residual, attn_fn, attn_scale, attn_base
                    )
                    self._assert_pre_close(attn_actual, attn_ref)
                    fake_attn = (attn_ref[2].float() * 0.75 + 0.125).bfloat16()
                    after_attn_ref = mhc._mhc_post_torch(
                        fake_attn, residual, attn_ref[0], attn_ref[1]
                    )
                    after_attn_actual = mhc._try_aiter_mhc_post(
                        fake_attn,
                        residual,
                        attn_actual[0],
                        attn_actual[1],
                    )
                    self.assertIsNotNone(after_attn_actual)
                    torch.testing.assert_close(
                        after_attn_actual,
                        after_attn_ref,
                        atol=2e-2,
                        rtol=2e-2,
                    )

                    ffn_ref = self._torch_pre(
                        after_attn_ref, ffn_fn, ffn_scale, ffn_base
                    )
                    ffn_actual = self._aiter_pre(
                        after_attn_actual, ffn_fn, ffn_scale, ffn_base
                    )
                    self._assert_pre_close(ffn_actual, ffn_ref)
                    fake_ffn = (ffn_ref[2].float() * 0.5 - 0.0625).bfloat16()
                    after_ffn_ref = mhc._mhc_post_torch(
                        fake_ffn, after_attn_ref, ffn_ref[0], ffn_ref[1]
                    )
                    after_ffn_actual = mhc._try_aiter_mhc_post(
                        fake_ffn,
                        after_attn_actual,
                        ffn_actual[0],
                        ffn_actual[1],
                    )
                    self.assertIsNotNone(after_ffn_actual)
                    torch.testing.assert_close(
                        after_ffn_actual,
                        after_ffn_ref,
                        atol=4e-2,
                        rtol=4e-2,
                    )
                    self.assertTrue(torch.isfinite(after_ffn_actual).all())


if __name__ == "__main__":
    unittest.main()
