import itertools
import unittest

import torch

from sglang.srt.layers.layernorm import (
    Gemma3RMSNorm,
    GemmaRMSNorm,
    LayerNorm,
    RMSNorm,
)
from sglang.test.test_utils import CustomTestCase


class TestRMSNorm(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
    ADD_RESIDUAL = [False, True]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_rms_norm_test(self, num_tokens, hidden_size, add_residual, dtype, seed):
        torch.manual_seed(seed)

        layer = RMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.1)
        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
        residual = torch.randn_like(x) * scale if add_residual else None

        with torch.inference_mode():
            ref_out = layer.forward_native(x, residual)
            out = layer(x, residual)

        if add_residual:
            self.assertTrue(torch.allclose(out[0], ref_out[0], atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(out[1], ref_out[1], atol=1e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2))

    def test_rms_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.ADD_RESIDUAL,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                add_residual=params[2],
                dtype=params[3],
                seed=params[4],
            ):
                self._run_rms_norm_test(*params)


class TestGemmaRMSNorm(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]
    ADD_RESIDUAL = [False, True]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_gemma_rms_norm_test(
        self, num_tokens, hidden_size, add_residual, dtype, seed
    ):
        torch.manual_seed(seed)

        layer = GemmaRMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=1.0, std=0.1)
        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale
        residual = torch.randn_like(x) * scale if add_residual else None

        with torch.inference_mode():
            ref_out = layer.forward_native(x, residual)
            out = layer(x, residual)

        if add_residual:
            self.assertTrue(torch.allclose(out[0], ref_out[0], atol=1e-3, rtol=1e-3))
            self.assertTrue(torch.allclose(out[1], ref_out[1], atol=1e-3, rtol=1e-3))
        else:
            self.assertTrue(torch.allclose(out, ref_out, atol=1e-3, rtol=1e-3))

    def test_gemma_rms_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.ADD_RESIDUAL,
            self.DTYPES,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                add_residual=params[2],
                dtype=params[3],
                seed=params[4],
            ):
                self._run_gemma_rms_norm_test(*params)


class TestGemma3RMSNorm(CustomTestCase):
    """Covers 2D/3D/4D inputs, including the non-contiguous ("unflatten")
    shape that Gemma3's q_norm/k_norm actually feed in: a head slice cut out
    of a wider qkv-style tensor via .split()+.unflatten(), so the leading
    dims are not flattenable to 2D."""

    DTYPES = [torch.half, torch.bfloat16]
    ADD_RESIDUAL = [False, True]
    SEEDS = [0]

    # (batch_size, hidden_size, dtype) combos for the 2D case.
    SHAPE_DTYPE_2D = [
        (batch_size, hidden_size, torch.float16)
        for batch_size in [1, 19, 99, 989]
        for hidden_size in [111, 500, 1024, 3072, 3584, 4096, 8192, 16384]
    ] + [
        (19, 1024, torch.bfloat16),
        (19, 1024, torch.float32),
        (2, 32768, torch.float16),
    ]

    BATCH_SIZES_3D = [1, 4, 19]
    SEQ_LENS_3D = [1, 7, 32]
    # hidden_size=1 exercises the "other dim == 1" shape (excluding the leading
    # batch dim) that bypasses the flattenable fast-path check but still
    # yields a correct result.
    HIDDEN_SIZES_3D = [1, 111, 1024, 4096]
    DTYPES_3D = [torch.float16]

    NUM_TOKENS_4D = [1, 7]
    # num_heads=1 and head_dim=1 exercise the "other dim == 1" shapes
    # (excluding the leading batch/token dims) that bypass the flattenable
    # fast-path check but still yield a correct result.
    NUM_HEADS_4D = [1, 4, 8]
    HEAD_DIMS_4D = [1, 64, 128]
    DTYPES_4D = [torch.float16]

    @classmethod
    def setUpClass(cls):
        if not (torch.cuda.is_available() or torch.xpu.is_available()):
            raise unittest.SkipTest("Neither CUDA nor XPU is available")
        device = "cuda" if torch.cuda.is_available() else "xpu"
        torch.set_default_device(device)

    def _run_gemma3_rms_norm_test(
        self, shape, add_residual, dtype, seed, non_contiguous=False
    ):
        torch.manual_seed(seed)
        hidden_size = shape[-1]
        layer = Gemma3RMSNorm(hidden_size).to(dtype=dtype)
        layer.weight.data.normal_(mean=0.0, std=0.1)
        scale = 1 / (2 * hidden_size)

        if non_contiguous:
            *lead, num_heads, head_dim = shape
            total_heads = num_heads + 3
            full = torch.randn(*lead, total_heads * head_dim, dtype=dtype) * scale
            x = full[..., : num_heads * head_dim].unflatten(-1, (num_heads, head_dim))
        else:
            x = torch.randn(*shape, dtype=dtype) * scale

        residual = torch.randn_like(x) * scale if add_residual else None

        with torch.inference_mode():
            ref_out = layer.forward_native(x, residual)
            out = layer(x, residual)

        if add_residual:
            self.assertTrue(torch.allclose(out[0], ref_out[0], atol=1e-2, rtol=1e-2))
            self.assertTrue(torch.allclose(out[1], ref_out[1], atol=1e-2, rtol=1e-2))
        else:
            self.assertTrue(torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2))

    def test_gemma3_rms_norm_2d(self):
        for (batch_size, hidden_size, dtype), add_residual, seed in itertools.product(
            self.SHAPE_DTYPE_2D, self.ADD_RESIDUAL, self.SEEDS
        ):
            with self.subTest(
                batch_size=batch_size,
                hidden_size=hidden_size,
                add_residual=add_residual,
                dtype=dtype,
            ):
                self._run_gemma3_rms_norm_test(
                    (batch_size, hidden_size), add_residual, dtype, seed
                )

    def test_gemma3_rms_norm_3d(self):
        for (
            batch_size,
            seq_len,
            hidden_size,
            dtype,
            add_residual,
            seed,
        ) in itertools.product(
            self.BATCH_SIZES_3D,
            self.SEQ_LENS_3D,
            self.HIDDEN_SIZES_3D,
            self.DTYPES_3D,
            self.ADD_RESIDUAL,
            self.SEEDS,
        ):
            with self.subTest(
                batch_size=batch_size,
                seq_len=seq_len,
                hidden_size=hidden_size,
                add_residual=add_residual,
                dtype=dtype,
            ):
                self._run_gemma3_rms_norm_test(
                    (batch_size, seq_len, hidden_size), add_residual, dtype, seed
                )

    def test_gemma3_rms_norm_4d(self):
        for (
            num_tokens,
            num_heads,
            head_dim,
            dtype,
            add_residual,
            seed,
        ) in itertools.product(
            self.NUM_TOKENS_4D,
            self.NUM_HEADS_4D,
            self.HEAD_DIMS_4D,
            self.DTYPES_4D,
            self.ADD_RESIDUAL,
            self.SEEDS,
        ):
            with self.subTest(
                num_tokens=num_tokens,
                num_heads=num_heads,
                head_dim=head_dim,
                add_residual=add_residual,
                dtype=dtype,
            ):
                self._run_gemma3_rms_norm_test(
                    (1, num_tokens, num_heads, head_dim), add_residual, dtype, seed
                )

    def test_gemma3_rms_norm_3d_unflatten(self):
        for head_dim, add_residual, dtype, seed in itertools.product(
            [64, 128], self.ADD_RESIDUAL, self.DTYPES, self.SEEDS
        ):
            with self.subTest(
                head_dim=head_dim, add_residual=add_residual, dtype=dtype
            ):
                self._run_gemma3_rms_norm_test(
                    (19, 4, head_dim), add_residual, dtype, seed, non_contiguous=True
                )

    def test_gemma3_rms_norm_4d_unflatten(self):
        for head_dim, add_residual, dtype, seed in itertools.product(
            [64, 128], self.ADD_RESIDUAL, self.DTYPES, self.SEEDS
        ):
            with self.subTest(
                head_dim=head_dim, add_residual=add_residual, dtype=dtype
            ):
                self._run_gemma3_rms_norm_test(
                    (1, 19, 4, head_dim),
                    add_residual,
                    dtype,
                    seed,
                    non_contiguous=True,
                )


class TestLayerNorm(CustomTestCase):
    DTYPES = [torch.half, torch.bfloat16]
    PARAM_DTYPES = [torch.bfloat16, torch.float32]
    NUM_TOKENS = [7, 83, 1024]
    HIDDEN_SIZES = [128, 512, 1536, 5120, 5124, 5125, 5126, 7168]
    USE_AFFINE = [False, True]
    USE_BIAS = [False, True]
    SEEDS = [0]

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_layer_norm_test(
        self, num_tokens, hidden_size, use_affine, use_bias, dtype, seed, param_dtype
    ):
        torch.manual_seed(seed)

        layer = LayerNorm(
            hidden_size, elementwise_affine=use_affine, bias=use_bias, dtype=param_dtype
        )
        if use_affine:
            layer.weight.data.normal_(mean=1.0, std=0.1)
            if use_bias:
                layer.bias.data.normal_(mean=0.0, std=0.1)

        scale = 1 / (2 * hidden_size)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype) * scale

        with torch.inference_mode():
            ref_out = layer.forward_native(x)
            out = layer(x)

        self.assertTrue(torch.allclose(out, ref_out, atol=1e-2, rtol=1e-3))

        if (
            use_affine
            and use_bias
            and not (dtype == torch.bfloat16 and param_dtype == torch.float32)
        ):
            layer.dtype = torch.float32
            layer.weight.data = layer.weight.data.to(torch.float32)
            layer.bias.data = layer.bias.data.to(torch.float32)
            with torch.inference_mode():
                cuda_out = layer(x.to(torch.bfloat16)).to(x.dtype)

            self.assertTrue(torch.allclose(cuda_out, ref_out, atol=2e-2, rtol=1e-3))

    def test_layer_norm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.USE_AFFINE,
            self.USE_BIAS,
            self.DTYPES,
            self.SEEDS,
            self.PARAM_DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                use_affine=params[2],
                use_bias=params[3],
                dtype=params[4],
                seed=params[5],
                param_dtype=params[6],
            ):
                self._run_layer_norm_test(*params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
