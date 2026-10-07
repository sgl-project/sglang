"""Pool dispatch and real DFLASH initialization for FP8 prefix commits."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, is_fp8_fnuz
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, _resolve_fused_scale
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kernels.prefix_valid import assert_prefix_commit, make_kv_cache
from sglang.test.test_utils import CustomTestCase

# Preserve the original GPU lanes; the kernel file retains the other 25 seconds.
register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# backend-specific: loaded FP8 scales are doubled for ROCm's E4M3FNUZ format.
register_amd_ci(est_time=15, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"


def _make_pool(row_dim, total_slots, dtype=fp8_dtype):
    # Isolate dispatch from the allocator; real DFLASH initialization is tested below.
    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.dtype = dtype
    pool.store_dtype = torch.uint8 if dtype == fp8_dtype else dtype
    pool.start_layer = 0
    pool.row_dim = pool.v_row_dim = pool.head_dim = pool.v_head_dim = row_dim
    k, v = make_kv_cache(total_slots, row_dim, dtype)
    pool.k_buffer, pool.v_buffer = (
        [k.view(pool.store_dtype)],
        [v.view(pool.store_dtype)],
    )
    return pool


def _make_layer(k_scale, v_scale):
    scales = [
        torch.nn.Parameter(
            torch.tensor(value, dtype=torch.float32, device=DEVICE), requires_grad=False
        )
        for value in (k_scale, v_scale)
    ]
    return SimpleNamespace(
        layer_id=0,
        k_scale=scales[0],
        v_scale=scales[1],
        k_scale_float=scales[0].item(),
        v_scale_float=scales[1].item(),
    )


def _check_pool_commit(
    pool, k, v, loc, lengths, k_scale, v_scale, *, inputs_mutated=False
):
    return assert_prefix_commit(
        k,
        v,
        pool.k_buffer[0].view(pool.dtype),
        pool.v_buffer[0].view(pool.dtype),
        loc,
        lengths,
        k_scale,
        v_scale,
        inputs_mutated=inputs_mutated,
    )


class TestPrefixValidFp8Pool(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0)

    def test_pool_dispatch_uses_gpu_parameter_float_shadows(self):
        """GPU scale Parameters must fuse without synchronizing through item()."""
        pool, layer = _make_pool(65, 12), _make_layer(0.625, 1.375)
        loc = torch.tensor([[2, 7, 9]], dtype=torch.int32, device=DEVICE)
        lengths = torch.tensor([2], dtype=torch.int64, device=DEVICE)
        k = torch.randn((3, 1, 65), dtype=torch.bfloat16, device=DEVICE)
        v = torch.randn_like(k)
        with _check_pool_commit(pool, k, v, loc, lengths, layer.k_scale, layer.v_scale):
            with patch.object(
                torch.Tensor,
                "item",
                side_effect=AssertionError("dispatch called item()"),
            ):
                pool.set_kv_buffer_prefix_valid(
                    layer, loc, lengths, k, v, layer.k_scale, layer.v_scale
                )

    def test_scale_kind_preserves_eager_division(self):
        """Preserve the BF16 0.1 counterexample and independent K/V scale kinds."""
        loc = torch.tensor([[2, -1, -1], [5, 1, -1]], device=DEVICE)
        lengths = torch.tensor([1, 2], dtype=torch.int32, device=DEVICE)
        kinds = (
            ("gpu", "gpu"),
            ("gpu", "python"),
            ("python", "gpu"),
            ("cpu", "gpu"),
            ("gpu", "cpu"),
        )
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for k_kind, v_kind in kinds:
                with self.subTest(dtype=dtype, k_kind=k_kind, v_kind=v_kind):
                    pool, layer = _make_pool(65, 8), _make_layer(0.1, 0.3)
                    k_scale = {
                        "gpu": layer.k_scale,
                        "python": layer.k_scale_float,
                        "cpu": layer.k_scale.cpu(),
                    }[k_kind]
                    v_scale = {
                        "gpu": layer.v_scale,
                        "python": layer.v_scale_float,
                        "cpu": layer.v_scale.cpu(),
                    }[v_kind]
                    k = (
                        torch.linspace(-2, 2, 390, device=DEVICE)
                        .to(dtype)
                        .view(6, 1, 65)
                    )
                    k[:, :, 0] = 0.0966796875
                    v = k.flip(-1)
                    with _check_pool_commit(pool, k, v, loc, lengths, k_scale, v_scale):
                        pool.set_kv_buffer_prefix_valid(
                            layer, loc, lengths, k, v, k_scale, v_scale
                        )

    def test_host_scale_rounding_through_pool(self):
        """Host reciprocal rounding must not move FP32 inputs onto FP8 ties."""
        loc = torch.tensor([[3, -1]], device=DEVICE)
        lengths = torch.tensor([1], dtype=torch.int32, device=DEVICE)
        for scale in (10000.0, torch.tensor(10000.0)):
            with self.subTest(scale_type=type(scale).__name__):
                pool, layer = _make_pool(65, 8), SimpleNamespace(layer_id=0)
                k = torch.full(
                    (2, 1, 65), 29.296873092651367, dtype=torch.float32, device=DEVICE
                )
                v = -k
                with _check_pool_commit(pool, k, v, loc, lengths, scale, scale):
                    pool.set_kv_buffer_prefix_valid(
                        layer, loc, lengths, k, v, scale, scale
                    )

    def test_eager_fallbacks(self):
        """Unresolved/missing scales and non-FP8 storage retain the old writer."""
        unresolved = SimpleNamespace(
            layer_id=0,
            k_scale=torch.tensor(0.625, device=DEVICE),
            v_scale=torch.tensor(1.375, device=DEVICE),
            k_scale_float=None,
            v_scale_float=None,
        )
        missing = SimpleNamespace(
            layer_id=0,
            k_scale=None,
            v_scale=1.375,
            k_scale_float=None,
            v_scale_float=1.375,
        )
        cases = (
            (
                "unresolved_gpu_scales",
                fp8_dtype,
                unresolved,
                unresolved.k_scale,
                unresolved.v_scale,
                True,
            ),
            ("missing_k_scale", fp8_dtype, missing, None, missing.v_scale, True),
            (
                "non_fp8_cache",
                torch.bfloat16,
                SimpleNamespace(layer_id=0),
                None,
                None,
                False,
            ),
        )
        loc = torch.tensor([[2, 7, 9]], device=DEVICE)
        lengths = torch.tensor([2], dtype=torch.int32, device=DEVICE)
        for name, dtype, layer, k_scale, v_scale, mutated in cases:
            with self.subTest(case=name):
                pool = _make_pool(65, 12, dtype)
                k = torch.randn((3, 1, 65), dtype=torch.bfloat16, device=DEVICE)
                v = torch.randn_like(k)
                with _check_pool_commit(
                    pool, k, v, loc, lengths, k_scale, v_scale, inputs_mutated=mutated
                ):
                    pool.set_kv_buffer_prefix_valid(
                        layer, loc, lengths, k, v, k_scale, v_scale
                    )


class TestDflashFp8PrefixCommit(CustomTestCase):
    def test_dflash_constructor_initializes_fp8_prefix_scales(self):
        """A real FP8 draft must initialize scales and reach the fused writer."""
        from transformers import LlamaConfig

        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8KVCacheMethod
        from sglang.srt.models.dflash import DFlashAttention
        from sglang.srt.runtime_context import get_context, get_parallel

        config = LlamaConfig(
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=64,
            max_position_embeddings=128,
        )
        with (
            get_context().override_server_args(tp_size=1),
            get_parallel().override(tp_rank=0),
        ):
            with torch.device(DEVICE):
                plain = DFlashAttention(config, layer_id=0).attn
            self.assertIsNone(plain.quant_method)
            self.assertIsNone(plain.k_scale)
            self.assertIsNone(plain.v_scale)

            for loaded_scales in (None, (0.625, 1.375)):
                with self.subTest(loaded_scales=loaded_scales):
                    with torch.device(DEVICE):
                        layer = DFlashAttention(
                            config, layer_id=0, quant_config=Fp8Config()
                        ).attn
                    self.assertIsInstance(layer.quant_method, Fp8KVCacheMethod)
                    for scale in (layer.k_scale, layer.v_scale):
                        self.assertIsInstance(scale, torch.nn.Parameter)
                        self.assertTrue(scale.is_cuda)
                        self.assertEqual(scale.dtype, torch.float32)
                        self.assertEqual(scale.ndim, 0)
                        self.assertEqual(scale.item(), -1.0)
                    if loaded_scales is not None:
                        with torch.no_grad():
                            layer.k_scale.fill_(loaded_scales[0])
                            layer.v_scale.fill_(loaded_scales[1])
                    layer.quant_method.process_weights_after_loading(layer)
                    factor = 2 if loaded_scales and is_fp8_fnuz() else 1
                    expected_scales = tuple(
                        value * factor for value in (loaded_scales or (1.0, 1.0))
                    )
                    self.assertEqual(
                        (layer.k_scale_float, layer.v_scale_float), expected_scales
                    )
                    self.assertEqual(
                        (layer.k_scale.item(), layer.v_scale.item()), expected_scales
                    )

                    pool = _make_pool(64, 8)
                    loc = torch.tensor([[2, 5, -1]], device=DEVICE)
                    lengths = torch.tensor([2], dtype=torch.int32, device=DEVICE)
                    k = (
                        torch.linspace(-2, 2, 192, device=DEVICE)
                        .to(torch.bfloat16)
                        .view(3, 1, 64)
                    )
                    v = k.flip(-1)
                    with _check_pool_commit(
                        pool, k, v, loc, lengths, layer.k_scale, layer.v_scale
                    ):
                        # Run real GPU stores; reject only the unintended fallback.
                        with patch(
                            "sglang.srt.mem_cache.memory_pool._set_kv_buffer_prefix_valid_impl",
                            side_effect=AssertionError(
                                "FP8 draft took the eager fallback"
                            ),
                        ):
                            pool.set_kv_buffer_prefix_valid(
                                layer, loc, lengths, k, v, layer.k_scale, layer.v_scale
                            )


class TestResolveFusedScale(CustomTestCase):
    def test_supported_and_fallback_scale_forms(self):
        """Only synchronization-free scalar forms may enter the fused kernel."""
        layer_scale = torch.tensor(0.5, dtype=torch.float32, device=DEVICE)
        unrelated_scale = torch.tensor(0.75, dtype=torch.float32, device=DEVICE)
        vector_scale = layer_scale.view(1)
        cases = (
            (0.25, None, None, 0.25),
            (2, None, None, 2.0),
            (torch.tensor(0.5), None, None, 0.5),
            (layer_scale, layer_scale, 0.5, 0.5),
            (unrelated_scale, layer_scale, 0.5, None),
            (None, None, 0.5, None),
            (vector_scale, vector_scale, 0.5, None),
            (torch.tensor([0.5]), None, None, None),
        )
        for scale, owner, shadow, expected in cases:
            with self.subTest(
                scale_type=type(scale).__name__,
                has_owner=owner is not None,
                shadow=shadow,
            ):
                self.assertEqual(_resolve_fused_scale(scale, owner, shadow), expected)


if __name__ == "__main__":
    unittest.main()
