import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, call, patch

import torch
import torch.nn as nn

import sglang.srt.utils as srt_utils

fake_aiter_for_import = ModuleType("aiter")
fake_aiter_for_import.__path__ = []
fake_aiter_ops = ModuleType("aiter.ops")
fake_aiter_ops.__path__ = []
fake_aiter_triton = ModuleType("aiter.ops.triton")
fake_aiter_triton.__path__ = []
fake_aiter_quant = ModuleType("aiter.ops.triton.quant")
fake_aiter_quant.dynamic_mxfp4_quant = Mock()

fake_aiter_modules = {
    "aiter": fake_aiter_for_import,
    "aiter.ops": fake_aiter_ops,
    "aiter.ops.triton": fake_aiter_triton,
    "aiter.ops.triton.quant": fake_aiter_quant,
}
missing_module = object()
previous_aiter_modules = {
    name: sys.modules.get(name, missing_module) for name in fake_aiter_modules
}
sys.modules.update(fake_aiter_modules)
try:
    with (
        patch.dict(os.environ, {"SGLANG_USE_AITER": "0"}),
        patch.object(srt_utils, "is_hip", return_value=False),
        patch.object(
            torch.cuda,
            "get_device_properties",
            return_value=SimpleNamespace(gcnArchName="gfx950", major=9, minor=5),
        ),
    ):
        import sglang.kernels.ops.attention.dsv4.aiter_fp4_indexer as aiter_fp4_indexer
        import sglang.srt.layers.attention.dsv4.compressor as compressor_module
        import sglang.srt.layers.attention.dsv4.compressor_v2 as compressor_v2
        import sglang.srt.layers.attention.dsv4.indexer as indexer_module
        from sglang.srt.layers.attention.dsv4.compressor_v2 import (
            CompressorBackendMixin,
        )
        from sglang.srt.layers.attention.dsv4.indexer import C4Indexer
        from sglang.test.ci.ci_register import register_cpu_ci
finally:
    for module_name, previous_module in previous_aiter_modules.items():
        if previous_module is missing_module:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _StaticLinear(nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.output = output

    def forward(self, _: torch.Tensor):
        return self.output, None


class _StubLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class TestDSV4FP4QProducer(unittest.TestCase):
    def setUp(self):
        self.num_tokens = 2
        self.projected_q = torch.arange(
            self.num_tokens * 64 * 128, dtype=torch.float32
        ).to(torch.bfloat16)
        self.projected_q = self.projected_q.view(self.num_tokens, 64 * 128)
        real = torch.randn(128, 32)
        imag = torch.randn(128, 32)
        self.freqs_cis = torch.complex(real, imag)

    def _make_indexer(self, use_fp4_indexer: bool) -> C4Indexer:
        indexer = C4Indexer.__new__(C4Indexer)
        nn.Module.__init__(indexer)
        indexer.n_local_heads = 64
        indexer.head_dim = 128
        indexer.use_fp4_indexer = use_fp4_indexer
        indexer.weight_scale = 0.125
        indexer.freqs_cis = self.freqs_cis
        indexer.compressor = SimpleNamespace(
            aiter_fp4_cos=self.freqs_cis.real.to(torch.bfloat16).contiguous(),
            aiter_fp4_sin=self.freqs_cis.imag.to(torch.bfloat16).contiguous(),
        )
        indexer.wq_b = _StaticLinear(self.projected_q)
        return indexer

    def test_hip_fp4_uses_exact_aiter_q_contract(self):
        payload_dtype = torch.float4_e2m1fn_x2
        fake_aiter = ModuleType("aiter")
        fake_aiter.dtypes = SimpleNamespace(fp4x2=payload_dtype)
        fake_aiter.rope_rotate_activation = Mock()

        config = SimpleNamespace(
            hidden_size=16,
            index_n_heads=64,
            index_head_dim=128,
            qk_rope_head_dim=64,
            index_topk=512,
            q_lora_rank=8,
            rms_norm_eps=1e-6,
        )
        with (
            patch.object(indexer_module, "ReplicatedLinear", _StubLinear),
            patch.object(compressor_module, "ReplicatedLinear", _StubLinear),
            patch.object(
                indexer_module,
                "get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(enable_deepseek_v4_fp4_indexer=True)
                ),
            ),
            patch.object(indexer_module, "is_hip", return_value=True),
            patch.object(
                indexer_module,
                "prepare_aiter_fp4_indexer_cos_sin",
                wraps=aiter_fp4_indexer.prepare_aiter_fp4_indexer_cos_sin,
            ) as prepare_cos_sin,
        ):
            indexer = C4Indexer(config, layer_id=7, freqs_cis=self.freqs_cis)

        prepare_cos_sin.assert_called_once_with(self.freqs_cis)
        indexer.wq_b = _StaticLinear(self.projected_q)
        constructed_cos = indexer.compressor.aiter_fp4_cos
        constructed_sin = indexer.compressor.aiter_fp4_sin
        indexer._apply(lambda tensor: tensor.clone())
        cos = indexer.compressor.aiter_fp4_cos
        sin = indexer.compressor.aiter_fp4_sin
        self.assertIsNot(cos, constructed_cos)
        self.assertIsNot(sin, constructed_sin)
        self.assertIsNotNone(cos)
        self.assertIsNotNone(sin)
        self.assertEqual(cos.dtype, torch.bfloat16)
        self.assertEqual(sin.dtype, torch.bfloat16)
        self.assertTrue(cos.is_contiguous())
        self.assertTrue(sin.is_contiguous())
        self.assertIn("aiter_fp4_cos", indexer.compressor._non_persistent_buffers_set)
        self.assertIn("aiter_fp4_sin", indexer.compressor._non_persistent_buffers_set)
        self.assertNotIn("compressor.aiter_fp4_cos", indexer.state_dict())
        self.assertNotIn("compressor.aiter_fp4_sin", indexer.state_dict())
        self.assertNotIn("_FREQS_CIS_TO_COS_SIN", vars(aiter_fp4_indexer))
        q_lora = torch.empty(self.num_tokens, 1, dtype=torch.bfloat16)
        raw_weights = torch.randn(self.num_tokens, 64, dtype=torch.bfloat16)
        expected_weights = raw_weights.clone()
        positions = torch.arange(self.num_tokens * 2, dtype=torch.int32)[::2]
        real_empty = torch.empty
        tensor_to = torch.Tensor.to

        def reject_full_table_to(tensor, *args, **kwargs):
            if tensor.shape == self.freqs_cis.shape:
                raise AssertionError(
                    "full RoPE tables must not be converted in forward"
                )
            return tensor_to(tensor, *args, **kwargs)

        with (
            patch.dict(sys.modules, {"aiter": fake_aiter}),
            patch.object(indexer_module, "is_hip", return_value=True),
            patch.object(
                indexer_module, "fused_q_indexer_rope_hadamard_fp4_quant"
            ) as cuda_fp4,
            patch.object(
                aiter_fp4_indexer.torch, "empty", wraps=real_empty
            ) as empty_mock,
            patch.object(torch.Tensor, "to", reject_full_table_to),
        ):
            (q_fp4, q_scale), returned_weights = indexer.compute_q(
                q_lora, positions, raw_weights
            )

        self.assertIs(returned_weights, raw_weights)
        self.assertEqual(returned_weights.dtype, torch.bfloat16)
        torch.testing.assert_close(returned_weights, expected_weights)
        self.assertEqual(q_fp4.shape, (self.num_tokens, 64, 64))
        self.assertEqual(q_fp4.dtype, payload_dtype)
        self.assertEqual(q_scale.shape, (self.num_tokens, 1, 4, 16, 4))
        self.assertEqual(q_scale.dtype, torch.uint8)
        self.assertEqual(
            empty_mock.call_args_list,
            [
                call(
                    (self.num_tokens, 64, 64),
                    dtype=payload_dtype,
                    device=torch.device("cpu"),
                ),
                call(
                    (self.num_tokens, 1, 4, 16, 4),
                    dtype=torch.uint8,
                    device=torch.device("cpu"),
                ),
            ],
        )

        rope_args = fake_aiter.rope_rotate_activation.call_args.args
        rope_kwargs = fake_aiter.rope_rotate_activation.call_args.kwargs
        self.assertIs(rope_args[0], q_fp4)
        self.assertEqual(rope_args[1].shape, (self.num_tokens, 64, 128))
        self.assertEqual(rope_args[1].dtype, torch.bfloat16)
        self.assertTrue(rope_args[1].is_contiguous())
        torch.testing.assert_close(
            rope_args[1], self.projected_q.view(self.num_tokens, 64, 128)
        )
        torch.testing.assert_close(rope_args[2], self.freqs_cis.real.to(torch.bfloat16))
        torch.testing.assert_close(rope_args[3], self.freqs_cis.imag.to(torch.bfloat16))
        self.assertTrue(rope_args[2].is_contiguous())
        self.assertTrue(rope_args[3].is_contiguous())
        self.assertIs(rope_args[2], cos)
        self.assertIs(rope_args[3], sin)
        self.assertEqual(rope_args[4].dtype, torch.int64)
        self.assertTrue(rope_args[4].is_contiguous())
        torch.testing.assert_close(rope_args[4], positions.to(torch.int64))
        self.assertEqual(
            rope_kwargs,
            {
                "rope_dim": 64,
                "out_scale": q_scale,
                "group_size": 32,
                "shuffle_scale": True,
                "do_rotate_act": True,
            },
        )
        cuda_fp4.assert_not_called()

        backend = CompressorBackendMixin.__new__(CompressorBackendMixin)
        backend._get_paged_compress_metadata = Mock(return_value=object())
        compressed = torch.empty((self.num_tokens, 128), dtype=torch.bfloat16)
        with (
            patch.object(compressor_v2, "compress_forward", return_value=compressed),
            patch.object(
                aiter_fp4_indexer, "aiter_k_indexer_fp4_cache_write"
            ) as k_store,
        ):
            backend._forward_compress_all_in_one(
                kv_score_buffer=torch.empty((2, 4, 512)),
                kv_score_input=torch.empty((2, 256)),
                ape=indexer.compressor.ape,
                head_dim=128,
                norm=indexer.compressor.norm,
                freqs_cis_cache=self.freqs_cis,
                kv_cache=torch.empty((2, 1, 4, 64, 16), dtype=torch.float4_e2m1fn_x2),
                kv_scale_cache=torch.empty((2, 1, 4, 64), dtype=torch.uint8),
                is_indexer=True,
                rotate=True,
                compress_ratio=4,
                page_size=64,
                out_loc=torch.tensor([3, 7]),
                use_fp4_indexer=True,
                use_aiter_fp4_indexer=True,
                aiter_fp4_cos=cos,
                aiter_fp4_sin=sin,
            )

        self.assertIs(k_store.call_args.kwargs["cos"], rope_args[2])
        self.assertIs(k_store.call_args.kwargs["sin"], rope_args[3])

    def test_cuda_fp4_delegation_is_unchanged(self):
        expected = Mock()
        indexer = self._make_indexer(use_fp4_indexer=True)
        raw_weights = torch.randn(self.num_tokens, 64, dtype=torch.bfloat16)
        positions = torch.arange(self.num_tokens, dtype=torch.int64)

        with (
            patch.object(indexer_module, "is_hip", return_value=False),
            patch.object(
                indexer_module,
                "fused_q_indexer_rope_hadamard_fp4_quant",
                return_value=expected,
            ) as cuda_fp4,
            patch.object(indexer_module, "fused_q_indexer_rope_hadamard_quant") as fp8,
        ):
            actual = indexer.compute_q(
                torch.empty(self.num_tokens, 1), positions, raw_weights
            )

        self.assertIs(actual, expected)
        args = cuda_fp4.call_args.args
        self.assertEqual(args[0].shape, (self.num_tokens, 64, 128))
        self.assertTrue(args[0].is_contiguous())
        self.assertIs(args[1], raw_weights)
        self.assertEqual(args[2], indexer.weight_scale)
        self.assertIs(args[3], self.freqs_cis)
        self.assertIs(args[4], positions)
        fp8.assert_not_called()

    def test_fp8_delegation_is_unchanged(self):
        expected = Mock()
        indexer = self._make_indexer(use_fp4_indexer=False)
        raw_weights = torch.randn(self.num_tokens, 64, dtype=torch.bfloat16)
        positions = torch.arange(self.num_tokens, dtype=torch.int64)

        with (
            patch.object(indexer_module, "is_hip", return_value=False) as is_hip,
            patch.object(
                indexer_module, "fused_q_indexer_rope_hadamard_fp4_quant"
            ) as cuda_fp4,
            patch.object(
                indexer_module,
                "fused_q_indexer_rope_hadamard_quant",
                return_value=expected,
            ) as fp8,
        ):
            actual = indexer.compute_q(
                torch.empty(self.num_tokens, 1), positions, raw_weights
            )

        self.assertIs(actual, expected)
        args = fp8.call_args.args
        self.assertEqual(args[0].shape, (self.num_tokens, 64, 128))
        self.assertIs(args[1], raw_weights)
        self.assertEqual(args[2], indexer.weight_scale)
        self.assertIs(args[3], self.freqs_cis)
        self.assertIs(args[4], positions)
        is_hip.assert_not_called()
        cuda_fp4.assert_not_called()

    def test_hip_fp4_validates_q_positions_and_rope_shapes(self):
        positions = torch.arange(self.num_tokens, dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, r"q shape \[T, 64, 128\]"):
            aiter_fp4_indexer.aiter_q_indexer_rope_hadamard_fp4_quant(
                torch.empty(self.num_tokens, 63, 128, dtype=torch.bfloat16),
                self.freqs_cis.real.to(torch.bfloat16).contiguous(),
                self.freqs_cis.imag.to(torch.bfloat16).contiguous(),
                positions,
            )
        with self.assertRaisesRegex(ValueError, r"positions shape \[T\]"):
            aiter_fp4_indexer.aiter_q_indexer_rope_hadamard_fp4_quant(
                torch.empty(self.num_tokens, 64, 128, dtype=torch.bfloat16),
                self.freqs_cis.real.to(torch.bfloat16).contiguous(),
                self.freqs_cis.imag.to(torch.bfloat16).contiguous(),
                positions[:1],
            )
        with self.assertRaisesRegex(
            ValueError, r"cos/sin with shape \[max_position, 32\]"
        ):
            aiter_fp4_indexer.aiter_q_indexer_rope_hadamard_fp4_quant(
                torch.empty(self.num_tokens, 64, 128, dtype=torch.bfloat16),
                torch.empty(128, 64, dtype=torch.bfloat16),
                torch.empty(128, 64, dtype=torch.bfloat16),
                positions,
            )
        with self.assertRaisesRegex(ValueError, r"dtype torch.bfloat16"):
            aiter_fp4_indexer.aiter_q_indexer_rope_hadamard_fp4_quant(
                torch.empty(self.num_tokens, 64, 128, dtype=torch.bfloat16),
                torch.empty(128, 32, dtype=torch.float32),
                torch.empty(128, 32, dtype=torch.bfloat16),
                positions,
            )
        with self.assertRaisesRegex(ValueError, r"contiguous cos"):
            aiter_fp4_indexer.aiter_q_indexer_rope_hadamard_fp4_quant(
                torch.empty(self.num_tokens, 64, 128, dtype=torch.bfloat16),
                torch.empty(32, 128, dtype=torch.bfloat16).T,
                torch.empty(128, 32, dtype=torch.bfloat16),
                positions,
            )
        with self.assertRaisesRegex(ValueError, r"same device"):
            aiter_fp4_indexer.aiter_q_indexer_rope_hadamard_fp4_quant(
                torch.empty(self.num_tokens, 64, 128, dtype=torch.bfloat16),
                torch.empty(128, 32, dtype=torch.bfloat16, device="meta"),
                torch.empty(128, 32, dtype=torch.bfloat16, device="meta"),
                positions,
            )


if __name__ == "__main__":
    unittest.main()
