import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

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
        import sglang.srt.layers.attention.dsv4.compressor_v2 as compressor_v2
        from sglang.kernels.ops.attention.dsv4.compress import (
            CompressorDecodePlan,
            CompressorPrefillPlan,
        )
        from sglang.srt.layers.attention.dsv4.compressor_v2 import (
            CompressorBackendMixin,
        )
        from sglang.test.ci.ci_register import register_cpu_ci
finally:
    for module_name, previous_module in previous_aiter_modules.items():
        if previous_module is missing_module:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module


register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _decode_plan(seq_lens):
    words = torch.zeros((len(seq_lens), 4), dtype=torch.int32)
    words[:, 0] = torch.tensor(seq_lens, dtype=torch.int32)
    return CompressorDecodePlan(4, words.view(torch.uint8))


def _prefill_plan(rows):
    words = torch.zeros((len(rows), 4), dtype=torch.int32)
    for row, (seq_len, ragged_id) in enumerate(rows):
        words[row, 0] = seq_len
        words[row, 1] = ragged_id
    plan_w = torch.empty((0, 8), dtype=torch.uint8)
    return CompressorPrefillPlan(4, words.view(torch.uint8), plan_w)


class TestAITERFP4KCacheWriter(unittest.TestCase):
    def setUp(self):
        real = torch.randn(128, 32)
        imag = torch.randn(128, 32)
        self.freqs_cis = torch.complex(real, imag)
        self.cos = self.freqs_cis.real.to(torch.bfloat16).contiguous()
        self.sin = self.freqs_cis.imag.to(torch.bfloat16).contiguous()
        self.norm_weight = torch.randn(256, dtype=torch.float32)[::2]
        self.payload = torch.empty((2, 1, 4, 64, 16), dtype=torch.float4_e2m1fn_x2)
        self.scale = torch.empty((2, 1, 4, 64), dtype=torch.uint8)

    def _run_writer(self, plan, out_loc, write_metadata=None):
        num_rows = plan[1].shape[0]
        k = torch.randn(128, num_rows, dtype=torch.float32).T
        self.assertFalse(k.is_contiguous())
        fake_aiter = ModuleType("aiter")
        fake_aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache = Mock()
        tensor_to = torch.Tensor.to

        def reject_full_table_to(tensor, *args, **kwargs):
            if tensor.shape == self.freqs_cis.shape:
                raise AssertionError(
                    "full RoPE tables must not be converted in forward"
                )
            return tensor_to(tensor, *args, **kwargs)

        with (
            patch.dict(sys.modules, {"aiter": fake_aiter}),
            patch.object(torch.Tensor, "to", reject_full_table_to),
        ):
            aiter_fp4_indexer.aiter_k_indexer_fp4_cache_write(
                k=k,
                norm_weight=self.norm_weight,
                norm_epsilon=1e-6,
                cos=self.cos,
                sin=self.sin,
                plan=plan,
                out_loc=out_loc,
                k_payload=self.payload,
                k_scale=self.scale,
                write_metadata=write_metadata,
            )

        call_args = fake_aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache.call_args
        self.assertIsNotNone(call_args)
        args = call_args.args
        self.assertIs(args[0], self.payload)
        self.assertIs(args[1], self.scale)
        self.assertEqual(args[2].shape, (num_rows, 1, 128))
        self.assertEqual(args[2].dtype, torch.bfloat16)
        self.assertTrue(args[2].is_contiguous())
        torch.testing.assert_close(args[2].squeeze(1), k.to(torch.bfloat16))
        self.assertEqual(args[3].shape, (128,))
        self.assertEqual(args[3].dtype, torch.bfloat16)
        self.assertTrue(args[3].is_contiguous())
        torch.testing.assert_close(args[3], self.norm_weight.to(torch.bfloat16))
        self.assertEqual(args[4].dtype, torch.bfloat16)
        self.assertEqual(args[5].dtype, torch.bfloat16)
        self.assertTrue(args[4].is_contiguous())
        self.assertTrue(args[5].is_contiguous())
        self.assertIs(args[4], self.cos)
        self.assertIs(args[5], self.sin)
        torch.testing.assert_close(args[4], self.freqs_cis.real.to(torch.bfloat16))
        torch.testing.assert_close(args[5], self.freqs_cis.imag.to(torch.bfloat16))
        self.assertEqual(args[6].dtype, torch.int64)
        self.assertEqual(args[7].dtype, torch.int64)
        self.assertTrue(args[6].is_contiguous())
        self.assertTrue(args[7].is_contiguous())
        self.assertEqual(args[8], 1e-6)
        self.assertEqual(
            call_args.kwargs,
            {
                "rope_dim": 64,
                "kv_block_size": 64,
                "group_size": 32,
                "shuffle_scale": True,
                "do_rotate_act": True,
            },
        )
        return args[6], args[7]

    def test_decode_masks_non_boundary_and_padded_rows(self):
        plan = _decode_plan([4, 5, 8, -1, 0])
        positions, slots = self._run_writer(
            plan, torch.tensor([11, 22, 33, 44, 55], dtype=torch.int32)
        )

        torch.testing.assert_close(positions, torch.tensor([0, 1, 4, 0, 0]))
        torch.testing.assert_close(slots, torch.tensor([11, -1, 33, -1, -1]))

    def test_prefill_uses_plan_ragged_order_and_skips_invalid_rows(self):
        plan = _prefill_plan([(8, 2), (12, 0), (-1, 1)])
        positions, slots = self._run_writer(
            plan, torch.tensor([101, 202, 303], dtype=torch.int64)
        )

        torch.testing.assert_close(positions, torch.tensor([4, 8, 0]))
        torch.testing.assert_close(slots, torch.tensor([303, 101, -1]))

    def test_precomputed_metadata_is_forwarded(self):
        plan = _decode_plan([4, 8])
        expected_positions = torch.tensor([0, 4], dtype=torch.int64)
        expected_slots = torch.tensor([11, 22], dtype=torch.int64)

        positions, slots = self._run_writer(
            plan,
            torch.tensor([11, 22], dtype=torch.int32),
            (expected_positions, expected_slots),
        )

        self.assertIs(positions, expected_positions)
        self.assertIs(slots, expected_slots)

    def test_empty_prefill_is_a_no_op(self):
        fake_aiter = ModuleType("aiter")
        fake_aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache = Mock()
        plan = _prefill_plan([])

        with patch.dict(sys.modules, {"aiter": fake_aiter}):
            aiter_fp4_indexer.aiter_k_indexer_fp4_cache_write(
                k=torch.empty((0, 128)),
                norm_weight=self.norm_weight,
                norm_epsilon=1e-6,
                cos=self.cos,
                sin=self.sin,
                plan=plan,
                out_loc=torch.empty((0,), dtype=torch.int64),
                k_payload=self.payload,
                k_scale=self.scale,
            )

        fake_aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache.assert_not_called()


class TestFP4KCompressorRouting(unittest.TestCase):
    def _make_forward_objects(self, *, is_indexer=True, ratio=4, enabled=True):
        kv_score_input = torch.empty((2, 256), dtype=torch.bfloat16)
        state_buffer = torch.empty((2, ratio, 256), dtype=torch.bfloat16)
        out_loc = torch.tensor([3, 7], dtype=torch.int64)

        token_pool = SimpleNamespace(
            get_index_k_with_scale_buffer=Mock(
                return_value=torch.empty((2, 4352), dtype=torch.uint8)
            ),
            get_index_k_fp4_payload_buffer=Mock(
                return_value=torch.empty(
                    (2, 1, 4, 64, 16), dtype=torch.float4_e2m1fn_x2
                )
            ),
            get_index_k_fp4_scale_buffer=Mock(
                return_value=torch.empty((2, 1, 4, 64), dtype=torch.uint8)
            ),
            get_index_k_page_size=Mock(return_value=64),
            get_extra_key_buffer=Mock(
                return_value=torch.empty((2, 128, 512), dtype=torch.bfloat16)
            ),
            get_extra_key_page_size=Mock(return_value=128),
            layer_mapping={7: (ratio, 0, SimpleNamespace())},
        )
        compressor = SimpleNamespace(
            ratio=ratio,
            is_in_indexer=is_indexer,
            head_dim=128 if is_indexer else 512,
            rotate=is_indexer,
            compute_kv_score=Mock(return_value=kv_score_input),
            get_state_pool=Mock(
                return_value=SimpleNamespace(
                    kv_score_buffer=SimpleNamespace(kv_score=state_buffer)
                )
            ),
            ape=torch.empty((ratio, 128 if is_indexer else 512)),
            norm=SimpleNamespace(
                weight=torch.empty(128 if is_indexer else 512),
                variance_epsilon=1e-6,
            ),
            freqs_cis=torch.empty((128, 32), dtype=torch.complex64),
            aiter_fp4_cos=(
                torch.empty((128, 32), dtype=torch.bfloat16) if is_indexer else None
            ),
            aiter_fp4_sin=(
                torch.empty((128, 32), dtype=torch.bfloat16) if is_indexer else None
            ),
        )
        backend = CompressorBackendMixin.__new__(CompressorBackendMixin)
        backend.token_to_kv_pool = token_pool
        backend.enable_deepseek_v4_fp4_indexer = enabled
        backend._get_out_loc = Mock(return_value=out_loc)
        backend._forward_compress_all_in_one = Mock()
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=Mock(return_value=False))
        )
        return backend, token_pool, compressor, forward_batch

    def test_forward_unified_fetches_split_buffers_before_legacy_accessor(self):
        backend, pool, compressor, forward_batch = self._make_forward_objects()

        with patch.object(compressor_v2, "_is_hip", True):
            backend.forward_unified(
                forward_batch=forward_batch,
                x=Mock(),
                layer_id=7,
                compressor=compressor,
            )

        pool.get_index_k_with_scale_buffer.assert_not_called()
        pool.get_index_k_fp4_payload_buffer.assert_called_once_with(7)
        pool.get_index_k_fp4_scale_buffer.assert_called_once_with(7)
        kwargs = backend._forward_compress_all_in_one.call_args.kwargs
        self.assertIs(
            kwargs["kv_cache"], pool.get_index_k_fp4_payload_buffer.return_value
        )
        self.assertIs(
            kwargs["kv_scale_cache"], pool.get_index_k_fp4_scale_buffer.return_value
        )
        self.assertEqual(kwargs["page_size"], 64)
        self.assertTrue(kwargs["use_fp4_indexer"])
        self.assertTrue(kwargs["use_aiter_fp4_indexer"])
        self.assertIs(kwargs["aiter_fp4_cos"], compressor.aiter_fp4_cos)
        self.assertIs(kwargs["aiter_fp4_sin"], compressor.aiter_fp4_sin)

    def test_forward_unified_keeps_cuda_fp4_and_hip_fp8_combined(self):
        for is_hip, enabled in ((False, True), (True, False)):
            with self.subTest(is_hip=is_hip, enabled=enabled):
                backend, pool, compressor, forward_batch = self._make_forward_objects(
                    enabled=enabled
                )
                with patch.object(compressor_v2, "_is_hip", is_hip):
                    backend.forward_unified(
                        forward_batch=forward_batch,
                        x=Mock(),
                        layer_id=7,
                        compressor=compressor,
                    )

                pool.get_index_k_with_scale_buffer.assert_called_once_with(7)
                pool.get_index_k_fp4_payload_buffer.assert_not_called()
                pool.get_index_k_fp4_scale_buffer.assert_not_called()
                kwargs = backend._forward_compress_all_in_one.call_args.kwargs
                self.assertIsNone(kwargs["kv_scale_cache"])
                self.assertFalse(kwargs["use_aiter_fp4_indexer"])
                self.assertEqual(kwargs["use_fp4_indexer"], enabled)

    def test_forward_unified_keeps_c128_extra_key_path(self):
        backend, pool, compressor, forward_batch = self._make_forward_objects(
            is_indexer=False, ratio=128, enabled=True
        )

        with (
            patch.object(compressor_v2, "_is_hip", True),
            patch(
                "sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate.is_unified_kv_triton",
                return_value=False,
            ),
        ):
            backend.forward_unified(
                forward_batch=forward_batch,
                x=Mock(),
                layer_id=7,
                compressor=compressor,
            )

        pool.get_extra_key_buffer.assert_called_once_with(7)
        pool.get_extra_key_page_size.assert_called_once_with(7)
        pool.get_index_k_with_scale_buffer.assert_not_called()
        kwargs = backend._forward_compress_all_in_one.call_args.kwargs
        self.assertEqual(kwargs["page_size"], 128)
        self.assertFalse(kwargs["use_fp4_indexer"])
        self.assertFalse(kwargs["use_aiter_fp4_indexer"])
        self.assertIsNone(kwargs["kv_scale_cache"])

    def test_all_in_one_routes_only_aiter_fp4_to_split_writer(self):
        plan = _decode_plan([4, 8])
        compressed = torch.randn((2, 128), dtype=torch.float32)
        payload = torch.empty((2, 1, 4, 64, 16), dtype=torch.float4_e2m1fn_x2)
        scale = torch.empty((2, 1, 4, 64), dtype=torch.uint8)
        out_loc = torch.tensor([3, 7], dtype=torch.int64)
        norm = SimpleNamespace(
            weight=torch.randn(128),
            variance_epsilon=1e-6,
            _aiter_fp4_weight_bf16=torch.randn(128, dtype=torch.bfloat16),
        )
        freqs_cis = torch.empty((128, 32), dtype=torch.complex64)
        cos = torch.empty((128, 32), dtype=torch.bfloat16)
        sin = torch.empty((128, 32), dtype=torch.bfloat16)
        backend = CompressorBackendMixin.__new__(CompressorBackendMixin)
        backend._get_paged_compress_metadata = Mock(return_value=plan)

        with (
            patch.object(
                compressor_v2, "compress_forward", return_value=compressed
            ) as compress,
            patch.object(compressor_v2, "compress_norm_rope_store") as legacy_store,
            patch.object(
                aiter_fp4_indexer, "aiter_k_indexer_fp4_cache_write"
            ) as aiter_store,
        ):
            backend._forward_compress_all_in_one(
                kv_score_buffer=torch.empty((2, 4, 512)),
                kv_score_input=torch.empty((2, 256)),
                ape=torch.empty((4, 128)),
                head_dim=128,
                norm=norm,
                freqs_cis_cache=freqs_cis,
                kv_cache=payload,
                kv_scale_cache=scale,
                is_indexer=True,
                rotate=True,
                compress_ratio=4,
                page_size=64,
                out_loc=out_loc,
                use_fp4_indexer=True,
                use_aiter_fp4_indexer=True,
                aiter_fp4_cos=cos,
                aiter_fp4_sin=sin,
            )

        legacy_store.assert_not_called()
        compress_out = compress.call_args.kwargs["out"]
        self.assertEqual(compress_out.shape, (2, 128))
        self.assertEqual(compress_out.dtype, torch.bfloat16)
        aiter_store.assert_called_once_with(
            k=compressed,
            norm_weight=norm._aiter_fp4_weight_bf16,
            norm_epsilon=norm.variance_epsilon,
            cos=cos,
            sin=sin,
            plan=plan,
            out_loc=out_loc,
            k_payload=payload,
            k_scale=scale,
        )

    def test_all_in_one_forwards_cached_aiter_write_metadata(self):
        plan = _decode_plan([4, 8])
        compressed = torch.randn((2, 128), dtype=torch.float32)
        payload = torch.empty((2, 1, 4, 64, 16), dtype=torch.float4_e2m1fn_x2)
        scale = torch.empty((2, 1, 4, 64), dtype=torch.uint8)
        write_metadata = (
            torch.tensor([0, 4], dtype=torch.int64),
            torch.tensor([3, 7], dtype=torch.int64),
        )
        backend = CompressorBackendMixin.__new__(CompressorBackendMixin)
        backend._get_paged_compress_metadata = Mock(return_value=plan)
        backend.forward_metadata = SimpleNamespace(
            aiter_fp4_k_write_metadata=write_metadata
        )

        with (
            patch.object(compressor_v2, "compress_forward", return_value=compressed),
            patch.object(
                aiter_fp4_indexer, "aiter_k_indexer_fp4_cache_write"
            ) as aiter_store,
        ):
            backend._forward_compress_all_in_one(
                kv_score_buffer=torch.empty((2, 4, 512)),
                kv_score_input=torch.empty((2, 256)),
                ape=torch.empty((4, 128)),
                head_dim=128,
                norm=SimpleNamespace(weight=torch.randn(128), variance_epsilon=1e-6),
                freqs_cis_cache=torch.empty((128, 32), dtype=torch.complex64),
                kv_cache=payload,
                kv_scale_cache=scale,
                is_indexer=True,
                rotate=True,
                compress_ratio=4,
                page_size=64,
                out_loc=torch.tensor([3, 7]),
                use_fp4_indexer=True,
                use_aiter_fp4_indexer=True,
                aiter_fp4_cos=torch.empty((128, 32), dtype=torch.bfloat16),
                aiter_fp4_sin=torch.empty((128, 32), dtype=torch.bfloat16),
            )

        self.assertIs(aiter_store.call_args.kwargs["write_metadata"], write_metadata)

    def test_all_in_one_keeps_legacy_fp4_and_fp8_store(self):
        plan = _decode_plan([4, 8])
        compressed = torch.randn((2, 128), dtype=torch.float32)
        combined = torch.empty((2, 4352), dtype=torch.uint8)
        norm = SimpleNamespace(weight=torch.randn(128), variance_epsilon=1e-6)
        backend = CompressorBackendMixin.__new__(CompressorBackendMixin)
        backend._get_paged_compress_metadata = Mock(return_value=plan)

        for use_fp4 in (True, False):
            with (
                self.subTest(use_fp4=use_fp4),
                patch.object(
                    compressor_v2, "compress_forward", return_value=compressed
                ),
                patch.object(compressor_v2, "compress_norm_rope_store") as legacy_store,
                patch.object(
                    aiter_fp4_indexer, "aiter_k_indexer_fp4_cache_write"
                ) as aiter_store,
            ):
                backend._forward_compress_all_in_one(
                    kv_score_buffer=torch.empty((2, 4, 512)),
                    kv_score_input=torch.empty((2, 256)),
                    ape=torch.empty((4, 128)),
                    head_dim=128,
                    norm=norm,
                    freqs_cis_cache=torch.empty((128, 32), dtype=torch.complex64),
                    kv_cache=combined,
                    kv_scale_cache=None,
                    is_indexer=True,
                    rotate=True,
                    compress_ratio=4,
                    page_size=64,
                    out_loc=torch.tensor([3, 7]),
                    use_fp4_indexer=use_fp4,
                    use_aiter_fp4_indexer=False,
                )

                aiter_store.assert_not_called()
                self.assertEqual(legacy_store.call_args.kwargs["use_fp4"], use_fp4)
                self.assertIs(legacy_store.call_args.kwargs["kvcache"], combined)


if __name__ == "__main__":
    unittest.main()
