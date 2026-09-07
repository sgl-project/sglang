import os
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
from sglang.srt.mem_cache.kvbit_dsv4 import (
    DSV4_INT4_LAYOUT,
    DSV4KVBitPackedSWAPool,
    DSV4KVBitRuntimeCapability,
    decode_dsv4_int4_reference,
    dsv4_kvbit_enabled_for_worker,
    dsv4_kvbit_sparse_decode,
    encode_dsv4_int4_reference,
    merge_attention_states_natural_log,
    require_dsv4_kvbit_runtime_capability,
    validate_dsv4_int4_geometry,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestDSV4INT4Layout(CustomTestCase):
    def test_layout_is_exactly_368_bytes(self):
        self.assertEqual(
            DSV4_INT4_LAYOUT.offsets(),
            {
                "codes": (0, 224),
                "header": (224, 240),
                "rope": (240, 368),
            },
        )
        self.assertEqual(DSV4_INT4_LAYOUT.row_bytes, 368)
        self.assertEqual(DSV4_INT4_LAYOUT.group_size, 32)

    def test_reference_codec_round_trip_and_padding(self):
        torch.manual_seed(6)
        kv = torch.randn(3, 512, dtype=torch.bfloat16) * 0.1
        packed = encode_dsv4_int4_reference(kv)
        decoded = decode_dsv4_int4_reference(packed)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(tuple(packed.shape), (3, 368))
        self.assertTrue(torch.equal(packed[:, 238:240], torch.zeros((3, 2))))
        self.assertEqual(
            tuple(packed[:, 224:238].contiguous().view(torch.float8_e4m3fn).shape),
            (3, 14),
        )
        self.assertTrue(torch.equal(decoded[:, 448:], kv[:, 448:]))
        torch.testing.assert_close(decoded.float(), kv.float(), atol=0.04, rtol=0.2)

    def test_reference_codec_uses_g32_e4m3_rne_signed_range_and_zero_scale(self):
        stored = torch.zeros(2, 512, dtype=torch.float32)
        stored[0, :10] = torch.tensor(
            [-7.0, -1.0, 0.0, 1.0, 7.0, -6.5, -5.5, 0.5, 1.5, 2.5]
        )
        stored[0, 32:64] = 14.0
        stored[0, 64:96] = 9.0
        packed = encode_dsv4_int4_reference(stored)
        low = packed[0, :5] & 0x0F
        high = packed[0, :5] >> 4
        scales = (
            packed[:, 224:238].contiguous().view(torch.float8_e4m3fn).reshape(2, 14)
        )
        self.assertEqual(low.tolist(), [9, 0, 7, 10, 2])
        self.assertEqual(high.tolist(), [15, 1, 10, 0, 2])
        nibbles = torch.stack((packed[0, :224] & 0x0F, packed[0, :224] >> 4))
        self.assertFalse(torch.any(nibbles == 8).item())
        self.assertEqual(scales[0, 0].item(), 1.0)
        self.assertEqual(scales[0, 1].item(), 2.0)
        self.assertEqual(scales[0, 2].item(), 1.25)
        self.assertEqual(packed[0, 226].item(), 0x3A)
        self.assertTrue(
            torch.equal(
                scales[1],
                torch.zeros(14, dtype=torch.float8_e4m3fn),
            )
        )
        self.assertTrue(
            torch.equal(
                decode_dsv4_int4_reference(packed[1:]),
                torch.zeros(1, 512, dtype=torch.bfloat16),
            )
        )

    def test_reference_codec_keeps_nope_in_original_domain(self):
        kv = torch.zeros(1, 512, dtype=torch.float32)
        kv[0, :32] = torch.arange(-16, 16, dtype=torch.float32)
        packed = encode_dsv4_int4_reference(kv)
        decoded = decode_dsv4_int4_reference(packed)
        scale = packed[0, 224:225].view(torch.float8_e4m3fn).float()
        expected_codes = torch.round(kv[0, :32] / scale).clamp(-7, 7)
        torch.testing.assert_close(
            decoded[0, :32].float(),
            expected_codes * scale,
            atol=0,
            rtol=0,
        )
        self.assertTrue(torch.equal(decoded[0, 32:], torch.zeros(480)))

    def test_reference_codec_rejects_invalid_shape_or_dtype(self):
        with self.assertRaisesRegex(ValueError, "last dimension must be 512"):
            encode_dsv4_int4_reference(torch.zeros(1, 511))
        with self.assertRaisesRegex(TypeError, "floating point"):
            encode_dsv4_int4_reference(torch.zeros(1, 512, dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "row width must be 368 or 384"):
            decode_dsv4_int4_reference(torch.zeros(1, 367, dtype=torch.uint8))

    def test_geometry_rejects_non_dsv4_shape(self):
        validate_dsv4_int4_geometry(448, 64)
        with self.assertRaisesRegex(ValueError, "448-nope/64-rope"):
            validate_dsv4_int4_geometry(512, 64)


class TestDSV4INT4Capability(CustomTestCase):
    def test_only_target_worker_is_eligible(self):
        self.assertTrue(
            dsv4_kvbit_enabled_for_worker(kv_cache_dtype="int4", is_draft_worker=False)
        )
        self.assertFalse(
            dsv4_kvbit_enabled_for_worker(kv_cache_dtype="int4", is_draft_worker=True)
        )
        for legacy in ("kvbit", "kvbit-mxint4", "kvbit-sint4-fp16step"):
            with self.subTest(legacy=legacy):
                self.assertFalse(
                    dsv4_kvbit_enabled_for_worker(
                        kv_cache_dtype=legacy, is_draft_worker=False
                    )
                )

    def test_target_int4_keeps_tag_and_resolves_backing_dtype_as_auto(self):
        tag, dtype = configure_kv_cache_dtype(
            server_args_kv_cache_dtype="int4",
            model=SimpleNamespace(quant_config=None),
            model_dtype=torch.bfloat16,
            is_draft_worker=False,
            is_dflash=False,
            speculative_draft_attention_backend="fa3",
        )
        self.assertEqual(tag, "int4")
        self.assertEqual(dtype, torch.bfloat16)

    def test_draft_defaults_to_fp8_instead_of_inheriting_int4(self):
        tag, dtype = configure_kv_cache_dtype(
            server_args_kv_cache_dtype="int4",
            model=SimpleNamespace(quant_config=None),
            model_dtype=torch.bfloat16,
            is_draft_worker=True,
            is_dflash=False,
            speculative_draft_attention_backend="fa3",
        )
        self.assertEqual(tag, "fp8_e4m3")
        self.assertEqual(dtype, torch.float8_e4m3fn)

    def test_capability_gate_rejects_missing_direct_operation(self):
        with self.assertRaisesRegex(RuntimeError, "scratch fallback is disabled"):
            require_dsv4_kvbit_runtime_capability(
                DSV4KVBitRuntimeCapability(
                    direct_packed_write=False,
                    direct_packed_decode=True,
                )
            )


class TestDSV4INT4PackedSWAPool(CustomTestCase):
    def _owner(self, *, enabled):
        owner = object.__new__(DeepSeekV4TokenToKVPool)
        owner.enable_kvbit_swa = enabled
        owner.qk_nope_head_dim = 448
        owner.qk_rope_head_dim = 64
        return owner

    def _pool_kwargs(self, *, page_size=8):
        return {
            "size": 9,
            "page_size": page_size,
            "dtype": torch.uint8,
            "layer_num": 2,
            "device": "cpu",
            "enable_memory_saver": False,
            "global_page_size": 256,
        }

    def test_factories_use_368_byte_rows_for_all_pool_kinds(self):
        owner = self._owner(enabled=True)
        swa = owner._make_swa_kv_pool(**self._pool_kwargs())
        c4 = owner._make_compressed_kv_pool(**self._pool_kwargs(page_size=64))
        c128 = owner._make_compressed_kv_pool(**self._pool_kwargs(page_size=2))
        for pool, page_size in ((swa, 256), (c4, 64), (c128, 2)):
            with self.subTest(page_size=page_size):
                self.assertIsInstance(pool, DSV4KVBitPackedSWAPool)
                self.assertEqual(pool.get_bytes_per_token(), 368)
                self.assertEqual(pool.bytes_per_page_padded, page_size * 368)

    def test_disabled_factories_stay_native(self):
        pool = self._owner(enabled=False)._make_swa_kv_pool(**self._pool_kwargs())
        self.assertIsInstance(pool, DeepSeekV4SingleKVPool)
        self.assertNotIsInstance(pool, DSV4KVBitPackedSWAPool)

    def test_aligned_factories_account_for_padding(self):
        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_DSV4_INT4_LAYOUT": "aos_384"}
        ):
            owner = self._owner(enabled=True)
            pools = (
                owner._make_swa_kv_pool(**self._pool_kwargs()),
                owner._make_compressed_kv_pool(**self._pool_kwargs(page_size=64)),
                owner._make_compressed_kv_pool(**self._pool_kwargs(page_size=2)),
            )
        for pool in pools:
            with self.subTest(page_size=pool.page_size):
                self.assertEqual(pool.get_bytes_per_token(), 384)
                self.assertEqual(pool.bytes_per_page_padded, pool.page_size * 384)
                self.assertEqual(
                    pool.kv_buffer[0].numel(), pool.num_pages * pool.page_size * 384
                )


class TestDSV4INT4Attention(CustomTestCase):
    def test_sparse_decode_matches_reference(self):
        torch.manual_seed(7)
        keys = torch.randn(3, 512, dtype=torch.bfloat16) * 0.1
        encoded = encode_dsv4_int4_reference(keys)
        packed = torch.zeros(2, 256 * 368, dtype=torch.uint8)
        physical_locs = torch.tensor([2, 255, 257], dtype=torch.int64)
        packed.view(-1, 368)[physical_locs] = encoded
        q = torch.randn(1, 1, 2, 512, dtype=torch.bfloat16) * 0.1
        indices = physical_locs.to(torch.int32).view(1, 1, -1)
        lengths = torch.tensor([3], dtype=torch.int32)
        output, lse = dsv4_kvbit_sparse_decode(
            q=q,
            packed=packed,
            indices=indices,
            lengths=lengths,
            attn_sink=None,
            page_size=256,
            softmax_scale=512**-0.5,
        )
        decoded = decode_dsv4_int4_reference(encoded).float()
        scores = torch.einsum("qhd,kd->qhk", q[0].float(), decoded) * (512**-0.5)
        expected = torch.einsum("qhk,kd->qhd", torch.softmax(scores, dim=-1), decoded)
        torch.testing.assert_close(output[0].float(), expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(
            lse[0],
            torch.logsumexp(scores, dim=-1).transpose(0, 1),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_int4_routes_all_ratios_to_single_flashmla_op(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DeepseekV4AttnBackend,
        )

        backend = object.__new__(DeepseekV4AttnBackend)
        backend.page_size = 256
        backend.softmax_scale = 512**-0.5
        backend.head_dim_v = 512
        indices = torch.zeros(1, 64, dtype=torch.int32)
        lengths = torch.ones(1, dtype=torch.int32)
        core = SimpleNamespace(
            swa_page_indices=indices,
            swa_topk_lengths=lengths,
            c4_sparse_page_indices=indices,
            c4_sparse_topk_lengths=lengths,
            c128_page_indices=indices,
            c128_topk_lengths_clamp1=lengths,
            get_flashmla_metadata=lambda _ratio: object(),
        )
        q = torch.zeros(1, 2, 512, dtype=torch.bfloat16)
        expected = torch.ones(1, 1, 2, 512, dtype=torch.bfloat16)
        packed = torch.zeros(1, 256 * 368, dtype=torch.uint8)

        for ratio, extra_page_size, row_bytes in (
            (0, None, 368),
            (4, 64, 368),
            (128, 2, 368),
            (0, None, 384),
            (4, 64, 384),
            (128, 2, 384),
        ):
            with self.subTest(compress_ratio=ratio, row_bytes=row_bytes):
                packed = torch.zeros(1, 256 * row_bytes, dtype=torch.uint8)
                extra = (
                    None
                    if extra_page_size is None
                    else torch.zeros(1, extra_page_size * row_bytes, dtype=torch.uint8)
                )
                backend.token_to_kv_pool = SimpleNamespace(
                    get_extra_key_buffer=lambda _layer_id: extra,
                    get_extra_key_page_size=lambda _layer_id: extra_page_size,
                )
                flashmla = MagicMock(return_value=(expected, torch.zeros(1, 2, 1)))
                module = ModuleType("sgl_kernel.kvbit_flash_mla")
                module.kvbit_int4_flash_mla_with_kvcache = flashmla
                package = ModuleType("sgl_kernel")
                package.kvbit_flash_mla = module
                with unittest.mock.patch.dict(
                    sys.modules,
                    {
                        "sgl_kernel": package,
                        "sgl_kernel.kvbit_flash_mla": module,
                    },
                ):
                    output = backend._forward_kvbit(
                        q=q,
                        layer_id=0,
                        compress_ratio=ratio,
                        packed_swa_cache=packed,
                        core_attn_metadata=core,
                        attn_sink=torch.zeros(2),
                    )
                torch.testing.assert_close(output, expected.squeeze(1))
                flashmla.assert_called_once()
                call = flashmla.call_args.kwargs
                self.assertEqual(tuple(call["packed_kcache"].shape), (256, row_bytes))

    def test_natural_log_lse_merge_matches_concatenated_attention(self):
        left_scores = torch.tensor([1.0, -2.0])
        right_scores = torch.tensor([20.0, 19.0])
        left_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        right_values = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        left_output = (torch.softmax(left_scores, dim=0) @ left_values).reshape(
            1, 1, 1, 2
        )
        right_output = (torch.softmax(right_scores, dim=0) @ right_values).reshape(
            1, 1, 1, 2
        )
        output, lse = merge_attention_states_natural_log(
            left_output,
            torch.logsumexp(left_scores, dim=0).reshape(1, 1, 1),
            right_output,
            torch.logsumexp(right_scores, dim=0).reshape(1, 1, 1),
        )
        all_scores = torch.cat((left_scores, right_scores))
        all_values = torch.cat((left_values, right_values))
        torch.testing.assert_close(
            output.reshape(-1), torch.softmax(all_scores, dim=0) @ all_values
        )
        torch.testing.assert_close(lse.reshape(()), torch.logsumexp(all_scores, dim=0))


if __name__ == "__main__":
    unittest.main()
