import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.layers.attention.dsv4.torch_quant import (
    dequantize_dsv41_packed_main_kv,
    fake_quant_compressed_kv,
    quantize_dsv41_packed_main_kv,
)
from sglang.srt.mem_cache.dsv41_main_kv_layout import (
    DSV41_MAIN_KV_LAYOUT,
    PackedMainKVView,
    make_dsv41_packed_main_kv_spec,
    resolve_dsv41_main_kv_layout_specs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDSV41PackedMainKVLayout(CustomTestCase):
    def test_layout_identity_and_legacy_api_rejection(self):
        layout = KVLayout.parse("dsv41_main_kv_e2m1_block16_rope_bf16_v1")
        self.assertIs(layout, DSV41_MAIN_KV_LAYOUT)
        self.assertTrue(layout.is_fp4)
        self.assertTrue(layout.is_packed_main_kv)
        for attr in ("data_bytes", "scale_bytes", "tile_size", "page_align"):
            with (
                self.subTest(attr=attr),
                self.assertRaisesRegex(ValueError, "MainKVLayoutSpec"),
            ):
                getattr(layout, attr)
        with self.assertRaisesRegex(ValueError, "MainKVLayoutSpec"):
            layout.page_bytes(256)

    def test_spec_geometry(self):
        expected = {
            256: (57_344, 65_536, 98_304),
            128: (28_672, 32_768, 49_152),
        }
        for page_slots, (
            scale_offset,
            rope_offset,
            page_bytes,
        ) in expected.items():
            with self.subTest(page_slots=page_slots):
                spec = make_dsv41_packed_main_kv_spec(page_slots)
                self.assertIs(spec.layout_id, DSV41_MAIN_KV_LAYOUT)
                self.assertEqual(spec.version, 1)
                self.assertEqual(spec.payload_offset, 0)
                self.assertEqual(spec.scale_offset, scale_offset)
                self.assertEqual(spec.rope_offset, rope_offset)
                self.assertEqual(spec.page_bytes, page_bytes)
                self.assertEqual(spec.bytes_per_slot, 384)
                self.assertEqual(page_bytes % 4096, 0)
        with self.assertRaisesRegex(ValueError, "128 or 256"):
            make_dsv41_packed_main_kv_spec(64)

    def test_layout_option_resolution(self):
        self.assertIsNone(resolve_dsv41_main_kv_layout_specs("auto", 256))
        self.assertIsNone(resolve_dsv41_main_kv_layout_specs("flashmla_fp8", 256))
        specs = resolve_dsv41_main_kv_layout_specs("packed_fp4", 256)
        self.assertEqual(specs[1].page_slots, 256)
        self.assertEqual(specs[2].page_slots, 128)
        with self.assertRaisesRegex(ValueError, "unknown"):
            resolve_dsv41_main_kv_layout_specs("invalid", 256)

    def test_server_args_reserve_packed_layout_options(self):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(model_path="dummy")
        self.assertEqual(args.dsv41_main_kv_layout, "auto")
        self.assertEqual(args.dsv41_main_kv_consumer, "auto")

    def test_packed_layout_requires_explicit_direct_consumer(self):
        from sglang.srt.arg_groups.deepseek_v4_hook import (
            validate_deepseek_v41_features,
        )

        config = SimpleNamespace(
            dsv41_main_kv_layout="packed_fp4",
            dsv41_main_kv_consumer="auto",
        )
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        )
        with (
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.resolving_view",
                return_value=config,
            ),
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.model_config_of",
                return_value=model_config,
            ),
            self.assertRaisesRegex(ValueError, "currently requires.*direct"),
        ):
            validate_deepseek_v41_features(object())

    def test_sm90_direct_consumer_is_accepted(self):
        from sglang.srt.arg_groups.deepseek_v4_hook import (
            validate_deepseek_v41_features,
        )

        config = SimpleNamespace(
            dsv41_main_kv_layout="packed_fp4",
            dsv41_main_kv_consumer="direct",
            dsv4_attn_backend="flashmla",
            enable_encoder_swa_bounded_replay=False,
        )
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type="deepseek_v41")
        )
        with (
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.resolving_view",
                return_value=config,
            ),
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.model_config_of",
                return_value=model_config,
            ),
            patch(
                "sglang.srt.arg_groups.deepseek_v4_hook.get_platform",
                return_value=SimpleNamespace(is_sm90=True),
            ),
        ):
            validate_deepseek_v41_features(object())

    def test_views_share_storage(self):
        spec = make_dsv41_packed_main_kv_spec(128)
        storage = torch.zeros((2, spec.page_bytes), dtype=torch.uint8)
        view = PackedMainKVView(storage, spec)

        self.assertEqual(view.payload.shape, (2, 128, 224))
        self.assertEqual(view.scales.shape, (2, 128, 32))
        self.assertEqual(view.rope.shape, (2, 128, 64))
        self.assertEqual(view.rope.dtype, torch.bfloat16)
        self.assertEqual(view.payload.data_ptr(), storage.data_ptr())
        self.assertEqual(view.scales.data_ptr(), storage.data_ptr() + spec.scale_offset)
        self.assertEqual(view.rope.data_ptr(), storage.data_ptr() + spec.rope_offset)

        view.payload[1, 2, 3] = 17
        payload_offset = spec.page_bytes + 2 * 224 + 3
        self.assertEqual(int(storage.view(-1)[payload_offset]), 17)

        malformed = msgspec.structs.replace(spec, rope_offset=spec.rope_offset + 1)
        with self.assertRaisesRegex(ValueError, "canonical"):
            PackedMainKVView(storage, malformed)

    def test_reference_round_trip_matches_fake_quant(self):
        generator = torch.Generator().manual_seed(17)
        for page_slots in (128, 256):
            with self.subTest(page_slots=page_slots):
                values = torch.randn(
                    (1, page_slots, 512),
                    generator=generator,
                    dtype=torch.bfloat16,
                )
                pages = quantize_dsv41_packed_main_kv(values)
                actual = dequantize_dsv41_packed_main_kv(pages, page_slots)
                expected = fake_quant_compressed_kv(values)
                self.assertTrue(torch.equal(actual, expected))

                spec = make_dsv41_packed_main_kv_spec(page_slots)
                view = PackedMainKVView(pages, spec)
                self.assertEqual(int(view.scales[..., 28:].count_nonzero()), 0)

    def test_reference_nibble_order_and_signed_zero(self):
        values = torch.zeros((1, 128, 512), dtype=torch.bfloat16)
        magnitudes = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.bfloat16,
        )
        values[0, 0, :8] = magnitudes
        values[0, 0, 8:16] = -magnitudes
        values[0, 0, 16] = torch.tensor(-0.0, dtype=torch.bfloat16)
        values[0, 0, 17] = torch.tensor(-0.125, dtype=torch.bfloat16)
        values[0, 0, 31] = 6.0

        view = PackedMainKVView(
            quantize_dsv41_packed_main_kv(values),
            make_dsv41_packed_main_kv_spec(128),
        )
        self.assertEqual(view.payload[0, 0, :4].tolist(), [0x10, 0x32, 0x54, 0x76])
        self.assertEqual(view.payload[0, 0, 8].item() & 0xF, 0)
        self.assertEqual(view.payload[0, 0, 8].item() >> 4, 0x8)

    def test_reference_rejects_invalid_input(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            quantize_dsv41_packed_main_kv(torch.zeros((1, 128, 511)))
        with self.assertRaisesRegex(ValueError, "128 or 256"):
            quantize_dsv41_packed_main_kv(torch.zeros((1, 64, 512)))
        values = torch.zeros((1, 128, 512))
        values[0, 0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            quantize_dsv41_packed_main_kv(values)

    def test_pool_allocation_uses_physical_page_bytes(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV41PackedMainKVPool,
        )

        spec = make_dsv41_packed_main_kv_spec(256)
        pool = DeepSeekV41PackedMainKVPool(
            size=512,
            page_size=256,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            spec=spec,
        )
        expected_pages = (512 + 256 + 1) // 256
        self.assertEqual(pool.get_bytes_per_slot(), 384)
        self.assertEqual(pool.get_bytes_per_page(), 98_304)
        self.assertEqual(pool.get_kv_size_bytes(), 2 * expected_pages * 98_304)
        with self.assertRaisesRegex(NotImplementedError, "get_view"):
            pool.get_key_buffer(0)

    def test_outer_pool_preserves_source_sharing_and_typed_access(self):
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.server_args import (
            ServerArgs,
            set_global_server_args_for_scheduler,
        )

        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=256)
        )
        specs = {
            1: make_dsv41_packed_main_kv_spec(256),
            2: make_dsv41_packed_main_kv_spec(128),
        }
        pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=4,
            swa_size=512,
            c4_size=0,
            c128_size=0,
            c4_state_pool_size=0,
            c128_state_pool_size=0,
            page_size=256,
            swa_page_size=256,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=448,
            qk_rope_head_dim=64,
            indexer_head_dim=128,
            layer_num=5,
            device="cpu",
            enable_memory_saver=False,
            compression_ratios=[0, 2, 2, 1, 1],
            kv_source_layers=[1, 3],
            full_size=512,
            main_kv_layout_specs=specs,
        )

        c2_source = pool.get_extra_key_view(1)
        c2_reader = pool.get_extra_key_view(2)
        c1_source = pool.get_extra_key_view(3)
        c1_reader = pool.get_extra_key_view(4)
        self.assertEqual(c2_source.storage.data_ptr(), c2_reader.storage.data_ptr())
        self.assertEqual(c1_source.storage.data_ptr(), c1_reader.storage.data_ptr())
        self.assertEqual(c2_source.spec.page_slots, 128)
        self.assertEqual(c1_source.spec.page_slots, 256)
        self.assertIs(
            pool.get_extra_key_layout(1),
            KVLayout.DSV41_MAIN_KV_E2M1_BLOCK16_ROPE_BF16_V1,
        )
        with self.assertRaisesRegex(TypeError, "legacy FlashMLA"):
            pool.get_extra_key_buffer(1)
        with self.assertRaisesRegex(NotImplementedError, "transfer descriptors"):
            pool.get_contiguous_buf_infos()

        packed_pool = pool.kv_pools[2]
        with patch.object(packed_pool, "set_key_buffer_fused") as writer:
            loc = torch.tensor([1], dtype=torch.int64)
            latent = torch.zeros((1, 512), dtype=torch.bfloat16)
            freqs = torch.zeros((1, 64), dtype=torch.float32)
            pool.set_extra_key_buffer_fused(
                layer_id=1,
                loc=loc,
                cache_k=latent,
                freqs_cis=freqs,
            )
            writer.assert_called_once_with(0, loc, latent, freqs)

    def test_physical_sizing_uses_page_stride(self):
        from sglang.srt.model_executor.pool_configurator import (
            get_dsv4_main_kv_physical_bytes_per_slot,
        )

        self.assertEqual(
            get_dsv4_main_kv_physical_bytes_per_slot(
                ratio=1,
                full_page_size=256,
                kv_layout=KVLayout.V4,
                compressed_kv_layout=None,
                packed_specs=None,
            ),
            585,
        )
        specs = {
            1: make_dsv41_packed_main_kv_spec(256),
            2: make_dsv41_packed_main_kv_spec(128),
        }
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                self.assertEqual(
                    get_dsv4_main_kv_physical_bytes_per_slot(
                        ratio=ratio,
                        full_page_size=256,
                        kv_layout=KVLayout.V4,
                        compressed_kv_layout=None,
                        packed_specs=specs,
                    ),
                    384,
                )


if __name__ == "__main__":
    unittest.main()
