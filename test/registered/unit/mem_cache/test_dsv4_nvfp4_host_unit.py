import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt import server_args as server_args_module
from sglang.srt.layers.attention import deepseek_v4_backend as dsv4_backend_module
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    _view_dsv4_nvfp4_cache,
)
from sglang.srt.layers.attention.dsv4.nvfp4_k_cache import (
    DSV4_NVFP4_BYTES_PER_TOKEN,
    DSV4_NVFP4_NOPE_DIM,
    DSV4_NVFP4_PACKED_NOPE_BYTES,
    DSV4_NVFP4_ROPE_BYTES,
    DSV4_NVFP4_ROPE_DIM,
    DSV4_NVFP4_SCALE_BYTES,
    dequantize_dsv4_nvfp4_k_cache_paged,
    quantize_dsv4_nvfp4_k_cache_into,
)
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    default_cuda_graph_config,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _dsv4_config():
    return SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])


class TestDSV4NVFP4HostUnit(unittest.TestCase):
    def test_layout_and_cpu_page_scatter_roundtrip(self):
        self.assertEqual(DSV4_NVFP4_PACKED_NOPE_BYTES, 224)
        self.assertEqual(DSV4_NVFP4_SCALE_BYTES, 28)
        self.assertEqual(DSV4_NVFP4_ROPE_BYTES, 128)
        self.assertEqual(DSV4_NVFP4_BYTES_PER_TOKEN, 380)

        page_size = 4
        cache = torch.full(
            (2, page_size * DSV4_NVFP4_BYTES_PER_TOKEN),
            0xA5,
            dtype=torch.uint8,
        )
        e2m1_values = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.bfloat16,
        )
        nope = e2m1_values.repeat(2, DSV4_NVFP4_NOPE_DIM // 8)
        rope = torch.arange(2 * DSV4_NVFP4_ROPE_DIM, dtype=torch.bfloat16).reshape(
            2, DSV4_NVFP4_ROPE_DIM
        )
        cache_k = torch.cat((nope, rope), dim=-1)

        quantize_dsv4_nvfp4_k_cache_into(
            cache_k,
            cache,
            torch.tensor([5, 0], dtype=torch.int32),
            page_size=page_size,
            global_scale=1.0,
        )

        rows = cache.view(-1, DSV4_NVFP4_BYTES_PER_TOKEN)
        self.assertTrue(torch.all(rows[1:5] == 0xA5))
        out = dequantize_dsv4_nvfp4_k_cache_paged(
            cache,
            torch.tensor([5, -1, 0, 8], dtype=torch.int32),
            page_size=page_size,
            global_scale=1.0,
        )
        torch.testing.assert_close(out[0, 0], cache_k[0], rtol=0, atol=0)
        torch.testing.assert_close(out[2, 0], cache_k[1], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(out[1]).item(), 0)
        self.assertEqual(torch.count_nonzero(out[3]).item(), 0)

    def test_nvfp4_cache_views_are_zero_copy_for_all_v4_page_sizes(self):
        for page_size in (256, 64, 2):
            with self.subTest(page_size=page_size):
                raw = torch.zeros(
                    (3, page_size * DSV4_NVFP4_BYTES_PER_TOKEN),
                    dtype=torch.uint8,
                )
                viewed = _view_dsv4_nvfp4_cache(raw, page_size)
                self.assertEqual(tuple(viewed.shape), (3, page_size, 1, 380))
                self.assertEqual(viewed.data_ptr(), raw.data_ptr())

        with self.assertRaisesRegex(ValueError, "requires"):
            _view_dsv4_nvfp4_cache(torch.zeros(2, 380, dtype=torch.uint8), 256)

    def test_fused_decode_dispatches_c0_c4_and_c128_sources(self):
        import sgl_kernel.flash_mla as flash_mla

        variants = (
            ("c0_flash", 0, None, 64, 0),
            ("c4_flash", 4, 64, 64, 512),
            ("c4_pro", 4, 64, 128, 1024),
            ("c128_flash", 128, 2, 64, 1024),
        )
        with (
            patch.object(
                dsv4_backend_module,
                "dequantize_dsv4_nvfp4_k_cache_paged",
                side_effect=AssertionError("decode must not dequantize to BF16"),
            ) as dequantize,
            patch.object(
                flash_mla,
                "flash_mla_sparse_fwd",
                side_effect=AssertionError("decode must not use sparse prefill"),
            ) as sparse_prefill,
        ):
            for name, compress_ratio, extra_page_size, h_q, extra_topk in variants:
                with self.subTest(name=name):
                    q = torch.zeros((2, 1, h_q, 512), dtype=torch.bfloat16)
                    swa_indices = torch.zeros((2, 1, 128), dtype=torch.int32)
                    swa_topk_lengths = torch.tensor([128, 96], dtype=torch.int32)
                    attn_sink = torch.zeros(h_q, dtype=torch.float32)
                    self._assert_fused_decode_dispatch(
                        q=q,
                        swa_indices=swa_indices,
                        swa_topk_lengths=swa_topk_lengths,
                        attn_sink=attn_sink,
                        compress_ratio=compress_ratio,
                        extra_page_size=extra_page_size,
                        extra_topk=extra_topk,
                    )

        dequantize.assert_not_called()
        sparse_prefill.assert_not_called()

    def _assert_fused_decode_dispatch(
        self,
        *,
        q: torch.Tensor,
        swa_indices: torch.Tensor,
        swa_topk_lengths: torch.Tensor,
        attn_sink: torch.Tensor,
        compress_ratio: int,
        extra_page_size: int | None,
        extra_topk: int,
    ) -> None:
        backend = object.__new__(DeepseekV4AttnBackend)
        backend.softmax_scale = 0.125
        backend.head_dim_v = 512
        expected_o = torch.full(q.shape, compress_ratio, dtype=torch.bfloat16)
        decode_fwd = Mock(return_value=(expected_o, torch.empty(0)))
        backend._dsv4_nvfp4_decode_fwd = decode_fwd

        swa_raw = torch.zeros((3, 256 * DSV4_NVFP4_BYTES_PER_TOKEN), dtype=torch.uint8)
        swa_scale = torch.tensor([1.25], dtype=torch.float32)
        extra_scale = torch.tensor([2.5], dtype=torch.float32)
        pool = SimpleNamespace(
            swa_page_size=256,
            get_swa_nvfp4_global_scale=lambda _layer_id: swa_scale,
            get_extra_key_page_size=lambda _layer_id: extra_page_size,
            get_extra_nvfp4_global_scale=lambda _layer_id: extra_scale,
        )

        extra_raw = None
        extra_indices = None
        extra_topk_lengths = None
        if extra_page_size is not None:
            extra_raw = torch.zeros(
                (5, extra_page_size * DSV4_NVFP4_BYTES_PER_TOKEN),
                dtype=torch.uint8,
            )
            extra_indices = torch.ones(
                (q.shape[0], q.shape[1], extra_topk), dtype=torch.int32
            )
            extra_topk_lengths = torch.tensor(
                [extra_topk, extra_topk // 2], dtype=torch.int32
            )

        flashmla_metadata = object()
        actual = backend._forward_nvfp4_sparse(
            q=q,
            layer_id=3,
            token_to_kv_pool=pool,
            swa_k_cache=swa_raw,
            swa_indices=swa_indices,
            swa_topk_lengths=swa_topk_lengths,
            extra_k_cache=extra_raw,
            extra_indices=extra_indices,
            extra_topk_lengths=extra_topk_lengths,
            attn_sink=attn_sink,
            flashmla_metadata=flashmla_metadata,
        )

        torch.testing.assert_close(actual, expected_o.squeeze(1))
        kwargs = decode_fwd.call_args.kwargs
        self.assertEqual(tuple(kwargs["k_cache"].shape), (3, 256, 1, 380))
        self.assertEqual(kwargs["k_cache"].data_ptr(), swa_raw.data_ptr())
        self.assertIs(kwargs["kv_global_scale"], swa_scale)
        self.assertIs(kwargs["tile_scheduler_metadata"], flashmla_metadata)
        self.assertIs(kwargs["topk_length"], swa_topk_lengths)
        self.assertIs(kwargs["attn_sink"], attn_sink)

        if extra_page_size is None:
            self.assertIsNone(kwargs["extra_k_cache"])
            self.assertIsNone(kwargs["extra_kv_global_scale"])
            self.assertIsNone(kwargs["extra_indices_in_kvcache"])
            self.assertIsNone(kwargs["extra_topk_length"])
        else:
            assert extra_raw is not None
            self.assertEqual(
                tuple(kwargs["extra_k_cache"].shape),
                (5, extra_page_size, 1, 380),
            )
            self.assertEqual(kwargs["extra_k_cache"].data_ptr(), extra_raw.data_ptr())
            self.assertIs(kwargs["extra_kv_global_scale"], extra_scale)
            self.assertIs(kwargs["extra_indices_in_kvcache"], extra_indices)
            self.assertIs(kwargs["extra_topk_length"], extra_topk_lengths)

    def test_flashmla_wrapper_caches_scheduler_outputs(self):
        import sgl_kernel.flash_mla as flash_mla

        q = torch.zeros((2, 1, 64, 512), dtype=torch.bfloat16)
        cache = torch.zeros((3, 256, 1, 380), dtype=torch.uint8)
        scale = torch.tensor([1.0], dtype=torch.float32)
        indices = torch.zeros((2, 1, 64), dtype=torch.int32)
        lengths = torch.full((2,), 64, dtype=torch.int32)
        sink = torch.zeros(64, dtype=torch.float32)
        extra_cache = torch.zeros((4, 64, 1, 380), dtype=torch.uint8)
        extra_scale = torch.tensor([2.0], dtype=torch.float32)
        extra_indices = torch.zeros((2, 1, 128), dtype=torch.int32)
        extra_lengths = torch.full((2,), 128, dtype=torch.int32)
        metadata_1 = torch.tensor([11], dtype=torch.int32)
        splits_1 = torch.tensor([12], dtype=torch.int32)
        metadata_2 = torch.tensor([21], dtype=torch.int32)
        splits_2 = torch.tensor([22], dtype=torch.int32)
        out = torch.zeros((2, 1, 64, 512), dtype=torch.bfloat16)
        lse = torch.zeros((2, 64, 1), dtype=torch.float32)
        op = Mock(
            side_effect=[
                (out, lse, metadata_1, splits_1),
                (out, lse, metadata_2, splits_2),
            ]
        )
        sched = flash_mla.FlashMLASchedMeta()

        with (
            patch.object(flash_mla, "_flashmla_import_error", None),
            patch.object(flash_mla, "_get_dsv4_nvfp4_decode_op", return_value=op),
        ):
            for _ in range(2):
                actual_out, actual_lse = flash_mla.flash_mla_with_kvcache_dsv4_nvfp4(
                    q,
                    cache,
                    scale,
                    indices,
                    lengths,
                    sink,
                    sched,
                    extra_k_cache=extra_cache,
                    extra_kv_global_scale=extra_scale,
                    extra_indices_in_kvcache=extra_indices,
                    extra_topk_length=extra_lengths,
                )
                self.assertIs(actual_out, out)
                self.assertIs(actual_lse, lse)

            with self.assertRaisesRegex(AssertionError, "inconsistent"):
                flash_mla.flash_mla_with_kvcache_dsv4_nvfp4(
                    q[:1],
                    cache,
                    scale,
                    indices[:1],
                    lengths[:1],
                    sink,
                    sched,
                    extra_k_cache=extra_cache,
                    extra_kv_global_scale=extra_scale,
                    extra_indices_in_kvcache=extra_indices[:1],
                    extra_topk_length=extra_lengths[:1],
                )

        self.assertEqual(sched.config.page_block_size, 256)
        self.assertEqual(sched.config.extra_page_block_size, 64)
        self.assertEqual(sched.config.topk, 64)
        self.assertEqual(sched.config.extra_topk, 128)
        first_args = op.call_args_list[0].args
        second_args = op.call_args_list[1].args
        self.assertIsNone(first_args[6])
        self.assertIsNone(first_args[7])
        self.assertIs(second_args[6], metadata_1)
        self.assertIs(second_args[7], splits_1)
        self.assertIs(sched.tile_scheduler_metadata, metadata_2)
        self.assertIs(sched.num_splits, splits_2)

    def test_missing_fused_op_fails_during_backend_initialization(self):
        class DummyNVFP4Pool:
            swa_page_size = 256
            dsv4_kv_cache_store_nvfp4 = True

        runner = SimpleNamespace(
            device="cpu",
            model_config=SimpleNamespace(head_dim=512, v_head_dim=512),
            page_size=256,
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((1, 1), dtype=torch.int32)
            ),
            token_to_kv_pool=DummyNVFP4Pool(),
            hisparse_coordinator=None,
        )
        with (
            patch(
                "sglang.srt.layers.attention.deepseek_v4_backend."
                "DeepSeekV4TokenToKVPool",
                DummyNVFP4Pool,
            ),
            patch(
                "sglang.srt.layers.attention.deepseek_v4_backend."
                "_load_dsv4_nvfp4_decode_fwd",
                side_effect=RuntimeError(
                    "does not provide dsv4_sparse_decode_fwd_nvfp4"
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "does not provide"),
        ):
            DeepseekV4AttnBackend(runner)

    def test_server_gate_accepts_only_sm90_nvfp4_page256(self):
        args = SimpleNamespace(
            kv_cache_dtype="fp4_e2m1",
            fp4_kv_cache_recipe="nvfp4",
            enable_hisparse=False,
            speculative_algorithm=None,
            enable_prefill_cp=False,
            page_size=256,
            disable_cuda_graph=False,
            disable_decode_cuda_graph=False,
            disable_prefill_cuda_graph=False,
            cuda_graph_config=default_cuda_graph_config(),
            get_model_config=lambda: SimpleNamespace(hf_config=_dsv4_config()),
        )
        with (
            patch.object(server_args_module, "is_cuda", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            ServerArgs._handle_kv4_compatibility(args)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.FULL)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)
        self.assertFalse(args.disable_cuda_graph)
        self.assertFalse(args.disable_decode_cuda_graph)
        self.assertTrue(args.disable_prefill_cuda_graph)

        # The DSV4 model override declares page_size=256 before the declaration
        # pipeline materializes it onto ServerArgs. The compatibility gate must
        # accept that resolved value even while the raw field is still None.
        args.page_size = None
        args._resolved_overrides = [("_deepseek_v4_overrides", {"page_size": 256})]
        args.disable_prefill_cuda_graph = False
        args.cuda_graph_config = default_cuda_graph_config()
        with (
            patch.object(server_args_module, "is_cuda", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
        ):
            ServerArgs._handle_kv4_compatibility(args)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.FULL)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)

        args._resolved_overrides = []
        args.page_size = 128
        with (
            patch.object(server_args_module, "is_cuda", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            self.assertRaisesRegex(ValueError, "page-size=256"),
        ):
            ServerArgs._handle_kv4_compatibility(args)

    def test_memory_sizing_does_not_reserve_disabled_nvfp4_prefill_graph(self):
        config = default_cuda_graph_config()
        config.decode.max_bs = 32
        config.prefill.backend = Backend.BREAKABLE
        args = SimpleNamespace(
            kv_cache_dtype="fp4_e2m1",
            fp4_kv_cache_recipe="nvfp4",
            enable_hisparse=False,
            speculative_algorithm=None,
            enable_prefill_cp=False,
            page_size=256,
            disable_cuda_graph=False,
            disable_decode_cuda_graph=False,
            disable_prefill_cuda_graph=False,
            cuda_graph_config=config,
            disaggregation_mode=None,
            get_model_config=lambda: SimpleNamespace(hf_config=_dsv4_config()),
            _resolved=lambda: SimpleNamespace(enable_dp_attention=False),
            use_mla_backend=lambda: True,
        )

        self.assertEqual(ServerArgs.reserve_for_graph_mb(args), 32 * 2 + 1536)
        ServerArgs._disable_dsv4_nvfp4_prefill_graph_for_memory_sizing(args)
        self.assertEqual(config.prefill.backend, Backend.DISABLED)
        self.assertEqual(ServerArgs.reserve_for_graph_mb(args), 32 * 2)
        self.assertTrue(args._dsv4_nvfp4_prefill_graph_was_enabled_for_sizing)

        # The formal gate still owns validation and emits the same warning even
        # though sizing has already projected the disabled prefill backend.
        with (
            patch.object(server_args_module, "is_cuda", return_value=True),
            patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)),
            self.assertLogs("sglang.srt.server_args", level="WARNING") as logs,
        ):
            ServerArgs._handle_kv4_compatibility(args)
        self.assertTrue(
            any("disabling prefill CUDA graphs" in message for message in logs.output)
        )
        self.assertFalse(args._dsv4_nvfp4_prefill_graph_was_enabled_for_sizing)

        # FP8 DeepSeek V4 keeps the existing graph configuration unchanged.
        args.kv_cache_dtype = "fp8_e4m3"
        config.prefill.backend = Backend.BREAKABLE
        ServerArgs._disable_dsv4_nvfp4_prefill_graph_for_memory_sizing(args)
        self.assertEqual(config.prefill.backend, Backend.BREAKABLE)


if __name__ == "__main__":
    unittest.main()
