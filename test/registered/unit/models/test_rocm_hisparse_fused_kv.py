"""Exercise the ROCm writer's slot contract with a CPU kernel boundary."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import dsa_backend
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DSAMetadata,
)
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_attn_backend,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.models.deepseek_common.attention_forward_methods import forward_mla_rocm
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_device_sm, is_blackwell
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRocmHiSparseFusedKV(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.cache = object()
        self.backend = self.make_backend(self.make_pool(DSATokenToKVPool))
        self.attn = SimpleNamespace(
            kv_cache_dtype="bfloat16",
            current_attention_backend="dsa",
            attn_mqa=SimpleNamespace(layer_id=7, k_scale=1.0),
            rotary_emb=SimpleNamespace(
                cos_cache=None, sin_cache=None, is_neox_style=False
            ),
        )

    @property
    def pool(self):
        return self.backend.token_to_kv_pool

    def make_pool(self, pool_type):
        # Supply storage without allocating a model's cache; keep real accessors.
        pool = pool_type.__new__(pool_type)
        pool.kv_buffer = [self.cache]
        pool.start_layer = 7
        pool.layer_transfer_counter = None
        pool.dtype = pool.store_dtype = torch.bfloat16
        return pool

    def use_hisparse(self):
        self.backend = self.make_backend(self.make_pool(HiSparseDSATokenToKVPool))
        mapping = torch.zeros(65, dtype=torch.int64)
        mapping[17], mapping[18], mapping[-1] = 3, 5, -1
        self.pool.register_mapping(mapping)
        return mapping

    def invoke(self, locations, forward_batch=None):
        if forward_batch is None:
            forward_batch = SimpleNamespace(
                out_cache_loc=locations, forward_mode=ForwardMode.EXTEND
            )
        with (
            forward_context(ForwardContext(attn_backend=self.backend)),
            patch.object(
                forward_mla_rocm,
                "fused_qk_rope_cat_and_cache_mla",
                lambda *args, **kwargs: args,
                create=True,
            ),
        ):
            args = forward_mla_rocm._fused_rope_cat_and_cache(
                self.attn,
                torch.empty(0),
                None,
                None,
                None,
                None,
                forward_batch,
            )
        self.assertIs(args[4], self.cache)
        return args[5]

    def make_backend(self, pool=None):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.token_to_kv_pool = pool if pool is not None else self.pool
        backend.req_to_token_pool = None
        backend.kv_index_translator = None
        backend.forward_metadata = self.make_metadata()
        return backend

    @staticmethod
    def make_metadata():
        lengths = torch.ones(2, dtype=torch.int32)
        offsets = torch.arange(3, dtype=torch.int32)
        page_table = torch.zeros((2, 1), dtype=torch.int32)
        return DSAMetadata(
            page_size=1,
            cache_seqlens_int32=lengths,
            max_seq_len_q=1,
            max_seq_len_k=1,
            cu_seqlens_q=offsets,
            cu_seqlens_k=offsets,
            page_table_1=page_table,
            real_page_table=page_table,
            dsa_cache_seqlens_int32=lengths,
            dsa_cu_seqlens_q=offsets,
            dsa_cu_seqlens_k=offsets,
            dsa_extend_seq_lens_list=[1, 1],
            dsa_seqlens_expanded=lengths,
        )

    def prepare(self, batch, backend=None):
        backend = backend or self.backend
        with (
            patch.object(dsa_backend, "_is_hip", True),
            patch.object(dsa_backend, "_IS_GFX95", True),
        ):
            backend.init_forward_metadata_in_graph(batch)

    def test_metadata_prepares_once_before_writes_and_refreshes_next_batch(self):
        mapping = self.use_hisparse()
        locations = torch.tensor([17, -1], dtype=torch.int64)
        batch = SimpleNamespace(
            out_cache_loc=locations, forward_mode=ForwardMode.DECODE
        )
        with patch.object(
            self.pool,
            "translate_loc_to_hisparse_device",
            wraps=self.pool.translate_loc_to_hisparse_device,
        ) as translate:
            self.prepare(batch)
            translate.assert_called_once()
            first_metadata = self.backend.forward_metadata
            outputs = [self.invoke(locations, batch), self.invoke(locations, batch)]
            translate.assert_called_once()
            self.assertIs(outputs[0], outputs[1])
            torch.testing.assert_close(outputs[0], torch.tensor([3, -1]))
            torch.testing.assert_close(locations, torch.tensor([17, -1]))

            # Slot assignment and logical input contents change before preparation.
            mapping[17] = 5
            locations[1] = 18
            self.backend.forward_metadata = self.make_metadata()
            self.prepare(batch)
            self.assertEqual(translate.call_count, 2)
            outputs = [self.invoke(locations, batch), self.invoke(locations, batch)]
            self.assertIs(outputs[0], outputs[1])
            torch.testing.assert_close(outputs[0], torch.tensor([5, 5]))
            torch.testing.assert_close(
                first_metadata.kv_write_locations, torch.tensor([3, -1])
            )

    def test_eager_runner_prepares_fresh_metadata_before_writing(self):
        mapping = self.use_hisparse()
        backend = self.backend
        backend.real_page_size = backend.dsa_index_kpool = 1
        backend.dsa_index_topk = 8
        backend.dsa_kv_cache_store_fp8 = False
        backend.dsa_decode_impl = backend.dsa_prefill_impl = "triton"
        backend.dsa_topk_backend = dsa_backend.DSATopKBackend.TORCH
        backend.enable_auto_select_prefill_impl = False
        backend._arange_buf = torch.arange(3, dtype=torch.int32)
        backend._is_in_breakable_cuda_graph = is_in_breakable_cuda_graph
        backend._is_in_tc_piecewise_cuda_graph = is_in_tc_piecewise_cuda_graph
        backend._get_device_sm = get_device_sm
        backend._is_blackwell = is_blackwell
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor([[17, 18]])
        )
        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.tensor([1]),
            positions=torch.tensor([1]),
            seq_lens=torch.tensor([2], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([2], dtype=torch.int32),
            seq_lens_sum=2,
            req_pool_indices=torch.tensor([0]),
            out_cache_loc=torch.tensor([18]),
        )

        def model_forward(input_ids, positions, forward_batch):
            # Check the context published by the runner, not a patched getter.
            self.assertIs(get_attn_backend(), backend)
            self.assertIsNotNone(backend.forward_metadata.kv_write_locations)
            result = forward_mla_rocm._fused_rope_cat_and_cache(
                self.attn, torch.empty(0), None, None, None, positions, forward_batch
            )
            self.assertIs(result[4], self.cache)
            return result[5]

        runner = EagerRunner.__new__(EagerRunner)
        # PDmux's eager path accepts an already-loaded batch and publishes its
        # decode backend. Exercise that real path without a server or registry.
        runner.enable_pdmux = True
        runner.model_runner = SimpleNamespace(
            decode_attn_backend=backend,
            model=SimpleNamespace(forward=model_forward),
            device_timer=None,
            _pp_kwargs=lambda _: {},
        )
        backend.forward_metadata = None
        with (
            patch.object(
                forward_mla_rocm,
                "fused_qk_rope_cat_and_cache_mla",
                lambda *args, **kwargs: args,
                create=True,
            ),
            get_parallel().override(attn_cp_size=1),
            patch.object(dsa_backend, "_is_hip", True),
            patch.object(dsa_backend, "_IS_GFX95", True),
        ):
            torch.testing.assert_close(runner._execute_decode(batch), torch.tensor([5]))
            first = backend.forward_metadata
            mapping[18] = 6
            torch.testing.assert_close(runner._execute_decode(batch), torch.tensor([6]))
            self.assertIsNot(backend.forward_metadata, first)
            torch.testing.assert_close(first.kv_write_locations, torch.tensor([5]))

    def test_modes_without_prepared_locations_translate_in_the_backend(self):
        self.use_hisparse()
        locations = torch.tensor([17, -1])
        for mode in (
            ForwardMode.DECODE,
            ForwardMode.EXTEND,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            with self.subTest(mode=mode):
                batch = SimpleNamespace(out_cache_loc=locations, forward_mode=mode)
                self.backend.forward_metadata = self.make_metadata()
                if mode != ForwardMode.DECODE:
                    self.prepare(batch)
                self.assertIsNone(self.backend.forward_metadata.kv_write_locations)
                with patch.object(
                    self.pool,
                    "translate_loc_to_hisparse_device",
                    wraps=self.pool.translate_loc_to_hisparse_device,
                ) as translate:
                    for _ in range(2):
                        torch.testing.assert_close(
                            self.invoke(locations, batch), torch.tensor([3, -1])
                        )
                    self.assertEqual(translate.call_count, 2)

    def test_default_backend_resolves_hisparse_without_prepared_metadata(self):
        # The same fused writer is also used by backends without DSA metadata.
        self.use_hisparse()
        batch = SimpleNamespace(out_cache_loc=torch.tensor([17, -1]))
        backend = SimpleNamespace(token_to_kv_pool=self.pool)
        torch.testing.assert_close(
            AttentionBackend.get_kv_write_locations(backend, batch),
            torch.tensor([3, -1]),
        )

    def test_writer_requires_initialized_attention_metadata(self):
        self.use_hisparse()
        self.backend.forward_metadata = None
        with self.assertRaisesRegex(AssertionError, "metadata must be initialized"):
            self.invoke(torch.tensor([17]))

    def test_metadata_preparation_leaves_other_pools_and_devices_unchanged(self):
        batch = SimpleNamespace(
            out_cache_loc=torch.tensor([17]), forward_mode=ForwardMode.DECODE
        )
        self.prepare(batch)
        self.assertIs(
            self.backend.forward_metadata.kv_write_locations, batch.out_cache_loc
        )
        self.use_hisparse()
        for hip, gfx95 in ((False, False), (True, False)):
            with (
                self.subTest(hip=hip, gfx95=gfx95),
                patch.object(dsa_backend, "_is_hip", hip),
                patch.object(dsa_backend, "_IS_GFX95", gfx95),
            ):
                self.backend.init_forward_metadata_in_graph(batch)
                self.assertIsNone(self.backend.forward_metadata.kv_write_locations)

    def test_tbo_children_prepare_their_own_slices(self):
        self.use_hisparse()
        primary, left, right = [self.make_backend() for _ in range(3)]
        wrapper = TboAttnBackend(primary, [left, right])
        locations = torch.tensor([17, -1, 18, 17])
        children = [
            SimpleNamespace(
                out_cache_loc=locations[sli],
                forward_mode=ForwardMode.DECODE,
                batch_size=2,
            )
            for sli in (slice(0, 2), slice(2, 4))
        ]
        parent = SimpleNamespace(
            out_cache_loc=locations,
            forward_mode=ForwardMode.DECODE,
            tbo_children=children,
        )
        self.prepare(parent, wrapper)
        self.assertIs(
            wrapper.get_kv_write_locations(parent),
            primary.forward_metadata.kv_write_locations,
        )
        for backend, child, expected in (
            (left, children[0], [3, -1]),
            (right, children[1], [5, 3]),
        ):
            self.backend = backend  # The TBO forward context selects this child.
            torch.testing.assert_close(
                self.invoke(child.out_cache_loc, child), torch.tensor(expected)
            )
            self.assertIsNot(
                backend.forward_metadata.kv_write_locations,
                primary.forward_metadata.kv_write_locations,
            )

    def test_hybrid_dispatch_reads_the_selected_backends_metadata(self):
        self.use_hisparse()
        decode, prefill = self.make_backend(), self.make_backend()
        wrapper = HybridAttnBackend.__new__(HybridAttnBackend)
        wrapper.decode_backend, wrapper.prefill_backend = decode, prefill
        decode_batch = SimpleNamespace(
            out_cache_loc=torch.tensor([17]), forward_mode=ForwardMode.DECODE
        )
        prefill_batch = SimpleNamespace(
            out_cache_loc=torch.tensor([18]), forward_mode=ForwardMode.EXTEND
        )
        self.prepare(decode_batch, wrapper)
        self.prepare(prefill_batch, wrapper)
        self.assertIs(
            wrapper.get_kv_write_locations(decode_batch),
            decode.forward_metadata.kv_write_locations,
        )
        torch.testing.assert_close(
            wrapper.get_kv_write_locations(prefill_batch), torch.tensor([5])
        )

    def test_resident_locations_are_unchanged(self):
        locations = torch.tensor([17, 0, -1], dtype=torch.int64)
        self.assertIs(self.invoke(locations), locations)

    def test_resident_strided_locations(self):
        """Strided locations must not send interleaved storage values to AITER."""
        storage = torch.tensor([3, 21, 5, 22, -1, 23])
        locations = storage[::2]
        actual = self.invoke(locations)
        with self.subTest(layout="strided"):
            self.assertTrue(actual.is_contiguous())
            torch.testing.assert_close(actual, torch.tensor([3, 5, -1]))
            torch.testing.assert_close(storage, torch.tensor([3, 21, 5, 22, -1, 23]))

    def test_hisparse_maps_logical_slots(self):
        """Logical slots beyond device capacity must write their physical rows."""
        self.use_hisparse()
        locations = torch.tensor([17, 18], dtype=torch.int64)
        with self.subTest(logical_slots=[17, 18]):
            torch.testing.assert_close(self.invoke(locations), torch.tensor([3, 5]))
            torch.testing.assert_close(locations, torch.tensor([17, 18]))

    def test_hisparse_padding_and_unmapped_slots(self):
        self.use_hisparse()
        locations = torch.tensor([17, -1, 19, 0])
        with self.subTest(padding=-1, unmapped=19):
            torch.testing.assert_close(
                self.invoke(locations), torch.tensor([3, -1, 0, 0])
            )
            torch.testing.assert_close(locations, torch.tensor([17, -1, 19, 0]))


if __name__ == "__main__":
    unittest.main()
