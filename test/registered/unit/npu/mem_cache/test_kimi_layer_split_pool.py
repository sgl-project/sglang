"""CPU regressions for hybrid NPU pool selection and real buffer allocation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu import dsa_cache_layer_split as npu_split
from sglang.srt.hardware_backend.npu.dsa_cache_layer_split import (
    LayerSplitNPUDSATokenToKVPool,
)
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUDSATokenToKVPool
from sglang.srt.layers.attention.dsa import utils as dsa_utils
from sglang.srt.mem_cache import kv_cache_configurator
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestKimiLayerSplitPool(CustomTestCase):
    def _pool_class(self, *, rank=0, split=True, draft=False, npu=True):
        configurator = object.__new__(kv_cache_configurator.KVCacheConfigurator)
        configurator.use_mla_backend = True
        configurator.is_draft_worker = draft
        configurator.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                architectures=["KimiK31ForConditionalGeneration"], index_topk=8
            )
        )
        with (
            patch.object(kv_cache_configurator, "_is_npu", npu),
            get_parallel().override(
                enable_dsa_cache_layer_split=split, attn_cp_rank=rank, attn_cp_size=16
            ),
        ):
            return configurator._hybrid_full_attention_pool_class(
                mha_pool_class=object, mla_pool_class=object, dsa_pool_class=object
            )

    def _allocate(self, rank):
        # Invoke the configurator-selected constructor and real NPU buffer
        # allocator on CPU, including NPUDSA's skip-top-k translation.
        with (
            get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0),
            patch(
                "sglang.srt.hardware_backend.npu.memory_pool_npu.get_bool_env_var",
                return_value=False,
            ),
            patch(
                "sglang.srt.hardware_backend.npu.memory_pool_npu.envs."
                "SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD.get",
                return_value=False,
            ),
        ):
            return self._pool_class(rank=rank)(
                size=16,
                page_size=8,
                dtype=torch.bfloat16,
                kv_lora_rank=8,
                qk_rope_head_dim=4,
                layer_num=5,
                device="cpu",
                enable_memory_saver=False,
                index_head_dim=128,
                skip_topk_layers=[False, True, False, False, True],
            )

    def test_selected_constructor_allocates_owned_layers_and_indexer_slots(self):
        # Five Full Attention layers over PCP16 includes skipped indexer
        # layers and eleven ranks without any owned layer.
        for rank in (0, 1, 2, 3, 4, 5, 15):
            with self.subTest(rank=rank):
                pool = self._allocate(rank)
                self.assertIs(type(pool), LayerSplitNPUDSATokenToKVPool)
                self.assertEqual(
                    (pool.layer_shard_rank, pool.layer_shard_size), (rank, 16)
                )
                owns = rank < 5
                indexed = rank in (0, 2, 3)
                self.assertEqual(tuple(pool.k_buffer.shape), (int(owns), 3, 8, 1, 8))
                self.assertEqual(tuple(pool.v_buffer.shape), (int(owns), 3, 8, 1, 4))
                self.assertEqual(
                    tuple(pool.index_k_buffer.shape), (int(indexed), 3, 8, 1, 128)
                )
                self.assertEqual(pool.indexer_layer_ids, (rank,) if indexed else ())
                self.assertEqual(
                    pool.indexer_layer_id_to_slot, {rank: 0} if indexed else {}
                )
                expected_ids = ([rank, rank] if owns else []) + (
                    [rank] if indexed else []
                )
                self.assertEqual(pool.get_kv_layer_ids(), expected_ids)
                self.assertEqual(
                    len(pool.get_contiguous_buf_infos()[0]), len(expected_ids)
                )
                # Even empty owners need scratch for reading remote layers.
                self.assertEqual(tuple(pool.remote_k_buffer.shape), (3, 8, 1, 8))
                self.assertEqual(
                    tuple(pool.remote_index_k_buffer.shape), (3, 8, 1, 128)
                )

    def test_memory_report_includes_scratch_once_on_owned_and_empty_ranks(self):
        for rank, expected_bytes in ((0, 13440), (1, 7296), (15, 6720)):
            with self.subTest(rank=rank):
                pool = self._allocate(rank)
                # 24 padded rows; BF16 K/V width 8+4 and Indexer width 128.
                # Scratch always has both; rank 1 owns K/V but skips top-k.
                self.assertEqual(pool.get_kv_size_bytes(), expected_bytes)
                self.assertEqual(
                    pool.remote_k_buffer.untyped_storage().data_ptr(),
                    pool.remote_v_buffer.untyped_storage().data_ptr(),
                )

    def test_disabled_split_and_draft_keep_regular_npu_pool(self):
        for split, draft in ((False, False), (True, True)):
            with self.subTest(split=split, draft=draft):
                self.assertIs(
                    self._pool_class(split=split, draft=draft), NPUDSATokenToKVPool
                )

    def test_non_npu_keeps_supplied_dsa_pool(self):
        self.assertIs(self._pool_class(npu=False), object)

    def test_prefetched_prefix_and_recycled_chunk_survive_hybrid_write_and_read(self):
        child = self._allocate(15)
        pool = object.__new__(HybridLinearKVPool)
        pool.use_mla = True
        pool.start_layer = 0
        pool.full_kv_pool = child
        pool.full_attention_layer_id_mapping = {3: 0, 7: 1, 11: 2, 15: 3, 19: 4}
        events = []
        pool.layer_transfer_counter = SimpleNamespace(
            wait_until=lambda layer: events.append(("layer-ready", layer))
        )
        references = {
            "k": torch.full((24, 1, 8), 3, dtype=torch.bfloat16),
            "v": torch.full((24, 1, 4), 5, dtype=torch.bfloat16),
            "index": torch.full((24, 1, 128), 7, dtype=torch.bfloat16),
        }
        for buffer in (
            child.remote_k_buffer,
            child.remote_v_buffer,
            child.remote_index_k_buffer,
        ):
            buffer.fill_(-1)

        class Work:
            def __init__(self, tensor, kind):
                self.tensor, self.kind = tensor, kind

            def wait(self):
                events.append(("wait", self.kind))
                self.tensor.copy_(
                    references[self.kind]
                    .reshape(-1)[: self.tensor.numel()]
                    .view_as(self.tensor)
                )

        def broadcast(tensor, *, src, group, async_op):
            self.assertEqual(src, 2)  # Compact layer 2's owner in PCP16.
            self.assertEqual(group, "cp-group")
            self.assertTrue(async_op)
            if tensor.data_ptr() == child.remote_k_buffer.data_ptr():
                kind = "k"
            elif tensor.data_ptr() == child.remote_v_buffer.data_ptr():
                kind = "v"
            else:
                self.assertEqual(
                    tensor.data_ptr(), child.remote_index_k_buffer.data_ptr()
                )
                kind = "index"
            self.assertEqual(tensor.numel(), 20 * references[kind].shape[-1])
            events.append(("broadcast", kind))
            # Do not deliver bytes until the actual consumer waits on Work.
            return Work(tensor, kind)

        def scatter(destination, indices, updates):
            events.append(("scatter", destination.shape[-1]))
            destination.index_copy_(0, indices.flatten().long(), updates)

        prefix_rows = torch.tensor([16, 17, 18, 19])
        new_rows = torch.tensor([8, 9])  # Recycled below the cached prefix.
        batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0]),
            extend_prefix_lens_cpu=[4],
            out_cache_loc=new_rows,
            extend_seq_lens_cpu=[2],
        )
        ctx = ForwardContext(
            attn_backend=SimpleNamespace(
                token_to_kv_pool=pool,
                req_to_token_pool=SimpleNamespace(
                    req_to_token=prefix_rows.unsqueeze(0)
                ),
            )
        )
        new_k = torch.full((2, 1, 8), 11, dtype=torch.bfloat16)
        new_v = torch.full((2, 1, 4), 13, dtype=torch.bfloat16)
        new_index = torch.full((2, 1, 128), 17, dtype=torch.bfloat16)
        layer = SimpleNamespace(layer_id=11)
        with (
            forward_context(ctx),
            patch.object(dsa_utils, "is_npu", return_value=True),
            patch.object(dsa_utils, "dsa_use_prefill_cp", return_value=True),
            patch.object(
                dsa_utils,
                "get_memory",
                return_value=SimpleNamespace(disable_radix_cache=True),
            ),
            patch.object(
                npu_split,
                "get_parallel",
                return_value=SimpleNamespace(
                    attn_cp_group=SimpleNamespace(
                        ranks=list(range(16)), device_group="cp-group"
                    )
                ),
            ),
            patch.object(
                npu_split.torch.distributed, "broadcast", side_effect=broadcast
            ),
            patch.object(
                npu_split,
                "torch_npu",
                SimpleNamespace(npu_scatter_nd_update_=scatter),
                create=True,
            ),
        ):
            dsa_utils.maybe_prefetch_next_kimi_full_attention_kv(batch, 11)
            self.assertEqual(
                events,
                [
                    ("layer-ready", 11),
                    ("broadcast", "k"),
                    ("broadcast", "v"),
                    ("broadcast", "index"),
                ],
            )
            pool.set_kv_buffer(layer, new_rows, new_k, new_v, cp_gathered_chunk=True)
            child.set_index_k_buffer(2, new_rows, new_index, cp_gathered_chunk=True)
            k, v = pool.get_kv_buffer(11)
            index = child.get_index_k_buffer(2)

        for actual, kind, updates in (
            (k, "k", new_k),
            (v, "v", new_v),
            (index, "index", new_index),
        ):
            rows = actual.reshape(24, 1, -1)
            torch.testing.assert_close(rows[prefix_rows], references[kind][prefix_rows])
            torch.testing.assert_close(rows[new_rows], updates)
            self.assertTrue(torch.all(rows[20:] == -1))
        self.assertEqual(
            [e for e in events if e[0] == "broadcast"],
            [("broadcast", "k"), ("broadcast", "v"), ("broadcast", "index")],
        )
        for kind, width in (("k", 8), ("v", 4), ("index", 128)):
            self.assertLess(
                events.index(("wait", kind)), events.index(("scatter", width))
            )
        self.assertEqual(layer.layer_id, 11)


if __name__ == "__main__":
    unittest.main()
