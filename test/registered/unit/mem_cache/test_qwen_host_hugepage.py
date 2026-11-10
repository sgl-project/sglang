"""Qwen host slabs prefer 1 GiB hugepages and fall back when the pool cannot serve them.

The node choice is the part that is easy to get wrong: a slab under one page
must not consume a page, the GPU node wins only when it actually has room,
and a short pool selects nothing so the caller stays on ordinary pinned memory.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.mem_cache import host_hugepage
from sglang.srt.mem_cache.dsv41_host_placement import GIB
from sglang.srt.mem_cache.host_hugepage import (
    select_1g_hugepage_node,
    set_qwen_host_hugepage_preference,
    try_alloc_pinned_1g_hugepage,
)
from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator,
    alloc_with_host_register,
)
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbeddingShardIndices,
)
from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding
from sglang.srt.models.qwen4_exp_ple_table import allocate_ple_host_table
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _NullPolicy:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestHugepageNodeChoice(CustomTestCase):
    def test_gpu_node_wins_only_when_it_has_enough_pages(self):
        # Node 0 has more free pages, but the GPU node fits, so it is used.
        self.assertEqual(
            select_1g_hugepage_node(8 * GIB, gpu_node=1, free_by_node={0: 40, 1: 8}),
            1,
        )
        # GPU node is short; the only node that can hold the slab is used.
        self.assertEqual(
            select_1g_hugepage_node(8 * GIB, gpu_node=0, free_by_node={0: 4, 1: 190}),
            1,
        )

    def test_short_pool_and_sub_page_slabs_select_nothing(self):
        self.assertIsNone(
            select_1g_hugepage_node(8 * GIB, gpu_node=1, free_by_node={0: 4, 1: 7})
        )
        self.assertIsNone(
            select_1g_hugepage_node(8 * GIB, gpu_node=None, free_by_node={})
        )
        self.assertIsNone(
            select_1g_hugepage_node(GIB - 1, gpu_node=1, free_by_node={1: 190})
        )

    def test_byte_past_a_page_requires_the_next_page(self):
        self.assertIsNone(
            select_1g_hugepage_node(GIB + 1, gpu_node=None, free_by_node={1: 1})
        )
        self.assertEqual(
            select_1g_hugepage_node(GIB + 1, gpu_node=None, free_by_node={1: 2}),
            1,
        )


class TestHugepageFallback(CustomTestCase):
    def tearDown(self):
        set_qwen_host_hugepage_preference(False)
        super().tearDown()

    def test_short_pool_does_not_mmap(self):
        with (
            mock.patch.object(
                host_hugepage, "free_1g_pages_by_node", return_value={1: 1}
            ),
            mock.patch.object(host_hugepage, "_preferred_gpu_node", return_value=1),
            mock.patch.object(host_hugepage, "alloc_1g_hugepage") as alloc,
        ):
            got = try_alloc_pinned_1g_hugepage((2 * GIB,), torch.uint8, purpose="slab")
        self.assertIsNone(got)
        alloc.assert_not_called()

    def test_mmap_failure_returns_none(self):
        with (
            mock.patch.object(
                host_hugepage, "free_1g_pages_by_node", return_value={1: 4}
            ),
            mock.patch.object(host_hugepage, "_preferred_gpu_node", return_value=1),
            mock.patch.object(
                host_hugepage, "_bind_mempolicy", return_value=_NullPolicy()
            ),
            mock.patch.object(
                host_hugepage, "alloc_1g_hugepage", side_effect=OSError("ENOMEM")
            ),
        ):
            got = try_alloc_pinned_1g_hugepage((GIB,), torch.uint8, purpose="slab")
        self.assertIsNone(got)

    def test_disabled_env_does_not_mmap(self):
        with (
            envs.SGLANG_QWEN_HOST_HUGETLB.override(False),
            mock.patch.object(host_hugepage, "alloc_1g_hugepage") as alloc,
        ):
            got = try_alloc_pinned_1g_hugepage((GIB,), torch.uint8, purpose="slab")
        self.assertIsNone(got)
        alloc.assert_not_called()

    def test_successful_mapping_is_the_returned_tensor(self):
        mapped = torch.empty(4, dtype=torch.uint8)

        def _register(tensor, registration_granularity_bytes=None, **kwargs):
            tensor._sglang_cuda_host_registered_ranges = []

        with (
            mock.patch.object(
                host_hugepage, "free_1g_pages_by_node", return_value={1: 4}
            ),
            mock.patch.object(host_hugepage, "_preferred_gpu_node", return_value=1),
            mock.patch.object(
                host_hugepage, "_bind_mempolicy", return_value=_NullPolicy()
            ),
            mock.patch.object(host_hugepage, "alloc_1g_hugepage", return_value=mapped),
            mock.patch(
                "sglang.srt.mem_cache.pool_host.common._cuda_host_register",
                side_effect=_register,
            ),
        ):
            got = try_alloc_pinned_1g_hugepage((GIB,), torch.uint8, purpose="slab")
        self.assertIs(got, mapped)
        self.assertEqual(mapped._sglang_hugetlb_node, 1)

    def test_ple_pinned_uses_hugepage_or_ordinary_pin(self):
        huge = torch.empty(2, dtype=torch.uint8)
        ordinary = torch.empty(2, dtype=torch.uint8)
        with (
            mock.patch(
                "sglang.srt.models.qwen4_exp_ple_table.try_alloc_pinned_1g_hugepage",
                return_value=huge,
            ),
            mock.patch.object(torch, "empty", return_value=ordinary) as empty,
        ):
            got = allocate_ple_host_table((8, 8), torch.bfloat16, "pinned")
        self.assertIs(got, huge)
        empty.assert_not_called()

        with (
            mock.patch(
                "sglang.srt.models.qwen4_exp_ple_table.try_alloc_pinned_1g_hugepage",
                return_value=None,
            ),
            mock.patch.object(torch, "empty", return_value=ordinary) as empty,
        ):
            got = allocate_ple_host_table((8, 8), torch.bfloat16, "pinned")
        self.assertIs(got, ordinary)
        self.assertTrue(empty.call_args.kwargs["pin_memory"])

    def test_parameter_keeps_hugepage_tensor_registered(self):
        host = torch.empty((4, 4), dtype=torch.bfloat16)
        host._sglang_hugetlb_node = 1
        shard = VocabParallelEmbeddingShardIndices(
            padded_org_vocab_start_index=0,
            padded_org_vocab_end_index=4,
            padded_added_vocab_start_index=4,
            padded_added_vocab_end_index=4,
            org_vocab_start_index=0,
            org_vocab_end_index=4,
            added_vocab_start_index=4,
            added_vocab_end_index=4,
        )
        source = SimpleNamespace(
            weight=nn.Parameter(
                torch.empty((4, 4), dtype=torch.bfloat16), requires_grad=False
            ),
            quant_config=None,
            enable_tp=True,
            use_attn_tp_group=False,
            tp_size=1,
            num_embeddings=4,
            org_vocab_size=4,
            padding_size=1,
            num_added_embeddings=0,
            use_presharded_weights=False,
            org_vocab_size_padded=4,
            num_embeddings_padded=4,
            shard_indices=shard,
            embedding_dim=4,
            weight_scale=None,
            quant_method=UnquantizedEmbeddingMethod(),
            num_embeddings_per_partition=4,
            num_org_embeddings_per_partition=4,
            num_added_embeddings_per_partition=0,
        )
        with mock.patch(
            "sglang.srt.models.qwen4_exp.allocate_ple_host_table",
            return_value=host,
        ):
            offloaded = Qwen4ExpPinnedHostEmbedding(source)
        self.assertIs(offloaded.weight._sglang_hugepage_owner, host)

    def test_hicache_uses_hugepage_only_for_qwen_and_falls_back(self):
        ordinary = torch.empty(4, dtype=torch.uint8)
        huge = torch.empty(4, dtype=torch.uint8)
        allocator = HostTensorAllocator()
        set_qwen_host_hugepage_preference(False)
        with (
            mock.patch.object(allocator, "allocate", return_value=ordinary) as allocate,
            mock.patch(
                "sglang.srt.mem_cache.pool_host.common._cuda_host_register"
            ) as register,
            mock.patch(
                "sglang.srt.mem_cache.host_hugepage.try_alloc_pinned_1g_hugepage",
                return_value=huge,
            ) as try_huge,
        ):
            got = alloc_with_host_register(
                (4,), torch.uint8, "cpu", True, allocator
            )
            self.assertIs(got, ordinary)
            try_huge.assert_not_called()
            allocate.assert_called_once()
            register.assert_called_once()

            set_qwen_host_hugepage_preference(True)
            got = alloc_with_host_register(
                (4,), torch.uint8, "cpu", True, allocator
            )
            self.assertIs(got, huge)

            try_huge.return_value = None
            got = alloc_with_host_register(
                (4,), torch.uint8, "cpu", True, allocator
            )
            self.assertIs(got, ordinary)
            self.assertEqual(allocate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
