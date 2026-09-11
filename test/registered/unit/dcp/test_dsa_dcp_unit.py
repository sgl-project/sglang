"""DSA DCP address partitioning and gathered-prefix layout regressions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.layers.dcp.comm import all_gather_kv_cache_for_mha_extend
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    is_dcp_mla_decode_phase,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDSADCP(CustomTestCase):
    def test_sparse_partition_covers_each_virtual_location_once(self):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        # Include sentinels and locations well beyond the per-rank capacity.
        virtual = torch.tensor([[-1, 64, 129, 257], [519, 1022, 1023, -1]])
        for size in (1, 2, 4, 8):
            covered = torch.zeros_like(virtual)
            for rank in range(size):
                with get_parallel().override(attn_dcp_size=size, attn_dcp_rank=rank):
                    local = backend._dcp_localize_page_table(virtual)
                owned = local >= 0
                covered += owned
                torch.testing.assert_close(local[owned] * size + rank, virtual[owned])
                self.assertTrue((local[~owned] == -1).all())
                self.assertTrue((local[owned] < 1024 // size).all())
            torch.testing.assert_close(covered, (virtual >= 0).to(covered.dtype))

    def test_replicated_draft_skips_decode_collectives(self):
        with get_parallel().override(dcp_enabled=True, attn_dcp_size=4):
            for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                batch = SimpleNamespace(forward_mode=mode)
                self.assertTrue(is_dcp_mla_decode_phase(batch))
                self.assertFalse(is_dcp_mla_decode_phase(batch, is_dsa_draft=True))
            self.assertFalse(
                is_dcp_mla_decode_phase(
                    SimpleNamespace(forward_mode=ForwardMode.EXTEND)
                )
            )

    def test_only_target_extend_uses_ragged_indices(self):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend.dsa_kv_cache_store_fp8 = False
        for sharded in (False, True):
            backend._dcp_sharded_kv = sharded
            for mode in (
                ForwardMode.EXTEND,
                ForwardMode.MIXED,
                ForwardMode.DECODE,
                ForwardMode.TARGET_VERIFY,
            ):
                expected = (
                    TopkTransformMethod.RAGGED
                    if sharded and mode.is_extend_without_speculative()
                    else TopkTransformMethod.PAGED
                )
                self.assertEqual(backend.get_topk_transform_method(mode), expected)

    def test_decode_and_verify_pass_local_selections_and_request_lse(self):
        backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
        backend._dcp_sharded_kv = True
        backend.use_fused_topk = True
        backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
        backend.dsa_prefill_impl = backend.dsa_decode_impl = "tilelang"
        backend.dsa_index_kpool = 4
        backend.dsa_kv_cache_store_fp8 = False
        backend.use_mha = False
        backend.hisparse_coordinator = None
        backend.forward_metadata = SimpleNamespace()
        cache = torch.zeros(256, 1, 512, dtype=torch.bfloat16)
        backend.token_to_kv_pool = SimpleNamespace(get_key_buffer=lambda _: cache)
        backend._forward_tilelang = Mock(return_value=("output", "lse"))
        layer = SimpleNamespace(
            is_cross_attention=False,
            layer_id=3,
            tp_q_head_num=16,
            v_head_dim=512,
            head_dim=512,
            scaling=512**-0.5,
        )
        # Deliberately above the per-rank buffer size, including a KPool tail.
        indices = torch.tensor([[800, 801, 802, 803, -1]], dtype=torch.int32)
        q = torch.zeros(1, 16, 512, dtype=torch.bfloat16)
        with (
            get_parallel().override(attn_dcp_size=4, attn_dcp_rank=2),
            patch(
                "sglang.srt.layers.attention.dsa_backend.concat_mla_absorb_q_general",
                side_effect=lambda a, b: torch.cat([a, b], dim=-1),
            ),
        ):
            for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
                fn = (
                    backend.forward_decode
                    if mode.is_decode()
                    else backend.forward_extend
                )
                result = fn(
                    q,
                    None,
                    None,
                    layer,
                    SimpleNamespace(forward_mode=mode),
                    topk_indices=indices,
                )
                self.assertEqual(result, ("output", "lse"))
                kwargs = backend._forward_tilelang.call_args.kwargs
                self.assertTrue(kwargs["return_lse"])
                torch.testing.assert_close(
                    kwargs["page_table_1"],
                    torch.tensor([[-1, -1, 200, -1, -1]], dtype=torch.int32),
                )
        torch.testing.assert_close(
            indices, torch.tensor([[800, 801, 802, 803, -1]], dtype=torch.int32)
        )

    def test_norope_prefix_gather_orders_each_request_contiguously(self):
        prefix = torch.tensor([10, 11, 20], dtype=torch.bfloat16).view(3, 1, 1)
        extend = torch.tensor([12, 21, 22], dtype=torch.bfloat16).view(3, 1)
        pool = SimpleNamespace(get_mla_kv_buffer=lambda *a, **kw: (prefix, None))
        # Stub just the collective; exercise the request-wise prefix/extend
        # reassembly used by sparse RAGGED top-k with unequal prefix lengths.
        with patch(
            "sglang.srt.layers.dcp.comm.all_gather_kv_cache_for_dcp",
            return_value=prefix,
        ):
            kv, rope = all_gather_kv_cache_for_mha_extend(
                pool,
                None,
                torch.tensor([1, 2, 3]),
                torch.tensor([3, 3]),
                torch.tensor([2, 1]),
                [2, 1],
                torch.tensor([1, 2]),
                extend,
                extend.new_empty((3, 1, 0)),
            )
        torch.testing.assert_close(
            kv.flatten(), torch.tensor([10, 11, 12, 20, 21, 22], dtype=kv.dtype)
        )
        self.assertEqual(rope.shape, (6, 1, 0))

    def test_replicated_draft_writes_do_not_localize_twice(self):
        pool = MLATokenToKVPool.__new__(MLATokenToKVPool)
        with (
            patch("sglang.srt.mem_cache.memory_pool.set_mla_kv_buffer_triton") as raw,
            patch(
                "sglang.srt.mem_cache.memory_pool.set_mla_kv_buffer_dcp_sharded_triton"
            ) as sharded,
        ):
            for replicated in (False, True):
                raw.reset_mock()
                sharded.reset_mock()
                pool.dcp_replicated = replicated
                pool._scatter_mla_rows(None, None, None, None)
                self.assertEqual(raw.call_count, int(replicated))
                self.assertEqual(sharded.call_count, int(not replicated))


if __name__ == "__main__":
    unittest.main()
