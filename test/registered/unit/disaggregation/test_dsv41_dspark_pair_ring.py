"""Validate DSpark PD pair-ring addressing and request-slot reuse."""

import unittest
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.utils import get_dsv4_request_state_indices
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

PAGE_SIZE = 256
FULL_SIZE = 2 * PAGE_SIZE

V41_RATIOS = [0] * 2 + [2] * 18 + [1] * 20
V41_SOURCES = [2, 8, 14, 20]


def _make_pool(*, compression_ratios, kv_source_layers=(), c4_size=0, c128_size=0):
    return DeepSeekV4TokenToKVPool(
        max_num_reqs=1,
        swa_size=FULL_SIZE,
        c4_size=c4_size,
        c128_size=c128_size,
        c4_state_pool_size=8 if c4_size else 0,
        c128_state_pool_size=8 if c128_size else 0,
        page_size=PAGE_SIZE,
        swa_page_size=128,
        dtype=torch.float8_e4m3fn,
        c4_state_dtype=torch.float32,
        c128_state_dtype=torch.float32,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        indexer_head_dim=128,
        layer_num=len(compression_ratios),
        device="cuda",
        enable_memory_saver=False,
        compression_ratios=compression_ratios,
        kv_source_layers=kv_source_layers,
        full_size=FULL_SIZE,
    )


@unittest.skipUnless(torch.cuda.is_available(), "allocates device KV buffers")
class TestDSV41DSparkPairRing(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        set_global_server_args_for_scheduler(
            ServerArgs(model_path="dummy", page_size=PAGE_SIZE)
        )

    def test_dspark_pair_ring_transfer_uses_request_slots(self):
        for gamma in (1, 5, 7):
            with (
                self.subTest(gamma=gamma),
                get_context().override_server_args(
                    speculative_algorithm="DSPARK",
                    speculative_num_draft_tokens=gamma + 1,
                ),
            ):
                pools = []
                for role in ("prefill", "decode"):
                    with get_context().override_server_args(
                        disaggregation_mode=role,
                        speculative_algorithm="DSPARK",
                        speculative_num_draft_tokens=gamma + 1,
                    ):
                        pools.append(
                            _make_pool(
                                compression_ratios=V41_RATIOS,
                                kv_source_layers=V41_SOURCES,
                            )
                        )
                prefill, decode = pools
                src_ptrs, _, item_lens = prefill.get_c128_state_buf_infos()
                dst_ptrs, _, dst_item_lens = decode.get_c128_state_buf_infos()
                self.assertEqual(item_lens, dst_item_lens)
                ring_size = prefill.get_ring_size(2)
                self.assertGreaterEqual(ring_size, gamma + 3)
                manager = object.__new__(MooncakeKVManager)
                manager.is_mla_backend = True
                manager.is_hybrid_mla_backend = False
                manager.enable_custom_mem_pool = False
                manager._transfer_data = Mock(return_value=0)
                for seq_len in (255, 256, 257, 511, 512, 513):
                    manager._transfer_data.reset_mock()
                    manager._send_kvcache_generic(
                        "decode",
                        src_ptrs,
                        dst_ptrs,
                        item_lens,
                        get_dsv4_request_state_indices(prefill, 0, seq_len),
                        get_dsv4_request_state_indices(decode, 1, seq_len),
                        executor=None,
                        state_type=StateType.C128_STATE,
                    )
                    blocks = manager._transfer_data.call_args.args[1]
                    expected = (
                        [
                            (src, dst + size, size)
                            for src, dst, size in zip(src_ptrs, dst_ptrs, item_lens)
                        ]
                        if seq_len % 2
                        else []
                    )
                    self.assertEqual(blocks, expected)

                # Reusing a decode slot must clear the expanded speculative ring,
                # including rejected candidate rows, without clearing its neighbor.
                for source in decode.sources_by_ratio[2]:
                    state = decode.compress_state_pools[source].kv_score_buffer.kv_score
                    self.assertEqual(
                        item_lens[decode.sources_by_ratio[2].index(source)],
                        state[0].nbytes * ring_size,
                    )
                    state.fill_(1)
                decode.clear_c128_req_state(1)
                for source in decode.sources_by_ratio[2]:
                    state = decode.compress_state_pools[source].kv_score_buffer.kv_score
                    half = state.shape[-1] // 2
                    self.assertTrue(bool((state[:ring_size] == 1).all()))
                    cleared = state[ring_size : 2 * ring_size]
                    self.assertTrue(bool((cleared[:, :half] == 0).all()))
                    self.assertTrue(bool((cleared[:, half:] == -torch.inf).all()))


if __name__ == "__main__":
    unittest.main()
