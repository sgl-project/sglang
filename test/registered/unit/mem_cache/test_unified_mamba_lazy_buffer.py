import unittest
from array import array
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.mem_cache.unified_memory_pool import (
    init_unified_mamba_pools,
    init_unified_mamba_swa_pools,
)
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestUnifiedMambaLazyBuffer(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def _build(self, shape, lazy):
        from test_unified_tri_pool import TestTriFactorySizing

        kwargs = TestTriFactorySizing()._factory_kwargs(
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=lazy,
            disable_overlap_schedule=False,
            max_mamba_cache_size=16,
        )
        if shape == "tri":
            return init_unified_mamba_swa_pools(**kwargs)
        for key in (
            "v_head_dim",
            "swa_head_num",
            "swa_head_dim",
            "swa_v_head_dim",
            "swa_attention_layer_ids",
            "swa_max_total_num_tokens",
        ):
            kwargs.pop(key)
        kwargs["max_total_num_tokens"] = kwargs.pop("full_max_total_num_tokens")
        kwargs.update(
            is_draft_worker=False,
            use_mla_backend=shape == "mla",
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            speculative_num_draft_tokens=None,
        )
        return init_unified_mamba_pools(**kwargs)

    def test_allocate_boundary_and_cleanup(self):
        for shape in ("mha", "mla", "tri"):
            for lazy in (False, True):
                for advance_boundary in (False, True):
                    with self.subTest(
                        shape=shape, lazy=lazy, boundary=advance_boundary
                    ):
                        bundle = self._build(shape, lazy)
                        pool = bundle.req_to_token_pool
                        allocator = pool.mamba_allocator
                        available = allocator.available_size()
                        req = Req("lazy-buffer", "", array("q", [1]), SamplingParams())
                        rows = pool.alloc([req])
                        self.assertIsNotNone(rows)
                        self.assertEqual(pool.enable_mamba_extra_buffer_lazy, lazy)
                        self.assertEqual(
                            available - allocator.available_size(), 2 if lazy else 3
                        )
                        buf = req.kv.mamba_ping_pong_track_buffer
                        self.assertEqual(int((buf == -1).sum()), int(lazy))
                        if lazy and advance_boundary:
                            next_slot = allocator.alloc(1)
                            self.assertIsNotNone(next_slot)
                            pool.set_mamba_ping_pong_slot(req, 1, next_slot[0])
                            SchedulerBatchResultProcessor.mamba_lazy_post_decode_at_boundary(
                                None,
                                req,
                                SimpleNamespace(req_to_token_pool=pool),
                                track_idx=1,
                            )
                            self.assertEqual(buf[0].item(), -1)
                            self.assertEqual(req.kv.mamba_last_track_idx, 1)
                        # Finish/abort before or after a boundary must reclaim
                        # every request-owned slot, excluding absent (-1) slots.
                        pool.free_mamba_cache(req)
                        pool.free(req)
                        self.assertEqual(allocator.available_size(), available)
                        self.assertIsNone(req.kv.mamba_pool_idx)
                        self.assertIsNone(req.kv.mamba_ping_pong_track_buffer)
                        self.assertEqual(
                            bundle.token_to_kv_pool_allocator.verify_byte_accounting(),
                            [],
                        )


if __name__ == "__main__":
    unittest.main()
