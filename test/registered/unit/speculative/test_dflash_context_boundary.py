"""Model-position admission and target-wide eager phase decisions."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_tail import select_eager_phase
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ContextBoundaryTests(CustomTestCase):
    def test_independent_layout_rejects_native_ring_before_initialization(self):
        from sglang.srt.arg_groups.speculative_hook import _handle_dflash

        cfg = SimpleNamespace(
            speculative_draft_kv_layout="hybrid",
            speculative_dflash_draft_kv_ring=True,
        )
        with patch(
            "sglang.srt.arg_groups.speculative_hook.resolving_view", return_value=cfg
        ):
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                _handle_dflash(object())

    def test_dsa_tail_uses_live_width_without_changing_b12_configuration(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.speculative_num_draft_tokens = 12
        backend.dsa_shards_index_k = False
        for width in [12, 1, 12]:
            spec = DFlashVerifyInput(
                draft_token=torch.empty((width,), dtype=torch.int64),
                positions=torch.arange(width),
                draft_token_num=width,
            )
            batch = SimpleNamespace(spec_info=spec)
            self.assertEqual(backend._runtime_verify_width(batch), width)
            cache = torch.tensor([50, 100], dtype=torch.int32)
            expanded = (cache[:, None] - width + 1 + torch.arange(width)).flatten()
            with patch(
                "sglang.srt.layers.attention.dsa_backend.get_platform",
                return_value=SimpleNamespace(is_sm100=True),
            ):
                schedule = backend._build_paged_mqa_schedule_2d_ctx_lens(
                    ForwardMode.TARGET_VERIFY, cache, expanded, 2, draft_token_num=width
                )
            self.assertEqual(tuple(schedule.shape), (2, width))
            self.assertEqual(backend.speculative_num_draft_tokens, 12)

    def batch(self, prefix=7, greedy=True, extend=False, idle=False):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_extend=lambda: extend),
            is_extend_in_batch=False,
            sampling_info=SimpleNamespace(is_all_greedy=greedy),
            reqs=(
                []
                if idle
                else [SimpleNamespace(kv=SimpleNamespace(kv_committed_len=prefix))]
            ),
        )

    def worker(self):
        return SimpleNamespace(
            block_size=12,
            model_runner=SimpleNamespace(model_config=SimpleNamespace(context_len=32)),
        )

    @patch(
        "sglang.srt.runtime_context.get_parallel",
        return_value=SimpleNamespace(tp_group=SimpleNamespace(world_size=1)),
    )
    def test_tail_starts_before_the_first_invalid_rope_position(self, parallel):
        worker = self.worker()
        select_eager_phase(worker, self.batch(prefix=20))
        self.assertFalse(worker._target_only_tail_this_step)
        select_eager_phase(worker, self.batch(prefix=21))
        self.assertTrue(worker._target_only_tail_this_step)
        select_eager_phase(worker, self.batch(prefix=31))
        self.assertTrue(worker._target_only_tail_this_step)

    @patch(
        "sglang.srt.runtime_context.get_parallel",
        return_value=SimpleNamespace(tp_group=SimpleNamespace(world_size=1)),
    )
    def test_single_prefill_health_request_is_allowed_but_sampling_decode_is_rejected(
        self, parallel
    ):
        worker = self.worker()
        select_eager_phase(worker, self.batch(greedy=False, extend=True))
        with self.assertRaisesRegex(ValueError, "greedy"):
            select_eager_phase(worker, self.batch(greedy=False))

    @patch(
        "sglang.srt.runtime_context.get_parallel",
        return_value=SimpleNamespace(
            tp_group=SimpleNamespace(world_size=2, cpu_group="target")
        ),
    )
    @patch("torch.distributed.all_reduce")
    def test_idle_owner_obeys_the_target_wide_phase_vote(self, reduce, parallel):
        def remote_tail(flags, **kwargs):
            self.assertEqual(kwargs["group"], "target")
            flags[1] = 1

        reduce.side_effect = remote_tail
        worker = self.worker()
        select_eager_phase(worker, self.batch(idle=True))
        self.assertTrue(worker._target_only_tail_this_step)

        def remote_unsupported(flags, **kwargs):
            flags[0] = 1

        reduce.side_effect = remote_unsupported
        with self.assertRaisesRegex(ValueError, "greedy"):
            select_eager_phase(worker, self.batch(idle=True))

    def test_independent_admission_exposes_exact_context_and_respects_actual_kv_capacity(
        self,
    ):
        context = 1048576
        for mode, capacity, expected in [
            ("shared", context + 128, context - 1),
            ("hybrid", context + 128, context),
            ("hybrid", 65536, 65536),
        ]:
            config = SimpleNamespace(context_len=context)
            runner = SimpleNamespace(
                model_config=config,
                effective_logical_max_total_num_tokens=capacity,
                logical_max_total_num_tokens=capacity,
                token_to_kv_pool_allocator=object(),
                spec_algorithm=SimpleNamespace(is_dflash=lambda: True),
                max_running_requests=1,
                forward_stream=None,
                req_to_token_pool=SimpleNamespace(size=1, max_context_len=context + 64),
                token_to_kv_pool=SimpleNamespace(size=capacity),
            )
            worker = SimpleNamespace(
                model_config=config,
                model_runner=runner,
                dllm_algorithm=None,
                random_seed=0,
                device="cpu",
            )
            with (
                patch(
                    "sglang.srt.managers.tp_worker.get_spec",
                    return_value=SimpleNamespace(speculative_draft_kv_layout=mode),
                ),
                patch(
                    "sglang.srt.managers.tp_worker.get_schedule",
                    return_value=SimpleNamespace(
                        max_prefill_tokens=512, max_queued_requests=None
                    ),
                ),
                patch(
                    "sglang.srt.managers.tp_worker.page_interleave_shard_size",
                    return_value=1,
                ),
                patch(
                    "sglang.srt.managers.tp_worker.is_cp_sharded_padded",
                    return_value=False,
                ),
            ):
                info = TpModelWorker.get_worker_info(worker)
            self.assertEqual(info[4], expected)
            self.assertEqual(info[5], expected - 5)


if __name__ == "__main__":
    unittest.main()
