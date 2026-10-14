import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.utils import validate_input_length
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestContextOutputBudget(unittest.TestCase):
    def setUp(self):
        override = get_context().override_server_args(min_output_token_reserve=1)
        override.install()
        self.addCleanup(override.restore)

    def test_admission_boundaries_and_no_prompt_truncation(self):
        for reserve in (1, 16):
            override = get_context().override_server_args(
                min_output_token_reserve=reserve
            )
            override.install()
            try:
                manager = TokenizerManager.__new__(TokenizerManager)
                manager.model_config = SimpleNamespace(
                    hf_config=SimpleNamespace(), vocab_size=1000, joint_head_config=None
                )
                manager.context_len = 128
                manager.num_reserved_tokens = 4  # EAGLE scratch is not public context.
                manager.allow_auto_truncate = True
                manager._validate_token_ids_logprob = Mock()
                for prompt in (
                    128 - reserve - 1,
                    128 - reserve,
                    128 - reserve + 1,
                    129,
                ):
                    for requested in (None, 0, 1, 1000):
                        ids = [1] * prompt
                        req = GenerateReqInput(
                            input_ids=ids, sampling_params={"max_new_tokens": requested}
                        )
                        if prompt > 128 - reserve:
                            with self.assertRaisesRegex(
                                ValueError, "maximum allowed prompt length"
                            ):
                                manager._validate_one_request(req, ids)
                        else:
                            manager._validate_one_request(req, ids)
                            self.assertEqual(
                                req.sampling_params["max_new_tokens"],
                                (
                                    None
                                    if requested is None
                                    else min(requested, 128 - prompt)
                                ),
                            )
                        self.assertEqual(ids, [1] * prompt)
            finally:
                override.restore()

    def test_one_million_context_budget(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(), vocab_size=1000, joint_head_config=None
        )
        manager.context_len = 1_048_576
        manager.num_reserved_tokens = 4
        manager.allow_auto_truncate = False
        manager._validate_token_ids_logprob = Mock()
        for prompt, expected in [(990_045, 58_531), (1_048_575, 1)]:
            req = GenerateReqInput(
                input_ids=[1] * prompt, sampling_params={"max_new_tokens": 64_000}
            )
            manager._validate_one_request(req, req.input_ids)
            self.assertEqual(req.sampling_params["max_new_tokens"], expected)

    def test_scheduler_preserves_last_output_slot(self):
        scheduler = SimpleNamespace(
            max_req_len=128,
            kv_shard_widening=1,
            page_size=64,
            max_new_tokens_limit=None,
            token_to_kv_pool_allocator=SimpleNamespace(
                max_new_tokens_for_memory=lambda n, m, **kw: m
            ),
            max_total_num_tokens=10000,
            sliding_window_size=512,
            chunked_prefill_size=64,
        )
        for prompt, requested, expected in [(127, 100, 1), (112, 100, 16), (112, 1, 1)]:
            req = SimpleNamespace(
                origin_input_ids=[1] * prompt,
                sampling_params=SimpleNamespace(
                    max_new_tokens=requested, min_new_tokens=0
                ),
                set_finish_with_abort=Mock(),
            )
            Scheduler.init_req_max_new_tokens(scheduler, req)
            self.assertEqual(req.sampling_params.max_new_tokens, expected)
            req.set_finish_with_abort.assert_not_called()
        self.assertIsNone(validate_input_length(req, 112, False))

    def test_memory_exhaustion_aborts_instead_of_success_with_zero_budget(self):
        scheduler = SimpleNamespace(
            max_req_len=128,
            kv_shard_widening=1,
            page_size=64,
            max_new_tokens_limit=None,
            token_to_kv_pool_allocator=SimpleNamespace(
                max_new_tokens_for_memory=lambda *a, **kw: 0
            ),
            max_total_num_tokens=128,
            sliding_window_size=512,
            chunked_prefill_size=64,
        )
        req = SimpleNamespace(
            origin_input_ids=[1] * 127,
            sampling_params=SimpleNamespace(max_new_tokens=1, min_new_tokens=0),
            set_finish_with_abort=Mock(),
        )
        Scheduler.init_req_max_new_tokens(scheduler, req)
        req.set_finish_with_abort.assert_called_once()


if __name__ == "__main__":
    unittest.main()
