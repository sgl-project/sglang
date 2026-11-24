import unittest
from functools import partial
from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context, get_schedule
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestBidirPrefixLenValidation(CustomTestCase):
    def _manager(self, **server_args):
        override = get_context().override_server_args(
            **{"attention_backend": "triton", "page_size": 1, **server_args}
        )
        override.install()
        self.addCleanup(override.restore)
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.context_len = 128
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = False
        manager.validate_total_tokens = False
        manager.is_generation = True
        manager.model_config = SimpleNamespace(
            joint_head_config=None, pointer_head=None, is_hybrid_swa=False
        )
        return manager

    def _validate(self, manager, prompt_len=8, bidir_prefix_len=4, **fields):
        req = GenerateReqInput(
            input_ids=list(range(prompt_len)),
            sampling_params={"max_new_tokens": 0},
            bidir_prefix_len=bidir_prefix_len,
            **fields,
        )
        manager._validate_one_request(req, req.input_ids)

    def test_accepts_triton_page_size_one(self):
        self._validate(self._manager())

    def test_rejects_other_backends_and_page_sizes(self):
        for server_args in (
            {"attention_backend": "trtllm_mha"},
            {"page_size": 64},
            {"enable_deterministic_inference": True},
            {"dcp_size": 2},
            {"enable_two_batch_overlap": True},
            # The load-scaled budget can cap every pass below the prompt.
            {
                "schedule_policy": "shortest-prefill-first",
                "shortest_prefill_boundary_halving_chunks": 1,
            },
        ):
            with self.subTest(**server_args):
                with self.assertRaisesRegex(ValueError, "requires --attention-backend"):
                    self._validate(self._manager(**server_args))

    def test_rejects_hybrid_swa_models(self):
        # Window pools price a re-prefill by its chunk and can leave only a cut
        # inside the prompt; the target models keep no windows.
        manager = self._manager()
        manager.model_config.is_hybrid_swa = True
        with self.assertRaisesRegex(ValueError, "sliding-window"):
            self._validate(manager)

    def test_rejects_values_that_are_not_a_boundary(self):
        # A list on a single prompt used to fail as a 500; -2 ran causal; a bool
        # is an int subclass.
        for bidir_prefix_len in ([4], -2, True, False):
            with self.subTest(bidir_prefix_len=bidir_prefix_len):
                with self.assertRaisesRegex(ValueError, "integer >= -1"):
                    self._validate(self._manager(), bidir_prefix_len=bidir_prefix_len)

    def test_overlong_prompt_is_refused_not_truncated(self):
        # Auto-truncation keeps the head, so it would cut the question off a
        # block-bidirectional prompt; a causal prompt is still truncated.
        manager = self._manager()
        manager.allow_auto_truncate = True
        manager.context_len = 8
        manager._embedding_preserve_last_token_id = None
        with get_schedule().override(chunked_prefill_size=None):
            for bidir_prefix_len, kept in ((4, 10), (-1, 8)):
                with self.subTest(bidir_prefix_len=bidir_prefix_len):
                    req = GenerateReqInput(
                        input_ids=list(range(10)),
                        sampling_params={"max_new_tokens": 0},
                        bidir_prefix_len=bidir_prefix_len,
                    )
                    if bidir_prefix_len >= 0:
                        with self.assertRaisesRegex(ValueError, "context length"):
                            manager._validate_one_request(req, req.input_ids)
                    else:
                        manager._validate_one_request(req, req.input_ids)
                    self.assertEqual(len(req.input_ids), kept)

    def test_rejects_legacy_session_params(self):
        with self.assertRaisesRegex(ValueError, "session_params"):
            self._validate(
                self._manager(), session_params={"id": "s", "rid": None, "offset": 0}
            )

    def test_prompt_must_fit_the_resolved_chunk(self):
        # The raw flag is unset; the resolved schedule still has a chunk size.
        manager = self._manager(chunked_prefill_size=None)
        with get_schedule().override(chunked_prefill_size=6):
            self._validate(manager, prompt_len=6)
            with self.assertRaisesRegex(ValueError, "chunked-prefill-size"):
                self._validate(manager, prompt_len=7)


def _scheduler(window_pool_names=()):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        window_pool_names=partial(tuple, window_pool_names)
    )
    return scheduler


class TestBidirPromptCut(CustomTestCase):
    def test_window_pools_refuse_a_bidir_prompt(self):
        # Admission prices a re-prefill by its chunk in every window pool, a fused
        # draft's on a target without windows too, so the largest cut that fits
        # can land inside the prompt.
        for windows in ((), ("window_512",)):
            scheduler = _scheduler(windows)
            for boundary in (-1, 0):
                with self.subTest(windows=windows, boundary=boundary):
                    req = SimpleNamespace(
                        origin_input_ids=list(range(8)), bidir_prefix_len=boundary
                    )
                    error = scheduler.validate_bidir_prompt_cut(req)
                    if windows and boundary >= 0:
                        self.assertIsNotNone(error)
                        self.assertIn("sliding-window KV pools", error)
                    else:
                        self.assertIsNone(error)

    def test_prompt_must_fit_the_largest_aligned_cut(self):
        # A retracted request re-prefills in cuts of chunk // align * align
        # tokens, and admission never cuts inside a block-bidirectional prompt.
        scheduler = _scheduler()
        for align, chunk, prompt, boundary, max_len in (
            (64, 100, 64, 0, None),
            (64, 100, 65, 0, 64),
            (64, 100, 65, -1, None),
            (64, 130, 128, 0, None),
            (64, 130, 129, 0, 128),
            (None, 130, 129, 0, None),
        ):
            with self.subTest(
                align=align, chunk=chunk, prompt=prompt, boundary=boundary
            ):
                scheduler.truncation_align_size = align
                scheduler.chunked_prefill_size = chunk
                req = SimpleNamespace(
                    origin_input_ids=list(range(prompt)), bidir_prefix_len=boundary
                )
                error = scheduler.validate_bidir_prompt_cut(req)
                if max_len is None:
                    self.assertIsNone(error)
                else:
                    self.assertIn(f"at most {max_len} tokens", error)


if __name__ == "__main__":
    unittest.main()
