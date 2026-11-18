import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput
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
            joint_head_config=None, pointer_head=None
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
        ):
            with self.subTest(**server_args):
                with self.assertRaisesRegex(ValueError, "requires --attention-backend"):
                    self._validate(self._manager(**server_args))

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


if __name__ == "__main__":
    unittest.main()
