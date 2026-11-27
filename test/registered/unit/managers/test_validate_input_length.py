"""Unit tests for the scheduler's prompt-length check (validate_input_length).

Turns red if the rejection loses OpenAI's context-length sentence, which
OpenAI-compatible clients use to tell a context overflow from other 400s, if
the auto-truncate and in-range paths change, if a multimodal prompt gets
truncated, or if an over-long embedding request is queued without an abort.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.utils import validate_input_length

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(num_tokens):
    return SimpleNamespace(origin_input_ids=array("q", range(num_tokens)))


class TestValidateInputLength(CustomTestCase):
    def test_rejection_uses_openai_context_length_wording(self):
        req = _req(1500)
        msg = validate_input_length(
            req, max_req_input_len=1018, allow_auto_truncate=False, is_multimodal=False
        )
        self.assertIn(
            "Input length (1500 tokens) exceeds the maximum allowed length (1018 tokens).",
            msg,
        )
        self.assertIn("This model's maximum context length is 1018 tokens", msg)
        self.assertIn("--allow-auto-truncate", msg)
        self.assertEqual(len(req.origin_input_ids), 1500)

    def test_limit_is_inclusive(self):
        """The limit itself is accepted untruncated; one token more is not."""
        for allow_auto_truncate in (False, True):
            req = _req(1018)
            self.assertIsNone(
                validate_input_length(
                    req,
                    max_req_input_len=1018,
                    allow_auto_truncate=allow_auto_truncate,
                    is_multimodal=False,
                )
            )
            self.assertEqual(len(req.origin_input_ids), 1018)
        msg = validate_input_length(
            _req(1019),
            max_req_input_len=1018,
            allow_auto_truncate=False,
            is_multimodal=False,
        )
        self.assertIn("This model's maximum context length is 1018 tokens", msg)

    def test_auto_truncate_unchanged(self):
        req = _req(1500)
        self.assertIsNone(
            validate_input_length(
                req,
                max_req_input_len=1018,
                allow_auto_truncate=True,
                is_multimodal=False,
            )
        )
        self.assertEqual(list(req.origin_input_ids), list(range(1018)))

    def test_in_range_unchanged(self):
        req = _req(1017)
        for allow_auto_truncate in (False, True):
            self.assertIsNone(
                validate_input_length(
                    req,
                    max_req_input_len=1018,
                    allow_auto_truncate=allow_auto_truncate,
                    is_multimodal=False,
                )
            )
        self.assertEqual(len(req.origin_input_ids), 1017)

    def test_multimodal_prompt_is_rejected_never_truncated(self):
        for allow_auto_truncate in (False, True):
            req = _req(1019)
            msg = validate_input_length(
                req,
                max_req_input_len=1018,
                allow_auto_truncate=allow_auto_truncate,
                is_multimodal=True,
            )
            self.assertIn(
                "Multimodal prompt is too long after expanding multimodal tokens.", msg
            )
            self.assertIn("This model's maximum context length is 1018 tokens", msg)
            self.assertNotIn("--allow-auto-truncate", msg)
            self.assertEqual(len(req.origin_input_ids), 1019)
            self.assertIsNone(
                validate_input_length(
                    _req(1018),
                    max_req_input_len=1018,
                    allow_auto_truncate=allow_auto_truncate,
                    is_multimodal=True,
                )
            )


class TestSchedulerEmbeddingLengthCheck(CustomTestCase):
    def test_over_long_embedding_request_is_aborted(self):
        from sglang.srt.managers import scheduler as scheduler_module

        scheduler = object.__new__(scheduler_module.Scheduler)
        scheduler.tokenizer = object()
        scheduler.max_req_input_len = 4
        scheduler._embedding_preserve_last_token_id = None
        scheduler._maybe_namespace_elastic_radix_cache = MagicMock()
        scheduler._record_kv_event_lora_name = MagicMock()
        scheduler._add_request_to_queue = MagicMock()
        req = MagicMock(origin_input_ids=array("q", range(5)))
        recv_req = MagicMock(mm_inputs=None)

        with (
            patch.object(scheduler_module, "Req", return_value=req),
            patch.object(
                scheduler_module,
                "get_serving",
                return_value=SimpleNamespace(allow_auto_truncate=False),
            ),
        ):
            scheduler.handle_embedding_request(recv_req)

        (error_msg,), _ = req.set_finish_with_abort.call_args
        self.assertIn("This model's maximum context length is 4 tokens", error_msg)
        scheduler._add_request_to_queue.assert_called_once_with(req)


if __name__ == "__main__":
    unittest.main()
