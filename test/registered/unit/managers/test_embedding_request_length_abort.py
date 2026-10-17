"""Regression: an embedding request whose input reaches the scheduler-side
max_req_input_len must be aborted (finished with a 400) before it is queued,
matching handle_generate_request. Drives the real
`Scheduler.handle_embedding_request` with a stubbed `self`; pure CPU."""

import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_module
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_parallel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MAX_REQ_INPUT_LEN = 8


def _make_recv_req(num_tokens):
    return SimpleNamespace(
        rid="rid-0",
        input_text="x",
        input_ids=list(range(num_tokens)),
        sampling_params=MagicMock(),
        positional_embed_overrides=None,
        token_type_ids=None,
        routed_dp_rank=None,
        priority=None,
        dimensions=None,
        lora_id=None,
        http_worker_ipc=None,
        time_stats=None,
        return_pooled_hidden_states=False,
        multi_item_delimiter_indices=None,
        token_indices_to_pool=None,
        mm_inputs=None,
    )


class TestEmbeddingRequestLengthAbort(CustomTestCase):
    def _handle(self, num_tokens):
        fake_self = MagicMock()
        fake_self.max_req_input_len = MAX_REQ_INPUT_LEN
        with (
            patch.object(
                scheduler_module,
                "get_serving",
                return_value=SimpleNamespace(allow_auto_truncate=False),
            ),
            get_parallel().override(tp_rank=1),
        ):
            Scheduler.handle_embedding_request(fake_self, _make_recv_req(num_tokens))
        fake_self._add_request_to_queue.assert_called_once()
        return fake_self._add_request_to_queue.call_args.args[0]

    def test_over_length_request_is_aborted(self):
        req = self._handle(MAX_REQ_INPUT_LEN + 1)
        self.assertIsInstance(req.to_finish, FINISH_ABORT)
        self.assertEqual(req.to_finish.status_code, HTTPStatus.BAD_REQUEST)

    def test_within_limit_request_is_not_aborted(self):
        req = self._handle(MAX_REQ_INPUT_LEN - 1)
        self.assertIsNone(req.to_finish)


if __name__ == "__main__":
    unittest.main()
