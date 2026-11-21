"""The prompt fingerprint must cover every input the transferred KV depends on.

Two requests that share a bootstrap_room must get different fingerprints whenever their KV differs.
The prefill request and the decode request of one transfer must get the same fingerprint.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation import decode as decode_module
from sglang.srt.disaggregation import prefill as prefill_module
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.utils import TransferBackend
from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import Session
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOM = 1000
PROMPT = [11, 12, 13, 14]


def _decode_req(origin_input_ids=PROMPT, output_ids=(), **fields):
    req = SimpleNamespace(
        origin_input_ids=array("q", origin_input_ids),
        output_ids=list(output_ids),
        bootstrap_host="10.0.0.9",
        bootstrap_port=8998,
        bootstrap_room=ROOM,
        lora_name=None,
    )
    req.__dict__.update(fields)
    return req


def _prefill_req(origin_input_ids=PROMPT, **fields):
    req = SimpleNamespace(
        origin_input_ids=array("q", origin_input_ids),
        bootstrap_host="10.0.0.9",
        bootstrap_room=ROOM,
        disagg_prefill_dp_rank=None,
        lora_name=None,
    )
    req.__dict__.update(fields)
    return req


def _fingerprints(decode_req=None, prefill_req=None, is_rebootstrap=False):
    """The fingerprints the real decode and prefill call sites hand to the KV classes."""
    kv_class = MagicMock()
    decode_queue = decode_module.DecodePreallocQueue.__new__(
        decode_module.DecodePreallocQueue
    )
    decode_queue.transfer_backend = TransferBackend.NIXL
    decode_queue.kv_manager = decode_queue.queue = None
    prefill_queue = prefill_module.PrefillBootstrapQueue.__new__(
        prefill_module.PrefillBootstrapQueue
    )
    prefill_queue.transfer_backend = TransferBackend.NIXL
    prefill_queue.kv_manager = None
    prefill_queue.bootstrap_port = 8998
    with (
        patch.object(decode_module, "get_kv_class", return_value=kv_class),
        patch.object(decode_module, "move"),
        patch.object(prefill_module, "get_kv_class", return_value=kv_class),
    ):
        if decode_req is not None:
            decode_queue._create_receiver_and_enqueue(
                decode_req, is_rebootstrap=is_rebootstrap
            )
        if prefill_req is not None:
            prefill_queue.create_sender(prefill_req, num_kv_heads=1)
    return [c.kwargs["prompt_fingerprint"] for c in kv_class.call_args_list]


class TestLoraInFingerprint(CustomTestCase):
    def test_same_ids_with_another_adapter_differ(self):
        (a,) = _fingerprints(prefill_req=_prefill_req(lora_name="adapter-a"))
        (b,) = _fingerprints(decode_req=_decode_req(lora_name="adapter-b"))
        (base,) = _fingerprints(decode_req=_decode_req())
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, base)

    def test_same_adapter_on_both_sides_agrees(self):
        decode_fp, prefill_fp = _fingerprints(
            _decode_req(lora_name="adapter-a"), _prefill_req(lora_name="adapter-a")
        )
        self.assertEqual(decode_fp, prefill_fp)


class TestSchedulerKeepsLoraName(CustomTestCase):
    """The fingerprint reads req.lora_name, so every scheduler path must set it from the wire."""

    @staticmethod
    def _scheduler(sessions, queued):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.session_controller = sessions
        scheduler.enable_session_radix_cache = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.model_config = SimpleNamespace(
            hf_eos_token_id=None, vocab_size=128, token_id_vocab_size=128
        )
        scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
        scheduler.kv_events_publisher = SimpleNamespace(enable_kv_cache_events=False)
        scheduler.tokenizer = None
        scheduler.dllm_config = None
        scheduler._maybe_namespace_elastic_radix_cache = lambda req: None
        scheduler.init_req_max_new_tokens = lambda req: None
        scheduler._add_request_to_queue = queued.append
        return scheduler

    def test_plain_and_session_requests_carry_the_adapter_name(self):
        live = Session(capacity_of_str_len=0, session_id="live", streaming=True)
        paths = {
            "plain": (None, {}),
            "session": (SessionParams(id="live"), {"live": live}),
        }
        for name, (session_params, sessions) in paths.items():
            with self.subTest(path=name):
                queued = []
                recv = TokenizedGenerateReqInput(
                    rid=f"rid-{name}",
                    input_text="hello",
                    input_ids=array("q", [1, 2, 3]),
                    input_embeds=None,
                    mm_inputs=None,
                    token_type_ids=None,
                    sampling_params=SamplingParams(max_new_tokens=4),
                    return_logprob=False,
                    logprob_start_len=-1,
                    top_logprobs_num=0,
                    token_ids_logprob=None,
                    stream=False,
                    session_params=session_params,
                    bootstrap_port=1,
                    lora_name="adapter-a",
                )
                # An mm_input_error stops every path right after construction.
                with get_parallel().override(tp_rank=0):
                    self._scheduler(sessions, queued).handle_generate_request(
                        recv, mm_input_error="stop"
                    )
                (req,) = queued
                self.assertEqual(req.lora_name, "adapter-a")


if __name__ == "__main__":
    unittest.main()
