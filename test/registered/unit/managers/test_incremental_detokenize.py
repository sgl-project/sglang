import unittest
from array import array
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs
from sglang.srt.managers import multi_tokenizer_mixin
from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.incremental_detokenize import (
    DetokenizeHeader,
    DetokenizeStateTable,
)
from sglang.srt.managers.io_struct import (
    BatchTokenIDOutput,
    BeamSearchOutput,
    DetokenizeReleaseReq,
    flatten_output_ids,
)
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SPECIAL = 100


class _Tokenizer:
    """Renders each id; ids from SPECIAL up are special tokens."""

    is_fast = True

    def decode(self, ids, skip_special_tokens, spaces_between_special_tokens):
        return "".join(
            f"<{t}>" if t >= SPECIAL else f"{t}."
            for t in ids
            if not (skip_special_tokens and t >= SPECIAL)
        )

    def batch_decode(self, ids_list, **kwargs):
        return [self.decode(ids, **kwargs) for ids in ids_list]


class _StopTokenizer:
    is_fast = True
    _pieces = {
        1: "S",
        2: "<special>",
        3: "T",
        4: ".",
        5: "ST",
        6: "S.T",
        7: "X",
    }

    def decode(
        self,
        ids,
        skip_special_tokens=False,
        spaces_between_special_tokens=True,
    ):
        return "".join(
            piece
            for token_id in ids
            for piece in [self._pieces[token_id]]
            if not (skip_special_tokens and token_id in (2, 5))
        )

    def batch_decode(self, ids_list, **kwargs):
        return [self.decode(ids, **kwargs) for ids in ids_list]


def _header(prompt_tail_ids, *, skip_special_tokens=True, ipc="worker-a"):
    return DetokenizeHeader(
        prompt_tail_ids=array("q", prompt_tail_ids),
        skip_special_tokens=skip_special_tokens,
        spaces_between_special_tokens=True,
        no_stop_trim=False,
        http_worker_ipc=ipc,
    )


def _output(sent, rows, *, ipc="worker-a"):
    """rows: (rid, output_ids, header or None, finished). sent: by rid, the
    output ids the scheduler sent so far, which sets each row's completion_tokens."""
    n = len(rows)
    completion_tokens = []
    for rid, ids, header, _ in rows:
        sent[rid] = (0 if header is not None else sent.get(rid, 0)) + len(ids)
        completion_tokens.append(sent[rid])
    headers = [header for _, _, header, _ in rows]
    output_ids_flat, output_ids_offsets = flatten_output_ids(
        ids for _, ids, _, _ in rows
    )
    return BatchTokenIDOutput(
        rids=[rid for rid, _, _, _ in rows],
        finished_reasons=[
            {"type": "length"} if finished else None for *_, finished in rows
        ],
        output_ids_flat=output_ids_flat,
        output_ids_offsets=output_ids_offsets,
        detokenize_headers=headers if any(headers) else None,
        http_worker_ipcs=(
            [ipc if finished else None for *_, finished in rows]
            if any(finished for *_, finished in rows)
            else None
        ),
        reasoning_tokens=[0] * n,
        completion_tokens=completion_tokens,
        input_token_logprobs_val=None,
        input_token_logprobs_idx=None,
        output_token_logprobs_val=None,
        output_token_logprobs_idx=None,
        input_top_logprobs_val=None,
        input_top_logprobs_idx=None,
        output_top_logprobs_val=None,
        output_top_logprobs_idx=None,
        input_token_ids_logprobs_val=None,
        input_token_ids_logprobs_idx=None,
        output_token_ids_logprobs_val=None,
        output_token_ids_logprobs_idx=None,
        output_token_entropy_val=None,
        output_token_sampling_mask=None,
        output_hidden_states=None,
        routed_experts=None,
        indexer_topk=None,
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
    )


def _detokenizer(capacity=16, tokenizer=None):
    detok = DetokenizerManager.__new__(DetokenizerManager)
    detok.tokenizer = tokenizer or _Tokenizer()
    detok.vocab_size = None
    detok.detokenize_state_table = DetokenizeStateTable(capacity=capacity)
    detok.disable_tokenizer_batch_decode = False
    detok.is_tool_call_parser_gpt_oss = False
    detok.tool_close_token_ids = frozenset()
    return detok


class _Drained(Exception):
    pass


def _deliver(detok, outputs):
    """Run the multi-tokenizer event loop over outputs; return each row that
    reaches a tokenizer worker as (worker, rid, text, finished)."""
    queue = list(outputs)
    received = []

    def recv(_socket):
        if not queue:
            raise _Drained
        return queue.pop(0)

    def send(worker, out):
        received.extend(
            (worker, rid, text, reason is not None)
            for rid, text, reason in zip(
                out.rids, out.output_strs, out.finished_reasons
            )
        )

    detok.recv_from_scheduler = None
    detok.soft_watchdog = mock.MagicMock()
    detok.init_request_dispatcher()
    with (
        mock.patch.object(multi_tokenizer_mixin, "sock_recv", side_effect=recv),
        mock.patch.object(
            multi_tokenizer_mixin,
            "get_zmq_socket",
            side_effect=lambda _context, _type, ipc, _bind: ipc,
        ),
        mock.patch.object(multi_tokenizer_mixin, "sock_send", side_effect=send),
    ):
        try:
            detok.multi_http_worker_event_loop()
        except _Drained:
            pass
    return received


class TestDetokenizeState(CustomTestCase):
    def setUp(self) -> None:
        self.enterContext(get_context().override_server_args(kv_canary="none"))

    def test_later_rows_decode_with_the_first_rows_header(self):
        detok = _detokenizer()
        sent = {}
        header = _header([1, 2], skip_special_tokens=False)

        steps = [
            ([3], header, False),
            ([SPECIAL, 4], None, False),
            ([5], None, True),
        ]
        texts = []
        for ids, row_header, finished in steps:
            out = detok.handle_batch_token_id_out(
                _output(sent, [("r", ids, row_header, finished)])
            )
            self.assertEqual(out.http_worker_ipcs, ["worker-a"])
            texts += out.output_strs

        self.assertEqual(texts, ["3.", f"<{SPECIAL}>4.", "5."])
        self.assertIsNone(detok.detokenize_state_table.get("r", "worker-a"))

    def test_new_header_replaces_the_state_of_a_reused_rid(self):
        detok = _detokenizer()
        sent = {}
        detok.handle_batch_token_id_out(
            _output(sent, [("r", [3], _header([1]), False)])
        )

        out = detok.handle_batch_token_id_out(
            _output(sent, [("r", [7], _header([6], ipc="worker-b"), False)])
        )

        self.assertEqual(out.output_strs, ["7."])
        self.assertEqual(out.http_worker_ipcs, ["worker-b"])
        self.assertEqual(
            list(detok.detokenize_state_table.get("r", "worker-b").decode_ids),
            [6, 7],
        )

    def test_a_row_without_state_drops_only_its_text(self):
        """A row still in flight when its request was released (overlap scheduling)
        loses only its own text."""
        detok = _detokenizer()
        sent = {}
        detok.handle_batch_token_id_out(
            _output(
                sent,
                [
                    ("gone", [3], _header([1]), False),
                    ("kept", [4], _header([2]), False),
                ],
            )
        )

        detok.init_request_dispatcher()
        detok._request_dispatcher(
            DetokenizeReleaseReq(rid="gone", http_worker_ipc="worker-a")
        )
        self.assertIsNone(detok.detokenize_state_table.get("gone", "worker-a"))
        with self.assertLogs(
            "sglang.srt.managers.detokenizer_manager", level="WARNING"
        ) as logs:
            out = detok.handle_batch_token_id_out(
                _output(sent, [("gone", [], None, False), ("kept", [6], None, False)])
            )

        self.assertIn("rid='gone'", logs.output[0])
        self.assertEqual(out.output_strs, ["", "6."])
        self.assertEqual(out.http_worker_ipcs, [None, "worker-a"])

    def test_a_finished_row_without_state_reaches_its_tokenizer_worker(self):
        detok = _detokenizer(capacity=1)
        sent = {}
        outputs = [
            _output(sent, [("a", [3], _header([1]), False)]),
            _output(sent, [("b", [4], _header([2], ipc="worker-b"), False)]),
            _output(sent, [("a", [5], None, True)]),
        ]

        with self.assertLogs(
            "sglang.srt.managers.detokenizer_manager", level="WARNING"
        ):
            delivered = _deliver(detok, outputs)

        self.assertEqual(
            delivered,
            [
                ("worker-a", "a", "3.", False),
                ("worker-b", "b", "4.", False),
                ("worker-a", "a", "", True),
            ],
        )

    def test_a_finished_row_routes_by_its_ipc_not_the_stored_headers(self):
        """One rid is in flight on two tokenizer workers at once."""
        detok = _detokenizer()
        sent = {}
        outputs = [
            _output(sent, [("r", [3], _header([1]), False)]),
            _output(sent, [("r", [4], _header([2], ipc="worker-b"), False)]),
            _output(sent, [("r", [5], None, True)], ipc="worker-a"),
        ]

        delivered = _deliver(detok, outputs)

        self.assertEqual(
            [(worker, finished) for worker, _, _, finished in delivered],
            [("worker-a", False), ("worker-b", False), ("worker-a", True)],
        )

    def test_an_idle_batch_reaches_no_worker(self):
        """A DP rank with no work still sends a batch; its fan-out must not crash."""
        detok = _detokenizer()
        sent = {}
        outputs = [
            _output(sent, []),
            _output(sent, [("r", [3], _header([1]), True)]),
        ]

        delivered = _deliver(detok, outputs)

        self.assertEqual(delivered, [("worker-a", "r", "3.", True)])

    def test_header_prompt_tail_is_clamped_to_the_vocab(self):
        """Multimodal placeholder ids in the prompt tail decode as 0."""
        table = DetokenizeStateTable(capacity=1)

        s = table.start(rid="r", header=_header([-101, 3, 12]), vocab_size=10)

        self.assertEqual(s.decode_ids, [0, 3, 0])

    def test_capacity_does_not_evict_an_active_stream(self):
        table = DetokenizeStateTable(capacity=1)
        active = table.start(rid="active", header=_header([1]), vocab_size=None)

        self.assertIsNotNone(active)
        self.assertIsNone(table.start(rid="new", header=_header([2]), vocab_size=None))
        self.assertIs(table.get("active", "worker-a"), active)

        table.release("active", "worker-a")
        self.assertIsNotNone(
            table.start(rid="new", header=_header([2]), vocab_size=None)
        )

    def test_same_rid_on_two_workers_has_independent_state(self):
        table = DetokenizeStateTable(capacity=2)
        worker_a = table.start(
            rid="r", header=_header([1], ipc="worker-a"), vocab_size=None
        )
        worker_b = table.start(
            rid="r", header=_header([2], ipc="worker-b"), vocab_size=None
        )

        self.assertIsNot(worker_a, worker_b)
        self.assertEqual(worker_a.decode_ids, [1])
        self.assertEqual(worker_b.decode_ids, [2])
        table.release("r", "worker-a")
        self.assertIsNone(table.get("r", "worker-a"))
        self.assertIs(table.get("r", "worker-b"), worker_b)

    def test_skipped_special_token_does_not_move_terminal_stop(self):
        for terminal_ids in ([2, 3], [5]):
            with self.subTest(terminal_ids=terminal_ids):
                detok = _detokenizer(tokenizer=_StopTokenizer())
                sent = {}
                first = detok.handle_batch_token_id_out(
                    _output(
                        sent,
                        [("r", [1], _header([], ipc="worker-a"), False)],
                    )
                )
                terminal = detok.handle_batch_token_id_out(
                    _output(sent, [("r", terminal_ids, None, True)])
                )

                self.assertEqual(first.output_strs, ["S"])
                self.assertEqual(terminal.output_strs, [""])
                self.assertEqual(terminal.http_worker_ipcs, ["worker-a"])

    def test_regex_like_stop_does_not_trim_an_earlier_literal(self):
        detok = _detokenizer(tokenizer=_StopTokenizer())
        sent = {}
        first = detok.handle_batch_token_id_out(
            _output(sent, [("r", [6], _header([], ipc="worker-a"), False)])
        )
        terminal = detok.handle_batch_token_id_out(
            _output(sent, [("r", [1, 7, 3], None, True)])
        )

        self.assertEqual(first.output_strs, ["S.T"])
        self.assertEqual(terminal.output_strs, ["SXT"])

    def test_a_row_that_does_not_continue_the_received_ids_raises_an_alarm(self):
        """A break warns (raises in CI) and still decodes; beam rows are exempt."""
        detok = _detokenizer()
        sent = {}
        detok.handle_batch_token_id_out(
            _output(sent, [("r", [3], _header([1]), False)])
        )
        with envs.SGLANG_IS_IN_CI.override(True):
            detok.handle_batch_token_id_out(
                _output(sent, [("r", [4], _header([2]), False)])
            )

        skipped = _output({"r": 5}, [("r", [5], None, False)])
        with envs.SGLANG_IS_IN_CI.override(True), self.assertRaises(RuntimeError):
            detok.handle_batch_token_id_out(skipped)
        with envs.SGLANG_IS_IN_CI.override(False):
            with self.assertLogs(
                "sglang.srt.managers.incremental_detokenize", level="WARNING"
            ):
                out = detok.handle_batch_token_id_out(skipped)
            self.assertEqual(out.output_strs, ["5."])
            self.assertEqual(out.http_worker_ipcs, ["worker-a"])

            beam = _output(sent, [("r", [8], None, True)])
            beam.completion_tokens = [9]
            beam.beam_search_output = [BeamSearchOutput(sequences=[])]
            with self.assertNoLogs("sglang.srt.managers", level="WARNING"):
                out = detok.handle_batch_token_id_out(
                    _output({"r": 6}, [("r", [7], None, False)])
                )
                self.assertEqual(out.output_strs, ["7."])
                out = detok.handle_batch_token_id_out(beam)
            self.assertEqual(out.output_strs, ["8."])


if __name__ == "__main__":
    unittest.main()
