"""Unit tests for the session context forward primitives in srt/managers."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.schedule_batch import _resolve_query_attention
from sglang.srt.managers.scheduler import (
    _read_shm_input_embeds,
    _tail_coverage_intact,
    _token_positions_dims,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(**fields):
    req = MagicMock()
    for name, value in fields.items():
        setattr(req, name, value)
    return req


class TestResolveQueryAttention(CustomTestCase):
    def test_single_mode_and_empty_batch(self):
        self.assertEqual(
            _resolve_query_attention([_req(query_attention="bidirectional")]),
            "bidirectional",
        )
        self.assertIsNone(_resolve_query_attention([_req(query_attention=None)]))
        # DP idle batches build ScheduleBatch.init_new([]) routinely
        self.assertIsNone(_resolve_query_attention([]))

    def test_mixed_modes_raise(self):
        with self.assertRaises(ValueError):
            _resolve_query_attention(
                [_req(query_attention=None), _req(query_attention="bidirectional")]
            )


class TestBatchPurityHelpers(CustomTestCase):
    def test_shm_embeds_reject_nonfinite_values_after_read_and_cast(self):
        ref = {"transport": "shm"}
        invalid = (
            np.array([[float("nan")]], dtype=np.float32),
            np.array([[float("inf")]], dtype=np.float32),
            np.array([[1e308]], dtype=np.float64),
        )
        for values in invalid:
            with (
                self.subTest(values=values),
                patch(
                    "sglang.srt.managers.scheduler.read_shm_tensor",
                    return_value=values,
                ),
                self.assertRaisesRegex(ValueError, "finite numbers"),
            ):
                _read_shm_input_embeds(ref)

        with patch(
            "sglang.srt.managers.scheduler.read_shm_tensor",
            return_value=np.array([[1.0]], dtype=np.float16),
        ):
            embeds = _read_shm_input_embeds(ref)
        self.assertEqual(embeds.dtype, np.float32)

    def test_token_positions_dims(self):
        self.assertIsNone(_token_positions_dims(_req(token_positions=None)))
        self.assertEqual(_token_positions_dims(_req(token_positions=[1, 2, 3])), 1)
        self.assertEqual(
            _token_positions_dims(_req(token_positions=[[9], [0], [0]])), 3
        )

    def test_tail_coverage(self):
        plain = _req(input_embeds=None, query_attention=None)
        self.assertTrue(_tail_coverage_intact(plain))

        covered = _req(
            input_embeds=[[0.0]] * 4,
            query_attention="bidirectional",
            origin_input_ids=list(range(10)),
            prefix_indices=list(range(6)),
            host_hit_length=0,
        )
        self.assertTrue(_tail_coverage_intact(covered))

        host_covered = _req(
            input_embeds=[[0.0]] * 4,
            query_attention="bidirectional",
            origin_input_ids=list(range(10)),
            prefix_indices=list(range(3)),
            host_hit_length=3,
        )
        self.assertTrue(_tail_coverage_intact(host_covered))

        evicted = _req(
            input_embeds=[[0.0]] * 4,
            query_attention="bidirectional",
            origin_input_ids=list(range(10)),
            prefix_indices=list(range(3)),
            host_hit_length=2,
        )
        self.assertFalse(_tail_coverage_intact(evicted))

    def test_evicted_context_forward_leaves_queue_and_streams_abort(self):
        from http import HTTPStatus
        from types import SimpleNamespace

        from sglang.srt.managers.scheduler import Scheduler

        req = _req(
            rid="evicted",
            input_embeds=[[0.0]],
            token_positions=[0],
            return_hidden_states=True,
            hidden_states_buffer={"transport": "shm"},
            return_logprob=False,
            mamba_cow_src_index=object(),
            mamba_needs_clear=True,
            mamba_pool_idx=None,
        )
        survivor = _req(rid="survivor")
        scheduler = SimpleNamespace(
            waiting_queue=[req, survivor],
            enable_hicache_storage=True,
            tree_cache=MagicMock(),
            output_streamer=MagicMock(),
        )

        Scheduler._abort_evicted_context_forwards(scheduler, [req])

        self.assertEqual(scheduler.waiting_queue, [survivor])
        scheduler.tree_cache.release_aborted_request.assert_called_once_with("evicted")
        scheduler.output_streamer.stream_output.assert_called_once_with([req], False)
        self.assertEqual(req.finished_reason.status_code, HTTPStatus.BAD_REQUEST)
        self.assertIn("resubmit", req.finished_reason.message)
        self.assertIsNone(req.input_embeds)
        self.assertIsNone(req.token_positions)
        self.assertFalse(req.return_hidden_states)
        self.assertIsNone(req.hidden_states_buffer)
        self.assertIsNone(req.mamba_cow_src_index)
        self.assertFalse(req.mamba_needs_clear)


class TestGenerateReqInputEmbeds(CustomTestCase):
    def test_http_schema_keeps_numeric_values_strict(self):
        from pydantic import TypeAdapter, ValidationError

        adapter = TypeAdapter(GenerateReqInput)
        for value in (True, "1"):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                adapter.validate_python({"input_embeds": [[value]]})

        for value in (1, 1.0):
            obj = adapter.validate_python({"input_embeds": [[value]]})
            self.assertEqual(obj.input_embeds, [[1.0]])

    def test_integer_embeds_are_a_single_request(self):
        obj = GenerateReqInput(input_embeds=[[0, 0], [1, 1]])
        obj.normalize_batch_and_arguments()
        self.assertTrue(obj.is_single)

    def test_empty_embeds_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, "input_embeds"):
            GenerateReqInput(input_embeds=[]).normalize_batch_and_arguments()

    def test_ids_win_over_embeds_without_query_attention(self):
        obj = GenerateReqInput(
            input_ids=[1, 2, 3],
            input_embeds=[[0.0], [0.0], [0.0]],
        )
        obj.normalize_batch_and_arguments()
        self.assertIsNone(obj.input_embeds)

    def test_context_forwards_keep_ids_and_embeds(self):
        # regression: the batch-size normalization used to clear input_embeds
        # whenever input_ids was present, silently degrading context forwards
        # to plain embedding lookups
        obj = GenerateReqInput(
            input_ids=[1, 2, 3],
            input_embeds=[[0.0], [0.0]],
            query_attention="bidirectional",
            token_positions=[1, 2],
        )
        obj.normalize_batch_and_arguments()
        self.assertEqual(obj.input_embeds, [[0.0], [0.0]])
        self.assertTrue(obj.is_single)

    def test_context_forward_rejects_parallel_sampling_before_expansion(self):
        obj = GenerateReqInput(
            input_ids=[1, 2, 3],
            input_embeds=[[0.0], [0.0]],
            query_attention="bidirectional",
            sampling_params={"n": 2},
        )
        with self.assertRaisesRegex(ValueError, "query_attention requires n=1"):
            obj.normalize_batch_and_arguments()

    def test_legacy_shm_transport_does_not_fall_back_inline(self):
        obj = GenerateReqInput(
            input_ids=[1],
            hidden_states_transport="shm",
        )
        with self.assertRaisesRegex(ValueError, "caller-owned hidden_states_buffer"):
            obj.normalize_batch_and_arguments()


class TestPrefillHiddenStateSlicing(CustomTestCase):
    """Guards the batched-context-forward fix: rows are per-request extend
    widths, and the offset advances for every processed request."""

    def _req(self, *, returning, origin_len, buffer=None):
        req = MagicMock()
        req.return_hidden_states = returning
        req.hidden_states = []
        req.hidden_states_buffer = buffer
        req.extend_batch_idx = 1
        req.origin_input_ids = list(range(origin_len))
        return req

    def _append(self, req, logits_output, offset, extend_len):
        from sglang.srt.managers.scheduler_components.batch_result_processor import (
            SchedulerBatchResultProcessor,
        )
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return SchedulerBatchResultProcessor._append_prefill_hidden_states(
            None,
            req=req,
            logits_output=logits_output,
            hidden_state_offset=offset,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            extend_input_len=extend_len,
            store=True,
        )

    def test_two_requests_get_their_own_rows(self):
        from types import SimpleNamespace

        hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        logits_output = SimpleNamespace(hidden_states=hidden)
        # origin lengths deliberately differ from extend widths (radix hits)
        req0 = self._req(returning=True, origin_len=9)
        req1 = self._req(returning=True, origin_len=7)

        offset = self._append(req0, logits_output, 0, 2)
        offset = self._append(req1, logits_output, offset, 3)
        self.assertEqual(offset, 5)
        self.assertEqual(req0.hidden_states, [hidden[0:2].tolist()])
        self.assertEqual(req1.hidden_states, [hidden[2:5].tolist()])

    def test_non_returning_request_still_advances_the_offset(self):
        from types import SimpleNamespace

        hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        logits_output = SimpleNamespace(hidden_states=hidden)
        skipped = self._req(returning=False, origin_len=5)
        returning = self._req(returning=True, origin_len=5)

        offset = self._append(skipped, logits_output, 0, 3)
        self._append(returning, logits_output, offset, 1)
        self.assertEqual(skipped.hidden_states, [])
        self.assertEqual(returning.hidden_states, [hidden[3:4].tolist()])

    def test_caller_owned_buffer_keeps_tensor_until_streaming(self):
        from types import SimpleNamespace

        hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        req = self._req(
            returning=True,
            origin_len=3,
            buffer={"transport": "shm"},
        )
        self._append(req, SimpleNamespace(hidden_states=hidden), 0, 3)
        torch.testing.assert_close(req.hidden_states[0], hidden)


class TestHiddenStatesBufferOutput(CustomTestCase):
    def test_late_write_failure_becomes_a_bad_request(self):
        from http import HTTPStatus
        from types import SimpleNamespace

        from sglang.srt.managers.scheduler_components.output_streamer import (
            _write_hidden_states_buffer,
        )

        req = SimpleNamespace(
            rid="request",
            hidden_states_buffer={"transport": "shm"},
            finished_reason=None,
        )
        with patch(
            "sglang.srt.managers.scheduler_components.output_streamer."
            "package_hidden_states",
            side_effect=FileNotFoundError("caller removed buffer"),
        ):
            result = _write_hidden_states_buffer(req, [torch.ones(1, 4)])

        self.assertIsNone(result)
        self.assertEqual(req.finished_reason.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(req.finished_reason.err_type, "BadRequestError")
        self.assertIn("caller removed buffer", req.finished_reason.message)


class TestContextForwardEnvelope(CustomTestCase):
    """Each v1 envelope pin must reject on its own."""

    def _stub(self, *, model_overrides=None, **overrides):
        from types import SimpleNamespace

        server_args = SimpleNamespace(
            attention_backend="fa3",
            prefill_attention_backend=None,
            decode_attention_backend=None,
            disaggregation_mode="null",
            speculative_algorithm=None,
            page_size=1,
            chunked_prefill_size=8192,
            disable_radix_cache=False,
            tp_size=1,
            pp_size=1,
        )
        server_args.get_attention_backends = lambda: (
            server_args.prefill_attention_backend or server_args.attention_backend,
            server_args.decode_attention_backend or server_args.attention_backend,
        )
        for name, value in overrides.items():
            setattr(server_args, name, value)
        model_config = SimpleNamespace(
            context_len=8192,
            hidden_size=4,
            is_local_attention_model=False,
            is_hybrid_swa=False,
            sliding_window_size=None,
            attention_chunk_size=None,
        )
        for name, value in (model_overrides or {}).items():
            setattr(model_config, name, value)
        return SimpleNamespace(
            server_args=server_args,
            model_config=model_config,
            context_len=model_config.context_len,
        )

    def _obj(self, **overrides):
        obj = GenerateReqInput(
            input_ids=[1, 2, 3],
            input_embeds=[[0.0] * 4],
            query_attention="bidirectional",
            token_positions=[2],
            sampling_params={"max_new_tokens": 0},
            return_hidden_states=True,
        )
        for name, value in overrides.items():
            setattr(obj, name, value)
        return obj

    def _validate(self, stub, obj):
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        TokenizerManager._validate_context_forward_fields(stub, obj)

    def test_base_envelope_passes(self):
        self._validate(self._stub(), self._obj())

    def test_each_pin_rejects(self):
        cases = [
            (self._stub(), self._obj(query_attention="diagonal")),
            (self._stub(), self._obj(input_embeds=None)),
            (self._stub(), self._obj(session_id="radix-session")),
            (self._stub(), self._obj(session_params={"id": "session"})),
            (self._stub(), self._obj(sampling_params={"max_new_tokens": 1})),
            (self._stub(attention_backend="triton"), self._obj()),
            (self._stub(prefill_attention_backend="triton"), self._obj()),
            (self._stub(disaggregation_mode="prefill"), self._obj()),
            (self._stub(speculative_algorithm="EAGLE"), self._obj()),
            (self._stub(page_size=16), self._obj()),
            (self._stub(tp_size=2), self._obj()),
            (self._stub(pp_size=2), self._obj()),
            (
                self._stub(chunked_prefill_size=2),
                self._obj(
                    input_ids=[1, 2, 3],
                    input_embeds=[[0.0] * 4] * 3,
                    token_positions=[0, 1, 2],
                ),
            ),
        ]
        for stub, obj in cases:
            with self.assertRaises(ValueError):
                self._validate(stub, obj)

    def test_integer_and_empty_embed_rows_report_a_shape(self):
        """JSON clients emit `0`, not `0.0`. A float-only probe returned None
        for those rows, and the caller treats None as "skip the hidden-size
        check", which re-opened the in-kernel scheduler crash the pin exists
        to prevent."""
        from sglang.srt.managers.tokenizer_manager import _input_embeds_shape

        self.assertEqual(_input_embeds_shape([[0, 0, 0]]), (1, 3))
        self.assertEqual(_input_embeds_shape([[0] * 4]), (1, 4))
        self.assertEqual(_input_embeds_shape([[0, 1.0, 2.0, 3.0]]), (1, 4))
        self.assertEqual(_input_embeds_shape([[]]), (1, 0))
        self.assertEqual(_input_embeds_shape([[0.0] * 4]), (1, 4))
        self.assertEqual(
            _input_embeds_shape(
                {
                    "transport": "shm",
                    "shape": [4, 8],
                    "dtype": "float32",
                    "name": "sgl_shm_test_123_deadbeef",
                }
            ),
            (4, 8),
        )
        with self.assertRaisesRegex(ValueError, "finite numbers"):
            _input_embeds_shape([["a", "b"]])

    def test_embed_rows_are_numeric_and_rectangular(self):
        for embeds in (
            [[0.0] * 4, [0.0] * 3],
            [[0.0, 0.0, 0.0, "x"]],
            [[0.0, 0.0, 0.0, True]],
            [[0.0, 0.0, 0.0, float("nan")]],
            [[0.0, 0.0, 0.0, float("inf")]],
            [[0.0, 0.0, 0.0, -float("inf")]],
            [[0.0, 0.0, 0.0, 1e308]],
        ):
            with self.subTest(embeds=embeds), self.assertRaises(ValueError):
                self._validate(self._stub(), self._obj(input_embeds=embeds))

        self._validate(
            self._stub(),
            self._obj(input_embeds=[[float(np.finfo(np.float32).max)] * 4]),
        )

    def test_token_positions_are_integer_1d_or_2d(self):
        for positions in (
            [],
            [[0], []],
            [[0], [1, 2]],
            [0.5],
            [True],
            [-1],
            [2**63 - 1],
            [2**63],
            [[0], [2**63]],
            [[[0]]],
        ):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                self._validate(self._stub(), self._obj(token_positions=positions))

    def test_token_positions_respect_active_context_length(self):
        stub = self._stub(model_overrides={"context_len": 4})
        self._validate(stub, self._obj(token_positions=[3]))
        self._validate(stub, self._obj(token_positions=[[3], [0], [1]]))

        for positions in ([4], [[3], [4], [0]]):
            with (
                self.subTest(positions=positions),
                self.assertRaisesRegex(ValueError, "context length \\(4\\)"),
            ):
                self._validate(stub, self._obj(token_positions=positions))

    def test_split_prefill_backend_is_honored(self):
        # base backend non-fa is fine when the resolved prefill backend is fa3
        self._validate(
            self._stub(attention_backend="triton", prefill_attention_backend="fa3"),
            self._obj(),
        )

    def test_local_attention_models_are_rejected(self):
        cases = (
            {"is_local_attention_model": True},
            {"is_hybrid_swa": True},
            {"sliding_window_size": 4096},
            {"sliding_window_size": [4096, None]},
            {"attention_chunk_size": 1024},
        )
        for model_overrides in cases:
            with (
                self.subTest(model_overrides=model_overrides),
                self.assertRaisesRegex(ValueError, "sliding-window or local-attention"),
            ):
                self._validate(
                    self._stub(model_overrides=model_overrides),
                    self._obj(),
                )

        for disabled_window in (0, -1):
            self._validate(
                self._stub(
                    model_overrides={
                        "sliding_window_size": disabled_window,
                        "attention_chunk_size": disabled_window,
                    }
                ),
                self._obj(),
            )

    def test_hidden_states_buffer_is_validated(self):
        ref = {
            "transport": "shm",
            "name": "sgl_shm_hs_123_deadbeef",
            "dtype": "float32",
            "shape": [1, 4],
        }
        with patch(
            "sglang.srt.managers.tokenizer_manager.validate_shm_tensor_buffer"
        ) as validate:
            self._validate(self._stub(), self._obj(hidden_states_buffer=ref))
        validate.assert_called_once_with(ref, shape=(1, 4), dtype="float32")

        with (
            patch(
                "sglang.srt.managers.tokenizer_manager.validate_shm_tensor_buffer",
                side_effect=FileNotFoundError("missing"),
            ),
            self.assertRaisesRegex(ValueError, "invalid hidden_states_buffer"),
        ):
            self._validate(self._stub(), self._obj(hidden_states_buffer=ref))

    def test_hidden_states_buffer_rejects_parallel_sampling(self):
        obj = self._obj(
            hidden_states_buffer={"transport": "shm"},
            sampling_params={"max_new_tokens": 0, "n": 2},
        )
        with self.assertRaises(ValueError):
            self._validate(self._stub(), obj)

    def test_hidden_states_buffer_requires_context_forward(self):
        obj = self._obj(
            query_attention=None,
            input_embeds=None,
            token_positions=None,
            hidden_states_buffer={"transport": "shm"},
        )
        with self.assertRaisesRegex(ValueError, "requires.*context forward"):
            self._validate(self._stub(), obj)

    def test_hidden_states_buffer_rejects_last_hidden_state_mode(self):
        obj = self._obj(
            hidden_states_buffer={"transport": "shm"},
            return_hidden_states="last",
        )
        with self.assertRaisesRegex(ValueError, "return_hidden_states=true"):
            self._validate(self._stub(), obj)


if __name__ == "__main__":
    unittest.main()
