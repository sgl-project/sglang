"""Exercise dLLM metric wiring through the real output components on CPU."""

import asyncio
import unittest
from array import array
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from test_incremental_detokenize import _detokenizer
from test_output_streamer_customized_info import _enter_streamer_scope
from test_tokenizer_manager_rid_cleanup import (
    _make_abort_req,
    _make_req_state,
    _make_tokenizer_manager,
)

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.managers.req_types import FINISH_ABORT, FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDllmMetricsPipeline(unittest.TestCase):
    def _pipeline(
        self, stream=True, incremental=False, skip_tokenizer=False, dllm=True
    ):
        _enter_streamer_scope(self)
        tm = _make_tokenizer_manager(self)
        tm.incremental_streaming_output = incremental
        detok = _detokenizer()
        emitted = []
        scheduler = SimpleNamespace(
            get_output_streamer_class=lambda: SchedulerOutputStreamer,
            ipc_channels=SimpleNamespace(
                send_to_detokenizer=SimpleNamespace(send_output=emitted.append)
            ),
            tree_cache=None,
            server_args=SimpleNamespace(enable_lmcache=False),
            is_generation=True,
            dllm_config=object() if dllm else None,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            skip_tokenizer_init=skip_tokenizer,
            enable_hicache_storage=False,
            enable_lmcache=False,
            rust_server=None,
            uocr_runtime=None,
        )
        Scheduler.init_output_streamer(scheduler)
        reqs, states = [], []
        for rid in ("first", "second"):
            params = SamplingParams()
            params.normalize(None)
            reqs.append(Req(rid, "", array("q", [1, 2]), params, stream=stream))
            state = _make_req_state(rid)
            state.obj.stream = stream
            state.obj.return_sampling_mask = False
            tm.rid_to_state[rid] = state
            states.append(state)

        def deliver(rows):
            scheduler.output_streamer.stream_output(rows, False)
            if not emitted:
                self.assertTrue(
                    all(not req.stream and not req.finished() for req in rows)
                )
                return
            payload = emitted.pop()
            self.assertEqual(payload.rids, [req.rid for req in rows])
            if not skip_tokenizer:
                payload = detok.handle_batch_token_id_out(payload)
            asyncio.run(tm._handle_batch_output(payload))

        return tm, reqs, states, deliver

    def test_cumulative_metrics_across_output_modes_and_reordered_rows(self):
        for stream, incremental in ((False, False), (True, False), (True, True)):
            for skip_tokenizer in (False, True):
                with self.subTest(
                    stream=stream,
                    incremental=incremental,
                    skip_tokenizer=skip_tokenizer,
                ):
                    tm, reqs, states, deliver = self._pipeline(
                        stream, incremental, skip_tokenizer
                    )
                    # Zero-token rounds must retain work and define zero-denominator TPF.
                    self._set_row(reqs[0], [], (0, 0, 0, 0))
                    self._set_row(reqs[1], [], (2, 2, 0, 0))
                    deliver(reqs)
                    if stream:
                        self._assert_metrics(states[0], (0, 0, 0, 0), 0)
                        self._assert_metrics(states[1], (2, 2, 0, 0), 0)
                    # Different values expose accidental row or field reuse.
                    self._set_row(reqs[0], [10], (6, 1, 3, 2))
                    self._set_row(reqs[1], [20, 21], (11, 2, 5, 4))
                    deliver(reqs)
                    if stream:
                        self._assert_metrics(states[0], (6, 1, 3, 2), 1)
                        self._assert_metrics(states[1], (11, 2, 5, 4), 2)
                    self._set_row(reqs[0], [10, 11], (8, 1, 5, 2))
                    self._set_row(reqs[1], [20, 21, 22], (14, 2, 8, 4))
                    reqs[0].finished_reason = FINISH_LENGTH(length=2)
                    reqs[1].finished_reason = FINISH_ABORT("test abort")
                    deliver(reqs[::-1])
                    self._assert_metrics(states[0], (8, 1, 5, 2), 2)
                    self._assert_metrics(states[1], (14, 2, 8, 4), 3)
                    self.assertEqual(
                        states[1].out_list[-1]["meta_info"]["finish_reason"]["type"],
                        "abort",
                    )
                    self.assertEqual(tm.rid_to_state, {})
                    if incremental:
                        for state in states:
                            state.out_list[0]["meta_info"].pop("dllm_metrics")
                            merged = tm._coalesce_streaming_chunks(
                                state.out_list, "metrics"
                            )
                            self.assertEqual(
                                merged["meta_info"]["dllm_metrics"],
                                state.out_list[-1]["meta_info"]["dllm_metrics"],
                            )

    def test_abort_echo_does_not_inherit_earlier_metrics(self):
        tm, reqs, states, deliver = self._pipeline(incremental=True)
        self._set_row(reqs[0], [10, 11], (5, 1, 3, 1))
        deliver(reqs[:1])
        self._assert_metrics(states[0], (5, 1, 3, 1), 2)
        tm._handle_abort_req(_make_abort_req(reqs[0].rid))
        merged = tm._coalesce_streaming_chunks(states[0].out_list, reqs[0].rid)
        self.assertEqual(merged["meta_info"]["finish_reason"]["type"], "abort")
        self.assertEqual(merged["meta_info"]["completion_tokens"], 2)
        self.assertNotIn("dllm_metrics", merged["meta_info"])

    def test_ar_output_has_no_dllm_metrics(self):
        _, reqs, states, deliver = self._pipeline(dllm=False)
        reqs[0].finished_reason = FINISH_LENGTH(length=0)
        deliver(reqs[:1])
        self.assertNotIn("dllm_metrics", states[0].out_list[-1]["meta_info"])

    @staticmethod
    def _set_row(req, token_ids, counts):
        req.output_ids = array("q", token_ids)
        (
            req.dllm_forward_count,
            req.dllm_prompt_forward_count,
            req.dllm_generation_forward_count,
            req.dllm_recompute_forward_count,
        ) = counts

    def _assert_metrics(self, state, counts, tokens):
        total, prompt, generation, recompute = counts
        meta = state.out_list[-1]["meta_info"]
        self.assertEqual(meta["completion_tokens"], tokens)
        self.assertEqual(
            meta["dllm_metrics"],
            {
                "forward_count": total,
                "prompt_forward_count": prompt,
                "generation_forward_count": generation,
                "recompute_forward_count": recompute,
                "completion_tokens": tokens,
                "tokens_per_forward": tokens / generation if generation else 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
