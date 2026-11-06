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
from test_tokenizer_manager_rid_cleanup import _make_req_state, _make_tokenizer_manager

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
    def test_reordered_rows_keep_metrics_through_streamer_and_detokenizer(self):
        for incremental in (False, True):
            with self.subTest(incremental=incremental):
                self._run_pipeline(incremental)

    def _run_pipeline(self, incremental):
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
            dllm_config=object(),
            spec_algorithm=SpeculativeAlgorithm.NONE,
            disaggregation_mode=DisaggregationMode.NULL,
            skip_tokenizer_init=False,
            enable_hicache_storage=False,
            enable_lmcache=False,
            rust_server=None,
            uocr_runtime=None,
        )
        Scheduler.init_output_streamer(scheduler)
        reqs, states = {}, {}
        for rid in ("first", "second"):
            params = SamplingParams()
            params.normalize(None)
            reqs[rid] = Req(rid, "", array("q", [1, 2]), params, stream=True)
            state = _make_req_state(rid)
            state.obj.stream = True
            state.obj.return_sampling_mask = False
            tm.rid_to_state[rid] = states[rid] = state

        # Different values in every column expose accidental row or field reuse.
        self._set_row(reqs["first"], [10], (6, 1, 3, 2))
        self._set_row(reqs["second"], [20, 21], (11, 2, 5, 4))
        scheduler.output_streamer.stream_output(list(reqs.values()), False)
        asyncio.run(
            tm._handle_batch_output(detok.handle_batch_token_id_out(emitted.pop()))
        )
        self._assert_metrics(states["first"], (6, 1, 3, 2), 1)
        self._assert_metrics(states["second"], (11, 2, 5, 4), 2)

        self._set_row(reqs["first"], [10, 11], (8, 1, 5, 2))
        self._set_row(reqs["second"], [20, 21, 22], (14, 2, 8, 4))
        reqs["first"].finished_reason = FINISH_LENGTH(length=2)
        reqs["second"].finished_reason = FINISH_ABORT("test abort")
        scheduler.output_streamer.stream_output([reqs["second"], reqs["first"]], False)
        payload = emitted.pop()
        self.assertEqual(payload.rids, ["second", "first"])
        asyncio.run(tm._handle_batch_output(detok.handle_batch_token_id_out(payload)))
        self._assert_metrics(states["first"], (8, 1, 5, 2), 2)
        self._assert_metrics(states["second"], (14, 2, 8, 4), 3)
        self.assertEqual(
            states["second"].out_list[-1]["meta_info"]["finish_reason"]["type"],
            "abort",
        )
        self.assertEqual(tm.rid_to_state, {})

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
                "tokens_per_forward": tokens / generation,
            },
        )


if __name__ == "__main__":
    unittest.main()
