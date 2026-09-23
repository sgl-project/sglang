"""Mixed prefill/decode batches for native Beam through the scripted runtime.

The scheduler folds running decodes into a prefill chunk only when no request
asks for logprobs, so the KL helpers never reach this path. Here two greedy
decoders, one past the 512-token window, keep decoding while a 1200-token
prompt prefills in chunks of at most 256 tokens. The batch log must show a
MIXED forward that carries a non-final chunk beside both decoders. A custom
logit processor keeps every pre-sampling row, and each generated position must
match the eager BF16 oracle at the same prefix under the fixture's KL bound.
Greedy ids are not compared across batch shapes: BF16 logits on this fixture
often tie exactly, so the argmax depends on kernel rounding. The harness
disables prefill CUDA graphs, so mixed batches run the eager prefill path.
"""

import os
import tempfile
import unittest
import uuid

import torch

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.utils import is_flashinfer_available
from sglang.srt.utils.common import (
    is_sm90_supported,
    is_sm100_supported,
    is_sm120_supported,
)
from sglang.test.beam_fixture import (
    assert_logprobs_close,
    eager_beam_logits,
    make_beam_fixture,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.context.http_post import (
    _http_post_and_await_recv_msg,
)
from sglang.test.scripted_runtime.req_handle import ScriptedReqHandle
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    DEFAULT_MAX_STEPS,
    advance_to_decode_step,
    base_engine_kwargs,
)

register_cuda_ci(est_time=150, stage="base-b", runner_config="1-gpu-large")

# (prompt token, prompt length, new tokens). Decoders must outlast every chunk.
_DECODERS = ((5, 64, 32), (17, 513, 32))
_CHUNKED = (29, 1200, 8)


class _RowRecorder(CustomLogitProcessor):
    """Append each pre-sampling logits row to its own request's params."""

    def __call__(self, logits, custom_param_list=None):
        for row, params in zip(logits, custom_param_list):
            params["rows"].append(row.float().cpu())
        return logits


def _start(t: ScriptedContext, spec):
    # start_req's payload plus the recorder; logprobs stay off so mixing is legal.
    token, length, new_tokens = spec
    rid = f"beam-mixed-{uuid.uuid4().hex}"
    payload = {
        "input_ids": [token] * length,
        "sampling_params": {
            "max_new_tokens": new_tokens,
            "ignore_eos": True,
            "temperature": 0.0,
            "custom_params": {"rows": []},
        },
        "rid": rid,
        "stream": True,
        "return_logprob": False,
        "custom_logit_processor": _RowRecorder.to_str(),
    }
    _http_post_and_await_recv_msg(
        t,
        path="/generate",
        json=payload,
        predicate=lambda obj: getattr(obj, "rid", None) == rid,
        description=f"request with rid {rid!r}",
    )
    return ScriptedReqHandle(rid=rid, context=t)


def _run_to_idle(t: ScriptedContext, handles):
    """Drive until every request finishes and the scheduler idles; return each Req."""
    reqs = {}
    for _ in range(DEFAULT_MAX_STEPS):
        for handle in handles:
            req = handle.req
            if req is not None:
                reqs[handle.rid] = req
        if all(handle.finished for handle in handles) and t.is_fully_idle:
            return reqs
        yield
    raise AssertionError(f"requests did not finish in {DEFAULT_MAX_STEPS} steps")


class TestBeamMixedChunk(ScriptedTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Beam mixed-chunk test requires a CUDA GPU")
        if not is_flashinfer_available() or not (
            is_sm90_supported() or is_sm100_supported() or is_sm120_supported()
        ):
            raise unittest.SkipTest(
                "Beam TRT-LLM MHA test requires FlashInfer and SM90, SM100, or SM120"
            )
        cls.checkpoint = tempfile.TemporaryDirectory(prefix="beam-fixture-")
        cls.config, state = make_beam_fixture(cls.checkpoint.name)
        cls.state = {name: value.cuda() for name, value in state.items()}
        cls.ENGINE_KWARGS = base_engine_kwargs(
            model_path=cls.checkpoint.name,
            chunked_prefill_size=256,
            dtype="bfloat16",
            attention_backend="trtllm_mha",
            page_size=64,
            max_total_tokens=4096,
            mem_fraction_static=0.2,
            enable_mixed_chunk=True,
            enable_custom_logit_processor=True,
        )
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        try:
            if hasattr(cls, "server"):
                super().tearDownClass()
        finally:
            if hasattr(cls, "state"):
                del cls.state
            if hasattr(cls, "checkpoint"):
                cls.checkpoint.cleanup()

    def test_mixed_batches_match_oracle(self):
        # The script runs in the scheduler process; rows come back via a file.
        with tempfile.TemporaryDirectory(prefix="beam-mixed-rows-") as rows_dir:
            rows_path = os.path.join(rows_dir, "rows.pt")
            self.server.execute_script(self._script_mixed_batches, args=(rows_path,))
            results = torch.load(rows_path)
        for (token, length, new_tokens), output_ids, rows, modes in results:
            with self.subTest(prompt_token=token, prompt_length=length):
                self.assertEqual(len(output_ids), new_tokens)
                self.assertEqual(rows.shape[0], new_tokens)
                if (token, length, new_tokens) in _DECODERS:
                    self.assertIn("mixed", modes, "no compared row was mixed")
                prompt = [token] * length
                logits = eager_beam_logits(
                    self.config, self.state, prompt + output_ids[:-1]
                )
                assert_logprobs_close(
                    self,
                    rows.cuda().log_softmax(-1),
                    logits[length - 1 :].log_softmax(-1),
                    tag=f"mixed prompt={token}x{length}",
                )

    @staticmethod
    def _script_mixed_batches(t: ScriptedContext, rows_path: str):
        decoders = [_start(t, spec) for spec in _DECODERS]
        for handle in decoders:
            yield from advance_to_decode_step(handle, 1)
        chunked = _start(t, _CHUNKED)
        handles = [*decoders, chunked]
        reqs = yield from _run_to_idle(t, handles)

        decoder_rids = {handle.rid for handle in decoders}
        log = t._scheduler_hook._batch_log
        assert any(
            record.mode == "mixed"
            and record.chunked_rid == chunked.rid
            and decoder_rids <= set(record.rids)
            for record in log
        ), (
            "no MIXED forward ran a prompt chunk beside both decoders; modes: "
            f"{[record.mode for record in log]}"
        )
        results = []
        for handle, spec in zip(handles, (*_DECODERS, _CHUNKED)):
            req = reqs[handle.rid]
            records = [record for record in log if handle.rid in record.rids]
            rows = req.sampling_params.custom_params["rows"]
            # One recorder call per batch holding the request, in launch order.
            assert len(rows) == len(records), f"{len(rows)} rows, {len(records)} runs"
            # Non-final chunk rows yield no token; an overlap step may trail the last.
            produced = [
                (row, record.mode)
                for row, record in zip(rows, records)
                if record.chunked_rid != handle.rid
            ][: spec[2]]
            results.append(
                (
                    spec,
                    list(req.output_ids),
                    torch.stack([row for row, _ in produced]),
                    [mode for _, mode in produced],
                )
            )
        torch.save(results, rows_path)


if __name__ == "__main__":
    unittest.main()
