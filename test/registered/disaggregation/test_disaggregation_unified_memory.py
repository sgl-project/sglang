"""PD disaggregation with --enable-unified-memory (MLA hybrid-Mamba).

Guards the unified-memory PD transfer scheme end to end: whole page-envelope
KV registration (`UnifiedMLATokenToKVPool.get_contiguous_buf_infos`), whole
slot-envelope KDA/mamba state transfer, virtual->physical index translation at
the prefill send / decode prealloc sites, and the compaction move gate. A
regression in any of them shifts the decode-side KV/state bytes and breaks
logprob parity with the non-PD unified-memory reference.

`--attention-backend` is deliberately NOT pinned, matching
`models_e2e/test_kimi_linear_unified_memory.py`, which documents that pinning
hides defects reachable only under the resolved default. The transferred bytes
are backend-independent, so the default (fa3 on this suite's H100 runner) covers
this file's subject either way. The linear-attn/Mamba backends stay pinned to
triton -- the page-major layout requires them.

`--enable-deterministic-inference` is deliberately NOT set. It would only guard
against batch-shape-dependent kernel variation, and the reference and P+D paths
run the same shapes: measured, two fresh servers on separate GPUs produce
bit-identical logits without it. Setting it would narrow the test to the
batch-invariant op set and a non-default sampling backend -- a less
representative config -- and couple a PD-transfer test to the deterministic code
path, so a defect there would fail this file for an unrelated reason.
"""

import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)

register_cuda_ci(est_time=900, stage="base-c", runner_config="4-gpu-h100")

KIMI_LINEAR_MODEL = "yujiepan/kimi-linear-tiny-random"
SERVER_ENV = {"SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM": "0"}
SERVER_ARGS = [
    "--skip-tokenizer-init",
    "--random-seed",
    "1",
    "--enable-unified-memory",
    "--linear-attn-backend",
    "triton",
    "--mamba-backend",
    "triton",
    "--max-mamba-cache-size",
    "32",
    "--max-total-tokens",
    "4096",
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
]


class TestUnifiedMemoryDisaggregation(PDDisaggregationServerBase):
    """1 prefill + 1 decode, both with --enable-unified-memory, vs a non-PD
    unified-memory reference server."""

    prefill_tp_size = 1
    decode_tp_size = 1
    decode_base_gpu_id = 1
    extra_prefill_args = SERVER_ARGS
    extra_decode_args = SERVER_ARGS
    extra_prefill_env = SERVER_ENV
    extra_decode_env = SERVER_ENV
    baseline_args = SERVER_ARGS

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = KIMI_LINEAR_MODEL

    @staticmethod
    def generate(base_url):
        response = requests.post(
            base_url + "/generate",
            json={
                "input_ids": [1] + [100 + i % 1000 for i in range(256)],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 4,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "top_logprobs_num": 5,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["meta_info"]

    def test_logprob_parity(self):
        baseline = popen_launch_server(
            self.model,
            self.lb_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code"] + self.baseline_args,
            env=SERVER_ENV,
        )
        try:
            reference = self.generate(self.lb_url)
        finally:
            kill_process_tree(baseline.pid, wait_timeout=60)
        time.sleep(5)

        self.launch_all()
        disaggregated = self.generate(self.lb_url)

        reference_logprobs = reference["output_token_logprobs"]
        disaggregated_logprobs = disaggregated["output_token_logprobs"]
        self.assertEqual(
            [item[1] for item in reference_logprobs],
            [item[1] for item in disaggregated_logprobs],
        )
        self.assertEqual(len(reference_logprobs), 4)
        for reference_item, disaggregated_item in zip(
            reference_logprobs, disaggregated_logprobs
        ):
            self.assertAlmostEqual(reference_item[0], disaggregated_item[0], delta=0.05)

        assert_process_healthy(self, "load balancer", self.process_lb, self.lb_url)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


class TestUnifiedMemoryDisaggregationChunkedPrefill(TestUnifiedMemoryDisaggregation):
    """Multi-chunk prefill (257-token prompt, 64-token chunks): each chunk's KV
    pages are translated to physical ids and shipped while later chunks still
    run, exercising the chunked send path and the prefill-side move gate
    (`chunked_req.start_send_idx > 0`). The reference server uses the same
    chunk size so any parity break isolates to the PD transfer.
    """

    _chunked_args = SERVER_ARGS + ["--chunked-prefill-size", "64"]
    extra_prefill_args = _chunked_args
    extra_decode_args = _chunked_args
    baseline_args = _chunked_args


if __name__ == "__main__":
    unittest.main()
