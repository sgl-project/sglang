import json
import time
import unittest
import uuid

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.customized_info_sampler import (
    CUSTOMIZED_INFO_ARRAY_FIELD,
    CUSTOMIZED_INFO_FIELD,
    CUSTOMIZED_INFO_SAMPLER_BACKEND,
)
from sglang.test.kits.pd_parity_kit import PDLogprobParityMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_cuda_ci(est_time=315, stage="extra-a", runner_config="2-gpu-large")

KIMI_LINEAR_MODEL = "yujiepan/kimi-linear-tiny-random"
# Smallest in-tree GDN hybrid: MHA full attention + gated-delta-net linear
# layers, i.e. the unified pool's MHA sub-pool rather than the MLA one.
QWEN_GDN_MODEL = "Qwen/Qwen3.5-0.8B"
SERVER_ENV = {
    "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM": "0",
    "SGLANG_DISAGG_CUSTOMIZED_INFO_MAX_BYTES": "4096",
}

# --attention-backend and --enable-deterministic-inference are deliberately
# absent: bytes are backend-independent and both paths run identical shapes.
UNIFIED_MEMORY_ARGS = [
    "--skip-tokenizer-init",
    "--random-seed",
    "1",
    "--enable-unified-memory",
    # Page-major layout requires triton for both.
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


class TestUnifiedMemoryDisaggregation(PDLogprobParityMixin, PDDisaggregationServerBase):
    """1 prefill + 1 decode, both with --enable-unified-memory, vs a non-PD
    unified-memory reference server."""

    model = KIMI_LINEAR_MODEL
    extra_prefill_env = SERVER_ENV
    extra_decode_env = SERVER_ENV
    prefill_tp_size = 1
    decode_tp_size = 1
    decode_base_gpu_id = 1
    launch_module = "sglang.test.customized_info_sampler"
    baseline_args = UNIFIED_MEMORY_ARGS
    extra_prefill_args = UNIFIED_MEMORY_ARGS + [
        "--sampling-backend",
        CUSTOMIZED_INFO_SAMPLER_BACKEND,
    ]
    extra_decode_args = extra_prefill_args

    def _assert_logprob_parity(self, *, reference, actual, label):
        super()._assert_logprob_parity(reference=reference, actual=actual, label=label)
        token_ids = [item[1] for item in actual["output_token_logprobs"]]
        self.assertEqual(actual[CUSTOMIZED_INFO_FIELD], token_ids, label)
        self.assertEqual(
            actual[CUSTOMIZED_INFO_ARRAY_FIELD], [[token] for token in token_ids], label
        )


class TestUnifiedMemoryDisaggregationChunkedPrefill(TestUnifiedMemoryDisaggregation):
    """Multi-chunk prefill (1025-token prompt, 64-token chunks): each chunk's KV
    pages are translated to physical ids and shipped while later chunks still
    run, exercising the chunked send path and the prefill-side move gate
    (`chunked_req.start_send_idx > 0`). The reference server uses the same
    chunk size so any parity break isolates to the PD transfer. A streaming
    cancellation after handoff must also return decode capacity for reuse.
    """

    parity_prompts = [[1] + [100 + i % 1000 for i in range(1024)]]
    extra_prefill_env = {
        **SERVER_ENV,
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY": "1",
    }
    extra_decode_env = extra_prefill_env
    _chunked_args = UNIFIED_MEMORY_ARGS + ["--chunked-prefill-size", "64"]
    baseline_args = _chunked_args
    extra_prefill_args = TestUnifiedMemoryDisaggregation.extra_prefill_args + [
        "--chunked-prefill-size",
        "64",
        "--enable-metrics",
    ]
    extra_decode_args = extra_prefill_args

    def _wait_for_decode_load(self, predicate, message):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            response = requests.get(
                self.decode_url + "/v1/loads?include=core,disagg", timeout=10
            )
            response.raise_for_status()
            (load,) = response.json()["loads"]
            if predicate(load):
                return load
            time.sleep(0.05)
        self.fail(f"{message}: {load}")

    def _wait_for_decode_idle(self, after):
        return self._wait_for_decode_load(
            lambda load: (
                load["timestamp"] > after
                and load["num_running_reqs"] == 0
                and load["num_waiting_reqs"] == 0
                and load["num_used_tokens"] == 0
            ),
            "Decode did not return its active KV capacity",
        )

    def _cancel_after_handoff(self, prompt):
        rid = "unified-pd-abort-" + uuid.uuid4().hex
        with requests.post(
            self.lb_url + "/generate",
            json={
                "rid": rid,
                "input_ids": prompt,
                "stream": True,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 2048,
                    "ignore_eos": True,
                },
            },
            stream=True,
            timeout=60,
        ) as response:
            response.raise_for_status()
            events = (
                json.loads(line[6:])
                for line in response.iter_lines()
                if line.startswith(b"data: ") and line != b"data: [DONE]"
            )
            for event in events:
                if event["meta_info"]["completion_tokens"] > 0:
                    break
            else:
                self.fail("No token arrived before cancellation")

            # A generated token plus an active decode allocation establishes
            # that the normal PD handoff has completed, without a sleep race.
            active = self._wait_for_decode_load(
                lambda load: (
                    load["num_running_reqs"] == 1
                    and load["num_used_tokens"] >= len(prompt)
                ),
                "Did not observe the streamed request in decode",
            )
            requests.post(
                self.decode_url + "/abort_request",
                json={"rid": rid, "abort_all": False},
                timeout=10,
            ).raise_for_status()
            for event in events:
                reason = event["meta_info"]["finish_reason"]
                if reason is not None:
                    self.assertEqual(reason["type"], "abort", event)
                    break
            else:
                self.fail("No terminal abort was emitted")
        return active

    def _check_prompt_parity(self, *, prompt, reference):
        super()._check_prompt_parity(prompt=prompt, reference=reference)
        self._flush_cache(self.prefill_url)
        self._flush_cache(self.decode_url)
        active = self._cancel_after_handoff(prompt)
        idle = self._wait_for_decode_idle(active["timestamp"])

        # The decode prefix cache is disabled by default. Without resetting
        # either worker, cycle more prompt tokens than fit in its 4096-token
        # budget and compare every new result with the non-PD reference.
        capacity = active["max_total_num_tokens"]
        self.assertGreater(capacity, 0)
        self.assertLessEqual(capacity, 4096)
        for i in range(capacity // len(prompt) + 1):
            actual = self._generate(base_url=self.lb_url, input_ids=prompt)
            self._assert_logprob_parity(
                reference=reference, actual=actual, label=f"after_cancel_{i}"
            )
        self._wait_for_decode_idle(idle["timestamp"])


class TestUnifiedMemoryDisaggregationMHA(TestUnifiedMemoryDisaggregation):
    """The MHA full-attention sub-pool over the wire.

    Kimi-Linear above exercises the MLA sub-pool, whose whole-envelope
    registration has always been the one PD supports. An MHA envelope is a
    different shape -- `2 * layer_num` row-blocks per page instead of
    `layer_num` -- and it reaches a different branch of
    `_send_kvcache_generic`: without `force_flat` the MHA branch halves the
    single registered region into K and V, computes `num_kv_layers = 0` and
    transfers NOTHING, which shows up as garbage decode rather than an error.
    Logprob parity against a non-PD unified reference is what catches that.
    """

    model = QWEN_GDN_MODEL


if __name__ == "__main__":
    unittest.main()
