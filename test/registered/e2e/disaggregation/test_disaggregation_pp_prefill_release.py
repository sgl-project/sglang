"""PD disaggregation with a PP2 prefill server: PP0 releases a transfer only once
every stage has concluded it.

Each prefill PP stage sends the KV of its own layers, so its transfer of a request
can conclude at a different step than the other stage's. An abort also reaches
PP0 before PP1, so for a while the request is Failed on one stage while the other
stage is still transferring. PP0 decides the release for both stages; releasing
before both have concluded would let a stage free KV pages and the metadata slot
its transfer still reads.

The stress mixes completed requests with requests aborted at spread-out delays,
so aborts land during prefill, during the KV transfer and after it. Rounds repeat
until the decode has logged at least one request aborted while it was receiving
its KV. Afterwards both servers must be up and idle with no leaked pages (the
strict idle memory check is on by default), and fixed prompts must give the same
greedy output as before the stress.
"""

import re
import threading
import time
import unittest
import uuid

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_cuda_ci(est_time=240, stage="extra-b", runner_config="4-gpu-h100")

_MIN_ROUNDS = 3
# Measured: 8 of 48 aborts landed mid-transfer in 3 rounds; 8 rounds leave margin.
_MAX_ROUNDS = 8
_REQUESTS_PER_ROUND = 32
# Delays for the aborted half: from mid-prefill to after the KV has landed.
_ABORT_DELAYS_S = (0.2, 0.35, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0)
_MAX_NEW_TOKENS = 32
# About 3k tokens: several prefill chunks, so each stage sends several KV chunks.
_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "Sphinx of black quartz, judge my vow. "
) * 100
_REFERENCE_PROMPTS = [
    f"Reference {i}: summarize the following text in one sentence.\n{_FILLER}"
    for i in range(4)
]


def _generate(url, rid, text, timeout=180):
    return requests.post(
        url + "/generate",
        json={
            "rid": rid,
            "text": text,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": _MAX_NEW_TOKENS,
                "ignore_eos": True,
            },
        },
        timeout=timeout,
    )


# decode.py logs this when a request in the decode transfer queue concludes
# Failed; an abort there records "Aborted by AbortReq." on its receiver.
_TRANSFER_ABORT = re.compile(
    r"Decode transfer failed for request .*rid='([^']+)'.*Aborted by AbortReq"
)


def _finish_type(response):
    if response.status_code != 200:
        return None
    reason = response.json().get("meta_info", {}).get("finish_reason")
    return reason.get("type") if isinstance(reason, dict) else None


class TestDisaggregationPPPrefillRelease(PDDisaggregationServerBase):
    capture_per_side_logs = True
    decode_tp_size = 1
    decode_base_gpu_id = 2

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)
        cls.extra_decode_args = ["--max-running-requests", "64"]
        cls.launch_all()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--nccl-port",
            cls.prefill_nccl_port,
            "--tp-size",
            "1",
            "--pp-size",
            "2",
            "--chunked-prefill-size",
            "1024",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices_for(range(2))
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    def _reference_outputs(self):
        # Start from an empty prefill cache, so both passes compute the same way.
        response = requests.post(self.prefill_url + "/flush_cache", timeout=60)
        self.assertEqual(response.status_code, 200, response.text)
        outputs = []
        for i, prompt in enumerate(_REFERENCE_PROMPTS):
            response = _generate(
                self.lb_url, f"pp-release-ref-{i}-{uuid.uuid4().hex}", prompt
            )
            self.assertEqual(response.status_code, 200, response.text)
            outputs.append(response.json()["text"])
        return outputs

    def _abort(self, rid):
        for url in (self.prefill_url, self.decode_url):
            requests.post(
                url + "/abort_request",
                json={"rid": rid, "abort_all": False},
                timeout=10,
            )

    def _stress_round(self, round_idx):
        results = {}

        def run(rid, text):
            try:
                results[rid] = _generate(self.lb_url, rid, text)
            except requests.RequestException as exc:
                results[rid] = exc

        threads, aborted = [], []
        for i in range(_REQUESTS_PER_ROUND):
            rid = f"pp-release-{round_idx}-{i}-{uuid.uuid4().hex}"
            # A distinct first line keeps the prompts from sharing a cached prefix.
            thread = threading.Thread(target=run, args=(rid, f"{rid}\n{_FILLER}"))
            thread.start()
            threads.append(thread)
            if i % 2:
                delay = _ABORT_DELAYS_S[(i // 2) % len(_ABORT_DELAYS_S)]
                timer = threading.Timer(delay, self._abort, args=(rid,))
                timer.start()
                threads.append(timer)
                aborted.append(rid)
        for thread in threads:
            thread.join(timeout=300)
            self.assertFalse(thread.is_alive(), "a request or abort hung")
        return results, set(aborted)

    def _aborted_in_transfer(self):
        log = self._decode_stdout_buf.getvalue() + self._decode_stderr_buf.getvalue()
        return set(_TRANSFER_ABORT.findall(log))

    def test_release_waits_for_every_stage_under_aborts(self):
        before = self._reference_outputs()

        in_transfer = set()
        for round_idx in range(_MAX_ROUNDS):
            results, aborted = self._stress_round(round_idx)
            for rid, response in results.items():
                # An aborted request may end in any way, even a dropped connection.
                if rid in aborted:
                    continue
                self.assertNotIsInstance(response, Exception, f"{rid}: {response!r}")
                self.assertEqual(response.status_code, 200, f"{rid}: {response.text}")
                self.assertEqual(_finish_type(response), "length", response.text)
            in_transfer |= aborted & self._aborted_in_transfer()
            if round_idx + 1 >= _MIN_ROUNDS and in_transfer:
                break
        print(f"{len(in_transfer)} requests aborted while decode received their KV")
        self.assertTrue(in_transfer, "no abort landed while decode received the KV")

        # Idle long enough for both servers' idle memory check to run.
        time.sleep(10)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)
        self.assertEqual(self._reference_outputs(), before)


if __name__ == "__main__":
    unittest.main()
