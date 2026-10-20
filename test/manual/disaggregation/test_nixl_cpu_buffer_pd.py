"""Small, download-free PD integration test with deterministic tiny Llama weights.

Requires two CUDA devices and NIXL/UCX. Set NIXL_TEST_TP_SIZE=2 to use four devices. This checks greedy token parity with a
unified server, including chunked prefill and repeated prefixes. It is not a
model-quality or throughput benchmark.
"""

import os
import socket
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_pd_server,
    popen_launch_server,
    terminate_and_kill_process_tree,
)
from sglang.utils import wait_for_http_ready


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def generate(url, body):
    response = requests.post(url + "/generate", json=body, timeout=120)
    response.raise_for_status()
    return response.json()["output_ids"]


class TestNixlCpuBufferPD(CustomTestCase):
    def test_greedy_parity_with_chunked_prefill(self):
        self._run_test()

    def test_pool_exhaustion_fails_both_peers_and_recovers(self):
        self._run_test(exhaustion=True)

    def test_source_pool_exhaustion_recovers(self):
        self._run_test(exhaustion="source")

    def test_pool_pressure_preserves_inflight_request(self):
        self._run_test(exhaustion="pressure")

    def _run_test(self, exhaustion=False):
        tp_size = int(os.environ.get("NIXL_TEST_TP_SIZE", "1"))
        self.assertIn(tp_size, (1, 2))
        with tempfile.TemporaryDirectory(prefix="nixl-cpu-pd-") as model:
            config = LlamaConfig(
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                vocab_size=128,
                max_position_embeddings=512,
                rms_norm_eps=1e-6,
                hidden_act="silu",
                bos_token_id=1,
                eos_token_id=2,
                tie_word_embeddings=False,
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                LlamaForCausalLM(config).save_pretrained(model)
            common = [
                "--tp",
                str(tp_size),
                "--load-format",
                "safetensors",
                "--skip-tokenizer-init",
                "--random-seed",
                "42",
                "--dtype",
                "float16",
                "--attention-backend",
                "triton",
                "--cuda-graph-backend-decode",
                "disabled",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--disable-overlap-schedule",
                "--mem-fraction-static",
                "0.04",
                "--max-total-tokens",
                "2048",
                "--max-running-requests",
                "4",
                "--context-length",
                "512",
                "--chunked-prefill-size",
                "32",
                "--page-size",
                "1",
                "--disable-custom-all-reduce",
            ]
            if exhaustion:
                common.append("--disable-radix-cache")
            prompts = [[1] + [3 + i % 100 for i in range(n)] for n in (15, 95, 191, 95)]
            bodies = [
                {
                    "input_ids": ids,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                        "ignore_eos": True,
                    },
                }
                for ids in prompts
            ]
            baseline_url = f"http://127.0.0.1:{free_port()}"
            baseline = popen_launch_server(
                model, baseline_url, timeout=180, other_args=common, device="cuda"
            )
            try:
                expected = [generate(baseline_url, body) for body in bodies]
            finally:
                terminate_and_kill_process_tree(baseline)
            prefill_url = f"http://127.0.0.1:{free_port()}"
            decode_url = f"http://127.0.0.1:{free_port()}"
            bootstrap = free_port()
            env = {
                "UCX_TLS": "tcp,cuda_copy,self",
                "UCX_ACC_DEVICES": "",
                "SGLANG_DISAGGREGATION_NIXL_BACKEND": "UCX",
                "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS": '{"num_threads":"1"}',
            }
            pd_args = common + [
                "--disaggregation-transfer-backend",
                "nixl",
                "--nixl-use-cpu-buffer",
                "--disaggregation-enable-kv-checksum",
                "--nixl-cpu-buffer-size-gb",
                (
                    str(65536 / tp_size / 1024**3)
                    if exhaustion in (True, "pressure")
                    else "0.01"
                ),
                "--disaggregation-bootstrap-port",
                str(bootstrap),
            ]
            processes = []
            try:
                for mode, url, gpu in (
                    ("prefill", prefill_url, "0"),
                    ("decode", decode_url, str(tp_size)),
                ):
                    processes.append(
                        popen_launch_pd_server(
                            model,
                            url,
                            timeout=180,
                            env=env,
                            other_args=pd_args
                            + ["--disaggregation-mode", mode, "--base-gpu-id", gpu]
                            + (
                                [
                                    "--nixl-cpu-buffer-size-gb",
                                    str(8192 / tp_size / 1024**3),
                                ]
                                if exhaustion == "source" and mode == "prefill"
                                else []
                            ),
                        )
                    )
                for url, process in zip((prefill_url, decode_url), processes):
                    wait_for_http_ready(
                        url=url + "/health", timeout=180, process=process
                    )
                with ThreadPoolExecutor(max_workers=2) as executor:
                    if exhaustion:

                        def expect_abort(url, body):
                            response = requests.post(
                                url + "/generate", json=body, timeout=15
                            )
                            payload = response.json()
                            self.assertTrue(
                                response.status_code >= 400
                                or payload.get("meta_info", {})
                                .get("finish_reason", {})
                                .get("type")
                                == "abort",
                                payload,
                            )
                            return payload

                        if exhaustion == "pressure":
                            for cycle in range(4):
                                held = {
                                    **bodies[1],
                                    "bootstrap_host": "127.0.0.1",
                                    "bootstrap_port": bootstrap,
                                    "bootstrap_room": 3000 + cycle * 2,
                                }
                                live = executor.submit(generate, decode_url, held)
                                deadline = time.monotonic() + 10
                                while True:
                                    response = requests.get(
                                        decode_url + "/v1/loads?include=disagg",
                                        timeout=5,
                                    )
                                    response.raise_for_status()
                                    load = response.json()["loads"][0]["disaggregation"]
                                    if load["decode_transfer_queue_reqs"] == 1:
                                        break
                                    self.assertLess(time.monotonic(), deadline)
                                    time.sleep(0.02)
                                # Each request fits by itself; the live receive
                                # region makes the second allocation fail.
                                rejected = {
                                    **held,
                                    "bootstrap_room": held["bootstrap_room"] + 1,
                                }
                                payload = expect_abort(decode_url, rejected)
                                self.assertIn("pool exhausted", str(payload))
                                expect_abort(prefill_url, rejected)
                                self.assertFalse(live.done())
                                generate(prefill_url, held)
                                self.assertEqual(live.result(timeout=15), expected[1])
                            print(
                                "Pool pressure passed: 4 rejected pairs, "
                                "4 intact in-flight requests, and capacity reuse"
                            )
                            return

                        # Oversized receive regions fail deterministically,
                        # regardless of transport speed or concurrent timing.
                        # Each failure is followed by a successful request;
                        # more cycles than max_running_requests catch leaked
                        # request, auxiliary, KV and host-pool capacity.
                        for cycle in range(12):
                            failed = {
                                **bodies[2],
                                "bootstrap_host": "127.0.0.1",
                                "bootstrap_port": bootstrap,
                                "bootstrap_room": 2000 + cycle * 2,
                            }

                            if cycle % 2 == 0 and exhaustion is True:
                                # Decode fails before prefill even sees HTTP.
                                payload = expect_abort(decode_url, failed)
                                self.assertIn("pool exhausted", str(payload))
                                expect_abort(prefill_url, failed)
                            else:
                                pending = executor.submit(
                                    expect_abort, prefill_url, failed
                                )
                                payload = expect_abort(decode_url, failed)
                                prefill_payload = pending.result(timeout=15)
                                # TP consensus may report a peer-rank failure
                                # before rank 0 receives its detailed status.
                                self.assertIn(
                                    "pool exhausted", str((payload, prefill_payload))
                                )
                            healthy = {
                                **failed,
                                **bodies[0],
                                "bootstrap_room": failed["bootstrap_room"] + 1,
                            }
                            pending = executor.submit(generate, prefill_url, healthy)
                            actual = generate(decode_url, healthy)
                            pending.result(timeout=15)
                            self.assertEqual(actual, expected[0])
                        print(
                            f"Pool exhaustion recovery passed ({exhaustion=}): "
                            "12 failed pairs and 12 healthy successors"
                        )
                        return
                    for i, body in enumerate(bodies):
                        request = {
                            **body,
                            "bootstrap_host": "127.0.0.1",
                            "bootstrap_port": bootstrap,
                            "bootstrap_room": 1000 + i,
                        }
                        prefill = executor.submit(generate, prefill_url, request)
                        actual = generate(decode_url, request)
                        prefill.result(timeout=120)
                        self.assertEqual(actual, expected[i], f"prompt {i}")
                        self.assertEqual(len(actual), 16)
                print(
                    "PD greedy parity passed: four requests, multi-chunk and repeated prefix"
                )
            finally:
                for process in reversed(processes):
                    terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
