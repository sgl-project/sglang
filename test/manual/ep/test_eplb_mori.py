import os
import unittest
from types import SimpleNamespace

from sglang.srt.server_args import ZMQ_TCP_PORT_DELTA
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import is_port_available
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def wait_all_ports_release(base_url, timeout_s=60):
    import time

    port = int(base_url.split(":")[-1])
    offsets = [
        0,
        ZMQ_TCP_PORT_DELTA,
        ZMQ_TCP_PORT_DELTA + 1,
        ZMQ_TCP_PORT_DELTA + 2,
        ZMQ_TCP_PORT_DELTA + 3,
        ZMQ_TCP_PORT_DELTA + 4,
    ]
    for _ in range(timeout_s):
        if all(is_port_available(port + off) for off in offsets):
            return
        time.sleep(1)
    print(f"Warning: some ports still occupied after {timeout_s}s")


mori_env = {
    **os.environ,
    "SGLANG_USE_AITER": "1",
    "SGLANG_MORI_DISPATCH_DTYPE": "bf16",
    "SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "4096",
    "SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE": "32",
    "MORI_SHMEM_MODE": "ISOLATION",
}

common_args = [
    "--tp-size",
    "8",
    "--ep-size",
    "8",
    "--dp-size",
    "8",
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "mori",
    "--trust-remote-code",
    "--load-balance-method",
    "round_robin",
    "--moe-dense-tp-size",
    "1",
    "--enable-dp-lm-head",
    "--mem-fraction-static",
    "0.6",
    "--chunked-prefill-size",
    "32768",
    "--max-running-requests",
    "128",
    "--context-length",
    "12288",
    "--attention-backend",
    "aiter",
    "--cuda-graph-max-bs",
    "32",
]

eplb_args = [
    "--enable-eplb",
    "--ep-num-redundant-experts",
    "32",
    "--eplb-rebalance-num-iterations",
    "50",
    "--expert-distribution-recorder-buffer-size",
    "50",
    "--ep-dispatch-algorithm",
    "static",
]


mtp_args = [
    "--speculative-algo",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]


class TestEPLBMoriStat(CustomTestCase):
    """EPLB with mori backend, stat mode (on_select_experts path)."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = (
            common_args
            + eplb_args
            + [
                "--deepep-mode",
                "normal",
                "--expert-distribution-recorder-mode",
                "stat",
            ]
        )

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
            env=mori_env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        wait_all_ports_release(cls.base_url)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1209,
            max_new_tokens=512,
            parallel=1209,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["accuracy"], 0.9)


class TestEPLBMoriStatApprox(CustomTestCase):
    """EPLB with mori backend, stat_approx mode (local_expert_count kernel)."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = (
            common_args
            + eplb_args
            + [
                "--deepep-mode",
                "normal",
                "--expert-distribution-recorder-mode",
                "stat_approx",
            ]
        )

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
            env=mori_env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        wait_all_ports_release(cls.base_url)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1209,
            max_new_tokens=512,
            parallel=1209,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["accuracy"], 0.9)


class TestEPLBMoriMultiChunk(CustomTestCase):
    """EPLB with mori backend, chunked layer updates."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = (
            common_args
            + eplb_args
            + [
                "--deepep-mode",
                "normal",
                "--expert-distribution-recorder-mode",
                "stat",
                "--eplb-rebalance-layers-per-chunk",
                "1",
            ]
        )

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
            env=mori_env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        wait_all_ports_release(cls.base_url)

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1209,
            max_new_tokens=512,
            parallel=1209,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()
