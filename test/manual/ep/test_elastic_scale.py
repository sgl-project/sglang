"""Manual tests for elastic EP scale-up.

Test classes:
  TestElasticScaleUp4To6                primary + joiner scale-up (6 GPUs)
  TestElasticScaleUp4To5To6             two consecutive single-rank scale-ups
  TestElasticScaleUp4To8                full primary + joiner scale-up (8 GPUs)

Run (8-GPU full scale-up):

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m pytest \\
        test/manual/ep/test_elastic_scale.py::TestElasticScaleUp4To8 \\
        -v -s
"""

import os
import subprocess
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import get_rdma_devices_args
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

TEST_MODEL = os.environ.get("NIXL_EP_TEST_MODEL", DEFAULT_MODEL_NAME_FOR_TEST_MLA)
os.environ.setdefault("SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")

ib_devices = get_rdma_devices_args()


def _extra_server_args() -> list[str]:
    """Extra `--flag [value]` tokens appended to every spawned server.

    Set via ``SGLANG_ELASTIC_EXTRA_SERVER_ARGS`` as a single space-separated
    string, e.g. ``--disable-overlap-schedule``.
    """
    raw = os.environ.get("SGLANG_ELASTIC_EXTRA_SERVER_ARGS", "").strip()
    return raw.split() if raw else []


DISABLED_CUDA_GRAPH_ARGS = [
    "--cuda-graph-backend-decode",
    "disabled",
    "--cuda-graph-backend-prefill",
    "disabled",
]


def _assert_generate_logprob_ok(testcase: unittest.TestCase, base_url: str) -> None:
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": "The answer is",
            "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
            "return_logprob": True,
            "top_logprobs_num": 1,
            "logprob_start_len": 0,
        },
        timeout=60,
    )
    testcase.assertEqual(response.status_code, 200, response.text)
    input_logprobs = response.json()["meta_info"]["input_token_logprobs"]
    testcase.assertGreater(len(input_logprobs), 0)


def _count_visible_gpus() -> int:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return len([x for x in env.split(",") if x.strip()])
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _visible_device_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [device.strip() for device in visible.split(",") if device.strip()]
    return [str(index) for index in range(_count_visible_gpus())]


LAUNCH_EP_SIZE = 4
MAX_EP_SIZE = 8

DIST_INIT_ADDR = os.environ.get("SGLANG_ELASTIC_SCALE_DIST_INIT", "127.0.0.1:24555")
PORT_A = int(os.environ.get("SGLANG_ELASTIC_SCALE_PORT_A", "21000"))
PORT_B = int(os.environ.get("SGLANG_ELASTIC_SCALE_PORT_B", "10000"))
PORT_C = int(os.environ.get("SGLANG_ELASTIC_SCALE_PORT_C", "11000"))
HOST_A = os.environ.get("SGLANG_ELASTIC_SCALE_HOST_A", "127.0.0.1")
BASE_URL_A = f"http://{HOST_A}:{PORT_A}"
PRE_SCALE_JOINER_DELAY_SEC = float(
    os.environ.get("SGLANG_ELASTIC_PRE_SCALE_JOINER_DELAY_SEC", "0")
)


def _scale_up_common_args(
    dist_init_addr: str,
    tp_size: int,
    nnodes: int,
    node_rank: int,
    cuda_graph_args: list[str],
    moe_dense_tp_size: int | None,
) -> list[str]:
    args = [
        "--trust-remote-code",
        "--moe-a2a-backend",
        "nixl",
        "--deepep-mode",
        "low_latency",
        "--tp",
        str(tp_size),
        "--dp",
        str(tp_size),
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--elastic-ep-backend",
        "mooncake",
        "--mooncake-ib-device",
        ib_devices,
        "--enable-eplb",
        "--ep-num-redundant-experts",
        "24",
        "--elastic-ep-initial-size",
        str(LAUNCH_EP_SIZE),
        "--max-ep-size",
        str(MAX_EP_SIZE),
        "--mem-fraction-static",
        "0.5",
        "--chunked-prefill-size",
        "1024",
        "--nnodes",
        str(nnodes),
        "--node-rank",
        str(node_rank),
        "--dist-init-addr",
        dist_init_addr,
    ]
    if moe_dense_tp_size is not None:
        args.extend(["--moe-dense-tp-size", str(moe_dense_tp_size)])
    return args + cuda_graph_args + _extra_server_args()


class _ElasticScaleUpEndToEndBase(CustomTestCase):
    """Shared scale-up E2E plumbing. Subclasses set JOIN_TP/JOIN_NNODES/JOIN_NODE_RANK."""

    JOIN_TP: int
    JOIN_NNODES: int
    JOIN_NODE_RANK: int
    TARGET_EP_SIZE: int
    CUDA_GRAPH_ARGS: list[str]
    MOE_DENSE_TP_SIZE: int | None = 1

    def setUp(self):
        if (
            not hasattr(type(self), "JOIN_TP")
            or type(self) is _ElasticScaleUpEndToEndBase
        ):
            self.skipTest("Abstract base — run a concrete subclass instead")

    @classmethod
    def setUpClass(cls):
        if cls is _ElasticScaleUpEndToEndBase:
            raise unittest.SkipTest("Abstract base")
        cls.model = TEST_MODEL
        cls.base_url = BASE_URL_A
        cls._joining_procs = []
        cls._joining_log_fhs = []

        primary_args = _scale_up_common_args(
            DIST_INIT_ADDR,
            tp_size=LAUNCH_EP_SIZE,
            nnodes=1,
            node_rank=0,
            cuda_graph_args=cls.CUDA_GRAPH_ARGS,
            moe_dense_tp_size=cls.MOE_DENSE_TP_SIZE,
        )
        primary_env = os.environ.copy()
        visible_devices = _visible_device_ids()
        if len(visible_devices) < LAUNCH_EP_SIZE:
            raise RuntimeError(
                f"Scale-up requires {LAUNCH_EP_SIZE} visible GPUs, got "
                f"{len(visible_devices)}"
            )
        primary_env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices[:LAUNCH_EP_SIZE])
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=primary_args,
            env=primary_env,
        )

    @classmethod
    def _launch_joining_group(
        cls,
        *,
        rank_offset: int,
        join_tp: int,
        port: int,
    ) -> subprocess.Popen:
        cmd = [
            "sglang",
            "serve",
            "--model-path",
            cls.model,
            *_scale_up_common_args(
                DIST_INIT_ADDR,
                tp_size=join_tp,
                nnodes=cls.JOIN_NNODES,
                node_rank=cls.JOIN_NODE_RANK,
                cuda_graph_args=cls.CUDA_GRAPH_ARGS,
                moe_dense_tp_size=cls.MOE_DENSE_TP_SIZE,
            ),
            "--elastic-ep-join-mode",
            "scale",
            "--elastic-ep-join-rank-offset",
            str(rank_offset),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--device",
            "cuda",
        ]
        env = os.environ.copy()
        visible_devices = _visible_device_ids()
        join_end = rank_offset + join_tp
        if join_end > len(visible_devices):
            raise RuntimeError(
                f"Scale-up requires {join_end} visible GPUs, got "
                f"{len(visible_devices)}"
            )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices[rank_offset:join_end])
        base_joining_log = os.environ.get(
            "SGLANG_ELASTIC_SCALE_JOINING_LOG",
            f"/tmp/elastic_scale_joining_nnodes{cls.JOIN_NNODES}_{int(time.time())}.log",
        )
        if cls._joining_procs:
            root, ext = os.path.splitext(base_joining_log)
            joining_log = f"{root}_step{len(cls._joining_procs) + 1}{ext}"
        else:
            joining_log = base_joining_log
        joining_log_fh = open(joining_log, "w")
        joining_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=joining_log_fh,
            stderr=subprocess.STDOUT,
        )
        cls._joining_procs.append(joining_proc)
        cls._joining_log_fhs.append(joining_log_fh)
        return joining_proc

    @classmethod
    def tearDownClass(cls):
        processes = [
            *reversed(cls._joining_procs),
            getattr(cls, "process", None),
        ]
        for proc in processes:
            if proc is None:
                continue
            try:
                kill_process_tree(proc.pid)
            except Exception:
                pass
        for proc in processes:
            if proc is None:
                continue
            try:
                proc.wait(timeout=15)
            except Exception:
                pass
        for fh in cls._joining_log_fhs:
            try:
                fh.close()
            except Exception:
                pass
        time.sleep(2)

    def _post(self, path: str, **kwargs) -> requests.Response:
        return requests.post(f"{self.base_url}{path}", timeout=60, **kwargs)

    def _generate_ok(self, msg_suffix: str, routed_dp_rank: int | None = None) -> None:
        payload = {
            "text": "Hello",
            "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
        }
        if routed_dp_rank is not None:
            payload["routed_dp_rank"] = routed_dp_rank
        resp = self._post(
            "/generate",
            json=payload,
        )
        self.assertEqual(
            resp.status_code,
            200,
            f"/generate {msg_suffix} failed: {resp.text}",
        )

    def _generate_logprob_ok(self, msg_suffix: str) -> None:
        try:
            _assert_generate_logprob_ok(self, self.base_url)
        except AssertionError as exc:
            raise AssertionError(
                f"/generate logprob {msg_suffix} failed: {exc}"
            ) from exc

    def _scale_once(
        self,
        *,
        old_ep_size: int,
        target_ep_size: int,
        join_tp: int,
        port: int,
    ) -> None:
        joining_proc = self._launch_joining_group(
            rank_offset=old_ep_size,
            join_tp=join_tp,
            port=port,
        )
        self.assertIsNone(
            joining_proc.poll(),
            "Joining group exited before scale request; see joining log",
        )
        if PRE_SCALE_JOINER_DELAY_SEC > 0:
            time.sleep(PRE_SCALE_JOINER_DELAY_SEC)
            self.assertIsNone(
                joining_proc.poll(),
                "Joining group exited before scale request; see joining log",
            )

        resp = self._post("/scale_elastic_ep", json={"new_ep_size": target_ep_size})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["old_ep_size"], old_ep_size)
        self.assertEqual(body["new_ep_size"], target_ep_size)

        deadline = time.time() + 300
        while time.time() < deadline:
            resp = requests.get(f"{self.base_url}/is_scaling_elastic_ep", timeout=60)
            state = resp.json() if resp.ok else None
            if state is not None and not state.get("is_scaling_elastic_ep", True):
                self.assertEqual(state.get("effective_ep_size"), target_ep_size)
                self.assertEqual(state.get("scale_phase"), "serving_expanded")
                self.assertIsNone(state.get("last_error"))
                self._generate_ok(
                    "on newest joiner",
                    routed_dp_rank=target_ep_size - 1,
                )
                return
            try:
                self._post(
                    "/generate",
                    json={
                        "text": "ping",
                        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
                    },
                )
            except Exception:
                pass
            time.sleep(2)
        self.fail("Timed out waiting for scaling to complete (300s)")

    def _run_post_scale_gsm8k(self) -> None:
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=256,
                num_threads=50,
            )
        )
        print(f"[TEST] Post-scale GSM8K accuracy: {metrics['score']:.2%}")
        self.assertGreater(
            metrics["score"],
            0.50,
            f"Post-scale GSM8K accuracy too low: {metrics['score']:.2%}",
        )

    def test_scale_up_on_demand(self):
        """Scale the primary group to the configured target."""
        self._generate_ok("pre-scale")

        self._scale_once(
            old_ep_size=LAUNCH_EP_SIZE,
            target_ep_size=self.TARGET_EP_SIZE,
            join_tp=self.JOIN_TP,
            port=PORT_B,
        )

        self._generate_ok("post-scale")
        self._generate_logprob_ok("post-scale")

        self._run_post_scale_gsm8k()
        self._generate_logprob_ok("after post-scale workload")


@unittest.skipUnless(
    _count_visible_gpus() >= 6,
    "4-to-6 scale-up E2E needs 6 GPUs.",
)
class TestElasticScaleUp4To6(_ElasticScaleUpEndToEndBase):
    """Scale from four to six ranks."""

    JOIN_TP = 2
    JOIN_NNODES = 2
    JOIN_NODE_RANK = 1
    TARGET_EP_SIZE = 6
    CUDA_GRAPH_ARGS = DISABLED_CUDA_GRAPH_ARGS
    MOE_DENSE_TP_SIZE = None


@unittest.skipUnless(
    _count_visible_gpus() >= 6,
    "4-to-5-to-6 scale-up E2E needs 6 GPUs.",
)
class TestElasticScaleUp4To5To6(_ElasticScaleUpEndToEndBase):
    """Scale from four to five and then from five to six ranks."""

    JOIN_TP = 1
    JOIN_NNODES = 2
    JOIN_NODE_RANK = 1
    TARGET_EP_SIZE = 6
    CUDA_GRAPH_ARGS = DISABLED_CUDA_GRAPH_ARGS

    def test_scale_up_on_demand(self):
        self._generate_ok("pre-scale")

        self._scale_once(
            old_ep_size=4,
            target_ep_size=5,
            join_tp=1,
            port=PORT_B,
        )
        self._generate_ok("after first scale")
        self._generate_logprob_ok("after first scale")

        self._scale_once(
            old_ep_size=5,
            target_ep_size=6,
            join_tp=1,
            port=PORT_C,
        )
        self._generate_ok("after second scale")
        self._generate_logprob_ok("after second scale")
        self._run_post_scale_gsm8k()
        self._generate_logprob_ok("after post-scale workload")


@unittest.skipUnless(
    _count_visible_gpus() >= MAX_EP_SIZE,
    f"Full scale-up E2E needs {MAX_EP_SIZE} GPUs.",
)
class TestElasticScaleUp4To8(_ElasticScaleUpEndToEndBase):
    """Scale from four to eight ranks."""

    JOIN_TP = LAUNCH_EP_SIZE
    JOIN_NNODES = 2
    JOIN_NODE_RANK = 1
    TARGET_EP_SIZE = MAX_EP_SIZE
    CUDA_GRAPH_ARGS = DISABLED_CUDA_GRAPH_ARGS
    MOE_DENSE_TP_SIZE = None


if __name__ == "__main__":
    unittest.main()
