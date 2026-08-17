"""Minimal end-to-end example for Mooncake-native scale-down.

Demonstrates the two-step ``shrink then regrow`` flow that most callers
will use in production:

    1. Launch primary at ``ep_size=4`` with ``--max-ep-size 5``. The +1
       headroom keeps a recoverable slot pool for the regrow step.
    2. POST ``/scale_elastic_ep`` with ``new_ep_size=3``. Server enters
       the DRAIN -> RETIRE FSM; the retired rank calls ``os._exit(0)``.
    3. Launch a joiner subprocess with ``--elastic-ep-join-mode recover``
       and ``--elastic-ep-join-rank-offset 3`` (the freshly-retired slot).
    4. POST ``/scale_elastic_ep`` with ``new_ep_size=4``. Survivor calls
       ``recover_ranks`` which pairs with the joiner's ``join_process_groups``
       over the DPC socket that ``remove_elastic_workers`` kept bound.
    5. Confirm ``/generate`` still serves after both scale steps.

This file intentionally has no fan-out into other MC0N chained-shrink /
multi-node / a2a-variant coverage. Those live in an out-of-tree harness.

Requires >= 4 GPUs and ``--elastic-ep-backend mooncake``. Run with::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest \\
        test/manual/ep/test_elastic_shrink.py -v -s
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.server_fixtures.disaggregation_fixture import get_rdma_devices_args
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

TEST_MODEL = os.environ.get("SGLANG_MC_TEST_MODEL", DEFAULT_MODEL_NAME_FOR_TEST_MLA)

# Mooncake EP buffer bounds per-rank in-flight tokens; 1024 is the max.
os.environ.setdefault("SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")
os.environ.setdefault("SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")

LAUNCH_EP_SIZE = 4
MAX_EP_SIZE = 5  # +1 headroom keeps a recoverable slot pool for regrow.
HOST = "127.0.0.1"
PORT_PRIMARY = int(os.environ.get("SGLANG_MC_PORT_A", "21100"))
PORT_JOINER = int(os.environ.get("SGLANG_MC_PORT_B", "10100"))
DIST_INIT_ADDR = os.environ.get("SGLANG_MC_DIST_INIT", "127.0.0.1:24655")
BASE_URL = f"http://{HOST}:{PORT_PRIMARY}"

# 24 fills EPLB layout at ep=4 (72 logical experts + 24 redundant = 96,
# divisible by 4) and keeps ep=3 feasible (72/3 = 24 experts per rank).
EP_NUM_REDUNDANT_EXPERTS = 24


def _visible_devices() -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if env:
        return [d.strip() for d in env.split(",") if d.strip()]
    try:
        import torch  # noqa: E402
        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:
        return []


def _common_args(tp: int) -> list[str]:
    """Server flags shared by primary and joiner."""
    return [
        "--trust-remote-code",
        "--moe-a2a-backend", "nixl",
        "--deepep-mode", "low_latency",
        "--tp", str(tp),
        "--dp", str(tp),
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--elastic-ep-backend", "mooncake",
        "--mooncake-ib-device", get_rdma_devices_args(),
        "--enable-eplb",
        "--ep-num-redundant-experts", str(EP_NUM_REDUNDANT_EXPERTS),
        "--max-ep-size", str(MAX_EP_SIZE),
        "--mem-fraction-static", "0.5",
        "--chunked-prefill-size", "1024",
        "--nnodes", "1",
        "--node-rank", "0",
        "--dist-init-addr", DIST_INIT_ADDR,
        "--moe-dense-tp-size", "1",
        "--cuda-graph-backend-decode", "disabled",
        "--cuda-graph-backend-prefill", "disabled",
    ]


@unittest.skipUnless(
    len(_visible_devices()) >= LAUNCH_EP_SIZE,
    f"Needs >= {LAUNCH_EP_SIZE} GPUs.",
)
class TestElasticShrinkThenRegrow(CustomTestCase):
    """End-to-end 4 -> 3 -> 4 exercise for Mooncake-native scale-down."""

    @classmethod
    def setUpClass(cls):
        cls.model = TEST_MODEL
        cls.base_url = BASE_URL
        devices = _visible_devices()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[:LAUNCH_EP_SIZE])
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_common_args(tp=LAUNCH_EP_SIZE),
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        proc = getattr(cls, "process", None)
        if proc is not None:
            try:
                kill_process_tree(proc.pid)
                proc.wait(timeout=15)
            except Exception:
                pass
        time.sleep(2)

    def _post(self, path: str, **kw) -> requests.Response:
        return requests.post(f"{self.base_url}{path}", timeout=60, **kw)

    def _generate_ok(self, msg: str, *, routed_dp_rank: int | None = None) -> None:
        payload = {
            "text": "Hello",
            "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
        }
        if routed_dp_rank is not None:
            payload["routed_dp_rank"] = routed_dp_rank
        resp = self._post("/generate", json=payload)
        self.assertEqual(resp.status_code, 200, f"/generate {msg}: {resp.text}")

    def _poll_until_serving(
        self, *, expected_ep_size: int, expected_phase: str, timeout_s: float = 600.0
    ) -> None:
        # 600s matches primary-side elastic_ep_scale_timeout so slow joiner
        # cold-starts on scale-up-v1 do not race the harness.
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            resp = requests.get(
                f"{self.base_url}/is_scaling_elastic_ep", timeout=60
            )
            state = resp.json() if resp.ok else None
            if state and not state.get("is_scaling_elastic_ep", True):
                self.assertEqual(state.get("effective_ep_size"), expected_ep_size)
                self.assertEqual(state.get("scale_phase"), expected_phase)
                self.assertIsNone(state.get("last_error"))
                return
            # Keep busy path warm so the retire handler ticks.
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
        self.fail(f"Timed out waiting for scale to reach {expected_phase}")

    def _scale_to(self, *, old_ep_size: int, target_ep_size: int) -> None:
        resp = self._post("/scale_elastic_ep", json={"new_ep_size": target_ep_size})
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["old_ep_size"], old_ep_size, body)
        self.assertEqual(body["new_ep_size"], target_ep_size, body)
        phase = "serving_shrunk" if target_ep_size < old_ep_size else "serving_expanded"
        self._poll_until_serving(
            expected_ep_size=target_ep_size, expected_phase=phase
        )

    @classmethod
    def _launch_recover_joiner(
        cls, *, rank_offset: int, join_tp: int, port: int
    ) -> subprocess.Popen:
        """Launch a joiner into a previously-retired slot.

        The joiner runs as ``--nnodes 2 --node-rank 1`` of a logical
        (primary=0, joiner=1) view (what DPC expects for offset joiners).
        """
        args = _common_args(tp=join_tp)
        for i, tok in enumerate(args):
            if tok == "--nnodes":
                args[i + 1] = "2"
            elif tok == "--node-rank":
                args[i + 1] = "1"
        cmd = [
            "sglang", "serve",
            "--model-path", cls.model,
            *args,
            "--elastic-ep-initial-size", str(LAUNCH_EP_SIZE),
            "--elastic-ep-join-mode", "recover",
            "--elastic-ep-join-rank-offset", str(rank_offset),
            "--host", HOST,
            "--port", str(port),
            "--device", "cuda",
        ]
        devices = _visible_devices()
        end = rank_offset + join_tp
        if end > len(devices):
            raise RuntimeError(f"Recover joiner needs {end} GPUs, got {len(devices)}")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[rank_offset:end])
        env.setdefault("PYTHONUNBUFFERED", "1")
        return subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)

    def test_shrink_then_regrow(self):
        # 1. Baseline: primary is serving at ep_size=4.
        self._generate_ok("pre-shrink")

        # 2. Shrink 4 -> 3. Retiree exits with os._exit(0); survivors flip
        #    active_ranks[retiree]=False and rebuild the EPLB layout.
        self._scale_to(old_ep_size=LAUNCH_EP_SIZE, target_ep_size=LAUNCH_EP_SIZE - 1)
        self._generate_ok("post-shrink")

        # 3. Launch joiner into the freshly-retired slot (rank 3).
        joiner = self._launch_recover_joiner(
            rank_offset=LAUNCH_EP_SIZE - 1, join_tp=1, port=PORT_JOINER
        )
        try:
            self.assertIsNone(joiner.poll(), "joiner exited before scale request")

            # 4. Grow 3 -> 4. Survivor's recover_ranks pairs with the joiner's
            #    join_process_groups over the DPC socket.
            self._scale_to(
                old_ep_size=LAUNCH_EP_SIZE - 1, target_ep_size=LAUNCH_EP_SIZE
            )

            # 5. Confirm every DP slot (including the recovered one) serves.
            self._generate_ok(
                "post-regrow", routed_dp_rank=LAUNCH_EP_SIZE - 1
            )
        finally:
            try:
                kill_process_tree(joiner.pid)
                joiner.wait(timeout=10)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
