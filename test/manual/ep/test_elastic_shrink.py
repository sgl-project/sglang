"""End-to-end 4 -> 3 -> 4 example for Mooncake-native scale-down.

Launches a primary at ``ep_size=4`` with ``--max-ep-size 5`` (+1 headroom keeps a
recoverable slot pool), shrinks to 3 via ``/scale_elastic_ep``, starts a joiner with
``--elastic-ep-join-mode recover --elastic-ep-join-rank-offset 3`` into the freed slot,
grows back to 4, then checks ``/generate`` still serves. Requires >= 4 GPUs::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest \\
        test/manual/ep/test_elastic_shrink.py -v -s
"""

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

# The a2a buffer bounds per-rank in-flight tokens; 1024 is the max.
os.environ.setdefault("SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")

LAUNCH_EP_SIZE = 4
MAX_EP_SIZE = 5  # +1 headroom keeps a recoverable slot pool for regrow.
HOST = "127.0.0.1"
PORT_PRIMARY = int(os.environ.get("SGLANG_MC_PORT_A", "21100"))
PORT_JOINER = int(os.environ.get("SGLANG_MC_PORT_B", "10100"))
DIST_INIT_ADDR = os.environ.get("SGLANG_MC_DIST_INIT", "127.0.0.1:24655")
BASE_URL = f"http://{HOST}:{PORT_PRIMARY}"

# 24 fills the EPLB layout at ep=4 (72 + 24 = 96) and keeps ep=3 feasible.
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


def _kill(proc, timeout: float = 15) -> None:
    if proc is None:
        return
    try:
        kill_process_tree(proc.pid)
        proc.wait(timeout=timeout)
    except Exception:
        pass


def _common_args(tp: int, *, nnodes: int = 1, node_rank: int = 0) -> list[str]:
    """Server flags shared by primary and joiner."""
    return (
        f"--trust-remote-code --moe-a2a-backend nixl --deepep-mode low_latency "
        f"--tp {tp} --dp {tp} --enable-dp-attention --enable-dp-lm-head "
        f"--elastic-ep-backend mooncake --enable-eplb "
        f"--mooncake-ib-device {get_rdma_devices_args()} "
        f"--ep-num-redundant-experts {EP_NUM_REDUNDANT_EXPERTS} "
        f"--max-ep-size {MAX_EP_SIZE} --mem-fraction-static 0.5 "
        f"--chunked-prefill-size 1024 --nnodes {nnodes} --node-rank {node_rank} "
        f"--dist-init-addr {DIST_INIT_ADDR} --moe-dense-tp-size 1 "
        f"--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled"
    ).split()


_NEEDS_GPUS = unittest.skipUnless(
    len(_visible_devices()) >= LAUNCH_EP_SIZE, f"Needs >= {LAUNCH_EP_SIZE} GPUs."
)


class _ElasticShrinkBase(CustomTestCase):
    """Fixtures shared by the shrink scenarios; each gets its own ep_size=4 primary."""

    @classmethod
    def setUpClass(cls):
        cls.model = TEST_MODEL
        cls.base_url = BASE_URL
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(_visible_devices()[:LAUNCH_EP_SIZE])
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_common_args(tp=LAUNCH_EP_SIZE),
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        _kill(getattr(cls, "process", None))
        time.sleep(2)

    def _post(self, path: str, **kw) -> requests.Response:
        return requests.post(f"{self.base_url}{path}", timeout=60, **kw)

    _PAYLOAD = {
        "text": "Hello",
        "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
    }

    def _generate(self, **extra) -> requests.Response:
        return self._post("/generate", json={**self._PAYLOAD, **extra})

    def _generate_ok(self, msg: str, **extra) -> None:
        resp = self._generate(**extra)
        self.assertEqual(resp.status_code, 200, f"/generate {msg}: {resp.text}")

    def _poll_until_serving(
        self, *, expected_ep_size: int, expected_phase: str, timeout_s: float = 600.0
    ) -> None:
        # 600s matches elastic_ep_scale_timeout so a slow joiner cold-start
        # does not race the harness.
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            resp = requests.get(f"{self.base_url}/is_scaling_elastic_ep", timeout=60)
            state = resp.json() if resp.ok else None
            if state and not state.get("is_scaling_elastic_ep", True):
                self.assertEqual(state.get("effective_ep_size"), expected_ep_size)
                self.assertEqual(state.get("scale_phase"), expected_phase)
                self.assertIsNone(state.get("last_error"))
                return
            try:
                self._generate()  # Keep busy path warm so the retire handler ticks.
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
        self._poll_until_serving(expected_ep_size=target_ep_size, expected_phase=phase)

    @classmethod
    def _launch_recover_joiner(
        cls, *, rank_offset: int, join_tp: int, port: int
    ) -> subprocess.Popen:
        # A joiner into a retired slot runs as ``--nnodes 2 --node-rank 1`` of a
        # logical (primary=0, joiner=1) view, which is what DPC expects.
        join_args = (
            f"--elastic-ep-initial-size {LAUNCH_EP_SIZE} "
            f"--elastic-ep-join-mode recover "
            f"--elastic-ep-join-rank-offset {rank_offset} "
            f"--host {HOST} --port {port} --device cuda"
        ).split()
        devices = _visible_devices()
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[rank_offset:][:join_tp])
        env.setdefault("PYTHONUNBUFFERED", "1")
        args = _common_args(tp=join_tp, nnodes=2, node_rank=1) + join_args
        cmd = ["sglang", "serve", "--model-path", cls.model, *args]
        return subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)


@_NEEDS_GPUS
class TestElasticShrinkThenRegrow(_ElasticShrinkBase):
    def test_shrink_then_regrow(self):
        # 1. Baseline: primary is serving at ep_size=4.
        self._generate_ok("pre-shrink")

        # 2. Shrink 4 -> 3: retiree exits, survivors clear its active_ranks bit.
        self._scale_to(old_ep_size=LAUNCH_EP_SIZE, target_ep_size=LAUNCH_EP_SIZE - 1)
        self._generate_ok("post-shrink")

        # 3. Launch joiner into the freshly-retired slot (rank 3).
        joiner = self._launch_recover_joiner(
            rank_offset=LAUNCH_EP_SIZE - 1, join_tp=1, port=PORT_JOINER
        )
        self.addCleanup(_kill, joiner, 10)
        self.assertIsNone(joiner.poll(), "joiner exited before scale request")

        # 4. Grow 3 -> 4: recover_ranks pairs with the joiner over the DPC socket.
        self._scale_to(old_ep_size=LAUNCH_EP_SIZE - 1, target_ep_size=LAUNCH_EP_SIZE)

        # 5. Confirm every DP slot (including the recovered one) serves.
        self._generate_ok("post-regrow", routed_dp_rank=LAUNCH_EP_SIZE - 1)


if __name__ == "__main__":
    unittest.main()
