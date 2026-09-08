"""End-to-end examples for Mooncake-native scale-down.

Shrink 4 -> 3 then regrow through a recover joiner; a long decode pinned to the
retiring rank that the drain gate must return whole; and the expert slots a shrink
cannot source over p2p (no GPU needed). The first two need >= 4 GPUs::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest \\
        test/manual/ep/test_elastic_shrink.py -v -s
"""

import os
import subprocess
import sys
import threading
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
    import torch

    return [str(i) for i in range(torch.cuda.device_count())]


def _kill(proc, timeout: float = 15) -> None:
    if proc is None:
        return
    try:
        kill_process_tree(proc.pid)
        proc.wait(timeout=timeout)
    except Exception:
        pass


def _common_args(tp: int, *, nnodes: int = 1, node_rank: int = 0) -> list[str]:
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
    """Each scenario gets its own ep_size=4 primary."""

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

    def _post(self, path: str, timeout: float = 60, **kw) -> requests.Response:
        return requests.post(f"{self.base_url}{path}", timeout=timeout, **kw)

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
        # 600s matches elastic_ep_scale_timeout: a slow joiner cold-start must not race us.
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
                self._generate()  # Keep the busy path warm so the retire handler ticks.
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
        # A joiner into a retired slot runs as ``--nnodes 2 --node-rank 1`` of a logical
        # (primary=0, joiner=1) view, which is what DPC expects.
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
        self._generate_ok("pre-shrink")
        self._scale_to(old_ep_size=LAUNCH_EP_SIZE, target_ep_size=LAUNCH_EP_SIZE - 1)
        self._generate_ok("post-shrink")

        joiner = self._launch_recover_joiner(
            rank_offset=LAUNCH_EP_SIZE - 1, join_tp=1, port=PORT_JOINER
        )
        self.addCleanup(_kill, joiner, 10)
        self.assertIsNone(joiner.poll(), "joiner exited before scale request")

        self._scale_to(old_ep_size=LAUNCH_EP_SIZE - 1, target_ep_size=LAUNCH_EP_SIZE)
        # Route at the recovered slot specifically: a grow that reports done before the
        # joiner can serve still passes an unrouted /generate.
        self._generate_ok("post-regrow", routed_dp_rank=LAUNCH_EP_SIZE - 1)


@_NEEDS_GPUS
class TestElasticShrinkDrainsPinnedDecode(_ElasticShrinkBase):
    """A decode already running on the retiring rank must come back whole. The tokenizer
    gate blocks only *new* admissions and a retiree's terminal is ``sys.exit()``, so an
    in-flight request survives only because the retiree withholds its drain-barrier
    arrival until its queues empty."""

    # Decode is eager here (CUDA graphs disabled), so 1024 steps outlast the shrink, and
    # ignore_eos fixes the length so a truncated answer is unambiguous.
    DECODE_TOKENS = int(os.environ.get("SGLANG_MC_DRAIN_TOKENS", "1024"))
    PREROLL_S = 5.0

    def test_shrink_waits_for_pinned_decode(self):
        retiree = LAUNCH_EP_SIZE - 1  # retirees are the contiguous tail
        self._generate_ok("pre-shrink", routed_dp_rank=retiree)

        result: dict = {}

        def _pinned_decode() -> None:
            try:
                resp = self._post(
                    "/generate",
                    timeout=600,  # Outlasts the shrink itself.
                    json={
                        "text": "Describe the number one in exhaustive detail: ",
                        "sampling_params": {
                            "max_new_tokens": self.DECODE_TOKENS,
                            "temperature": 0.0,
                            "ignore_eos": True,
                        },
                        "routed_dp_rank": retiree,
                    },
                )
                result["status"] = resp.status_code
                result["body"] = resp.json() if resp.ok else resp.text
            except Exception as exc:  # noqa: BLE001
                result["error"] = f"{type(exc).__name__}: {exc}"

        decoder = threading.Thread(target=_pinned_decode, daemon=True)
        decoder.start()
        time.sleep(self.PREROLL_S)
        # Tripping here means the shrink never overlaps the decode: raise DECODE_TOKENS.
        self.assertTrue(
            decoder.is_alive(),
            f"pinned decode finished inside the {self.PREROLL_S}s preroll: {result}",
        )

        self._scale_to(old_ep_size=LAUNCH_EP_SIZE, target_ep_size=retiree)

        decoder.join(timeout=600)
        self.assertFalse(decoder.is_alive(), "pinned decode never returned")
        self.assertNotIn(
            "error",
            result,
            f"decode on the retired rank did not survive the shrink: "
            f"{result.get('error')}",
        )
        self.assertEqual(result.get("status"), 200, f"pinned decode failed: {result}")
        meta = result["body"].get("meta_info", {})
        self.assertEqual(
            meta.get("completion_tokens"),
            self.DECODE_TOKENS,
            f"decode on the retired rank was truncated; ignore_eos was set so the "
            f"length should be exact: meta={meta}",
        )


class TestUnsourcedSlotsAreNotPublished(CustomTestCase):
    """A p2p recv from a retired rank must not publish its unfilled buffer. No GPUs.

    The recv and the buffer->weight copy that consumes it are appended together, so
    dropping only the recv leaves the copy to publish a buffer nothing wrote: an
    empty_like reused across layers, giving uninitialised memory on the first layer and
    the previous layer's weights after. Neither is a NaN, which is why a shrink that
    corrupts a share of its experts still serves and still scores."""

    NUM_LOCAL = 3
    NUM_RANKS = 4
    RETIRED_RANK = 3

    # Logical 9 lives only on the retired rank, and rank 0's new map wants it in two
    # slots -- slot 2 free-rides off slot 0's buffer, so both must be held back.
    OLD_MAP = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9]
    NEW_MAP = [9, 1, 9, 3, 4, 5, 6, 7, 8, 0, 2, 8]
    MARKER = 1234.5

    def _run(self):
        import torch

        from sglang.srt.elastic_ep import elastic_ep as ee
        from sglang.srt.eplb import expert_location_updater as elu

        class _FakeP2POp:
            # The real one resolves a process group at construction. Every op here is
            # filtered before execution, so recording the peer is enough.
            def __init__(self, op, tensor, peer):
                self.op, self.tensor, self.peer = op, tensor, peer

        active = torch.ones(self.NUM_RANKS, dtype=torch.bool)
        active[self.RETIRED_RANK] = False

        class _FakeState:
            active_ranks_cpu = active

        # Distinct per slot so a wrong-slot copy is visible, offset by a marker so an
        # uninitialised buffer cannot coincide with a pass.
        weights = [
            torch.arange(self.NUM_LOCAL, dtype=torch.float32).reshape(self.NUM_LOCAL, 1)
            + self.MARKER
        ]
        before = weights[0].clone()
        missing = []

        orig_instance = ee.ElasticEPStateManager.instance
        orig_op = elu.P2POp
        ee.ElasticEPStateManager.instance = classmethod(lambda cls: _FakeState())
        elu.P2POp = _FakeP2POp
        try:
            elu.update_expert_weights_single_layer(
                routed_experts_weights=weights,
                temp_buffers=elu.create_temp_buffers(weights),
                old_physical_to_logical_map=list(self.OLD_MAP),
                new_physical_to_logical_map=list(self.NEW_MAP),
                num_local_physical_experts=self.NUM_LOCAL,
                num_gpu_per_node=self.NUM_RANKS,
                rank=0,
                world_size=self.NUM_RANKS,
                missing_logical_experts_info=missing,
            )
        finally:
            ee.ElasticEPStateManager.instance = orig_instance
            elu.P2POp = orig_op

        return weights[0], before, missing

    def test_unsourced_slots_are_left_for_the_reload(self):
        import torch

        after, before, missing = self._run()
        # Slot 0 took the dropped recv and slot 2 free-rode off the same buffer. Leaving
        # every slot at its old contents is what makes the caller's reload a repair
        # rather than a race against a slot that already went out wrong.
        torch.testing.assert_close(after, before)
        # And those slots stay stale until that reload, so silence here is a shrink
        # quietly serving the wrong experts.
        self.assertEqual(missing, [9])


if __name__ == "__main__":
    unittest.main()
