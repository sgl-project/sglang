"""Manual end-to-end tests for Mooncake-native scale-down (MC01-MC14).

Requires --elastic-ep-backend mooncake and >= 4 GPUs (8 for multi-node).

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest \\
        test/manual/ep/test_mooncake_scale_down.py::TestMooncakeScaleDown4To3 \\
        -v -s
"""

from __future__ import annotations

import importlib.util
import math
import os
import statistics
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
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


def _load_sibling_module(name: str, path: Path):
    # ``test.manual.ep`` is not always on ``sys.path`` when pytest picks
    # this file up directly, so load the sibling helpers module by path.
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ELASTIC_SCALE = _load_sibling_module(
    "sglang_elastic_scale_helpers",
    Path(__file__).with_name("test_elastic_scale.py"),
)
DISABLED_CUDA_GRAPH_ARGS = _ELASTIC_SCALE.DISABLED_CUDA_GRAPH_ARGS
_count_visible_gpus = _ELASTIC_SCALE._count_visible_gpus
_extra_server_args = _ELASTIC_SCALE._extra_server_args
_visible_device_ids = _ELASTIC_SCALE._visible_device_ids

TEST_MODEL = os.environ.get("NIXL_EP_TEST_MODEL", DEFAULT_MODEL_NAME_FOR_TEST_MLA)
# Mooncake EP buffer asserts on-the-fly at ``mooncake_ep_buffer.cpp:221``
# that per-rank in-flight tokens fit its ring; 1024 is the max the C++
# side accepts and is needed for a GSM8K prefill with chunked-prefill 256.
os.environ.setdefault("SGLANG_NIXL_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")
os.environ.setdefault("SGLANG_MOONCAKE_EP_NUM_MAX_DISPATCH_TOKENS_PER_RANK", "1024")

ib_devices = get_rdma_devices_args()

LAUNCH_EP_SIZE = 4
_MAX_EP_ENV = os.environ.get("SGLANG_MC_MAX_EP_SIZE", "").strip()
MAX_EP_SIZE = int(_MAX_EP_ENV) if _MAX_EP_ENV else LAUNCH_EP_SIZE
GSM8K_MIN_SCORE = float(os.environ.get("SGLANG_MC_GSM8K_MIN", "0.50"))
# Relative-drop tolerance for single-round post-scale GSM8K parity tests
# (MC02, MC04, MC05, MC12). MC06 opts into deterministic mode instead
# and asserts byte-identical scores across rounds. 0.10 covers observed
# post-fork numeric drift (max delta ~7.5%) while still catching
# wholesale quality collapse.
GSM8K_REL_TOL = float(os.environ.get("SGLANG_MC_GSM8K_REL_TOL", "0.10"))
GSM8K_NUM_EXAMPLES = int(os.environ.get("SGLANG_MC_GSM8K_NUM", "128"))
# Deterministic-mode workload (MC06). Smaller than GSM8K_NUM_EXAMPLES
# because deterministic mode forces num_threads=1 (~90s per 32-question
# serial pass on DSV3-lite fp8).
GSM8K_NUM_EXAMPLES_DETERMINISTIC = int(
    os.environ.get("SGLANG_MC_GSM8K_NUM_DETERMINISTIC", "32")
)
DETERMINISTIC_SEED = int(os.environ.get("SGLANG_MC_DETERMINISTIC_SEED", "42"))
# Loose safety margin -- fires only for wholesale corruption; primary
# signal is exact byte-equality across post-shrink rounds.
GSM8K_DETERMINISTIC_SAFETY_MARGIN = float(
    os.environ.get("SGLANG_MC_GSM8K_DETERMINISTIC_SAFETY_MARGIN", "0.15")
)

NUM_LOGICAL_EXPERTS = 72  # DeepSeek-V3-Lite n_routed_experts


def _min_redundant_experts_for_shrink(
    launch_ep: int,
    min_target_ep: int,
    num_logical: int = NUM_LOGICAL_EXPERTS,
) -> int:
    """Minimum ``--ep-num-redundant-experts`` for a shrink test.

    Simultaneously satisfies EPLB layout divisibility at fresh launch
    ((num_logical + n) % launch_ep == 0) and scheduler shrink
    feasibility (num_local * min_target_ep >= num_logical). Returns
    ceil(num_logical / min_target_ep) * launch_ep - num_logical."""
    assert launch_ep > 0 and min_target_ep > 0 and num_logical > 0
    assert min_target_ep <= launch_ep, (
        f"min_target_ep ({min_target_ep}) cannot exceed launch_ep "
        f"({launch_ep}) for a shrink test."
    )
    k_min = math.ceil(num_logical / min_target_ep)
    return k_min * launch_ep - num_logical


DIST_INIT_ADDR = os.environ.get("SGLANG_MC_DIST_INIT", "127.0.0.1:24655")
PORT_A = int(os.environ.get("SGLANG_MC_PORT_A", "21100"))
PORT_B = int(os.environ.get("SGLANG_MC_PORT_B", "10100"))
# PORT_C is used when a test needs a SECOND joiner subprocess in the
# same run (e.g. MC14's shrink-recover-then-append chains a recover
# joiner on PORT_B with an append joiner on PORT_C).
PORT_C = int(os.environ.get("SGLANG_MC_PORT_C", "10101"))
HOST_A = os.environ.get("SGLANG_MC_HOST_A", "127.0.0.1")
BASE_URL_A = f"http://{HOST_A}:{PORT_A}"


def _shrink_common_args(
    *,
    dist_init_addr: str,
    tp_size: int,
    max_ep_size: int,
    moe_dense_tp_size: int | None,
    ep_num_redundant_experts: int = 24,
    moe_a2a_backend: str = "mooncake",
    deterministic: bool = False,
) -> list[str]:
    """Common `sglang serve` args for Mooncake-native shrink tests.

    Grow-direction tests MUST pass moe_a2a_backend="nixl" per PR #30164
    (Mooncake a2a on the grow path is not validated upstream).
    ep_num_redundant_experts controls minimum shrink cohort size.

    deterministic=True enables --enable-deterministic-inference + a
    fixed random seed + explicit fa3 attention backend so MC06 can
    assert exact-equality across post-shrink rounds."""
    args = [
        "--trust-remote-code",
        "--moe-a2a-backend",
        moe_a2a_backend,
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
        str(ep_num_redundant_experts),
        "--max-ep-size",
        str(max_ep_size),
        "--mem-fraction-static",
        "0.5",
        "--chunked-prefill-size",
        "1024",
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--dist-init-addr",
        dist_init_addr,
    ]
    if moe_dense_tp_size is not None:
        args.extend(["--moe-dense-tp-size", str(moe_dense_tp_size)])
    if deterministic:
        args.extend(
            [
                "--enable-deterministic-inference",
                "--random-seed",
                str(DETERMINISTIC_SEED),
                # Pin fa3 (what upstream's resolver picks anyway on
                # Hopper + DeepSeek) so future resolver changes cannot
                # silently move MC06 off fa3.
                "--attention-backend",
                "fa3",
                # Push EPLB auto-rebalance past ~6k combined forward
                # passes for a three-round MC06 soak. Do NOT go higher:
                # this value also sizes expert_distribution_recorder_
                # buffer_size at ~11 KB/step (would OOM at 10M).
                "--eplb-rebalance-num-iterations",
                "100000",
                # Detach the recorder buffer from the flag above so it
                # stays a fixed ~11 MB on DSV3-lite.
                "--expert-distribution-recorder-buffer-size",
                "1024",
            ]
        )
    return args + DISABLED_CUDA_GRAPH_ARGS + _extra_server_args()


class _MooncakeShrinkEndToEndBase(CustomTestCase):
    """Shared plumbing for MC0N shrink E2E tests."""

    CUDA_GRAPH_ARGS = DISABLED_CUDA_GRAPH_ARGS
    MOE_DENSE_TP_SIZE: int | None = 1
    EP_NUM_REDUNDANT_EXPERTS: int = 24  # override for shrinks below ceil(num_logical/num_local)
    MOE_A2A_BACKEND: str = "nixl"  # Mooncake-a2a variants override
    LAUNCH_EP: int = LAUNCH_EP_SIZE  # override on non-4-rank fresh launches (MC03A etc.)
    # Elastic pool ceiling reserved at primary launch. Grow-back tests
    # must set MAX_EP > LAUNCH_EP so Mooncake keeps a recoverable slot
    # pool for try_recover_ranks.
    MAX_EP: int = MAX_EP_SIZE
    # Opt-in for deterministic mode (see _shrink_common_args). Only
    # MC06 uses this; off by default because deterministic mode disables
    # aiter fusion and slows unrelated MC0N tests.
    DETERMINISTIC: bool = False

    @classmethod
    def setUpClass(cls):
        if cls is _MooncakeShrinkEndToEndBase:
            raise unittest.SkipTest("Abstract base")
        cls.model = TEST_MODEL
        cls.base_url = BASE_URL_A

        primary_args = _shrink_common_args(
            dist_init_addr=DIST_INIT_ADDR,
            tp_size=cls.LAUNCH_EP,
            max_ep_size=cls.MAX_EP,
            moe_dense_tp_size=cls.MOE_DENSE_TP_SIZE,
            ep_num_redundant_experts=cls.EP_NUM_REDUNDANT_EXPERTS,
            moe_a2a_backend=cls.MOE_A2A_BACKEND,
            deterministic=cls.DETERMINISTIC,
        )
        primary_env = os.environ.copy()
        visible_devices = _visible_device_ids()
        if len(visible_devices) < cls.LAUNCH_EP:
            raise RuntimeError(
                f"MC shrink tests need {cls.LAUNCH_EP} visible GPUs, got "
                f"{len(visible_devices)}"
            )
        primary_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            visible_devices[: cls.LAUNCH_EP]
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=primary_args,
            env=primary_env,
        )

    @classmethod
    def tearDownClass(cls):
        proc = getattr(cls, "process", None)
        if proc is not None:
            try:
                kill_process_tree(proc.pid)
            except Exception:
                pass
            try:
                proc.wait(timeout=15)
            except Exception:
                pass
        time.sleep(2)

    # -------------------------------- helpers -------------------------------

    def _post(self, path: str, **kwargs) -> requests.Response:
        return requests.post(f"{self.base_url}{path}", timeout=60, **kwargs)

    def _generate_ok(self, msg_suffix: str, routed_dp_rank: int | None = None) -> None:
        payload = {
            "text": "Hello",
            "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
        }
        if routed_dp_rank is not None:
            payload["routed_dp_rank"] = routed_dp_rank
        resp = self._post("/generate", json=payload)
        self.assertEqual(
            resp.status_code, 200, f"/generate {msg_suffix} failed: {resp.text}"
        )

    def _run_gsm8k(
        self,
        tag: str,
        *,
        num_threads: int | None = None,
        num_examples: int | None = None,
    ) -> float:
        """Run GSM8K against the currently-serving cohort.

        Default: 128 questions at num_threads=16. MC06 (deterministic)
        overrides both to run a serial pass at
        num_examples=GSM8K_NUM_EXAMPLES_DETERMINISTIC."""
        if num_threads is None:
            num_threads = 16
        if num_examples is None:
            num_examples = GSM8K_NUM_EXAMPLES
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                api="completion",
                max_tokens=512,
                num_examples=num_examples,
                num_threads=num_threads,
            )
        )
        score = float(metrics["score"])
        print(
            f"[TEST] GSM8K accuracy ({tag}): {score:.2%} "
            f"(n={num_examples}, threads={num_threads})"
        )
        return score

    def _flush_kv_cache(self, *, timeout_s: float = 60.0) -> None:
        """Flush server radix + KV caches between deterministic-mode
        GSM8K rounds. Without this, FP8 KV residue breaks byte-identical
        scores across rounds even under deterministic-inference."""
        resp = requests.post(
            f"{self.base_url}/flush_cache",
            timeout=timeout_s,
        )
        self.assertEqual(
            resp.status_code,
            200,
            f"/flush_cache failed: HTTP {resp.status_code} body={resp.text!r}",
        )

    def _poll_until_serving(
        self,
        *,
        expected_ep_size: int,
        expected_phase: str,
        timeout_s: float = 600.0,
    ) -> None:
        # 600s matches primary-side elastic_ep_scale_timeout to
        # accommodate slow joiner cold-start on scale-up-v1 (MC02B/MC14).
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            resp = requests.get(
                f"{self.base_url}/is_scaling_elastic_ep", timeout=60
            )
            state = resp.json() if resp.ok else None
            if state is not None and not state.get("is_scaling_elastic_ep", True):
                self.assertEqual(state.get("effective_ep_size"), expected_ep_size)
                self.assertEqual(state.get("scale_phase"), expected_phase)
                self.assertIsNone(state.get("last_error"))
                return
            # Keep the busy path warm so the retire handler ticks.
            try:
                self._post(
                    "/generate",
                    json={
                        "text": "ping",
                        "sampling_params": {
                            "max_new_tokens": 1,
                            "temperature": 0.0,
                        },
                    },
                )
            except Exception:
                pass
            time.sleep(2)
        self.fail(f"Timed out waiting for scale to reach {expected_phase}")

    # Post-scale survivability floor: number of /generate probes after
    # every _scale_to to catch "serving_* but next generate crashes"
    # regressions. Subclasses that already exercise the new topology
    # (MC08/MC09/MC07) can override.
    POST_SCALE_MIN_PROBES: int = 2

    def _assert_post_scale_survives(self, new_ep_size: int) -> None:
        """Run max(new_ep_size, POST_SCALE_MIN_PROBES) /generate probes
        striped across live DP slots, so every slot in the new topology
        receives at least one forward pass immediately after the scale."""
        probes = max(new_ep_size, self.POST_SCALE_MIN_PROBES)
        for i in range(probes):
            dp_rank = i % new_ep_size if new_ep_size > 0 else None
            self._generate_ok(
                f"post-scale-survivability probe {i + 1}/{probes} "
                f"(target_ep={new_ep_size}, dp={dp_rank})",
                routed_dp_rank=dp_rank,
            )

    def _scale_to(
        self,
        *,
        old_ep_size: int,
        target_ep_size: int,
    ) -> None:
        resp = self._post(
            "/scale_elastic_ep", json={"new_ep_size": target_ep_size}
        )
        # 200 = accepted; body carries ``{message, old_ep_size, new_ep_size}``.
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["old_ep_size"], old_ep_size, body)
        self.assertEqual(body["new_ep_size"], target_ep_size, body)

        expected_phase = (
            "serving_shrunk"
            if target_ep_size < old_ep_size
            else "serving_expanded"
        )
        self._poll_until_serving(
            expected_ep_size=target_ep_size,
            expected_phase=expected_phase,
        )
        # Enforce the "cohort must serve N more forward passes after every
        # scale completes" invariant on every _scale_to caller uniformly.
        # Subclasses whose downstream flow already probes every slot
        # (MC08 / MC09 sustained-probe sweeps) still get this cheap
        # sanity check for free; extra probes are idempotent.
        self._assert_post_scale_survives(new_ep_size=target_ep_size)

    def _assert_no_orphan_processes(self, retired_slots: int) -> None:
        """Sanity: retiree Python processes have actually exited.

        Uses ``ps`` since ``nvidia-smi`` isn't available in every env.
        Counts scheduler processes bound to this test's cohort and
        asserts we lost exactly ``retired_slots`` of them.
        """
        try:
            out = subprocess.check_output(
                ["pgrep", "-a", "-f", "sglang.srt"], text=True
            )
        except subprocess.CalledProcessError:
            out = ""
        scheduler_lines = [
            line for line in out.splitlines() if "run_scheduler_process" in line
        ]
        # Heuristic threshold: we can't perfectly correlate pids to slots
        # here without more scaffolding; just log and continue.
        print(
            f"[TEST] active scheduler-like pids after shrink "
            f"(retired_slots={retired_slots}): {len(scheduler_lines)}"
        )


    # ---- offset-joiner helpers ------------------------------------------

    @classmethod
    def _launch_offset_joiner(
        cls,
        *,
        rank_offset: int,
        join_tp: int,
        port: int,
        join_mode: str,
    ) -> subprocess.Popen:
        """Launch a joiner subprocess targeting a specific rank offset.

        ``join_mode="scale"`` is append-only scale-up: the primary was
        launched with ``max_ep_size > launch_ep_size`` and the joiner
        slots into a fresh, never-active slot beyond the launch cohort.
        ``join_mode="recover"`` is grow-into-retired-slot: the joiner
        slots into a slot inside the launch cohort that was previously
        retired via a scale-down.
        """
        primary_args = _shrink_common_args(
            dist_init_addr=DIST_INIT_ADDR,
            tp_size=join_tp,
            max_ep_size=cls.MAX_EP,
            moe_dense_tp_size=cls.MOE_DENSE_TP_SIZE,
            # Joiner must use the same ep_num_redundant_experts as
            # the primary; its own EPLB layout check asserts
            # (num_logical + n) % elastic_ep_initial_size == 0.
            ep_num_redundant_experts=cls.EP_NUM_REDUNDANT_EXPERTS,
            moe_a2a_backend=cls.MOE_A2A_BACKEND,
        )
        # Joiner runs as --nnodes 2 --node-rank 1 of a logical
        # (primary=0, joiner=1) view (what DPC expects for offset joiners).
        _rewrote_nnodes = False
        _rewrote_node_rank = False
        for idx, tok in enumerate(primary_args):
            if tok == "--nnodes":
                primary_args[idx + 1] = "2"
                _rewrote_nnodes = True
            elif tok == "--node-rank":
                primary_args[idx + 1] = "1"
                _rewrote_node_rank = True
            if _rewrote_nnodes and _rewrote_node_rank:
                break

        cmd = [
            "sglang",
            "serve",
            "--model-path",
            cls.model,
            *primary_args,
            # Joiner's elastic_ep_initial_size must match the primary's
            # LAUNCH_EP (per subclass) so pg_world_size lines up.
            "--elastic-ep-initial-size",
            str(cls.LAUNCH_EP),
            "--elastic-ep-join-mode",
            join_mode,
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
        env.setdefault("PYTHONUNBUFFERED", "1")
        visible_devices = _visible_device_ids()
        join_end = rank_offset + join_tp
        if join_end > len(visible_devices):
            raise RuntimeError(
                f"MC slot-reuse needs {join_end} visible GPUs, got "
                f"{len(visible_devices)}"
            )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            visible_devices[rank_offset:join_end]
        )
        # Prefer the persistent log dir so a failed run leaves the
        # joiner side of the hang inspectable after ``/tmp`` on the
        # compute node is torn down with the container.
        default_log_dir = os.environ.get(
            "SGLANG_ELASTIC_LOG_DIR",
            "/lustre/fsw/portfolios/network/users/qkang/logs",
        )
        default_log_path = os.path.join(
            default_log_dir,
            f"mc_joiner_off{rank_offset}_{int(time.time())}.log",
        )
        log_path = os.environ.get("SGLANG_MC_JOINER_LOG", default_log_path)
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        except OSError:
            log_path = f"/tmp/mc_joiner_off{rank_offset}_{int(time.time())}.log"
        fh = open(log_path, "w")
        proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)
        return proc


@unittest.skipUnless(
    _count_visible_gpus() >= LAUNCH_EP_SIZE,
    f"MC01 shrink E2E needs {LAUNCH_EP_SIZE} GPUs.",
)
class TestMooncakeScaleDown4To3(_MooncakeShrinkEndToEndBase):
    """MC01: 4 -> 3 shrink with pre/post GSM8K parity check."""

    def test_scale_down_on_demand(self):
        self._generate_ok("pre-shrink")

        pre_score = self._run_gsm8k("pre-shrink 4-rank")

        self._scale_to(old_ep_size=4, target_ep_size=3)

        self._generate_ok("post-shrink")
        self._assert_no_orphan_processes(retired_slots=1)

        post_score = self._run_gsm8k("post-shrink 3-rank")

        # vLLM-style relative-tolerance parity: post must not regress more
        # than GSM8K_REL_TOL from the pre baseline.
        rel_delta = (pre_score - post_score) / max(pre_score, 1e-9)
        print(
            f"[TEST] GSM8K parity: pre={pre_score:.2%} post={post_score:.2%} "
            f"rel_delta={rel_delta:.2%} tol={GSM8K_REL_TOL:.2%}"
        )
        self.assertLess(
            rel_delta,
            GSM8K_REL_TOL,
            f"Post-shrink GSM8K regressed more than {GSM8K_REL_TOL:.0%}: "
            f"pre={pre_score:.2%} post={post_score:.2%}",
        )
        self.assertGreater(
            post_score,
            GSM8K_MIN_SCORE,
            f"Post-shrink GSM8K accuracy too low: {post_score:.2%}",
        )


@unittest.skipUnless(
    _count_visible_gpus() >= LAUNCH_EP_SIZE,
    f"MC02 slot-reuse E2E needs {LAUNCH_EP_SIZE} GPUs.",
)
class TestMooncakeScaleDown4To3To4(_MooncakeShrinkEndToEndBase):
    """MC02: 4 -> 3 -> 4 grow-back via recover-into-retired-slot.

    Survivor's recover_ranks pairs with joiner's join_process_groups
    over the DPC socket that remove_elastic_workers kept bound."""

    # PR #30164 validated grow collectives only against NIXL a2a.
    MOE_A2A_BACKEND = os.environ.get("SGLANG_MC02_A2A_BACKEND", "nixl")
    # +1 headroom so Mooncake keeps a recoverable pool for the retired
    # slot after 4->3 (otherwise recover_ranks returns False forever).
    MAX_EP = LAUNCH_EP_SIZE + 1

    def test_shrink_then_regrow(self):
        self._generate_ok("pre-shrink")
        self._scale_to(old_ep_size=4, target_ep_size=3)
        self._generate_ok("post-shrink")

        joiner = self._launch_offset_joiner(
            rank_offset=3, join_tp=1, port=PORT_B, join_mode="recover",
        )
        try:
            self.assertIsNone(
                joiner.poll(),
                "MC02 grow-back joiner exited before scale request",
            )
            self._scale_to(old_ep_size=3, target_ep_size=4)
            self._generate_ok("post-regrow", routed_dp_rank=3)
        finally:
            try:
                kill_process_tree(joiner.pid)
                joiner.wait(timeout=10)
            except Exception:
                pass


@unittest.skipUnless(
    _count_visible_gpus() >= LAUNCH_EP_SIZE,
    f"MC02 (Mooncake a2a) slot-reuse E2E needs {LAUNCH_EP_SIZE} GPUs.",
)
class TestMooncakeScaleDown4To3To4MooncakeA2A(TestMooncakeScaleDown4To3To4):
    """4 -> 3 -> 4 grow-back on Mooncake a2a (known-fail regression fence).

    Same topology as :class:`TestMooncakeScaleDown4To3To4`, but with
    ``--moe-a2a-backend mooncake`` on the grow half. Confirms PR #30164's
    stated limitation (grow direction not supported on Mooncake a2a):
    scale-joiner's DeepGEMM warmup asserts num_groups mismatch before
    any collective runs. Should pass in step with the NIXL variant
    once the joiner-side EPMoE fix lands upstream."""

    MOE_A2A_BACKEND = "mooncake"


@unittest.skipUnless(
    _count_visible_gpus() >= LAUNCH_EP_SIZE,
    f"NIXL shrink test needs {LAUNCH_EP_SIZE} GPUs.",
)
class TestMooncakeScaleDownNixlShrink(_MooncakeShrinkEndToEndBase):
    """4->3 shrink only, NIXL a2a variant."""

    MOE_A2A_BACKEND = "nixl"
    MAX_EP = LAUNCH_EP_SIZE + 1  # +1 keeps retired slot in Mooncake pool

    def test_shrink_only(self):
        self._generate_ok("pre-shrink")
        self._scale_to(old_ep_size=4, target_ep_size=3)
        self._generate_ok("post-shrink")
        self._assert_no_orphan_processes(retired_slots=1)


class _MooncakeGrowFromShrunkBase(CustomTestCase):
    """Fresh-launch primary at "as-if-shrunk" size that exercises only
    the grow half of a shrink+grow scenario, isolating scale-up
    regressions from scale-down regressions."""

    CUDA_GRAPH_ARGS = DISABLED_CUDA_GRAPH_ARGS
    MOE_DENSE_TP_SIZE: int | None = 1
    LAUNCH_EP: int
    TARGET_EP: int
    JOIN_TP: int = 1
    EP_NUM_REDUNDANT_EXPERTS: int = 24
    # PR #30164 known limitation: "Runtime scale-up supports Mooncake
    # as the Elastic EP backend and NIXL as the MoE all-to-all
    # backend." Mooncake a2a on the grow path is outside the tested
    # upstream envelope, so grow tests default to NIXL.
    MOE_A2A_BACKEND: str = "nixl"
    # Subclasses can override to allocate max_ep_size larger than
    # target_ep. Needed because Mooncake's ``bootstrapLocalPeer`` in the
    # joiner requires ``max_world_size > world_size`` on the joiner side.
    _MAX_EP_HEADROOM: int | None = None

    @classmethod
    def _max_ep(cls) -> int:
        return cls._MAX_EP_HEADROOM if cls._MAX_EP_HEADROOM is not None else cls.TARGET_EP

    @classmethod
    def setUpClass(cls):
        if cls is _MooncakeGrowFromShrunkBase:
            raise unittest.SkipTest("Abstract base")
        cls.model = TEST_MODEL
        cls.base_url = BASE_URL_A

        primary_args = _shrink_common_args(
            dist_init_addr=DIST_INIT_ADDR,
            tp_size=cls.LAUNCH_EP,
            max_ep_size=cls._max_ep(),
            moe_dense_tp_size=cls.MOE_DENSE_TP_SIZE,
            ep_num_redundant_experts=cls.EP_NUM_REDUNDANT_EXPERTS,
            moe_a2a_backend=cls.MOE_A2A_BACKEND,
        )
        # Pin launch-size so the joiner boots with the same
        # ``elastic_ep_initial_size`` view.
        primary_args += ["--elastic-ep-initial-size", str(cls.LAUNCH_EP)]
        primary_env = os.environ.copy()
        visible_devices = _visible_device_ids()
        if len(visible_devices) < cls.LAUNCH_EP:
            raise RuntimeError(
                f"MC03B needs {cls.LAUNCH_EP} visible GPUs, got "
                f"{len(visible_devices)}"
            )
        primary_env["CUDA_VISIBLE_DEVICES"] = ",".join(
            visible_devices[: cls.LAUNCH_EP]
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=primary_args,
            env=primary_env,
        )
        cls._joiner_proc = None

    @classmethod
    def tearDownClass(cls):
        joiner = getattr(cls, "_joiner_proc", None)
        for proc in [joiner, getattr(cls, "process", None)]:
            if proc is None:
                continue
            try:
                kill_process_tree(proc.pid)
            except Exception:
                pass
            try:
                proc.wait(timeout=15)
            except Exception:
                pass
        time.sleep(2)

    def _post(self, path: str, **kwargs) -> requests.Response:
        return requests.post(f"{self.base_url}{path}", timeout=60, **kwargs)

    def _generate_ok(self, msg_suffix: str) -> None:
        resp = self._post(
            "/generate",
            json={
                "text": "Hello",
                "sampling_params": {"max_new_tokens": 4, "temperature": 0.0},
            },
        )
        self.assertEqual(
            resp.status_code, 200, f"/generate {msg_suffix} failed: {resp.text}"
        )

    def _launch_scale_joiner(
        self, *, rank_offset: int, join_tp: int, port: int
    ) -> subprocess.Popen:
        primary_args = _shrink_common_args(
            dist_init_addr=DIST_INIT_ADDR,
            tp_size=join_tp,
            max_ep_size=self._max_ep(),
            moe_dense_tp_size=self.MOE_DENSE_TP_SIZE,
            moe_a2a_backend=self.MOE_A2A_BACKEND,
        )
        # See ``_launch_offset_joiner`` for why the joiner needs
        # ``--nnodes 2 --node-rank 1`` (two-node joiner convention).
        _rewrote_nnodes = False
        _rewrote_node_rank = False
        for idx, tok in enumerate(primary_args):
            if tok == "--nnodes":
                primary_args[idx + 1] = "2"
                _rewrote_nnodes = True
            elif tok == "--node-rank":
                primary_args[idx + 1] = "1"
                _rewrote_node_rank = True
            if _rewrote_nnodes and _rewrote_node_rank:
                break
        cmd = [
            "sglang",
            "serve",
            "--model-path",
            self.model,
            *primary_args,
            "--elastic-ep-initial-size",
            str(self.LAUNCH_EP),
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
                f"MC03B needs {join_end} visible GPUs, got "
                f"{len(visible_devices)}"
            )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(
            visible_devices[rank_offset:join_end]
        )
        log_path = os.environ.get(
            "SGLANG_MC_JOINER_LOG",
            f"/tmp/mc03b_joiner_{int(time.time())}.log",
        )
        fh = open(log_path, "w")
        return subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT)

    def _scale_to(self, *, old_ep_size: int, target_ep_size: int) -> None:
        resp = self._post(
            "/scale_elastic_ep", json={"new_ep_size": target_ep_size}
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["old_ep_size"], old_ep_size, body)
        self.assertEqual(body["new_ep_size"], target_ep_size, body)

        # PR #30164 admission runs from forward(); an idle cluster hangs.
        # Poll /is_scaling_elastic_ep interleaved with a cheap /generate
        # ping to keep the busy path ticking. 600s matches primary-side
        # elastic_ep_scale_timeout for cold-start joiner budget.
        deadline = time.time() + 600.0
        while time.time() < deadline:
            resp = requests.get(
                f"{self.base_url}/is_scaling_elastic_ep", timeout=60
            )
            state = resp.json() if resp.ok else None
            if state is not None and not state.get("is_scaling_elastic_ep", True):
                self.assertEqual(state.get("effective_ep_size"), target_ep_size)
                self.assertEqual(
                    state.get("scale_phase"), "serving_expanded"
                )
                self.assertIsNone(state.get("last_error"))
                return
            try:
                self._post(
                    "/generate",
                    json={
                        "text": "ping",
                        "sampling_params": {
                            "max_new_tokens": 1,
                            "temperature": 0.0,
                        },
                    },
                )
            except Exception:
                pass
            time.sleep(2)
        self.fail("Timed out waiting for grow to reach serving_expanded")


@unittest.skipUnless(
    _count_visible_gpus() >= 4,
    "Fresh 3->4 grow test needs 4 GPUs.",
)
class TestMooncakeScaleUpFreshGrow(_MooncakeGrowFromShrunkBase):
    """MC02B: launch ep=3 with headroom, append slot 3 via scale-up-v1.

    Companion to TestMooncakeScaleDownNixlShrink (shrink-only baseline)."""

    LAUNCH_EP = 3
    TARGET_EP = 4
    _MAX_EP_HEADROOM = 5

    def test_grow_from_fresh(self):
        self._generate_ok("pre-grow")

        joiner = self._launch_scale_joiner(
            rank_offset=self.LAUNCH_EP,
            join_tp=self.JOIN_TP,
            port=PORT_B,
        )
        type(self)._joiner_proc = joiner
        try:
            self.assertIsNone(
                joiner.poll(),
                "MC02B joiner exited before scale request; see joiner log",
            )
            self._scale_to(
                old_ep_size=self.LAUNCH_EP, target_ep_size=self.TARGET_EP
            )
            self._generate_ok("post-grow")
        finally:
            try:
                kill_process_tree(joiner.pid)
                joiner.wait(timeout=10)
            except Exception:
                pass


@unittest.skipUnless(
    _count_visible_gpus() >= 4,
    "MC02B (Mooncake a2a) differential grow-only needs 4 GPUs.",
)
class TestMooncakeGrow3To4OnlyMooncakeA2A(TestMooncakeScaleUpFreshGrow):
    """MC02B (Mooncake a2a variant): 3->4 grow-only on Mooncake a2a.

    Known-fail regression fence: scale-joiner's DeepGEMM warmup asserts
    num_groups mismatch before any collective runs. Should pass once
    the Mooncake-a2a EPMoE runner is fixed upstream to boot with a
    self-consistent num_groups when elastic_ep_initial_size != ep_size."""

    MOE_A2A_BACKEND = "mooncake"


@unittest.skipUnless(
    _count_visible_gpus() >= 5 and MAX_EP_SIZE >= 5,
    "MC03 grow-then-shrink E2E needs 5 GPUs and SGLANG_MC_MAX_EP_SIZE>=5.",
)
class TestMooncakeScaleDown4To5To4(_MooncakeShrinkEndToEndBase):
    """4 -> 5 -> 4 grow-then-shrink; retires a freshly-joined rank."""

    def test_grow_then_shrink(self):
        self._generate_ok("pre-grow")

        joiner = self._launch_offset_joiner(
            rank_offset=4, join_tp=1, port=PORT_B, join_mode="scale",
        )
        try:
            self.assertIsNone(joiner.poll(), "MC03 joiner exited early")
            self._scale_to(old_ep_size=4, target_ep_size=5)
            self._generate_ok("post-grow", routed_dp_rank=4)

            self._scale_to(old_ep_size=5, target_ep_size=4)
            self._generate_ok("post-shrink")
        finally:
            try:
                kill_process_tree(joiner.pid)
                joiner.wait(timeout=10)
            except Exception:
                pass


@unittest.skipUnless(
    os.environ.get("SGLANG_MC13_FORCE_RUN") == "1",
    # See class docstring: requires kvcache-ai/Mooncake PR #2623 (0.3.12+).
    "MC13 requires kvcache-ai/Mooncake PR #2623 (>=0.3.12); "
    "set SGLANG_MC13_FORCE_RUN=1 to run after upgrading the wheel."
)
class TestMooncakeScaleDown4To5To3(_MooncakeShrinkEndToEndBase):
    """MC13: 4 -> 5 -> 3 scale-up-v1 append, then shrink retiring both a
    launch-cohort slot and the ex-append-joiner slot.

    Env-gated by SGLANG_MC13_FORCE_RUN. Mooncake <= 0.3.11.post1 sizes
    P2PProxy peer arrays to the launch cohort, tripping resetPeerState
    out-of-range when retiring the appended rank; fixed upstream in
    kvcache-ai/Mooncake PR #2623 (>= 0.3.12).

    Single-node needs 5 GPUs (4 primary + 1 joiner)."""

    LAUNCH_EP = 4
    MAX_EP = 5
    JOIN_TP = 1

    def test_scale_up_then_shrink(self):
        self._generate_ok("pre-grow")

        joiner = self._launch_offset_joiner(
            rank_offset=self.LAUNCH_EP,
            join_tp=self.JOIN_TP,
            port=PORT_B,
            join_mode="scale",
        )
        try:
            self.assertIsNone(
                joiner.poll(),
                "MC13 joiner exited before scale-up request; see joiner log",
            )
            self._scale_to(
                old_ep_size=self.LAUNCH_EP,
                target_ep_size=self.LAUNCH_EP + self.JOIN_TP,
            )
            self._generate_ok("post-grow", routed_dp_rank=4)

            self._scale_to(
                old_ep_size=self.LAUNCH_EP + self.JOIN_TP,
                target_ep_size=3,
            )
            self._generate_ok("post-shrink")
        finally:
            try:
                kill_process_tree(joiner.pid)
                joiner.wait(timeout=10)
            except Exception:
                pass


MULTINODE_LAUNCH_EP_SIZE = int(
    os.environ.get("SGLANG_MC_MN_LAUNCH_EP_SIZE", "8")
)
MULTINODE_TARGET_EP_SIZE = int(
    os.environ.get("SGLANG_MC_MN_TARGET_EP_SIZE", "6")
)
MULTINODE_MODE = os.environ.get("SGLANG_MC_MN_ROLE", "").lower()  # "primary" | "worker"
MULTINODE_MASTER_HOST = os.environ.get("SGLANG_MC_MN_MASTER_HOST", "127.0.0.1")
MULTINODE_MASTER_PORT = int(os.environ.get("SGLANG_MC_MN_MASTER_PORT", "24855"))


def _multinode_completion_marker() -> str:
    """Shared-fs handshake path used by primary to signal test completion.

    Placed on lustre so the worker (different physical node) can poll it.
    Keyed by ``SLURM_JOBID`` and ``SGLANG_MC_MN_MASTER_PORT`` so
    concurrent MC05 runs never see each other's markers.
    """
    slurm_id = os.environ.get(
        "SLURM_JOB_ID", os.environ.get("SLURM_JOBID", "local")
    )
    return (
        "/lustre/fsw/portfolios/network/users/qkang/logs/"
        f"mc05_done_{slurm_id}_{MULTINODE_MASTER_PORT}.marker"
    )


@unittest.skipUnless(
    MULTINODE_MODE in ("primary", "worker"),
    "MC05 multi-node shrink E2E only runs when SGLANG_MC_MN_ROLE is set "
    "(via scripts/_run_mc05_multinode.sh).",
)
class TestMooncakeScaleDown8To6MultiNode(_MooncakeShrinkEndToEndBase):
    """MC05: 8 -> 6 shrink across two 4-GPU nodes.

    Runs under scripts/_run_mc05_multinode.sh which sets SGLANG_MC_MN_ROLE
    to primary or worker. Retired ranks (6, 7) live on the worker node;
    verifies retiree sys.exit propagates cleanly across nodes and that
    the survivor cohort serves GSM8K post-shrink."""

    MOE_DENSE_TP_SIZE = None
    MOE_A2A_BACKEND: str = "nixl"

    @classmethod
    def setUpClass(cls):
        cls.model = TEST_MODEL
        cls.base_url = BASE_URL_A

        # ``--tp`` is the TOTAL TP across all nodes (SGLang enforces
        # ``tp_size % nnodes == 0``), not the per-node count. Passing
        # per-node here silently creates a smaller cohort and breaks
        # Mooncake's intra-node CUDA IPC path.
        launch_ep = MULTINODE_LAUNCH_EP_SIZE
        node_rank = 0 if MULTINODE_MODE == "primary" else 1

        primary_args = [
            "--trust-remote-code",
            "--moe-a2a-backend",
            cls.MOE_A2A_BACKEND,
            "--deepep-mode",
            "low_latency",
            "--tp",
            str(launch_ep),
            "--dp",
            str(launch_ep),
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--elastic-ep-backend",
            "mooncake",
            "--mooncake-ib-device",
            ib_devices,
            "--enable-eplb",
            "--ep-num-redundant-experts",
            "24",
            "--max-ep-size",
            str(launch_ep),
            "--mem-fraction-static",
            "0.5",
            "--chunked-prefill-size",
            "1024",
            "--nnodes",
            "2",
            "--node-rank",
            str(node_rank),
            "--dist-init-addr",
            f"{MULTINODE_MASTER_HOST}:{MULTINODE_MASTER_PORT}",
            *DISABLED_CUDA_GRAPH_ARGS,
            *_extra_server_args(),
        ]
        env = os.environ.copy()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=primary_args,
            env=env,
        )

    def test_scale_down_across_nodes(self):
        marker = _multinode_completion_marker()
        if MULTINODE_MODE == "worker":
            # Park on the shared-fs completion marker so both nodes tear
            # down together; avoids Mooncake / TCPStore broken-pipe
            # noise from an asymmetric SIGTERM.
            deadline = time.time() + 600
            while time.time() < deadline:
                if os.path.exists(marker):
                    return
                time.sleep(2)
            self.fail(
                f"worker: timed out waiting for primary completion marker {marker}"
            )

        self._generate_ok("pre-shrink")
        pre_score = self._run_gsm8k(f"pre-shrink {MULTINODE_LAUNCH_EP_SIZE}-rank")

        self._scale_to(
            old_ep_size=MULTINODE_LAUNCH_EP_SIZE,
            target_ep_size=MULTINODE_TARGET_EP_SIZE,
        )

        self._generate_ok("post-shrink")
        post_score = self._run_gsm8k(
            f"post-shrink {MULTINODE_TARGET_EP_SIZE}-rank"
        )

        rel_delta = (pre_score - post_score) / max(pre_score, 1e-9)
        print(
            f"[TEST] MC05 GSM8K parity: pre={pre_score:.2%} "
            f"post={post_score:.2%} rel_delta={rel_delta:.2%}"
        )
        self.assertLess(
            rel_delta,
            GSM8K_REL_TOL,
            f"MC05 post-shrink GSM8K regressed more than {GSM8K_REL_TOL:.0%}",
        )

        # Signal the worker that primary work is complete BEFORE tearDown
        # so both sides return from pytest at roughly the same time and
        # their tearDownClass paths kill sglang serve concurrently. This
        # keeps the survivor ranks on the worker node from outliving the
        # WORLD master's TCPStore.
        os.makedirs(os.path.dirname(marker), exist_ok=True)
        with open(marker, "w") as fh:
            fh.write(
                f"done pre={pre_score:.4f} post={post_score:.4f} "
                f"rel_delta={rel_delta:.4f}\n"
            )


@unittest.skipUnless(
    MULTINODE_MODE in ("primary", "worker"),
    "MC05P NIXL a2a variant only runs when SGLANG_MC_MN_ROLE is set "
    "(via scripts/_run_mc05_multinode.sh).",
)
class TestMooncakeScaleDown8To6MultiNodeNixl(TestMooncakeScaleDown8To6MultiNode):
    """MC05P: 8 -> 6 shrink across two nodes with NIXL a2a.

    Guards the post-shrink hang fix in NixlEPDispatcher._combine_core
    (gate query_mask_buffer + sync_active_to_cpu on
    NIXL._connected_ep_size != effective_ep_size)."""

    MOE_A2A_BACKEND = "nixl"


# MC08 topology constants (env-keyed for co-scheduled MC05P + MC08 runs).
MC08_LAUNCH_EP = int(os.environ.get("SGLANG_MC08_LAUNCH_EP", "8"))
MC08_SHRINK_TARGET = int(os.environ.get("SGLANG_MC08_SHRINK_TARGET", "4"))
# Full grow-back only: partial regrow deadlocks Mooncake's
# join_group(WORLD) rendezvous (joiner blocks until every survivor
# reactivates every retiree via recover_ranks).
MC08_REGROW_TARGET = int(
    os.environ.get("SGLANG_MC08_REGROW_TARGET", str(MC08_LAUNCH_EP))
)
# NIXL EP requires num_ranks < 8 or % 8 == 0; 8-launch needs ceiling 16.
MC08_MAX_EP = 16
# Per-slot post-regrow probes -- sweep all DP slots since a grow-back
# corruption in the shared peer table would surface on survivors too.
MC08_POST_REGROW_PROBES_PER_SLOT = int(
    os.environ.get("SGLANG_MC08_POST_REGROW_PROBES_PER_SLOT", "10")
)


def _mc_signal_marker(
    prefix: str, kind: str, cycle: int | None = None
) -> str:
    """Shared-fs handshake path for MCXX primary <-> worker signaling.

    Keyed by ``SLURM_JOBID`` and ``SGLANG_MC_MN_MASTER_PORT`` so
    concurrent multi-node runs never see each other's markers.
    ``cycle`` is optional for tests with multiple shrink-regrow rounds
    (e.g. MC09).
    """
    slurm_id = os.environ.get(
        "SLURM_JOB_ID", os.environ.get("SLURM_JOBID", "local")
    )
    cycle_tag = "" if cycle is None else f"_c{cycle}"
    return (
        "/lustre/fsw/portfolios/network/users/qkang/logs/"
        f"{prefix}_{kind}{cycle_tag}_{slurm_id}_{MULTINODE_MASTER_PORT}.marker"
    )


@unittest.skipUnless(
    MULTINODE_MODE in ("primary", "worker"),
    "MC08 multi-node shrink-then-regrow only runs when SGLANG_MC_MN_ROLE "
    "is set (via scripts/_run_mc08_multinode.sh).",
)
class TestMooncakeScaleDown8To4To6MultiNodeNixl(_MooncakeShrinkEndToEndBase):
    """MC08: 8 -> 4 -> N shrink-then-regrow across two nodes on NIXL a2a.

    N defaults to MC08_LAUNCH_EP (8, a full grow-back); partial grow-back
    is blocked by a Mooncake _join_world_group deadlock (see
    :data:`MC08_REGROW_TARGET`).

    Two nodes launch an 8-rank cohort with max_ep_size = launch_ep + 1;
    shrink 8 -> 4 retires all worker-side ranks; the worker spawns a
    single recover-mode joiner (--tp 2 --dp 2) that re-attaches to slots
    4/5 via try_recover_ranks. Pre-shrink/post-shrink assert GSM8K parity
    (MC01 tolerance); post-regrow asserts every DP slot serves
    MC08_POST_REGROW_PROBES_PER_SLOT sequential /generate requests
    (concurrent-load parity is out-of-scope for the regrow path).

    Requires NIXL a2a: Mooncake a2a on the grow path is known-broken
    and additionally deadlocks the shrink half here."""

    MOE_DENSE_TP_SIZE = None
    MOE_A2A_BACKEND = "nixl"
    MAX_EP = MC08_MAX_EP
    # 72 = minimum to keep the 8->4 shrink feasible
    # (num_local * shrink_target >= num_logical).
    EP_NUM_REDUNDANT_EXPERTS = 72

    @classmethod
    def setUpClass(cls):
        cls.model = TEST_MODEL
        cls.base_url = BASE_URL_A

        launch_ep = MC08_LAUNCH_EP
        node_rank = 0 if MULTINODE_MODE == "primary" else 1

        primary_args = [
            "--trust-remote-code",
            "--moe-a2a-backend",
            cls.MOE_A2A_BACKEND,
            "--deepep-mode",
            "low_latency",
            "--tp",
            str(launch_ep),
            "--dp",
            str(launch_ep),
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--elastic-ep-backend",
            "mooncake",
            "--mooncake-ib-device",
            ib_devices,
            "--enable-eplb",
            "--ep-num-redundant-experts",
            str(cls.EP_NUM_REDUNDANT_EXPERTS),
            "--max-ep-size",
            str(cls.MAX_EP),
            "--mem-fraction-static",
            "0.5",
            "--chunked-prefill-size",
            "1024",
            "--nnodes",
            "2",
            "--node-rank",
            str(node_rank),
            "--dist-init-addr",
            f"{MULTINODE_MASTER_HOST}:{MULTINODE_MASTER_PORT}",
            *DISABLED_CUDA_GRAPH_ARGS,
            *_extra_server_args(),
        ]
        env = os.environ.copy()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=primary_args,
            env=env,
        )
        cls._joiner_proc = None

    @classmethod
    def tearDownClass(cls):
        joiner = getattr(cls, "_joiner_proc", None)
        proc = getattr(cls, "process", None)
        for target in (joiner, proc):
            if target is None:
                continue
            try:
                kill_process_tree(target.pid)
            except Exception:
                pass
            try:
                target.wait(timeout=15)
            except Exception:
                pass
        time.sleep(2)

    def _launch_mc08_recover_joiner(
        self, *, rank_offset: int, join_tp: int
    ) -> subprocess.Popen:
        """Launch a recover-mode joiner subprocess on the worker node.

        Same shape as _launch_offset_joiner but targets the multi-node
        master addr (MC08's primary is not on 127.0.0.1)."""
        cmd = [
            "sglang",
            "serve",
            "--model-path",
            self.model,
            "--trust-remote-code",
            "--moe-a2a-backend",
            self.MOE_A2A_BACKEND,
            "--deepep-mode",
            "low_latency",
            "--tp",
            str(join_tp),
            "--dp",
            str(join_tp),
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--elastic-ep-backend",
            "mooncake",
            "--mooncake-ib-device",
            ib_devices,
            "--enable-eplb",
            "--ep-num-redundant-experts",
            str(self.EP_NUM_REDUNDANT_EXPERTS),
            "--max-ep-size",
            str(self.MAX_EP),
            "--mem-fraction-static",
            "0.5",
            "--chunked-prefill-size",
            "1024",
            "--nnodes",
            "2",
            "--node-rank",
            "1",
            "--dist-init-addr",
            f"{MULTINODE_MASTER_HOST}:{MULTINODE_MASTER_PORT}",
            *DISABLED_CUDA_GRAPH_ARGS,
            *_extra_server_args(),
            "--elastic-ep-initial-size",
            str(MC08_LAUNCH_EP),
            "--elastic-ep-join-mode",
            "recover",
            "--elastic-ep-join-rank-offset",
            str(rank_offset),
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT_B),
            "--device",
            "cuda",
        ]
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        # Pin joiner to the first join_tp worker-local GPUs -- the same
        # ones the pre-shrink retirees occupied, so IB/NUMA topology
        # stays stable across shrink+regrow.
        visible = _visible_device_ids()
        if len(visible) < join_tp:
            raise RuntimeError(
                f"MC08 joiner needs {join_tp} worker-local GPUs; got "
                f"{len(visible)} in CUDA_VISIBLE_DEVICES"
            )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(visible[:join_tp])

        log_dir = os.environ.get(
            "SGLANG_ELASTIC_LOG_DIR",
            "/lustre/fsw/portfolios/network/users/qkang/logs",
        )
        log_path = os.path.join(
            log_dir, f"mc08_joiner_off{rank_offset}_{int(time.time())}.log"
        )
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
        except OSError:
            log_path = (
                f"/tmp/mc08_joiner_off{rank_offset}_{int(time.time())}.log"
            )
        fh = open(log_path, "w")
        proc = subprocess.Popen(
            cmd, env=env, stdout=fh, stderr=subprocess.STDOUT
        )
        print(
            f"[TEST] MC08 worker spawned joiner pid={proc.pid} "
            f"rank_offset={rank_offset} join_tp={join_tp} log={log_path}"
        )
        return proc

    def _wait_for_marker(
        self,
        marker: str,
        *,
        timeout_s: float,
        abort_marker: str | None = None,
    ) -> bool:
        """Poll ``marker`` on the shared filesystem.

        Returns True on success, False if ``abort_marker`` appears
        first (primary aborted mid-test), and raises via ``self.fail``
        on timeout so the responsible side surfaces the failure.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if os.path.exists(marker):
                return True
            if abort_marker is not None and os.path.exists(abort_marker):
                return False
            time.sleep(2)
        self.fail(f"timed out waiting for marker {marker}")
        return False  # unreachable; keeps type-checkers happy

    def test_shrink_then_regrow_across_nodes(self):
        launch_joiner_marker = _mc_signal_marker("mc08", "launch_joiner")
        joiner_ready_marker = _mc_signal_marker("mc08", "joiner_ready")
        done_marker = _mc_signal_marker("mc08", "done")

        if MULTINODE_MODE == "worker":
            # Worker: wait for primary to signal joiner launch, then
            # spawn the joiner subprocess and park until completion.
            # If the primary aborts before signaling (e.g. shrink
            # itself hung), the done marker appears first and we exit
            # cleanly so both nodes tear down together.
            if not self._wait_for_marker(
                launch_joiner_marker,
                timeout_s=600,
                abort_marker=done_marker,
            ):
                return  # primary aborted; nothing to spawn

            join_tp = MC08_REGROW_TARGET - MC08_SHRINK_TARGET
            rank_offset = MC08_SHRINK_TARGET  # First retired global rank.
            joiner = self._launch_mc08_recover_joiner(
                rank_offset=rank_offset, join_tp=join_tp
            )
            type(self)._joiner_proc = joiner

            os.makedirs(os.path.dirname(joiner_ready_marker), exist_ok=True)
            with open(joiner_ready_marker, "w") as fh:
                fh.write(
                    f"joiner pid={joiner.pid} rank_offset={rank_offset} "
                    f"join_tp={join_tp}\n"
                )

            # Wait for primary to finish (grow-back + parity check).
            # Break early if the joiner subprocess dies -- the primary
            # will hang otherwise and hit its own scale-poll timeout.
            deadline = time.time() + 900
            while time.time() < deadline:
                if os.path.exists(done_marker):
                    return
                if joiner.poll() is not None:
                    self.fail(
                        f"worker: joiner subprocess exited early with "
                        f"code {joiner.returncode}; see joiner log"
                    )
                time.sleep(2)
            self.fail(
                f"worker: timed out waiting for primary done marker "
                f"{done_marker}"
            )
            return

        # Primary path
        try:
            self._generate_ok("pre-shrink")
            pre_score = self._run_gsm8k(f"pre-shrink {MC08_LAUNCH_EP}-rank")

            self._scale_to(
                old_ep_size=MC08_LAUNCH_EP,
                target_ep_size=MC08_SHRINK_TARGET,
            )
            self._generate_ok("post-shrink")
            mid_score = self._run_gsm8k(
                f"post-shrink {MC08_SHRINK_TARGET}-rank"
            )

            # Tell the worker to spawn the recover joiner. Includes
            # target coordinates so a stale marker from a previous run
            # doesn't silently mis-target.
            os.makedirs(
                os.path.dirname(launch_joiner_marker), exist_ok=True
            )
            with open(launch_joiner_marker, "w") as fh:
                fh.write(
                    f"rank_offset={MC08_SHRINK_TARGET} "
                    f"join_tp={MC08_REGROW_TARGET - MC08_SHRINK_TARGET}\n"
                )

            self._wait_for_marker(joiner_ready_marker, timeout_s=300)

            self._scale_to(
                old_ep_size=MC08_SHRINK_TARGET,
                target_ep_size=MC08_REGROW_TARGET,
            )
            # Post-regrow criterion: every DP slot in the restored
            # cohort serves MC08_POST_REGROW_PROBES_PER_SLOT sequential
            # /generate calls. Concurrent-load parity is intentionally
            # out of scope for the regrow path.
            probes = MC08_POST_REGROW_PROBES_PER_SLOT
            for dp_rank in range(MC08_REGROW_TARGET):
                for i in range(probes):
                    self._generate_ok(
                        f"post-regrow slot={dp_rank} probe={i + 1}/{probes}",
                        routed_dp_rank=dp_rank,
                    )
            print(
                f"[TEST] MC08 sequential probe parity: pre={pre_score:.2%} "
                f"mid={mid_score:.2%} "
                f"post-regrow slots [0, {MC08_REGROW_TARGET}) each served "
                f"{probes} sequential /generate requests OK "
                f"(recovered slots: "
                f"[{MC08_SHRINK_TARGET}, {MC08_REGROW_TARGET}))"
            )
        finally:
            # Always drop the done marker so the worker exits pytest
            # even on primary-side failure -- otherwise the worker
            # blocks on its 900s poll and Slurm eventually SIGTERMs
            # the whole step, obscuring the real failure.
            os.makedirs(os.path.dirname(done_marker), exist_ok=True)
            with open(done_marker, "w") as fh:
                fh.write("done\n")


# MC09: consecutive MC08 cycles in one program (8 -> 4 -> 8 -> 4 -> 8).
# Reuses every MC08 topology knob (MC08_LAUNCH_EP / MC08_SHRINK_TARGET /
# MC08_REGROW_TARGET / MC08_MAX_EP / MC08_POST_REGROW_PROBES_PER_SLOT)
# so the two cycles are byte-identical shrink+regrow steps and any
# divergence is attributable to cycle-1 residue, not to different
# topology or accept-criterion tuning.
MC09_NUM_CYCLES = int(os.environ.get("SGLANG_MC09_NUM_CYCLES", "2"))


@unittest.skipUnless(
    MULTINODE_MODE in ("primary", "worker"),
    "MC09 multi-node double shrink-regrow only runs when SGLANG_MC_MN_ROLE "
    "is set (via scripts/_run_mc09_multinode.sh).",
)
class TestMooncakeScaleDown8To4To8To4To8MultiNodeNixl(
    TestMooncakeScaleDown8To4To6MultiNodeNixl
):
    """MC09: 8 -> 4 -> 8 -> 4 -> 8 double shrink-then-regrow on NIXL a2a.

    Runs MC08's shrink+regrow cycle twice back-to-back in one server
    process. Cycle 2 should be topologically identical to a fresh
    8-rank launch; a regression vs MC08 indicates retire/recover
    residue leaking across the grow-back boundary. Each phase is
    tagged (``cycle{N} pre-shrink`` etc.) for localization.

    Worker spawns a fresh joiner per cycle: after cycle-1's joiner
    ranks retire, its HTTP wrapper still holds the worker GPUs, so
    kill_process_tree + short sleep are required before cycle 2."""

    NUM_CYCLES = MC09_NUM_CYCLES

    def _mc09_kill_joiner(self, joiner: subprocess.Popen | None) -> None:
        """Kill the previous cycle's idle joiner and pause so the CUDA
        driver releases the worker GPUs before the next cycle."""
        if joiner is None:
            return
        try:
            kill_process_tree(joiner.pid)
        except Exception:
            pass
        try:
            joiner.wait(timeout=15)
        except Exception:
            pass
        time.sleep(5)

    def test_shrink_then_regrow_across_nodes(self):
        done_marker = _mc_signal_marker("mc09", "done")

        if MULTINODE_MODE == "worker":
            current_joiner: subprocess.Popen | None = None
            for cycle_idx in range(self.NUM_CYCLES):
                cycle = cycle_idx + 1
                launch_marker = _mc_signal_marker("mc09", "launch_joiner", cycle)
                ready_marker = _mc_signal_marker("mc09", "joiner_ready", cycle)

                if not self._wait_for_marker(
                    launch_marker, timeout_s=900, abort_marker=done_marker
                ):
                    return  # primary aborted mid-test

                # Tear down the previous cycle's idle joiner (its
                # scheduler ranks were just retired by the shrink that
                # triggered this launch marker; only the HTTP wrapper
                # remains alive holding the worker GPUs).
                self._mc09_kill_joiner(current_joiner)
                current_joiner = None
                type(self)._joiner_proc = None

                join_tp = MC08_REGROW_TARGET - MC08_SHRINK_TARGET
                rank_offset = MC08_SHRINK_TARGET
                current_joiner = self._launch_mc08_recover_joiner(
                    rank_offset=rank_offset, join_tp=join_tp
                )
                type(self)._joiner_proc = current_joiner
                print(
                    f"[TEST] MC09 worker cycle {cycle}/{self.NUM_CYCLES} "
                    f"spawned joiner pid={current_joiner.pid}"
                )

                os.makedirs(os.path.dirname(ready_marker), exist_ok=True)
                with open(ready_marker, "w") as fh:
                    fh.write(
                        f"joiner pid={current_joiner.pid} cycle={cycle} "
                        f"rank_offset={rank_offset} join_tp={join_tp}\n"
                    )

                # Park until the next cycle's launch marker or done.
                # Do NOT poll current_joiner.poll(): the primary's next
                # shrink legitimately retires this joiner with rc=3
                # before writing the next marker. Real joiner crashes
                # surface as a hung primary or an early `done` marker.
                next_launch_marker = (
                    _mc_signal_marker("mc09", "launch_joiner", cycle + 1)
                    if cycle_idx + 1 < self.NUM_CYCLES
                    else None
                )
                deadline = time.time() + 1200
                advanced = False
                while time.time() < deadline:
                    if os.path.exists(done_marker):
                        return
                    if (
                        next_launch_marker is not None
                        and os.path.exists(next_launch_marker)
                    ):
                        advanced = True
                        break
                    time.sleep(2)
                if not advanced and next_launch_marker is not None:
                    self.fail(
                        f"MC09 worker cycle {cycle}: timed out waiting for "
                        f"next-cycle launch marker {next_launch_marker}"
                    )

            # All cycles serviced. Park on the done marker so the
            # primary's final tear-down window (post-regrow probes)
            # gets its full budget.
            deadline = time.time() + 1200
            while time.time() < deadline:
                if os.path.exists(done_marker):
                    return
                if (
                    current_joiner is not None
                    and current_joiner.poll() is not None
                ):
                    self.fail(
                        "MC09 worker: final-cycle joiner exited early "
                        f"(rc={current_joiner.returncode})"
                    )
                time.sleep(2)
            self.fail(
                f"MC09 worker: timed out waiting for done marker {done_marker}"
            )
            return

        # ---- Primary path ----
        try:
            self._generate_ok("pre-test")
            pre_score = self._run_gsm8k(f"pre-test {MC08_LAUNCH_EP}-rank")
            cycle_mid_scores: list[float] = []

            for cycle_idx in range(self.NUM_CYCLES):
                cycle = cycle_idx + 1
                launch_marker = _mc_signal_marker("mc09", "launch_joiner", cycle)
                ready_marker = _mc_signal_marker("mc09", "joiner_ready", cycle)

                # Shrink 8 -> 4 (retires ranks 4..7). Cycle 1: the
                # retirees are the original worker schedulers; cycle 2:
                # they are cycle-1's joiner schedulers.
                self._scale_to(
                    old_ep_size=MC08_LAUNCH_EP,
                    target_ep_size=MC08_SHRINK_TARGET,
                )
                self._generate_ok(f"cycle{cycle} post-shrink")
                mid_score = self._run_gsm8k(
                    f"cycle{cycle} post-shrink {MC08_SHRINK_TARGET}-rank"
                )
                cycle_mid_scores.append(mid_score)

                # Signal worker to (kill old joiner if any and) spawn
                # this cycle's recover joiner. Include coordinates so
                # a stale marker from a prior aborted run can't
                # silently mis-target.
                os.makedirs(os.path.dirname(launch_marker), exist_ok=True)
                with open(launch_marker, "w") as fh:
                    fh.write(
                        f"cycle={cycle} rank_offset={MC08_SHRINK_TARGET} "
                        f"join_tp={MC08_REGROW_TARGET - MC08_SHRINK_TARGET}\n"
                    )

                self._wait_for_marker(ready_marker, timeout_s=300)

                # Regrow 4 -> 8. Recover-mode: survivors' try_recover_
                # ranks pairs with joiner's join_process_groups.
                self._scale_to(
                    old_ep_size=MC08_SHRINK_TARGET,
                    target_ep_size=MC08_REGROW_TARGET,
                )
                probes = MC08_POST_REGROW_PROBES_PER_SLOT
                for dp_rank in range(MC08_REGROW_TARGET):
                    for i in range(probes):
                        self._generate_ok(
                            f"cycle{cycle} post-regrow slot={dp_rank} "
                            f"probe={i + 1}/{probes}",
                            routed_dp_rank=dp_rank,
                        )
                print(
                    f"[TEST] MC09 cycle {cycle}/{self.NUM_CYCLES} PASS: "
                    f"post-shrink={mid_score:.2%}, "
                    f"post-regrow {probes} probes/slot OK for slots "
                    f"[0, {MC08_REGROW_TARGET})"
                )

            print(
                f"[TEST] MC09 all {self.NUM_CYCLES} cycles PASS: "
                f"pre={pre_score:.2%}, "
                f"per-cycle post-shrink={cycle_mid_scores}"
            )
        finally:
            os.makedirs(os.path.dirname(done_marker), exist_ok=True)
            with open(done_marker, "w") as fh:
                fh.write("done\n")


@unittest.skipUnless(
    _count_visible_gpus() >= LAUNCH_EP_SIZE,
    f"MC04 chained shrink E2E needs {LAUNCH_EP_SIZE} GPUs.",
)
class TestMooncakeScaleDown4To3To2(_MooncakeShrinkEndToEndBase):
    """MC04: 4 -> 3 -> 2 chained shrink.

    Exercises ElasticEPStateManager reset between shrinks and the
    survivor's _finalize_scale_down starting from an already-shrunk
    cohort. 72 redundant experts = 36 local per rank so 2-rank cohort
    still covers all 72 logical experts."""

    EP_NUM_REDUNDANT_EXPERTS = 72
    STAGE2_REL_TOL = 0.15  # 2-rank cohort has minimal EPLB slack

    def test_two_consecutive_shrinks(self):
        self._generate_ok("pre-shrink")
        pre_score = self._run_gsm8k("pre-shrink 4-rank")

        self._scale_to(old_ep_size=4, target_ep_size=3)
        self._generate_ok("after-first-shrink")
        mid_score = self._run_gsm8k("post-first-shrink 3-rank")
        rel_mid = (pre_score - mid_score) / max(pre_score, 1e-9)
        print(
            f"[TEST] MC04 stage1 parity: pre={pre_score:.2%} "
            f"mid={mid_score:.2%} rel_delta={rel_mid:.2%} "
            f"tol={GSM8K_REL_TOL:.2%}"
        )
        self.assertLess(
            rel_mid,
            GSM8K_REL_TOL,
            f"MC04 3-rank stage regressed more than {GSM8K_REL_TOL:.0%}: "
            f"pre={pre_score:.2%} mid={mid_score:.2%}",
        )

        self._scale_to(old_ep_size=3, target_ep_size=2)
        self._generate_ok("after-second-shrink")
        post_score = self._run_gsm8k("post-second-shrink 2-rank")
        rel_post = (pre_score - post_score) / max(pre_score, 1e-9)
        print(
            f"[TEST] MC04 stage2 parity: pre={pre_score:.2%} "
            f"post={post_score:.2%} rel_delta={rel_post:.2%} "
            f"tol={self.STAGE2_REL_TOL:.2%}"
        )
        self.assertLess(
            rel_post,
            self.STAGE2_REL_TOL,
            f"MC04 2-rank stage regressed more than "
            f"{self.STAGE2_REL_TOL:.0%}: pre={pre_score:.2%} "
            f"post={post_score:.2%}",
        )
        self.assertGreater(
            post_score,
            0.30,
            f"MC04 post-chained-shrink accuracy too low: {post_score:.2%}",
        )


@unittest.skipUnless(
    _count_visible_gpus() >= LAUNCH_EP_SIZE,
    f"MC06 post-shrink soak needs {LAUNCH_EP_SIZE} GPUs.",
)
class TestMooncakeScaleDown4To3Soak(_MooncakeShrinkEndToEndBase):
    """MC06: 4->3 shrink followed by three sustained inference rounds.

    Runs three back-to-back GSM8K rounds after shrink with brief idle
    windows to catch issues that only surface after the shrunk cohort
    has served for minutes (slow KV leaks, EPLB reweight confusion,
    scheduler-timer stalls).

    Uses --enable-deterministic-inference + fixed --random-seed +
    num_threads=1 + KV flush between rounds. Primary assertion is
    exact equality across the three post-shrink scores; a loose
    (GSM8K_DETERMINISTIC_SAFETY_MARGIN) wholesale-corruption safety
    net catches reproducible-but-wrong cohorts."""

    DETERMINISTIC = True

    def test_soak_after_shrink(self):
        self._generate_ok("pre-shrink")
        self._flush_kv_cache()
        pre_score = self._run_gsm8k(
            "pre-shrink 4-rank",
            num_threads=1,
            num_examples=GSM8K_NUM_EXAMPLES_DETERMINISTIC,
        )

        self._scale_to(old_ep_size=4, target_ep_size=3)
        self._generate_ok("post-shrink initial ping")
        self._assert_no_orphan_processes(retired_slots=1)

        post_scores: list[float] = []
        for i in range(3):
            self._flush_kv_cache()
            score = self._run_gsm8k(
                f"post-shrink round {i+1}/3",
                num_threads=1,
                num_examples=GSM8K_NUM_EXAMPLES_DETERMINISTIC,
            )
            post_scores.append(score)
            if i < 2:
                # Idle window between rounds -- catches bugs that only
                # surface when the scheduler / EPLB timer sees a warm
                # -> cold -> warm transition after a scale event.
                time.sleep(30)

        print(
            f"[TEST] MC06 pre={pre_score:.4%} "
            f"post_scores={[f'{s:.4%}' for s in post_scores]}"
        )

        # Primary signal: byte-identical scores across the three
        # deterministic post-shrink rounds.
        unique_scores = sorted(set(post_scores))
        self.assertEqual(
            len(unique_scores),
            1,
            "MC06 post-shrink GSM8K is non-deterministic across "
            f"rounds under --enable-deterministic-inference: "
            f"scores={post_scores}. Expected byte-identical scores; "
            f"got {len(unique_scores)} distinct values "
            f"({unique_scores}). This indicates either a genuine "
            "batch-order-dependent correctness bug in the shrunk "
            "cohort, or upstream's deterministic-inference guarantee "
            "is no longer holding for our (backend, model, kernel) "
            "combination.",
        )

        # Wholesale-corruption safety net (loose floor).
        deterministic_score = post_scores[0]
        rel_delta = (pre_score - deterministic_score) / max(pre_score, 1e-9)
        print(
            f"[TEST] MC06 safety-margin check: pre={pre_score:.2%} "
            f"post={deterministic_score:.2%} rel_delta={rel_delta:.2%} "
            f"margin={GSM8K_DETERMINISTIC_SAFETY_MARGIN:.0%}"
        )
        self.assertLess(
            rel_delta,
            GSM8K_DETERMINISTIC_SAFETY_MARGIN,
            "MC06 post-shrink accuracy collapsed vs pre-shrink: "
            f"pre={pre_score:.2%} post={deterministic_score:.2%} "
            f"rel_delta={rel_delta:.2%} "
            f"(safety margin={GSM8K_DETERMINISTIC_SAFETY_MARGIN:.0%}). "
            "Byte-identical across rounds but categorically worse "
            "than pre-shrink indicates a reproducible-but-corrupt "
            "shrunk cohort.",
        )
        self.assertGreater(
            deterministic_score,
            GSM8K_MIN_SCORE,
            f"MC06 post-shrink accuracy too low: {deterministic_score:.2%}",
        )


@unittest.skipUnless(
    _count_visible_gpus() >= LAUNCH_EP_SIZE,
    f"MC07 concurrent-traffic shrink needs {LAUNCH_EP_SIZE} GPUs.",
)
class TestMooncakeScaleDown4To3ConcurrentTraffic(_MooncakeShrinkEndToEndBase):
    """MC07: 4->3 shrink under concurrent client traffic.

    Pumps N concurrent /generate streams before, during, and after the
    shrink event. Requests are bucketed by client start_ts into
    pre-shrink / transition / post-shrink windows. All three must be
    100% clean; a leak in any window indicates the scheduler admission
    gate (see Scheduler._elastic_scale_down_in_transition) let a stale
    batch through and would trip the NIXL device-side expert-bound
    assertion at nixl_ep_ll.cu:178.

    Also asserts post-shrink median latency stays within a bounded
    multiple of pre-shrink median."""

    # 8 concurrent streams each hitting /generate every 200ms -> ~40
    # requests/s sustained. Enough to keep every rank busy without
    # saturating the Mooncake in-flight token budget (1024 per rank).
    NUM_WORKERS = 8
    REQUEST_INTERVAL_S = 0.2
    # ``WARMUP_S`` is the discard window at the start of the pumped load
    # Warmup discard window (cold-start jitter) then a steady-state
    # window that we assert on.
    WARMUP_S = 20.0
    PRE_SHRINK_ASSERT_S = 10.0
    POST_SHRINK_WORKLOAD_S = 30.0
    REQUEST_TIMEOUT_S = 120.0  # > any observed drain-barrier window
    LATENCY_REGRESSION_TOL = 5.0  # post/pre median cap
    # Default 0.0 = zero-tolerance for transition-window failures with
    # admission gate closed. Subclasses may widen for infra flakes.
    TRANSITION_FAILURE_TOL_FRAC = 0.0
    SCHEDULER_CRASH_ERR_SUBSTRINGS = (
        "Connection refused",
        "ConnectionResetError",
        "RemoteDisconnected",
    )

    def test_shrink_under_load(self):
        # Prime with a small concurrent burst before starting the
        # timed pump; a single small /generate is not enough to warm
        # the KV allocator / CUDA graph replay.
        prime_ok = 0
        prime_errs: list[str] = []
        prime_threads: list[threading.Thread] = []
        prime_lock = threading.Lock()

        def _prime() -> None:
            nonlocal prime_ok
            try:
                resp = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": "prime",
                        "sampling_params": {
                            "max_new_tokens": 8,
                            "temperature": 0.0,
                        },
                    },
                    timeout=180.0,
                )
                if resp.status_code == 200:
                    with prime_lock:
                        prime_ok += 1
                else:
                    with prime_lock:
                        prime_errs.append(f"HTTP {resp.status_code}")
            except Exception as exc:  # pragma: no cover - runtime path
                with prime_lock:
                    prime_errs.append(f"{type(exc).__name__}: {exc!r}")

        for _ in range(self.NUM_WORKERS * 2):
            t = threading.Thread(target=_prime, daemon=True)
            t.start()
            prime_threads.append(t)
        for t in prime_threads:
            t.join(timeout=300.0)
        self.assertGreaterEqual(
            prime_ok,
            self.NUM_WORKERS * 2 - 2,
            f"MC07 warm-priming failed: ok={prime_ok} errs={prime_errs[:5]}",
        )

        stop_event = threading.Event()
        # Each entry is (start_ts, ok, latency_or_None, err_or_None).
        results: list[tuple[float, bool, float | None, str | None]] = []
        lock = threading.Lock()

        def _pump(worker_id: int) -> None:
            while not stop_event.is_set():
                start_ts = time.perf_counter()
                try:
                    resp = requests.post(
                        f"{self.base_url}/generate",
                        json={
                            "text": f"Q from worker {worker_id}:",
                            "sampling_params": {
                                "max_new_tokens": 8,
                                "temperature": 0.0,
                            },
                        },
                        timeout=self.REQUEST_TIMEOUT_S,
                    )
                    dt = time.perf_counter() - start_ts
                    if resp.status_code == 200:
                        with lock:
                            results.append((start_ts, True, dt, None))
                    else:
                        err = f"HTTP {resp.status_code}: {resp.text[:100]}"
                        with lock:
                            results.append((start_ts, False, None, err))
                except Exception as exc:  # pragma: no cover - runtime path
                    err = f"{type(exc).__name__}: {exc!r}"
                    with lock:
                        results.append((start_ts, False, None, err))
                time.sleep(self.REQUEST_INTERVAL_S)

        threads = [
            threading.Thread(target=_pump, args=(i,), daemon=True)
            for i in range(self.NUM_WORKERS)
        ]
        pump_start_ts = time.perf_counter()
        for t in threads:
            t.start()

        # Warmup window: pumped requests started here are dropped from
        # the counting buckets. The subsequent PRE_SHRINK_ASSERT_S
        # window is the steady-state slice we DO assert on.
        time.sleep(self.WARMUP_S)
        pre_assert_start_ts = time.perf_counter()
        time.sleep(self.PRE_SHRINK_ASSERT_S)

        shrink_start_ts = time.perf_counter()
        self._scale_to(old_ep_size=4, target_ep_size=3)
        shrink_end_ts = time.perf_counter()

        time.sleep(self.POST_SHRINK_WORKLOAD_S)
        stop_event.set()
        for t in threads:
            t.join(timeout=10)

        with lock:
            snapshot = list(results)

        # Bucket by start timestamp. Requests started before
        # pre_assert_start_ts are still "warming up" (cold graph,
        # KV cache allocation) and are discarded.
        pre_bucket: list[tuple[bool, float | None, str | None]] = []
        transition_bucket: list[tuple[bool, float | None, str | None]] = []
        post_bucket: list[tuple[bool, float | None, str | None]] = []
        warmup_dropped = 0
        for ts, ok, latency, err in snapshot:
            entry = (ok, latency, err)
            if ts < pre_assert_start_ts:
                warmup_dropped += 1
                continue
            if ts < shrink_start_ts:
                pre_bucket.append(entry)
            elif ts < shrink_end_ts:
                transition_bucket.append(entry)
            else:
                post_bucket.append(entry)
        print(f"[TEST] MC07 warmup dropped {warmup_dropped} req(s)")

        def _summary(name: str, bucket) -> tuple[int, int, list[str]]:
            succ = sum(1 for ok, _, _ in bucket if ok)
            fail = len(bucket) - succ
            errs = [err for ok, _, err in bucket if not ok and err][:5]
            print(
                f"[TEST] MC07 {name:>10s}: success={succ} failure={fail}"
            )
            return succ, fail, errs

        pre_succ, pre_fail, pre_errs = _summary("pre-shrink", pre_bucket)
        trans_succ, trans_fail, trans_errs = _summary(
            "transition", transition_bucket
        )
        post_succ, post_fail, post_errs = _summary("post-shrink", post_bucket)

        # Scheduler-crash sentinel: fail-fast on client-side errors
        # indicating a scheduler process died mid-test. Zero-tolerance:
        # even one crash-signature is a regression of the admission gate.
        crash_hits: list[tuple[str, str]] = []
        for bucket_name, bucket in (
            ("pre-shrink", pre_bucket),
            ("transition", transition_bucket),
            ("post-shrink", post_bucket),
        ):
            for ok, _, err in bucket:
                if ok or not err:
                    continue
                if any(
                    marker in err
                    for marker in self.SCHEDULER_CRASH_ERR_SUBSTRINGS
                ):
                    crash_hits.append((bucket_name, err[:200]))
        self.assertEqual(
            crash_hits,
            [],
            "MC07 detected scheduler crash(es) mid-test -- the "
            "elastic-EP admission gate likely failed to close before "
            "NIXL's dispatch_ep dropped (regression of the MC07 fix "
            "at nixl_ep_ll.cu:178). Sample hits: "
            f"{crash_hits[:5]}",
        )

        self.assertEqual(
            pre_fail, 0,
            f"MC07 pre-shrink saw {pre_fail} failure(s): {pre_errs}",
        )
        self.assertEqual(
            post_fail, 0,
            f"MC07 post-shrink saw {post_fail} failure(s) after "
            f"serving_shrunk (drain complete): {post_errs}",
        )
        # Transition window: default TRANSITION_FAILURE_TOL_FRAC=0 gives
        # zero-tolerance (assert on absolute count, not fraction, since
        # assertLess(0, 0.0) is unsatisfiable). Subclasses can widen.
        trans_total = trans_succ + trans_fail
        if trans_total > 0:
            trans_frac = trans_fail / trans_total
            print(
                f"[TEST] MC07 transition failure frac: "
                f"{trans_frac:.2%} (tol={self.TRANSITION_FAILURE_TOL_FRAC:.0%})"
            )
            if self.TRANSITION_FAILURE_TOL_FRAC <= 0.0:
                self.assertEqual(
                    trans_fail,
                    0,
                    f"MC07 transition window saw {trans_fail} failure(s) "
                    f"out of {trans_total} (zero-tolerance mode): "
                    f"{trans_errs}",
                )
            else:
                self.assertLess(
                    trans_frac,
                    self.TRANSITION_FAILURE_TOL_FRAC,
                    f"MC07 transition window failure rate {trans_frac:.2%} "
                    f"exceeds tol {self.TRANSITION_FAILURE_TOL_FRAC:.0%}: "
                    f"{trans_errs}",
                )
        self.assertGreater(
            post_succ, 0,
            "MC07 saw no successful post-shrink requests",
        )

        pre_lat = [
            l for ok, l, _ in pre_bucket if ok and l is not None
        ]
        post_lat = [
            l for ok, l, _ in post_bucket if ok and l is not None
        ]
        if len(pre_lat) >= 10 and len(post_lat) >= 10:
            pre_median = statistics.median(pre_lat)
            post_median = statistics.median(post_lat)
            print(
                f"[TEST] MC07 median latency: "
                f"pre={pre_median*1000:.0f}ms "
                f"post={post_median*1000:.0f}ms "
                f"tol={self.LATENCY_REGRESSION_TOL:.1f}x"
            )
            self.assertLess(
                post_median,
                pre_median * self.LATENCY_REGRESSION_TOL,
                f"MC07 latency degraded > "
                f"{self.LATENCY_REGRESSION_TOL:.1f}x after shrink: "
                f"pre={pre_median*1000:.0f}ms "
                f"post={post_median*1000:.0f}ms",
            )


# ---------------------------------------------------------------------------
# NIXL a2a variants of the shrink-only shrink matrix (MC01 / MC04 / MC06 /
# MC07). Each variant only overrides ``MOE_A2A_BACKEND`` so the shrink
# topology + parity assertions stay identical to the mooncake-a2a baseline;
# both a2a data planes must hold the same GSM8K parity + soak invariants.
# MC02A (:class:`TestMooncakeScaleDownNixlShrink`) and MC05P
# (:class:`TestMooncakeScaleDown8To6MultiNodeNixl`) already exist as their
# own dedicated NIXL variants.
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    _count_visible_gpus() >= LAUNCH_EP_SIZE,
    f"MC01 (NIXL a2a) shrink E2E needs {LAUNCH_EP_SIZE} GPUs.",
)
class TestMooncakeScaleDown4To3Nixl(TestMooncakeScaleDown4To3):
    """MC01 (NIXL a2a): 4 -> 3 shrink with pre/post GSM8K parity check.

    Same as :class:`TestMooncakeScaleDown4To3` but with ``--moe-a2a-backend
    nixl``. Guards the a2a-backend-agnostic shrink correctness invariant
    on the shortest test in the matrix.
    """

    MOE_A2A_BACKEND = "nixl"


@unittest.skipUnless(
    _count_visible_gpus() >= LAUNCH_EP_SIZE,
    f"MC04 (NIXL a2a) chained shrink E2E needs {LAUNCH_EP_SIZE} GPUs.",
)
class TestMooncakeScaleDown4To3To2Nixl(TestMooncakeScaleDown4To3To2):
    """MC04 (NIXL a2a): 4 -> 3 -> 2 chained shrink."""

    MOE_A2A_BACKEND = "nixl"


MC12_LAUNCH_EP = 6


@unittest.skipUnless(
    _count_visible_gpus() >= MC12_LAUNCH_EP,
    f"MC12 3-step chained shrink E2E needs {MC12_LAUNCH_EP} GPUs.",
)
class TestMooncakeScaleDown6To5To4To3(_MooncakeShrinkEndToEndBase):
    """MC12: 6 -> 5 -> 4 -> 3 3-step chained shrink.

    Exercises the NIXL retire barrier's epoch derivation across the
    cycle-3 arrival-bucket boundary. The pre-fix cumulative formula
    ``epoch = (arrival - 1) // world_size + 1`` would split cycle 3's
    posters across two epochs (world_size=launch_ep=6), leaving one
    subgroup racing alone into on_flip_mask and stalling the rest on
    a stale ready_key.

    LAUNCH_EP = 6 (not 8): NIXL EP asserts num_ranks < 8 or
    num_ranks % 8 == 0 at every cohort size, and 8-launch would need
    max_ep_size=16.

    Cycle 1 / 2 assert GSM8K parity (GSM8K_REL_TOL / STAGE2_REL_TOL);
    cycle 3 only requires /generate = 200 (post-fix survival signal)."""

    LAUNCH_EP = MC12_LAUNCH_EP
    # MAX_EP > LAUNCH_EP forces the elastic-EP code path so
    # elastic_ep_initial_size is auto-populated; without it the
    # non-elastic branch enforces num_physical_experts % ep_size == 0
    # which breaks on non-divisor target sizes.
    MAX_EP = MC12_LAUNCH_EP + 1
    EP_NUM_REDUNDANT_EXPERTS = 72
    STAGE2_REL_TOL = 0.15

    def test_three_consecutive_shrinks(self):
        self._generate_ok("pre-shrink")
        pre_score = self._run_gsm8k("pre-shrink 6-rank")

        self._scale_to(old_ep_size=6, target_ep_size=5)
        self._generate_ok("after-first-shrink")
        mid_score = self._run_gsm8k("post-first-shrink 5-rank")
        rel_mid = (pre_score - mid_score) / max(pre_score, 1e-9)
        print(
            f"[TEST] MC12 stage1 parity: pre={pre_score:.2%} "
            f"mid={mid_score:.2%} rel_delta={rel_mid:.2%} "
            f"tol={GSM8K_REL_TOL:.2%}"
        )
        self.assertLess(
            rel_mid,
            GSM8K_REL_TOL,
            f"MC12 5-rank stage regressed more than {GSM8K_REL_TOL:.0%}: "
            f"pre={pre_score:.2%} mid={mid_score:.2%}",
        )

        self._scale_to(old_ep_size=5, target_ep_size=4)
        self._generate_ok("after-second-shrink")
        stage2_score = self._run_gsm8k("post-second-shrink 4-rank")
        rel_stage2 = (pre_score - stage2_score) / max(pre_score, 1e-9)
        print(
            f"[TEST] MC12 stage2 parity: pre={pre_score:.2%} "
            f"post={stage2_score:.2%} rel_delta={rel_stage2:.2%} "
            f"tol={self.STAGE2_REL_TOL:.2%}"
        )
        self.assertLess(
            rel_stage2,
            self.STAGE2_REL_TOL,
            f"MC12 4-rank stage regressed more than "
            f"{self.STAGE2_REL_TOL:.0%}: pre={pre_score:.2%} "
            f"post={stage2_score:.2%}",
        )

        self._scale_to(old_ep_size=4, target_ep_size=3)
        self._generate_ok("after-third-shrink (Bug-A cycle)")
        # If Bug A regresses, the ``_scale_to`` HTTP call above
        # times out before we reach this line; if it landed and
        # ``/generate`` is answering, the cohort is intact.
        print("[TEST] MC12 stage3 survived cycle-3 chained shrink (Bug A path)")


@unittest.skipUnless(
    _count_visible_gpus() >= MC12_LAUNCH_EP,
    f"MC12 (NIXL a2a) 3-step chained shrink E2E needs {MC12_LAUNCH_EP} GPUs.",
)
class TestMooncakeScaleDown6To5To4To3Nixl(TestMooncakeScaleDown6To5To4To3):
    """MC12 (NIXL a2a): 6 -> 5 -> 4 -> 3 3-step chained shrink."""

    MOE_A2A_BACKEND = "nixl"


@unittest.skipUnless(
    _count_visible_gpus() >= LAUNCH_EP_SIZE,
    f"MC06 (NIXL a2a) post-shrink soak needs {LAUNCH_EP_SIZE} GPUs.",
)
class TestMooncakeScaleDown4To3SoakNixl(TestMooncakeScaleDown4To3Soak):
    """MC06 (NIXL a2a): 4->3 shrink + three sustained GSM8K rounds."""

    MOE_A2A_BACKEND = "nixl"


@unittest.skipUnless(
    _count_visible_gpus() >= LAUNCH_EP_SIZE,
    f"MC07 (NIXL a2a) concurrent-traffic shrink needs {LAUNCH_EP_SIZE} GPUs.",
)
class TestMooncakeScaleDown4To3ConcurrentTrafficNixl(
    TestMooncakeScaleDown4To3ConcurrentTraffic
):
    """MC07 (NIXL a2a): 4->3 shrink under concurrent client traffic."""

    MOE_A2A_BACKEND = "nixl"


# ---------------------------------------------------------------------------
# MC03 family: shrink baseline + grow-back + round-trip
#
# These tests all use the v1 recover path (``/scale_elastic_ep`` +
# ``--elastic-ep-join-mode recover``) because the "always launch at
# ``max_ep_size`` + pre-shrink" design principle means every "grow"
# is topologically a re-join of a previously retired launch-cohort
# slot -- exactly what :func:`try_recover_ranks` handles. See
# :class:`TestMooncakeScaleDown8To4To6MultiNodeNixl` (MC08) for the
# multi-node validation of the same pattern.
# ---------------------------------------------------------------------------



@unittest.skipUnless(
    _count_visible_gpus() >= 5,
    "MC03A shrink 5->4 needs 5 GPUs.",
)
class TestMooncakeScaleDown5To4(_MooncakeShrinkEndToEndBase):
    """MC03A: fresh-launch ep=5, shrink to ep=4 via /scale_elastic_ep.

    Shrink-only counterpart to MC03B (recover-grow 4->5) and MC03
    (round-trip 5->4->5->4). Every rank is a launch-cohort member,
    so any crash isolates to the pure shrink path.

    EP_NUM_REDUNDANT_EXPERTS derived from
    _min_redundant_experts_for_shrink to satisfy EPLB layout
    divisibility at tp=5 (base=90 vs default 96 which fails)."""

    MOE_A2A_BACKEND = "nixl"
    LAUNCH_EP = 5
    TARGET_EP = 4
    MAX_EP = 6  # +1 headroom keeps retired slot in Mooncake pool
    EP_NUM_REDUNDANT_EXPERTS = _min_redundant_experts_for_shrink(
        launch_ep=LAUNCH_EP, min_target_ep=TARGET_EP
    )

    def test_shrink_only(self):
        self._generate_ok("pre-shrink")
        self._scale_to(
            old_ep_size=self.LAUNCH_EP, target_ep_size=self.TARGET_EP
        )
        self._generate_ok("post-shrink")


@unittest.skipUnless(
    _count_visible_gpus() >= 5,
    "MC03B recover-mode grow 4->5 needs 5 GPUs.",
)
class TestMooncakeGrow4To5RecoverOnly(_MooncakeShrinkEndToEndBase):
    """MC03B: launch at 5, pre-shrink to 4, recover-grow back to 5.

    Grow-only companion to MC03A (shrink 5->4) and MC03 (round-trip).
    Exercises the "launch at max_ep_size, pre-shrink, then recover"
    pattern in isolation: retires rank 4, then a --tp 1 recover-mode
    joiner boots at rank_offset=4 and pairs with the survivor cohort's
    try_recover_ranks to flip active_ranks[4] back on."""

    MOE_A2A_BACKEND = "nixl"
    LAUNCH_EP = 5
    TARGET_EP = 5
    MAX_EP = 6  # +1 headroom keeps retired slot recoverable
    EP_NUM_REDUNDANT_EXPERTS = _min_redundant_experts_for_shrink(
        launch_ep=LAUNCH_EP, min_target_ep=4
    )

    def test_recover_grow(self):
        self._generate_ok("pre-shrink")
        self._scale_to(old_ep_size=self.LAUNCH_EP, target_ep_size=4)
        self._generate_ok("post-shrink 4-rank")

        joiner = self._launch_offset_joiner(
            rank_offset=4, join_tp=1, port=PORT_B, join_mode="recover",
        )
        try:
            self.assertIsNone(
                joiner.poll(),
                "MC03B recover joiner exited before scale request",
            )
            self._scale_to(old_ep_size=4, target_ep_size=self.LAUNCH_EP)
            self._generate_ok("post-regrow", routed_dp_rank=4)
        finally:
            try:
                kill_process_tree(joiner.pid)
                joiner.wait(timeout=10)
            except Exception:
                pass


@unittest.skipUnless(
    _count_visible_gpus() >= 5,
    "MC03 round-trip 5->4->5->4 needs 5 GPUs.",
)
class TestMooncakeScaleDown5To4To5To4(_MooncakeShrinkEndToEndBase):
    """MC03: round-trip 5 -> 4 -> 5 -> 4 on the recover path.

    Composes MC03A (shrink) and MC03B (recover-grow) with a final
    shrink that retires the just-recovered slot. Protects the
    invariant: "a slot recovered via recover_ranks can be retired
    again without stale Mooncake / NIXL survivor state"."""

    MOE_A2A_BACKEND = "nixl"
    LAUNCH_EP = 5
    MAX_EP = 6
    EP_NUM_REDUNDANT_EXPERTS = _min_redundant_experts_for_shrink(
        launch_ep=LAUNCH_EP, min_target_ep=4
    )

    def test_shrink_regrow_shrink(self):
        self._generate_ok("pre-shrink")
        self._scale_to(old_ep_size=self.LAUNCH_EP, target_ep_size=4)
        self._generate_ok("post-shrink 4-rank")

        joiner = self._launch_offset_joiner(
            rank_offset=4, join_tp=1, port=PORT_B, join_mode="recover",
        )
        try:
            self.assertIsNone(
                joiner.poll(),
                "MC03 recover joiner exited before regrow request",
            )
            self._scale_to(old_ep_size=4, target_ep_size=self.LAUNCH_EP)
            self._generate_ok("post-regrow 5-rank", routed_dp_rank=4)
            # Extra forward passes absorb first-use NIXL peer connect
            # and DeepGEMM JIT latency BEFORE the second shrink starts,
            # otherwise the retire path can race a still-warming NIXL
            # peer connect on the incumbent side (same rationale as
            # MC02B's f1/f2 probes).
            self._generate_ok("post-regrow f1")
            self._generate_ok("post-regrow f2")

            # Second shrink retires the slot that was JUST recovered
            # by the joiner. This is the invariant the round-trip test
            # protects.
            self._scale_to(old_ep_size=self.LAUNCH_EP, target_ep_size=4)
            self._generate_ok("post-second-shrink 4-rank")
            self._generate_ok("post-second-shrink f1")
            self._generate_ok("post-second-shrink f2")

            # The joiner subprocess should self-exit via
            # ``local_cleanup`` once its slot is marked retired. Give
            # it a short grace window then confirm.
            try:
                joiner.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            self.assertIsNotNone(
                joiner.poll(),
                "MC03 recover joiner did not exit after its slot was "
                "retired in the second shrink",
            )
        finally:
            try:
                if joiner.poll() is None:
                    kill_process_tree(joiner.pid)
                    joiner.wait(timeout=10)
            except Exception:
                pass


@unittest.skipUnless(
    _count_visible_gpus() >= 5,
    "MC14 shrink-recover-then-append needs 5 GPUs.",
)
class TestMooncakeScaleDown4To3To4To5(_MooncakeShrinkEndToEndBase):
    """MC14: 4 -> 3 -> 4 -> 5 shrink, recover-back, then scale-up-v1 append.

    Chains MC02 (recover round-trip) with MC02B (append) to verify a
    completed shrink+recover leaves elastic-EP state clean for a
    subsequent append. Ends with rank 4 as a WORLD-only append slot;
    does NOT retire it (that shape is MC13 and requires Mooncake >=
    0.3.12; see :class:`TestMooncakeScaleDown4To5To3`).

    Invariant: post-commit_scale state for recover is indistinguishable
    (from the scale-request handler's view) from a fresh boot at the
    same cohort size, so the next request can be an append."""

    MOE_A2A_BACKEND = "nixl"  # scale-up-v1 requires NIXL a2a (per PR #30164)
    MAX_EP = 5                # WORLD's P2PProxy must be >= 5-wide for the append

    def test_shrink_recover_then_append(self):
        self._generate_ok("pre-cycle 4-rank")

        # Half 1: shrink 4 -> 3 (MC01 shape).
        self._scale_to(old_ep_size=4, target_ep_size=3)
        self._generate_ok("post-shrink 3-rank")

        # Half 2: recover 3 -> 4 (MC02 shape).
        recover_joiner = self._launch_offset_joiner(
            rank_offset=3, join_tp=1, port=PORT_B, join_mode="recover",
        )
        append_joiner = None
        try:
            self.assertIsNone(
                recover_joiner.poll(),
                "MC14 recover joiner exited before the regrow request",
            )
            self._scale_to(old_ep_size=3, target_ep_size=4)
            self._generate_ok("post-recover 4-rank", routed_dp_rank=3)
            # Absorb first-use NIXL peer-connect + DeepGEMM JIT before
            # the second scale-up starts (same rationale as MC03/MC03B):
            # otherwise the append handshake can race a still-warming
            # NIXL peer connect on the incumbent side.
            self._generate_ok("post-recover f1")
            self._generate_ok("post-recover f2")

            # Half 3: scale-up-v1 append 4 -> 5 (MC02B shape).
            append_joiner = self._launch_offset_joiner(
                rank_offset=4, join_tp=1, port=PORT_C, join_mode="scale",
            )
            self.assertIsNone(
                append_joiner.poll(),
                "MC14 append joiner exited before the scale-up request",
            )
            self._scale_to(old_ep_size=4, target_ep_size=5)
            self._generate_ok("post-append 5-rank", routed_dp_rank=4)
            self._generate_ok("post-append f1")
            self._generate_ok("post-append f2")
            # Deliberately end here. Retiring rank 4 next would trip
            # the Mooncake C++ P2PProxy invariant on cpu_group /
            # device_group (see TestMooncakeScaleDown4To5To3 / MC13,
            # which is the skipped diagnostic for that crash class).
        finally:
            for j in (append_joiner, recover_joiner):
                if j is None:
                    continue
                try:
                    if j.poll() is None:
                        kill_process_tree(j.pid)
                        j.wait(timeout=10)
                except Exception:
                    pass


# MC10: multi-node partial-recovery. Extends MC08 with a second shrink
# down to 6 so the final cohort is a strict subset of the recovered
# cohort (retire ranks that were re-populated by try_recover_ranks).
# Two-step shape (not a direct 8->4->6 partial regrow) because
# Mooncake's join_group(WORLD) deadlocks on partial regrow.
MC10_LAUNCH_EP = int(os.environ.get("SGLANG_MC10_LAUNCH_EP", "8"))
MC10_SHRINK_TARGET = int(os.environ.get("SGLANG_MC10_SHRINK_TARGET", "4"))
# Full regrow target. Must equal ``MC10_LAUNCH_EP`` per the
# recover-mode joiner rendezvous constraint documented on
# :data:`MC08_REGROW_TARGET`.
MC10_REGROW_TARGET = int(
    os.environ.get("SGLANG_MC10_REGROW_TARGET", str(MC10_LAUNCH_EP))
)
# Final shrink target after the full regrow. Must satisfy
# ``SHRINK_TARGET < FINAL_SHRINK_TARGET < REGROW_TARGET`` so it retires
# strictly-newly-recovered ranks without collapsing back to the
# initial-shrink survivor set (which would make the second shrink
# equivalent to just skipping the regrow).
MC10_FINAL_SHRINK_TARGET = int(
    os.environ.get("SGLANG_MC10_FINAL_SHRINK_TARGET", "6")
)
# Elastic pool ceiling. Reuses MC08's reasoning: NIXL EP's per-peer
# memory allocator asserts ``num_ranks < NUM_MAX_NVL_PEERS or
# num_ranks % NUM_MAX_NVL_PEERS == 0`` with ``NUM_MAX_NVL_PEERS = 8``,
# so an 8-rank launch cohort needs the next legal ceiling 16.
MC10_MAX_EP = 16
# Post-final-shrink acceptance criterion: every DP slot in the FINAL
# cohort ([0, FINAL_SHRINK_TARGET)) must serve this many sequential
# /generate probes without error. Deliberately does NOT include slots
# [FINAL_SHRINK_TARGET, LAUNCH_EP) -- those are retired by the final
# shrink, so ``routed_dp_rank>=FINAL_SHRINK_TARGET`` would 500 with
# ``dp_size=FINAL_SHRINK_TARGET``.
MC10_POST_REGROW_PROBES_PER_SLOT = int(
    os.environ.get("SGLANG_MC10_POST_REGROW_PROBES_PER_SLOT", "10")
)


@unittest.skipUnless(
    MULTINODE_MODE in ("primary", "worker"),
    "MC10 multi-node partial-recovery only runs when SGLANG_MC_MN_ROLE "
    "is set (via scripts/_run_mc10_multinode.sh).",
)
class TestMooncakeScaleDown8To4To8To6MultiNodeNixl(
    TestMooncakeScaleDown8To4To6MultiNodeNixl
):
    """MC10: 8 -> 4 -> 8 -> 6 partial-recovery across two nodes on NIXL a2a.

    Extends MC08 (see :class:`TestMooncakeScaleDown8To4To6MultiNodeNixl`)
    with a final 8 -> 6 shrink that retires ranks 6, 7 which were just
    re-populated by the recover-mode grow-back. Covers the
    "retire ranks that were previously rejoined via recover_ranks"
    contract on multi-node NIXL a2a.

    Two-step (full-regrow then shrink) rather than a direct 8 -> 4 -> 6
    partial regrow because direct partial regrow deadlocks in
    Mooncake's C++ join_group(WORLD) (see :data:`MC08_REGROW_TARGET`)."""

    def test_shrink_regrow_shrink_across_nodes(self):
        launch_joiner_marker = _mc_signal_marker("mc10", "launch_joiner")
        joiner_ready_marker = _mc_signal_marker("mc10", "joiner_ready")
        done_marker = _mc_signal_marker("mc10", "done")

        if MULTINODE_MODE == "worker":
            if not self._wait_for_marker(
                launch_joiner_marker,
                timeout_s=600,
                abort_marker=done_marker,
            ):
                return

            # Recover joiner covers the ENTIRE retired range
            # [SHRINK_TARGET, LAUNCH_EP). The subsequent shrink to
            # FINAL_SHRINK_TARGET retires the top slots.
            join_tp = MC10_LAUNCH_EP - MC10_SHRINK_TARGET
            rank_offset = MC10_SHRINK_TARGET
            joiner = self._launch_mc08_recover_joiner(
                rank_offset=rank_offset, join_tp=join_tp
            )
            type(self)._joiner_proc = joiner

            os.makedirs(
                os.path.dirname(joiner_ready_marker), exist_ok=True
            )
            with open(joiner_ready_marker, "w") as fh:
                fh.write(
                    f"joiner pid={joiner.pid} rank_offset={rank_offset} "
                    f"join_tp={join_tp}\n"
                )

            deadline = time.time() + 1200
            while time.time() < deadline:
                if os.path.exists(done_marker):
                    return
                if joiner.poll() is not None:
                    self.fail(
                        f"worker: MC10 joiner subprocess exited early "
                        f"with code {joiner.returncode}; see joiner log"
                    )
                time.sleep(2)
            self.fail(
                f"worker: timed out waiting for primary done marker "
                f"{done_marker}"
            )
            return

        # Primary path
        try:
            self._generate_ok("pre-shrink")
            pre_score = self._run_gsm8k(f"pre-shrink {MC10_LAUNCH_EP}-rank")

            self._scale_to(
                old_ep_size=MC10_LAUNCH_EP,
                target_ep_size=MC10_SHRINK_TARGET,
            )
            self._generate_ok("post-shrink")
            mid_score = self._run_gsm8k(
                f"post-shrink {MC10_SHRINK_TARGET}-rank"
            )

            # Tell the worker to spawn the recover joiner covering
            # the full retired range.
            os.makedirs(
                os.path.dirname(launch_joiner_marker), exist_ok=True
            )
            with open(launch_joiner_marker, "w") as fh:
                fh.write(
                    f"rank_offset={MC10_SHRINK_TARGET} "
                    f"join_tp={MC10_LAUNCH_EP - MC10_SHRINK_TARGET}\n"
                )

            self._wait_for_marker(joiner_ready_marker, timeout_s=300)

            # FULL regrow ``SHRINK_TARGET -> LAUNCH_EP``: recovers
            # every retired rank via the v1 recover path (same
            # primitive MC08 validates).
            self._scale_to(
                old_ep_size=MC10_SHRINK_TARGET,
                target_ep_size=MC10_REGROW_TARGET,
            )
            self._generate_ok("post-regrow full")

            # Second shrink ``REGROW_TARGET -> FINAL_SHRINK_TARGET``
            # retires the top slots that were JUST recovered
            # (default 8 -> 6 retires ranks 6, 7). This is the step
            # that specifically validates the partial-recovery
            # invariant "retire ranks that were previously rejoined
            # via recover".
            self._scale_to(
                old_ep_size=MC10_REGROW_TARGET,
                target_ep_size=MC10_FINAL_SHRINK_TARGET,
            )

            # Final-cohort acceptance: every DP slot in the FINAL
            # cohort ([0, FINAL_SHRINK_TARGET)) must serve
            # MC10_POST_REGROW_PROBES_PER_SLOT sequential /generate
            # requests. See class docstring for why slots
            # [FINAL_SHRINK_TARGET, LAUNCH_EP) are NOT probed.
            probes = MC10_POST_REGROW_PROBES_PER_SLOT
            for dp_rank in range(MC10_FINAL_SHRINK_TARGET):
                for i in range(probes):
                    self._generate_ok(
                        f"post-final slot={dp_rank} "
                        f"probe={i + 1}/{probes}",
                        routed_dp_rank=dp_rank,
                    )
            print(
                f"[TEST] MC10 partial-recovery parity: "
                f"pre={pre_score:.2%} mid={mid_score:.2%} "
                f"post-final slots "
                f"[0, {MC10_FINAL_SHRINK_TARGET}) each served "
                f"{probes} sequential /generate requests OK "
                f"(recovered slots: "
                f"[{MC10_SHRINK_TARGET}, {MC10_REGROW_TARGET}) via "
                f"recover; retired-again slots: "
                f"[{MC10_FINAL_SHRINK_TARGET}, {MC10_LAUNCH_EP}) via "
                f"second shrink)"
            )
        finally:
            os.makedirs(os.path.dirname(done_marker), exist_ok=True)
            with open(done_marker, "w") as fh:
                fh.write("done\n")


if __name__ == "__main__":
    unittest.main()
