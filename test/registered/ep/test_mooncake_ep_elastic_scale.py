"""End-to-end coverage for Mooncake elastic EP capacity and scale-up."""

import json
import math
import os
import re
import subprocess
import tempfile
import time
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    get_rdma_devices_args,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=300, stage="base-c", runner_config="4-gpu-h100")

INITIAL_EP_SIZE = 2
MAX_EP_SIZE = 4
SCALE_TIMEOUT_SEC = 300
DECODE_GRAPH_METRIC = "sglang:cuda_graph_passes_total"
_METRIC_LINE = re.compile(r"^([^{}]+)\{(.*)\}\s+([0-9.eE+-]+)$")
_METRIC_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z_0-9]*)="((?:\\.|[^"\\])*)"')
LOGPROB_ABS_TOL = 0.05


def _server_args(
    *,
    dist_init_addr: str,
    tp_size: int,
    attn_dp_size: int,
    nnodes: int,
    node_rank: int,
    base_gpu_id: int,
    cohort_offset: int,
    ib_devices: str,
) -> list[str]:
    args = [
        "--device",
        "cuda",
        "--trust-remote-code",
        "--tp",
        str(tp_size),
        "--attn-dp-size",
        str(attn_dp_size),
        "--nnodes",
        str(nnodes),
        "--node-rank",
        str(node_rank),
        "--base-gpu-id",
        str(base_gpu_id),
        "--dist-init-addr",
        dist_init_addr,
        "--moe-a2a-backend",
        "mooncake",
        "--elastic-ep-backend",
        "mooncake",
        "--disable-custom-all-reduce",
        "--mooncake-ib-device",
        ib_devices,
        "--extra-metric-labels",
        json.dumps({"ep_cohort_offset": str(cohort_offset)}),
        "--deepep-mode",
        "low_latency",
        "--enable-dp-lm-head",
        "--enable-eplb",
        "--ep-num-redundant-experts",
        "24",
        "--load-balance-method",
        "round_robin",
        "--moe-dense-tp-size",
        "1",
        "--elastic-ep-initial-size",
        str(INITIAL_EP_SIZE),
        "--max-ep-size",
        str(MAX_EP_SIZE),
        "--mem-fraction-static",
        "0.5",
        "--chunked-prefill-size",
        "512",
        "--max-running-requests",
        "4",
        "--enable-metrics",
        "--cuda-graph-backend-decode",
        "full",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--cuda-graph-bs-decode",
        "1",
    ]
    return args


def _assert_generation_matches(
    reference: dict, candidate: dict, *, context: str
) -> None:
    """Compare generated token IDs and their output logprobs only."""
    reference_logprobs = reference.get("meta_info", {}).get("output_token_logprobs")
    candidate_logprobs = candidate.get("meta_info", {}).get("output_token_logprobs")
    if not reference_logprobs or not candidate_logprobs:
        raise AssertionError(f"{context}: missing output_token_logprobs")
    if len(reference_logprobs) != len(candidate_logprobs):
        raise AssertionError(
            f"{context}: output logprob lengths differ: "
            f"reference={len(reference_logprobs)} candidate={len(candidate_logprobs)}"
        )

    reference_ids = [int(item[1]) for item in reference_logprobs]
    candidate_ids = [int(item[1]) for item in candidate_logprobs]
    reference_values = [float(item[0]) for item in reference_logprobs]
    candidate_values = [float(item[0]) for item in candidate_logprobs]
    if not all(math.isfinite(value) for value in reference_values + candidate_values):
        raise AssertionError(f"{context}: output logprob contains non-finite values")
    max_delta = max(
        abs(reference_value - candidate_value)
        for reference_value, candidate_value in zip(reference_values, candidate_values)
    )
    print(
        f"[elastic-ep-check] {context} max_abs_logprob_delta={max_delta:.6f}",
        flush=True,
    )
    if reference_ids != candidate_ids:
        raise AssertionError(
            f"{context}: output token IDs differ: "
            f"reference={reference_ids} candidate={candidate_ids}"
        )
    if max_delta > LOGPROB_ABS_TOL:
        raise AssertionError(
            f"{context}: output logprob delta {max_delta:.6f} exceeds "
            f"absolute tolerance {LOGPROB_ABS_TOL:.2f}"
        )


class TestMooncakeElasticEpScale(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = f"http://127.0.0.1:{find_available_port(20000)}"
        cls.dist_port = find_available_port(23000)
        cls.dist_init_addr = f"127.0.0.1:{cls.dist_port}"
        cls._owned_processes = []
        cls._prometheus_tempdir = tempfile.TemporaryDirectory(
            prefix="sglang-mooncake-metrics-"
        )

        # Keep each worker's ordinal equal to its local CUDA ordinal. The CI
        # runner exposes all four GPUs; --base-gpu-id assigns disjoint cohorts.
        cls.server_env = os.environ.copy()
        cls.server_env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] = "0"
        cls.server_env["SGLANG_JIT_DEEPGEMM_FAST_WARMUP"] = "1"
        cls.server_env["PROMETHEUS_MULTIPROC_DIR"] = cls._prometheus_tempdir.name

        primary_args = _server_args(
            dist_init_addr=cls.dist_init_addr,
            tp_size=INITIAL_EP_SIZE,
            attn_dp_size=INITIAL_EP_SIZE,
            nnodes=1,
            node_rank=0,
            base_gpu_id=0,
            cohort_offset=0,
            ib_devices=get_rdma_devices_args([0, 1]),
        )
        cls.primary_process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=primary_args,
            env=cls.server_env,
            device="cuda",
        )
        cls._owned_processes.append(("primary", cls.primary_process))

    @classmethod
    def tearDownClass(cls):
        cleanup_errors = []
        all_processes_stopped = True
        # The primary broadcasts ShutdownReq before the joiners shut down.
        for _, process in getattr(cls, "_owned_processes", []):
            try:
                terminate_and_kill_process_tree(process)
            except Exception as exc:
                cleanup_errors.append(exc)
                all_processes_stopped = False
            if process.poll() is None:
                cleanup_errors.append(
                    RuntimeError("Owned SGLang process remained alive after teardown")
                )
                all_processes_stopped = False
        if all_processes_stopped:
            prometheus_tempdir = getattr(cls, "_prometheus_tempdir", None)
            if prometheus_tempdir is not None:
                try:
                    prometheus_tempdir.cleanup()
                except Exception as exc:
                    cleanup_errors.append(exc)
        if cleanup_errors:
            raise cleanup_errors[0]

    def _assert_owned_processes_alive(self) -> None:
        for name, process in self.__class__._owned_processes:
            self.assertIsNone(
                process.poll(),
                f"Owned SGLang process {name} exited unexpectedly",
            )

    def _launch_joiner(self, rank_offset: int) -> subprocess.Popen:
        joiner_port = find_available_port(25000 + rank_offset * 1000)
        ib_devices = get_rdma_devices_args([rank_offset])
        args = _server_args(
            dist_init_addr=self.dist_init_addr,
            tp_size=1,
            attn_dp_size=1,
            nnodes=2,
            node_rank=1,
            base_gpu_id=rank_offset,
            cohort_offset=rank_offset,
            ib_devices=ib_devices,
        )
        command = [
            "sglang",
            "serve",
            "--model-path",
            self.model,
            *args,
            "--elastic-ep-join-mode",
            "scale",
            "--elastic-ep-join-rank-offset",
            str(rank_offset),
            "--host",
            "127.0.0.1",
            "--port",
            str(joiner_port),
        ]
        process = subprocess.Popen(command, env=self.server_env)
        self.__class__._owned_processes.append((f"joiner-rank-{rank_offset}", process))
        self.assertIsNone(
            process.poll(),
            f"Scale joiner at EP rank {rank_offset} exited during startup",
        )
        return process

    def _generate_on_rank(
        self,
        routed_dp_rank: int,
        *,
        return_logprob: bool = True,
        timeout: float = 120,
    ) -> dict:
        payload = {
            "text": "The answer is",
            "sampling_params": {
                "max_new_tokens": 3,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "routed_dp_rank": routed_dp_rank,
        }
        if return_logprob:
            payload.update(
                {
                    "return_logprob": True,
                    "top_logprobs_num": 1,
                    "logprob_start_len": 0,
                }
            )
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=timeout,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        meta_info = result["meta_info"]
        # Scale joiners report the cohort-local DP rank. The routed request
        # selects the global rank; the metric cohort label identifies its owner.
        self.assertEqual(meta_info.get("dp_rank"), 0)
        self.assertGreater(meta_info.get("completion_tokens", 0), 1)

        if return_logprob:
            output_logprobs = meta_info.get("output_token_logprobs") or []
            self.assertGreaterEqual(len(output_logprobs), 2)
            for item in output_logprobs:
                self.assertGreaterEqual(len(item), 1)
                self.assertTrue(
                    math.isfinite(float(item[0])),
                    f"Non-finite output token logprob: {item[0]!r}",
                )
        return result

    def _decode_graph_passes(self, cohort_offset: int, dp_rank: int = 0) -> float:
        response = requests.get(f"{self.base_url}/metrics", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        total = 0.0
        for line in response.text.splitlines():
            match = _METRIC_LINE.match(line)
            if match is None or match.group(1) != DECODE_GRAPH_METRIC:
                continue
            labels = dict(_METRIC_LABEL.findall(match.group(2)))
            if (
                labels.get("mode") == "decode_cuda_graph"
                and labels.get("dp_rank") == str(dp_rank)
                and labels.get("ep_cohort_offset") == str(cohort_offset)
            ):
                total += float(match.group(3))
        return total

    def _assert_graph_replay(self, routed_dp_rank: int, cohort_offset: int) -> dict:
        before = self._decode_graph_passes(cohort_offset=cohort_offset)
        result = self._generate_on_rank(routed_dp_rank=routed_dp_rank)
        deadline = time.monotonic() + 10
        after = self._decode_graph_passes(cohort_offset=cohort_offset)
        while after <= before and time.monotonic() < deadline:
            time.sleep(0.25)
            after = self._decode_graph_passes(cohort_offset=cohort_offset)
        self.assertGreater(
            after,
            before,
            "A routed decode did not increment the decode_cuda_graph counter "
            f"for EP cohort offset {cohort_offset}",
        )
        return result

    def _scale_to(self, old_ep_size: int, new_ep_size: int) -> None:
        self._launch_joiner(rank_offset=old_ep_size)
        response = requests.post(
            f"{self.base_url}/scale_elastic_ep",
            json={"new_ep_size": new_ep_size},
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["old_ep_size"], old_ep_size)
        self.assertEqual(body["new_ep_size"], new_ep_size)

        deadline = time.monotonic() + SCALE_TIMEOUT_SEC
        last_state = None
        while time.monotonic() < deadline:
            self._assert_owned_processes_alive()
            state_response = requests.get(
                f"{self.base_url}/is_scaling_elastic_ep", timeout=15
            )
            self.assertEqual(state_response.status_code, 200, state_response.text)
            last_state = state_response.json()
            self.assertIsNone(
                last_state.get("last_error"),
                f"Elastic EP scale failed: {last_state}",
            )
            self.assertNotEqual(
                last_state.get("scale_phase"),
                "failed",
                f"Elastic EP scale failed: {last_state}",
            )
            if (
                not last_state.get("is_scaling_elastic_ep", True)
                and last_state.get("effective_ep_size") == new_ep_size
                and last_state.get("scale_phase") == "serving_expanded"
            ):
                return
            self.assertTrue(
                last_state.get("is_scaling_elastic_ep"),
                f"Scale stopped before reaching EP{new_ep_size}: {last_state}",
            )

            try:
                # Workers call maybe_join_ep_ranks() from a forward pass. This
                # request keeps an otherwise idle server moving through scale.
                self._generate_on_rank(
                    routed_dp_rank=0,
                    return_logprob=False,
                    timeout=30,
                )
            except requests.Timeout:
                # A forward can wait for the joiner inside the rendezvous.
                # Recheck process health and scale state on the next iteration.
                pass
            time.sleep(1)

        self.fail(f"Scale to EP{new_ep_size} timed out; last state was {last_state}")

    def test_reserved_capacity_grows_twice_with_decode_graph(self):
        # Use the pre-scale output as the baseline for both expanded cohorts.
        baseline = self._assert_graph_replay(routed_dp_rank=0, cohort_offset=0)

        for old_ep_size, new_ep_size in ((2, 3), (3, 4)):
            self._scale_to(old_ep_size, new_ep_size)
            old_cohort_result = self._assert_graph_replay(
                routed_dp_rank=0,
                cohort_offset=0,
            )
            _assert_generation_matches(
                baseline,
                old_cohort_result,
                context=f"EP{new_ep_size} old cohort 0",
            )
            # The routed rank reports cohort-local dp_rank=0; the extra metric
            # label proves this request and graph replay ran in its new cohort.
            new_cohort_result = self._assert_graph_replay(
                routed_dp_rank=new_ep_size - 1,
                cohort_offset=old_ep_size,
            )
            _assert_generation_matches(
                baseline,
                new_cohort_result,
                context=f"EP{new_ep_size} new cohort {old_ep_size}",
            )


if __name__ == "__main__":
    unittest.main()
