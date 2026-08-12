import json
import os
import unittest
import uuid
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
    assert_process_healthy,
    configure_nixl_pd_backend,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_ci,
    popen_launch_pd_server,
    start_subprocess_fail_fast_watcher,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_cuda_ci(est_time=1000, stage="base-c", runner_config="8-gpu-h20")
# base-c 8-GPU runner is required for TP4 prefill + TP4 decode.

NIXL_PREFILL_TP_SIZE = 4
NIXL_DECODE_TP_SIZE = 4
NIXL_DECODE_BASE_GPU_ID = 4
NIXL_PREFILL_UCX_NUM_THREADS = 2

# This is a PD transfer functional gate, not the standalone model-quality gate.
# The standalone Llama-3.1-8B GSM8K threshold is 0.80, while existing PD
# disaggregation coverage gates at 0.62. Keep NIXL above catastrophic transfer
# failure while leaving margin for PD/router/evaluator variance on 200 examples.
NIXL_GSM8K_SCORE_THRESHOLD = 0.62


def _nixl_backend_config(
    backend,
    backend_params_json,
    ucx_num_threads=NIXL_PREFILL_UCX_NUM_THREADS,
):
    backend_params = json.loads(backend_params_json)
    if not isinstance(backend_params, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in backend_params.items()
    ):
        raise ValueError(
            "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS must be a JSON object "
            "with string keys and string values"
        )

    if backend == "UCX":
        backend_params["num_threads"] = str(ucx_num_threads)
    elif backend == "OBJ":
        backend_params.setdefault("num_threads", "8")
    elif backend == "GDS_MT":
        backend_params.setdefault("thread_count", "8")
    elif backend == "UCCL":
        backend_params.setdefault("num_cpus", "8")

    return backend, backend_params


def _nixl_prefill_ucx_backend_env():
    backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
    if backend != "UCX":
        return {}

    _, backend_params = _nixl_backend_config(
        backend,
        envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS.get(),
        ucx_num_threads=NIXL_PREFILL_UCX_NUM_THREADS,
    )
    return {"SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS": json.dumps(backend_params)}


def _get_configured_nixl_backend_probe_error():
    backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
    backend_params_json = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS.get()

    try:
        from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t
    except ImportError as e:
        return f"NIXL import failed: {e}"

    try:
        backend, backend_params = _nixl_backend_config(backend, backend_params_json)
    except (json.JSONDecodeError, ValueError) as e:
        return str(e)

    try:
        probe_num_threads = NIXL_PREFILL_UCX_NUM_THREADS if backend == "UCX" else 8
        agent_config = nixl_agent_config(
            backends=[],
            num_threads=probe_num_threads,
            sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
        )
        agent = nixl_agent(f"sglang_nixl_probe_{uuid.uuid4()}", agent_config)
        available_plugins = agent.get_plugin_list()
        if backend not in available_plugins:
            return (
                f"NIXL backend {backend!r} not found. "
                f"Available plugins: {available_plugins}."
            )
        agent.create_backend(backend, backend_params)
    except Exception as e:
        return f"NIXL backend probe failed: {e}"

    return None


def _has_configured_nixl_backend():
    return _get_configured_nixl_backend_probe_error() is None


def _require_configured_nixl_backend():
    error = _get_configured_nixl_backend_probe_error()
    if error is not None:
        raise RuntimeError(error)


def _clear_disagg_failure_env():
    # Failure injection is scoped to the failure test; clear inherited env so
    # Basic/Accuracy cannot run with injected transfer failures.
    os.environ.pop("SGLANG_TEST_DISAGG_FAILURE_PROB", None)
    os.environ.pop("DISAGGREGATION_TEST_FAILURE_PROB", None)


_HAS_CONFIGURED_NIXL_BACKEND = is_in_ci() or _has_configured_nixl_backend()


class NixlPDDisaggregationServerBase(PDDisaggregationServerBase):
    prefill_tp_size = NIXL_PREFILL_TP_SIZE
    decode_tp_size = NIXL_DECODE_TP_SIZE
    decode_base_gpu_id = NIXL_DECODE_BASE_GPU_ID

    @classmethod
    def start_prefill(cls):
        prefill_env = {
            **cls.extra_prefill_env,
            **_nixl_prefill_ucx_backend_env(),
        }
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.prefill_tp_size),
        ] + list(cls.extra_prefill_args)
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=prefill_env,
            return_stdout_stderr=(
                (cls._prefill_stdout_buf, cls._prefill_stderr_buf)
                if cls.capture_per_side_logs
                else None
            ),
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.decode_tp_size),
            "--base-gpu-id",
            str(cls.decode_base_gpu_id),
        ] + list(cls.extra_decode_args)
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=cls.extra_decode_env,
            return_stdout_stderr=(
                (cls._decode_stdout_buf, cls._decode_stderr_buf)
                if cls.capture_per_side_logs
                else None
            ),
        )

    @classmethod
    def replace_role(cls, role):
        """Restart one PD role in place while leaving its peer and router alive."""
        if cls._fail_fast_stop is not None:
            cls._fail_fast_stop.set()
            cls._fail_fast_stop = None

        process_attr = f"process_{role}"
        process = getattr(cls, process_attr)
        if process is not None:
            # SIGTERM first: a bare hard kill leaves CUDA context and pinned
            # memory to kernel reclaim, which can hold GPU memory for minutes
            # while the replacement launches on the same GPUs (callers may
            # replace several times in a row). Also matches how Kubernetes
            # actually replaces a pod.
            terminate_and_kill_process_tree(process, wait_timeout=60)
        setattr(cls, process_attr, None)

        getattr(cls, f"start_{role}")()
        cls.wait_server_ready(
            getattr(cls, f"{role}_url") + "/health",
            process=getattr(cls, process_attr),
        )
        cls._fail_fast_stop = start_subprocess_fail_fast_watcher(
            [
                ("prefill", cls.process_prefill),
                ("decode", cls.process_decode),
                ("lb", cls.process_lb),
            ]
        )


@unittest.skipUnless(
    _HAS_CONFIGURED_NIXL_BACKEND,
    "NIXL with the configured backend is required for this test.",
)
class TestDisaggregationNixlBasic(NixlPDDisaggregationServerBase):
    """Small NIXL PD E2E coverage.

    Mooncake already owns the broad disaggregation functional matrix in
    test_disaggregation_basic.py. This class intentionally mirrors only the
    subset that proves NIXL can launch, transfer KV, serve a request, return
    logprobs, and keep all workers alive.
    """

    # Mask the connection-refused heartbeat path for the replacement test:
    # with the default 5s interval the heartbeat clears the stale cache while
    # the prefill is still down, so replacement recovery would pass even
    # without #33789's failure-driven invalidation. The short waiting timeout
    # makes each stale-cache request fail fast instead of in 300s.
    # Class-level on purpose (one 8-GPU launch for the whole class), so the
    # other tests here also run with the heartbeat masked and a 30s waiting
    # timeout — if a healthy test flakes on queueing >30s, look here first.
    extra_decode_env = {
        "SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL": "3600",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "30",
    }

    @classmethod
    def setUpClass(cls):
        _require_configured_nixl_backend()
        _clear_disagg_failure_env()
        super().setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)
        configure_nixl_pd_backend(cls)
        cls.launch_all()

    def test_completion_returns_text_and_workers_stay_alive(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)

        data = response.json()
        self.assertIn("text", data, f"Unexpected response shape: {data}")
        self.assertGreater(len(data["text"]), 0, "Generated text should not be empty")

        assert_process_healthy(self, "load balancer", self.process_lb, self.lb_url)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)

    def test_logprob(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of france is ",
                "sampling_params": {"temperature": 0},
                "return_logprob": True,
                "return_input_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)

        meta_info = response.json()["meta_info"]
        completion_tokens = meta_info["completion_tokens"]
        input_logprobs = meta_info["input_token_logprobs"]
        output_logprobs = meta_info["output_token_logprobs"]

        self.assertEqual(len(output_logprobs), completion_tokens)
        self.assertGreater(len(input_logprobs), 0)

    def _prefill_rank_endpoints(self):
        """Per-rank (ip, port) endpoint map from the bootstrap /route table."""
        endpoints = {}
        for tp_rank in range(self.prefill_tp_size):
            response = requests.get(
                f"http://{self.base_host}:{self.bootstrap_port}/route",
                params={
                    "prefill_dp_rank": 0,
                    "prefill_cp_rank": 0,
                    "target_tp_rank": tp_rank,
                    "target_pp_rank": 0,
                },
                timeout=30,
            )
            self.assertEqual(response.status_code, 200, response.text)
            info = response.json()
            endpoints[tp_rank] = (info["rank_ip"], info["rank_port"])
        return endpoints

    def test_role_replacements_reconnect(self):
        """#33789: a replaced prefill behind a still-reachable bootstrap
        address strands the decode's cached rank endpoints. The heartbeat is
        masked (see extra_decode_env), so recovery must come from the failure
        path: requests that reuse the stale cache time out and drop it, after
        which registration happens afresh and serving resumes."""

        def send_request(stage):
            try:
                return requests.post(
                    self.lb_url + "/generate",
                    json={
                        "text": f"NIXL replacement check after {stage}",
                        "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    },
                    timeout=90,
                )
            except requests.RequestException:
                return None

        def assert_request_succeeds(stage):
            response = send_request(stage)
            self.assertIsNotNone(response, f"request errored after {stage}")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertGreater(len(response.json()["text"]), 0)

        assert_request_succeeds("initial startup")

        self.replace_role("decode")
        assert_request_succeeds("decode replacement")

        # A replacement that rebinds the exact same rank endpoints would let
        # the stale cache accidentally target the replacement, and stock code
        # would pass too. Rank ports are ephemeral, so a full collision is
        # rare; re-replace until the topology discriminates.
        stale_endpoints = self._prefill_rank_endpoints()
        for _ in range(3):
            self.replace_role("prefill")
            if self._prefill_rank_endpoints() != stale_endpoints:
                break
        else:
            self.fail(
                "prefill replacement kept identical rank endpoints three "
                "times; the test cannot discriminate fixed from stock code"
            )
        # Each failed request invalidates the cache on the ranks that timed
        # out locally (propagated failures do not, see failure_exception), so
        # full repair can take up to one failed request per decode TP rank.
        outcomes = []
        for attempt in range(self.decode_tp_size + 1):
            response = send_request(f"prefill replacement attempt {attempt}")
            outcomes.append(response is not None and response.status_code == 200)
            if outcomes[-1]:
                break
        # The first request usually fails on the stale cached endpoints (no
        # automatic retry today) -- but that is not asserted: a future
        # automatic re-bootstrap would make it legitimately succeed, and
        # this test only pins the recovery contract.
        if outcomes[0]:
            print(
                "NOTE: first request after prefill replacement succeeded; "
                "recovery contract still verified below."
            )
        self.assertTrue(
            outcomes[-1],
            f"serving did not recover after {len(outcomes)} requests "
            "following prefill replacement",
        )


@unittest.skipUnless(
    _HAS_CONFIGURED_NIXL_BACKEND,
    "NIXL with the configured backend is required for this test.",
)
class TestDisaggregationNixlAccuracy(NixlPDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        _require_configured_nixl_backend()
        _clear_disagg_failure_env()
        super().setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)
        configure_nixl_pd_backend(cls)
        cls.launch_all()

    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_shots=5,
            num_threads=128,
            temperature=0.0,
        )

        metrics = run_eval(args)
        print(f"Evaluation metrics: {metrics}")
        self.assertGreater(
            metrics["score"],
            NIXL_GSM8K_SCORE_THRESHOLD,
            f"Expected NIXL PD transfer to preserve GSM8K accuracy, got {metrics}",
        )

        assert_process_healthy(self, "load balancer", self.process_lb, self.lb_url)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


@unittest.skipUnless(
    _HAS_CONFIGURED_NIXL_BACKEND,
    "NIXL with the configured backend is required for this test.",
)
class TestDisaggregationNixlFailure(NixlPDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        _require_configured_nixl_backend()
        super().setUpClass()
        os.environ["SGLANG_TEST_DISAGG_FAILURE_PROB"] = "0.05"
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)
        configure_nixl_pd_backend(cls)
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SGLANG_TEST_DISAGG_FAILURE_PROB", None)
        super().tearDownClass()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )

        # Match TestDisaggregationMooncakeFailure: inject many transfer failures
        # and tolerate eval/request errors as long as workers remain healthy.
        try:
            metrics = run_eval(args)
            print(f"Evaluation metrics: {metrics}")
        except Exception as e:
            print(f"Test encountered expected errors: {e}")

        assert_process_healthy(self, "load balancer", self.process_lb, self.lb_url)
        assert_process_healthy(
            self, "prefill", self.process_prefill, self.prefill_url, "/health_generate"
        )
        assert_process_healthy(
            self, "decode", self.process_decode, self.decode_url, "/health_generate"
        )


if __name__ == "__main__":
    unittest.main()
