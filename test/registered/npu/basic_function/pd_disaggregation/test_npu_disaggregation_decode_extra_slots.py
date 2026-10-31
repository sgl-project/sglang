"""Test --disaggregation-decode-extra-slots on the NPU PD disaggregation pipeline.

Launches a DSV4 prefill + decode pipeline twice:

1. Flag unset: the decode server auto-infers the default extra slots
   (2x the per-worker batch, see pd_disaggregation_hook).
2. Flag set to 64: verifies the resolved value on the running decode server,
   that c128_state_fixed grows by the formula-derived amount, and that real
   requests still flow through the prefill -> decode pipeline.
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest

import requests

from sglang.srt.mem_cache.deepseek_v4_memory_pool import get_compress_state_ring_size
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V4_FLASH_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_npu_ci(est_time=600, suite="full-16-npu-a3", nightly=True)

# Regex pattern to extract c128_state_fixed value from DSV4 memory pool logs
C128_PATTERN = re.compile(r"c128_state_fixed\s*[=:]\s*([\d.]+)\s*GB")

# Same process patterns as sglang.cli.killall, so the cleanup also catches
# servers started via `sglang serve` (this test's launch method).
_SGLANG_PROCESS_PATTERN = (
    "sglang::|sglang.launch_server|sglang.bench|sglang.data_parallel"
    "|sglang.srt|sglang serve"
)

# Decode server launch constants, also feeding the c128 formula below.
TP_SIZE = 8
DP_SIZE = 8
MAX_RUNNING_REQUESTS = 32
SPEC_NUM_DRAFT_TOKENS = 3
EXPLICIT_EXTRA_SLOTS = 64


def _cleanup_residual_sglang():
    """Kill any leftover sglang processes before starting a new test round."""
    print("[CLEANUP] Killing residual sglang processes...")
    os.system(f"pkill -9 -f '{_SGLANG_PROCESS_PATTERN}' 2>/dev/null || true")
    time.sleep(5)


def _per_worker_max_running_requests() -> int:
    """Per-worker request cap, as resolved by the pool configurator."""
    return max(1, MAX_RUNNING_REQUESTS // DP_SIZE)


def _auto_inferred_extra_slots() -> int:
    """Extra slots the decode server infers when the flag is not set.

    Mirrors pd_disaggregation_hook: with --max-running-requests set and a
    per-worker batch of at most 32, the server reserves 2x the per-worker
    batch as in-transfer headroom.
    """
    per_worker = _per_worker_max_running_requests()
    return per_worker * 2 if per_worker <= 32 else 0


def _expected_c128_state_rows(extra_slots: int) -> int:
    """c128 state rows the pool configurator reserves for extra_slots.

    Mirrors pool_configurator._get_c128_state_fixed_bytes (offline path):

      num_req_slots = per_worker_max_running_requests + extra_slots + 1
      state_rows = ceil_div(num_req_slots * ring_size + ring_size + 1, 128) * 128
      c128_bytes  = state_rows * (2 * attn_head_dim) * dtype_size * num_layers

    The per-row byte cost is model-specific but identical across rounds, so
    the measured c128 growth ratio must match this row growth ratio.
    """
    ring_size = get_compress_state_ring_size(128, SPEC_NUM_DRAFT_TOKENS)
    num_req_slots = _per_worker_max_running_requests() + extra_slots + 1
    rows = num_req_slots * ring_size + ring_size + 1
    return -(-rows // 128) * 128


class DisaggregationExtraSlotsBase(PDDisaggregationServerBase):
    """Base class for disaggregation extra slots validation tests.
    Reuses the standard PD disaggregation server lifecycle and provides
    log capture and c128_state_fixed extraction utilities.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = os.environ.get("MODEL_PATH", DEEPSEEK_V4_FLASH_WEIGHTS_PATH)
        cls.temp_dir = tempfile.mkdtemp()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.tp_size = int(os.environ.get("TP", str(TP_SIZE)))

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    @classmethod
    def _start_prefill_with_capture(cls):
        """Launch prefill server with stdout/stderr captured to a temp log file."""
        prefill_args = [
            "--attention-backend",
            "dsv4",
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            str(cls.tp_size),
            "--dtype",
            "bfloat16",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--page-size",
            "128",
            "--dp-size",
            str(cls.tp_size),
            "--enable-dp-attention",
            "--device",
            "npu",
            "--watchdog-timeout",
            "9000",
            "--chunked-prefill-size",
            "-1",
            "--max-running-requests",
            "128",
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "auto",
            "--quantization",
            "modelslim",
            "--enable-dp-lm-head",
            "--kv-cache-dtype",
            "auto",
            "--skip-server-warmup",
            "--disaggregation-transfer-backend",
            "ascend",
            "--mem-fraction-static",
            "0.8",
            "--prefill-max-requests",
            "1",
            "--disable-overlap-schedule",
            "--disaggregation-bootstrap-port",
            "8998",
        ]
        prefill_args += cls.transfer_backend

        env = {
            **os.environ,
            "HCCL_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "HCCL_BUFFSIZE": "2000",
            "SGLANG_SET_CPU_AFFINITY": "1",
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
            "STREAMS_PER_DEVICE": "32",
            "INF_NAN_MODE_FORCE_DISABLE": "1",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
            "USE_NPU_MOE_GATING_TOP_K": "1",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
            "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
            "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
            "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
            "IS_DEEPSEEK_V4": "1",
            "SGLANG_DEBUG_LAYER_NORM": "1",
            "SGLANG_DEBUG_FWD_INPUT": "1",
            "USE_FUSED_HC_PRE_ASCENDC": "1",
            "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
            "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "0",
            "SGLANG_DSV4_NPU_FUSED_INDEXER": "1",
            "SGLANG_OPT_FP8_WO_A_GEMM": "0",
            "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
            "FORCE_DRAFT_MODEL_NON_QUANT": "1",
            "SGLANG_DSV4_FP4_EXPERTS": "False",
            "SGLANG_OPT_FUSE_WQA_WKV": "0",
            "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
            "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
            "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
            "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
            "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
            "SGLANG_NPU_PROFILING": "0",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
        }

        _, host, port = cls.prefill_url.split(":")
        host = host[2:]

        command = [
            "sglang",
            "serve",
            "--model-path",
            cls.model,
            *prefill_args,
            "--host",
            host,
            "--port",
            port,
        ]
        print(f"[PREFILL] command={' '.join(command)}")

        cls._prefill_stdout = tempfile.NamedTemporaryFile(
            mode="w+", suffix="_prefill.log", delete=False, dir=cls.temp_dir
        )
        cls._prefill_stdout_name = cls._prefill_stdout.name

        cls.process_prefill = subprocess.Popen(
            command,
            stdout=cls._prefill_stdout,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

    @classmethod
    def _start_decode_with_capture(cls, extra_slots=None):
        """Launch decode server with stdout/stderr captured to a temp log file.
        Applies --disaggregation-decode-extra-slots when specified.
        """
        decode_args = [
            "--attention-backend",
            "dsv4",
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            str(cls.tp_size),
            "--dtype",
            "bfloat16",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--page-size",
            "128",
            "--dp-size",
            str(cls.tp_size),
            "--enable-dp-attention",
            "--device",
            "npu",
            "--watchdog-timeout",
            "9000",
            "--chunked-prefill-size",
            "-1",
            "--max-running-requests",
            str(MAX_RUNNING_REQUESTS),
            "--moe-a2a-backend",
            "deepep",
            "--deepep-mode",
            "auto",
            "--quantization",
            "modelslim",
            "--enable-dp-lm-head",
            "--kv-cache-dtype",
            "auto",
            "--skip-server-warmup",
            "--disaggregation-transfer-backend",
            "ascend",
            "--mem-fraction-static",
            "0.85",
            "--cuda-graph-bs-decode",
            "1",
            "2",
            "3",
            "4",
            # Prefill owns the first half of the visible devices; decode takes
            # the second half. The CI runner controls the visible device set,
            # so no device ids are hardcoded here.
            "--base-gpu-id",
            str(cls.tp_size),
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            str(SPEC_NUM_DRAFT_TOKENS),
        ]

        if extra_slots is not None:
            decode_args.extend(["--disaggregation-decode-extra-slots", str(extra_slots)])
            print(f"[DECODE] Using --disaggregation-decode-extra-slots={extra_slots}")

        decode_args += cls.transfer_backend

        env = {
            **os.environ,
            "HCCL_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
            "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24667",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "HCCL_BUFFSIZE": "2000",
            "SGLANG_SET_CPU_AFFINITY": "1",
            "STREAMS_PER_DEVICE": "32",
            "INF_NAN_MODE_FORCE_DISABLE": "1",
            "HCCL_OP_EXPANSION_MODE": "AIV",
            "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
            "USE_NPU_MOE_GATING_TOP_K": "1",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "64",
            "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
            "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
            "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
            "IS_DEEPSEEK_V4": "1",
            "SGLANG_DEBUG_LAYER_NORM": "1",
            "SGLANG_DEBUG_FWD_INPUT": "1",
            "USE_FUSED_HC_PRE_ASCENDC": "1",
            "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
            "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "0",
            "SGLANG_DSV4_NPU_FUSED_INDEXER": "1",
            "SGLANG_OPT_FP8_WO_A_GEMM": "0",
            "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
            "FORCE_DRAFT_MODEL_NON_QUANT": "1",
            "SGLANG_DSV4_FP4_EXPERTS": "False",
            "SGLANG_OPT_FUSE_WQA_WKV": "0",
            "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
            "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
            "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
            "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
            "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
            "SGLANG_NPU_PROFILING": "0",
            "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }

        _, host, port = cls.decode_url.split(":")
        host = host[2:]

        command = [
            "sglang",
            "serve",
            "--model-path",
            cls.model,
            *decode_args,
            "--host",
            host,
            "--port",
            port,
        ]
        print(f"[DECODE] command={' '.join(command)}")

        cls._decode_stdout = tempfile.NamedTemporaryFile(
            mode="w+", suffix="_decode.log", delete=False, dir=cls.temp_dir
        )
        cls._decode_stdout_name = cls._decode_stdout.name

        cls.process_decode = subprocess.Popen(
            command,
            stdout=cls._decode_stdout,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

    def _extract_c128(self, log_path, max_wait=60):
        """Poll the log file and extract the c128_state_fixed value in GB."""
        for i in range(max_wait):
            try:
                with open(log_path, "r") as f:
                    content = f.read()
                matches = C128_PATTERN.findall(content)
                if matches:
                    val = float(matches[-1])
                    print(f"  [EXTRACT] c128_state_fixed = {val} GB")
                    return val
            except Exception:
                pass
            time.sleep(1)

        print(f"[WARN] c128_state_fixed not found in {log_path}, last 30 lines:")
        self._tail_log(log_path, 30)
        return None

    @staticmethod
    def _tail_log(log_path, n=30):
        """Print the last n lines of a log file for debugging."""
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
            for line in lines[-n:]:
                print("    LOG> " + line.rstrip())
        except Exception as e:
            print(f"    Could not read log: {e}")

    @staticmethod
    def _kill_gracefully(proc):
        """Terminate a process gracefully, falling back to kill if needed."""
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _cleanup_round(self):
        """Terminate server processes and release resources between test rounds."""
        print("[CLEANUP] Terminating decode and prefill...")
        if getattr(self, "process_lb", None):
            try:
                kill_process_tree(self.process_lb.pid, wait_timeout=60)
            except Exception as e:
                print(f"[CLEANUP] Error killing lb: {e}")
            self.process_lb = None
        self._kill_gracefully(getattr(self, "process_decode", None))
        self._kill_gracefully(getattr(self, "process_prefill", None))

        for attr in ["_prefill_stdout", "_decode_stdout"]:
            if hasattr(self, attr):
                try:
                    getattr(self, attr).close()
                except Exception:
                    pass

        print("[CLEANUP] Sleeping 10s for NPU resource release...")
        time.sleep(10)

    def _run_round(self, extra_slots):
        """Run a full test round: start servers, wait for ready, extract c128 value."""
        print(f"\n{'=' * 60}")
        print(f"ROUND: extra_slots={extra_slots}")
        print(f"{'=' * 60}")

        _cleanup_residual_sglang()
        self._start_prefill_with_capture()
        self._start_decode_with_capture(extra_slots)

        print("[WAIT] Waiting for prefill /health...")
        self.wait_server_ready(self.prefill_url + "/health")
        print("[WAIT] Waiting for decode /health...")
        self.wait_server_ready(self.decode_url + "/health")

        print("[WAIT] Services ready, sleeping 10s for memory calc log...")
        time.sleep(10)

        val = self._extract_c128(self._decode_stdout_name)
        return val

    def _verify_extra_slots_in_use(self, expected_slots):
        """The running decode server must expose the resolved extra slots."""
        info = requests.get(self.decode_url + "/server_info", timeout=30).json()
        self.assertEqual(
            info.get("disaggregation_decode_extra_slots"),
            expected_slots,
            f"decode server resolved disaggregation_decode_extra_slots="
            f"{info.get('disaggregation_decode_extra_slots')}, "
            f"expected {expected_slots}",
        )

    def _verify_serving_through_pd_pipeline(self):
        """Serve real requests through the pipeline with the extra slots active.

        Each request transits the in-transfer state on the decode side, so a
        successful completion proves the parameterized request-slot pool
        actually serves traffic, not just that memory accounting changed.
        """
        self.launch_lb()
        try:
            for i in range(4):
                response = requests.post(
                    f"{self.lb_url}/generate",
                    json={
                        "text": "The capital of France is",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 32,
                        },
                    },
                    timeout=120,
                )
                self.assertEqual(
                    response.status_code,
                    200,
                    f"Request {i} failed: {response.status_code} - {response.text}",
                )
                data = response.json()
                self.assertTrue(
                    isinstance(data["text"], str) and data["text"].strip(),
                    f"Request {i} returned an empty completion.",
                )
                self.assertIn(
                    data["meta_info"]["finish_reason"]["type"],
                    ("stop", "length"),
                    f"Request {i} finished abnormally.",
                )
        finally:
            if self.process_lb:
                try:
                    kill_process_tree(self.process_lb.pid, wait_timeout=60)
                except Exception as e:
                    print(f"[CLEANUP] Error killing lb: {e}")
                self.process_lb = None


class TestDisaggregationDecodeExtraSlots(DisaggregationExtraSlotsBase):
    """Testcase: --disaggregation-decode-extra-slots sizes the decode c128 state pool.

    [Test Category] Parameter
    [Test Target] --disaggregation-decode-extra-slots
    """

    def test_c128_matches_formula_with_extra_slots(self):
        """Verify c128_state_fixed follows the sizing formula for extra slots."""

        # Round 1: flag not set. The decode server auto-infers the default
        # extra slots (2x the per-worker batch) instead of using zero.
        c128_default = self._run_round(extra_slots=None)
        self.assertIsNotNone(
            c128_default,
            "Failed to capture c128_state_fixed with auto-inferred "
            "--disaggregation-decode-extra-slots",
        )
        self.assertGreater(
            c128_default,
            0,
            "c128_state_fixed should be positive for the DSV4 decode pool",
        )
        print(
            f"[RESULT] Auto-inferred extra_slots="
            f"{_auto_inferred_extra_slots()}: c128_state_fixed = {c128_default} GB"
        )

        self._cleanup_round()

        # Round 2: explicit extra_slots=64
        c128_explicit = self._run_round(extra_slots=EXPLICIT_EXTRA_SLOTS)
        self.assertIsNotNone(
            c128_explicit,
            "Failed to capture c128_state_fixed with "
            f"--disaggregation-decode-extra-slots={EXPLICIT_EXTRA_SLOTS}",
        )
        print(
            f"[RESULT] With extra_slots={EXPLICIT_EXTRA_SLOTS}: "
            f"c128_state_fixed = {c128_explicit} GB"
        )

        # The parameter is actually in force and usable on the running server.
        self._verify_extra_slots_in_use(EXPLICIT_EXTRA_SLOTS)
        self._verify_serving_through_pd_pipeline()

        self._cleanup_round()

        # Formula-derived expectation (see _expected_c128_state_rows): the
        # per-row byte cost is model-specific but identical across rounds, so
        # the growth must match the row-growth ratio exactly, up to the
        # 0.01 GB log rounding.
        rows_default = _expected_c128_state_rows(_auto_inferred_extra_slots())
        rows_explicit = _expected_c128_state_rows(EXPLICIT_EXTRA_SLOTS)
        expected_c128_explicit = c128_default * rows_explicit / rows_default
        tolerance_gb = max(0.1, 0.05 * expected_c128_explicit)
        self.assertAlmostEqual(
            c128_explicit,
            expected_c128_explicit,
            delta=tolerance_gb,
            msg=(
                f"c128_state_fixed with extra_slots={EXPLICIT_EXTRA_SLOTS} "
                f"({c128_explicit} GB) should be ~{expected_c128_explicit:.2f} GB, "
                f"derived from {c128_default} GB x {rows_explicit}/{rows_default} "
                f"rows (per-worker max_running_requests="
                f"{_per_worker_max_running_requests()})"
            ),
        )
        print("[PASS] Test completed successfully!")


if __name__ == "__main__":
    unittest.main()
