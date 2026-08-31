"""Test --disaggregation-decode-extra-slots effect on c128_state_fixed in PD disaggregation.
Fully reuses the original test script structure (DisaggregationHiCacheBase pattern).
"""

import os
import re
import shutil
import subprocess
import tempfile
import time
import unittest

from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V4_FLASH_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)

register_npu_ci(est_time=600, suite="full-16-npu-a3", nightly=True)

# Regex pattern to extract c128_state_fixed value from DSV4 memory pool logs
C128_PATTERN = re.compile(r"c128_state_fixed\s*[=:]\s*([\d.]+)\s*GB")


def _cleanup_residual_sglang():
    """Kill any leftover sglang processes before starting a new test round."""
    print("[CLEANUP] Killing residual sglang.launch_server processes...")
    os.system("pkill -9 -f 'sglang.launch_server' 2>/dev/null || true")
    time.sleep(5)


class DisaggregationExtraSlotsBase(PDDisaggregationServerBase):
    """Base class for disaggregation extra slots validation tests.
    Reuses the standard PD disaggregation server lifecycle and providesv
    log capture and c128_state_fixed extraction utilities.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = os.environ.get("MODEL_PATH", DEEPSEEK_V4_FLASH_WEIGHTS_PATH)
        cls.temp_dir = tempfile.mkdtemp()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.tp_size = int(os.environ.get("TP", "8"))

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
            "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
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
            "32",
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
            "--cuda-graph-bs",
            "1",
            "2",
            "3",
            "4",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "3",
        ]

        if extra_slots is not None:
            decode_args.extend(
                ["--disaggregation-decode-extra-slots", str(extra_slots)]
            )
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
            "ASCEND_RT_VISIBLE_DEVICES": "8,9,10,11,12,13,14,15",
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


class TestDisaggregationDecodeExtraSlots(DisaggregationExtraSlotsBase):
    """Testcase: --disaggregation-decode-extra-slots increases c128_state_fixed.

    [Test Category] Functional
    [Test Target] PD disaggregation decode extra slots on NPU
    --disaggregation-decode-extra-slots;
    """

    def test_c128_increases_with_extra_slots(self):
        """Verify c128_state_fixed grows when extra slots are configured."""

        # Round 1: baseline without extra slots
        c128_without = self._run_round(extra_slots=None)
        self.assertIsNotNone(
            c128_without,
            "Failed to capture c128_state_fixed without "
            "--disaggregation-decode-extra-slots",
        )
        print(f"[RESULT] Without extra_slots: c128_state_fixed = {c128_without} GB")

        self._cleanup_round()

        # Round 2: with extra_slots=64
        c128_with = self._run_round(extra_slots=64)
        self.assertIsNotNone(
            c128_with,
            "Failed to capture c128_state_fixed with "
            "--disaggregation-decode-extra-slots=64",
        )
        print(f"[RESULT] With extra_slots=64: c128_state_fixed = {c128_with} GB")

        self._cleanup_round()

        # Validate that extra slots increase the c128 allocation
        self.assertGreater(
            c128_with,
            c128_without,
            f"c128_state_fixed with extra_slots(64)={c128_with} GB should be "
            f"greater than without extra_slots={c128_without} GB",
        )
        print("[PASS] Test completed successfully!")


if __name__ == "__main__":
    unittest.main()
