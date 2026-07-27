import os
import re
import subprocess
import tempfile
import unittest
from typing import List, Optional
from urllib.parse import urlparse

from sglang.bench_serving import run_benchmark
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
)
from sglang.utils import wait_for_http_ready

register_npu_ci(est_time=1200, suite="full-2-npu-a3", nightly=True)

MODEL = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
_LAUNCH_TIMEOUT = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

_BS_LOG_RE = re.compile(r"Capture cuda graph bs \[([^\]]+)\]")
_MEM_LOG_RE = re.compile(r"mem usage=([\d.]+) GB")


def _pd_ports():
    p = urlparse(DEFAULT_URL_FOR_TEST)
    host = p.hostname
    bp = str(p.port)
    return {
        "host": host,
        "lb": bp,
        "prefill": str(int(bp) + 100),
        "decode": str(int(bp) + 200),
        "bootstrap": str(int(bp) + 500),
    }


def _pd_transport_args():
    # NPU uses ascend transfer backend (no RDMA/IB devices needed).
    return ["--disaggregation-transfer-backend", "ascend"]


def _launch_pd_server(url, *, mode, bootstrap_port, extra_args, base_gpu_id="0"):
    """Launch one PD server (prefill or decode), capturing stderr to a temp file.

    Returns (process, stderr_file_path).
    """
    _, host, port = url.split(":")
    host = host[2:]
    err_fd, err_path = tempfile.mkstemp(suffix=".log", prefix=f"pd_{mode}_")
    os.close(err_fd)

    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--host",
        host,
        "--port",
        port,
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        "0.8",
        "--disaggregation-mode",
        mode,
        "--disaggregation-bootstrap-port",
        bootstrap_port,
        "--base-gpu-id",
        base_gpu_id,
        "--tp",
        "1",
        *extra_args,
        *_pd_transport_args(),
    ]
    env = {
        **os.environ,
        "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:26666",
    }
    with open(err_path, "w") as err_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=err_file,
            text=True,
            env=env,
        )
    wait_for_http_ready(url + "/health", timeout=_LAUNCH_TIMEOUT, process=proc)
    return proc, err_path


def _launch_router(prefill_url, decode_url, host, lb_port):
    cmd = [
        "python3",
        "-m",
        "sglang_router.launch_router",
        "--pd-disaggregation",
        "--prefill",
        prefill_url,
        "--decode",
        decode_url,
        "--host",
        host,
        "--port",
        lb_port,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    lb_url = f"http://{host}:{lb_port}"
    wait_for_http_ready(lb_url + "/health", timeout=_LAUNCH_TIMEOUT, process=proc)
    return proc, lb_url


def _launch_pd(*, prefill_args=None, decode_args=None):
    """Launch full PD stack. Returns (prefill_proc, decode_proc, lb_proc, lb_url,
    prefill_err_path, decode_err_path).
    """
    ports = _pd_ports()
    prefill_url = f"http://{ports['host']}:{ports['prefill']}"
    decode_url = f"http://{ports['host']}:{ports['decode']}"

    os.environ["MC_TCP_ENABLE_CONNECTION_POOL"] = "true"

    pp, pe = _launch_pd_server(
        prefill_url,
        mode="prefill",
        bootstrap_port=ports["bootstrap"],
        extra_args=prefill_args or [],
        base_gpu_id="0",
    )
    dp, de = _launch_pd_server(
        decode_url,
        mode="decode",
        bootstrap_port=ports["bootstrap"],
        extra_args=decode_args or [],
        base_gpu_id="1",
    )
    lp, lb_url = _launch_router(prefill_url, decode_url, ports["host"], ports["lb"])
    return pp, dp, lp, lb_url, pe, de


def _cleanup_pd(pp, dp, lp, pe, de):
    for proc in (lp, dp, pp):
        if proc:
            kill_process_tree(proc.pid)
    for path in (pe, de):
        try:
            os.remove(path)
        except OSError:
            pass


def _read_log(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _parse_capture_bs(log_text: str) -> Optional[List[int]]:
    for line in log_text.splitlines():
        m = _BS_LOG_RE.search(line)
        if m:
            return [int(x.strip()) for x in m.group(1).split(",")]
    return None


def _parse_graph_memory_gb(log_text: str) -> Optional[float]:
    for line in log_text.splitlines():
        m = _MEM_LOG_RE.search(line)
        if m:
            return float(m.group(1))
    return None


def _has_graph_begin(log_text: str) -> bool:
    return any("graph begin" in line for line in log_text.splitlines())


def _run_bench(base_url):
    bench_args = get_benchmark_args(
        base_url=base_url,
        backend="sglang",
        dataset_name="random",
        tokenizer=MODEL,
        num_prompts=10,
        random_input_len=256,
        random_output_len=32,
        request_rate=float("inf"),
    )
    bench_args.warmup_requests = 0
    return run_benchmark(bench_args)


class TestCudaGraphBsPD(CustomTestCase):
    """Testcase: verify per-phase CUDA-graph BS parameters in PD disaggregation.

    PD hook (pd_disaggregation_hook.py:77) force-disables CG on the prefill
    server, so --cuda-graph-max-bs-decode / --cuda-graph-bs-decode are parsed
    but never trigger graph capture on the prefill side.  The decode server
    captures CUDA graphs normally and respects the parameters.

    All tests launch a full PD stack (prefill + decode + LB), send traffic
    through the LB with bench_serving, and verify CG behaviour on each side
    independently.

    [Test Category] Parameter
    [Test Target] --cuda-graph-max-bs-decode; --cuda-graph-max-bs-prefill;
                  --cuda-graph-bs-decode; --cuda-graph-bs-prefill;
                  --disaggregation-mode
    """

    # max_bs only, bs auto-generated on decode side
    def test_max_bs_auto_generates_bs(self):
        pp, dp, lp, lb_url, pe, de = _launch_pd(
            decode_args=["--cuda-graph-max-bs-decode", "8"],
        )
        try:
            res = _run_bench(lb_url)
            self.assertEqual(res["completed"], 10)

            prefill_log = _read_log(pe)
            decode_log = _read_log(de)
        finally:
            _cleanup_pd(pp, dp, lp, pe, de)

        # Prefill: CG disabled by PD hook
        self.assertFalse(
            _has_graph_begin(prefill_log), "Prefill CG must be disabled by PD hook"
        )
        # Decode: bs auto-generated, all ≤ 8
        decode_bs = _parse_capture_bs(decode_log)
        self.assertIsNotNone(decode_bs, "Expected capture bs in decode log")
        self.assertEqual(max(decode_bs), 8)
        self.assertTrue(all(b <= 8 for b in decode_bs))

    # explicit bs only, max_bs derived on decode side
    def test_explicit_bs_derives_max_bs(self):
        pp, dp, lp, lb_url, pe, de = _launch_pd(
            decode_args=["--cuda-graph-bs-decode", "1", "2", "4", "8"],
        )
        try:
            res = _run_bench(lb_url)
            self.assertEqual(res["completed"], 10)
            decode_log = _read_log(de)
            prefill_log = _read_log(pe)
        finally:
            _cleanup_pd(pp, dp, lp, pe, de)

        self.assertFalse(_has_graph_begin(prefill_log))
        decode_bs = _parse_capture_bs(decode_log)
        self.assertEqual(decode_bs, [1, 2, 4, 8])
        mem = _parse_graph_memory_gb(decode_log)
        self.assertIsNotNone(mem)
        self.assertGreater(mem, 0)

    # both max_bs and bs set, max_bs silently overwritten
    def test_max_bs_overwritten_when_bs_set(self):
        pp, dp, lp, lb_url, pe, de = _launch_pd(
            decode_args=[
                "--cuda-graph-max-bs-decode",
                "4",
                "--cuda-graph-bs-decode",
                "1",
                "2",
                "8",
            ],
        )
        try:
            res = _run_bench(lb_url)
            self.assertEqual(res["completed"], 10)
            decode_log = _read_log(de)
            prefill_log = _read_log(pe)
        finally:
            _cleanup_pd(pp, dp, lp, pe, de)

        self.assertFalse(_has_graph_begin(prefill_log))
        decode_bs = _parse_capture_bs(decode_log)
        self.assertEqual(decode_bs, [1, 2, 8])
        self.assertEqual(max(decode_bs), 8, "max_bs should be 8 (overwritten), not 4")

    # disable cuda graph padding, sequential bs generated
    def test_disable_padding_sequential_bs(self):
        pp, dp, lp, lb_url, pe, de = _launch_pd(
            decode_args=[
                "--cuda-graph-max-bs-decode",
                "8",
                "--disable-cuda-graph-padding",
            ],
        )
        try:
            res = _run_bench(lb_url)
            self.assertEqual(res["completed"], 10)
            decode_log = _read_log(de)
            prefill_log = _read_log(pe)
        finally:
            _cleanup_pd(pp, dp, lp, pe, de)

        self.assertFalse(_has_graph_begin(prefill_log))
        decode_bs = _parse_capture_bs(decode_log)
        self.assertEqual(decode_bs, list(range(1, 9)))

    # cuda graph disabled, no graph capture, serving works
    def test_disable_cuda_graph_serving_works(self):
        pp, dp, lp, lb_url, pe, de = _launch_pd(
            decode_args=["--cuda-graph-max-bs-decode", "8", "--disable-cuda-graph"],
        )
        try:
            res = _run_bench(lb_url)
            self.assertEqual(res["completed"], 10)
            decode_log = _read_log(de)
            prefill_log = _read_log(pe)
        finally:
            _cleanup_pd(pp, dp, lp, pe, de)

        self.assertFalse(
            _has_graph_begin(prefill_log), "Prefill CG must be disabled by PD hook"
        )
        self.assertFalse(
            _has_graph_begin(decode_log),
            "Decode CG must be disabled by --disable-cuda-graph",
        )

    # prefill CG disabled + decode CG behaviour verified by tests above

    # TTFT comparison with different max_bs values
    def test_max_bs_ttft_comparison(self):
        # max_bs=1
        pp1, dp1, lp1, lb1, pe1, de1 = _launch_pd(
            decode_args=["--cuda-graph-max-bs-decode", "1"],
        )
        try:
            r1 = _run_bench(lb1)
            self.assertEqual(r1["completed"], 10)
        finally:
            _cleanup_pd(pp1, dp1, lp1, pe1, de1)

        # max_bs=8
        pp8, dp8, lp8, lb8, pe8, de8 = _launch_pd(
            decode_args=["--cuda-graph-max-bs-decode", "8"],
        )
        try:
            r8 = _run_bench(lb8)
            self.assertEqual(r8["completed"], 10)
        finally:
            _cleanup_pd(pp8, dp8, lp8, pe8, de8)

        t1, t8 = r1["mean_ttft_ms"], r8["mean_ttft_ms"]
        p1, p8 = r1["p99_ttft_ms"], r8["p99_ttft_ms"]
        print(
            f"\n=== TTFT comparison (PD mode): max_bs=1 vs max_bs=8 ===\n"
            f"  Mean TTFT: {t1:.1f} ms (max_bs=1) vs {t8:.1f} ms (max_bs=8)\n"
            f"  P99  TTFT: {p1:.1f} ms (max_bs=1) vs {p8:.1f} ms (max_bs=8)"
        )
        self.assertGreater(t1, 0)
        self.assertGreater(t8, 0)


if __name__ == "__main__":
    unittest.main()
