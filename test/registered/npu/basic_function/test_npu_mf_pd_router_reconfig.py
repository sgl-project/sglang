"""MemFabric PD 分离服务 router 动态重组功能测试（自动化自 MemFabric_PDSeparate_Func_01/02/03）。

覆盖三个功能用例，共享一套预拉起的 worker 实例池（1P+1P+1D+1D，TP=1，共 4 卡）：
  - test_01_scale_up_1p1d_to_2p2d   -> Func_01: 1P1D -> 终止旧 router -> 新增 P/D 各一个 -> 2P2D 新 router
  - test_02_kill_single_prefill     -> Func_02: 2P1D -> 终止旧 router -> kill 单个 P 实例 -> 1P1D 新 router
  - test_03_kill_single_decode      -> Func_03: 1P2D -> 终止旧 router -> kill 单个 D 实例 -> 1P1D 新 router

环境对齐手工用例（.github/workflows/1.log）：
  - Qwen3-4B，TP=1，单机 A3，ASCEND_MF_TRANSFER_PROTOCOL=sdma
  - prefill 实例带 --enable-hierarchical-cache --max-total-tokens 8192
  - router 使用 --pd-disaggregation --policy random

注意：用例之间存在顺序依赖（test_02 杀 prefill_2、test_03 杀 decode_2），
     必须按编号顺序执行（unittest 默认按方法名字母序，已用零填充编号保证）。
"""

import os
import time
import unittest
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_4B_WEIGHTS_PATH,
    logger,
    popen_with_error_check,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=1500,
    suite="full-4-npu-a3",
    nightly=True,
)

MODEL_PATH = QWEN3_4B_WEIGHTS_PATH
MEM_FRACTION = "0.8"

# PD 分离 server 公共参数（对齐 1.log 手工用例）
COMMON_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    MEM_FRACTION,
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
]

# 4 卡布局：2 个 prefill + 2 个 decode，各占 1 卡（TP=1）
WORKER_LAYOUT = {
    "prefill_1": {"mode": "prefill", "base_gpu_id": 0},
    "prefill_2": {"mode": "prefill", "base_gpu_id": 1},
    "decode_1": {"mode": "decode", "base_gpu_id": 2},
    "decode_2": {"mode": "decode", "base_gpu_id": 3},
}

PROMPT_TEXT = "The capital of France is"
EXPECTED_ANSWER = "Paris"


def send_request(url):
    """对 /generate 发送一条请求（等价于手工用例的 curl 命令），带超时防挂起。"""
    return requests.post(
        f"{url}/generate",
        json={
            "text": PROMPT_TEXT,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 100,
            },
        },
        timeout=60,
    )


class MFPDRouterReconfigTest(CustomTestCase):
    """验证 PD 分离服务在 router 重组 / worker 故障场景下的可用性。

    [Test Category] Functional
    [Test Target] MemFabric PD Disaggregation Router Reconfig (HA)
    ASCEND_MF_STORE_URL; ASCEND_MF_TRANSFER_PROTOCOL; --pd-disaggregation;
    --health-failure-threshold; --health-success-threshold
    """

    @classmethod
    def setUpClass(cls):
        cls._init_ports()
        cls.log_files = {}
        cls.processes = {}

        # 预拉起全部 4 个 worker 实例。
        # popen_launch_server 内部会阻塞等待每个实例就绪，串行拉起保证日志清晰、端口无竞争。
        for name in WORKER_LAYOUT:
            cls._start_worker(name)

    @classmethod
    def _init_ports(cls):
        """端口规划（基于 DEFAULT_URL_FOR_TEST 的基础端口 bp）：
        lb=bp, prefill_1=bp+100, prefill_2=bp+200, decode_1=bp+300, decode_2=bp+350,
        bootstrap: prefill_1=bp+400, prefill_2=bp+500, MF store=bp+600
        """
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed.hostname
        bp = int(parsed.port)

        cls.lb_port = bp
        cls.ports = {
            "prefill_1": bp + 100,
            "prefill_2": bp + 200,
            "decode_1": bp + 300,
            "decode_2": bp + 350,
        }
        cls.bootstrap_ports = {
            "prefill_1": bp + 400,
            "prefill_2": bp + 500,
        }
        cls.ascend_mf_store_url = f"tcp://{cls.base_host}:{bp + 600}"
        cls.urls = {
            name: f"http://{cls.base_host}:{port}" for name, port in cls.ports.items()
        }
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"

    @classmethod
    def _open_logs(cls, name):
        out = open(f"./{name}_out_log.txt", "w+", encoding="utf-8")
        err = open(f"./{name}_err_log.txt", "w+", encoding="utf-8")
        cls.log_files[name] = (out, err)
        return out, err

    @classmethod
    def _mf_env(cls):
        """PD 分离 + MemFabric 所需环境变量（对齐 1.log 手工用例）。
        A3 单机场景传输协议使用 sdma；其余 CANN/HCCL 环境变量由 CI 镜像环境提供。
        """
        return {
            **os.environ,
            "ASCEND_MF_STORE_URL": cls.ascend_mf_store_url,
            "ASCEND_MF_TRANSFER_PROTOCOL": "sdma",
            "ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE": "true",
            "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
            "TASK_QUEUE_ENABLE": "2",
            "STREAMS_PER_DEVICE": "32",
            "DP_ROUND_ROBIN": "1",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "48",
        }

    @classmethod
    def _start_worker(cls, name):
        """按 WORKER_LAYOUT 拉起单个 prefill/decode 实例。"""
        spec = WORKER_LAYOUT[name]
        out, err = cls._open_logs(name)

        args = COMMON_ARGS + [
            "--tp-size",
            "1",
            "--base-gpu-id",
            str(spec["base_gpu_id"]),
            "--disaggregation-transfer-backend",
            "ascend",
            "--disaggregation-mode",
            spec["mode"],
        ]

        # prefill 实例专属参数：bootstrap 端口必须两两不同（对齐 1.log 中 8998/8899 的做法）
        if spec["mode"] == "prefill":
            args += [
                "--disaggregation-bootstrap-port",
                str(cls.bootstrap_ports[name]),
                "--enable-hierarchical-cache",
                "--max-total-tokens",
                "8192",
                "--max-running-requests",
                "372",
            ]

        cls.processes[name] = popen_launch_server(
            MODEL_PATH,
            cls.urls[name],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=cls._mf_env(),
            return_stdout_stderr=(out, err),
        )

    @classmethod
    def _kill_router(cls):
        """终止当前 router（幂等）。worker 实例全部保留。"""
        proc = cls.processes.get("lb")
        if proc:
            try:
                kill_process_tree(proc.pid)
                logger.info(f"Killed old router pid={proc.pid}")
            except Exception as e:
                logger.warning(f"Failed to kill old router: {e}")
            cls.processes["lb"] = None
            time.sleep(5)

        # 重新打开 lb 日志，保证每次重组后的日志独立可读
        logs = cls.log_files.pop("lb", None)
        if logs:
            for f in logs:
                try:
                    f.close()
                except Exception:
                    pass

    @classmethod
    def _launch_router(cls, prefill_names, decode_names):
        """按指定拓扑拉起新 router 并等待就绪（自动先终止旧 router）。

        prefill_names: 注册到 router 的 prefill 实例名列表，如 ["prefill_1", "prefill_2"]
        decode_names:  注册到 router 的 decode 实例名列表
        """
        cls._kill_router()
        out, err = cls._open_logs("lb")

        cmd = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--policy",
            "random",
        ]
        for name in prefill_names:
            cmd += ["--prefill", cls.urls[name], str(cls.bootstrap_ports[name])]
        for name in decode_names:
            cmd += ["--decode", cls.urls[name]]

        cmd += [
            "--host",
            cls.base_host,
            "--port",
            str(cls.lb_port),
            "--health-failure-threshold",
            "2",
            "--health-success-threshold",
            "2",
            "--health-check-timeout-secs",
            "30",
            "--health-check-interval-secs",
            "15",
        ]

        cls.processes["lb"] = popen_with_error_check(
            cmd, return_stdout_stderr=(out, err)
        )
        cls._wait_ready(f"{cls.lb_url}/health")

    @classmethod
    def _wait_ready(cls, url, timeout=120):
        start = time.perf_counter()
        while True:
            try:
                if requests.get(url, timeout=10).status_code == 200:
                    logger.info(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")

            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        # 统一清理：先杀 router，再杀所有仍存活的 worker，防止残留进程挂住批量测试管道
        for name, proc in cls.processes.items():
            if proc:
                try:
                    kill_process_tree(proc.pid)
                    logger.info(f"Killed {name} pid={proc.pid}")
                except Exception as e:
                    logger.warning(f"Failed to kill {name}: {e}")

        time.sleep(5)

        for out, err in cls.log_files.values():
            for f in (out, err):
                try:
                    f.close()
                except Exception:
                    pass

        for name in list(cls.log_files.keys()):
            for suffix in ("out_log.txt", "err_log.txt"):
                path = f"./{name}_{suffix}"
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to remove {path}: {e}")

    # ---------------- 断言辅助 ----------------

    def _send_and_assert(self):
        """发送一条请求并断言 200 + 答案正确（对应手工用例的 E2/E4）。"""
        resp = send_request(self.lb_url)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(EXPECTED_ANSWER, resp.text)
        logger.info(resp.json())

    def _assert_workers_healthy(self, names):
        """直连 worker 的 /health 端口探活（绕过 router），验证"原有实例不中断"（E3）。"""
        for name in names:
            resp = requests.get(f"{self.urls[name]}/health", timeout=10)
            self.assertEqual(
                resp.status_code, 200, f"Worker {name} is not healthy"
            )
        logger.info(f"Workers {names} are healthy")

    def _kill_worker(self, name):
        """kill 指定 worker 并确认进程已终止。"""
        proc = self.processes.get(name)
        self.assertIsNotNone(proc, f"Worker {name} already killed")
        kill_process_tree(proc.pid)
        time.sleep(5)
        self.assertIsNotNone(proc.poll(), f"Worker {name} not terminated yet")
        self.processes[name] = None
        logger.info(f"Worker {name} killed")

    # ---------------- 测试用例 ----------------

    def test_01_scale_up_1p1d_to_2p2d(self):
        """Func_01: 1P1D -> 终止旧 router -> 新增 P/D 各一 -> 2P2D 新 router。

        S1 拉起 1P1D 服务; S2 发送请求; S3 终止旧 router, 拉起 2P2D 新 router
        E1 服务拉起成功; E2 请求 200; E3 原有实例不中断, 新 router 拉起成功; E4 请求 200
        """
        # S1: 初始 1P1D 拓扑（prefill_2/decode_2 已预拉起但未注册）
        self._launch_router(["prefill_1"], ["decode_1"])

        # E1: 初始服务就绪 + E2: 请求成功
        self._assert_workers_healthy(["prefill_1", "decode_1"])
        self._send_and_assert()

        # S3: 终止旧 router（_launch_router 内部先执行），扩容为 2P2D 新 router
        self._launch_router(["prefill_1", "prefill_2"], ["decode_1", "decode_2"])

        # E3: 原有实例（prefill_1/decode_1）不中断
        self._assert_workers_healthy(["prefill_1", "decode_1"])
        # E4: 扩容后请求成功（random 策略下请求应正常分发）
        self._send_and_assert()

    def test_02_kill_single_prefill(self):
        """Func_02: 2P1D -> 终止旧 router -> kill 单个 P 实例 -> 1P1D 新 router。

        E3: 观测到报错日志, 原有实例不中断, 服务不崩溃; E4: 新 router 下请求 200
        """
        # S1/S2: 2P1D 拓扑 + 请求成功
        self._launch_router(["prefill_1", "prefill_2"], ["decode_1"])
        self._send_and_assert()

        # S3: 终止旧 router 后手动 kill prefill_2
        self._kill_router()
        self._kill_worker("prefill_2")

        # E3: 原有实例不中断（服务不崩溃）
        self._assert_workers_healthy(["prefill_1", "decode_1", "decode_2"])

        # S3: 拉起 1P1D 新 router
        self._launch_router(["prefill_1"], ["decode_1"])

        # E4: 请求成功
        self._send_and_assert()

    def test_03_kill_single_decode(self):
        """Func_03: 1P2D -> 终止旧 router -> kill 单个 D 实例 -> 1P1D 新 router。

        E3: 观测到报错日志, 原有实例不中断, 服务不崩溃; E4: 新 router 下请求 200
        """
        # S1/S2: 1P2D 拓扑 + 请求成功
        self._launch_router(["prefill_1"], ["decode_1", "decode_2"])
        self._send_and_assert()

        # S3: 终止旧 router 后手动 kill decode_2
        self._kill_router()
        self._kill_worker("decode_2")

        # E3: 原有实例不中断（服务不崩溃）
        # 注意：prefill_2 已在 test_02 中被杀
        self._assert_workers_healthy(["prefill_1", "decode_1"])

        # S3: 拉起 1P1D 新 router
        self._launch_router(["prefill_1"], ["decode_1"])

        # E4: 请求成功
        self._send_and_assert()


if __name__ == "__main__":
    unittest.main()
