"""E2E (NPU): --enable-int8-mamba-checkpoint 的缓存容量扩展验证（参考 PR 28185）。

参考 PR 28185 (bench_int8_checkpoint_reuse.py) 的 WARM/PROBE 手法，验证 int8
checkpoint 池让"可缓存的 distinct 前缀数"约翻倍：

  * WARM：发送 K 个互不相同、长度跨过 mamba chunk 粒度(约 512)的前缀，
    各自在 mamba radix cache 中占用 cached state slot。
  * PROBE：重放每个前缀但换用不同的 suffix —— suffix 不命中，命中只覆盖
    前缀本身 —— 读取 meta_info.cached_tokens 计算 reuse_frac
    = sum(cached_tokens) / sum(prompt_tokens)。

原理：
  * 每个 distinct 前缀的 cached state 存在缓存池里。一旦 distinct 前缀数
    超过池容量，最早的 state 被淘汰，PROBE 时该前缀命中不到，reuse 崩塌。
  * 不开启 int8 时，cached state 存在 active bf16 池（--max-mamba-cache-size
    个 slot）；开启后存在独立的 int8 checkpoint 池，其 slot 数默认
    = 2 * max_mamba_cache_size（见 mamba_checkpoint_pool.py），因此
    int8 的"崩塌点"(reuse 开始下降的 K) 更远，约 2 倍。

断言：
  * 每个配置内 reuse_frac 随 K 单调不增（容量上限导致）。
  * int8 开启时的崩塌点 K 不早于关闭时（更晚崩塌）。
  * 强断言：扫描各 K，int8 相对关闭的最大 reuse 增益显著（说明关闭端已崩塌、
    开启端仍保持复用）。

[Test Category] Memory and Scheduling
"""

import os
import random
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ascend.test_ascend_utils import QWEN3_5_35B_A3B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_npu_ci(est_time=3600, suite="full-2-npu-a3", nightly=True)

# 模型路径可用环境变量覆盖，默认对齐本地手动脚本；CI 上按需注入。
MODEL = QWEN3_5_35B_A3B_WEIGHTS_PATH
BASE_URL = DEFAULT_URL_FOR_TEST
# 与手动脚本一致的 NPU 运行环境变量。
NPU_ENV = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1536",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "SGLANG_DEEPEP_BF16_DISPATCH": "1",
    "ENABLE_ASCEND_MOE_NZ": "1",
}
# 可选：本地多机复用脚本时指定起始 NPU 卡（CI 不设置，交由调度器分配）。
BASE_GPU_ID = os.environ.get("BASE_GPU_ID")

# 前缀需跨过 mamba chunk 粒度(约 512)，否则不可缓存、reuse 恒 0（PR 28185 备注）。
PREFIX_TOKENS = 1024
SUFFIX_TOKENS = 16
# active bf16 池 slot 数；int8 checkpoint 池默认 = 2 * 该值。
MAX_MAMBA_CACHE_SIZE = 256
# 扫描的 distinct 前缀数；reuse 崩塌点应落在该区间内（off ~256，on ~512）。
K_SCAN = [64, 128, 256, 512]
PARALLEL = 8
# reuse 低于该值视为"崩塌"（该 K 处缓存已溢出）。
COLLAPSE_THRESHOLD = 0.5
# int8 相对关闭的最大 reuse 增益须超过该值。
REUSE_GAP = 0.3


def _random_ids(length, seed):
    rng = random.Random(seed)
    return [rng.randint(1, 30000) for _ in range(length)]


def _make_requests():
    """预生成 K 组 WARM/PROBE 请求（确定性，与 int8 开关无关）。

    两次测量使用同一份请求，保证 off/on 对比公平。
    WARM：每个前缀 + 专属 suffix，独占一个 cached state slot。
    PROBE：同一前缀 + 不同 suffix，只命中前缀本身。
    """
    plan = {}
    for K in K_SCAN:
        # 每个前缀带唯一 tag 头 + 长随机段，保证互不相同。
        prefixes = [
            _random_ids(1, seed=1000 + i) + _random_ids(PREFIX_TOKENS, seed=2000 + i)
            for i in range(K)
        ]
        warm = [
            p + _random_ids(SUFFIX_TOKENS, seed=3000 + i)
            for i, p in enumerate(prefixes)
        ]
        probe = [
            p + _random_ids(SUFFIX_TOKENS, seed=4000 + i)
            for i, p in enumerate(prefixes)
        ]
        plan[K] = (warm, probe)
    return plan


def _generate(input_ids):
    resp = requests.post(
        BASE_URL + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 1,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def _flush():
    try:
        requests.post(BASE_URL + "/flush_cache", timeout=60)
    except requests.RequestException:
        pass
    time.sleep(1.5)


def _measure_config(int8_enabled, plan):
    """启动一个 server，扫描 K，返回 {K: reuse_frac}。"""
    other_args = [
        "--trust-remote-code",
        "--device",
        "npu",
        "--tp-size",
        "2",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        "0.8",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--max-mamba-cache-size",
        str(MAX_MAMBA_CACHE_SIZE),
    ]
    if BASE_GPU_ID is not None:
        other_args += ["--base-gpu-id", BASE_GPU_ID]
    if int8_enabled:
        other_args.append("--enable-int8-mamba-checkpoint")

    proc = popen_launch_server(
        MODEL,
        BASE_URL,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
        env=NPU_ENV,
        device="npu",
    )
    try:
        results = {}
        for K, (warm, probe) in plan.items():
            _flush()
            with ThreadPoolExecutor(PARALLEL) as ex:
                list(ex.map(_generate, warm))  # WARM
            with ThreadPoolExecutor(PARALLEL) as ex:
                metas = [r["meta_info"] for r in ex.map(_generate, probe)]

            sum_prompt = sum(m["prompt_tokens"] for m in metas)
            sum_cached = sum(m["cached_tokens"] for m in metas)
            reuse = sum_cached / max(1, sum_prompt)
            results[K] = reuse
            print(
                f"[int8={'on' if int8_enabled else 'off'}] K={K}: reuse_frac={reuse:.3f}"
            )
        return results
    finally:
        terminate_and_kill_process_tree(proc, terminate_timeout=60)
        time.sleep(2)


def _collapse_point(results):
    """reuse 首次低于阈值的 K；若都未崩塌，返回比最大 K 更大的哨兵值。"""
    for K in K_SCAN:
        if results[K] < COLLAPSE_THRESHOLD:
            return K
    return K_SCAN[-1] * 2


class TestNpuInt8MambaCheckpointReuse(CustomTestCase):
    """int8 checkpoint 池应让前缀缓存容量约翻倍，reuse 崩塌点更远。"""

    def test_int8_delays_prefix_reuse_collapse(self):
        plan = _make_requests()
        off = _measure_config(int8_enabled=False, plan=plan)
        on = _measure_config(int8_enabled=True, plan=plan)

        print(f"off={off}")
        print(f"on ={on}")
        print(f"collapse point: off={_collapse_point(off)}, on={_collapse_point(on)}")

        # 1) 每个配置内 reuse 随 K 单调不增（容量上限）。
        for results, tag in ((off, "off"), (on, "on")):
            for a, b in zip(K_SCAN, K_SCAN[1:]):
                self.assertGreaterEqual(
                    results[a] + 1e-6,
                    results[b],
                    f"[{tag}] reuse should be non-increasing in K "
                    f"({results[a]:.3f}@{a} -> {results[b]:.3f}@{b})",
                )

        # 2) int8 崩塌点不早于关闭时（更晚崩塌）。
        self.assertGreaterEqual(
            _collapse_point(on),
            _collapse_point(off),
            "int8 checkpoint pool should not collapse earlier than bf16",
        )

        # 3) 强断言：扫描各 K 取 int8 相对关闭的最大 reuse 增益，须显著。
        #    不依赖固定 K（崩塌点位置受每前缀 slot 占用影响而浮动）。
        diffs = {K: on[K] - off[K] for K in K_SCAN}
        max_k = max(diffs, key=lambda k: diffs[k])
        max_diff = diffs[max_k]
        print(f"max reuse gain: {max_diff:.3f} at K={max_k}")
        self.assertGreater(
            max_diff,
            REUSE_GAP,
            f"expected int8 to keep reuse higher than off at some K by > {REUSE_GAP}, "
            f"max gain {max_diff:.3f} @K={max_k}",
        )


if __name__ == "__main__":
    unittest.main()
