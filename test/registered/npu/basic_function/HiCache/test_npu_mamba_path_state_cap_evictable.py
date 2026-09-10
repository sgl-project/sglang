"""
E2E 黑盒验证 ``--mamba-max-states-per-path`` 的显存生效性（不修改任何功能代码）。

核心观测指标
-------------
使用 server 已暴露的 prometheus gauge ``sglang:mamba_evictable_tokens``
（= radix 树中可淘汰的 cached mamba state 数，见 metrics_collector.py 与
mamba_radix_cache.mamba_evictable_size）。

为什么用这个指标而不是 ``sglang:mamba_usage``:
  * ``mamba_usage``/``mamba_num_used`` 只统计 running 请求占用的 ACTIVE slot；
    淘汰树中 cached state 时 ``available +1`` 与 ``evictable -1`` 相互抵消，
    usage 保持不变 -> 无法区分不同 cap。
  * ``mamba_evictable_tokens`` 直接统计树中 cached state 数，path-cap 每淘汰
    一个中间节点就令它 -1，是观测淘汰的唯一敏感指标。

原理
-----
  * radix cache 策略默认 ``auto``，对 Qwen3.5（GDN）解析为 ``extra_buffer``
    （见 arg_groups/overrides.py）。extra_buffer 会在路径中间节点缓存 state，
    从而让 path-cap 有中间节点可淘汰；no_buffer 只在 full-sequence leaf 存
    state，单链路径无中间 state 可淘汰，cap 无法触发。
  * 多轮"共享前缀"请求：每轮在前一轮前缀上追加内容，命中已有缓存后 append
    新节点并 donate 一个 state，在同一条路径上累积多个 state 节点。
  * cap 越小 -> 路径上保留的 state 越少 -> ``mamba_evictable_tokens`` 越小。
  * 固定工作负载、仅扫描 cap 值，验证该指标随 cap 单调不减，反证淘汰生效。

前提
-----
  * ``--enable-metrics`` 必须开启，否则 /metrics 不可访问。
  * 不启用 ``--enable-int8-mamba-checkpoint``（int8 下 mamba_evictable 被
    强制置 0，指标失效）。
  * mamba radix cache 由模型架构自动启用（uses_mamba_radix_cache），无需
    额外 env。若所在环境的 auto 解析为 no_buffer，需显式加
    ``--mamba-radix-cache-strategy extra_buffer`` 才能触发 path-cap。
  * 需 2 卡（Qwen3.5-35B-A3B，TP=2）。

用法（NPU 手动运行）:
    python3 -m unittest test_npu_mamba_path_state_cap_evictable

[Test Category] HiCache
"""

import os
import re
import time
import unittest

import requests

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
MODEL = os.environ.get("MAMBA_MODEL_PATH", "Qwen/Qwen3.5-35B-A3B")
BASE_URL = DEFAULT_URL_FOR_TEST
METRIC = "sglang:mamba_evictable_tokens"
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
# 可选：本地手动运行时指定起始 NPU 卡（CI 不设置，交由调度器分配）。
BASE_GPU_ID = os.environ.get("BASE_GPU_ID")

# 每轮"新增"的词数。mamba state 只在跨过 track boundary（seq_len % 256 == 0）
# 时才 donate（见 batch_result_processor._mamba_prefix_cache_update）。每轮
# 新增 ~500 词（~600 token）可跨过多个 256 边界，保证路径上累积多个可淘汰
# state。若每轮总长 < 256 且不跨边界，全程 0 个 state 落盘，evictable 恒 0。
PER_TURN_ADDED_WORDS = 500
NUM_TURNS = 6
# 每轮生成的 output token：稍长一些，让 decode 阶段也跨过边界、辅助 donate。
MAX_NEW_TOKENS = 32
# 扫描的 cap 值：1（最严格，只保留 tail）-> 3 -> 64（几乎不淘汰，作上限参考）
CAPS = [1, 3, 64]

_BASE_SEGMENT = (
    "The quick brown fox jumps over the lazy dog and then runs across the "
    "sunny meadow while the birds sing in the tall green trees. "
)


def _build_shared_prefix_turns(
    num_turns=NUM_TURNS, per_turn_added_words=PER_TURN_ADDED_WORDS
):
    """构造多轮共享前缀，每轮在前一轮基础上追加内容。

    返回形如 [T1, T2, ..., Tn]，其中 Ti 是 T(i-1) 的严格超集：Ti 的词数 =
    i * per_turn_added_words。这让每轮都在上轮缓存的节点后 append 新节点
    （而非重复同一段文本），并在同一条 radix 路径上持续累积 mamba state。
    """
    turns = []
    prefix = ""
    for i in range(1, num_turns + 1):
        while len(prefix.split()) < i * per_turn_added_words:
            prefix += _BASE_SEGMENT
        turns.append(prefix)
    return turns


def _wait_metrics_ready(base_url, timeout=120):
    """确认 /metrics 真实可访问后再进入负载阶段。

    若一直拿不到 200，说明 --enable-metrics 未生效（或命中了未开 metrics 的
    旧 server），给出可操作的报错，避免把问题掩盖到负载之后。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(base_url + "/metrics", timeout=5)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise AssertionError(
        f"/metrics 持续返回非 200（端口 {base_url}）。请检查："
        "(1) 是否传了 --enable-metrics；(2) 该端口是否被未开 metrics 的"
        "遗留 server 占用（可先 pkill sglang 或换端口重试）。"
    )


def _get_metric_lines(base_url, name, timeout=10):
    """从 /metrics 拉取某个 metric 的全部原始行。

    TP2 下同一 gauge 会按 tp_rank/pp_rank/moe_ep_rank 拆分多行输出
    （如 sglang:mamba_evictable_tokens{tp_rank="0",...} 0.0 与
    tp_rank="1" 一行）。返回 (值列表, 完整文本)。
    """
    resp = requests.get(base_url + "/metrics", timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    values = []
    for line in text.splitlines():
        if line.startswith(name):
            m = re.match(rf"{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.e+-]+)", line)
            if m:
                values.append(float(m.group(1)))
    return values, text


def _get_metric_value(base_url, name, timeout=10):
    """从 /metrics 拉取 gauge 的跨 rank 求和值。

    TP2 下 mamba state 可能只落在部分 rank 上，只读第一行会误得 0；
    这里对同名的所有 rank 行求和，保证读数稳定（跨 cap 比较时比例一致）。
    """
    values, _ = _get_metric_lines(base_url, name, timeout=timeout)
    if not values:
        raise AssertionError(f"metric '{name}' not found in /metrics")
    return sum(values)


def _dump_mamba_metrics(base_url, cap):
    """打印全部 mamba 相关原始指标，便于定位为何某 rank 为 0。"""
    for name in (
        "sglang:mamba_evictable_tokens",
        "sglang:mamba_available_tokens",
        "sglang:mamba_used_tokens",
    ):
        values, text = _get_metric_lines(base_url, name)
        lines = [l for l in text.splitlines() if l.startswith(name)]
        print(f"[cap={cap}] {name}  (sum={sum(values)})")
        for l in lines:
            print(f"        {l}")


def _wait_metric_stable(base_url, name, stable_reads=2, poll_interval=2.0, timeout=30):
    """轮询直到连续两次读数相同，返回该稳定值（等待 pool stats 周期刷新）。"""
    last, stable = None, 0
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            cur = _get_metric_value(base_url, name)
        except AssertionError:
            cur = None
        if cur == last and cur is not None:
            stable += 1
            if stable >= stable_reads:
                return cur
        else:
            stable = 0
        last = cur
        time.sleep(poll_interval)
    return last


class TestMambaPathStateCapEvictable(CustomTestCase):
    """对比不同 cap 值下 cached mamba state 数，反证淘汰生效。"""

    def _send_workload(self, base_url):
        """发送多轮共享前缀请求，全部阻塞完成后返回。"""
        for turn in _build_shared_prefix_turns():
            requests.post(
                base_url + "/generate",
                json={
                    "text": turn,
                    "sampling_params": {
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "temperature": 0,
                    },
                },
                timeout=180,
            ).raise_for_status()

    def _measure_evictable_tokens(self, cap):
        """在固定端口启动指定 cap 的 server，跑相同负载，返回空闲时 evictable tokens。

        容器内端口由网络命名空间隔离，用固定端口即可；_wait_metrics_ready
        仍负责在端口被残留 server 占用时快速报错（/generate 正常但 /metrics
        404 的场景）。
        """
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
            "--enable-metrics",
            "--mamba-max-states-per-path",
            str(cap),
        ]
        if BASE_GPU_ID is not None:
            other_args += ["--base-gpu-id", BASE_GPU_ID]

        process = popen_launch_server(
            MODEL,
            BASE_URL,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=NPU_ENV,
            device="npu",
        )
        try:
            # 先确认 /metrics 真实可用（排除端口被旧 server 占用的情况），
            # 再进入负载，避免把问题掩盖到负载之后。
            _wait_metrics_ready(BASE_URL)
            self._send_workload(BASE_URL)
            # 请求全部完成后，pool stats 异步刷新，需等待指标稳定
            stable = _wait_metric_stable(BASE_URL, METRIC)
            _dump_mamba_metrics(BASE_URL, cap)  # 打印各 rank 原始值便于诊断
            return stable
        finally:
            terminate_and_kill_process_tree(process, terminate_timeout=60)
            time.sleep(2)

    def test_evictable_tokens_monotonic_in_cap(self):
        """cap 越小，路径上保留的 cached mamba state 越少，evictable 指标越小。

        相同工作负载下，该指标应随 cap 单调不减（严格递减更理想）。任何
        一个 cap 值使 evictable 反而更大，都说明淘汰路径未按预期工作。
        """
        values = {}
        for cap in CAPS:
            values[cap] = self._measure_evictable_tokens(cap)
            print(f"cap={cap} -> {METRIC}={values[cap]}")

        # 单调性：cap 依次增大，evictable 不得减少
        for a, b in zip(CAPS, CAPS[1:]):
            self.assertGreaterEqual(
                values[b],
                values[a],
                f"cap={b} 应保留 >= cap={a} 的 cached state "
                f"({values[a]} -> {values[b]})，淘汰未按预期生效",
            )

        # 最强断言：严格限制 cap=1 必须显著小于 cap=64（淘汰确实发生）
        self.assertLess(
            values[CAPS[0]],
            values[CAPS[-1]],
            f"cap=1({values[CAPS[0]]}) 应显著小于 cap={CAPS[-1]}"
            f"({values[CAPS[-1]]})，证明 path-cap 淘汰生效",
        )


if __name__ == "__main__":
    unittest.main()
