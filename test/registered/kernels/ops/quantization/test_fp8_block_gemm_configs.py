import json
import re
import sys
from pathlib import Path

import pytest
import torch
import triton

from sglang.kernels.ops.quantization import fp8_kernel
from sglang.kernels.ops.quantization.fp8_kernel import (
    _w8a8_block_fp8_matmul,
    get_w8a8_block_fp8_configs,
)
from sglang.srt.utils import get_device, get_device_name
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

REPO_ROOT = Path(__file__).parents[5]
CONFIGS_DIR = Path(fp8_kernel.__file__).parent / "configs"

# DeepSeek-V4.1-Flash 的非专家权重：fp8 e4m3 + ue8m0，block_shape [32, 32]。
# （这是 W8A8 block-FP8 GEMM 在 H20-3e 上占 decode 38.6% / prefill 22.7% 的那个 shape。）
DSV41_BLOCK_SHAPE = (32, 32)
# 服务启动时实际查表的全部 (N, K)：从 `--log-level info` 的
# "Using configuration from ..." / "Config file not found at ..." 去重得到。
DSV41_SHAPES = [
    (512, 5120), (576, 5120), (1280, 5120), (1536, 5120), (1792, 5120),
    (5120, 5120), (4096, 1280), (5120, 1024), (5120, 15360), (25600, 6144),
    (5120, 288),
]


_NAME_RE = re.compile(
    r"^N=(?P<N>\d+),K=(?P<K>\d+),device_name=(?P<dev>.+?),"
    r"dtype=(?P<dtype>[a-z0-9_]+)"
    r"(?:,block_shape=\[(?P<bn>\d+),\s*(?P<bk>\d+)\])?$"
)


def _parse_config_name(path: Path):
    """`N=<n>,K=<k>,device_name=<dev>,dtype=fp8_w8a8,block_shape=[<bn>, <bk>].json`

    注意 block_shape 里也有逗号，不能用 `split(",")`。
    """
    m = _NAME_RE.match(path.stem)
    assert m, f"无法解析配置表文件名: {path.name}"
    block = (int(m["bn"]), int(m["bk"])) if m["bn"] else None
    return int(m["N"]), int(m["K"]), m["dev"], m["dtype"], block


def _config_files():
    return sorted(CONFIGS_DIR.glob("*.json"))


def _invariant_violations(config: dict, block_n: int, block_k: int):
    """`_w8a8_block_fp8_matmul` 对 BLOCK_SIZE_* 的隐含约束。

    kernel 里与 scale 步进相关的只有 K 方向：

        n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K      # group_k == block_k
        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)

    所以 BLOCK_SIZE_K 必须整除 block_k（否则 `n_tiles_k_per_group_k` 归零，
    取模直接除零；即便不为零，scale 步进也会错位）。

    N 方向**不需要**整除：`offs_bsn = offs_bn // group_n` 是 gather 语义，
    BLOCK_SIZE_N 小于 block_n 时同一个量化块被多个 tile 重复读取，结果依然正确
    （随仓库发布的 int8_w8a8 A100/A800 表就用了 BLOCK_SIZE_N=32/64 < block_n=128）。

    再加上 `tl.dot` 的下界要求（M/N 方向 BLOCK_SIZE >= 16，K 方向 >= 32）。
    """
    bad = []
    if block_k % config["BLOCK_SIZE_K"] != 0:
        bad.append(f"block_k({block_k}) % BLOCK_SIZE_K({config['BLOCK_SIZE_K']}) != 0")
    if config["BLOCK_SIZE_K"] < 32:
        bad.append(f"BLOCK_SIZE_K({config['BLOCK_SIZE_K']}) < 32 (tl.dot)")
    if config["BLOCK_SIZE_M"] < 16:
        bad.append(f"BLOCK_SIZE_M({config['BLOCK_SIZE_M']}) < 16 (tl.dot)")
    if config["BLOCK_SIZE_N"] < 16:
        bad.append(f"BLOCK_SIZE_N({config['BLOCK_SIZE_N']}) < 16 (tl.dot)")
    return bad


def test_shipped_block_fp8_configs_satisfy_kernel_invariants():
    """所有随仓库发布的 W8A8 block-FP8 配置表都必须满足 kernel 的隐含约束。

    回归背景（2026-09-23）：`tuning_block_wise_kernel.py` 的搜索空间只有
    `BLOCK_SIZE_K ∈ {64, 128}`，而 block_k=32（DS-V4.1 的权重）要求
    `block_k % BLOCK_SIZE_K == 0` → 过滤后空间为空，调优器直接
    `assert best_config is not None` 挂掉。这个测试把那条约束钉在**配置表**上。
    """
    files = _config_files()
    assert files, f"{CONFIGS_DIR} 里没有任何配置表"
    checked = 0
    problems = []
    for path in files:
        _, _, _, dtype, block = _parse_config_name(path)
        if block is None:
            continue  # *_channelwise 之类没有 block_shape，不受本约束
        if not (dtype.startswith("fp8_w8a8") or dtype.startswith("int8_w8a8")):
            continue
        block_n, block_k = block
        table = json.loads(path.read_text())
        assert table, f"{path.name} 是空表"
        for m, config in table.items():
            assert int(m) > 0, f"{path.name}: 非法 batch size {m}"
            for msg in _invariant_violations(config, block_n, block_k):
                problems.append(f"{path.name} M={m}: {msg}")
            checked += 1
    assert not problems, "配置表违反 kernel 约束：\n  " + "\n  ".join(problems[:20])
    assert checked > 0


def test_dsv41_block_shape_configs_are_present_for_this_device():
    """若当前设备是 H20-3e，则 DS-V4.1 的 10 个 shape 都必须有配置表。

    没有表时 `get_w8a8_block_fp8_configs` 会静默回退默认 config
    （BLOCK_SIZE_N=32 / num_stages=3），服务启动日志里会有
    "Config file not found ... Performance might be sub-optimal!"。
    """
    device = get_device_name().replace(" ", "_")
    if device != "NVIDIA_H20-3e":
        pytest.skip(f"当前设备 {device} 不是 H20-3e")
    missing = [
        (N, K)
        for (N, K) in DSV41_SHAPES
        if not (
            CONFIGS_DIR
            / f"N={N},K={K},device_name={device},dtype=fp8_w8a8,"
            f"block_shape=[{DSV41_BLOCK_SHAPE[0]}, {DSV41_BLOCK_SHAPE[1]}].json"
        ).is_file()
    ]
    assert not missing, f"H20-3e 缺少 DS-V4.1 配置表: {missing}"


def test_config_lookup_finds_shipped_table_for_this_device():
    """`get_w8a8_block_fp8_configs()` 必须能按真实 `get_device_name()` 找到已发布的表。

    回归背景：查表文件名用的是 `get_device_name().replace(" ", "_")`。H20-3e 的
    device name 带 `-3e` 后缀，而仓库里只有 `NVIDIA_H20` / `NVIDIA_H200` 的表 →
    按名字查不到、静默回退默认 config。
    """
    device = get_device_name().replace(" ", "_")
    candidates = [
        p for p in _config_files()
        if _parse_config_name(p)[2] == device and _parse_config_name(p)[3].startswith("fp8_w8a8")
    ]
    if not candidates:
        pytest.skip(f"当前设备 {device} 没有随仓库发布的 fp8_w8a8 配置表")
    N, K, _, _, (block_n, block_k) = _parse_config_name(candidates[0])
    found = get_w8a8_block_fp8_configs(N, K, block_n, block_k)
    assert found, (
        f"{device} 上 N={N},K={K},block_shape=[{block_n}, {block_k}] 查表失败——"
        "文件名与 get_device_name() 的约定不一致"
    )
    assert all(_invariant_violations(c, block_n, block_k) == [] for c in found.values())


@pytest.mark.xfail(
    reason="上游 get_configs_compute_bound() 只有 BLOCK_SIZE_K∈{64,128}，"
           "block_k=32 过滤后为空（调优器对 DS-V4.1 的 32×32 权重不可用）；"
           "workaround 见 benchmark 侧补候选",
    strict=False,
)
def test_tuning_search_space_is_nonempty_for_every_block_k():
    """调优器的候选集必须对每种 block_k 都非空（否则 assert best_config is not None 挂掉）。"""
    bench_dir = REPO_ROOT / "benchmark" / "kernels" / "quantization"
    sys.path.insert(0, str(bench_dir))
    from tuning_block_wise_kernel import get_configs_compute_bound

    space = get_configs_compute_bound()
    for block_k in (32, 64, 128):
        kept = [c for c in space if block_k % c["BLOCK_SIZE_K"] == 0]
        assert kept, f"block_k={block_k} 的搜索空间为空"


def _reference_block_fp8_matmul(A, B, As, Bs, block_n, block_k):
    """独立参考：按 32×32 块反量化后做 fp64 matmul，再按输出 dtype 舍入。"""
    M, K = A.shape
    N = B.shape[0]
    a = A.double() * As.repeat_interleave(block_k, dim=1).double()
    b = (
        B.double()
        * Bs.repeat_interleave(block_n, dim=0).repeat_interleave(block_k, dim=1).double()
    )
    return (a @ b.T).to(torch.bfloat16)


def _dyadic_inputs(M, N, K, block_n, block_k, seed=719):
    """用二进制可精确表示的输入，隔离 fp32 累加顺序的影响。"""
    g = torch.Generator(device="cuda").manual_seed(seed)
    # e4m3 能精确表示 k/4（k ∈ [-8, 8]）
    A = (torch.randint(-8, 9, (M, K), generator=g, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    B = (torch.randint(-8, 9, (N, K), generator=g, device="cuda") / 4).to(
        torch.float8_e4m3fn
    )
    # scale 取 2 的幂 → 反量化精确
    As = torch.ldexp(
        torch.ones(M, K // block_k, device="cuda"), -torch.randint(1, 9, (M, K // block_k), generator=g, device="cuda")
    )
    Bs = torch.ldexp(
        torch.ones(N // block_n, K // block_k, device="cuda"),
        -torch.randint(1, 9, (N // block_n, K // block_k), generator=g, device="cuda"),
    )
    return A, B, As, Bs


@pytest.mark.parametrize(
    "M,N,K,block_n,block_k",
    [
        (1, 512, 5120, 32, 32),      # DS-V4.1 wkv（decode）
        (64, 5120, 5120, 32, 32),    # DS-V4.1 最大 shape（decode 上限）
        (128, 1280, 5120, 32, 32),   # DS-V4.1 wq_a
        (33, 576, 5120, 32, 32),     # 非 2 次幂 M
        (16, 5120, 288, 32, 32),     # K=288（K/BLOCK_SIZE_K=9，非 2 次幂）
        (8, 256, 256, 128, 128),     # 其它 block_shape 也要能跑
    ],
)
def test_block_fp8_matmul_matches_reference(M, N, K, block_n, block_k):
    """用**配置表里的 config**跑 kernel，结果必须与独立参考一致。

    这同时覆盖了"调优出来的 config 是否数值正确"——`tune()` 只看时间、不看对错，
    BLOCK_SIZE_K 选错会让 `group_k // BLOCK_SIZE_K` 归零、结果直接错掉。
    """
    configs = get_w8a8_block_fp8_configs(N, K, block_n, block_k)
    config = (
        configs[min(configs, key=lambda m: abs(m - M))]
        if configs
        else {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    )
    assert _invariant_violations(config, block_n, block_k) == []

    A, B, As, Bs = _dyadic_inputs(M, N, K, block_n, block_k)
    C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    _w8a8_block_fp8_matmul[(triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),)](
        A, B, C, As, Bs, M, N, K, block_n, block_k,
        A.stride(-2), A.stride(-1), B.stride(1), B.stride(0), C.stride(-2), C.stride(-1),
        As.stride(-2), As.stride(-1), Bs.stride(1), Bs.stride(0),
        needs_masking=bool(K % config["BLOCK_SIZE_K"] != 0),
        **config,
    )
    expected = _reference_block_fp8_matmul(A, B, As, Bs, block_n, block_k)
    torch.testing.assert_close(C, expected, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
