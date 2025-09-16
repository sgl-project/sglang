import triton
import logging

import torch
from typing import Any, List, Dict

import torch.distributed as dist

logger = logging.getLogger(__name__)


def execute():
    local_metrics = {bench_name: _compute_local_metric(bench_name) for bench_name in _BENCH_NAMES}

    all_metrics = [None for _ in range(dist.get_world_size())]
    dist.gather_object(local_metrics, all_metrics if dist.get_rank() == 0 else None)

    if dist.get_rank() == 0:
        _analyze_metrics(all_metrics)


class _GemmExecutor:
    def __init__(self):
        self.lhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')
        self.rhs = torch.randn((8192, 8192), dtype=torch.float, device='cuda')

    def __call__(self):
        self.lhs @ self.rhs


class _ElementwiseExecutor:
    def __init__(self):
        self.value = torch.randn(128 * 1024 ** 2, dtype=torch.int32, device='cuda')

    def __call__(self):
        self.value += 1


_EXECUTOR_CLS_OF_BENCH = {
    "gemm": _GemmExecutor,
    "elementwise": _ElementwiseExecutor,
}

_BENCH_NAMES = list(_EXECUTOR_CLS_OF_BENCH.keys())


def _compute_local_metric(bench_name):
    executor = _EXECUTOR_CLS_OF_BENCH[bench_name]()
    ms, _, _ = triton.testing.do_bench_cudagraph(executor, return_mode="mean")
    return ms


def _analyze_metrics(all_metrics: List[Dict[str, Any]]):
    for bench_name in _BENCH_NAMES:
        time_of_rank = torch.tensor([m[bench_name] for m in all_metrics])
        speed_of_rank = 1 / time_of_rank
        rel_speed_of_rank = speed_of_rank / speed_of_rank.max()
        slowest_rel_speed = rel_speed_of_rank.min().item()
        logger.info(f"[slow_rank_detector] {bench_name=} {slowest_rel_speed=} {rel_speed_of_rank=} {time_of_rank=}")
        if slowest_rel_speed < 0.9:
            logger.warning("[slow_rank_detector] Some ranks are too slow compared with others")
