#!/usr/bin/env python3
"""Correctness-checked gfx1250 cooperative TopK v2 A/B benchmark."""

import os
import sys

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4.topk import (
    _gfx1250_cluster_width,
    plan_topk_v2,
    topk_transform_paged_v2,
)

SHAPES = (
    (1, 131072),
    (4, 131072),
    (1, 262144),
    (31, 131072),
    (64, 131072),
    (64, 262144),
)
TOPK = 512
PAGE_SIZE = 64


def set_cluster(enabled: bool) -> None:
    if enabled:
        os.environ["SGLANG_GFX1250_TOPK_CLUSTER"] = "1"
    else:
        os.environ.pop("SGLANG_GFX1250_TOPK_CLUSTER", None)


def bench_shape(batch: int, seq: int) -> tuple[float, float, int]:
    torch.manual_seed(batch * 100003 + seq)
    scores = torch.randn(batch, seq, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    out_off = torch.full((batch, TOPK), -1, dtype=torch.int32, device="cuda")
    out_on = torch.full_like(out_off, -1)

    set_cluster(False)
    metadata_off = plan_topk_v2(seq_lens)
    torch.cuda.synchronize()
    set_cluster(True)
    metadata_on = plan_topk_v2(seq_lens)
    torch.cuda.synchronize()

    def run_off() -> None:
        set_cluster(False)
        topk_transform_paged_v2(
            scores, seq_lens, None, out_off, PAGE_SIZE, metadata_off
        )

    def run_on() -> None:
        set_cluster(True)
        topk_transform_paged_v2(
            scores, seq_lens, None, out_on, PAGE_SIZE, metadata_on
        )

    run_off()
    run_on()
    torch.cuda.synchronize()
    reference = torch.topk(scores, TOPK, dim=1, sorted=False).indices.cpu().tolist()
    for row, (expected, got_off, got_on) in enumerate(
        zip(reference, out_off.cpu().tolist(), out_on.cpu().tolist())
    ):
        expected_set = set(expected)
        if set(got_off) != expected_set:
            raise AssertionError(f"default mismatch at row {row}")
        if set(got_on) != expected_set:
            raise AssertionError(f"cluster mismatch at row {row}")

    kwargs = {
        "use_cuda_graph": False,
        "warmup_iters": 20,
        "replay_iters": 100,
        "metrics": (0.5, "avg"),
        "memory_args": (scores, seq_lens),
    }
    off_time = marker.do_bench(run_off, **kwargs).times[0] * 1e6
    on_time = marker.do_bench(run_on, **kwargs).times[0] * 1e6
    floor = 32768 if batch <= 15 else 65536
    cluster_width = (
        _gfx1250_cluster_width(batch, seq)
        if 1 < batch <= 512 and seq > floor
        else 0
    )
    return off_time, on_time, cluster_width


def main() -> None:
    props = torch.cuda.get_device_properties(0)
    print(f"device={props.name} arch={getattr(props, 'gcnArchName', None)}")
    print("| batch | seq | cluster width | default us | cluster us | speedup |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    shapes = (
        ((int(sys.argv[1]), int(sys.argv[2])),)
        if len(sys.argv) == 3
        else SHAPES
    )
    for batch, seq in shapes:
        off, on, width = bench_shape(batch, seq)
        print(
            f"| {batch} | {seq} | {width} | {off:.2f} | {on:.2f} | "
            f"{off / on:.3f}x |"
        )


if __name__ == "__main__":
    main()
