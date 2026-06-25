from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch
import triton
import triton.testing
from sgl_kernel import fast_topk_transform_fused

from sglang.jit_kernel.benchmark.utils import (
    DEFAULT_DEVICE,
    get_benchmark_range,
    run_benchmark_no_cudagraph,
)
from sglang.srt.layers.attention.dsa.transform_index import (
    write_dsa_only_k_topk_paged,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="base-b-kernel-benchmark-1-gpu-large")

TOPK = 2048

SHAPES = get_benchmark_range(
    full_range=[
        (128, 128),
        (128, 256),
        (512, 512),
        (1000, 1024),
        (1024, 1024),
        (2048, 2048),
    ],
    ci_range=[(1000, 1024)],
)


def _make_case(valid_rows: int, padded_rows: int) -> Dict[str, torch.Tensor]:
    assert valid_rows <= padded_rows
    device = DEFAULT_DEVICE
    generator = torch.Generator(device=device).manual_seed(42)

    lengths = torch.randint(
        1,
        TOPK + 1,
        (valid_rows,),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    page_table = torch.arange(TOPK, dtype=torch.int32, device=device).view(1, TOPK)
    token_to_batch_idx = torch.zeros(valid_rows, dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, valid_rows], dtype=torch.int32, device=device)

    dummy_logits = torch.empty((valid_rows, TOPK), dtype=torch.float32, device=device)
    raw_topk_result = torch.empty((valid_rows, TOPK), dtype=torch.int32, device=device)
    output = torch.empty((padded_rows, TOPK), dtype=torch.int32, device=device)

    cols = torch.arange(TOPK, dtype=torch.int32, device=device)
    valid_ref = torch.where(
        cols.unsqueeze(0) < lengths.unsqueeze(1),
        page_table[0].unsqueeze(0),
        -1,
    )
    reference = torch.full(
        (padded_rows, TOPK), -1, dtype=torch.int32, device=device
    )
    reference[:valid_rows].copy_(valid_ref)

    return {
        "lengths": lengths,
        "page_table": page_table,
        "token_to_batch_idx": token_to_batch_idx,
        "cu_seqlens_q": cu_seqlens_q,
        "dummy_logits": dummy_logits,
        "raw_topk_result": raw_topk_result,
        "output": output,
        "reference": reference,
    }


def _current_path(case: Dict[str, torch.Tensor], valid_rows: int) -> torch.Tensor:
    dummy_logits = torch.zeros(
        (valid_rows, TOPK), dtype=torch.float32, device=DEFAULT_DEVICE
    )
    raw_topk_result = fast_topk_transform_fused(
        dummy_logits,
        case["lengths"],
        case["page_table"],
        case["cu_seqlens_q"],
        TOPK,
    )
    output = torch.full(
        case["output"].shape, -1, dtype=torch.int32, device=DEFAULT_DEVICE
    )
    output[:valid_rows].copy_(raw_topk_result)
    return output


def _current_prealloc(case: Dict[str, torch.Tensor], valid_rows: int) -> torch.Tensor:
    case["dummy_logits"].zero_()
    torch.ops.sgl_kernel.fast_topk_transform_fused(
        case["dummy_logits"],
        case["lengths"],
        case["raw_topk_result"],
        case["page_table"],
        case["cu_seqlens_q"],
        None,
    )
    case["output"].fill_(-1)
    case["output"][:valid_rows].copy_(case["raw_topk_result"])
    return case["output"]


def _triton_writer(case: Dict[str, torch.Tensor], valid_rows: int) -> torch.Tensor:
    del valid_rows
    return write_dsa_only_k_topk_paged(
        page_table=case["page_table"],
        lengths=case["lengths"],
        token_to_batch_idx=case["token_to_batch_idx"],
        output=case["output"],
        topk=TOPK,
    )


def _check_correctness(
    fn: Callable[[Dict[str, torch.Tensor], int], torch.Tensor],
    case: Dict[str, torch.Tensor],
    valid_rows: int,
) -> None:
    actual = fn(case, valid_rows)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, case["reference"], rtol=0, atol=0)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["valid_rows", "padded_rows"],
        x_vals=SHAPES,
        line_arg="provider",
        line_vals=["current_path", "current_prealloc", "triton_writer"],
        line_names=[
            "Current path",
            "Current path, preallocated",
            "Triton writer",
        ],
        styles=[("blue", "-"), ("orange", "--"), ("green", "-")],
        ylabel="us",
        plot_name="dsa-only-k-topk-paged",
        args={},
    )
)
def benchmark(
    valid_rows: int, padded_rows: int, provider: str
) -> Tuple[float, float, float]:
    case = _make_case(valid_rows, padded_rows)
    fn_map = {
        "current_path": _current_path,
        "current_prealloc": _current_prealloc,
        "triton_writer": _triton_writer,
    }
    fn = fn_map[provider]
    _check_correctness(fn, case, valid_rows)

    def bench_fn():
        fn(case, valid_rows)

    return run_benchmark_no_cudagraph(bench_fn)


if __name__ == "__main__":
    benchmark.run(print_data=True)
