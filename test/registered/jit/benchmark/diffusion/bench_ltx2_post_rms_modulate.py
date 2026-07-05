import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.diffusion.ltx2_post_rms_modulate import (
    can_use_ltx2_post_rms_dual_modulate_cuda,
    can_use_ltx2_post_rms_modulate_cuda,
    ltx2_post_rms_dual_modulate_cuda,
    ltx2_post_rms_modulate_cuda,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="4-gpu-b200"
)


def _require_b200() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        marker.skip("LTX2 post-RMS modulation benchmark requires B200/SM100")


def _param(batch: int, seq: int, hidden: int, param_seq: int) -> torch.Tensor:
    packed = torch.randn(
        batch, param_seq, 3, hidden, device="cuda", dtype=torch.bfloat16
    )
    return packed.unbind(dim=2)[1]


def torch_single(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    return x * (1 + scale) + shift


def cuda_single(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
    if not can_use_ltx2_post_rms_modulate_cuda(x, scale, shift):
        marker.skip("unsupported single modulation shape")
    return ltx2_post_rms_modulate_cuda(x, scale, shift)


def torch_dual(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
):
    return x * (1 + scale0) + shift0, x * (1 + scale1) + shift1


def cuda_dual(
    x: torch.Tensor,
    scale0: torch.Tensor,
    shift0: torch.Tensor,
    scale1: torch.Tensor,
    shift1: torch.Tensor,
):
    if not can_use_ltx2_post_rms_dual_modulate_cuda(x, scale0, shift0, scale1, shift1):
        marker.skip("unsupported dual modulation shape")
    return ltx2_post_rms_dual_modulate_cuda(x, scale0, shift0, scale1, shift1)


FN_MAP = {
    "torch": {
        "single": torch_single,
        "dual": torch_dual,
    },
    "cuda": {
        "single": cuda_single,
        "dual": cuda_dual,
    },
}


@marker.parametrize(
    "mode,batch,seq,hidden,param_seq",
    [
        ("single", 1, 161, 4096, 1),
        ("single", 1, 529, 4096, 529),
        ("dual", 1, 32640, 4096, 32640),
        ("dual", 1, 1024, 2048, 1024),
    ],
    [
        ("single", 1, 161, 4096, 1),
        ("dual", 1, 1024, 2048, 1024),
    ],
)
@marker.benchmark("impl", ["torch", "cuda"])
def benchmark(
    mode: str,
    batch: int,
    seq: int,
    hidden: int,
    param_seq: int,
    impl: str,
):
    _require_b200()
    x = torch.randn(batch, seq, hidden, device="cuda", dtype=torch.bfloat16)
    scale0 = _param(batch, seq, hidden, param_seq)
    shift0 = _param(batch, seq, hidden, param_seq)
    if mode == "single":
        return marker.do_bench(
            FN_MAP[impl][mode],
            input_args=(x, scale0, shift0),
            graph_clone_args=(0, 1, 2),
        )

    scale1 = _param(batch, seq, hidden, param_seq)
    shift1 = _param(batch, seq, hidden, param_seq)
    return marker.do_bench(
        FN_MAP[impl][mode],
        input_args=(x, scale0, shift0, scale1, shift1),
        graph_clone_args=(0, 1, 2, 3, 4),
    )


if __name__ == "__main__":
    benchmark.run()
