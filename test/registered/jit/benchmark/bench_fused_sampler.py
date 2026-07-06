import torch

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.fused_sampler import fused_topk_sample
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def _flashinfer_topk_topp_sample(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    if top_k == 1:
        return torch.argmax(logits, dim=-1)

    try:
        from flashinfer.sampling import top_k_top_p_sampling_from_probs
    except ImportError:
        marker.skip("flashinfer.sampling is unavailable")

    probs = torch.softmax(logits / temperatures.view(-1, 1), dim=-1)
    return top_k_top_p_sampling_from_probs(
        probs.contiguous(),
        top_ks,
        top_ps,
        filter_apply_order="joint",
    )


def _jit_fused_sample(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    uniforms: torch.Tensor,
    out: torch.Tensor,
    scratch_scores: torch.Tensor,
    scratch_indices: torch.Tensor,
    scratch_sums: torch.Tensor,
    top_k: int,
    needs_top_p: bool,
) -> torch.Tensor:
    return fused_topk_sample(
        logits,
        temperatures,
        top_ps,
        top_k,
        uniforms,
        out=out,
        scratch_scores=scratch_scores,
        scratch_indices=scratch_indices,
        scratch_sums=scratch_sums,
        needs_top_p=needs_top_p,
    )


FN_MAP = {
    "baseline": _flashinfer_topk_topp_sample,
    "jit": _jit_fused_sample,
}


@marker.parametrize("top_p", [1.0, 0.95], [1.0])
@marker.parametrize("top_k", [2, 4, 8], [4])
@marker.parametrize("vocab_size", [128256, 151552], [128256])
@marker.parametrize("batch_size", [1, 16, 64, 256, 1024], [16])
@marker.benchmark("impl", ["baseline", "jit"])
def benchmark(
    batch_size: int,
    vocab_size: int,
    top_k: int,
    top_p: float,
    impl: str,
):
    logits = torch.randn((batch_size, vocab_size), device="cuda", dtype=torch.float32)
    temperatures = torch.ones((batch_size,), device="cuda", dtype=torch.float32)
    top_ps = torch.full((batch_size,), top_p, device="cuda", dtype=torch.float32)
    uniforms = torch.rand((batch_size,), device="cuda", dtype=torch.float32)
    top_ks = torch.full((batch_size,), top_k, device="cuda", dtype=torch.int32)
    needs_top_p = top_p != 1.0

    out = torch.empty((batch_size,), device="cuda", dtype=torch.int64)
    if top_k == 1:
        scratch_scores = torch.empty((0,), device="cuda", dtype=torch.float32)
        scratch_indices = torch.empty((0,), device="cuda", dtype=torch.int32)
        scratch_sums = torch.empty((0,), device="cuda", dtype=torch.float32)
    else:
        num_tiles = (vocab_size + 1023) // 1024
        scratch_scores = torch.empty(
            (batch_size, num_tiles, top_k), device="cuda", dtype=torch.float32
        )
        scratch_indices = torch.empty(
            (batch_size, num_tiles, top_k), device="cuda", dtype=torch.int32
        )
        scratch_sums = (
            torch.empty((batch_size, num_tiles), device="cuda", dtype=torch.float32)
            if needs_top_p
            else torch.empty((0,), device="cuda", dtype=torch.float32)
        )

    if impl == "baseline":
        return marker.do_bench(
            FN_MAP[impl],
            input_args=(logits, temperatures, top_ps, top_ks, top_k),
            graph_clone_args=(0,),
            disable_log_bandwidth=True,
        )

    return marker.do_bench(
        FN_MAP[impl],
        input_args=(
            logits,
            temperatures,
            top_ps,
            uniforms,
            out,
            scratch_scores,
            scratch_indices,
            scratch_sums,
            top_k,
            needs_top_p,
        ),
        graph_clone_args=(0,),
        memory_output=(out,),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
