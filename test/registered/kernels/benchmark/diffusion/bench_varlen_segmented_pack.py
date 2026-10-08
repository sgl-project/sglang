import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.diffusion import fused_pack_qkv, fused_pack_segmented_qkv
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=12, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _materialized_pack(q_prefix, k_prefix, v_prefix, q_main, k_main, v_main, indices):
    return fused_pack_qkv(
        torch.cat([q_prefix, q_main], dim=1),
        torch.cat([k_prefix, k_main], dim=1),
        torch.cat([v_prefix, v_main], dim=1),
        indices,
    )


def _segmented_pack(q_prefix, k_prefix, v_prefix, q_main, k_main, v_main, indices):
    return fused_pack_segmented_qkv(
        q_prefix, k_prefix, v_prefix, q_main, k_main, v_main, indices
    )


@marker.parametrize(
    "batch,prefix_rows,main_rows,heads,head_dim,valid_prefix_rows",
    [
        (1, 64, 4096, 12, 128, 24),
        (2, 64, 1024, 8, 128, 40),
    ],
    ci_vals=[(1, 64, 4096, 12, 128, 24)],
)
@marker.benchmark("provider", ["materialized", "segmented"])
def benchmark(
    batch: int,
    prefix_rows: int,
    main_rows: int,
    heads: int,
    head_dim: int,
    valid_prefix_rows: int,
    provider: str,
) -> marker.BenchResult:
    generator = torch.Generator(device=DEVICE).manual_seed(42)
    prefixes = tuple(
        torch.randn(
            batch,
            prefix_rows,
            heads,
            head_dim,
            dtype=DTYPE,
            device=DEVICE,
            generator=generator,
        )
        for _ in range(3)
    )
    mains = tuple(
        torch.randn(
            batch,
            main_rows,
            heads,
            head_dim,
            dtype=DTYPE,
            device=DEVICE,
            generator=generator,
        )
        for _ in range(3)
    )
    mask = torch.zeros(batch, prefix_rows + main_rows, dtype=torch.bool, device=DEVICE)
    mask[:, :valid_prefix_rows] = True
    mask[:, prefix_rows:] = True
    indices = mask.flatten().nonzero(as_tuple=False).flatten()
    args = (*prefixes, *mains, indices)

    expected = _materialized_pack(*args)
    actual = _segmented_pack(*args)
    assert all(
        torch.equal(got, want) for got, want in zip(actual, expected, strict=True)
    )

    fn = _materialized_pack if provider == "materialized" else _segmented_pack
    return marker.do_bench(
        fn,
        input_args=args,
        graph_clone_args=tuple(range(len(args))),
        disable_log_bandwidth=True,
    )


if __name__ == "__main__":
    benchmark.run()
