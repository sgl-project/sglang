import torch
from sgl_kernel.flash_mla_mega import flash_mla_mega_prefill

from sglang.kernels.jit.benchmark import marker


@marker.parametrize("tokens", [1, 32, 128, 512], [128])
@marker.parametrize("heads", [64, 128], [64])
@marker.benchmark("provider", ["flashmla_mega"])
def benchmark(tokens: int, heads: int, provider: str):
    q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(4096, 1, 512, device="cuda", dtype=torch.bfloat16)
    indices = torch.randint(4096, (tokens, 1, 640), device="cuda", dtype=torch.int32)
    positions = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    table = torch.cat((torch.ones(1, 32), torch.zeros(1, 32)), dim=-1).cuda()
    return marker.do_bench(
        flash_mla_mega_prefill,
        input_args=(q, kv, indices, positions, table, 512**-0.5),
    )


if __name__ == "__main__":
    benchmark.run()
