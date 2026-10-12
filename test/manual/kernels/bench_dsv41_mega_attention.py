import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.dsv4.mega_attention import (
    flashmla_mega_decode,
    pack_mega_query,
)
from sglang.kernels.ops.attention.dsv4.torch_quant import quantize_k_cache_v41
from sglang.srt.layers.attention.dsv4.mega_attention import (
    prepare_mega_wo_a,
    project_mega_attention,
)
from sglang.test.kernels.deepseek_v4.dsv41_kv_quant_reference import (
    quantize_k_cache_v41_fp4,
)


def run(q, kv, indices, extra_kv, extra_indices, positions, table, weight, scale):
    out = flashmla_mega_decode(
        pack_mega_query(q, 64),
        kv,
        indices,
        positions,
        table,
        softmax_scale=512**-0.5,
        extra_kv=extra_kv,
        extra_indices=extra_indices,
    )
    return project_mega_attention(out.data, out.scale, weight, scale)


@marker.parametrize("tokens", [1, 32, 128, 512], [128])
@marker.parametrize("heads", [16, 32, 64], [64])
@marker.benchmark("provider", ["mega_nvfp4"])
def benchmark(tokens: int, heads: int, provider: str):
    q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16)
    main = quantize_k_cache_v41(
        torch.randn(1, 128, 512, device="cuda", dtype=torch.bfloat16)
    )
    main = main[:, : 128 * 528].view(1, 128, 1, 528)
    extra = quantize_k_cache_v41_fp4(
        torch.randn(128, 64, 512, device="cuda", dtype=torch.bfloat16)
    )
    extra = extra[:, : 64 * 288].view(128, 64, 1, 288)
    indices = torch.arange(128, device="cuda", dtype=torch.int32).repeat(tokens, 1)
    extra_indices = torch.randint(8192, (tokens, 512), device="cuda", dtype=torch.int32)
    positions = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    table = torch.cat((torch.ones(1, 32), torch.zeros(1, 32)), dim=-1).cuda()
    weight, scale = prepare_mega_wo_a(
        torch.randn(heads // 8, 1024, 4096, device="cuda", dtype=torch.bfloat16)
    )
    return marker.do_bench(
        run,
        input_args=(
            q,
            main,
            indices,
            extra,
            extra_indices,
            positions,
            table,
            weight,
            scale,
        ),
    )


if __name__ == "__main__":
    benchmark.run()
