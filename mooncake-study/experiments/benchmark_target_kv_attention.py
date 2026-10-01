"""CUDA-graph attention microbenchmark; excludes model, KV writes and serving SLO."""

import json

import torch
import triton.testing
from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.kernels.ops.speculative.dspark.target_kv_attention import (
    target_kv_attention,
)


@torch.no_grad()
def run(batch, prefix, width):
    heads, kv_heads, dim = 16, 8, 128
    device, dtype = "cuda", torch.bfloat16
    num_slots = batch * (prefix + width) * 2
    q = torch.randn(batch * width, heads, dim, device=device, dtype=dtype)
    k = torch.randn(num_slots, kv_heads, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    slots = torch.randperm(num_slots, device=device)[
        : batch * (prefix + width)
    ].reshape(batch, -1)
    indices = slots[:, :prefix].flatten().contiguous()
    extend_indices = slots[:, prefix:].flatten().contiguous()
    k_extend, v_extend = k[extend_indices], v[extend_indices]
    qo = torch.arange(batch + 1, device=device, dtype=torch.int32) * width
    ki = torch.arange(batch + 1, device=device, dtype=torch.int32) * prefix
    old_output, new_output = torch.empty_like(q), torch.empty_like(q)

    def legacy():
        extend_attention_fwd(
            q,
            k_extend,
            v_extend,
            old_output,
            k,
            v,
            qo,
            ki,
            indices,
            custom_mask=None,
            is_causal=False,
            mask_indptr=None,
            max_len_extend=width,
            k_scale=1.0,
            v_scale=1.0,
            sm_scale=dim**-0.5,
        )

    def logical():
        target_kv_attention(
            q,
            k,
            v,
            new_output,
            qo,
            ki,
            indices,
            max_query=width,
            scale=dim**-0.5,
            extend_indices=extend_indices,
        )

    legacy()
    logical()
    torch.testing.assert_close(new_output, old_output, rtol=0.03, atol=0.015)
    times = {
        name: triton.testing.do_bench_cudagraph(call, rep=200) * 1000
        for name, call in (
            ("legacy_two_stage_us", legacy),
            ("target_kv_logical_us", logical),
        )
    }
    print(
        json.dumps(
            {
                "microbenchmark_only": True,
                "device": torch.cuda.get_device_name(),
                "torch_version": torch.__version__,
                "dtype": str(dtype),
                "batch": batch,
                "prefix": prefix,
                "width": width,
                "heads": heads,
                "kv_heads": kv_heads,
                "dim": dim,
                **times,
                "logical_over_legacy": times["target_kv_logical_us"]
                / times["legacy_two_stage_us"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    torch.manual_seed(918)
    for shape in (
        (1, 160, 3),
        (4, 160, 3),
        (1, 1024, 3),
        (4, 1024, 16),
        (1, 8192, 3),
        (16, 8192, 16),
    ):
        run(*shape)
