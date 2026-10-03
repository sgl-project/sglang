"""FP8 Marlin CP decode attention TP shards produce the rank-local GEMM.

The rank-local packed views must compute exactly what the logical TP shard of
the full linear computes: output-split shards give a column block of the full
output; input-split partial products sum to the full output.
"""

import unittest

import torch

from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpShard
from sglang.srt.layers.quantization.marlin_utils_fp8 import (
    apply_fp8_marlin_linear,
    prepare_fp8_layer_for_marlin,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

N, K, SIZE, M = 1024, 1024, 4, 16


def _make_layer(weight, scales, block, size_k_first, shard):
    layer = torch.nn.Module()
    layer.output_size_per_partition = N
    layer.input_size_per_partition = K
    layer.orig_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    if block is not None:
        layer.weight_block_size = block
        layer.weight_scale_inv = torch.nn.Parameter(scales.clone(), requires_grad=False)
    else:
        layer.weight_scale = torch.nn.Parameter(scales.clone(), requires_grad=False)
    layer.cp_decode_attn_tp_shard = shard
    prepare_fp8_layer_for_marlin(layer, size_k_first)
    return layer


def _gemm(x, weight, weight_scale, workspace, n, k):
    return apply_fp8_marlin_linear(
        input=x,
        weight=weight,
        weight_scale=weight_scale,
        workspace=workspace,
        size_n=n,
        size_k=k,
        bias=None,
    )


@unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
class TestFp8MarlinCpDecodeAttnTp(CustomTestCase):
    def _inputs(self, block, size_k_first):
        torch.manual_seed(0)
        dev = "cuda"
        if size_k_first:
            weight = torch.randn(K, N, device=dev) * 0.1
        else:
            weight = torch.randn(N, K, device=dev) * 0.1
        weight = weight.to(torch.float8_e4m3fn)
        if block is not None:
            scales = torch.rand(N // block[0], K // block[1], device=dev) * 0.02 + 0.01
        else:
            scales = torch.rand(N, device=dev) * 0.02 + 0.01
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        return weight, scales, x

    def _dequant(self, weight, scales, block, size_k_first):
        w = weight.float()
        if size_k_first:
            w = w.T  # (N, K)
        if block is not None:
            s = scales.repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
        else:
            s = scales.view(N, 1)
        return w * s

    def _check(self, split, block, size_k_first):
        weight, scales, x = self._inputs(block, size_k_first)
        ref = x.float() @ self._dequant(weight, scales, block, size_k_first).T

        partial_sum = torch.zeros(M, N, device="cuda", dtype=torch.float32)
        for rank in range(SIZE):
            layer = _make_layer(
                weight,
                scales,
                block,
                size_k_first,
                CpDecodeAttnTpShard(split=split, rank=rank, size=SIZE),
            )
            y_full = _gemm(x, layer.weight, layer.weight_scale, layer.workspace, N, K)
            torch.testing.assert_close(y_full.float(), ref, atol=5e-2, rtol=2e-2)

            views = layer.cp_decode_attn_tp_packed_views
            if split == "output":
                chunk = N // SIZE
                y = _gemm(
                    x, views["weight"], views["weight_scale"], layer.workspace, chunk, K
                )
                torch.testing.assert_close(
                    y,
                    y_full[:, rank * chunk : (rank + 1) * chunk],
                    atol=2e-2,
                    rtol=1e-2,
                )
            else:
                chunk = K // SIZE
                x_local = x[:, rank * chunk : (rank + 1) * chunk].contiguous()
                y = _gemm(
                    x_local,
                    views["weight"],
                    views["weight_scale"],
                    layer.workspace,
                    N,
                    chunk,
                )
                partial_sum += y.float()
        if split == "input":
            torch.testing.assert_close(partial_sum, ref, atol=5e-2, rtol=2e-2)

    def test_block_quant_output_split(self):
        self._check("output", [128, 128], size_k_first=False)

    def test_block_quant_input_split(self):
        self._check("input", [128, 128], size_k_first=False)

    def test_channelwise_output_split(self):
        self._check("output", None, size_k_first=True)

    def test_channelwise_input_split(self):
        self._check("input", None, size_k_first=True)


if __name__ == "__main__":
    unittest.main(verbosity=3)
