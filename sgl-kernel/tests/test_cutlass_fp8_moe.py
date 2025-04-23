# Modified from https://github.com/vllm-project/vllm/blob/main/tests/kernels/moe/test_cutlass_moe.py
import random
from typing import Optional
import pytest
import torch
from sgl_kernel import fp8_cutlass_moe_mm, cutlass_moe_fp8
from sglang.srt.layers.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
from sglang.srt.layers.moe.topk import fused_topk

# For testing quantized linear kernels
def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fn)

def baseline_scaled_mm(a: torch.Tensor,
                       b: torch.Tensor,
                       scale_a: torch.Tensor,
                       scale_b: torch.Tensor,
                       out_dtype: type[torch.dtype],
                       bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    # We treat N-dimensional group scaling as extended numpy-style broadcasting
    # in numpy simply stretches dimensions with an extent of 1 to match the
    # the target shape by repeating the data along that dimension (broadcasting)
    # , we extend these semantics to say if the extent of a dimension in the
    # source shape is not 1 and does not match the target shape we repeat each
    # element along that dimension src_shape[dim] // target_shape[dim] times
    # example if we have:
    #       a = [[1, 2], and target_shape = (2, 4)
    #            [3, 4]]
    # then we would expand a to:
    #       a = [[1, 1, 2, 2],
    #            [3, 3, 4, 4]]
    # NOTE this function this function does not explicitly broadcast dimensions
    # with an extent of 1, since this can be done implicitly by pytorch
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = t.unsqueeze(i + 1)\
                  .expand(*t.shape[:i+1], s // t.shape[i], *t.shape[i+1:])\
                  .flatten(i, i + 1)
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)

    output = torch.mm((scale_a * a.to(dtype=torch.float32)),
                      (scale_b * b.to(dtype=torch.float32))).to(out_dtype)

    if bias is not None:
        output = output + bias

    return output

@pytest.mark.parametrize("num_experts", [8, 64])
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
@pytest.mark.parametrize("use_bias", [False])
def test_cutlass_fp8_group_gemm(num_experts: int, per_act_token: bool,
                                per_out_ch: bool, use_bias: bool):

    # if torch.cuda.get_device_capability()[0] < 9:
    #     return

    # Device and dtype setup
    device = "cuda"
    out_dtype = torch.half

    # Create separate A, B, C tensors for each group
    a_tensors = []
    b_tensors = []
    a_scales_tensors = []
    b_scales_tensors = []
    baseline_tensors = []

    expert_offsets = torch.zeros((num_experts + 1),
                                 device=device,
                                 dtype=torch.int32)

    problem_sizes = torch.zeros((num_experts, 3),
                                device=device,
                                dtype=torch.int32)

    if not per_act_token:
        one_scale_a = torch.randn((1, 1), device=device, dtype=torch.float32)

    alignment = 16  # 128 // 8
    # For variation, each group has dimensions
    n_g = alignment * random.randint(1, 64)
    k_g = alignment * random.randint(1, 64)
    for g in range(num_experts):
        m_g = alignment * random.randint(1, 64)

        expert_offsets[g + 1] = expert_offsets[g] + m_g
        problem_sizes[g][0] = m_g
        problem_sizes[g][1] = n_g
        problem_sizes[g][2] = k_g

        m_a_scales = m_g if per_act_token else 1
        n_b_scales = n_g if per_out_ch else 1

        print("shape:", m_g, n_g, k_g)

        # Create group-specific A and B (FP8) and output (FP16/FP32)
        a_g = to_fp8(torch.randn((m_g, k_g), device=device))
        b_g = to_fp8(torch.randn((n_g, k_g), device=device).t())
        a_tensors.append(a_g)
        b_tensors.append(b_g)

        # Set up A/B scales
        scale_b = torch.randn((1, n_b_scales),
                              device=device,
                              dtype=torch.float32)
        b_scales_tensors.append(scale_b)

        if per_act_token:
            scale_a = torch.randn((m_a_scales, 1),
                                  device=device,
                                  dtype=torch.float32)
            a_scales_tensors.append(scale_a)
        else:
            scale_a = one_scale_a

        # Compute baseline result for this group
        baseline_g = baseline_scaled_mm(a_g, b_g, scale_a, scale_b, out_dtype,
                                        None)
        baseline_tensors.append(baseline_g)

    a_tensors_stacked = torch.empty((expert_offsets[num_experts], k_g),
                                    device=device,
                                    dtype=torch.float8_e4m3fn)
    b_tensors_stacked = torch.empty((num_experts, n_g, k_g),
                                    device=device,
                                    dtype=torch.float8_e4m3fn)

    for g in range(num_experts):
        a_tensors_stacked[expert_offsets[g]:expert_offsets[g +
                                                           1]] = a_tensors[g]
        b_tensors_stacked[g] = b_tensors[g].t()
    b_tensors_stacked = b_tensors_stacked.transpose(1, 2)

    if per_act_token:
        a_scales_tensors_stacked = torch.empty(
            (expert_offsets[num_experts], 1),
            device=device,
            dtype=torch.float32)
        for g in range(num_experts):
            a_scales_tensors_stacked[
                expert_offsets[g]:expert_offsets[g + 1]] = a_scales_tensors[g]
    else:
        a_scales_tensors_stacked = one_scale_a

    b_scales_tensors_stacked = torch.empty((num_experts, n_b_scales),
                                           device=device,
                                           dtype=torch.float32)
    for g in range(num_experts):
        b_scales_tensors_stacked[g] = b_scales_tensors[g]

    out_tensors_stacked = torch.zeros((expert_offsets[num_experts], n_g),
                                      device=device,
                                      dtype=out_dtype)

    ab_strides = torch.full((num_experts, ),
                            a_tensors_stacked.stride(0),
                            device="cuda",
                            dtype=torch.int64)
    c_strides = torch.full((num_experts, ),
                           out_tensors_stacked.stride(0),
                           device="cuda",
                           dtype=torch.int64)

    fp8_cutlass_moe_mm(out_tensors_stacked, a_tensors_stacked,
                       b_tensors_stacked, a_scales_tensors_stacked,
                       b_scales_tensors_stacked, expert_offsets[:-1],
                       problem_sizes, ab_strides, ab_strides, c_strides)

    # Validate each group's result against the baseline
    for g in range(num_experts):
        baseline = baseline_tensors[g]
        c = out_tensors_stacked[expert_offsets[g]:expert_offsets[g + 1]]
        print(baseline)
        print(c)
        print("*")
        torch.testing.assert_close(c, baseline, rtol=1e-2, atol=5e-4)

NUM_EXPERTS = [40, 64]
TOP_KS = [6, 8]

@pytest.mark.parametrize("m", [2, 64, 224])
@pytest.mark.parametrize("n", [1024, 3072])
@pytest.mark.parametrize("k", [1024, 1536])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
def test_cutlass_moe_no_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
):
    dtype = torch.half
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    # Get the right scale for tests.
    _, a_scale1 = scaled_fp8_quant(
        a, use_per_token_if_dynamic=per_act_token)
    a_q, _ = scaled_fp8_quant(a,
                                    a_scale1,
                                    use_per_token_if_dynamic=per_act_token)

    a_d = a_q.float().mul(a_scale1).to(dtype)

    n_b_scales = 2 * n if per_out_ch else 1
    k_b_scales = k if per_out_ch else 1

    w1_q = torch.empty((e, 2 * n, k),
                        device="cuda",
                        dtype=torch.float8_e4m3fn)
    w2_q = torch.empty((e, k, n), device="cuda", dtype=torch.float8_e4m3fn)
    w1_scale = torch.empty((e, n_b_scales, 1),
                            device="cuda",
                            dtype=torch.float32)
    w2_scale = torch.empty((e, k_b_scales, 1),
                            device="cuda",
                            dtype=torch.float32)

    ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
    c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
    ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
    c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)

    for expert in range(e):
        w1_q[expert], w1_scale[expert] = scaled_fp8_quant(
            w1[expert], use_per_token_if_dynamic=per_out_ch)
        w2_q[expert], w2_scale[expert] = scaled_fp8_quant(
            w2[expert], use_per_token_if_dynamic=per_out_ch)
    w1_q = w1_q.transpose(1, 2)
    w2_q = w2_q.transpose(1, 2)

    ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
    c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
    ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
    c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)

    w1_d = torch.empty_like(w1)
    w2_d = torch.empty_like(w2)
    for expert in range(e):
        w1_d[expert] = (w1_q[expert].t().float() * w1_scale[expert]).half()
        w2_d[expert] = (w2_q[expert].t().float() * w2_scale[expert]).half()

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

    triton_output = fused_experts(a_d, w1_d, w2_d, topk_weights, topk_ids)

    cutlass_output = cutlass_moe_fp8(a,
                                        w1_q,
                                        w2_q,
                                        w1_scale,
                                        w2_scale,
                                        topk_weights,
                                        topk_ids,
                                        ab_strides1,
                                        c_strides1,
                                        ab_strides2,
                                        c_strides2,
                                        a1_scale=a_scale1)

    print(triton_output)
    print(cutlass_output)
    print("*")

    torch.testing.assert_close(triton_output,
                                cutlass_output,
                                atol=5e-2,
                                rtol=1e-2)



def run(a: torch.Tensor, a_scale: torch.Tensor, w1_q: torch.Tensor,
        w2_q: torch.Tensor, w1_scale: torch.Tensor, w2_scale: torch.Tensor,
        topk_weights: torch.Tensor, topk_ids: torch.Tensor,
        ab_strides1: torch.Tensor, c_strides1: torch.Tensor,
        ab_strides2: torch.Tensor, c_strides2: torch.Tensor):
    return cutlass_moe_fp8(a,
                            w1_q,
                            w2_q,
                            w1_scale,
                            w2_scale,
                            topk_weights,
                            topk_ids,
                            ab_strides1,
                            c_strides1,
                            ab_strides2,
                            c_strides2,
                            a1_scale=a_scale)


@pytest.mark.parametrize("m", [2, 64, 224])
@pytest.mark.parametrize("n", [1024, 3072])
@pytest.mark.parametrize("k", [1024, 1536])
@pytest.mark.parametrize("e", NUM_EXPERTS)
@pytest.mark.parametrize("topk", TOP_KS)
@pytest.mark.parametrize("per_act_token", [True, False])
@pytest.mark.parametrize("per_out_ch", [True, False])
def test_cutlass_moe_cuda_graph(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    per_act_token: bool,
    per_out_ch: bool,
):
    dtype = torch.half

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10

    # Get the right scale for tests.
    _, a_scale1 = scaled_fp8_quant(
        a, use_per_token_if_dynamic=per_act_token)
    a_q, _ = scaled_fp8_quant(a,
                                    a_scale1,
                                    use_per_token_if_dynamic=per_act_token)

    a_d = a_q.float().mul(a_scale1).to(dtype)

    n_b_scales = 2 * n if per_out_ch else 1
    k_b_scales = k if per_out_ch else 1

    w1_q = torch.empty((e, 2 * n, k),
                        device="cuda",
                        dtype=torch.float8_e4m3fn)
    w2_q = torch.empty((e, k, n), device="cuda", dtype=torch.float8_e4m3fn)
    w1_scale = torch.empty((e, n_b_scales, 1),
                            device="cuda",
                            dtype=torch.float32)
    w2_scale = torch.empty((e, k_b_scales, 1),
                            device="cuda",
                            dtype=torch.float32)

    ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
    c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
    ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
    c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)

    for expert in range(e):
        w1_q[expert], w1_scale[expert] = scaled_fp8_quant(
            w1[expert], use_per_token_if_dynamic=per_out_ch)
        w2_q[expert], w2_scale[expert] = scaled_fp8_quant(
            w2[expert], use_per_token_if_dynamic=per_out_ch)
    w1_q = w1_q.transpose(1, 2)
    w2_q = w2_q.transpose(1, 2)

    ab_strides1 = torch.full((e, ), k, device="cuda", dtype=torch.int64)
    c_strides1 = torch.full((e, ), 2 * n, device="cuda", dtype=torch.int64)
    ab_strides2 = torch.full((e, ), n, device="cuda", dtype=torch.int64)
    c_strides2 = torch.full((e, ), k, device="cuda", dtype=torch.int64)

    w1_d = torch.empty_like(w1)
    w2_d = torch.empty_like(w2)
    for expert in range(e):
        w1_d[expert] = (w1_q[expert].t().float() * w1_scale[expert]).half()
        w2_d[expert] = (w2_q[expert].t().float() * w2_scale[expert]).half()

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    topk_weights, topk_ids = fused_topk(a, score, topk, renormalize=False)

    triton_output = fused_experts(a_d, w1_d, w2_d, topk_weights, topk_ids)

    stream = torch.cuda.Stream()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        cutlass_output = run(a, a_scale1, w1_q, w2_q, w1_scale, w2_scale,
                                topk_weights, topk_ids, ab_strides1,
                                c_strides1, ab_strides2, c_strides2)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    print(triton_output)
    print(cutlass_output)
    print("*")

    torch.testing.assert_close(triton_output,
                                cutlass_output,
                                atol=9e-2,
                                rtol=1e-2)

if __name__ == "__main__":
    pytest.main([__file__])
