import torch
# import tensorrt_llm
import pytest
from sgl_kernel import cutlass_w4a8_moe_mm


def print_tensor_info(name, tensor):
    print(f"\n{name}:")
    print(f"  shape: {tensor.shape}")
    print(f"  values: {tensor.flatten()[:10]}")  # Print first 10 values


def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError(
            "int4_values_interleaved 的最后一个维度的大小必须是偶数。"
        )

    input_tensor_int8 = int4_values_interleaved.to(torch.int8)

    # 分离低位和高位半字节的值
    # a[..., 0::2] 取最后一个维度上索引为偶数的元素
    # a[..., 1::2] 取最后一个维度上索引为奇数的元素
    low_nibbles = input_tensor_int8[..., 0::2]
    high_nibbles = input_tensor_int8[..., 1::2]

    packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)
    
    return packed_tensor.to(torch.int8)


def pack_interleave(num_experts, ref_weight, ref_scale):
    n, k = ref_weight.shape[1], ref_weight.shape[2]
    # packer = torch.ops.trtllm.pack_int8_tensor_to_packed_int4

    # weight = packer(ref_weight.cpu()).cuda()
    weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8)
    # w_q = w_q.contiguous().transpose(1, 2)
    w_q = w_q.contiguous()

    ###############################################################
    # scale interleave, [E, K, N]
    # scale = ref_scale.permute(0, 2, 1)  # [E, N, K]
    scale = ref_scale
    scale_interleaved = scale.reshape(
        scale.shape[0], scale.shape[1], (scale.shape[2] // 4), 4
    )  # [E, N, K/4, 4]
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)  # [E, K/4, N, 4]
    scale_interleaved = scale_interleaved.reshape(
        scale.shape[0], scale.shape[2] // 4, scale.shape[1] * 4
    )  # [E, K/4, N*4]
    w_scale = scale_interleaved.contiguous()

    return w_q, w_scale


def woq_assert_near_eq(ref, act, wTypeId):
    # match the scale in cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp
    if wTypeId == 1:
        bits_in_type = 8
    else:
        bits_in_type = 4
    quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))

    max_val = torch.max(abs(ref)).item()
    atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
    torch.testing.assert_close(ref, act, atol=atol, rtol=1e-7)


@pytest.mark.parametrize("batch_size", [2])
def test_int4_fp8_grouped_gemm(batch_size):
    # Test parameters
    num_experts = 2
    m = batch_size  # batch size
    k = 512  # input dimension
    n = 1024  # output dimension
    # torch.manual_seed(0)
    dtype = torch.bfloat16
    debug = False

    print(f"\nTesting with batch_size={batch_size}")

    # Create input tensors with ones
    if debug:
        a = torch.ones(m, k, dtype=torch.bfloat16, device="cuda")
        # a[1:] = 2
        ref_w = torch.ones(num_experts, n, k, dtype=torch.int8, device="cuda")
        a_scale = torch.ones(1, dtype=torch.float, device="cuda")
        ref_w_scale = torch.ones(num_experts, n, k // 128, dtype=dtype, device="cuda")
    else:
        a = torch.randn(m, k, dtype=dtype, device="cuda")
        ref_w = torch.randint(
            -8, 8, (num_experts, n, k), dtype=torch.int8, device="cuda"
        )
        affine_coeff = 0.005
        a_scale = torch.randn(1, dtype=torch.float32).cuda() * 0.02
        ref_w_scale = (
            torch.randn(num_experts, n, k // 128, dtype=dtype, device="cuda")
            * affine_coeff
        )

    w, w_scale = pack_interleave(num_experts, ref_w, ref_w_scale)

    # Create expert offsets and problem sizes
    expert_offsets = torch.tensor([0, 0, m], dtype=torch.int32, device="cuda")
    problem_sizes = torch.tensor(
        [[n, 0, k], [n, m, k]], dtype=torch.int32, device="cuda"
    )

    device = "cuda"
    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = a_strides
    s_strides = c_strides

    # Print all input parameters
    print_tensor_info("Input a", a)
    print_tensor_info("Weights w", w)
    print_tensor_info("Ref Weights ref_w", ref_w)
    print_tensor_info("Input scale a_scale", a_scale)
    print_tensor_info("Weight scale w_scale", w_scale)
    print_tensor_info("Expert offsets", expert_offsets)
    print_tensor_info("Problem sizes", problem_sizes)
    print_tensor_info("A strides", a_strides)
    print_tensor_info("B strides", b_strides)
    print_tensor_info("C strides", c_strides)
    print_tensor_info("S strides", s_strides)

    # Quantize input
    # a_q, a_scale = ops.scaled_fp8_quant(a, a_scale)
    a_q = torch.clamp((a / a_scale), -448.0, 448.0).to(torch.float8_e4m3fn).to(device)
    print_tensor_info("Quantized input a_q", a_q)
    print_tensor_info("Updated input scale a_scale", a_scale)

    # Create output tensor
    c = torch.empty((m, n), dtype=torch.float16, device="cuda")
    # c = c.T

    # Run the operator
    # w = w.view(torch.quint4x2)
    cutlass_w4a8_moe_mm(
        c,
        a_q,
        w,
        a_scale,
        w_scale,
        expert_offsets[:-1],
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        s_strides,
        128,
        8,
    )
    c = c.to(dtype)
    # c = c.T

    print_tensor_info("Output c", c)

    # Reference implementation
    e_idx = 1
    a = torch.clamp((a / a_scale), -448.0, 448.0).to(torch.float8_e4m3fn)
    ref_w_scale_repeat = ref_w_scale[e_idx].repeat_interleave(128, dim=1).to(float)
    print_tensor_info("ref_w_scale_repeat", ref_w_scale_repeat)
    ref_w_one_expert = (ref_w[e_idx].to(float) * ref_w_scale_repeat).to(dtype)
    print_tensor_info("ref_w_one_expert", ref_w_one_expert)
    c_ref = torch.matmul(a.to(dtype), ref_w_one_expert.t().to(dtype)) * a_scale
    c_ref = c_ref.to(dtype)
    print_tensor_info("Reference output c_ref", c_ref)

    # Compare results
    max_diff = torch.max(torch.abs(c - c_ref))
    mean_diff = torch.mean(torch.abs(c - c_ref))
    print(f"\nMax difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    print("relative diff: ", torch.mean(torch.abs(c - c_ref) / torch.abs(c_ref)))

    # woq_assert_near_eq(c_ref, c, 2)
    try:
        torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=0.1)
    except AssertionError as e:
        # torch.set_printoptions(threshold=10_000)
        print(f"  FAILURE: tensors are NOT close.")
        print(f"    Ref tensor: {c_ref.flatten()}")
        print(f"    Cutlass tensor: {c.flatten()}")
        print(
            f"    Max absolute difference: {torch.max(torch.abs(c.to(c_ref.dtype) - c_ref))}"
        )
        print(
            f"    Mean absolute difference: {torch.mean(torch.abs(c.to(c_ref.dtype) - c_ref))}"
        )
        print(f"    AssertionError: {e}")
        raise

    # Basic shape checks
    assert c.shape == (m, n)
    assert not torch.isnan(c).any()
    assert not torch.isinf(c).any()

    # Assert close


if __name__ == "__main__":
    pytest.main([__file__])
