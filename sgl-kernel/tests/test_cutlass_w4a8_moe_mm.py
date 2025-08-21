import pytest
import torch
from sgl_kernel import cutlass_w4a8_moe_mm


def pack_int4_values_to_int8(int4_values_interleaved: torch.Tensor) -> torch.Tensor:
    if int4_values_interleaved.shape[-1] % 2 != 0:
        raise ValueError(
            "the last dim size of int4_values_interleaved tensor must be even."
        )

    input_tensor_int8 = int4_values_interleaved.to(torch.int8)

    low_nibbles = input_tensor_int8[..., 0::2]
    high_nibbles = input_tensor_int8[..., 1::2]

    packed_tensor = (high_nibbles << 4) | (low_nibbles & 0x0F)

    return packed_tensor.to(torch.int8)


def pack_interleave(num_experts, ref_weight, ref_scale):
    n, k = ref_weight.shape[1], ref_weight.shape[2]

    weight = pack_int4_values_to_int8(ref_weight.cpu()).cuda()
    w_q = weight.view((num_experts, n, k // 2)).view(torch.int8)
    w_q = w_q.contiguous()

    scale_interleaved = ref_scale.reshape(
        ref_scale.shape[0], ref_scale.shape[1], (ref_scale.shape[2] // 4), 4
    )  # [E, N, K/4, 4]
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)  # [E, K/4, N, 4]
    scale_interleaved = scale_interleaved.reshape(
        ref_scale.shape[0], ref_scale.shape[2] // 4, ref_scale.shape[1] * 4
    )  # [E, K/4, N*4]
    w_scale = scale_interleaved.contiguous()

    return w_q, w_scale


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
def test_int4_fp8_grouped_gemm_single_expert(batch_size):
    # Test parameters
    num_experts = 1
    m = batch_size  # batch size
    k = 512  # input dimension
    n = 1024  # output dimension
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = "cuda"
    debug = False

    print(f"\nTesting with batch_size={batch_size}")

    # Create input tensors with ones
    if debug:
        a = torch.ones(m, k, dtype=torch.bfloat16, device=device)
        ref_w = torch.ones(num_experts, n, k, dtype=torch.int8, device=device)
        a_scale = torch.ones(1, dtype=torch.float, device=device)
        ref_w_scale = torch.ones(num_experts, n, k // 128, dtype=dtype, device=device)
    else:
        a = torch.randn(m, k, dtype=dtype, device=device)
        ref_w = torch.randint(
            -8, 8, (num_experts, n, k), dtype=torch.int8, device=device
        )
        affine_coeff = 0.005
        a_scale = torch.randn(1, dtype=torch.float32).cuda() * 0.02
        ref_w_scale = (
            torch.randn(num_experts, n, k // 128, dtype=dtype, device=device)
            * affine_coeff
        )

    w, w_scale = pack_interleave(num_experts, ref_w, ref_w_scale)

    # Create expert offsets and problem sizes
    expert_offsets = torch.tensor([0, m], dtype=torch.int32, device=device)
    problem_sizes = torch.tensor([[n, m, k]], dtype=torch.int32, device=device)

    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = a_strides
    s_strides = c_strides

    # Quantize input
    a_q = torch.clamp((a / a_scale), -448.0, 448.0).to(torch.float8_e4m3fn).to(device)

    # Create output tensor
    c = torch.empty((m, n), dtype=torch.float16, device=device)
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

    # Reference implementation
    experts_selection_result = torch.full((m,), 0)
    c_ref = ref_grouped_gemm(
        c, a, a_scale, ref_w, ref_w_scale, num_experts, experts_selection_result
    )

    # Compare results
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


@pytest.mark.parametrize("batch_size", [2, 4, 8, 16])
@pytest.mark.parametrize("k", [512, 1024])
@pytest.mark.parametrize("n", [1024, 2048])
@pytest.mark.parametrize("num_experts", [2, 4, 6, 8])
def test_int4_fp8_grouped_gemm_multi_experts(batch_size, k, n, num_experts):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = "cuda"
    debug = False

    print(
        f"\nTesting with batch_size={batch_size}, k={k}, n={n}, num_experts={num_experts}"
    )

    if debug:
        a = torch.ones(batch_size, k, dtype=torch.bfloat16, device=device)
        ref_w = torch.ones(num_experts, n, k, dtype=torch.int8, device=device)
        a_scale = torch.ones(1, dtype=torch.float, device=device)
        ref_w_scale = torch.ones(num_experts, n, k // 128, dtype=dtype, device=device)
    else:
        a = torch.randn(batch_size, k, dtype=dtype, device=device)
        ref_w = torch.randint(
            -8, 8, (num_experts, n, k), dtype=torch.int8, device=device
        )
        affine_coeff = 0.005
        a_scale = torch.randn(1, dtype=torch.float32).cuda() * 0.02
        ref_w_scale = (
            torch.randn(num_experts, n, k // 128, dtype=dtype, device=device)
            * affine_coeff
        )

    w, w_scale = pack_interleave(num_experts, ref_w, ref_w_scale)

    # random select experts
    experts_selection_result = torch.randint(
        0, num_experts, (batch_size,), device=device
    )
    permutation = torch.argsort(experts_selection_result)
    expert_token_counts = torch.bincount(
        experts_selection_result, minlength=num_experts
    )

    # Create problem sizes and offsets for active experts
    problem_sizes = []
    for i in range(num_experts):
        problem_sizes.append([n, expert_token_counts[i].item(), k])
    problem_sizes = torch.tensor(problem_sizes, dtype=torch.int32, device=device)

    expert_offsets = []
    offset = 0
    for i in range(num_experts):
        expert_offsets.append(offset)
        offset += problem_sizes[i][1].item()
    expert_offsets = torch.tensor(expert_offsets, dtype=torch.int32, device=device)

    # Permute input and quantize
    a_perm = a[permutation]
    a_q_perm = (
        torch.clamp((a_perm / a_scale), -448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .to(device)
    )

    # Create stride tensors
    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = a_strides
    s_strides = c_strides

    c_perm = torch.empty((batch_size, n), dtype=torch.float16, device=device)
    cutlass_w4a8_moe_mm(
        c_perm,
        a_q_perm,
        w,
        a_scale,
        w_scale,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        s_strides,
        128,
        8,
    )

    # Un-permute the result
    c = torch.empty_like(c_perm)
    c[permutation] = c_perm
    c = c.to(dtype)

    c_ref = ref_grouped_gemm(
        c, a, a_scale, ref_w, ref_w_scale, num_experts, experts_selection_result
    )

    # Compare results
    try:
        torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=0.1)
    except AssertionError as e:
        print(f"  FAILURE: tensors are NOT close.")
        print(
            f"    Max absolute difference: {torch.max(torch.abs(c.to(c_ref.dtype) - c_ref))}"
        )
        print(
            f"    Mean absolute difference: {torch.mean(torch.abs(c.to(c_ref.dtype) - c_ref))}"
        )
        print(f"    AssertionError: {e}")
        raise


def ref_grouped_gemm(c, a, a_scale, w, w_scale, num_experts, experts_selection_result):
    dtype = torch.bfloat16
    c_ref = torch.zeros_like(c)
    a_q = torch.clamp((a / a_scale), -448.0, 448.0).to(torch.float8_e4m3fn)
    for i in range(num_experts):
        token_idx = torch.where(experts_selection_result == i)[0]
        if len(token_idx) == 0:
            continue
        a = a_q[token_idx]

        ref_w_scale_repeat = w_scale[i].repeat_interleave(128, dim=1).to(float)
        ref_w = (w[i].to(float) * ref_w_scale_repeat).to(dtype)
        c = torch.matmul(a.to(dtype), ref_w.t().to(dtype)) * a_scale
        c = c.to(dtype)
        c_ref[token_idx] = c.to(dtype)

    return c_ref


if __name__ == "__main__":
    pytest.main([__file__])
