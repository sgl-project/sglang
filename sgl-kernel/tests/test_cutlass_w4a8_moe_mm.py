import pytest
import torch
from sgl_kernel import cutlass_w4a8_moe_mm, sgl_per_tensor_quant_fp8
from utils import is_hopper

from sglang.srt.layers.quantization.fp8_kernel import (
    sglang_per_token_group_quant_fp8,
    sglang_per_token_group_quant_8bit,
)
from sglang.srt.layers.moe.ep_moe.kernels import (
    deepep_ll_get_cutlass_w4a8_moe_mm_data,
)
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

    alignment = 4 if k % 512 == 0 else 1
    scale_interleaved = ref_scale.reshape(
        ref_scale.shape[0],
        ref_scale.shape[1],
        (ref_scale.shape[2] // alignment),
        alignment,
    )  # [E, N, K/4, 4]
    scale_interleaved = scale_interleaved.permute(0, 2, 1, 3)  # [E, K/4, N, 4]
    scale_interleaved = scale_interleaved.reshape(
        ref_scale.shape[0],
        ref_scale.shape[2] // alignment,
        ref_scale.shape[1] * alignment,
    )  # [E, K/4, N*4]
    w_scale = scale_interleaved.contiguous()

    return w_q, w_scale


@pytest.mark.skipif(
    not is_hopper(),
    reason="cutlass_w4a8_moe_mm is only supported on sm90",
)
@pytest.mark.parametrize("num_experts", [8, 16, 32])
@pytest.mark.parametrize("m", [32, 128, 512])
@pytest.mark.parametrize("k", [128, 256, 512, 768, 7168])
@pytest.mark.parametrize("n", [ 512, 1024, 7168])
def test_int4_fp8_grouped_gemm3d_expert(num_experts, m, k, n): # low-latency 3d input
    # Test parameters
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = "cuda"
    debug = False

    # Create input tensors with ones
    if debug:
        ref_w = torch.ones(num_experts, n, k, dtype=torch.int8, device=device)
        ref_w_scale = torch.ones(num_experts, n, k // 128, dtype=dtype, device=device)
    else:
        ref_w = torch.randint(
            -8, 8, (num_experts, n, k), dtype=torch.int8, device=device
        )
        affine_coeff = 0.005
        ref_w_scale = (
            torch.randn(num_experts, n, k // 128, dtype=dtype, device=device)
            * affine_coeff
        )

    w, w_scale = pack_interleave(num_experts, ref_w, ref_w_scale)
    a = torch.zeros((num_experts, m, k), dtype=dtype, device=device)
    # Create expert offsets and problem sizes
    expert_offsets = []
    offset = 0
    masked_m_list = []
    for i in range(num_experts):
        m_i = torch.randint(0, m + 1, (1,)).item()
        masked_m_list.append(m_i)
        if debug:
            a_i = torch.ones(m_i, k, dtype=dtype, device=device)
        else:
            a_i = torch.randn(m_i, k, dtype=dtype, device=device)
        a[i, : m_i, :] = a_i

        expert_offsets.append(offset)
        offset += m_i
        

    expert_offsets = torch.tensor(expert_offsets, dtype=torch.int32, device=device)
    masked_m = torch.tensor(masked_m_list, dtype=torch.int32, device=device)

    problem_sizes1 = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=device
    )
    problem_sizes2 = torch.empty(
        (num_experts, 3), dtype=torch.int32, device=device
    )
    problem_sizes1, problem_sizes2 = deepep_ll_get_cutlass_w4a8_moe_mm_data(
        masked_m,
        problem_sizes1,
        problem_sizes2,
        num_experts,
        n/2,
        k,
    )
    

    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = a_strides
    sb_strides = c_strides

    # Quantize input
    # a_q, a_scale = sglang_per_token_group_quant_fp8(a, 128)

    a_q, a_scale = sglang_per_token_group_quant_8bit(
        x=a,
        dst_dtype=torch.float8_e4m3fn,
        group_size=128,
        masked_m=masked_m,
        column_major_scales=False,
        scale_tma_aligned=False,
        fuse_silu_and_mul=False,
        enable_v2=True,
    )

    if k % 512 != 0:  # K need padding to 4 for B scale
        assert k % 128 == 0, "k must be multiple of 128 or 512"
        a_scale_padded = torch.empty(
            (a_scale.shape[0], a_scale.shape[1], a_scale.shape[2] * 4), device=device, dtype=a_scale.dtype
        )
        for i in range(a_scale.shape[0]):
            for j in range(a_scale.shape[1]):
                a_scale_padded[i, j] = a_scale[i, j].repeat_interleave(4, dim=0)
        sa_strides = torch.full(
            (num_experts, 3), k // 128 * 4, device=device, dtype=torch.int64
        )
    else:
        a_scale_padded = a_scale
        sa_strides = torch.full(
            (num_experts, 3), k // 128, device=device, dtype=torch.int64
        )

    # Create output tensor
    c = torch.zeros((num_experts, m, n), dtype=torch.bfloat16, device=device)
    cutlass_w4a8_moe_mm( 
        c,
        a_q,
        w,
        a_scale_padded,
        w_scale,
        expert_offsets,
        problem_sizes1,
        a_strides,
        b_strides,
        c_strides,
        sa_strides,
        sb_strides,
        128,
        8,
    )

    # Reference implementation
    c_ref = ref_grouped_gemm3d(
        c,
        a_q,
        a_scale,
        ref_w,
        ref_w_scale,
        num_experts,
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


def _per_tensor_quant_fp8(
    x: torch.Tensor,
    dtype: torch.dtype = torch.float8_e4m3fn,
):
    assert x.is_contiguous(), "`x` is not contiguous"

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    x_s = torch.empty(
        1,
        device=x.device,
        dtype=torch.float32,
    )
    sgl_per_tensor_quant_fp8(x, x_q, x_s, is_static=False)
    return x_q, x_s


@pytest.mark.skipif(
    not is_hopper(),
    reason="cutlass_w4a8_moe_mm is only supported on sm90",
)
@pytest.mark.parametrize("batch_size", [2, 4, 8, 16, 32])
@pytest.mark.parametrize("k", [128, 256, 512, 1024, 2048, 4096, 7168])
@pytest.mark.parametrize("n", [128, 256, 512, 1024, 2048, 7168])
@pytest.mark.parametrize("num_experts", [2, 4, 6, 8])
def test_int4_fp8_grouped_gemm_multi_experts(batch_size, k, n, num_experts): # normal 2d input
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
        ref_w_scale = torch.ones(num_experts, n, k // 128, dtype=dtype, device=device)
    else:
        a = torch.randn(batch_size, k, dtype=dtype, device=device)
        ref_w = torch.randint(
            -8, 8, (num_experts, n, k), dtype=torch.int8, device=device
        )
        affine_coeff = 0.005
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
    a_q, a_scale = sglang_per_token_group_quant_fp8(a, 128)

    if k % 512 != 0:  # K need padding to 4 for B scale
        assert k % 128 == 0, "k must be multiple of 128 or 512"
        a_scale_padded = torch.empty(
            (a_scale.shape[0], a_scale.shape[1] * 4), device=device, dtype=a_scale.dtype
        )
        for i in range(a_scale.shape[0]):
            a_scale_padded[i] = a_scale[i].repeat_interleave(4, dim=0)
        sa_strides = torch.full(
            (num_experts, 3), k // 128 * 4, device=device, dtype=torch.int64
        )
    else:
        a_scale_padded = a_scale
        sa_strides = torch.full(
            (num_experts, 3), k // 128, device=device, dtype=torch.int64
        )

    a_q_perm = a_q[permutation]
    a_scale_perm = a_scale_padded[permutation]

    # Create stride tensors
    a_strides = torch.full((num_experts, 3), k, device=device, dtype=torch.int64)
    c_strides = torch.full((num_experts, 3), n, device=device, dtype=torch.int64)
    b_strides = a_strides
    sb_strides = c_strides

    c_perm = torch.empty((batch_size, n), dtype=torch.bfloat16, device=device)
    cutlass_w4a8_moe_mm(
        c_perm,
        a_q_perm,
        w,
        a_scale_perm,
        w_scale,
        expert_offsets,
        problem_sizes,
        a_strides,
        b_strides,
        c_strides,
        sa_strides,
        sb_strides,
        128,
        8,
    )

    # Un-permute the result
    c = torch.empty_like(c_perm)
    c[permutation] = c_perm
    c = c.to(dtype)

    c_ref = ref_grouped_gemm(
        c, a_q, a_scale, ref_w, ref_w_scale, num_experts, experts_selection_result
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


def ref_grouped_gemm(
    c, a_q, a_scale, w, w_scale, num_experts, experts_selection_result
):
    dtype = torch.bfloat16
    c_ref = torch.zeros_like(c)
    a_q = a_q.to(torch.float32) * a_scale.repeat_interleave(128, dim=1).to(
        torch.float32
    )
    for i in range(num_experts):
        token_idx = torch.where(experts_selection_result == i)[0]
        if len(token_idx) == 0:
            continue
        a = a_q[token_idx]

        ref_w_scale_repeat = w_scale[i].repeat_interleave(128, dim=1).to(torch.float32)
        ref_w = w[i].to(torch.float32) * ref_w_scale_repeat
        c = torch.matmul(a.to(torch.float32), ref_w.t())
        c_ref[token_idx] = c.to(dtype)

    return c_ref

def ref_grouped_gemm3d(
    c, a_q, a_scale, w, w_scale, num_experts
):
    dtype = torch.bfloat16
    c_ref = torch.zeros_like(c)    
    for i in range(num_experts):
        a = a_q[i].to(torch.float32) * a_scale[i].repeat_interleave(128, dim=-1).to(
            torch.float32
        )
        ref_w_scale_repeat = w_scale[i].repeat_interleave(128, dim=1).to(torch.float32)
        ref_w = w[i].to(torch.float32) * ref_w_scale_repeat
        c = torch.matmul(a, ref_w.t())
        c_ref[i] = c.to(dtype)
    return c_ref


if __name__ == "__main__":
    pytest.main([__file__])
