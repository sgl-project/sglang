import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Added num_stages and more num_warps options
        triton.Config(
            {"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 64}, num_warps=8, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 512, "BLOCK_SIZE_N": 64}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64}, num_warps=16, num_stages=2
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64}, num_warps=8, num_stages=3
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_warps=4, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64}, num_warps=8, num_stages=4
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32}, num_warps=4, num_stages=5
        ),
    ],
    key=["M", "N"],
)
@triton.jit
def fused_qkv_quant_pad_tma_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    qp_ptr,
    kp_ptr,
    vp_ptr,
    sq_ptr,
    sk_ptr,
    sv_ptr,
    M,
    N,
    aligned_N,
    stride_im,
    stride_in,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load scales (scalars)
    sq = tl.load(sq_ptr)
    sk = tl.load(sk_ptr)
    sv = tl.load(sv_ptr)

    iq = 1.0 / (sq + 1e-12)
    ik = 1.0 / (sk + 1e-12)
    iv = 1.0 / (sv + 1e-12)

    # Define Input Block Pointers for Q, K, V (TMA core logic)
    # Input data shape is (M, N), using boundary checks
    q_in_block_ptr = tl.make_block_ptr(
        base=q_ptr,
        shape=(M, N),
        strides=(stride_im, stride_in),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    k_in_block_ptr = tl.make_block_ptr(
        base=k_ptr,
        shape=(M, N),
        strides=(stride_im, stride_in),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    v_in_block_ptr = tl.make_block_ptr(
        base=v_ptr,
        shape=(M, N),
        strides=(stride_im, stride_in),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    # Define Output Block Pointers for Q, K, V
    # Output data shape is (M, aligned_N)
    qp_out_block_ptr = tl.make_block_ptr(
        base=qp_ptr,
        shape=(M, aligned_N),
        strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    kp_out_block_ptr = tl.make_block_ptr(
        base=kp_ptr,
        shape=(M, aligned_N),
        strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    vp_out_block_ptr = tl.make_block_ptr(
        base=vp_ptr,
        shape=(M, aligned_N),
        strides=(stride_om, stride_on),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    # Use tl.load to asynchronously load data blocks.
    # Triton 3.5 automatically compiles this to TMA instructions on Hopper.
    xq = tl.load(q_in_block_ptr, boundary_check=(0, 1), padding_option="zero")
    xk = tl.load(k_in_block_ptr, boundary_check=(0, 1), padding_option="zero")
    xv = tl.load(v_in_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Quantize and store back
    tl.store(qp_out_block_ptr, (xq * iq).to(tl.float8e4nv), boundary_check=(0, 1))
    tl.store(kp_out_block_ptr, (xk * ik).to(tl.float8e4nv), boundary_check=(0, 1))
    tl.store(vp_out_block_ptr, (xv * iv).to(tl.float8e4nv), boundary_check=(0, 1))


def fused_qkv_per_tensor_quant_pad(q, k, v, qp, kp, vp, sq, sk, sv):
    """
    q, k, v: (tokens, heads, head_dim) - BF16
    qp, kp, vp: (tokens, heads, aligned_head_dim) - FP8
    sq, sk, sv: (1,) - Float32
    """
    # 1. Calculate scales in Python
    sq.copy_(torch.max(torch.abs(q)) / 448.0)
    sk.copy_(torch.max(torch.abs(k)) / 448.0)
    sv.copy_(torch.max(torch.abs(v)) / 448.0)

    # Ensure contiguity to avoid TMA page faults or stride errors
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    M = q.shape[0] * q.shape[1]
    N = q.shape[2]
    aligned_N = qp.shape[2]

    # Flatten to 2D for easier processing
    q_2d = q.view(-1, N)
    k_2d = k.view(-1, N)
    v_2d = v.view(-1, N)
    qp_2d = qp.view(-1, aligned_N)
    kp_2d = kp.view(-1, aligned_N)
    vp_2d = vp.view(-1, aligned_N)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(aligned_N, META["BLOCK_SIZE_N"]),
    )

    fused_qkv_quant_pad_tma_kernel[grid](
        q_2d,
        k_2d,
        v_2d,
        qp_2d,
        kp_2d,
        vp_2d,
        sq,
        sk,
        sv,
        M,
        N,
        aligned_N,
        q_2d.stride(0),
        q_2d.stride(1),
        qp_2d.stride(0),
        qp_2d.stride(1),
    )
