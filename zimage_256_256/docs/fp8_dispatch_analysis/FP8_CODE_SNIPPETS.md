# FP8 GEMM Forward Path - Actual Code Snippets

## 1. Entry Point: Fp8LinearMethod.apply()

**File:** `python/sglang/multimodal_gen/runtime/layers/quantization/fp8.py` (lines 442-498)

```python
def apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if self.use_marlin:
        return apply_fp8_marlin_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            workspace=layer.workspace,
            size_n=layer.output_size_per_partition,
            size_k=layer.input_size_per_partition,
            bias=bias,
        )

    if self.block_quant:
        if use_intel_amx_backend(layer):
            return torch.ops.sgl_kernel.fp8_scaled_mm_cpu(
                x,
                layer.weight,
                layer.weight_scale_inv,
                self.quant_config.weight_block_size,
                bias,
                x.dtype,
                True,  # is_vnni
            )

        if isinstance(x, tuple):
            return self.w8a8_block_fp8_linear(
                input=x[0],
                weight=layer.weight,
                block_size=self.quant_config.weight_block_size,
                weight_scale=layer.weight_scale_inv,
                input_scale=x[1],
                bias=bias,
            )

        return self.w8a8_block_fp8_linear(
            input=x,
            weight=layer.weight,
            block_size=self.quant_config.weight_block_size,
            weight_scale=layer.weight_scale_inv,
            input_scale=None,
            bias=bias,
        )

    return apply_fp8_linear(
        input=x,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        input_scale=layer.input_scale,
        bias=bias,
        cutlass_fp8_supported=self.cutlass_fp8_supported,
        use_per_token_if_dynamic=False,
    )
```

---

## 2. Block Quantization Dispatcher

**File:** `python/sglang/srt/layers/quantization/fp8_utils.py` (lines 647-692)

```python
def deepgemm_w8a8_block_fp8_linear_with_fallback(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert input_scale is None

    output_dtype = input.dtype
    dtype_supported = output_dtype == torch.bfloat16

    # TODO: https://github.com/sgl-project/sglang/pull/6890#issuecomment-2943395737
    shape_supported = weight.shape[0] % 64 == 0 and weight.shape[1] % 128 == 0

    if not (shape_supported and dtype_supported):
        # fall back to triton
        # If weight_scale is in UE8M0 packed format (int32), convert back to float32
        # UE8M0 format has shape (N, K//block_k//4) with dtype int32
        # Triton expects shape (N//block_n, K//block_k) with dtype float32
        if weight_scale.dtype == torch.int32:
            weight_scale = _unpack_ue8m0_scale_for_triton(
                weight_scale, weight.shape, block_size
            )
        return triton_w8a8_block_fp8_linear(
            input, weight, block_size, weight_scale, input_scale, bias
        )

    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]

    q_input, x_scale = sglang_per_token_group_quant_fp8(
        input_2d,
        block_size[1],
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )

    output = w8a8_block_fp8_matmul_deepgemm(
        q_input, weight, x_scale, weight_scale, block_size, output_dtype=output_dtype
    )
    if bias is not None:
        output += bias
    return output.to(dtype=output_dtype).view(*output_shape)
```

---

## 3. Activation Quantization

**File:** `python/sglang/srt/layers/quantization/fp8_kernel.py` (lines 479-541)

```python
def sglang_per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    column_major_scales: bool = False,
    scale_tma_aligned: bool = False,
    scale_ue8m0: bool = False,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[torch.Tensor] = None,
    enable_v2: Optional[bool] = None,
):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    out_shape = (*x.shape[:-1], x.shape[-1] // (2 if fuse_silu_and_mul else 1))

    x_q = torch.empty(out_shape, device=x.device, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=out_shape,
        device=x.device,
        group_size=group_size,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
    )

    if x.shape[0] > 0:
        # Temporary
        if enable_sgl_per_token_group_quant_8bit:
            if enable_v2:
                sgl_per_token_group_quant_8bit(
                    x,
                    x_q,
                    x_s,
                    group_size,
                    eps,
                    fp8_min,
                    fp8_max,
                    scale_ue8m0,
                    fuse_silu_and_mul,
                    masked_m,
                    enable_v2=True,
                )
            else:
                sgl_per_token_group_quant_8bit_jit(
                    input=x,
                    output_q=x_q,
                    output_s=x_s,
                    group_size=group_size,
                    eps=eps,
                    fp8_min=fp8_min,
                    fp8_max=fp8_max,
                    scale_ue8m0=scale_ue8m0,
                )
        else:
            assert not enable_v2
            sgl_per_token_group_quant_fp8(
                x, x_q, x_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
            )

    return x_q, x_s
```

---

## 4. Scale Tensor Creation

**File:** `python/sglang/srt/layers/quantization/fp8_kernel.py` (lines 435-476)

```python
def create_per_token_group_quant_fp8_output_scale(
    x_shape,
    device,
    group_size,
    column_major_scales: bool,
    scale_tma_aligned: bool,
    scale_ue8m0: bool,
):
    if scale_ue8m0:
        assert column_major_scales and scale_tma_aligned
        *x_batch, x_q_mn, x_q_k = x_shape
        x_s_mn, x_s_k = x_q_mn, x_q_k // 128
        aligned_mn = ceil_align(x_s_mn, 4)
        aligned_k = ceil_align(x_s_k, 4)
        # TODO(FIXME): Fix cuda kernel and recover here to empty.
        return torch.empty(
            (*x_batch, aligned_k // 4, aligned_mn),
            device=device,
            dtype=torch.int,
        ).transpose(-1, -2)[..., :x_s_mn, :]
    elif column_major_scales:
        if scale_tma_aligned:
            # TODO extract "align" function
            # aligned to 4 * sizeof(float)
            aligned_size = (x_shape[-2] + 3) // 4 * 4
            return torch.empty(
                x_shape[:-2] + (x_shape[-1] // group_size, aligned_size),
                device=device,
                dtype=torch.float32,
            ).transpose(-1, -2)[: x_shape[-2], :]
        else:
            return torch.empty(
                (x_shape[-1] // group_size,) + x_shape[:-1],
                device=device,
                dtype=torch.float32,
            ).permute(-1, -2)
    else:
        return torch.empty(
            x_shape[:-1] + (x_shape[-1] // group_size,),
            device=device,
            dtype=torch.float32,
        )
```

---

## 5. DeepGEMM GEMM Execution

**File:** `python/sglang/srt/layers/quantization/fp8_kernel.py` (lines 1091-1106)

```python
def w8a8_block_fp8_matmul_deepgemm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    M, N, K, C = prepare_block_fp8_matmul_inputs(A, B, As, Bs, block_size, output_dtype)

    # Deepgemm only supports output tensor type as bfloat16
    assert C.dtype == torch.bfloat16 and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM

    deep_gemm_fp8_fp8_bf16_nt(A, As, B, Bs, C)

    return C
```

---

## 6. Output Tensor Allocation

**File:** `python/sglang/srt/layers/quantization/fp8_kernel.py` (lines 1043-1088)

```python
def prepare_block_fp8_matmul_inputs(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: List[int],
    output_dtype: torch.dtype = torch.float16,
) -> Tuple[int, int, int]:
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    assert A.shape[-1] == B.shape[-1]
    assert A.shape[:-1] == As.shape[:-1]
    assert A.is_contiguous()

    if As.dtype == torch.float:
        assert triton.cdiv(A.shape[-1], block_k) == As.shape[-1]
    elif As.dtype == torch.int:
        assert (
            triton.cdiv(triton.cdiv(A.shape[-1], block_k), 4) == As.shape[-1]
        ), f"{A.shape=} {As.shape=} {block_size=}"
    else:
        raise NotImplementedError

    M = A.numel() // A.shape[-1]

    assert B.ndim == 2
    assert B.is_contiguous()
    assert Bs.ndim == 2
    N, K = B.shape

    if Bs.dtype == torch.float:
        assert triton.cdiv(N, block_n) == Bs.shape[0]
        assert triton.cdiv(K, block_k) == Bs.shape[1]
    elif Bs.dtype == torch.int:
        assert N == Bs.shape[0], f"{B.shape=} {Bs.shape=} {block_size=}"
        assert (
            triton.cdiv(triton.cdiv(K, block_k), 4) == Bs.shape[1]
        ), f"{B.shape=} {Bs.shape=} {block_size=}"
    else:
        raise NotImplementedError

    C_shape = A.shape[:-1] + (N,)
    C = A.new_empty(C_shape, dtype=output_dtype)

    return M, N, K, C
```

---

## 7. DeepGEMM Kernel Call

**File:** `python/sglang/srt/layers/quantization/fp8_kernel.py` (lines 113-120)

```python
def deep_gemm_fp8_fp8_bf16_nt(
    A: torch.Tensor,
    As: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    C: torch.Tensor,
) -> None:
    deep_gemm_wrapper.gemm_nt_f8f8bf16((A, As), (B, Bs), C)
```

---

## 8. DeepGEMM Wrapper

**File:** `python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py` (lines 84-102)

```python
def gemm_nt_f8f8bf16(
    lhs: Tuple[torch.Tensor, torch.Tensor],
    rhs: Tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
):
    m, k = lhs[0].shape
    n, _ = rhs[0].shape
    num_groups = 1
    kernel_type = compile_utils.DeepGemmKernelType.GEMM_NT_F8F8BF16

    _sanity_check_input(lhs)
    _sanity_check_input(rhs)

    with compile_utils.deep_gemm_execution_hook(m, n, k, num_groups, kernel_type):
        deep_gemm.fp8_gemm_nt(
            lhs,
            rhs,
            out,
        )
```

---

## 9. BF16 Reference Path

**File:** `python/sglang/multimodal_gen/runtime/layers/linear.py` (lines 152-160)

```python
def apply(
    self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    output = (
        F.linear(x, layer.weight, bias)
        if current_platform.is_amp_supported() or bias is None
        else F.linear(x, layer.weight, bias.to(x.dtype))
    )  # NOTE: this line assumes that we are using amp when using cuda and is needed to account for the fact that amp isn't supported in mps
    return output
```

---

## 10. Triton Quantization Kernel

**File:** `python/sglang/srt/layers/quantization/fp8_kernel.py` (lines 167-212)

```python
@triton.jit
def _per_token_group_quant_8bit_colmajor(
    # Pointers to inputs and output
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    # Num columns of y
    y_num_columns,
    # Stride from one column to the next of y_s
    y_s_col_stride,
    # Avoid to divide zero
    eps,
    # Information for float8
    bit8_min,
    bit8_max,
    # Meta-parameters
    BLOCK: tl.constexpr,
    SCALE_UE8M0: tl.constexpr,
):
    """A Triton-accelerated function to perform per-token-group
    quantization on a tensor.
    This function converts the tensor values into float8 values.
    """
    # Map the program id to the row of X and Y it should compute.
    g_id = tl.program_id(0)
    y_ptr += g_id.to(tl.int64) * group_size
    y_q_ptr += g_id.to(tl.int64) * group_size

    # Convert g_id the flattened block coordinate to 2D so we can index
    # into the output y_scales matrix
    blocks_per_row = y_num_columns // group_size
    scale_col = g_id % blocks_per_row
    scale_row = g_id // blocks_per_row
    y_s_ptr += scale_col * y_s_col_stride + scale_row

    cols = tl.arange(0, BLOCK)  # group_size <= BLOCK
    mask = cols < group_size

    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Quant
    _absmax = tl.maximum(tl.max(tl.abs(y)), eps)
    y_s = _absmax / bit8_max
    if SCALE_UE8M0:
        y_s = tl.exp2(tl.ceil(tl.log2(tl.abs(y_s))))
    y_q = tl.clamp(y / y_s, bit8_min, bit8_max).to(y_q_ptr.dtype.element_ty)

    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)
```

