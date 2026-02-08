![ALT](../../images/gemm-hierarchy-with-epilogue-no-labels.png "CUTLASS Functionality")


# Functionality

Note : CUTLASS-3 requires users to use CUDA 11.4 or newer, and SM70 or newer, for the target toolkit and architecture, respectively.

- N - Column Major Matrix
- T - Row Major matrix
- {N,T} x {N,T} - All combinations, i.e., NN, NT, TN, TT
- [NHWC](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/layout/tensor.h#L63-206) - 4 dimension tensor used for convolution
- [NCxHWx](https://github.com/NVIDIA/cutlass/tree/main/include/cutlass/layout/tensor.h#L290-395) - Interleaved 4 dimension tensor used for convolution
- f - floating point
- s - signed int
- b - bit
- cf - complex float
- bf16 - bfloat16
- tf32 - tfloat32
- Simt - Use Simt CUDA Core MMA
- TensorOp - Use Tensor Core MMA
- SpTensorOp - Use Sparse Tensor Core MMA
- WmmaTensorOp - Use WMMA abstraction to use Tensor Core MMA

## Device-level GEMM

The following tables summarize device-level GEMM kernels in CUTLASS, organized by opcode class, data type, and layout.
Hyperlinks to relevant unit tests demonstrate how specific template instances may be defined.

### CUTLASS 3.x Kernels

|**Opcode Class** | **Compute Capability** | **CUDA Toolkit** | **Data Type**                  | **Layouts**            | **Unit Test**    |
|-----------------|------------------------|------------------|--------------------------------|------------------------|------------------|
| **TensorOp**        | 90a                 |  12.0+           | `f16 * f16 + { f16, f32 } => { f16, f32 }`       | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op_f32_cluster_warpspecialized.cu) |
| **TensorOp**        | 90a                 |  12.0+           | `bf16 * bf16 + { f16, f32 } => { bf16, f32 }`| {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu) |
| **TensorOp**        | 90a                 |  12.0+           | `{f32, tf32} * {f32, tf32} + f32 => f32`| { T } x { N } => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_f32_f32_f32_tensor_op_f32.cu) |
| **TensorOp**        | 90a                 |  12.0+           | `s8 * s8 + s32 => {s32, s8}`   | { T } x { N } => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/sm90_gemm_s8_s8_s8_tensor_op_s32.cu) |
 

### CUTLASS 2.x Kernels

|**Opcode Class** | **Compute Capability** | **CUDA Toolkit** | **Data Type**                  | **Layouts**            | **Unit Test**    |
|-----------------|------------------------|------------------|--------------------------------|------------------------|------------------|
| **Simt**        | 50+                    |  11.4+            | `f32 * f32 + f32 => f32`       | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/simt_sgemm_nt_sm50.cu)                |
| **Simt**        | 50+                    |  11.4+            | `f64 * f64 + f64 => f64`       | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/simt_dgemm_nt_sm50.cu)                |
| **Simt**        | 60+                    |  11.4+            | `f16 * f16 + f16 => f16`       | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/simt_hgemm_nt_sm50.cu)                |
| **Simt**        | 61+                    |  11.4+            | `s8 * s8 + s32 => {s32,s8}`    | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/simt_igemm_nt_sm50.cu)              |
| **WmmaTensorOp**    | 70+                |  11.4+            | `f16 * f16 + f16 => f16`       | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16t_f16t_f16n_wmma_tensor_op_f16_sm70.cu)     |
| **WmmaTensorOp**    | 70+                |  11.4+            | `f16 * f16 + f32 => {f16, f32}`| {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16t_f16t_f16n_wmma_tensor_op_f32_sm70.cu)                |
| **WmmaTensorOp**    | 75+                |  11.4+           | `s8 * s8 + s32 => {s32, s8}`   | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s8t_wmma_tensor_op_s32_sm72.cu) |
| **WmmaTensorOp**    | 75+                |  11.4+           | `s4 * s4 + s32 => {s32, s4}`   | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s4t_s4n_s4t_wmma_tensor_op_s32_sm75.cu)                |
| **WmmaTensorOp**    | 75+                |  11.4+           | `b1 ^ b1 + s32 => {s32, b1}`   | { T } x { N } => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_b1t_b1n_b1t_wmma_tensor_op_s32_sm75.cu)                |
| **TensorOp**        | 70+                |  11.4+           | `f16 * f16 + f16 => f16`       | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f16t_volta_tensor_op_f16_sm70.cu)                |
| **TensorOp**        | 70+                |  11.4+           | `f16 * f16 + f32 => {f16, f32}`| {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f16t_volta_tensor_op_f32_sm70.cu)                |
| **TensorOp**        | 75+                |  11.4+           | `f16 * f16 + f16 => f16`       | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f16t_tensor_op_f16_sm75.cu) |
| **TensorOp**        | 75+                |  11.4+           | `f16 * f16 + f32 => {f16, f32}`| {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f16t_tensor_op_f32_sm75.cu) |
| **TensorOp**        | 75+                |  11.4+           | `s8 * s8 + s32 => {s32, s8}`   | { T } x { N } => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s32n_tensor_op_s32_sm75.cu) |
| **TensorOp**        | 75+                |  11.4+           | `s4 * s4 + s32 => {s32, s4}`   | { T } x { N } => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s4t_s4n_s32n_tensor_op_s32_sm75.cu) |
| **TensorOp**        | 75+                |  11.4+           | `b1 ^ b1 + s32 => {s32, b1}`   | { T } x { N } => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_b1t_b1n_s32n_tensor_op_s32_sm75.cu) |
| **TensorOp**        | 80+                |  11.4+           | `f16 * f16 + f16 => f16`       | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f16t_tensor_op_f16_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `f16 * f16 + f32 => {f16, f32}`| {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16t_f16t_tensor_op_f32_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `bf16 * bf16 + f32 => {bf16, f32}`| {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_bf16n_bf16t_bf16t_tensor_op_f32_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `tf32 * tf32 + f32 => f32`| {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f32n_f32t_f32t_tensor_op_f32_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `s8 * s8 + s32 => {s32, s8}`   | { T } x { N } => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s32n_tensor_op_s32_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `s4 * s4 + s32 => {s32, s4}`   | { T } x { N } => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s4t_s4n_s32n_tensor_op_s32_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `b1 ^ b1 + s32 => {s32, b1}`   | { T } x { N } => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_b1t_b1n_s32n_tensor_op_s32_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `f64 * f64 + f64 => f64`       | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f64n_f64t_f64t_tensor_op_f64_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `cf32 * cf32 + cf32 => cf32`       | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_cf32n_cf32t_cf32t_tensor_op_tf32_f32_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `cf64 * cf64 + cf64 => cf64`       | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_cf64n_cf64t_cf64t_tensor_op_f64_sm80.cu), [Gaussian 3m](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_cf64n_cf64t_cf64t_tensor_op_f64_gaussian_sm80.cu) |
| **SpTensorOp**      | 80+                |  11.4+           | `f16 * f16 + f32 => {f16, f32}`    | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16n_f32t_tensor_op_f32_sparse_sm80.cu) |
| **SpTensorOp**      | 80+                |  11.4+           | `bf16 * bf16 + f32 => {bf16, f32}` | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f16n_f16n_f32t_tensor_op_f32_sparse_sm80.cu) |
| **SpTensorOp**      | 80+                |  11.4+           | `tf32 * tf32 + f32 => f32`         | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f32n_f32n_f32t_tensor_op_f32_sparse_sm80.cu) |
| **SpTensorOp**      | 80+                |  11.4+           | `s8 * s8 + s32 => {s8, s32}`       | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s8t_s8n_s32t_tensor_op_s32_sparse_sm80.cu) |
| **SpTensorOp**      | 80+                |  11.4+           | `s4 * s4 + s32 => {s4, s32}`       | {N,T} x {N,T} => {N,T} | [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_s4t_s4n_s32t_tensor_op_s32_sparse_sm80.cu) |
| **TensorOp**        | 90+                |  11.8+           | `f64 * f64 + f64 => f64`       | {N,T} x {N,T} => {N,T} |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/gemm/device/gemm_f64n_f64t_f64t_tensor_op_f64_sm90.cu) |


## Device-level Implicit GEMM convolution

The following table summarizes device-level implicit GEMM convolution kernels in CUTLASS, organized by opcode class, data type, and layout.
Hyperlinks to relevant conv2d fprop unit tests demonstrate how specific template instances may be defined. 
One can find and/or create equivalent dgrad and wgrad convolutional operators.

|**Opcode Class** | **Compute Capability** | **CUDA Toolkit** | **Data Type**                  | **Layouts**      | **Unit Test**    |
|-----------------|------------------------|------------------|--------------------------------|------------------|------------------|
| **Simt**            | 50+                |  11.4+            | `f32 * f32 + f32 => f32`       | NHWC             |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32_sm50.cu)                |
| **Simt**            | 50+                |  11.4+            | `cf32 * cf32 + cf32 => cf32`   | NHWC             |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_cf32nhwc_cf32nhwc_cf32nhwc_simt_f32_sm50.cu)                |
| **TensorOp**        | 70+                |  11.4+           | `f16 * f16 + f32 => {f16, f32}`| NHWC             |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_sm70.cu) |
| **TensorOp**        | 75+                |  11.4+           | `f16 * f16 + f32 => {f16, f32}`| NHWC             |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_sm75.cu) |
| **TensorOp**        | 75+                |  11.4+           | `s8 * s8 + s32 => {s32, s8}`   | NHWC, NCxHWx     |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s8nhwc_s8nhwc_s32nhwc_tensor_op_s32_sm75.cu), [ncxhwx](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm75.cu) |
| **TensorOp**        | 75+                |  11.4+           | `s4 * s4 + s32 => {s32, s4}`   | NHWC, NCxHWx     |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s4nhwc_s4nhwc_s32nhwc_tensor_op_s32_sm75.cu), [ncxhwx](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s4ncxhwx_s4cxrskx_s4ncxhwx_tensor_op_s32_sm75.cu) |
| **Simt**            | 80+                |  11.4+           | `f32 * f32 + f32 => f32`       | NHWC             |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f32nhwc_f32nhwc_f32nhwc_simt_f32_sm80.cu)                |
| **Simt**            | 80+                |  11.4+           | `cf32 * cf32 + cf32 => cf32`   | NHWC             |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_cf32nhwc_cf32nhwc_cf32nhwc_simt_f32_sm80.cu)                |
| **TensorOp**        | 80+                |  11.4+           | `f16 * f16 + f32 => {f16, f32}`| NHWC             |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `f16 * f16 + f16 => f16`       | NHWC             |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `tf32 * tf32 + f32 => f32`     | NHWC             |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `s8 * s8 + s32 => {s32, s8}`   | NHWC, NCxHWx     |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s8nhwc_s8nhwc_s32nhwc_tensor_op_s32_sm80.cu), [ncxhwx](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s8ncxhwx_s8cxrskx_s8ncxhwx_tensor_op_s32_sm80.cu) |
| **TensorOp**        | 80+                |  11.4+           | `s4 * s4 + s32 => {s32, s4}`   | NHWC, NCxHWx     |  [example](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s4nhwc_s4nhwc_s32nhwc_tensor_op_s32_sm80.cu), [ncxhwx](https://github.com/NVIDIA/cutlass/tree/main/test/unit/conv/device/conv2d_fprop_implicit_gemm_s4ncxhwx_s4cxrskx_s4ncxhwx_tensor_op_s32_sm80.cu) |



## Warp-level Matrix Multiply with Tensor Cores

The following table summarizes supported warp level shapes for each TensorOp instruction.

|**Opcode Class** | **Instruction Shape** | **Warp Shapes**                            |
|-----------------|-----------------------|--------------------------------------------|
| **TensorOp**    | 8-by-8-by-4           | 32x32x4, 32x64x4, 64x32x4, 64x64x4         |
| **TensorOp**    | 16-by-8-by-8          | 32x32x8, 32x64x8, 64x32x8, 64x64x8         |
| **TensorOp**    | 16-by-8-by-16         | 32x32x16, 32x64x16, 64x32x16, 64x64x16     |
| **TensorOp**    | 8-by-8-by-16          | 32x32x16, 32x64x16, 64x32x16, 64x64x16     |
| **TensorOp**    | 8-by-8-by-32          | 32x32x32, 32x64x32, 64x32x32, 64x64x32     |
| **TensorOp**    | 16-by-8-by-32         | 32x32x32, 32x64x32, 64x32x32, 64x64x32     |
| **TensorOp**    | 16-by-8-by-64         | 32x32x64, 32x64x64, 64x32x64, 64x64x64     |
| **TensorOp**    | 8-by-8-by-128         | 32x32x128, 32x64x128, 64x32x128, 64x64x128 |
| **TensorOp**    | 16-by-8-by-256        | 32x32x256, 32x64x256, 64x32x256, 64x64x256 |
| **SpTensorOp**  | 16-by-8-by-16         | 64x64x16, 64x32x16, 32x64x16, 32x32x16     |
| **SpTensorOp**  | 16-by-8-by-32         | 64x64x32, 64x32x32, 32x64x32, 32x32x32     |
| **SpTensorOp**  | 16-by-8-by-64         | 64x64x64, 64x32x64, 32x64x64, 32x32x64     |
| **SpTensorOp**  | 16-by-8-by-128        | 64x64x128, 64x32x128, 32x64x128, 32x32x128 |


TensorOp instructions depend on a permuted shared memory layout that can be efficiently
loaded from. The following tables summarize the destination shared memory layout that
can be targeted by matrix operands. It is assumed that each thread loads 128b vectors
from global memory with layout specified in the column "GMEM Layout."

**TensorOp 8-by-8-by-4.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                         |
|-----------|--------------|-----------------|-----------------------------------------|
|  **A**    | `half_t`     | `ColumnMajor`   | `ColumnMajorVoltaTensorOpCongruous<16>` |
|  **A**    | `half_t`     | `RowMajor`      | `RowMajorVoltaTensorOpCrosswise<16>`    |
|  **B**    | `half_t`     | `ColumnMajor`   | `ColumnMajorVoltaTensorOpCrosswise<16>` |
|  **B**    | `half_t`     | `RowMajor`      | `RowMajorVoltaTensorOpCongruous<16>`    |
|  **C**    | `half_t`     | `RowMajor`      | `RowMajor`                              |
|  **C**    | `float`      | `RowMajor`      | `RowMajor`                              |

**TensorOp 16-by-8-by-8.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                    |
|-----------|--------------|-----------------|------------------------------------|
|  **A**    | `half_t`     | `ColumnMajor`   | `ColumnMajorTensorOpCongruous<16>` |
|  **A**    | `half_t`     | `RowMajor`      | `RowMajorTensorOpCrosswise<16>`    |
|  **B**    | `half_t`     | `ColumnMajor`   | `ColumnMajorTensorOpCrosswise<16>` |
|  **B**    | `half_t`     | `RowMajor`      | `RowMajorTensorOpCongruous<16>`    |
|  **C**    | `half_t`     | `RowMajor`      | `RowMajor`                         |
|  **C**    | `float`      | `RowMajor`      | `RowMajor`                         |

**TensorOp 16-by-8-by-8.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                    |
|-----------|--------------|-----------------|------------------------------------|
|  **A**    | `tfloat32_t`     | `ColumnMajor`   | `ColumnMajorTensorOpCongruous<32>` |
|  **A**    | `tfloat32_t`     | `RowMajor`      | `RowMajorTensorOpCrosswise<32>`    |
|  **B**    | `tfloat32_t`     | `ColumnMajor`   | `ColumnMajorTensorOpCrosswise<32>` |
|  **B**    | `tfloat32_t`     | `RowMajor`      | `RowMajorTensorOpCongruous<32>`    |
|  **C**    | `float`          | `RowMajor`      | `RowMajor`                         |


**TensorOp 16-by-8-by-16.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                    |
|-----------|--------------|-----------------|------------------------------------|
|  **A**    | `half_t`, `bfloat16_t`     | `ColumnMajor`   | `ColumnMajorTensorOpCongruous<16>` |
|  **A**    | `half_t`, `bfloat16_t`     | `RowMajor`      | `RowMajorTensorOpCrosswise<16>`    |
|  **B**    | `half_t`, `bfloat16_t`     | `ColumnMajor`   | `ColumnMajorTensorOpCrosswise<16>` |
|  **B**    | `half_t`, `bfloat16_t`     | `RowMajor`      | `RowMajorTensorOpCongruous<16>`    |
|  **C**    | `half_t`     | `RowMajor`      | `RowMajor`                         |
|  **C**    | `float`      | `RowMajor`      | `RowMajor`                         |

**TensorOp 8-by-8-by-4.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                    |
|-----------|--------------|-----------------|------------------------------------|
|  **A**    | `double`     | `ColumnMajor`   | `ColumnMajorTensorOpCongruous<64>` |
|  **A**    | `double`     | `RowMajor`      | `RowMajorTensorOpCrosswise<64>`    |
|  **B**    | `double`     | `ColumnMajor`   | `ColumnMajorTensorOpCrosswise<64>` |
|  **B**    | `double`     | `RowMajor`      | `RowMajorTensorOpCongruous<64>`    |
|  **C**    | `double`     | `RowMajor`      | `RowMajor`                         |

**TensorOp 8-by-8-by-16.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                    |
|-----------|--------------|-----------------|------------------------------------|
|  **A**    | `int8_t`     | `RowMajor`      | `RowMajorTensorOpCrosswise<8>`     |
|  **B**    | `int8_t`     | `ColumnMajor`   | `ColumnMajorTensorOpCongruous<8>`  |
|  **C**    | `int32_t`    | `RowMajor`      | `RowMajor`                         |

**TensorOp 16-by-8-by-32.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                    |
|-----------|--------------|-----------------|------------------------------------|
|  **A**    | `int8_t`     | `RowMajor`      | `RowMajorTensorOpCrosswise<8>`     |
|  **B**    | `int8_t`     | `ColumnMajor`   | `ColumnMajorTensorOpCongruous<8>`  |
|  **C**    | `int32_t`    | `RowMajor`      | `RowMajor`                         |

**TensorOp 8-by-8-by-32.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                    |
|-----------|--------------|-----------------|------------------------------------|
|  **A**    | `int4b_t`    | `RowMajor`      | `RowMajorTensorOpCrosswise<4>`     |
|  **B**    | `int4b_t`    | `ColumnMajor`   | `ColumnMajorTensorOpCongruous<4>`  |
|  **C**    | `int32_t`    | `RowMajor`      | `RowMajor`                         |

**TensorOp 16-by-8-by-64.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                    |
|-----------|--------------|-----------------|------------------------------------|
|  **A**    | `int4b_t`    | `RowMajor`      | `RowMajorTensorOpCrosswise<4>`     |
|  **B**    | `int4b_t`    | `ColumnMajor`   | `ColumnMajorTensorOpCongruous<4>`  |
|  **C**    | `int32_t`    | `RowMajor`      | `RowMajor`                         |

**TensorOp 8-by-8-by-128.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                    |
|-----------|--------------|-----------------|------------------------------------|
|  **A**    | `bin1_t`     | `RowMajor`      | `RowMajorTensorOpCrosswise<4>`     |
|  **B**    | `bin1_t`     | `ColumnMajor`   | `ColumnMajorTensorOpCongruous<4>`  |
|  **C**    | `int32_t`    | `RowMajor`      | `RowMajor`                         |


**SpTensorOp 16-by-8-by-16.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                    |
|-----------|--------------|-----------------|------------------------------------|
|  **A**    | `tfloat32_t` | `RowMajor`      | `RowMajorTensorOpCrosswise<32, 32>`   |
|  **B**    | `tfloat32_t` | `ColumnMajor`   | `ColumnMajorTensorOpCrosswise<32, 32>`|
|  **C**    | `float`      | `RowMajor`      | `RowMajor`                            |

**SpTensorOp 16-by-8-by-32.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                       |
|-----------|--------------|-----------------|---------------------------------------|
|  **A**    | `half_t`     | `RowMajor`      | `RowMajorTensorOpCrosswise<16, 64>`   |
|  **B**    | `half_t`     | `ColumnMajor`   | `ColumnMajorTensorOpCrosswise<16, 64>`|
|  **C**    | `float`      | `RowMajor`      | `RowMajor`                            |

**SpTensorOp 16-by-8-by-64.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                       |
|-----------|--------------|-----------------|---------------------------------------|
|  **A**    | `int8_t`     | `RowMajor`      | `RowMajorTensorOpCrosswise<8, 128>`   |
|  **B**    | `int8_t`     | `ColumnMajor`   | `ColumnMajorTensorOpCrosswise<8, 128>`|
|  **C**    | `int32_t`    | `RowMajor`      | `RowMajor`                            |

**SpTensorOp 16-by-8-by-128.**

|**Operand**|**Element**   | **GMEM Layout** | **SMEM Layout**                    |
|-----------|--------------|-----------------|------------------------------------|
|  **A**    | `int4b_t`    | `RowMajor`      | `RowMajorTensorOpCrosswise<4, 256>`   |
|  **B**    | `int4b_t`    | `ColumnMajor`   | `ColumnMajorTensorOpCrosswise<4, 256>`|
|  **C**    | `int32_t`    | `RowMajor`      | `RowMajor`                           |



## Warp-level Matrix Multiply with CUDA WMMA API

The following table summarizes supported warp level shapes for each WmmaTensorOp instruction.

|**Opcode Class**     | **Instruction Shape** | **Warp Shapes**                            |
|---------------------|-----------------------|--------------------------------------------|
| **WmmaTensorOp**    | 16-by-16-by-16        | 32x32x16, 32x64x16, 64x32x16               |
| **WmmaTensorOp**    | 8-by-32-by-16         | 32x32x16, 32x64x16, 64x32x16               |
| **WmmaTensorOp**    | 32-by-8-by-16         | 32x32x16, 32x64x16, 64x32x16               |
| **WmmaTensorOp**    | 8-by-8-by-32          | 32x32x32, 32x64x32, 64x32x32, 64x64x32     |
| **WmmaTensorOp**    | 8-by-8-by-128         | 32x32x128, 32x64x128, 64x32x128, 64x64x128 |


CUDA exposes warp-level matrix operations in the CUDA C++ WMMA API. The CUDA C++ WMMA API exposes Tensor Cores via a set of functions and types in the `nvcuda::wmma` namespace. The functions and types in `nvcuda::wmma` provide target-independent APIs and implement architecture-specific tensor operation using TensorOp instruction underneath. CUTLASS exposes WMMA API through WmmaTensorOp. The WmmaTensorOp supports canonical shared memory layouts. The following table summarizes the destination shared memory layout that can be targeted by matrix operands. The WMMA API expects that matrices in shared memory loaded by `nvcuda::wmma::load_matrix_sync()` satisfy 128 bit alignment.


**WmmaTensorOp (all matrix sizes and data types).**

|**Operand** |       **GMEM Layout**      |       **SMEM Layout**        |
|------------|----------------------------|------------------------------|
|  **A**     | `RowMajor`, `ColumnMajor`  | `RowMajor`, `ColumnMajor`    |
|  **B**     | `RowMajor`, `ColumnMajor`  | `RowMajor`, `ColumnMajor`    |
|  **C**     | `RowMajor`, `ColumnMajor`  | `RowMajor`, `ColumnMajor`    |

### Copyright

Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: BSD-3-Clause

```
  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
  DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
  OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
