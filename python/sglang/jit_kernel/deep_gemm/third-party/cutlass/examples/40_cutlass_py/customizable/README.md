# Customizable Python Interface Examples
This directory contains examples of using the CUTLASS Python interface with a variety of configurations for kernels.

For all the tests, add `--print_cuda` to print the underlying CUDA kernel. Use `-h` or `--help` to display the help message.

## GEMM Examples
The GEMM examples use numpy to create input tensors and verify the results.
### GEMM F64 Example
Example 1: SM80_Device_Gemm_f64t_f64n_f64n_tensor_op_f64_32x32x16_16x16x16
```python
python gemm.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 32 32 16 -s 4 -w 2 2 1 -cc 80 -la ColumnMajor -aa 1 -lb RowMajor -ab 1 -lc RowMajor -ac 1 -te float64 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 1
```
Example 2: SM80_Device_Gemm_f64n_f64t_f64n_tensor_op_f64_64x64x16_32x32x16, split_k(2)_serial
```python
python gemm.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 64 64 16 -s 4 -w 2 2 1 -cc 80 -la RowMajor -aa 1 -lb ColumnMajor -ab 1 -lc RowMajor -ac 1 -te float64 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 2
```

### GEMM F32 Example
Example 1: SM80_Device_Gemm_f32n_f32t_f32n_tensor_op_bf16_f32_128x128x32_64x64x32
```python
python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add_fast_bf16 -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la RowMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 1
```
Example 2: SM80_Device_Gemm_f32t_f32t_f32n_tensor_op_f32_128x128x32_64x64x32, split_k(2)_parallel
```python
python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm GemmSplitKParallel -k 2
```
Example 3: SM80_Device_Gemm_f32t_f32t_f32n_tensor_op_fast_accurate_f32_64x64x32_32x32x32, split_k(4)_serial
```python
python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add_fast_f32 -op TensorOp -b 64 64 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 4
```

### GEMM F16 Example
Example 1: SM80_Device_Gemm_f32t_f32n_f32t_tensor_op_bf16_f32_128x128x32_64x64x32
```python
python gemm.py -i 16 8 16 -ta float16 -tb float16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb RowMajor -ab 8 -lc ColumnMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle4 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 1
```
Example 2: SM80_Device_Gemm_f16t_f16t_f16n_tensor_op_f32_128x128x64_64x64x64, split_k(2)_serial
```python
python gemm.py -i 16 8 16 -ta float16 -tb float16 -tc float16 -tacc float32 -m multiply_add -op TensorOp -b 128 128 64 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc RowMajor -ac 8 -te float32 -ep LinearCombination -sw IdentitySwizzle2 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 2
```
Example 3: SM80_Device_Gemm_f16t_f16t_f32n_tensor_op_f32_256x128x64_64x64x64, split_k(3)_serial
```python
python gemm.py -i 16 8 16 -ta float16 -tb float16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 256 128 64 -s 3 -w 4 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm GemmSplitKParallel -k 3
```

### GEMM BF16 Example
Example 1: Device_Gemm_bf16t_bf16t_f32n_tensor_op_f32_64x128x64_32x64x64, split_k(5)_parallel
```python
python gemm.py -i 16 8 16 -ta bfloat16 -tb bfloat16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 64 128 64 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle2 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm GemmSplitKParallel -k 5
```

### GEMM Int8 Example
Example 1: SM80_Device_Gemm_s8n_s8t_s8n_tensor_op_s32_256x128x128_64x64x128
```python
python gemm.py -i 16 8 32 -ta int8 -tb int8 -tc int8 -tacc int32 -m multiply_add -op TensorOp -b 128 128 128 -s 3 -w 2 2 1 -cc 80 -la RowMajor -aa 16 -lb ColumnMajor -ab 16 -lc RowMajor -ac 16 -te float32 -ep FastLinearCombinationClamp -sw IdentitySwizzle2 -p 512 512 512 -alpha 1.0 -beta 0.0 -gm Gemm -k 1
```

### Batched & Array GEMM
Example 1: Batched GEMM
```python
python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add_fast_bf16 -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la RowMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw BatchedIdentitySwizzle -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Batched -k 1 -batch 3
```
Example 2: Array GEMM
```python
python gemm.py -i 16 8 16 -ta float16 -tb float16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb RowMajor -ab 8 -lc ColumnMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle4 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Array -k 1 -batch 2
```
***
## GEMM Grouped Examples
The GEMM Grouped examples use numpy to create input tensors and verify the results.

Example 1: SM80_Device_GemmGrouped_f16t_f16t_f32t_tensor_op_f32_128x128x32_64x64x32, device schedule
```python
python gemm_grouped.py -i 16 8 16 -ta float16 -tb float16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc ColumnMajor -ac 4 -te float32 -ep LinearCombination -p ./grouped_gemm_problem_size.csv -alpha 1.0 -beta 0.0 -pm Device
```
Example 2: SM80_Device_GemmGrouped_f64n_f64n_f64t_tensor_op_f64_64x64x16_32x32x16, host schedule
```python
python gemm_grouped.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 64 64 16 -s 4 -w 2 2 1 -cc 80 -la RowMajor -aa 1 -lb RowMajor -ab 1 -lc ColumnMajor -ac 1 -te float64 -ep LinearCombination -p ./grouped_gemm_problem_size.csv -alpha 1.0 -beta 1.0 -pm Host
```
Example 3: SM80_Device_GemmGrouped_f32n_f32n_f32n_simt_f32_128x64x8_64x32x1, device schedule
```python
python gemm_grouped.py -i 1 1 1 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op Simt -b 128 64 8 -s 4 -w 2 2 1 -cc 80 -la RowMajor -aa 1 -lb RowMajor -ab 1 -lc RowMajor -ac 1 -te float32 -ep LinearCombination -p ./grouped_gemm_problem_size.csv -alpha 2.0 -beta 1.0 -pm Device
```
Example 4: SM80_Device_GemmGrouped_f16t_f16t_f32t_tensor_op_f32_128x128x32_64x64x32, device schedule
```python
python gemm_grouped.py -i 16 8 16 -ta float16 -tb float16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc ColumnMajor -ac 4 -te float32 -ep LinearCombination -p ./grouped_gemm_problem_size.csv -alpha 2.0 -beta 1.0 -pm Device
```
***
## Conv2d Example
The Conv2d examples use pytorch to create input tensors and verify the results. Pytorch can be installed following the [official website](https://pytorch.org/get-started/locally/).
### Conv2d F32 Fprop
Example 1: SM80_Device_Conv2d_Fprop_Analytic_ImplicitGemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32
```python
python conv2d.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 16 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 4 -lb TensorNHWC -ab 4 -lc TensorNHWC -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -co fprop -st Strided -ia optimized -sm Serial -k 1 -nhwc 1 13 17 8 -krsc 24 3 3 8 -pad 0 0 0 0 -stride 2 2 -dilation 1 1 -alpha 1.0 -beta 0.0
```
Example 2: SM80_Device_Conv2d_Fprop_Optimized_ImplicitGemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32_align2
```python
python conv2d.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 16 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 2 -lb TensorNHWC -ab 2 -lc TensorNHWC -ac 2 -te float32 -ep LinearCombination -sw IdentitySwizzle2 -co fprop -st Strided -ia optimized -sm Serial -k 2 -nhwc 1 4 4 12 -krsc 8 3 3 12 -pad 0 0 0 0 -stride 3 3 -dilation 1 1 -alpha 1.0 -beta 1.0
```
Example 3: SM80_Device_Conv2d_Fprop_Analytic_ImplicitGemm_f32nhwc_f32nhwc_f32nhwc_simt_f32
```python
python conv2d.py -i 1 1 1 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op Simt -b 128 128 8 -s 4 -w 4 2 1 -cc 80 -la TensorNHWC -aa 4 -lb TensorNHWC -ab 4 -lc TensorNHWC -ac 1 -te float32 -ep LinearCombination -sw IdentitySwizzle4 -co fprop -st Strided -ia analytic -sm Parallel -k 3 -nhwc 1 71 80 32 -krsc 64 5 5 32 -pad 2 2 2 2 -stride 2 2 -dilation 1 1 -alpha 1.0 -beta 1.0
```
### Conv2d F32 Wgrad
Example 1: Device_Conv2d_Wgrad_Optimized_ImplicitGemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32_align1
```python
python conv2d.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 1 -lb TensorNHWC -ab 1 -lc TensorNHWC -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -co wgrad -st Strided -ia optimized -sm Serial -k 1 -nhwc 1 8 8 1 -krsc 1 3 3 1 -pad 1 1 1 1 -stride 1 1 -dilation 1 1 -alpha 1.0 -beta 0.0
```
Example 2: Device_Conv2d_Wgrad_Analytic_ImplicitGemm_f32nhwc_f32nhwc_f32nhwc_simt_f32
```python
python conv2d.py -i 1 1 1 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op Simt -b 128 128 8 -s 4 -w 2 4 1 -cc 80 -la TensorNHWC -aa 4 -lb TensorNHWC -ab 4 -lc TensorNHWC -ac 1 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -co wgrad -st Strided -ia optimized -sm Serial -k 2 -nhwc 1 27 27 256 -krsc 512 3 3 256 -pad 1 1 1 1 -stride 2 1 -dilation 1 1 -alpha 1.0 -beta 0.0
```
### Conv2d F32 Dgrad
Example 1: Device_Conv2d_Dgrad_Analytic_ImplicitGemm_tf32nhwc_tf32nhwc_f32nhwc_tensor_op_f32
```python
python conv2d.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 16 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 4 -lb TensorNHWC -ab 4 -lc TensorNHWC -ac 4 -te float32 -ep LinearCombination -sw StridedDgradIdentitySwizzle1 -co dgrad -st Strided -ia optimized -sm Serial -k 2 -nhwc 1 27 27 256 -krsc 512 3 3 256 -pad 1 1 1 1 -stride 2 1 -dilation 1 1 -alpha 1.0 -beta 0.0
```

### Conv2d F16 Fprop
Example 1: SM80_Device_Conv2d_Fprop_Analytic_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32
```python
python conv2d.py -i 16 8 16 -ta float16 -tb float16 -tc float16 -tacc float32 -m multiply_add -op TensorOp -b 128 128 64 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 8 -lb TensorNHWC -ab 8 -lc TensorNHWC -ac 8 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -co fprop -st Strided -ia optimized -sm Serial -k 1 -nhwc 1 27 27 256 -krsc 512 3 3 256 -pad 1 1 1 1 -stride 2 1 -dilation 1 1 -alpha 1.0 -beta 0.0
```
Example 2: SM80_Device_Conv2d_Fprop_Few_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_2
```python
python conv2d.py -i 16 8 16 -ta float16 -tb float16 -tc float16 -tacc float32 -m multiply_add -op TensorOp -b 128 128 64 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 2 -lb TensorNHWC -ab 2 -lc TensorNHWC -ac 8 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -co fprop -st Strided -ia few_channels -sm Serial -k 1 -nhwc 1 16 16 2 -krsc 16 3 3 2 -pad 1 1 1 1 -stride 2 2 -dilation 1 1 -alpha 1.0 -beta 0.0
```
Example 3: SM80_Device_Conv2d_Fprop_Fixed_Channels_ImplicitGemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f32_channels_8
```python
python conv2d.py -i 16 8 16 -ta float16 -tb float16 -tc float16 -tacc float32 -m multiply_add -op TensorOp -b 128 128 64 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 8 -lb TensorNHWC -ab 8 -lc TensorNHWC -ac 8 -te float32 -ep LinearCombination -sw IdentitySwizzle2 -co fprop -st Strided -ia fixed_channels -sm Serial -k 1 -nhwc 1 8 8 8 -krsc 16 3 3 8 -pad 1 1 1 1 -stride 2 2 -dilation 1 1 -alpha 1.0 -beta 0.0
```
Example 4: SM80_Device_Conv2d_Strided_Dgrad_Optimized_ImplicitGemm_f16nhwc_f16nhwc_f32nhwc_tensor_op_f32_128x128_32x3_64x64x32_align4
```python
python conv2d.py -i 16 8 16 -ta float16 -tb float16 -tc float16 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 4 -lb TensorNHWC -ab 4 -lc TensorNHWC -ac 4 -te float32 -ep LinearCombination -sw StridedDgradIdentitySwizzle1 -co dgrad -st Strided -ia optimized -sm Serial -k 1 -nhwc 1 56 56 12 -krsc 8 1 1 12 -pad 0 0 0 0 -stride 2 2 -dilation 1 1 -alpha 1.0 -beta 0.0
```

## Epilogue
### Bias 
To replace C with a bias vector, add `-bias` flag.
### Activation function
Example 1: ReLU
```python
python gemm.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 32 32 16 -s 4 -w 2 2 1 -cc 80 -la ColumnMajor -aa 1 -lb RowMajor -ab 1 -lc RowMajor -ac 1 -te float64 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 1 -bias -activ relu
```
Example 2: leaky ReLU
```python
python gemm.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 64 64 16 -s 4 -w 2 2 1 -cc 80 -la RowMajor -aa 1 -lb ColumnMajor -ab 1 -lc RowMajor -ac 1 -te float64 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm Gemm -k 2 -bias -activ leaky_relu -activ_arg 0.2
```
Example 3: tanh (alpha=0 to avoid saturation)
```python
python gemm.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 32 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 4 -lb ColumnMajor -ab 4 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -p 512 256 128 -alpha 1.0 -beta 0.5 -gm GemmSplitKParallel -k 2 -bias -activ tanh
```
Example 4: sigmoid
```python
python gemm_grouped.py -i 8 8 4 -ta float64 -tb float64 -tc float64 -tacc float64 -m multiply_add -op TensorOp -b 64 64 16 -s 4 -w 2 2 1 -cc 80 -la RowMajor -aa 1 -lb RowMajor -ab 1 -lc ColumnMajor -ac 1 -te float64 -ep LinearCombination -p ./grouped_gemm_problem_size.csv -alpha 0.0 -beta 0.5 -pm Host -bias -activ sigmoid -bias -activ sigmoid
```
Example 5: SiLU
```python
python conv2d.py -i 16 8 8 -ta float32 -tb float32 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 128 128 16 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 2 -lb TensorNHWC -ab 2 -lc TensorNHWC -ac 2 -te float32 -ep LinearCombination -sw IdentitySwizzle2 -co fprop -st Strided -ia optimized -sm Serial -k 2 -nhwc 1 4 4 12 -krsc 8 3 3 12 -pad 0 0 0 0 -stride 3 3 -dilation 1 1 -alpha 0.0 -beta 0.5 -bias -activ silu
```
Example 6: HardSwish
```python
python conv2d.py -i 16 8 16 -ta float16 -tb float16 -tc float16 -tacc float32 -m multiply_add -op TensorOp -b 128 128 64 -s 3 -w 2 2 1 -cc 80 -la TensorNHWC -aa 2 -lb TensorNHWC -ab 2 -lc TensorNHWC -ac 8 -te float32 -ep LinearCombination -sw IdentitySwizzle1 -co fprop -st Strided -ia few_channels -sm Serial -k 1 -nhwc 1 16 16 2 -krsc 16 3 3 2 -pad 1 1 1 1 -stride 2 2 -dilation 1 1 -alpha 0.0 -beta 0.5 -bias -activ hardswish
```
Example 7: GELU
```python
python gemm.py -i 16 8 16 -ta bfloat16 -tb bfloat16 -tc float32 -tacc float32 -m multiply_add -op TensorOp -b 64 128 64 -s 3 -w 2 2 1 -cc 80 -la ColumnMajor -aa 8 -lb ColumnMajor -ab 8 -lc RowMajor -ac 4 -te float32 -ep LinearCombination -sw IdentitySwizzle2 -p 512 256 128 -alpha 0.0 -beta 0.5 -gm GemmSplitKParallel -k 5 -bias -activ gelu
```

# Copyright

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
