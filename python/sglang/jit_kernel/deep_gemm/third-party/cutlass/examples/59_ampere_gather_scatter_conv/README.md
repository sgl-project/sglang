# Example 59: Ampere gather/scatter convolution

CuTe and CUTLASS 3.x based Ampere convolution forward propagation kernel capable of operating on both affine and gather/scatter tensors.

Example executions:
```sh
./59_ampere_gather_scatter_conv
./59_ampere_gather_scatter_conv --n=108
./59_ampere_gather_scatter_conv --n=4096 --i=1
./59_ampere_gather_scatter_conv --n=1080 --i=1000
./59_ampere_gather_scatter_conv --n=131072 --i=1000 --no-check
```

This example demonstrates a few super cool features of CUTLASS and CuTe. It shows off
1. A dense conv 3D fprop kernel written as a single file ...
2. ... that leverages off-the-shelf CUTLASS collectives to show how custom kernels can use collectives ...
3. ... and uses the exact same templated kernel to also stamp out a gather/scatter 3D fprop conv ...
4. ... while getting near peak performance of the Ampere class tensor core on Ampere and Ada GPUs ...
5. ... by using static cute shapes and strides in case problem shapes are known at compile time.

## A dense conv 3D fprop kernel written in CUTLASS 3.x and CuTe

The most common strategy for implementing high performance convolution kernels on the GPU is to transform
the activation tensor in such a way that we can perform the computation as a GEMM. This is called the 
image to column (im2col) transformation. [CUTLASS 2.x implementation of im2col based convolutions is
documented separately](../../media/docs/implicit_gemm_convolution.md), and here we consider a fresh approach for CuTe.

A 3D convolution has the following input tensors:
- Activation tensor (Act): `((N,(D,H,W)), (C,(1,1,1)))`
- Filter tensor     (Flt): `( K,          (C,(T,R,S)))`
- Output tensor     (Out): `((N,(Z,P,Q)),  K         )`

Where
- N   := number of images
- DHW := spatial dimensions of the activation tensor
- C   := channel dimension of the activation tensor
- K   := channel dimension of the filter and output tensor
- TRS := spoke dimensions of the filter tensor
- ZPQ := spatial dimensions of the output tensor

As is evident in the tensor shapes, these cannot be issued to a GEMM just yet, since there is no
logical M, N, and K modes we can group the tensor modes into.

Notice that every spoke of the filter tensor (TRS) will be applied to some (offset) view of the
activation tensor, thus expanding the logical size of the activation tensor. 
Additionally, a similar logical transform of the spatial dimensions can be encoded as a function of the
padding, dilations, traversal strides, and filter spokes. This gets us to our im2col transform:

im2col transform affects the component shapes/strides of the activation tensor in the following way:
- ZPQ Shape   : changes DHW domain with formula `(1 + (DHW + pad - (((TRS-1) * dilation) + 1)) / traversal_stride)`
- TRS Shape   : TRS domain instead of `(1,1,1)`
- ZPQ Strides : Original DHW strides get `elem_scale()`-ed by traversal strides DHW
- TRS Strides : Original DHW strides get `elem_scale()`-ed by dilation DHW

With this transform applied, we end up with a set of input and output tensors that
are logically consistent in their MNK dimensions, thus allowing us to dispatch to a GEMM.
im2col activation layout: ((N,(Z,P,Q)), (C,(T,R,S))) // logical (M,K)
filter layout           : ( K,          (C,(T,R,S))) // logical (N,K)
output layout           : ((N,(Z,P,Q)),  K         ) // logical (M,N)

CuTe's layout representation and algebra make these folded tensors easy to represent and manipulate.
This is most evident in the reference check code used in this example:

```cpp
for (size_t logical_m = 0; logical_m < size<0>(mOutputRef); ++logical_m) {
  for (size_t logical_n = 0; logical_n < size<1>(mOutputRef); ++logical_n) {
    auto accumulator = float(0);
    for (size_t logical_k = 0; logical_k < size<1>(mStencil); ++logical_k) {
      accumulator += mStencil(logical_m, logical_k) * mActivation(logical_n, logical_k);
    }
    mOutputRef(logical_m, logical_n) = accumulator;
  }
}
```

Which succinctly demonstrates how im2col transform allows us to implement convolutions
as GEMMs with special layout transformations on the input tensor.

Note: in the example kernel's implementation we treat activations as the B tensor
and filter as the A tensor, thus making their logical dimensions NK and MK respectively.

## Leveraging CUTLASS collectives off the shelf in a custom kernel

Now that we have transformed our problem in such a way that allows us to dispatch to a GEMM,
we can reuse much of the machinery CUTLASS offers to implement this forward pass convolution
operator. CUTLASS decomposes these "moving parts" of GPU linear algebra into reusable,
modular software components abstracted by C++ template classes. This example
demonstrates how some of the lower layers of the hierarchy can be re-used for custom kernels
by writing a custom kernel for convolution that re-uses the Ampere/Ada GEMM collectives
from CUTLASS 3.

A kernel author is free to compose their custom components with any of the existing templates
in the CUTLASS hierarchy to leverage existing high performance implementations from the CUTLASS
team. In this example, we write a custom kernel layer and compose with an existing collective.
However, any of the CUTLASS kernels can be composed with bespoke collectives if the desired
customization is a mainloop or epilogue fusion without changes to the grid planning,
tile scheduling, load balancing, or thread marshalling.

## Implementing gather/scatter and dense convolution with the same kernel

Functionality and correctness of the implemented kernel, as a virtue of using
CuTe and off the shelf CUTLASS collectives, only relies on the logical consistency of
the layouts of input and output tensors. This means that we can freely change how
the logical coordinates of the tensors map into the index space, and even how these dereferences
happen. [CUTLASS example 52](../52_hopper_gather_scatter_fusion/) demonstrates this by implementing a custom stride that
supports indexed indirection for tensor data accesses. This allows for example 52
to implement a GEMM where inputs are gathered and output is scattered based on an index buffer.

We re-use the same custom stride utilities in this example to implement a convolution kernel
that gathers along the NDHW dimensions of the activation tensor and scatters the output along the
NZPQ dimensions of the output tensor, treating the channel dimensions as the dense vectors.

Our dense affine im2col transformed activation tensor:

```cpp
// im2col transformed activation layout: ((nzpq), (ctrs)) => idx
auto xformed_act_layout = make_layout(
  make_shape (make_shape (      N,     Z,   P, Q), make_shape (  C,      T,   R, S)),
  make_stride(make_stride(D*H*W*C, H*W*C, W*C, C), make_stride(_1{}, H*W*C, W*C, C)));
```

now becomes a composed layout that uses `IndexedGather`:

```cpp
// Inner layout of the composition:
// ((nzpq), (csrt)) => (idx_buffer_idx, dense_offset)
auto EG = E<0>{};  // Gather basis     (1,0) (idx_buffer_idx) 
auto EC = E<1>{};  // Contiguous basis (0,1) (dense_offset)    
auto xformed_act_logical_inner = make_layout(
  make_shape (make_shape (       N,      Z,    P,  Q), make_shape ( C,      T,    R,  S)),
  make_stride(make_stride(D*H*W*EG, H*W*EG, W*EG, EG), make_stride(EC, H*W*EG, W*EG, EG)));

// Outer layout of the composition:
// (idx_buffer_idx, dense_offset) => idx
// IndexedGather obtains idx by applying (gmem_base_ptr + gather_idx_buf[idx_buffer_idx] + dense_offset)
auto xformed_act_gather_outer = make_layout(
  make_shape(_1{},_1{}),
  make_stride(CustomStride{IndexedGather{gather_idx_buf}, C}, _1{}));

// Compose the inner and outer layouts
// ((nzpq), (ctrs)) => idx
auto xformed_act_composed_layout = composition(
  xformed_act_gather_outer,
  make_arithmetic_tuple(_0{}, _0{}),
  xformed_act_logical_inner);
```

Here, we create a composed layout whose inner layout has the same logical MK shape as earlier,
but with an outer layout that uses the custom strides with an index buffer to access memory with
indirections. A custom stride requires two inputs to compute the index that a certain coordinate maps to:
the index buffer offset and the dense offset into the vector. This entails that our inner layout
(the one with the logical MK shape) has a rank-2 codomain `(idx_buffer_idx, dense_offset)`.
We can set up such a layout with scaled basis strides, which allow us to map a domain onto a
codomain with multiple orthogonal bases. The two codomain basis are the
index buffer offsets (rank 0 basis), and the dense vector offsets (rank 1 basis).
A similar composed layout is set up for the output scatter tensor.

This tensor still has a logical MK shape and is backed by a CuTe layout, which means we can still
tile, partition, and otherwise manipulate it with CuTe's layout algebra in the same way we would any
other tensor. Substituting the activation tensor's affine layout for this gather layout requires
no changes to the implementation of the kernel whatsoever. Everything composes. This example
stamps out a dense 3D convolution as well as gather/scatter 3D convolution using the same kernel template,
with the only difference between them being the layouts of the input and output tensors.

Convolutions are just a special case of tensor contractions, and as [example 51](../51_hopper_gett)
demonstrates, the exact same collective used in this example can also be used to implement arbitrary GETTs.
Of course, this also means that the same kernel can implement gather/scatter GETTs as well!

This demonstrates the composition power of not just CuTe, but also CUTLASS 3's two level
micro kernel abstraction. A single highly tuned temporal micro-kernel (collective) can be implemented once
and applied to compute dense GETTs, gather/scatter GETTs, dense convolutions, and gather/scatter convolutions.

## Peak performance on Ampere and Ada GPUs by leveraging domain specific knowledge

Often, when implementing custom kernels, a user has more knowledge of the problem domain that can be
exploited to deliver higher performance than otherwise could be through general kernels. In this example
we presume that the shape of each of the images (DHWC dimensions) as well as the filter (TRS) are available
a-priori and that the tile shape evenly divides the problem. Number of images (N) is still left as a runtime
parameter.

Knowing the extents of our tensors at compile time allows us to encode them as static cute shapes rather than
a dynamic problem shape, resulting in the elimination of most of the index computation instructions such as
expensive div/mods. Knowing that the problem shape is divisible by the tile shape allows us to use the 
Ampere collective that does not perform predication on global memory loads, further reducing overheads
and allowing us to achieve near peak performance on RTX Ampere and Ada GPUs.

Running this example on an RTX 3080Ti prints the following performance numbers (some output culled for brevity):

```
$> ./examples/59_ampere_gather_scatter_conv/59_ampere_gather_scatter_conv --n=131072 --i=128 --no-check
Ampere convolution forward propagation kernel supporting both affine and gather/scatter tensors.

Allocating tensors ... done.
Initializing data ... done.
Initializing gather/scatter index buffers ... done.

Running dense fprop kernel
Conv TFLOP count = 0.927713
Conv dense perf: 31.027376ms | TFLOP/s = 29.899819

Running gather/scatter fprop kernel
Conv TFLOP count = 0.927713
Conv gather/scatter perf: 28.973721ms | TFLOP/s = 32.019117
```

With this in mind, this example kernel has the following limitations:
- This example kernel only supports dynamic image count, all other conv problem shape must be defined as `cute::Constant<>`s
- Problem shapes (including dynamic image count `N`) must be evenly divisible by the tile shape
- It does not perform fp32->tf32 numeric conversion, gmem inputs must be rounded to tf32 already

## Copyright

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
