# Predication: What to do when tiling isn't perfect

The [GEMM tutorial](./0x_gemm_tutorial.md) shows how
we compute a matrix-matrix multiply
by iterating over tiles of the input matrices and output matrix.
The examples all assume that the tiles fit evenly into the matrices,
with no remainder.
What do we do if this is not the case?
For example, we might want to tile a 41 x 55 matrix into 4 x 8 tiles,
but 41 / 4 is 10 remainder 1, and 55 / 8 is 6 remainder 7.
What do we do with those "leftover" parts of the matrix?

To start, we note that `logical_divide`
(CuTe's way of tiling layouts) "rounds up."
For example, if `N` is the layout `1000:1` and `B` is the layout `128:1`,
then `logical_divide(N, B)` is the layout `(128, 8):(1, 128)`.
This effectively rounds up the original shape `N = 1000`
into an `128 x 8` matrix (as if `N = 1024`).
What about those last 24 elements,
that aren't part of the original data? How is the last tile handled and how do we avoid indexing out-of-bounds?

Like other introductions to CUDA programming, the idiomatic CuTe way to address these issues is through "predication."
Rather than attempting to reason about the "remainder tiles" by trying to represent "7 tiles of size-128 and 1 tile of size-104,"
CuTe instead rounds up to "8 tiles of size-128" and constructs predicates so that the kernel
only tries to access data in each tile that are valid within the matrix.
This corresponds well with how our GPUs optimize:
branches without warp divergence are relatively fast.
It also matches the usual CUDA idiom
when dividing N work items in 1-D fashion over B thread blocks:
first test if "my thread" is out of bounds before doing work.

Consider a generic tiling wherein a size-1000 vector is tiled into size-128 chunks. Then a predication tensor can be constructed as follows:

```c++
Tensor gmem = ...     // e.g. size 1000
Tensor smem = ...     // e.g. size 128

// Tile the gmem for smem
Tensor gmem_tiled = logical_divide(gmem, size(smem));      // e.g. (128,8)

// Create an identity layout for gmem and tile it similarly
Layout id_layout = make_layout(shape(gmem));               // e.g. 1000:1, explicitly constructed as identity function
Layout id_tiled  = logical_divide(id_layout, size(smem));  // e.g. (128,8):(1,128), but many elements aren't "valid"

// Create a predicate tensor
Tensor pred = make_tensor<bool>(shape(id_tiled));          // e.g. (128,8)
for (int i = 0; i < size(pred); ++i) {
  pred(i) = id_tiled(i) < size(id_layout);  // Predicate: Is the offset within the original shape?
}

// ... intervening code ...

// Note that gmem_tiled, id_tiled, and pred tensors are all congruent
// For tile tile_i, determine if element value_j is in-bounds and copy to smem
if (pred(value_j,tile_i)) { smem(value_j) = gmem_tiled(value_j,tile_i); }
```

The general procedure is that we

1. create an "identity" layout (`Layout id_layout = make_layout(shape(gmem))`,
   in the above example) with the same shape as our original data;

2. repeat the same tiling/partitioning/slicing (possibly rounding up)
   on that identity layout (`Layout id_tiled  = logical_divide(id_layout, size(smem));`);

3. create a "predicate tensor" by comparing the coordinates
   of that reference layout with the bounds of the original layout;
   and then

4. use the predicate tensor to mask off accesses to out-of-bounds elements.

As a relatively simple example, consider predicating the epilogue of a GEMM.
Suppose that we've partitioned `mC` into cta tiles and across threads of an mma as follows.

```cpp
// CTA partitioning
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

// Thread partitioning
auto thr_mma = mma.get_slice(threadIdx.x);
Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)
Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

// ... Compute gemms and accumulate into tCrC ...

// axpby epilogue
for (int i = 0; i < size(tCgC); ++i) {
  tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
}
```

Then, following the predication procedure is straightforward,

```cpp
// A coordinate tensor the same shape as mC: (m,n) -> (m,n)
Tensor cC     = make_identity_tensor(shape(mC));

// Repeat partitioning steps applied to mC to our coordinate tensor cC
// CTA partitioning
Tensor cta_cC = local_tile(cC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N) -> (m,n)
// Thread partitioning
Tensor tCcC   = thr_mma.partition_C(cta_cC);                             // (MMA,MMA_M,MMA_N) -> (m,n)

// Predicated axpby epilogue
for (int i = 0; i < size(tCgC); ++i) {
  if (elem_less(tCcC(i), shape(mC))) {  // if coord is in-bounds
    tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
  }
}
```

Above, the cta is responsible for tiling/partitioning `mC` and the mma is responsible for tiling/partitioning `gC`,
so both steps are also applied to the identity tensor.
The coordinate tensor `tCcC` is congruent with the register fragment `tCrC` and the partitioned global memory tensor `tCgC`, which are this threads' subtensors of the tile of data. However, the `tCcC` tensor retains it's original codomain when evaluated: a global coordinate into the original tensor `mC`. This global coordinate is compared to the shape of `mC` to determine validity of the operation.

Advantages of this "reference identity tensor" or "coordinate tensor" approach include:

1. There is no dependence on the layout/strides of the tensor
   being predicated, just the logical bounds imposed.

2. The partitioning stage(s) can be anything. A CTA tiling, a thread partitioning, a TiledMMA, and a TiledCopy can all be applied to any tensor, including a coordinate tensor.

3. It naturally extends to any-dimensional predication.

4. It's a natural generalization of a typical CUDA 1-D
   parallel vector access pattern,
   which computes an access index `idx` and predicates access to the vector's `idx`-th element, determining if `idx` is in-bounds.
```cpp
int idx = blockDim.x * blockIdx.x + threadIdx.x;
if (idx < N)  // idx is a "coord" into gmem and N is the "bound"
  gmem_ptr[idx] = ...;
```

In a SIMT programming model, the tensor extents should not be modified so that loops don't overrun.
Instead, predication is a general method to query the original coordinate and determine if that coordinate overruns.
This avoids variable/dynamic loop bounds in favor of instruction-level predication, preservation of thread coherence, and maintaining load balance.
It's also general enough to extend to all ranks, all layouts of threads and data, and all tiling/partitioning patterns.
Assumptions can be built into the coordinate tensors or the predicate tensors to account for special cases.

As another slightly more complex example, consider the m- and n-predication of A and B loads in a GEMM. Suppose that we've partitioned A and B tiles across ctas and threads as follows.

```c++
// CTA partitioning
auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)

Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

// Thread partitioning
Tensor tAgA = local_partition(gA, tA, thread_idx);                   // (THR_M,THR_K,k)
Tensor tAsA = local_partition(sA, tA, thread_idx);                   // (THR_M,THR_K)

Tensor tBgB = local_partition(gB, tB, thread_idx);                   // (THR_N,THR_K,k)
Tensor tBsB = local_partition(sB, tB, thread_idx);                   // (THR_N,THR_K)
```

`gA` and `gB` are tiles of `mA` resp. `mB` according to `cta_tiler` and the `cta_coord`.
`tAgA` and `tBgB` are partitions of `gA` resp. `gB` according the the thread-layouts `tA` and `tB`
and `thread_idx`.

The following code creates "identity tensors" that map coordinates `(m,k) -> (m,k)` and `(n,k) -> (n,k)`.

```c++
// Coordinate tensors
Tensor cA = make_identity_tensor(shape(mA));   // (m,k) -> (m,k)
Tensor cB = make_identity_tensor(shape(mB));   // (n,k) -> (n,k)
```

Then, the reference tensors are tiled and partitioned
in exactly the same way the `mA` and `mB` tensors were tiled and partitioned
into `tAgA` and `tBgB`.

```c++
// CTA partitioning
Tensor cta_cA = local_tile(cA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k) -> (m,k)
Tensor cta_cB = local_tile(cB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k) -> (n,k)

// Thread partitioning
Tensor tAcA = local_partition(cta_cA, tA, thread_idx);                   // (THR_M,THR_K,k) -> (m,k)
Tensor tBcB = local_partition(cta_cB, tB, thread_idx);                   // (THR_N,THR_K,k) -> (m,k)
```

The following code creates predicate tensors
corresponding to `tAgA` and `tBgB`.
They will be computed once in the prologue.
and will be used to mask off instructions in the inner loop.

```c++
Tensor tApA = make_tensor<bool>(make_shape (size<0>(tAcA), size<1>(tAcA)),
                                make_stride(     Int<1>{},      Int<0>{}));
Tensor tBpB = make_tensor<bool>(make_shape (size<0>(tBcB), size<1>(tBcB)),
                                make_stride(     Int<1>{},      Int<0>{}));
```

Here, we make a few assumptions: we're only interested in predicates for one tile of data at a time and we're only interested in predicates for the m- and n-modes and will handle the k-mode predicates differently.
The m- and n- predicates will be considered constant across every tile and will be reused in every iteration of the mainloop.
Thus, we only store the predicates for the m- and n-modes and broadcast them across the k-mode.
When populating the tensors, we carry the same assumption through:

```c++
// Populate the m- and n-predicates
CUTE_UNROLL
for (int m = 0; m < size<0>(tApA); ++m) {
  tApA(m,0) = elem_less(get<0>(tAcA(m,0,0)), shape<0>(mA));  // Compare the m-coordinate
}
CUTE_UNROLL
for (int n = 0; n < size<0>(tBpB); ++n) {
  tBpB(n,0) = elem_less(get<0>(tBcB(n,0,0)), shape<0>(mB));  // Compare the n-coordinate
}
```

and only compare the m- and n-coordinates of the 0th k-tile and 0th k-block. The stride-0 broadcasting mode still allows us to treat this data as a predicate tensor for each and every element of the tile to be loaded.

Finally, we can then use the predicate tensors in `copy_if` to copy only the elements for which the corresponding predicate tensor elements are `true`.

```c++
// Copy a k_tile from global memory to shared memory
copy_if(tApA, tAgA(_,_,k_tile), tAsA);
copy_if(tBpB, tBgB(_,_,k_tile), tBsB);
```
