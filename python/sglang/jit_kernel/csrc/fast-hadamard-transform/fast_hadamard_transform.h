/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

// Copied from https://github.com/sgl-project/fast-hadamard-transform

#pragma once

////////////////////////////////////////////////////////////////////////////////////////////////////

struct HadamardParamsBase {
  using index_t = int64_t;

  int batch, dim, log_N;

  index_t x_batch_stride;
  index_t out_batch_stride;

  float scale;

  // Common data pointers.
  void* __restrict__ x_ptr;
  void* __restrict__ out_ptr;

  // Optional per-element sign vectors for fused WHT rotation.
  // When non-null, load multiplies by signs1 and store multiplies by signs2.
  // Shape: (dim,) float32. Set to nullptr to skip.
  const float* __restrict__ signs1_ptr;
  const float* __restrict__ signs2_ptr;
};
