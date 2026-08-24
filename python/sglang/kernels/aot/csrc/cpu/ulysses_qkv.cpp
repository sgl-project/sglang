#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <cstring>

/*
 * Pack Q/K/V into destination-major layout for Ulysses all-to-all.
 *
 * Input:
 *   q, k, v:
 *     [rows, global_heads, head_size]
 *
 * Output:
 *   [world_size, rows, local_heads, 3 * head_size]
 *
 * where:
 *   local_heads = global_heads / world_size
 *
 * Mapping:
 *
 *   global_head = destination * local_heads + local_head
 *
 *   output[destination, row, local_head, 0:D]
 *       = q[row, global_head, :]
 *
 *   output[destination, row, local_head, D:2D]
 *       = k[row, global_head, :]
 *
 *   output[destination, row, local_head, 2D:3D]
 *       = v[row, global_head, :]
 *
 */

void pack_qkv_destination_major_cpu(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, int64_t world_size, at::Tensor& output) {
  TORCH_CHECK(
      q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && q.sizes() == k.sizes() && q.sizes() == v.sizes(),
      "q, k, and v must have the same 3D shape");
  TORCH_CHECK(q.device().is_cpu() && k.device().is_cpu() && v.device().is_cpu(), "q, k, and v must be CPU tensors");
  TORCH_CHECK(
      q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(), "q, k, and v must have the same dtype");
  TORCH_CHECK(
      q.layout() == at::kStrided && k.layout() == at::kStrided && v.layout() == at::kStrided,
      "q, k, and v must be strided tensors");
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1, "q, k, and v must be contiguous in head_size");
  TORCH_CHECK(world_size >= 1, "world_size must be positive");

  const int64_t rows = q.size(0);
  const int64_t global_heads = q.size(1);
  const int64_t head_size = q.size(2);
  TORCH_CHECK(global_heads % world_size == 0, "world_size must divide global_heads");
  const int64_t local_heads = global_heads / world_size;
  TORCH_CHECK(output.device().is_cpu(), "output must be a CPU tensor");
  TORCH_CHECK(output.scalar_type() == q.scalar_type(), "output must have the same dtype as q/k/v");
  TORCH_CHECK(
      output.size(0) == world_size && output.size(1) == rows && output.size(2) == local_heads &&
          output.size(3) == 3 * head_size,
      "invalid output shape");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  if (q.numel() == 0) {
    return;
  }
  const int64_t q_stride_row = q.stride(0);
  const int64_t q_stride_head = q.stride(1);
  const int64_t k_stride_row = k.stride(0);
  const int64_t k_stride_head = k.stride(1);
  const int64_t v_stride_row = v.stride(0);
  const int64_t v_stride_head = v.stride(1);
  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "pack_qkv_destination_major_cpu", [&] {
    const scalar_t* q_ptr = q.data_ptr<scalar_t>();
    const scalar_t* k_ptr = k.data_ptr<scalar_t>();
    const scalar_t* v_ptr = v.data_ptr<scalar_t>();
    scalar_t* out_ptr = output.data_ptr<scalar_t>();
    const size_t head_bytes = static_cast<size_t>(head_size) * sizeof(scalar_t);
    const int64_t total_head_slots = rows * global_heads;
    at::parallel_for(0, total_head_slots, 0, [&](int64_t begin, int64_t end) {
      for (int64_t head_slot = begin; head_slot < end; ++head_slot) {
        const int64_t local_head = head_slot % local_heads;
        const int64_t row_slot = head_slot / local_heads;
        const int64_t row = row_slot % rows;
        const int64_t destination = row_slot / rows;
        const int64_t global_head = destination * local_heads + local_head;
        const scalar_t* q_src = q_ptr + row * q_stride_row + global_head * q_stride_head;
        const scalar_t* k_src = k_ptr + row * k_stride_row + global_head * k_stride_head;
        const scalar_t* v_src = v_ptr + row * v_stride_row + global_head * v_stride_head;
        scalar_t* out = out_ptr + head_slot * (3 * head_size);

        // [Q | K | V]
        std::memcpy(out, q_src, head_bytes);
        std::memcpy(out + head_size, k_src, head_bytes);
        std::memcpy(out + 2 * head_size, v_src, head_bytes);
      }
    });
  });
}
