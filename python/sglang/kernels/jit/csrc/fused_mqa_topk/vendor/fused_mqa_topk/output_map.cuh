// SPDX-License-Identifier: MIT
// Output index transform. Every valid output index is a key index: absolute
// (the logits column), or with `relative` x = key - start, optionally shifted
// by out_offsets[row] (x = key - start + out_offsets[row]) or mapped through
// the row's page-table row (table[row_batch[row] * stride + x], a page_size =
// 1 table of KV slots). The selection writes absolute key indices and a short
// row (at most top-k visible keys) first + j for its j-th key (first = start,
// 0 or out_offsets[row]); finish_row then rewrites a row in place (L2-hot).
#pragma once
#include <cstdint>

struct PageTable {
  const int* table;      // nullptr: no page table
  int64_t stride;        // elements between consecutive rows of table
  const int* row_batch;  // [rows] table row of each query row
};

namespace fused_output {

// One warp rewrites output row r in place: every valid value v becomes
// x = v + shift, or with a page table the table row's entry x. 4 int4 per lane
// in flight (topk / 4 is a multiple of 4 * 32).
__device__ __forceinline__ void finish_row(const PageTable& p, int* output,
                                           unsigned r, unsigned topk, int shift,
                                           unsigned lane) {
  const int* table =
      p.table ? p.table + int64_t(p.row_batch[r]) * p.stride : nullptr;
  if (shift == 0 && !table) return;
  int4* q = reinterpret_cast<int4*>(output + (uint64_t)r * topk);
  const unsigned quads = topk / 4;
  for (unsigned i = lane; i < quads; i += 4 * 32) {
    int4 v[4];
#pragma unroll
    for (unsigned u = 0; u < 4; ++u) v[u] = q[i + 32 * u];
#pragma unroll
    for (unsigned u = 0; u < 4; ++u) {
      int* e = reinterpret_cast<int*>(&v[u]);
#pragma unroll
      for (unsigned c = 0; c < 4; ++c) {
        const int x = e[c] + shift;
        e[c] = e[c] < 0 ? -1 : table ? __ldg(table + x) : x;
      }
      q[i + 32 * u] = v[u];
    }
  }
}

}  // namespace fused_output
