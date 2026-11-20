#pragma once

#include <cstdint>

namespace sglang {

// Staged pages lie back to back on the device. Their host pages lie back to back
// too wherever `dst_indices` steps by one page, and then one copy can move them
// all. Calls `emit(first_page, num_pages)` once per maximal run of such pages, in
// page order. With `merge` false every page is a run of its own.
template <typename Emit>
inline void
for_each_host_page_run(const int64_t* dst_indices, int64_t num_pages, int64_t page_size, bool merge, Emit&& emit) {
  int64_t run_begin = 0;
  for (int64_t page = 1; page <= num_pages; ++page) {
    const bool adjacent =
        merge && page < num_pages && dst_indices[page * page_size] == dst_indices[(page - 1) * page_size] + page_size;
    if (!adjacent) {
      emit(run_begin, page - run_begin);
      run_begin = page;
    }
  }
}

}  // namespace sglang
