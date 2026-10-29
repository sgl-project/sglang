#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

static constexpr uint32_t INPUT_STRIDE_ALIGNMENT_REQUIREMENT = 1024; // In number of bytes
static constexpr uint32_t OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT = 32; // In number of bytes

static constexpr uint32_t MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION = 1u << 23;
// [SGLang] Overridable so a module that only ever sees short rows can shrink the window-delta
// table in `incoming_topk_pairs`, which is what pins NUM_EXTRA_SLOTS and so the plan size.
#ifndef TOPK_SELECT_MAX_VOCAB_SIZE
#define TOPK_SELECT_MAX_VOCAB_SIZE (1u << 23)
#endif
static constexpr uint32_t MAX_VOCAB_SIZE = TOPK_SELECT_MAX_VOCAB_SIZE;
static_assert(MAX_VOCAB_SIZE <= MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION);

struct TopkSelectArgs {
    uint32_t batch_size;
    uint32_t vocab_size;
    uint32_t topk;

    void* input;
    void* output_value;
    void* output_index;
    int* begin_ptr;
    int* end_ptr;
    int* output_idx_offset;

    // All strides are in number of elements, not bytes
    uint64_t stride_input_batch;
    uint64_t stride_output_value_batch;
    uint64_t stride_output_index_batch;

    bool sorted_value;
    bool sorted_index;
    bool return_value;
    int idx_oob_fill_value;
    float value_oob_fill_value;
    bool abort_when_nan_found;

    uint64_t shared_memory_size_per_sm;
    cudaStream_t stream;

    // [SGLang] Page transform, read only when `Config::page_transform` is set:
    // index `i` is written as `(page_table[i >> page_bits] << page_bits) | (i & ((1 << page_bits) - 1))`
    const int* page_table;
    uint64_t stride_page_table_batch;
    uint32_t page_bits;
};
