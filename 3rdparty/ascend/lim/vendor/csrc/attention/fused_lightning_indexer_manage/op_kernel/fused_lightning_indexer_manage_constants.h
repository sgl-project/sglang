/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */

#ifndef FUSED_LIGHTNING_INDEXER_MANAGE_CONSTANTS_H
#define FUSED_LIGHTNING_INDEXER_MANAGE_CONSTANTS_H

#include <cstdint>

namespace LIMConfig {

constexpr uint32_t TOPK = 2048U;
constexpr uint32_t MISS_CAPACITY = 32768U;
constexpr uint32_t CACHE_BLOCK_SIZE = 128U;
constexpr uint32_t HEAD_DIM = 128U;
constexpr uint32_t KEY_HEADS = 1U;
constexpr uint32_t QUERY_HEADS_SMALL = 32U;
constexpr uint32_t QUERY_HEADS_LARGE = 64U;
constexpr uint32_t MAX_ROUTES = 14U;
constexpr uint32_t MATURE_UNION_MAX_ROUTES = 4U;
constexpr uint32_t MAX_BLOCKS_PER_REQUEST = 1U << 14U;
constexpr uint32_t MAX_SOURCE_CAPACITY = 1U << 21U;
constexpr uint32_t MAX_CACHE_TOKENS = 32640U;

constexpr int32_t REQUEST_STATE_NON_OFFLOAD = -3;
constexpr int32_t REQUEST_STATE_FIRST_DECODE = -2;
constexpr int32_t REQUEST_STATE_STEADY = -1;
constexpr int32_t PADDING_SOURCE_ID = -1;

constexpr uint32_t AIVS_PER_AIC = 2U;
constexpr uint32_t LD_PARAM_COUNT = 16U;
constexpr uint32_t ROUTE_COUNT_BUFFER_ELEMENTS = 16U;
constexpr uint32_t THRESHOLD_WORKSPACE_STRIDE = 8U;
constexpr uint32_t ROUTE_COUNT_WORKSPACE_STRIDE = 8U;

constexpr uint32_t TOPK_SOURCE_OUTPUT_INDEX = 0U;
constexpr uint32_t TOPK_SLOT_OUTPUT_INDEX = 1U;
constexpr uint32_t TOPK_MISS_COUNT_OUTPUT_INDEX = 2U;
constexpr uint32_t MISS_SOURCE_OUTPUT_INDEX = 3U;
constexpr uint32_t MISS_SLOT_OUTPUT_INDEX = 4U;
constexpr uint32_t MISS_COUNT_OUTPUT_INDEX = 5U;
constexpr uint32_t CACHE_SLOTS_OUTPUT_INDEX = 6U;
constexpr uint32_t OUTPUT_COUNT = 7U;

} // namespace LIMConfig

#endif
