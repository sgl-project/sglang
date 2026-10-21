/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Stage 4: official LightningIndexer scheduling, union eviction, and cache update.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_lightning_indexer_manage_template_tiling_key.h"
#include "fused_lightning_indexer_manage_constants.h"
#include "lightning_indexer_kernel.h"
#include "fused_lightning_indexer_manage_union.h"
#include "fused_lightning_indexer_manage_workspace.h"

using namespace LIKernel;
using namespace AscendC;

__aicore__ inline void PrepareNonOffloadRows(
    __gm__ uint8_t *requestStateAddr, __gm__ uint8_t *reqPoolEntriesAddr,
    __gm__ uint8_t *actualSeqQueryAddr, __gm__ uint8_t *actualSeqKeyAddr,
    __gm__ uint8_t *offloadSeqKeyAddr, __gm__ uint8_t *cacheTokensAddr,
    __gm__ uint8_t *cacheSlotsAddr,
    uint32_t batchSize, uint32_t totalQueries, uint32_t poolSize,
    uint32_t sourceCapacity, TPipe *pipe)
{
    if ASCEND_IS_AIV {
    if (GetBlockIdx() % LIMConfig::AIVS_PER_AIC != 0U) return;
    // Preserve 4096-element chunks for short rows so they retain enough AIV
    // owners.  Long rows use 8192 elements to halve each owner's serialized
    // vector/DMA rounds; the 32768-byte MTE3 transfer remains representable by
    // DataCopyPad's uint16_t byte count.
    constexpr uint32_t IDENTITY_SHORT_CHUNK = 4096U;
    constexpr uint32_t IDENTITY_LONG_CHUNK = 8192U;
    constexpr uint32_t IDENTITY_LONG_CUTOFF = 512U * 1024U;
    constexpr uint32_t IDENTITY_BUFFER_CAPACITY = IDENTITY_LONG_CHUNK;
    // Keep the transition cleanup independent from the -3 identity-fill
    // buffers.  Start with a conservative 512-entry DMA block; only a row
    // positively identified as an identity row uses this buffer.
    constexpr uint32_t TRANSITION_CLEANUP_CHUNK = 512U;
    GlobalTensor<int32_t> states;
    GlobalTensor<int32_t> entries;
    GlobalTensor<int32_t> slots;
    GlobalTensor<int32_t> queryEnds;
    GlobalTensor<int32_t> keyLengths;
    GlobalTensor<int32_t> offloadLengths;
    GlobalTensor<int32_t> cacheTokens;
    states.SetGlobalBuffer((__gm__ int32_t *)requestStateAddr);
    entries.SetGlobalBuffer((__gm__ int32_t *)reqPoolEntriesAddr);
    slots.SetGlobalBuffer((__gm__ int32_t *)cacheSlotsAddr);
    queryEnds.SetGlobalBuffer((__gm__ int32_t *)actualSeqQueryAddr);
    keyLengths.SetGlobalBuffer((__gm__ int32_t *)actualSeqKeyAddr);
    offloadLengths.SetGlobalBuffer((__gm__ int32_t *)offloadSeqKeyAddr);
    cacheTokens.SetGlobalBuffer((__gm__ int32_t *)cacheTokensAddr);
    const uint32_t owner = GetBlockIdx() / LIMConfig::AIVS_PER_AIC;
    // KERNEL_TYPE_MIX_AIC_1_2 exposes 2 * GetBlockNum() AIV blocks.  Even
    // AIVs therefore provide exactly GetBlockNum() independent owners.
    const uint32_t owners = GetBlockNum();
    TBuf<TPosition::VECCALC> identityBaseBuf;
    TBuf<TPosition::VECCALC> identityValuesBuf;
    TBuf<TPosition::VECCALC> transitionInvalidBuf;
    bool identityBufferReady = false;
    bool transitionCleanupBufferReady = false;
    // Every owner participates in every standard row.  The previous
    // batch-strided assignment left all but one owner idle for BS=1, making a
    // large identity row a single-AIV GM write.  Standard LI does not consume
    // cacheSlots, so owners can publish disjoint chunks before the outer
    // SyncAll without introducing a data dependency.  Keep the transition
    // cleanup on its established one-owner-per-request assignment below.
    for (uint32_t batch = 0U; batch < batchSize; ++batch) {
        const int32_t state = states.GetValue(batch);
        if (state != LIMConfig::REQUEST_STATE_NON_OFFLOAD &&
            state != LIMConfig::REQUEST_STATE_STEADY) continue;
        if (state == LIMConfig::REQUEST_STATE_STEADY &&
            batch % owners != owner) continue;
        const int32_t queryEnd = queryEnds.GetValue(batch);
        const int32_t queryStart = batch == 0U ? 0 : queryEnds.GetValue(batch - 1U);
        const int32_t routes = queryEnd - queryStart;
        const int32_t keyLength = keyLengths.GetValue(batch);
        if (routes < 1 || routes > static_cast<int32_t>(LIMConfig::MAX_ROUTES) ||
            queryEnd > static_cast<int32_t>(totalQueries) ||
            keyLength < routes || keyLength > static_cast<int32_t>(sourceCapacity)) {
            continue;
        }
        const int32_t row = entries.GetValue(batch);
        if (row < 0 || static_cast<uint32_t>(row) >= poolSize) continue;
        const uint64_t base = static_cast<uint64_t>(row) * sourceCapacity;
        if (state == LIMConfig::REQUEST_STATE_STEADY) {
            const int32_t length = offloadLengths.GetValue(batch);
            const int32_t cacheCount = cacheTokens.GetValue(batch);
            if (length < static_cast<int32_t>(LIMConfig::TOPK) || length > keyLength ||
                length > static_cast<int32_t>(sourceCapacity) ||
                cacheCount < static_cast<int32_t>(LIMConfig::TOPK) || cacheCount > length ||
                (length & static_cast<int32_t>(LIMConfig::CACHE_BLOCK_SIZE - 1U)) != 0 ||
                (cacheCount & static_cast<int32_t>(LIMConfig::CACHE_BLOCK_SIZE - 1U)) != 0 ||
                (length <= routes * static_cast<int32_t>(LIMConfig::TOPK) && cacheCount != length) ||
                (length > routes * static_cast<int32_t>(LIMConfig::TOPK) &&
                 (cacheCount < routes * static_cast<int32_t>(LIMConfig::TOPK) ||
                  cacheCount > static_cast<int32_t>(LIMConfig::MAX_CACHE_TOKENS))) ||
                static_cast<uint32_t>(length) >
                    ((static_cast<uint32_t>(keyLength) - static_cast<uint32_t>(routes)) /
                     LIMConfig::CACHE_BLOCK_SIZE) * LIMConfig::CACHE_BLOCK_SIZE) {
                continue;
            }
            // A steady offload row contains only slots in [0, C) or a
            // negative sentinel.  An identity row left by state=-3 has
            // cacheSlots[C] == C, so only that lifecycle transition needs
            // the full-row cleanup.  Avoid scanning the row on every decode.
            if (static_cast<uint32_t>(cacheCount) >= sourceCapacity) {
                continue;
            }
            if (slots.GetValue(base + static_cast<uint32_t>(cacheCount)) <
                cacheCount) {
                continue;
            }
            // The scalar probe populated its DCache line with the identity
            // value.  Invalidate it before DMA overwrites the suffix, or a
            // later cache writeback could restore slot C and corrupt -1.
            const uint32_t probeLine = static_cast<uint32_t>(cacheCount) & ~31U;
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE,
                                     DcciDst::CACHELINE_OUT>(
                slots[base + probeLine]);
            if (!transitionCleanupBufferReady) {
                pipe->InitBuffer(transitionInvalidBuf,
                                 TRANSITION_CLEANUP_CHUNK * sizeof(int32_t));
                LocalTensor<float> invalidValues =
                    transitionInvalidBuf.Get<int32_t>().ReinterpretCast<float>();
                // IEEE -0.0F has the INT32_MIN bit pattern; reinterpreting
                // the buffer produces the required invalid slot sentinel.
                Duplicate(invalidValues, -0.0F, TRANSITION_CLEANUP_CHUNK);
                PipeBarrier<PIPE_V>();
                transitionCleanupBufferReady = true;
            }
            LocalTensor<int32_t> invalidValues = transitionInvalidBuf.Get<int32_t>();
            LIServiceVec::SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            // A proven identity row already has the correct initial resident
            // mapping in [0, C).  DMA-fill only [C, N) instead of scalar
            // scanning or vector-comparing every source slot.
            for (uint32_t source = static_cast<uint32_t>(cacheCount);
                 source < sourceCapacity;
                 source += TRANSITION_CLEANUP_CHUNK) {
                const uint32_t valid = sourceCapacity - source < TRANSITION_CLEANUP_CHUNK
                    ? sourceCapacity - source
                    : TRANSITION_CLEANUP_CHUNK;
                DataCopyPad(
                    slots[base + source], invalidValues,
                    {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0});
                LIServiceVec::SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
            continue;
        }
        const uint32_t identityChunk =
            sourceCapacity >= IDENTITY_LONG_CUTOFF
                ? IDENTITY_LONG_CHUNK : IDENTITY_SHORT_CHUNK;
        const uint32_t firstSource = owner * identityChunk;
        if (firstSource >= sourceCapacity) continue;
        if (!identityBufferReady) {
            pipe->InitBuffer(identityBaseBuf,
                             IDENTITY_BUFFER_CAPACITY * sizeof(int32_t));
            pipe->InitBuffer(identityValuesBuf,
                             IDENTITY_BUFFER_CAPACITY * sizeof(int32_t));
            LocalTensor<int32_t> identityBase =
                identityBaseBuf.Get<int32_t>();
            Arange<int32_t>(identityBase, 0, 1,
                            static_cast<int32_t>(identityChunk));
            PipeBarrier<PIPE_V>();
            identityBufferReady = true;
        }
        LocalTensor<int32_t> identityBase = identityBaseBuf.Get<int32_t>();
        LocalTensor<int32_t> identityValues = identityValuesBuf.Get<int32_t>();
        for (uint32_t source = firstSource;
             source < sourceCapacity;
             source += owners * identityChunk) {
            const uint32_t valid =
                sourceCapacity - source < identityChunk
                    ? sourceCapacity - source
                    : identityChunk;
            Adds(identityValues, identityBase, static_cast<int32_t>(source),
                 valid);
            PipeBarrier<PIPE_V>();
            LIServiceVec::SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyPad(
                slots[base + source], identityValues,
                {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0});
            LIServiceVec::SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
    if (identityBufferReady || transitionCleanupBufferReady) {
        pipe->Reset();
    }
    }
}

// Standard LI has already written both TopK outputs.  A homogeneous state=-3
// owner only needs to publish zero request/route miss counts, so it can avoid
// constructing the full union/sort implementation and its UB layout.
__aicore__ inline bool FinalizeStandardCounts(
    __gm__ uint8_t *requestStateAddr, __gm__ uint8_t *actualSeqQueryAddr,
    __gm__ uint8_t *missCountAddr, __gm__ uint8_t *topkMissCountsAddr,
    uint32_t batchSize, uint32_t totalQueries, uint32_t first,
    uint32_t stride, TPipe *pipe)
{
    GlobalTensor<int32_t> states;
    GlobalTensor<int32_t> queryEnds;
    GlobalTensor<int32_t> requestCounts;
    GlobalTensor<int32_t> routeCounts;
    states.SetGlobalBuffer((__gm__ int32_t *)requestStateAddr);
    queryEnds.SetGlobalBuffer((__gm__ int32_t *)actualSeqQueryAddr);
    requestCounts.SetGlobalBuffer((__gm__ int32_t *)missCountAddr);
    routeCounts.SetGlobalBuffer((__gm__ int32_t *)topkMissCountsAddr);

    // Validate every request assigned to this owner before allocating UB or
    // publishing output.  Mixed owners retain the established union path.
    for (uint32_t batch = first; batch < batchSize; batch += stride) {
        const uint32_t queryStart = batch == 0U
            ? 0U : static_cast<uint32_t>(queryEnds.GetValue(batch - 1U));
        const uint32_t queryEnd =
            static_cast<uint32_t>(queryEnds.GetValue(batch));
        if (states.GetValue(batch) != LIMConfig::REQUEST_STATE_NON_OFFLOAD ||
            queryEnd <= queryStart || queryEnd > totalQueries ||
            queryEnd - queryStart > LIMConfig::MAX_ROUTES) {
            return false;
        }
    }

    TBuf<TPosition::VECCALC> countBuf;
    pipe->InitBuffer(countBuf,
                     LIMConfig::ROUTE_COUNT_BUFFER_ELEMENTS * sizeof(int32_t));
    LocalTensor<int32_t> zeros = countBuf.Get<int32_t>();
    Duplicate(zeros, 0, LIMConfig::ROUTE_COUNT_BUFFER_ELEMENTS);
    PipeBarrier<PIPE_V>();
    LIServiceVec::SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    for (uint32_t batch = first; batch < batchSize; batch += stride) {
        const uint32_t queryStart = batch == 0U
            ? 0U : static_cast<uint32_t>(queryEnds.GetValue(batch - 1U));
        const uint32_t queryEnd =
            static_cast<uint32_t>(queryEnds.GetValue(batch));
        DataCopyPad(requestCounts[batch], zeros,
                    {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        DataCopyPad(routeCounts[queryStart], zeros,
                    {1, static_cast<uint16_t>((queryEnd - queryStart) *
                                              sizeof(int32_t)), 0, 0});
    }
    LIServiceVec::SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    return true;
}

#define LI_MTP_COPY_TILING()                                                                                           \
    GET_TILING_DATA_WITH_STRUCT(FusedLightningIndexerManageTilingData, tiling_data_in, tiling);                                      \
    const FusedLightningIndexerManageTilingData *__restrict tiling_data = &tiling_data_in

#define INVOKE_LI_MTP_TOPK(...)                                                                                        \
    do {                                                                                                               \
        LI_MTP_COPY_TILING();                                                                                           \
        PrepareNonOffloadRows(requestState, reqPoolEntries, actualSeqLengthsQuery, actualSeqLengthsKey,               \
                              offloadSeqLengthsKey, cacheTokens,                                                       \
                              cacheSlots, tiling_data->bSize, tiling_data->tSize,                                      \
                              tiling_data->poolSize, tiling_data->cacheSlotsSize, &tPipe);                             \
        if ASCEND_IS_AIV { SyncAll(); }                                                                                \
        __gm__ uint8_t *unionPair0 =                                                                                    \
            user + MtpWorkspace::Pair0Offset(GetBlockNum(), tiling_data->n1Size);                                      \
        __gm__ uint8_t *unionPair1 =                                                                                    \
            user + MtpWorkspace::Pair1Offset(GetBlockNum(), tiling_data->n1Size, tiling_data->bSize);                  \
        __gm__ uint8_t *scoreScratch =                                                                                  \
            user + MtpWorkspace::ScoreOffset(GetBlockNum(), tiling_data->n1Size, tiling_data->bSize);                  \
        __gm__ uint8_t *thresholdScratch =                                                                              \
            user + MtpWorkspace::ThresholdOffset(GetBlockNum(), tiling_data->n1Size, tiling_data->bSize,               \
                                                   tiling_data->tSize,                                                   \
                                                   tiling_data->cacheSlotsSize);                                         \
        __gm__ uint8_t *routeMissCounts =                                                                               \
            user + MtpWorkspace::RouteCountOffset(GetBlockNum(), tiling_data->n1Size, tiling_data->bSize,              \
                                                    tiling_data->tSize,                                                  \
                                                    tiling_data->cacheSlotsSize);                                        \
        LIPreload<LIType<__VA_ARGS__, int32_t, true, LI_LAYOUT::TND, LI_LAYOUT::PA_BSND>> op;                           \
        op.Init(query, key, weights, reqPoolEntries, cacheSlots, actualSeqLengthsQuery, actualSeqLengthsKey,           \
                offloadSeqLengthsKey, requestState, blockTable,                                                       \
                topkSourceIds, topkSlots, unionPair0, unionPair1, scoreScratch, thresholdScratch, routeMissCounts,      \
                user, tiling_data, &tPipe);                                                                             \
        op.Process();                                                                                                   \
        if ASCEND_IS_AIV {                                                                                              \
            SyncAll();                                                                                                  \
            if (GetBlockIdx() % LIMConfig::AIVS_PER_AIC == 0U) {                                                       \
                tPipe.Reset();                                                                                          \
                const uint32_t unionOwner = GetBlockIdx() / LIMConfig::AIVS_PER_AIC;                                   \
                const uint32_t unionOwners = GetBlockNum();                                                            \
                if (!FinalizeStandardCounts(requestState, actualSeqLengthsQuery, missCount,                            \
                                            topkMissCounts, tiling_data->bSize, tiling_data->tSize,                    \
                                            unionOwner, unionOwners, &tPipe)) {                                         \
                    MtpUnion::MtpMissUnion unionOp;                                                                     \
                    unionOp.Init(unionPair0, unionPair1, offloadSeqLengthsKey, actualSeqLengthsQuery,                 \
                                 actualSeqLengthsKey, requestState,                                                   \
                                 cacheSlots, cacheTokens,                                                              \
                                 reqPoolEntries, scoreScratch, thresholdScratch, routeMissCounts,                      \
                                 missSrcIds,                                                                           \
                                 missDstSlots, missCount, topkSourceIds, topkSlots, topkMissCounts,                    \
                                 tiling_data->bSize, tiling_data->tSize, tiling_data->s2Size,                         \
                                 tiling_data->poolSize,                                                                \
                                 tiling_data->cacheSlotsSize, &tPipe);                                                   \
                    unionOp.Process(unionOwner, unionOwners);                                                          \
                }                                                                                                       \
            }                                                                                                           \
        }                                                                                                               \
    } while (0)

template <int DT>
__global__ __aicore__ void fused_lightning_indexer_manage(
    __gm__ uint8_t *weights, __gm__ uint8_t *queryDequantScale,
    __gm__ uint8_t *query, __gm__ uint8_t *keyDequantScale,
    __gm__ uint8_t *key, __gm__ uint8_t *blockTable,
    __gm__ uint8_t *actualSeqLengthsQuery,
    __gm__ uint8_t *actualSeqLengthsKey,
    __gm__ uint8_t *offloadSeqLengthsKey,
    __gm__ uint8_t *cacheTokens, __gm__ uint8_t *requestState,
    __gm__ uint8_t *reqPoolEntries, __gm__ uint8_t *cacheSlots,
    __gm__ uint8_t *topkSourceIds, __gm__ uint8_t *topkSlots,
    __gm__ uint8_t *topkMissCounts, __gm__ uint8_t *missSrcIds,
    __gm__ uint8_t *missDstSlots, __gm__ uint8_t *missCount,
    __gm__ uint8_t *cacheSlotsOut, __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__CCE_AICORE__ == 200)
#else
    TPipe tPipe;
    (void)queryDequantScale;
    (void)keyDequantScale;
    (void)actualSeqLengthsKey;
    (void)requestState;
    (void)cacheSlotsOut;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (DT == LI_MTP_TPL_FP16) {
        INVOKE_LI_MTP_TOPK(half, half);
    } else {
        INVOKE_LI_MTP_TOPK(bfloat16_t, bfloat16_t);
    }
#endif
}
