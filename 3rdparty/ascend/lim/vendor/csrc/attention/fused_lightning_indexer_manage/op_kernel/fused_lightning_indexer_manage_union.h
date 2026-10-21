#ifndef FUSED_LIGHTNING_INDEXER_MANAGE_UNION_H
#define FUSED_LIGHTNING_INDEXER_MANAGE_UNION_H

#include "kernel_operator.h"
#include "fused_lightning_indexer_manage_constants.h"
#include "lightning_indexer_vector.h"

namespace MtpUnion {
using namespace AscendC;

constexpr uint32_t ROUTES = LIMConfig::MATURE_UNION_MAX_ROUTES;
constexpr uint32_t LOCAL_MAX_ROUTES = 7U;
constexpr uint32_t MAX_ROUTES = LIMConfig::MAX_ROUTES;
constexpr uint32_t TOPK = LIMConfig::TOPK;
constexpr uint32_t MISS_CAPACITY = LIMConfig::MISS_CAPACITY;
constexpr uint32_t MAX_CACHE_TOKENS = LIMConfig::MAX_CACHE_TOKENS;
constexpr uint32_t PAIR_WORDS = TOPK * 2U;
constexpr uint32_t CAPACITY = ROUTES * TOPK;
// Keep the mature victim payload codec separate from the logical source-ID
// codec. The payload remains [slot:15 | source_low17]; long steady eviction
// carries source_high4 in the sort-key tag while union keys use the full ID.
constexpr uint32_t PACKED_SOURCE_BITS = LIServiceVec::PACKED_SOURCE_BITS;
constexpr uint32_t PACKED_SOURCE_MASK = LIServiceVec::PACKED_SOURCE_MASK;
constexpr uint32_t LONG_SOURCE_BITS = LIServiceVec::LONG_SOURCE_BITS;
constexpr uint32_t LONG_SOURCE_MASK = LIServiceVec::LONG_SOURCE_MASK;
constexpr int32_t INVALID_CACHE_SLOT = static_cast<int32_t>(0x80000000U);
constexpr uint32_t MISS_KEY_BASE_BITS = 0x40000000U;
constexpr uint32_t EVICT_CHUNK = 512U;
constexpr uint32_t EVICT_PAIR_WORDS = EVICT_CHUNK * 2U;
constexpr uint32_t EVICT_SORT_REPEATS = EVICT_CHUNK / 32U;
// AscendC compare masks must own an aligned vector-work region.  Keep the
// same full-chunk spacing as the mature single-query LIM implementation;
// packing the uint8 view down to its logical byte count aliases the following
// invalid-key and Sort32 work tensors on device.
constexpr uint32_t EVICT_MASK_WORK_FLOATS = EVICT_CHUNK;
constexpr uint32_t EVICT_SCRATCH_FLOATS = EVICT_CHUNK * 12U;
constexpr uint32_t GENERAL_EVICT_SCRATCH_FLOATS =
    EVICT_CHUNK * (MAX_ROUTES + 8U);
constexpr uint32_t THRESHOLD_STRIDE = LIMConfig::THRESHOLD_WORKSPACE_STRIDE;
constexpr uint32_t ROUTE_COUNT_STRIDE = LIMConfig::ROUTE_COUNT_WORKSPACE_STRIDE;
constexpr int32_t ROUTE_COUNTS_FROM_WORKSPACE = -1;
constexpr uint32_t ROUTE_POSITION_BITS = 11U;
constexpr uint32_t ROUTE_POSITION_MASK = (1U << ROUTE_POSITION_BITS) - 1U;
constexpr uint32_t OCCURRENCE_ROUTE_BITS = 4U;
constexpr uint32_t OCCURRENCE_ROUTE_MASK =
    (1U << OCCURRENCE_ROUTE_BITS) - 1U;
constexpr uint32_t OCCURRENCE_UNION_SHIFT =
    ROUTE_POSITION_BITS + OCCURRENCE_ROUTE_BITS;
// Leave 1 KiB of UB headroom for pipe bookkeeping.  The steady path is far
// below this limit; an all-four-routes-miss boundary falls back to the proven
// source join instead of allocating the full 192-KiB UB capacity.
constexpr uint32_t OCCURRENCE_CAPACITY = CAPACITY - 256U;
// The compact destination publication shares the expanded union UB layout.
// Keep it within the same 4096-occurrence contract as SortAll; beyond this
// boundary route-row publication can overwrite unionSources[4096].  Larger
// unions remain in this generalized function and use its exact source join.
constexpr uint32_t OCCURRENCE_FAST_CAPACITY = TOPK * 2U;
constexpr uint32_t GENERAL_SORT_CAPACITY = TOPK * 2U;
constexpr uint32_t GENERAL_ROUTE_ALIGN = 8U;
constexpr uint32_t FIRST_DECODE_LIST_CAPACITY = CAPACITY * 2U;
// First decode can keep eight complete TopK rows in the existing 16K pair
// buffers.  Keep this limit independent from LOCAL_MAX_ROUTES: the steady
// occurrence path still needs its seven-route alignment headroom.
constexpr uint32_t LOCAL_FIRST_DECODE_ROUTES = 8U;
constexpr uint32_t LOCAL_FIRST_DECODE_MAX =
    LOCAL_FIRST_DECODE_ROUTES * TOPK;
// DataCopyPad's block length is uint16 bytes.  Q1-Q7 retain their proven
// single-DMA publication; the exact Q8 list is 65536 bytes and is chunked.
constexpr uint32_t FIRST_DECODE_SINGLE_DMA_MAX = 16256U;
constexpr uint32_t FIRST_DECODE_CLEAR_CHUNK = CAPACITY;
// Arange has a proven 4096-element vector form in the -3 identity path.
// Keep first-decode destination publication within that form: cache budgets
// can be 8192--16256, while a single large Arange does not reliably publish
// the complete identity range on the target platform.
constexpr uint32_t FIRST_DECODE_LIST_CHUNK = 4096U;
static_assert(GENERAL_SORT_CAPACITY <= OCCURRENCE_CAPACITY,
              "sorted generalized misses must fit occurrenceBuf");
static_assert(LOCAL_MAX_ROUTES * (TOPK + GENERAL_ROUTE_ALIGN - 1U) <=
                  CAPACITY * 2U,
              "local generalized route prefixes must fit pairInBuf");
static_assert(TOPK == (1U << ROUTE_POSITION_BITS),
              "route position encoding must cover one TopK row");
static_assert(MAX_ROUTES <= (1U << OCCURRENCE_ROUTE_BITS),
              "occurrence route encoding must cover every MTP route");
static_assert(LOCAL_MAX_ROUTES * TOPK <= CAPACITY * 2U,
              "local generalized TopK rows must fit pairInBuf");
static_assert(LOCAL_FIRST_DECODE_MAX <= FIRST_DECODE_LIST_CAPACITY,
              "maximum first-decode cache budget must fit pair buffers");
static_assert(LOCAL_FIRST_DECODE_ROUTES * TOPK <=
                  FIRST_DECODE_LIST_CAPACITY,
              "local first-decode TopK rows must fit pair buffers");
static_assert(FIRST_DECODE_SINGLE_DMA_MAX * sizeof(int32_t) <= 65535U,
              "single-DMA first-decode source list must fit block length");
static_assert(FIRST_DECODE_CLEAR_CHUNK * sizeof(int32_t) <= 65535U,
              "first-decode clear chunk must fit one DataCopyPad block");
static_assert(FIRST_DECODE_LIST_CHUNK * 2U <= CAPACITY,
              "first-decode identity base and values must fit unionSourceBuf");
static_assert(EVICT_PAIR_WORDS + GENERAL_EVICT_SCRATCH_FLOATS +
                      EVICT_PAIR_WORDS <=
                  CAPACITY * 2U,
              "generalized eviction scratch must fit pairInBuf");
// Sort32/MrgSort do not provide a useful ordering guarantee for infinities.
// Keep a wide finite gap between the invalid key and the scan stop threshold.
constexpr float INVALID_EVICT_KEY = -1.0e20F;
constexpr float EVICT_STOP_KEY = -5.0e19F;

__aicore__ inline uint32_t SourceMaskForLength(uint32_t length)
{
    return length > (1U << PACKED_SOURCE_BITS)
        ? LONG_SOURCE_MASK
        : PACKED_SOURCE_MASK;
}

__aicore__ inline int32_t MissKeyDecodeBase(uint32_t sourceMask)
{
    return static_cast<int32_t>(MISS_KEY_BASE_BITS + sourceMask);
}

__aicore__ inline bool UseCompactOccurrenceMap(uint32_t total)
{
    return total <= OCCURRENCE_FAST_CAPACITY;
}

// The hardware sort pair payload has the same 17-bit source-ID contract as
// the LI TopK payload.  For a 21-bit source, retain the low 17 bits there and
// place the four high bits in otherwise insignificant low bits of the sort
// key.  EVICT_CHUNK divides 2^17, so every candidate chunk has one shared
// high-bit value.  This is the eviction equivalent of LIVector::TagLongIndex.
__aicore__ inline void TagLongEvictSource(
    LocalTensor<float> key, uint32_t sourceBase, uint32_t count)
{
    LocalTensor<uint32_t> keyBits = key.ReinterpretCast<uint32_t>();
    ShiftRight(keyBits, keyBits, LIServiceVec::SCORE_TAG_CLEAR_SHIFT, count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(keyBits, keyBits, LIServiceVec::SCORE_TAG_CLEAR_SHIFT, count);
    PipeBarrier<PIPE_V>();
    Adds(keyBits.ReinterpretCast<int32_t>(), keyBits.ReinterpretCast<int32_t>(),
         static_cast<int32_t>(
             (sourceBase >> PACKED_SOURCE_BITS) &
             ((1U << LIServiceVec::INDEX_HIGH_BITS) - 1U)),
         count);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline uint32_t DecodeLongEvictSource(uint32_t keyBits,
                                                   uint32_t payload)
{
    const uint32_t high =
        (keyBits & ((1U << LIServiceVec::INDEX_HIGH_BITS) - 1U)) <<
        PACKED_SOURCE_BITS;
    return high | (payload & PACKED_SOURCE_MASK);
}

template <HardEvent event>
__aicore__ inline void Sync(HardEvent e)
{
    event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(e));
    AscendC::SetFlag<event>(id);
    AscendC::WaitFlag<event>(id);
}

class MtpMissUnion {
public:
    __aicore__ inline void Init(GM_ADDR pair0, GM_ADDR pair1,
                                GM_ADDR candidateLengths, GM_ADDR actualSeqLengthsQuery,
                                GM_ADDR actualSeqLengthsKey, GM_ADDR requestState,
                                GM_ADDR cacheSlots,
                                GM_ADDR cacheTokens, GM_ADDR reqEntries,
                                GM_ADDR scoreScratch, GM_ADDR thresholdScratch,
                                GM_ADDR routeMissCounts,
                                GM_ADDR missSources, GM_ADDR missDestinations,
                                GM_ADDR counts, GM_ADDR topkSources,
                                GM_ADDR topkDestinations,
                                GM_ADDR topkMissCounts,
                                uint32_t batch, uint32_t totalQueries,
                                uint32_t scoreCapacity, uint32_t poolCapacity,
                                uint32_t cacheCapacity,
                                TPipe *pipe)
    {
        pair0Gm.SetGlobalBuffer((__gm__ float *)pair0);
        pair1Gm.SetGlobalBuffer((__gm__ float *)pair1);
        candidateLengthsGm.SetGlobalBuffer((__gm__ int32_t *)candidateLengths);
        actualSeqLengthsQueryGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsQuery);
        actualSeqLengthsKeyGm.SetGlobalBuffer((__gm__ int32_t *)actualSeqLengthsKey);
        requestStateGm.SetGlobalBuffer((__gm__ int32_t *)requestState);
        cacheSlotsGm.SetGlobalBuffer((__gm__ int32_t *)cacheSlots);
        cacheTokensGm.SetGlobalBuffer((__gm__ int32_t *)cacheTokens);
        reqEntriesGm.SetGlobalBuffer((__gm__ int32_t *)reqEntries);
        scoreScratchGm.SetGlobalBuffer((__gm__ float *)scoreScratch);
        thresholdScratchGm.SetGlobalBuffer((__gm__ float *)thresholdScratch);
        routeMissCountsGm.SetGlobalBuffer((__gm__ int32_t *)routeMissCounts);
        missSourcesGm.SetGlobalBuffer((__gm__ int32_t *)missSources);
        missDestinationsGm.SetGlobalBuffer((__gm__ int32_t *)missDestinations);
        countsGm.SetGlobalBuffer((__gm__ int32_t *)counts);
        topkSourcesGm.SetGlobalBuffer((__gm__ int32_t *)topkSources);
        topkDestinationsGm.SetGlobalBuffer((__gm__ int32_t *)topkDestinations);
        topkMissCountsGm.SetGlobalBuffer((__gm__ int32_t *)topkMissCounts);
        batchSize = batch;
        tSize = totalQueries;
        scoreStride = ((scoreCapacity + EVICT_CHUNK - 1U) / EVICT_CHUNK) * EVICT_CHUNK;
        sourceCapacity = cacheCapacity;
        poolSize = poolCapacity;
        pipe->InitBuffer(pairInBuf, CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(pairOutBuf, CAPACITY * 2U * sizeof(float));
        pipe->InitBuffer(unionSourceBuf, CAPACITY * sizeof(int32_t));
        // This buffer holds route counts during request setup, then becomes the
        // packed (union index, route, route position) occurrence map.  Sharing
        // it keeps total UB use within the 192-KiB C220 limit.
        pipe->InitBuffer(occurrenceBuf,
                         OCCURRENCE_CAPACITY * sizeof(uint32_t));
    }

    __aicore__ inline void Process(uint32_t first, uint32_t stride)
    {
        for (uint32_t batch = first; batch < batchSize; batch += stride) {
            ProcessBatch(batch);
        }
    }

private:
    // Match the mature Q=4 publication scheme: every request owner uses MTE3
    // to write its disjoint count cells directly.  This avoids scalar-cache
    // line sharing and removes the kernel-wide post-union finalizer.
    __aicore__ inline void PublishCounts(uint32_t batch, uint32_t queryStart,
                                         uint32_t queryEnd,
                                         int32_t requestCount,
                                         int32_t fixedRouteCount)
    {
        LocalTensor<int32_t> values =
            occurrenceBuf.Get<uint32_t>().ReinterpretCast<int32_t>();
        values.SetValue(0U, requestCount);
        const bool validRange = queryEnd > queryStart && queryStart < tSize &&
                                queryEnd <= tSize;
        const uint32_t routes = validRange ? queryEnd - queryStart : 0U;
        for (uint32_t route = 0U; route < routes; ++route) {
            int32_t value = fixedRouteCount;
            if (fixedRouteCount == ROUTE_COUNTS_FROM_WORKSPACE) {
                value = routeMissCountsGm.GetValue(
                    static_cast<uint64_t>(queryStart + route) *
                    ROUTE_COUNT_STRIDE);
                if (value < 0 || value > static_cast<int32_t>(TOPK)) {
                    value = 0;
                }
            }
            values.SetValue(ROUTE_COUNT_STRIDE + route, value);
        }
        Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(countsGm[batch], values,
                    {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        if (routes != 0U) {
            DataCopyPad(
                topkMissCountsGm[queryStart], values[ROUTE_COUNT_STRIDE],
                {1, static_cast<uint16_t>(routes * sizeof(int32_t)), 0, 0});
        }
        Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline bool RouteContains(uint32_t route, int32_t source)
    {
        int32_t splitValue = routeMissCountsGm.GetValue(
            static_cast<uint64_t>(route) * ROUTE_COUNT_STRIDE);
        uint32_t split = splitValue >= 0 && splitValue <= static_cast<int32_t>(TOPK)
            ? static_cast<uint32_t>(splitValue) : 0U;
        const uint64_t base = static_cast<uint64_t>(route) * TOPK;
        const uint32_t starts[2] = {0U, split};
        const uint32_t ends[2] = {split, TOPK};
        for (uint32_t part = 0U; part < 2U; ++part) {
            uint32_t low = starts[part];
            uint32_t high = ends[part];
            while (low < high) {
                const uint32_t middle = (low + high) >> 1U;
                if (topkSourcesGm.GetValue(base + middle) < source) low = middle + 1U;
                else high = middle;
            }
            if (low < ends[part] && topkSourcesGm.GetValue(base + low) == source) return true;
        }
        return false;
    }

    __aicore__ inline bool TopkContains(uint32_t queryStart,
                                        uint32_t queryEnd,
                                        int32_t source)
    {
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            if (RouteContains(route, source)) return true;
        }
        return false;
    }

    // Merge up to eight complete TopK rows in UB.  Each LI row is published
    // as two source-ID-sorted ranges: the miss prefix and the hit suffix.
    // Keeping union construction separate lets Q9-Q14 invoke this same
    // proven operation for two route groups without changing steady state.
    __aicore__ inline bool BuildFirstDecodeUnion(
        uint32_t queryStart, uint32_t queryEnd, uint32_t length,
        LocalTensor<int32_t> routeSources,
        LocalTensor<int32_t> unionSources, uint32_t &unionCount)
    {
        const uint32_t routes = queryEnd - queryStart;
        if (routes == 0U || routes > LOCAL_FIRST_DECODE_ROUTES) {
            return false;
        }
        uint32_t cursors[MAX_ROUTES * 2U] = {0U};
        uint32_t ends[MAX_ROUTES * 2U] = {0U};

        for (uint32_t route = 0U; route < routes; ++route) {
            const int32_t splitValue = routeMissCountsGm.GetValue(
                static_cast<uint64_t>(queryStart + route) *
                ROUTE_COUNT_STRIDE);
            if (splitValue < 0 || splitValue > static_cast<int32_t>(TOPK)) {
                return false;
            }
            const uint32_t split = static_cast<uint32_t>(splitValue);
            const uint32_t rowBase = route * TOPK;
            cursors[route * 2U] = rowBase;
            ends[route * 2U] = rowBase + split;
            cursors[route * 2U + 1U] = rowBase + split;
            ends[route * 2U + 1U] = rowBase + TOPK;
            DataCopyPad(
                routeSources[rowBase],
                topkSourcesGm[
                    static_cast<uint64_t>(queryStart + route) * TOPK],
                AscendC::DataCopyExtParams{
                    1, TOPK * sizeof(int32_t), 0, 0, 0},
                AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
        Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        unionCount = 0U;
        const uint32_t segments = routes * 2U;
        while (true) {
            int32_t nextSource = INT32_MAX;
            for (uint32_t segment = 0U; segment < segments; ++segment) {
                if (cursors[segment] >= ends[segment]) continue;
                const int32_t source =
                    routeSources.GetValue(cursors[segment]);
                if (source < 0 || static_cast<uint32_t>(source) >= length) {
                    return false;
                }
                if (source < nextSource) nextSource = source;
            }
            if (nextSource == INT32_MAX) break;
            if (unionCount >= FIRST_DECODE_LIST_CAPACITY) {
                return false;
            }
            unionSources.SetValue(unionCount, nextSource);
            ++unionCount;
            for (uint32_t segment = 0U; segment < segments; ++segment) {
                while (cursors[segment] < ends[segment] &&
                       routeSources.GetValue(cursors[segment]) ==
                           nextSource) {
                    ++cursors[segment];
                }
            }
        }
        return true;
    }

    // The local first-decode path loads all rows once, then appends the
    // lowest non-union source IDs until the resident cache budget is full.
    // No pool row is touched until the complete selection is validated.
    __aicore__ inline bool BuildFirstDecodeSelection(
        uint32_t queryStart, uint32_t queryEnd, uint32_t length,
        uint32_t cacheCount, LocalTensor<int32_t> residentSources,
        LocalTensor<int32_t> unionSources, uint32_t &unionCount)
    {
        if (!BuildFirstDecodeUnion(queryStart, queryEnd, length,
                                   residentSources, unionSources,
                                   unionCount) ||
            unionCount > cacheCount) {
            return false;
        }

        // pairInBuf is no longer needed as route storage after the union has
        // been materialized in pairOutBuf, so reuse it for the complete
        // resident source list.
        for (uint32_t index = 0U; index < unionCount; ++index) {
            residentSources.SetValue(index, unionSources.GetValue(index));
        }
        uint32_t unionCursor = 0U;
        uint32_t residentCount = unionCount;
        for (uint32_t source = 0U;
             source < length && residentCount < cacheCount; ++source) {
            if (unionCursor < unionCount &&
                unionSources.GetValue(unionCursor) ==
                    static_cast<int32_t>(source)) {
                ++unionCursor;
                continue;
            }
            residentSources.SetValue(residentCount,
                                     static_cast<int32_t>(source));
            ++residentCount;
        }
        return residentCount == cacheCount;
    }

    // Q9-Q14 use the same UB union operation as Q1-Q8, split into an
    // eight-route group and one remaining group.  Only the two already
    // deduplicated group unions are merged through GM, reducing the old
    // per-output scan from up to 28 GM segment heads to one GM head and one
    // UB head.  missDestinations temporarily holds the first group while the
    // merged union and fillers are written directly to public missSources.
    __aicore__ inline bool BuildWideFirstDecodeSelection(
        uint32_t batch, uint32_t queryStart, uint32_t queryEnd,
        uint32_t length, uint32_t cacheCount,
        LocalTensor<int32_t> routeSources,
        LocalTensor<int32_t> groupUnion, uint32_t &unionCount)
    {
        const uint32_t routeCount = queryEnd - queryStart;
        if (routeCount <= LOCAL_FIRST_DECODE_ROUTES ||
            routeCount > MAX_ROUTES || cacheCount > MISS_CAPACITY) {
            return false;
        }
        const uint64_t missBase =
            static_cast<uint64_t>(batch) * MISS_CAPACITY;
        const uint32_t groupEnd = queryStart + LOCAL_FIRST_DECODE_ROUTES;
        uint32_t firstCount = 0U;
        if (!BuildFirstDecodeUnion(queryStart, groupEnd, length,
                                   routeSources, groupUnion, firstCount)) {
            return false;
        }

        Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        for (uint32_t offset = 0U; offset < firstCount;
             offset += FIRST_DECODE_LIST_CHUNK) {
            const uint32_t valid =
                firstCount - offset < FIRST_DECODE_LIST_CHUNK
                    ? firstCount - offset
                    : FIRST_DECODE_LIST_CHUNK;
            DataCopyPad(
                missDestinationsGm[missBase + offset], groupUnion[offset],
                {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0});
        }
        Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);

        uint32_t secondCount = 0U;
        if (!BuildFirstDecodeUnion(groupEnd, queryEnd, length,
                                   routeSources, groupUnion, secondCount)) {
            return false;
        }

        uint32_t firstCursor = 0U;
        uint32_t secondCursor = 0U;
        uint32_t output = 0U;
        int32_t firstSource = firstCursor < firstCount
            ? missDestinationsGm.GetValue(missBase + firstCursor)
            : INT32_MAX;
        int32_t secondSource = secondCursor < secondCount
            ? groupUnion.GetValue(secondCursor)
            : INT32_MAX;
        while (firstSource != INT32_MAX || secondSource != INT32_MAX) {
            const int32_t source = firstSource < secondSource
                ? firstSource : secondSource;
            if (source < 0 || static_cast<uint32_t>(source) >= length ||
                output >= cacheCount || output >= MISS_CAPACITY) {
                return false;
            }
            missSourcesGm.SetValue(missBase + output, source);
            ++output;
            if (firstSource == source) {
                ++firstCursor;
                firstSource = firstCursor < firstCount
                    ? missDestinationsGm.GetValue(missBase + firstCursor)
                    : INT32_MAX;
            }
            if (secondSource == source) {
                ++secondCursor;
                secondSource = secondCursor < secondCount
                    ? groupUnion.GetValue(secondCursor)
                    : INT32_MAX;
            }
        }
        unionCount = output;

        uint32_t unionCursor = 0U;
        uint32_t residentCount = unionCount;
        int32_t unionSource = unionCursor < unionCount
            ? missSourcesGm.GetValue(missBase + unionCursor)
            : INT32_MAX;
        for (uint32_t source = 0U;
             source < length && residentCount < cacheCount; ++source) {
            if (unionSource == static_cast<int32_t>(source)) {
                ++unionCursor;
                unionSource = unionCursor < unionCount
                    ? missSourcesGm.GetValue(
                          missBase + unionCursor)
                    : INT32_MAX;
                continue;
            }
            missSourcesGm.SetValue(
                missBase + residentCount, static_cast<int32_t>(source));
            ++residentCount;
        }
        if (residentCount != cacheCount) {
            return false;
        }

        return true;
    }

    // Generate one immutable invalid block and reuse it to clear the complete
    // physical pool row.  Keep resident mapping and transfer-list writes on
    // their proven scalar path so this change measures only pool clearing.
    __aicore__ inline void ClearFirstDecodePool(uint32_t row)
    {
        const uint64_t cacheBase =
            static_cast<uint64_t>(row) * sourceCapacity;
        LocalTensor<int32_t> output = unionSourceBuf.Get<int32_t>();
        LocalTensor<float> invalid = output.ReinterpretCast<float>();

        // IEEE -0.0F has the INT32_MIN bit pattern used by invalid slots.
        // Do not rebuild the block or wait between chunks: it is immutable
        // until all MTE3 copies have completed.
        Duplicate(invalid, -0.0F, FIRST_DECODE_CLEAR_CHUNK);
        Sync<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        for (uint32_t source = 0U; source < sourceCapacity;
             source += FIRST_DECODE_CLEAR_CHUNK) {
            const uint32_t valid =
                sourceCapacity - source < FIRST_DECODE_CLEAR_CHUNK
                    ? sourceCapacity - source
                    : FIRST_DECODE_CLEAR_CHUNK;
            DataCopyPad(
                cacheSlotsGm[cacheBase + source], output,
                {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0});
        }
        Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void PublishFirstDecodeLists(
        uint32_t batch, uint32_t cacheCount,
        LocalTensor<int32_t> residentSources, bool sourcesAlreadyPublished)
    {
        const uint64_t missBase =
            static_cast<uint64_t>(batch) * MISS_CAPACITY;
        LocalTensor<int32_t> identityBase = unionSourceBuf.Get<int32_t>();
        LocalTensor<int32_t> destinations =
            identityBase[FIRST_DECODE_LIST_CHUNK];

        // residentSources is already laid out as sorted union sources followed
        // by sorted fillers.  Preserve the Q1-Q7 single-DMA fast path.  The
        // exact Q8 list occupies 65536 bytes, one byte beyond DataCopyPad's
        // uint16 block-length range, so publish only that boundary in chunks.
        if (!sourcesAlreadyPublished) {
            Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            if (cacheCount <= FIRST_DECODE_SINGLE_DMA_MAX) {
                DataCopyPad(
                    missSourcesGm[missBase], residentSources,
                    {1, static_cast<uint16_t>(
                            cacheCount * sizeof(int32_t)), 0, 0});
            } else {
                for (uint32_t source = 0U; source < cacheCount;
                     source += FIRST_DECODE_LIST_CHUNK) {
                    const uint32_t valid =
                        cacheCount - source < FIRST_DECODE_LIST_CHUNK
                            ? cacheCount - source
                            : FIRST_DECODE_LIST_CHUNK;
                    DataCopyPad(
                        missSourcesGm[missBase + source],
                        residentSources[source],
                        {1, static_cast<uint16_t>(
                                valid * sizeof(int32_t)), 0, 0});
                }
            }
        }

        // Destination slots are the identity range [0, C).  Build the
        // immutable 0..4095 base once, then use Adds for each chunk.  This is
        // the same 4096-element Arange form used by the proven -3 identity
        // fill; generating 8192+ elements in one Arange corrupts some -2
        // transfer lists on the target platform.
        Arange<int32_t>(identityBase, 0, 1,
                        static_cast<int32_t>(FIRST_DECODE_LIST_CHUNK));
        PipeBarrier<PIPE_V>();
        for (uint32_t slot = 0U; slot < cacheCount;
             slot += FIRST_DECODE_LIST_CHUNK) {
            const uint32_t valid =
                cacheCount - slot < FIRST_DECODE_LIST_CHUNK
                    ? cacheCount - slot
                    : FIRST_DECODE_LIST_CHUNK;
            LocalTensor<int32_t> values = identityBase;
            if (slot != 0U) {
                Adds(destinations, identityBase, static_cast<int32_t>(slot),
                     valid);
                PipeBarrier<PIPE_V>();
                values = destinations;
            }
            Sync<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DataCopyPad(
                missDestinationsGm[missBase + slot], values,
                {1, static_cast<uint16_t>(valid * sizeof(int32_t)), 0, 0});
            Sync<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
        Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void ResolveFirstDecodeTopk(
        uint32_t queryStart, uint32_t queryEnd, uint32_t unionCount,
        LocalTensor<int32_t> unionSources, uint32_t cacheRow,
        bool resolveFromPool)
    {
        LocalTensor<int32_t> routeStorage = unionSourceBuf.Get<int32_t>();
        LocalTensor<int32_t> routeSources = routeStorage;
        LocalTensor<int32_t> routeDestinations = routeStorage[TOPK];
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            const uint64_t topkBase = static_cast<uint64_t>(route) * TOPK;
            DataCopyPad(
                routeSources, topkSourcesGm[topkBase],
                AscendC::DataCopyExtParams{
                    1, TOPK * sizeof(int32_t), 0, 0, 0},
                AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

            if (resolveFromPool) {
                const uint64_t cacheBase =
                    static_cast<uint64_t>(cacheRow) * sourceCapacity;
                for (uint32_t position = 0U; position < TOPK; ++position) {
                    const int32_t source = routeSources.GetValue(position);
                    const int32_t slot = source >= 0 &&
                            static_cast<uint32_t>(source) < sourceCapacity
                        ? cacheSlotsGm.GetValue(
                              cacheBase + static_cast<uint32_t>(source))
                        : -1;
                    routeDestinations.SetValue(position, slot);
                }
                Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(
                    topkDestinationsGm[topkBase], routeDestinations,
                    {1, static_cast<uint16_t>(TOPK * sizeof(int32_t)), 0, 0});
                Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                continue;
            }

            const int32_t splitValue = routeMissCountsGm.GetValue(
                static_cast<uint64_t>(route) * ROUTE_COUNT_STRIDE);
            const uint32_t split = splitValue >= 0 &&
                                           splitValue <=
                                               static_cast<int32_t>(TOPK)
                                       ? static_cast<uint32_t>(splitValue)
                                       : 0U;
            const uint32_t starts[2] = {0U, split};
            const uint32_t ends[2] = {split, TOPK};
            for (uint32_t part = 0U; part < 2U; ++part) {
                uint32_t unionCursor = 0U;
                for (uint32_t position = starts[part];
                     position < ends[part]; ++position) {
                    const int32_t source = routeSources.GetValue(position);
                    while (unionCursor < unionCount &&
                           unionSources.GetValue(unionCursor) < source) {
                        ++unionCursor;
                    }
                    const int32_t slot =
                        unionCursor < unionCount &&
                                unionSources.GetValue(unionCursor) == source
                            ? static_cast<int32_t>(unionCursor)
                            : -1;
                    routeDestinations.SetValue(position, slot);
                }
            }
            Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(
                topkDestinationsGm[topkBase], routeDestinations,
                {1, static_cast<uint16_t>(TOPK * sizeof(int32_t)), 0, 0});
            Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        }
    }

    __aicore__ inline void InitializeFirstDecode(uint32_t batch)
    {
        const uint32_t queryStart = batch == 0U ? 0U :
            static_cast<uint32_t>(actualSeqLengthsQueryGm.GetValue(batch - 1U));
        const uint32_t queryEnd =
            static_cast<uint32_t>(actualSeqLengthsQueryGm.GetValue(batch));
        if (queryStart >= queryEnd || queryEnd > tSize ||
            queryEnd - queryStart > MAX_ROUTES) {
            PublishCounts(batch, queryStart, queryEnd, 0, 0);
            return;
        }
        int32_t row = -1;
        uint32_t length = 0U;
        uint32_t cacheCount = 0U;
        if (!ValidateOffloadRequest(batch, queryStart, queryEnd, row, length,
                                    cacheCount)) {
            PublishSafeFailure(batch, queryStart, queryEnd);
            return;
        }
        const uint64_t cacheBase =
            static_cast<uint64_t>(row) * sourceCapacity;
        LocalTensor<int32_t> residentSources =
            pairInBuf.Get<float>().ReinterpretCast<int32_t>();
        LocalTensor<int32_t> unionSources =
            pairOutBuf.Get<float>().ReinterpretCast<int32_t>();
        uint32_t unionCount = 0U;
        const uint32_t routeCount = queryEnd - queryStart;
        const bool useLocalSelection =
            (routeCount <= LOCAL_MAX_ROUTES &&
             cacheCount <= FIRST_DECODE_SINGLE_DMA_MAX) ||
            (routeCount == LOCAL_FIRST_DECODE_ROUTES &&
             cacheCount <= LOCAL_FIRST_DECODE_MAX);
        const uint64_t missBase =
            static_cast<uint64_t>(batch) * MISS_CAPACITY;
        if (useLocalSelection) {
            if (!BuildFirstDecodeSelection(queryStart, queryEnd, length,
                                           cacheCount, residentSources,
                                           unionSources, unionCount)) {
                PublishSafeFailure(batch, queryStart, queryEnd);
                return;
            }
        } else if (routeCount > LOCAL_FIRST_DECODE_ROUTES) {
            if (!BuildWideFirstDecodeSelection(
                    batch, queryStart, queryEnd, length, cacheCount,
                    residentSources, unionSources, unionCount)) {
                PublishSafeFailure(batch, queryStart, queryEnd);
                return;
            }
        } else {
            uint64_t cursors[MAX_ROUTES * 2U] = {0U};
            uint64_t ends[MAX_ROUTES * 2U] = {0U};
            for (uint32_t route = 0U; route < routeCount; ++route) {
                const int32_t splitValue = routeMissCountsGm.GetValue(
                    static_cast<uint64_t>(queryStart + route) *
                    ROUTE_COUNT_STRIDE);
                if (splitValue < 0 ||
                    splitValue > static_cast<int32_t>(TOPK)) {
                    PublishSafeFailure(batch, queryStart, queryEnd);
                    return;
                }
                const uint64_t routeBase =
                    static_cast<uint64_t>(queryStart + route) * TOPK;
                const uint32_t split = static_cast<uint32_t>(splitValue);
                cursors[route * 2U] = routeBase;
                ends[route * 2U] = routeBase + split;
                cursors[route * 2U + 1U] = routeBase + split;
                ends[route * 2U + 1U] = routeBase + TOPK;
            }
            const uint32_t segments = routeCount * 2U;
            while (true) {
                int32_t nextSource = INT32_MAX;
                for (uint32_t segment = 0U; segment < segments; ++segment) {
                    if (cursors[segment] >= ends[segment]) continue;
                    const int32_t source =
                        topkSourcesGm.GetValue(cursors[segment]);
                    if (source < 0 ||
                        static_cast<uint32_t>(source) >= length) {
                        PublishSafeFailure(batch, queryStart, queryEnd);
                        return;
                    }
                    if (source < nextSource) nextSource = source;
                }
                if (nextSource == INT32_MAX) break;
                if (unionCount >= cacheCount ||
                    unionCount >= MISS_CAPACITY) {
                    PublishSafeFailure(batch, queryStart, queryEnd);
                    return;
                }
                missSourcesGm.SetValue(missBase + unionCount, nextSource);
                ++unionCount;
                for (uint32_t segment = 0U; segment < segments; ++segment) {
                    while (cursors[segment] < ends[segment] &&
                           topkSourcesGm.GetValue(cursors[segment]) ==
                               nextSource) {
                        ++cursors[segment];
                    }
                }
            }
            uint32_t unionCursor = 0U;
            uint32_t residentCount = unionCount;
            for (uint32_t source = 0U;
                 source < length && residentCount < cacheCount; ++source) {
                if (unionCursor < unionCount &&
                    missSourcesGm.GetValue(missBase + unionCursor) ==
                        static_cast<int32_t>(source)) {
                    ++unionCursor;
                    continue;
                }
                missSourcesGm.SetValue(
                    missBase + residentCount, static_cast<int32_t>(source));
                ++residentCount;
            }
            if (residentCount != cacheCount) {
                PublishSafeFailure(batch, queryStart, queryEnd);
                return;
            }
        }
        ClearFirstDecodePool(static_cast<uint32_t>(row));
        for (uint32_t written = 0U; written < cacheCount; ++written) {
            const uint32_t source = static_cast<uint32_t>(
                useLocalSelection
                    ? residentSources.GetValue(written)
                    : missSourcesGm.GetValue(missBase + written));
            cacheSlotsGm.SetValue(cacheBase + source,
                                  static_cast<int32_t>(written));
        }
        PublishFirstDecodeLists(batch, cacheCount, residentSources,
                                !useLocalSelection);
        ResolveFirstDecodeTopk(queryStart, queryEnd, unionCount,
                               unionSources, static_cast<uint32_t>(row),
                               !useLocalSelection);
        PublishCounts(batch, queryStart, queryEnd,
                      static_cast<int32_t>(cacheCount),
                      static_cast<int32_t>(TOPK));
    }

    __aicore__ inline bool ValidateOffloadRequest(
        uint32_t batch, uint32_t queryStart, uint32_t queryEnd,
        int32_t &row, uint32_t &length, uint32_t &cacheCount)
    {
        if (queryEnd <= queryStart ||
            queryEnd - queryStart > MAX_ROUTES) return false;
        row = reqEntriesGm.GetValue(batch);
        const int32_t lengthValue = candidateLengthsGm.GetValue(batch);
        const int32_t cacheValue = cacheTokensGm.GetValue(batch);
        const int32_t actualValue = actualSeqLengthsKeyGm.GetValue(batch);
        const uint32_t routes = queryEnd - queryStart;
        if (row < 0 || static_cast<uint32_t>(row) >= poolSize ||
            lengthValue < static_cast<int32_t>(TOPK) ||
            static_cast<uint32_t>(lengthValue) > sourceCapacity ||
            cacheValue < static_cast<int32_t>(TOPK) || cacheValue > lengthValue ||
            actualValue < static_cast<int32_t>(routes) || lengthValue > actualValue ||
            static_cast<uint32_t>(lengthValue) >
                ((static_cast<uint32_t>(actualValue) - routes) / 128U) * 128U ||
            (lengthValue & 127) != 0 || (cacheValue & 127) != 0) return false;
        if ((static_cast<uint32_t>(lengthValue) <= routes * TOPK && cacheValue != lengthValue) ||
            (static_cast<uint32_t>(lengthValue) > routes * TOPK &&
             (static_cast<uint32_t>(cacheValue) < routes * TOPK ||
              cacheValue > static_cast<int32_t>(MAX_CACHE_TOKENS)))) return false;
        length = static_cast<uint32_t>(lengthValue);
        cacheCount = static_cast<uint32_t>(cacheValue);
        return true;
    }

    __aicore__ inline void PublishSafeFailure(uint32_t batch,
                                              uint32_t queryStart,
                                              uint32_t queryEnd)
    {
        const uint32_t safeStart = queryStart < tSize ? queryStart : tSize;
        const uint32_t safeEnd = queryEnd < tSize ? queryEnd : tSize;
        for (uint32_t route = safeStart; route < safeEnd; ++route) {
            const uint64_t base = static_cast<uint64_t>(route) * TOPK;
            for (uint32_t position = 0U; position < TOPK; ++position) {
                topkSourcesGm.SetValue(base + position, -1);
                topkDestinationsGm.SetValue(base + position, -1);
            }
        }
        PublishCounts(batch, safeStart, safeEnd, 0, 0);
    }

    __aicore__ inline void ProcessGenericSteady(uint32_t batch,
                                                uint32_t queryStart,
                                                uint32_t queryEnd)
    {
        int32_t row = -1;
        uint32_t length = 0U;
        uint32_t cacheCount = 0U;
        if (!ValidateOffloadRequest(batch, queryStart, queryEnd,
                                    row, length, cacheCount)) {
            PublishSafeFailure(batch, queryStart, queryEnd);
            return;
        }
        const uint64_t cacheBase = static_cast<uint64_t>(row) * sourceCapacity;
        const uint64_t missBase =
            static_cast<uint64_t>(batch) * MISS_CAPACITY;
        // Identity-to-offload normalization is handled once by
        // PrepareNonOffloadRows.  Steady decode must not rescan the row.
        uint32_t missCount = 0U;
        const bool singleRoute = queryEnd - queryStart == 1U;
        if (singleRoute) {
            const uint64_t topkBase = static_cast<uint64_t>(queryStart) * TOPK;
            const int32_t routeMissValue = routeMissCountsGm.GetValue(
                static_cast<uint64_t>(queryStart) * ROUTE_COUNT_STRIDE);
            const uint32_t routeMiss = routeMissValue >= 0 && routeMissValue <= static_cast<int32_t>(TOPK)
                ? static_cast<uint32_t>(routeMissValue) : 0U;
            for (uint32_t position = 0U; position < routeMiss; ++position) {
                const int32_t source = topkSourcesGm.GetValue(topkBase + position);
                if (source >= 0 && static_cast<uint32_t>(source) < length &&
                    cacheSlotsGm.GetValue(cacheBase + static_cast<uint32_t>(source)) < 0) {
                    missSourcesGm.SetValue(missBase + missCount, source);
                    ++missCount;
                }
            }
        } else {
            // Every route's miss prefix is already source-ID sorted by the LI
            // producer.  Merge those prefixes directly instead of scanning
            // all L sources and repeatedly searching Q TopK rows.
            uint32_t cursors[MAX_ROUTES] = {0U};
            uint32_t routeMisses[MAX_ROUTES] = {0U};
            const uint32_t routes = queryEnd - queryStart;
            for (uint32_t route = 0U; route < routes; ++route) {
                const int32_t value = routeMissCountsGm.GetValue(
                    static_cast<uint64_t>(queryStart + route) *
                    ROUTE_COUNT_STRIDE);
                routeMisses[route] = value >= 0 &&
                        value <= static_cast<int32_t>(TOPK)
                    ? static_cast<uint32_t>(value) : 0U;
            }
            while (true) {
                int32_t nextSource = INT32_MAX;
                for (uint32_t route = 0U; route < routes; ++route) {
                    if (cursors[route] >= routeMisses[route]) continue;
                    const int32_t source = topkSourcesGm.GetValue(
                        static_cast<uint64_t>(queryStart + route) * TOPK +
                        cursors[route]);
                    if (source < nextSource) nextSource = source;
                }
                if (nextSource == INT32_MAX) break;
                missSourcesGm.SetValue(missBase + missCount, nextSource);
                ++missCount;
                for (uint32_t route = 0U; route < routes; ++route) {
                    while (cursors[route] < routeMisses[route] &&
                           topkSourcesGm.GetValue(
                               static_cast<uint64_t>(queryStart + route) *
                                   TOPK + cursors[route]) == nextSource) {
                        ++cursors[route];
                    }
                }
            }
        }
        uint32_t victim = 0U;
        uint32_t missPosition = 0U;
        const int32_t singleMissValue = routeMissCountsGm.GetValue(
            static_cast<uint64_t>(queryStart) * ROUTE_COUNT_STRIDE);
        const uint32_t singleMissCount = singleMissValue >= 0 && singleMissValue <= static_cast<int32_t>(TOPK)
            ? static_cast<uint32_t>(singleMissValue) : 0U;
        uint32_t hitPosition = singleMissCount;
        const uint64_t singleTopkBase = static_cast<uint64_t>(queryStart) * TOPK;
        for (uint32_t source = 0U; source < length && victim < missCount; ++source) {
            const int32_t slot = cacheSlotsGm.GetValue(cacheBase + source);
            bool selected = false;
            if (singleRoute) {
                while (missPosition < singleMissCount &&
                       topkSourcesGm.GetValue(singleTopkBase + missPosition) <
                           static_cast<int32_t>(source)) ++missPosition;
                while (hitPosition < TOPK &&
                       topkSourcesGm.GetValue(singleTopkBase + hitPosition) <
                           static_cast<int32_t>(source)) ++hitPosition;
                selected = (missPosition < singleMissCount &&
                    topkSourcesGm.GetValue(singleTopkBase + missPosition) == static_cast<int32_t>(source)) ||
                    (hitPosition < TOPK &&
                     topkSourcesGm.GetValue(singleTopkBase + hitPosition) == static_cast<int32_t>(source));
            } else {
                selected = TopkContains(queryStart, queryEnd,
                                        static_cast<int32_t>(source));
            }
            if (slot < 0 || selected) continue;
            const int32_t incoming = missSourcesGm.GetValue(missBase + victim);
            cacheSlotsGm.SetValue(cacheBase + source, INVALID_CACHE_SLOT);
            cacheSlotsGm.SetValue(cacheBase + static_cast<uint32_t>(incoming), slot);
            missDestinationsGm.SetValue(missBase + victim, slot);
            ++victim;
        }
        if (victim != missCount) {
            PublishSafeFailure(batch, queryStart, queryEnd);
            return;
        }
        for (uint32_t route = queryStart; route < queryEnd; ++route) {
            const uint64_t base = static_cast<uint64_t>(route) * TOPK;
            for (uint32_t position = 0U; position < TOPK; ++position) {
                const int32_t source = topkSourcesGm.GetValue(base + position);
                const int32_t slot = source >= 0 && static_cast<uint32_t>(source) < length
                    ? cacheSlotsGm.GetValue(cacheBase + static_cast<uint32_t>(source)) : -1;
                topkDestinationsGm.SetValue(base + position, slot);
            }
            const int32_t routeMiss = routeMissCountsGm.GetValue(
                static_cast<uint64_t>(route) * ROUTE_COUNT_STRIDE);
        }
        PublishCounts(batch, queryStart, queryEnd,
                      static_cast<int32_t>(missCount),
                      ROUTE_COUNTS_FROM_WORKSPACE);
    }

    // Q=1..14 steady path. Q=1..4 reuse the mature MrgSort4/vector-dedup path.
    // Q=5..14 concatenate up to 4096 miss occurrences and reuse the proven
    // fixed-size SortAll pipeline; larger correctness boundaries retain the
    // source-sorted k-way union. Both paths carry occurrence metadata so slot
    // Q1-Q7 also carry occurrence metadata; wider routes stream their
    // destination publication through one reusable local row.
    __aicore__ inline void ProcessGeneralizedSteady(uint32_t batch,
                                                    uint32_t queryStart,
                                                    uint32_t queryEnd)
    {
        int32_t row = -1;
        uint32_t length = 0U;
        uint32_t cacheCount = 0U;
        if (!ValidateOffloadRequest(batch, queryStart, queryEnd, row, length,
                                    cacheCount)) {
            PublishSafeFailure(batch, queryStart, queryEnd);
            return;
        }
        const uint32_t sourceMask = SourceMaskForLength(length);
        const uint32_t routeCount = queryEnd - queryStart;
        LocalTensor<float> input = pairInBuf.Get<float>();
        LocalTensor<float> workspace = pairOutBuf.Get<float>();
        LocalTensor<int32_t> routeSources =
            input.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> unionSources = unionSourceBuf.Get<int32_t>();
        LocalTensor<int32_t> unionDestinations =
            workspace.ReinterpretCast<int32_t>()[CAPACITY];
        LocalTensor<float> accumulator = workspace;
        LocalTensor<int32_t> countLocal =
            occurrenceBuf.Get<uint32_t>().ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> occurrences = occurrenceBuf.Get<uint32_t>();
        const bool useMaturePairUnion = routeCount <= ROUTES;

        uint32_t lengths[MAX_ROUTES] = {0U};
        uint32_t cursors[MAX_ROUTES] = {0U};
        uint32_t routeOffsets[MAX_ROUTES] = {0U};
        uint32_t total = 0U;
        uint32_t routeStorageCount = 0U;
        for (uint32_t route = 0U; route < routeCount; ++route) {
            const int32_t value = routeMissCountsGm.GetValue(
                static_cast<uint64_t>(queryStart + route) *
                ROUTE_COUNT_STRIDE);
            lengths[route] = value > 0 &&
                                     value <= static_cast<int32_t>(TOPK)
                                 ? static_cast<uint32_t>(value)
                                 : 0U;
            routeOffsets[route] = routeStorageCount;
            routeStorageCount +=
                (lengths[route] + GENERAL_ROUTE_ALIGN - 1U) /
                GENERAL_ROUTE_ALIGN * GENERAL_ROUTE_ALIGN;
            total += lengths[route];
        }
        if (total == 0U) {
            PublishCounts(batch, queryStart, queryEnd, 0, 0);
            return;
        }
        // pairInBuf retains the Q1-Q7 fast-path footprint.  Wider or highly
        // fragmented requests stay in this same generalized entry point and
        // use its GM-backed k-way implementation.
        if (!useMaturePairUnion && routeStorageCount > CAPACITY * 2U) {
            ProcessGenericSteady(batch, queryStart, queryEnd);
            return;
        }
        if (!useMaturePairUnion) {
            for (uint32_t route = 0U; route < routeCount; ++route) {
                if (lengths[route] == 0U) continue;
                DataCopyPad(
                    routeSources[routeOffsets[route]],
                    topkSourcesGm[
                        static_cast<uint64_t>(queryStart + route) * TOPK],
                    AscendC::DataCopyExtParams{
                        1,
                        static_cast<uint32_t>(lengths[route] *
                                              sizeof(int32_t)),
                        0, 0, 0},
                    AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            }
        }
        uint32_t unionCount = 0U;
        bool overflow = false;
        bool capturedOccurrences = false;
        if (useMaturePairUnion) {
            const uint64_t pairBase =
                static_cast<uint64_t>(batch) * CAPACITY;
            for (uint32_t route = 0U; route < routeCount; ++route) {
                if (lengths[route] == 0U) {
                    continue;
                }
                const uint64_t gmOffset =
                    pairBase + (route % 2U) * PAIR_WORDS;
                const AscendC::DataCopyExtParams copy{
                    1,
                    static_cast<uint32_t>(lengths[route] * 2U *
                                          sizeof(float)),
                    0, 0, 0};
                const AscendC::DataCopyPadExtParams<float> pad{
                    false, 0, 0, 0.0F};
                if (route < 2U) {
                    DataCopyPad(input[route * PAIR_WORDS],
                                pair0Gm[gmOffset], copy, pad);
                } else {
                    DataCopyPad(input[route * PAIR_WORDS],
                                pair1Gm[gmOffset], copy, pad);
                }
            }
            Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            Sync<HardEvent::S_V>(HardEvent::S_V);
            MrgSort4Info params;
            params.elementLengths[0] = lengths[0];
            params.elementLengths[1] = lengths[1];
            params.elementLengths[2] = lengths[2];
            params.elementLengths[3] = routeCount == ROUTES
                                           ? lengths[3]
                                           : 0U;
            params.ifExhaustedSuspension = false;
            params.validBit = (lengths[0] > 0U ? 0b0001 : 0U) |
                              (lengths[1] > 0U ? 0b0010 : 0U) |
                              (lengths[2] > 0U ? 0b0100 : 0U) |
                              (routeCount == ROUTES && lengths[3] > 0U
                                   ? 0b1000
                                   : 0U);
            params.repeatTimes = 1;
            MrgSortSrcList<float> sources;
            sources.src1 = input;
            sources.src2 = input[PAIR_WORDS];
            sources.src3 = input[PAIR_WORDS * 2U];
            sources.src4 = input[PAIR_WORDS * 3U];
            MrgSort<float>(workspace, sources, params);
            PipeBarrier<PIPE_V>();
            capturedOccurrences = routeCount <= LOCAL_MAX_ROUTES &&
                                  UseCompactOccurrenceMap(total);
            unionCount = DeduplicateMergedMisses(
                workspace, input, total, unionSources, occurrences,
                capturedOccurrences, sourceMask);
        } else {
            Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
            if (total <= GENERAL_SORT_CAPACITY) {
                uint32_t sortLength = 128U;
                if (total > 2048U) {
                    sortLength = 4096U;
                } else if (total > 1024U) {
                    sortLength = 2048U;
                } else if (total > 512U) {
                    sortLength = 1024U;
                } else if (total > 384U) {
                    sortLength = 512U;
                } else if (total > 256U) {
                    sortLength = 384U;
                } else if (total > 128U) {
                    sortLength = 256U;
                }

                // SortAll consumes [keys, payloads] and emits interleaved
                // (key, payload) pairs.  Encode the same descending source
                // key used by the mature LI producer, while carrying the
                // four-bit route and final miss-prefix position as payload.
                LocalTensor<uint32_t> sortWords =
                    workspace.ReinterpretCast<uint32_t>();
                // Vector instructions require 32-byte-aligned local starts.
                // routeOffsets already provide that alignment.  Retain their
                // small zero-filled gaps when they fit in the SortAll bucket;
                // the valid source keys sort ahead of every zero gap, so the
                // first `total` output pairs keep the original semantics.
                const bool useAlignedVectorBuild =
                    routeStorageCount <= sortLength;
                if (useAlignedVectorBuild) {
                    // Each miss prefix is source-ID sorted by the fused LI
                    // producer.  Endpoint validation therefore proves every
                    // source in the internal row is inside [0, length).
                    for (uint32_t route = 0U; route < routeCount; ++route) {
                        if (lengths[route] == 0U) continue;
                        const uint32_t offset = routeOffsets[route];
                        const int32_t first = routeSources.GetValue(offset);
                        const int32_t last = routeSources.GetValue(
                            offset + lengths[route] - 1U);
                        if (first < 0 || last < first ||
                            static_cast<uint32_t>(last) >= length) {
                            PublishSafeFailure(batch, queryStart, queryEnd);
                            return;
                        }
                    }

                    LocalTensor<int32_t> sortInts =
                        workspace.ReinterpretCast<int32_t>();
                    // The route prefixes arrived through MTE2.  The scalar
                    // endpoint checks above are covered by MTE2_S, while the
                    // following vector key construction needs its own direct
                    // MTE2_V dependency before reading the same UB rows.
                    Sync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                    Duplicate(sortInts, 0, sortLength);
                    Duplicate(sortInts[sortLength], 0, sortLength);
                    PipeBarrier<PIPE_V>();
                    // Generate the common 0..2047 route-position row once.
                    // Reissuing variable-length ArithProgression operations
                    // for adjacent payload ranges leaves a target-dependent
                    // V-pipe hazard (observed on the FP16 Q5/2049 boundary).
                    // pairInBuf's upper half is outside SortAll's temporary
                    // range and is later reused by deduplication, so it is a
                    // safe transient home for this fixed position row.
                    LocalTensor<int32_t> positionBase =
                        input[CAPACITY].ReinterpretCast<int32_t>();
                    ArithProgression<int32_t>(positionBase, 0, 1, TOPK);
                    PipeBarrier<PIPE_V>();
                    for (uint32_t route = 0U; route < routeCount; ++route) {
                        if (lengths[route] == 0U) continue;
                        const uint32_t offset = routeOffsets[route];
                        Muls(sortInts[offset], routeSources[offset], -1,
                             lengths[route]);
                    }
                    PipeBarrier<PIPE_V>();
                    const int32_t keyBase = static_cast<int32_t>(
                        MISS_KEY_BASE_BITS + sourceMask);
                    for (uint32_t route = 0U; route < routeCount; ++route) {
                        if (lengths[route] == 0U) continue;
                        const uint32_t offset = routeOffsets[route];
                        Adds(sortInts[offset], sortInts[offset], keyBase,
                             lengths[route]);
                        Adds(sortInts[sortLength + offset], positionBase,
                             static_cast<int32_t>(
                                 route << ROUTE_POSITION_BITS),
                             lengths[route]);
                    }
                    PipeBarrier<PIPE_V>();
                } else {
                    // A padded route layout can cross a SortAll capacity
                    // boundary even though the compact occurrence count does
                    // not.  Keep the existing scalar compaction there rather
                    // than doubling the sorting work for a few alignment
                    // gaps (for example, 4095 occurrences across six routes).
                    uint32_t packed = 0U;
                    for (uint32_t route = 0U; route < routeCount; ++route) {
                        for (uint32_t position = 0U;
                             position < lengths[route]; ++position) {
                            const int32_t source = routeSources.GetValue(
                                routeOffsets[route] + position);
                            if (source < 0 ||
                                static_cast<uint32_t>(source) >= length) {
                                PublishSafeFailure(batch, queryStart,
                                                   queryEnd);
                                return;
                            }
                            sortWords.SetValue(
                                packed,
                                MISS_KEY_BASE_BITS + sourceMask -
                                    static_cast<uint32_t>(source));
                            sortWords.SetValue(
                                sortLength + packed,
                                (route << ROUTE_POSITION_BITS) | position);
                            ++packed;
                        }
                    }
                    for (uint32_t index = total; index < sortLength; ++index) {
                        sortWords.SetValue(index, 0U);
                        sortWords.SetValue(sortLength + index, 0U);
                    }
                    Sync<HardEvent::S_V>(HardEvent::S_V);
                }
                LIServiceVec::SortAll(workspace, input, sortLength);
                PipeBarrier<PIPE_V>();
                capturedOccurrences = routeCount <= LOCAL_MAX_ROUTES;
                unionCount = DeduplicateMergedMisses(
                    workspace, input, total, unionSources, occurrences,
                    capturedOccurrences, sourceMask);
            } else {
                const bool captureOccurrences =
                    routeCount <= LOCAL_MAX_ROUTES &&
                    UseCompactOccurrenceMap(total);
                uint32_t occurrenceCount = 0U;
                while (true) {
                    int32_t nextSource = INT32_MAX;
                    for (uint32_t route = 0U; route < routeCount; ++route) {
                        if (cursors[route] >= lengths[route]) {
                            continue;
                        }
                        const int32_t source = routeSources.GetValue(
                            routeOffsets[route] + cursors[route]);
                        if (source < nextSource) {
                            nextSource = source;
                        }
                    }
                    if (nextSource == INT32_MAX) {
                        break;
                    }
                    if (unionCount == CAPACITY) {
                        overflow = true;
                        break;
                    }
                    const uint32_t unionIndex = unionCount;
                    unionSources.SetValue(unionCount++, nextSource);
                    for (uint32_t route = 0U; route < routeCount; ++route) {
                        while (cursors[route] < lengths[route] &&
                               routeSources.GetValue(
                                   routeOffsets[route] + cursors[route]) ==
                                   nextSource) {
                            if (captureOccurrences) {
                                occurrences.SetValue(
                                    occurrenceCount++,
                                    (unionIndex << OCCURRENCE_UNION_SHIFT) |
                                        (route << ROUTE_POSITION_BITS) |
                                        cursors[route]);
                            }
                            ++cursors[route];
                        }
                    }
                }
                capturedOccurrences = captureOccurrences &&
                                      occurrenceCount == total;
            }
        }
        if (overflow) {
            ProcessGenericSteady(batch, queryStart, queryEnd);
            return;
        }

        const uint32_t updated = FindGeneralEvictSlotsAndUpdateCache(
            queryStart, routeCount, length, cacheCount,
            static_cast<uint32_t>(row), unionCount, lengths, input,
            accumulator, unionSources, unionDestinations);

        LocalTensor<int32_t> topkDestinationRows = routeSources;
        bool topkDestinationsPublished = false;
        if (capturedOccurrences) {
            // unionDestinations occupies workspace's upper half.  Q5-Q7
            // destination rows extend past route 3 and would overwrite that
            // union-slot list before it is published to missDestinationsGm.
            // Eviction has finished using input at this point, and pairInBuf
            // is statically large enough for all seven TopK rows, so publish
            // the occurrence expansion there instead.
            PrepareTopkRows(total, updated, occurrences, unionDestinations,
                            input);
            topkDestinationRows = input.ReinterpretCast<int32_t>();
        } else {
            // Stream each miss prefix through one reusable row so Q8-Q14
            // never require all destination rows to coexist in local UB.
            for (uint32_t route = 0U; route < routeCount; ++route) {
                if (lengths[route] == 0U) {
                    continue;
                }
                DataCopyPad(
                    routeSources,
                    topkSourcesGm[
                        static_cast<uint64_t>(queryStart + route) * TOPK],
                    AscendC::DataCopyExtParams{
                        1,
                        static_cast<uint32_t>(lengths[route] *
                                              sizeof(int32_t)),
                        0, 0, 0},
                    AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
                Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
                uint32_t unionCursor = 0U;
                for (uint32_t position = 0U;
                     position < lengths[route]; ++position) {
                    const int32_t source = routeSources.GetValue(position);
                    while (unionCursor < updated &&
                           unionSources.GetValue(unionCursor) < source) {
                        ++unionCursor;
                    }
                    routeSources.SetValue(
                        position,
                        unionCursor < updated &&
                                unionSources.GetValue(unionCursor) == source
                            ? unionDestinations.GetValue(unionCursor)
                            : -1);
                }
                Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(
                    topkDestinationsGm[
                        static_cast<uint64_t>(queryStart + route) * TOPK],
                    routeSources,
                    {1, static_cast<uint16_t>(lengths[route] *
                                               sizeof(int32_t)),
                     0, 0});
                Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
            }
            topkDestinationsPublished = true;
        }

        countLocal.SetValue(0U, static_cast<int32_t>(updated));
        for (uint32_t route = 0U; route < routeCount; ++route) {
            countLocal.SetValue(ROUTE_COUNT_STRIDE + route,
                                static_cast<int32_t>(lengths[route]));
        }
        Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        if (updated != 0U) {
            const uint64_t missBase =
                static_cast<uint64_t>(batch) * MISS_CAPACITY;
            const uint16_t unionBytes =
                static_cast<uint16_t>(updated * sizeof(int32_t));
            DataCopyPad(missSourcesGm[missBase], unionSources,
                        {1, unionBytes, 0, 0});
            DataCopyPad(missDestinationsGm[missBase], unionDestinations,
                        {1, unionBytes, 0, 0});
            for (uint32_t route = 0U;
                 !topkDestinationsPublished && route < routeCount; ++route) {
                if (lengths[route] == 0U) {
                    continue;
                }
                DataCopyPad(
                    topkDestinationsGm[
                        static_cast<uint64_t>(queryStart + route) * TOPK],
                    topkDestinationRows[route * TOPK],
                    {1,
                     static_cast<uint16_t>(lengths[route] *
                                           sizeof(int32_t)),
                     0, 0});
            }
        }
        DataCopyPad(countsGm[batch], countLocal,
                    {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        DataCopyPad(
            topkMissCountsGm[queryStart], countLocal[ROUTE_COUNT_STRIDE],
            {1, static_cast<uint16_t>(routeCount * sizeof(int32_t)), 0, 0});
        Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline uint32_t HashEvictScanSeed(uint32_t actual, uint32_t cacheRow)
    {
        uint32_t value = actual ^ ((cacheRow + 1U) * 0x9e3779b9U);
        value ^= value >> 16U;
        value *= 0x7feb352dU;
        value ^= value >> 15U;
        value *= 0x846ca68bU;
        value ^= value >> 16U;
        return value;
    }

    __aicore__ inline void SortEvictChunk(LocalTensor<float> dst,
                                          LocalTensor<float> key,
                                          LocalTensor<uint32_t> payload,
                                          LocalTensor<float> tmp)
    {
        Sort32(tmp, key, payload, EVICT_SORT_REPEATS);
        PipeBarrier<PIPE_V>();
        MrgSort4Info params;
        params.elementLengths[0] = 32U;
        params.elementLengths[1] = 32U;
        params.elementLengths[2] = 32U;
        params.elementLengths[3] = 32U;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;
        params.repeatTimes = 1;
        for (uint32_t group = 0U; group < 4U; ++group) {
            const uint32_t offset = group * 256U;
            MrgSortSrcList<float> sources;
            sources.src1 = tmp[offset];
            sources.src2 = tmp[offset + 64U];
            sources.src3 = tmp[offset + 128U];
            sources.src4 = tmp[offset + 192U];
            MrgSort<float>(dst[offset], sources, params);
        }
        PipeBarrier<PIPE_V>();
        params.elementLengths[0] = 128U;
        params.elementLengths[1] = 128U;
        params.elementLengths[2] = 128U;
        params.elementLengths[3] = 128U;
        MrgSortSrcList<float> sources;
        sources.src1 = dst;
        sources.src2 = dst[256U];
        sources.src3 = dst[512U];
        sources.src4 = dst[768U];
        MrgSort<float>(tmp, sources, params);
        PipeBarrier<PIPE_V>();
        DataCopy(dst, tmp, EVICT_PAIR_WORDS);
        PipeBarrier<PIPE_V>();
    }

    // Build one 512-token victim block.  A valid key is the negated sum of
    // the four route scores, so a descending sort places the lowest aggregate
    // score first.  Tokens in any route TopK and uncached tokens are masked.
    __aicore__ inline void BuildEvictCandidateChunk(
        uint32_t batch, uint32_t cacheRow, uint32_t start, uint32_t valid,
        const float thresholds[ROUTES], LocalTensor<float> chunkPair,
        LocalTensor<float> scratch, bool longSource)
    {
        LocalTensor<float> score0 = scratch;
        LocalTensor<float> score1 = scratch[EVICT_CHUNK];
        LocalTensor<float> score2 = scratch[EVICT_CHUNK * 2U];
        LocalTensor<float> score3 = scratch[EVICT_CHUNK * 3U];
        LocalTensor<float> key = scratch[EVICT_CHUNK * 4U];
        LocalTensor<float> temp = scratch[EVICT_CHUNK * 5U];
        LocalTensor<int32_t> slots = scratch[EVICT_CHUNK * 6U].ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> payload = scratch[EVICT_CHUNK * 7U].ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> invalidMask = scratch[EVICT_CHUNK * 8U].ReinterpretCast<uint8_t>();
        LocalTensor<float> invalidKey =
            scratch[EVICT_CHUNK * 8U + EVICT_MASK_WORK_FLOATS];
        LocalTensor<float> sortTmp = scratch[EVICT_CHUNK * 10U];

        const uint64_t requestRoute = static_cast<uint64_t>(batch) * ROUTES;
        DataCopyPad(score0, scoreScratchGm[requestRoute * scoreStride + start],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<float>{
                        true, 0, static_cast<uint8_t>((8U - valid % 8U) % 8U), 0.0F});
        DataCopyPad(score1, scoreScratchGm[(requestRoute + 1U) * scoreStride + start],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<float>{
                        true, 0, static_cast<uint8_t>((8U - valid % 8U) % 8U), 0.0F});
        DataCopyPad(score2, scoreScratchGm[(requestRoute + 2U) * scoreStride + start],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<float>{
                        true, 0, static_cast<uint8_t>((8U - valid % 8U) % 8U), 0.0F});
        DataCopyPad(score3, scoreScratchGm[(requestRoute + 3U) * scoreStride + start],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<float>{
                        true, 0, static_cast<uint8_t>((8U - valid % 8U) % 8U), 0.0F});
        DataCopyPad(slots,
                    cacheSlotsGm[static_cast<uint64_t>(cacheRow) * sourceCapacity + start],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(valid * sizeof(int32_t)), 0, 0, 0},
                    AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        Sync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        // Preserve the requested eviction ordering: lower aggregate score is
        // better, hence a descending sort over the negated four-route sum.
        Add(key, score0, score1, valid);
        PipeBarrier<PIPE_V>();
        Add(temp, score2, score3, valid);
        PipeBarrier<PIPE_V>();
        Add(key, key, temp, valid);
        PipeBarrier<PIPE_V>();
        Muls(key, key, -1.0F, valid);
        PipeBarrier<PIPE_V>();

        Duplicate(invalidKey, INVALID_EVICT_KEY, EVICT_CHUNK);
        PipeBarrier<PIPE_V>();

        // Do not use a signed vector comparison against INT32_MIN here.  On
        // the target this path must recognize both the standardized INT32_MIN
        // pool sentinel and the historical -1 sentinel.  Extracting the sign
        // bit produces exactly 0 or 1, so the following equality comparison
        // is scalar-safe and prevents cold sources from entering victim sort.
        LocalTensor<int32_t> slotSign = temp.ReinterpretCast<int32_t>();
        ShiftRight(slotSign.ReinterpretCast<uint32_t>(),
                   slots.ReinterpretCast<uint32_t>(), 31U, valid);
        PipeBarrier<PIPE_V>();
        CompareScalar(invalidMask, slotSign, 1, CMPMODE::EQ, valid);
        PipeBarrier<PIPE_V>();
        Select(key, invalidMask, invalidKey, key,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, valid);
        PipeBarrier<PIPE_V>();

        // Pack only after invalid cache entries have been excluded.  Their
        // payload bit pattern is irrelevant because their sort key is masked.
        // The payload is [slot:15 | source_low17].  Preserve the historical
        // direct source construction on the short path; long-source high
        // bits belong exclusively in TagLongEvictSource's sort-key tag.
        const uint32_t payloadStart = longSource
            ? (start & PACKED_SOURCE_MASK) : start;
        ArithProgression<int32_t>(
            payload.ReinterpretCast<int32_t>(),
            static_cast<int32_t>(payloadStart), 1,
            EVICT_CHUNK);
        PipeBarrier<PIPE_V>();
        if (longSource) {
            // Keep the mature [slot:15 | source_low17] payload and carry
            // only the missing source high bits in the sort-key tag.
            TagLongEvictSource(key, start, valid);
        }
        ShiftLeft(slots.ReinterpretCast<uint32_t>(),
                  slots.ReinterpretCast<uint32_t>(),
                  PACKED_SOURCE_BITS, valid);
        PipeBarrier<PIPE_V>();
        Add(payload.ReinterpretCast<int32_t>(),
            payload.ReinterpretCast<int32_t>(), slots, valid);
        PipeBarrier<PIPE_V>();

        // Build one protection margin instead of repeatedly selecting into
        // key.  margin >= 0 means the token belongs to at least one route's
        // TopK (ties included) and therefore cannot be evicted.
        Adds(score0, score0, -thresholds[0], valid);
        PipeBarrier<PIPE_V>();
        Adds(score1, score1, -thresholds[1], valid);
        PipeBarrier<PIPE_V>();
        Max(score0, score0, score1, valid);
        PipeBarrier<PIPE_V>();
        Adds(score2, score2, -thresholds[2], valid);
        PipeBarrier<PIPE_V>();
        Max(score0, score0, score2, valid);
        PipeBarrier<PIPE_V>();
        Adds(score3, score3, -thresholds[3], valid);
        PipeBarrier<PIPE_V>();
        Max(score0, score0, score3, valid);
        PipeBarrier<PIPE_V>();

        // AscendC's proven protection path compares the negated margin with
        // zero using LE: margin >= 0 iff -margin <= 0.  This protects every
        // token selected by at least one of the four TopK routes, including
        // threshold ties.
        Muls(score0, score0, -1.0F, valid);
        PipeBarrier<PIPE_V>();
        CompareScalar(invalidMask, score0, 0.0F, CMPMODE::LE, valid);
        PipeBarrier<PIPE_V>();
        Select(key, invalidMask, invalidKey, key,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, valid);
        PipeBarrier<PIPE_V>();
        if (valid < EVICT_CHUNK) {
            Duplicate(key[valid], INVALID_EVICT_KEY, EVICT_CHUNK - valid);
            PipeBarrier<PIPE_V>();
        }
        SortEvictChunk(chunkPair, key, payload, sortTmp);
    }

    // Runtime-route extension of the mature four-route victim builder.  Keep
    // the same aggregate-score ordering, threshold protection, slot codec and
    // 512-token SortEvictChunk implementation; only the number and GM origin
    // of score rows are generalized.
    __aicore__ inline void BuildGeneralEvictCandidateChunk(
        uint32_t queryStart, uint32_t routeCount, uint32_t cacheRow,
        uint32_t start, uint32_t valid,
        const float thresholds[MAX_ROUTES], LocalTensor<float> chunkPair,
        LocalTensor<float> scratch, bool longSource)
    {
        LocalTensor<float> key = scratch[EVICT_CHUNK * MAX_ROUTES];
        LocalTensor<float> temp = scratch[EVICT_CHUNK * (MAX_ROUTES + 1U)];
        LocalTensor<int32_t> slots =
            scratch[EVICT_CHUNK * (MAX_ROUTES + 2U)].ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> payload =
            scratch[EVICT_CHUNK * (MAX_ROUTES + 3U)].ReinterpretCast<uint32_t>();
        LocalTensor<uint8_t> invalidMask =
            scratch[EVICT_CHUNK * (MAX_ROUTES + 4U)].ReinterpretCast<uint8_t>();
        LocalTensor<float> invalidKey =
            scratch[EVICT_CHUNK * (MAX_ROUTES + 4U) +
                    EVICT_MASK_WORK_FLOATS];
        LocalTensor<float> sortTmp =
            scratch[EVICT_CHUNK * (MAX_ROUTES + 6U)];

        for (uint32_t route = 0U; route < routeCount; ++route) {
            LocalTensor<float> score = scratch[route * EVICT_CHUNK];
            DataCopyPad(
                score,
                scoreScratchGm[
                    static_cast<uint64_t>(queryStart + route) * scoreStride +
                    start],
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{
                    true, 0,
                    static_cast<uint8_t>((8U - valid % 8U) % 8U), 0.0F});
        }
        DataCopyPad(
            slots,
            cacheSlotsGm[
                static_cast<uint64_t>(cacheRow) * sourceCapacity + start],
            AscendC::DataCopyExtParams{
                1, static_cast<uint32_t>(valid * sizeof(int32_t)), 0, 0, 0},
            AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        Sync<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        Adds(key, scratch, 0.0F, valid);
        PipeBarrier<PIPE_V>();
        for (uint32_t route = 1U; route < routeCount; ++route) {
            Add(key, key, scratch[route * EVICT_CHUNK], valid);
            PipeBarrier<PIPE_V>();
        }
        Muls(key, key, -1.0F, valid);
        PipeBarrier<PIPE_V>();
        Duplicate(invalidKey, INVALID_EVICT_KEY, EVICT_CHUNK);
        PipeBarrier<PIPE_V>();

        // See the mature four-route builder above.  A sign-bit extraction is
        // required because direct signed CompareScalar(slot, 0, LT) does not
        // reliably mask the standardized INT32_MIN sentinel on this target.
        LocalTensor<int32_t> slotSign = temp.ReinterpretCast<int32_t>();
        ShiftRight(slotSign.ReinterpretCast<uint32_t>(),
                   slots.ReinterpretCast<uint32_t>(), 31U, valid);
        PipeBarrier<PIPE_V>();
        CompareScalar(invalidMask, slotSign, 1, CMPMODE::EQ, valid);
        PipeBarrier<PIPE_V>();
        Select(key, invalidMask, invalidKey, key,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, valid);
        PipeBarrier<PIPE_V>();

        // Reuse scores[0] as the running max threshold margin.  A nonnegative
        // margin means that at least one route protects this source.
        Adds(scratch, scratch, -thresholds[0], valid);
        PipeBarrier<PIPE_V>();
        for (uint32_t route = 1U; route < routeCount; ++route) {
            Adds(temp, scratch[route * EVICT_CHUNK], -thresholds[route],
                 valid);
            PipeBarrier<PIPE_V>();
            Max(scratch, scratch, temp, valid);
            PipeBarrier<PIPE_V>();
        }

        // Preserve direct source construction on the short path.  Keep long
        // source high bits out of the packed slot field, matching Q=4.
        const uint32_t payloadStart = longSource
            ? (start & PACKED_SOURCE_MASK) : start;
        ArithProgression<int32_t>(
            payload.ReinterpretCast<int32_t>(),
            static_cast<int32_t>(payloadStart), 1,
            EVICT_CHUNK);
        PipeBarrier<PIPE_V>();
        if (longSource) {
            TagLongEvictSource(key, start, valid);
        }
        ShiftLeft(slots.ReinterpretCast<uint32_t>(),
                  slots.ReinterpretCast<uint32_t>(),
                  PACKED_SOURCE_BITS, valid);
        PipeBarrier<PIPE_V>();
        Add(payload.ReinterpretCast<int32_t>(),
            payload.ReinterpretCast<int32_t>(), slots, valid);
        PipeBarrier<PIPE_V>();
        Muls(scratch, scratch, -1.0F, valid);
        PipeBarrier<PIPE_V>();
        CompareScalar(invalidMask, scratch, 0.0F, CMPMODE::LE, valid);
        PipeBarrier<PIPE_V>();
        Select(key, invalidMask, invalidKey, key,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, valid);
        PipeBarrier<PIPE_V>();
        if (valid < EVICT_CHUNK) {
            Duplicate(key[valid], INVALID_EVICT_KEY, EVICT_CHUNK - valid);
            PipeBarrier<PIPE_V>();
        }
        SortEvictChunk(chunkPair, key, payload, sortTmp);
    }

    // This scalar prefix walk is reserved for correctness/fallback paths.  The
    // steady union-miss<=512 path checks its exact kth key in O(1) instead.
    __aicore__ inline uint32_t ValidCandidatePrefix(LocalTensor<float> pairs,
                                                    uint32_t capacity)
    {
        uint32_t count = 0U;
        while (count < capacity &&
               pairs.GetValue(count * 2U) > EVICT_STOP_KEY) {
            ++count;
        }
        return count;
    }

    __aicore__ inline uint32_t ScalarDeduplicateKeys(
        LocalTensor<uint32_t> keyBits, uint32_t total,
        LocalTensor<int32_t> unionSources, uint32_t sourceMask)
    {
        uint32_t count = 0U;
        int32_t last = -1;
        for (uint32_t index = 0U; index < total; ++index) {
            const uint32_t key = keyBits.GetValue(index);
            const int32_t sourceId = static_cast<int32_t>(
                sourceMask - (key - MISS_KEY_BASE_BITS));
            if (sourceId != last) {
                unionSources.SetValue(count++, sourceId);
                last = sourceId;
            }
        }
        return count;
    }

    // MrgSort leaves the four miss prefixes as one source-key-sorted pair
    // list.  Extract the keys, compare each key with its predecessor, and
    // compact only first occurrences.  This replaces the typical 4*N scalar
    // walk with vector Gather/Compare/GatherMask operations.  The input and
    // merged tensors are scratch after the key extraction, so this path adds
    // neither UB nor GM workspace.
    __aicore__ inline uint32_t DeduplicateMergedMisses(
        LocalTensor<float> merged, LocalTensor<float> scratch,
        uint32_t total, LocalTensor<int32_t> unionSources,
        LocalTensor<uint32_t> occurrences, bool captureOccurrences,
        uint32_t sourceMask)
    {
        // The LI producer stores (route, final miss-prefix position) in the
        // pair payload.  Extract it before merged is reused for predecessor
        // keys and compare-mask work below.
        if (captureOccurrences) {
            LIServiceVec::ExtractIndex(
                occurrences, merged.ReinterpretCast<uint32_t>(), total);
        }
        LocalTensor<uint32_t> keyBits =
            scratch.ReinterpretCast<uint32_t>();
        LIServiceVec::ExtractScoreBits(
            keyBits, merged.ReinterpretCast<uint32_t>(), total);
        if (total == 1U) {
            Sync<HardEvent::V_S>(HardEvent::V_S);
            unionSources.SetValue(
                0U, static_cast<int32_t>(
                        sourceMask -
                        (keyBits.GetValue(0U) - MISS_KEY_BASE_BITS)));
            return 1U;
        }

        // Compare/GatherMask consume B32 vectors in 64-lane repeats.  Pad the
        // last repeat with equal zero key/predecessor pairs, so lanes outside
        // the merged miss prefix can never be marked unique.
        const uint32_t alignedTotal =
            (total + LIKernel::B32_VEC_ELM_NUM - 1U) /
            LIKernel::B32_VEC_ELM_NUM * LIKernel::B32_VEC_ELM_NUM;
        const uint32_t tail = alignedTotal - total;
        if (tail > 0U) {
            // `total` is an arbitrary sum of route miss counts and therefore
            // is not necessarily 32-byte aligned.  A vector Duplicate
            // starting at keyBits[total] faults on C220 for such workloads.
            Sync<HardEvent::V_S>(HardEvent::V_S);
            for (uint32_t index = total; index < alignedTotal; ++index) {
                keyBits.SetValue(index, 0U);
            }
            Sync<HardEvent::S_V>(HardEvent::S_V);
        }

        // Gather a one-element-shifted predecessor row.  Gather offsets are
        // byte offsets: [0, 0, 4, 8, ...].
        LocalTensor<int32_t> predecessorOffsets =
            scratch.ReinterpretCast<int32_t>()[CAPACITY];
        ArithProgression(predecessorOffsets, 0, 1, total);
        PipeBarrier<PIPE_V>();
        Adds(predecessorOffsets, predecessorOffsets, -1, total);
        PipeBarrier<PIPE_V>();
        Relu(predecessorOffsets, predecessorOffsets, total);
        PipeBarrier<PIPE_V>();
        Muls(predecessorOffsets, predecessorOffsets,
             static_cast<int32_t>(sizeof(uint32_t)), total);
        PipeBarrier<PIPE_V>();

        LocalTensor<uint32_t> predecessorBits =
            merged.ReinterpretCast<uint32_t>();
        // Reinterpret as float to use the same proven C220 Gather overload as
        // the existing kernels; the 32-bit key payload is copied bit-for-bit.
        Gather(predecessorBits.ReinterpretCast<float>(),
               keyBits.ReinterpretCast<float>(),
               predecessorOffsets.ReinterpretCast<uint32_t>(), 0U, total);
        PipeBarrier<PIPE_V>();

        // Make lane 0 distinct so it is always retained.  Writing that one
        // sentinel directly is both cheaper and less error-prone than
        // constructing a vector [1, 0, 0, ...].  In particular, the latter
        // exposed a lane-1 dependency hazard on C220 and could turn an equal
        // first key pair into two unique keys.
        Sync<HardEvent::V_S>(HardEvent::V_S);
        predecessorBits.SetValue(0U, predecessorBits.GetValue(0U) + 1U);
        if (tail > 0U) {
            for (uint32_t index = total; index < alignedTotal; ++index) {
                predecessorBits.SetValue(index, 0U);
            }
        }
        Sync<HardEvent::S_V>(HardEvent::S_V);

        LocalTensor<uint8_t> uniqueMask =
            merged[CAPACITY].ReinterpretCast<uint8_t>();
        Compare(uniqueMask, keyBits.ReinterpretCast<float>(),
                predecessorBits.ReinterpretCast<float>(), CMPMODE::NE,
                alignedTotal);
        PipeBarrier<PIPE_V>();

        GatherMaskParams compactParams;
        compactParams.repeatTimes = 1;
        compactParams.src0BlockStride = 1;
        compactParams.src0RepeatStride = LIKernel::B32_VEC_REPEAT_STRIDE;
        compactParams.src1RepeatStride = LIKernel::B32_VEC_REPEAT_STRIDE;
        uint64_t uniqueCount = 0U;
        GatherMask(unionSources.ReinterpretCast<uint32_t>(), keyBits,
                   uniqueMask.ReinterpretCast<uint32_t>(), true, alignedTotal,
                   compactParams, uniqueCount);
        PipeBarrier<PIPE_V>();
        Sync<HardEvent::V_S>(HardEvent::V_S);

        const uint32_t count = static_cast<uint32_t>(uniqueCount);
        if (count == 0U || count > total) {
            // Defensive fallback for an unsupported/abnormal GatherMask
            // result.  It is not part of the steady target path.
            const uint32_t fallbackCount =
                ScalarDeduplicateKeys(keyBits, total, unionSources,
                                      sourceMask);
            if (captureOccurrences) {
                uint32_t unionIndex = 0U;
                uint32_t previousKey = keyBits.GetValue(0U);
                for (uint32_t index = 0U; index < total; ++index) {
                    const uint32_t key = keyBits.GetValue(index);
                    if (index != 0U && key != previousKey) {
                        ++unionIndex;
                        previousKey = key;
                    }
                    occurrences.SetValue(
                        index, (unionIndex << OCCURRENCE_UNION_SHIFT) |
                                   occurrences.GetValue(index));
                }
            }
            return fallbackCount;
        }
        Muls(unionSources, unionSources, -1, count);
        PipeBarrier<PIPE_V>();
        Adds(unionSources, unionSources, MissKeyDecodeBase(sourceMask), count);
        Sync<HardEvent::V_S>(HardEvent::V_S);

        // Equal source keys are adjacent in the merged list.  Attach their
        // compact union index to the producer's route/position payload once;
        // after eviction this is a direct unionDestinations gather map.
        if (captureOccurrences) {
            uint32_t unionIndex = 0U;
            uint32_t previousKey = keyBits.GetValue(0U);
            for (uint32_t index = 0U; index < total; ++index) {
                const uint32_t key = keyBits.GetValue(index);
                if (index != 0U && key != previousKey) {
                    ++unionIndex;
                    previousKey = key;
                }
                occurrences.SetValue(
                    index, (unionIndex << OCCURRENCE_UNION_SHIFT) |
                               occurrences.GetValue(index));
            }
        }
        return count;
    }

    __aicore__ inline uint32_t ApplyCandidateUpdates(
        uint32_t actual, uint32_t cacheRow,
        uint32_t cacheTokenCount, uint32_t outputOffset,
        LocalTensor<float> pairs, uint32_t count,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations)
    {
        LocalTensor<uint32_t> pairBits = pairs.ReinterpretCast<uint32_t>();
        const uint64_t cacheBase =
            static_cast<uint64_t>(cacheRow) * sourceCapacity;
        uint32_t updated = 0U;
        for (uint32_t index = 0U; index < count; ++index) {
            const uint32_t payload = pairBits.GetValue(index * 2U + 1U);
            const uint32_t victimSource = payload & PACKED_SOURCE_MASK;
            const int32_t victimSlot =
                static_cast<int32_t>(payload >> PACKED_SOURCE_BITS);
            const int32_t missSourceValue =
                unionSources.GetValue(outputOffset + index);
            if (victimSource >= actual || missSourceValue < 0 ||
                static_cast<uint32_t>(missSourceValue) >= actual ||
                victimSlot < 0 ||
                static_cast<uint32_t>(victimSlot) >= cacheTokenCount) {
                break;
            }
            unionDestinations.SetValue(outputOffset + updated, victimSlot);
            cacheSlotsGm.SetValue(cacheBase + victimSource,
                                  INVALID_CACHE_SLOT);
            cacheSlotsGm.SetValue(cacheBase +
                                      static_cast<uint32_t>(missSourceValue),
                                  victimSlot);
            ++updated;
        }
        return updated;
    }

    // The mature payload remains [slot:15 | source_low17] for long sources.
    // The sort-key tag carries source_high4, allowing the final candidate to
    // recover both its complete source ID and the slot captured by the same
    // MTE2 cache-row load that qualified it as an eviction candidate.  This
    // deliberately avoids a second scalar GM pool lookup after sorting.
    // Validation is two-phase: no cache row update occurs until every selected
    // victim and incoming source has a legal mapping.
    __aicore__ inline uint32_t ApplyLongCandidateUpdates(
        uint32_t actual, uint32_t cacheRow,
        uint32_t cacheTokenCount, uint32_t outputOffset,
        LocalTensor<float> pairs, uint32_t count,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations)
    {
        LocalTensor<uint32_t> pairBits = pairs.ReinterpretCast<uint32_t>();
        const uint64_t cacheBase =
            static_cast<uint64_t>(cacheRow) * sourceCapacity;

        // Phase 1: resolve and validate all slots without touching the pool.
        // unionDestinations is the final output buffer as well as temporary
        // storage for this bounded (<= CAPACITY) selected-victim list.
        for (uint32_t index = 0U; index < count; ++index) {
            const uint32_t keyBits = pairBits.GetValue(index * 2U);
            const uint32_t payload = pairBits.GetValue(index * 2U + 1U);
            const uint32_t victimSource =
                DecodeLongEvictSource(keyBits, payload);
            const int32_t missSourceValue =
                unionSources.GetValue(outputOffset + index);
            if (victimSource >= actual || missSourceValue < 0 ||
                static_cast<uint32_t>(missSourceValue) >= actual) {
                return 0U;
            }
            const int32_t victimSlot = static_cast<int32_t>(
                payload >> PACKED_SOURCE_BITS);
            if (victimSlot < 0 ||
                static_cast<uint32_t>(victimSlot) >= cacheTokenCount) {
                return 0U;
            }
            unionDestinations.SetValue(outputOffset + index, victimSlot);
        }

        // Phase 2: all mappings are known to be valid, so update atomically
        // with respect to validation failures from the first phase.
        for (uint32_t index = 0U; index < count; ++index) {
            const uint32_t victimSource = DecodeLongEvictSource(
                pairBits.GetValue(index * 2U),
                pairBits.GetValue(index * 2U + 1U));
            const int32_t missSourceValue =
                unionSources.GetValue(outputOffset + index);
            const int32_t victimSlot =
                unionDestinations.GetValue(outputOffset + index);
            cacheSlotsGm.SetValue(cacheBase + victimSource,
                                  INVALID_CACHE_SLOT);
            cacheSlotsGm.SetValue(
                cacheBase + static_cast<uint32_t>(missSourceValue),
                victimSlot);
        }
        return count;
    }

    __aicore__ inline bool SegmentContains(LocalTensor<int32_t> topkSources,
                                            uint32_t route, uint32_t begin,
                                            uint32_t end, uint32_t source)
    {
        const uint32_t segmentEnd = end;
        LocalTensor<int32_t> row = topkSources[route * TOPK];
        while (begin < end) {
            const uint32_t middle = (begin + end) >> 1U;
            const uint32_t current =
                static_cast<uint32_t>(row.GetValue(middle));
            if (current < source) {
                begin = middle + 1U;
            } else {
                end = middle;
            }
        }
        return begin < segmentEnd &&
               static_cast<uint32_t>(row.GetValue(begin)) == source;
    }

    __aicore__ inline bool IsProtectedTopk(LocalTensor<int32_t> topkSources,
                                           uint32_t source,
                                           const uint32_t lengths[ROUTES])
    {
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            if (SegmentContains(topkSources, route, 0U, lengths[route], source) ||
                SegmentContains(topkSources, route, lengths[route], TOPK, source)) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline bool IsProtectedTopkGeneral(
        LocalTensor<int32_t> topkSources, uint32_t source,
        const uint32_t lengths[MAX_ROUTES], uint32_t routeCount)
    {
        for (uint32_t route = 0U; route < routeCount; ++route) {
            if (SegmentContains(topkSources, route, 0U, lengths[route],
                                source) ||
                SegmentContains(topkSources, route, lengths[route], TOPK,
                                source)) {
                return true;
            }
        }
        return false;
    }

    // Threshold ties are rare on the target BF16 workload.  The vector fast
    // path masks every tie to guarantee that no selected TopK token can be
    // evicted.  If that conservative rule leaves too few victims, resolve
    // only tied tokens against the exact four sorted TopK rows.
    __aicore__ inline uint32_t AppendThresholdTieFallback(
        uint32_t batch, uint32_t actual, uint32_t cacheRow,
        uint32_t cacheTokenCount, const uint32_t lengths[ROUTES],
        const float thresholds[ROUTES], uint32_t written, uint32_t required,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<float> fallbackStorage)
    {
        if (written >= required) {
            return written;
        }
        LocalTensor<int32_t> allTopkSources =
            fallbackStorage.ReinterpretCast<int32_t>();
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            const uint64_t rowOffset =
                (static_cast<uint64_t>(batch) * ROUTES + route) * TOPK;
            DataCopyPad(
                allTopkSources[route * TOPK], topkSourcesGm[rowOffset],
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(TOPK * sizeof(int32_t)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
        Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        const uint64_t cacheBase = static_cast<uint64_t>(cacheRow) * sourceCapacity;
        const uint64_t requestRoute = static_cast<uint64_t>(batch) * ROUTES;
        for (uint32_t source = 0U; source < actual && written < required; ++source) {
            const int32_t slot = cacheSlotsGm.GetValue(cacheBase + source);
            if (slot < 0 || static_cast<uint32_t>(slot) >= cacheTokenCount) {
                continue;
            }
            bool above = false;
            bool tied = false;
            for (uint32_t route = 0U; route < ROUTES; ++route) {
                const float score = scoreScratchGm.GetValue(
                    (requestRoute + route) * scoreStride + source);
                if (score > thresholds[route]) {
                    above = true;
                    break;
                }
                tied = tied || score == thresholds[route];
            }
            if (above || !tied ||
                IsProtectedTopk(allTopkSources, source, lengths)) {
                continue;
            }
            const int32_t missSource = unionSources.GetValue(written);
            if (missSource < 0 || static_cast<uint32_t>(missSource) >= actual) {
                continue;
            }
            cacheSlotsGm.SetValue(cacheBase + source, INVALID_CACHE_SLOT);
            cacheSlotsGm.SetValue(cacheBase +
                                      static_cast<uint32_t>(missSource),
                                  slot);
            unionDestinations.SetValue(written, slot);
            ++written;
        }
        return written;
    }

    // Runtime-route counterpart of AppendThresholdTieFallback.  This is only
    // entered when conservative threshold masking leaves too few candidates;
    // the common path remains entirely vectorized.
    __aicore__ inline uint32_t AppendGeneralThresholdTieFallback(
        uint32_t queryStart, uint32_t routeCount, uint32_t actual,
        uint32_t cacheRow, uint32_t cacheTokenCount,
        const uint32_t lengths[MAX_ROUTES],
        const float thresholds[MAX_ROUTES], uint32_t written,
        uint32_t required, LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<float> fallbackStorage)
    {
        if (written >= required) {
            return written;
        }
        const bool useLocalTopk = routeCount <= LOCAL_MAX_ROUTES;
        LocalTensor<int32_t> allTopkSources =
            fallbackStorage.ReinterpretCast<int32_t>();
        if (useLocalTopk) {
            for (uint32_t route = 0U; route < routeCount; ++route) {
                const uint64_t rowOffset =
                    static_cast<uint64_t>(queryStart + route) * TOPK;
                DataCopyPad(
                    allTopkSources[route * TOPK], topkSourcesGm[rowOffset],
                    AscendC::DataCopyExtParams{
                        1, static_cast<uint32_t>(TOPK * sizeof(int32_t)),
                        0, 0, 0},
                    AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
            }
            Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        }
        const uint64_t cacheBase =
            static_cast<uint64_t>(cacheRow) * sourceCapacity;
        for (uint32_t source = 0U;
             source < actual && written < required; ++source) {
            const int32_t slot = cacheSlotsGm.GetValue(cacheBase + source);
            if (slot < 0 || static_cast<uint32_t>(slot) >= cacheTokenCount) {
                continue;
            }
            bool above = false;
            bool tied = false;
            for (uint32_t route = 0U; route < routeCount; ++route) {
                const float score = scoreScratchGm.GetValue(
                    static_cast<uint64_t>(queryStart + route) * scoreStride +
                    source);
                if (score > thresholds[route]) {
                    above = true;
                    break;
                }
                tied = tied || score == thresholds[route];
            }
            const bool protectedTopk = useLocalTopk
                ? IsProtectedTopkGeneral(allTopkSources, source, lengths,
                                         routeCount)
                : TopkContains(queryStart, queryStart + routeCount,
                               static_cast<int32_t>(source));
            if (above || !tied || protectedTopk) {
                continue;
            }
            const int32_t missSource = unionSources.GetValue(written);
            if (missSource < 0 ||
                static_cast<uint32_t>(missSource) >= actual) {
                continue;
            }
            cacheSlotsGm.SetValue(cacheBase + source, INVALID_CACHE_SLOT);
            cacheSlotsGm.SetValue(
                cacheBase + static_cast<uint32_t>(missSource), slot);
            unionDestinations.SetValue(written, slot);
            ++written;
        }
        return written;
    }

    __aicore__ inline uint32_t FindGeneralEvictSlotsAndUpdateCache(
        uint32_t queryStart, uint32_t routeCount, uint32_t actual,
        uint32_t cacheTokenCount, uint32_t cacheRow, uint32_t count,
        const uint32_t lengths[MAX_ROUTES],
        LocalTensor<float> input, LocalTensor<float> accumulator,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations)
    {
        if (count == 0U) {
            return 0U;
        }
        const bool longSource = actual > (1U << PACKED_SOURCE_BITS);
        const uint32_t chunks =
            (actual + EVICT_CHUNK - 1U) / EVICT_CHUNK;
        if (chunks == 0U) {
            return 0U;
        }
        const uint32_t startChunk =
            HashEvictScanSeed(actual, cacheRow) % chunks;
        LocalTensor<float> chunkPair = input;
        LocalTensor<float> scratch = input[EVICT_PAIR_WORDS];
        LocalTensor<float> mergeTmp =
            input[EVICT_PAIR_WORDS + GENERAL_EVICT_SCRATCH_FLOATS];

        LocalTensor<float> thresholdLocal = mergeTmp;
        for (uint32_t route = 0U; route < routeCount; ++route) {
            DataCopyPad(
                thresholdLocal[route * 8U],
                thresholdScratchGm[
                    static_cast<uint64_t>(queryStart + route) *
                    THRESHOLD_STRIDE],
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{true, 0, 7U, 0.0F});
        }
        Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        float thresholds[MAX_ROUTES] = {0.0F};
        for (uint32_t route = 0U; route < routeCount; ++route) {
            thresholds[route] = thresholdLocal.GetValue(route * 8U);
        }

        if (count <= EVICT_CHUNK) {
            Duplicate(accumulator, INVALID_EVICT_KEY, EVICT_PAIR_WORDS);
            PipeBarrier<PIPE_V>();
            bool found = false;
            for (uint32_t scan = 0U; scan < chunks; ++scan) {
                const uint32_t chunk = (startChunk + scan) % chunks;
                const uint32_t start = chunk * EVICT_CHUNK;
                const uint32_t valid = start + EVICT_CHUNK > actual
                                           ? actual - start
                                           : EVICT_CHUNK;
                BuildGeneralEvictCandidateChunk(
                    queryStart, routeCount, cacheRow, start, valid,
                    thresholds, chunkPair, scratch, longSource);
                LIServiceVec::MergeSort(accumulator, EVICT_CHUNK, chunkPair,
                                        EVICT_CHUNK, mergeTmp);
                Sync<HardEvent::V_S>(HardEvent::V_S);
                if (accumulator.GetValue((count - 1U) * 2U) >
                    EVICT_STOP_KEY) {
                    found = true;
                    break;
                }
            }
            const uint32_t vectorCount = found
                                             ? count
                                             : ValidCandidatePrefix(
                                                   accumulator, count);
            const uint32_t updated = longSource
                ? ApplyLongCandidateUpdates(
                    actual, cacheRow, cacheTokenCount, 0U, accumulator,
                    vectorCount, unionSources, unionDestinations)
                : ApplyCandidateUpdates(
                    actual, cacheRow, cacheTokenCount, 0U, accumulator,
                    vectorCount, unionSources, unionDestinations);
            return AppendGeneralThresholdTieFallback(
                queryStart, routeCount, actual, cacheRow, cacheTokenCount,
                lengths, thresholds, updated, count, unionSources,
                unionDestinations, input);
        }

        uint32_t written = 0U;
        for (uint32_t scan = 0U; scan < chunks && written < count; ++scan) {
            const uint32_t chunk = (startChunk + scan) % chunks;
            const uint32_t start = chunk * EVICT_CHUNK;
            const uint32_t valid = start + EVICT_CHUNK > actual
                                       ? actual - start
                                       : EVICT_CHUNK;
            BuildGeneralEvictCandidateChunk(
                queryStart, routeCount, cacheRow, start, valid, thresholds,
                chunkPair, scratch, longSource);
            Sync<HardEvent::V_S>(HardEvent::V_S);
            const uint32_t remaining = count - written;
            const uint32_t capacity = remaining < EVICT_CHUNK
                                          ? remaining
                                          : EVICT_CHUNK;
            const uint32_t take =
                ValidCandidatePrefix(chunkPair, capacity);
            const uint32_t updated = longSource
                ? ApplyLongCandidateUpdates(
                    actual, cacheRow, cacheTokenCount, written, chunkPair,
                    take, unionSources, unionDestinations)
                : ApplyCandidateUpdates(
                    actual, cacheRow, cacheTokenCount, written, chunkPair,
                    take, unionSources, unionDestinations);
            written += updated;
        }
        return AppendGeneralThresholdTieFallback(
            queryStart, routeCount, actual, cacheRow, cacheTokenCount,
            lengths, thresholds, written, count, unionSources,
            unionDestinations, input);
    }

    __aicore__ inline uint32_t FindEvictSlotsAndUpdateCache(
        uint32_t batch, uint32_t actual, uint32_t cacheTokenCount,
        uint32_t cacheRow, uint32_t count,
        const uint32_t lengths[ROUTES],
        LocalTensor<float> input, LocalTensor<float> accumulator,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations)
    {
        if (count == 0U) {
            return 0U;
        }
        const bool longSource = actual > (1U << PACKED_SOURCE_BITS);
        const uint32_t chunks = (actual + EVICT_CHUNK - 1U) / EVICT_CHUNK;
        if (chunks == 0U) {
            return 0U;
        }
        const uint32_t startChunk = HashEvictScanSeed(actual, cacheRow) % chunks;
        LocalTensor<float> chunkPair = input;
        LocalTensor<float> scratch = input[EVICT_PAIR_WORDS];
        // BuildEvictCandidateChunk uses 6144 floats from scratch.  Keep the
        // two-list merge output disjoint from both the chunk and its scratch.
        LocalTensor<float> mergeTmp = input[EVICT_PAIR_WORDS + EVICT_SCRATCH_FLOATS];

        // Thresholds are produced by whichever AIV owns each final TopK row,
        // while this request's eviction scan always runs on one even AIV.
        // Fetch the four strided values through MTE2 after the kernel-wide
        // barrier.  This avoids stale scalar-cache lines across invocations
        // and makes the producer-MTE3 -> SyncAll -> consumer-MTE2 dependency
        // explicit.
        LocalTensor<float> thresholdLocal = mergeTmp;
        const uint64_t requestRoute = static_cast<uint64_t>(batch) * ROUTES;
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            DataCopyPad(
                thresholdLocal[route * 8U],
                thresholdScratchGm[(requestRoute + route) * THRESHOLD_STRIDE],
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<float>{true, 0, 7U, 0.0F});
        }
        Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        float thresholds[ROUTES];
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            thresholds[route] = thresholdLocal.GetValue(route * 8U);
        }

        if (count <= EVICT_CHUNK) {
            Duplicate(accumulator, INVALID_EVICT_KEY, EVICT_PAIR_WORDS);
            PipeBarrier<PIPE_V>();
            bool found = false;
            for (uint32_t scan = 0U; scan < chunks; ++scan) {
                const uint32_t chunk = (startChunk + scan) % chunks;
                const uint32_t start = chunk * EVICT_CHUNK;
                const uint32_t valid = start + EVICT_CHUNK > actual
                                           ? actual - start
                                           : EVICT_CHUNK;
                BuildEvictCandidateChunk(batch, cacheRow, start, valid,
                                         thresholds, chunkPair, scratch,
                                         longSource);
                LIServiceVec::MergeSort(accumulator, EVICT_CHUNK, chunkPair,
                                        EVICT_CHUNK, mergeTmp);
                Sync<HardEvent::V_S>(HardEvent::V_S);
                if (accumulator.GetValue((count - 1U) * 2U) >
                    EVICT_STOP_KEY) {
                    found = true;
                    break;
                }
            }
            const uint32_t vectorCount = found
                                             ? count
                                             : ValidCandidatePrefix(accumulator, count);
            const uint32_t updated = longSource
                ? ApplyLongCandidateUpdates(
                    actual, cacheRow, cacheTokenCount, 0U, accumulator,
                    vectorCount, unionSources, unionDestinations)
                : ApplyCandidateUpdates(
                    actual, cacheRow, cacheTokenCount, 0U, accumulator,
                    vectorCount, unionSources, unionDestinations);
            const uint32_t finalCount = AppendThresholdTieFallback(
                batch, actual, cacheRow, cacheTokenCount, lengths, thresholds,
                updated, count, unionSources, unionDestinations, input);
            return finalCount;
        }

        // Large union misses are correctness boundaries rather than the
        // steady-decode target.  Keep each 512-token block vectorized and
        // sorted, and append its legal unique slots to the local union map.
        // Source chunks are disjoint and a valid cache row owns each slot once.
        uint32_t written = 0U;
        for (uint32_t scan = 0U; scan < chunks && written < count; ++scan) {
            const uint32_t chunk = (startChunk + scan) % chunks;
            const uint32_t start = chunk * EVICT_CHUNK;
            const uint32_t valid = start + EVICT_CHUNK > actual
                                       ? actual - start
                                       : EVICT_CHUNK;
            BuildEvictCandidateChunk(batch, cacheRow, start, valid,
                                     thresholds, chunkPair, scratch,
                                     longSource);
            Sync<HardEvent::V_S>(HardEvent::V_S);
            const uint32_t remaining = count - written;
            const uint32_t capacity = remaining < EVICT_CHUNK
                                          ? remaining
                                          : EVICT_CHUNK;
            const uint32_t take = ValidCandidatePrefix(chunkPair, capacity);
            const uint32_t updated = longSource
                ? ApplyLongCandidateUpdates(
                    actual, cacheRow, cacheTokenCount, written, chunkPair,
                    take, unionSources, unionDestinations)
                : ApplyCandidateUpdates(
                    actual, cacheRow, cacheTokenCount, written, chunkPair,
                    take, unionSources, unionDestinations);
            if (updated != 0U) {
                written += updated;
            }
        }
        const uint32_t finalCount = AppendThresholdTieFallback(
            batch, actual, cacheRow, cacheTokenCount, lengths, thresholds,
            written, count, unionSources, unionDestinations, input);
        return finalCount;
    }

    // The LI owner tagged every merged miss occurrence with its final route
    // and miss-prefix position.  Distribute assigned union slots through that
    // compact map, avoiding four topk_src_ids GM reads and four source joins.
    __aicore__ inline void PrepareTopkRows(
        uint32_t total, uint32_t unionCount,
        LocalTensor<uint32_t> occurrences,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<float> destinationStorage)
    {
        if (unionCount == 0U) {
            return;
        }
        LocalTensor<int32_t> allDestinations =
            destinationStorage.ReinterpretCast<int32_t>();
        for (uint32_t index = 0U; index < total; ++index) {
            const uint32_t occurrence = occurrences.GetValue(index);
            const uint32_t unionIndex =
                occurrence >> OCCURRENCE_UNION_SHIFT;
            const uint32_t route =
                (occurrence >> ROUTE_POSITION_BITS) & OCCURRENCE_ROUTE_MASK;
            const uint32_t position = occurrence & ROUTE_POSITION_MASK;
            const int32_t destination = unionIndex < unionCount
                                            ? unionDestinations.GetValue(unionIndex)
                                            : -1;
            allDestinations.SetValue(route * TOPK + position, destination);
        }
    }

    __aicore__ inline void PrepareTopkRowsFallback(
        uint32_t batch, const uint32_t lengths[ROUTES], uint32_t unionCount,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<float> sourceStorage,
        LocalTensor<float> destinationStorage)
    {
        LocalTensor<int32_t> allSources =
            sourceStorage.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> allDestinations =
            destinationStorage.ReinterpretCast<int32_t>();
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            if (lengths[route] == 0U) {
                continue;
            }
            const uint64_t rowOffset =
                (static_cast<uint64_t>(batch) * ROUTES + route) * TOPK;
            DataCopyPad(
                allSources[route * TOPK], topkSourcesGm[rowOffset],
                AscendC::DataCopyExtParams{
                    1, static_cast<uint32_t>(lengths[route] * sizeof(int32_t)),
                    0, 0, 0},
                AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
        }
        Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            LocalTensor<int32_t> rowSources = allSources[route * TOPK];
            LocalTensor<int32_t> rowDestinations =
                allDestinations[route * TOPK];
            uint32_t unionCursor = 0U;
            for (uint32_t miss = 0U; miss < lengths[route]; ++miss) {
                const int32_t source = rowSources.GetValue(miss);
                while (unionCursor < unionCount &&
                       unionSources.GetValue(unionCursor) < source) {
                    ++unionCursor;
                }
                const int32_t destination =
                    unionCursor < unionCount &&
                            unionSources.GetValue(unionCursor) == source
                        ? unionDestinations.GetValue(unionCursor)
                        : -1;
                rowDestinations.SetValue(miss, destination);
            }
        }
    }

    // All caller-visible union/update outputs are independent until the
    // request-local computation is complete.  Publish them under one MTE3
    // fence instead of serializing miss sources, destinations, count, and
    // four TopK prefixes separately.
    __aicore__ inline void PublishFinalOutputs(
        uint32_t batch, const uint32_t lengths[ROUTES], uint32_t count,
        LocalTensor<int32_t> countLocal,
        LocalTensor<int32_t> unionSources,
        LocalTensor<int32_t> unionDestinations,
        LocalTensor<float> topkDestinationStorage)
    {
        countLocal.SetValue(0, static_cast<int32_t>(count));
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            countLocal.SetValue(ROUTE_COUNT_STRIDE + route,
                                static_cast<int32_t>(lengths[route]));
        }
        Sync<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        if (count != 0U) {
            const uint64_t unionOffset =
                static_cast<uint64_t>(batch) * MISS_CAPACITY;
            const uint16_t unionBytes =
                static_cast<uint16_t>(count * sizeof(int32_t));
            DataCopyPad(missSourcesGm[unionOffset], unionSources,
                        {1, unionBytes, 0, 0});
            DataCopyPad(missDestinationsGm[unionOffset], unionDestinations,
                        {1, unionBytes, 0, 0});

            LocalTensor<int32_t> allDestinations =
                topkDestinationStorage.ReinterpretCast<int32_t>();
            for (uint32_t route = 0U; route < ROUTES; ++route) {
                if (lengths[route] == 0U) {
                    continue;
                }
                const uint64_t rowOffset =
                    (static_cast<uint64_t>(batch) * ROUTES + route) * TOPK;
                DataCopyPad(
                    topkDestinationsGm[rowOffset],
                    allDestinations[route * TOPK],
                    {1, static_cast<uint16_t>(lengths[route] * sizeof(int32_t)),
                     0, 0});
            }
        }
        DataCopyPad(countsGm[batch], countLocal,
                    {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
        DataCopyPad(topkMissCountsGm[batch * ROUTES],
                    countLocal[ROUTE_COUNT_STRIDE],
                    {1, static_cast<uint16_t>(ROUTES * sizeof(int32_t)),
                     0, 0});
        Sync<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    }

    __aicore__ inline void ProcessBatch(uint32_t batch)
    {
        const uint32_t queryStart = batch == 0U ? 0U :
            static_cast<uint32_t>(actualSeqLengthsQueryGm.GetValue(batch - 1U));
        const uint32_t queryEnd =
            static_cast<uint32_t>(actualSeqLengthsQueryGm.GetValue(batch));
        const int32_t state = requestStateGm.GetValue(batch);
        if (state == LIMConfig::REQUEST_STATE_NON_OFFLOAD) {
            if (queryStart >= queryEnd || queryEnd > tSize ||
                queryEnd - queryStart > MAX_ROUTES) {
                PublishCounts(batch, queryStart, queryEnd, 0, 0);
                return;
            }
            // Standard LI already writes source IDs to both TopK outputs.
            // Avoid a second GM -> UB -> GM copy of every 2048-entry route.
            PublishCounts(batch, queryStart, queryEnd, 0, 0);
            return;
        }
        if (state == LIMConfig::REQUEST_STATE_FIRST_DECODE) {
            InitializeFirstDecode(batch);
            return;
        }
        if (state != LIMConfig::REQUEST_STATE_STEADY) {
            PublishSafeFailure(batch, queryStart, queryEnd);
            return;
        }
        // Preserve the stable optimized MTP3 implementation when its score
        // and route workspace layout is identical to the legacy branch.
        if (queryEnd - queryStart != ROUTES || queryStart != batch * ROUTES) {
            ProcessGeneralizedSteady(batch, queryStart, queryEnd);
            return;
        }
        int32_t validatedRow = -1;
        uint32_t validatedLength = 0U;
        uint32_t validatedCache = 0U;
        if (!ValidateOffloadRequest(batch, queryStart, queryEnd, validatedRow,
                                    validatedLength, validatedCache)) {
            PublishSafeFailure(batch, queryStart, queryEnd);
            return;
        }
        const uint32_t sourceMask = SourceMaskForLength(validatedLength);
        LocalTensor<float> input = pairInBuf.Get<float>();
        LocalTensor<float> merged = pairOutBuf.Get<float>();
        // The eviction merge temporary extends beyond input's lower half, so
        // union sources use a dedicated 32-KiB UB buffer.  The accumulator
        // only uses merged's lower half; its upper half safely holds union
        // destinations across eviction and final TopK patching.
        LocalTensor<int32_t> unionSources = unionSourceBuf.Get<int32_t>();
        LocalTensor<int32_t> unionDestinations =
            merged.ReinterpretCast<int32_t>()[CAPACITY];
        LocalTensor<uint32_t> occurrences = occurrenceBuf.Get<uint32_t>();
        LocalTensor<int32_t> countLocal = occurrences.ReinterpretCast<int32_t>();
        const uint64_t countBase =
            static_cast<uint64_t>(batch) * ROUTES * ROUTE_COUNT_STRIDE;
        DataCopy(countLocal, routeMissCountsGm[countBase],
                 ROUTES * ROUTE_COUNT_STRIDE);
        Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        uint32_t lengths[ROUTES] = {0U, 0U, 0U, 0U};
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            const int32_t value =
                countLocal.GetValue(route * ROUTE_COUNT_STRIDE);
            lengths[route] = value > 0 && value <= static_cast<int32_t>(TOPK)
                                 ? static_cast<uint32_t>(value)
                                 : 0U;
        }

        const uint32_t total = lengths[0] + lengths[1] + lengths[2] + lengths[3];
        if (total == 0U) {
            PublishFinalOutputs(batch, lengths, 0U, countLocal, unionSources,
                                unionDestinations, merged);
            return;
        }

        const uint64_t pairBase = static_cast<uint64_t>(batch) * CAPACITY;
        for (uint32_t route = 0U; route < ROUTES; ++route) {
            if (lengths[route] == 0U) {
                continue;
            }
            const uint64_t gmOffset =
                pairBase + (route % 2U) * PAIR_WORDS;
            const AscendC::DataCopyExtParams copy{
                1, static_cast<uint32_t>(lengths[route] * 2U * sizeof(float)),
                0, 0, 0};
            const AscendC::DataCopyPadExtParams<float> pad{
                false, 0, 0, 0.0F};
            if (route < 2U) {
                DataCopyPad(input[route * PAIR_WORDS], pair0Gm[gmOffset],
                            copy, pad);
            } else {
                DataCopyPad(input[route * PAIR_WORDS], pair1Gm[gmOffset],
                            copy, pad);
            }
        }
        Sync<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        Sync<HardEvent::S_V>(HardEvent::S_V);
        MrgSort4Info params;
        params.elementLengths[0] = lengths[0];
        params.elementLengths[1] = lengths[1];
        params.elementLengths[2] = lengths[2];
        params.elementLengths[3] = lengths[3];
        params.ifExhaustedSuspension = false;
        params.validBit = (lengths[0] > 0U ? 0b0001 : 0U) |
                          (lengths[1] > 0U ? 0b0010 : 0U) |
                          (lengths[2] > 0U ? 0b0100 : 0U) |
                          (lengths[3] > 0U ? 0b1000 : 0U);
        params.repeatTimes = 1;
        MrgSortSrcList<float> sources;
        sources.src1 = input;
        sources.src2 = input[PAIR_WORDS];
        sources.src3 = input[PAIR_WORDS * 2U];
        sources.src4 = input[PAIR_WORDS * 3U];
        MrgSort<float>(merged, sources, params);
        PipeBarrier<PIPE_V>();

        const bool captureOccurrences = UseCompactOccurrenceMap(total);
        const uint32_t count = DeduplicateMergedMisses(
            merged, input, total, unionSources, occurrences,
            captureOccurrences, sourceMask);

        const uint32_t updated = FindEvictSlotsAndUpdateCache(
            batch, validatedLength, validatedCache,
            static_cast<uint32_t>(validatedRow), count, lengths, input,
            merged, unionSources, unionDestinations);
        if (captureOccurrences) {
            PrepareTopkRows(total, updated, occurrences, unionDestinations,
                            merged);
        } else {
            PrepareTopkRowsFallback(batch, lengths, updated, unionSources,
                                    unionDestinations, input, merged);
        }
        PublishFinalOutputs(batch, lengths, updated, countLocal,
                            unionSources, unionDestinations, merged);
    }

    GlobalTensor<float> pair0Gm;
    GlobalTensor<float> pair1Gm;
    GlobalTensor<int32_t> candidateLengthsGm;
    GlobalTensor<int32_t> actualSeqLengthsQueryGm;
    GlobalTensor<int32_t> actualSeqLengthsKeyGm;
    GlobalTensor<int32_t> requestStateGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> cacheTokensGm;
    GlobalTensor<int32_t> reqEntriesGm;
    GlobalTensor<float> scoreScratchGm;
    GlobalTensor<float> thresholdScratchGm;
    GlobalTensor<int32_t> routeMissCountsGm;
    GlobalTensor<int32_t> missSourcesGm;
    GlobalTensor<int32_t> missDestinationsGm;
    GlobalTensor<int32_t> countsGm;
    GlobalTensor<int32_t> topkSourcesGm;
    GlobalTensor<int32_t> topkDestinationsGm;
    GlobalTensor<int32_t> topkMissCountsGm;
    TBuf<TPosition::VECCALC> pairInBuf;
    TBuf<TPosition::VECCALC> pairOutBuf;
    TBuf<TPosition::VECCALC> unionSourceBuf;
    TBuf<TPosition::VECCALC> occurrenceBuf;
    uint32_t batchSize = 0U;
    uint32_t tSize = 0U;
    uint32_t scoreStride = 0U;
    uint32_t sourceCapacity = 0U;
    uint32_t poolSize = 0U;
};
} // namespace MtpUnion

#endif
