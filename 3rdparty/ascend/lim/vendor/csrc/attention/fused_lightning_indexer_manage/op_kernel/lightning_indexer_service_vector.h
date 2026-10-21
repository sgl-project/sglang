/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_service_vector.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_SERVICE_VECTOR_H
#define LIGHTNING_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "fused_lightning_indexer_manage_constants.h"
#include "lightning_indexer_common.h"
#include "lightning_indexer_vector.h"

namespace LIKernel {
using namespace LICommon;
using namespace LIServiceVec;
constexpr uint32_t BASE_TOPK = LIMConfig::TOPK;
constexpr uint32_t LD_PARAM_NUM = LIMConfig::LD_PARAM_COUNT;
constexpr uint32_t EXACT_PACKED_SOURCE_TOKENS = 1U << PACKED_SOURCE_BITS;
constexpr uint32_t MTP_THRESHOLD_STRIDE = LIMConfig::THRESHOLD_WORKSPACE_STRIDE;
constexpr uint32_t MTP_ROUTE_COUNT_STRIDE = LIMConfig::ROUTE_COUNT_WORKSPACE_STRIDE;
constexpr uint32_t MTP_MISS_KEY_BASE_BITS = 0x40000000U;
constexpr uint32_t LD_NEED_FD_INDEX = 0U;
constexpr uint32_t LD_S2_ACT_SEQ_INDEX = 1U;
constexpr uint32_t LD_S2_START_INDEX = 2U;
constexpr uint32_t LD_S2_END_INDEX = 3U;
constexpr uint32_t LD_IS_S2_END_INDEX = 4U;
constexpr uint32_t LD_BN2_INDEX = 5U;
constexpr uint32_t LD_S1_INDEX = 6U;
constexpr uint32_t LD_S1_PROCESS_COUNT_INDEX = 7U;
constexpr uint32_t LD_OUTPUT_OFFSET_INDEX = 8U;
constexpr uint32_t LD_ROUTE_COUNT_INDEX = 9U;

template <typename LIT>
class LIVector {
public:
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using K_T = typename LIT::keyType;
    static constexpr LI_LAYOUT LAYOUT_T = LIT::layout;

    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline LIVector(){};
    __aicore__ inline void ProcessVec(const LICommon::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct LICommon::ConstInfo &constInfo,
                                      const FusedLightningIndexerManageTilingData *__restrict tilingData);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<K_T> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<K_T> valueOutGm,
                                                GlobalTensor<int32_t> reqPoolEntriesGm,
                                                GlobalTensor<int32_t> cacheSlotsGm,
                                                GlobalTensor<int32_t> slotOutGm,
                                                GlobalTensor<uint32_t> actualSeqLengthsQueryGm,
                                                GlobalTensor<int32_t> requestStateGm,
                                                uint32_t batchSize,
                                                __gm__ uint8_t *unionPair0, __gm__ uint8_t *unionPair1,
                                                __gm__ uint8_t *scoreScratch,
                                                __gm__ uint8_t *thresholdScratch,
                                                __gm__ uint8_t *routeMissCounts);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<K_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<K_T> valueOutGm;
    GlobalTensor<int32_t> reqPoolEntriesGm;
    GlobalTensor<int32_t> cacheSlotsGm;
    GlobalTensor<int32_t> slotOutGm;
    GlobalTensor<uint32_t> actualSeqLengthsQueryGm;
    GlobalTensor<int32_t> requestStateGm;
    GlobalTensor<float> unionPair0Gm;
    GlobalTensor<float> unionPair1Gm;
    GlobalTensor<float> scoreScratchGm;
    GlobalTensor<float> thresholdScratchGm;
    GlobalTensor<int32_t> routeMissCountsGm;
    // =================================常量区=================================

private:
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;
    TBuf<TPosition::VECCALC> payloadBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> SortedBasicBlock_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;

    // para for LD
    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = LD_PARAM_NUM;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    struct LICommon::ConstInfo constInfo_;
    uint32_t cacheSlotsSize_ = 0;
    uint32_t poolSize_ = 0;
    uint32_t scoreStride_ = 0;
    uint32_t batchSize_ = 0;

    __aicore__ inline void PrepareChunkPayload(const LocalTensor<int32_t> &payloadLocal,
                                               uint32_t batchIdx, int32_t sourceBase,
                                               int32_t validLen, int32_t alignedLen,
                                               bool standardMode);
    __aicore__ inline void TagLongIndex(const LocalTensor<float> &scoreLocal,
                                       int32_t sourceBase, int32_t validLen);
    __aicore__ inline void DecodeTopkHitMiss(const LocalTensor<float> &pairLocal,
                                             const LocalTensor<int32_t> &indexLocal,
                                             const LocalTensor<int32_t> &slotLocal,
                                             const LocalTensor<int32_t> &scratchLocal,
                                             int64_t outputOffset, bool hasLongIndexTag,
                                             uint32_t batch, uint32_t routeInBatch,
                                             uint32_t routeCount, int64_t visibleKeyCount);
    __aicore__ inline void SortTopkBySlotIndex(const LocalTensor<float> &pairLocal,
                                               const LocalTensor<float> &workspaceLocal,
                                               bool hasLongIndexTag);
};

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitBuffers(TPipe *pipe)
{
    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;

    pipe->InitBuffer(inQueue_, 2,
                     groupInner_ * s2BaseSize_ * sizeof(float) + s2BaseSize_ * sizeof(float)); // 69KB mm_out_ub
    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);                                            // 32KB  extract
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK * 2 * sizeof(float));    // 64KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));                                // 2KB
    // Two blocks are reserved for the generic TopK pair output; two disjoint
    // raw-score blocks ping-pong so an MTE3 score-scratch write can overlap
    // the current TopK sort/merge and the next route's vector work.
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 4 * sizeof(float));
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));
    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));
    pipe->InitBuffer(payloadBuf_, s2BaseSize_ * sizeof(int32_t));

    //
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    SortedBasicBlock_ = globalTopkUb_[BASE_TOPK * 2 * 2];
    globalTopkNum_ = 0;

    // 基本块执行前初始化UB和GM
    // step1. 初始化一个有序索引 0 - s2BaseSize_
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // step2. globalTopkUb_ [CeilDiv(s1BaseSize_, 2), BASE_TOPK, 2]   -inf,-1
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK * 2);

    // step3. 初始化vec1ParamGm，是否进行LD的标志位设为-1(needFd=-1)
    // vec1ResIn32Gm = [aic, 2, s1BaseSize_, 16] int32
    // ws清零 [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, ......]
    LocalTensor<float> tmpfBuff = outQueue_.AllocTensor<float>();
    Duplicate(tmpfBuff.template ReinterpretCast<int32_t>(), -1, 2 * (s1BaseSize_ / 2) * paramNum_ * 2);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
    int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +      // 2个AIV共同地址偏移
                           (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_; // 每个AIV的地址偏移，S1方向
    DataCopyPad(vec1ParamGm[wsInfoOffset], tmpfBuff.template ReinterpretCast<int64_t>(),
                {1, static_cast<uint16_t>((s1BaseSize_ / 2) * 2 * paramNum_ * sizeof(int64_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(tmpfBuff);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float)); // 2：value + index
    pipe->InitBuffer(ldTmpBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float));     // 2：value + index
    pipe->InitBuffer(ldOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::InitParams(const struct LICommon::ConstInfo &constInfo,
                                                 const FusedLightningIndexerManageTilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;
    cacheSlotsSize_ = tilingData->cacheSlotsSize;
    poolSize_ = tilingData->poolSize;
    // Match the uint32 stride used by the eviction reader.  The generic
    // uint64 CeilDiv path produced a zero stride on Ascend910_93.
    const uint32_t scoreBlock = static_cast<uint32_t>(s2BaseSize_);
    scoreStride_ = ((tilingData->s2Size + scoreBlock - 1U) / scoreBlock) * scoreBlock;

    // group ub 切分因子当前按照UB空间强制为16
    groupInner_ = 16;

    blockId_ = GetBlockIdx();
}

template <typename LIT>
__aicore__ inline void
LIVector<LIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                    GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<K_T> weightsGm,
                                    GlobalTensor<int32_t> indiceOutGm, GlobalTensor<K_T> valueOutGm,
                                    GlobalTensor<int32_t> reqPoolEntriesGm,
                                    GlobalTensor<int32_t> cacheSlotsGm,
                                    GlobalTensor<int32_t> slotOutGm,
                                    GlobalTensor<uint32_t> actualSeqLengthsQueryGm,
                                    GlobalTensor<int32_t> requestStateGm,
                                    uint32_t batchSize,
                                    __gm__ uint8_t *unionPair0, __gm__ uint8_t *unionPair1,
                                    __gm__ uint8_t *scoreScratch,
                                    __gm__ uint8_t *thresholdScratch,
                                    __gm__ uint8_t *routeMissCounts)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
    this->reqPoolEntriesGm = reqPoolEntriesGm;
    this->cacheSlotsGm = cacheSlotsGm;
    this->slotOutGm = slotOutGm;
    this->actualSeqLengthsQueryGm = actualSeqLengthsQueryGm;
    this->requestStateGm = requestStateGm;
    this->batchSize_ = batchSize;
    this->unionPair0Gm.SetGlobalBuffer((__gm__ float *)unionPair0);
    this->unionPair1Gm.SetGlobalBuffer((__gm__ float *)unionPair1);
    this->scoreScratchGm.SetGlobalBuffer((__gm__ float *)scoreScratch);
    this->thresholdScratchGm.SetGlobalBuffer((__gm__ float *)thresholdScratch);
    this->routeMissCountsGm.SetGlobalBuffer((__gm__ int32_t *)routeMissCounts);
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::PrepareChunkPayload(
    const LocalTensor<int32_t> &payloadLocal, uint32_t batchIdx, int32_t sourceBase,
    int32_t validLen, int32_t alignedLen, bool standardMode)
{
    if (standardMode) {
        // Full 512-token chunks overwrite every sort payload entry.  Only a
        // partial tail needs INVALID_IDX padding for the aligned sort range.
        if (validLen < alignedLen) {
            Duplicate(payloadLocal, constInfo_.INVALID_IDX, alignedLen);
            PipeBarrier<PIPE_V>();
        }
        // Standard LI only needs the source payload.  Long-index high bits are
        // carried by TagLongIndex, so generate the low INDEX_BITS directly
        // without reading or packing the identity cache row.
        Adds(payloadLocal, globalTopkIndice_, sourceBase, validLen);
        PipeBarrier<PIPE_V>();
        return;
    }
    Duplicate(payloadLocal, constInfo_.INVALID_IDX, alignedLen);
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
    int32_t cacheRowIdx = reqPoolEntriesGm.GetValue(batchIdx);
    if (cacheRowIdx < 0 || static_cast<uint32_t>(cacheRowIdx) >= poolSize_) {
        return;
    }
    uint64_t cacheRowBase = static_cast<uint64_t>(cacheRowIdx) * cacheSlotsSize_;
    DataCopyPad(payloadLocal, cacheSlotsGm[cacheRowBase + static_cast<uint32_t>(sourceBase)],
                AscendC::DataCopyExtParams{1, static_cast<uint32_t>(validLen * sizeof(int32_t)), 0, 0, 0},
                AscendC::DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

    Maxs(payloadLocal, payloadLocal, constInfo_.INVALID_IDX, validLen);
    PipeBarrier<PIPE_V>();
    ShiftLeft(payloadLocal.template ReinterpretCast<uint32_t>(),
              payloadLocal.template ReinterpretCast<uint32_t>(), INDEX_BITS, validLen);
    PipeBarrier<PIPE_V>();
    Add(payloadLocal, payloadLocal, globalTopkIndice_, validLen);
    PipeBarrier<PIPE_V>();
    Adds(payloadLocal, payloadLocal, sourceBase & static_cast<int32_t>((1U << INDEX_BITS) - 1U), validLen);
    PipeBarrier<PIPE_V>();

}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::TagLongIndex(
    const LocalTensor<float> &scoreLocal, int32_t sourceBase, int32_t validLen)
{
    LocalTensor<uint32_t> scoreBits = scoreLocal.template ReinterpretCast<uint32_t>();
    ShiftRight(scoreBits, scoreBits, SCORE_TAG_CLEAR_SHIFT, validLen);
    PipeBarrier<PIPE_V>();
    ShiftLeft(scoreBits, scoreBits, SCORE_TAG_CLEAR_SHIFT, validLen);
    PipeBarrier<PIPE_V>();
    Adds(scoreBits.template ReinterpretCast<int32_t>(), scoreBits.template ReinterpretCast<int32_t>(),
         (sourceBase >> INDEX_BITS) & ((1 << INDEX_HIGH_BITS) - 1), validLen);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::SortTopkBySlotIndex(
    const LocalTensor<float> &pairLocal, const LocalTensor<float> &workspaceLocal,
    bool hasLongIndexTag)
{
    constexpr int32_t HIT_KEY_BASE_BITS = static_cast<int32_t>(0x3F800000U);
    constexpr int32_t MISS_KEY_DELTA_BITS = static_cast<int32_t>(0x00800000U);
    constexpr int32_t SHORT_INDEX_MASK =
        static_cast<int32_t>(PACKED_SOURCE_MASK);
    constexpr int32_t LONG_INDEX_MASK =
        static_cast<int32_t>(LONG_SOURCE_MASK);
    LocalTensor<float> keyLocal = workspaceLocal;
    LocalTensor<uint32_t> payloadLocal = workspaceLocal[BASE_TOPK].template ReinterpretCast<uint32_t>();
    LocalTensor<int32_t> missFlagLocal = workspaceLocal[BASE_TOPK * 2].template ReinterpretCast<int32_t>();
    LocalTensor<float> sortTmpLocal = workspaceLocal[BASE_TOPK * 2];

    ExtractIndex(payloadLocal, pairLocal.template ReinterpretCast<uint32_t>(), BASE_TOPK);
    ShiftLeft(keyLocal.template ReinterpretCast<uint32_t>(), payloadLocal,
              32U - INDEX_BITS, BASE_TOPK);
    PipeBarrier<PIPE_V>();
    ShiftRight(keyLocal.template ReinterpretCast<uint32_t>(),
               keyLocal.template ReinterpretCast<uint32_t>(),
               32U - INDEX_BITS, BASE_TOPK);
    PipeBarrier<PIPE_V>();
    if (hasLongIndexTag) {
        ExtractScoreBits(missFlagLocal.template ReinterpretCast<uint32_t>(),
                         pairLocal.template ReinterpretCast<uint32_t>(), BASE_TOPK);
        ShiftLeft(missFlagLocal.template ReinterpretCast<uint32_t>(),
                  missFlagLocal.template ReinterpretCast<uint32_t>(),
                  SCORE_TAG_EXTRACT_SHIFT, BASE_TOPK);
        PipeBarrier<PIPE_V>();
        ShiftRight(missFlagLocal.template ReinterpretCast<uint32_t>(),
                   missFlagLocal.template ReinterpretCast<uint32_t>(),
                   SCORE_TAG_EXTRACT_SHIFT, BASE_TOPK);
        PipeBarrier<PIPE_V>();
        ShiftLeft(missFlagLocal.template ReinterpretCast<uint32_t>(),
                  missFlagLocal.template ReinterpretCast<uint32_t>(), INDEX_BITS, BASE_TOPK);
        PipeBarrier<PIPE_V>();
        Add(keyLocal.template ReinterpretCast<int32_t>(),
            keyLocal.template ReinterpretCast<int32_t>(), missFlagLocal, BASE_TOPK);
        PipeBarrier<PIPE_V>();
    }
    Muls(keyLocal.template ReinterpretCast<int32_t>(),
         keyLocal.template ReinterpretCast<int32_t>(), -1, BASE_TOPK);
    PipeBarrier<PIPE_V>();
    Adds(keyLocal.template ReinterpretCast<int32_t>(),
         keyLocal.template ReinterpretCast<int32_t>(),
         hasLongIndexTag ? LONG_INDEX_MASK : SHORT_INDEX_MASK, BASE_TOPK);
    PipeBarrier<PIPE_V>();

    ShiftRight(missFlagLocal.template ReinterpretCast<uint32_t>(), payloadLocal,
               INDEX_BITS, BASE_TOPK);
    PipeBarrier<PIPE_V>();
    Adds(missFlagLocal, missFlagLocal, 1, BASE_TOPK);
    PipeBarrier<PIPE_V>();
    ShiftRight(missFlagLocal.template ReinterpretCast<uint32_t>(),
               missFlagLocal.template ReinterpretCast<uint32_t>(),
               INVALID_FLAG_SHIFT, BASE_TOPK);
    PipeBarrier<PIPE_V>();
    Muls(missFlagLocal, missFlagLocal, MISS_KEY_DELTA_BITS, BASE_TOPK);
    PipeBarrier<PIPE_V>();
    Add(keyLocal.template ReinterpretCast<int32_t>(),
        keyLocal.template ReinterpretCast<int32_t>(), missFlagLocal, BASE_TOPK);
    PipeBarrier<PIPE_V>();
    Adds(keyLocal.template ReinterpretCast<int32_t>(),
         keyLocal.template ReinterpretCast<int32_t>(), HIT_KEY_BASE_BITS, BASE_TOPK);
    PipeBarrier<PIPE_V>();

    LocalTensor<float> sortedPair = pairLocal;
    LIServiceVec::SortAll(sortedPair, keyLocal, payloadLocal, sortTmpLocal, BASE_TOPK);
    PipeBarrier<PIPE_V>();
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::DecodeTopkHitMiss(
    const LocalTensor<float> &pairLocal, const LocalTensor<int32_t> &indexLocal,
    const LocalTensor<int32_t> &slotLocal, const LocalTensor<int32_t> &scratchLocal,
    int64_t outputOffset, bool hasLongIndexTag, uint32_t batch,
    uint32_t routeInBatch, uint32_t routeCount, int64_t visibleKeyCount)
{
    const int32_t requestState = batch < batchSize_
        ? requestStateGm.GetValue(batch) : 0;
    ExtractIndex(indexLocal.template ReinterpretCast<uint32_t>(),
                 pairLocal.template ReinterpretCast<uint32_t>(), constInfo_.sparseCount);
    if (requestState == LIMConfig::REQUEST_STATE_NON_OFFLOAD) {
        // Causal masking changes scores to -inf, but the shared chunk payload
        // still contains later queries' source IDs.  Invalidate the fixed
        // TopK suffix when this route sees fewer than TopK keys.
        if (visibleKeyCount < constInfo_.sparseCount) {
            const uint32_t validCount = visibleKeyCount > 0
                ? static_cast<uint32_t>(visibleKeyCount) : 0U;
            // Vector stores require a 32-byte-aligned start.  Fill the aligned
            // suffix in bulk, then repair the at-most-seven boundary entries.
            const uint32_t alignedCount = (validCount + 7U) & ~7U;
            PipeBarrier<PIPE_V>();
            if (alignedCount < constInfo_.sparseCount) {
                Duplicate(indexLocal[alignedCount], LIMConfig::PADDING_SOURCE_ID,
                          constInfo_.sparseCount - alignedCount);
            }
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            for (uint32_t position = validCount; position < alignedCount; ++position) {
                indexLocal.SetValue(position, LIMConfig::PADDING_SOURCE_ID);
            }
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        }
        // Standard payload already is the source ID (including -1 padding).
        // Preserve score order and publish the identity destination directly.
        LIServiceVec::CopyOut(slotOutGm[outputOffset], indexLocal,
                              constInfo_.sparseCount);
        return;
    }
    DecodePackedSlot(slotLocal, indexLocal.template ReinterpretCast<uint32_t>(),
                     scratchLocal, constInfo_.sparseCount);
    DecodePackedIndex(indexLocal.template ReinterpretCast<uint32_t>(),
                      scratchLocal.template ReinterpretCast<uint32_t>(),
                      constInfo_.sparseCount, false);
    if (hasLongIndexTag) {
        ExtractScoreBits(scratchLocal.template ReinterpretCast<uint32_t>(),
                         pairLocal.template ReinterpretCast<uint32_t>(), constInfo_.sparseCount);
        ShiftLeft(scratchLocal.template ReinterpretCast<uint32_t>(),
                  scratchLocal.template ReinterpretCast<uint32_t>(),
                  32U - INDEX_BITS - INDEX_HIGH_BITS, constInfo_.sparseCount);
        PipeBarrier<PIPE_V>();
        ShiftRight(scratchLocal.template ReinterpretCast<uint32_t>(),
                   scratchLocal.template ReinterpretCast<uint32_t>(),
                   32U - INDEX_HIGH_BITS, constInfo_.sparseCount);
        PipeBarrier<PIPE_V>();
        Muls(scratchLocal, scratchLocal, -1, constInfo_.sparseCount);
        PipeBarrier<PIPE_V>();
        Adds(scratchLocal, scratchLocal, (1 << INDEX_HIGH_BITS) - 1, constInfo_.sparseCount);
        PipeBarrier<PIPE_V>();
        ShiftLeft(scratchLocal.template ReinterpretCast<uint32_t>(),
                  scratchLocal.template ReinterpretCast<uint32_t>(), INDEX_BITS,
                  constInfo_.sparseCount);
        PipeBarrier<PIPE_V>();
        Add(indexLocal, indexLocal, scratchLocal, constInfo_.sparseCount);
        PipeBarrier<PIPE_V>();
    }
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    LocalTensor<uint32_t> pairBits = pairLocal.ReinterpretCast<uint32_t>();
    uint32_t low = 0U;
    uint32_t high = BASE_TOPK;
    while (low < high) {
        const uint32_t middle = (low + high) >> 1U;
        if (pairBits.GetValue(middle * 2U) >= MTP_MISS_KEY_BASE_BITS) {
            low = middle + 1U;
        } else {
            high = middle;
        }
    }
    const uint32_t routeMissCount = low;
    const uint32_t route = static_cast<uint32_t>(outputOffset / BASE_TOPK);
    const uint64_t pairOffset = static_cast<uint64_t>(batch) * BASE_TOPK * 4U +
                                (routeInBatch % 2U) * BASE_TOPK * 2U;
    const bool useMaturePairUnion = batch < batchSize_ &&
        requestState == LIMConfig::REQUEST_STATE_STEADY &&
        routeCount >= 1U && routeCount <= LIMConfig::MATURE_UNION_MAX_ROUTES;
    if (routeMissCount != 0U && useMaturePairUnion) {
        // The union stage only consumes the source-sorted key word.  Preserve
        // the final route/position of every miss in the otherwise-dead payload
        // word, so it can distribute the assigned union slot directly instead
        // of reading topk_src_ids back from GM and joining by source again.
        constexpr uint32_t MTP_ROUTE_POSITION_BITS = 11U;
        for (uint32_t position = 0U; position < routeMissCount; ++position) {
            pairBits.SetValue(
                position * 2U + 1U,
                (static_cast<uint32_t>(routeInBatch) << MTP_ROUTE_POSITION_BITS) |
                    position);
        }
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        if (routeInBatch < 2U) {
            LIServiceVec::CopyOut(unionPair0Gm[pairOffset], pairLocal,
                                  routeMissCount * 2U);
        } else {
            LIServiceVec::CopyOut(unionPair1Gm[pairOffset], pairLocal,
                                  routeMissCount * 2U);
        }
    }
    LIServiceVec::CopyOut(slotOutGm[outputOffset], slotLocal,
                          constInfo_.sparseCount);
    scratchLocal.SetValue(0U, static_cast<int32_t>(routeMissCount));
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    DataCopyPad(routeMissCountsGm[route * MTP_ROUTE_COUNT_STRIDE], scratchLocal,
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);

    // Misses form the prefix and hits form the suffix.  Keep the complete
    // decoded source-ID row for both, matching the non-MTP LIM interface.
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::AllocEventID()
{
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::FreeEventID()
{
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    LIServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
    idxULocal1.SetValue(0U, 0);
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    const uint64_t route =
        static_cast<uint64_t>(invalidS1offset) / BASE_TOPK;
    DataCopyPad(routeMissCountsGm[route * MTP_ROUTE_COUNT_STRIDE], idxULocal1,
                {1, static_cast<uint16_t>(sizeof(int32_t)), 0, 0});
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    outQueue_.FreeTensor(valueULocal);

    if (constInfo_.returnValue) {
        uint16_t negInf = 0;
        if constexpr(std::is_same<K_T, float16_t>::value) {
            negInf = 0xFC00;
        } else {
            negInf = 0xFF80;
        }
        LocalTensor<uint16_t> valueULocal = outQueue_.AllocTensor<uint16_t>();
        Duplicate(valueULocal, negInf, constInfo_.sparseCount);
        outQueue_.EnQue<uint16_t>(valueULocal);
        valueULocal = outQueue_.DeQue<uint16_t>();
        GlobalTensor<uint16_t> valueOutGmTmp;
        valueOutGmTmp.SetGlobalBuffer((__gm__ uint16_t *)valueOutGm.GetPhyAddr());
        LIServiceVec::CopyOut(valueOutGmTmp[invalidS1offset], valueULocal, constInfo_.sparseCount);
        outQueue_.FreeTensor(valueULocal);
    }
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::ProcessVec(const LICommon::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 512*512 + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * ((s1BaseSize_ * gSize_) * s2BaseSize_);
    // (B,S1,N1,1);(T,N1,1) -> (B,S1,N2,G,1) 当前只切分到S1轴
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    PipeBarrier<PIPE_V>();
    // cuS1BeginIdxPerAiv: 每个AIV的S1起始偏移
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // cuS1ProcNumPerAiv: 每个AIv的S1计算量
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    // 基本块基地址偏移奇数核加一个S1地址偏移
    weightGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    // cut G
    int32_t outerG = CeilDiv(gSize_, groupInner_);

    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK * 2);
        blockS2StartIdx_ = 0;
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq
    int32_t cuRealAcSeq = info.actS2Size;
    if (info.causal) {
        // attenMask true场景
        cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
    }
    int32_t sharedS2Len = cuBaseS2Idx + s2BaseSize_ >= info.actS2Size
                              ? info.actS2Size - cuBaseS2Idx
                              : s2BaseSize_;
    LocalTensor<int32_t> payloadLocal = payloadBuf_.Get<int32_t>();
    if (sharedS2Len > 0) {
        PrepareChunkPayload(payloadLocal, info.bIdx, cuBaseS2Idx,
                            sharedS2Len, s2BaseSize_, info.causal);
    }
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    event_t scoreMte3ToV[2] = {
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V)),
        static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V))};
    bool scoreWritePending[2] = {false, false};
    // LD输出S1方向偏移，保证2个Vector输出的内容连续
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (info.causal) {
            cuRealAcSeq += 1;
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            int32_t mmUbStride = (cuS2LenVecAlign - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
            const uint32_t scoreBufferIndex =
                static_cast<uint32_t>(innerS1Idx) & 1U;
            if (scoreWritePending[scoreBufferIndex]) {
                WaitFlag<HardEvent::MTE3_V>(
                    scoreMte3ToV[scoreBufferIndex]);
                scoreWritePending[scoreBufferIndex] = false;
            }
            LocalTensor<float> reduceOutInner =
                reduceOutBuff[s2BaseSize_ * (2U + scoreBufferIndex)];
            PipeBarrier<PIPE_V>();
            LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
            for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
                int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;
                LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
                LocalTensor<float> weightsInUb = mmInUb[procGnum * s2BaseSize_];
                LocalTensor<K_T> weightsInTUb = weightsInUb.template ReinterpretCast<K_T>();
                if constexpr (!IsSameType<K_T, float>::value) {
                    weightsInTUb = weightsInTUb[groupInner_];
                }
                LIServiceVec::CopyIn(mmInUb, weightsInTUb, mm1ResGm, weightsGm,
                                     mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                         outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign,
                                     weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_, procGnum,
                                     info.actualSingleProcessSInnerSizeAlign, mmUbStride);

                inQueue_.EnQue<float>(mmInUb);
                mmInUb = inQueue_.DeQue<float>();
                weightsInUb = mmInUb[procGnum * s2BaseSize_];
                LIServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], mmInUb, weightsInUb, weightsInTUb,
                                      brcBuf, procGnum, s2BaseSize_, outerGidx);
                // confused reduceOp in DoScale
                // neednot use LIServiceVec::doReduce(mmInUb, reduceOutInner, procGnum, (s2BaseSize_+8));
                inQueue_.FreeTensor(mmInUb);
            }

            int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            LIServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, gRedCnt, s2BaseSize_);
            outQueue_.FreeTensor(reduceCacheBuf);

            // Materialize the TopK input before publishing the immutable raw
            // score.  The following sort uses this disjoint first block while
            // MTE3 drains one of the two raw-score blocks.
            LocalTensor<float> sortScoreUb = reduceOutBuff;
            PipeBarrier<PIPE_V>();
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
            PipeBarrier<PIPE_V>();
            // Standard (-3) payloads keep the complete source ID directly.
            // Tagging their float score would overwrite its low bits and can
            // perturb the order of otherwise-close TopK scores.  Packed
            // offload payloads alone need the score-side high source bits.
            const bool hasLongIndexTag =
                !info.causal && info.actS2Size > EXACT_PACKED_SOURCE_TOKENS;
            if (hasLongIndexTag) {
                TagLongIndex(sortScoreUb, cuBaseS2Idx, cuS2Len);
            }

            // Raw scores are consumed only by offload eviction.  Avoid the
            // full L-float GM write for state=-3.
            const uint64_t routeIndex =
                static_cast<uint64_t>(info.indiceOutOffset) /
                    static_cast<uint64_t>(constInfo_.sparseCount) +
                static_cast<uint32_t>(cuS1Idx);
            if (!info.causal) {
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                LIServiceVec::CopyOut(
                    scoreScratchGm[static_cast<uint64_t>(routeIndex) * scoreStride_ +
                                   static_cast<uint32_t>(cuBaseS2Idx)],
                    reduceOutInner, cuS2LenVecAlign);
                SetFlag<HardEvent::MTE3_V>(scoreMte3ToV[scoreBufferIndex]);
                scoreWritePending[scoreBufferIndex] = true;
            }
            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            if (info.actS1Size > 4) {
                // info.actS1Size > 4 则单个vector核内处理的 s1>2，缓存方案无法处理
                // The in-place SortAll overload consumes [score, source-id].
                // Q<=4 passes payloadLocal to Sort directly; materialize the
                // same source IDs in the second half for the Q=5..7 path.
                DataCopy(
                    reduceOutBuff[cuS2LenVecAlign]
                        .template ReinterpretCast<int32_t>(),
                    payloadLocal, cuS2LenVecAlign);
                PipeBarrier<PIPE_V>();
                LIServiceVec::SortAll(reduceOutBuff, tmpSortBuf,
                                      cuS2LenVecAlign); //  cuS2LenVecAlign <= s2BaseSize_, fill -inf
                PipeBarrier<PIPE_V>();
                LIServiceVec::MergeSortExistingFirst(
                    globalTopkUb_[innerS1Idx * BASE_TOPK * 2], BASE_TOPK, reduceOutBuff,
                    cuS2LenVecAlign, tmpSortBuf);
            } else {
                int64_t globalTopkUbCacheIdx = (info.s2Idx - blockS2StartIdx_) % 4;
                Sort<float, true>(
                    SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2 + globalTopkUbCacheIdx * s2BaseSize_ * 2],
                    reduceOutBuff, payloadLocal.template ReinterpretCast<uint32_t>(), tmpSortBuf,
                    cuS2LenVecAlign / 32);
                // 缓存4块512或者S2结束, 需要进行精排
                if (globalTopkUbCacheIdx == 3 || isS2End || info.isAllLoopEnd) {
                    LocalTensor<float> tt = SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2];
                    // 前4块直接精排覆盖到globalTopkUb_
                    if (info.s2Idx - blockS2StartIdx_ < 4) {
                        MrgBasicBlock(globalTopkUb_[innerS1Idx * BASE_TOPK * 2], tt,
                                      static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                    } else { // 后面缓存在 SortedBasicBlock_, 先精排, 再merge到globalTopkUb_
                        if (globalTopkUbCacheIdx > 0) {
                            MrgBasicBlock(tmpSortBuf, tt, static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                            PipeBarrier<PIPE_V>();
                            DataCopy(SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf,
                                     (globalTopkUbCacheIdx + 1) * s2BaseSize_ * 2);
                        }
                        PipeBarrier<PIPE_V>();
                        SparseTopK(globalTopkUb_[innerS1Idx * BASE_TOPK * 2],
                                   SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf, BASE_TOPK,
                                   s2BaseSize_ * (globalTopkUbCacheIdx + 1));
                    }
                }
            }

            PipeBarrier<PIPE_V>();
            outQueue_.FreeTensor(tmpSortBuf);

            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;

            // 中间结果保存
            bool needCopyWsGm = info.isAllLoopEnd || isS2End;

            // Later output paths allocate their own MTE3->V events.  Retire
            // the score events first so FetchEventID cannot recycle a still
            // pending ID on the final/workspace-producing iteration.
            if (needCopyOutGm || needCopyWsGm) {
                for (uint32_t pendingIndex = 0U; pendingIndex < 2U;
                     ++pendingIndex) {
                    if (scoreWritePending[pendingIndex]) {
                        WaitFlag<HardEvent::MTE3_V>(
                            scoreMte3ToV[pendingIndex]);
                        scoreWritePending[pendingIndex] = false;
                    }
                }
            }

            if (needCopyOutGm) {
                if (!constInfo_.returnValue) {
                    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
                    if (!info.causal) {
                        // Threshold and slot ordering are used only by the
                        // offload eviction path.
                        valueULocal.SetValue(
                            0U, globalTopkUb_[innerS1Idx * BASE_TOPK * 2]
                                    .GetValue((BASE_TOPK - 1U) * 2U));
                        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                        DataCopyPad(
                            thresholdScratchGm[
                                (static_cast<uint64_t>(info.indiceOutOffset) /
                                     BASE_TOPK +
                                 static_cast<uint32_t>(cuS1Idx)) *
                                MTP_THRESHOLD_STRIDE],
                            valueULocal,
                            {1, static_cast<uint16_t>(sizeof(float)), 0, 0});
                        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                        SortTopkBySlotIndex(globalTopkUb_[innerS1Idx * BASE_TOPK * 2], valueULocal,
                                            hasLongIndexTag);
                    }
                    LocalTensor<int32_t> slotLocal = valueULocal.template ReinterpretCast<int32_t>();
                    LocalTensor<int32_t> indexLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
                    LocalTensor<int32_t> scratchLocal = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK * 2];
                    int64_t outputOffset = info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount;
                    DecodeTopkHitMiss(globalTopkUb_[innerS1Idx * BASE_TOPK * 2], indexLocal,
                                      slotLocal, scratchLocal, outputOffset,
                                      hasLongIndexTag,
                                      info.bIdx, static_cast<uint32_t>(cuS1Idx),
                                      info.actS1Size, cuRealAcSeq);
                    InitSortOutBuf(globalTopkUb_[innerS1Idx * BASE_TOPK * 2], BASE_TOPK * 2);
                    outQueue_.EnQue<float>(valueULocal);
                    valueULocal = outQueue_.DeQue<float>();
                    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>()[BASE_TOPK];
                    LIServiceVec::CopyOut(indiceOutGm[outputOffset], idxULocal1, constInfo_.sparseCount);
                    outQueue_.FreeTensor(valueULocal);
                } else {
                    LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                    LocalTensor<uint32_t> outIdxUb = outValueUb[BASE_TOPK].template ReinterpretCast<uint32_t>();
                    Extract(outValueUb, outIdxUb, globalTopkUb_[innerS1Idx * BASE_TOPK * 2], (BASE_TOPK / 32));
                    PipeBarrier<PIPE_V>();
                    LocalTensor<K_T> valueULocal1 = outValueUb.template ReinterpretCast<K_T>();
                    Cast(valueULocal1, outValueUb, RoundMode::CAST_ROUND, constInfo_.sparseCount);
                    PipeBarrier<PIPE_V>();
                    outQueue_.EnQue<float>(outValueUb);
                    outValueUb = outQueue_.DeQue<float>();
                    LocalTensor<int32_t> idxULocal1 = outValueUb[BASE_TOPK].template ReinterpretCast<int32_t>();
                    LIServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount],
                                        idxULocal1, constInfo_.sparseCount);
                    LIServiceVec::CopyOut(valueOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount],
                                        valueULocal1, constInfo_.sparseCount);
                    outQueue_.FreeTensor(outValueUb);
                }
            } else if (needCopyWsGm) {
                // vec1Res Gm = [aic, s1BaseSize_, 2, 2, topkOut_] float32
                // vec1Param Gm = [aic, s1BaseSize_, 2, 16] int64
                //     16 = [needFd, s2AcSeq, s2Start, s2End, isS2End, bn2idx, s1Idx, S1ProcNum, ......]

                int64_t wsOffset = (blockId_ / 2) * s1BaseSize_ * 2 * 2 * BASE_TOPK +       // 2个AIV共同地址偏移
                                   (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * 2 * BASE_TOPK + // 每个AIV的地址偏移，S1方向
                                   (ldS1Offset + innerS1Idx) * 2 * 2 * BASE_TOPK;
                int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +       // 2个AIV共同地址偏移
                                       (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_ + // 每个AIV的地址偏移，S1方向
                                       (ldS1Offset + innerS1Idx) * 2 * paramNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                tmpiBuff.SetValue(LD_NEED_FD_INDEX, static_cast<int64_t>(1));
                tmpiBuff.SetValue(LD_S2_ACT_SEQ_INDEX, static_cast<int64_t>(cuRealAcSeq));
                tmpiBuff.SetValue(LD_S2_START_INDEX, static_cast<int64_t>(blockS2StartIdx_));
                tmpiBuff.SetValue(LD_S2_END_INDEX, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(LD_IS_S2_END_INDEX, static_cast<int64_t>(isS2End));
                tmpiBuff.SetValue(LD_BN2_INDEX, static_cast<int64_t>(info.bN2Idx));
                tmpiBuff.SetValue(LD_S1_INDEX, static_cast<int64_t>(cuS1Idx));
                tmpiBuff.SetValue(LD_S1_PROCESS_COUNT_INDEX, static_cast<int64_t>(cuS1ProcNum));
                tmpiBuff.SetValue(LD_OUTPUT_OFFSET_INDEX,
                                  static_cast<int64_t>(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount));
                tmpiBuff.SetValue(LD_ROUTE_COUNT_INDEX, static_cast<int64_t>(info.actS1Size));
                // 写入头尾判断
                // [head, tail]
                // head: 与前面规约，与前后规约
                // tail: 与后面规约
                bool isTailReduce = blockS2StartIdx_ == 0; // 一定是isLastTile
                // WS偏移规则 blockS2StartIdx_ != 0
                // 跟前面块做规约 写到0偏移 不用做计算 blockS2StartIdx_ == 0 and !isS2End
                // 跟后面块做规约 写到1偏移  需要 + s1BaseSize_, BASE_TOPK*2
                if (isTailReduce) { // S2不是最后结束的数据就需要往后做规约，放入第二块ws
                    wsInfoOffset += paramNum_;
                    wsOffset += 2 * BASE_TOPK;
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                LIServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, LD_PARAM_NUM);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                LIServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * BASE_TOPK * 2], 2 * BASE_TOPK);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        } else if (cuRealAcSeq <= 0) {
            for (uint32_t pendingIndex = 0U; pendingIndex < 2U;
                 ++pendingIndex) {
                if (scoreWritePending[pendingIndex]) {
                    WaitFlag<HardEvent::MTE3_V>(
                        scoreMte3ToV[pendingIndex]);
                    scoreWritePending[pendingIndex] = false;
                }
            }
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
        }
    }

    // ProcessVec may return before either ping-pong block is reused.  Drain
    // both completion events so the caller can safely recycle the TBuf and so
    // the following cross-core union barrier observes all score writes.
    for (uint32_t scoreBufferIndex = 0U; scoreBufferIndex < 2U;
         ++scoreBufferIndex) {
        if (scoreWritePending[scoreBufferIndex]) {
            WaitFlag<HardEvent::MTE3_V>(scoreMte3ToV[scoreBufferIndex]);
        }
    }

    // BNSD场景无效S1 输出-1
    if (LAYOUT_T == LI_LAYOUT::BSND) {
        // 最后一个S1的基本块, 需要 >= info.actS1Size
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        // blockS2StartIdx_ == 0 控制S2从开始的核去做冗余清理
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.sparseCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2Size;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && info.causal) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.sparseCount);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        // S2最后一个Loop后, 下一个基本块初始从0开始
        blockS2StartIdx_ = 0;
    }
}

template <typename LIT>
__aicore__ inline void LIVector<LIT>::ProcessLD()
{
    int32_t curCubeId = blockId_ / 2;
    int32_t tmpCubeId = curCubeId;

    int64_t s2ActSeq;
    int64_t s2Start;
    int64_t s2End;
    int64_t isS2End;
    int64_t bn2Idx;
    int64_t s1Idx;
    uint32_t acc_list_num = 0;
    int64_t bIdx = 0;
    int64_t needFd;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    int64_t nextneedFd;
    int64_t valueOffset = 0;
    int64_t outOffset = 0;
    int64_t routeCount = 0;

    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    // S2开头信息
    // 开始必然没有头规约，因此从尾规约开始处理，while循环读取下一个核的头规约
    // 存满4个list或者遇到S2结尾，则做merge，直到做完S2
    // 每个核都忽略自己的头规约，因为必然由前面的核做完
    uint32_t s1LdStartIdx = 0;
    uint32_t s1ProcNum = 0;
    uint64_t paramGmCoreOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_;
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        needFd = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_);
        if (needFd == 1) {
            s1LdStartIdx = (s1ProcNum == 0) ? innerS1Idx : s1LdStartIdx;
            s1ProcNum++;
        }
    }

    if (s1ProcNum == 0) {
        return;
    }

    // S1逐行计算
    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) {
        s1LdStartIdx = s1LdStartIdx + s1VecNum;
        s1VecNum = s1ProcNum - s1VecNum;
    }
    for (uint32_t innerS1Idx = s1LdStartIdx; innerS1Idx < s1LdStartIdx + s1VecNum; innerS1Idx++) {
        // 重置偏移
        tmpCubeId = curCubeId;
        acc_list_num = 0;
        valueOffset = 0;

        // 搬入数据
        wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                   innerS1Idx * 2 * 2 * BASE_TOPK + 2 * BASE_TOPK;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(curValueIdxUb, vec1ResGm[wsOffset],
                    {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
        acc_list_num++;
        valueOffset += 2 * BASE_TOPK;

        // 获取下一个核规约信息
        tmpCubeId++;
        wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
        needFd = vec1ParamGm.GetValue(wsInfoOffset + LD_NEED_FD_INDEX);
        s2ActSeq = vec1ParamGm.GetValue(wsInfoOffset + LD_S2_ACT_SEQ_INDEX);
        isS2End = vec1ParamGm.GetValue(wsInfoOffset + LD_IS_S2_END_INDEX);
        s1Idx = vec1ParamGm.GetValue(wsInfoOffset + LD_S1_INDEX);
        outOffset = vec1ParamGm.GetValue(wsInfoOffset + LD_OUTPUT_OFFSET_INDEX);
        bIdx = vec1ParamGm.GetValue(wsInfoOffset + LD_BN2_INDEX) /
               static_cast<int64_t>(constInfo_.kHeadNum);
        routeCount = vec1ParamGm.GetValue(wsInfoOffset + LD_ROUTE_COUNT_INDEX);

        while (needFd == 1) {
            // 搬入头规约数据
            wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + // 2个AIV共同地址偏移
                       innerS1Idx * 2 * 2 * BASE_TOPK;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset],
                        {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
            valueOffset += 2 * BASE_TOPK;
            acc_list_num++;

            // 每满4个list，聚合  前2K为mrg结果
            if (acc_list_num == mrgListNum_) {
                // MrgSort 四条2048的队列，Mrg成一条
                AscendC::MrgSort4Info params;
                params.elementLengths[0] = BASE_TOPK;
                params.elementLengths[1] = BASE_TOPK;
                params.elementLengths[2] = BASE_TOPK;
                params.elementLengths[3] = BASE_TOPK;
                params.ifExhaustedSuspension = true;
                params.validBit = 0b1111;
                params.repeatTimes = 1;

                AscendC::MrgSortSrcList<float> srcList;
                srcList.src1 = curValueIdxUb[0];
                srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
                srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
                srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                MrgSort(tmpUb, srcList, params);
                PipeBarrier<PIPE_V>();
                DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
                PipeBarrier<PIPE_V>();
                acc_list_num = 1;
                valueOffset = 2 * BASE_TOPK;
            }

            // reduce到S2末尾，则跳出
            if (isS2End == 1) {
                break;
            }

            tmpCubeId++;
            wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
            needFd = vec1ParamGm.GetValue(wsInfoOffset + LD_NEED_FD_INDEX);
            isS2End = vec1ParamGm.GetValue(wsInfoOffset + LD_IS_S2_END_INDEX);
        }

        // mrg不足4个list的数据
        if (acc_list_num != 1) {
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = BASE_TOPK;
            params.elementLengths[1] = BASE_TOPK;
            params.elementLengths[2] = BASE_TOPK;
            params.elementLengths[3] = BASE_TOPK;
            params.ifExhaustedSuspension = true;
            if (acc_list_num == 2) {
                params.validBit = 0b0011;
            } else if (acc_list_num == 3) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
            srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
            srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
            PipeBarrier<PIPE_V>();
        }

        // 搬出
        LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
        LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();
        if (!constInfo_.returnValue) {
            const bool standardMode =
                requestStateGm.GetValue(bIdx) == LIMConfig::REQUEST_STATE_NON_OFFLOAD;
            if (!standardMode) {
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
                // The final LD owner may differ from the later eviction owner.
                // Publish through DMA before the following outer SyncAll.
                outValueUb.SetValue(
                    0U, curValueIdxUb.GetValue((BASE_TOPK - 1U) * 2U));
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                DataCopyPad(
                    thresholdScratchGm[
                        (static_cast<uint64_t>(outOffset) / BASE_TOPK) *
                        MTP_THRESHOLD_STRIDE],
                    outValueUb,
                    {1, static_cast<uint16_t>(sizeof(float)), 0, 0});
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                SortTopkBySlotIndex(curValueIdxUb, tmpUb,
                                    s2ActSeq > EXACT_PACKED_SOURCE_TOKENS);
            }
            LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
            DecodeTopkHitMiss(curValueIdxUb, idxULocal1,
                              outValueUb.template ReinterpretCast<int32_t>(),
                              tmpUb.template ReinterpretCast<int32_t>(), outOffset,
                              s2ActSeq > EXACT_PACKED_SOURCE_TOKENS,
                              static_cast<uint32_t>(bIdx),
                              static_cast<uint32_t>(s1Idx),
                              static_cast<uint32_t>(routeCount), s2ActSeq);
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        } else {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
            LocalTensor<K_T> valueULocal1 = outValueUb.template ReinterpretCast<K_T>();
            Cast(valueULocal1, outValueUb, RoundMode::CAST_ROUND, constInfo_.sparseCount);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(int32_t)), 0, 0});
            DataCopyPad(valueOutGm[outOffset], valueULocal1,
                        {1, static_cast<uint16_t>(constInfo_.sparseCount * sizeof(K_T)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}
} // namespace LIKernel
#endif
