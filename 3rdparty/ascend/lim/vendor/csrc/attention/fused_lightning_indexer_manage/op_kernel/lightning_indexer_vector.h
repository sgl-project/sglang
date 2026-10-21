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
 * \file lightning_indexer_vector.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_VECTOR_H
#define LIGHTNING_INDEXER_VECTOR_H

#include "kernel_operator.h"

namespace LIServiceVec {
using namespace AscendC;

// The sort payload reserves 15 bits for a cache slot, leaving 17 bits for
// the directly packed portion of a source ID. Long-source LI carries the
// remaining bits in the score tag; keep both widths explicit so consumers do
// not accidentally use the packed width as the logical source-ID limit.
constexpr uint32_t PACKED_SOURCE_BITS = 17;
constexpr uint32_t INDEX_BITS = PACKED_SOURCE_BITS;
constexpr uint32_t INDEX_MASK_SHIFT = 32U - INDEX_BITS;
constexpr uint32_t INDEX_HIGH_BITS = 4;
constexpr uint32_t LONG_SOURCE_BITS = INDEX_BITS + INDEX_HIGH_BITS;
constexpr uint32_t PACKED_SOURCE_MASK = (1U << PACKED_SOURCE_BITS) - 1U;
constexpr uint32_t LONG_SOURCE_MASK = (1U << LONG_SOURCE_BITS) - 1U;
constexpr uint32_t SCORE_TAG_CLEAR_SHIFT = INDEX_HIGH_BITS;
constexpr uint32_t SCORE_TAG_EXTRACT_SHIFT = 32U - INDEX_HIGH_BITS;
constexpr uint32_t INVALID_FLAG_SHIFT = 15;
constexpr int32_t INVALID_SLOT15 = (1 << INVALID_FLAG_SHIFT) - 1;
constexpr int32_t INVALID_SLOT_DELTA = -32768;

constexpr int32_t NEG_INF = 0xFF800000;
constexpr int32_t INVALID_INDEX = -1;
constexpr uint8_t VEC_REPEAT_MAX = 255;
constexpr uint8_t B32_VEC_ELM_NUM = 64;
constexpr uint8_t B32_BLOCK_ALIGN_NUM = 8;
constexpr uint8_t B32_VEC_REPEAT_STRIDE = 8;
constexpr uint64_t VEC_REPEAT_BYTES = 256;
constexpr int32_t CONST_TWO = 2;
constexpr int64_t VALUE_AND_INDEX_NUM = 2;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t MRG_QUE_0 = 0;
constexpr int64_t MRG_QUE_1 = 1;
constexpr int64_t MRG_QUE_2 = 2;
constexpr int64_t MRG_QUE_3 = 3;
constexpr int64_t MRG_BLOCK_2 = 2;
constexpr int64_t MRG_BLOCK_3 = 3;
constexpr int64_t MRG_BLOCK_4 = 4;

template <typename T>
__aicore__ inline void CopyIn(LocalTensor<float> &mmOutUb, LocalTensor<T> &weightsUb, GlobalTensor<float> &mMoutGm,
                              GlobalTensor<T> &weightScaleGm, int64_t MMout_gmoffset, int64_t weights_gmoffset,
                              int64_t groupInner, int64_t s2Inner, int64_t mmUbStride)
{
    // 将MMout_gmoffset copy到UB上
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopymMoutParams;
    dataCopymMoutParams.blockCount = groupInner;
    dataCopymMoutParams.blockLen = s2Inner * sizeof(float);
    dataCopymMoutParams.srcStride = 0;
    dataCopymMoutParams.dstStride = mmUbStride;
    dataCopymMoutParams.rsv = 0;
    AscendC::DataCopyPad(mmOutUb, mMoutGm[MMout_gmoffset], dataCopymMoutParams, padParams);

    // 将weights_gmoffset copy到UB
    AscendC::DataCopyPadExtParams<T> padTParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams dataCopyweightParams;
    dataCopyweightParams.blockCount = 1;
    dataCopyweightParams.blockLen = groupInner * sizeof(T);
    dataCopyweightParams.srcStride = 0;
    dataCopyweightParams.dstStride = 0;
    dataCopyweightParams.rsv = 0;
    AscendC::DataCopyPad(weightsUb, weightScaleGm[weights_gmoffset], dataCopyweightParams, padTParams);
}


template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb, int64_t copyCount)
{
    AscendC::DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = copyCount * sizeof(T);
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    AscendC::DataCopyPad(dstGm, srcUb, dataCopyOutyParams);
}


template <typename T>
__aicore__ inline void DoScale(const LocalTensor<float> &reduceCacheBuf, LocalTensor<float> &mmOutUb,
                               LocalTensor<float> &weightsUb, LocalTensor<T> &weightsTUb, LocalTensor<float> &tmpBuff,
                               int64_t groupInner, int64_t s2Inner, int32_t outerGidx)
{
    // cast bfloat16_t to float
    if constexpr (!IsSameType<T, float>::value) {
        AscendC::Cast(weightsUb, weightsTUb, RoundMode::CAST_NONE, groupInner);
        AscendC::PipeBarrier<PIPE_V>();
    }

    // weight broadcast: [groupInner, 1] -> [groupInner, 8]
    AscendC::Brcb(tmpBuff, weightsUb, LICommon::CeilDiv(groupInner, static_cast<int64_t>(B32_BLOCK_ALIGN_NUM)),
                  {1, B32_VEC_REPEAT_STRIDE});
    AscendC::PipeBarrier<PIPE_V>();

    // do scale: [groupInner, 8] * [groupInner, s2Inner]
    uint64_t countPerRepeat = VEC_REPEAT_BYTES / sizeof(float);
    uint64_t repeatTimes = s2Inner / countPerRepeat;
    for (int32_t i = 0; i < groupInner; i++) {
        if (outerGidx == 0) {
            AscendC::Mul(reduceCacheBuf[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM],
                         countPerRepeat, repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        } else {
            AscendC::Mul(mmOutUb[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * B32_BLOCK_ALIGN_NUM], countPerRepeat,
                         repeatTimes, {1, 1, 0, B32_VEC_REPEAT_STRIDE, B32_VEC_REPEAT_STRIDE, 0});
        }
    }

    if (outerGidx != 0) {
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(reduceCacheBuf, mmOutUb, reduceCacheBuf, groupInner * s2Inner);
    }
    AscendC::PipeBarrier<PIPE_V>();
}


__aicore__ inline uint64_t FindNearestPower2(uint64_t value)
{
    if (value <= CONST_TWO) {
        return value;
    } else {
        const uint64_t pow = 63 - AscendC::ScalarCountLeadingZero(value); // 返回前导0的个数，对于64位整数，最大有效位位置 = 63 - 前导0个数
        return (1 << pow);
    }
}


// dstTensor 需要初始化0
__aicore__ inline void DoReduce(const LocalTensor<float> &srcTensor, LocalTensor<float> &dstTensor, int32_t rNum,
                                int32_t aNum)
{
    if (rNum == 1) {
        AscendC::Adds<float>(dstTensor, srcTensor, 0, aNum);
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }

    uint32_t dichotomizeAddPow = FindNearestPower2(rNum);
    uint32_t dichotomizeAddDiffSize = rNum - dichotomizeAddPow;
    if (dichotomizeAddDiffSize != 0) {
        AscendC::Add(srcTensor, srcTensor, srcTensor[dichotomizeAddPow * aNum], dichotomizeAddDiffSize * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    int32_t nowRows = dichotomizeAddPow;
    while (nowRows > CONST_TWO) {
        nowRows = nowRows / CONST_TWO;
        AscendC::Add(srcTensor, srcTensor, srcTensor[nowRows * aNum], nowRows * aNum);
        AscendC::PipeBarrier<PIPE_V>();
    }
    AscendC::Add(dstTensor, srcTensor, srcTensor[aNum], aNum);
    AscendC::PipeBarrier<PIPE_V>();
}


/**
  src: 传入的初始化空间
  eleNum: 需要初始化的元素个数需为64整数倍，元素将被初始化为交错排布的-inf，-1
 */
__aicore__ inline void InitSortOutBuf(const LocalTensor<float> &src, int64_t eleNum)
{
    uint64_t mask1[2] = {0x5555555555555555, 0};
    uint64_t mask0[2] = {0xaaaaaaaaaaaaaaaa, 0};
    int64_t repeatNum = eleNum / B32_VEC_ELM_NUM;
    int64_t forLoop = repeatNum / VEC_REPEAT_MAX;
    int64_t forRemain = repeatNum % VEC_REPEAT_MAX;
    for (int i = 0; i < forLoop; i++) {
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>(), NEG_INF, mask1, VEC_REPEAT_MAX, 1,
                           B32_VEC_REPEAT_STRIDE);
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>(), INVALID_INDEX, mask0, VEC_REPEAT_MAX, 1,
                           B32_VEC_REPEAT_STRIDE);
    }
    if (forRemain > 0) {
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX * B32_VEC_ELM_NUM], NEG_INF,
                           mask1, forRemain, 1, B32_VEC_REPEAT_STRIDE);
        AscendC::Duplicate(src.template ReinterpretCast<int32_t>()[forLoop * VEC_REPEAT_MAX * B32_VEC_ELM_NUM],
                           INVALID_INDEX, mask0, forRemain, 1, B32_VEC_REPEAT_STRIDE);
    }
    AscendC::PipeBarrier<PIPE_V>();
}


/**
  src: logits和索引，前logitsNum为logits，后logitsNum为索引
  tmp: 计算使用到的临时空间，大小与src一致
  logitsNum: 排序的元素个数, 暂只支持[128,256,384,512,1024,2048,4096]
 */
__aicore__ inline void SortAll(LocalTensor<float> &src, LocalTensor<float> &tmp, int64_t logitsNum)
{
    int64_t sort32Repeats = logitsNum / BLOCK_BYTES;
    AscendC::Sort32(tmp, src, src[logitsNum].ReinterpretCast<uint32_t>(), sort32Repeats);
    AscendC::PipeBarrier<PIPE_V>();

    int64_t mrgGroups = sort32Repeats;
    int64_t mrgElements = BLOCK_BYTES;
    int64_t i = 0;
    AscendC::LocalTensor<float> srcTensor;
    AscendC::LocalTensor<float> dstTensor;
    while (true) {
        if (i % CONST_TWO == 0) {
            srcTensor = tmp;
            dstTensor = src;
        } else {
            srcTensor = src;
            dstTensor = tmp;
        }
        AscendC::MrgSort4Info params;
        params.elementLengths[0] = mrgElements;
        params.elementLengths[MRG_QUE_1] = mrgElements;
        params.elementLengths[MRG_QUE_2] = mrgElements;
        params.elementLengths[MRG_QUE_3] = mrgElements;
        params.ifExhaustedSuspension = false;
        params.validBit = 0b1111;

        AscendC::MrgSortSrcList<float> srcList;
        srcList.src1 = srcTensor[0];
        srcList.src2 = srcTensor[MRG_QUE_1 * VALUE_AND_INDEX_NUM * mrgElements];
        srcList.src3 = srcTensor[MRG_QUE_2 * VALUE_AND_INDEX_NUM * mrgElements];
        srcList.src4 = srcTensor[MRG_QUE_3 * VALUE_AND_INDEX_NUM * mrgElements];
        if (mrgGroups <= MRG_BLOCK_4) {
            params.repeatTimes = 1;
            if (mrgGroups == 1) {
                break;
            } else if (mrgGroups == MRG_BLOCK_2) {
                params.validBit = 0b0011;
            } else if (mrgGroups == MRG_BLOCK_3) {
                params.validBit = 0b0111;
            } else if (mrgGroups == MRG_BLOCK_4) {
                params.validBit = 0b1111;
            }
            AscendC::MrgSort<float>(dstTensor, srcList, params);
            i += 1;
            break;
        } else {
            params.repeatTimes = mrgGroups / MRG_BLOCK_4;
            AscendC::MrgSort<float>(dstTensor, srcList, params);
            i += 1;
            mrgElements = mrgElements * MRG_BLOCK_4;
            mrgGroups = mrgGroups / MRG_BLOCK_4;
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
    if (i % CONST_TWO == 0) {
        AscendC::DataCopy(src, tmp, logitsNum * VALUE_AND_INDEX_NUM);
        AscendC::PipeBarrier<PIPE_V>();
    }
}


/**
  dst: 输出全排序的结果，排布方式为value，index
  srcValue：输入的待排序浮点数
  srcIndex：浮点数的索引
  tmp: 计算使用到的临时空间，大小为srcValue+srcIndex
  logitsNum: 排序的元素个数
 */
__aicore__ inline void SortAll(LocalTensor<float> &dst, LocalTensor<float> &srcValue, LocalTensor<uint32_t> &srcIndex,
                               LocalTensor<float> &tmpTensor, int64_t logitsNum)
{
    int64_t sort32Repeats = logitsNum / BLOCK_BYTES;
    AscendC::Sort<float, true>(dst, srcValue, srcIndex, tmpTensor, sort32Repeats);
    AscendC::PipeBarrier<PIPE_V>();
}


/**
  mrgDst: 合并进的Tensor
  mrgSrc: 待合并的Tensor
  tmpTensor：空间为mrgDst+mrgSrc
 */
__aicore__ inline void MergeSort(const LocalTensor<float> &mrgDst, int32_t mrgDstNum, LocalTensor<float> &mrgSrc,
                                 int32_t mrgSrcNum, LocalTensor<float> &tmpTensor)
{
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = mrgSrcNum;
    params.elementLengths[1] = mrgDstNum;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b0011;
    params.repeatTimes = 1;

    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = mrgSrc;
    srcList.src2 = mrgDst;

    AscendC::MrgSort<float>(tmpTensor, srcList, params);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopy(mrgDst, tmpTensor, mrgDstNum * VALUE_AND_INDEX_NUM);
    AscendC::PipeBarrier<PIPE_V>();
}

/**
 * @brief Merge a new TopK chunk into an existing TopK while preserving the
 *        existing sequence's order when score keys compare equal.
 *
 * The standard LI interface exposes ordered TopK source IDs.  For FP16,
 * quantization makes equal scores common; the hardware merge resolves those
 * ties in source-list order.  Keep the already-processed (earlier source
 * chunk) list first so Q=5..7 follows the native LI stable order.
 */
__aicore__ inline void MergeSortExistingFirst(const LocalTensor<float> &mrgDst, int32_t mrgDstNum,
                                              LocalTensor<float> &mrgSrc, int32_t mrgSrcNum,
                                              LocalTensor<float> &tmpTensor)
{
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = mrgDstNum;
    params.elementLengths[1] = mrgSrcNum;
    params.ifExhaustedSuspension = false;
    params.validBit = 0b0011;
    params.repeatTimes = 1;

    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = mrgDst;
    srcList.src2 = mrgSrc;

    AscendC::MrgSort<float>(tmpTensor, srcList, params);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::DataCopy(mrgDst, tmpTensor, mrgDstNum * VALUE_AND_INDEX_NUM);
    AscendC::PipeBarrier<PIPE_V>();
}


/**
 * @brief 合并基础块函数
 * @param dst 归并后的输出, 大小为blockNum * basicBlockSize * 2 * sizeof(float)
 * @param src 基本块输入
 * @param blockNum 基本块的数量
 * @param basicBlockSize 基础块的大小
 * @return 无
 */
__aicore__ inline void MrgBasicBlock(const LocalTensor<float> &dst, const LocalTensor<float> &src, int64_t blockNum,
                                     int64_t basicBlockSize)
{
    // 初始化合并排序参数
    AscendC::MrgSort4Info params;
    params.elementLengths[MRG_QUE_0] = basicBlockSize;
    params.elementLengths[MRG_QUE_1] = basicBlockSize;
    params.elementLengths[MRG_QUE_2] = basicBlockSize;
    params.elementLengths[MRG_QUE_3] = basicBlockSize;
    params.ifExhaustedSuspension = false;
    // 根据块的数量设置有效位
    if (blockNum == MRG_BLOCK_2) {
        params.validBit = 0b0011;
    } else if (blockNum == MRG_BLOCK_3) {
        params.validBit = 0b0111;
    } else if (blockNum == MRG_BLOCK_4) {
        params.validBit = 0b1111;
    } else {
        AscendC::DataCopy(dst, src, basicBlockSize * VALUE_AND_INDEX_NUM);
        return;
    }
    // 初始化源列表
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = src[0];
    srcList.src2 = src[basicBlockSize * VALUE_AND_INDEX_NUM * MRG_QUE_1];
    srcList.src3 = src[basicBlockSize * VALUE_AND_INDEX_NUM * MRG_QUE_2];
    srcList.src4 = src[basicBlockSize * VALUE_AND_INDEX_NUM * MRG_QUE_3];
    // 执行合并排序
    AscendC::MrgSort<float>(dst, srcList, params);
}


/**
 * @brief 从两个队列中选择topk
 * @param dst 已经归并好的topk数据
 * @param needsMerging 需要合并的有序数据
 * @param tmp 临时空间
 * @param topk topk的元素个数
 * @param mergSize 待合并的元素个数
 * @return 无
 */
template <bool needMrg = true>
__aicore__ inline void SparseTopK(const LocalTensor<float> &dst, const LocalTensor<float> &needsMerging,
                                  const LocalTensor<float> &tmp, int64_t topk, int64_t mergSize)
{
    // 如果不需要合并，则直接复制数据
    if (!needMrg) {
        AscendC::DataCopy(dst, needsMerging, mergSize * VALUE_AND_INDEX_NUM);
        return;
    }
    // 初始化合并排序参数
    AscendC::MrgSort4Info params;
    params.elementLengths[0] = topk;
    params.elementLengths[1] = mergSize;
    params.ifExhaustedSuspension = (topk == mergSize);
    params.validBit = 0b0011;
    // 初始化源列表
    AscendC::MrgSortSrcList<float> srcList;
    srcList.src1 = dst;
    srcList.src2 = needsMerging;
    // 执行合并排序
    AscendC::MrgSort<float>(tmp, srcList, params);
    // 将结果复制到目标张量
    AscendC::DataCopy(dst, tmp, topk * VALUE_AND_INDEX_NUM);
}


__aicore__ inline void ExtractIndex(const LocalTensor<uint32_t> &idxULocal, const LocalTensor<uint32_t> &sortLocal,
                                    int64_t extractNum)
{
    AscendC::GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = Ceil(extractNum * sizeof(float) * VALUE_AND_INDEX_NUM, VEC_REPEAT_BYTES);
    gatherMaskParams.src0BlockStride = 1;
    gatherMaskParams.src0RepeatStride = B32_VEC_REPEAT_STRIDE;
    gatherMaskParams.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;    // 用于保存筛选后保留下来的元素个数
    uint8_t src1Pattern = 2; // 固定模式2,表示筛选出奇数索引的数
    AscendC::GatherMask(idxULocal, sortLocal, src1Pattern, false, static_cast<uint32_t>(0), gatherMaskParams, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ExtractScoreBits(const LocalTensor<uint32_t> &scoreBitsLocal,
                                        const LocalTensor<uint32_t> &sortLocal, int64_t extractNum)
{
    AscendC::GatherMaskParams params;
    params.repeatTimes = Ceil(extractNum * sizeof(float) * VALUE_AND_INDEX_NUM, VEC_REPEAT_BYTES);
    params.src0BlockStride = 1;
    params.src0RepeatStride = B32_VEC_REPEAT_STRIDE;
    params.src1RepeatStride = 0;
    uint64_t rsvdCnt = 0;
    AscendC::GatherMask(scoreBitsLocal, sortLocal, static_cast<uint8_t>(1), false,
                        static_cast<uint32_t>(0), params, rsvdCnt);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void DecodePackedSlot(const LocalTensor<int32_t> &slotLocal,
                                        const LocalTensor<uint32_t> &payloadLocal,
                                        const LocalTensor<int32_t> &scratchLocal, int64_t count)
{
    AscendC::ShiftRight(slotLocal.template ReinterpretCast<uint32_t>(), payloadLocal, INDEX_BITS, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(scratchLocal, slotLocal, static_cast<int32_t>(1), count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(scratchLocal.template ReinterpretCast<uint32_t>(),
                        scratchLocal.template ReinterpretCast<uint32_t>(), INVALID_FLAG_SHIFT, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(scratchLocal, scratchLocal, INVALID_SLOT_DELTA, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(slotLocal, slotLocal, scratchLocal, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void DecodePackedIndex(const LocalTensor<uint32_t> &indexLocal,
                                         const LocalTensor<uint32_t> &scoreTagLocal, int64_t count,
                                         bool hasLongIndexTag)
{
    AscendC::ShiftLeft(indexLocal, indexLocal, INDEX_MASK_SHIFT, count);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ShiftRight(indexLocal, indexLocal, INDEX_MASK_SHIFT, count);
    AscendC::PipeBarrier<PIPE_V>();
    if (hasLongIndexTag) {
        AscendC::ShiftLeft(scoreTagLocal, scoreTagLocal, SCORE_TAG_EXTRACT_SHIFT, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftRight(scoreTagLocal, scoreTagLocal, SCORE_TAG_EXTRACT_SHIFT, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(scoreTagLocal, scoreTagLocal, INDEX_BITS, count);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(indexLocal.template ReinterpretCast<int32_t>(),
                     indexLocal.template ReinterpretCast<int32_t>(),
                     scoreTagLocal.template ReinterpretCast<int32_t>(), count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}


template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt)
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<event>(eventId);
    AscendC::WaitFlag<event>(eventId);
}

} // namespace LIServiceVec
#endif // LIGHTNING_INDEXER_VECTOR_H
