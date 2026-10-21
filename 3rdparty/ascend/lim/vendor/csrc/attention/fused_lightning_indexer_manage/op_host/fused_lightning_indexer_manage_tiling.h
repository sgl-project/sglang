/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef FUSED_LIGHTNING_INDEXER_MANAGE_TILING_H_
#define FUSED_LIGHTNING_INDEXER_MANAGE_TILING_H_

#include "error/ops_error.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "../op_kernel/fused_lightning_indexer_manage_constants.h"

namespace optiling {

struct MtpRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct MtpTensorParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

constexpr uint32_t WEIGHTS_INDEX = 0;
constexpr uint32_t QUERY_DEQUANT_SCALE_INDEX = 1;
constexpr uint32_t QUERY_INDEX = 2;
constexpr uint32_t KEY_DEQUANT_SCALE_INDEX = 3;
constexpr uint32_t KEY_INDEX = 4;
constexpr uint32_t BLOCK_TABLE_INDEX = 5;
constexpr uint32_t ACTUAL_SEQ_Q_INDEX = 6;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 7;
constexpr uint32_t OFFLOAD_SEQ_K_INDEX = 8;
constexpr uint32_t CACHE_TOKENS_INDEX = 9;
constexpr uint32_t REQUEST_STATE_INDEX = 10;
constexpr uint32_t REQ_POOL_ENTRIES_INDEX = 11;
constexpr uint32_t CACHE_SLOTS_INDEX = 12;
constexpr uint32_t TOPK_INDEX = LIMConfig::TOPK_SOURCE_OUTPUT_INDEX;
constexpr uint32_t TOPK_SLOTS_INDEX = LIMConfig::TOPK_SLOT_OUTPUT_INDEX;
constexpr uint32_t MISS_COUNT_INDEX = 2;
constexpr uint32_t CACHE_SLOTS_OUT_INDEX = 3;
constexpr uint32_t MTP_TOPK_MISS_COUNT_INDEX = LIMConfig::TOPK_MISS_COUNT_OUTPUT_INDEX;
constexpr uint32_t MTP_MISS_SRC_INDEX = LIMConfig::MISS_SOURCE_OUTPUT_INDEX;
constexpr uint32_t MTP_MISS_SLOTS_INDEX = LIMConfig::MISS_SLOT_OUTPUT_INDEX;
constexpr uint32_t MTP_MISS_COUNT_INDEX = LIMConfig::MISS_COUNT_OUTPUT_INDEX;
constexpr uint32_t MTP_CACHE_SLOTS_OUT_INDEX = LIMConfig::CACHE_SLOTS_OUTPUT_INDEX;

constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
constexpr uint32_t DIM_NUM_ONE = 1;
constexpr uint32_t DIM_NUM_TWO = 2;
constexpr uint32_t DIM_NUM_THREE = 3;
constexpr uint32_t DIM_NUM_FOUR = 4;

constexpr uint32_t DECODE_N2 = LIMConfig::KEY_HEADS;
constexpr uint32_t DECODE_HEAD_DIM = LIMConfig::HEAD_DIM;
constexpr uint32_t DECODE_SPARSE_COUNT = LIMConfig::TOPK;
constexpr uint32_t DECODE_OUTPUT_CAPACITY = LIMConfig::TOPK;

BEGIN_TILING_DATA_DEF(FusedLightningIndexerManageTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, tSize)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, poolSize)
TILING_DATA_FIELD_DEF(uint32_t, n1Size)
TILING_DATA_FIELD_DEF(uint32_t, cacheSlotsSize)
TILING_DATA_FIELD_DEF(uint32_t, scheduleMode)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(FusedLightningIndexerManage, FusedLightningIndexerManageTilingData)

struct FusedLightningIndexerManageCompileInfo {};

struct FusedLightningIndexerManageParaInfo {
    MtpRequiredParaInfo query = {nullptr, nullptr};
    MtpRequiredParaInfo key = {nullptr, nullptr};
    MtpRequiredParaInfo weights = {nullptr, nullptr};
    MtpTensorParaInfo reqPoolEntries = {nullptr, nullptr};
    MtpRequiredParaInfo cacheSlots = {nullptr, nullptr};
    MtpTensorParaInfo cacheTokens = {nullptr, nullptr};
    MtpTensorParaInfo actualSeqLengths = {nullptr, nullptr};
    MtpTensorParaInfo blockTable = {nullptr, nullptr};
    MtpRequiredParaInfo topkIndexOut = {nullptr, nullptr};
    MtpRequiredParaInfo topkSlotsOut = {nullptr, nullptr};
    MtpRequiredParaInfo topkMissCountOut = {nullptr, nullptr};
    MtpRequiredParaInfo missCountOut = {nullptr, nullptr};
    MtpRequiredParaInfo missSrcOut = {nullptr, nullptr};
    MtpRequiredParaInfo missSlotsOut = {nullptr, nullptr};
    MtpRequiredParaInfo cacheSlotsOut = {nullptr, nullptr};
};

class FusedLightningIndexerManageTilingInfo {
public:
    const char *opName = nullptr;
    fe::PlatFormInfos *platformInfo = nullptr;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
    FusedLightningIndexerManageParaInfo opParamInfo;

    uint32_t bSize = 0;
    uint32_t tSize = 0;
    uint32_t n1Size = LIMConfig::QUERY_HEADS_SMALL;
    uint32_t n2Size = DECODE_N2;
    uint32_t s2Size = 0;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t poolSize = 0;
    uint32_t cacheSlotsSize = 0;
    uint32_t usedCoreNum = 0;

    ge::DataType inputQType = ge::DT_FLOAT16;
};

class FusedLightningIndexerManageTiling {
public:
    explicit FusedLightningIndexerManageTiling(gert::TilingContext *context, bool mtp = true)
        : context_(context), mtp_(mtp) {};
    ge::graphStatus ParseAndCheck(FusedLightningIndexerManageTilingInfo &tilingInfo);
    ge::graphStatus DoTiling(FusedLightningIndexerManageTilingInfo *tilingInfo);

private:
    ge::graphStatus GetNpuInfo(FusedLightningIndexerManageTilingInfo &tilingInfo) const;
    ge::graphStatus GetTensorInfo(FusedLightningIndexerManageTilingInfo &tilingInfo) const;
    ge::graphStatus CheckDtype(const FusedLightningIndexerManageTilingInfo &tilingInfo) const;
    ge::graphStatus CheckShape(FusedLightningIndexerManageTilingInfo &tilingInfo) const;

    gert::TilingContext *context_ = nullptr;
    FusedLightningIndexerManageTilingData tilingData_;
    bool mtp_ = true;
};

} // namespace optiling
#endif // FUSED_LIGHTNING_INDEXER_MANAGE_TILING_H_
