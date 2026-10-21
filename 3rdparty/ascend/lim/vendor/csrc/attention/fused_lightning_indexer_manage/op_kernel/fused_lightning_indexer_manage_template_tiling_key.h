/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 */

#ifndef TEMPLATE_TILING_KEY_LI_DECODE_UPDATE_MTP_H_
#define TEMPLATE_TILING_KEY_LI_DECODE_UPDATE_MTP_H_

#include "ascendc/host_api/tiling/template_argument.h"

#define LI_MTP_TPL_FP16 1
#define LI_MTP_TPL_BF16 27

ASCENDC_TPL_ARGS_DECL(FusedLightningIndexerManage,
                      ASCENDC_TPL_DTYPE_DECL(DT, LI_MTP_TPL_FP16, LI_MTP_TPL_BF16));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT, LI_MTP_TPL_FP16)),
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_DTYPE_SEL(DT, LI_MTP_TPL_BF16)), );

#endif
