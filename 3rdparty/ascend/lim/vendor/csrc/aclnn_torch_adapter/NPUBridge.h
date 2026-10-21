// Copyright (c) 2020, Huawei Technologies Co., Ltd
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <c10/core/StorageImpl.h>
#include "NPUStorageImpl.h"

namespace vllm_ascend
{

    class NPUBridge
    {
    public:
        // at::tensor to NPUStorageImpl
        static NPUStorageImpl *GetNpuStorageImpl(const at::Tensor &tensor);

        // c10::StorageImpl to NPUStorageImpl
        static NPUStorageImpl *GetNpuStorageImpl(c10::StorageImpl *storageImpl);

        // c10::Storage to NPUStorageImpl
        static NPUStorageImpl *GetNpuStorageImpl(c10::Storage &&storage);

        // tensor to NPUStorageDesc
        static NPUStorageDesc &GetNpuStorageImplDesc(const at::Tensor &tensor);
    };
}
