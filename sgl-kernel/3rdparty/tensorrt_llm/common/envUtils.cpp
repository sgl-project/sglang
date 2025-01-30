/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "envUtils.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include <cstdlib>

namespace tensorrt_llm::common
{

std::optional<int32_t> getIntEnv(char const* name)
{
    char const* const env = std::getenv(name);
    if (env == nullptr)
    {
        return std::nullopt;
    }
    int32_t const val = std::stoi(env);
    if (val <= 0)
    {
        return std::nullopt;
    }
    return {val};
};

// Returns true if the env variable exists and is set to "1"
static bool getBoolEnv(char const* name)
{
    char const* env = std::getenv(name);
    return env && env[0] == '1' && env[1] == '\0';
}

// XQA kernels (optimized kernels for generation phase).
bool forceXQAKernels()
{
    static bool const forceXQA = (getIntEnv("TRTLLM_FORCE_XQA").value_or(0) != 0);
    return forceXQA;
}

std::optional<bool> getEnvEnableXQAJIT()
{
    static bool init = false;
    static bool exists = false;
    static bool enableXQAJIT = false;
    if (!init)
    {
        init = true;
        char const* enable_xqa_jit_var = std::getenv("TRTLLM_ENABLE_XQA_JIT");
        if (enable_xqa_jit_var)
        {
            exists = true;
            if (enable_xqa_jit_var[0] == '1' && enable_xqa_jit_var[1] == '\0')
            {
                enableXQAJIT = true;
            }
        }
    }
    if (exists)
    {
        return enableXQAJIT;
    }
    else
    {
        return std::nullopt;
    }
}

// Tune the number of blocks per sequence for accuracy/performance purpose.
bool getEnvMmhaMultiblockDebug()
{
    static bool init = false;
    static bool forceMmhaMaxSeqLenTile = false;
    if (!init)
    {
        init = true;
        char const* enable_mmha_debug_var = std::getenv("TRTLLM_ENABLE_MMHA_MULTI_BLOCK_DEBUG");
        if (enable_mmha_debug_var)
        {
            if (enable_mmha_debug_var[0] == '1' && enable_mmha_debug_var[1] == '\0')
            {
                forceMmhaMaxSeqLenTile = true;
            }
        }
    }
    return forceMmhaMaxSeqLenTile;
}

int getEnvMmhaBlocksPerSequence()
{
    static bool init = false;
    static int mmhaBlocksPerSequence = 0;
    if (!init)
    {
        init = true;
        char const* mmhaBlocksPerSequenceEnv = std::getenv("TRTLLM_MMHA_BLOCKS_PER_SEQUENCE");
        if (mmhaBlocksPerSequenceEnv)
        {
            mmhaBlocksPerSequence = std::atoi(mmhaBlocksPerSequenceEnv);
            if (mmhaBlocksPerSequence <= 0)
            {
                TLLM_LOG_WARNING("Invalid value for TRTLLM_MMHA_BLOCKS_PER_SEQUENCE. Will use default values instead!");
            }
        }
    }
    return mmhaBlocksPerSequence;
}

int getEnvMmhaKernelBlockSize()
{
    static bool init = false;
    static int mmhaKernelBlockSize = 0;
    if (!init)
    {
        init = true;
        char const* mmhaKernelBlockSizeEnv = std::getenv("TRTLLM_MMHA_KERNEL_BLOCK_SIZE");
        if (mmhaKernelBlockSizeEnv)
        {
            mmhaKernelBlockSize = std::atoi(mmhaKernelBlockSizeEnv);
            if (mmhaKernelBlockSize <= 0)
            {
                TLLM_LOG_WARNING("Invalid value for TRTLLM_MMHA_KERNEL_BLOCK_SIZE. Will use default values instead!");
            }
        }
    }
    return mmhaKernelBlockSize;
}

bool getEnvEnablePDL()
{
    static bool init = false;
    static bool enablePDL = false;
    if (!init)
    {
        init = true;
        // PDL only available when arch >= 90
        if (getSMVersion() >= 90)
        {
            // PDL will be enabled by setting the env variables `TRTLLM_ENABLE_PDL` to `1`
            enablePDL = getBoolEnv("TRTLLM_ENABLE_PDL");
        }
    }
    return enablePDL;
}

bool getEnvUseUCXKvCache()
{
    static bool const useUCXKVCache = getBoolEnv("TRTLLM_USE_UCX_KVCACHE");
    return useUCXKVCache;
}

std::string getEnvUCXInterface()
{
    static bool init = false;
    static std::string ucxInterface;
    if (!init)
    {
        init = true;
        {
            char const* ucx_interface = std::getenv("TRTLLM_UCX_INTERFACE");
            if (ucx_interface)
            {
                ucxInterface = ucx_interface;
            }
        }
    }
    return ucxInterface;
}

bool getEnvDisaggLayerwise()
{
    static bool const disaggLayerwise = getBoolEnv("TRTLLM_DISAGG_LAYERWISE");
    return disaggLayerwise;
}

bool getEnvParallelCacheSend()
{
    static bool const parallelCacheSend = getBoolEnv("TRTLLM_PARALLEL_CACHE_SEND");
    return parallelCacheSend;
}

bool getEnvRequestKVCacheSerial()
{
    static bool const requestKVCacheSerial = getBoolEnv("TRTLLM_REQUEST_KV_CACHE_SERIAL");
    return requestKVCacheSerial;
}

bool getEnvDisableKVCacheTransferOverlap()
{
    static bool const disableKVCacheTransferOverlap = getBoolEnv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP");
    return disableKVCacheTransferOverlap;
}

bool getEnvDisableReceiveKVCacheParallel()
{
    static bool const disableReceiveParallel = getBoolEnv("TRTLLM_DISABLE_KVCACHE_RECEIVE_PARALLEL");
    return disableReceiveParallel;
}

} // namespace tensorrt_llm::common
