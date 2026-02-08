/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once
#include <cuda_fp16.h>

template <typename T>
__device__
T add(T const & a, T const &b){
    return  (a + b);
}

template <>
__device__
half2 add(half2 const & a, half2 const &b){
    return (__hadd2(a,b));
}

template <typename T>
struct RELU{
    __device__
    T operator()(T const & a){
        return  a > T(0) ? a : T(0);
    }
    __device__
    half2 operator()(half2 const & a){
        float2 a_fp32x2 = __half22float2(a);
        a_fp32x2.x = a_fp32x2.x > 0.f ? a_fp32x2.x : 0.f;
        a_fp32x2.y = a_fp32x2.y > 0.f ? a_fp32x2.y : 0.f;
        if(a_fp32x2.x < 0.f || a_fp32x2.y < 0.f)
        printf(" %f %f\n", a_fp32x2.x ,a_fp32x2.y);
        return __float22half2_rn(a_fp32x2);
    }
};

template <typename T>
struct LEAKY_RELU{
    __device__
    T operator()(T const & a, T const & scale = half(1)){
        return  a > T(0) ? a : scale * a;
    }
    __device__
    half2 operator()(half2 const & a, half const & scale = half(1)){
        half2 zero = __half2half2(half(0));
        half2 gt_zero = __hge2(a, zero);
        half2 le_zero = __hle2(a, zero);


        half2 scale_f16x2 = __half2half2(scale);
        half2 mask_scale_f16x2 = __hfma2(le_zero, scale_f16x2, gt_zero);
        return __hmul2(a, mask_scale_f16x2);
    }
};

template <int N, int BLOCKDIM>
__global__ void leaky_and_activation(half* inout, half* bias, half scale, bool mat_bias){

    constexpr bool N_MOD_2 = N & 1 ? false : true;

    using Access_tp = typename std::conditional<N_MOD_2, half2, half>::type;

    constexpr int Access_elements = sizeof(Access_tp) / sizeof(half);

    constexpr int iter = (N + (BLOCKDIM * Access_elements) - 1 ) / (BLOCKDIM * Access_elements);

    LEAKY_RELU<half> Act;
    Access_tp src_v[iter];
    Access_tp bias_v[iter];

    int batch_id = blockIdx.y;
    int batch_offset = batch_id * gridDim.x * N;

    for(int i = 0; i < iter; i++){
        int idx = (i * BLOCKDIM + threadIdx.x) * Access_elements;
        if (idx < N){
            src_v[i] = *reinterpret_cast<Access_tp*>(inout + blockIdx.x * N + idx + batch_offset);
            if (mat_bias)
                bias_v[i] = *reinterpret_cast<Access_tp*>(bias + blockIdx.x * N + idx + batch_offset);
            else
                bias_v[i] = *reinterpret_cast<Access_tp*>(bias + idx + batch_id * N);
            *reinterpret_cast<Access_tp*>(inout + blockIdx.x * N + idx + batch_offset) = Act(add(src_v[i],bias_v[i]),scale); 
        }
        
    }
}



template <int N, int BLOCKDIM>
__global__ void leaky_and_activation(half* inout, half scale){

    constexpr bool N_MOD_2 = N & 1 ? false : true;

    using Access_tp = typename std::conditional<N_MOD_2, half2, half>::type;

    constexpr int Access_elements = sizeof(Access_tp) / sizeof(half);

    constexpr int iter = (N + (BLOCKDIM * Access_elements) - 1 ) / (BLOCKDIM * Access_elements);

    int batch_id = blockIdx.y;
    int batch_offset = batch_id * gridDim.x * N;

    LEAKY_RELU<half> Act;
    Access_tp src_v[iter];

    for(int i = 0; i < iter; i++){
        int idx = (i * BLOCKDIM + threadIdx.x) * Access_elements;
        if (idx < N){
            src_v[i] = *reinterpret_cast<Access_tp*>(inout + blockIdx.x * N + idx + batch_offset);
            *reinterpret_cast<Access_tp*>(inout + blockIdx.x * N + idx + batch_offset) = Act(src_v[i], scale);
        }
        
    }
}



template <int N, int BLOCKDIM>
void leaky_and_activation(half* inout, half* bias, int m, int b, half scale, bool mat_bias){

    dim3 grid(m, b);
    if (bias == nullptr)
        leaky_and_activation<N, BLOCKDIM><<<grid , BLOCKDIM>>>(inout, scale);
    else
        leaky_and_activation<N, BLOCKDIM><<<grid , BLOCKDIM>>>(inout, bias, scale, mat_bias);
}

template <int N, int BLOCKDIM>
__global__ void relu_and_activation(half* inout, half* bias, bool mat_bias){

    constexpr bool N_MOD_2 = N & 1 ? false : true;

    using Access_tp = typename std::conditional<N_MOD_2, half2, half>::type;

    constexpr int Access_elements = sizeof(Access_tp) / sizeof(half);

    constexpr int iter = (N + (BLOCKDIM * Access_elements) - 1 ) / (BLOCKDIM * Access_elements);

    RELU<half> Act;
    Access_tp src_v[iter];
    Access_tp bias_v[iter];
    
    int batch_id = blockIdx.y;
    int batch_offset = batch_id * gridDim.x * N;

    for(int i = 0; i < iter; i++){
        int idx = (i * BLOCKDIM + threadIdx.x) * Access_elements;
        if (idx < N){
            src_v[i] = *reinterpret_cast<Access_tp*>(inout + blockIdx.x * N + idx + batch_offset);
            if (mat_bias)
                bias_v[i] = *reinterpret_cast<Access_tp*>(bias + blockIdx.x * N + idx + batch_offset);
            else
                bias_v[i] = *reinterpret_cast<Access_tp*>(bias + idx + batch_id * N);
            *reinterpret_cast<Access_tp*>(inout + blockIdx.x * N + idx + batch_offset) = Act(add(src_v[i],bias_v[i])); 
        }
        
    }
}



template <int N, int BLOCKDIM>
__global__ void relu_and_activation(half* inout){

    constexpr bool N_MOD_2 = N & 1 ? false : true;

    using Access_tp = typename std::conditional<N_MOD_2, half2, half>::type;

    constexpr int Access_elements = sizeof(Access_tp) / sizeof(half);

    constexpr int iter = (N + (BLOCKDIM * Access_elements) - 1 ) / (BLOCKDIM * Access_elements);

    int batch_id = blockIdx.y;
    int batch_offset = batch_id * gridDim.x * N;

    RELU<half> Act;
    Access_tp src_v[iter];

    for(int i = 0; i < iter; i++){
        int idx = (i * BLOCKDIM + threadIdx.x) * Access_elements;
        if (idx < N){
            src_v[i] = *reinterpret_cast<Access_tp*>(inout + blockIdx.x * N + idx + batch_offset);
            *reinterpret_cast<Access_tp*>(inout + blockIdx.x * N + idx + batch_offset) = Act(src_v[i]);
        }
        
    }
}



template <int N, int BLOCKDIM>
void relu_and_activation(half* inout, half* bias, int m, int b, bool mat_bias){
    dim3 grid(m, b);
    if (bias == nullptr)
        relu_and_activation<N, BLOCKDIM><<<grid , BLOCKDIM>>>(inout);
    else
        relu_and_activation<N, BLOCKDIM><<<grid , BLOCKDIM>>>(inout, bias, mat_bias);
}


template <int N, int BLOCKDIM>
__global__ void identity_and_activation(half* inout, half* bias, bool mat_bias){

    constexpr bool N_MOD_2 = N & 1 ? false : true;

    using Access_tp = typename std::conditional<N_MOD_2, half2, half>::type;

    constexpr int Access_elements = sizeof(Access_tp) / sizeof(half);

    constexpr int iter = (N + (BLOCKDIM * Access_elements) - 1 ) / (BLOCKDIM * Access_elements);

    int batch_id = blockIdx.y;
    int batch_offset = batch_id * gridDim.x * N;

    Access_tp src_v[iter];
    Access_tp bias_v[iter];

    for(int i = 0; i < iter; i++){
        int idx = (i * BLOCKDIM + threadIdx.x) * Access_elements;
        if (idx < N){
            src_v[i] = *reinterpret_cast<Access_tp*>(inout + blockIdx.x * N + idx + batch_offset);
            if (mat_bias)
                bias_v[i] = *reinterpret_cast<Access_tp*>(bias + blockIdx.x * N + idx + batch_offset);
            else
                bias_v[i] = *reinterpret_cast<Access_tp*>(bias + idx + batch_id * N);
            *reinterpret_cast<Access_tp*>(inout + blockIdx.x * N + idx + batch_offset) = (add(src_v[i],bias_v[i])); 
        }
        
    }
}

template <int N, int BLOCKDIM>
__global__ void identity_and_activation(half* inout){

    constexpr bool N_MOD_2 = N & 1 ? false : true;

    using Access_tp = typename std::conditional<N_MOD_2, half2, half>::type;

    constexpr int Access_elements = sizeof(Access_tp) / sizeof(half);

    constexpr int iter = (N + (BLOCKDIM * Access_elements) - 1 ) / (BLOCKDIM * Access_elements);

    int batch_id = blockIdx.y;
    int batch_offset = batch_id * gridDim.x * N;
    Access_tp src_v[iter];

    for(int i = 0; i < iter; i++){
        int idx = (i * BLOCKDIM + threadIdx.x) * Access_elements;
        if (idx < N){
            src_v[i] = *reinterpret_cast<Access_tp*>(inout + blockIdx.x * N + idx + batch_offset);
            *reinterpret_cast<Access_tp*>(inout + blockIdx.x * N + idx + batch_offset) = (src_v[i]);
        }
        
    }
}

template <int N, int BLOCKDIM>
void identity_and_activation(half* inout, half* bias, int m, int b, bool mat_bias){
    dim3 grid(m, b);
    if (bias == nullptr)
        identity_and_activation<N, BLOCKDIM><<<grid , BLOCKDIM>>>(inout);
    else
        identity_and_activation<N, BLOCKDIM><<<grid , BLOCKDIM>>>(inout, bias, mat_bias);
}
