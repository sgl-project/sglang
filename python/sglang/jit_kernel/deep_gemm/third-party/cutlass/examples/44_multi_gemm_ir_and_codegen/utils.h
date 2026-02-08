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
#define TI(tag) \
    cudaEvent_t _event_start_ ##tag; \
    cudaEvent_t _event_end_ ##tag; \
    float _event_time_ ##tag; \
    cudaEventCreate(& _event_start_ ##tag); \
    cudaEventCreate(& _event_end_ ##tag); \
    cudaEventRecord(_event_start_ ##tag);

#define TO(tag, str, times) \
    cudaEventRecord(_event_end_ ##tag); \
    cudaEventSynchronize(_event_end_ ##tag); \
    cudaEventElapsedTime(&_event_time_ ##tag, _event_start_ ##tag, _event_end_ ##tag); \
    float _event_time_once_ ##tag = _event_time_ ##tag / times; \
    printf("%20s:\t %10.3fus\t", str, _event_time_once_ ##tag * 1000); \
    cudaDeviceSynchronize(); \
    printf("%20s string: %s\n",str, cudaGetErrorString(cudaGetLastError()));

template<typename T>
struct memory_unit{
    T* host_ptr;
    T* device_ptr;
    int size_bytes;
    int elements;
    void h2d(){
        cudaMemcpy(device_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice);
    }
    void d2h(){
        cudaMemcpy(host_ptr, device_ptr, size_bytes, cudaMemcpyDeviceToHost);
    }
    void free_all(){
        free(host_ptr);
        cudaFree(device_ptr);
    }
    memory_unit(int elements_): size_bytes(elements_ * sizeof(T)), elements(elements_){
        host_ptr = (T*) malloc(elements_ * sizeof(T));
        cudaMalloc((void**)&device_ptr, elements_ * sizeof(T));
    }
    void init(int abs_range = 1){
        for(int i = 0; i < elements; i++){
            host_ptr[i] = T(rand() % 100 / float(100)  * 2 * abs_range - abs_range);
        }
        h2d();
    }
};

template<typename T>
int check_result(T * a, T * b, int N){
    int cnt = 0;
    for(int i = 0; i < N; i ++){
        float std = float(a[i]);
        float my = float(b[i]);

        if(abs(std - my) / abs(std) > 1e-2)
        {
            // printf("my: %f , std: %f\n", my, std);
            cnt++;
        }

    }
    printf("total err: %d / %d\n", cnt, N);
    return cnt;
}
