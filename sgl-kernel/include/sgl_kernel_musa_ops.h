/*
 * Copyright (c) 2020-2026, Moore Threads Technology Co., Ltd("Moore Threads").
 * All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/torch.h>

#include <optional>

void batched_rotary_embedding_contiguous(
    torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox,
    int64_t rot_dim,
    torch::Tensor& cos_sin_cache_offsets);

void rotary_embedding_contiguous(
    torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox);

void fused_moe_gemv(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& C,
    const c10::optional<torch::Tensor>& A_scale,
    const c10::optional<torch::Tensor>& B_scale,
    torch::Tensor& topk_weights,
    torch::Tensor& topk_ids,
    bool mul_routed_weight,
    int64_t topk,
    bool use_int4_w4a16,
    bool use_swigelu);

void musa_fused_gemv(
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& C,
    const c10::optional<torch::Tensor>& A_scale,
    const c10::optional<torch::Tensor>& B_scale,
    bool use_int4_w4a16,
    bool use_swigelu,
    bool use_rms_norm,
    const c10::optional<torch::Tensor>& gamma,
    double eps);

void fused_mul_add(torch::Tensor& output, torch::Tensor& self, torch::Tensor& bias, double scale);

void musa_top_k_top_p_sampling_from_probs(
    at::Tensor probs,
    at::Tensor output,
    std::optional<at::Tensor> maybe_indices,
    std::optional<at::Tensor> maybe_top_k_arr,
    double top_k_val,
    std::optional<at::Tensor> maybe_top_p_arr,
    double top_p_val,
    bool deterministic,
    std::optional<at::Generator> gen);
