/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cute/arch/copy_sm90.hpp>
#include <cute/atom/copy_traits.hpp>
#include <cute/atom/copy_traits_sm75.hpp>

#include <cute/layout.hpp>

namespace cute
{

template <>
struct Copy_Traits<SM90_U32x1_STSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID   = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = typename Copy_Traits<SM75_U32x1_LDSM_N>::DstLayout;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = typename Copy_Traits<SM75_U32x1_LDSM_N>::SrcLayout;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template <>
struct Copy_Traits<SM90_U32x2_STSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID   = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = typename Copy_Traits<SM75_U32x2_LDSM_N>::DstLayout;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = typename Copy_Traits<SM75_U32x2_LDSM_N>::SrcLayout;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template <>
struct Copy_Traits<SM90_U32x4_STSM_N>
{
  // Logical thread id to thread idx (warp)
  using ThrID   = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = typename Copy_Traits<SM75_U32x4_LDSM_N>::DstLayout;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = typename Copy_Traits<SM75_U32x4_LDSM_N>::SrcLayout;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template <>
struct Copy_Traits<SM90_U16x2_STSM_T>
{
  // Logical thread id to thread idx (warp)
  using ThrID   = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = typename Copy_Traits<SM75_U16x2_LDSM_T>::DstLayout;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = typename Copy_Traits<SM75_U16x2_LDSM_T>::SrcLayout;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template <>
struct Copy_Traits<SM90_U16x4_STSM_T>
{
  // Logical thread id to thread idx (warp)
  using ThrID   = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = typename Copy_Traits<SM75_U16x4_LDSM_T>::DstLayout;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = typename Copy_Traits<SM75_U16x4_LDSM_T>::SrcLayout;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

template <>
struct Copy_Traits<SM90_U16x8_STSM_T>
{
  // Logical thread id to thread idx (warp)
  using ThrID   = Layout<_32>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = typename Copy_Traits<SM75_U16x8_LDSM_T>::DstLayout;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = typename Copy_Traits<SM75_U16x8_LDSM_T>::SrcLayout;

  // Reference map from (thr,val) to bit
  using RefLayout = SrcLayout;
};

} // end namespace cute
