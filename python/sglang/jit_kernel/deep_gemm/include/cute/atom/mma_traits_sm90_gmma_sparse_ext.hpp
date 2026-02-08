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
  
#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/mma_traits.hpp>

namespace cute {

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_24,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 24, 32>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_24,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 24, 32>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_40,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 40, 32>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_40,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 40, 32>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_48,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 48, 32>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_48,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 48, 32>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_56,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 56, 32>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_56,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 56, 32>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_72,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 72, 32>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_72,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 72, 32>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_80,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 80, 32>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_80,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 80, 32>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_88,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 88, 32>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_88,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 88, 32>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_104,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<104, 32>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_104,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<104, 32>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_112,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<112, 32>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_112,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<112, 32>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_120,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<120, 32>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_120,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<120, 32>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_136,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<136, 32>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_136,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<136, 32>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_144,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<144, 32>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_144,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<144, 32>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_152,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<152, 32>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_152,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<152, 32>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_160,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<160, 32>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_160,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<160, 32>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_168,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<168, 32>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_168,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<168, 32>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_176,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<176, 32>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_176,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<176, 32>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_184,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<184, 32>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_184,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<184, 32>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_200,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<200, 32>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_200,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<200, 32>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_208,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<208, 32>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_208,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<208, 32>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_216,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<216, 32>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_216,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<216, 32>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_224,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<224, 32>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_224,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<224, 32>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_232,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<232, 32>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_232,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<232, 32>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_240,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<240, 32>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_240,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<240, 32>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x32_F16F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_248,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<248, 32>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x32_F16F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_248,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<248, 32>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_24,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 24, 32>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_24,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 24, 32>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_40,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 40, 32>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_40,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 40, 32>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_48,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 48, 32>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_48,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 48, 32>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_56,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 56, 32>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_56,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 56, 32>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_72,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 72, 32>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_72,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 72, 32>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_80,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 80, 32>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_80,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 80, 32>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_88,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 88, 32>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_88,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 88, 32>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_104,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<104, 32>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_104,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<104, 32>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_112,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<112, 32>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_112,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<112, 32>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_120,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<120, 32>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_120,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<120, 32>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_136,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<136, 32>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_136,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<136, 32>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_144,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<144, 32>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_144,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<144, 32>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_152,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<152, 32>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_152,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<152, 32>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_160,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<160, 32>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_160,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<160, 32>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_168,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<168, 32>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_168,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<168, 32>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_176,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<176, 32>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_176,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<176, 32>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_184,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<184, 32>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_184,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<184, 32>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_200,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<200, 32>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_200,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<200, 32>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_208,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<208, 32>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_208,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<208, 32>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_216,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<216, 32>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_216,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<216, 32>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_224,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<224, 32>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_224,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<224, 32>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_232,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<232, 32>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_232,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<232, 32>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_240,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<240, 32>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_240,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<240, 32>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x32_F32F16F16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_248,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<248, 32>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x32_F32F16F16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, half_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = half_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_248,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<248, 32>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_24,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 24, 32>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_24,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 24, 32>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_40,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 40, 32>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_40,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 40, 32>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_48,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 48, 32>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_48,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 48, 32>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_56,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 56, 32>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_56,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 56, 32>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_72,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 72, 32>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_72,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 72, 32>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_80,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 80, 32>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_80,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 80, 32>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_88,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 88, 32>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_88,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout< 88, 32>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_104,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<104, 32>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_104,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<104, 32>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_112,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<112, 32>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_112,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<112, 32>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_120,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<120, 32>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_120,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<120, 32>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_136,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<136, 32>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_136,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<136, 32>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_144,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<144, 32>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_144,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<144, 32>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_152,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<152, 32>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_152,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<152, 32>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_160,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<160, 32>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_160,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<160, 32>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_168,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<168, 32>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_168,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<168, 32>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_176,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<176, 32>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_176,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<176, 32>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_184,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<184, 32>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_184,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<184, 32>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_200,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<200, 32>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_200,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<200, 32>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_208,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<208, 32>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_208,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<208, 32>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_216,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<216, 32>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_216,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<216, 32>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_224,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<224, 32>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_224,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<224, 32>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_232,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<232, 32>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_232,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<232, 32>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_240,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<240, 32>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_240,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<240, 32>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x32_F32BF16BF16_SS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<tnspA>;
  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_248,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 32>;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<248, 32>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::Major tnspA, GMMA::Major tnspB, GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x32_F32BF16BF16_RS<tnspA, tnspB, scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, bfloat16_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = bfloat16_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<tnspB>;

  using Shape_MNK = Shape<_64,_248,_32>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x32;
  using ELayout = GMMA::ELayout_64x32;
  using BLayout = GMMA::ABLayout<248, 32>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 24, 16>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 24, 16>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 40, 16>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 40, 16>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 48, 16>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 48, 16>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 56, 16>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 56, 16>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 72, 16>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 72, 16>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 80, 16>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 80, 16>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 88, 16>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout< 88, 16>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<104, 16>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<104, 16>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<112, 16>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<112, 16>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<120, 16>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<120, 16>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<136, 16>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<136, 16>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<144, 16>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<144, 16>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<152, 16>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<152, 16>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<160, 16>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<160, 16>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<168, 16>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<168, 16>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<176, 16>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<176, 16>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<184, 16>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<184, 16>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<200, 16>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<200, 16>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<208, 16>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<208, 16>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<216, 16>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<216, 16>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<224, 16>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<224, 16>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<232, 16>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<232, 16>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<240, 16>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<240, 16>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x16_F32TF32TF32_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 16>;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<248, 16>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x16_F32TF32TF32_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, tfloat32_t>;
  using ValTypeE = sparse_elem<4, uint8_t>;
  using ValTypeB = tfloat32_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_16>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x16;
  using ELayout = GMMA::ELayout_64x16;
  using BLayout = GMMA::ABLayout<248, 16>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32S8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32S8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32S8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32S8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32S8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32S8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32S8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32S8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, int8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32U8S8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32U8S8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32U8S8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32U8S8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = int8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32U8U8_SS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32U8U8_SS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32U8U8_RS_TN<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_S32U8U8_RS_TN_SATURATE<spsel>>
{
  using ValTypeD = int32_t;
  using ValTypeA = sparse_elem<2, uint8_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = uint8_t;
  using ValTypeC = int32_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F16E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F16E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F32E4M3E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F32E4M3E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F16E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F16E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F32E4M3E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F32E4M3E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e4m3_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F16E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F16E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F32E5M2E4M3_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F32E5M2E4M3_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e4m3_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x24x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_24,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 24, 64>;
  using CLayout = GMMA::CLayout_64x24;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x40x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_40,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 40, 64>;
  using CLayout = GMMA::CLayout_64x40;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x48x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_48,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 48, 64>;
  using CLayout = GMMA::CLayout_64x48;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x56x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_56,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 56, 64>;
  using CLayout = GMMA::CLayout_64x56;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x72x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_72,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 72, 64>;
  using CLayout = GMMA::CLayout_64x72;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x80x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_80,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 80, 64>;
  using CLayout = GMMA::CLayout_64x80;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x88x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_88,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout< 88, 64>;
  using CLayout = GMMA::CLayout_64x88;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x104x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_104,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<104, 64>;
  using CLayout = GMMA::CLayout_64x104;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x112x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_112,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<112, 64>;
  using CLayout = GMMA::CLayout_64x112;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x120x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_120,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<120, 64>;
  using CLayout = GMMA::CLayout_64x120;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x136x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_136,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<136, 64>;
  using CLayout = GMMA::CLayout_64x136;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x144x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_144,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<144, 64>;
  using CLayout = GMMA::CLayout_64x144;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x152x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_152,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<152, 64>;
  using CLayout = GMMA::CLayout_64x152;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x160x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_160,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<160, 64>;
  using CLayout = GMMA::CLayout_64x160;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x168x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_168,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<168, 64>;
  using CLayout = GMMA::CLayout_64x168;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x176x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_176,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<176, 64>;
  using CLayout = GMMA::CLayout_64x176;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x184x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_184,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<184, 64>;
  using CLayout = GMMA::CLayout_64x184;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x200x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_200,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<200, 64>;
  using CLayout = GMMA::CLayout_64x200;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x208x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_208,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<208, 64>;
  using CLayout = GMMA::CLayout_64x208;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x216x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_216,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<216, 64>;
  using CLayout = GMMA::CLayout_64x216;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x224x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_224,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<224, 64>;
  using CLayout = GMMA::CLayout_64x224;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x232x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_232,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<232, 64>;
  using CLayout = GMMA::CLayout_64x232;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x240x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_240,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<240, 64>;
  using CLayout = GMMA::CLayout_64x240;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F16E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F16E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = half_t;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = half_t;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F32E5M2E5M2_SS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeA = GMMA::smem_desc<GMMA::Major::K>;
  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ABLayout< 64, 64>;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <GMMA::ScaleIn scaleA, GMMA::ScaleIn scaleB, GMMA::SparseSel spsel>
struct MMA_Traits<SM90::GMMA::SPARSE::GMMA_64x248x64_F32E5M2E5M2_RS_TN<scaleA, scaleB, spsel>>
{
  using ValTypeD = float;
  using ValTypeA = sparse_elem<2, float_e5m2_t>;
  using ValTypeE = sparse_elem<8, uint8_t>;
  using ValTypeB = float_e5m2_t;
  using ValTypeC = float;

  using FrgTypeB = GMMA::smem_desc<GMMA::Major::K>;

  using Shape_MNK = Shape<_64,_248,_64>;
  using ThrID   = Layout<_128>;
  using ALayout = GMMA::ALayout_64x64;
  using ELayout = GMMA::ELayout_64x64;
  using BLayout = GMMA::ABLayout<248, 64>;
  using CLayout = GMMA::CLayout_64x248;

  GMMA::ScaleOut accumulate_ = GMMA::ScaleOut::One;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute
