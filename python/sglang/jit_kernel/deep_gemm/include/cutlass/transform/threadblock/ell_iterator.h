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
/*! \file
    \brief Ell iterator for matrix of indices (ellColInd matrix) 
*/

#pragma once

namespace cutlass {
namespace transform {
namespace threadblock {

namespace ell{

constexpr unsigned int SmemPow = 8;
constexpr unsigned int SmemStages = 2;
constexpr unsigned int SmemSize = 1 << SmemPow;
constexpr unsigned int SmemMask = (SmemSize*SmemStages-1);

class SharedStorage{
  public:
    Array<int, SmemSize*SmemStages> array;
};

class Iterator{
  public:
  using Layout = layout::PitchLinear;
  using LongIndex = typename Layout::LongIndex;

  private:
    const int *gmem_col_idx_;
    int *smem_col_idx_;
    const int  block_size_;
    const int  base_idx_;
    const int  k_shape_;
    const int  ell_increment_;
    const int  array_length_;
    int  col_idx_base_;
    int  residue_;
    int  counter_;

    int  pow2_;
    int  residue_shape_;

    int  smem_offset_;
    int  smem_stage_;
    int  gmem_offset_;

    int  lane_;

    bool is_pow2_;
    bool is_residue_tile_;

  public:
    CUTLASS_DEVICE
    void load_ell_indices(){
      for(int i=threadIdx.x; i<SmemSize; i+=blockDim.x){
        int idx = (gmem_offset_+i < array_length_) ? gmem_offset_+i : array_length_-1;
        int gmem_col_idx = gmem_col_idx_[idx] - base_idx_;
        smem_col_idx_[i + smem_stage_ * SmemSize] = 
          (gmem_col_idx >= 0) ? gmem_col_idx : -1;
      }
      gmem_offset_ += SmemSize;
      smem_stage_ ^= 1;
    }

    CUTLASS_DEVICE
    Iterator(
        SharedStorage& shared_storage_base,
        const int* col_idx,
        const int& block_size,
        const int& base_idx,
        const int  k_shape,
        const int& problem_size_k,
        const int& ell_stride,
        const int& thread_idx)
        : residue_(0),
          counter_(0),
          smem_offset_(0),
          smem_stage_(0),
          gmem_offset_(0),
          block_size_(block_size),
          base_idx_(base_idx),
          k_shape_(k_shape),
          ell_increment_(ell_stride * block_size),
          array_length_((problem_size_k + block_size_ - 1) / block_size_), 
          residue_shape_(problem_size_k % k_shape_),
          is_residue_tile_(residue_shape_ != 0),
          smem_col_idx_(reinterpret_cast<int*>(&shared_storage_base.array)),
          gmem_col_idx_(const_cast<int*>(col_idx)),
          lane_(thread_idx % 32) {

      load_ell_indices();
      __syncthreads();
          
      is_pow2_ = ((block_size_ & (block_size_ - 1)) == 0);
      if( is_pow2_ && k_shape <= block_size_ ) lane_ = 0;
      
      col_idx_base_ = smem_col_idx_[(smem_offset_ + lane_) & SmemMask] * ell_increment_;

      pow2_ = 0;
      while(block_size_ >> (pow2_ + 1)) ++pow2_;
    }

    CUTLASS_DEVICE
    int get_blocksize(){
      return block_size_;
    }

    CUTLASS_DEVICE
    Iterator &operator++(){
      if(is_residue_tile_){
        residue_ += residue_shape_;
        is_residue_tile_ = false;
      } else {
        residue_ += k_shape_;
      }

      if(residue_ < block_size_){
        return *this;
      }

      if((array_length_ > SmemSize) && (((smem_offset_ >> SmemPow) & 1) != smem_stage_)) 
        load_ell_indices();

      if(residue_ == block_size_){
        ++smem_offset_;
        counter_ += ell_increment_;
        residue_ = 0;
        col_idx_base_ = smem_col_idx_[(smem_offset_ + lane_) & SmemMask] * ell_increment_ - counter_;
        return *this;
      }
      
      if(is_pow2_){
        smem_offset_ += residue_ >> pow2_; 
        counter_ += (residue_ >> pow2_) * ell_increment_;
        residue_ = residue_ & ((1 << pow2_) - 1);
      }
      else {
        smem_offset_ += residue_ / block_size_; 
        counter_ += (residue_ / block_size_) * ell_increment_;
        residue_ %= block_size_;
      }
      
      col_idx_base_ = smem_col_idx_[(smem_offset_ + lane_) & SmemMask] * ell_increment_ - counter_;
      
      return *this;
    }
    
    CUTLASS_DEVICE
    LongIndex get_offset(const int& idx) {
      int num_jump_tiles;
      if(is_pow2_)
        num_jump_tiles = (idx + residue_) >> pow2_;
      else 
        num_jump_tiles = (idx + residue_) / block_size_;

      int tmp = __shfl_sync(0xffffffff, col_idx_base_, num_jump_tiles); 
      return tmp - num_jump_tiles * ell_increment_;
    }
    
    CUTLASS_DEVICE
    LongIndex get_offset_fast() {
      return col_idx_base_;
    }
};

}
}
}
}
