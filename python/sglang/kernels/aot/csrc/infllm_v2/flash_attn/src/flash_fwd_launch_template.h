/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "flash.h"
#include "flash_fwd_kernel.h"
#include "hardware_info.h"
#include "static_switch.h"

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH \
  printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
  template <typename Kernel_traits, __VA_ARGS__>     \
  __global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(
    flash_fwd_kernel,
    bool Is_dropout,
    bool Is_causal,
    bool Is_local,
    bool Has_alibi,
    bool Is_even_MN,
    bool Is_even_K,
    bool Is_softcap,
    bool Return_softmax) {
#if defined(ARCH_SUPPORTS_FLASH)
  static_assert(!(Is_causal && Is_local));  // Enforce constraints
  flash::compute_attn<
      Kernel_traits,
      Is_dropout,
      Is_causal,
      Is_local,
      Has_alibi,
      Is_even_MN,
      Is_even_K,
      Is_softcap,
      Return_softmax>(params);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

DEFINE_FLASH_FORWARD_KERNEL(
    flash_fwd_splitkv_kernel,
    bool Is_causal,
    bool Is_local,
    bool Has_alibi,
    bool Is_even_MN,
    bool Is_even_K,
    bool Is_softcap,
    bool Split,
    bool Append_KV) {
#if defined(ARCH_SUPPORTS_FLASH)
  flash::compute_attn_splitkv<
      Kernel_traits,
      Is_causal,
      Is_local,
      Has_alibi,
      Is_even_MN,
      Is_even_K,
      Is_softcap,
      Split,
      Append_KV>(params);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

DEFINE_FLASH_FORWARD_KERNEL(
    flash_fwd_splitkv_stage1_kernel,
    bool Is_causal,
    bool Is_local,
    bool Has_alibi,
    bool Is_even_MN,
    bool Is_even_K,
    bool Is_softcap,
    bool Split,
    bool Append_KV) {
#if defined(ARCH_SUPPORTS_FLASH)
  flash::compute_attn_splitkv_stage1<
      Kernel_traits,
      Is_causal,
      Is_local,
      Has_alibi,
      Is_even_MN,
      Is_even_K,
      Is_softcap,
      Split,
      Append_KV>(params);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, int kBlockM, int Log_max_splits, bool Is_even_K) {
  static_assert(Log_max_splits >= 1);
  flash::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}

template <typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr size_t smem_size = Kernel_traits::kSmemSize;
  // printf("smem_size = %d\n", smem_size);

  // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
  // https://github.com/kokkos/kokkos-kernels/issues/349
  // https://github.com/HazyResearch/flash-attention/issues/21

  const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(num_m_block, params.b, params.h);
  const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
                          params.seqlen_k % Kernel_traits::kBlockN == 0 &&
                          params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;
  // const bool return_softmax = params.p_ptr != nullptr;
  BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
      // LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
      constexpr static bool Is_local = false;
      {  // TODO remove debug info
        // BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
        constexpr static bool ReturnSoftmaxConst = false;
        {  // TODO remove debug info
          // ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
          constexpr static bool Has_alibi = false;
          {  // TODO remove debug info
            // SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
            constexpr static bool Is_softcap = false;
            {
              // Will only return softmax if dropout, to reduce compilation time.
              // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
              // If return_softmax, set IsEvenMNConst to false to reduce number of templates
              // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
              // If Is_local, set Is_causal to false
              auto kernel = &flash_fwd_kernel < Kernel_traits, Is_dropout && !Is_softcap, Is_causal,
                   Is_local && !Is_causal, Has_alibi,
                   IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128,
                   IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap > ;
              // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, false, true, true, false>;
              // printf("IsEvenMNConst = %d, IsEvenKConst = %d, Is_local = %d, Is_causal = %d, ReturnSoftmaxConst = %d,
              // Is_dropout = %d\n", int(IsEvenMNConst), int(IsEvenKConst), int(Is_local), int(Is_causal),
              // int(ReturnSoftmaxConst), int(Is_dropout)); auto kernel = &flash_fwd_kernel<Kernel_traits, false,
              // Is_causal, false, true, true, false>;
              if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
              }
              // int ctas_per_sm;
              // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
              // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
              kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
              C10_CUDA_KERNEL_LAUNCH_CHECK();
            }
          }
        }
      }
    });
  });
}

template <typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
  static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
  constexpr size_t smem_size = Kernel_traits::kSmemSize;
  const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(
      num_m_block,
      params.num_splits > 1 ? params.num_splits : params.b,
      params.num_splits > 1 ? params.b * params.h : params.h);
  const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
                          params.seqlen_k % Kernel_traits::kBlockN == 0 &&
                          params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;
  BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
      // LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
      constexpr static bool Is_local = false;
      {  // TODO remove debug info
        BOOL_SWITCH(params.num_splits > 1, Split, [&] {
          BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
            // ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
            constexpr static bool Has_alibi = false;
            {  // TODO remove debug info
              // SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
              constexpr static bool Is_softcap = false;
              {  // TODO remove debug info
                // If Append_KV, then we must have seqlen_offsets, which means cu_seqlens_k != nullptr.
                // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                // If Is_local, set Is_causal to false
                auto kernel = &flash_fwd_splitkv_kernel < Kernel_traits, Is_causal, Is_local && !Is_causal, Has_alibi,
                     IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128,
                     IsEvenKConst, Is_softcap, Split, Append_KV > ;
                // auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, true, Split, Append_KV>;
                // auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, IsEvenKConst>;
                if (smem_size >= 48 * 1024) {
                  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
            }
          });
        });
      }
    });
  });
  if (params.num_splits > 1) {
    // We want kBlockM to be as small as possible for more parallelism.
    // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
    // If headdim is divisible by 64, then we set kBlockM = 8, etc.
    constexpr static int kBlockM =
        Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
    dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
      if (params.num_splits <= 2) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 4) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 8) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 16) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 32) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 64) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 128) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }
}

template <typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd_stage1(Flash_fwd_params& params, cudaStream_t stream) {
  static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
  static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
  constexpr size_t smem_size = Kernel_traits::kSmemSize;
  const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(
      num_m_block,
      params.num_splits > 1 ? params.num_splits : params.b,
      params.num_splits > 1 ? params.b * params.h : params.h);
  const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
                          params.seqlen_k % Kernel_traits::kBlockN == 0 &&
                          params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;
  BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
      // LOCAL_SWITCH((params.window_size_left >= 0 || params.window_size_right >= 0) && !Is_causal, Is_local, [&] {
      constexpr static bool Is_local = false;
      {  // TODO remove debug info
        // BOOL_SWITCH(params.num_splits > 1, Split, [&] {
        constexpr static bool Split = false;
        {  // TODO remove debug info
          // BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV, [&] {
          constexpr static bool Append_KV = false;
          {  // TODO remove debug info
            // ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
            constexpr static bool Has_alibi = false;
            {  // TODO remove debug info
              // SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
              constexpr static bool Is_softcap = false;
              {  // TODO remove debug info
                // If Append_KV, then we must have seqlen_offsets, which means cu_seqlens_k != nullptr.
                // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                // If Is_local, set Is_causal to false
                auto kernel = &flash_fwd_splitkv_stage1_kernel < Kernel_traits, Is_causal, Is_local && !Is_causal,
                     Has_alibi,
                     IsEvenMNConst && !Append_KV && IsEvenKConst && !Is_local && Kernel_traits::kHeadDim <= 128,
                     IsEvenKConst, Is_softcap, Split, Append_KV > ;
                if (smem_size >= 48 * 1024) {
                  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                }
                kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                C10_CUDA_KERNEL_LAUNCH_CHECK();
              }
            }
          }
        }
      }
    });
  });
}

template <typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params& params, cudaStream_t stream) {
  if (params.blockmask == nullptr) {
    constexpr static int kBlockM = 64;  // Fixed for all head dimensions
    // TD [2023-08-28]: nvcc segfaults for headdim 96 with block size 64 x 256,
    // and for headdim 192 with block size 64 x 128.
    // Also for headdim 160 with block size 64 x 128 after the rotary addition.
    constexpr static int kBlockN = Headdim <= 64 ? 256 : (Headdim <= 128 ? 128 : 64);
    if (params.m_block_dim == 1) {
      run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>, Is_causal>(
          params, stream);
    } else {
      run_flash_splitkv_fwd_stage1<Flash_fwd_kernel_traits<Headdim, 16, 64, 1, false, false, T>, Is_causal>(
          params, stream);
    }
  } else if (params.cu_seqlens_q != nullptr) {
    constexpr static int kBlockM = 16;
    constexpr static int kBlockN = 64;
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 1, false, false, T>, Is_causal>(
        params, stream);
  } else {
    constexpr static int kBlockM = 64;
    constexpr static int kBlockN = 64;
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>, Is_causal>(
        params, stream);
  }
}
