# Middle layers sequence-position aggregation

- Source ordered timeline: `/sgl-workspace/sglang/decode_tp0_middle_kernel_timeline_ordered.csv`
- Aggregation key: `(layer_type, kernel_order_in_layer)`
- Meaning: all layers of the same type have their N-th kernel summed together.

## Type summaries

| type | layers | sequence length | total ms |
| --- | ---: | ---: | ---: |
| odd_long | 26 | 77 | 30.804 |
| even_short | 26 | 100 | 19.290 |

## odd_long: sequence order

| order | pct | total us | mean us | median us | tag | name |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 0.41% | 126.618 us | 4.870 us | 4.840 us | mhc_pre | `triton_red_fused__to_copy_mean_pow_view_0` |
| 1 | 0.64% | 197.656 us | 7.602 us | 7.599 us | other | `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA4_GR...WSGRB0_WS64_WG16_4_4` |
| 2 | 0.52% | 160.577 us | 6.176 us | 6.179 us | other | `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW1` |
| 3 | 0.36% | 112.178 us | 4.315 us | 4.299 us | mhc_pre | `triton_per_fused_mean_pow_1` |
| 4 | 0.36% | 110.575 us | 4.253 us | 4.239 us | mhc_pre | `triton_poi_fused_add_mean_mul_pow_rsqrt_2` |
| 5 | 0.53% | 162.776 us | 6.261 us | 6.279 us | mhc_pre | `hc_split_sinkhorn_kernel__kernel` |
| 6 | 0.36% | 111.897 us | 4.304 us | 4.319 us | other | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bina...ambda(int, bool)#1})` |
| 7 | 0.39% | 121.375 us | 4.668 us | 4.619 us | other | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator(...d int, float, 4, 4>)` |
| 8 | 0.37% | 114.974 us | 4.422 us | 4.439 us | copy_cast | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> ...::array<char*, 2ul>)` |
| 9 | 0.39% | 118.735 us | 4.567 us | 4.559 us | norm | `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi32ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_diiiiiiib` |
| 10 | 0.35% | 109.254 us | 4.202 us | 4.199 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 11 | 0.36% | 111.256 us | 4.279 us | 4.319 us | other | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)` |
| 12 | 0.37% | 114.734 us | 4.413 us | 4.399 us | other | `__amd_rocclr_fillBufferAligned` |
| 13 | 0.58% | 179.415 us | 6.901 us | 6.859 us | linear_quant_gemm | `aiter::fp8gemm_bf16_blockscale_BpreShuffle_48x128` |
| 14 | 0.37% | 112.735 us | 4.336 us | 4.339 us | norm | `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi8ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_diiiiiiib` |
| 15 | 0.36% | 109.655 us | 4.218 us | 4.199 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 16 | 0.56% | 172.698 us | 6.642 us | 6.620 us | linear_quant_gemm | `void ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle<ck::GridwiseGemmMultiD_blockscale_xdl_cshuffle_v3_b_preshuffle<ck::tensor_layout::gemm:...8_fnuz_t>::Argument)` |
| 17 | 0.38% | 115.894 us | 4.457 us | 4.439 us | norm | `_rms_normalize_kernel` |
| 18 | 0.38% | 118.458 us | 4.556 us | 4.559 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 19 | 0.4% | 124.015 us | 4.770 us | 4.759 us | linear_quant_gemm | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK_SIZE_K_128_GROUP_SIZE_M_4_NUM_KSPLIT_14_SPLITK_BLOCK_SIZE_512_EVE...32_cache_modifier_CG` |
| 20 | 0.39% | 119.859 us | 4.610 us | 4.639 us | linear_quant_gemm | `_gemm_a8w8_blockscale_reduce_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_ACTUAL_KSPLIT_14_MAX_KSPLIT_16` |
| 21 | 0.38% | 117.414 us | 4.516 us | 4.519 us | norm | `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_diiiiiiib` |
| 22 | 0.38% | 115.696 us | 4.450 us | 4.439 us | rope | `apply_rotary_emb_triton_kernel` |
| 23 | 0.35% | 108.657 us | 4.179 us | 4.199 us | rope | `apply_rotary_emb_triton_kernel` |
| 24 | 0.36% | 109.815 us | 4.224 us | 4.239 us | copy_cast | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::a...::array<char*, 2ul>)` |
| 25 | 0.83% | 256.295 us | 9.857 us | 9.679 us | copy_cast | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::a...::array<char*, 2ul>)` |
| 26 | 0.79% | 244.136 us | 9.390 us | 9.299 us | other | `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT64x16x64_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA4_GR...WSGRB0_WS64_WG16_4_4` |
| 27 | 0.61% | 187.175 us | 7.199 us | 7.199 us | other | `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` |
| 28 | 0.37% | 113.095 us | 4.350 us | 4.339 us | copy_cast | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(...ambda(int, bool)#1})` |
| 29 | 3.74% | 1152.714 us | 44.335 us | 44.139 us | compressor | `_compress_c128_decode_old_kernel` |
| 30 | 0.35% | 108.735 us | 4.182 us | 4.159 us | other | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<in...::array<char*, 2ul>)` |
| 31 | 0.34% | 106.094 us | 4.081 us | 4.099 us | other | `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIter...::array<char*, 2ul>)` |
| 32 | 0.35% | 107.255 us | 4.125 us | 4.119 us | other | `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<int, int, int, at::native::binary_internal::MulFunctor<int> >, std::array<char*,...::array<char*, 2ul>)` |
| 33 | 0.35% | 106.814 us | 4.108 us | 4.079 us | copy_cast | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(...ambda(int, bool)#1})` |
| 34 | 0.38% | 116.295 us | 4.473 us | 4.479 us | other | `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` |
| 35 | 0.36% | 110.055 us | 4.233 us | 4.239 us | rope | `_fused_norm_rope_kernel` |
| 36 | 0.38% | 118.575 us | 4.561 us | 4.559 us | copy_cast | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> ...::array<char*, 2ul>)` |
| 37 | 0.38% | 115.934 us | 4.459 us | 4.439 us | other | `_quant_k_cache_fused_kernel` |
| 38 | 0.34% | 105.975 us | 4.076 us | 4.079 us | other | `_set_k_and_s_triton_kernel` |
| 39 | 0.37% | 112.855 us | 4.341 us | 4.319 us | copy_cast | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{...ambda(int, bool)#1})` |
| 40 | 0.39% | 119.457 us | 4.595 us | 4.599 us | other | `_quant_k_cache_fused_kernel` |
| 41 | 0.36% | 111.095 us | 4.273 us | 4.279 us | other | `_set_k_and_s_triton_kernel` |
| 42 | 38.86% | 11970.644 us | 460.409 us | 460.704 us | attention_main | `main_kernel corr=33755 runtime=hipGraphLaunch` |
| 43 | 11.8% | 3633.663 us | 139.756 us | 139.841 us | attention_main | `main_kernel corr=33755 runtime=hipGraphLaunch` |
| 44 | 0.39% | 118.894 us | 4.573 us | 4.559 us | rope | `apply_rotary_emb_triton_kernel` |
| 45 | 0.82% | 251.174 us | 9.661 us | 9.599 us | other | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVW...WSGRB0_WS64_WG16_4_4` |
| 46 | 0.43% | 132.457 us | 5.095 us | 5.080 us | copy_cast | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{...ambda(int, bool)#1})` |
| 47 | 0.38% | 116.855 us | 4.494 us | 4.439 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 48 | 0.49% | 152.335 us | 5.859 us | 5.739 us | linear_quant_gemm | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_32_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_NUM_KSPLIT_4_SPLITK_BLOCK_SIZE_512_EVEN...24_cache_modifier_CG` |
| 49 | 0.38% | 116.176 us | 4.468 us | 4.479 us | linear_quant_gemm | `_gemm_a8w8_blockscale_reduce_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_ACTUAL_KSPLIT_4_MAX_KSPLIT_4` |
| 50 | 2.39% | 736.068 us | 28.310 us | 28.180 us | all_reduce | `_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8RankDataES2_NS_11RankSignalsEPNS_6SignalEPT_ii` |
| 51 | 1.79% | 551.099 us | 21.196 us | 21.179 us | mhc_post_boundary | `mhc_post_tilelang_kernel` |
| 52 | 0.42% | 130.374 us | 5.014 us | 5.019 us | mhc_pre | `triton_red_fused__to_copy_mean_pow_view_0` |
| 53 | 0.68% | 210.416 us | 8.093 us | 8.040 us | other | `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA4_GR...WSGRB0_WS64_WG16_4_4` |
| 54 | 0.61% | 186.736 us | 7.182 us | 7.139 us | other | `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW1` |
| 55 | 0.36% | 112.175 us | 4.314 us | 4.319 us | mhc_pre | `triton_per_fused_mean_pow_1` |
| 56 | 0.36% | 110.536 us | 4.251 us | 4.259 us | mhc_pre | `triton_poi_fused_add_mean_mul_pow_rsqrt_2` |
| 57 | 0.53% | 162.698 us | 6.258 us | 6.279 us | mhc_pre | `hc_split_sinkhorn_kernel__kernel` |
| 58 | 0.4% | 124.215 us | 4.777 us | 4.779 us | other | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bina...ambda(int, bool)#1})` |
| 59 | 0.51% | 156.017 us | 6.001 us | 5.959 us | other | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator(...d int, float, 4, 4>)` |
| 60 | 0.4% | 121.937 us | 4.690 us | 4.679 us | copy_cast | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> ...::array<char*, 2ul>)` |
| 61 | 0.38% | 117.855 us | 4.533 us | 4.539 us | norm | `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi32ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_diiiiiiib` |
| 62 | 0.36% | 109.496 us | 4.211 us | 4.199 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 63 | 3.7% | 1140.144 us | 43.852 us | 43.799 us | linear_quant_gemm | `void ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle<ck::GridwiseGemmMultiD_blockscale_xdl_cshuffle_v3_b_preshuffle<ck::tensor_layout::gemm:...8_fnuz_t>::Argument)` |
| 64 | 0.38% | 116.496 us | 4.481 us | 4.479 us | other | `_ZN7sgl_hip10activation18act_and_mul_kernelI14__hip_bfloat16TnPFT_RKS3_EXadL_Z4siluIS2_ES3_S5_EEEEvPS3_PS4_i` |
| 65 | 0.36% | 110.496 us | 4.250 us | 4.279 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 66 | 0.47% | 143.336 us | 5.513 us | 5.420 us | linear_quant_gemm | `void ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle<ck::GridwiseGemmMultiD_blockscale_xdl_cshuffle_v3_b_preshuffle<ck::tensor_layout::gemm:...8_fnuz_t>::Argument)` |
| 67 | 0.77% | 236.699 us | 9.104 us | 9.079 us | other | `hgemm_bf16_32x64x64_S2TN_AS_SPK16_BS_0` |
| 68 | 0.7% | 215.657 us | 8.294 us | 8.279 us | moe | `void aiter::grouped_topk_kernel<c10::BFloat16, float __vector(4), 8, true, true, false>(c10::BFloat16*, c10::BFloat16 const*, float*, int*, unsigned long, i...nt, int, int, float)` |
| 69 | 1.27% | 391.419 us | 15.055 us | 14.999 us | moe | `void ck_tile::kentry<2, ck_tile::MoeSortingKernel<ck_tile::MoeSortingProblemEx<int, float, 2, true, false, false, true, 0> >, ck_tile::MoeSortingKernel<ck_t..., true, 0> >::Kargs)` |
| 70 | 0.48% | 147.856 us | 5.687 us | 5.599 us | moe | `_ZN5aiter27mxfp4_quant_moe_sort_kernelIDF16bN4opus5fp4_tELi256ELi32EEEvPT0_PhPKT_PKiSA_iiiiiiiii` |
| 71 | 2.96% | 910.942 us | 35.036 us | 34.279 us | moe | `void ck::kernel_moe_mxgemm_2lds<ck::GridwiseMoeGemmMX_BPreshuffle<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, ck::Tuple<ck::ten...4x2_pk_t>::Argument)` |
| 72 | 0.44% | 135.777 us | 5.222 us | 5.199 us | moe | `_ZN5aiter27mxfp4_quant_moe_sort_kernelIDF16bN4opus5fp4_tELi64ELi8EEEvPT0_PhPKT_PKiSA_iiiiiiiii` |
| 73 | 1.64% | 506.502 us | 19.481 us | 19.459 us | moe | `void ck::kernel_moe_mxgemm_2lds<ck::GridwiseMoeGemmMX_BPreshuffle<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, ck::Tuple<ck::ten...4x2_pk_t>::Argument)` |
| 74 | 0.37% | 114.535 us | 4.405 us | 4.399 us | other | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10:...::array<char*, 3ul>)` |
| 75 | 1.06% | 326.698 us | 12.565 us | 12.739 us | all_reduce | `_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8RankDataES2_NS_11RankSignalsEPNS_6SignalEPT_ii` |
| 76 | 1.68% | 516.618 us | 19.870 us | 19.799 us | mhc_post_boundary | `mhc_post_tilelang_kernel` |

## even_short: sequence order

| order | pct | total us | mean us | median us | tag | name |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 0 | 0.65% | 124.935 us | 4.805 us | 4.759 us | mhc_pre | `triton_red_fused__to_copy_mean_pow_view_0` |
| 1 | 1.02% | 197.657 us | 7.602 us | 7.579 us | other | `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA4_GR...WSGRB0_WS64_WG16_4_4` |
| 2 | 0.82% | 157.617 us | 6.062 us | 6.120 us | other | `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW1` |
| 3 | 0.58% | 112.096 us | 4.311 us | 4.299 us | mhc_pre | `triton_per_fused_mean_pow_1` |
| 4 | 0.57% | 109.336 us | 4.205 us | 4.199 us | mhc_pre | `triton_poi_fused_add_mean_mul_pow_rsqrt_2` |
| 5 | 0.83% | 159.696 us | 6.142 us | 6.099 us | mhc_pre | `hc_split_sinkhorn_kernel__kernel` |
| 6 | 0.58% | 112.295 us | 4.319 us | 4.319 us | other | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bina...ambda(int, bool)#1})` |
| 7 | 0.59% | 112.895 us | 4.342 us | 4.339 us | other | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator(...d int, float, 4, 4>)` |
| 8 | 0.6% | 115.257 us | 4.433 us | 4.439 us | copy_cast | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> ...::array<char*, 2ul>)` |
| 9 | 0.62% | 118.696 us | 4.565 us | 4.559 us | norm | `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi32ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_diiiiiiib` |
| 10 | 0.57% | 110.134 us | 4.236 us | 4.239 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 11 | 0.58% | 111.536 us | 4.290 us | 4.299 us | other | `void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, std::array<char*, 1ul> >(int, at::native::FillFunctor<float>, std::array<char*, 1ul>)` |
| 12 | 0.6% | 116.216 us | 4.470 us | 4.479 us | other | `__amd_rocclr_fillBufferAligned` |
| 13 | 0.95% | 183.295 us | 7.050 us | 6.939 us | linear_quant_gemm | `aiter::fp8gemm_bf16_blockscale_BpreShuffle_48x128` |
| 14 | 0.59% | 113.294 us | 4.357 us | 4.339 us | norm | `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi8ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_diiiiiiib` |
| 15 | 0.56% | 108.895 us | 4.188 us | 4.199 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 16 | 0.91% | 175.336 us | 6.744 us | 6.699 us | linear_quant_gemm | `void ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle<ck::GridwiseGemmMultiD_blockscale_xdl_cshuffle_v3_b_preshuffle<ck::tensor_layout::gemm:...8_fnuz_t>::Argument)` |
| 17 | 0.61% | 117.536 us | 4.521 us | 4.500 us | norm | `_rms_normalize_kernel` |
| 18 | 0.63% | 120.694 us | 4.642 us | 4.599 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 19 | 0.64% | 123.378 us | 4.745 us | 4.739 us | linear_quant_gemm | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK_SIZE_K_128_GROUP_SIZE_M_4_NUM_KSPLIT_14_SPLITK_BLOCK_SIZE_512_EVE...32_cache_modifier_CG` |
| 20 | 0.63% | 121.175 us | 4.661 us | 4.639 us | linear_quant_gemm | `_gemm_a8w8_blockscale_reduce_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_ACTUAL_KSPLIT_14_MAX_KSPLIT_16` |
| 21 | 0.63% | 120.856 us | 4.648 us | 4.639 us | norm | `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi64ELi8ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_diiiiiiib` |
| 22 | 0.59% | 114.736 us | 4.413 us | 4.399 us | rope | `apply_rotary_emb_triton_kernel` |
| 23 | 0.57% | 110.896 us | 4.265 us | 4.239 us | rope | `apply_rotary_emb_triton_kernel` |
| 24 | 0.55% | 106.694 us | 4.104 us | 4.079 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 25 | 0.83% | 160.857 us | 6.187 us | 6.159 us | linear_quant_gemm | `void ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle<ck::GridwiseGemmMultiD_blockscale_xdl_cshuffle_v3_b_preshuffle<ck::tensor_layout::gemm:...8_fnuz_t>::Argument)` |
| 26 | 0.6% | 114.936 us | 4.421 us | 4.420 us | rope | `apply_rotary_emb_triton_kernel` |
| 27 | 0.62% | 119.175 us | 4.584 us | 4.579 us | other | `void fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16, 7, c10::BFloat16> >(HadamardParamsBase)` |
| 28 | 0.64% | 123.336 us | 4.744 us | 4.759 us | other | `_act_quant_kernel` |
| 29 | 1.61% | 311.176 us | 11.968 us | 11.999 us | other | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVW...WSGRB0_WS64_WG16_4_4` |
| 30 | 0.55% | 107.015 us | 4.116 us | 4.099 us | other | `_fused_scale_kernel` |
| 31 | 0.55% | 107.055 us | 4.117 us | 4.099 us | copy_cast | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::a...::array<char*, 2ul>)` |
| 32 | 0.82% | 158.735 us | 6.105 us | 6.099 us | copy_cast | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::a...::array<char*, 2ul>)` |
| 33 | 0.88% | 170.617 us | 6.562 us | 6.500 us | other | `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT32x16x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA4_GR...WSGRB0_WS64_WG16_4_4` |
| 34 | 1.01% | 194.738 us | 7.490 us | 7.439 us | other | `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW4` |
| 35 | 0.61% | 117.015 us | 4.501 us | 4.439 us | copy_cast | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(...ambda(int, bool)#1})` |
| 36 | 0.72% | 139.216 us | 5.354 us | 5.339 us | compressor | `_compress_c4_decode_old_kernel` |
| 37 | 0.56% | 108.936 us | 4.190 us | 4.199 us | other | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<in...::array<char*, 2ul>)` |
| 38 | 0.55% | 105.695 us | 4.065 us | 4.079 us | other | `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIter...::array<char*, 2ul>)` |
| 39 | 0.56% | 107.417 us | 4.131 us | 4.139 us | other | `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<int, int, int, at::native::binary_internal::MulFunctor<int> >, std::array<char*,...::array<char*, 2ul>)` |
| 40 | 0.57% | 109.294 us | 4.204 us | 4.199 us | copy_cast | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(...ambda(int, bool)#1})` |
| 41 | 0.61% | 117.736 us | 4.528 us | 4.519 us | other | `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` |
| 42 | 0.58% | 111.735 us | 4.298 us | 4.299 us | rope | `_fused_norm_rope_kernel` |
| 43 | 0.6% | 115.976 us | 4.461 us | 4.460 us | other | `void fast_hadamard_transform_kernel<fast_hadamard_transform_kernel_traits<16, 7, float> >(HadamardParamsBase)` |
| 44 | 0.58% | 112.175 us | 4.314 us | 4.319 us | other | `_act_quant_kernel` |
| 45 | 0.57% | 110.095 us | 4.234 us | 4.239 us | other | `_set_k_and_s_triton_kernel` |
| 46 | 0.88% | 170.054 us | 6.541 us | 6.519 us | other | `fp8_paged_mqa_logits_kernel` |
| 47 | 1.43% | 275.497 us | 10.596 us | 10.479 us | other | `(anonymous namespace)::deepseek_v4_topk_transform_512_kernel((anonymous namespace)::TopK512Params)` |
| 48 | 0.57% | 110.216 us | 4.239 us | 4.239 us | copy_cast | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::a...::array<char*, 2ul>)` |
| 49 | 2.17% | 419.180 us | 16.122 us | 16.040 us | copy_cast | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::a...::array<char*, 2ul>)` |
| 50 | 2.44% | 471.382 us | 18.130 us | 18.140 us | other | `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x512_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA4_G...WSGRB0_WS64_WG16_4_4` |
| 51 | 0.58% | 112.496 us | 4.327 us | 4.319 us | copy_cast | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(...ambda(int, bool)#1})` |
| 52 | 0.86% | 166.455 us | 6.402 us | 6.359 us | compressor | `_compress_c4_decode_old_kernel` |
| 53 | 0.56% | 108.256 us | 4.164 us | 4.159 us | other | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul> >(int, at::native::CUDAFunctorOnSelf_add<in...::array<char*, 2ul>)` |
| 54 | 0.55% | 105.415 us | 4.054 us | 4.039 us | other | `void at::native::vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<int, int, int, at::native::binary_internal::div_floor_kernel_cuda(at::TensorIter...::array<char*, 2ul>)` |
| 55 | 0.55% | 106.975 us | 4.114 us | 4.119 us | other | `void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<int, int, int, at::native::binary_internal::MulFunctor<int> >, std::array<char*,...::array<char*, 2ul>)` |
| 56 | 0.55% | 106.174 us | 4.084 us | 4.079 us | copy_cast | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(...ambda(int, bool)#1})` |
| 57 | 0.6% | 116.615 us | 4.485 us | 4.479 us | other | `void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)` |
| 58 | 0.58% | 111.135 us | 4.274 us | 4.260 us | rope | `_fused_norm_rope_kernel` |
| 59 | 0.6% | 116.095 us | 4.465 us | 4.479 us | copy_cast | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> ...::array<char*, 2ul>)` |
| 60 | 0.59% | 114.616 us | 4.408 us | 4.399 us | other | `_quant_k_cache_fused_kernel` |
| 61 | 0.55% | 106.815 us | 4.108 us | 4.119 us | other | `_set_k_and_s_triton_kernel` |
| 62 | 0.61% | 117.898 us | 4.535 us | 4.459 us | copy_cast | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{...ambda(int, bool)#1})` |
| 63 | 0.6% | 114.854 us | 4.417 us | 4.419 us | other | `_quant_k_cache_fused_kernel` |
| 64 | 0.57% | 110.496 us | 4.250 us | 4.239 us | other | `_set_k_and_s_triton_kernel` |
| 65 | 9.25% | 1783.515 us | 68.597 us | 68.420 us | attention_main | `main_kernel corr=33755 runtime=hipGraphLaunch` |
| 66 | 1.1% | 212.337 us | 8.167 us | 8.159 us | attention_main | `main_kernel corr=33755 runtime=hipGraphLaunch` |
| 67 | 0.61% | 118.136 us | 4.544 us | 4.559 us | rope | `apply_rotary_emb_triton_kernel` |
| 68 | 1.23% | 237.498 us | 9.135 us | 9.099 us | other | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI16x16x1_SN_LDSB1_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVW...WSGRB0_WS64_WG16_4_4` |
| 69 | 0.64% | 123.256 us | 4.741 us | 4.719 us | copy_cast | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{...ambda(int, bool)#1})` |
| 70 | 0.6% | 114.935 us | 4.421 us | 4.399 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 71 | 0.77% | 148.656 us | 5.718 us | 5.599 us | linear_quant_gemm | `_gemm_a8w8_blockscale_kernel_GROUP_K_128_GROUP_N_128_BLOCK_SIZE_M_16_BLOCK_SIZE_N_32_BLOCK_SIZE_K_128_GROUP_SIZE_M_1_NUM_KSPLIT_4_SPLITK_BLOCK_SIZE_512_EVEN...24_cache_modifier_CG` |
| 72 | 0.6% | 115.696 us | 4.450 us | 4.439 us | linear_quant_gemm | `_gemm_a8w8_blockscale_reduce_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_ACTUAL_KSPLIT_4_MAX_KSPLIT_4` |
| 73 | 3.5% | 676.099 us | 26.004 us | 25.619 us | all_reduce | `_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8RankDataES2_NS_11RankSignalsEPNS_6SignalEPT_ii` |
| 74 | 2.64% | 509.342 us | 19.590 us | 19.599 us | mhc_post_boundary | `mhc_post_tilelang_kernel` |
| 75 | 0.64% | 123.294 us | 4.742 us | 4.719 us | mhc_pre | `triton_red_fused__to_copy_mean_pow_view_0` |
| 76 | 1.06% | 204.659 us | 7.872 us | 7.839 us | other | `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT16x16x64_MI16x16x1_SN_LDSB0_AFC0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_GRPM1_GRVWA4_GR...WSGRB0_WS64_WG16_4_4` |
| 77 | 0.86% | 165.336 us | 6.359 us | 6.339 us | other | `Cijk_SS_BiasS_HAS_ScaleAlphaVec_PostGSU16_VW1` |
| 78 | 0.58% | 112.776 us | 4.338 us | 4.319 us | mhc_pre | `triton_per_fused_mean_pow_1` |
| 79 | 0.57% | 109.416 us | 4.208 us | 4.199 us | mhc_pre | `triton_poi_fused_add_mean_mul_pow_rsqrt_2` |
| 80 | 0.84% | 162.215 us | 6.239 us | 6.219 us | mhc_pre | `hc_split_sinkhorn_kernel__kernel` |
| 81 | 0.63% | 120.857 us | 4.648 us | 4.639 us | other | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bina...ambda(int, bool)#1})` |
| 82 | 0.66% | 127.015 us | 4.885 us | 4.879 us | other | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator(...d int, float, 4, 4>)` |
| 83 | 0.63% | 121.294 us | 4.665 us | 4.639 us | copy_cast | `void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> ...::array<char*, 2ul>)` |
| 84 | 0.61% | 116.814 us | 4.493 us | 4.499 us | norm | `_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDF16bLi256ELi32ELb0ELb0ELb1ELi1EEEvPT0_PT_PfS4_S4_S4_diiiiiiib` |
| 85 | 0.57% | 109.176 us | 4.199 us | 4.199 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 86 | 5.82% | 1121.910 us | 43.150 us | 43.119 us | linear_quant_gemm | `void ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle<ck::GridwiseGemmMultiD_blockscale_xdl_cshuffle_v3_b_preshuffle<ck::tensor_layout::gemm:...8_fnuz_t>::Argument)` |
| 87 | 0.61% | 117.255 us | 4.510 us | 4.519 us | other | `_ZN7sgl_hip10activation18act_and_mul_kernelI14__hip_bfloat16TnPFT_RKS3_EXadL_Z4siluIS2_ES3_S5_EEEEvPS3_PS4_i` |
| 88 | 0.57% | 110.456 us | 4.248 us | 4.239 us | linear_quant_gemm | `_ZN5aiter37dynamic_per_group_scaled_quant_kernelIDF16bDB8_Li32EEEvPT0_PfPKT_PKfiliibPKii` |
| 89 | 0.72% | 138.015 us | 5.308 us | 5.279 us | linear_quant_gemm | `void ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle<ck::GridwiseGemmMultiD_blockscale_xdl_cshuffle_v3_b_preshuffle<ck::tensor_layout::gemm:...8_fnuz_t>::Argument)` |
| 90 | 1.21% | 233.618 us | 8.985 us | 8.999 us | other | `hgemm_bf16_32x64x64_S2TN_AS_SPK16_BS_0` |
| 91 | 1.1% | 211.414 us | 8.131 us | 8.059 us | moe | `void aiter::grouped_topk_kernel<c10::BFloat16, float __vector(4), 8, true, true, false>(c10::BFloat16*, c10::BFloat16 const*, float*, int*, unsigned long, i...nt, int, int, float)` |
| 92 | 2.03% | 391.778 us | 15.068 us | 14.980 us | moe | `void ck_tile::kentry<2, ck_tile::MoeSortingKernel<ck_tile::MoeSortingProblemEx<int, float, 2, true, false, false, true, 0> >, ck_tile::MoeSortingKernel<ck_t..., true, 0> >::Kargs)` |
| 93 | 0.75% | 145.135 us | 5.582 us | 5.579 us | moe | `_ZN5aiter27mxfp4_quant_moe_sort_kernelIDF16bN4opus5fp4_tELi256ELi32EEEvPT0_PhPKT_PKiSA_iiiiiiiii` |
| 94 | 3.83% | 737.907 us | 28.381 us | 28.359 us | moe | `void ck::kernel_moe_mxgemm_2lds<ck::GridwiseMoeGemmMX_BPreshuffle<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, ck::Tuple<ck::ten...4x2_pk_t>::Argument)` |
| 95 | 0.69% | 133.416 us | 5.131 us | 5.119 us | moe | `_ZN5aiter27mxfp4_quant_moe_sort_kernelIDF16bN4opus5fp4_tELi64ELi8EEEvPT0_PhPKT_PKiSA_iiiiiiiii` |
| 96 | 2.33% | 449.220 us | 17.278 us | 17.299 us | moe | `void ck::kernel_moe_mxgemm_2lds<ck::GridwiseMoeGemmMX_BPreshuffle<ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::ColumnMajor, ck::Tuple<ck::ten...4x2_pk_t>::Argument)` |
| 97 | 0.61% | 117.214 us | 4.508 us | 4.519 us | other | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10:...::array<char*, 3ul>)` |
| 98 | 1.56% | 300.497 us | 11.558 us | 11.579 us | all_reduce | `_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8RankDataES2_NS_11RankSignalsEPNS_6SignalEPT_ii` |
| 99 | 2.65% | 510.698 us | 19.642 us | 19.639 us | mhc_post_boundary | `mhc_post_tilelang_kernel` |
