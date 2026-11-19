# Previous tree vs this head: kernel launches per branch variant

`RUN=r9 scope_r6.sh`: per branch variant, `trace_sgl_kernel.py record` (every benchmark shape) and `trace_entry_points.py record` (every `TopKSelector` entry point) on `sglang-prev/` (`git archive` of `b4861e8`, before the rebase; same `python/` as `e887cac`) and `sglang/` (this head after the rebase onto `d75d741`), one GPU per process, then `compare --names previous this`.

## `sgl-kernel`

Benchmark shapes:

| suite | shapes | identical kernel sequence (name, grid, block, smem, regs) | identical output | identical output, rows sorted | valid top-k, previous | valid top-k, this |
|---|---:|---:|---:|---:|---:|---:|
| paged | 117 | 117 | 0 | 117 | 117 | 117 |
| paged16 | 39 | 39 | 16 | 18 | 39 | 39 |
| sparse | 8 | 8 | 0 | 0 | 8 | 8 |
| blocks | 20 | 20 | 0 | 10 | 20 | 20 |

shapes missing on one side: 0
kernels launched by the previous variant: [
 "sglang::topk_mode_1::topk_plan_cluster(unsigned int const*, sglang::topk_mode_1::PlanItem*, unsigned int, int)",
 "void sglang::amax8_varlen_kernel<true, __nv_bfloat16, float>(sglang::BlockAmaxParams<__nv_bfloat16, float>)",
 "void sglang::topk_bf16_small_kernel<true>(sglang::TopKBF16Params)",
 "void sglang::topk_mode_1::topk_main_kernel<true, 0, (sglang::topk_mode_1::TopKMode)0>(sglang::topk_mode_1::TopKPagedParams)",
 "void sglang::topk_mode_1::topk_main_kernel<true, 0, (sglang::topk_mode_1::TopKMode)1>(sglang::topk_mode_1::TopKPagedParams)",
 "void sglang::topk_mode_1::topk_main_kernel<true, 1, (sglang::topk_mode_1::TopKMode)0>(sglang::topk_mode_1::TopKPagedParams)",
 "void sglang::topk_mode_1::topk_main_kernel<true, 1, (sglang::topk_mode_1::TopKMode)1>(sglang::topk_mode_1::TopKPagedParams)",
 "void sglang::topk_mode_1::topk_main_kernel<true, 3, (sglang::topk_mode_1::TopKMode)0>(sglang::topk_mode_1::TopKPagedParams)",
 "void sglang::topk_mode_1::topk_main_kernel<true, 3, (sglang::topk_mode_1::TopKMode)1>(sglang::topk_mode_1::TopKPagedParams)",
 "void sglang::topk_mode_1::topk_persistent_cluster_kernel<true, 8u>(sglang::topk_mode_1::TopKPagedParams)",
 "void sglang::topk_mode_1::topk_small_batch_cluster_kernel<true, (sglang::topk_mode_1::TopKMode)0, 16u, 1u>(sglang::topk_mode_1::TopKPagedParams)",
 "void sglang::topk_mode_1::topk_small_batch_cluster_kernel<true, (sglang::topk_mode_1::TopKMode)0, 8u, 2u>(sglang::topk_mode_1::TopKPagedParams)",
 "void sglang::topk_mode_1::topk_small_batch_cluster_kernel<true, (sglang::topk_mode_1::TopKMode)1, 16u, 1u>(sglang::topk_mode_1::TopKPagedParams)",
 "void sglang::topk_mode_1::topk_small_batch_cluster_kernel<true, (sglang::topk_mode_1::TopKMode)1, 8u, 2u>(sglang::topk_mode_1::TopKPagedParams)",
 "void topk_select_bf16_cluster::topk_kernel<topk_select_bf16_cluster::TopkSelectKernelBF16Cluster<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 1024u, 512u, 1u, 8192u, 4096u, 5u, 512u, 16u, true> > >(TopkSelectArgs, topk_select_bf16_cluster::TopkSelectKernelBF16Cluster<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 1024u, 512u, 1u, 8192u, 4096u, 5u, 512u, 16u, true> >::TmaParams)",
 "void topk_select_bf16_cluster::topk_kernel<topk_select_bf16_cluster::TopkSelectKernelBF16Cluster<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 1024u, 512u, 1u, 8192u, 4096u, 5u, 512u, 2u, true> > >(TopkSelectArgs, topk_select_bf16_cluster::TopkSelectKernelBF16Cluster<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 1024u, 512u, 1u, 8192u, 4096u, 5u, 512u, 2u, true> >::TmaParams)",
 "void topk_select_bf16_cluster::topk_kernel<topk_select_bf16_cluster::TopkSelectKernelBF16Cluster<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 1024u, 512u, 1u, 8192u, 4096u, 5u, 512u, 4u, true> > >(TopkSelectArgs, topk_select_bf16_cluster::TopkSelectKernelBF16Cluster<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 1024u, 512u, 1u, 8192u, 4096u, 5u, 512u, 4u, true> >::TmaParams)",
 "void topk_select_bf16_normal::topk_kernel<topk_select_bf16_normal::TopkSelectKernelBF16Normal<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 512u, 512u, 1u, 8192u, 4096u, 5u, 512u, 1u, true> > >(TopkSelectArgs, topk_select_bf16_normal::TopkSelectKernelBF16Normal<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 512u, 512u, 1u, 8192u, 4096u, 5u, 512u, 1u, true> >::TmaParams)"
]

Entry points:

| entry point | kernels, previous | identical kernel sequence | identical output | identical output, rows sorted |
|---|---:|---|---|---|
| ragged bf16 64x8192 | 1 | yes | yes | yes |
| ragged bf16 8x65536 | 1 | yes | no | no |
| ragged fp32 64x8192 | 1 | yes | no | yes |
| ragged fp32 8x65536 | 1 | yes | no | yes |
| ragged fp32 block keys 32x4096 k2048 | 1 | yes | no | yes |
| raw fp32 32x16384 | 1 | yes | no | yes |
| dense fp32 32x3000 sorted=False | 1 | yes | yes | yes |
| dense fp32 32x3000 sorted=True | 2 | yes | yes | yes |
| sparse bf16 64x16384 outputs=1 | 1 | yes | no | no |
| sparse bf16 64x16384 outputs=2 | 2 | yes | no | no |
| order_candidates 64x64 | 0 | yes | yes | yes |
| paged bf16 32x16384 | 1 | yes | yes | yes |
| paged fp32 v2 32x16384 | 1 | yes | no | yes |
| paged fp32 v2 row chunk + plan 32x16384 | 1 | yes | no | yes |
| paged fp32 v1 32x16384 | 1 | yes | no | yes |
| amax_blocks bf16 16x65536 k2048 | 4 | yes | no | yes |
| select_candidate_block_ids 16x4000 | 12 | yes | yes | yes |
| topk_among_blocks 16x4000 n96 | 19 | yes | yes | yes |
| candidate_topk 16x768 | 10 | yes | yes | yes |

entry points with identical kernel sequences: 19 / 19; only in this: []

## `sgl-kernel-mode2`

Benchmark shapes:

| suite | shapes | identical kernel sequence (name, grid, block, smem, regs) | identical output | identical output, rows sorted | valid top-k, previous | valid top-k, this |
|---|---:|---:|---:|---:|---:|---:|
| paged | 117 | 117 | 0 | 117 | 117 | 117 |
| paged16 | 39 | 39 | 16 | 17 | 39 | 39 |
| sparse | 8 | 8 | 0 | 0 | 8 | 8 |
| blocks | 20 | 20 | 0 | 20 | 20 | 20 |

shapes missing on one side: 0
kernels launched by the previous variant: [
 "sglang::topk_mode_2::topk_plan_cluster(unsigned int const*, sglang::topk_mode_2::PlanItem*, unsigned int, int)",
 "void sglang::amax8_varlen_kernel<true, __nv_bfloat16, float>(sglang::BlockAmaxParams<__nv_bfloat16, float>)",
 "void sglang::topk_bf16_small_kernel<true>(sglang::TopKBF16Params)",
 "void sglang::topk_mode_2::topk_main_kernel<true, 0, (sglang::topk_mode_2::TopKMode)0>(sglang::topk_mode_2::TopKPagedParams)",
 "void sglang::topk_mode_2::topk_main_kernel<true, 0, (sglang::topk_mode_2::TopKMode)1>(sglang::topk_mode_2::TopKPagedParams)",
 "void sglang::topk_mode_2::topk_main_kernel<true, 1, (sglang::topk_mode_2::TopKMode)0>(sglang::topk_mode_2::TopKPagedParams)",
 "void sglang::topk_mode_2::topk_main_kernel<true, 1, (sglang::topk_mode_2::TopKMode)1>(sglang::topk_mode_2::TopKPagedParams)",
 "void sglang::topk_mode_2::topk_main_kernel<true, 3, (sglang::topk_mode_2::TopKMode)0>(sglang::topk_mode_2::TopKPagedParams)",
 "void sglang::topk_mode_2::topk_main_kernel<true, 3, (sglang::topk_mode_2::TopKMode)1>(sglang::topk_mode_2::TopKPagedParams)",
 "void sglang::topk_mode_2::topk_persistent_cluster_kernel<true, 8u>(sglang::topk_mode_2::TopKPagedParams)",
 "void sglang::topk_mode_2::topk_small_batch_cluster_kernel<true, (sglang::topk_mode_2::TopKMode)0, 16u, 1u>(sglang::topk_mode_2::TopKPagedParams)",
 "void sglang::topk_mode_2::topk_small_batch_cluster_kernel<true, (sglang::topk_mode_2::TopKMode)0, 8u, 2u>(sglang::topk_mode_2::TopKPagedParams)",
 "void sglang::topk_mode_2::topk_small_batch_cluster_kernel<true, (sglang::topk_mode_2::TopKMode)1, 16u, 1u>(sglang::topk_mode_2::TopKPagedParams)",
 "void sglang::topk_mode_2::topk_small_batch_cluster_kernel<true, (sglang::topk_mode_2::TopKMode)1, 8u, 2u>(sglang::topk_mode_2::TopKPagedParams)",
 "void topk_select_bf16_cluster::topk_kernel<topk_select_bf16_cluster::TopkSelectKernelBF16Cluster<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 1024u, 512u, 1u, 8192u, 4096u, 5u, 512u, 16u, true> > >(TopkSelectArgs, topk_select_bf16_cluster::TopkSelectKernelBF16Cluster<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 1024u, 512u, 1u, 8192u, 4096u, 5u, 512u, 16u, true> >::TmaParams)",
 "void topk_select_bf16_cluster::topk_kernel<topk_select_bf16_cluster::TopkSelectKernelBF16Cluster<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 1024u, 512u, 1u, 8192u, 4096u, 5u, 512u, 2u, true> > >(TopkSelectArgs, topk_select_bf16_cluster::TopkSelectKernelBF16Cluster<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 1024u, 512u, 1u, 8192u, 4096u, 5u, 512u, 2u, true> >::TmaParams)",
 "void topk_select_bf16_cluster::topk_kernel<topk_select_bf16_cluster::TopkSelectKernelBF16Cluster<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 1024u, 512u, 1u, 8192u, 4096u, 5u, 512u, 4u, true> > >(TopkSelectArgs, topk_select_bf16_cluster::TopkSelectKernelBF16Cluster<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 1024u, 512u, 1u, 8192u, 4096u, 5u, 512u, 4u, true> >::TmaParams)",
 "void topk_select_bf16_normal::topk_kernel<topk_select_bf16_normal::TopkSelectKernelBF16Normal<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 512u, 512u, 1u, 8192u, 4096u, 5u, 512u, 1u, true> > >(TopkSelectArgs, topk_select_bf16_normal::TopkSelectKernelBF16Normal<TopkSelectConfig<__nv_bfloat16, int, false, false, false, 512u, 512u, 1u, 8192u, 4096u, 5u, 512u, 1u, true> >::TmaParams)"
]

Entry points:

| entry point | kernels, previous | identical kernel sequence | identical output | identical output, rows sorted |
|---|---:|---|---|---|
| ragged bf16 64x8192 | 1 | yes | yes | yes |
| ragged bf16 8x65536 | 1 | yes | no | no |
| ragged fp32 64x8192 | 1 | yes | no | yes |
| ragged fp32 8x65536 | 1 | yes | no | yes |
| ragged fp32 block keys 32x4096 k2048 | 1 | yes | no | yes |
| raw fp32 32x16384 | 1 | yes | no | yes |
| dense fp32 32x3000 sorted=False | 1 | yes | yes | yes |
| dense fp32 32x3000 sorted=True | 2 | yes | yes | yes |
| sparse bf16 64x16384 outputs=1 | 1 | yes | no | no |
| sparse bf16 64x16384 outputs=2 | 2 | yes | no | no |
| order_candidates 64x64 | 0 | yes | yes | yes |
| paged bf16 32x16384 | 1 | yes | yes | yes |
| paged fp32 v2 32x16384 | 1 | yes | no | yes |
| paged fp32 v2 row chunk + plan 32x16384 | 1 | yes | no | yes |
| paged fp32 v1 32x16384 | 1 | yes | no | yes |
| amax_blocks bf16 16x65536 k2048 | 4 | yes | no | yes |
| select_candidate_block_ids 16x4000 | 12 | yes | yes | yes |
| topk_among_blocks 16x4000 n96 | 19 | yes | yes | yes |
| candidate_topk 16x768 | 10 | yes | yes | yes |

entry points with identical kernel sequences: 19 / 19; only in this: []

## `flashinfer`

Benchmark shapes:

| suite | shapes | identical kernel sequence (name, grid, block, smem, regs) | identical output | identical output, rows sorted | valid top-k, previous | valid top-k, this |
|---|---:|---:|---:|---:|---:|---:|
| paged | 117 | 117 | 0 | 117 | 117 | 117 |
| paged16 | 39 | 39 | 0 | 14 | 39 | 39 |
| sparse | 8 | 8 | 0 | 1 | 8 | 8 |
| blocks | 20 | 20 | 0 | 1 | 20 | 20 |

shapes missing on one side: 0
kernels launched by the previous variant: [
 "(anonymous namespace)::CUBFillTopKTailsKernel(int*, int*, int const*, long, long)",
 "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int)#1})",
 "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul>, false>(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)",
 "void cub::_V_300600_SM_1030::detail::batched_topk::device_batched_topk_kernel<cub::_V_300600_SM_1030::detail::batched_topk::policy_selector_from_types<__nv_bfloat16, int, 2097152l, 2097152l, (cuda::execution::__4::determinism::__determinism_t)0, (cuda::execution::__4::tie_break::__tie_break_t)0, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 2097152> >, cuda::__4::strided_iterator<cuda::__4::counting_iterator<__nv_bfloat16 const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 2097152l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1> const, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, long>, cuda::__4::strided_iterator<cuda::__4::counting_iterator<__nv_bfloat16 const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 2097152> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 2097152l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1>, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, long, (cuda::execution::__4::determinism::__determinism_t)0, (cuda::execution::__4::tie_break::__tie_break_t)0>(cuda::__4::strided_iterator<cuda::__4::counting_iterator<__nv_bfloat16 const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 2097152> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 2097152l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1>, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, cub::_V_300600_SM_1030::detail::batched_topk::baseline_kernel_args<cuda::args::__4::__traits<cuda::args::__4::immediate<long, cuda::args::__4::no_bounds> >::element_type, long>, cub::_V_300600_SM_1030::detail::batched_topk::cluster_kernel_args)",
 "void cub::_V_300600_SM_1030::detail::batched_topk::device_batched_topk_kernel<cub::_V_300600_SM_1030::detail::batched_topk::policy_selector_from_types<float, int, 2097152l, 2097152l, (cuda::execution::__4::determinism::__determinism_t)0, (cuda::execution::__4::tie_break::__tie_break_t)0, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 2097152> >, cuda::__4::strided_iterator<cuda::__4::counting_iterator<float const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 2097152l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1> const, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, long>, cuda::__4::strided_iterator<cuda::__4::counting_iterator<float const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 2097152> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 2097152l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1>, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, long, (cuda::execution::__4::determinism::__determinism_t)0, (cuda::execution::__4::tie_break::__tie_break_t)0>(cuda::__4::strided_iterator<cuda::__4::counting_iterator<float const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 2097152> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 2097152l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1>, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, cub::_V_300600_SM_1030::detail::batched_topk::baseline_kernel_args<cuda::args::__4::__traits<cuda::args::__4::immediate<long, cuda::args::__4::no_bounds> >::element_type, long>, cub::_V_300600_SM_1030::detail::batched_topk::cluster_kernel_args)",
 "void flashinfer::sampling::FilteredTopKUnifiedKernel<__nv_bfloat16, int, 1, false, (flashinfer::sampling::FilteredTopKMode)1, (flashinfer::sampling::TopKTieBreak)0, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int> >(__nv_bfloat16 const*, int*, __nv_bfloat16*, int const*, long, int const*, int const*, int const*, int const*, unsigned int, unsigned int, unsigned int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>)",
 "void flashinfer::sampling::FilteredTopKUnifiedKernel<float, int, 1, false, (flashinfer::sampling::FilteredTopKMode)1, (flashinfer::sampling::TopKTieBreak)0, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int> >(float const*, int*, float*, int const*, long, int const*, int const*, int const*, int const*, unsigned int, unsigned int, unsigned int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>)",
 "void sglang::amax8_varlen_kernel<true, __nv_bfloat16, float>(sglang::BlockAmaxParams<__nv_bfloat16, float>)"
]

Entry points:

| entry point | kernels, previous | identical kernel sequence | identical output | identical output, rows sorted |
|---|---:|---|---|---|
| ragged bf16 64x8192 | 7 | yes | no | no |
| ragged bf16 8x65536 | 8 | yes | no | yes |
| ragged fp32 64x8192 | 7 | yes | no | yes |
| ragged fp32 8x65536 | 8 | yes | no | yes |
| ragged fp32 block keys 32x4096 k2048 | 7 | yes | no | yes |
| raw fp32 32x16384 | 3 | yes | no | yes |
| dense fp32 32x3000 sorted=False | 1 | yes | no | yes |
| dense fp32 32x3000 sorted=True | 5 | yes | yes | yes |
| sparse bf16 64x16384 outputs=1 | 1 | yes | no | no |
| sparse bf16 64x16384 outputs=2 | 11 | yes | no | no |
| order_candidates 64x64 | 8 | yes | yes | yes |
| paged bf16 32x16384 | 1 | yes | no | no |
| paged fp32 v2 32x16384 | 1 | yes | no | yes |
| paged fp32 v2 row chunk + plan 32x16384 | 1 | yes | no | yes |
| paged fp32 v1 32x16384 | 1 | yes | no | yes |
| amax_blocks bf16 16x65536 k2048 | 6 | yes | no | no |
| select_candidate_block_ids 16x4000 | 15 | yes | yes | yes |
| topk_among_blocks 16x4000 n96 | 19 | yes | no | yes |
| candidate_topk 16x768 | 10 | yes | no | yes |

entry points with identical kernel sequences: 19 / 19; only in this: []

## `flashinfer-small`

Benchmark shapes:

| suite | shapes | identical kernel sequence (name, grid, block, smem, regs) | identical output | identical output, rows sorted | valid top-k, previous | valid top-k, this |
|---|---:|---:|---:|---:|---:|---:|
| paged | 117 | 117 | 0 | 117 | 117 | 117 |
| paged16 | 39 | 39 | 0 | 39 | 39 | 39 |
| sparse | 8 | 8 | 0 | 8 | 8 | 8 |
| blocks | 20 | 20 | 0 | 20 | 20 | 20 |

shapes missing on one side: 0
kernels launched by the previous variant: [
 "(anonymous namespace)::CUBFillTopKTailsKernel(int*, int*, int const*, long, long)",
 "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int)#1})",
 "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul>, false>(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)",
 "void cub::_V_300600_SM_1030::detail::batched_topk::device_batched_topk_kernel<cub::_V_300600_SM_1030::detail::batched_topk::policy_selector_from_types<__nv_bfloat16, int, 2097152l, 2097152l, (cuda::execution::__4::determinism::__determinism_t)2, (cuda::execution::__4::tie_break::__tie_break_t)1, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 2097152> >, cuda::__4::strided_iterator<cuda::__4::counting_iterator<__nv_bfloat16 const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 2097152l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1> const, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, long>, cuda::__4::strided_iterator<cuda::__4::counting_iterator<__nv_bfloat16 const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 2097152> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 2097152l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1>, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, long, (cuda::execution::__4::determinism::__determinism_t)2, (cuda::execution::__4::tie_break::__tie_break_t)1>(cuda::__4::strided_iterator<cuda::__4::counting_iterator<__nv_bfloat16 const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 2097152> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 2097152l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1>, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, cub::_V_300600_SM_1030::detail::batched_topk::baseline_kernel_args<cuda::args::__4::__traits<cuda::args::__4::immediate<long, cuda::args::__4::no_bounds> >::element_type, long>, cub::_V_300600_SM_1030::detail::batched_topk::cluster_kernel_args)",
 "void cub::_V_300600_SM_1030::detail::batched_topk::device_batched_topk_kernel<cub::_V_300600_SM_1030::detail::batched_topk::policy_selector_from_types<float, int, 2097152l, 2097152l, (cuda::execution::__4::determinism::__determinism_t)2, (cuda::execution::__4::tie_break::__tie_break_t)1, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 2097152> >, cuda::__4::strided_iterator<cuda::__4::counting_iterator<float const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 2097152l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1> const, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, long>, cuda::__4::strided_iterator<cuda::__4::counting_iterator<float const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 2097152> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 2097152l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1>, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, long, (cuda::execution::__4::determinism::__determinism_t)2, (cuda::execution::__4::tie_break::__tie_break_t)1>(cuda::__4::strided_iterator<cuda::__4::counting_iterator<float const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 2097152> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 2097152l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1>, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, cub::_V_300600_SM_1030::detail::batched_topk::baseline_kernel_args<cuda::args::__4::__traits<cuda::args::__4::immediate<long, cuda::args::__4::no_bounds> >::element_type, long>, cub::_V_300600_SM_1030::detail::batched_topk::cluster_kernel_args)",
 "void cub::_V_300600_SM_1030::detail::batched_topk::device_batched_topk_kernel<cub::_V_300600_SM_1030::detail::batched_topk::policy_selector_from_types<float, int, 8192l, 8192l, (cuda::execution::__4::determinism::__determinism_t)2, (cuda::execution::__4::tie_break::__tie_break_t)1, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 8192> >, cuda::__4::strided_iterator<cuda::__4::counting_iterator<float const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 8192l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1> const, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, long>, cuda::__4::strided_iterator<cuda::__4::counting_iterator<float const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 8192> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 8192l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1>, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, long, (cuda::execution::__4::determinism::__determinism_t)2, (cuda::execution::__4::tie_break::__tie_break_t)1>(cuda::__4::strided_iterator<cuda::__4::counting_iterator<float const*, long, 0, 0, 0>, long>, cuda::__4::constant_iterator<cuda::__4::discard_iterator, long>, cuda::__4::constant_iterator<cuda::__4::counting_iterator<int, long, 0, 0, 0>, long>, cuda::__4::transform_iterator<(anonymous namespace)::CUBMakePageTableRowOut<false>, cuda::__4::counting_iterator<long, __int128, 0, 0, 0> >, cuda::args::__4::deferred_sequence<int const*, cuda::args::__4::static_bounds<-2147483648, 8192> >, cuda::args::__4::immediate<long, cuda::args::__4::static_bounds<1l, 8192l> >, cub::_V_300600_SM_1030::detail::params::static_discrete_param<cub::_V_300600_SM_1030::detail::topk::select, (cub::_V_300600_SM_1030::detail::topk::select)1>, cuda::args::__4::immediate<long, cuda::args::__4::no_bounds>, cub::_V_300600_SM_1030::detail::batched_topk::baseline_kernel_args<cuda::args::__4::__traits<cuda::args::__4::immediate<long, cuda::args::__4::no_bounds> >::element_type, long>, cub::_V_300600_SM_1030::detail::batched_topk::cluster_kernel_args)",
 "void flashinfer::sampling::FilteredTopKUnifiedKernel<__nv_bfloat16, int, 1, true, (flashinfer::sampling::FilteredTopKMode)1, (flashinfer::sampling::TopKTieBreak)1, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int> >(__nv_bfloat16 const*, int*, __nv_bfloat16*, int const*, long, int const*, int const*, int const*, int const*, unsigned int, unsigned int, unsigned int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>)",
 "void flashinfer::sampling::FilteredTopKUnifiedKernel<float, int, 1, true, (flashinfer::sampling::FilteredTopKMode)1, (flashinfer::sampling::TopKTieBreak)1, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int> >(float const*, int*, float*, int const*, long, int const*, int const*, int const*, int const*, unsigned int, unsigned int, unsigned int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>)",
 "void flashinfer::sampling::FinalizeTopKIndicesKernel<false, (flashinfer::sampling::FilteredTopKMode)1, 128u, 8u, unsigned char, int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int> >(int*, unsigned char*, int const*, long, int const*, int const*, unsigned int, unsigned int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>)",
 "void flashinfer::sampling::FinalizeTopKIndicesKernel<false, (flashinfer::sampling::FilteredTopKMode)1, 256u, 8u, unsigned char, int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int> >(int*, unsigned char*, int const*, long, int const*, int const*, unsigned int, unsigned int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>)",
 "void flashinfer::sampling::FinalizeTopKIndicesKernel<false, (flashinfer::sampling::FilteredTopKMode)1, 64u, 8u, unsigned char, int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int> >(int*, unsigned char*, int const*, long, int const*, int const*, unsigned int, unsigned int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>)",
 "void sglang::amax8_varlen_kernel<true, __nv_bfloat16, float>(sglang::BlockAmaxParams<__nv_bfloat16, float>)"
]

Entry points:

| entry point | kernels, previous | identical kernel sequence | identical output | identical output, rows sorted |
|---|---:|---|---|---|
| ragged bf16 64x8192 | 8 | yes | no | yes |
| ragged bf16 8x65536 | 8 | yes | no | yes |
| ragged fp32 64x8192 | 8 | yes | no | yes |
| ragged fp32 8x65536 | 8 | yes | no | yes |
| ragged fp32 block keys 32x4096 k2048 | 8 | yes | no | yes |
| raw fp32 32x16384 | 4 | yes | no | yes |
| dense fp32 32x3000 sorted=False | 1 | yes | no | yes |
| dense fp32 32x3000 sorted=True | 5 | yes | yes | yes |
| sparse bf16 64x16384 outputs=1 | 2 | yes | no | yes |
| sparse bf16 64x16384 outputs=2 | 12 | yes | no | yes |
| order_candidates 64x64 | 8 | yes | yes | yes |
| paged bf16 32x16384 | 2 | yes | no | yes |
| paged fp32 v2 32x16384 | 2 | yes | no | yes |
| paged fp32 v2 row chunk + plan 32x16384 | 2 | yes | no | yes |
| paged fp32 v1 32x16384 | 2 | yes | no | yes |
| amax_blocks bf16 16x65536 k2048 | 7 | yes | no | yes |
| select_candidate_block_ids 16x4000 | 15 | yes | yes | yes |
| topk_among_blocks 16x4000 n96 | 19 | yes | no | yes |
| candidate_topk 16x768 | 10 | yes | no | yes |

entry points with identical kernel sequences: 19 / 19; only in this: []

## `flashinfer-small-det`

Benchmark shapes:

| suite | shapes | identical kernel sequence (name, grid, block, smem, regs) | identical output | identical output, rows sorted | valid top-k, previous | valid top-k, this |
|---|---:|---:|---:|---:|---:|---:|
| paged | 117 | 117 | 117 | 117 | 117 | 117 |
| paged16 | 39 | 39 | 39 | 39 | 39 | 39 |
| sparse | 8 | 8 | 8 | 8 | 8 | 8 |
| blocks | 20 | 20 | 20 | 20 | 20 | 20 |

shapes missing on one side: 0
kernels launched by the previous variant: [
 "void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int)#1})",
 "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul>, false>(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)",
 "void flashinfer::sampling::FilteredTopKUnifiedKernel<__nv_bfloat16, int, 1, true, (flashinfer::sampling::FilteredTopKMode)1, (flashinfer::sampling::TopKTieBreak)1, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int> >(__nv_bfloat16 const*, int*, __nv_bfloat16*, int const*, long, int const*, int const*, int const*, int const*, unsigned int, unsigned int, unsigned int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>)",
 "void flashinfer::sampling::FilteredTopKUnifiedKernel<float, int, 1, true, (flashinfer::sampling::FilteredTopKMode)1, (flashinfer::sampling::TopKTieBreak)1, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int> >(float const*, int*, float*, int const*, long, int const*, int const*, int const*, int const*, unsigned int, unsigned int, unsigned int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>)",
 "void flashinfer::sampling::FinalizeTopKIndicesKernel<true, (flashinfer::sampling::FilteredTopKMode)1, 128u, 8u, unsigned char, int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int> >(int*, unsigned char*, int const*, long, int const*, int const*, unsigned int, unsigned int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>)",
 "void flashinfer::sampling::FinalizeTopKIndicesKernel<true, (flashinfer::sampling::FilteredTopKMode)1, 256u, 8u, unsigned char, int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int> >(int*, unsigned char*, int const*, long, int const*, int const*, unsigned int, unsigned int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>)",
 "void flashinfer::sampling::FinalizeTopKIndicesKernel<true, (flashinfer::sampling::FilteredTopKMode)1, 64u, 8u, unsigned char, int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int> >(int*, unsigned char*, int const*, long, int const*, int const*, unsigned int, unsigned int, flashinfer::sampling::ConfigurablePageTableKernelPolicy<int>)",
 "void sglang::amax8_varlen_kernel<true, __nv_bfloat16, float>(sglang::BlockAmaxParams<__nv_bfloat16, float>)"
]

Entry points:

| entry point | kernels, previous | identical kernel sequence | identical output | identical output, rows sorted |
|---|---:|---|---|---|
| ragged bf16 64x8192 | 8 | yes | yes | yes |
| ragged bf16 8x65536 | 8 | yes | yes | yes |
| ragged fp32 64x8192 | 8 | yes | yes | yes |
| ragged fp32 8x65536 | 8 | yes | yes | yes |
| ragged fp32 block keys 32x4096 k2048 | 8 | yes | yes | yes |
| raw fp32 32x16384 | 4 | yes | yes | yes |
| dense fp32 32x3000 sorted=False | 3 | yes | yes | yes |
| dense fp32 32x3000 sorted=True | 7 | yes | yes | yes |
| sparse bf16 64x16384 outputs=1 | 2 | yes | yes | yes |
| sparse bf16 64x16384 outputs=2 | 12 | yes | yes | yes |
| order_candidates 64x64 | 8 | yes | yes | yes |
| paged bf16 32x16384 | 2 | yes | yes | yes |
| paged fp32 v2 32x16384 | 2 | yes | yes | yes |
| paged fp32 v2 row chunk + plan 32x16384 | 2 | yes | yes | yes |
| paged fp32 v1 32x16384 | 2 | yes | yes | yes |
| amax_blocks bf16 16x65536 k2048 | 7 | yes | yes | yes |
| select_candidate_block_ids 16x4000 | 17 | yes | yes | yes |
| topk_among_blocks 16x4000 n96 | 21 | yes | yes | yes |
| candidate_topk 16x768 | 12 | yes | yes | yes |

entry points with identical kernel sequences: 19 / 19; only in this: []
