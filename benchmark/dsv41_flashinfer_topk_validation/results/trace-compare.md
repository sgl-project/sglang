| suite | shapes | identical kernel sequence (name, grid, block, smem, regs) | identical output | identical output, rows sorted | valid top-k, base | valid top-k, branch |
|---|---:|---:|---:|---:|---:|---:|
| paged | 117 | 117 | 0 | 117 | 117 | 117 |
| paged16 | 39 | 39 | 16 | 17 | 39 | 39 |
| sparse | 8 | 8 | 0 | 0 | 8 | 8 |
| blocks | 20 | 20 | 0 | 10 | 20 | 20 |

Base run twice (same tree, same inputs):

| suite | shapes | identical kernel sequence (name, grid, block, smem, regs) | identical output | identical output, rows sorted | valid top-k, base | valid top-k, base again |
|---|---:|---:|---:|---:|---:|---:|
| paged | 117 | 117 | 0 | 117 | 117 | 117 |
| paged16 | 39 | 39 | 16 | 16 | 39 | 39 |
| sparse | 8 | 8 | 0 | 0 | 8 | 8 |
| blocks | 20 | 20 | 0 | 10 | 20 | 20 |

shapes missing on one side: 0
kernels launched by the base variant: [
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
