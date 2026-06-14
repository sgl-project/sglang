// Copyright 2026 SGLang Team. Licensed under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with the
// License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// CUTLASS blockwise-scaled FP8 grouped-GEMM kernels for IFMOE on Hopper
// (SM90). Originally embedded as a heredoc in kernel.cu's inline build
// script; extracted here so the sgl-kernel CMake build can compile it at
// wheel-build time (eliminating the runtime nvcc + bash + /tmp .so dance
// of the previous in-tree JIT path).

// CUTLASS blockwise FP8 grouped GEMM (Sm90/Hopper) — dual tile 64x128 + 128x128
#include <cuda_runtime.h>
#include <algorithm>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/util/packed_stride.hpp"
using namespace cute;
using EA = cutlass::float_e4m3_t; using EB = cutlass::float_e4m3_t;
using EAcc = float; using EC = cutlass::bfloat16_t;
using LA = cutlass::layout::RowMajor; using LB = cutlass::layout::ColumnMajor;
using LC = cutlass::layout::RowMajor;
// Sm90 blockwise scale config (MN-major for SFA/SFB; bundled CUTLASS comment:
// "Sm90 only supports MN major for SFA and SFB for now").
using SC = cutlass::detail::Sm90BlockwiseScaleConfig<1,128,128,cute::GMMA::Major::MN,cute::GMMA::Major::MN>;
using LSFA = decltype(SC::deduce_layoutSFA()); using LSFB = decltype(SC::deduce_layoutSFB());
using PS = cutlass::gemm::GroupProblemShape<Shape<int,int,int>>;
// Large-M cooperative tile 128x128x128
using Tl128 = Shape<_128,_128,_128>; using Cl1 = Shape<_1,_1,_1>;
using Ep128 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tl128,Cl1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    EAcc,EAcc,EC,LC*,8,EC,LC*,8,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
using Ml128 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    EA,cute::tuple<LA*,LSFA*>,16,EB,cute::tuple<LB*,LSFB*>,16,
    EAcc,Tl128,Cl1,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Ep128::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum>::CollectiveOp;
using Kn128 = cutlass::gemm::kernel::GemmUniversal<PS,Ml128,Ep128>;
using Gm128 = cutlass::gemm::device::GemmUniversalAdapter<Kn128>;
// Large-M cooperative tile 128x128x128 with Cluster<2,1,1> — 2-CTA cluster enables
// TMA multicast of B across the pair (gated on max_M_estimate > 2048).
using Cl2 = Shape<_2,_1,_1>;
using Ep128c = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tl128,Cl2,
    cutlass::epilogue::collective::EpilogueTileAuto,
    EAcc,EAcc,EC,LC*,8,EC,LC*,8,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
using Ml128c = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    EA,cute::tuple<LA*,LSFA*>,16,EB,cute::tuple<LB*,LSFB*>,16,
    EAcc,Tl128,Cl2,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Ep128c::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperativeFP8BlockScaledAccum>::CollectiveOp;
using Kn128c = cutlass::gemm::kernel::GemmUniversal<PS,Ml128c,Ep128c>;
using Gm128c = cutlass::gemm::device::GemmUniversalAdapter<Kn128c>;
// Small-M tile 64x128x128 using pingpong (cooperative requires M>=128)
using Tl64 = Shape<_64,_128,_128>;
using Ep64 = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,Tl64,Cl1,
    cutlass::epilogue::collective::EpilogueTileAuto,
    EAcc,EAcc,EC,LC*,8,EC,LC*,8,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong>::CollectiveOp;
using Ml64 = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,cutlass::arch::OpClassTensorOp,
    EA,cute::tuple<LA*,LSFA*>,16,EB,cute::tuple<LB*,LSFB*>,16,
    EAcc,Tl64,Cl1,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Ep64::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpongFP8BlockScaledAccum>::CollectiveOp;
using Kn64 = cutlass::gemm::kernel::GemmUniversal<PS,Ml64,Ep64>;
using Gm64 = cutlass::gemm::device::GemmUniversalAdapter<Kn64>;
using StA = typename Gm128::GemmKernel::InternalStrideA;
using StB = typename Gm128::GemmKernel::InternalStrideB;
using StD = typename Gm128::GemmKernel::InternalStrideD;
using ILSFA = cute::remove_pointer_t<typename Ml128::LayoutSFA>;
using ILSFB = cute::remove_pointer_t<typename Ml128::LayoutSFB>;
__global__ void prep(EA* A, EB* B, EAcc* SFA, EAcc* SFB, EC* D,
    int* mi, int* expert_ids, int n, int k, int ng,
    PS::UnderlyingProblemShape* ps, const EA** pA, const EB** pB,
    const EAcc** pSFA, const EAcc** pSFB, EC** pD,
    StA* sA, StB* sB, StD* sD, ILSFA* lA, ILSFB* lB) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=ng) return;
#if (__CUDACC_VER_MAJOR__>=12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__>=900))
    asm volatile("griddepcontrol.wait;");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    int sk=k/128, sn=n/128, off=mi[i], m=mi[i+1]-off;
    int eid = expert_ids[i];
    ps[i]=PS::UnderlyingProblemShape(m,n,k);
    sA[i]=cutlass::make_cute_packed_stride(StA{},{m,k,1});
    sB[i]=cutlass::make_cute_packed_stride(StB{},{n,k,1});
    sD[i]=cutlass::make_cute_packed_stride(StD{},{m,n,1});
    pA[i]=A+int64_t(off)*k; pB[i]=B+int64_t(eid)*n*k;
    pD[i]=D+int64_t(off)*n;
    lA[i]=SC::tile_atom_to_shape_SFA(make_shape(m,n,k,1));
    // SFA/SFB pointers reference MN-major repacked scratch; scratch is laid out
    // per-group (not per-expert), so group i's slab starts at off*sk (SFA) / i*sn*sk (SFB).
    pSFA[i]=SFA+int64_t(off)*sk;
    lB[i]=SC::tile_atom_to_shape_SFB(make_shape(m,n,k,1));
    pSFB[i]=SFB+int64_t(i)*sn*sk;
}
// Element-wise scale-repack kernels (K-major model layout -> MN-major Sm90 layout).
// SFA: [T, K/128] K-major -> per-group [m_i, K/128] MN-major, contiguously in row order.
__global__ void repack_sfa(const EAcc* __restrict__ src, EAcc* __restrict__ dst,
                           const int* __restrict__ mi, int sk) {
    int i = blockIdx.x;
    int k_blk = blockIdx.y;
    int off = mi[i], m = mi[i+1]-off;
    int64_t gbase = int64_t(off)*sk;
    for (int lm = threadIdx.x; lm < m; lm += blockDim.x) {
        int row = off + lm;
        dst[gbase + int64_t(k_blk)*m + lm] = src[int64_t(row)*sk + k_blk];
    }
}
// SFB: [E, N/128, K/128] K-major -> per-group [N/128, K/128] MN-major slabs.
__global__ void repack_sfb(const EAcc* __restrict__ src, EAcc* __restrict__ dst,
                           const int* __restrict__ expert_ids, int sn, int sk) {
    int i = blockIdx.y;
    int k_blk = blockIdx.x;
    int eid = expert_ids[i];
    int64_t gbase = int64_t(i)*sn*sk;
    int64_t sbase = int64_t(eid)*sn*sk;
    for (int nb = threadIdx.x; nb < sn; nb += blockDim.x) {
        dst[gbase + int64_t(k_blk)*sn + nb] = src[sbase + int64_t(nb)*sk + k_blk];
    }
}
// Launches repack kernels onto `stream`. Source SFA/SFB are the model's K-major scales;
// dst_sfa/dst_sfb must be pre-allocated MN-major scratch.
static inline void launch_repack(const EAcc* src_sfa, const EAcc* src_sfb,
                                 EAcc* dst_sfa, EAcc* dst_sfb,
                                 int G, int N, int K,
                                 int* d_mi, int* d_eids,
                                 cudaStream_t stream) {
    int sk = K/128, sn = N/128;
    dim3 ga(G, sk, 1);
    repack_sfa<<<ga, 128, 0, stream>>>(src_sfa, dst_sfa, d_mi, sk);
    dim3 gb(sk, G, 1);
    int bx = sn < 128 ? sn : 128;
    repack_sfb<<<gb, bx, 0, stream>>>(src_sfb, dst_sfb, d_eids, sn, sk);
}
static Gm128 g_gemm128; static Gm64 g_gemm64;
static Gm128c g_gemm128c;  // Tl128 + Cluster<2,1,1> cooperative variant
static cutlass::KernelHardwareInfo g_hw;
static void* g_ws128=nullptr; static size_t g_ws128_sz=0;
static void* g_ws64=nullptr; static size_t g_ws64_sz=0;
static void* g_ws128c=nullptr; static size_t g_ws128c_sz=0;  // Tl128+Cluster cooperative workspace
static bool g_init=false; static int g_mx=0;
// Scale-repack scratch buffers (grown on demand based on observed sizes).
static EAcc* g_sfa_scratch=nullptr; static size_t g_sfa_sz=0;
static EAcc* g_sfb_scratch=nullptr; static size_t g_sfb_sz=0;
static EAcc* g_sfa_scratch2=nullptr; static size_t g_sfa_sz2=0;
static EAcc* g_sfb_scratch2=nullptr; static size_t g_sfb_sz2=0;
// Scratch sizes: SFA covers T up to ~300K at sk=56 (64 MB). SFB covers ng*sn*sk up
// to 2M floats (8 MB) — well above the 32*56*56 = 100K used here. Over-alloc cost is
// negligible on 96GB HBM and avoids any host-side size sync per call.
static constexpr size_t kSFAScratchFloats = 16ull * 1024 * 1024;
static constexpr size_t kSFBScratchFloats = 2ull  * 1024 * 1024;
static void ensure_scratch() {
    if (!g_sfa_scratch) { cudaMalloc(&g_sfa_scratch,  kSFAScratchFloats*sizeof(EAcc)); g_sfa_sz =kSFAScratchFloats; }
    if (!g_sfb_scratch) { cudaMalloc(&g_sfb_scratch,  kSFBScratchFloats*sizeof(EAcc)); g_sfb_sz =kSFBScratchFloats; }
    if (!g_sfa_scratch2){ cudaMalloc(&g_sfa_scratch2, kSFAScratchFloats*sizeof(EAcc)); g_sfa_sz2=kSFAScratchFloats; }
    if (!g_sfb_scratch2){ cudaMalloc(&g_sfb_scratch2, kSFBScratchFloats*sizeof(EAcc)); g_sfb_sz2=kSFBScratchFloats; }
}
static PS::UnderlyingProblemShape* d_ps=nullptr;
static EA const** d_pA=nullptr; static EB const** d_pB=nullptr; static EC** d_pD=nullptr;
static EAcc const** d_pSFA=nullptr; static EAcc const** d_pSFB=nullptr;
static StA* d_sA=nullptr; static StB* d_sB=nullptr; static StD* d_sD=nullptr;
static ILSFA* d_lA=nullptr; static ILSFB* d_lB=nullptr;
static PS::UnderlyingProblemShape* d_ps2=nullptr;
static EA const** d_pA2=nullptr; static EB const** d_pB2=nullptr; static EC** d_pD2=nullptr;
static EAcc const** d_pSFA2=nullptr; static EAcc const** d_pSFB2=nullptr;
static StA* d_sA2=nullptr; static StB* d_sB2=nullptr; static StD* d_sD2=nullptr;
static ILSFA* d_lA2=nullptr; static ILSFB* d_lB2=nullptr;
static void grow(int G){if(G<=g_mx)return;int n=std::max(G,64);
    if(d_ps){cudaFree(d_ps);cudaFree(d_pA);cudaFree(d_pB);cudaFree(d_pD);
             cudaFree(d_pSFA);cudaFree(d_pSFB);cudaFree(d_sA);cudaFree(d_sB);
             cudaFree(d_sD);cudaFree(d_lA);cudaFree(d_lB);
             cudaFree(d_ps2);cudaFree(d_pA2);cudaFree(d_pB2);cudaFree(d_pD2);
             cudaFree(d_pSFA2);cudaFree(d_pSFB2);cudaFree(d_sA2);cudaFree(d_sB2);
             cudaFree(d_sD2);cudaFree(d_lA2);cudaFree(d_lB2);}
    cudaMalloc(&d_ps,n*sizeof(*d_ps));cudaMalloc(&d_pA,n*sizeof(*d_pA));
    cudaMalloc(&d_pB,n*sizeof(*d_pB));cudaMalloc(&d_pD,n*sizeof(*d_pD));
    cudaMalloc(&d_pSFA,n*sizeof(*d_pSFA));cudaMalloc(&d_pSFB,n*sizeof(*d_pSFB));
    cudaMalloc(&d_sA,n*sizeof(*d_sA));cudaMalloc(&d_sB,n*sizeof(*d_sB));
    cudaMalloc(&d_sD,n*sizeof(*d_sD));
    cudaMalloc(&d_lA,n*sizeof(*d_lA));cudaMalloc(&d_lB,n*sizeof(*d_lB));
    cudaMalloc(&d_ps2,n*sizeof(*d_ps2));cudaMalloc(&d_pA2,n*sizeof(*d_pA2));
    cudaMalloc(&d_pB2,n*sizeof(*d_pB2));cudaMalloc(&d_pD2,n*sizeof(*d_pD2));
    cudaMalloc(&d_pSFA2,n*sizeof(*d_pSFA2));cudaMalloc(&d_pSFB2,n*sizeof(*d_pSFB2));
    cudaMalloc(&d_sA2,n*sizeof(*d_sA2));cudaMalloc(&d_sB2,n*sizeof(*d_sB2));
    cudaMalloc(&d_sD2,n*sizeof(*d_sD2));
    cudaMalloc(&d_lA2,n*sizeof(*d_lA2));cudaMalloc(&d_lB2,n*sizeof(*d_lB2));g_mx=n;}
struct GemmArgs {
    int num_groups; int N, K;
    void* A; void* B; void* D; void* SFA; void* SFB;
    int* m_indptr; int* expert_ids;
};
template <class Gm>
static int run_prepared_gemm(Gm& gemm, void*& ws, size_t& ws_sz, bool& validated,
    int G, PS::UnderlyingProblemShape* ps, const EA** pA, const EB** pB,
    const EAcc** pSFA, const EAcc** pSFB, EC** pD,
    StA* sA, StB* sB, StD* sD, ILSFA* lA, ILSFB* lB,
    cudaStream_t stream) {
    typename Gm::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
        {G,ps,nullptr},{pA,sA,pB,sB,pSFA,lA,pSFB,lB},
        {{},nullptr,nullptr,pD,sD},g_hw};
    args.epilogue.thread.alpha=1.f; args.epilogue.thread.beta=0.f;
    if (!validated) {
        auto st=gemm.can_implement(args); if(st!=cutlass::Status::kSuccess) return -1;
        size_t need=Gm::get_workspace_size(args);
        if(need>ws_sz){if(ws)cudaFree(ws);cudaMalloc(&ws,need);ws_sz=need;}
        validated = true;
    }
    auto st=gemm.initialize(args,ws,stream); if(st!=cutlass::Status::kSuccess) return -2;
    st=gemm.run(stream); return st==cutlass::Status::kSuccess?0:-3;
}
extern "C" int cutlass_blockwise_fp8_gemm(GemmArgs* a, cudaStream_t stream) {
    if(!g_init){g_hw.device_id=0;g_hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);g_init=true;}
    int G=a->num_groups; grow(G); ensure_scratch();
    // Repack model's K-major SFA/SFB scales into MN-major scratch (required by Sm90).
    launch_repack((const EAcc*)a->SFA,(const EAcc*)a->SFB,g_sfa_scratch,g_sfb_scratch,
                  G,a->N,a->K,a->m_indptr,a->expert_ids,stream);
    {cudaLaunchConfig_t c; c.gridDim=(G+255)/256; c.blockDim=std::min(G,256);
     c.dynamicSmemBytes=0; c.stream=stream;
     cudaLaunchAttribute at[1]; at[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
     at[0].val.programmaticStreamSerializationAllowed=true; c.numAttrs=1; c.attrs=at;
     cudaLaunchKernelEx(&c,prep,(EA*)a->A,(EB*)a->B,g_sfa_scratch,g_sfb_scratch,(EC*)a->D,
        a->m_indptr,a->expert_ids,a->N,a->K,G,
        d_ps,(const EA**)d_pA,(const EB**)d_pB,(const EAcc**)d_pSFA,(const EAcc**)d_pSFB,
        (EC**)d_pD,d_sA,d_sB,d_sD,d_lA,d_lB);}
    typename Gm64::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
        {G,d_ps,nullptr},{(const EA**)d_pA,d_sA,(const EB**)d_pB,d_sB,
         (const EAcc**)d_pSFA,d_lA,(const EAcc**)d_pSFB,d_lB},
        {{},nullptr,nullptr,(EC**)d_pD,d_sD},g_hw};
    args.epilogue.thread.alpha=1.f; args.epilogue.thread.beta=0.f;
    static bool s_validated64 = false;
    if (!s_validated64) {
        auto st=g_gemm64.can_implement(args); if(st!=cutlass::Status::kSuccess) return -1;
        size_t need=Gm64::get_workspace_size(args);
        if(need>g_ws64_sz){if(g_ws64)cudaFree(g_ws64);cudaMalloc(&g_ws64,need);g_ws64_sz=need;}
        s_validated64 = true;
    }
    auto st=g_gemm64.initialize(args,g_ws64,stream); if(st!=cutlass::Status::kSuccess) return -2;
    st=g_gemm64.run(stream); return st==cutlass::Status::kSuccess?0:-3;
}
extern "C" int cutlass_blockwise_fp8_gemm_128(GemmArgs* a, cudaStream_t stream) {
    if(!g_init){g_hw.device_id=0;g_hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);g_init=true;}
    int G=a->num_groups; grow(G); ensure_scratch();
    launch_repack((const EAcc*)a->SFA,(const EAcc*)a->SFB,g_sfa_scratch,g_sfb_scratch,
                  G,a->N,a->K,a->m_indptr,a->expert_ids,stream);
    {cudaLaunchConfig_t c; c.gridDim=(G+255)/256; c.blockDim=std::min(G,256);
     c.dynamicSmemBytes=0; c.stream=stream;
     cudaLaunchAttribute at[1]; at[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
     at[0].val.programmaticStreamSerializationAllowed=true; c.numAttrs=1; c.attrs=at;
     cudaLaunchKernelEx(&c,prep,(EA*)a->A,(EB*)a->B,g_sfa_scratch,g_sfb_scratch,(EC*)a->D,
        a->m_indptr,a->expert_ids,a->N,a->K,G,
        d_ps,(const EA**)d_pA,(const EB**)d_pB,(const EAcc**)d_pSFA,(const EAcc**)d_pSFB,
        (EC**)d_pD,d_sA,d_sB,d_sD,d_lA,d_lB);}
    typename Gm128::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
        {G,d_ps,nullptr},{(const EA**)d_pA,d_sA,(const EB**)d_pB,d_sB,
         (const EAcc**)d_pSFA,d_lA,(const EAcc**)d_pSFB,d_lB},
        {{},nullptr,nullptr,(EC**)d_pD,d_sD},g_hw};
    args.epilogue.thread.alpha=1.f; args.epilogue.thread.beta=0.f;
    static bool s_validated128 = false;
    if (!s_validated128) {
        auto st=g_gemm128.can_implement(args); if(st!=cutlass::Status::kSuccess) return -1;
        size_t need=Gm128::get_workspace_size(args);
        if(need>g_ws128_sz){if(g_ws128)cudaFree(g_ws128);cudaMalloc(&g_ws128,need);g_ws128_sz=need;}
        s_validated128 = true;
    }
    auto st=g_gemm128.initialize(args,g_ws128,stream); if(st!=cutlass::Status::kSuccess) return -2;
    st=g_gemm128.run(stream); return st==cutlass::Status::kSuccess?0:-3;
}
__global__ void prep_dual(
    EA* A1, EB* B1, EAcc* SFA1, EAcc* SFB1, EC* D1, int n1, int k1,
    EA* A2, EB* B2, EAcc* SFA2, EAcc* SFB2, EC* D2, int n2, int k2,
    int* mi, int* expert_ids, int ng,
    PS::UnderlyingProblemShape* ps1, const EA** pA1, const EB** pB1,
    const EAcc** pSFA1, const EAcc** pSFB1, EC** pD1,
    StA* sA1, StB* sB1, StD* sD1, ILSFA* lA1, ILSFB* lB1,
    PS::UnderlyingProblemShape* ps2, const EA** pA2, const EB** pB2,
    const EAcc** pSFA2, const EAcc** pSFB2, EC** pD2,
    StA* sA2, StB* sB2, StD* sD2, ILSFA* lA2, ILSFB* lB2) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=ng) return;
#if (__CUDACC_VER_MAJOR__>=12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__>=900))
    asm volatile("griddepcontrol.wait;");
    asm volatile("griddepcontrol.launch_dependents;");
#endif
    int off=mi[i], m=mi[i+1]-off;
    int eid = expert_ids[i];
    {int sk=k1/128, sn=n1/128;
    ps1[i]=PS::UnderlyingProblemShape(m,n1,k1);
    sA1[i]=cutlass::make_cute_packed_stride(StA{},{m,k1,1});
    sB1[i]=cutlass::make_cute_packed_stride(StB{},{n1,k1,1});
    sD1[i]=cutlass::make_cute_packed_stride(StD{},{m,n1,1});
    pA1[i]=A1+int64_t(off)*k1; pB1[i]=B1+int64_t(eid)*n1*k1;
    pD1[i]=D1+int64_t(off)*n1;
    lA1[i]=SC::tile_atom_to_shape_SFA(make_shape(m,n1,k1,1));
    pSFA1[i]=SFA1+int64_t(off)*sk;
    lB1[i]=SC::tile_atom_to_shape_SFB(make_shape(m,n1,k1,1));
    pSFB1[i]=SFB1+int64_t(i)*sn*sk;}
    {int sk=k2/128, sn=n2/128;
    ps2[i]=PS::UnderlyingProblemShape(m,n2,k2);
    sA2[i]=cutlass::make_cute_packed_stride(StA{},{m,k2,1});
    sB2[i]=cutlass::make_cute_packed_stride(StB{},{n2,k2,1});
    sD2[i]=cutlass::make_cute_packed_stride(StD{},{m,n2,1});
    pA2[i]=A2+int64_t(off)*k2; pB2[i]=B2+int64_t(eid)*n2*k2;
    pD2[i]=D2+int64_t(off)*n2;
    lA2[i]=SC::tile_atom_to_shape_SFA(make_shape(m,n2,k2,1));
    pSFA2[i]=SFA2+int64_t(off)*sk;
    lB2[i]=SC::tile_atom_to_shape_SFB(make_shape(m,n2,k2,1));
    pSFB2[i]=SFB2+int64_t(i)*sn*sk;}
}
struct GemmArgsDual {
    int num_groups;
    int N1, K1; void* A1; void* B1; void* D1; void* SFA1; void* SFB1;
    int N2, K2; void* A2; void* B2; void* D2; void* SFA2; void* SFB2;
    int* m_indptr; int* expert_ids;
};
extern "C" int cutlass_prep_dual(GemmArgsDual* a, cudaStream_t stream) {
    if(!g_init){g_hw.device_id=0;g_hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);g_init=true;}
    int G=a->num_groups; grow(G); ensure_scratch();
    // Repack scales for both GEMMs. GEMM1 uses scratch set #1, GEMM2 uses set #2.
    launch_repack((const EAcc*)a->SFA1,(const EAcc*)a->SFB1,g_sfa_scratch, g_sfb_scratch,
                  G,a->N1,a->K1,a->m_indptr,a->expert_ids,stream);
    launch_repack((const EAcc*)a->SFA2,(const EAcc*)a->SFB2,g_sfa_scratch2,g_sfb_scratch2,
                  G,a->N2,a->K2,a->m_indptr,a->expert_ids,stream);
    cudaLaunchConfig_t c; c.gridDim=(G+255)/256; c.blockDim=std::min(G,256);
    c.dynamicSmemBytes=0; c.stream=stream;
    cudaLaunchAttribute at[1]; at[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
    at[0].val.programmaticStreamSerializationAllowed=true; c.numAttrs=1; c.attrs=at;
    cudaLaunchKernelEx(&c,prep_dual,
        (EA*)a->A1,(EB*)a->B1,g_sfa_scratch, g_sfb_scratch, (EC*)a->D1,a->N1,a->K1,
        (EA*)a->A2,(EB*)a->B2,g_sfa_scratch2,g_sfb_scratch2,(EC*)a->D2,a->N2,a->K2,
        a->m_indptr,a->expert_ids,G,
        d_ps,(const EA**)d_pA,(const EB**)d_pB,(const EAcc**)d_pSFA,(const EAcc**)d_pSFB,
        (EC**)d_pD,d_sA,d_sB,d_sD,d_lA,d_lB,
        d_ps2,(const EA**)d_pA2,(const EB**)d_pB2,(const EAcc**)d_pSFA2,(const EAcc**)d_pSFB2,
        (EC**)d_pD2,d_sA2,d_sB2,d_sD2,d_lA2,d_lB2);
    return 0;
}
extern "C" int cutlass_blockwise_fp8_gemm_noprep(GemmArgs* a, cudaStream_t stream) {
    if(!g_init){g_hw.device_id=0;g_hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);g_init=true;}
    int G=a->num_groups; grow(G);
    typename Gm64::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
        {G,d_ps,nullptr},{(const EA**)d_pA,d_sA,(const EB**)d_pB,d_sB,
         (const EAcc**)d_pSFA,d_lA,(const EAcc**)d_pSFB,d_lB},
        {{},nullptr,nullptr,(EC**)d_pD,d_sD},g_hw};
    args.epilogue.thread.alpha=1.f; args.epilogue.thread.beta=0.f;
    static bool s_validated64np = false;
    if (!s_validated64np) {
        auto st=g_gemm64.can_implement(args); if(st!=cutlass::Status::kSuccess) return -1;
        size_t need=Gm64::get_workspace_size(args);
        if(need>g_ws64_sz){if(g_ws64)cudaFree(g_ws64);cudaMalloc(&g_ws64,need);g_ws64_sz=need;}
        s_validated64np = true;
    }
    auto st=g_gemm64.initialize(args,g_ws64,stream); if(st!=cutlass::Status::kSuccess) return -2;
    st=g_gemm64.run(stream); return st==cutlass::Status::kSuccess?0:-3;
}
extern "C" int cutlass_blockwise_fp8_gemm_128_noprep(GemmArgs* a, cudaStream_t stream) {
    if(!g_init){g_hw.device_id=0;g_hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);g_init=true;}
    int G=a->num_groups; grow(G);
    typename Gm128::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
        {G,d_ps,nullptr},{(const EA**)d_pA,d_sA,(const EB**)d_pB,d_sB,
         (const EAcc**)d_pSFA,d_lA,(const EAcc**)d_pSFB,d_lB},
        {{},nullptr,nullptr,(EC**)d_pD,d_sD},g_hw};
    args.epilogue.thread.alpha=1.f; args.epilogue.thread.beta=0.f;
    static bool s_validated128np = false;
    if (!s_validated128np) {
        auto st=g_gemm128.can_implement(args); if(st!=cutlass::Status::kSuccess) return -1;
        size_t need=Gm128::get_workspace_size(args);
        if(need>g_ws128_sz){if(g_ws128)cudaFree(g_ws128);cudaMalloc(&g_ws128,need);g_ws128_sz=need;}
        s_validated128np = true;
    }
    auto st=g_gemm128.initialize(args,g_ws128,stream); if(st!=cutlass::Status::kSuccess) return -2;
    st=g_gemm128.run(stream); return st==cutlass::Status::kSuccess?0:-3;
}
// Tl128 + Cluster<2,1,1> cooperative no-prep (array set 1). Selected when
// max_M_estimate > 2048. Uses the same pre-filled d_ps/d_pA/... produced by dual prep.
extern "C" int cutlass_blockwise_fp8_gemm_128c_noprep(GemmArgs* a, cudaStream_t stream) {
    if(!g_init){g_hw.device_id=0;g_hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);g_init=true;}
    int G=a->num_groups; grow(G);
    typename Gm128c::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
        {G,d_ps,nullptr},{(const EA**)d_pA,d_sA,(const EB**)d_pB,d_sB,
         (const EAcc**)d_pSFA,d_lA,(const EAcc**)d_pSFB,d_lB},
        {{},nullptr,nullptr,(EC**)d_pD,d_sD},g_hw};
    args.epilogue.thread.alpha=1.f; args.epilogue.thread.beta=0.f;
    static bool s_v128c = false;
    if (!s_v128c) {
        auto st=g_gemm128c.can_implement(args); if(st!=cutlass::Status::kSuccess) return -1;
        size_t need=Gm128c::get_workspace_size(args);
        if(need>g_ws128c_sz){if(g_ws128c)cudaFree(g_ws128c);cudaMalloc(&g_ws128c,need);g_ws128c_sz=need;}
        s_v128c = true;
    }
    auto st=g_gemm128c.initialize(args,g_ws128c,stream); if(st!=cutlass::Status::kSuccess) return -2;
    st=g_gemm128c.run(stream); return st==cutlass::Status::kSuccess?0:-3;
}
extern "C" int cutlass_blockwise_fp8_gemm_noprep2(GemmArgs* a, cudaStream_t stream) {
    if(!g_init){g_hw.device_id=0;g_hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);g_init=true;}
    int G=a->num_groups; grow(G);
    typename Gm64::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
        {G,d_ps2,nullptr},{(const EA**)d_pA2,d_sA2,(const EB**)d_pB2,d_sB2,
         (const EAcc**)d_pSFA2,d_lA2,(const EAcc**)d_pSFB2,d_lB2},
        {{},nullptr,nullptr,(EC**)d_pD2,d_sD2},g_hw};
    args.epilogue.thread.alpha=1.f; args.epilogue.thread.beta=0.f;
    static bool s_v64np2 = false;
    if (!s_v64np2) {
        auto st=g_gemm64.can_implement(args); if(st!=cutlass::Status::kSuccess) return -1;
        size_t need=Gm64::get_workspace_size(args);
        if(need>g_ws64_sz){if(g_ws64)cudaFree(g_ws64);cudaMalloc(&g_ws64,need);g_ws64_sz=need;}
        s_v64np2 = true;
    }
    auto st=g_gemm64.initialize(args,g_ws64,stream); if(st!=cutlass::Status::kSuccess) return -2;
    st=g_gemm64.run(stream); return st==cutlass::Status::kSuccess?0:-3;
}
extern "C" int cutlass_blockwise_fp8_gemm_128_noprep2(GemmArgs* a, cudaStream_t stream) {
    if(!g_init){g_hw.device_id=0;g_hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);g_init=true;}
    int G=a->num_groups; grow(G);
    typename Gm128::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
        {G,d_ps2,nullptr},{(const EA**)d_pA2,d_sA2,(const EB**)d_pB2,d_sB2,
         (const EAcc**)d_pSFA2,d_lA2,(const EAcc**)d_pSFB2,d_lB2},
        {{},nullptr,nullptr,(EC**)d_pD2,d_sD2},g_hw};
    args.epilogue.thread.alpha=1.f; args.epilogue.thread.beta=0.f;
    static bool s_v128np2 = false;
    if (!s_v128np2) {
        auto st=g_gemm128.can_implement(args); if(st!=cutlass::Status::kSuccess) return -1;
        size_t need=Gm128::get_workspace_size(args);
        if(need>g_ws128_sz){if(g_ws128)cudaFree(g_ws128);cudaMalloc(&g_ws128,need);g_ws128_sz=need;}
        s_v128np2 = true;
    }
    auto st=g_gemm128.initialize(args,g_ws128,stream); if(st!=cutlass::Status::kSuccess) return -2;
    st=g_gemm128.run(stream); return st==cutlass::Status::kSuccess?0:-3;
}
// Tl128 + Cluster<2,1,1> cooperative no-prep (array set 2).
extern "C" int cutlass_blockwise_fp8_gemm_128c_noprep2(GemmArgs* a, cudaStream_t stream) {
    if(!g_init){g_hw.device_id=0;g_hw.sm_count=cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);g_init=true;}
    int G=a->num_groups; grow(G);
    typename Gm128c::Arguments args{cutlass::gemm::GemmUniversalMode::kGrouped,
        {G,d_ps2,nullptr},{(const EA**)d_pA2,d_sA2,(const EB**)d_pB2,d_sB2,
         (const EAcc**)d_pSFA2,d_lA2,(const EAcc**)d_pSFB2,d_lB2},
        {{},nullptr,nullptr,(EC**)d_pD2,d_sD2},g_hw};
    args.epilogue.thread.alpha=1.f; args.epilogue.thread.beta=0.f;
    static bool s_v128c2 = false;
    if (!s_v128c2) {
        auto st=g_gemm128c.can_implement(args); if(st!=cutlass::Status::kSuccess) return -1;
        size_t need=Gm128c::get_workspace_size(args);
        if(need>g_ws128c_sz){if(g_ws128c)cudaFree(g_ws128c);cudaMalloc(&g_ws128c,need);g_ws128c_sz=need;}
        s_v128c2 = true;
    }
    auto st=g_gemm128c.initialize(args,g_ws128c,stream); if(st!=cutlass::Status::kSuccess) return -2;
    st=g_gemm128c.run(stream); return st==cutlass::Status::kSuccess?0:-3;
}
