#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

using namespace cute;

#include "named_barrier.h"
#include "utils.h"
#include "softmax.h"
#include "static_switch.h"
#include "flash_mla.h"
#include "fp8_transpose_v.h"

namespace flashmla {

template<typename PrecType, int DIM, int DIM2 = DIM, cute::GMMA::Major major = GMMA::Major::K>
constexpr auto getSmemLayoutK() {
    constexpr int headSizeBytes = sizeof(PrecType) * DIM;
    constexpr int headSizeBytes2 = sizeof(PrecType) * DIM2;

    if constexpr (major == GMMA::Major::K) {
        if constexpr (headSizeBytes % 128 == 0 && headSizeBytes2 % 128 == 0) {
            return GMMA::Layout_K_SW128_Atom<PrecType>{};
        } else if constexpr (headSizeBytes % 64 == 0 && headSizeBytes2 % 64 == 0) {
            return GMMA::Layout_K_SW64_Atom<PrecType>{};
        } else {
            return GMMA::Layout_K_SW32_Atom<PrecType>{};
        }
    } else {
        if constexpr (headSizeBytes % 128 == 0 && headSizeBytes2 % 128 == 0) {
            return GMMA::Layout_MN_SW128_Atom<PrecType>{};
        } else if constexpr (headSizeBytes % 64 == 0 && headSizeBytes2 % 64 == 0) {
            return GMMA::Layout_MN_SW64_Atom<PrecType>{};
        } else {
            return GMMA::Layout_MN_SW32_Atom<PrecType>{};
        }
    }
    
}

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type=cutlass::bfloat16_t, typename elem_type_o = cutlass::bfloat16_t, int kHeadDimV_ = 0>
struct Flash_fwd_kernel_traits_mla {
    using Element = elem_type;
    using ElementO = elem_type_o;
    using ElementAccum = float;
    using index_t = int64_t;
    
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t>;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;
    static constexpr int kNWarpsS = 4;
    static constexpr int kNThreadsS = kNWarpsS * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);
    static constexpr int kHeadDimV = kHeadDimV_ != 0 ? kHeadDimV_ : kHeadDim;
    static_assert(kHeadDimV % 32 == 0);
    static_assert(kHeadDimV <= kHeadDim);

    static constexpr int kBlockKSmem = Is_FP8 ? (kHeadDim % 128 == 0 ? 128 : 64) : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kBlockKSmemO = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzleO = kBlockKSmemO == 32 ? 2 : 3;

    static constexpr cute::GMMA::Major MmaMajorV = !Is_FP8 ? GMMA::Major::MN : GMMA::Major::K;

    using TiledMma = decltype(make_tiled_mma(
            cute::GMMA::ss_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>,
                    GMMA::Major::K, GMMA::Major::K>(),
            Layout<Shape<Int<kNWarpsS / 4>, _1, _1>>{}));

    static constexpr int AtomLayoutNO = kNThreads / kNThreadsS;
    using TiledMmaO = decltype(make_tiled_mma(
            cute::GMMA::rs_op_selector<Element, Element, ElementAccum, Shape<Int<kBlockM>, Int<kHeadDimV / AtomLayoutNO>, Int<kBlockN>>,
                    GMMA::Major::K, MmaMajorV>(),
            Layout<Shape<Int<kNWarpsS / 4>, Int<AtomLayoutNO>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim>(),
            Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutV = decltype(tile_to_shape(
            getSmemLayoutK<Element, kHeadDim, kHeadDimV>(),
            Shape<Int<kBlockN>, Int<kHeadDimV>>{}));
    using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDimV>, Int<kBlockN>>{}, GenRowMajor{})));

    using SmemLayoutP = std::conditional_t<
        Is_FP8,
        Layout<Shape<Shape<_4, _2>, Int<kNThreadsS>, _1, _2, Int<kBlockN / 32>>>,
        Layout<Shape<Shape<_2, _2>, Int<kNThreadsS>, _1, _2, Int<kBlockN / 16>>>
    >;
    using SmemLayoutRow = Layout<Shape<_2, Int<kNThreadsS>>, Stride<_1, _2>>;

    using SmemLayoutAtomO = decltype(composition(
            Swizzle<kSwizzleO, 3, 3>{},
            Layout<Shape<Int<8>, Int<kBlockKSmemO>>, Stride<Int<kBlockKSmemO>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
            SmemLayoutAtomO{},
            Shape<Int<kBlockM>, Int<kHeadDimV>>{}));
    using SmemCopyAtomO = Copy_Atom<SM90_U32x4_STSM_N, ElementO>;
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    static constexpr int kNThreadsLoad = kNThreads - kNThreadsS;
    static_assert(kNThreadsLoad % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");

    static constexpr int kGmemElemsPerLoadO = sizeof(cute::uint128_t) / sizeof(ElementO);
    static_assert(kHeadDim % kGmemElemsPerLoadO == 0, "kHeadDim must be a multiple of kGmemElemsPerLoadO");
    static constexpr int kGmemThreadsPerRowO = kBlockKSmemO / kGmemElemsPerLoadO;
    static_assert(kNThreadsLoad % kGmemThreadsPerRowO == 0, "kNThreads must be a multiple of kGmemThreadsPerRowO");

    using GmemLayoutAtom = Layout<
            Shape<Int<kNThreadsLoad / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
            Stride<Int<kGmemThreadsPerRow>, _1>>;


    using GmemTiledCopy = decltype(make_tiled_copy(
            Copy_Atom<Gmem_copy_struct, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 vals per read

    using GmemLayoutAtomO = Layout<
            Shape<Int<kNThreadsS / kGmemThreadsPerRowO>, Int<kGmemThreadsPerRowO>>,
            Stride<Int<kGmemThreadsPerRowO>, _1>>;
    using GmemTiledCopyO = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementO>{},
            GmemLayoutAtomO{},
            Layout<Shape<_1, Int<kGmemElemsPerLoadO>>>{}));  // Val layout, 8 vals per store

    static constexpr int kGmemElemsPerLoadAccum = sizeof(cute::uint128_t) / sizeof(ElementAccum);
    static constexpr int kGmemThreadsPerRowAccum = kBlockKSmemO / kGmemElemsPerLoadAccum;
    using GmemLayoutAtomOaccum = Layout<
            Shape<Int<kNThreadsS / kGmemThreadsPerRowAccum>, Int<kGmemThreadsPerRowAccum>>,
            Stride<Int<kGmemThreadsPerRowAccum>, _1>>;
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
            GmemLayoutAtomOaccum{},
            Layout<Shape<_1, Int<kGmemElemsPerLoadAccum>>>{}));  // Val layout, 4 vals per store


    // ------ for f8 ------
    using SmemFp8Tranpose = SmemTransposeFp8_64x64<kBlockN, kHeadDimV, SmemLayoutK>;
    using SmemLayoutVtMMa = typename SmemFp8Tranpose::SmemLayoutVt;
};

namespace flash {

using namespace cute;

template<typename Kernel_traits>
struct SharedStorageMLA {
    using SmemV_t = std::conditional_t<Kernel_traits::Is_FP8,
                                    cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutVtMMa>>,
                                    cute::array_aligned<typename Kernel_traits::Element, 0>>;
    union {
        struct {
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutQ>> smem_q;
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutK> * 2> smem_k;  // Double buffer
            SmemV_t smem_vt;
            cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutP>> smem_p;
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutRow>> smem_scale;
        };
        struct {
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutRow>> smem_max;
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutRow>> smem_sum;
            cute::array_aligned<typename Kernel_traits::ElementAccum, cute::cosize_v<typename Kernel_traits::SmemLayoutO>> smem_o;
        };
    };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Split, typename SharedStorage, typename AccO, typename Softmax>
__forceinline__ __device__ void store(const Flash_fwd_mla_params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx,
                                      SharedStorage &shared_storage, AccO tOrO, Softmax softmax, float descale_k, float scale_softmax) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    using Element = typename Kernel_traits::ElementO;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;

    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);

    // Epilogue

    const int split_offset = __ldg(params.num_splits_ptr + bidb);

    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(tOrO, scale_softmax, descale_k);

    using ElementO = std::conditional_t<!Split, Element, ElementAccum>;
    Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(shared_storage.smem_o.data())), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    using SmemTiledCopyO = std::conditional_t<
            !Split,
            typename Kernel_traits::SmemCopyAtomO,
            typename Kernel_traits::SmemCopyAtomOaccum
    >;
    auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma_o);
    auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor rO = flashmla::convert_type<ElementO>(tOrO);
    Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    __syncthreads();

    cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

    const index_t row_offset_o = bidb * params.o_batch_stride + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_oaccum = (((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_v;
    const index_t row_offset_lse = (bidb * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    const index_t row_offset_lseaccum = ((split_offset + n_split_idx) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                                 Shape<Int<kBlockM>, Int<kHeadDimV>>{},
                                 make_stride(Split ? kHeadDimV : params.o_row_stride, _1{}));
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + (Split ? row_offset_lseaccum : row_offset_lse)),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});

    using GmemTiledCopyO = std::conditional_t<!Split, typename Kernel_traits::GmemTiledCopyO, typename Kernel_traits::GmemTiledCopyOaccum>;
    GmemTiledCopyO gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

    __syncthreads();

    if (tidx >= kNThreadsS) { return; }

    Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
    cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDimV>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma_o.partition_C(caccO);                           // ((MMA=4, X), MMA_M, MMA_K=1)
    Tensor taccOcO_row = taccOcO(make_coord(0, _, 0), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
#pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < params.seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flashmla::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, params.seqlen_q - m_block * kBlockM
    );
}

template<typename Kernel_traits, bool Is_causal, typename SharedStorage>
__forceinline__ __device__ void compute_attn_1rowblock_splitkv_mla(const Flash_fwd_mla_params &params,
                                                                   const int bidb, const int bidh, const int m_block,
                                                                   const int n_split_idx, const int seqlen_k,
                                                                   const int n_block_min, const int n_block_max, const bool NoSplit,
                                                                   SharedStorage &shared_storage, const float descale_k, const float scale_softmax, const float scale_softmax_log2) {
    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kHeadDimV = Kernel_traits::kHeadDimV;
    constexpr int kNThreads = Kernel_traits::kNThreads;
    constexpr int kNThreadsS = Kernel_traits::kNThreadsS;
    static_assert(kNThreads == 256 and kNThreadsS == 128);
    using Element = typename Kernel_traits::Element;
    using index_t = typename Kernel_traits::index_t;

    const int tidx = threadIdx.x;
    int n_block = n_block_max - 1;

    Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), typename Kernel_traits::SmemLayoutQ{}); // [BlockM, HeadDim]
    Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutK{}); // [BlockN, HeadDim]

    auto sV = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutV{}); // [BlockN, HeadDimV]
    auto sVt = [&](){
        if constexpr(Kernel_traits::Is_FP8){
            return make_tensor(make_smem_ptr(shared_storage.smem_vt.data()), typename Kernel_traits::SmemLayoutVtMMa{});
        } else {
            return make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename Kernel_traits::SmemLayoutVtransposed{}); // [HeadDimV, BlockN]
        }
    }();

    Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), typename Kernel_traits::SmemLayoutP{});
    Tensor tPsP = sP(_, tidx % kNThreadsS, _, _, _);
    Tensor sScale_o = make_tensor(make_smem_ptr(shared_storage.smem_scale.data()), typename Kernel_traits::SmemLayoutRow{});
    Tensor tScale_osScale_o = sScale_o(_, tidx % kNThreadsS);
    Tensor sRow_max = make_tensor(make_smem_ptr(shared_storage.smem_max.data()), typename Kernel_traits::SmemLayoutRow{});
    Tensor tRow_maxsRow_max = sRow_max(_, tidx % kNThreadsS);
    Tensor sRow_sum = make_tensor(make_smem_ptr(shared_storage.smem_sum.data()), typename Kernel_traits::SmemLayoutRow{});
    Tensor tRow_sumsRow_sum = sRow_sum(_, tidx % kNThreadsS);

    typename Kernel_traits::TiledMmaO tiled_mma_o;
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    Tensor tOrVt = thr_mma_o.partition_fragment_B(sVt);                // (MMA, MMA_K,MMA_N)
    Tensor tOrO = partition_fragment_C(tiled_mma_o, Shape<Int<kBlockM>, Int<kHeadDimV>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)
    clear(tOrO);

    // 初始化softmax计算器（处理每行的2个元素）
    flashmla::Softmax<2 * size<1>(tOrO)> softmax;

    int warp_group_idx = cutlass::canonical_warp_group_idx();
    if (warp_group_idx == 0) {
        // 计算组：处理QK^T、softmax和PV计算
        typename Kernel_traits::TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_thread_slice(tidx);
        Tensor tSrQ = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
        Tensor tSrK = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)

        if (n_block % 2 == 1) {
            // Double buffer for sK
            // 调整指针到第二个缓冲区
            constexpr int sK_offset = size(sK);
            
            if constexpr (Kernel_traits::Is_FP8) {
                tSrK.data() = tSrK.data() + sK_offset / 16;
            } else {
                tSrK.data() = tSrK.data() + sK_offset / 8;
                tOrVt.data() = tOrVt.data() + sK_offset / 8;
            }
        }

        // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
        // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
        // We will have at least 1 "masking" iteration.
        // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
        // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
        constexpr int n_masking_steps = !Is_causal ? 1 : cute::ceil_div(kBlockM, kBlockN) + 1;
#pragma unroll 1
        for (int masking_step = n_masking_steps; n_block >= n_block_min; --masking_step, --n_block) {
            __syncthreads();

            Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // ((MMA=4, X), MMA_M, MMA_N=1)
            flashmla::gemm</*zero_init=*/true, /*wg_wait=*/0>(tiled_mma, tSrQ, tSrK, tSrS);

            const bool is_masking_step = masking_step > 0;
            const bool is_first_masking_step = masking_step == n_masking_steps;

            if (is_masking_step) {
                Tensor cS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
                Tensor tScS = thr_mma.partition_C(cS);
#pragma unroll
                for (int i = 0; i < size(tSrS); ++i) {
                    if constexpr (!Is_causal) {  // Just masking based on col
                        if (int(get<1>(tScS(i))) >= int(seqlen_k - n_block * kBlockN)) tSrS(i) = -INFINITY;
                    } else {
                        // Ensure seqlen_k - 1 - (n_block * kBlockN + col) >= (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
                        // col <= seqlen_k - 1 - n_block * kBlockN - (seqlen_q - 1 - (m_block * kBlockM + row)) / ngroups
                        int row = int(get<0>(tScS(i)));
                        int col_limit_right = seqlen_k - 1 - n_block * kBlockN - (params.seqlen_q - 1 - (m_block * kBlockM + row)) / params.ngroups;
                        if (int(get<1>(tScS(i))) > col_limit_right) tSrS(i) = -INFINITY;
                    }
                }
            }

            // We have key_padding_mask so we'll need to Check_inf
            Tensor scale_o = is_first_masking_step
                             ? softmax.template softmax</*Is_first=*/true,  /*Check_inf=*/Is_causal>(tSrS, scale_softmax_log2)
                             : is_masking_step ?
                               softmax.template softmax</*Is_first=*/false, /*Check_inf=*/Is_causal>(tSrS, scale_softmax_log2)
                                               : softmax.template softmax</*Is_first=*/false, /*Check_inf=*//*Is_local=*/false>(tSrS, scale_softmax_log2);

            if constexpr (Kernel_traits::Is_FP8) { flashmla::permute_Cregs_fp8(tSrS); }
            Tensor tOrP_acc = make_tensor(tSrS.data(), flashmla::convert_layout_acc_Aregs<typename Kernel_traits::TiledMmaO>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            convert_type_out(tOrP_acc, tOrP);
            
            cute::copy(tOrP, tPsP); // send Aregs of MMA1 instead of Cregs of MMA0
            cute::copy(scale_o, tScale_osScale_o);

            cutlass::arch::NamedBarrier::arrive(kNThreads, static_cast<int>(NamedBarriers::SReady));

            flashmla::rescale_o(tOrO, scale_o);

            if constexpr (Kernel_traits::Is_FP8) {
                cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<int>(NamedBarriers::TransVReady));
                __syncthreads();
            }
            flashmla::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_o, tOrP, tOrVt, tOrO);

            // Double buffer for sK
            const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
            if constexpr (Kernel_traits::Is_FP8) {
                tSrK.data() = tSrK.data() + sK_offset / 16;
            } else {
                tSrK.data() = tSrK.data() + sK_offset / 8;
                tOrVt.data() = tOrVt.data() + sK_offset / 8;
            }
        }

        cute::copy(softmax.row_max, tRow_maxsRow_max);
        cute::copy(softmax.row_sum, tRow_sumsRow_sum);
        cutlass::arch::NamedBarrier::arrive(kNThreads, static_cast<int>(NamedBarriers::SoftmaxReady));
    } else {
        // 加载组：负责将Q、K、V从全局内存加载到共享内存
        const int *block_table = params.block_table + bidb * params.block_table_batch_stride;
        int cur_block_table = __ldg(&block_table[n_block]);

        const index_t row_offset_q = bidb * params.q_batch_stride + m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
        Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                make_stride(params.q_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_Q;
        auto gmem_thr_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx - kNThreadsS);
        Tensor tQgQ = gmem_thr_copy_Q.partition_S(gQ); // (ACPY, ACPY_M, ACPY_K)
        Tensor tQsQ = gmem_thr_copy_Q.partition_D(sQ); // (ACPY, ACPY_M, ACPY_K)
        Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor tQcQ = gmem_thr_copy_Q.partition_S(cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

        // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
        flashmla::copy</*Is_even_MN=*/false, /*Is_even_K=*/true>(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, tQpQ,
                                                              params.seqlen_q - m_block * kBlockM);

        const index_t row_offset_k = (bidh / params.h_h_k_ratio) * params.k_head_stride;
        Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                make_stride(params.k_row_stride, _1{}));
        typename Kernel_traits::GmemTiledCopy gmem_tiled_copy_K;
        auto gmem_thr_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx - kNThreadsS);
        Tensor tKgK = gmem_thr_copy_K.partition_S(gK); // (ACPY, ACPY_N, ACPY_K)
        Tensor tKsK = gmem_thr_copy_K.partition_D(sK); // (ACPY, ACPY_N, ACPY_K)
        Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_K.partition_S(cK);  // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
        Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK)));

        if (n_block % 2 == 1) {
            // Double buffer for sK
            constexpr int sK_offset = size(sK);
            tKsK.data() = tKsK.data() + sK_offset;
            if constexpr (!Kernel_traits::Is_FP8) {
                tOrVt.data() = tOrVt.data() + sK_offset / 8;
            }
        }

        // We need to clear the sK smem tiles because K is V.
        const index_t offset_k = cur_block_table * params.k_batch_stride;
        tKgK.data() = tKgK.data() + offset_k;
        flashmla::copy</*Is_even_MN=*/false, /*Is_even_K=*/true, /*Clear_OOB_MN=*/true>(gmem_tiled_copy_K, tKgK, tKsK, tKcK, tKpK,
                                                                                        seqlen_k - n_block * kBlockN);
        tKgK.data() = tKgK.data() + -offset_k;
        cute::cp_async_fence();

        if (n_block - 1 >= n_block_min) {
            cur_block_table = __ldg(&block_table[n_block - 1]);
        }

#pragma unroll 1
        for (; n_block >= n_block_min; --n_block) {
            flashmla::cp_async_wait<0>();
            __syncthreads();

            if (n_block - 1 >= n_block_min) {
                // Double buffer for sK
                const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
                tKsK.data() = tKsK.data() + sK_offset;

                const index_t offset_k = cur_block_table * params.k_batch_stride;
                tKgK.data() = tKgK.data() + offset_k;
                flashmla::copy</*Is_even_MN=*/true, /*Is_even_K=*/true>(gmem_tiled_copy_K, tKgK, tKsK, tKcK, tKpK);
                tKgK.data() = tKgK.data() + -offset_k;
                cute::cp_async_fence();
            }

            if constexpr (Kernel_traits::Is_FP8) {
                auto TransV = [&]() {
                    using SmemFp8Tranpose = typename Kernel_traits::SmemFp8Tranpose;
                    SmemFp8Tranpose smem_transpose_V;
                    Tensor sV_divide = as_position_independent_swizzle_tensor(
                        make_tensor(make_smem_ptr(shared_storage.smem_k.data()), typename SmemFp8Tranpose::SmemLayoutTransposeV{}));
                    Tensor sVt_divide = as_position_independent_swizzle_tensor(
                        make_tensor(make_smem_ptr(shared_storage.smem_vt.data()), typename SmemFp8Tranpose::SmemLayoutTransposeVt{}));

                    if (n_block % 2 == 1) {
                        sV_divide.data() = sV_divide.data() + size(sK);   
                    }
                    
                    CUTLASS_PRAGMA_UNROLL
                    for (int j = 0; j < shape<2>(typename SmemFp8Tranpose::SmemLayoutTransposeV{}); ++j) {
                        CUTLASS_PRAGMA_UNROLL
                        for (int i = 0; i < shape<1>(typename SmemFp8Tranpose::SmemLayoutTransposeV{}); ++i) {
                            smem_transpose_V.transpose(flatten(sV_divide(_, i, j)), flatten(sVt_divide(_, i, j)));
                        }
                    }
                };

                TransV();
                cutlass::arch::NamedBarrier::arrive(kNThreads, static_cast<int>(NamedBarriers::TransVReady));
            }
            
            // 同步至 P = Softmax(QK^T) 已经计算完成 (warpgroup 0)
            cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<int>(NamedBarriers::SReady));

            if (n_block - 2 >= n_block_min) {
                cur_block_table = __ldg(&block_table[n_block - 2]);
            }

            typename Kernel_traits::TiledMma tiled_mma;
            auto tSrS_layout = flashmla::convert_layout_acc_Aregs<Kernel_traits::TiledMmaO>(partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{}).layout());
            Tensor tOrP = make_tensor<Element>(tSrS_layout);
            Tensor scale_o = make_tensor<float>(Shape<_2>{});
            cute::copy(tScale_osScale_o, scale_o); // 加载缩放因子
            cute::copy(tPsP, tOrP); // 加载注意力权重

            flashmla::rescale_o(tOrO, scale_o); // 调整累加器数值范围

            if constexpr (Kernel_traits::Is_FP8) __syncthreads();
            flashmla::gemm</*zero_init=*/false, /*wg_wait=*/0>(tiled_mma_o, tOrP, tOrVt, tOrO);

            if constexpr (!Kernel_traits::Is_FP8) {
                // Double buffer for sK
                const int sK_offset = n_block % 2 == 0 ? size(sK) : -size(sK);
                tOrVt.data() = tOrVt.data() + sK_offset / 8;
            }
        }

        cutlass::arch::NamedBarrier::sync(kNThreads, static_cast<int>(NamedBarriers::SoftmaxReady));
        cute::copy(tRow_maxsRow_max, softmax.row_max);
        cute::copy(tRow_sumsRow_sum, softmax.row_sum);
    }

    if (NoSplit)
        store<Kernel_traits, false>(params, bidb, bidh, m_block, n_split_idx, shared_storage, tOrO, softmax, descale_k, scale_softmax);
    else
        store<Kernel_traits, true>(params, bidb, bidh, m_block, n_split_idx, shared_storage, tOrO, softmax, descale_k, scale_softmax);
}

template<typename Kernel_traits, bool Is_causal, typename SharedStorage>
__global__ void __launch_bounds__(Kernel_traits::kNThreads, 1, 1)
flash_fwd_splitkv_mla_kernel(__grid_constant__ const Flash_fwd_mla_params params) {
    constexpr int kBlockN = Kernel_traits::kBlockN;
    const int m_block = blockIdx.x;
    const int bidh = blockIdx.y;
    const int partition_idx = blockIdx.z;

    extern __shared__ char shared_memory[];
    auto &shared_storage = *reinterpret_cast<SharedStorage *>(shared_memory);

    int *tile_scheduler_metadata_ptr = params.tile_scheduler_metadata_ptr + partition_idx * TileSchedulerMetaDataSize;
    int4 tile_scheduler_metadata = __ldg(reinterpret_cast<int4 *>(tile_scheduler_metadata_ptr));
    int begin_idx = tile_scheduler_metadata.x; // begin batch idx
    int begin_seqlen = tile_scheduler_metadata.y; // begin batch seqlen_k idx
    int end_idx = tile_scheduler_metadata.z; // end batch idx
    int end_seqlen = tile_scheduler_metadata.w; // end batch seqlen_k idx
    if (begin_idx >= params.b) return;
    int begin_n_split_idx = __ldg(tile_scheduler_metadata_ptr + 4); // 当前batch被拆分到不同分区后，当前sm分区所在的分区index

    float descale_k = 1.f;
    float scale_softmax = params.scale_softmax;
    float scale_softmax_log2 = params.scale_softmax_log2;
    if constexpr (Kernel_traits::Is_FP8) {
        float descale_q = __ldg(params.descale_q_ptr);
        descale_k = __ldg(params.descale_k_ptr);
        scale_softmax = scale_softmax * descale_q * descale_k;
        scale_softmax_log2 = scale_softmax_log2 * descale_q * descale_k;
    }

#pragma unroll 1
    for (int batch_id = begin_idx; batch_id <= end_idx; ++batch_id) {
        const int n_split_idx = batch_id == begin_idx ? begin_n_split_idx : 0;
        const int seqlen_k = __ldg(params.cu_seqlens_k + batch_id); // params.cu_seqlens_k == seqlens_k
        const int n_block_min = batch_id == begin_idx ? begin_seqlen / kBlockN : 0;
        const int n_block_max = batch_id == end_idx ? cute::ceil_div(end_seqlen, kBlockN) : cute::ceil_div(seqlen_k, kBlockN);
        const bool NoSplit = n_block_min == 0 && n_block_max == cute::ceil_div(seqlen_k, kBlockN); // 当前batch是否只需要一个sm分区就能完成所有计算
        if (batch_id > begin_idx) {
            __syncthreads();  // Barrier between two tiles.
        }
        flash::compute_attn_1rowblock_splitkv_mla<Kernel_traits, Is_causal>(params, batch_id, bidh, m_block, n_split_idx, seqlen_k, n_block_min, n_block_max, NoSplit, shared_storage, descale_k, scale_softmax, scale_softmax_log2);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Element, typename ElementAccum, typename index_t, int kHeadDimV, int kMaxSplits>
__global__ void __launch_bounds__(256, 1, 1)
flash_fwd_splitkv_mla_combine_kernel(__grid_constant__ const Flash_fwd_mla_params params) {
    constexpr int kNThreads = 128;

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;
    const int hs = params.h * params.seqlen_q;
    const int batch_idx = bidx / hs;
    const int hs_idx = bidx % hs;

    const int split_offset = __ldg(params.num_splits_ptr + batch_idx);
    const int actual_num_splits = __ldg(params.num_splits_ptr + batch_idx + 1) - split_offset;
    FLASH_DEVICE_ASSERT(actual_num_splits <= kMaxSplits);
    if (actual_num_splits == 1) return;

    __shared__ ElementAccum sLseScale[kMaxSplits];

    const index_t row_offset_lseaccum = split_offset * hs + hs_idx;
    const index_t row_offset_lse = bidx;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                   Shape<Int<kMaxSplits>>{}, make_stride(hs));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<_1>{}, Stride<_1>{});

    int warp_idx = cutlass::canonical_warp_idx_sync();
    if (warp_idx == 0) {
        constexpr int kNLsePerThread = cute::ceil_div(kMaxSplits, 32);

        float local_lse[kNLsePerThread];
        for (int i = 0; i < kNLsePerThread; ++i) {
            const int split = i * 32 + tidx;
            local_lse[i] = split < actual_num_splits ? gLSEaccum(split) : -INFINITY;
        }

        float max_lse = -INFINITY;
        for (int i = 0; i < kNLsePerThread; ++i) max_lse = max(max_lse, local_lse[i]);
        for (int offset = 16; offset >= 1; offset /= 2) max_lse = max(max_lse, __shfl_xor_sync(uint32_t(-1), max_lse, offset));
        max_lse = max_lse == -INFINITY ? 0.0f : max_lse;  // In case all local LSEs are -inf

        float sum_lse = 0;
        for (int i = 0; i < kNLsePerThread; ++i) sum_lse = sum_lse + expf(local_lse[i] - max_lse);
        for (int offset = 16; offset >= 1; offset /= 2) sum_lse = sum_lse + __shfl_xor_sync(uint32_t(-1), sum_lse, offset);

        float global_lse = (sum_lse == 0.f || sum_lse != sum_lse) ? INFINITY : logf(sum_lse) + max_lse;
        if (tidx == 0) gLSE(0) = global_lse;

        for (int i = 0; i < kNLsePerThread; ++i) {
            const int split = i * 32 + tidx;
            if (split < actual_num_splits) sLseScale[split] = expf(local_lse[i] - global_lse);
        }
    }
    __syncthreads();

    static_assert(kHeadDimV % kNThreads == 0);
    constexpr int Elements = kHeadDimV / kNThreads;
    const index_t row_offset_oaccum = (split_offset * hs + hs_idx) * kHeadDimV;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kHeadDimV>>{}, Stride<_1>{});
    using GmemTiledCopyOaccum = decltype(make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
            Layout<Shape<Int<kNThreads>>>{},
            Layout<Shape<Int<Elements>>>{}));
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    for (int split = 0; split < actual_num_splits; ++split) {
        cute::copy(tOgOaccum, tOrOaccum);
        ElementAccum lse_scale = sLseScale[split];
        for (int i = 0; i < size(tOrO); ++i) {
            tOrO(i) += lse_scale * tOrOaccum(i);
        }
        tOgOaccum.data() = tOgOaccum.data() + hs * kHeadDimV;
    }

    Tensor rO = flashmla::convert_type<Element>(tOrO);
    const int head_idx = (bidx - batch_idx * hs) / params.seqlen_q;
    const int row = bidx - batch_idx * hs - head_idx * params.seqlen_q;
    auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride + head_idx * params.o_head_stride + row * params.o_row_stride;
    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + tidx * Elements), Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
    cute::copy(rO, gO);
}

} // namespace flash

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, typename SharedStorage>
void run_flash_splitkv_fwd_mla(Flash_fwd_mla_params &params, cudaStream_t stream) {
    FLASH_ASSERT(params.page_block_size == Kernel_traits::kBlockN);
    const int num_m_block = cute::ceil_div(params.seqlen_q, Kernel_traits::kBlockM);
    BOOL_SWITCH(params.is_causal, Is_causal, [&] {
        auto kernel = &flash::flash_fwd_splitkv_mla_kernel<Kernel_traits, Is_causal, SharedStorage>;
        constexpr size_t smem_size = sizeof(SharedStorage);
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        // num_m_block = 128 / 64 = 2, params.h = 1, params.num_sm_parts = sm_count / 2
        kernel<<<dim3(num_m_block, params.h, params.num_sm_parts), Kernel_traits::kNThreads, smem_size, stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();

    dim3 grid_combine(params.b * params.h * params.seqlen_q);
    MLA_NUM_SPLITS_SWITCH(params.num_sm_parts, kMaxSplits, [&] {
        auto combine_kernel = &flash::flash_fwd_splitkv_mla_combine_kernel<
                typename Kernel_traits::ElementO, typename Kernel_traits::ElementAccum, typename Kernel_traits::index_t, Kernel_traits::kHeadDimV, kMaxSplits>;
        combine_kernel<<<grid_combine, 128, 0, stream>>>(params);
    });
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<typename T, typename To, int Headdim>
void run_mha_fwd_splitkv_mla(Flash_fwd_mla_params &params, cudaStream_t stream) {
    static_assert(Headdim == 576);
    FLASH_ASSERT(params.d_v == 512);
    FLASH_ASSERT(params.k_ptr == params.v_ptr);  // Shared_KV
    using Kernel_traits = Flash_fwd_kernel_traits_mla<576, 64, 64, 8, T, To, 512>;
    run_flash_splitkv_fwd_mla<Kernel_traits, flash::SharedStorageMLA<Kernel_traits>>(params, stream);
}

} // flashmla