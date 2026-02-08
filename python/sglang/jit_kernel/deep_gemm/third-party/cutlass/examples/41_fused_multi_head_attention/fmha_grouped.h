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
    \brief Grouped FMHA kernel
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/complex.h"
#include "cutlass/semaphore.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"

#include "fmha_grouped_problem_visitor.h"
#include "gemm_kernel_utils.h"
#include "gemm/mma_accum_lambda_iterator.h"
#include "epilogue/epilogue_rescale_output.h"


namespace {
  static CUTLASS_DEVICE float atomicMaxFloat(float* addr, float value) {
  // source: https://stackoverflow.com/a/51549250
  return (value >= 0)
      ? __int_as_float(atomicMax((int*)addr, __float_as_int(value)))
      : __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));
}
}

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename MM0_,                           ///! Structure for computing P = Q @ K
  typename MM1_,                           ///! Structure for computing O = P @ V
  typename scalar_t_,
  typename accum_t_,
  typename output_t_,
  typename output_accum_t_,
  bool kKeepOutputInRF,                    ///! Whether the intermediate output from MM0_ should be kept in the register file
  GroupScheduleMode GroupScheduleMode_     ///! Type of scheduling to perform
>
struct FMHAGrouped {
public:
  using MM0 = MM0_;
  using MM1 = MM1_;

  using scalar_t = scalar_t_;
  using accum_t = accum_t_;
  using output_t = output_t_;
  using output_accum_t = output_accum_t_;

  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;

  static constexpr bool kNeedsOutputAccumulatorBuffer = !kKeepOutputInRF &&
      !cutlass::platform::is_same<output_accum_t, output_t>::value;

  // Parameters to satisfy BaseGrouped
  using ElementA = scalar_t;
  using ElementB = scalar_t;
  using ElementC = accum_t;
  using LayoutA = typename MM0::LayoutA;
  using LayoutB = typename MM0::ElementB;
  using LayoutC = typename MM1::ElementC;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;
  static int const kAlignmentA = MM0::kAlignmentA;
  static int const kAlignmentB = MM0::kAlignmentB;
  static int const kAlignmentC = 1;
  using Mma = typename MM1::Mma;
  using EpilogueOutputOp = typename MM1::EpilogueOutputOp;
  using ThreadblockSwizzle = void;
  using Operator = typename MM1::Operator;
  using WarpShape = typename MM1::WarpShape;
  using InstructionShape = typename MM1::InstructionShape;

  using ElementQ = scalar_t;
  using ElementK = scalar_t;
  using ElementP = accum_t;
  using ElementV = scalar_t;
  using ElementO = output_t;
  using ElementOAccum = output_accum_t;
  using ElementAccumulator = accum_t;

  using LayoutQ = typename MM0::LayoutA;
  using LayoutK = typename MM0::LayoutB;
  using LayoutP = typename MM0::LayoutC;
  using LayoutV = typename MM1::LayoutB;
  using LayoutO = typename MM1::LayoutC;

  static bool const kPreloadV = (MM1::Mma::ArchTag::kMinComputeCapability >= 80 &&
                                 cutlass::sizeof_bits<ElementV>::value == 16);

  static int const kAlignmentQ = MM0::kAlignmentA;
  static int const kAlignmentK = MM0::kAlignmentB;
  static int const kAlignmentV = 1;

  using ThreadblockShape = typename MM0::ThreadblockShape;

  static int const kQueriesPerBlock = ThreadblockShape::kM;
  static int const kKeysPerBlock = ThreadblockShape::kN;

  static constexpr bool kSupportsDropout = false;
  static constexpr bool kSupportsBias = false;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename MM1::WarpCount;
  static int const kThreadsPerWarp = 32;
  static int const kThreadCount = kThreadsPerWarp * WarpCount::kCount;

  static constexpr int kNumWarpsPerBlock =
    kQueriesPerBlock * kKeysPerBlock / (kThreadsPerWarp * kThreadsPerWarp);

  using ProblemVisitor = FMHAGroupedProblemVisitor<
                            ThreadblockShape,
                            kGroupScheduleMode,
                            kThreadCount,
                            kThreadCount>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord *problem_sizes0{nullptr};
    GemmCoord *problem_sizes1{nullptr};

    int problem_count{0};
    int threadblock_count{0};

    ElementQ ** ptr_Q{nullptr};
    ElementK ** ptr_K{nullptr};
    ElementP ** ptr_P{nullptr};
    ElementV ** ptr_V{nullptr};
    ElementO ** ptr_O{nullptr};
    ElementOAccum ** ptr_O_accum{nullptr};

    typename LayoutQ::Stride::LongIndex *ldq{nullptr};
    typename LayoutK::Stride::LongIndex *ldk{nullptr};
    typename LayoutP::Stride::LongIndex *ldv{nullptr};
    typename LayoutO::Stride::LongIndex *ldo{nullptr};

    // Whether causal masking is to be performed
    bool causal{false};

    // Scale
    ElementAccumulator scale{0};

    // Only used by device-level operator
    GemmCoord *host_problem_sizes{nullptr};

    //
    // Methods
    //
  
      /// Default ctor
    Arguments() = default;

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord *problem_sizes0,
      GemmCoord *problem_sizes1,
      int problem_count,
      int threadblock_count,
      ElementQ ** ptr_Q,
      ElementK ** ptr_K,
      ElementP ** ptr_P,
      ElementV ** ptr_V,
      ElementO ** ptr_O,
      ElementOAccum ** ptr_O_accum,
      typename LayoutQ::Stride::LongIndex *ldq,
      typename LayoutK::Stride::LongIndex *ldk,
      typename LayoutP::Stride::LongIndex *ldp,
      typename LayoutV::Stride::LongIndex *ldv,
      typename LayoutO::Stride::LongIndex *ldo,
      bool causal,
      ElementAccumulator scale,
      GemmCoord *host_problem_sizes=nullptr
    ):
      problem_sizes0(problem_sizes0),
      problem_sizes1(problem_sizes1),
      problem_count(problem_count),
      threadblock_count(threadblock_count),
      ptr_Q(ptr_Q),
      ptr_K(ptr_K),
      ptr_P(ptr_P),
      ptr_V(ptr_V),
      ptr_O(ptr_O),
      ptr_O_accum(kNeedsOutputAccumulatorBuffer ? ptr_O_accum : (accum_t**)ptr_O),
      ldq(ldq),
      ldk(ldk),
      ldv(ldv),
      ldo(ldo),
      causal(causal),
      scale(scale),
      host_problem_sizes(host_problem_sizes)
    {

    }

    bool __host__ check_supported() {
      CHECK_ALIGNED_PTR(ptr_Q, kAlignmentQ);
      CHECK_ALIGNED_PTR(ptr_K, kAlignmentK);
      CHECK_ALIGNED_PTR(ptr_V, kAlignmentV);
      XFORMERS_CHECK(ldq % kAlignmentQ == 0, "query is not correctly aligned");
      XFORMERS_CHECK(ldk % kAlignmentK == 0, "key is not correctly aligned");
      XFORMERS_CHECK(ldv % kAlignmentV == 0, "value is not correctly aligned");
      return true;
    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {

    typename ProblemVisitor::Params problem_visitor;
    int threadblock_count;

    ElementQ ** ptr_Q;
    ElementK ** ptr_K;
    ElementP ** ptr_P;
    ElementV ** ptr_V;
    ElementO ** ptr_O;
    ElementOAccum ** ptr_O_accum;

    typename LayoutQ::Stride::LongIndex *ldq;
    typename LayoutK::Stride::LongIndex *ldk;
    typename LayoutP::Stride::LongIndex *ldv;
    typename LayoutO::Stride::LongIndex *ldo;

    ElementAccumulator scale;
    bool causal;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params():
      ptr_Q(nullptr),
      ptr_K(nullptr),
      ptr_P(nullptr),
      ptr_V(nullptr),
      ptr_O(nullptr),
      ptr_O_accum(nullptr),
      ldq(nullptr),
      ldk(nullptr),
      ldv(nullptr),
      ldo(nullptr),
      causal(false),
      scale(0)
    { }

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args,
          void *workspace = nullptr,
          int tile_count = 0):
      problem_visitor(args.problem_sizes0, args.problem_sizes1, args.problem_count, workspace, tile_count),
      threadblock_count(args.threadblock_count),
      ptr_Q(args.ptr_Q),
      ptr_K(args.ptr_K),
      ptr_P(args.ptr_P),
      ptr_V(args.ptr_V),
      ptr_O(args.ptr_O),
      ptr_O_accum(kNeedsOutputAccumulatorBuffer ? args.ptr_O_accum : (accum_t**)args.ptr_O),
      ldq(args.ldq),
      ldk(args.ldk),
      ldv(args.ldv),
      ldo(args.ldo),
      causal(args.causal),
      scale(args.scale)
    { 

    }

    CUTLASS_HOST_DEVICE
    void update(
      Arguments const &args,
      void *workspace = nullptr,
      int tile_count = 0) {

      problem_visitor = typename ProblemVisitor::Params(args.problem_sizes0,
                                                        args.problem_sizes1,
                                                        args.problem_count,
                                                        workspace, tile_count);
      threadblock_count = args.threadblock_count;
      ptr_Q = args.ptr_Q;
      ptr_K = args.ptr_K;
      ptr_P = args.ptr_P;
      ptr_V = args.ptr_V;
      ptr_O = args.ptr_O;
      ptr_O_accum = kNeedsOutputAccumulatorBuffer ? args.ptr_O_accum : (accum_t**)args.ptr_O;
      ldq = args.ldq;
      ldk = args.ldk;
      ldv = args.ldv;
      ldo = args.ldo;
      causal = args.causal;
      scale = args.scale;
    }
  };

  // Shared storage - depends on kernel params
  struct ScalingCoefs {
    cutlass::Array<ElementAccumulator, kQueriesPerBlock> m_prime;
    cutlass::Array<ElementAccumulator, kQueriesPerBlock> s_prime;
    cutlass::Array<ElementAccumulator, kQueriesPerBlock> mi;
    cutlass::Array<ElementAccumulator, kQueriesPerBlock> out_rescale;
    cutlass::Array<ElementAccumulator, kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>
        addition_storage;
  };

  struct SharedStorageEpilogueAtEnd : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      // Everything here might be overwritten during MM0
      typename MM0::AccumulatorSharedStorage si;
      typename MM1::Mma::SharedStorage mm1;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return epilogue;
    }

    // ProblemVisitor shared storage can't be overlapped with others
    typename ProblemVisitor::SharedStorage problem_visitor;
  };

  struct SharedStorageEpilogueInLoop : ScalingCoefs {
    struct SharedStorageAfterMM0 {
      // Everything here might be overwritten during MM0
      typename MM0::AccumulatorSharedStorage si;
      typename MM1::Mma::SharedStorage mm1;
      typename MM1::DefaultEpilogue::SharedStorage epilogue;
    };

    union {
      typename MM0::Mma::SharedStorage mm0;
      SharedStorageAfterMM0 after_mm0;
    };

    CUTLASS_DEVICE typename MM1::DefaultEpilogue::SharedStorage&
    epilogue_shared_storage() {
      return after_mm0.epilogue;
    }

    // ProblemVisitor shared storage can't be overlapped with others
    typename ProblemVisitor::SharedStorage problem_visitor;
  };

  using SharedStorage = typename cutlass::platform::conditional<
      kKeepOutputInRF,
      SharedStorageEpilogueAtEnd,
      SharedStorageEpilogueInLoop>::type;

private:

  // Parameters to be used by an individual tile
  struct TileParams {

    CUTLASS_HOST_DEVICE
    static int query_start(int threadblock_idx) {
      return threadblock_idx * kQueriesPerBlock;
    }

    // Returns whether this threadblock computes within the number of queries,
    // which is determined by the M dimension of problem 0
    CUTLASS_HOST_DEVICE
    static bool can_compute(int threadblock_idx, const GemmCoord& problem_size0) {
      return query_start(threadblock_idx) < problem_size0.m();
    }

    CUTLASS_HOST_DEVICE
    static int num_queries(int threadblock_idx, const GemmCoord& problem_size0) {
      return problem_size0.m() - query_start(threadblock_idx);
    }

    CUTLASS_HOST_DEVICE
    static int num_keys(int threadblock_idx, const GemmCoord& problem_size0, bool causal) {
      int nk = problem_size0.n();
      if (causal) {
        nk = cutlass::fast_min(int32_t(query_start(threadblock_idx) + kQueriesPerBlock), nk);
      }
      return nk;
    }

  };

public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  FMHAGrouped() { }

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const & problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const &args) {
    return Status::kSuccess;
  }

  static CUTLASS_DEVICE int16_t thread_id() {
    return threadIdx.x;
  }

  static CUTLASS_DEVICE int8_t warp_id() {
    return threadIdx.x / kThreadsPerWarp;
  }

  static CUTLASS_DEVICE int8_t lane_id() {
    return threadIdx.x % kThreadsPerWarp;
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {
    auto& m_prime = shared_storage.m_prime;
    auto& s_prime = shared_storage.s_prime;
    [[maybe_unused]] auto& si = shared_storage.after_mm0.si;
    auto& mi = shared_storage.mi;
    auto& out_rescale = shared_storage.out_rescale;

    ProblemVisitor problem_visitor(
      params.problem_visitor,
      shared_storage.problem_visitor,
      blockIdx.x);

    // Outer 'persistent' loop to iterate over tiles
    while (problem_visitor.next_tile()) {

      GemmCoord problem_size0 = problem_visitor.problem_size0();
      GemmCoord problem_size1 = problem_visitor.problem_size1();
      const int32_t threadblock_idx = int32_t(problem_visitor.threadblock_idx());

      if (!TileParams::can_compute(threadblock_idx, problem_size0)) {
        problem_visitor.advance(gridDim.x);
        continue;
      }

      const int32_t problem_idx = problem_visitor.problem_index();

      if (thread_id() < kQueriesPerBlock) {
        s_prime[thread_id()] = ElementAccumulator(0);
        out_rescale[thread_id()] = accum_t(1.0);
        m_prime[thread_id()] =
            -cutlass::platform::numeric_limits<ElementAccumulator>::infinity();
        mi[thread_id()] = -cutlass::platform::numeric_limits<ElementAccumulator>::infinity();
      }

      ElementO *ptr_O = params.ptr_O[problem_idx]  + TileParams::query_start(threadblock_idx) * params.ldo[problem_idx];
      ElementOAccum *ptr_O_accum = params.ptr_O_accum[problem_idx]  + TileParams::query_start(threadblock_idx) * params.ldo[problem_idx];
      const int num_queries = TileParams::num_queries(threadblock_idx, problem_size0);

      auto createOutputIter = [&](int col) -> typename MM1::OutputTileIterator {
        using OutputTileIterator = typename MM1::OutputTileIterator;
        return OutputTileIterator(
            typename OutputTileIterator::Params{(int32_t)params.ldo[problem_idx]},
            ptr_O,
            typename OutputTileIterator::TensorCoord{
                num_queries, problem_size1.n()},
            thread_id(),
            {0, col});
      };

      auto createOutputAccumIter = [&](int col) ->
        typename MM1::OutputTileIteratorAccum {
          using OutputTileIteratorAccum = typename MM1::OutputTileIteratorAccum;
          return OutputTileIteratorAccum(
              typename OutputTileIteratorAccum::Params{(int32_t)params.ldo[problem_idx]},
              ptr_O_accum,
              typename OutputTileIteratorAccum::TensorCoord{
                  num_queries, problem_size1.n()},
              thread_id(),
              {0, col});
        };

      typename MM1::Mma::FragmentC accum_o;
      accum_o.clear();

      const int num_keys = TileParams::num_keys(threadblock_idx, problem_size0, params.causal);

      for (int32_t iter_key_start = 0; iter_key_start < num_keys;
           iter_key_start += kKeysPerBlock) {
        int32_t problem_size_0_m =
            cutlass::fast_min((int32_t)kQueriesPerBlock, num_queries);
        int32_t problem_size_0_n = cutlass::fast_min(
            (int32_t)kKeysPerBlock, num_keys - iter_key_start);
        int32_t const& problem_size_0_k = problem_size0.k();
        int32_t const& problem_size_1_n = problem_size1.n();
        int32_t const& problem_size_1_k = problem_size_0_n;

        auto prologueV = [&](int blockN) {
          typename MM1::Mma::IteratorB iterator_V(
              typename MM1::IteratorB::Params{typename MM1::LayoutB(params.ldv[problem_idx])},
              params.ptr_V[problem_idx] + iter_key_start * params.ldv[problem_idx],
              {problem_size_1_k, problem_size_1_n},
              thread_id(),
              cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});

          MM1::Mma::prologue(
              shared_storage.after_mm0.mm1,
              iterator_V,
              thread_id(),
              problem_size_1_k);
        };

        __syncthreads(); // Need to have shared memory initialized, and `m_prime`
                         // updated from end of prev iter

        //
        // MATMUL: Q.K_t
        //
        // Computes the block-matrix product of:
        // (a) query[query_start:query_end, :]
        // with
        // (b) key[iter_key_start:iter_key_start + kKeysPerBlock]
        // and stores that into `shared_storage.si`
        //

        ElementQ *ptr_Q = params.ptr_Q[problem_idx] + TileParams::query_start(threadblock_idx) * params.ldq[problem_idx];

        // Construct iterators to A and B operands
        typename MM0::IteratorA iterator_A(
          typename MM0::IteratorA::Params(
              typename MM0::MmaCore::LayoutA(params.ldq[problem_idx])),
          ptr_Q,
          {problem_size_0_m, problem_size_0_k},
          thread_id(),
          {0, 0});

        typename MM0::IteratorB iterator_B(
            typename MM0::IteratorB::Params(
                typename MM0::MmaCore::LayoutB(params.ldk[problem_idx])),
            params.ptr_K[problem_idx] + iter_key_start * params.ldk[problem_idx],
            {problem_size_0_k, problem_size_0_n},
            thread_id(),
            {0, 0});

        // Construct thread-scoped matrix multiply
        typename MM0::Mma mma(
            shared_storage.mm0, thread_id(), warp_id(), lane_id());

        typename MM0::Mma::FragmentC accum;

        accum.clear();

        auto gemm_k_iterations =
            (problem_size_0_k + MM0::Mma::Shape::kK - 1) / MM0::Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add
        mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
        __syncthreads();

        if (kPreloadV) {
          prologueV(0);
        } else {
          MM1::Mma::drain_cp_asyncs();
        }

        typename MM0::Mma::Operator::IteratorC::TensorCoord
          iteratorC_tile_offset = {
              (warp_id() % MM0::Mma::WarpCount::kM),
              (warp_id() / MM0::Mma::WarpCount::kM)
            };

        // Mask out last if causal
        if (params.causal && num_keys - iter_key_start <= kKeysPerBlock) {
          auto lane_offset = MM0::AccumLambdaIterator::get_lane_offset(
              lane_id(), warp_id(), iteratorC_tile_offset);
          int32_t last_col;
          MM0::AccumLambdaIterator::iterateRows(
              lane_offset,
              [&](int accum_m) {
                last_col = TileParams::query_start(threadblock_idx) + accum_m - iter_key_start;
              },
              [&](int accum_m, int accum_n, int idx) {
                if (accum_n > last_col) {
                  accum[idx] =
                      -cutlass::platform::numeric_limits<accum_t>::infinity();
                }
              },
              [&](int accum_m) {});
        }
        // DISPATCH_BOOL(iter_key_start == 0, kIsFirst, ([&] {
        //         DISPATCH_BOOL(
        //             num_keys - iter_key_start >= kKeysPerBlock,
        //             kFullColumns,
        //             ([&] {
        //               // Update `mi` from accum stored in registers
        //               // Also does accum[i] <- exp(accum[i] - mi)
        //               iterative_softmax<
        //                   typename MM0::Mma::Operator::IteratorC,
        //                   kFullColumns,
        //                   kIsFirst>(
        //                   accum_o,
        //                   accum,
        //                   mi,
        //                   m_prime,
        //                   s_prime,
        //                   lane_id(),
        //                   thread_id(),
        //                   warp_id(),
        //                   num_keys - iter_key_start,
        //                   iteratorC_tile_offset,
        //                   kSupportsBias ? 1.0f : params.scale);
        //             }));
        //       }));

        // Update `mi` from accum stored in registers
        // Also does accum[i] <- exp(accum[i] - mi)
        iterative_softmax<typename MM0::Mma::Operator::IteratorC>(
            accum_o,
            accum,
            mi,
            m_prime,
            s_prime,
            out_rescale,
            shared_storage.addition_storage,
            lane_id(),
            thread_id(),
            warp_id(),
            num_keys - iter_key_start,
            iter_key_start == 0,
            iteratorC_tile_offset,
            kSupportsBias ? 1.0f : params.scale);

        // Output results to shared-memory
        int warp_idx_mn_0 = warp_id() %
            (MM0::Mma::Base::WarpCount::kM * MM0::Mma::Base::WarpCount::kN);
        auto output_tile_coords = cutlass::MatrixCoord{
            warp_idx_mn_0 % MM0::Mma::Base::WarpCount::kM,
            warp_idx_mn_0 / MM0::Mma::Base::WarpCount::kM};

        MM0::B2bGemm::accumToSmem(
            shared_storage.after_mm0.si, accum, lane_id(), output_tile_coords);

        __syncthreads();

        //
        // MATMUL: Attn . V
        // Run the matmul `attn @ V` for a block of attn and V.
        // `attn` is read from shared memory (in `shared_storage_si`)
        // `V` is read from global memory (with iterator_B)
        //

        const int64_t nBlockN = kKeepOutputInRF ? 1
                                                : ceil_div(
                                                      (int64_t)problem_size_1_n,
                                                      int64_t(MM1::ThreadblockShape::kN));

        // Iterate over the N dimension of GEMM1
        for (int blockN = 0; blockN < nBlockN; ++blockN) {
          int gemm_k_iterations =
              (problem_size_1_k + MM1::Mma::Shape::kK - 1) / MM1::Mma::Shape::kK;

          // Compute threadblock-scoped matrix multiply-add and store it in accum
          // (in registers)
          if (!kPreloadV) {
            __syncthreads(); // we share shmem between mma and epilogue
          }

          typename MM1::Mma::IteratorB iterator_V(
            typename MM1::IteratorB::Params{typename MM1::LayoutB(params.ldv[problem_idx])},
            params.ptr_V[problem_idx] + iter_key_start * params.ldv[problem_idx],
            {problem_size_1_k, problem_size_1_n},
            thread_id(),
            cutlass::MatrixCoord{0, blockN * MM1::Mma::Shape::kN});

          typename MM1::Mma mma_pv(
            // operand A: Pij_dropped in shared memory
            shared_storage.after_mm0.si.accum_ref(),
            // operand B: shared memory staging area for Vj, which is loaded
            // from global memory
            shared_storage.after_mm0.mm1.operand_B_ref(),
            (int)thread_id(),
            (int)warp_id(),
            (int)lane_id());

          mma_pv.set_prologue_done(kPreloadV);
          if (!kKeepOutputInRF) {
            accum_o.clear();
          }

          mma_pv(gemm_k_iterations, accum_o, iterator_V, accum_o);
          __syncthreads();

          if (kPreloadV && !kKeepOutputInRF && blockN + 1 < nBlockN) {
            prologueV(blockN + 1);
          }

          if (!kKeepOutputInRF) {
            MM1::Mma::drain_cp_asyncs();
            DISPATCH_BOOL(
                iter_key_start == 0, kIsFirst, ([&] {
                  DISPATCH_BOOL(
                      (iter_key_start + kKeysPerBlock) >= num_keys,
                      kIsLast,
                      ([&] {
                        using DefaultEpilogue = typename MM1::DefaultEpilogue;
                        using DefaultOp = typename MM1::DefaultConfig::EpilogueOutputOp;
                        using ElementCompute = typename DefaultOp::ElementCompute;
                        using EpilogueOutputOp = typename cutlass::epilogue::
                            thread::MemoryEfficientAttentionNormalize<
                                typename cutlass::platform::conditional<
                                    kIsLast::value,
                                    output_t,
                                    output_accum_t>::type,
                                output_accum_t,
                                DefaultOp::kCount,
                                typename DefaultOp::ElementAccumulator,
                                output_accum_t,
                                kIsFirst::value,
                                kIsLast::value,
                                cutlass::Array<ElementCompute, kQueriesPerBlock>>;
                        using Epilogue = typename cutlass::epilogue::threadblock::
                            EpiloguePipelined<
                                typename DefaultEpilogue::Shape,
                                typename MM1::Mma::Operator,
                                DefaultEpilogue::kPartitionsK,
                                typename cutlass::platform::conditional<
                                    kIsLast::value,
                                    typename MM1::OutputTileIterator,
                                    typename MM1::OutputTileIteratorAccum>::type,
                                typename DefaultEpilogue::
                                    AccumulatorFragmentIterator,
                                typename DefaultEpilogue::WarpTileIterator,
                                typename DefaultEpilogue::SharedLoadIterator,
                                EpilogueOutputOp,
                                typename DefaultEpilogue::Padding,
                                DefaultEpilogue::kFragmentsPerIteration,
                                true, // IterationsUnroll
                                typename MM1::OutputTileIteratorAccum // Read
                                                                      // iterator
                                >;

                        int col = blockN * MM1::Mma::Shape::kN;
                        auto source_iter = createOutputAccumIter(col);
                        auto dest_iter = gemm_kernel_utils::call_conditional<
                            kIsLast::value,
                            decltype(createOutputIter),
                            decltype(createOutputAccumIter)>::
                            apply(createOutputIter, createOutputAccumIter, col);
                        EpilogueOutputOp rescale(s_prime, out_rescale);
                        Epilogue epilogue(
                            shared_storage.epilogue_shared_storage(),
                            thread_id(),
                            warp_id(),
                            lane_id());
                        epilogue(rescale, dest_iter, accum_o, source_iter);
                      }));
                }));
            if (!kKeepOutputInRF) {
              __syncthreads();
            }
          }
        }
         __syncthreads(); // we modify `m_prime` after
      }

      if (kKeepOutputInRF) {
        constexpr bool kIsFirst = true;
        constexpr bool kIsLast = true;
        using DefaultEpilogue = typename MM1::DefaultEpilogue;
        using DefaultOp = typename MM1::DefaultConfig::EpilogueOutputOp;
        using ElementCompute = typename DefaultOp::ElementCompute;
        using EpilogueOutputOp =
            typename cutlass::epilogue::thread::MemoryEfficientAttentionNormalize<
                output_t,       // output
                output_accum_t, // source
                DefaultOp::kCount,
                typename DefaultOp::ElementAccumulator, // accum
                output_accum_t, // compute
                kIsFirst,
                kIsLast,
                cutlass::Array<ElementCompute, kQueriesPerBlock>>;
        using Epilogue =
            typename cutlass::epilogue::threadblock::EpiloguePipelined<
                typename DefaultEpilogue::Shape,
                typename MM1::Mma::Operator,
                DefaultEpilogue::kPartitionsK,
                typename MM1::OutputTileIterator, // destination
                typename DefaultEpilogue::AccumulatorFragmentIterator,
                typename DefaultEpilogue::WarpTileIterator,
                typename DefaultEpilogue::SharedLoadIterator,
                EpilogueOutputOp,
                typename DefaultEpilogue::Padding,
                DefaultEpilogue::kFragmentsPerIteration,
                true, // IterationsUnroll
                typename MM1::OutputTileIteratorAccum // source tile
                >;
        auto dest_iter = createOutputIter(0);
        EpilogueOutputOp rescale(s_prime, out_rescale);
        Epilogue epilogue(
            shared_storage.epilogue_shared_storage(),
            thread_id(),
            warp_id(),
            lane_id());
        MM1::Mma::drain_cp_asyncs();
        epilogue(rescale, dest_iter, accum_o);
      }

      // Next tile
      problem_visitor.advance(gridDim.x);
      __syncthreads(); // Don't start the next iteration until all threads are done using shared memory.
    }
  }

  template <typename WarpIteratorC>
  CUTLASS_DEVICE static void iterative_softmax(
      typename WarpIteratorC::Fragment& frag_o, // output so far
      typename WarpIteratorC::Fragment& frag,
      cutlass::Array<accum_t, kQueriesPerBlock>& mi,
      cutlass::Array<accum_t, kQueriesPerBlock>& m_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& s_prime,
      cutlass::Array<accum_t, kQueriesPerBlock>& out_rescale,
      cutlass::Array<accum_t, kQueriesPerBlock * MM0::MmaCore::WarpCount::kN>&
          addition_storage,
      int8_t lane_id,
      int8_t thread_id,
      int8_t warp_id,
      int max_col,
      bool is_first,
      typename WarpIteratorC::TensorCoord const& tile_offset,
      float scaling) {
    /* Iterates on the accumulator and corresponding position on result matrix

    (1) Update `mi[r]` to the max value of the row `r`
    (2) In a second iteration do the following:
        (a) accum   <- exp(accum - mi)
        (b) m_prime <- exp(m_prime - mi)
        (c) s_prime <- s_prime * m_prime + sum(accum)

    All of this is done on registers, before we store all of this
    on shared memory for the next matmul with Value.
    */
    using Fragment = typename WarpIteratorC::Fragment;
    using LambdaIterator = typename DefaultMmaAccumLambdaIterator<
        WarpIteratorC,
        accum_t,
        kThreadsPerWarp>::Iterator;
    // Convert to `accum_t` (rather than double)
    constexpr float kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E

    static_assert(kQueriesPerBlock % kNumWarpsPerBlock == 0, "");
    static constexpr int kLinesPerWarp = kQueriesPerBlock / kNumWarpsPerBlock;

    frag = cutlass::multiplies<Fragment>()(scaling * kLog2e, frag);

    auto lane_offset =
        LambdaIterator::get_lane_offset(lane_id, warp_id, tile_offset);

    // First update `mi` to the max per-row
    {
      accum_t max;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) {
            max = -cutlass::platform::numeric_limits<accum_t>::infinity();
          },
          [&](int accum_m, int accum_n, int idx) {
            if (accum_n < max_col) {
              max = cutlass::fast_max(max, frag[idx]);
            }
          },
          [&](int accum_m) {
            // Having 4x atomicMax seems faster than reduce within warp
            // first...
            atomicMaxFloat(&mi[accum_m], max);
          });
    }

    // Make sure we all share the update values for `mi`
    __syncthreads();

    // Doing this `exp` is quite expensive. Let's
    // split it across the warps
    bool restore_mi_to_minus_inf = false;
    if (lane_id < kLinesPerWarp) {
      int id = warp_id * kLinesPerWarp + lane_id;
      auto m_prime_id = m_prime[id];
      auto mi_id = mi[id];
      bool changed = m_prime_id < mi_id; // `false` if both are -inf
      if (changed) {
        auto m_prime_exp = exp2f(m_prime_id - mi_id);
        out_rescale[id] = m_prime_exp;
        s_prime[id] *= m_prime_exp;
      } else {
        // Only when bias is enabled, it's possible that all the first values
        // of attention are masked to `-inf`. In that case we want to avoid
        // `nan = exp2f(-inf - (-inf))` so we temporarily set `mi` to 0
        if (kSupportsBias &&
            mi_id == -cutlass::platform::numeric_limits<accum_t>::infinity()) {
          restore_mi_to_minus_inf = true;
          mi[id] = 0.0f;
        }
        out_rescale[id] = 1.0f;
      }
    }
    __syncthreads(); // Update output fragments
    if (kKeepOutputInRF && !is_first) {
      accum_t line_rescale;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { line_rescale = out_rescale[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag_o[idx] = frag_o[idx] * line_rescale;
          },
          [&](int accum_m) {});
    }
    // Update accum_m, accum_n, ...
    {
      accum_t mi_row, total_row;
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { mi_row = mi[accum_m]; },
          [&](int accum_m, int accum_n, int idx) {
            frag[idx] =
                (accum_n < max_col) ? exp2f(frag[idx] - mi_row) : accum_t(0.0);
          },
          [&](int accum_m) {});
      LambdaIterator::iterateRows(
          lane_offset,
          [&](int accum_m) { total_row = 0.0; },
          [&](int accum_m, int accum_n, int idx) { total_row += frag[idx]; },
          [&](int accum_m) {
            if (LambdaIterator::reduceSameRow(
                    lane_id, total_row, [](accum_t a, accum_t b) {
                      return a + b;
                    })) {
              // NOTE: we could atomically add `total_row` to `s_prime`, but
              // it's faster (and deterministic) to avoid atomics here
              addition_storage
                  [accum_m + kQueriesPerBlock * tile_offset.column()] =
                      total_row;
            }
          });
    }

    __syncthreads();
    if (lane_id < kLinesPerWarp) {
      int id = warp_id * kLinesPerWarp + lane_id;
      accum_t total_row = s_prime[id];
      if (restore_mi_to_minus_inf) {
        // Restore `mi`, see above when we set `restore_mi_to_minus_inf=true`
        mi[id] = -cutlass::platform::numeric_limits<accum_t>::infinity();
      } else {
        m_prime[id] = mi[id];
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < MM0::MmaCore::WarpCount::kN; ++i) {
        total_row += addition_storage[id + kQueriesPerBlock * i];
      }
      s_prime[id] = total_row;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
