// K3 fused o_proj GEMM + all-reduce (bf16, TP row-parallel) — HOST SHIM.
//
// The kernels themselves are NOT in this tree: they ship as a prebuilt cubin
// whose path comes from SGLANG_K3_GEMM_AR_CUBIN (resolved in gemm_ar.py).
//
// CONTRACT (per rank r of R):  out[M, 7168] = sum_r x_r[M, K] @ W_r[7168, K]^T
//   bf16 in/out, fp32 accumulate, partials round to bf16 pre-sum (same
//   semantics as the unfused GEMM + bf16 ring AR).
// M in [1, 512], rounded up to a tuned cell {8,16,32,64,128,256,512}; out
// must have `cell` rows — rows [M, cell) are clobbered with zeros.
// Comm plane: pure NVLink P2P (unicast pushes + per-rank flag reductions);
// one-shot AR below the two-shot threshold, two-shot RS+AG above.
// Requires SM100+ with full P2P; tuned on GB300 (sm_103a).
#include <sgl_kernel/cuda_check.h>
#include <sgl_kernel/tensor_map.h>

#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <string>

namespace oproj_ar {

// ---------------------------------------------------------------- constants
#ifndef OPROJ_N       // output dim (columns of W / of out). The
#define OPROJ_N 7168  // default is the Kimi-K3 o_proj shape; other
#endif                // shapes compile via -DOPROJ_N (see asserts).
constexpr int kN = OPROJ_N;
constexpr int kBK = 64;                       // K per stage (128 B rows → swizzle-128B)
constexpr int kBNRows = 48;                   // B-box rows per stage (6 n8-tiles)
constexpr int kCWarps = 6;                    // consumer warp w owns n8-tile w
constexpr int kThreads = (kCWarps + 1) * 32;  // +1 dedicated TMA producer warp
constexpr int kMMax = 512;                    // bs64 — sizes the shared slot layout
constexpr int kRing = 64;                     // epoch flag/gather ring (monotonic values)

enum class Comm { kNone, kMc, kPeer, kMcPull, kTwoShot, kTwoShotPeer };

// Shared-region layout (BYTES, M-independent: sized at kMMax so every arm and
// every bs cell reuses one region). Parity-double-buffered payloads; the flag
// ring lives on its own 2 MB page.
constexpr size_t kSlotBytes1 = size_t(kMMax) * kN * 2;  // one [M,N] bf16
constexpr size_t flags_off(int R) {
  const size_t end = (2 * size_t(R) + 2) * kSlotBytes1;
  return (end + (size_t(2) << 20)) & ~((size_t(2) << 20) - 1);
}
constexpr int kMaxCTA = 256;
constexpr int kFams = 7;  // flag/gather/done ring FAMILY per dispatch
                          // CELL {8,16,32,64,128,256,512}
constexpr size_t done_off(int R) {
  return flags_off(R) + size_t(kFams) * 512;
}
constexpr size_t region_bytes(int R) {
  return done_off(R) + size_t(kFams) * kRing * kMaxCTA * 4 + 512;
}

// ------------------------------------------------------------------ params
// Byte-for-byte the kernel-side struct; see ABI LOCKSTEP above.
template <int R>
struct Params {
  uint8_t* mc_base;                 // MC VA (null when no MC object — kPeer runs)
  uint8_t* uc_base[R];              // per-rank unicast VAs of the shared region
  uint32_t* gather;                 // device-local u32[kRing]
  __nv_bfloat16* out;               // [M,N] local output
  const __nv_bfloat16* partial_in;  // GEMM_ON=false input [M,N]
  uint32_t* epoch_base;             // per-fam CTA ticket counters (device state)
  int my_rank;
  int fam;  // ring family (dispatch cell) — see kFams
};

// -------------------------------------------------------- dense (member 3)
constexpr int kD3BM = 128, kD3BK = 64;
constexpr int d3_ns(int bn) {  // ring depth by stage size
  const int stage = kD3BM * kD3BK * 2 + bn * kD3BK * 2;
  const int ns = 220 * 1024 / stage;
  return ns > 12 ? 12 : ns;
}
constexpr int kD3Threads = 8 * 32;  // warps 0,1,4-7 live
constexpr int kD3ABytes = kD3BM * kD3BK * 2;
constexpr int d3_bn(int Mp) {
  return Mp <= 128 ? 64 : (Mp == 256 ? 128 : 256);
}

// ------------------------------------------------------------- cubin plane
inline CUmodule& cubin_module() {
  static CUmodule m = nullptr;
  return m;
}

inline void load_cubin(const std::string& path) {
  CUDA_CHECK(cudaFree(nullptr));  // force the primary context current for the driver API
  CUmodule m = nullptr;
  CU_CHECK(cuModuleLoad(&m, path.c_str()));
  cubin_module() = m;
}

// Itanium mangling of the kernel template instantiations. Params<R> mangles
// as the dependent expression T1_ (the third template param) in both.
inline std::string ar_sym(int M, int K, int R, Comm comm, bool gemm_on, int S, int CH, int C) {
  char buf[256];
  std::snprintf(
      buf,
      sizeof(buf),
      "_ZN8oproj_ar15oproj_ar_kernelILi%dELi%dELi%dELNS_4CommE%dELb%dELi%dELi%dELi%dEEEv14CUtensorMap_stS2_NS_"
      "6ParamsIXT1_EEE",
      M,
      K,
      R,
      int(comm),
      int(gemm_on),
      S,
      CH,
      C);
  return buf;
}

inline std::string dense_sym(int M, int K, int R, Comm comm) {
  char buf[256];
  std::snprintf(
      buf,
      sizeof(buf),
      "_ZN8oproj_ar21oproj_dense_ar_kernelILi%dELi%dELi%dELNS_4CommE%dEEEv14CUtensorMap_stS2_NS_6ParamsIXT1_EEE",
      M,
      K,
      R,
      int(comm));
  return buf;
}

// 100% carveout → the SM smem config fits TWO CTAs (this grid's tail + the
// next PDL grid's feed); the default config blocks dual residency and with
// it the whole tail-hiding scheme.
inline CUfunction cubin_fn(const std::string& sym, size_t smem) {
  if (cubin_module() == nullptr) {
    std::fprintf(stderr, "gemm_ar: cubin not loaded (set SGLANG_K3_GEMM_AR_CUBIN)\n");
    std::abort();
  }
  CUfunction f = nullptr;
  CU_CHECK(cuModuleGetFunction(&f, cubin_module(), sym.c_str()));
  CU_CHECK(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, int(smem)));
  CU_CHECK(cuFuncSetAttribute(f, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, 100));
  return f;
}

// pdl=false = the NON-COOPERATIVE-neighbor regime: no programmatic
// serialization attribute, so successive kernels fully serialize — the AR
// tail is exposed. In-kernel griddepcontrol ops are no-ops without the
// attribute; the done-guard is trivially met.
inline void launch_cubin(CUfunction fn, int grid, int block, size_t smem, void** args, cudaStream_t stream, bool pdl) {
  CUlaunchConfig cfg{};
  CUlaunchAttribute attr[1];
  cfg.gridDimX = unsigned(grid);
  cfg.gridDimY = 1;
  cfg.gridDimZ = 1;
  cfg.blockDimX = unsigned(block);
  cfg.blockDimY = 1;
  cfg.blockDimZ = 1;
  cfg.sharedMemBytes = unsigned(smem);
  cfg.hStream = reinterpret_cast<CUstream>(stream);
  if (pdl) {
    attr[0].id = CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION;
    attr[0].value.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  CU_CHECK(cuLaunchKernelEx(&cfg, fn, args, nullptr));
}

template <typename T>
void* as_arg(const T& v) {
  return const_cast<void*>(static_cast<const void*>(&v));
}

// ------------------------------------------------------------- launch glue
// Ring geometry per Mp (smem-budget-fit): batching CH k-chunks per stage
// amortizes the per-stage mbar/wake overhead; ring depth S divides the fixed
// feed latency F. smem ≤ ~113 KB/CTA so the NEXT PDL launch co-resides with
// this grid's boundary spin (2 CTA/SM transiently) — the tail hides under
// its feed.
// ponytail: the cluster-multicast A axis (C>1) is dropped with the cubin
// split — C=1 is the only shipped instantiation (LEDGER R9 refuted C=2).
template <
    int M,
    int K,
    int R,
    Comm COMM,
    bool GEMM_ON,
    int S = ((M + 15) & ~15) <= 16 ? 6 : (((M + 15) & ~15) == 32 ? 4 : 3),
    int CH = 2>
struct Launcher {
  static constexpr int Mp = (M + 15) & ~15;
  static constexpr size_t kSmem = GEMM_ON
                                      ? size_t(S) * CH * (kBNRows * kBK * 2 + Mp * kBK * 2) + 2 * S * sizeof(uint64_t)
                                      : 4096;  // AR-only path never touches the feed ring
  static CUfunction fn() {
    static CUfunction f = cubin_fn(ar_sym(M, K, R, COMM, GEMM_ON, S, CH, 1), kSmem);
    return f;
  }
  static void launch(
      const CUtensorMap& w_map,
      const CUtensorMap& x_map,
      const Params<R>& prm,
      int ncta,
      cudaStream_t stream,
      bool pdl) {
    void* args[] = {as_arg(w_map), as_arg(x_map), as_arg(prm)};
    launch_cubin(fn(), ncta, kThreads, kSmem, args, stream, pdl);
  }
};

template <int M, int K, int R, Comm COMM>
struct Launcher3 {
  static constexpr bool kSwap = (M <= 64);
  static constexpr int Mp = kSwap ? (M < 16 ? 16 : ((M + 15) & ~15)) : (M + kD3BM - 1) / kD3BM * kD3BM;
  static constexpr int kBN = kSwap ? (Mp <= 32 ? 16 : 32) : d3_bn(Mp);
  static constexpr int kTiles = kSwap ? (kN / kD3BM) * (Mp / kBN) : (Mp / kD3BM) * (kN / kBN);
  static constexpr int kGrid = kTiles < 152 ? kTiles : 152;
  static constexpr size_t kSmem = size_t(d3_ns(kBN)) * (kD3ABytes + kBN * kD3BK * 2);
  static CUfunction fn() {
    static CUfunction f = cubin_fn(dense_sym(M, K, R, COMM), kSmem);
    return f;
  }
  static void
  launch(const CUtensorMap& x_tmap, const CUtensorMap& w_tmap, const Params<R>& prm, cudaStream_t stream, bool pdl) {
    void* args[] = {as_arg(x_tmap), as_arg(w_tmap), as_arg(prm)};
    launch_cubin(fn(), kGrid, kD3Threads, kSmem, args, stream, pdl);
  }
};

}  // namespace oproj_ar

// ================= sglang tvm-ffi adapter =================

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For CHECK_HOST

#include <sgl_kernel/utils.cuh>  // For bf16_t, TVMFFIEnvGetStream

#include <tvm/ffi/container/tensor.h>

#include <array>
#include <mutex>
#include <unordered_map>

namespace oproj_ar_ffi {

using namespace oproj_ar;
using tvm::ffi::TensorView;

constexpr int kCellList[7] = {8, 16, 32, 64, 128, 256, 512};

inline int cell_of(int m) {
  for (int c : kCellList)
    if (m <= c) return c;
  return -1;
}

template <int K, int R, bool kUsePDL>
struct GemmArKernel {
  static_assert(R >= 2 && R <= 8, "R outside the validated 2..8 range");
  static_assert(K % 128 == 0 && K >= 128, "K must be a multiple of 128");
  static_assert(
      kN % 256 == 0 && kN >= 256,
      "N must be a multiple of 256 (member tile table: 128-row "
      "strips + BN up to 256); relaxing this needs a BN-table edit");

  static constexpr int kTwoShotMinM = R >= 8 ? 128 : 256;

  // one ring family per dispatch cell (see kFams)
  static int64_t fam_of(int64_t m) {
    const int cell = cell_of(int(m));
    CHECK_HOST(cell > 0) << "gemm_ar: M=" << m << " outside [1, " << kMMax << "]";
    for (int i = 0; i < 7; ++i)
      if (kCellList[i] == cell) return i;
    return -1;
  }

  static int64_t cell_of_ffi(int64_t m) {
    return cell_of(int(m));
  }
  static int64_t region_nbytes() {
    return int64_t(region_bytes(R));
  }
  static int64_t gather_words() {
    return int64_t(kFams) * 2 * kRing;
  }
  static int64_t num_fams() {
    return kFams;
  }
  static int64_t max_tokens() {
    return kMMax;
  }

  // The device plane. Loaded once at init from SGLANG_K3_GEMM_AR_CUBIN;
  // every kernel lookup below then resolves out of this module.
  static void set_cubin(std::string path) {
    CHECK_HOST(!path.empty()) << "gemm_ar: empty cubin path";
    load_cubin(path);
  }

  // Per-weight-pointer W tensor maps (encode once; weights are static).
  struct WMaps {
    CUtensorMap w48, w64, w128, w256;
  };

  static const WMaps& w_maps(void* w) {
    static std::unordered_map<void*, WMaps> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    auto it = cache.find(w);
    if (it == cache.end()) {
      auto enc = [&](uint32_t box_rows) {
        return tmap::encode_tiled_2d(
            w, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, kN, K, size_t(K) * 2, box_rows, kBK, CU_TENSOR_MAP_SWIZZLE_128B);
      };
      it = cache.emplace(w, WMaps{enc(kBNRows), enc(64), enc(128), enc(256)}).first;
    }
    return it->second;
  }

  // x tensor map over the caller's [M, K] tensor: global rows = M, TMA
  // zero-fills the [M, cell) padding rows out-of-bounds.
  static CUtensorMap x_map(void* x, int m, uint32_t box_rows) {
    return tmap::encode_tiled_2d(
        x, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, uint64_t(m), K, size_t(K) * 2, box_rows, kBK, CU_TENSOR_MAP_SWIZZLE_128B);
  }

  template <int CELL>
  static void enqueue_cell(const WMaps& wm, void* x, int m, const Params<R>& prm, cudaStream_t stream, bool pdl) {
    if constexpr (CELL <= 16) {
      const CUtensorMap xm = x_map(x, m, 16);
      Launcher<CELL, K, R, Comm::kPeer, true>::launch(wm.w48, xm, prm, kM1Grid_(), stream, pdl);
    } else if constexpr (CELL < kTwoShotMinM) {
      const CUtensorMap xm = x_map(x, m, CELL <= 32 ? 16 : (CELL <= 64 ? 32 : 128));
      const CUtensorMap& wmap = CELL <= 64 ? wm.w128 : wm.w64;
      Launcher3<CELL, K, R, Comm::kPeer>::launch(xm, wmap, prm, stream, pdl);
    } else {
      const CUtensorMap xm = x_map(x, m, 128);
      const CUtensorMap& wmap = CELL == 128 ? wm.w64 : (CELL == 256 ? wm.w128 : wm.w256);
      Launcher3<CELL, K, R, Comm::kTwoShotPeer>::launch(xm, wmap, prm, stream, pdl);
    }
  }

  static constexpr int kM1Grid_() {
    return 152;
  }

  // per-rank UC VAs of the comm region, stashed host-side ONCE at init:
  // per-call CPU-tensor derefs from inside the op are not reliable in every
  // execution context (observed dangling under the sglang scheduler).
  static std::array<uint8_t*, R>& bases_store() {
    static std::array<uint8_t*, R> a{};
    return a;
  }

  static void set_bases(TensorView uc_bases) {
    using namespace host;
    TensorMatcher({R}).with_dtype<int64_t>().verify(uc_bases);
    const int64_t* b = static_cast<const int64_t*>(uc_bases.data_ptr());
    for (int r = 0; r < R; ++r)
      bases_store()[r] = reinterpret_cast<uint8_t*>(b[r]);
  }

  static void
  run(TensorView out,
      TensorView x,
      TensorView w,
      TensorView gather,  // [kFams * 2 * kRing] int32 CUDA, device-local
      TensorView epochs,  // [kFams] int32 CUDA: device-resident CTA ticket counters
      int64_t my_rank) {
    using namespace host;

    auto M = SymbolicSize{"num_tokens"};
    auto CellRows = SymbolicSize{"cell_rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({M, K}).with_dtype<bf16_t>().with_device(device).verify(x);
    TensorMatcher({kN, K}).with_dtype<bf16_t>().with_device(device).verify(w);
    TensorMatcher({CellRows, kN}).with_dtype<bf16_t>().with_device(device).verify(out);

    const int m = int(M.unwrap());
    const int cell = cell_of(m);
    CHECK_HOST(cell > 0) << "gemm_ar: M=" << m << " outside [1, " << kMMax << "]";
    CHECK_HOST(int64_t(CellRows.unwrap()) == cell)
        << "out must have cell(M)=" << cell << " rows, got " << CellRows.unwrap();
    CHECK_HOST(my_rank >= 0 && my_rank < R);
    CHECK_HOST(bases_store()[0] != nullptr) << "gemm_ar: set_bases not called";
    TensorMatcher({int64_t(kFams) * 2 * kRing}).with_dtype<int32_t>().verify(gather);
    TensorMatcher({kFams}).with_dtype<int32_t>().verify(epochs);

    const DLDevice dev = device.unwrap();
    const auto stream = static_cast<cudaStream_t>(::TVMFFIEnvGetStream(dev.device_type, dev.device_id));

    Params<R> prm{};
    prm.mc_base = nullptr;  // pure-P2P plane
    for (int r = 0; r < R; ++r)
      prm.uc_base[r] = bases_store()[r];
    prm.gather = static_cast<uint32_t*>(gather.data_ptr());
    prm.out = static_cast<__nv_bfloat16*>(out.data_ptr());
    prm.partial_in = nullptr;
    prm.epoch_base = static_cast<uint32_t*>(epochs.data_ptr());
    prm.my_rank = int(my_rank);
    prm.fam = int(fam_of(m));

    const bool pdl = kUsePDL;
    const WMaps& wm = w_maps(w.data_ptr());
    switch (cell) {
      case 8:
        enqueue_cell<8>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 16:
        enqueue_cell<16>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 32:
        enqueue_cell<32>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 64:
        enqueue_cell<64>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 128:
        enqueue_cell<128>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      case 256:
        enqueue_cell<256>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
      default:
        enqueue_cell<512>(wm, x.data_ptr(), m, prm, stream, pdl);
        break;
    }
    CHECK_CUDA(cudaGetLastError()) << "gemm_ar launch (cell=" << cell << ")";
  }
};

}  // namespace oproj_ar_ffi

using oproj_ar_ffi::GemmArKernel;
