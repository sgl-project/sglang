/// \file debug/trace.cuh
/// \brief In-kernel event tracing for debugging and profiling. Never enabled in production builds.
///
/// Off by default: every macro expands to nothing. With `SGLANG_JIT_DEBUG_TRACE=iket` (the loader
/// then defines `SGL_TRACE_IKET`) the macros emit NVIDIA IKET (In-Kernel Event Tracing) placeholders
/// and CUBIN metadata, which IKET (nvidia-cutlass-dsl) records under `sglang.kernels.jit.debug.trace`
/// or `run-iket profile -- ...`. Without IKET attached, an instrumented build still executes each
/// placeholder (a globaltimer read and a store to reserved shared memory).
///
/// Declare every name once at namespace scope, then use it in device code:
///
///   SGL_TRACE_DECLARE_RANGE(load);   // push / pop range
///   SGL_TRACE_DECLARE_SPAN(stage);   // start / end range (may cross loop iterations)
///   SGL_TRACE_DECLARE_MARK(issued);  // point event
///
///   SGL_TRACE_SCOPE(load);           // push now, pop at the end of the C++ scope
///   SGL_TRACE_PUSH(load); ...; SGL_TRACE_POP();
///   SGL_TRACE_START(stage); ...; SGL_TRACE_END(stage);
///   SGL_TRACE_MARK(issued);
///
/// `SGLANG_JIT_DEBUG_TRACE=naive` instead records `%clock` cycles into a buffer owned by
/// `sglang.kernels.jit.debug.trace` (any architecture, no IKET). It needs two more macros:
///
///   SGL_TRACE_KERNEL();              // first statement of the kernel: per-warp cursor and anchors
///   SGL_TRACE_SYNC();                // where the warp reconverges after marks made by some lanes only
///
/// Naive marks keep a per-lane cursor that every active lane advances while one elected lane writes,
/// so the cursor must stay warp-uniform: inside a divergent region only one path may mark, and the
/// warp must call SGL_TRACE_SYNC() (a max-reduction of the cursor) once it reconverges.
/// `naive-debug` checks this without waiting on anything: records are added (fire-and-forget atomic)
/// to zeroed memory and carry a writer count of 1, so an overwrite leaves a count of 2 or more, which
/// the host reports when it decodes the buffer.
/// Marks run in code inlined into the kernel (the cursor is a kernel-scope PTX register).
///
/// IKET events are per warp. Records are 4 bytes (NativeDump), which allows 30 names per translation unit;
/// define `SGL_TRACE_IKET_EXTENDED` as well for 8-byte records and up to 4031 names.
/// Event IDs come from `__COUNTER__`: declare all names right after this include, since any other
/// `__COUNTER__` expansion in between (tvm-ffi and CCCL headers use it) skips IDs and uses up the budget.
/// The placeholder and metadata layout follow NVIDIA cuda-oxide `iket-lower` (Apache-2.0).

#pragma once

#if defined(SGL_TRACE_IKET)

#include <cstdint>

namespace sglang::device::trace::iket {

// Every layout and constant below is copied from NVIDIA cuda-oxide (Apache-2.0),
// crates/iket-lower/src: metadata.rs (encode_meta_info, encode_event, encode_range, method_id),
// physical.rs (build_placeholder_ptx, placeholder_config, validate_event_id) and method.rs.
// IKET publishes no specification; fields marked "unknown" are copied verbatim.

// metadata.rs method_id: NativeDump = 3 (4 B records), ExtendedNativeDump = 5 (8 B records).
// physical.rs validate_event_id: user IDs 1..30 (NativeDump) or 64..4094 (Extended); 31 = range pop.
#if defined(SGL_TRACE_IKET_EXTENDED)
inline constexpr uint32_t kMethod = 5;
inline constexpr uint32_t kFirstUserId = 64;
inline constexpr uint32_t kLastUserId = 4094;
inline constexpr uint32_t kMaxEventId = 4095;
#else
inline constexpr uint32_t kMethod = 3;
inline constexpr uint32_t kFirstUserId = 1;
inline constexpr uint32_t kLastUserId = 30;
inline constexpr uint32_t kMaxEventId = 31;
#endif
inline constexpr uint32_t kRangePopId = 31;

// metadata.rs EventPosition and RangeType.
enum : uint32_t { kNotInRange = 0, kRangeStart = 1, kRangeStartEnd = 4 };
enum : uint32_t { kStartEnd = 1, kPushPop = 2 };

// metadata.rs encode_event (60 B, symbol prefix __iket_evt_decl_).
struct EventDecl {
  uint32_t size;       // 60
  uint32_t event_id;   // the ID in the placeholder
  uint32_t method;     // kMethod
  uint32_t payload;    // payload_id: 0 = none
  uint32_t position;   // kNotInRange / kRangeStart / kRangeStartEnd
  uint32_t range_id;   // fnv1a_32(range name); 0 for a point event
  uint32_t name_size;  // strlen + 1
  char name[32];       // inline name, at most 31 bytes + NUL
};
// metadata.rs encode_range (72 B, symbol prefix __iket_range_decl_).
struct alignas(8) RangeDecl {
  uint32_t size;          // 72
  uint32_t scope;         // 0 = kIntraWarp
  uint32_t range_id;      // fnv1a_32(name)
  uint32_t color;         // 0xffffffff = kAuto
  uint32_t type;          // kStartEnd / kPushPop
  uint32_t pair_mode;     // 1 = kNameOnly for start / end, 0 = kInvalid for push / pop
  uint64_t reserved;      // 0
  uint32_t name_size;     // strlen + 1
  char name[32];          // inline name
  uint32_t marker_count;  // 0
};
// metadata.rs encode_meta_info (48 B, symbol __iket_meta_info, one per CUBIN). The CuTe DSL 4.8.1
// compiler emits {48, 0, 5, 31, 32, 60, 0xbabef19d, 0} for the words instead; IKET accepts both.
struct alignas(8) MetaInfo {
  uint32_t size;          // 48
  uint32_t unknown1;      // 0
  uint32_t unknown2;      // 7 (CuTe DSL: 5)
  uint32_t max_event_id;  // 31 (NativeDump) or 4095 (Extended)
  uint32_t unknown4;      // 0xffffffff (CuTe DSL: 32)
  uint32_t event_size;    // 60 = sizeof(EventDecl)
  uint32_t magic;         // 0xbabef19d
  uint32_t unknown7;      // 0
  uint64_t features;      // 3 = kBasic | kPayload
  uint32_t reserved[2];   // 0
};
static_assert(sizeof(EventDecl) == 60 && sizeof(RangeDecl) == 72 && sizeof(MetaInfo) == 48);

// metadata.rs fnv1a_32: the range-ID hash of CUDA C++ IKET.
constexpr uint32_t fnv1a_32(const char* s) {
  uint32_t h = 2166136261u;
  for (; *s; ++s)
    h = (h ^ static_cast<uint8_t>(*s)) * 16777619u;
  return h;
}

inline constexpr int kCounterBase = __COUNTER__;

}  // namespace sglang::device::trace::iket

// One per translation unit; IKET reads the event-ID budget from it.
extern "C" __device__ __attribute__((used)) const ::sglang::device::trace::iket::MetaInfo __iket_meta_info = {
    48, 0, 7, ::sglang::device::trace::iket::kMaxEventId, 0xffffffffu, 60, 0xbabef19du, 0, 3, {0, 0}};

#define SGL_TRACE_IMPL_ID(name) _sgl_trace_id_##name

#define SGL_TRACE_IMPL_DECLARE_EVENT(name, position, range_id)                                                     \
  inline constexpr uint32_t SGL_TRACE_IMPL_ID(name) =                                                              \
      ::sglang::device::trace::iket::kFirstUserId - 1 + __COUNTER__ - ::sglang::device::trace::iket::kCounterBase; \
  static_assert(                                                                                                   \
      SGL_TRACE_IMPL_ID(name) <= ::sglang::device::trace::iket::kLastUserId,                                       \
      "SGL_TRACE: more than 30 names in this translation unit; use SGLANG_JIT_DEBUG_TRACE=iket-extended");         \
  extern "C" __device__ __attribute__((used))                                                                      \
  const ::sglang::device::trace::iket::EventDecl __iket_evt_decl_sgl_##name##_attrs = {                            \
      60,                                                                                                          \
      SGL_TRACE_IMPL_ID(name),                                                                                     \
      ::sglang::device::trace::iket::kMethod,                                                                      \
      0,                                                                                                           \
      position,                                                                                                    \
      range_id,                                                                                                    \
      sizeof(#name),                                                                                               \
      #name}

#define SGL_TRACE_IMPL_DECLARE_RANGE(name, type)                                          \
  extern "C" __device__ __attribute__((used))                                             \
  const ::sglang::device::trace::iket::RangeDecl __iket_range_decl_sgl_##name##_attrs = { \
      72,                                                                                 \
      0,                                                                                  \
      ::sglang::device::trace::iket::fnv1a_32(#name),                                     \
      0xffffffffu,                                                                        \
      type,                                                                               \
      type == ::sglang::device::trace::iket::kStartEnd ? 1u : 0u,                         \
      0,                                                                                  \
      sizeof(#name),                                                                      \
      #name,                                                                              \
      0}

#define SGL_TRACE_DECLARE_MARK(name) SGL_TRACE_IMPL_DECLARE_EVENT(name, ::sglang::device::trace::iket::kNotInRange, 0)
#define SGL_TRACE_DECLARE_RANGE(name)                                                                    \
  SGL_TRACE_IMPL_DECLARE_EVENT(                                                                          \
      name, ::sglang::device::trace::iket::kRangeStart, ::sglang::device::trace::iket::fnv1a_32(#name)); \
  SGL_TRACE_IMPL_DECLARE_RANGE(name, ::sglang::device::trace::iket::kPushPop)
#define SGL_TRACE_DECLARE_SPAN(name)                                                                        \
  SGL_TRACE_IMPL_DECLARE_EVENT(                                                                             \
      name, ::sglang::device::trace::iket::kRangeStartEnd, ::sglang::device::trace::iket::fnv1a_32(#name)); \
  SGL_TRACE_IMPL_DECLARE_RANGE(name, ::sglang::device::trace::iket::kStartEnd)

namespace sglang::device::trace::iket {

// physical.rs placeholder_config: the deliberately unused shared address the placeholder stores to.
// SM90: cluster rank << 24 | 0x3f0; SM10x / SM11x: cluster rank << 24 | 0x20; SM12x: 0x20.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
#define SGL_TRACE_IMPL_ADDR "mov.b32 %%r, 0x20; "
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
#define SGL_TRACE_IMPL_ADDR "mov.b32 %%r, %%cluster_ctarank; mad.lo.u32 %%r, %%r, 0x1000000, 0x20; "
#else
#define SGL_TRACE_IMPL_ADDR "mov.b32 %%r, %%cluster_ctarank; mad.lo.u32 %%r, %%r, 0x1000000, 0x3f0; "
#endif

/// physical.rs build_placeholder_ptx (no payload). IKET finds the placeholder by its PMTRIG and
/// rewrites the shared store into a store to its per-warp buffer. NativeDump packs the ID into the
/// low bits of %globaltimer_lo (it ticks every 32 ns); Extended stores {timestamp, ID} as 8 bytes.
template <uint32_t kId>
__device__ __forceinline__ void emit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#if defined(SGL_TRACE_IKET_EXTENDED)
  asm volatile(
      "{ .reg .b32 %%r, %%t, %%e; .reg .b64 %%te; mov.u32 %%t, %%globaltimer_lo; " SGL_TRACE_IMPL_ADDR
      "mov.b32 %%e, %0; mov.b64 %%te, {%%t, %%e}; st.weak.shared.u64 [%%r], %%te; pmevent.mask %0; }" ::"n"(kId)
      : "memory");
#else
  asm volatile("{ .reg .b32 %%r, %%t; mov.u32 %%t, %%globaltimer_lo; or.b32 %%t, %%t, %0; " SGL_TRACE_IMPL_ADDR
               "st.weak.shared.u32 [%%r], %%t; pmevent.mask %0; }" ::"n"(kId)
               : "memory");
#endif
#endif
}

struct ScopeGuard {
  __device__ __forceinline__ ~ScopeGuard() {
    emit<kRangePopId>();
  }
};

}  // namespace sglang::device::trace::iket

#define SGL_TRACE_MARK(name) ::sglang::device::trace::iket::emit<SGL_TRACE_IMPL_ID(name)>()
#define SGL_TRACE_PUSH(name) ::sglang::device::trace::iket::emit<SGL_TRACE_IMPL_ID(name)>()
#define SGL_TRACE_POP() ::sglang::device::trace::iket::emit<::sglang::device::trace::iket::kRangePopId>()
#define SGL_TRACE_START(name) ::sglang::device::trace::iket::emit<SGL_TRACE_IMPL_ID(name)>()
#define SGL_TRACE_END(name) ::sglang::device::trace::iket::emit<SGL_TRACE_IMPL_ID(name)>()
#define SGL_TRACE_IMPL_CAT2(a, b) a##b
#define SGL_TRACE_IMPL_CAT(a, b) SGL_TRACE_IMPL_CAT2(a, b)
#define SGL_TRACE_SCOPE(name) \
  SGL_TRACE_PUSH(name);       \
  ::sglang::device::trace::iket::ScopeGuard SGL_TRACE_IMPL_CAT(_sgl_trace_scope_, __LINE__)
#define SGL_TRACE_KERNEL() ((void)0)
#define SGL_TRACE_SYNC() ((void)0)

#elif defined(SGL_TRACE_NAIVE)

#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace sglang::device::trace::naive {

enum Kind : uint32_t { kMark = 0, kPush = 1, kPop = 2, kStart = 3, kEnd = 4 };

/// Written by the host before a capture; base 0 disables recording.
struct Config {
  uint64_t base;              // device buffer of slots
  uint32_t records_per_warp;  // records stored per slot; later ones are only counted
  uint32_t cta_begin;         // linear CTA IDs in [cta_begin, cta_end) record
  uint32_t cta_end;
  uint32_t reserved;
};

/// One slot per recording warp: this header, then records_per_warp records of 8 bytes,
/// {%clock, tag}, tag = writers << 26 | id << 3 | kind. writers is 1 in every record; two guard bits
/// (24, 25) take the carries of summed tags, so naive-debug's added records keep an exact count.
struct SlotHeader {
  uint32_t count;         // records issued, stored or not
  uint32_t location;      // smid << 16 | warp in CTA
  uint32_t cta;           // linear CTA ID
  uint32_t clock_begin;   // %clock at SGL_TRACE_KERNEL()
  uint64_t ns_begin;      // %globaltimer at SGL_TRACE_KERNEL()
  uint32_t clock_end;     // %clock when the kernel scope ends
  uint32_t ns_end_delta;  // %globaltimer at that point minus ns_begin
};
static_assert(sizeof(SlotHeader) == 32);

inline constexpr int kCounterBase = __COUNTER__;

}  // namespace sglang::device::trace::naive

// One per translation unit; set through the exported sgl_trace_set_config.
static __constant__ ::sglang::device::trace::naive::Config sgl_trace_config;

namespace sglang::host::trace::naive {

/// "id:kind:name" per declared name, filled by static initializers when the module loads.
inline std::vector<std::string>& registry() {
  static std::vector<std::string> names;
  return names;
}

inline bool add(uint32_t id, const char* kind, const char* name) {
  registry().push_back(std::to_string(id) + ":" + kind + ":" + name);
  return true;
}

inline void set_config(int64_t base, int64_t records_per_warp, int64_t cta_begin, int64_t cta_end) {
  const ::sglang::device::trace::naive::Config config = {
      static_cast<uint64_t>(base),
      static_cast<uint32_t>(records_per_warp),
      static_cast<uint32_t>(cta_begin),
      static_cast<uint32_t>(cta_end),
      0};
  const cudaError_t status = cudaMemcpyToSymbol(sgl_trace_config, &config, sizeof(config));
  if (status != cudaSuccess) {
    TVM_FFI_THROW(RuntimeError) << "sgl_trace_set_config: " << cudaGetErrorString(status);
  }
}

inline tvm::ffi::String events() {
  std::string joined;
  for (const auto& entry : registry()) {
    joined += entry + ";";
  }
  return tvm::ffi::String(joined);
}

}  // namespace sglang::host::trace::naive

// A module holds one traced translation unit: a second one would export these names again.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_trace_set_config, ::sglang::host::trace::naive::set_config);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(sgl_trace_events, ::sglang::host::trace::naive::events);

#define SGL_TRACE_IMPL_ID(name) _sgl_trace_id_##name

#define SGL_TRACE_IMPL_DECLARE(name, kind)                                                                        \
  inline constexpr uint32_t SGL_TRACE_IMPL_ID(name) = __COUNTER__ - ::sglang::device::trace::naive::kCounterBase; \
  [[maybe_unused]] inline const bool _sgl_trace_registered_##name =                                               \
      ::sglang::host::trace::naive::add(SGL_TRACE_IMPL_ID(name), kind, #name)

#define SGL_TRACE_DECLARE_MARK(name) SGL_TRACE_IMPL_DECLARE(name, "mark")
#define SGL_TRACE_DECLARE_RANGE(name) SGL_TRACE_IMPL_DECLARE(name, "range")
#define SGL_TRACE_DECLARE_SPAN(name) SGL_TRACE_IMPL_DECLARE(name, "span")

// %%p = one elected lane among the active ones (%%m = activemask). elect.sync needs sm_90; before
// that, the lowest active lane.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define SGL_TRACE_IMPL_ELECT                                                                                      \
  "activemask.b32 %%m; { .reg .b32 %%l, %%b, %%n; mov.u32 %%l, %%laneid; shl.b32 %%b, 1, %%l; neg.s32 %%n, %%m; " \
  "and.b32 %%n, %%n, %%m; setp.eq.u32 %%p, %%n, %%b; } "
#else
#define SGL_TRACE_IMPL_ELECT "activemask.b32 %%m; elect.sync _|%%p, %%m; "
#endif

namespace sglang::device::trace::naive {

inline constexpr uint32_t kOneWriter = 1u << 26;

// The bookkeeping (election, room check, address) runs outside the measured interval: an event that
// opens a range reads %clock last, one that closes a range (or a mark) reads it first, so an empty
// range holds only the opening event's store.
#define SGL_TRACE_IMPL_PREP                                         \
  SGL_TRACE_IMPL_ELECT                                              \
  "setp.lt.u32 %%q, %%sgl_off, %%sgl_end; and.pred %%p, %%p, %%q; " \
  "cvt.u64.u32 %%a, %%sgl_off; add.u64 %%a, %%a, %0; "
#if defined(SGL_TRACE_NAIVE_DEBUG)
// naive-debug adds the record instead (red: no result, no wait), so an overwrite shows as 2 writers.
#define SGL_TRACE_IMPL_WRITE \
  "mov.b64 %%r, {%%c, %1}; @%%p red.global.add.u64 [%%a], %%r; add.u32 %%sgl_off, %%sgl_off, 8; }"
#else
#define SGL_TRACE_IMPL_WRITE "@%%p st.global.v2.b32 [%%a], {%%c, %1}; add.u32 %%sgl_off, %%sgl_off, 8; }"
#endif
#define SGL_TRACE_IMPL_REGS "{ .reg .pred %%p, %%q; .reg .b32 %%c, %%m; .reg .b64 %%a, %%r; "

/// One record: every active lane advances the kernel-scope cursor, one elected lane writes while the
/// slot has room.
template <uint32_t kTag, bool kOpens>
__device__ __forceinline__ void record(uint64_t base) {
  if constexpr (kOpens) {
    asm volatile(SGL_TRACE_IMPL_REGS SGL_TRACE_IMPL_PREP "mov.u32 %%c, %%clock; " SGL_TRACE_IMPL_WRITE ::"l"(base),
                 "n"(kTag | kOneWriter)
                 : "memory");
  } else {
    asm volatile(SGL_TRACE_IMPL_REGS "mov.u32 %%c, %%clock; " SGL_TRACE_IMPL_PREP SGL_TRACE_IMPL_WRITE ::"l"(base),
                 "n"(kTag | kOneWriter)
                 : "memory");
  }
}

template <uint32_t kId, uint32_t kKind>
__device__ __forceinline__ void emit() {
  static_assert(kId < (1u << 21), "SGL_TRACE: event ID out of range");
  record<(kId << 3) | kKind, kKind == kPush || kKind == kStart>(sgl_trace_config.base);
}

__device__ __forceinline__ uint32_t elect() {
  uint32_t elected;
  asm volatile("{ .reg .pred %%p; .reg .b32 %%m; " SGL_TRACE_IMPL_ELECT "selp.u32 %0, 1, 0, %%p; }" : "=r"(elected));
  return elected;
}

/// SGL_TRACE_KERNEL(): declares the kernel-scope cursor, writes the begin anchors, and at the end of
/// the kernel scope writes the end anchors and the record count.
struct KernelGuard {
  SlotHeader* header = nullptr;
  uint32_t records_begin = 0;

  __device__ __forceinline__ KernelGuard() {
    const Config config = sgl_trace_config;
    const uint32_t cta = blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z);
    const uint32_t thread = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const uint32_t warps_per_cta = (blockDim.x * blockDim.y * blockDim.z + 31) / 32;
    const bool enabled = config.base != 0 && cta >= config.cta_begin && cta < config.cta_end;
    const uint32_t slot_bytes = sizeof(SlotHeader) + 8 * config.records_per_warp;
    const uint32_t slot = enabled ? (cta - config.cta_begin) * warps_per_cta + thread / 32 : 0;
    records_begin = slot * slot_bytes + sizeof(SlotHeader);
    const uint32_t records_end = enabled ? records_begin + 8 * config.records_per_warp : records_begin;
    asm volatile(
        ".reg .b32 %%sgl_off;\n\t.reg .b32 %%sgl_end;\n\tmov.b32 %%sgl_off, %0;\n\tmov.b32 %%sgl_end, %1;" ::"r"(
            records_begin),
        "r"(records_end)
        : "memory");
    if (enabled) {
      header = reinterpret_cast<SlotHeader*>(config.base + records_begin - sizeof(SlotHeader));
    }
    if (enabled && elect()) {
      uint32_t clock, smid;
      uint64_t ns;
      asm volatile("mov.u32 %0, %%clock; mov.u64 %1, %%globaltimer; mov.u32 %2, %%smid;"
                   : "=r"(clock), "=l"(ns), "=r"(smid));
      header->location = (smid << 16) | (thread / 32);
      header->cta = cta;
      header->clock_begin = clock;
      header->ns_begin = ns;
    }
  }

  __device__ __forceinline__ ~KernelGuard() {
    uint32_t clock, cursor;
    uint64_t ns;
    asm volatile(
        "{ .reg .b32 %%m; activemask.b32 %%m; redux.sync.max.u32 %%sgl_off, %%sgl_off, %%m; }\n\t"
        "mov.u32 %0, %%clock;\n\tmov.u64 %1, %%globaltimer;\n\tmov.b32 %2, %%sgl_off;"
        : "=r"(clock), "=l"(ns), "=r"(cursor)
        :
        : "memory");
    const uint32_t count = (cursor - records_begin) / 8;
    if (header != nullptr && elect()) {
      header->count = count;
      header->clock_end = clock;
      header->ns_end_delta = static_cast<uint32_t>(ns - header->ns_begin);
    }
  }
};

/// Pops at the end of the C++ scope.
struct ScopeGuard {
  __device__ __forceinline__ ~ScopeGuard() {
    record<kPop, false>(sgl_trace_config.base);
  }
};

}  // namespace sglang::device::trace::naive

#define SGL_TRACE_MARK(name) \
  ::sglang::device::trace::naive::emit<SGL_TRACE_IMPL_ID(name), ::sglang::device::trace::naive::kMark>()
#define SGL_TRACE_PUSH(name) \
  ::sglang::device::trace::naive::emit<SGL_TRACE_IMPL_ID(name), ::sglang::device::trace::naive::kPush>()
#define SGL_TRACE_POP() \
  ::sglang::device::trace::naive::record<::sglang::device::trace::naive::kPop, false>(sgl_trace_config.base)
#define SGL_TRACE_START(name) \
  ::sglang::device::trace::naive::emit<SGL_TRACE_IMPL_ID(name), ::sglang::device::trace::naive::kStart>()
#define SGL_TRACE_END(name) \
  ::sglang::device::trace::naive::emit<SGL_TRACE_IMPL_ID(name), ::sglang::device::trace::naive::kEnd>()
#define SGL_TRACE_IMPL_CAT2(a, b) a##b
#define SGL_TRACE_IMPL_CAT(a, b) SGL_TRACE_IMPL_CAT2(a, b)
#define SGL_TRACE_SCOPE(name) \
  SGL_TRACE_PUSH(name);       \
  ::sglang::device::trace::naive::ScopeGuard SGL_TRACE_IMPL_CAT(_sgl_trace_scope_, __LINE__)
#define SGL_TRACE_KERNEL() ::sglang::device::trace::naive::KernelGuard _sgl_trace_kernel
// Clobber-only asm is basic asm: no %% escaping.
#define SGL_TRACE_SYNC() \
  asm volatile("{ .reg .b32 %m; activemask.b32 %m; redux.sync.max.u32 %sgl_off, %sgl_off, %m; }" ::: "memory")

#else  // tracing off

#define SGL_TRACE_DECLARE_MARK(name) static_assert(true)
#define SGL_TRACE_DECLARE_RANGE(name) static_assert(true)
#define SGL_TRACE_DECLARE_SPAN(name) static_assert(true)
#define SGL_TRACE_MARK(name) ((void)0)
#define SGL_TRACE_PUSH(name) ((void)0)
#define SGL_TRACE_POP() ((void)0)
#define SGL_TRACE_START(name) ((void)0)
#define SGL_TRACE_END(name) ((void)0)
#define SGL_TRACE_SCOPE(name) ((void)0)
#define SGL_TRACE_KERNEL() ((void)0)
#define SGL_TRACE_SYNC() ((void)0)

#endif
