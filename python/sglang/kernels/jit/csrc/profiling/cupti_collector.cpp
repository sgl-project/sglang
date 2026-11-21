// Minimal CUPTI activity collector (experimental). Not header-only: built from
// this source by sglang/kernels/jit/debug/cupti_collector.py against a chosen
// CUPTI install and exported through tvm-ffi. CUPTI state is process-global,
// so every entry point acts on one collector. Kernel / memcpy / memset records
// are reduced to fixed int64 rows kept in memory; graph exec creation is
// tracked to map exec ids to the user graphs they were instantiated from.

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/string.h>

// Not <cupti.h>: it pulls runtime-API metadata headers that need a matching
// CUDA toolkit; these are enough here.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cupti_activity.h>
#include <cupti_callbacks.h>
#include <cupti_version.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if CUPTI_API_VERSION >= 130400
typedef CUpti_ActivityKernel13 KernelRecord;
typedef CUpti_ActivityMemcpy7 MemcpyRecord;
typedef CUpti_ActivityMemset5 MemsetRecord;
#else
typedef CUpti_ActivityKernel10 KernelRecord;
typedef CUpti_ActivityMemcpy6 MemcpyRecord;
typedef CUpti_ActivityMemset4 MemsetRecord;
#endif

namespace sglang::cupti_collector {

using sglang::host::RuntimeCheck;
using tvm::ffi::TensorView;

// Column order of a record row; RECORD_FIELDS in cupti_collector.py names them.
enum Field : int64_t {
  kStart,
  kEnd,
  kGraphNodeId,
  kGraphId,
  kCorrelationId,
  kNameId,
  kStreamId,
  kKind,  // 0 kernel, 1 memcpy, 2 memset
  kGridX,
  kGridY,
  kGridZ,
  kNumFields,
};

using Row = std::array<int64_t, kNumFields>;

constexpr size_t kBufferSize = 8 << 20;

std::mutex g_mu;
std::vector<Row> g_rows;
std::vector<std::string> g_names;
std::unordered_map<std::string, int64_t> g_name_index;
// (exec id, graph handle) in creation order; graphs are identified by handle,
// since the capture-time op map records the graph under capture without CUPTI.
std::vector<std::pair<int64_t, int64_t>> g_execs;
std::unordered_map<int64_t, int64_t> g_clone_source;  // cloned graph -> original
int64_t g_dropped = 0;
int64_t g_callback_errors = 0;
CUpti_SubscriberHandle g_subscriber = nullptr;
bool g_registered = false;

const CUpti_ActivityKind kKinds[] = {
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET};

void check_cupti(CUptiResult result, const char* what) {
  const char* message = nullptr;
  cuptiGetResultString(result, &message);
  RuntimeCheck(result == CUPTI_SUCCESS, what, " failed: ", message ? message : "unknown CUPTI error");
}

int64_t intern_name(const char* name) {
  std::string key = name ? name : "";
  auto it = g_name_index.find(key);
  if (it != g_name_index.end()) return it->second;
  const auto id = static_cast<int64_t>(g_names.size());
  g_names.push_back(key);
  g_name_index.emplace(std::move(key), id);
  return id;
}

template <typename Record>
Row make_row(const Record* r, int64_t name_id, int64_t kind) {
  Row row{};
  row[kStart] = static_cast<int64_t>(r->start);
  row[kEnd] = static_cast<int64_t>(r->end);
  row[kGraphNodeId] = static_cast<int64_t>(r->graphNodeId);
  row[kGraphId] = r->graphId;
  row[kCorrelationId] = r->correlationId;
  row[kNameId] = name_id;
  row[kStreamId] = r->streamId;
  row[kKind] = kind;
  return row;
}

void CUPTIAPI buffer_requested(uint8_t** buffer, size_t* size, size_t* max_num_records) {
  *buffer = static_cast<uint8_t*>(aligned_alloc(8, kBufferSize));
  *size = kBufferSize;
  *max_num_records = 0;
}

void CUPTIAPI buffer_completed(CUcontext, uint32_t, uint8_t* buffer, size_t, size_t valid_size) {
  CUpti_Activity* record = nullptr;
  std::lock_guard<std::mutex> lock(g_mu);
  while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
    switch (record->kind) {
      case CUPTI_ACTIVITY_KIND_KERNEL:
      case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
        auto* k = reinterpret_cast<KernelRecord*>(record);
        Row row = make_row(k, intern_name(k->name), 0);
        row[kGridX] = k->gridX;
        row[kGridY] = k->gridY;
        row[kGridZ] = k->gridZ;
        g_rows.push_back(row);
        break;
      }
      case CUPTI_ACTIVITY_KIND_MEMCPY:
        g_rows.push_back(make_row(reinterpret_cast<MemcpyRecord*>(record), intern_name("memcpy"), 1));
        break;
      case CUPTI_ACTIVITY_KIND_MEMSET:
        g_rows.push_back(make_row(reinterpret_cast<MemsetRecord*>(record), intern_name("memset"), 2));
        break;
      default:
        break;
    }
  }
  size_t dropped = 0;
  cuptiActivityGetNumDroppedRecords(nullptr, 0, &dropped);
  g_dropped += static_cast<int64_t>(dropped);
  free(buffer);
}

void CUPTIAPI resource_callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void* cbdata) {
  if (domain != CUPTI_CB_DOMAIN_RESOURCE) return;
  auto* rd = static_cast<const CUpti_ResourceData*>(cbdata);
  auto* gd = static_cast<CUpti_GraphData*>(rd->resourceDescriptor);
  std::lock_guard<std::mutex> lock(g_mu);
  if (cbid == CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED) {
    uint32_t exec_id = 0;
    if (cuptiGetGraphExecId(gd->graphExec, &exec_id) != CUPTI_SUCCESS) {
      ++g_callback_errors;
      return;
    }
    g_execs.emplace_back(exec_id, reinterpret_cast<int64_t>(gd->graph));
  } else if (cbid == CUPTI_CBID_RESOURCE_GRAPH_CLONED) {
    g_clone_source[reinterpret_cast<int64_t>(gd->graph)] = reinterpret_cast<int64_t>(gd->originalGraph);
  }
}

// Call before the CUDA context exists when `hes` is set (HES must precede it).
void init(bool hes, bool track_graphs) {
  // Records are cast to the struct versions of the compile-time headers, so
  // an older runtime CUPTI would be read with the wrong layout.
  uint32_t version = 0;
  check_cupti(cuptiGetVersion(&version), "cuptiGetVersion");
  RuntimeCheck(
      version >= CUPTI_API_VERSION,
      "runtime CUPTI API version ",
      version,
      " is older than the headers this collector was built with (",
      CUPTI_API_VERSION,
      ")");
  if (hes) check_cupti(cuptiActivityEnableHWTrace(1), "cuptiActivityEnableHWTrace");
  if (track_graphs && g_subscriber == nullptr) {
    check_cupti(cuptiSubscribe(&g_subscriber, resource_callback, nullptr), "cuptiSubscribe");
    for (CUpti_CallbackId cbid : {CUPTI_CBID_RESOURCE_GRAPHEXEC_CREATED, CUPTI_CBID_RESOURCE_GRAPH_CLONED}) {
      check_cupti(cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE, cbid), "cuptiEnableCallback");
    }
  }
  if (!g_registered) {
    check_cupti(cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed), "cuptiActivityRegisterCallbacks");
    g_registered = true;
  }
}

void start(bool with_memory) {
  const int n = with_memory ? 3 : 1;
  for (int i = 0; i < n; ++i)
    check_cupti(cuptiActivityEnable(kKinds[i]), "cuptiActivityEnable");
}

void flush() {
  check_cupti(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "cuptiActivityFlushAll");
}

// Disables the kinds, then forces out every buffer so all records are in
// memory. CUPTI stays attached: finalizing breaks later sessions.
void stop() {
  for (auto kind : kKinds)
    cuptiActivityDisable(kind);
  flush();
}

int64_t num_records() {
  std::lock_guard<std::mutex> lock(g_mu);
  return static_cast<int64_t>(g_rows.size());
}

// Moves the oldest rows into `out` ([N, kNumFields] int64 on the host) and
// returns how many were written (at most N).
int64_t take_records(TensorView out) {
  using namespace sglang::host;
  auto N = SymbolicSize{"num_rows"};
  TensorMatcher({N, kNumFields}).with_dtype<int64_t>().with_device<kDLCPU>().verify(out);
  std::lock_guard<std::mutex> lock(g_mu);
  const auto m = std::min<size_t>(static_cast<size_t>(N.unwrap()), g_rows.size());
  auto* dst = static_cast<Row*>(out.data_ptr());
  std::copy(g_rows.begin(), g_rows.begin() + m, dst);
  g_rows.erase(g_rows.begin(), g_rows.begin() + m);
  return static_cast<int64_t>(m);
}

tvm::ffi::Array<tvm::ffi::String> names() {
  std::lock_guard<std::mutex> lock(g_mu);
  tvm::ffi::Array<tvm::ffi::String> out;
  for (const auto& name : g_names)
    out.push_back(name);
  return out;
}

// Flat (exec id, user graph handle) pairs in creation order, clones resolved
// to the graph they were cloned from.
tvm::ffi::Array<int64_t> graph_execs() {
  std::lock_guard<std::mutex> lock(g_mu);
  tvm::ffi::Array<int64_t> out;
  for (const auto& [exec_id, handle] : g_execs) {
    int64_t graph = handle;
    for (auto it = g_clone_source.find(graph); it != g_clone_source.end(); it = g_clone_source.find(graph))
      graph = it->second;
    out.push_back(exec_id);
    out.push_back(graph);
  }
  return out;
}

int64_t dropped() {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_dropped;
}

int64_t callback_errors() {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_callback_errors;
}

int64_t num_fields() {
  return kNumFields;
}

int64_t cupti_version() {
  uint32_t version = 0;
  check_cupti(cuptiGetVersion(&version), "cuptiGetVersion");
  return version;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(init, init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(start, start);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(stop, stop);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(flush, flush);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(num_records, num_records);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(take_records, take_records);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(names, names);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(graph_execs, graph_execs);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dropped, dropped);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(callback_errors, callback_errors);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(num_fields, num_fields);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(cupti_version, cupti_version);

}  // namespace sglang::cupti_collector
