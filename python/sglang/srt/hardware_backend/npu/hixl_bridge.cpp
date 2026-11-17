// Copyright 2026 SGLang Team. Licensed under Apache-2.0.
// A C ABI bridge compiled against the installed CANN headers. No C++ object
// layouts or mangled symbols are reproduced in Python.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <hixl/hixl.h>
#include <memory>
#include <thread>
#include <vector>

namespace {
using namespace hixl;
template <class F> unsigned checked(F fn) noexcept {
  try {
    return fn();
  } catch (...) {
    return FAILED;
  }
}
} // namespace

extern "C" {
unsigned sgl_hixl_create(const char *endpoint, void **output) noexcept {
  return checked([&]() -> unsigned {
    auto engine = std::make_unique<Hixl>();
    auto status = engine->Initialize(AscendString(endpoint), {});
    if (status == SUCCESS)
      *output = engine.release();
    return status;
  });
}
unsigned sgl_hixl_register(void *engine, uint64_t address, uint64_t bytes,
                           void **handle) noexcept {
  return checked([&]() {
    return static_cast<Hixl *>(engine)->RegisterMem(MemDesc{address, bytes, {}},
                                                    MEM_DEVICE, *handle);
  });
}
unsigned sgl_hixl_connect(void *engine, const char *peer,
                          int timeout_ms) noexcept {
  return checked([&]() {
    return static_cast<Hixl *>(engine)->Connect(AscendString(peer), timeout_ms);
  });
}
unsigned sgl_hixl_read(void *engine, const char *const *peers,
                       const uint64_t *const *triples, const size_t *counts,
                       size_t peer_count, int timeout_ms) noexcept {
  return checked([&]() -> unsigned {
    auto *hixl = static_cast<Hixl *>(engine);
    std::vector<std::vector<TransferOpDesc>> operations(peer_count);
    std::vector<TransferReq> requests(peer_count, nullptr);
    std::vector<bool> completed(peer_count, false);
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    // Submit every peer before waiting, so independent READs can progress.
    for (size_t peer = 0; peer < peer_count; ++peer) {
      auto &ops = operations[peer];
      ops.reserve(counts[peer]);
      for (size_t i = 0; i < counts[peer]; ++i)
        ops.push_back({triples[peer][3 * i], triples[peer][3 * i + 1],
                       triples[peer][3 * i + 2]});
      auto status = hixl->TransferAsync(AscendString(peers[peer]), READ, ops,
                                        {}, requests[peer]);
      if (status != SUCCESS)
        return status;
    }
    size_t remaining = peer_count;
    while (remaining != 0) {
      for (size_t peer = 0; peer < peer_count; ++peer) {
        if (completed[peer])
          continue;
        TransferStatus state;
        auto status = hixl->GetTransferStatus(requests[peer], state);
        if (status != SUCCESS)
          return status;
        if (state == TransferStatus::COMPLETED) {
          // Terminal status consumes the backend handle. Never query it again.
          completed[peer] = true;
          --remaining;
        } else if (state != TransferStatus::WAITING) {
          return FAILED;
        }
      }
      if (remaining == 0)
        return SUCCESS;
      if (std::chrono::steady_clock::now() >= deadline)
        return TIMEOUT;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    return SUCCESS;
  });
}
unsigned sgl_hixl_disconnect(void *engine, const char *peer,
                             int timeout_ms) noexcept {
  return checked([&]() {
    return static_cast<Hixl *>(engine)->Disconnect(AscendString(peer),
                                                   timeout_ms);
  });
}
unsigned sgl_hixl_deregister(void *engine, void *handle) noexcept {
  return checked(
      [&]() { return static_cast<Hixl *>(engine)->DeregisterMem(handle); });
}
unsigned sgl_hixl_destroy(void *engine) noexcept {
  return checked([&]() -> unsigned {
    auto *hixl = static_cast<Hixl *>(engine);
    hixl->Finalize();
    delete hixl;
    return SUCCESS;
  });
}
}
