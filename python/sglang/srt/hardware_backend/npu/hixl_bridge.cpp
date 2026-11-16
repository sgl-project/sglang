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
unsigned sgl_hixl_read(void *engine, const char *peer, const uint64_t *triples,
                       size_t count, int timeout_ms) noexcept {
  return checked([&]() -> unsigned {
    std::vector<TransferOpDesc> ops;
    ops.reserve(count);
    for (size_t i = 0; i < count; ++i)
      ops.push_back({triples[3 * i], triples[3 * i + 1], triples[3 * i + 2]});
    TransferReq request = nullptr;
    auto *hixl = static_cast<Hixl *>(engine);
    auto status =
        hixl->TransferAsync(AscendString(peer), READ, ops, {}, request);
    if (status != SUCCESS)
      return status;
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    for (;;) {
      TransferStatus state;
      status = hixl->GetTransferStatus(request, state);
      if (status != SUCCESS)
        return status;
      if (state == TransferStatus::COMPLETED)
        return SUCCESS;
      if (state != TransferStatus::WAITING)
        return FAILED;
      if (std::chrono::steady_clock::now() >= deadline)
        return TIMEOUT;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
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
