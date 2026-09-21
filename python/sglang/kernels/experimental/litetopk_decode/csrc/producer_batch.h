// Batched decode producer with a compile-time 1024/2048-bin histogram.
// DeepGEMM-derived host/scheduler code: Copyright (c) 2025 DeepSeek, MIT.
#pragma once

#include <torch/custom_class.h>
#include <memory>

#ifndef LITETOPK_PRODUCER_NAMESPACE
#error "LITETOPK_PRODUCER_NAMESPACE must name the isolated Torch namespace"
#endif

namespace LITETOPK_PRODUCER_NAMESPACE {

struct BatchProducerHandle : torch::CustomClassHolder {
  struct Impl;
  std::unique_ptr<Impl> impl;
  BatchProducerHandle(at::Tensor q, at::Tensor cache, at::Tensor weights,
                      int64_t max_seq_len, int64_t rows, int64_t stage, int64_t mode);
  ~BatchProducerHandle() override;
};

c10::intrusive_ptr<BatchProducerHandle> make_handle(
    at::Tensor q, at::Tensor cache, at::Tensor weights,
    int64_t max_seq_len, int64_t rows, int64_t stage, int64_t mode);

// One scorer launch for the WHOLE batch (148 CTAs, independent of B).
// hist is the int32[B,bins] global histogram: zero at entry, nonzero local bins are
// atomically added per row, and the consumer restores it. diag=[148] is CTA-indexed and
// overwritten on every path, including invalid/empty.
void produce(const c10::intrusive_ptr<BatchProducerHandle>& handle,
             const at::Tensor& table, const at::Tensor& lengths,
             const at::Tensor& schedule, const at::Tensor& indices,
             const at::Tensor& dense, const at::Tensor& hist,
             const at::Tensor& diag);

// Same launch, replacing only call-local descriptor addresses (no staging copies).
void produce_rebound(const c10::intrusive_ptr<BatchProducerHandle>& handle,
                     const at::Tensor& q, const at::Tensor& weights,
                     const at::Tensor& table, const at::Tensor& lengths,
                     const at::Tensor& schedule, const at::Tensor& indices,
                     const at::Tensor& dense, const at::Tensor& hist,
                     const at::Tensor& diag);

c10::Dict<std::string, int64_t> producer_info(
    const c10::intrusive_ptr<BatchProducerHandle>& handle);

}  // namespace LITETOPK_PRODUCER_NAMESPACE
