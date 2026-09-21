// Batched derivative of decode_topk_final/producer/producer.h (which is unchanged and
// stays the qualified B=1 artifact). Only the row extent differs: every operand carries
// B rows and the global histogram is int32[B,2048].
// DeepGEMM-derived host/scheduler code: Copyright (c) 2025 DeepSeek, MIT.
#pragma once

#include <torch/custom_class.h>
#include <memory>

namespace litetopk_batched_producer_20260920 {

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
// hist is the int32[B,2048] global histogram: zero at entry, nonzero local bins are
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

}  // namespace litetopk_batched_producer_20260920
