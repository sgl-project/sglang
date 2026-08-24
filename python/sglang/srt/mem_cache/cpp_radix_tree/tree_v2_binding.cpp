#include <ATen/ops/cat.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "tree_v2.h"

namespace {

radix_tree_v2::token_slice token_buffer(const pybind11::buffer& key, const std::optional<std::size_t> key_len) {
  namespace py = pybind11;
  const py::buffer_info info = key.request();
  if (info.ndim != 1 || info.itemsize != sizeof(radix_tree_v2::token_t) || info.strides[0] != info.itemsize) {
    throw std::invalid_argument("radix key must be a contiguous one-dimensional int64 buffer");
  }
  const auto available = static_cast<std::size_t>(info.shape[0]);
  const auto length = key_len.value_or(available);
  if (length > available) {
    throw std::invalid_argument("radix key length exceeds its backing buffer");
  }
  return {static_cast<const radix_tree_v2::token_t*>(info.ptr), length};
}

std::optional<at::Tensor> flatten_chunks(std::vector<at::Tensor> chunks) {
  if (chunks.empty()) return std::nullopt;
  if (chunks.size() == 1) return std::move(chunks.front());
  return at::cat(chunks);
}

}  // namespace

PYBIND11_MODULE(radix_tree_cpp, m) {
  using namespace radix_tree_v2;
  namespace py = pybind11;
  py::class_<RadixTree>(m, "RadixTree")
      .def(
          py::init<bool, std::optional<std::size_t>, std::size_t, std::size_t, std::size_t>(),
          py::arg("disabled"),
          py::arg("host_size"),
          py::arg("page_size"),
          py::arg("write_through_threshold"),
          py::arg("sliding_window_size") = 0)
      .def(
          "match_prefix",
          [](RadixTree& tree, const py::buffer& key, std::optional<std::size_t> key_len) {
            return tree.match_prefix(token_buffer(key, key_len));
          },
          py::arg("key"),
          py::arg("key_len") = py::none())
      .def(
          "match_prefix_flat",
          [](RadixTree& tree, const py::buffer& key, std::optional<std::size_t> key_len) {
            auto [chunks, host_hit_length, device_node, host_node] = tree.match_prefix(token_buffer(key, key_len));
            return std::make_tuple(flatten_chunks(std::move(chunks)), host_hit_length, device_node, host_node);
          },
          py::arg("key"),
          py::arg("key_len") = py::none())
      .def(
          "match_prefix_swa",
          [](RadixTree& tree, const py::buffer& key, std::optional<std::size_t> key_len) {
            return tree.match_prefix_swa(token_buffer(key, key_len));
          },
          py::arg("key"),
          py::arg("key_len") = py::none())
      .def(
          "match_prefix_swa_flat",
          [](RadixTree& tree, const py::buffer& key, std::optional<std::size_t> key_len) {
            auto [chunks, device_node, full_hit_length] = tree.match_prefix_swa(token_buffer(key, key_len));
            return std::make_tuple(flatten_chunks(std::move(chunks)), device_node, full_hit_length);
          },
          py::arg("key"),
          py::arg("key_len") = py::none())
      .def("evict", &RadixTree::evict)
      .def("lock_ref", &RadixTree::lock_ref)
      .def("lock_ref_swa", &RadixTree::lock_ref_swa)
      .def("unlock_ref_swa", &RadixTree::unlock_ref_swa)
      .def("unlock_swa_only", &RadixTree::unlock_swa_only)
      .def("evictable_size", &RadixTree::evictable_size)
      .def("protected_size", &RadixTree::protected_size)
      .def("total_size", &RadixTree::total_size)
      .def("component_evictable_size", &RadixTree::component_evictable_size)
      .def("component_protected_size", &RadixTree::component_protected_size)
      .def("all_values", &RadixTree::all_values)
      .def("debug_stats", &RadixTree::debug_stats)
      .def(
          "writing_through",
          [](RadixTree& tree, const py::buffer& key, at::Tensor value, std::optional<std::size_t> key_len) {
            return tree.writing_through(token_buffer(key, key_len), std::move(value));
          },
          py::arg("key"),
          py::arg("value"),
          py::arg("key_len") = py::none())
      .def(
          "writing_through_swa",
          [](RadixTree& tree,
             const py::buffer& key,
             at::Tensor value,
             std::size_t prev_prefix_len,
             std::size_t swa_evicted_seqlen,
             std::optional<std::size_t> key_len) {
            return tree.writing_through_swa(
                token_buffer(key, key_len), std::move(value), prev_prefix_len, swa_evicted_seqlen);
          },
          py::arg("key"),
          py::arg("value"),
          py::arg("prev_prefix_len"),
          py::arg("swa_evicted_seqlen"),
          py::arg("key_len") = py::none())
      .def("match_node_and_lock", &RadixTree::match_node_and_lock)
      .def(
          "match_node_and_lock_flat",
          [](RadixTree& tree, NodeHandle node_id, bool lock_full, bool lock_swa) {
            auto [chunks, best_node, full_hit_length, swa_uuid, skipped] =
                tree.match_node_and_lock(node_id, lock_full, lock_swa);
            return std::make_tuple(
                flatten_chunks(std::move(chunks)), best_node, full_hit_length, swa_uuid, std::move(skipped));
          },
          py::arg("node_id"),
          py::arg("lock_full"),
          py::arg("lock_swa"))
      .def(
          "match_node_range_and_lock_flat",
          [](RadixTree& tree,
             NodeHandle node_id,
             std::size_t range_start,
             std::size_t range_end,
             bool lock_full,
             bool lock_swa) {
            auto [chunks, best_node, best_prefix_length, full_hit_length, swa_uuid, skipped] =
                tree.match_node_range_and_lock(node_id, range_start, range_end, lock_full, lock_swa);
            return std::make_tuple(
                flatten_chunks(std::move(chunks)),
                best_node,
                best_prefix_length,
                full_hit_length,
                swa_uuid,
                std::move(skipped));
          },
          py::arg("node_id"),
          py::arg("range_start"),
          py::arg("range_end"),
          py::arg("lock_full"),
          py::arg("lock_swa"))
      .def("set_swa_value", &RadixTree::set_swa_value)
      .def("get_swa_value", &RadixTree::get_swa_value)
      .def("evict_component", &RadixTree::evict_component)
      .def("loading_onboard", &RadixTree::loading_onboard)
      .def("commit_writing_through", &RadixTree::commit_writing_through)
      .def("commit_loading_onboard", &RadixTree::commit_loading_onboard)
      .def("reset", &RadixTree::reset)
      .def("debug_print", &RadixTree::debug_print);
}
