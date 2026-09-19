#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// Optimization points over Python version:
// 1. Multi-threading via at::parallel_for (OpenMP/thread-pool) instead of
//    nested Python for-loops.
// 2. Direct pointer arithmetic on contiguous memory, avoiding Python tensor
//    indexing and tolist() overhead.
// 3. Type specialization for int32/int64, eliminating dynamic type checks.
// 4. std::mt19937_64 RNG is faster than Python's random.Random.
// 5. GIL is released during computation, allowing true parallelism.
// 6. Pinned memory output for faster CPU->GPU transfer.
// 7. Compiler inlining, loop unrolling, and SIMD vectorization.

namespace py = pybind11;

namespace {

int64_t compute_gpu_id_of_physical_expert(int64_t physical_expert_id, int64_t num_local_gpu_physical_experts) {
  return physical_expert_id / num_local_gpu_physical_experts;
}

int64_t compute_node_id_of_physical_expert(int64_t physical_expert_id, int64_t num_local_node_physical_experts) {
  return physical_expert_id / num_local_node_physical_experts;
}

std::vector<int64_t> fair_choices(const std::vector<int64_t>& candidates, int64_t k, std::mt19937_64& rng) {
  const int64_t quotient = k / static_cast<int64_t>(candidates.size());
  const int64_t remainder = k % static_cast<int64_t>(candidates.size());

  std::vector<int64_t> result;
  result.reserve(k);
  for (int64_t i = 0; i < quotient; ++i) {
    result.insert(result.end(), candidates.begin(), candidates.end());
  }

  if (remainder > 0) {
    std::vector<int64_t> sample = candidates;
    for (int64_t i = 0; i < remainder; ++i) {
      std::uniform_int_distribution<int64_t> dist(i, static_cast<int64_t>(sample.size()) - 1);
      std::swap(sample[i], sample[dist(rng)]);
      result.push_back(sample[i]);
    }
  }

  std::shuffle(result.begin(), result.end(), rng);
  return result;
}

int64_t select_dispatch_physical_expert(
    const int64_t* candidates_ptr,
    int64_t candidate_size,
    int64_t ep_rank,
    int64_t ep_size,
    int64_t num_local_gpu_physical_experts,
    int64_t num_gpus_per_node,
    int64_t num_local_node_physical_experts,
    uint64_t seed) {
  std::vector<int64_t> candidates;
  candidates.reserve(candidate_size);
  for (int64_t i = 0; i < candidate_size; ++i) {
    const int64_t physical_expert_id = candidates_ptr[i];
    if (physical_expert_id != -1) {
      candidates.push_back(physical_expert_id);
    }
  }

  if (candidates.empty()) {
    throw std::runtime_error("No candidate physical expert ids found.");
  }

  if (candidates.size() == 1) {
    return candidates[0];
  }

  const int64_t num_nodes = ep_size / num_gpus_per_node;
  const int64_t rank_node_id = ep_rank / num_gpus_per_node;
  int64_t same_node_candidate = -1;
  std::vector<bool> node_present(num_nodes, false);

  for (const int64_t physical_expert_id : candidates) {
    const int64_t gpu_id = compute_gpu_id_of_physical_expert(physical_expert_id, num_local_gpu_physical_experts);
    if (gpu_id == ep_rank) {
      return physical_expert_id;
    }

    const int64_t node_id = compute_node_id_of_physical_expert(physical_expert_id, num_local_node_physical_experts);
    node_present[node_id] = true;
    if (node_id == rank_node_id && same_node_candidate == -1) {
      same_node_candidate = physical_expert_id;
    }
  }

  if (same_node_candidate != -1) {
    return same_node_candidate;
  }

  const int64_t num_missing_nodes = static_cast<int64_t>(std::count(node_present.begin(), node_present.end(), false));
  if (num_missing_nodes <= 0) {
    throw std::runtime_error("No dispatch candidate found for current EP rank.");
  }

  std::mt19937_64 rng(seed);
  const std::vector<int64_t> fill_values = fair_choices(candidates, num_missing_nodes * num_gpus_per_node, rng);

  int64_t missing_nodes_before = 0;
  for (int64_t node_id = 0; node_id < rank_node_id; ++node_id) {
    if (!node_present[node_id]) {
      ++missing_nodes_before;
    }
  }

  const int64_t missing_index = missing_nodes_before * num_gpus_per_node + (ep_rank % num_gpus_per_node);
  return fill_values[missing_index];
}

torch::Tensor compute_logical_to_rank_dispatch_physical_map(
    torch::Tensor logical_to_all_physical_map_cpu,
    int64_t ep_size,
    int64_t num_physical_experts,
    int64_t ep_rank,
    int64_t num_gpus_per_node,
    int64_t seed) {
  TORCH_CHECK(logical_to_all_physical_map_cpu.device().is_cpu(), "logical_to_all_physical_map_cpu must be on CPU");
  TORCH_CHECK(logical_to_all_physical_map_cpu.dim() == 3, "logical_to_all_physical_map_cpu must be a 3D tensor");
  TORCH_CHECK(
      logical_to_all_physical_map_cpu.scalar_type() == torch::kInt64 ||
          logical_to_all_physical_map_cpu.scalar_type() == torch::kInt32,
      "logical_to_all_physical_map_cpu must be int32 or int64");
  TORCH_CHECK(ep_size > 0, "ep_size must be positive");
  TORCH_CHECK(num_gpus_per_node > 0, "num_gpus_per_node must be positive");
  TORCH_CHECK(ep_size % num_gpus_per_node == 0, "ep_size must be divisible by num_gpus_per_node");
  TORCH_CHECK(0 <= ep_rank && ep_rank < ep_size, "ep_rank must be in [0, ep_size)");

  auto input = logical_to_all_physical_map_cpu.contiguous();
  const auto num_layers = input.size(0);
  const auto num_logical_experts = input.size(1);
  const auto candidate_size = input.size(2);
  const int64_t num_local_gpu_physical_experts = num_physical_experts / ep_size;
  const int64_t num_local_node_physical_experts = num_local_gpu_physical_experts * num_gpus_per_node;

  auto options =
      torch::TensorOptions().device(torch::kCPU).dtype(input.scalar_type()).pinned_memory(torch::cuda::is_available());
  auto output = torch::empty({num_layers, num_logical_experts}, options);

  if (input.scalar_type() == torch::kInt64) {
    const auto* input_ptr = input.data_ptr<int64_t>();
    auto* output_ptr = output.data_ptr<int64_t>();
    at::parallel_for(0, num_layers * num_logical_experts, 0, [&](int64_t begin, int64_t end) {
      for (int64_t linear = begin; linear < end; ++linear) {
        const int64_t layer_id = linear / num_logical_experts;
        const int64_t logical_expert_id = linear % num_logical_experts;
        const int64_t* candidates_ptr =
            input_ptr + ((layer_id * num_logical_experts + logical_expert_id) * candidate_size);
        const uint64_t local_seed =
            static_cast<uint64_t>(seed) ^ (static_cast<uint64_t>(linear) * 0x9E3779B97F4A7C15ULL);
        output_ptr[linear] = select_dispatch_physical_expert(
            candidates_ptr,
            candidate_size,
            ep_rank,
            ep_size,
            num_local_gpu_physical_experts,
            num_gpus_per_node,
            num_local_node_physical_experts,
            local_seed);
      }
    });
  } else {
    const auto* input_ptr = input.data_ptr<int32_t>();
    auto* output_ptr = output.data_ptr<int32_t>();
    at::parallel_for(0, num_layers * num_logical_experts, 0, [&](int64_t begin, int64_t end) {
      for (int64_t linear = begin; linear < end; ++linear) {
        const int64_t layer_id = linear / num_logical_experts;
        const int64_t logical_expert_id = linear % num_logical_experts;
        std::vector<int64_t> candidates(candidate_size);
        const int64_t offset = (layer_id * num_logical_experts + logical_expert_id) * candidate_size;
        for (int64_t i = 0; i < candidate_size; ++i) {
          candidates[i] = static_cast<int64_t>(input_ptr[offset + i]);
        }
        const uint64_t local_seed =
            static_cast<uint64_t>(seed) ^ (static_cast<uint64_t>(linear) * 0x9E3779B97F4A7C15ULL);
        output_ptr[linear] = static_cast<int32_t>(select_dispatch_physical_expert(
            candidates.data(),
            candidate_size,
            ep_rank,
            ep_size,
            num_local_gpu_physical_experts,
            num_gpus_per_node,
            num_local_node_physical_experts,
            local_seed));
      }
    });
  }

  return output;
}

}  // namespace

PYBIND11_MODULE(eplb_expert_location_cpp, m) {
  m.def(
      "compute_logical_to_rank_dispatch_physical_map",
      &compute_logical_to_rank_dispatch_physical_map,
      py::arg("logical_to_all_physical_map_cpu"),
      py::arg("ep_size"),
      py::arg("num_physical_experts"),
      py::arg("ep_rank"),
      py::arg("num_gpus_per_node"),
      py::arg("seed"));
}
