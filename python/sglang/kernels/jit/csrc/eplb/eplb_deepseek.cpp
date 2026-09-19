#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

torch::Tensor inverse_perm(torch::Tensor perm) {
  auto input = perm.contiguous();
  auto inv = torch::empty_like(input);
  inv.scatter_(1, input, torch::arange(input.size(1), input.options()).expand(input.sizes()));
  return inv;
}

std::tuple<torch::Tensor, torch::Tensor> balanced_packing_cpu(torch::Tensor weight, int64_t num_packs) {
  TORCH_CHECK(weight.device().is_cpu(), "weight must be on CPU");
  TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");
  TORCH_CHECK(num_packs > 0, "num_packs must be positive");

  auto weight_f = weight.to(torch::kFloat32).contiguous();
  const auto num_layers = weight_f.size(0);
  const auto num_groups = weight_f.size(1);
  TORCH_CHECK(num_groups % num_packs == 0, "num_groups must be divisible by num_packs");
  const auto groups_per_pack = num_groups / num_packs;

  if (groups_per_pack == 1) {
    auto pack_index = torch::arange(num_groups, torch::TensorOptions().dtype(torch::kInt64)).expand(weight_f.sizes());
    auto rank_in_pack = torch::zeros_like(pack_index);
    return {pack_index, rank_in_pack};
  }

  auto indices = std::get<1>(weight_f.sort(-1, /*descending=*/true)).contiguous();
  auto pack_index =
      torch::full({num_layers, num_groups}, -1, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto rank_in_pack = torch::full_like(pack_index, -1);

  const auto* weight_ptr = weight_f.data_ptr<float>();
  const auto* indices_ptr = indices.data_ptr<int64_t>();
  auto* pack_index_ptr = pack_index.data_ptr<int64_t>();
  auto* rank_in_pack_ptr = rank_in_pack.data_ptr<int64_t>();

  at::parallel_for(0, num_layers, 0, [&](int64_t begin, int64_t end) {
    std::vector<float> pack_weights(num_packs);
    std::vector<int64_t> pack_items(num_packs);
    for (int64_t layer = begin; layer < end; ++layer) {
      std::fill(pack_weights.begin(), pack_weights.end(), 0.0f);
      std::fill(pack_items.begin(), pack_items.end(), 0);
      const auto row_offset = layer * num_groups;
      for (int64_t pos = 0; pos < num_groups; ++pos) {
        const int64_t group = indices_ptr[row_offset + pos];
        int64_t best_pack = -1;
        float best_weight = 0.0f;
        for (int64_t pack = 0; pack < num_packs; ++pack) {
          if (pack_items[pack] >= groups_per_pack) {
            continue;
          }
          if (best_pack == -1 || pack_weights[pack] < best_weight) {
            best_pack = pack;
            best_weight = pack_weights[pack];
          }
        }
        TORCH_CHECK(best_pack != -1, "No available pack found");
        pack_index_ptr[row_offset + group] = best_pack;
        rank_in_pack_ptr[row_offset + group] = pack_items[best_pack];
        pack_weights[best_pack] += weight_ptr[row_offset + group];
        pack_items[best_pack] += 1;
      }
    }
  });

  return {pack_index, rank_in_pack};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> replicate_experts_cpu(torch::Tensor weight, int64_t num_phy) {
  TORCH_CHECK(weight.device().is_cpu(), "weight must be on CPU");
  TORCH_CHECK(weight.dim() == 2, "weight must be a 2D tensor");

  auto weight_f = weight.to(torch::kFloat32).contiguous();
  const auto num_layers = weight_f.size(0);
  const auto num_log = weight_f.size(1);
  const auto num_redundant = num_phy - num_log;
  TORCH_CHECK(num_redundant >= 0, "num_phy must be >= num_logical_experts");

  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto phy2log = torch::arange(num_phy, options).repeat({num_layers, 1});
  auto rank = torch::zeros({num_layers, num_phy}, options);
  auto logcnt = torch::ones({num_layers, num_log}, options);

  const auto* weight_ptr = weight_f.data_ptr<float>();
  auto* phy2log_ptr = phy2log.data_ptr<int64_t>();
  auto* rank_ptr = rank.data_ptr<int64_t>();
  auto* logcnt_ptr = logcnt.data_ptr<int64_t>();

  at::parallel_for(0, num_layers, 0, [&](int64_t begin, int64_t end) {
    for (int64_t layer = begin; layer < end; ++layer) {
      const auto weight_offset = layer * num_log;
      const auto phy_offset = layer * num_phy;
      const auto logcnt_offset = layer * num_log;
      for (int64_t phy = num_log; phy < num_phy; ++phy) {
        int64_t best_log = 0;
        float best_score = weight_ptr[weight_offset] / static_cast<float>(logcnt_ptr[logcnt_offset]);
        for (int64_t log = 1; log < num_log; ++log) {
          const float score = weight_ptr[weight_offset + log] / static_cast<float>(logcnt_ptr[logcnt_offset + log]);
          if (score > best_score) {
            best_log = log;
            best_score = score;
          }
        }
        phy2log_ptr[phy_offset + phy] = best_log;
        rank_ptr[phy_offset + phy] = logcnt_ptr[logcnt_offset + best_log];
        logcnt_ptr[logcnt_offset + best_log] += 1;
      }
    }
  });

  return {phy2log, rank, logcnt};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rebalance_experts_hierarchical_cpu(
    torch::Tensor weight, int64_t num_physical_experts, int64_t num_groups, int64_t num_nodes, int64_t num_gpus) {
  auto weight_f = weight.to(torch::kFloat32).cpu().contiguous();
  const auto num_layers = weight_f.size(0);
  const auto num_logical_experts = weight_f.size(1);

  TORCH_CHECK(num_logical_experts % num_groups == 0, "num_logical_experts must be divisible by num_groups");
  const auto group_size = num_logical_experts / num_groups;
  TORCH_CHECK(num_groups % num_nodes == 0, "num_groups must be divisible by num_nodes");
  const auto groups_per_node = num_groups / num_nodes;
  TORCH_CHECK(num_gpus % num_nodes == 0, "num_gpus must be divisible by num_nodes");
  TORCH_CHECK(num_physical_experts % num_gpus == 0, "num_physical_experts must be divisible by num_gpus");
  const auto phy_experts_per_gpu = num_physical_experts / num_gpus;

  auto tokens_per_group = weight_f.view({num_layers, num_groups, group_size}).sum(-1);
  auto [group_pack_index, group_rank_in_pack] = balanced_packing_cpu(tokens_per_group, num_nodes);

  auto group_offsets = torch::arange(group_size, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
                           .view({1, 1, group_size});
  auto log2mlog =
      (((group_pack_index * groups_per_node) + group_rank_in_pack) * group_size).unsqueeze(-1) + group_offsets;
  log2mlog = log2mlog.view({num_layers, num_logical_experts});
  auto mlog2log = inverse_perm(log2mlog);

  auto tokens_per_mlog = weight_f.gather(-1, mlog2log).view({-1, num_logical_experts / num_nodes});
  auto [phy2mlog, phyrank, mlogcnt] = replicate_experts_cpu(tokens_per_mlog, num_physical_experts / num_nodes);

  auto tokens_per_phy = (tokens_per_mlog / mlogcnt.to(torch::kFloat32)).gather(-1, phy2mlog);
  auto [pack_index, rank_in_pack] = balanced_packing_cpu(tokens_per_phy, num_gpus / num_nodes);
  auto phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack;
  auto pphy2phy = inverse_perm(phy2pphy);

  auto pphy2mlog = phy2mlog.gather(-1, pphy2phy);
  auto node_offsets = torch::arange(
                          0,
                          num_logical_experts,
                          num_logical_experts / num_nodes,
                          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
                          .view({1, num_nodes, 1});
  pphy2mlog = (pphy2mlog.view({num_layers, num_nodes, -1}) + node_offsets).flatten(-2);
  auto pphy2log = mlog2log.gather(-1, pphy2mlog);
  auto pphyrank = phyrank.gather(-1, pphy2phy).view({num_layers, -1});
  auto logcnt = mlogcnt.view({num_layers, -1}).gather(-1, log2mlog);
  return {pphy2log, pphyrank, logcnt};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rebalance_experts_cpu(
    torch::Tensor weight,
    int64_t num_replicas,
    int64_t num_groups,
    int64_t num_nodes,
    int64_t num_gpus,
    bool enable_hierarchical) {
  auto weight_f = weight.to(torch::kFloat32).cpu().contiguous();
  const auto num_layers = weight_f.size(0);
  const auto num_logical_experts = weight_f.size(1);

  torch::Tensor phy2log;
  torch::Tensor phyrank;
  torch::Tensor logcnt;
  if (enable_hierarchical) {
    std::tie(phy2log, phyrank, logcnt) =
        rebalance_experts_hierarchical_cpu(weight_f, num_replicas, num_groups, num_nodes, num_gpus);
  } else {
    std::tie(phy2log, phyrank, logcnt) = rebalance_experts_hierarchical_cpu(weight_f, num_replicas, 1, 1, num_gpus);
  }

  const auto maxlogcnt = logcnt.max().item<int64_t>();
  auto log2phy = torch::full(
      {num_layers, num_logical_experts, maxlogcnt},
      -1,
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  log2phy.view({num_layers, -1})
      .scatter_(
          -1,
          phy2log * maxlogcnt + phyrank,
          torch::arange(num_replicas, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
              .expand({num_layers, num_replicas}));
  return {phy2log, log2phy, logcnt};
}

}  // namespace

PYBIND11_MODULE(eplb_deepseek_cpp, m) {
  m.def(
      "rebalance_experts",
      &rebalance_experts_cpu,
      py::arg("weight"),
      py::arg("num_replicas"),
      py::arg("num_groups"),
      py::arg("num_nodes"),
      py::arg("num_gpus"),
      py::arg("enable_hierarchical"),
      py::call_guard<py::gil_scoped_release>());
}
