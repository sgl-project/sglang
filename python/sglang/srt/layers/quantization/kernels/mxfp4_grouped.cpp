// mxfp4_grouped.cpp
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "mxfp4_grouped_common.h"
#include <vector>
#include <stdexcept>

static int to_layout(const std::string& s) {
  if (s == "mxfp4_row") return 0;
  if (s == "mxfp4_col") return 1;
  throw std::runtime_error("Unsupported pack_layout: " + s);
}

std::vector<at::Tensor> grouped_forward(
    std::vector<at::Tensor> X_list,
    std::vector<at::Tensor> Wq_list,
    std::vector<at::Tensor> S_list,
    int group_size,
    const std::string& pack_layout,
    int sm) {

  TORCH_CHECK(X_list.size() == Wq_list.size() && X_list.size() == S_list.size(),
              "Input lists must have the same length");

  if (X_list.empty()) return {};

  const int layout = to_layout(pack_layout);
  std::vector<GroupedDesc> descs;
  descs.reserve(X_list.size());
  std::vector<at::Tensor> Y_list;
  Y_list.reserve(X_list.size());

  for (size_t i = 0; i < X_list.size(); ++i) {
    auto X = X_list[i];
    auto Wq = Wq_list[i];
    auto S = S_list[i];

    TORCH_CHECK(X.is_cuda() && Wq.is_cuda() && S.is_cuda(), "All tensors must be on CUDA");
    TORCH_CHECK(X.dtype() == at::kBFloat16, "X must be BF16");
    TORCH_CHECK(S.dtype() == at::kBFloat16 || S.dtype() == at::kHalf, "Scales must be BF16/FP16");
    TORCH_CHECK(X.dim() == 2, "X must be 2D [M,K]");
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");

    const auto M = X.size(0);
    const auto K = X.size(1);

    // Expect Wq packed row-major by output rows; N derived from metadata or shape.
    TORCH_CHECK(Wq.dim() >= 2, "Wq must be at least 2D packed tensor");
    TORCH_CHECK(Wq.dtype() == at::kByte || Wq.dtype() == at::kUInt8, "Wq must be uint8 packed FP4");
    TORCH_CHECK(Wq.is_contiguous(), "Wq must be contiguous [K_packed, N]");
    TORCH_CHECK(S.dim() >= 1, "Scales must be 1D+ per-group");
    TORCH_CHECK(S.is_contiguous(), "Scales must be contiguous");
    // For MXFP4, Wq is [K_packed, N] where K_packed = K // 2
    const bool wq_is_kpacked_by_n = (Wq.dim() >= 2); // true in your pack
    const auto Kp = Wq.size(0);               // K_packed
    const auto N  = Wq.size(1);               // output channels
    TORCH_CHECK(Kp * 2 >= K, "Packed K must cover logical K"); // 2 FP4 per byte

    // Y allocation
    auto Y = at::empty({M, N}, X.options()); // BF16

    GroupedDesc d{};
    d.X = X.data_ptr();
    d.Wq = Wq.data_ptr();
    d.Scales = S.data_ptr();
    d.Y = Y.data_ptr();
    d.M = M; d.N = N; d.K = K;
    d.group_size = group_size;
    d.pack_layout = layout;
    // Leading dims (row-major):
    // A: [M,K]   lda = K
    // Wq: [Kp,N] ldwq = N   (advance by columns across contiguous N)
    // Y: [M,N]   ldy = N
    d.lda  = K;       // A: [M,K] row-major
    d.ldwq = N;       // Wq: [Kp,N] → leading dim is N (not stride(0))
    d.lds = S.numel() > 0 ? S.stride(0) : 0;
    d.ldy  = N;

    descs.push_back(d);
    Y_list.push_back(Y);
  }

  auto stream = at::cuda::getCurrentCUDAStream();
  launch_grouped_mxfp4_weightonly(descs, sm, stream.stream());

  return Y_list;
}

PYBIND11_MODULE(_mxfp4_kernels, m) {
  m.def("grouped_forward", &grouped_forward,
        "Grouped MXFP4 weight-only GEMM forward",
        py::arg("X_list"), py::arg("Wq_list"), py::arg("S_list"),
        py::arg("group_size"), py::arg("pack_layout"), py::arg("sm"));
}