#pragma once

#include "../utils/layout.hpp"
#include "../utils/compatibility.hpp"

#if DG_TENSORMAP_COMPATIBLE
#include "../jit_kernels/impls/smxx_layout.hpp"
#endif

namespace deep_gemm::layout {

#if DG_TENSORMAP_COMPATIBLE
static torch::Tensor transform_sf_into_required_layout(const torch::Tensor& sf,
                                                       const int& mn, const int& k,
                                                       const std::optional<std::tuple<int, int, int>>& recipe,
                                                       const std::optional<std::tuple<int, int>>& recipe_ab,
                                                       const std::optional<int>& num_groups,
                                                       const bool& is_sfa,
                                                       const bool& disable_ue8m0_cast) {
    const auto& arch_major = device_runtime->get_arch_major();

    int gran_mn, gran_k;
    if (recipe.has_value()) {
        DG_HOST_ASSERT(not recipe_ab.has_value());
        gran_mn = is_sfa ? std::get<0>(recipe.value()) : std::get<1>(recipe.value());
        gran_k = std::get<2>(recipe.value());
    } else {
        DG_HOST_ASSERT(recipe_ab.has_value());
        std::tie(gran_mn, gran_k) = recipe_ab.value();
    }

    // Pre-transform checks
    check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups);

    // (FP32, 1, 128) on SM90: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and gran_mn == 1 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return get_mn_major_tma_aligned_tensor(sf);

    // (FP32, 128, 128) on SM90: no need to transform, check SFB requirements
    if (sf.scalar_type() == torch::kFloat and gran_mn == 128 and gran_k == 128 and (arch_major == 9 or disable_ue8m0_cast))
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, false, true, torch::kFloat);

    // (FP32, x, gran_k) on SM100: transform to (INT, 1, gran_k), TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kFloat and (gran_k == 32 or gran_k == 128) and arch_major == 10) {
        DG_HOST_ASSERT(not disable_ue8m0_cast);
        const auto& broadcasted = gran_mn == 1 ? sf :
                                  sf.index_select(-2, torch::arange(mn, at::TensorOptions().device(sf.device())).floor_divide_(gran_mn));
        return get_mn_major_tma_aligned_packed_ue8m0_tensor(broadcasted);
    }

    // (INT, 1, gran_k) on SM100: transform to TMA-aligned and MN-major
    if (sf.scalar_type() == torch::kInt and gran_mn == 1 and (gran_k == 32 or gran_k == 128) and arch_major == 10)
        return check_sf_layout(sf, mn, k, gran_mn, gran_k, num_groups, true, false, torch::kInt);

    DG_HOST_UNREACHABLE("Unknown SF transformation");
}

static std::tuple<torch::Tensor, torch::Tensor, int, int> transform_sf_pair_into_required_layout(
        const torch::Tensor& sfa, const torch::Tensor& sfb,
        const int& m, const int& n, const int& k,
        std::optional<std::tuple<int, int, int>>& recipe,
        const std::optional<std::tuple<int, int>>& recipe_a,
        const std::optional<std::tuple<int, int>>& recipe_b,
        const std::optional<int>& num_groups_a,
        const std::optional<int>& num_groups_b,
        const bool& disable_ue8m0_cast = false) {
    DG_HOST_ASSERT(recipe_a.has_value() == recipe_b.has_value());
    if (not recipe_a.has_value() and not recipe.has_value())
        recipe = get_default_recipe(sfa.scalar_type(), sfb.scalar_type());
    const auto transformed_sfa = transform_sf_into_required_layout(sfa, m, k, recipe, recipe_a, num_groups_a, true, disable_ue8m0_cast);
    const auto transformed_sfb = transform_sf_into_required_layout(sfb, n, k, recipe, recipe_b, num_groups_b, false, disable_ue8m0_cast);
    const int gran_k_a = recipe_a.has_value() ? std::get<1>(recipe_a.value()) : std::get<2>(recipe.value());
    const int gran_k_b = recipe_b.has_value() ? std::get<1>(recipe_b.value()) : std::get<2>(recipe.value());
    return std::make_tuple(transformed_sfa, transformed_sfb, gran_k_a, gran_k_b);
}

static torch::Tensor transform_k_grouped_sf_into_required_layout(const torch::Tensor& sf,
                                                                 const std::vector<int>& ks,
                                                                 const torch::Tensor& ks_tensor,
                                                                 const std::tuple<int, int, int>& recipe) {
    DG_HOST_ASSERT(sf.dim() == 2);
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128));
    const auto& arch_major = device_runtime->get_arch_major();

    // FP32 on SM90
    if (sf.scalar_type() == torch::kFloat and arch_major == 9)
        return get_mn_major_tma_aligned_tensor(sf);

    // FP32 on SM100
    if (sf.scalar_type() == torch::kFloat and arch_major == 10)
        return get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(sf, ks_tensor, ks);

    // INT on SM100
    if (sf.scalar_type() == torch::kInt and arch_major == 10)
        DG_HOST_UNREACHABLE("Unimplemented");

    DG_HOST_UNREACHABLE("Unknown cases");
}

#endif

static void register_apis(pybind11::module_& m) {

#if DG_TENSORMAP_COMPATIBLE
    m.def("transform_sf_into_required_layout", &transform_sf_into_required_layout,
      py::arg("sf"), py::arg("mn"), py::arg("k"),
      py::arg("recipe") = std::nullopt, py::arg("recipe_ab") = std::nullopt,
      py::arg("num_groups") = std::nullopt, py::arg("is_sfa") = false,
      py::arg("disable_ue8m0_cast") = false);

    m.def("get_tma_aligned_size", &get_tma_aligned_size);
    m.def("get_mn_major_tma_aligned_tensor", &get_mn_major_tma_aligned_tensor);
    m.def("get_mn_major_tma_aligned_packed_ue8m0_tensor", &get_mn_major_tma_aligned_packed_ue8m0_tensor);
    m.def("get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor", &get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor);
#endif

    m.def("get_mk_alignment_for_contiguous_layout", &get_mk_alignment_for_contiguous_layout);
}

} // namespace deep_gemm::layout
