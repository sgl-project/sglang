#pragma once

#include <optional>
#include <string>

namespace deep_gemm {

static std::string get_default_epilogue_type(const std::optional<std::string>& epilogue_type) {
    return epilogue_type.value_or("EpilogueIdentity");
}

} // namespace deep_gemm
