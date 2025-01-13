#pragma once
#include <torch/extension.h>
#include "../../../3rdparty/nlohmann/json.hpp"
#include <fstream>
#include <sstream>
#include <filesystem>
// 在头文件中定义
#ifdef SGL_DEBUG_BUILD
constexpr int MAX_CONFIG_ID = 6;  // debug模式下的最大配置ID
#else
constexpr int MAX_CONFIG_ID = 90; // 正常模式下的最大配置ID
#endif

using json = nlohmann::json;

struct cuda_error : public std::runtime_error {
  /**
   * @brief Constructs a `cuda_error` object with the given `message`.
   *
   * @param message The error char array used to construct `cuda_error`
   */
  cuda_error(const char* message) : std::runtime_error(message) {}
  /**
   * @brief Constructs a `cuda_error` object with the given `message` string.
   *
   * @param message The `std::string` used to construct `cuda_error`
   */
  cuda_error(std::string const& message) : cuda_error{message.c_str()} {}
};

#define CHECK_CUDA_SUCCESS(cmd)                                         \
  do {                                                                  \
    cudaError_t e = cmd;                                                \
    if (e != cudaSuccess) {                                             \
      std::stringstream _message;                                       \
      auto s = cudaGetErrorString(e);                                   \
      _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
      throw cuda_error(_message.str());                                 \
    }                                                                   \
  } while (0)

#define CHECK_IS_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_IS_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) \
  CHECK_IS_CUDA(x);         \
  CHECK_IS_CONTIGUOUS(x)

inline int getSMVersion() {
  int device{-1};
  CHECK_CUDA_SUCCESS(cudaGetDevice(&device));
  int sm_major = 0;
  int sm_minor = 0;
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device));
  CHECK_CUDA_SUCCESS(cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device));
  return sm_major * 10 + sm_minor;
}

inline uint32_t next_pow_2(uint32_t const num) {
  if (num <= 1) return num;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

inline json read_json_config(const std::string& config_path) {
    std::ifstream f(config_path);
    if (!f.is_open()) {
        std::stringstream ss;
        ss << "Failed to open config file: " << config_path;
        throw std::runtime_error(ss.str());
    }
    json config;
    try {
        config = json::parse(f);
    } catch (const json::parse_error& e) {
        std::stringstream ss;
        ss << "Failed to parse config file: " << config_path << "\n"
           << "Error: " << e.what();
        throw std::runtime_error(ss.str());
    }
    return config;
}

inline json get_gemm_config(const std::string& config_path, int m, int n) {
    auto config = read_json_config(config_path);
    
    json* best_config = nullptr;
    int min_diff = std::numeric_limits<int>::max();
    
    for (auto& cfg : config["configs"]) {
        int cfg_m = cfg["m_range"][0];
        int cfg_m_end = cfg["m_range"][1];
        int cfg_n = cfg["n_range"][0];
        int cfg_n_end = cfg["n_range"][1];
        
        if (m >= cfg_m && m <= cfg_m_end && n >= cfg_n && n <= cfg_n_end) {
            int diff = std::abs(m - cfg_m) + std::abs(n - cfg_n);
            if (diff < min_diff) {
                min_diff = diff;
                best_config = &cfg;
            }
        }
    }
    
    if (!best_config) {
        throw std::runtime_error("No matching configuration found for m=" + 
                               std::to_string(m) + ", n=" + std::to_string(n));
    }
    
    return *best_config;
}

inline std::string get_config_path(int64_t N, int64_t K, const torch::Dtype& dtype) {
    static int device = -1;
    static std::string cached_device_name;
    
    // 只在第一次调用时获取设备信息
    if (device == -1) {
        CHECK_CUDA_SUCCESS(cudaGetDevice(&device));
        cudaDeviceProp prop;
        CHECK_CUDA_SUCCESS(cudaGetDeviceProperties(&prop, device));
        cached_device_name = prop.name;
        std::replace(cached_device_name.begin(), cached_device_name.end(), ' ', '_');
    }
    
    std::string dtype_str = (dtype == torch::kBFloat16) ? "bfloat16" : "float16";
    
    return "N=" + std::to_string(N) + 
           ",K=" + std::to_string(K) + 
           ",device=" + cached_device_name + 
           ",dtype=" + dtype_str + ".json";
}

// 添加一个辅助函数来找到最近的配置
inline int find_nearest_m(const json& config, int current_m) {
    int nearest_m = -1;
    int min_diff = std::numeric_limits<int>::max();
    
    for (auto& el : config.items()) {
        if (el.key().substr(0, 2) == "M=") {
            int m = std::stoi(el.key().substr(2));
            if (m <= current_m && (current_m - m) < min_diff) {
                min_diff = current_m - m;
                nearest_m = m;
            }
        }
    }
    
    return nearest_m;
}