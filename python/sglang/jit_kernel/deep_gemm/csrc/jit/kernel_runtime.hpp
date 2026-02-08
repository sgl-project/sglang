#pragma once

#include "../utils/exception.hpp"
#include "../utils/format.hpp"
#include "../utils/system.hpp"
#include "device_runtime.hpp"
#include "handle.hpp"

namespace deep_gemm {

struct LaunchArgs {
    std::pair<int, int> grid_dim;
    int num_threads;
    int smem_size;
    int cluster_dim;

    LaunchArgs(const int& grid_dim_x, const int& num_threads, const int& smem_size = 0, const int& cluster_dim = 1):
        grid_dim({grid_dim_x, 1}), num_threads(num_threads), smem_size(smem_size), cluster_dim(cluster_dim) {}

    LaunchArgs(const std::pair<int, int>& grid_dim, const int& num_threads, const int& smem_size = 0, const int& cluster_dim = 1):
        grid_dim(grid_dim), num_threads(num_threads), smem_size(smem_size), cluster_dim(cluster_dim) {}
};

class KernelRuntime final {
public:
    static std::filesystem::path cuda_home;

    LibraryHandle library;
    KernelHandle kernel;

    explicit KernelRuntime(const std::filesystem::path& dir_path) {
        // Check `prepare_init`
        DG_HOST_ASSERT(not cuda_home.empty());

        // NOLINT(*-pro-type-member-init)
        const auto& cuobjdump_path = cuda_home / "bin" / "cuobjdump";
        const auto& cubin_path = dir_path / "kernel.cubin";
        if (get_env<int>("DG_JIT_DEBUG"))
            printf("Loading CUBIN: %s\n", cubin_path.c_str());

        // Find the only symbol
        // TODO: use kernel enumeration for newer drivers
        const std::vector<std::string> illegal_names = {"vprintf", "__instantiate_kernel", "__internal", "__assertfail"};
        const auto& [exit_code, symbols] = call_external_command(fmt::format("{} -symbols {}", cuobjdump_path.c_str(), cubin_path.c_str()));
        DG_HOST_ASSERT(exit_code == 0);
        std::istringstream iss(symbols);
        std::vector<std::string> symbol_names;
        for (std::string line; std::getline(iss, line); ) {
            if (line.find("STT_FUNC") == 0 and line.find("STO_ENTRY") != std::string::npos and
                std::none_of(illegal_names.begin(), illegal_names.end(),
                [&](const auto& name) { return line.find(name) != std::string::npos; })) {
                const auto& last_space = line.rfind(' ');
                symbol_names.push_back(line.substr(last_space + 1));
            }
        }
        if (get_env<int>("DG_JIT_DEBUG")) {
            printf("Symbol names: ");
            for (const auto& symbol: symbol_names)
                printf("%s, ", symbol.c_str());
            printf("\n");
        }

        // Load from the library
        DG_HOST_ASSERT(symbol_names.size() == 1);
        kernel = load_kernel(cubin_path, symbol_names[0], &library);
    }

    static void prepare_init(const std::string& cuda_home_path_by_python) {
        cuda_home = cuda_home_path_by_python;
    }

    static bool check_validity(const std::filesystem::path& dir_path) {
        return std::filesystem::exists(dir_path / "kernel.cu") and
               std::filesystem::exists(dir_path / "kernel.cubin");
    }

    ~KernelRuntime() noexcept(false) {
        unload_library(library);
    }
};

DG_DECLARE_STATIC_VAR_IN_CLASS(KernelRuntime, cuda_home);

template <typename Derived>
class LaunchRuntime {
public:
    template <typename Args>
    static std::string generate(const Args& args) {
        const auto& code = Derived::generate_impl(args);
        if (get_env<int>("DG_JIT_DEBUG", 0))
            printf("Generated kernel code: %s\n", code.c_str());
        return code;
    }

    template <typename Args>
    static void launch(const std::shared_ptr<KernelRuntime>& kernel_runtime, const Args& args) {
        const auto& kernel = kernel_runtime->kernel;
        const auto& stream = at::cuda::getCurrentCUDAStream();
        const LaunchArgs& launch_args = args.launch_args;

        const dim3& grid_dim = {static_cast<unsigned>(launch_args.grid_dim.first),
                                static_cast<unsigned>(launch_args.grid_dim.second),
                                1};
        const dim3& block_dim = {static_cast<unsigned>(launch_args.num_threads), 1, 1};
        auto config = construct_launch_config(kernel, stream, launch_args.smem_size,
                                              grid_dim, block_dim, launch_args.cluster_dim);

        // Launch in the derived class
        if (get_env<int>("DG_JIT_DEBUG")) {
            printf("Launch kernel with {%d, %d} x %d, shared memory: %d bytes, cluster: %d, stream: %ld\n",
                   launch_args.grid_dim.first, launch_args.grid_dim.second, launch_args.num_threads,
                   launch_args.smem_size, launch_args.cluster_dim, stream.id());
        }
        Derived::launch_impl(kernel, config, args);
    }
};

} // namespace deep_gemm
