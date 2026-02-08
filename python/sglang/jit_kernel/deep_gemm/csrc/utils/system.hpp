#pragma once

#include <array>
#include <filesystem>
#include <functional>
#include <random>
#include <string>
#include <memory>
#include <unistd.h>

#include "exception.hpp"
#include "format.hpp"

namespace deep_gemm {

// ReSharper disable once CppNotAllPathsReturnValue
template <typename dtype_t>
static dtype_t get_env(const std::string& name, const dtype_t& default_value = dtype_t()) {
    const auto& c_str = std::getenv(name.c_str());
    if (c_str == nullptr)
        return default_value;

    // Read the env and convert to the desired type
    if constexpr (std::is_same_v<dtype_t, std::string>) {
        return std::string(c_str);
    } else if constexpr (std::is_same_v<dtype_t, int>) {
        int value;
        std::sscanf(c_str, "%d", &value);
        return value;
    } else {
        DG_HOST_ASSERT(false and "Unexpected type");
    }
}

static std::tuple<int, std::string> call_external_command(std::string command) {
    command = command + " 2>&1";
    const auto& deleter = [](FILE* f) { if (f) pclose(f); };
    std::unique_ptr<FILE, decltype(deleter)> pipe(popen(command.c_str(), "r"), deleter);
    DG_HOST_ASSERT(pipe != nullptr);

    std::array<char, 512> buffer;
    std::string output;
    while (fgets(buffer.data(), buffer.size(), pipe.get()))
        output += buffer.data();
    const auto& exit_code = WEXITSTATUS(pclose(pipe.release()));
    return {exit_code, output};
}

static std::vector<std::filesystem::path> collect_files(const std::filesystem::path& root) {
    std::vector<std::filesystem::path> files;
    std::function<void(const std::filesystem::path&)> impl;
    impl = [&](const std::filesystem::path& dir) {
        for (const auto& entry: std::filesystem::directory_iterator(dir)) {
            if (entry.is_directory()) {
                impl(entry.path());
            } else if (entry.is_regular_file() and entry.path().extension() == ".cuh") {
                files.emplace_back(entry.path());
            }
        }
    };
    impl(root);

    // Be consistent
    std::sort(files.begin(), files.end());
    return files;
}

static std::filesystem::path make_dirs(const std::filesystem::path& path) {
    // OK if existed
    std::error_code capture;
    const bool& created = std::filesystem::create_directories(path, capture);
    if (not (created or capture.value() == 0)) {
        DG_HOST_UNREACHABLE(fmt::format("Failed to make directory: {}, created: {}, value: {}",
                                        path.c_str(), created, capture.value()));
    }
    if (created and get_env<int>("DG_JIT_DEBUG"))
        printf("Create directory: %s\n", path.c_str());
    return path;
}

static std::string get_uuid() {
    static std::random_device rd;
    static std::mt19937 gen([]() {
        return rd() ^ std::chrono::steady_clock::now().time_since_epoch().count();
    }());
    static std::uniform_int_distribution<uint32_t> dist;

    std::stringstream ss;
    ss << getpid() << "-"
       << std::hex << std::setfill('0')
       << std::setw(8) << dist(gen) << "-"
       << std::setw(8) << dist(gen) << "-"
       << std::setw(8) << dist(gen);
    return ss.str();
}

} // deep_gemm
