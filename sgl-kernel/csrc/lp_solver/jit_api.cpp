#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <nvJitLink.h>
#include <nvrtc.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define NVRTC_SAFE_CALL(x)                                                                \
  do {                                                                                    \
    nvrtcResult result = x;                                                               \
    if (result != NVRTC_SUCCESS) {                                                        \
      std::ostringstream oss;                                                             \
      oss << "\nerror: " #x " failed with error " << nvrtcGetErrorString(result) << '\n'; \
      throw std::runtime_error(oss.str());                                                \
    }                                                                                     \
  } while (0)

#define CU_SAFE_CALL(x)                                           \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      std::ostringstream oss;                                     \
      const char* msg;                                            \
      cuGetErrorName(result, &msg);                               \
      oss << "\nerror: " #x " failed with error " << msg << '\n'; \
      throw std::runtime_error(oss.str());                        \
    }                                                             \
  } while (0)

#define CUDA_SAFE_CALL(x)                                                              \
  do {                                                                                 \
    cudaError_t result = x;                                                            \
    if (result != cudaSuccess) {                                                       \
      std::ostringstream oss;                                                          \
      oss << "\nerror: " #x " failed with error " << cudaGetErrorName(result) << '\n'; \
      throw std::runtime_error(oss.str());                                             \
    }                                                                                  \
  } while (0)

#define NVJITLINK_SAFE_CALL(h, x)                                    \
  do {                                                               \
    nvJitLinkResult result = x;                                      \
    if (result != NVJITLINK_SUCCESS) {                               \
      std::ostringstream oss;                                        \
      oss << "\nerror: " #x " failed with error " << result << '\n'; \
      size_t lsize;                                                  \
      result = nvJitLinkGetErrorLogSize(h, &lsize);                  \
      if (result == NVJITLINK_SUCCESS && lsize > 0) {                \
        char* log = (char*)malloc(lsize);                            \
        result = nvJitLinkGetErrorLog(h, log);                       \
        if (result == NVJITLINK_SUCCESS) {                           \
          oss << "error: " << log << '\n';                           \
          free(log);                                                 \
        }                                                            \
      }                                                              \
      throw std::runtime_error(oss.str());                           \
    }                                                                \
  } while (0)

struct compiled_solver {
  CUfunction kernel_ipm;
  CUmodule module;
  int NC;
  int NV;
  int block_dim;
  int smem_size;

  compiled_solver(const std::string& resource_path, int NC, int NV, int block_dim)
      : kernel_ipm(nullptr), module(nullptr), NC(NC), NV(NV), block_dim(block_dim), smem_size(-1) {
    // 1. Get Device Architecture
    CUdevice cuDevice = c10::cuda::current_device();
    int major = 0, minor = 0;
    CU_SAFE_CALL(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    CU_SAFE_CALL(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    int arch = major * 10 + minor;

    // std::cout << "Compiling ipm.cu for sm_" << arch << std::endl;

    // 2. Read Kernel Source
    std::string source_path = resource_path + "/templates/ipm.cu";
    std::ifstream kernel_file(source_path);
    if (!kernel_file.good()) {
      throw std::runtime_error("Cannot find templates/ipm.cu at " + source_path);
    }
    std::string kernel_source((std::istreambuf_iterator<char>(kernel_file)), std::istreambuf_iterator<char>());
    kernel_file.close();

    // 3. Create NVRTC Program
    nvrtcProgram prog;
    NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog, kernel_source.c_str(), "ipm.cu", 0, NULL, NULL));

    // 4. Compile to LTO IR
    std::string mathdx_path = resource_path + "/mathdx/";
    std::string arch_arg = "-arch=sm_" + std::to_string(arch);
    std::string sm_ver_def = "-DSM_Ver=" + std::to_string(arch * 10);
    std::string nc_def = "-DNC=" + std::to_string(NC);
    std::string nv_def = "-DNV=" + std::to_string(NV);
    std::string block_def = "-DBLOCK_DIM=" + std::to_string(block_dim);
    std::string inc_mathdx = "-I" + mathdx_path + "include";
    std::string inc_cublasdx = "-I" + mathdx_path + "include/cublasdx";
    std::string inc_cutlass = "-I" + mathdx_path + "external/cutlass/include";

    std::vector<const char*> opts = {
        "-dlto",
        "--relocatable-device-code=true",
        "-default-device",
        arch_arg.c_str(),
        sm_ver_def.c_str(),
        nc_def.c_str(),
        nv_def.c_str(),
        block_def.c_str(),
        inc_mathdx.c_str(),
        inc_cublasdx.c_str(),
        inc_cutlass.c_str(),
        "-I/usr/local/cuda/include"  // Fallback
    };

    // // Debug: Print compilation options
    // std::cout << "NVRTC Options:" << std::endl;
    // for (const auto& opt : opts) {
    //     std::cout << "  " << opt << std::endl;
    // }

    nvrtcResult compileResult = nvrtcCompileProgram(prog, opts.size(), opts.data());

    // Check Log
    size_t logSize;
    NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
    if (logSize > 1) {
      std::vector<char> log(logSize);
      NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log.data()));
      if (compileResult != NVRTC_SUCCESS) {
        std::cerr << log.data() << std::endl;
        throw std::runtime_error("NVRTC Compilation failed");
      }
    }

    // Get LTO IR
    size_t LTOIRSize;
    NVRTC_SAFE_CALL(nvrtcGetLTOIRSize(prog, &LTOIRSize));
    std::vector<char> LTOIR(LTOIRSize);
    NVRTC_SAFE_CALL(nvrtcGetLTOIR(prog, LTOIR.data()));
    NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

    // 5. Link with nvJitLink
    nvJitLinkHandle handle;
    char smbuf[16];
    sprintf(smbuf, "-arch=sm_%d", arch);
    const char* lopts[] = {"-lto", smbuf};
    NVJITLINK_SAFE_CALL(handle, nvJitLinkCreate(&handle, 2, lopts));

    // Add MathDx Fatbin
    std::string lib_path = mathdx_path + "/lib/libcusolverdx.fatbin";
    NVJITLINK_SAFE_CALL(handle, nvJitLinkAddFile(handle, NVJITLINK_INPUT_FATBIN, lib_path.c_str()));

    // Add our LTO IR
    NVJITLINK_SAFE_CALL(handle, nvJitLinkAddData(handle, NVJITLINK_INPUT_LTOIR, LTOIR.data(), LTOIRSize, "ipm_lto"));

    // Complete Link
    NVJITLINK_SAFE_CALL(handle, nvJitLinkComplete(handle));
    size_t cubinSize;
    NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubinSize(handle, &cubinSize));
    void* cubin = malloc(cubinSize);
    NVJITLINK_SAFE_CALL(handle, nvJitLinkGetLinkedCubin(handle, cubin));
    NVJITLINK_SAFE_CALL(handle, nvJitLinkDestroy(&handle));

    // 6. Load Module
    CU_SAFE_CALL(cuModuleLoadData(&module, cubin));
    free(cubin);
    CU_SAFE_CALL(cuModuleGetFunction(&kernel_ipm, module, "kernel_solve"));

    // 7. Calculate Shared Memory Size
    CUfunction get_smem_size;
    CU_SAFE_CALL(cuModuleGetFunction(&get_smem_size, module, "get_smem_size"));

    int h_smem_size = 0;
    int* d_smem_size;
    CUDA_SAFE_CALL(cudaMalloc(&d_smem_size, sizeof(int)));

    void* args[] = {&d_smem_size};
    CU_SAFE_CALL(cuLaunchKernel(get_smem_size, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
    CUDA_SAFE_CALL(cudaMemcpy(&h_smem_size, d_smem_size, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaFree(d_smem_size));

    this->smem_size = h_smem_size;
    // std::cout << "Smem size: " << smem_size << std::endl;

    CU_SAFE_CALL(cuFuncSetAttribute(kernel_ipm, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_size));
  }

  at::Tensor operator()(at::Tensor A, at::Tensor b, at::Tensor c, at::Tensor avail_num) {
    // Input Validation
    TORCH_CHECK(avail_num.scalar_type() == torch::kInt32, "avail_num must be int32");

    // Pointers
    void* p_avail_num = avail_num.data_ptr();
    void* p_A = A.data_ptr();
    void* p_b = b.data_ptr();
    void* p_c = c.data_ptr();

    // Create Result Tensor
    long result_size = NV;
    at::Tensor result = at::empty({result_size}, A.options());
    void* p_result = result.data_ptr();

    void* args[] = {&p_avail_num, &p_result, &p_A, &p_b, &p_c};

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Cooperative Launch
    // Grid (1,1,1), Block (block_dim, 1, 1)
    CU_SAFE_CALL(cuLaunchCooperativeKernel(kernel_ipm, 1, 1, 1, block_dim, 1, 1, smem_size, stream, args));
    return result;
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<compiled_solver>(m, "CompiledSolver")
      .def(py::init<const std::string&, int, int, int>())
      .def("solve", &compiled_solver::operator(), py::arg("A"), py::arg("b"), py::arg("c"), py::arg("avail_num"));
}
