/* Copyright 2026 SGLang Team. All Rights Reserved. */

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <limits>
#include <mutex>
#include <vector>

#include "dwdp/hsa_copy.h"

namespace {

std::once_flag hsa_init_once;
hsa_status_t hsa_init_status = HSA_STATUS_ERROR_NOT_INITIALIZED;

void ensure_hsa_initialized() {
  std::call_once(hsa_init_once, [] { hsa_init_status = hsa_init(); });
  TORCH_CHECK(hsa_init_status == HSA_STATUS_SUCCESS, "hsa_init failed: ", static_cast<int>(hsa_init_status));
}

void check_hsa(hsa_status_t status, const char* operation) {
  if (status == HSA_STATUS_SUCCESS) {
    return;
  }
  const char* detail = nullptr;
  hsa_status_string(status, &detail);
  TORCH_CHECK(
      false,
      operation,
      " failed with HSA status ",
      static_cast<int>(status),
      ": ",
      detail == nullptr ? "unknown" : detail);
}

hsa_signal_t signal_from_ticket(int64_t ticket) {
  TORCH_CHECK(ticket > 0, "HSA copy ticket must be positive, got ", ticket);
  return hsa_signal_t{static_cast<uint64_t>(ticket)};
}

hsa_status_t collect_gpu_agents(hsa_agent_t agent, void* data) {
  hsa_device_type_t type{};
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS) {
    return HSA_STATUS_SUCCESS;
  }
  if (type == HSA_DEVICE_TYPE_GPU) {
    static_cast<std::vector<hsa_agent_t>*>(data)->push_back(agent);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_agent_t agent_for_hip_device(int64_t device_id) {
  TORCH_CHECK(device_id >= 0 && device_id <= std::numeric_limits<int>::max(), "invalid HIP device ", device_id);
  char pci_bus_id[32] = {};
  const hipError_t hip_status = hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), static_cast<int>(device_id));
  TORCH_CHECK(hip_status == hipSuccess, "hipDeviceGetPCIBusId failed: ", hipGetErrorString(hip_status));

  unsigned int domain = 0;
  unsigned int bus = 0;
  unsigned int device = 0;
  unsigned int function = 0;
  TORCH_CHECK(
      std::sscanf(pci_bus_id, "%x:%x:%x.%x", &domain, &bus, &device, &function) == 4,
      "cannot parse HIP PCI bus id ",
      pci_bus_id);
  const uint32_t expected_bdf = (bus << 8) | (device << 3) | function;

  std::vector<hsa_agent_t> agents;
  check_hsa(hsa_iterate_agents(collect_gpu_agents, &agents), "hsa_iterate_agents");
  for (const hsa_agent_t agent : agents) {
    uint32_t bdf = 0;
    if (hsa_agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID), &bdf) ==
            HSA_STATUS_SUCCESS &&
        bdf == expected_bdf) {
      return agent;
    }
  }
  TORCH_CHECK(false, "no HSA GPU agent matches HIP device ", device_id, " (", pci_bus_id, ")");
  return hsa_agent_t{0};
}

hsa_amd_sdma_engine_id_t select_engine(hsa_agent_t destination, hsa_agent_t source) {
  uint32_t available = 0;
  check_hsa(
      hsa_amd_memory_copy_engine_status(destination, source, &available),
      "hsa_amd_memory_copy_engine_status");
  TORCH_CHECK(available != 0, "no HSA SDMA engine is available for the GPU pair");

  uint32_t preferred = 0;
  const hsa_status_t preferred_status =
      hsa_amd_memory_get_preferred_copy_engine(destination, source, &preferred);
  uint32_t candidates =
      preferred_status == HSA_STATUS_SUCCESS ? (available & preferred) : 0;
  if (candidates == 0) {
    candidates = available;
  }
  const uint32_t selected = candidates & (~candidates + 1);
  TORCH_CHECK(selected != 0, "failed to select an HSA SDMA engine");
  return static_cast<hsa_amd_sdma_engine_id_t>(selected);
}

}  // namespace

bool dwdp_hsa_copy_is_available() {
  try {
    ensure_hsa_initialized();
    return true;
  } catch (...) {
    return false;
  }
}

int64_t dwdp_hsa_copy_engine_for_devices(int64_t destination_device, int64_t source_device) {
  ensure_hsa_initialized();
  return static_cast<int64_t>(
      select_engine(agent_for_hip_device(destination_device), agent_for_hip_device(source_device)));
}

int64_t dwdp_hsa_copy_async(
    const at::Tensor& destination,
    const at::Tensor& source,
    int64_t destination_device,
    int64_t source_device) {
  ensure_hsa_initialized();
  TORCH_CHECK(destination.is_cuda() && source.is_cuda(), "HSA DWDP copy requires GPU tensors");
  TORCH_CHECK(destination.is_contiguous() && source.is_contiguous(), "HSA DWDP copy requires contiguous tensors");
  TORCH_CHECK(
      destination.numel() * destination.element_size() == source.numel() * source.element_size(),
      "HSA DWDP copy byte-size mismatch");

  hsa_signal_t completion{};
  check_hsa(hsa_signal_create(1, 0, nullptr, &completion), "hsa_signal_create");
  const hsa_agent_t destination_agent = agent_for_hip_device(destination_device);
  const hsa_agent_t source_agent = agent_for_hip_device(source_device);
  const auto engine = select_engine(destination_agent, source_agent);
  const size_t bytes = static_cast<size_t>(destination.numel()) * destination.element_size();

  const hsa_status_t status = hsa_amd_memory_async_copy_on_engine(
      destination.data_ptr(),
      destination_agent,
      source.data_ptr(),
      source_agent,
      bytes,
      0,
      nullptr,
      completion,
      engine,
      true);
  if (status != HSA_STATUS_SUCCESS) {
    hsa_signal_destroy(completion);
    check_hsa(status, "hsa_amd_memory_async_copy_on_engine");
  }
  TORCH_CHECK(
      completion.handle <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
      "HSA signal handle exceeds int64");
  return static_cast<int64_t>(completion.handle);
}

int64_t dwdp_hsa_copy_wait(int64_t ticket) {
  ensure_hsa_initialized();
  const hsa_signal_value_t value = hsa_signal_wait_scacquire(
      signal_from_ticket(ticket),
      HSA_SIGNAL_CONDITION_LT,
      1,
      std::numeric_limits<uint64_t>::max(),
      HSA_WAIT_STATE_BLOCKED);
  return static_cast<int64_t>(value);
}

void dwdp_hsa_copy_destroy(int64_t ticket) {
  ensure_hsa_initialized();
  check_hsa(hsa_signal_destroy(signal_from_ticket(ticket)), "hsa_signal_destroy");
}
