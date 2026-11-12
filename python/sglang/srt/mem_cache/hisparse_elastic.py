"""HS_ELASTIC: one HiSparse configuration for every load. The device memory that a static configuration reserves up
front for the indexer keys of tokens that might be admitted (index-K of every indexer layer, sized for the logical pool,
i.e. host_to_device_ratio x device pool) is lent to the device KV pool until those tokens arrive.

Layout per rank, all in CUDA VMM ranges whose base addresses never change (CUDA graphs and kernels keep their pointers):
  KV   78 layers x (K_max + page) rows of 656 B: rows [0, K_min + page) permanently mapped (the static pool), rows above
       that (the band) mapped in 2 MiB chunks while lent
  IK   the indexer layers x L_pages pages of 8,448 B: pages mapped in 2 MiB chunks as logical pages come into use
The band plus the mapped index-K never exceed the static index-K reservation E, so the device memory in use is the
static configuration's. Admission keeps the static view (the band is invisible to it): anything admitted fits the static
layout after demoting resident requests, so the band can always be taken back.
"""

import functools
import logging

import torch

logger = logging.getLogger(__name__)

_CPP = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <map>
#include <vector>
#define HS_CK(x) do { CUresult hs_ck_res_ = (x); if (hs_ck_res_ != CUDA_SUCCESS) { const char* s = nullptr; \
  cuGetErrorString(hs_ck_res_, &s); throw std::runtime_error(std::string(#x) + ": " + (s ? s : "?")); } } while (0)
static std::map<int64_t, CUmemGenericAllocationHandle> g_handles;   // mapped address -> handle
static std::map<int64_t, int64_t> g_sizes;                          // mapped address -> bytes
// Physical chunks are never released while the process runs: an unmapped chunk goes to this pool and is mapped again
// wherever the next chunk is needed (KV band or index-K). Released memory would go back to the driver, where the
// PyTorch caching allocator of the forward thread can take it before the next map (cuMemCreate out of memory).
static std::map<int64_t, std::vector<CUmemGenericAllocationHandle>> g_pool;  // bytes -> free handles
static int64_t g_creates = 0;  // cuMemCreate calls so far (v3: none after start-up)
int64_t granularity(int64_t dev) {
  CUmemAllocationProp p = {}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  p.location.id = (int)dev; size_t g = 0;
  HS_CK(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)); return (int64_t)g;
}
int64_t reserve(int64_t bytes, int64_t align) {
  CUdeviceptr base = 0; HS_CK(cuMemAddressReserve(&base, (size_t)bytes, (size_t)align, 0, 0)); return (int64_t)base;
}
// map [ptr, ptr + bytes) with one physical allocation (bytes a multiple of the granularity)
void map_range(int64_t dev, int64_t ptr, int64_t bytes) {
  CUmemAllocationProp p = {}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  p.location.id = (int)dev;
  CUmemGenericAllocationHandle h;
  auto& fl = g_pool[bytes];
  if (!fl.empty()) { h = fl.back(); fl.pop_back(); }
  else { HS_CK(cuMemCreate(&h, (size_t)bytes, &p, 0)); g_creates++; }
  HS_CK(cuMemMap((CUdeviceptr)ptr, (size_t)bytes, 0, h, 0));
  CUmemAccessDesc a = {}; a.location.type = CU_MEM_LOCATION_TYPE_DEVICE; a.location.id = (int)dev;
  a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  HS_CK(cuMemSetAccess((CUdeviceptr)ptr, (size_t)bytes, &a, 1));
  g_handles[ptr] = h; g_sizes[ptr] = bytes;
}
void unmap_range(int64_t ptr) {
  auto it = g_handles.find(ptr);
  if (it == g_handles.end()) throw std::runtime_error("unmap_range: not mapped");
  HS_CK(cuMemUnmap((CUdeviceptr)ptr, (size_t)g_sizes[ptr]));
  g_pool[g_sizes[ptr]].push_back(it->second);  // kept for the next map, never released
  g_handles.erase(it); g_sizes.erase(ptr);
}
int64_t pooled(int64_t bytes) { auto it = g_pool.find(bytes); return it == g_pool.end() ? 0 : (int64_t)it->second.size(); }
// v3: create n physical chunks of `bytes` now and keep them in the pool (mapped later without cuMemCreate)
void reserve_pool(int64_t dev, int64_t bytes, int64_t n) {
  CUmemAllocationProp p = {}; p.type = CU_MEM_ALLOCATION_TYPE_PINNED; p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  p.location.id = (int)dev;
  for (int64_t i = 0; i < n; i++) { CUmemGenericAllocationHandle h; HS_CK(cuMemCreate(&h, (size_t)bytes, &p, 0));
    g_creates++; g_pool[bytes].push_back(h); }
}
int64_t creates() { return g_creates; }
torch::Tensor byte_view(int64_t dev, int64_t ptr, int64_t nbytes) {
  auto d = torch::Device(torch::kCUDA, (int)dev);
  auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(d);
  return at::for_blob((void*)ptr, {nbytes}).options(opts).target_device(d).deleter([](void*) {}).make_tensor();
}
"""


@functools.lru_cache(maxsize=1)
def _ext():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        "hs_elastic_v3",
        _CPP,
        functions=[
            "granularity",
            "reserve",
            "map_range",
            "unmap_range",
            "byte_view",
            "pooled",
            "reserve_pool",
            "creates",
        ],
        extra_include_paths=["/usr/local/cuda/include"],
        extra_ldflags=["-L/usr/local/cuda/lib64/stubs", "-lcuda"],
        verbose=False,
    )


class ElasticFamily:
    """`layers` per-layer regions of `max_bytes` each in one VA reservation. Bytes [0, perm) of every layer are mapped
    at once and never unmapped; above that, 2 MiB chunks are mapped / unmapped together in every layer."""

    def __init__(self, device, layers: int, max_bytes: int, perm_bytes: int, name: str):
        dev = torch.device(device)
        self.dev = dev.index if dev.index is not None else torch.cuda.current_device()
        self.x = _ext()
        self.g = int(self.x.granularity(self.dev))
        self.layers, self.name = layers, name
        self.stride = -(-max_bytes // self.g) * self.g
        self.base = int(self.x.reserve(self.stride * max(1, layers), self.g))
        self.perm = -(-perm_bytes // self.g) * self.g if perm_bytes > 0 else 0
        for l in range(layers):
            if self.perm:
                self.x.map_range(self.dev, self.base + l * self.stride, self.perm)
        self.mapped = self.perm  # bytes mapped per layer (a multiple of g)

    def view(self, layer: int, nbytes: int) -> torch.Tensor:
        assert nbytes <= self.stride
        return self.x.byte_view(self.dev, self.base + layer * self.stride, nbytes)

    def handles(self) -> int:
        """Physical chunks mapped above the permanent region (the part counted against the budget)."""
        return (self.mapped - self.perm) // self.g * self.layers

    def chunk_bytes(self) -> int:
        return self.g * self.layers

    def grow_to(self, nbytes: int) -> int:
        """Map chunks so that at least [0, nbytes) of every layer is backed; returns chunks added per layer."""
        target = min(self.stride, -(-nbytes // self.g) * self.g)
        added = 0
        while self.mapped < target:
            for l in range(self.layers):
                self.x.map_range(
                    self.dev, self.base + l * self.stride + self.mapped, self.g
                )
            self.mapped += self.g
            added += 1
        return added

    def shrink_to(self, nbytes: int) -> int:
        """Unmap whole chunks above max(nbytes, perm) (rounded up); caller guarantees nothing there is in use and the
        device is idle on that memory. Returns chunks removed per layer."""
        target = max(self.perm, -(-nbytes // self.g) * self.g)
        removed = 0
        while self.mapped > target:
            self.mapped -= self.g
            for l in range(self.layers):
                self.x.unmap_range(self.base + l * self.stride + self.mapped)
            removed += 1
        return removed
