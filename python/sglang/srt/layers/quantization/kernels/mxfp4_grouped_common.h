#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

struct GroupedDesc {
  const void* X;
  const void* Wq;
  const void* Scales;
  void*       Y;
  int64_t M, N, K;
  int group_size;
  int pack_layout;
  int64_t lda, ldwq, lds, ldy;
};

void launch_grouped_mxfp4_weightonly(
    const std::vector<GroupedDesc>& descs,
    int sm_arch, cudaStream_t stream);