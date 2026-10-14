#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <climits>
// Qualified single-request planner/gather. Inactive tiered routines omitted.
namespace pair_swap_gather {
namespace cg = cooperative_groups;
constexpr int kPlanThreads = 256;

__global__ __launch_bounds__(kPlanThreads, 1) void cooperative_plan_kernel(
    const int64_t *__restrict__ hot, int *__restrict__ hot_epoch,
    int *__restrict__ permutation, int *__restrict__ swap_a,
    int *__restrict__ swap_b, int *__restrict__ counts, int hot_size,
    int window_start, int common_end, int epoch) {
  cg::grid_group grid = cg::this_grid();
  const int i = static_cast<int>(blockIdx.x) * kPlanThreads + threadIdx.x;
  const int lane = threadIdx.x & 31;

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    counts[1] = 0;
    counts[2] = 0;
  }
  const int old_count = max(0, min(counts[0], hot_size));
  if (i < old_count) {
    const int a = swap_a[i];
    const int b = swap_b[i];
    permutation[a] = a;
    permutation[b] = b;
  }
  if (i < hot_size) {
    const int64_t value = static_cast<int64_t>(hot[i]);
    if (value >= window_start && value < common_end) {
      const int previous =
          atomicExch(hot_epoch + static_cast<int>(value), epoch);
      if (previous == epoch) {
        atomicOr(counts + 3, 1);
      }
    } else {
      atomicOr(counts + 3, 2);
    }
  }
  grid.sync();

  const int64_t hot_value64 = static_cast<int64_t>(hot[i]);
  const bool hot_in_range =
      hot_value64 >= window_start && hot_value64 < common_end;
  const int hot_value =
      hot_in_range ? static_cast<int>(hot_value64) : window_start;
  const int window_value = window_start + i;
  const bool take_a = hot_in_range && hot_value >= window_start + hot_size;
  const bool take_b = hot_epoch[window_value] != epoch;

  const unsigned a_mask = __ballot_sync(0xffffffffu, take_a);
  int a_base = 0;
  if (lane == 0 && a_mask != 0) {
    a_base = atomicAdd(counts + 1, __popc(a_mask));
  }
  a_base = __shfl_sync(0xffffffffu, a_base, 0);
  if (take_a) {
    const int rank = __popc(a_mask & ((1u << lane) - 1u));
    swap_a[a_base + rank] = hot_value;
  }

  const unsigned b_mask = __ballot_sync(0xffffffffu, take_b);
  int b_base = 0;
  if (lane == 0 && b_mask != 0) {
    b_base = atomicAdd(counts + 2, __popc(b_mask));
  }
  b_base = __shfl_sync(0xffffffffu, b_base, 0);
  if (take_b) {
    const int rank = __popc(b_mask & ((1u << lane) - 1u));
    swap_b[b_base + rank] = window_value;
  }
  grid.sync();

  const int pair_count = min(counts[1], counts[2]);
  const int metadata_error = counts[3];
  if (metadata_error != 0 || counts[1] != counts[2]) {
    if (i == 0 && counts[1] != counts[2]) {
      atomicOr(counts + 3, 8);
    }
    asm volatile("trap;");
    return;
  }
  if (i < pair_count) {
    const int a = swap_a[i];
    const int b = swap_b[i];
    permutation[a] = b;
    permutation[b] = a;
  }
  if (i == 0) {
    counts[0] = pair_count;
  }
}





void launch_plan(const torch::Tensor &hot, const torch::Tensor &hot_epoch,
                 const torch::Tensor &permutation, const torch::Tensor &swap_a,
                 const torch::Tensor &swap_b, const torch::Tensor &counts,
                 int window_start, int common_end, int epoch,
                 cudaStream_t stream) {
  int hot_size = static_cast<int>(hot.numel());
  const int blocks = (hot_size + kPlanThreads - 1) / kPlanThreads;
  const int64_t *hot_ptr = hot.data_ptr<int64_t>();
  int *hot_epoch_ptr = hot_epoch.data_ptr<int>();
  int *permutation_ptr = permutation.data_ptr<int>();
  int *swap_a_ptr = swap_a.data_ptr<int>();
  int *swap_b_ptr = swap_b.data_ptr<int>();
  int *counts_ptr = counts.data_ptr<int>();
  void *args[] = {
      &hot_ptr,    &hot_epoch_ptr, &permutation_ptr, &swap_a_ptr, &swap_b_ptr,
      &counts_ptr, &hot_size,      &window_start,    &common_end, &epoch,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<const void *>(cooperative_plan_kernel), dim3(blocks),
      dim3(kPlanThreads), args, 0, stream));
}

// One private transaction publishes the new permutation before its gather.
// Public borrowed-input checks and fixed owner storage establish its layout.
template<auto Gather, typename... Fields>
void plan_and_gather(
    const torch::Tensor &hot, const torch::Tensor &hot_epoch,
    const torch::Tensor &permutation, const torch::Tensor &swap_a,
    const torch::Tensor &swap_b, const torch::Tensor &counts,
    int64_t window_start, int64_t common_end, int64_t epoch,
    const torch::Tensor &cache, const torch::Tensor &dst_k,
    const torch::Tensor &dst_scale, const torch::Tensor &table,
    const Fields &... fields) {
  // Private Few entry. valid_paged_inputs checked the borrowed64x132 byte
  // cache and I32 page table, including16B cache alignment and active width.
  // Active table entries name physical pages provided by SGLang allocation.
  // HOT/I32 epoch-permutation-swap state and D128/rho outputs belong to the
  // serial owner; the caller keeps every tensor alive through current-stream completion.
  const c10::cuda::CUDAGuard guard(hot.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  launch_plan(hot, hot_epoch, permutation, swap_a, swap_b, counts,
              static_cast<int>(window_start), static_cast<int>(common_end),
              static_cast<int>(epoch), stream);
  Gather(cache, dst_k, dst_scale, table, permutation, fields..., stream);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
// Same raw-byte RMS; no scale is applied to its value. Invalid scale joins
// the current-key bad flag, so a finite rho also certifies 0<KS<=2^64.
// Included inside each caller's existing namespace to preserve native symbols.
// Finite E4 magnitudes are magnitude_unit(code)/512; sign is irrelevant.
// The floating estimate only proposes a code. Integer checks certify the bound.
__device__ __forceinline__ uint32_t magnitude_unit(uint32_t code) {
  const uint32_t exponent=code>>3, mantissa=code&7u;
  return exponent==0?mantissa:(8u+mantissa)<<(exponent-1);
}

__device__ __forceinline__ uint32_t max4_magnitude(
    uint32_t packed,uint32_t& invalid) {
  uint32_t result=0;
  #pragma unroll
  for(int byte=0;byte<4;++byte){
    const uint32_t code=(packed>>(8*byte))&0x7fu;
    invalid|=code==0x7fu;
    result=max(result,code);
  }
  return result;
}

__device__ __forceinline__ unsigned long long sum4_square_units(uint32_t packed) {
  unsigned long long result=0;
  #pragma unroll
  for(int byte=0;byte<4;++byte){
    const uint32_t code=(packed>>(8*byte))&0x7fu;
    if(code!=0x7fu){
      const unsigned long long unit=magnitude_unit(code);
      result+=unit*unit;
    }
  }
  return result;
}

__device__ __forceinline__ uint32_t exact_rms_code(
    unsigned long long sum_square,uint32_t kappa,int dimensions) {
  // Floating point proposes a code; exact current/previous tests prove minimality.
  const float estimate=sqrtf(float(sum_square)/float(dimensions))*(1.0f/512.0f);
  uint32_t candidate=uint32_t(
      __nv_cvt_float_to_fp8(estimate,__NV_SATFINITE,__NV_E4M3));
  unsigned long long unit=magnitude_unit(candidate);
  if(static_cast<unsigned long long>(dimensions)*unit*unit<sum_square)++candidate;
  unit=magnitude_unit(candidate);
  const unsigned long long previous=magnitude_unit(candidate?candidate-1:0);
  if(candidate<=kappa
      &&static_cast<unsigned long long>(dimensions)*unit*unit>=sum_square
      &&(candidate==0
          ||static_cast<unsigned long long>(dimensions)*previous*previous<sum_square))
    return candidate;
  // An inaccurate proposal retains the original integer search below.
  uint32_t lo=0,hi=kappa;
  #pragma unroll
  for(int iteration=0;iteration<7;++iteration){
    const uint32_t mid=(lo+hi)>>1;
    const unsigned long long unit=magnitude_unit(mid);
    const bool enough=static_cast<unsigned long long>(dimensions)*unit*unit>=sum_square;
    if(lo<hi){
      if(enough)hi=mid;
      else lo=mid+1;
    }
  }
  return lo;
}

__global__ void paged_gather_rho_kernel(
    const char* __restrict__ cache, char* __restrict__ dst_k,
    char* __restrict__ dst_scale, const int* __restrict__ table,
    const int* __restrict__ permutation, int64_t block_stride,
    int page_size, int n, int* __restrict__ rho) {
  const int row=blockIdx.x*blockDim.y+threadIdx.y, x=threadIdx.x;
  const bool valid=row<n;
  int src=x==0 && valid?permutation[row]:-1;
  src=__shfl_sync(0xffffffffu,src,0,8);
  unsigned long long square=0;
  uint32_t bad=0,kappa=0;
  if(valid){
    if(src<0 || src>=n){asm volatile("trap;");return;}
    const int block=table[src/page_size];
    const int64_t base=int64_t(block)*block_stride;
    const int within=src%page_size;
    const uint4 words=*reinterpret_cast<const uint4*>(cache+base+within*128+x*16);
    *reinterpret_cast<uint4*>(dst_k+int64_t(row)*128+x*16)=words;
    const uint32_t v[4]={words.x,words.y,words.z,words.w};
#pragma unroll
    for(int i=0;i<4;++i){
      kappa=max(kappa,max4_magnitude(v[i],bad));
      square+=sum4_square_units(v[i]);
    }
    if(x==0){
      const uint32_t scale_bits=*reinterpret_cast<const uint32_t*>(cache+base+page_size*128+within*4);
      *reinterpret_cast<uint32_t*>(dst_scale+int64_t(row)*4)=scale_bits;
      // Positive F32 bit order: accepts subnormal through2^64 inclusive.
      bad|=!((scale_bits-1u)<0x5f800000u);
    }
  }
#pragma unroll
  for(int delta=4;delta;delta>>=1){
    square+=__shfl_down_sync(0xffffffffu,square,delta,8);
    kappa=max(kappa,__shfl_down_sync(0xffffffffu,kappa,delta,8));
    bad|=__shfl_down_sync(0xffffffffu,bad,delta,8);
  }
  if(x==0 && valid)rho[row]=bad?127:exact_rms_code(square,kappa,128);
}

void launch_gather_rho(const torch::Tensor& cache,const torch::Tensor& k,
    const torch::Tensor& scale,const torch::Tensor& table,
    const torch::Tensor& permutation,const torch::Tensor& rho,cudaStream_t stream){
  const int n=k.size(0);if(!n)return;
  paged_gather_rho_kernel<<<(n+31)/32,dim3(8,32),0,stream>>>(
    reinterpret_cast<const char*>(cache.data_ptr<uint8_t>()),
    reinterpret_cast<char*>(k.data_ptr<uint8_t>()),
    reinterpret_cast<char*>(scale.data_ptr<uint8_t>()),table.data_ptr<int>(),
    permutation.data_ptr<int>(),cache.stride(0),cache.size(1),n,rho.data_ptr<int>());
}

inline constexpr auto plan_and_permuted_paged_gather_rho_out =
    plan_and_gather<launch_gather_rho, torch::Tensor>;

} // namespace pair_swap_gather

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace {
constexpr int kCarryTileItems = 8192;
constexpr int kCarryMaxItems = 1 << 20;
constexpr int kCarryMaxBlocks =
    (kCarryMaxItems + kCarryTileItems - 1) / kCarryTileItems;
constexpr int kCarryThreads = 256;
constexpr int kCarryWarps = kCarryThreads / 32;

enum CarryStateOffset : int {
  kCarryTicket = 0,
  kCarryThreshold = 1,
  kCarryTieBlock = 2,
  kCarryTieTake = 3,
  kCarryOutK = 4,
  kCarryNumBlocks = 5,
  kCarryBlockOffsets = 6,
};

__device__ __forceinline__ int carry_warp_sum(int value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}
// Same input scan, publication and state ABI. Only partial representation and
// the final-CTA threshold/offset calculation differ from the original plan.
__global__ void carry_votes_plan_prefix_kernel(
    const int32_t* __restrict__ votes,int count,int min_index,int out_k,
    int max_vote,volatile int16_t* __restrict__ partial,int partial_stride,
    int32_t* __restrict__ state) {
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,block=blockIdx.x;
  const int begin=block*kCarryTileItems,end=min(begin+kCarryTileItems,count);
  const int bins=max_vote+1;
  extern __shared__ uint32_t freq[];
  __shared__ int warp_sum[kCarryWarps],scan_base,last,threshold;
  __shared__ int block_gt[kCarryMaxBlocks],block_eq[kCarryMaxBlocks];
  for(int b=tid;b<bins;b+=kCarryThreads)freq[b]=0;
  __syncthreads();
  int zeros=0;
  for(int index=begin+tid;index<end;index+=kCarryThreads) {
    if(index<min_index)continue;
    int value=votes[index];value=value<0?0:(value>max_vote?max_vote:value);
    if(value==0)++zeros;else atomicAdd(freq+value,1u);
  }
  zeros=carry_warp_sum(zeros);
  if(lane==0)warp_sum[warp]=zeros;
  __syncthreads();
  if(warp==0) {
    int value=lane<kCarryWarps?warp_sum[lane]:0;value=carry_warp_sum(value);
    if(lane==0)freq[0]=static_cast<uint32_t>(value);
  }
  if(tid==0)scan_base=0;
  __syncthreads();
  volatile int16_t* row=partial+static_cast<size_t>(block)*partial_stride;
  // P_b(v) is at most8192, even though it is a cumulative rather than raw bin.
  for(int tile=0;tile<bins;tile+=kCarryThreads) {
    const int bin=max_vote-tile-tid;
    int inclusive=bin>=0?static_cast<int>(freq[bin]):0;
    #pragma unroll
    for(int delta=1;delta<32;delta<<=1) {
      int other=__shfl_up_sync(0xffffffffu,inclusive,delta);
      if(lane>=delta)inclusive+=other;
    }
    if(lane==31)warp_sum[warp]=inclusive;
    __syncthreads();
    int before=0;
    #pragma unroll
    for(int w=0;w<kCarryWarps;++w)if(w<warp)before+=warp_sum[w];
    if(bin>=0)row[bin]=static_cast<int16_t>(scan_base+before+inclusive);
    __syncthreads();
    if(tid==0) {
      int total=0;
      #pragma unroll
      for(int w=0;w<kCarryWarps;++w)total+=warp_sum[w];
      scan_base+=total;
    }
    __syncthreads();
  }
  // Preserve the original every-writer release before the last-block ticket.
  __threadfence();
  __syncthreads();
  if(tid==0)last=atomicAdd(state+kCarryTicket,1)==gridDim.x-1;
  __syncthreads();
  if(!last)return;

  if(warp==0) {
    int lo=0,hi=bins; // S(lo)>=K, S(hi)=0<K (virtual high endpoint).
    while(hi-lo>1) {
      int mid=(lo+hi)/2,total=0;
      for(int b=lane;b<gridDim.x;b+=32)
        total+=static_cast<int>(partial[static_cast<size_t>(b)*partial_stride+mid]);
      total=carry_warp_sum(total);total=__shfl_sync(0xffffffffu,total,0);
      if(total>=out_k)lo=mid;else hi=mid;
    }
    if(lane==0)threshold=lo;
  }
  __syncthreads();
  if(tid<gridDim.x) {
    const size_t base=static_cast<size_t>(tid)*partial_stride;
    int ge=static_cast<int>(partial[base+threshold]);
    int gt=threshold<max_vote?static_cast<int>(partial[base+threshold+1]):0;
    block_gt[tid]=gt;block_eq[tid]=ge-gt;
  }
  __syncthreads();
  if(tid==0) {
    int gt=0;
    for(int b=0;b<gridDim.x;++b)gt+=block_gt[b];
    int need=out_k-gt,remaining=need,offset=0,tie_block=gridDim.x-1,tie_take=0;
    for(int b=0;b<gridDim.x;++b) {
      state[kCarryBlockOffsets+b]=offset;
      if(remaining>0&&remaining<=block_eq[b]) {tie_block=b;tie_take=remaining;}
      const int take=min(block_eq[b],max(remaining,0));
      offset+=block_gt[b]+take;remaining-=block_eq[b];
    }
    state[kCarryBlockOffsets+gridDim.x]=offset;
    state[kCarryThreshold]=threshold;state[kCarryTieBlock]=tie_block;
    state[kCarryTieTake]=tie_take;state[kCarryOutK]=out_k;state[kCarryNumBlocks]=gridDim.x;
    __threadfence();atomicExch(state+kCarryTicket,0);
  }
}
__global__ void carry_votes_emit_warp_kernel(
    int32_t *__restrict__ votes, int count, int min_index, int max_vote,
    int64_t *__restrict__ out_idx, const int32_t *__restrict__ state) {
  constexpr int kSteps = kCarryTileItems / kCarryThreads;
  constexpr int kWarpItems = 32 * kSteps;
  constexpr unsigned kFullMask = 0xffffffffu;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int block = blockIdx.x;
  const int begin = block * kCarryTileItems + warp * kWarpItems;
  const int threshold = state[kCarryThreshold];
  const int tie_block = state[kCarryTieBlock];
  const int tie_take = state[kCarryTieTake];
  const int output_base = state[kCarryBlockOffsets + block];
  const unsigned lane_bit = 1u << lane;
  const unsigned lane_mask = lane_bit - 1u;

  __shared__ unsigned s_greater[kCarryWarps][kSteps];
  __shared__ unsigned s_equal[kCarryWarps][kSteps];
  __shared__ int s_greater_count[kCarryWarps];
  __shared__ int s_equal_count[kCarryWarps];
  __shared__ int s_equal_quota[kCarryWarps];
  __shared__ int s_output_base[kCarryWarps];

  int greater_count = 0;
  int equal_count = 0;
#pragma unroll 1
  for (int step = 0; step < kSteps; ++step) {
    const int index = begin + step * 32 + lane;
    const bool valid = index < count;
    const int raw = valid ? votes[index] : 0;
    const int value = raw < 0 ? 0 : (raw > max_vote ? max_vote : raw);
    if (valid) votes[index] = 0;
    const bool eligible = valid && index >= min_index;
    const unsigned greater = __ballot_sync(kFullMask, eligible && value > threshold);
    const unsigned equal = __ballot_sync(kFullMask, eligible && value == threshold);
    greater_count += __popc(greater);
    equal_count += __popc(equal);
    if (lane == 0) {
      s_greater[warp][step] = greater;
      s_equal[warp][step] = equal;
    }
  }
  if (lane == 0) {
    s_greater_count[warp] = greater_count;
    s_equal_count[warp] = equal_count;
  }
  __syncthreads();

  if (tid == 0) {
    int equals_before = 0;
    int selected_before = 0;
#pragma unroll
    for (int source_warp = 0; source_warp < kCarryWarps; ++source_warp) {
      const int equal = s_equal_count[source_warp];
      int quota = 0;
      if (block < tie_block) {
        quota = equal;
      } else if (block == tie_block) {
        const int remaining = tie_take - equals_before;
        quota = remaining <= 0 ? 0 : (remaining < equal ? remaining : equal);
      }
      s_equal_quota[source_warp] = quota;
      s_output_base[source_warp] = selected_before;
      equals_before += equal;
      selected_before += s_greater_count[source_warp] + quota;
    }
  }
  __syncthreads();

  const int quota = s_equal_quota[warp];
  const int warp_output_base = output_base + s_output_base[warp];
  int equals_seen = 0;
  int selected_seen = 0;
#pragma unroll 1
  for (int step = 0; step < kSteps; ++step) {
    const unsigned greater = s_greater[warp][step];
    const unsigned equal = s_equal[warp][step];
    const int equal_rank = equals_seen + __popc(equal & lane_mask);
    const bool take_equal = (equal & lane_bit) != 0 && equal_rank < quota;
    const bool selected = (greater & lane_bit) != 0 || take_equal;
    const unsigned mask = __ballot_sync(kFullMask, selected);
    if (selected) {
      const int rank = selected_seen + __popc(mask & lane_mask);
      out_idx[warp_output_base + rank] =
          static_cast<int64_t>(begin + step * 32 + lane);
    }
    equals_seen += __popc(equal);
    selected_seen += __popc(mask);
  }
}

}

void carry_votes_topk_reset_litetopk_(
    torch::Tensor votes, torch::Tensor out_idx, torch::Tensor partial,
    torch::Tensor state, int64_t k64, int64_t max_vote64, int64_t min_index64) {
  // Owned I32[N] votes, I64[12288] HOT, I16[128,max_vote+1] histogram and
  // I32[136] state. N<=2^20 and max_vote is4 or the current recent-row count.
  // Both kernels run on the supplied owner's current stream; reset is explicit.
  const int64_t count64 = votes.numel();
  const int count = static_cast<int>(count64);
  const int min_index = static_cast<int>(min_index64);
  const int eligible = count - min_index;
  const int out_k = static_cast<int>(std::min(k64, static_cast<int64_t>(eligible)));
  const int max_vote = static_cast<int>(max_vote64);
  const int bins = max_vote + 1;
  const int blocks = (count + kCarryTileItems - 1) / kCarryTileItems;
  const int partial_stride = static_cast<int>(partial.size(1));
  const size_t dynamic_smem = static_cast<size_t>(bins) * sizeof(uint32_t);
  const c10::cuda::CUDAGuard device_guard(votes.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  carry_votes_plan_prefix_kernel<<<blocks, kCarryThreads, dynamic_smem,
                                     stream>>>(
      votes.data_ptr<int32_t>(), count, min_index, out_k, max_vote,
      partial.data_ptr<int16_t>(), partial_stride, state.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  carry_votes_emit_warp_kernel
      <<<blocks, kCarryThreads, 0, stream>>>(
          votes.data_ptr<int32_t>(), count, min_index, max_vote,
          out_idx.data_ptr<int64_t>(), state.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}





namespace dsa_litetopk { constexpr uint32_t kCandidateIndexMask=(1u<<20)-1u; }





__global__ void map_votes_kernel(
    int32_t *__restrict__ out_idx, const int32_t *__restrict__ index_map,
    const int32_t *__restrict__ status, int32_t *__restrict__ votes,
    int64_t total, int rows, int index_map_size, int topk, int votes_len,
    int vote_recent_rows) {
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int64_t global_thread =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t vote_begin =
      static_cast<int64_t>(rows - vote_recent_rows) * topk;
  // A bad/underfilled row may contain selector padding rather than K valid
  // physical indices.  Preserve fail-closed behavior before treating the
  // complete output matrix as mappable winners.
  for (int row = static_cast<int>(global_thread); row < rows;
       row += static_cast<int>(step)) {
    if (status[row] != 0) {
      asm volatile("trap;");
      return;
    }

  }
  // The previous always-present telemetry pass synchronized here. Keep
  // that status-to-output ordering after dropping its unconsumed statistics.
  __syncthreads();
  for (int64_t linear = global_thread; linear < total; linear += step) {
    const int32_t physical_idx = out_idx[linear];
    if (static_cast<uint32_t>(physical_idx) >=
        static_cast<uint32_t>(index_map_size)) {
      asm volatile("trap;");
      return;
    }
    const int32_t original_idx = index_map[physical_idx];
    if (static_cast<uint32_t>(original_idx) >=
            static_cast<uint32_t>(index_map_size) ||
        static_cast<uint32_t>(original_idx) >
            dsa_litetopk::kCandidateIndexMask) {
      asm volatile("trap;");
      return;
    }
    out_idx[linear] = original_idx;

    if (votes != nullptr && votes_len > 0) {
      // The next chunk is best predicted by the most recent query
      // window.  Mapping already visits every Q*K winner, so voting all
      // winners from the last 1536 rows only adds the atomics; it needs
      // no extra winner read or launch.
      if (linear >= vote_begin) {
        const int32_t vote_idx =
            original_idx < 0
                ? 0
                : (original_idx >= votes_len ? votes_len - 1 : original_idx);
        atomicAdd(votes + vote_idx, 1);
      }
    }
  }
}

void map_topk_vote_stats_litetopk_(
    const torch::Tensor& out_idx, const torch::Tensor& index_map, const torch::Tensor& status,
    const torch::Tensor& votes, int64_t vote_recent_rows64) {
  // Owned I32[rows,2048] output, I32[rows] status and I32[N] permutation/votes.
  // recent is in[0,rows]; publication and voting share one current-stream pass.
  constexpr int kThreads = 256;
  constexpr int kBlocksPerSm = 8;
  constexpr int kProductionSms = 148;
  const int64_t total = out_idx.numel();
  const int blocks = static_cast<int>(std::min<int64_t>(
      (total + kThreads - 1) / kThreads, kProductionSms * kBlocksPerSm));
  const int votes_len = static_cast<int>(votes.numel());
  const c10::cuda::CUDAGuard device_guard(out_idx.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  map_votes_kernel<<<blocks, kThreads, 0,
                                                          stream>>>(
      out_idx.data_ptr<int32_t>(), index_map.data_ptr<int32_t>(),
      status.data_ptr<int32_t>(),
      votes_len > 0 ? votes.data_ptr<int32_t>() : nullptr, total,
      static_cast<int>(out_idx.size(0)), static_cast<int>(index_map.numel()),
      static_cast<int>(out_idx.size(1)), votes_len,
      static_cast<int>(vote_recent_rows64));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}





// Exact inverse of the request's PAGED table. Only the existing stock
// selector's completed output is consumed; this does not select new scores.
namespace stock_prior {
constexpr int kThreads = 256;
constexpr int kInvalidTable = 1;
constexpr int kDuplicatePage = 2;
constexpr int kInvalidWinner = 4;
constexpr int kInvalidVotes = 8;
constexpr int kHashFull = 16;

__device__ __forceinline__ uint32_t slot_of(int page, int slots) {
  return (uint32_t(page) * 2654435761u) & uint32_t(slots - 1);
}

__global__ void initialize(int* keys, int* values, int slots, int* status) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < slots;
       i += blockDim.x * gridDim.x) {
    keys[i] = -1;
    values[i] = INT_MAX;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) *status = 0;
}

__global__ void build_inverse(const int* table, int pages, int* keys,
                              int* values, int slots, int* status) {
  for (int logical = blockIdx.x * blockDim.x + threadIdx.x; logical < pages;
       logical += blockDim.x * gridDim.x) {
    const int page = table[logical];
    if (page < 0 || page > INT_MAX / 64) {
      atomicOr(status, kInvalidTable);
      continue;
    }
    uint32_t slot = slot_of(page, slots);
    bool found = false;
    for (int n = 0; n < slots; ++n, slot = (slot + 1) & (slots - 1)) {
      const int old = atomicCAS(keys + slot, -1, page);
      if (old == -1) {
        // A duplicate may have marked this -1 before the owner arrives.
        // atomicMin preserves that rejection under either publication order.
        atomicMin(values + slot, logical);
        found = true;
        break;
      }
      if (old == page) {
        atomicMin(values + slot, -1);
        atomicOr(status, kDuplicatePage);
        found = true;
        break;
      }
    }
    if (!found) atomicOr(status, kHashFull);
  }
}

__global__ void inverse_votes(const int* paged, int items, const int* keys,
                              const int* values, int slots, int nv,
                              int* votes, int* status) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < items;
       i += blockDim.x * gridDim.x) {
    const int physical = paged[i];
    if (physical < 0) {
      atomicOr(status, kInvalidWinner);
      continue;
    }
    const int page = physical / 64;
    uint32_t slot = slot_of(page, slots);
    int logical_page = -1;
    for (int n = 0; n < slots; ++n, slot = (slot + 1) & (slots - 1)) {
      const int key = keys[slot];
      if (key == page) {
        logical_page = values[slot];
        break;
      }
      if (key == -1) break;
    }
    const int64_t logical = int64_t(logical_page) * 64 + physical % 64;
    if (logical_page < 0 || logical < 0 || logical >= nv) {
      atomicOr(status, kInvalidWinner);
      continue;
    }
    atomicAdd(votes + logical, 1);
  }
}

__global__ void bound_votes(int* votes, int nv, int max_vote, int* status) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nv;
       i += blockDim.x * gridDim.x) {
    const int v = votes[i];
    if (v < 0 || v > max_vote) {
      atomicOr(status, kInvalidVotes);
      // Keep the original carry selector's histogram access in bounds even
      // for rejected input. A flagged result is never adopted as a prior.
      votes[i] = min(max(v, 0), max_vote);
    }
  }
}

void paged_to_votes_(torch::Tensor snapshot, torch::Tensor hash_storage,
                     torch::Tensor votes, torch::Tensor status,
                     int64_t recent64) {
  // Public stock publication copied its winners and page table into this
  // owned I32[recent*2048+ceil(N/64)] snapshot. Histogram/hash/status are also
  // owned; the side-stream lease lasts through inverse construction and carry.
  const int64_t nv64 = votes.numel();
  const int nv = int(nv64), recent = int(recent64);
  const int pages = (nv + 63) / 64;
  int slots = 1;
  while (slots < pages * 2) slots *= 2;
  const c10::cuda::CUDAGuard guard(votes.device());
  auto stream = c10::cuda::getCurrentCUDAStream();
  int* keys = hash_storage.data_ptr<int>();
  int* values = keys + slots;
  int* flags = status.data_ptr<int>();
  initialize<<<std::min(128, (slots + 255) / 256), kThreads, 0, stream>>>(
      keys, values, slots, flags);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  build_inverse<<<(pages + 255) / 256, kThreads, 0, stream>>>(
      snapshot.data_ptr<int>() + recent * 2048, pages, keys, values, slots, flags);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  inverse_votes<<<std::min(256, (recent * 2048 + 255) / 256), kThreads, 0, stream>>>(
      snapshot.data_ptr<int>(), recent * 2048, keys, values, slots, nv,
      votes.data_ptr<int>(), flags);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  bound_votes<<<std::min(256, (nv + 255) / 256), kThreads, 0, stream>>>(
      votes.data_ptr<int>(), nv, recent, flags);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace stock_prior

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("plan_and_permuted_paged_gather_rho_out",
        pair_swap_gather::plan_and_permuted_paged_gather_rho_out);
  m.def("carry_votes_topk_reset_", &carry_votes_topk_reset_litetopk_);
  m.def("map_topk_vote_stats_litetopk_", &map_topk_vote_stats_litetopk_);
  m.def("stock_paged_to_votes_", &stock_prior::paged_to_votes_);
}
