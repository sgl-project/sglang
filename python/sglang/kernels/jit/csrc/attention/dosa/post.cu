// Exact completion of global20 score cells and conditional recovery.
// One original H32 arithmetic/TMEM implementation for both modes.
#include "score.cuh"
#include <cub/block/block_radix_sort.cuh>
#include <cfloat>



namespace fused_boundary_h32 {
using namespace certicascade_streaming_exact;
using Barrier=cutlass::arch::ClusterTransactionBarrier;
constexpr int BatchQueries=256/kHeads, KeysPerQuery=128/BatchQueries;
constexpr int QueriesPerWarp=32/KeysPerQuery, GroupDoneBit=1<<(BatchQueries/2);
using Sort=cub::BlockRadixSort<float,128,1,int>;
struct alignas(1024) Smem {
  union Work {alignas(1024) uint8_t keys[128*128];typename Sort::TempStorage sort;} work;
  alignas(1024) uint8_t q[kHeads*128];
  alignas(128) float weights[kHeads],scale[128];
  alignas(128) int ids[128],coordinates[128];
  Barrier gathered,done;
  uint32_t tmem;
};
struct Args {
  const uint8_t*q;
  const float*scale,*weights;
  const int*boundary_ids,*boundary_counts,*diag,*starts,*ends;
  int*error,*output;
  float*values;
  int*ids;
  int nk,stride;
  int64_t boundary_stride;
  int rows;
  int*group_done;
};
__device__ __forceinline__ int swizzle(int row,int vec){return row*8+(vec^(row%8));}
template<int Copies>__device__ __forceinline__ void copy_q_async(uint8_t* dst,const uint8_t* src,int t){
 #pragma unroll
 for(int it=0;it<Copies;++it){
  int i=t+it*128;
  uint32_t to=cute::cast_smem_ptr_to_uint(reinterpret_cast<uint4*>(dst)+swizzle(i/8,i%8));
  const uint4* from=reinterpret_cast<const uint4*>(src)+i;
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"::"r"(to),"l"(from):"memory");
  if((it&7)==7||it==Copies-1)asm volatile("cp.async.commit_group;":::"memory");
 }
}

__device__ __forceinline__ void gather(Smem&s,const cute::TmaDescriptor*map) {
  #pragma unroll
  for(int r=0;r<128;r+=4) {
    int4 coord=*reinterpret_cast<const int4*>(s.coordinates+r);
    deep_gemm::ptx::tma_gather4(map,s.gathered,s.work.keys+r*128,0,coord,
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL));
  }
  s.gathered.arrive_and_expect_tx(128*128);
}

struct alignas(1024) PairSmem {
  alignas(1024) uint8_t keys[128*128],q[2*kHeads*128];
  alignas(128) float weights[2*kHeads],scale[128],exchange_score[128];
  alignas(128) int ids[128],coordinates[128],exchange_id[128];
  int b[2],e[2],bad[2];Barrier gathered,done;uint32_t tmem;
};
__device__ __forceinline__ void complete_pair64(Args a,const cute::TmaDescriptor*map,int first) {
  const int t=threadIdx.x,warp=t/32,qr=t/64,slot=t%64,row=first+qr;
  extern __shared__ __align__(1024) uint8_t buffer[];
  auto&s=*reinterpret_cast<PairSmem*>(buffer);
  if(t<2){int r=first+t;s.b[t]=a.diag[r*2];s.e[t]=a.diag[r*2+1];s.bad[t]=a.error[r]!=0;}
  if(t==0){s.gathered.init(1);s.done.init(1);cutlass::arch::fence_barrier_init();cute::prefetch_tma_descriptor(map);}
  __syncthreads();
  bool active=slot<s.e[qr]&&!s.bad[qr];
  int id=active?a.boundary_ids[uint64_t(row)*a.boundary_stride+slot]:0;
  bool bad=active&&(id<a.starts[row]||id>=a.ends[row]||id<0||id>=a.nk);
  if(bad){atomicOr(s.bad+qr,1);atomicOr(a.error+row,1);}
  __syncthreads();
  // A late bad ID invalidates only its query. Both query columns still drain
  // the shared MMA; the healthy partner uses its complete original key set.
  active=slot<s.e[qr]&&!s.bad[qr];id=active?id:0;
  s.ids[t]=active?id:-1;s.coordinates[t]=id;s.scale[t]=active?a.scale[id]:0.f;
  copy_q_async<kHeads/8>(s.q,a.q+uint64_t(first)*kHeads*128,t);
  if(t<2*kHeads)s.weights[t]=a.weights[first*kHeads+t];
  if(warp==0){cute::TMEM::Allocator1Sm().allocate(2*kHeads,&s.tmem);cute::TMEM::Allocator1Sm().release_allocation_lock();}
  asm volatile("cp.async.wait_group 0;":::"memory");
  cutlass::arch::fence_view_async_shared();__syncthreads();
  const uint32_t tm=deep_gemm::ptx::ld_shared(&s.tmem);
  if(t==0){
    #pragma unroll
    for(int r=0;r<128;r+=4){int4 coordinates=*reinterpret_cast<const int4*>(s.coordinates+r);
      deep_gemm::ptx::tma_gather4(map,s.gathered,s.keys+r*128,0,coordinates,
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL));}
    s.gathered.arrive_and_expect_tx(128*128);s.gathered.wait(0);
    deep_gemm::ptx::tcgen05_after_thread_sync();
    issue_h32_mma<128,2*kHeads>(reinterpret_cast<cutlass::float_e4m3_t*>(s.keys),
      reinterpret_cast<cutlass::float_e4m3_t*>(s.q),0,tm,cute::cast_smem_ptr_to_uint(&s.done));
  }
  s.done.wait(0);deep_gemm::ptx::tcgen05_after_thread_sync();
  float accum[kHeads];tmem_load_heads(tm+qr*kHeads,accum);
  float score=exact_h32_reduce(accum,s.weights+qr*kHeads,s.scale[t]);
  if(active&&!isfinite(score)){atomicOr(s.bad+qr,1);atomicOr(a.error+row,1);}
  deep_gemm::ptx::tcgen05_before_thread_sync();__syncthreads();
  if(warp==0)cute::TMEM::Allocator1Sm().free(tm,2*kHeads);
  __syncthreads();
  score=active&&!s.bad[qr]?score:-INFINITY;id=active&&!s.bad[qr]?s.ids[t]:-1;
  #pragma unroll
  for(int width=2;width<=64;width*=2){
    #pragma unroll
    for(int step=width/2;step>0;step/=2){
      float other;int other_id;
      if(step==32){
        s.exchange_score[t]=score;s.exchange_id[t]=id;__syncthreads();
        other=s.exchange_score[t^32];other_id=s.exchange_id[t^32];
        // This is the only cross-warp exchange; these shared slots are never
        // overwritten again, so no second read-completion barrier is needed.
      }else{other=__shfl_xor_sync(0xffffffffu,score,step);other_id=__shfl_xor_sync(0xffffffffu,id,step);}
      bool take_best=((slot&step)==0)==((slot&width)==0);
      bool other_better=other>score||(other==score&&other_id<id);
      if(take_best?other_better:!other_better){score=other;id=other_id;}
    }
  }
  const int needed=2048-s.b[qr];
  if(!s.bad[qr]){
    if(slot<needed)a.output[uint64_t(row)*2048+s.b[qr]+slot]=id;
  }
}

__device__ __forceinline__ uint32_t exact_ordered(float x) {
 uint32_t u=__float_as_uint(x);return (u&0x80000000u)?~u:(u^0x80000000u);
}
__device__ __forceinline__ void finish_large_radix(Args a,int row,int b,int e,uint32_t* work){
 const int t=threadIdx.x,lane=t&31,needed=2048-b;uint32_t*state=work+256;
 bool bad=false;
 for(int j=t;j<b+e;j+=128){float v=a.values[uint64_t(row)*a.stride+j];int id=a.ids[uint64_t(row)*a.stride+j];
  bad|=!isfinite(v)||id<a.starts[row]||id>=a.ends[row];}
 if(__syncthreads_or(bad)){if(t==0)atomicOr(a.error+row,1);return;}
 if(t==0){state[0]=0;state[1]=needed;state[2]=0;state[3]=0;state[4]=0;state[5]=0;}
 __syncthreads();
 #pragma unroll
 for(int pass=0;pass<4;++pass){
  const int shift=24-8*pass;const uint32_t prefix_mask=shift==24?0u:(0xffffffffu<<(shift+8));
  work[t]=0;work[t+128]=0;__syncthreads();
  const uint32_t prefix=state[0];
  for(int j=t;j<e;j+=128){uint32_t code=exact_ordered(a.values[uint64_t(row)*a.stride+b+j]);
   if((code&prefix_mask)==prefix)atomicAdd(work+((code>>shift)&255u),1u);}
  __syncthreads();
  if(t<32){
   uint32_t local=0;const int first=255-lane*8;
   #pragma unroll
   for(int j=0;j<8;++j)local+=work[first-j];
   uint32_t inclusive=local;
   #pragma unroll
   for(int off=1;off<32;off<<=1){uint32_t x=__shfl_up_sync(0xffffffffu,inclusive,off);if(lane>=off)inclusive+=x;}
   uint32_t rank=state[1];unsigned hits=__ballot_sync(0xffffffffu,inclusive>=rank);
   int owner=hits?__ffs(hits)-1:0;uint32_t digit=0,next_rank=0,equal=0;
   if(lane==owner){uint32_t remaining=rank-(inclusive-local);
    #pragma unroll
    for(int j=0;j<8;++j){uint32_t count=work[first-j];if(remaining>count)remaining-=count;else{digit=first-j;next_rank=remaining;equal=count;break;}}
   }
   digit=__shfl_sync(0xffffffffu,digit,owner);next_rank=__shfl_sync(0xffffffffu,next_rank,owner);equal=__shfl_sync(0xffffffffu,equal,owner);
   if(lane==0){if(!hits||!rank||!next_rank||next_rank>equal)state[3]=1;state[0]|=digit<<shift;state[1]=next_rank;state[2]=equal;}
  }
  __syncthreads();
 }
 if(state[3]){if(t==0)atomicOr(a.error+row,1);return;}
 const uint32_t pivot=state[0],equal_take=state[1];const int strict=needed-int(equal_take);
 if(strict<0||equal_take==0){if(t==0)atomicOr(a.error+row,1);return;}
 for(int j=t;j<e;j+=128){
  int id=a.ids[uint64_t(row)*a.stride+b+j];uint32_t code=exact_ordered(a.values[uint64_t(row)*a.stride+b+j]);
  if(code>pivot){uint32_t pos=atomicAdd(state+4,1u);if(pos<uint32_t(strict))a.output[uint64_t(row)*2048+b+pos]=id;else atomicOr(state+3,1u);}
  else if(code==pivot){uint32_t pos=atomicAdd(state+5,1u);if(pos<equal_take)a.output[uint64_t(row)*2048+b+strict+pos]=id;}
 }
 for(int j=t;j<b;j+=128)a.output[uint64_t(row)*2048+j]=a.ids[uint64_t(row)*a.stride+j];
 __syncthreads();
 if(t==0){
  if(state[3]||state[4]!=uint32_t(strict)||state[5]!=state[2])atomicOr(a.error+row,1);
 }
}

CUTLASS_GLOBAL void __launch_bounds__(128,4) complete(
    Args a,const __grid_constant__ cute::TmaDescriptor map) {
  int row=blockIdx.x,t=threadIdx.x,warp=t/32;
  const int plan=a.group_done[row/BatchQueries];
  if(plan&GroupDoneBit)return;
  if(plan&(1<<((row%BatchQueries)/2))){
    if((row&1)==0)complete_pair64(a,&map,row);
    return;
  }
  int b=a.diag[row*2],e=a.diag[row*2+1];
  if(a.error[row]!=0)return;
  bool invalid=b<0||b>=2048||e<=0||e>4096||b+e<2048||
      b+e>a.stride||a.boundary_counts[row]!=e||a.starts[row]<0||
      a.ends[row]<a.starts[row]||a.ends[row]>a.nk;
  if(invalid){if(t==0)atomicOr(a.error+row,1);return;}
  extern __shared__ __align__(1024) uint8_t buffer[];
  auto&s=*reinterpret_cast<Smem*>(buffer);
  if(t==0) {
    s.gathered.init(1);s.done.init(1);
    cutlass::arch::fence_barrier_init();cute::prefetch_tma_descriptor(&map);
  }
  for(int i=t;i<kHeads*8;i+=128)
    reinterpret_cast<uint4*>(s.q)[swizzle(i/8,i%8)]=
        reinterpret_cast<const uint4*>(a.q+uint64_t(row)*kHeads*128)[i];
  if(t<kHeads)s.weights[t]=a.weights[row*kHeads+t];
  if(warp==0) {
    cute::TMEM::Allocator1Sm().allocate(kHeads,&s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  cutlass::arch::fence_view_async_shared();__syncthreads();
  const uint32_t tm=deep_gemm::ptx::ld_shared(&s.tmem);
  float small_score[1]={-INFINITY};int small_id[1]={-1};
  bool failed=false;
  for(int begin=0,packet=0;begin<e;begin+=128,++packet) {
    bool active=begin+t<e;
    int id=active?a.boundary_ids[uint64_t(row)*a.boundary_stride+begin+t]:0;
    bool bad=active&&(id<a.starts[row]||id>=a.ends[row]||id<0||id>=a.nk);
    // Validate logical coordinates before TMA or the scale load.
    if(__syncthreads_or(bad)) {if(t==0)atomicOr(a.error+row,1);failed=true;break;}
    s.ids[t]=id;s.coordinates[t]=id;s.scale[t]=active?a.scale[id]:0.f;
    cutlass::arch::fence_view_async_shared();__syncthreads();
    if(t==0) {
      gather(s,&map);s.gathered.wait(packet&1);
      deep_gemm::ptx::tcgen05_after_thread_sync();
      issue_h32_mma<128,kHeads>(reinterpret_cast<cutlass::float_e4m3_t*>(s.work.keys),
          reinterpret_cast<cutlass::float_e4m3_t*>(s.q),0,tm,cute::cast_smem_ptr_to_uint(&s.done));
    }
    s.done.wait(packet&1);deep_gemm::ptx::tcgen05_after_thread_sync();
    float accum[kHeads];
    tmem_load_heads(tm,accum);
    float score=exact_h32_reduce(accum,s.weights,s.scale[t]);
    if(__syncthreads_or(active&&!isfinite(score))) {
      if(t==0)atomicOr(a.error+row,1);failed=true;break;
    }
    if(e<=128) {small_score[0]=active?score:-INFINITY;small_id[0]=active?id:-1;}
    else if(active) {
      uint64_t at=uint64_t(row)*a.stride+b+begin+t;
      a.values[at]=score;a.ids[at]=id;
    }
    deep_gemm::ptx::tcgen05_before_thread_sync();__syncthreads();
  }
  // No MMA, TMA or TMEM consumer survives this barrier. Reuse dead K SMEM
  // for CUB and release TMEM before sorting the per-thread H32 registers.
  deep_gemm::ptx::tcgen05_before_thread_sync();__syncthreads();
  if(warp==0)cute::TMEM::Allocator1Sm().free(tm,kHeads);
  __syncthreads();
  if(failed)return;
  if(e<=128) {
    Sort(s.work.sort).SortDescending(small_score,small_id);
    int needed=2048-b;
    if(t<needed)a.output[uint64_t(row)*2048+b+t]=small_id[0];
  } else {
    __syncthreads();
    finish_large_radix(a,row,b,e,reinterpret_cast<uint32_t*>(s.work.keys));
  }
}
struct alignas(1024) BatchSmem {
  alignas(1024) uint8_t keys[128*128],q[256*128];
  alignas(128) float weights[256],scale[128];
  alignas(128) int ids[128],coordinates[128];
  int b[BatchQueries],e[BatchQueries];
  Barrier gathered,done;
  uint32_t tmem;
};

CUTLASS_GLOBAL void __launch_bounds__(128,2) complete_batch8(
    Args a,const __grid_constant__ cute::TmaDescriptor map) {
  const int group=blockIdx.x,first=group*BatchQueries,t=threadIdx.x,warp=t/32;
  // Freeze the Q2 decisions before any later completion can report a fault.
  // A complete B8 group remains first priority and uses its original body.
  bool invalid=false,eligible64=false;
  if(t<BatchQueries){
    int row=first+t;
    if(row>=a.rows)invalid=true;
    else{
      int b=a.diag[row*2],e=a.diag[row*2+1];
      bool valid=a.error[row]==0&&b>=0&&b<2048&&e>0&&b+e>=2048&&
        b+e<=a.stride&&a.boundary_counts[row]==e&&a.starts[row]>=0&&
        a.ends[row]>=a.starts[row]&&a.ends[row]<=a.nk;
      invalid=!valid||e>KeysPerQuery;eligible64=valid&&e<=64;
    }
  }
  const unsigned mask=__ballot_sync(0xffffffffu,eligible64);
  if(t==0){int plan=0;
    #pragma unroll
    for(int p=0;p<BatchQueries/2;++p)if(((mask>>(2*p))&3u)==3u)plan|=1<<p;
    a.group_done[group]=plan;
  }
  if(__syncthreads_or(invalid))return;
  extern __shared__ __align__(1024) uint8_t buffer[];
  auto&s=*reinterpret_cast<BatchSmem*>(buffer);
  if(t<BatchQueries) {
    s.b[t]=a.diag[(first+t)*2];s.e[t]=a.diag[(first+t)*2+1];
  }
  if(t==0) {
    s.gathered.init(1);s.done.init(1);
    cutlass::arch::fence_barrier_init();cute::prefetch_tma_descriptor(&map);
  }
  __syncthreads();
  int qr=t/KeysPerQuery,row=first+qr,slot=t%KeysPerQuery;
  const bool active=slot<s.e[qr];
  int id=active?a.boundary_ids[uint64_t(row)*a.boundary_stride+slot]:0;
  bool bad=active&&(id<a.starts[row]||id>=a.ends[row]||id<0||id>=a.nk);
  if(bad)atomicOr(a.error+row,1);
  if(__syncthreads_or(bad))return;
  s.ids[t]=active?id:-1;s.coordinates[t]=id;s.scale[t]=active?a.scale[id]:0.f;
  copy_q_async<16>(s.q,a.q+uint64_t(first)*kHeads*128,t);
  for(int i=t;i<256;i+=128)s.weights[i]=a.weights[first*kHeads+i];
  if(warp==0) {
    cute::TMEM::Allocator1Sm().allocate(256,&s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  asm volatile("cp.async.wait_group 0;":::"memory");
  cutlass::arch::fence_view_async_shared();__syncthreads();
  uint32_t tm=deep_gemm::ptx::ld_shared(&s.tmem);
  if(t==0) {
    #pragma unroll
    for(int r=0;r<128;r+=4) {
      int4 coordinates=*reinterpret_cast<const int4*>(s.coordinates+r);
      deep_gemm::ptx::tma_gather4(&map,s.gathered,s.keys+r*128,0,coordinates,
        static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL));
    }
    s.gathered.arrive_and_expect_tx(128*128);s.gathered.wait(0);
    deep_gemm::ptx::tcgen05_after_thread_sync();
    issue_h32_mma<128,256>(reinterpret_cast<cutlass::float_e4m3_t*>(s.keys),
      reinterpret_cast<cutlass::float_e4m3_t*>(s.q),0,tm,
      cute::cast_smem_ptr_to_uint(&s.done));
  }
  s.done.wait(0);deep_gemm::ptx::tcgen05_after_thread_sync();
  float score=-INFINITY;
  #pragma unroll
  for(int half=0;half<QueriesPerWarp;++half) {
    // Column addresses are uniform within each warp. Its lower/upper half
    // retains the corresponding query's original-H32 result.
    int query=warp*QueriesPerWarp+half;
    float accum[kHeads];
    tmem_load_heads(tm+query*kHeads,accum);
    float v=exact_h32_reduce(accum,s.weights+query*kHeads,s.scale[t]);
    if((t%32)/KeysPerQuery==half)score=active?v:-INFINITY;
  }
  bad=active&&!isfinite(score);
  if(bad)atomicOr(a.error+row,1);
  bool failed=__syncthreads_or(bad);
  deep_gemm::ptx::tcgen05_before_thread_sync();__syncthreads();
  if(warp==0)cute::TMEM::Allocator1Sm().free(tm,256);
  __syncthreads();
  if(failed)return;
  id=s.ids[t];
  #pragma unroll
  for(int width=2;width<=KeysPerQuery;width*=2) {
    #pragma unroll
    for(int step=width/2;step>0;step/=2) {
      float other=__shfl_xor_sync(0xffffffffu,score,step,KeysPerQuery);
      int other_id=__shfl_xor_sync(0xffffffffu,id,step,KeysPerQuery);
      bool take_best=((slot&step)==0)==((slot&width)==0);
      bool other_better=other>score||(other==score&&other_id<id);
      if(take_best?other_better:!other_better){score=other;id=other_id;}
    }
  }
  int needed=2048-s.b[qr];
  if(slot<needed)a.output[uint64_t(row)*2048+s.b[qr]+slot]=id;
  __syncthreads();
  if(t==0)a.group_done[group]=GroupDoneBit;
}
} // namespace fused_boundary_h32

void complete_h32(torch::Tensor q,torch::Tensor k,torch::Tensor scale,torch::Tensor weights,
    torch::Tensor boundary_ids,torch::Tensor boundary_counts,torch::Tensor diag,
    torch::Tensor starts,torch::Tensor ends,torch::Tensor error,torch::Tensor output,
    torch::Tensor values,torch::Tensor ids) {
  // Owned FP8 Q/K and F32 weights/scales; I32[Q,4096] boundary IDs,
  // I32[Q,2048] output, F32/I32[Q,8192] temporaries and row metadata.
  // The owner keeps their allocations through every current-stream consumer;
  // B/E fields use I32[Q,2]. Kernels still validate counts, IDs, intervals and scores.
  const int rows=values.size(0);
  using namespace fused_boundary_h32;
  c10::cuda::CUDAGuard guard(q.device());
  auto groups=torch::empty({(rows+BatchQueries-1)/BatchQueries},error.options());
  Args a{static_cast<const uint8_t*>(q.data_ptr()),scale.data_ptr<float>(),weights.data_ptr<float>(),
      boundary_ids.data_ptr<int>(),boundary_counts.data_ptr<int>(),diag.data_ptr<int>(),
      starts.data_ptr<int>(),ends.data_ptr<int>(),error.data_ptr<int>(),output.data_ptr<int>(),
      values.data_ptr<float>(),ids.data_ptr<int>(),int(k.size(0)),int(values.size(1)),boundary_ids.stride(0),rows,
      groups.data_ptr<int>()};
  auto map=make_map(k.data_ptr(),CU_TENSOR_MAP_DATA_TYPE_UINT8,{128,(cuuint64_t)k.size(0)},
      {128},{128,1},CU_TENSOR_MAP_SWIZZLE_128B);
  C10_CUDA_CHECK(cudaFuncSetAttribute(complete,cudaFuncAttributeMaxDynamicSharedMemorySize,std::max(sizeof(Smem),sizeof(PairSmem))));
  C10_CUDA_CHECK(cudaFuncSetAttribute(complete_batch8,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(BatchSmem)));
  complete_batch8<<<(rows+BatchQueries-1)/BatchQueries,128,sizeof(BatchSmem),c10::cuda::getCurrentCUDAStream()>>>(a,map);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  complete<<<rows,128,std::max(sizeof(Smem),sizeof(PairSmem)),c10::cuda::getCurrentCUDAStream()>>>(a,map);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void set_if_any(cudaGraphConditionalHandle handle,
                          const int32_t* values, int count, bool positive) {
  bool active = false;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    active |= (positive ? values[i] > 0 : values[i] != 0);
  const int any = __syncthreads_or(active);
  if (threadIdx.x == 0) cudaGraphSetConditional(handle, any != 0);
}

__global__ void set_if_any_not_in(cudaGraphConditionalHandle handle,
                                 const int32_t* values, int count,
                                 int inactive0, int inactive1) {
  bool active = false;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const int value = values[i];
    active |= value != inactive0 && value != inactive1;
  }
  const int any = __syncthreads_or(active);
  if (threadIdx.x == 0) cudaGraphSetConditional(handle, any != 0);
}

// The supplied child graph is immutable and its tensor allocations must remain
// alive. cudaGraphAddChildGraphNode copies its topology into the owning graph.
void attach_conditional(torch::Tensor values, bool positive, uint64_t body_address,
                        bool exclude_inactive, int inactive0, int inactive1) {
  // values is the owner's contiguous I32 error or Q4 budget vector. The
  // captured parent leases the immutable child graph and all of its tensors.
  TORCH_CHECK(body_address!=0, "conditional body must name a live child graph");
  c10::cuda::CUDAGuard guard(values.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus status;
  cudaGraph_t graph;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph,
                                         nullptr, nullptr, nullptr));
  TORCH_CHECK(status == cudaStreamCaptureStatusActive,
              "conditional attachment requires active stream capture");
  cudaGraphConditionalHandle handle;
  C10_CUDA_CHECK(cudaGraphConditionalHandleCreate(
      &handle, graph, 0, cudaGraphCondAssignDefault));
  if (exclude_inactive)
    set_if_any_not_in<<<1, 256, 0, stream>>>(
        handle, values.data_ptr<int32_t>(), int(values.numel()), inactive0, inactive1);
  else
    set_if_any<<<1, 256, 0, stream>>>(handle, values.data_ptr<int32_t>(),
                                     int(values.numel()), positive);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const cudaGraphNode_t* dependencies;
  const cudaGraphEdgeData* edges;
  size_t count;
  C10_CUDA_CHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr, &graph,
                                         &dependencies, &edges, &count));
  cudaGraphNodeParams params{};
  params.type = cudaGraphNodeTypeConditional;
  params.conditional.handle = handle;
  params.conditional.type = cudaGraphCondTypeIf;
  params.conditional.size = 1;
  cudaGraphNode_t conditional, child;
  C10_CUDA_CHECK(cudaGraphAddNode(&conditional, graph, dependencies, edges,
                                 count, &params));
  C10_CUDA_CHECK(cudaGraphAddChildGraphNode(
      &child, params.conditional.phGraph_out[0], nullptr, 0,
      reinterpret_cast<cudaGraph_t>(body_address)));
  C10_CUDA_CHECK(cudaStreamUpdateCaptureDependencies(
      stream, &conditional, nullptr, 1, cudaStreamSetCaptureDependencies));
}

void attach_if_any(torch::Tensor values, bool positive, uint64_t body_address) {
  attach_conditional(values, positive, body_address, false, 0, 0);
}

void attach_if_any_not_in(torch::Tensor values, int inactive0, int inactive1,
                         uint64_t body_address) {
  // Every value outside the explicitly inactive set executes the original body,
  // including malformed/unknown budgets. Original error handling stays inside it.
  attach_conditional(values, false, body_address, true, inactive0, inactive1);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("complete_h32_fused_large", &complete_h32);
  m.def("attach_if_any", &attach_if_any);
  m.def("attach_if_any_not_in", &attach_if_any_not_in);
}
