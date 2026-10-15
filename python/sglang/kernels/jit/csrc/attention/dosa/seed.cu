// Original-F32 seed selection. Retained min/max and four-way histogram follow
// the pinned seed producer; only the threshold bucket needs exact sorting.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <type_traits>
#include "record.cuh"
namespace exact_seed {
constexpr int Threads=256,Top=2048,Bins=256,Seed=12288;
using Sort=cub::BlockRadixSort<float,Threads,1,int>;
struct alignas(16) SuffixStorage {float4 values[1024];};
__device__ __forceinline__ void suffix_store(float4*p,float4 v){uint32_t a=uint32_t(__cvta_generic_to_shared(p));asm volatile("st.volatile.shared.v4.f32 [%0],{%1,%2,%3,%4};"::"r"(a),"f"(v.x),"f"(v.y),"f"(v.z),"f"(v.w):"memory");}
__device__ __forceinline__ float4 suffix_load(const float4*p){uint32_t a=uint32_t(__cvta_generic_to_shared(p));float4 v;asm volatile("ld.volatile.shared.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(a):"memory");return v;}

struct Output{void*values;int*ids,*counts;float*tau;int*error,*fi,*diag;int64_t vs,is;};
// Store an already-selected original score as its20-bit cell and exact ID.
// Selection, kth tau and the FI slow path use the original FP32 scores.
template<bool Check=true> __device__ __forceinline__ void store(
    Output a,int row,int slot,float score,int id) {
  uint32_t cp=h32_record5::encode(score);
  if constexpr(Check) {
   const bool valid=isfinite(score)&&__float_as_uint(score)!=0x80000000u&&
       id>=0&&uint32_t(id)<0x100000u&&
       cp>=h32_record5::Min&&cp<=h32_record5::Max;
   if(!valid){cp=h32_record5::Invalid;atomicOr(a.error+row,1);}
  }
  static_cast<uint8_t*>(a.values)[int64_t(row)*a.vs+slot]=uint8_t(cp);
  a.ids[int64_t(row)*a.is+slot]=int32_t(h32_record5::pack_word(cp,uint32_t(id)));

}
__device__ int bucket(float s,float inv,float offset,bool&bad){float q=fmaf(-s,inv,offset);bad|=!isfinite(q);return q<0.f?0:q<255.f?int(q):255;}
__device__ void soft(Output a,int row,int mode){if(threadIdx.x==0){a.fi[row]=Seed;a.counts[row]=0;a.tau[row]=-INFINITY;a.diag[row*4+3]=mode;}}
__global__ void prepare(const float*scores,int64_t stride,Output a){
 int row=gridDim.x-1-blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;
 __shared__ int hist[4*Bins],wsB[8],wsE[8],th,nb,ne,mode,bad_store;
 __shared__ float mnw[8],mxw[8],edge_values[Threads],exact_tau;
 __shared__ int edge_ids[Threads];
 __shared__ Sort::TempStorage sort;
 __shared__ SuffixStorage suffix;
 constexpr int Records=Seed/(Threads*4);using Mask=unsigned long long;
 float4 retained[8];float mn=INFINITY,mx=-INFINITY;bool nf=false,unsupported=false;
 #pragma unroll
 for(int i=0;i<Records;++i){float4 v=reinterpret_cast<const float4*>(scores+int64_t(row)*stride)[t+i*Threads];if(i<8)retained[i]=v;else suffix_store(&suffix.values[t+(i-8)*Threads],v);float x[4]={v.x,v.y,v.z,v.w};
  #pragma unroll
  for(int j=0;j<4;++j){uint32_t bits=__float_as_uint(x[j]),mag=bits&0x7fffffffu;nf|=!isfinite(x[j]);unsupported|=bits==0x80000000u||(mag&&mag<0x00800000u);if(isfinite(x[j])){mn=fminf(mn,x[j]);mx=fmaxf(mx,x[j]);}}
 }
 bool any_nf=__syncthreads_or(nf),any_unsupported=__syncthreads_or(unsupported);
 if(t==0){a.counts[row]=0;a.fi[row]=0;a.tau[row]=-INFINITY;for(int j=0;j<3;++j)a.diag[row*4+j]=-1;a.diag[row*4+3]=0;mode=0;bad_store=0;if(any_nf)atomicOr(a.error+row,1);}
 __syncthreads();
 if(any_nf||any_unsupported||a.error[row]!=0){soft(a,row,any_nf?3:any_unsupported?2:4);return;}
 #pragma unroll
 for(int d=16;d;d>>=1){mn=fminf(mn,__shfl_xor_sync(0xffffffffu,mn,d));mx=fmaxf(mx,__shfl_xor_sync(0xffffffffu,mx,d));}
 if(lane==0){mnw[warp]=mn;mxw[warp]=mx;}__syncthreads();
 if(t==0)for(int w=1;w<8;++w){mnw[0]=fminf(mnw[0],mnw[w]);mxw[0]=fmaxf(mxw[0],mxw[w]);}__syncthreads();
 float origin=-mxw[0],span=fmaxf(-mnw[0]-origin,1e-20f);origin-=0.f*span;float inv=255.f/(span*1.f),offset=__fmul_rn(-origin,inv);uint32_t im=__float_as_uint(inv)&0x7fffffffu;
 bool calibration=isfinite(origin)&&isfinite(offset)&&inv>0.f&&im>=0x00800000u&&im<0x7f800000u;
 if(!calibration){soft(a,row,2);return;}
 for(int i=t;i<4*Bins;i+=Threads)hist[i]=0;__syncthreads();
 bool bad_phi=false;int*mine=hist+(t/64)*Bins;
 #pragma unroll
 for(int i=0;i<Records;++i){float4 v;if(i<8)v=retained[i];else v=suffix_load(&suffix.values[t+(i-8)*Threads]);float x[4]={v.x,v.y,v.z,v.w};
  #pragma unroll
  for(int j=0;j<4;++j)atomicAdd(mine+bucket(x[j],inv,offset,bad_phi),1);
 }
 if(__syncthreads_or(bad_phi)){soft(a,row,2);return;}
 int h=hist[t]+hist[Bins+t]+hist[2*Bins+t]+hist[3*Bins+t];hist[t]=h;__syncthreads();
 int inclusive=h;
 #pragma unroll
 for(int d=1;d<32;d<<=1){int z=__shfl_up_sync(0xffffffffu,inclusive,d);if(lane>=d)inclusive+=z;}
 if(lane==31)wsB[warp]=inclusive;__syncthreads();int base=0;
 #pragma unroll
 for(int w=0;w<8;++w)if(w<warp)base+=wsB[w];
 int incl=base+inclusive,excl=incl-h;
 if(excl<Top&&Top<=incl){th=t;nb=excl;ne=h;}__syncthreads();
 if(t==0){a.diag[row*4]=nb;a.diag[row*4+1]=ne;a.diag[row*4+2]=Top-nb;}
 if(ne>Threads){soft(a,row,1);return;}
 Mask bm=0,em=0;
 #pragma unroll
 for(int i=0;i<Records;++i){float4 v;if(i<8)v=retained[i];else v=suffix_load(&suffix.values[t+(i-8)*Threads]);float x[4]={v.x,v.y,v.z,v.w};
  #pragma unroll
  for(int j=0;j<4;++j){int b=bucket(x[j],inv,offset,bad_phi);bm|=Mask(b<th)<<(4*i+j);em|=Mask(b==th)<<(4*i+j);}
 }
 int cb=__popcll(bm),ce=__popcll(em);int ib=cb,ie=ce;
 #pragma unroll
 for(int d=1;d<32;d<<=1){int b=__shfl_up_sync(0xffffffffu,ib,d),e=__shfl_up_sync(0xffffffffu,ie,d);if(lane>=d){ib+=b;ie+=e;}}
 if(lane==31){wsB[warp]=ib;wsE[warp]=ie;}__syncthreads();int before_b=ib-cb,before_e=ie-ce,total_b=0,total_e=0;
 #pragma unroll
 for(int w=0;w<8;++w){if(w<warp){before_b+=wsB[w];before_e+=wsE[w];}total_b+=wsB[w];total_e+=wsE[w];}
 // Fast prepare already checked every raw score for NF/negative zero above,
 // before this branch, and B IDs are generated in [0,Seed). The E store has
 // its original explicit ID/finite check. Only finish needs the store guard.
 int rb=0,re=0;
 #pragma unroll
 for(int i=0;i<Records;++i){float4 v;if(i<8)v=retained[i];else v=suffix_load(&suffix.values[t+(i-8)*Threads]);float x[4]={v.x,v.y,v.z,v.w};
  #pragma unroll
  for(int j=0;j<4;++j){int id=t*4+i*Threads*4+j;
   if(bm&(Mask(1)<<(4*i+j))){int at=before_b+rb++;if(at<Top){store<false>(a,row,at,x[j],id);}else atomicOr(&bad_store,1);}
   if(em&(Mask(1)<<(4*i+j))){int at=before_e+re++;if(at<Threads){edge_values[at]=x[j];edge_ids[at]=id;}else atomicOr(&bad_store,1);}
  }
 }
 __syncthreads();
 if(t==0&&(total_b!=nb||total_e!=ne||bad_store||bad_phi)){mode=5;atomicOr(a.error+row,1);}__syncthreads();
 if(mode){soft(a,row,mode);return;}
 float key[1]={t<ne?edge_values[t]:-INFINITY};int id[1]={t<ne?edge_ids[t]:-1};Sort(sort).SortDescending(key,id);
 int needed=Top-nb;
 if(t<needed){
  if(id[0]<0||id[0]>=Seed||!isfinite(key[0]))atomicOr(a.error+row,1);
  else{store<false>(a,row,nb+t,key[0],id[0]);}
  if(t==needed-1)exact_tau=key[0];
 }
 __syncthreads();
 if(t==0){if(a.error[row]){a.fi[row]=Seed;a.diag[row*4+3]=5;}else{a.counts[row]=Top;a.tau[row]=exact_tau;}}
}
__device__ uint32_t ordered(float x){uint32_t u=__float_as_uint(x);return u&0x80000000u?~u:u^0x80000000u;}
__device__ float decode(uint32_t u){return __uint_as_float(u&0x80000000u?u^0x80000000u:~u);}
__global__ void finish(const float*scores,int64_t stride,const int*selected,Output a){
 int row=blockIdx.x,t=threadIdx.x,lane=t&31,warp=t>>5;if(a.fi[row]==0)return;
 __shared__ uint32_t mins[8];uint32_t lo=UINT_MAX;bool nan=false,bad=false;
 for(int j=t;j<Top;j+=Threads){int raw=selected[int64_t(row)*Top+j],id=max(0,min(Seed-1,raw));bad|=raw!=id;float v=scores[int64_t(row)*stride+id];store<>(a,row,j,v,id);nan|=isnan(v);if(!isnan(v))lo=min(lo,ordered(v));}
 bool has_nan=__syncthreads_or(nan),bad_id=__syncthreads_or(bad);
 #pragma unroll
 for(int d=16;d;d>>=1)lo=min(lo,__shfl_down_sync(0xffffffffu,lo,d));
 if(lane==0)mins[warp]=lo;__syncthreads();
 if(t==0){uint32_t best=mins[0];for(int w=1;w<8;++w)best=min(best,mins[w]);float tau=has_nan?__uint_as_float(0x7fc00000u):decode(best);a.tau[row]=tau;a.counts[row]=Top;if(bad_id||!isfinite(tau))atomicOr(a.error+row,1);}
}
}
using torch::Tensor;
exact_seed::Output output(Tensor scores,Tensor values,Tensor ids,Tensor counts,Tensor tau,Tensor error,Tensor lengths,Tensor diag){
 // Owner-reserved F32[Q,12288] view has row stride12544 and16B-aligned
 // storage; U8/I32[Q,114688] frontier planes and row metadata are independent.
 // DeepGEMM's output binding is checked at its Python call boundary.
 return {values.data_ptr(),ids.data_ptr<int>(),counts.data_ptr<int>(),tau.data_ptr<float>(),error.data_ptr<int>(),lengths.data_ptr<int>(),diag.data_ptr<int>(),values.stride(0),ids.stride(0)};
}
// Fast selection and FI recovery both publish the same record format.
void prepare5(Tensor scores,Tensor low,Tensor words,Tensor counts,Tensor tau,
    Tensor error,Tensor lengths,Tensor diag) {
 c10::cuda::CUDAGuard guard(scores.device());
 auto a=output(scores,low,words,counts,tau,error,lengths,diag);
 exact_seed::prepare<<<scores.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(
     scores.data_ptr<float>(),scores.stride(0),a);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void finish5(Tensor scores,Tensor selected,Tensor low,Tensor words,Tensor counts,
    Tensor tau,Tensor error,Tensor lengths,Tensor diag) {
 c10::cuda::CUDAGuard guard(scores.device());
 auto a=output(scores,low,words,counts,tau,error,lengths,diag);
 // selected is the owner-reserved contiguous I32[Q,2048] FI output.
 exact_seed::finish<<<scores.size(0),256,0,at::cuda::getCurrentCUDAStream()>>>(
     scores.data_ptr<float>(),scores.stride(0),selected.data_ptr<int>(),a);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
 m.def("prepare5",&prepare5);m.def("finish5",&finish5);
}
