#include <cublasLt.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// W[N,K], X[M,K], Y[M,N] are seen as W^T[K,N], X^T[K,M], Y^T[N,M].
// Only FP32 compute is allowed. In-place partial sums require FP32 output;
// the Python caller includes the final BF16 cast in the measured graph.
static thread_local std::string error;
static void ck(cublasStatus_t s) { if(s) throw std::runtime_error("cublas status="+std::to_string(int(s))); }
struct Ctx {
  cublasLtHandle_t lt{};
  cublasLtMatmulDesc_t op{};
  cublasLtMatrixLayout_t a{},b{},c{};
  cublasLtMatmulPreference_t pref{};
  std::vector<cublasLtMatmulAlgo_t> algos;
  size_t workspace;
  int output_fp32, policy;
  Ctx(int m,int n,int k,int fp32,int mode,size_t ws,bool search=true):workspace(ws),output_fp32(fp32),policy(mode) {
    ck(cublasLtCreate(&lt));
    ck(cublasLtMatmulDescCreate(&op,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
    ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    ck(cublasLtMatrixLayoutCreate(&a,CUDA_R_16BF,k,n,k));
    ck(cublasLtMatrixLayoutCreate(&b,CUDA_R_16BF,k,m,k));
    ck(cublasLtMatrixLayoutCreate(&c,fp32?CUDA_R_32F:CUDA_R_16BF,n,m,n));
    ck(cublasLtMatmulPreferenceCreate(&pref));
    ck(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws)));
    uint32_t mask=mode==1?0:CUBLASLT_REDUCTION_SCHEME_INPLACE;
    ck(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,&mask,sizeof(mask)));
    if(!search)return;
    cublasLtMatmulHeuristicResult_t hs[128];int count=0;
    ck(cublasLtMatmulAlgoGetHeuristic(lt,op,a,b,c,c,pref,128,hs,&count));
    for(int i=0;i<count;i++)if(hs[i].state==CUBLAS_STATUS_SUCCESS)add(hs[i].algo);
    int ids[128],nc=0;
    ck(cublasLtMatmulAlgoGetIds(lt,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,
       fp32?CUDA_R_32F:CUDA_R_16BF,fp32?CUDA_R_32F:CUDA_R_16BF,128,ids,&nc));
    for(int q=0;q<nc && algos.size()<384;q++) {
      cublasLtMatmulAlgo_t seed;
      if(cublasLtMatmulAlgoInit(lt,CUBLAS_COMPUTE_32F,CUDA_R_32F,CUDA_R_16BF,CUDA_R_16BF,
        fp32?CUDA_R_32F:CUDA_R_16BF,fp32?CUDA_R_32F:CUDA_R_16BF,ids[q],&seed))continue;
      add(seed);
      int tiles[64];size_t sz=0;
      if(cublasLtMatmulAlgoCapGetAttribute(&seed,CUBLASLT_ALGO_CAP_TILE_IDS,tiles,sizeof(tiles),&sz))continue;
      int nt=std::min(12,int(sz/sizeof(int)));
      for(int t=0;t<nt;t++) {
        auto algo=seed;
        if(cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tiles[t],sizeof(int)))continue;
        if(mode==1) {
          uint32_t split=1,red=0;
          cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split));
          cublasLtMatmulAlgoConfigSetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&red,sizeof(red));add(algo);
        } else if(fp32) {
          for(uint32_t split:{2u,4u,8u}) {
            auto in=algo;uint32_t red=CUBLASLT_REDUCTION_SCHEME_INPLACE;
            if(cublasLtMatmulAlgoConfigSetAttribute(&in,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split)))continue;
            if(cublasLtMatmulAlgoConfigSetAttribute(&in,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&red,sizeof(red)))continue;
            add(in);
          }
        }
      }
    }
  }
  void add(const cublasLtMatmulAlgo_t& algo) {
    uint32_t split=0,red=0;size_t sz=0;
    cublasLtMatmulAlgoConfigGetAttribute(&algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,&split,sizeof(split),&sz);
    cublasLtMatmulAlgoConfigGetAttribute(&algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,&red,sizeof(red),&sz);
    if(red!=0 && red!=CUBLASLT_REDUCTION_SCHEME_INPLACE)return;
    if(!output_fp32 && red==CUBLASLT_REDUCTION_SCHEME_INPLACE)return;
    if(policy==1 && (split>1 || red!=0))return;
    // BF16 in-place candidates are never constructed by the Python caller.
    cublasLtMatmulHeuristicResult_t h{};
    if(cublasLtMatmulAlgoCheck(lt,op,a,b,c,c,&algo,&h) || h.workspaceSize>workspace)return;
    for(auto& x:algos)if(!std::memcmp(&x,&algo,sizeof(algo)))return;
    algos.push_back(algo);
  }
  ~Ctx() {if(pref)cublasLtMatmulPreferenceDestroy(pref);if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);if(op)cublasLtMatmulDescDestroy(op);if(lt)cublasLtDestroy(lt);}
};
extern "C" const char* lm_error(){return error.c_str();}
extern "C" size_t lm_lt_version(){return cublasLtGetVersion();}
extern "C" void* lm_create(int m,int n,int k,int fp32,int mode,size_t ws) {
  try {return new Ctx(m,n,k,fp32,mode,ws);}catch(const std::exception& e){error=e.what();return nullptr;}
}
extern "C" void* lm_create_fixed(int m,int n,int k,size_t ws) {
  try {return new Ctx(m,n,k,0,1,ws,false);}catch(const std::exception& e){error=e.what();return nullptr;}
}
extern "C" int lm_count(void* p){return ((Ctx*)p)->algos.size();}
extern "C" size_t lm_algo_size(){return sizeof(cublasLtMatmulAlgo_t);}
extern "C" void lm_export(void* p,int idx,void* data){auto* ctx=(Ctx*)p;std::memcpy(data,&ctx->algos.at(idx),sizeof(cublasLtMatmulAlgo_t));}
extern "C" int lm_import(void* p,void* data){
  auto* ctx=(Ctx*)p;cublasLtMatmulAlgo_t algo;std::memcpy(&algo,data,sizeof(algo));ctx->add(algo);
  for(size_t i=0;i<ctx->algos.size();i++)if(!std::memcmp(&algo,&ctx->algos[i],sizeof(algo)))return i;
  return -1;
}
extern "C" int lm_info(void* p,int idx,int* values) {
  auto* ctx=(Ctx*)p;size_t sz=0;
  cublasLtMatmulAlgoConfigAttributes_t attrs[]={CUBLASLT_ALGO_CONFIG_ID,CUBLASLT_ALGO_CONFIG_TILE_ID,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,CUBLASLT_ALGO_CONFIG_STAGES_ID,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION};
  for(int i=0;i<7;i++){values[i]=-1;cublasLtMatmulAlgoConfigGetAttribute(&ctx->algos.at(idx),attrs[i],&values[i],sizeof(int),&sz);}return 0;
}
extern "C" size_t lm_workspace(void* p,int idx) {
  auto* ctx=(Ctx*)p;cublasLtMatmulHeuristicResult_t h{};
  if(cublasLtMatmulAlgoCheck(ctx->lt,ctx->op,ctx->a,ctx->b,ctx->c,ctx->c,&ctx->algos.at(idx),&h))return size_t(-1);
  return h.workspaceSize;
}
extern "C" int lm_run_sized(void* p,int idx,void* w,void* x,void* y,void* workspace,size_t bytes,void* stream) {
  auto* ctx=(Ctx*)p;float alpha=1,beta=0;
  auto rc=cublasLtMatmul(ctx->lt,ctx->op,&alpha,w,ctx->a,x,ctx->b,&beta,y,ctx->c,y,ctx->c,
    &ctx->algos.at(idx),workspace,bytes,(cudaStream_t)stream);
  if(rc)error="matmul status="+std::to_string(int(rc));return rc;
}
extern "C" int lm_run(void* p,int idx,void* w,void* x,void* y,void* workspace,void* stream) {
  return lm_run_sized(p,idx,w,x,y,workspace,((Ctx*)p)->workspace,stream);
}
extern "C" void lm_destroy(void* p){delete (Ctx*)p;}
