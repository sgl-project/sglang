/* Copyright @2020-2026 Moore Threads Technology Co., Ltd("Moore Threads"). All
 * rights reserved.
 *
 * This software ("this software and its documentations" or "the software") is
 * protected by Copyright and the information contained herein is confidential.
 *
 * The software contained herein is PROPRIETARY to Moore Threads and is being
 * provided under the terms and conditions of a form of Moore Threads software
 * license agreement by and between Moore Threads and Licensee ("License
 * Agreement") or electronically accepted by Licensee. Notwithstanding any
 * terms or conditions to the contrary in the License Agreement, copy or
 * disclosure of the software to any third party without the express written
 * consent of Moore Threads is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE LICENSE
 * AGREEMENT, MOORE THREADS MAKES NO REPRESENTATION ABOUT ANY WARRANTIES,
 * INCLUDING BUT NOT LIMITED TO THE SUITABILITY OF THE SOFTWARE FOR ANY
 * PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF
 * ANY KIND. MOORE THREADS DISCLAIMS ALL WARRANTIES WITH REGARD TO THE
 * SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL MOORE THREADS BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THE SOFTWARE.
 */

 #include "musa.h"
 #include <iostream>
 #include <vector>
 #include <cmath>
 #include <musa_runtime.h>
 #include <musa_fp16.h>
 #include "musa_bf16.h"
 #include <musa_robust.h>
 #include <torch/torch.h>
 #include "torch_musa/csrc/core/MUSAGuard.h"
 #include "torch_musa/csrc/core/MUSAStream.h"

 typedef __half float16_t;
 typedef __mt_bfloat16 bfloat16_t;

 #define DEVICE_INLINE __device__ __forceinline__
 template <typename T, int width>
 __device__ __forceinline__ T mudnn_shfl_down_sync(T val, unsigned int delta) {
   return __shfl_down_sync(0xffffffff, val, delta, width);
 }

 __device__ __host__ __forceinline__ constexpr int ceil_div(int a, int b) {
   return (a + b - 1) / b;
 }

 __device__ __host__ __forceinline__ constexpr int64_t ceil_div(int64_t a,
                                                                int64_t b) {
   return (a + b - 1) / b;
 }

 #define WARP_THREADS 32
 #define SMEM_STOP (WARP_THREADS / 2)
 #define SHFL_START min(WARP_THREADS / 2, BLOCK_X / 2)
 #define __SYNCTHREADS_LM __syncthreads_lm()
 #define MACRO_UNROLL _Pragma("unroll")
 #define LD_BYP_SLC(_BITS, _BYTES)                                   \
   VecType dst;                                                      \
   const BaseType* addr = ptr + idx;                                 \
   asm volatile("LSU.LD.B" #_BITS " %0, %1, _, " #_BYTES             \
                ", 1, 1, inner_persist=0, "                          \
                "outer_persist=2, chrnt=l2_l3, slc=byp, persist=0, " \
                "stride_add_first=0"                                 \
                : "=R"(dst)                                          \
                : "R"(addr));                                        \
   return dst;

     #define ATTR_ALIGNED(v) __attribute__((aligned(v)))
     #define SELF_VEC_DEF(BASE_TYPE, VEC_TYPE_V2, VEC_TYPE_V4)                  \
       struct ATTR_ALIGNED(sizeof(BASE_TYPE) * 2) VEC_TYPE_V2 {                 \
         __device__ VEC_TYPE_V2() {}                                            \
         __device__ VEC_TYPE_V2(const VEC_TYPE_V2& t) {                         \
           this->x = t.x;                                                       \
           this->y = t.y;                                                       \
         }                                                                      \
         BASE_TYPE x, y;                                                        \
       };                                                                       \
                                                                                \
       __device__ __forceinline__ VEC_TYPE_V2 make_##VEC_TYPE_V2(BASE_TYPE x,   \
                                                                 BASE_TYPE y) { \
         VEC_TYPE_V2 t;                                                         \
         t.x = x, t.y = y;                                                      \
         return t;                                                              \
       }                                                                        \
                                                                                \
       struct ATTR_ALIGNED(sizeof(BASE_TYPE) * 4) VEC_TYPE_V4 {                 \
         __device__ VEC_TYPE_V4() {}                                            \
         __device__ VEC_TYPE_V4(const VEC_TYPE_V4& t) {                         \
           this->x = t.x;                                                       \
           this->y = t.y;                                                       \
           this->z = t.z;                                                       \
           this->w = t.w;                                                       \
         }                                                                      \
         BASE_TYPE x, y, z, w;                                                  \
       };                                                                       \
                                                                                \
       __device__ __forceinline__ VEC_TYPE_V4 make_##VEC_TYPE_V4(               \
           BASE_TYPE x, BASE_TYPE y, BASE_TYPE z, BASE_TYPE w) {                \
         VEC_TYPE_V4 t;                                                         \
         t.x = x, t.y = y, t.z = z, t.w = w;                                    \
         return t;                                                              \
       }

     SELF_VEC_DEF(float16_t, Half2, Half4)
     SELF_VEC_DEF(bfloat16_t, Bhalf2, Bhalf4)

     #define GEN_VECTYPE(_CTYPE, _VECTYPE, _BYTES, _VLEN) \
       struct ATTR_ALIGNED(_BYTES) _VECTYPE {             \
         __device__ _VECTYPE() {}                         \
         __device__ _VECTYPE(const _VECTYPE& t) {         \
           MACRO_UNROLL                                   \
           for (int i = 0; i < _VLEN; i++) {              \
             this->arr[i] = t.arr[i];                     \
           }                                              \
         }                                                \
         _CTYPE arr[_VLEN];                               \
       }

 GEN_VECTYPE(float16_t, Half8, 16, 8);
 GEN_VECTYPE(bfloat16_t, Bhalf8, 16, 8);
 template <typename type>
 class Dtype;

 #define INST(_type, _vec2, _vec4)                                        \
   template <>                                                            \
   class Dtype<_type> {                                                   \
    public:                                                               \
     using Scalar = _type;                                                \
     using Vec2 = _vec2;                                                  \
     using Vec4 = _vec4;                                                  \
     static __device__ __forceinline__ Vec2 make_vec2(_type x, _type y) { \
       return make_##_vec2(x, y);                                         \
     }                                                                    \
     static __device__ __forceinline__ Vec4 make_vec4(_type x, _type y,   \
                                                      _type z, _type w) { \
       return make_##_vec4(x, y, z, w);                                   \
     }                                                                    \
   }

 INST(float, float2, float4);
 INST(bfloat16_t, Bhalf2, Bhalf4);

 template <typename T, int bits = 16 * 8>
 struct VecType;

 template <typename T>
 struct DeduceVectorizedType {
   using Type = T;
 };

 template <>
 struct DeduceVectorizedType<half> {
   using Type = _Float16;
 };

 template <>
 struct DeduceVectorizedType<bfloat16_t> {
   using Type = _Float16;
 };

 #define DEF_VECT(_CTYPE, _VECTYPE)                                            \
     template <>                                                               \
     struct VecType<_CTYPE, sizeof(_VECTYPE) * 8> {                            \
     static constexpr int vec_bytes = sizeof(_VECTYPE);                        \
     static constexpr int bit_per_byte = 8;                                    \
     using BaseType = _CTYPE;                                                  \
     using RobustTypePtr = __musa::robust_ptr<_CTYPE>;                         \
     using Ttype = _VECTYPE;                                                   \
     static constexpr int bits = vec_bytes * bit_per_byte;                     \
     static constexpr int vlen = bits / (sizeof(BaseType) * bit_per_byte);     \
     using VectorizedType = typename DeduceVectorizedType<BaseType>::Type;     \
     typedef VectorizedType VxTy __attribute__((vector_size(vec_bytes)));      \
     template <typename OffsetType>                                            \
     static __device__ __forceinline__ VecType load(const BaseType* ptr,       \
                                                     OffsetType idx) {         \
         return *(VecType*)(ptr + idx);                                        \
     }                                                                         \
     template <typename OffsetType>                                            \
     static __device__ __forceinline__ VecType                                 \
     load_byp_slc(const BaseType* ptr, OffsetType idx) {                       \
         if constexpr (vec_bytes == 16) {                                      \
         LD_BYP_SLC(128, 16);                                                  \
         } else if constexpr (vec_bytes == 8) {                                \
         LD_BYP_SLC(64, 8);                                                    \
         } else if constexpr (vec_bytes == 4) {                                \
         LD_BYP_SLC(32, 4);                                                    \
         } else if constexpr (vec_bytes == 2) {                                \
         LD_BYP_SLC(16, 2);                                                    \
         } else {                                                              \
         LD_BYP_SLC(8, 1);                                                     \
         }                                                                     \
     }                                                                         \
     template <typename OffsetType>                                            \
     static __device__ __forceinline__ VecType                                 \
     robust_load(const RobustTypePtr ptr, OffsetType idx) {                    \
         return __musa::robust_load<VecType, BaseType>(ptr, idx);              \
     }                                                                         \
                                                                               \
     template <typename OffsetType>                                            \
     static __device__ __forceinline__ void store(BaseType* ptr,               \
                                                     OffsetType idx,           \
                                                     const VecType& dst) {     \
         *(VecType*)(ptr + idx) = dst;                                         \
     }                                                                         \
     template <typename OffsetType>                                            \
     static __device__ __forceinline__ void robust_store(RobustTypePtr ptr,    \
                                                         OffsetType idx,       \
                                                         const VecType& dst) { \
         __musa::robust_store<VecType, BaseType>(dst, ptr, idx);               \
     }                                                                         \
                                                                               \
     __device__ VecType() {                                                    \
         MACRO_UNROLL                                                          \
         for (int i = 0; i < sizeof(Ttype) / sizeof(BaseType); i++) {          \
         this->val_.elem[i] = 0;                                               \
         }                                                                     \
     }                                                                         \
     __device__ VecType(const VecType& t) {                                    \
         MACRO_UNROLL                                                          \
         for (int i = 0; i < sizeof(Ttype) / sizeof(BaseType); i++) {          \
         this->val_.elem[i] = t.val_.elem[i];                                  \
         }                                                                     \
     }                                                                         \
     __device__ VecType& operator=(const VecType& t) {                         \
         MACRO_UNROLL                                                          \
         for (int i = 0; i < sizeof(Ttype) / sizeof(BaseType); i++) {          \
         this->val_.elem[i] = t.val_.elem[i];                                  \
         }                                                                     \
         return *this;                                                         \
     }                                                                         \
     __device__ VecType(_CTYPE val) {                                          \
         MACRO_UNROLL                                                          \
         for (int i = 0; i < sizeof(Ttype) / sizeof(BaseType); i++) {          \
         this->val_.elem[i] = val;                                             \
         }                                                                     \
     }                                                                         \
     template <typename SrcVecType>                                            \
     friend __device__ VecType operator+(VecType lhs, const SrcVecType& rhs) { \
         MACRO_UNROLL                                                          \
         for (int i = 0; i < sizeof(Ttype) / sizeof(BaseType); i++) {          \
         lhs.val_.elem[i] += static_cast<BaseType>(rhs.val_.elem[i]);          \
         }                                                                     \
         return lhs;                                                           \
     }                                                                         \
     friend __device__ VecType operator+(VecType lhs, const _CTYPE& rhs) {     \
         MACRO_UNROLL                                                          \
         for (int i = 0; i < sizeof(Ttype) / sizeof(BaseType); i++) {          \
         lhs.val_.elem[i] += rhs;                                              \
         }                                                                     \
         return lhs;                                                           \
     }                                                                         \
     friend __device__ VecType operator-(VecType lhs, const VecType& rhs) {    \
         MACRO_UNROLL                                                          \
         for (int i = 0; i < sizeof(Ttype) / sizeof(BaseType); i++) {          \
         lhs.val_.elem[i] -= rhs.val_.elem[i];                                 \
         }                                                                     \
         return lhs;                                                           \
     }                                                                         \
     friend __device__ VecType operator*(VecType lhs, const VecType& rhs) {    \
         MACRO_UNROLL                                                          \
         for (int i = 0; i < sizeof(Ttype) / sizeof(BaseType); i++) {          \
         lhs.val_.elem[i] *= rhs.val_.elem[i];                                 \
         }                                                                     \
         return lhs;                                                           \
     }                                                                         \
     template <typename Func>                                                  \
     __device__ VecType& apply() {                                             \
         MACRO_UNROLL                                                          \
         for (int i = 0; i < sizeof(Ttype) / sizeof(BaseType); i++) {          \
         this->val_.elem[i] = Func::apply(this->val_.elem[i]);                 \
         }                                                                     \
         return *this;                                                         \
     }                                                                         \
     template <typename SrcVecType>                                            \
     static __device__ VecType cvt(const SrcVecType& src) {                    \
         VecType dst;                                                          \
         MACRO_UNROLL                                                          \
         for (int i = 0; i < sizeof(Ttype) / sizeof(BaseType); i++) {          \
         dst.val_.elem[i] = (BaseType)(src.val_.elem[i]);                      \
         }                                                                     \
         return dst;                                                           \
     }                                                                         \
     union U {                                                                 \
         __device__ U() {                                                      \
         MACRO_UNROLL                                                          \
         for (int i = 0; i < sizeof(Ttype) / sizeof(BaseType); i++) {          \
             this->elem[i] = 0;                                                \
         }                                                                     \
         }                                                                     \
         Ttype storage;                                                        \
         BaseType elem[sizeof(Ttype) / sizeof(BaseType)];                      \
         VxTy vt_elem;                                                         \
     };                                                                        \
     U val_{};                                                                 \
     }
 DEF_VECT(float16_t, float16_t);
 DEF_VECT(float16_t, Half2);
 DEF_VECT(bfloat16_t, bfloat16_t);
 DEF_VECT(bfloat16_t, Bhalf2);
 DEF_VECT(bfloat16_t, Bhalf8);
 DEF_VECT(float16_t, Half8);
 DEF_VECT(float, float4);

 enum class VarUpdateMode { WELFORD, WELFORD_ONLY_MEAN, CHAN, CHAN_ONLY_MEAN };

 static __device__ __forceinline__ float fast_rcpf(float x) {
   float y = __frcp_rn(x);
   y = y * (2.0 - x * y);
   return y;
 }

 static __device__ __forceinline__ float fast_divf(float a, float b) {
   return a * fast_rcpf(b);
 }

 static __device__ __forceinline__ float fast_rsqrtf(float a) {
   float x = 0.5 * a;
   float y = __frsqrt_rn(a);
   y = y * (1.5 - x * y * y);
   return y;
 }

 template <typename T, VarUpdateMode Mode>
 struct VarUpdate;

 template <typename T>
 struct VarUpdate<T, VarUpdateMode::WELFORD_ONLY_MEAN> {
     DEVICE_INLINE void apply(T curr, T* mu, T* cnt) {
     *cnt += 1;
     T delta = curr - *mu;
     *mu += fast_divf(delta, *cnt);
     }
 };

 template <typename T>
 struct VarUpdate<T, VarUpdateMode::CHAN_ONLY_MEAN> {
   DEVICE_INLINE void apply(T mu_B, T cnt_B, T* mu, T* cnt) {
     if (cnt_B > 0) {
       T n_AB = cnt_B + (*cnt);
       T delta = mu_B - (*mu);
       *mu += delta * fast_divf(cnt_B, n_AB);
       *cnt = n_AB;
     }
   }
 };

 template <typename ComputeType, int BLOCK_X, int BLOCK_Y,int Vlen>
 struct AllReduceOp {
   DEVICE_INLINE void apply(ComputeType* sum, int tx, int ty) {
     __shared__ ComputeType __attribute__((aligned(16)))
     smem[BLOCK_X * BLOCK_Y * Vlen];
     ComputeType* smem_sum = &smem[0];

       static_assert(Vlen == 1,
                     "Axis COLUMN doesn't support vlen greater than 1");
 #pragma unroll
       for (int offset = BLOCK_X / 2; offset > SMEM_STOP; offset /= 2) {
         if (tx >= offset && tx < 2 * offset) {
           smem_sum[ty * BLOCK_X + tx] = *sum;
         }
         __SYNCTHREADS_LM;
         if (tx < offset) {
           *sum += smem_sum[ty * BLOCK_X + tx + offset];
         }
       }
 #if ((defined __MUSA_ARCH__) && (__MUSA_ARCH__ >= 220))
 #pragma unroll
       for (int offset = SHFL_START; offset > 0; offset /= 2) {
         *sum += mudnn_shfl_down_sync<ComputeType, 32>(*sum, offset);
       }
 #endif
       if (tx == 0) {
         smem_sum[ty * BLOCK_X + tx] = *sum;
       }
       __SYNCTHREADS_LM;
       *sum = smem_sum[ty * BLOCK_X];
   }
 };

  template <typename SrcDtype, typename ComputeType, int BLOCK_X, int BLOCK_Y, int vlen>
  __global__ void LayerNormGlobalKernelVlen(
     SrcDtype* input, SrcDtype* residual, const SrcDtype* weight,
      const size_t M, const size_t N, const ComputeType eps) {
    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    size_t m_idx = blockIdx.x * blockDim.y + ty;
    size_t n_idx = tx * vlen;
    size_t n_step = (size_t)blockDim.x * vlen;

    using SrcVec = VecType<SrcDtype, vlen * sizeof(SrcDtype) * 8>;

    ComputeType var = 0;
    const SrcDtype* __restrict p_src = input + m_idx * N;
    SrcDtype* __restrict p_res = residual + m_idx * N;  // residual ptr

    // TODO(wuke): use robust_load, robust_store
    bool m_valid = m_idx < M;
    if (m_valid) {
      for (size_t j = n_idx; j < N; j += n_step) {
        SrcVec curr, res_vec, fused_vec;
        #if ((defined __MUSA_ARCH__) && (__MUSA_ARCH__ == 220))
            curr = *(SrcVec *)(p_src+j);
            res_vec = *(SrcVec *)(p_res+j);
        #elif ((defined __MUSA_ARCH__) && (__MUSA_ARCH__ == 310))
            curr = SrcVec::load_byp_slc(p_src, j);
            res_vec = SrcVec::load_byp_slc(p_res, j);
        #endif
        #pragma unroll
        for (int k = 0; k < vlen; k++) {
         fused_vec.val_.elem[k] = curr.val_.elem[k] + res_vec.val_.elem[k];
         var += (ComputeType)fused_vec.val_.elem[k] * (ComputeType)fused_vec.val_.elem[k];
        }
        *(SrcVec*)(p_res + j) = fused_vec;
      }
    }
    AllReduceOp<ComputeType, BLOCK_X, BLOCK_Y, 1> all_reduce_op;
    all_reduce_op.apply(&var, tx, ty);
    if (m_valid) {
      ComputeType inv_var = fast_rsqrtf(var / N + eps);
      SrcDtype* __restrict p_dst = input + m_idx * N;
      bool with_weight = (weight != NULL);
      if (with_weight) {
        for (size_t j = n_idx; j < N; j += n_step) {
          SrcVec fused_vec, weight_val, dst;
          #if ((defined __MUSA_ARCH__) && (__MUSA_ARCH__ == 220))
            fused_vec = *(SrcVec *)(p_res+j);
            weight_val = *(SrcVec *)(weight+j);
          #elif ((defined __MUSA_ARCH__) && (__MUSA_ARCH__ == 310))
            fused_vec = SrcVec::load_byp_slc(p_res, j);
            weight_val = SrcVec::load_byp_slc(weight, j);
          #endif
  #pragma unroll
          for (int k = 0; k < vlen; k++) {
             dst.val_.elem[k] = (SrcDtype)((ComputeType)fused_vec.val_.elem[k] * inv_var *
                             (ComputeType)weight_val.val_.elem[k]);
          }
          *(SrcVec*)(p_dst + j) = dst;
        }
      }
    }
  }

 #define CALL_KERN(_SRC_DTYPE,_KERN, _BLKX, _BLKY, _VLEN)                      \
 {                                                                             \
     const uint32_t block_x = _BLKX;                                           \
     const uint32_t block_y = _BLKY;                                           \
     const uint32_t nr_blocks = ceil_div(m, (size_t)block_y);                  \
     dim3 block_size{block_x, block_y, 1};                                     \
     dim3 grid_size{nr_blocks, 1, 1};                                          \
     LayerNorm##_KERN##KernelVlen<_SRC_DTYPE, float,                           \
                                      block_x, block_y, _VLEN>                 \
             <<<grid_size, block_size, 0, stream>>>(                           \
                 static_cast<_SRC_DTYPE*>(input),                              \
                 static_cast<_SRC_DTYPE*>(residual),                           \
                 static_cast<_SRC_DTYPE*>(weight),                             \
                 m, n, static_cast<float>(epsilon));                           \
 }

 #define DISPATCH_KERNEL(_KERN, _BLKX, _BLKY)                                 \
   if constexpr (std::is_same_v<SrcDtype,float16_t>) {                        \
      CALL_KERN(float16_t, _KERN, _BLKX, _BLKY, 8);                           \
   } else if constexpr (std::is_same_v<SrcDtype, bfloat16_t>) {               \
     CALL_KERN(bfloat16_t, _KERN, _BLKX, _BLKY, 8);                           \
   } else if constexpr (std::is_same_v<SrcDtype, float>) {                    \
     CALL_KERN(float, _KERN, _BLKX, _BLKY, 4);                                \
   }

 template <typename SrcDtype>
 void rms_fused_add_rms_norm(SrcDtype* input, SrcDtype* residual, SrcDtype* weight, int m, int n, double epsilon) {
   auto stream = c10::musa::getCurrentMUSAStream().stream();
   DISPATCH_KERNEL(Global, 1024, 1);
 }

void musa_fused_add_rms_norm(
    torch::Tensor &input,
    torch::Tensor &residual,
    torch::Tensor &weight,
    double epsilon,
    bool enable_pdl) {

    int m = input.size(0);
    int n = input.size(1);

    const at::musa::OptionalMUSAGuard device_guard(device_of(input));

    if (input.scalar_type() == at::ScalarType::BFloat16)
    {
        rms_fused_add_rms_norm<__mt_bfloat16>(
            static_cast<__mt_bfloat16*>(input.data_ptr()),
            static_cast<__mt_bfloat16*>(residual.data_ptr()),
            static_cast<__mt_bfloat16*>(weight.data_ptr()),
            m,
            n,
            epsilon);
    }
    else if (input.scalar_type() == at::ScalarType::Half)
    {
        rms_fused_add_rms_norm<__half>(
            static_cast<__half*>(input.data_ptr()),
            static_cast<__half*>(residual.data_ptr()),
            static_cast<__half*>(weight.data_ptr()),
            m,
            n,
            epsilon);
    }
    else if (input.scalar_type() == at::ScalarType::Float)
    {
        rms_fused_add_rms_norm<float>(
            static_cast<float*>(input.data_ptr()),
            static_cast<float*>(residual.data_ptr()),
            static_cast<float*>(weight.data_ptr()),
            m,
            n,
            epsilon);
    }
    else
    {
        TORCH_CHECK(false, "only support Float32, Half and BFloat16 dtype");
    }
}
