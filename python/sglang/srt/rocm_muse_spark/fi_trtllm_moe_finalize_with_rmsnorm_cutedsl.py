"""CuTeDSL kernel for fused MOE finalize with RMSNorm.

Used with FlashInfer TRTLLM NVFP4 MoE and Muse Spark post-expert
normalization.

The kernel fuses three operations per token:
1. Gather expert output from permuted positions (unpermute)
2. Apply weight-first RMSNorm (multiply by weight+1.0, then normalize)
3. Weighted accumulation across top-K experts

Uses vectorized 128-bit tiled copies (cute.copy with CopyUniversalOp) for
global memory loads and stores. TensorSSA is used for register-level
computation (weight scaling, sum-of-squares reduction, normalization).

Algorithm for each token (1 block per token):
  For each expert k in topK:
    - Gather expert output from gemm2_output[permuted_idx] (skip if -1)
    - Vectorized 128-bit load of expert row
    - Apply weight scaling: scaled = input * rms_norm_weight
    - Compute sum_sq via TensorSSA reduce + warp + cross-warp reduction
    - Compute inv_rms = rsqrt(sum_sq / hidden_dim + 1e-8)
    - Normalize: normalized = scaled * inv_rms
    - Accumulate: output += expert_weight * normalized

The public API is ``FusedPostExpertNormWeightedSumTRTLLM`` in
``fused_post_expert_norm_weighted_sum_trtllm.py``.  This module exposes
only the CuTeDSL kernel infrastructure and compilation cache.
"""

import functools
import operator

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    from cutlass import Float32, Int32, Int64, cute, utils
    from cutlass.cute.runtime import make_ptr

    HAS_CUTEDSL = True
except ImportError:
    HAS_CUTEDSL = False


# =====================================================================
# CuTeDSL kernel
# =====================================================================

if HAS_CUTEDSL:
    _torch_to_cutlass_dtype = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }

    @functools.lru_cache(maxsize=16)
    def _get_sm_version(device=None) -> int:
        if not torch.cuda.is_available():
            return 80
        props = torch.cuda.get_device_properties(device)
        return props.major * 10 + props.minor

    # =================================================================
    # Configuration
    # =================================================================

    class MOEFinalizeRMSNormConfig:
        """Configuration for the MOE finalize + RMSNorm kernel.

        One token per block, all threads process the same row.
        Uses vectorized 128-bit tiled copies for global memory access.
        """

        COPY_BITS = 128

        def __init__(self, dtype: type, hidden_dim: int, threads_per_row=None):
            self.dtype = dtype
            self.N = hidden_dim
            self.vec_size = self.COPY_BITS // dtype.width

            self.threads_per_row = (
                threads_per_row
                if threads_per_row is not None
                else self._compute_threads_per_row(hidden_dim, self.vec_size)
            )
            self.num_threads = self.threads_per_row
            self.rows_per_block = 1

            self.num_iters = (
                hidden_dim + self.threads_per_row * self.vec_size - 1
            ) // (self.threads_per_row * self.vec_size)
            self.elems_per_thread = self.vec_size * self.num_iters
            self.warps_per_row = max(self.threads_per_row // 32, 1)

        @staticmethod
        def _compute_threads_per_row(N: int, vec_size: int) -> int:
            # Initial preference based on dimension size
            if N <= 64:
                tpr = 8
            elif N <= 128:
                tpr = 16
            elif N <= 2048:
                tpr = 32
            elif N <= 2560:
                tpr = 64
            elif N <= 16384:
                tpr = 128
            else:
                tpr = 256
            # Ensure N is evenly divisible by (tpr * vec_size)
            while tpr > 1 and N % (tpr * vec_size) != 0:
                tpr //= 2
            return tpr

        def _tv_layout(self):
            """Compute tiled-copy thread-value layout for vectorized access.

            Returns (tiler_mn, tv_layout) where:
            - tiler_mn: tile shape (rows_per_block, cols) for local_tile
            - tv_layout: thread-value layout for make_tiled_copy
            """
            vecsize = self.vec_size
            tpr = self.threads_per_row
            num_threads = self.num_threads
            N = self.N

            num_cols_vec = cute.ceil_div(N, vecsize)
            num_blocks_N = cute.ceil_div(num_cols_vec, tpr)
            cols_per_block = num_threads // tpr
            tiler_mn = (cols_per_block, vecsize * num_blocks_N * tpr)
            tv_layout = cute.make_layout(
                ((tpr, cols_per_block), (vecsize, num_blocks_N)),
                stride=(
                    (vecsize * cols_per_block, 1),
                    (cols_per_block, cols_per_block * vecsize * tpr),
                ),
            )
            return tiler_mn, tv_layout

        def smem_size_in_bytes(self) -> int:
            """Shared memory for cross-warp reduction buffer only."""
            return self.rows_per_block * self.warps_per_row * 4

    # =================================================================
    # Kernel
    # =================================================================

    class MOEFinalizeRMSNormKernel:
        """CuTeDSL kernel: unpermute + RMSNorm + weighted accumulate.

        Supports both weight-first (``rms_norm(weight*x)``) and standard
        (``rms_norm(x)*weight``) forms, parametrised by ``weight_first`` and
        ``eps`` so the kernel reproduces the same math as the Python
        ``MuseSpark1xRMSNorm`` modules.

        Uses vectorized 128-bit tiled copies for global memory access and
        TensorSSA for register-level computation.
        """

        def __init__(
            self,
            dtype: type,
            hidden_dim: int,
            topk: int,
            hidden_dim_padded: int,
            threads_per_row=None,
            use_pdl=False,
            eps: float = 1e-8,
            weight_first: bool = True,
        ):
            self.cfg = MOEFinalizeRMSNormConfig(dtype, hidden_dim, threads_per_row)
            self.topk = topk
            self.hidden_dim_padded = hidden_dim_padded
            self.use_pdl = use_pdl
            self.eps = float(eps)
            self.weight_first = bool(weight_first)

        @cute.jit
        def __call__(
            self,
            gemm2_ptr: cute.Pointer,
            expert_weights_ptr: cute.Pointer,
            idx_map_ptr: cute.Pointer,
            rms_weight_ptr: cute.Pointer,
            out_ptr: cute.Pointer,
            num_tokens: Int32,
            stream: cuda.CUstream,
        ):
            """Host function to launch the kernel."""
            cfg = self.cfg
            hidden_dim = cfg.N
            topk = self.topk
            hidden_dim_padded = self.hidden_dim_padded

            mGemm2 = cute.make_tensor(
                gemm2_ptr,
                cute.make_layout(
                    (num_tokens * topk, hidden_dim),
                    stride=(hidden_dim_padded, 1),
                ),
            )

            mOut = cute.make_tensor(
                out_ptr,
                cute.make_layout((num_tokens, hidden_dim), stride=(hidden_dim, 1)),
            )

            mW = cute.make_tensor(
                rms_weight_ptr,
                cute.make_layout((1, hidden_dim), stride=(hidden_dim, 1)),
            )

            mExpW = cute.make_tensor(
                expert_weights_ptr,
                cute.make_layout((num_tokens * topk,), stride=(1,)),
            )

            mIdx = cute.make_tensor(
                idx_map_ptr,
                cute.make_layout((num_tokens * topk,), stride=(1,)),
            )

            self.kernel(mGemm2, mOut, mW, mExpW, mIdx).launch(
                grid=[num_tokens, 1, 1],
                block=[cfg.num_threads, 1, 1],
                smem=cfg.smem_size_in_bytes(),
                stream=stream,
                use_pdl=self.use_pdl,
            )

        @cute.kernel
        def kernel(
            self,
            mGemm2: cute.Tensor,
            mOut: cute.Tensor,
            mW: cute.Tensor,
            mExpW: cute.Tensor,
            mIdx: cute.Tensor,
        ):
            """Device kernel using vectorized 128-bit tiled copies."""
            cfg = self.cfg
            topk = self.topk
            tidx = cute.arch.thread_idx()[0]
            token_idx = cute.arch.block_idx()[0]

            # ---- PDL: wait for prior kernel to complete ----
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            hidden_dim = cfg.N
            threads_per_row = cfg.threads_per_row
            warps_per_row = cfg.warps_per_row

            tiler_mn, tv_layout = cfg._tv_layout()

            # Shared memory for reduction
            smem = utils.SmemAllocator()
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((1, warps_per_row)),
                byte_alignment=4,
            )

            # Tiled copy setup
            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                mGemm2.element_type,
                num_bits_per_copy=cfg.COPY_BITS,
            )
            thr_copy = cute.make_tiled_copy(copy_atom, tv_layout, tiler_mn).get_slice(
                tidx
            )

            # Load weight vectorized
            gW = cute.local_tile(mW, tiler_mn, (0, 0))
            tXgW = thr_copy.partition_S(gW)
            w_frag = cute.make_fragment_like(tXgW)
            cute.copy(copy_atom, tXgW, w_frag)

            # w as TensorSSA (already has +1.0 pre-applied on host)
            w_f32 = w_frag.load().to(cute.Float32)

            # Output tile
            gO = cute.local_tile(mOut, tiler_mn, (token_idx, 0))
            tXgO = thr_copy.partition_D(gO)

            # Accumulator and input fragments
            acc_frag = cute.make_fragment_like(tXgW)
            acc_frag.fill(0)
            input_frag = cute.make_fragment_like(tXgW)

            # Expert loop
            for k in range(topk):
                expanded_idx = token_idx * topk + k
                permuted_idx = Int32(mIdx[expanded_idx])
                expert_weight = Float32(mExpW[expanded_idx])

                if permuted_idx >= 0:
                    # Vectorized 128-bit load of expert row
                    gInput = cute.local_tile(mGemm2, tiler_mn, (Int64(permuted_idx), 0))
                    tXgInput = thr_copy.partition_S(gInput)
                    cute.copy(copy_atom, tXgInput, input_frag)

                    input_f32 = input_frag.load().to(cute.Float32)
                    if cutlass.const_expr(self.weight_first):
                        scaled_for_rms = input_f32 * w_f32
                    else:
                        scaled_for_rms = input_f32

                    # Preserve FMA reduction semantics across CuTeDSL versions.
                    sum_sq_local = Float32(0.0)
                    for value_idx in cutlass.range_constexpr(cfg.elems_per_thread):
                        value = scaled_for_rms[value_idx]
                        sum_sq_local = cute.fma(value, value, sum_sq_local)

                    # Warp reduction
                    warp_width = min(threads_per_row, 32)
                    warp_sum = cute.arch.warp_reduction(
                        sum_sq_local,
                        operator.add,
                        threads_in_group=warp_width,
                    )

                    # Cross-warp reduction
                    if cutlass.const_expr(warps_per_row > 1):
                        lane_idx = cute.arch.lane_idx()
                        warp_idx = cute.arch.warp_idx()
                        col_idx = warp_idx % warps_per_row
                        if lane_idx == 0:
                            reduction_buffer[0, col_idx] = warp_sum
                        cute.arch.barrier()
                        block_val = Float32(0.0)
                        if lane_idx < warps_per_row:
                            block_val = reduction_buffer[0, lane_idx]
                        sum_sq = cute.arch.warp_reduction(block_val, operator.add)
                    else:
                        sum_sq = warp_sum

                    # inv_rms — eps is a Python float, capture as compile-time const
                    eps_val = Float32(self.eps)
                    mean_sq = sum_sq / hidden_dim
                    inv_rms = cute.math.rsqrt(mean_sq + eps_val, fastmath=True)

                    # Barrier after reduction
                    if cutlass.const_expr(warps_per_row > 1):
                        cute.arch.barrier()

                    # Normalize, weight, and accumulate
                    if cutlass.const_expr(self.weight_first):
                        normalized = scaled_for_rms * inv_rms
                    else:
                        normalized = scaled_for_rms * inv_rms * w_f32
                    weighted = normalized * expert_weight

                    acc_f32 = acc_frag.load().to(cute.Float32)
                    new_acc = acc_f32 + weighted
                    acc_frag.store(new_acc.to(acc_frag.element_type))

            # Write output (vectorized store)
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    mOut.element_type,
                    num_bits_per_copy=cfg.COPY_BITS,
                ),
                acc_frag,
                tXgO,
            )

            # ---- PDL: signal next kernel can start ----
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()

    # =================================================================
    # Compilation cache
    # =================================================================

    _compile_cache: dict = {}

    def _get_compiled_kernel(
        dtype: type,
        hidden_dim: int,
        topk: int,
        hidden_dim_padded: int,
        stream: cuda.CUstream,
        threads_per_row=None,
        use_pdl=False,
        eps: float = 1e-8,
        weight_first: bool = True,
    ):
        key = (
            dtype,
            hidden_dim,
            topk,
            hidden_dim_padded,
            threads_per_row,
            use_pdl,
            eps,
            weight_first,
        )
        if key not in _compile_cache:
            kernel_obj = MOEFinalizeRMSNormKernel(
                dtype,
                hidden_dim,
                topk,
                hidden_dim_padded,
                threads_per_row,
                use_pdl=use_pdl,
                eps=eps,
                weight_first=weight_first,
            )
            compiled = cute.compile(
                kernel_obj,
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(
                    cutlass.BFloat16,
                    16,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(
                    cutlass.Int32,
                    16,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                Int32(1),  # num_tokens
                stream,
                options="--enable-tvm-ffi",
            )
            _compile_cache[key] = compiled
        return _compile_cache[key]
