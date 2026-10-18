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

Algorithm for each token (1 block or DSMEM cluster per token):
  For each group of experts in topK:
    - Gather expert output from gemm2_output[permuted_idx] (skip if -1)
    - Vectorized load of expert rows
    - Apply weight scaling: scaled = input * rms_norm_weight
    - Compute sum_sq per expert via local, warp, and CTA/DSMEM reduction
    - Compute inv_rms = rsqrt(sum_sq / hidden_dim + eps)
    - Normalize: normalized = scaled * inv_rms
    - Accumulate: output += expert_weight * normalized
  Convert the FP32 accumulator to the output dtype once.

The public API is ``FusedPostExpertNormWeightedSumTRTLLM`` in
``fused_post_expert_norm_weighted_sum_trtllm.py``.  This module exposes
the CuTeDSL kernel, compilation cache, and borrowed TP push workspace.
"""

import functools

import torch
import torch.distributed as dist


class FinalizePushWorkspace:
    """Borrow TP push buffers before capture; dependency-ordered launches only."""

    def __init__(
        self,
        communicator,
        device_group: dist.ProcessGroup,
        hidden_dim=4096,
        *,
        multicast=False,
    ):
        self.communicator = communicator  # Keep the allocation owner alive.
        self.group = device_group
        self.rank = communicator.rank
        self.world_size = communicator.world_size
        self.device = communicator.device
        self.sm_count = torch.cuda.get_device_properties(
            self.device
        ).multi_processor_count
        self.hidden_dim = hidden_dim
        self.slot_bytes = communicator.max_push_size
        self.peers = communicator._push_workspaces
        self.counter = communicator._push_counter
        self.multicast = multicast
        self.multicast_ptr = communicator._push_multicast_ptr
        config = (
            self.controller_count,
            self.slot_bytes,
            self.sm_count,
            hidden_dim,
            multicast,
        )
        configs = [None] * self.world_size
        dist.all_gather_object(configs, config, group=communicator.group)
        if any(peer != config for peer in configs):
            raise ValueError(
                "Finalize AR workspace configuration must match across ranks"
            )
        if multicast and not communicator.has_multicast:
            raise ValueError(
                "Finalize multicast requires a multicast-capable push workspace"
            )
        if not 0 < self.controller_count <= self.sm_count:
            raise ValueError("Unsupported finalize AR controller budget")
        if hidden_dim <= 0 or hidden_dim % 256 or self.slot_bytes < hidden_dim * 2:
            raise ValueError(
                "Finalize push requires H divisible by 256 and sufficient buffer capacity"
            )
        self.max_tokens = min(384, self.slot_bytes // (hidden_dim * 2))
        self.fallback_logged = False
        self.fusion_logged = False

    def uses_one_shot_push(self, num_tokens: int) -> bool:
        return self.communicator.uses_one_shot_push(num_tokens * self.hidden_dim * 2)

    def can_fuse(self, tp_group, num_tokens: int) -> bool:
        """Use custom AR storage while respecting explicit algorithm overrides.

        Fusion has a different crossover from a standalone AR launch. Allow
        push through 384 tokens when its allocated slot fits, even if ordinary
        CUDA-graph AR would select pull above its 288-token crossover (H=4096),
        but not where the standalone AR would pick 2shot_push.
        No standalone AR thresholds or buffer allocations are changed.
        """
        if (
            torch.compiler.is_compiling()
            or self.communicator is not tp_group.ca_comm
            or self.communicator.disabled
            or not 0 < num_tokens <= self.max_tokens
        ):
            return False
        if self.world_size > 4 and not self.uses_one_shot_push(num_tokens):
            return False
        if self.communicator.uses_two_shot_push(num_tokens * self.hidden_dim * 2):
            return False
        output = (
            self.peers[self.rank]
            .view(torch.bfloat16)
            .view(-1)[: num_tokens * self.hidden_dim]
            .view(num_tokens, self.hidden_dim)
        )
        return (
            not tp_group._can_use_flashinfer_allreduce(output)
            and tp_group._resolve_outplace_all_reduce_method(output) == "ca"
            and (
                self.communicator.override_algo is None
                or self.uses_one_shot_push(num_tokens)
            )
        )

    @property
    def controller_count(self) -> int:
        return self.counter.numel()

    def validate(self, output: torch.Tensor, cluster_size: int = 1) -> None:
        if (
            output.device != self.device
            or output.dtype != torch.bfloat16
            or output.ndim != 2
            or output.shape[1] != self.hidden_dim
            or output.shape[0] > self.max_tokens
            or not output.is_contiguous()
            or output.data_ptr() % 16
        ):
            raise ValueError("Unsupported output for finalize push workspace")
        if cluster_size not in (1, 2, 4, 8, 16) or cluster_size > self.sm_count:
            raise ValueError("Unsupported cluster size")

        if output.shape[0] == 0:
            raise ValueError("Finalize push requires M>0")
        if cluster_size > 1 and output.shape[0] * cluster_size > self.controller_count:
            raise ValueError(
                "Single-token clusters require M * C <= controller_count; use C=1 otherwise"
            )

    def arguments(self) -> tuple:
        import cutlass
        from cutlass.cute.runtime import make_ptr

        return (
            tuple(make_ptr(cutlass.Uint32, p.data_ptr()) for p in self.peers),
            make_ptr(cutlass.Uint32, self.counter.data_ptr()),
            cutlass.Int32(self.rank),
            cutlass.Int32(self.slot_bytes // 4),
            make_ptr(cutlass.Uint32, self.peers[self.rank].data_ptr()),
            make_ptr(
                cutlass.Uint32, self.multicast_ptr or self.peers[self.rank].data_ptr()
            ),
        )


try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Float32, Int32, Int64
    from cutlass._mlir.dialects import llvm
    from cutlass.cute.runtime import make_ptr
    from cutlass.cutlass_dsl import T, dsl_user_op

    HAS_CUTEDSL = True
except ImportError:
    HAS_CUTEDSL = False


# =====================================================================
# CuTeDSL kernel
# =====================================================================

if HAS_CUTEDSL:
    # =================================================================
    # One-shot push AR helpers
    # =================================================================
    # All ranks use identical collective order and M/H. Wait on predecessor
    # completion before reading shared epochs or reusing inboxes (PDL wait).
    # Two slots suffice: publishing N+2 requires consuming N+1 from every peer;
    # each peer publishes N+1 only after its previous launch cleared N.

    AR_COPY_BITS = 128
    AR_VEC_SIZE = AR_COPY_BITS // cutlass.BFloat16.width
    AR_VEC_WORDS = AR_COPY_BITS // cutlass.Uint32.width

    def make_ar_compile_args(ar_world_size: int):
        """Compilation placeholders matching FinalizePushWorkspace.arguments()."""
        if not ar_world_size:
            return None

        from cutlass.cute.runtime import make_ptr

        def uint32_ptr():
            return make_ptr(
                cutlass.Uint32, 16, cute.AddressSpace.gmem, assumed_align=16
            )

        return (
            tuple(uint32_ptr() for _ in range(ar_world_size)),  # peer inboxes
            uint32_ptr(),  # worker epoch controllers
            Int32(0),  # TP rank
            Int32(1),  # slot size in uint32 elements
            uint32_ptr(),  # local inbox
            uint32_ptr(),  # multicast inbox mapping
        )

    @dsl_user_op
    def _store_multicast(ptr, a, b, c, d, *, loc=None, ip=None):
        llvm.inline_asm(
            None,
            [
                ptr.toint(loc=loc, ip=ip).ir_value(),
                a.ir_value(),
                b.ir_value(),
                c.ir_value(),
                d.ir_value(),
            ],
            "multimem.st.relaxed.sys.global.v4.f32 [$0], {$1, $2, $3, $4};",
            "l,r,r,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @cute.jit
    def push_token(
        acc,
        ar,
        token,
        offset,
        threads: cutlass.Constexpr,
        hidden: cutlass.Constexpr,
        cluster: cutlass.Constexpr,
        epoch,
        multicast: cutlass.Constexpr,
    ):
        peers, _, rank, slot_words, _, mc_workspace = ar
        vec, iters = acc.shape
        assert cutlass.const_expr(vec == AR_VEC_SIZE), "Finalize AR requires vec=8"
        world = len(peers)
        base = Int32(epoch & 1) * world * slot_words
        expert_values = cute.make_rmem_tensor(
            cute.make_layout((vec,)), cutlass.BFloat16
        )
        packed = cute.recast_tensor(expert_values, cutlass.Uint32)
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            cutlass.Uint32,
            num_bits_per_copy=AR_COPY_BITS,
            memory_order=cute.nvgpu.MemoryOrder.RELAXED,
            memory_scope=cute.nvgpu.MemoryScope.SYS,
        )
        # Publish this token without polling. Its inbox slot remains distinct from
        # every later token assigned to this persistent cluster.
        for p in cutlass.range_constexpr(iters):
            # Round once at the finalize boundary, then view pairs as Lamport words.
            expert_values.store(acc[None, p].load().to(cutlass.BFloat16))
            for v in cutlass.range_constexpr(AR_VEC_WORDS):
                value = packed[v]
                if value == 0:
                    value = cutlass.Uint32(0x8000)
                packed[v] = value
            word = (token * hidden + offset + p * threads * cluster * vec) // 2
            if cutlass.const_expr(multicast):
                _store_multicast(
                    mc_workspace + base + rank * slot_words + word,
                    packed[0],
                    packed[1],
                    packed[2],
                    packed[3],
                )
            else:
                for dst in cutlass.range_constexpr(world):
                    ptr = peers[dst] + base + rank * slot_words + word
                    dst_tensor = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.Uint32,
                            ptr.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=AR_COPY_BITS // 8,
                        ),
                        cute.make_layout((AR_VEC_WORDS,)),
                    )
                    cute.copy(store_atom, packed, dst_tensor)

    @cute.jit
    def reduce_token(
        output,
        ar,
        token,
        tidx: Int32,
        cta_rank: Int32,
        threads: cutlass.Constexpr,
        hidden: cutlass.Constexpr,
        cluster: cutlass.Constexpr,
        epoch,
    ):
        peers, _, _, slot_words, poll_workspace, _ = ar
        world = len(peers)
        base = Int32(epoch & 1) * world * slot_words
        # Interleave AR warps across CTAs.
        vector = (
            tidx // cute.arch.WARP_SIZE * cluster + cta_rank
        ) * cute.arch.WARP_SIZE + tidx % cute.arch.WARP_SIZE
        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            cutlass.Uint32,
            num_bits_per_copy=AR_COPY_BITS,
            memory_order=cute.nvgpu.MemoryOrder.RELAXED,
            memory_scope=cute.nvgpu.MemoryScope.SYS,
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            cutlass.Uint32,
            num_bits_per_copy=AR_COPY_BITS,
            memory_order=cute.nvgpu.MemoryOrder.WEAK,
            memory_scope=cute.nvgpu.MemoryScope.CTA,
        )
        assert cutlass.const_expr(threads % cute.arch.WARP_SIZE == 0)
        assert cutlass.const_expr(hidden % (threads * cluster * AR_VEC_SIZE) == 0)
        num_iters = hidden // (threads * cluster * AR_VEC_SIZE)
        vector_stride = threads * cluster * AR_VEC_WORDS
        word = token * (hidden // AR_VEC_SIZE * AR_VEC_WORDS) + vector * AR_VEC_WORDS
        # Per-thread partitions: [word, iteration] and [word, source, iteration].
        output_tensor = cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint32,
                (output.iterator + token * hidden + vector * AR_VEC_SIZE).toint(),
                cute.AddressSpace.gmem,
                assumed_align=AR_COPY_BITS // 8,
            ),
            cute.make_layout((AR_VEC_WORDS, num_iters), stride=(1, vector_stride)),
        )
        poll_ptrs = cute.make_tensor(
            cute.make_ptr(
                cutlass.Uint32,
                (poll_workspace + base + word).toint(),
                cute.AddressSpace.gmem,
                assumed_align=AR_COPY_BITS // 8,
            ),
            cute.make_layout(
                (AR_VEC_WORDS, world, num_iters),
                stride=(1, cute.assume(slot_words, divby=AR_VEC_WORDS), vector_stride),
            ),
        )
        rank_values = cute.make_rmem_tensor(
            cute.make_layout((AR_VEC_WORDS, world)), cutlass.Uint32
        )
        rank_bf16 = cute.recast_tensor(rank_values, cutlass.BFloat16)
        result = cute.make_rmem_tensor(
            cute.make_layout((AR_VEC_SIZE,)), cutlass.BFloat16
        )
        zero = cute.make_rmem_tensor(cute.make_layout((AR_VEC_WORDS,)), cutlass.Uint32)
        zero.fill(0)
        for vector_iter in cutlass.range_constexpr(num_iters):
            # Issue all peer probes before any load-dependent readiness test.
            # Ready slots stay intact until this consumer clears them below.
            ready = False
            while not ready:
                for src in cutlass.range_constexpr(world):
                    cute.copy(
                        load_atom,
                        poll_ptrs[None, src, vector_iter],
                        rank_values[None, src],
                    )
                ready = True
                for src in cutlass.range_constexpr(world):
                    for a in cutlass.range_constexpr(AR_VEC_WORDS):
                        ready = ready & (rank_values[a, src] != 0)
            # Reinterpret BF16 payloads and accumulate ranks in the original order.
            total = rank_bf16[None, 0].load().to(Float32)
            for src in cutlass.range_constexpr(1, world):
                total = total + rank_bf16[None, src].load().to(Float32)
            result.store(total.to(cutlass.BFloat16))
            cute.copy(
                store_atom,
                cute.recast_tensor(result, cutlass.Uint32),
                output_tensor[None, vector_iter],
            )
            for src in cutlass.range_constexpr(world):
                cute.copy(store_atom, zero, poll_ptrs[None, src, vector_iter])

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
    # PTX Wrappers
    # =================================================================

    @dsl_user_op
    def _first_lane(mask, *, loc=None, ip=None):
        # Find the lowest set-bit lane in a nonzero mask.
        return Int32(
            llvm.inline_asm(
                T.i32(),
                [Int32(mask).ir_value(loc=loc, ip=ip)],
                "{ .reg .b32 rev; brev.b32 rev, $1; clz.b32 $0, rev; }",
                "=r,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
            )
        )

    @dsl_user_op
    def _store_dsmem(dst, bar, peer, a, b, c, d, *, loc=None, ip=None):
        # Store four FP32 sums to a peer CTA and complete 16 barrier bytes.
        # The caller includes self in the same DSMEM completion protocol.
        llvm.inline_asm(
            None,
            [
                dst.toint(loc=loc, ip=ip).ir_value(),
                bar.toint(loc=loc, ip=ip).ir_value(),
                peer.ir_value(),
                a.bitcast(Int32).ir_value(),
                b.bitcast(Int32).ir_value(),
                c.bitcast(Int32).ir_value(),
                d.bitcast(Int32).ir_value(),
            ],
            "{ .reg .b32 rd, rb; "
            "mapa.shared::cluster.u32 rd, $0, $2; "
            "mapa.shared::cluster.u32 rb, $1, $2; "
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
            "[rd], {$3, $4, $5, $6}, [rb]; }",
            "r,r,r,r,r,r,r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    @dsl_user_op
    def _wait_cluster_tx(bar, *, loc=None, ip=None):
        # Wait for phase-0 transactions and acquire visibility of DSMEM writes.
        llvm.inline_asm(
            None,
            [bar.toint(loc=loc, ip=ip).ir_value()],
            "{ .reg .pred ready; WAIT: "
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ready, [$0], 0; "
            "@!ready bra WAIT; }",
            "r,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )

    # =================================================================
    # Reduction helpers
    # =================================================================

    @cute.jit
    def _warp_sum(x: Float32, width: cutlass.Constexpr = 32):
        for i in cutlass.range_constexpr(5):
            if cutlass.const_expr((16 >> i) < width):
                x = x + cute.arch.shuffle_sync_bfly(x, offset=16 >> i)
        return x

    @cute.jit
    def _joint(s: cute.Tensor, num_valids: Int32):
        # Reduce 1-4 independent expert sums across a full warp, in place.
        # Input: each lane holds its partial sum for each expert in s[e].
        # Output: every lane holds sum_over_lanes(s[e]) for each valid expert.
        # Assign experts to lane residues (lane % 2 or lane % 4), reduce them
        # together, then broadcast from lanes 0..num_valids-1.
        lane = cute.arch.lane_idx()
        if num_valids == 1:
            s[0] = _warp_sum(s[0])
        else:
            a = Float32(0)
            b = Float32(0)
            if num_valids == 2:
                send = s[0] if lane & 1 else s[1]
                recv = cute.arch.shuffle_sync_bfly(send, offset=1)
                a = (s[1] if lane & 1 else s[0]) + recv
                for i in cutlass.range_constexpr(1, 5):
                    a = a + cute.arch.shuffle_sync_bfly(a, offset=1 << i)
            else:
                if num_valids == 3:
                    s[3] = Float32(0)
                send0 = s[0] if lane & 2 else s[2]
                send1 = s[1] if lane & 2 else s[3]
                a = (s[2] if lane & 2 else s[0]) + cute.arch.shuffle_sync_bfly(
                    send0, offset=2
                )
                b = (s[3] if lane & 2 else s[1]) + cute.arch.shuffle_sync_bfly(
                    send1, offset=2
                )
                send = a if lane & 1 else b
                recv = cute.arch.shuffle_sync_bfly(send, offset=1)
                a = (b if lane & 1 else a) + recv
                for i in cutlass.range_constexpr(2, 5):
                    a = a + cute.arch.shuffle_sync_bfly(a, offset=1 << i)
                s[2] = cute.arch.shuffle_sync(a, 2)
                s[3] = cute.arch.shuffle_sync(a, 3)
            s[0] = cute.arch.shuffle_sync(a, 0)
            s[1] = cute.arch.shuffle_sync(a, 1)

    # =================================================================
    # Configuration
    # =================================================================

    class MOEFinalizeRMSNormConfig:
        """Configuration for the MOE finalize + RMSNorm kernel.

        One token per block or cluster, all threads process the same row.
        Uses tiled copies vectorized up to 128 bits for global memory access.
        """

        COPY_BITS = 128

        def __init__(
            self,
            dtype: type,
            hidden_dim: int,
            hidden_dim_padded: int,
            threads_per_row=None,
            cluster_size=1,
        ):
            if cluster_size not in (1, 2, 4, 8, 16):
                raise ValueError("cluster_size must be 1, 2, 4, 8, or 16")
            self.dtype = dtype
            self.N = hidden_dim
            self.cluster_size = cluster_size
            self.threads_per_row, self.vec_size = self._compute_threads_per_row(
                hidden_dim, hidden_dim_padded, threads_per_row
            )
            self.num_threads = self.threads_per_row
            self.num_iters = hidden_dim // (
                self.num_threads * cluster_size * self.vec_size
            )
            self.warps_per_row = max(self.threads_per_row // 32, 1)

        def _compute_threads_per_row(
            self, N: int, hidden_dim_padded: int, threads_per_row=None
        ) -> tuple[int, int]:
            """Select compatible threads per row and vector width together."""
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
            if threads_per_row is not None:
                tpr = threads_per_row
            if tpr < 1 or tpr > 1024 or tpr & (tpr - 1):
                raise ValueError("threads_per_row must be a power of two up to 1024")
            # Select an aligned copy width; scalar copies handle odd row strides.
            while tpr >= (32 if self.cluster_size > 1 else 1):
                for vec_size in (8, 4, 2, 1):
                    if (
                        vec_size * self.dtype.width <= self.COPY_BITS
                        and N > 0
                        and N % (tpr * self.cluster_size * vec_size) == 0
                        and hidden_dim_padded % vec_size == 0
                    ):
                        return tpr, vec_size
                if threads_per_row is not None:
                    break
                tpr //= 2
            raise ValueError("No compatible thread/vector layout")

        def smem_size_in_bytes(self, topk_groups: int) -> int:
            """Shared memory for expert sums, transaction barriers, and warp sums."""
            return (
                topk_groups * self.cluster_size * self.warps_per_row * 16
                + topk_groups * 8
                + self.warps_per_row * 4
            )

    # =================================================================
    # Kernel
    # =================================================================

    class MOEFinalizeRMSNormKernel:
        """CuTeDSL kernel: unpermute + RMSNorm + weighted accumulate.

        Supports both weight-first (``rms_norm(weight*x)``) and standard
        (``rms_norm(x)*weight``) forms, parametrised by ``weight_first`` and
        ``eps`` so the kernel reproduces the same math as the Python
        ``MuseSpark1xRMSNorm`` modules.

        Uses tiled copies vectorized up to 128 bits for global memory access
        and FP32 register-level computation.
        """

        def __init__(
            self,
            dtype: type,
            num_tokens: int,
            hidden_dim: int,
            topk: int,
            hidden_dim_padded: int,
            threads_per_row=None,
            use_pdl=False,
            eps: float = 1e-8,
            weight_first: bool = True,
            cluster_size: int = 1,
            ar_world_size: int = 0,
            ar_controller_count: int = 0,
            ar_multicast: bool = False,
        ):
            self.cfg = MOEFinalizeRMSNormConfig(
                dtype, hidden_dim, hidden_dim_padded, threads_per_row, cluster_size
            )
            self.ar_controller_count = ar_controller_count
            self.cluster_size = cluster_size
            self.topk = topk
            self.hidden_dim_padded = hidden_dim_padded
            self.group_width = 4 if cluster_size > 1 else 1
            self.topk_groups = (topk + self.group_width - 1) // self.group_width
            self.use_pdl = use_pdl
            self.eps = float(eps)
            self.weight_first = bool(weight_first)
            self.ar_world_size = ar_world_size
            self.ar_multicast = ar_multicast
            self.num_tokens = num_tokens
            self.is_persistent = (
                ar_world_size > 0 and num_tokens * cluster_size > ar_controller_count
            )
            if ar_world_size > 0 and self.cfg.vec_size != AR_VEC_SIZE:
                raise ValueError("Finalize push requires eight BF16 values per lane")

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
            ar=None,
        ):
            """Host function to launch the kernel.

            ar is None for standalone finalize, otherwise a tuple from
            FinalizePushWorkspace.arguments(): (peer Uint32 pointers, controller
            Uint32 pointer, rank, slot_words, local inbox Uint32 pointer).
            """
            cfg = self.cfg
            hidden_dim = cfg.N
            topk = self.topk
            hidden_dim_padded = self.hidden_dim_padded
            # Both paths specialize M; the runtime argument is retained for the ABI.
            num_tokens = self.num_tokens

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

            num_blocks = num_tokens * self.cluster_size
            if cutlass.const_expr(self.is_persistent):
                num_blocks = min(self.ar_controller_count, num_blocks)
            self.kernel(mGemm2, mOut, mW, mExpW, mIdx, ar).launch(
                grid=[num_blocks, 1, 1],
                block=[cfg.num_threads, 1, 1],
                cluster=[self.cluster_size, 1, 1],
                smem=cfg.smem_size_in_bytes(self.topk_groups),
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
            ar=None,
        ):
            """Device kernel using tiled copies vectorized up to 128 bits."""
            cfg = self.cfg
            topk = self.topk
            hidden_dim = cfg.N
            threads_per_row = cfg.threads_per_row
            warps_per_row = cfg.warps_per_row

            tidx = cute.arch.thread_idx()[0]
            lane = tidx % 32
            warp = tidx // 32
            cta_rank = cute.arch.block_idx()[0] % self.cluster_size
            token_start = cute.arch.block_idx()[0] // self.cluster_size
            smem = utils.SmemAllocator()
            # DSMEM inbox: [expert group, source CTA, source warp, 4 expert sums].
            partials = smem.allocate_tensor(
                Float32,
                cute.make_layout(
                    (self.topk_groups * self.cluster_size * warps_per_row * 4,)
                ),
                byte_alignment=16,
            )
            # One transaction-completion barrier per expert group; no phase reuse.
            bars = smem.allocate_tensor(
                cutlass.Uint64, cute.make_layout((self.topk_groups,)), byte_alignment=8
            )
            # C=1 scratch: one sum per warp for the current expert.
            warp_sums = smem.allocate_tensor(
                Float32, cute.make_layout((warps_per_row,)), byte_alignment=4
            )

            # ---- Initialize mbarriers when cluster size > 1 ----
            if cutlass.const_expr(self.cluster_size > 1):
                # Each source warp sends four FP32 sums (16 bytes) to every CTA.
                for g in range(tidx, self.topk_groups, threads_per_row):
                    cute.arch.mbarrier_init(bars.iterator + g, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bars.iterator + g, self.cluster_size * warps_per_row * 16
                    )
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier()
                # All peer barriers must be initialized before any DSMEM stores.
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()

            token_end = token_start + 1
            token_stride = 1
            token_iterations = 1
            if cutlass.const_expr(self.is_persistent):
                token_end = self.num_tokens
                token_stride = self.ar_controller_count
                token_iterations = (
                    token_end - token_start + token_stride - 1
                ) // token_stride
            # Preserve the host-validated alignment for vectorized copies.
            offset = cute.assume(
                (cta_rank * threads_per_row + tidx) * cfg.vec_size, divby=cfg.vec_size
            )
            layout = cute.make_layout(
                (cfg.vec_size, cfg.num_iters),
                stride=(1, threads_per_row * self.cluster_size * cfg.vec_size),
            )
            reg_layout = cute.make_layout((cfg.vec_size, cfg.num_iters))
            atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cfg.dtype,
                num_bits_per_copy=cfg.vec_size * cfg.dtype.width,
            )
            norm = cute.make_rmem_tensor(reg_layout, cfg.dtype)
            # Read weights before PDL wait since weights are already fixed.
            cute.copy(atom, cute.make_tensor(mW.iterator + offset, layout), norm)
            if cutlass.const_expr(self.ar_world_size > 0):
                # Read the borrowed epoch after the PDL dependency completes.
                controllers = cute.make_tensor(
                    ar[1], cute.make_layout((self.ar_controller_count,))
                )
                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
                epoch = controllers[token_start]
            for token_iteration in cutlass.range(token_iterations):
                token_idx = token_start + token_iteration * token_stride
                # ---- PDL: wait for prior kernel to complete ----
                if cutlass.const_expr(self.use_pdl and self.ar_world_size == 0):
                    cute.arch.griddepcontrol_wait()

                acc = cute.make_rmem_tensor(reg_layout, Float32)
                acc.fill(0)
                input_frag = cute.make_rmem_tensor(
                    cute.make_layout((cfg.vec_size, cfg.num_iters, self.group_width)),
                    cfg.dtype,
                )
                scaled_frag = cute.make_rmem_tensor(
                    cute.make_layout((cfg.vec_size, cfg.num_iters, self.group_width)),
                    Float32,
                )
                sum_sq = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
                expert_weights = cute.make_rmem_tensor(
                    cute.make_layout((self.group_width,)), Float32
                )
                permuted_idxs = cute.make_rmem_tensor(
                    cute.make_layout((self.group_width,)), Int32
                )
                lane_row = Int32(-1)
                lane_weight = Float32(0)
                remaining = cutlass.Uint32(0)

                # Extract metadata parallelly for the current token.
                if cutlass.const_expr(topk <= min(32, threads_per_row)):
                    if lane < topk:
                        lane_row = mIdx[token_idx * topk + lane]
                        lane_weight = Float32(mExpW[token_idx * topk + lane])
                    remaining = cute.arch.vote_ballot_sync(lane_row >= 0)
                next_expert = Int32(0)
                # Iterate over all group of topK
                for group in cutlass.range_constexpr(self.topk_groups):
                    sum_sq.fill(0)
                    num_valids = Int32(0)
                    # Find valid experts in the current group.
                    for e in cutlass.range_constexpr(self.group_width):
                        row = Int32(-1)
                        route = Float32(0)
                        if cutlass.const_expr(topk <= min(32, threads_per_row)):
                            source = (
                                _first_lane(remaining) if remaining != 0 else Int32(0)
                            )
                            permuted_idx = cute.arch.shuffle_sync(lane_row, source)
                            expert_weight = cute.arch.shuffle_sync(lane_weight, source)
                            if remaining != 0:
                                row = permuted_idx
                                route = expert_weight
                            remaining = remaining & (remaining - cutlass.Uint32(1))
                        else:
                            searching = True
                            while next_expert < topk and searching:
                                candidate = mIdx[token_idx * topk + next_expert]
                                if candidate >= 0:
                                    row = candidate
                                    route = Float32(
                                        mExpW[token_idx * topk + next_expert]
                                    )
                                    searching = False
                                next_expert += 1
                        permuted_idxs[e] = row
                        expert_weights[e] = route
                        if row >= 0:
                            num_valids += 1
                            input_offset = cute.assume(
                                Int64(row) * self.hidden_dim_padded + Int64(offset),
                                divby=cfg.vec_size,
                            )
                            cute.copy(
                                atom,
                                cute.make_tensor(
                                    mGemm2.iterator + input_offset, layout
                                ),
                                input_frag[None, None, e],
                            )
                    # Load the whole group before consuming any expert's values.
                    for e in cutlass.range_constexpr(self.group_width):
                        if permuted_idxs[e] >= 0:
                            sq = Float32(0)
                            for p in cutlass.range_constexpr(cfg.num_iters):
                                for v in cutlass.range_constexpr(cfg.vec_size):
                                    value = Float32(input_frag[v, p, e])
                                    if cutlass.const_expr(self.weight_first):
                                        value = value * Float32(norm[v, p])
                                    scaled_frag[v, p, e] = value
                                    sq = cute.fma(value, value, sq)
                            sum_sq[e] = sq
                    if num_valids > 0:
                        if cutlass.const_expr(self.cluster_size > 1):
                            # ---- Reduce within each warp, then publish to all CTAs ----
                            _joint(sum_sq, num_valids)
                            if lane < self.cluster_size:
                                dst = (
                                    partials.iterator
                                    + (
                                        group * self.cluster_size * warps_per_row
                                        + cta_rank * warps_per_row
                                        + warp
                                    )
                                    * 4
                                )
                                _store_dsmem(
                                    dst,
                                    bars.iterator + group,
                                    lane,
                                    sum_sq[0],
                                    sum_sq[1],
                                    sum_sq[2],
                                    sum_sq[3],
                                )
                            # ---- Wait for the inbox, then reduce all CTA/warp sums ----
                            _wait_cluster_tx(bars.iterator + group)
                            sum_sq.fill(0)
                            for slot in range(
                                lane, self.cluster_size * warps_per_row, 32
                            ):
                                for e in cutlass.range_constexpr(self.group_width):
                                    if e < num_valids:
                                        sum_sq[e] += partials[
                                            (
                                                group
                                                * self.cluster_size
                                                * warps_per_row
                                                + slot
                                            )
                                            * 4
                                            + e
                                        ]
                            _joint(sum_sq, num_valids)
                        else:
                            # ---- Single-CTA reduction through warp_sums ----
                            s = _warp_sum(sum_sq[0], min(threads_per_row, 32))
                            if lane == 0:
                                warp_sums[warp] = s
                            cute.arch.barrier()
                            s = Float32(0)
                            if lane < warps_per_row:
                                s = warp_sums[lane]
                            sum_sq[0] = _warp_sum(s, min(threads_per_row, 32))
                            cute.arch.barrier()
                        # ---- Normalize each expert and accumulate router-weighted output ----
                        for e in cutlass.range_constexpr(self.group_width):
                            if permuted_idxs[e] >= 0:
                                inv = cute.math.rsqrt(
                                    sum_sq[e] / hidden_dim + Float32(self.eps),
                                    fastmath=True,
                                )
                                for p in cutlass.range_constexpr(cfg.num_iters):
                                    for v in cutlass.range_constexpr(cfg.vec_size):
                                        value = scaled_frag[v, p, e] * inv
                                        if cutlass.const_expr(not self.weight_first):
                                            value = value * Float32(norm[v, p])
                                        acc[v, p] = (
                                            acc[v, p] + expert_weights[e] * value
                                        )
                if cutlass.const_expr(self.ar_world_size > 0):
                    # Push finalized values to peer GPUs.
                    push_token(
                        acc,
                        ar,
                        token_idx,
                        offset,
                        threads_per_row,
                        hidden_dim,
                        self.cluster_size,
                        epoch,
                        self.ar_multicast,
                    )
                else:
                    out = cute.make_rmem_tensor(reg_layout, cfg.dtype)
                    out.store(acc.load().to(cfg.dtype))
                    output_offset = cute.assume(
                        Int64(token_idx) * hidden_dim + Int64(offset),
                        divby=cfg.vec_size,
                    )
                    cute.copy(
                        atom,
                        out,
                        cute.make_tensor(mOut.iterator + output_offset, layout),
                    )

            # ---- PDL: Trigger next kernel here to increase overlap
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            # Reduce received values for each token.
            if cutlass.const_expr(self.ar_world_size > 0):
                for token_iteration in cutlass.range(token_iterations):
                    token_idx = token_start + token_iteration * token_stride
                    reduce_token(
                        mOut,
                        ar,
                        token_idx,
                        tidx,
                        cta_rank,
                        threads_per_row,
                        hidden_dim,
                        self.cluster_size,
                        epoch,
                    )
            # Keep shared memory alive until peer DSMEM accesses finish.
            if cutlass.const_expr(self.cluster_size > 1):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            if cutlass.const_expr(self.ar_world_size > 0):
                # One writer per active worker; all CTAs share unused controllers.
                next_epoch = epoch ^ cutlass.Uint32(1)
                if cta_rank == 0 and tidx == 0:
                    controllers[token_start] = next_epoch
                global_thread = cute.arch.block_idx()[0] * threads_per_row + tidx
                # Advance unused controllers to match ordinary push AR.
                for controller in range(
                    cute.arch.grid_dim()[0] // self.cluster_size + global_thread,
                    self.ar_controller_count,
                    cute.arch.grid_dim()[0] * threads_per_row,
                ):
                    controllers[controller] = next_epoch

    # =================================================================
    # Compilation cache
    # =================================================================

    _compile_cache: dict = {}

    def _get_compiled_kernel(
        dtype: type,
        num_tokens: int,
        hidden_dim: int,
        topk: int,
        hidden_dim_padded: int,
        stream: cuda.CUstream,
        threads_per_row=None,
        use_pdl=False,
        eps: float = 1e-8,
        weight_first: bool = True,
        expert_weights_dtype: type = cutlass.BFloat16,
        cluster_size: int = 1,
        ar_world_size: int = 0,
        ar_controller_count: int = 0,
        ar_multicast: bool = False,
    ):
        if num_tokens <= 0:
            raise ValueError("Finalize requires a positive token count")
        if ar_world_size:
            if cluster_size > 1 and num_tokens * cluster_size > ar_controller_count:
                raise ValueError("Persistent finalize AR requires cluster_size=1")
        key = (
            dtype,
            hidden_dim,
            topk,
            hidden_dim_padded,
            threads_per_row,
            use_pdl,
            eps,
            weight_first,
            expert_weights_dtype,
            cluster_size,
            ar_world_size,
            ar_controller_count,
            num_tokens,
            ar_multicast,
        )
        if key not in _compile_cache:
            if ar_controller_count < 0 or (ar_controller_count and not ar_world_size):
                raise ValueError(
                    "Controller count requires an AR workspace and must be nonnegative"
                )
            kernel_obj = MOEFinalizeRMSNormKernel(
                dtype,
                num_tokens,
                hidden_dim,
                topk,
                hidden_dim_padded,
                threads_per_row,
                use_pdl=use_pdl,
                eps=eps,
                weight_first=weight_first,
                cluster_size=cluster_size,
                ar_world_size=ar_world_size,
                ar_controller_count=ar_controller_count,
                ar_multicast=ar_multicast,
            )
            compiled = cute.compile(
                kernel_obj,
                make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16),
                make_ptr(
                    expert_weights_dtype,
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
                make_ar_compile_args(ar_world_size),
                options="--enable-tvm-ffi",
            )
            _compile_cache[key] = compiled
        return _compile_cache[key]
