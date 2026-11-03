import os
import unittest
from datetime import timedelta
from itertools import accumulate
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)
from sglang.srt.layers.attention.linear.kernels.gdn_flashqla import (
    FlashQLAGDNKernel,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.cp.utils import (
    ContextParallelOutput,
)
from sglang.srt.layers.cp.utils import (
    ShardedContextParallelMetadata as ContextParallelMetadata,
)
from sglang.srt.layers.cp.utils import (
    _heads_to_context,
)
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import get_device, is_hip
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_xpu_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=11, stage="stage-b", runner_config="1-gpu-large-amd")
register_xpu_ci(est_time=900, suite="stage-b-test-1-gpu-xpu")


@unittest.skipIf(
    not (torch.cuda.is_available() or torch.xpu.is_available()),
    "Test requires CUDA or XPU",
)
class TestChunkGatedDeltaRule(unittest.TestCase):
    """Test chunk_gated_delta_rule against token-by-token fused_recurrent reference."""

    ATOL = 2e-2
    RTOL = 1e-2

    def _run_reference(self, pool_init, cache_indices, q, k, v, g, beta):
        """Per-batch token-by-token reference using fused_recurrent_gated_delta_rule.

        initial_state shape: [N, H, V, K] (native layout on this branch).
        """
        B = cache_indices.shape[0]
        T_per_seq = q.shape[1] // B
        pool = pool_init.clone()
        h_cur = pool[cache_indices].contiguous().clone()

        o_list = []
        for b in range(B):
            sl = slice(b * T_per_seq, (b + 1) * T_per_seq)
            o_b, h_b = fused_recurrent_gated_delta_rule(
                q=q[0, sl].unsqueeze(0),
                k=k[0, sl].unsqueeze(0),
                v=v[0, sl].unsqueeze(0),
                g=g[0, sl].unsqueeze(0),
                beta=beta[0, sl].unsqueeze(0),
                initial_state=h_cur[b : b + 1],
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
            o_list.append(o_b)
            h_cur[b] = h_b[0]

        pool[cache_indices] = h_cur
        return torch.cat(o_list, dim=1), pool

    def _run_chunk(self, pool_init, cache_indices, q, k, v, g, beta, cu_seqlens):
        """Run chunk_gated_delta_rule with native [V, K] pool."""
        pool = pool_init.clone()
        o, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=pool,
            initial_state_indices=cache_indices,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        return o, pool

    def _check_shape(
        self, B, T_per_seq, H, K, V, pool_size, sequential_indices=False, seed=42
    ):
        """Run correctness check for one (B, T_per_seq, H, K, V, pool_size) config."""
        device = get_device()
        dtype = torch.bfloat16
        T = B * T_per_seq

        torch.manual_seed(seed)

        if sequential_indices:
            cache_indices = torch.arange(B, dtype=torch.int32, device=device)
        else:
            perm = torch.randperm(pool_size, device=device)[:B]
            cache_indices = perm.to(torch.int32)

        pool_init = (
            torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device) * 0.1
        )
        cu_seqlens = torch.zeros(B + 1, dtype=torch.long, device=device)
        cu_seqlens[1:] = (
            torch.arange(1, B + 1, dtype=torch.long, device=device) * T_per_seq
        )

        q = torch.randn(1, T, H, K, dtype=dtype, device=device)
        k = torch.randn(1, T, H, K, dtype=dtype, device=device)
        v = torch.randn(1, T, H, V, dtype=dtype, device=device)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, T, H, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(torch.randn(1, T, H, dtype=dtype, device=device))

        o_ref, pool_ref = self._run_reference(
            pool_init, cache_indices, q, k, v, g, beta
        )
        o_new, pool_new = self._run_chunk(
            pool_init, cache_indices, q, k, v, g, beta, cu_seqlens
        )

        self.assertTrue(
            torch.allclose(
                o_ref.float(), o_new.float(), atol=self.ATOL, rtol=self.RTOL
            ),
            f"Output mismatch: max_diff="
            f"{(o_ref.float() - o_new.float()).abs().max().item():.2e}",
        )

        ref_slots = pool_ref[cache_indices].contiguous()
        new_slots = pool_new[cache_indices].contiguous()
        self.assertTrue(
            torch.allclose(
                ref_slots.float(), new_slots.float(), atol=self.ATOL, rtol=self.RTOL
            ),
            f"State mismatch: max_diff="
            f"{(ref_slots.float() - new_slots.float()).abs().max().item():.2e}",
        )

    # ------------------------------------------------------------------
    # Production-style configs (Qwen3-Next)
    # ------------------------------------------------------------------
    def test_production_nt1(self):
        self._check_shape(B=4, T_per_seq=64, H=16, K=128, V=128, pool_size=32)

    def test_production_nt2(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=128, V=128, pool_size=32)

    def test_production_nt4(self):
        self._check_shape(B=4, T_per_seq=256, H=16, K=128, V=128, pool_size=32)

    # ------------------------------------------------------------------
    # Batch size sweep
    # ------------------------------------------------------------------
    def test_batch_1(self):
        self._check_shape(B=1, T_per_seq=128, H=16, K=128, V=128, pool_size=32)

    def test_batch_2(self):
        self._check_shape(B=2, T_per_seq=128, H=16, K=128, V=128, pool_size=32)

    def test_batch_8(self):
        self._check_shape(B=8, T_per_seq=128, H=16, K=128, V=128, pool_size=64)

    def test_batch_16(self):
        self._check_shape(B=16, T_per_seq=64, H=16, K=128, V=128, pool_size=128)

    def test_batch_32(self):
        self._check_shape(B=32, T_per_seq=32, H=16, K=128, V=128, pool_size=256)

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "The read-only initial-state path is not implemented by the XPU chunk kernel",
    )
    def test_read_only_initial_state_supports_duplicate_indices(self):
        """MIS item branches may share one query-end state without updating it."""
        device = get_device()
        dtype = torch.bfloat16
        batch_size, tokens_per_item = 3, 65
        num_heads, key_dim, value_dim = 4, 32, 32
        total_tokens = batch_size * tokens_per_item

        torch.manual_seed(1234)
        pool_init = torch.randn(
            4,
            num_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=device,
        )
        duplicate_indices = torch.tensor([2, 2, 2], dtype=torch.int32, device=device)
        cu_seqlens = torch.arange(
            0,
            total_tokens + 1,
            tokens_per_item,
            dtype=torch.int32,
            device=device,
        )
        q = torch.randn(1, total_tokens, num_heads, key_dim, dtype=dtype, device=device)
        k = torch.randn_like(q)
        v = torch.randn(
            1, total_tokens, num_heads, value_dim, dtype=dtype, device=device
        )
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, total_tokens, num_heads, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(
            torch.randn(1, total_tokens, num_heads, dtype=dtype, device=device)
        )

        expected, _ = self._run_reference(
            pool_init, duplicate_indices, q, k, v, g, beta
        )
        actual_pool = pool_init.clone()
        actual, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=actual_pool,
            initial_state_indices=duplicate_indices,
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            inplace_update=False,
        )

        torch.testing.assert_close(
            actual.float(), expected.float(), atol=self.ATOL, rtol=self.RTOL
        )
        self.assertTrue(torch.equal(actual_pool, pool_init))

        updating_pool = pool_init.clone()
        chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=updating_pool,
            initial_state_indices=torch.arange(
                batch_size, dtype=torch.int32, device=device
            ),
            cu_seqlens=cu_seqlens,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )
        self.assertFalse(
            torch.equal(updating_pool[:batch_size], pool_init[:batch_size])
        )

    # ------------------------------------------------------------------
    # Head count sweep
    # ------------------------------------------------------------------
    def test_heads_4(self):
        self._check_shape(B=4, T_per_seq=128, H=4, K=128, V=128, pool_size=32)

    def test_heads_8(self):
        self._check_shape(B=4, T_per_seq=128, H=8, K=128, V=128, pool_size=32)

    def test_heads_32(self):
        self._check_shape(B=4, T_per_seq=128, H=32, K=128, V=128, pool_size=32)

    def test_heads_64(self):
        self._check_shape(B=4, T_per_seq=128, H=64, K=128, V=128, pool_size=32)

    # ------------------------------------------------------------------
    # K != V  (exercises that [V,K] != [K,V] byte-order matters)
    # ------------------------------------------------------------------
    def test_dim_64x64(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=64, V=64, pool_size=32)

    def test_dim_k_lt_v(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=64, V=128, pool_size=32)

    def test_dim_k_gt_v(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=128, V=64, pool_size=32)

    @unittest.skipIf(
        is_hip(),
        "K=V=256 head dim exceeds the FLA chunk triton kernel's shared-memory "
        "budget on ROCm (out-of-resource at launch); smaller head dims pass.",
    )
    def test_dim_256x256(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=256, V=256, pool_size=32)

    # ------------------------------------------------------------------
    # Short sequences (T < chunk_size=64)
    # ------------------------------------------------------------------
    def test_seqlen_1(self):
        self._check_shape(B=4, T_per_seq=1, H=16, K=128, V=128, pool_size=32)

    def test_seqlen_7(self):
        self._check_shape(B=4, T_per_seq=7, H=16, K=128, V=128, pool_size=32)

    def test_seqlen_16(self):
        self._check_shape(B=4, T_per_seq=16, H=16, K=128, V=128, pool_size=32)

    def test_seqlen_32(self):
        self._check_shape(B=4, T_per_seq=32, H=16, K=128, V=128, pool_size=32)

    # ------------------------------------------------------------------
    # Multi-chunk and large pool
    # ------------------------------------------------------------------
    def test_multi_chunk_nt8(self):
        self._check_shape(B=4, T_per_seq=512, H=16, K=128, V=128, pool_size=32)

    def test_large_pool(self):
        self._check_shape(B=4, T_per_seq=128, H=16, K=128, V=128, pool_size=512)

    # ------------------------------------------------------------------
    # Long prompts (many chunks; regression test for cross-chunk errors)
    # ------------------------------------------------------------------
    def test_long_prompt(self):
        for B, T_per_seq in [(1, 1024), (1, 1536), (1, 2048), (2, 1024)]:
            with self.subTest(T_per_seq=T_per_seq):
                self._check_shape(
                    B=B, T_per_seq=T_per_seq, H=16, K=128, V=128, pool_size=32
                )

    # ------------------------------------------------------------------
    # Combined stress
    # ------------------------------------------------------------------
    def test_stress(self):
        self._check_shape(B=32, T_per_seq=128, H=32, K=128, V=128, pool_size=256)

    # ------------------------------------------------------------------
    # Sequential-index variants (pool_size == B, indices = 0..B-1)
    # ------------------------------------------------------------------
    def test_seq_idx_b4(self):
        self._check_shape(
            B=4,
            T_per_seq=128,
            H=16,
            K=128,
            V=128,
            pool_size=4,
            sequential_indices=True,
        )

    def test_seq_idx_b8(self):
        self._check_shape(
            B=8,
            T_per_seq=128,
            H=16,
            K=128,
            V=128,
            pool_size=8,
            sequential_indices=True,
        )

    def test_seq_idx_h32(self):
        self._check_shape(
            B=4,
            T_per_seq=128,
            H=32,
            K=128,
            V=128,
            pool_size=4,
            sequential_indices=True,
        )

    def test_seq_idx_h64(self):
        self._check_shape(
            B=4,
            T_per_seq=128,
            H=64,
            K=128,
            V=128,
            pool_size=4,
            sequential_indices=True,
        )

    def test_seq_idx_stress(self):
        self._check_shape(
            B=32,
            T_per_seq=128,
            H=32,
            K=128,
            V=128,
            pool_size=32,
            sequential_indices=True,
        )

    # ------------------------------------------------------------------
    # Padded rows (idle DP ranks under breakable CUDA graph prefill)
    # ------------------------------------------------------------------

    def test_padded_state_index_is_skipped(self):
        """Rows carrying state index -1 must be skipped, not addressed.

        Without the guard the sentinel reaches pointer arithmetic and
        addresses before the state pool.
        """
        device = get_device()
        dtype = torch.bfloat16
        B, T_per_seq, H, K, V, pool_size = 4, 64, 4, 128, 128, 8
        T = B * T_per_seq
        torch.manual_seed(0)

        pool_init = (
            torch.randn(pool_size, H, V, K, dtype=torch.float32, device=device) * 0.1
        )
        cu_seqlens = torch.zeros(B + 1, dtype=torch.long, device=device)
        cu_seqlens[1:] = (
            torch.arange(1, B + 1, dtype=torch.long, device=device) * T_per_seq
        )
        q = torch.randn(1, T, H, K, dtype=dtype, device=device)
        k = torch.randn(1, T, H, K, dtype=dtype, device=device)
        v = torch.randn(1, T, H, V, dtype=dtype, device=device)
        g = torch.nn.functional.logsigmoid(
            torch.randn(1, T, H, dtype=dtype, device=device)
        )
        beta = torch.sigmoid(torch.randn(1, T, H, dtype=dtype, device=device))

        # "all padded" is the idle-DP-rank case that faulted.
        for label, indices in (("mixed", [0, -1, 2, -1]), ("all_padded", [-1] * B)):
            with self.subTest(label):
                cache_indices = torch.tensor(indices, dtype=torch.int32, device=device)
                o, pool = self._run_chunk(
                    pool_init, cache_indices, q, k, v, g, beta, cu_seqlens
                )
                # Forces a sync: an out-of-bounds access surfaces here.
                self.assertTrue(torch.isfinite(o.float()).all())

                untouched = [s for s in range(pool_size) if s not in indices]
                self.assertTrue(
                    torch.equal(pool[untouched], pool_init[untouched]),
                    f"{label}: padded rows wrote into the state pool",
                )


@unittest.skipUnless(
    int(os.environ.get("WORLD_SIZE", "1")) > 1, "Run with torchrun for CP"
)
class TestFlashQLACP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=timedelta(seconds=180))
        cls.rank = dist.get_rank()
        cls.world = dist.get_world_size()
        cls.triton = TritonGDNKernel()
        cls.tp = FlashQLAGDNKernel(fallback=cls.triton)
        cls.cp = FlashQLAGDNKernel(fallback=cls.triton, cp_group=dist.group.WORLD)
        from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher

        cls.dispatcher = object.__new__(GDNKernelDispatcher)
        cls.dispatcher.extend_kernel = cls.cp
        cls.dispatcher.use_context_parallel = True

    @classmethod
    def tearDownClass(cls):
        import gc

        cls.cp = cls.tp = cls.dispatcher = None
        gc.collect()
        if dist.is_initialized():
            dist.destroy_process_group()

    @torch.inference_mode()
    def test_startup_warmup_uses_sharded_metadata(self):
        from sglang.srt.layers.attention.linear.kernels.gdn_flashqla import (
            _warmup_flashqla_gdn,
        )

        _warmup_flashqla_gdn(
            torch.device("cuda", torch.cuda.current_device()),
            cp_group=dist.group.WORLD,
            cp_size=self.world,
            cp_rank=self.rank,
            prefix_algorithm="ring",
        )

    @torch.inference_mode()
    def test_default_symm_state_and_opt_out(self):

        if self.cp._symm_state_unavailable is not None:
            self.skipTest(self.cp._symm_state_unavailable)
        self.assertTrue(self.cp.symm_state_enabled())
        self.assertFalse(self.tp.symm_state_enabled())
        torch.manual_seed(20261017 + self.rank)
        q = torch.randn(1, 128, 16, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 128, 64, 128, device="cuda", dtype=q.dtype)
        inputs = [
            q,
            torch.randn_like(q),
            v,
            -torch.rand(1, 128, 64, device="cuda") * 0.1,
            torch.rand(1, 128, 64, device="cuda"),
        ]
        initial = torch.randn(1, 64, 128, 128, device="cuda") * 0.01
        for checkpoint in (None, 0, self.world * 128 - 64):
            with patch.object(self.cp, "_symm_state_unavailable", "NCCL comparison"):
                self.assertFalse(self.cp.symm_state_enabled())
                expected = self.dispatcher.extend(
                    *inputs,
                    ssm_states=initial,
                    cache_indices=None,
                    query_start_loc=None,
                    cp_metadata=ContextParallelMetadata(
                        group=self.cp._group, rank=self.cp._rank, world=self.cp._world
                    ),
                    initial_state=initial,
                    inplace_update=False,
                    checkpoint_indices=() if checkpoint is None else (checkpoint,),
                )
            actual = self.dispatcher.extend(
                *inputs,
                ssm_states=initial,
                cache_indices=None,
                query_start_loc=None,
                cp_metadata=ContextParallelMetadata(
                    group=self.cp._group, rank=self.cp._rank, world=self.cp._world
                ),
                initial_state=initial,
                inplace_update=False,
                checkpoint_indices=() if checkpoint is None else (checkpoint,),
            )
            self.assertIsNotNone(self.cp._state_broadcast_workspace)
            self.assertIsNotNone(self.cp._prefix_workspace)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            with patch.object(self.cp, "_prefix_algorithm", "ring"):
                ring_result = self.dispatcher.extend(
                    *inputs,
                    ssm_states=initial,
                    cache_indices=None,
                    query_start_loc=None,
                    cp_metadata=ContextParallelMetadata(
                        group=self.cp._group, rank=self.cp._rank, world=self.cp._world
                    ),
                    initial_state=initial,
                    inplace_update=False,
                    checkpoint_indices=() if checkpoint is None else (checkpoint,),
                )
            for actual_state, expected_state in zip(ring_result, expected):
                torch.testing.assert_close(actual_state, expected_state, atol=0, rtol=0)

    def test_symm_state_capability_consensus(self):
        from sglang.srt.distributed.device_communicators import torch_symm_mem

        communicator = torch_symm_mem.TorchSymmMemCommunicator

        # One unsupported rank must send the entire group down the old path.
        with patch.object(
            communicator,
            "_local_multicast_unavailable_reason",
            return_value="multicast unavailable" if self.rank == 0 else None,
        ):
            reason = communicator.multicast_unavailable_reason(dist.group.WORLD)
        self.assertEqual(reason, "Peer rank 0: multicast unavailable")
        with patch.object(
            torch_symm_mem.socket, "gethostname", return_value=f"host-{self.rank}"
        ):
            reason = communicator.multicast_unavailable_reason(dist.group.WORLD)
        self.assertIn("one host", reason)
        with self.assertRaisesRegex(ValueError, "Peer ranks disagree"):
            communicator.multicast_unavailable_reason(
                dist.group.WORLD, enabled=self.rank == 0
            )

    @torch.inference_mode()
    def test_fused_cp_inputs_preserve_outputs_and_states(self):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            _prepare_qkv_inputs,
        )

        layer = SimpleNamespace(
            num_q_heads=16,
            num_k_heads=16,
            num_v_heads=64,
            head_q_dim=128,
            head_k_dim=128,
            head_v_dim=128,
            q_dim=2048,
            k_dim=2048,
            v_dim=8192,
        )
        torch.manual_seed(20261016 + self.rank)
        for count, width in ((0, 64), (63, 64), (64, 64), (129, 192), (8192, 8192)):
            for strided in (False, True):
                packed = torch.randn(
                    count,
                    12288 * (2 if strided else 1),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                if strided:
                    packed = packed[:, ::2]
                expected = _prepare_qkv_inputs(packed, layer, width, fused=False)
                actual = _prepare_qkv_inputs(packed, layer, width, fused=True)
                for a, b in zip(actual, expected):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
                    self.assertTrue(a.is_contiguous())
                # Both ranks keep identical collective order, including empty
                # local shards. Padding must be an identity state transition.
                g = torch.zeros(1, width, 64, device="cuda")
                beta = torch.zeros_like(g)
                g[:, :count] = -0.1
                beta[:, :count] = 0.5
                initial = torch.randn(1, 64, 128, 128, device="cuda") * 0.01
                old = self.dispatcher.extend(
                    *[*expected, g, beta],
                    ssm_states=initial,
                    cache_indices=None,
                    query_start_loc=None,
                    cp_metadata=ContextParallelMetadata(
                        group=self.cp._group, rank=self.cp._rank, world=self.cp._world
                    ),
                    initial_state=initial,
                    inplace_update=False,
                )
                new = self.dispatcher.extend(
                    *[*actual, g, beta],
                    ssm_states=initial,
                    cache_indices=None,
                    query_start_loc=None,
                    cp_metadata=ContextParallelMetadata(
                        group=self.cp._group, rank=self.cp._rank, world=self.cp._world
                    ),
                    initial_state=initial,
                    inplace_update=False,
                )
                for a, b in zip(new, old):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)

    @torch.inference_mode()
    def test_sequence_metadata_reuse_does_not_reuse_activations(self):
        # Alternate shapes/heads and change both input and initial state. A
        # cached context must behave exactly like a newly planned invocation.
        for length, heads in ((64, 16), (128, 64), (4096, 64), (128, 64)):
            q = torch.randn(1, length, 16, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, length, heads, 128, device="cuda", dtype=q.dtype)
            inputs = [
                q,
                torch.randn_like(q),
                v,
                torch.full((1, length, heads), -0.1, device="cuda"),
                torch.full((1, length, heads), 0.5, device="cuda"),
            ]
            initial = torch.randn(1, heads, 128, 128, device="cuda") * 0.01
            self.dispatcher.extend(
                *inputs,
                ssm_states=initial,
                cache_indices=None,
                query_start_loc=None,
                cp_metadata=ContextParallelMetadata(
                    group=self.cp._group, rank=self.cp._rank, world=self.cp._world
                ),
                initial_state=initial,
                inplace_update=False,
            )
            changed = [x * 0.7 for x in inputs]
            cached = self.dispatcher.extend(
                *changed,
                ssm_states=initial * 0.5,
                cache_indices=None,
                query_start_loc=None,
                cp_metadata=ContextParallelMetadata(
                    group=self.cp._group, rank=self.cp._rank, world=self.cp._world
                ),
                initial_state=initial * 0.5,
                inplace_update=False,
            )
            self.cp._cp_contexts.clear()
            fresh = self.dispatcher.extend(
                *changed,
                ssm_states=initial * 0.5,
                cache_indices=None,
                query_start_loc=None,
                cp_metadata=ContextParallelMetadata(
                    group=self.cp._group, rank=self.cp._rank, world=self.cp._world
                ),
                initial_state=initial * 0.5,
                inplace_update=False,
            )
            for actual, expected in zip(cached, fresh):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def _close(self, actual, expected, label, *, rounding=False):
        delta = (actual.float() - expected.float()).abs()
        if rounding:
            # Each BF16 store can round by half an ULP. Remove only that
            # representational error before applying the same FP32 error bounds.
            for value in (actual, expected):
                if value.dtype == torch.bfloat16:
                    magnitude = value.abs()
                    spacing = (
                        torch.nextafter(
                            magnitude, torch.full_like(magnitude, float("inf"))
                        ).float()
                        - magnitude.float()
                    )
                    delta = (delta - 0.5 * spacing).clamp_min(0)
        scale = expected.float().square().mean().sqrt().clamp_min(1e-6)
        metrics = torch.stack(
            [delta.square().mean().sqrt() / scale, delta.max() / scale]
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        self.assertTrue(torch.isfinite(metrics).all().item(), label)
        self.assertLess(metrics[0].item(), 0.01, f"{label}: {metrics.tolist()}")
        self.assertLess(metrics[1].item(), 0.1, f"{label}: {metrics.tolist()}")
        if self.rank == 0:
            print(
                f"{label}: relative_rms={metrics[0].item():.6f} max/rms={metrics[1].item():.6f}",
                flush=True,
            )

    @torch.inference_mode()
    def test_symm_state_preserves_forward_and_checkpoints(self):

        # Exercise changing sequence length, checkpoint owner and state dtype;
        # results must remain independent of symmetric-buffer reuse.
        torch.manual_seed(20261016 + self.rank)
        for length in (64, 128, 8192):
            q = torch.randn(1, length, 16, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, length, 64, 128, device="cuda", dtype=q.dtype)
            inputs = [
                q,
                torch.randn_like(q),
                v,
                -torch.rand(1, length, 64, device="cuda") * 0.1,
                torch.rand(1, length, 64, device="cuda"),
            ]
            initial = torch.randn(1, 64, 128, 128, device="cuda") * 0.01
            for checkpoint in (None, 0, self.world * length - 64):
                for dtype in (torch.float32, torch.bfloat16):
                    with patch.object(
                        self.cp, "_symm_state_unavailable", "NCCL comparison"
                    ):
                        expected = self.dispatcher.extend(
                            *inputs,
                            ssm_states=initial,
                            cache_indices=None,
                            query_start_loc=None,
                            cp_metadata=ContextParallelMetadata(
                                group=self.cp._group,
                                rank=self.cp._rank,
                                world=self.cp._world,
                            ),
                            initial_state=initial,
                            inplace_update=False,
                            checkpoint_indices=(
                                () if checkpoint is None else (checkpoint,)
                            ),
                            state_dtype=dtype,
                        )
                    with patch.object(self.cp, "_symm_state_unavailable", None):
                        actual = self.dispatcher.extend(
                            *inputs,
                            ssm_states=initial,
                            cache_indices=None,
                            query_start_loc=None,
                            cp_metadata=ContextParallelMetadata(
                                group=self.cp._group,
                                rank=self.cp._rank,
                                world=self.cp._world,
                            ),
                            initial_state=initial,
                            inplace_update=False,
                            checkpoint_indices=(
                                () if checkpoint is None else (checkpoint,)
                            ),
                            state_dtype=dtype,
                        )
                    for a, b in zip(actual, expected):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)

    @torch.inference_mode()
    def test_checkpoint_export_preserves_forward(self):

        torch.manual_seed(1901 + self.rank)
        for length in (64, 128, 8192):
            q = torch.randn(1, length, 16, 128, device="cuda", dtype=torch.bfloat16)
            v = torch.randn(1, length, 64, 128, device="cuda", dtype=q.dtype)
            inputs = [
                q,
                torch.randn_like(q),
                v,
                -torch.rand(1, length, 64, device="cuda") * 0.1,
                torch.rand(1, length, 64, device="cuda"),
            ]
            initial = torch.randn(1, 64, 128, 128, device="cuda") * 0.01
            expected_output, expected_final, _ = self.dispatcher.extend(
                *inputs,
                ssm_states=initial,
                cache_indices=None,
                query_start_loc=None,
                cp_metadata=ContextParallelMetadata(
                    group=self.cp._group, rank=self.cp._rank, world=self.cp._world
                ),
                initial_state=initial,
                inplace_update=False,
            )
            for target, state_dtype in (
                (64, torch.float32),
                (self.world * length - 64, torch.float32),
                (64, torch.bfloat16),
                (self.world * length - 64, torch.bfloat16),
            ):
                # This assertion observes the non-RMA FlashQLA export entry.
                # RMA uses its own forward method; its output/checkpoint parity
                # is covered by test_symm_state_preserves_forward_and_checkpoints.
                with (
                    patch.object(self.cp, "_symm_state_unavailable", "NCCL comparison"),
                    patch.object(
                        self.cp, "_forward_states", wraps=self.cp._forward_states
                    ) as export,
                ):
                    output, final, checkpoint = self.dispatcher.extend(
                        *inputs,
                        ssm_states=initial,
                        cache_indices=None,
                        query_start_loc=None,
                        cp_metadata=ContextParallelMetadata(
                            group=self.cp._group,
                            rank=self.cp._rank,
                            world=self.cp._world,
                        ),
                        initial_state=initial,
                        inplace_update=False,
                        checkpoint_indices=(target,),
                        state_dtype=state_dtype,
                    )
                self.assertEqual(
                    export.call_count,
                    int(target % length != 0 and self.rank == target // length),
                )
                checkpoint = checkpoint[0]
                torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
                torch.testing.assert_close(
                    final, expected_final.to(state_dtype), atol=0, rtol=0
                )
                self.assertEqual(checkpoint.dtype, state_dtype)
                self.assertEqual(checkpoint.shape, initial.shape)
                copies = [torch.empty_like(checkpoint) for _ in range(self.world)]
                dist.all_gather(copies, checkpoint)
                for copy in copies:
                    torch.testing.assert_close(copy, checkpoint, atol=0, rtol=0)

    def _inputs(self, lengths, gate):
        heads, value_heads = 16 // self.world, 64 // self.world
        tokens = sum(lengths)
        q = torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(
            1, tokens, value_heads, 128, device="cuda", dtype=torch.bfloat16
        )
        g = -torch.rand(1, tokens, value_heads, device="cuda") * gate
        beta = torch.rand_like(g)
        return dict(q=q, k=k, v=v, g=g, beta=beta)

    def test_convolution_tail_and_padding(self):
        from sglang.srt.layers.attention.linear.gdn_backend import (
            _padded_inputs,
            _tail,
        )

        for dtype in (torch.bfloat16, torch.float32):
            for width in (0, 1, 3):
                history = torch.randn(128, width, device="cuda", dtype=dtype)
                for count in (0, 1, 2, 3, 64, 8192):
                    # Projection slices may have a non-contiguous row stride.
                    rows = torch.randn(count, 256, device="cuda", dtype=dtype)[:, ::2]
                    actual = _tail(history, rows, width)
                    expected = torch.cat((history, rows.T), dim=1)
                    expected = expected[:, -width:] if width else expected[:, :0]
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                    self.assertTrue(actual.is_contiguous())
                    padded = _padded_inputs([rows], 8192)[0]
                    torch.testing.assert_close(padded[0, :count], rows, atol=0, rtol=0)
                    self.assertEqual(torch.count_nonzero(padded[0, count:]).item(), 0)
                    self.assertTrue(padded.is_contiguous())

    @torch.inference_mode()
    def test_token_sharded_backend_cache_and_checkpoints(self):
        for state_dtype in (torch.float32, torch.bfloat16):
            self._run_token_sharded_backend_cache_and_checkpoints(state_dtype)

    @torch.inference_mode()
    def test_transport_preserves_backend_cache(self):
        for state_dtype in (torch.float32, torch.bfloat16):
            self._run_token_sharded_backend_cache_and_checkpoints(
                state_dtype, compare_transport=True
            )

    @torch.inference_mode()
    def test_backend_restores_sharded_state_and_checkpoints(self):
        for state_dtype in (torch.float32, torch.bfloat16):
            self._run_token_sharded_backend_cache_and_checkpoints(
                state_dtype, sharded_state_cache=True
            )

    def _run_token_sharded_backend_cache_and_checkpoints(
        self, state_dtype, *, compare_transport=False, sharded_state_cache=False
    ):
        import torch.nn.functional as F

        from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_fn,
        )
        from sglang.srt.layers.attention.linear.gdn_backend import (
            GDNAttnBackend,
            make_gdn_cp_conv_metadata,
        )
        from sglang.srt.layers.cp.utils import (
            ShardedContextParallelMetadata as ContextParallelMetadata,
        )
        from sglang.srt.layers.cp.utils import (
            gather_token_rows,
        )
        from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
        from sglang.test.test_utils import mock_parallel_groups

        torch.manual_seed(1701)
        heads, value_heads, dim, channels = 16, 64, 128, (16 * 2 + 64) * 128
        layer = RadixLinearAttention(
            0,
            heads,
            heads,
            value_heads,
            dim,
            dim,
            dim,
            conv_weights=torch.randn(channels, 4, device="cuda", dtype=torch.bfloat16)
            * 0.1,
            A_log=torch.zeros(value_heads, device="cuda"),
            dt_bias=torch.full((value_heads,), -3.0, device="cuda"),
        )
        # Short/empty local shards, non-zero initial state, and an intermediate
        # checkpoint whose ownership differs from the full chunk.
        for lengths, prefixes, checkpoints in (
            ((129, 1), (0, 0), (64, 0)),
            ((513, 65), (64, 128), (192, 64)),
            ((128, 256), (64, 0), (128, 256)),
            ((16384,), (128,), (16384,)),
            ((31121,), (0,), (31104,)),
            ((32768,), (0,), (32704,)),
        ):
            total = sum(lengths)
            raw = torch.randn(total, channels, device="cuda", dtype=torch.bfloat16)
            a = torch.randn(total, value_heads, device="cuda", dtype=torch.bfloat16)
            b = torch.randn_like(a)
            slots = 4 + self.world * len(lengths)
            initial = (
                torch.randn(slots, value_heads, dim, dim, device="cuda") * 0.01
            ).to(state_dtype)
            history = (
                torch.randn(slots, channels, 3, device="cuda", dtype=torch.bfloat16)
                * 0.1
            )
            for i, prefix in enumerate(prefixes):
                if not prefix:
                    initial[i].zero_()
                    history[i].zero_()
            cache = SimpleNamespace(temporal=initial.clone(), conv=[history.clone()])
            state_pool = object.__new__(MambaPool)
            state_pool.cp_size = 1
            state_pool.mamba_cache = MambaPool.State(
                temporal=cache.temporal.unsqueeze(0),
                conv=[cache.conv[0].unsqueeze(0)],
            )
            slot_offset = 0
            if sharded_state_cache:
                from sglang.srt.configs.mamba_utils import (
                    Mamba2CacheParams,
                    Mamba2StateDType,
                    Mamba2StateShape,
                )

                slot_offset = 2  # put the initial prefix on a nonzero owner
                shape = Mamba2StateShape.create(
                    tp_world_size=1,
                    intermediate_size=value_heads * dim,
                    n_groups=heads,
                    num_heads=value_heads,
                    head_dim=dim,
                    state_size=dim,
                    conv_kernel=4,
                )
                state_pool = MambaPool(
                    size=slots + slot_offset - 1,
                    spec_state_size=0,
                    device="cuda",
                    cache_params=Mamba2CacheParams(
                        shape=shape,
                        layers=[0],
                        dtype=Mamba2StateDType(
                            conv=torch.bfloat16, temporal=state_dtype
                        ),
                    ),
                    mamba_layer_ids=[0],
                    context_parallel_group=dist.group.WORLD,
                )
                self.addCleanup(state_pool.close)
                logical_slots = torch.arange(
                    slot_offset, slots + slot_offset, device="cuda"
                )

            def run_backend(*args):
                if sharded_state_cache:
                    state_pool.synchronize()
                    state_pool.store_layer_state(
                        0, logical_slots, conv=cache.conv, temporal=cache.temporal
                    )
                    state_pool.synchronize()
                output = GDNAttnBackend.forward_extend(*args)
                if sharded_state_cache:
                    state_pool.synchronize()
                    snapshot = state_pool.load_layer_state(0, logical_slots)
                    cache.temporal.copy_(snapshot.temporal)
                    cache.conv[0][...].copy_(snapshot.conv[0])
                    state_pool.synchronize()
                return output

            batch = SimpleNamespace(
                forward_mode=ForwardMode.EXTEND,
                extend_seq_lens_cpu=lengths,
                extend_prefix_lens_cpu=prefixes,
                num_token_non_padded=None,
            )
            with mock_parallel_groups(
                attn_cp_size=self.world,
                attn_cp_rank=self.rank,
                moe_tp_size=1,
                attn_cp_group=SimpleNamespace(device_group=dist.group.WORLD),
            ):
                frame = ContextParallelMetadata.from_prefill(batch, raw.device)
            batch.prefill_cp_metadata = frame
            targets = []
            for i, (length, checkpoint) in enumerate(zip(lengths, checkpoints)):
                width = (length + 64 * self.world - 1) // (64 * self.world) * 64
                targets.append(
                    [(checkpoint, i + 2)]
                    + [
                        (rank * width, 4 + i * self.world + rank - 1)
                        for rank in range(1, self.world)
                        if rank * width < checkpoint
                    ]
                )
            backend = SimpleNamespace(
                kernel_dispatcher=self.dispatcher,
                _conv_workspace=None,
                _forward_extend_cp=lambda *args: GDNAttnBackend._forward_extend_cp(
                    backend, *args
                ),
                req_to_token_pool=SimpleNamespace(
                    mamba2_layer_cache=lambda _: cache,
                    mamba_pool=state_pool,
                    mamba_map={0: 0},
                ),
                cp_state_slots=torch.arange(len(lengths), device="cuda") + slot_offset,
                cp_checkpoints=[
                    tuple(
                        (n, torch.tensor([slot + slot_offset], device="cuda"))
                        for n, slot in sequence
                    )
                    for sequence in targets
                ],
                cp_conv_metadata=make_gdn_cp_conv_metadata(frame),
            )
            rows = list(frame.contiguous[self.rank])

            def forward(batch, rows):
                args = (backend, layer, batch, raw[rows], a[rows], b[rows])
                if not compare_transport:
                    return run_backend(*args)
                saved_state, saved_conv = cache.temporal.clone(), cache.conv[0].clone()
                with (
                    patch.object(self.cp, "_symm_state_unavailable", "NCCL comparison"),
                ):
                    expected = run_backend(*args).clone()
                    expected_state = cache.temporal.clone()
                    expected_conv = cache.conv[0].clone()
                cache.temporal.copy_(saved_state)
                cache.conv[0].copy_(saved_conv)
                with (
                    patch.object(self.cp, "_symm_state_unavailable", None),
                ):
                    actual = run_backend(*args)
                for actual_value, expected_value in (
                    (actual, expected),
                    (cache.temporal, expected_state),
                    (cache.conv[0], expected_conv),
                ):
                    torch.testing.assert_close(
                        actual_value, expected_value, atol=0, rtol=0
                    )
                return actual

            actual = forward(batch, rows)
            if compare_transport:
                # Compare resumed requests using exactly the same exported
                # prefix. This separates transport regressions from the existing
                # BF16 checkpoint export error against a separate prefix launch.
                if len(lengths) == 1 and 0 < checkpoints[0] < lengths[0]:
                    target = checkpoints[0]
                    cache.temporal[0].copy_(cache.temporal[2])
                    cache.conv[0][0].copy_(cache.conv[0][2])
                    resume = SimpleNamespace(
                        forward_mode=ForwardMode.EXTEND,
                        extend_seq_lens_cpu=(lengths[0] - target,),
                        extend_prefix_lens_cpu=(prefixes[0] + target,),
                        num_token_non_padded=None,
                    )
                    with mock_parallel_groups(
                        attn_cp_size=self.world,
                        attn_cp_rank=self.rank,
                        moe_tp_size=1,
                        attn_cp_group=SimpleNamespace(device_group=dist.group.WORLD),
                    ):
                        resume.prefill_cp_metadata = (
                            ContextParallelMetadata.from_prefill(resume, raw.device)
                        )
                    tail_frame = resume.prefill_cp_metadata
                    backend.cp_checkpoints = [()]
                    backend.cp_conv_metadata = make_gdn_cp_conv_metadata(tail_frame)
                    forward(
                        resume,
                        [target + row for row in tail_frame.contiguous[self.rank]],
                    )
                continue
            actual = gather_token_rows(
                actual[0], frame.contiguous, group=dist.group.WORLD
            )
            start = 0
            reference_checkpoints = {}
            for i, count in enumerate(lengths):
                x = raw[start : start + count]
                conv_input = torch.cat((history[i], x.T), dim=1).float().unsqueeze(0)
                convolved = (
                    F.silu(
                        F.conv1d(
                            conv_input,
                            layer.conv_weights.float().unsqueeze(1),
                            groups=channels,
                        )
                    )
                    .squeeze(0)
                    .T.to(raw.dtype)
                )
                serial_conv = causal_conv1d_fn(
                    x.T,
                    layer.conv_weights,
                    None,
                    activation="silu",
                    conv_states=history[i : i + 1].clone(),
                    has_initial_state=torch.ones(1, device="cuda", dtype=torch.bool),
                    cache_indices=torch.zeros(1, device="cuda", dtype=torch.int32),
                    query_start_loc=torch.tensor(
                        [0, count], device="cuda", dtype=torch.int32
                    ),
                    seq_lens_cpu=[count],
                ).T
                # Keep the CP comparison on identical convolution arithmetic;
                # separately validate the convolution against the FP32 oracle.
                # Bound BF16 product/output rounding against the independent
                # FP32 oracle. SiLU's derivative is bounded by 1.1; summing
                # absolute products also covers cancellation near zero.
                sum_abs_products = (
                    F.conv1d(
                        conv_input.abs(),
                        layer.conv_weights.float().abs().unsqueeze(1),
                        groups=channels,
                    )
                    .squeeze(0)
                    .T
                )
                bound = (
                    1.1 * sum_abs_products + 2 * convolved.float().abs()
                ) / 256 + 2e-6
                self.assertTrue(
                    ((serial_conv.float() - convolved.float()).abs() <= bound)
                    .all()
                    .item(),
                    "Convolution exceeds BF16 rounding bound",
                )
                convolved = serial_conv
                q, k, v = convolved.split(
                    (heads * dim, heads * dim, value_heads * dim), -1
                )
                g, beta = fused_gdn_gating(
                    layer.A_log,
                    a[start : start + count],
                    b[start : start + count],
                    layer.dt_bias,
                )
                inputs = dict(
                    q=q.reshape(1, count, heads, dim),
                    k=k.reshape(1, count, heads, dim),
                    v=v.reshape(1, count, value_heads, dim),
                    g=g,
                    beta=beta,
                )
                for target, slot in ((count, i), *targets[i]):
                    if target:
                        # Compare with the established head-TP -> CP adapter.
                        # Both paths run identical padded FlashQLA arithmetic;
                        # this isolates token layout, halos and cache write-back.
                        lo, hi = (
                            self.rank * value_heads // self.world,
                            (self.rank + 1) * value_heads // self.world,
                        )
                        pool = initial[i : i + 1, lo:hi].clone()
                        sharded = {}
                        for name, value in inputs.items():
                            width = value.shape[2] // self.world
                            sharded[name] = value[
                                :, :target, self.rank * width : (self.rank + 1) * width
                            ].contiguous()
                        output, _, _ = self.cp.extend(
                            **sharded,
                            ssm_states=pool,
                            cache_indices=torch.tensor([0], device="cuda"),
                            query_start_loc=torch.tensor(
                                [0, target], device="cuda", dtype=torch.int32
                            ),
                        )
                        outputs = [torch.empty_like(output) for _ in range(self.world)]
                        states = [torch.empty_like(pool) for _ in range(self.world)]
                        dist.all_gather(outputs, output.contiguous())
                        dist.all_gather(states, pool)
                        expected_state = torch.cat(states, dim=1)[0]
                        shard_width = (
                            (count + 64 * self.world - 1) // (64 * self.world) * 64
                        )
                        if slot != i and target < count and target % shard_width == 0:
                            # A shorter CP launch repartitions the prefix and can
                            # round state again in padded ranks. Check shard ends
                            # against the existing token-by-token FP32 oracle.
                            reference_inputs = {
                                name: value[:, :target].contiguous()
                                for name, value in inputs.items()
                            }
                            for name in ("q", "k"):
                                reference_inputs[name], _ = self.cp._normalize(
                                    reference_inputs[name]
                                )
                            _, expected_state = fused_recurrent_gated_delta_rule(
                                **reference_inputs,
                                initial_state=initial[i : i + 1].float(),
                                output_final_state=True,
                            )
                            expected_state = expected_state[0].to(state_dtype)
                        if len(lengths) == 1 and slot != i:
                            reference_checkpoints[slot] = expected_state
                        if slot == i:
                            self._close(
                                actual[start : start + count],
                                torch.cat(outputs, dim=2)[0],
                                "GDN CP core",
                            )
                        self._close(
                            cache.temporal[slot],
                            expected_state,
                            f"GDN CP cached state length={count} target={target} slot={slot} dtype={state_dtype}",
                            rounding=True,
                        )
                    else:
                        self._close(
                            cache.temporal[slot], initial[i], "GDN CP initial snapshot"
                        )
                    expected_conv = torch.cat((history[i], x[:target].T), dim=1)[:, -3:]
                    torch.testing.assert_close(
                        cache.conv[0][slot], expected_conv, rtol=0, atol=0
                    )
                start += count

            # Resume a real cached prefix, including its convolution halo, and
            # compare the suffix with the original uninterrupted request.
            resume_targets = targets[0] if len(lengths) == 1 else []
            for target, checkpoint_slot in resume_targets:
                if not 0 < target < lengths[0]:
                    continue
                final_ref = cache.temporal[0].clone()
                conv_ref = cache.conv[0][0].clone()
                cache.temporal[0].copy_(cache.temporal[checkpoint_slot])
                cache.conv[0][0].copy_(cache.conv[0][checkpoint_slot])
                resume = SimpleNamespace(
                    forward_mode=ForwardMode.EXTEND,
                    extend_seq_lens_cpu=(lengths[0] - target,),
                    extend_prefix_lens_cpu=(prefixes[0] + target,),
                    num_token_non_padded=None,
                )
                with mock_parallel_groups(
                    attn_cp_size=self.world,
                    attn_cp_rank=self.rank,
                    moe_tp_size=1,
                    attn_cp_group=SimpleNamespace(device_group=dist.group.WORLD),
                ):
                    resume.prefill_cp_metadata = ContextParallelMetadata.from_prefill(
                        resume, raw.device
                    )
                tail_frame = resume.prefill_cp_metadata
                backend.cp_checkpoints = [()]
                backend.cp_conv_metadata = make_gdn_cp_conv_metadata(tail_frame)
                tail_rows = [target + row for row in tail_frame.contiguous[self.rank]]
                continued = run_backend(
                    backend, layer, resume, raw[tail_rows], a[tail_rows], b[tail_rows]
                )
                continued = gather_token_rows(
                    continued[0], tail_frame.contiguous, group=dist.group.WORLD
                )
                # Resuming changes the CP partition. Compare below with the
                # independently computed prefix and the identical tail plan.
                continued_state = cache.temporal[0].clone()
                torch.testing.assert_close(cache.conv[0][0], conv_ref, atol=0, rtol=0)
                # Isolate checkpoint-export rounding from the existing numerical
                # difference between a full request and a separate tail launch.
                # The control uses the established adapter's prefix snapshot but
                # exactly the same tail partition and convolution history.
                cache.temporal[0].copy_(reference_checkpoints[checkpoint_slot])
                cache.conv[0][0].copy_(cache.conv[0][checkpoint_slot])
                control = run_backend(
                    backend, layer, resume, raw[tail_rows], a[tail_rows], b[tail_rows]
                )
                control = gather_token_rows(
                    control[0], tail_frame.contiguous, group=dist.group.WORLD
                )
                self._close(
                    continued,
                    control,
                    "GDN CP cache-hit exported snapshot",
                    rounding=True,
                )
                self._close(
                    continued_state,
                    cache.temporal[0],
                    "GDN CP cache-hit state",
                    rounding=True,
                )
                if self.rank == 0:
                    for name, value in (
                        ("exported", continued_state),
                        ("control", cache.temporal[0]),
                    ):
                        delta = value.float() - final_ref.float()
                        scale = final_ref.float().square().mean().sqrt().clamp_min(1e-6)
                        print(
                            f"cache-hit/full {state_dtype} {lengths[0]} {name}: relative_rms={(delta.square().mean().sqrt() / scale).item():.6f} max/rms={(delta.abs().max() / scale).item():.6f}",
                            flush=True,
                        )

    def test_cp_packed_output_token_order(self):
        for lengths, padding in (([16384], 0), ([73, 0, 211], 0), ([129, 1], 2)):
            torch.manual_seed(47 + self.rank)
            inputs = self._inputs(lengths + ([padding] if padding else []), 0.1)
            heads = inputs["v"].shape[2]
            slots = torch.arange(len(lengths), device="cuda")
            pool = torch.randn(len(lengths), heads, 128, 128, device="cuda") * 0.1
            cu = torch.tensor(
                [0, *accumulate(lengths)], device="cuda", dtype=torch.int32
            )
            with torch.inference_mode():
                expected, final_ref, _ = self.cp.extend(
                    **inputs,
                    ssm_states=pool.clone(),
                    cache_indices=slots,
                    query_start_loc=cu,
                    seq_lens_cpu=lengths,
                )
                with patch(
                    "sglang.srt.layers.attention.linear.kernels.gdn_flashqla._context_to_heads",
                    side_effect=AssertionError("CP output must not restore TP heads"),
                ):
                    actual, final, _ = self.cp.extend(
                        **inputs,
                        ssm_states=pool.clone(),
                        cache_indices=slots,
                        query_start_loc=cu,
                        seq_lens_cpu=lengths,
                        return_cp_output=True,
                    )
                self.assertIsInstance(actual, ContextParallelOutput)
                expected_local = _heads_to_context(expected[0], group=dist.group.WORLD)
                self._close(
                    actual.hidden_states, expected_local, "CP packed token order"
                )
                self._close(final, final_ref, "CP direct final state")

    def _run(self, lengths, gate, *, inplace=True, state_dtype=torch.float32):
        torch.manual_seed(47 + self.rank)
        inputs = self._inputs(lengths, gate)
        heads = inputs["v"].shape[2]
        slots = torch.tensor(
            [3, 1, 4][: len(lengths)], device="cuda", dtype=torch.int64
        )
        # Deliberately leave holes and use noncontiguous page-major state slots.
        backing = (torch.randn(6, 2, heads, 128, 128, device="cuda") * 0.1).to(
            state_dtype
        )
        pool = backing[:, 1]
        before = pool.clone()
        ref_pool = before.clone()
        expected = torch.zeros_like(inputs["v"])
        offset = 0
        with torch.inference_mode():
            for row, length in enumerate(lengths):
                if length:
                    part = {
                        name: x[:, offset : offset + length].contiguous()
                        for name, x in inputs.items()
                    }
                    out, _, _ = self.triton.extend(
                        **part,
                        ssm_states=ref_pool,
                        cache_indices=slots[row : row + 1],
                        query_start_loc=torch.tensor(
                            [0, length], dtype=torch.int32, device="cuda"
                        ),
                    )
                    expected[:, offset : offset + length] = out
                offset += length
            cu = torch.tensor(
                [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                dtype=torch.int32,
                device="cuda",
            )
            for name, kernel in (("flashqla", self.tp), ("flashqla_cp", self.cp)):
                pool.copy_(before)
                output = torch.empty_like(expected)
                out, final, _ = kernel.extend(
                    **inputs,
                    ssm_states=pool,
                    cache_indices=slots,
                    query_start_loc=cu,
                    seq_lens_cpu=lengths,
                    inplace_update=inplace,
                    output=output,
                )
                label = f"{name} lengths={lengths} gate={gate} inplace={inplace} state={state_dtype}"
                self.assertEqual(out.data_ptr(), output.data_ptr())
                self._close(out, expected, label + " output")
                self._close(final, ref_pool[slots], label + " final state")
                self._close(pool, ref_pool if inplace else before, label + " pool")
                untouched = torch.tensor([0, 2, 5], device="cuda")
                torch.testing.assert_close(
                    pool[untouched], before[untouched], rtol=0, atol=0
                )
                # Restore TP heads before the existing row-parallel projection.
                weight = (
                    torch.randn(heads * 128, 256, dtype=torch.bfloat16, device="cuda")
                    / (heads * 128) ** 0.5
                )
                projected = out.flatten(0, 1).flatten(1) @ weight
                projected_ref = expected.flatten(0, 1).flatten(1) @ weight
                dist.all_reduce(projected)
                dist.all_reduce(projected_ref)
                self._close(projected, projected_ref, label + " projection+AR")
                if inplace:
                    continuation = self._inputs([17] * len(lengths), gate)
                    next_cu = (
                        torch.arange(len(lengths) + 1, device="cuda", dtype=torch.int32)
                        * 17
                    )
                    expected_next, _, _ = self.triton.extend(
                        **continuation,
                        ssm_states=ref_pool.clone(),
                        cache_indices=slots,
                        query_start_loc=next_cu,
                    )
                    actual_next, _, _ = self.triton.extend(
                        **continuation,
                        ssm_states=pool.contiguous(),
                        cache_indices=slots,
                        query_start_loc=next_cu,
                    )
                    self._close(actual_next, expected_next, label + " continuation")

    def test_16k_and_ragged_state_transfer(self):
        for lengths, gate, inplace in (
            ([16384], 0.0625, True),
            ([73, 0, 211], 0.0, True),
            ([1, 129], 0.0625, False),
        ):
            self._run(lengths, gate, inplace=inplace)

    def test_radix_tracking_falls_back_with_snapshots(self):
        torch.manual_seed(80 + self.rank)
        inputs = self._inputs([129], 0.0625)
        pool = torch.zeros(1, inputs["v"].shape[2], 128, 128, device="cuda")
        with torch.inference_mode():
            out, final, snapshots = self.cp.extend(
                **inputs,
                ssm_states=pool,
                cache_indices=torch.tensor([0], device="cuda"),
                query_start_loc=torch.tensor(
                    [0, 129], dtype=torch.int32, device="cuda"
                ),
                seq_lens_cpu=[129],
                output_intermediate_states=True,
            )
        self.assertIsNotNone(snapshots)
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertTrue(torch.isfinite(pool).all().item())

    def test_bfloat16_state_pool(self):
        self._run([257], 0.0, state_dtype=torch.bfloat16)

    def test_padding_sentinel_does_not_write_last_slot(self):
        torch.manual_seed(91 + self.rank)
        inputs = self._inputs([73, 64], 0.0625)
        pool = torch.randn(3, inputs["v"].shape[2], 128, 128, device="cuda")
        before = pool.clone()
        with torch.inference_mode():
            out, _, _ = self.cp.extend(
                **inputs,
                ssm_states=pool,
                cache_indices=torch.tensor([1, -1], device="cuda"),
                query_start_loc=torch.tensor(
                    [0, 73, 137], dtype=torch.int32, device="cuda"
                ),
                seq_lens_cpu=[73, 64],
            )
        torch.testing.assert_close(pool[[0, 2]], before[[0, 2]], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(out[:, 73:]).item(), 0)


if __name__ == "__main__":
    unittest.main()
