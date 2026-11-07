"""Tests for Int8CheckpointStore (int8-compressed cached linear-attn states).

CPU tests cover the codec error bound, store/load round-trip, and the active-pool
copy-on-write helpers. The CUDA test reproduces the validated decode-output error
(int8 checkpoint loaded once then decoded continues bf16) ~ 0.5%, far below the
bf16-baseline-relative threshold that GSM8K showed is quality-safe.

    python -m pytest test/registered/mem_cache/test_int8_checkpoint_store.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.mamba_checkpoint_pool import (
    Int8CheckpointStore,
    MambaCheckpointPool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# These are CPU codec / store / pool unit tests (the single CUDA decode-error
# case self-skips when no GPU is present), so they run on the cheapest CI tier.
# The int8 GPU decode path is covered end-to-end by
# test_int8_mamba_checkpoint_e2e (now in the extra stage).
register_cpu_ci(est_time=12, suite="base-a-test-cpu")

H, V, K = 32, 128, 128
L = 4


def _rand_state(n, device="cpu"):
    # KDA-like state magnitudes (see fp8_checkpoint_probe: |S| mean ~6e-2)
    return torch.randn(L, n, H, V, K, device=device) * 6e-2


class TestInt8CheckpointCodec(unittest.TestCase):
    def test_quantize_dequantize_error_bound(self):
        s = _rand_state(8)
        q, scale = Int8CheckpointStore.quantize(s)
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(scale.shape, (L, 8, H, 1, K))  # per (layer,slot,head,k-chan)
        deq = Int8CheckpointStore.dequantize(q, scale, torch.float32)
        rel = (deq - s).norm() / s.norm()
        # uniform int8 per-channel on a ~uniform state: well under 1%
        self.assertLess(rel.item(), 1e-2, f"int8 codec rel err too high: {rel}")

    def test_symmetric_and_zero(self):
        s = torch.zeros(L, 1, H, V, K)
        q, scale = Int8CheckpointStore.quantize(s)
        self.assertTrue(torch.equal(q, torch.zeros_like(q)))
        deq = Int8CheckpointStore.dequantize(q, scale, torch.float32)
        self.assertTrue(torch.equal(deq, s))

    def test_store_load_roundtrip(self):
        store = Int8CheckpointStore(
            num_layers=L,
            num_slots=16,
            num_heads=H,
            head_v_dim=V,
            head_k_dim=K,
            device="cpu",
        )
        s = _rand_state(4)
        slots = torch.tensor([1, 3, 5, 7])
        store.store(slots, s)
        out = store.load(slots, torch.float32)
        # load == dequant of stored
        q, scale = Int8CheckpointStore.quantize(s)
        ref = Int8CheckpointStore.dequantize(
            q, scale.to(store.scale.dtype), torch.float32
        )
        self.assertLess((out - ref).abs().max().item(), 1e-3)

    def test_cow_helpers(self):
        store = Int8CheckpointStore(
            num_layers=L,
            num_slots=16,
            num_heads=H,
            head_v_dim=V,
            head_k_dim=K,
            device="cpu",
        )
        active = torch.zeros(L, 10, H, V, K)  # bf16/fp32 active pool
        active[:, 2] = _rand_state(1).squeeze(1)
        # store active slot 2 -> ckpt slot 4
        store.store_from_pool(active, torch.tensor([2]), torch.tensor([4]))
        # load ckpt slot 4 -> active slot 6 (cache-hit COW)
        store.copy_to_pool(active, torch.tensor([4]), torch.tensor([6]))
        rel = (active[:, 6] - active[:, 2]).norm() / active[:, 2].norm()
        self.assertLess(rel.item(), 1e-2)

    def test_memory_is_half_of_bf16(self):
        store = Int8CheckpointStore(
            num_layers=L,
            num_slots=100,
            num_heads=H,
            head_v_dim=V,
            head_k_dim=K,
            device="cpu",
        )
        bf16_per_slot = L * H * V * K * 2
        # int8 data (1B) + small per-(head,k) bf16 scale -> well under bf16; ~2x slots
        self.assertLess(store.bytes_per_slot(), bf16_per_slot * 0.6)

    def test_estimate_matches_actual_mem(self):
        # the pre-allocation estimate (used to fit-check HBM before building the
        # pool) must equal the real allocated footprint, for any temporal dtype
        for tdt in (torch.bfloat16, torch.float32):
            kw = dict(
                num_layers=L,
                num_slots=64,
                num_heads=H,
                head_v_dim=V,
                head_k_dim=K,
                conv_shapes=[(4, K)],
                conv_dtype=torch.bfloat16,
                temporal_dtype=tdt,
            )
            est = MambaCheckpointPool.estimate_mem_usage_bytes(**kw)
            pool = MambaCheckpointPool(**kw, device="cpu")
            self.assertEqual(est["qdata"] + est["scale"] + est["conv"], est["total"])
            self.assertEqual(est["total"], pool.mem_usage_bytes())


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA + fla kernels")
class TestInt8CheckpointDecodeError(unittest.TestCase):
    def test_decode_error_within_bound(self):
        try:
            from sglang.kernels.ops.attention.fla.kda import fused_recurrent_kda
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"fla kernels unavailable: {e}")

        dev = "cuda"
        torch.manual_seed(0)

        def synth(T, s):
            torch.manual_seed(s)
            q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16) * 0.5
            k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16) * 0.5
            v = (torch.randn(1, T, H, V, device=dev) * 0.5).bfloat16()
            beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
            g = -torch.rand(1, T, H, K, device=dev, dtype=torch.float32) * 0.1 - 0.005
            return q, k, v, g, beta

        def decode(state, inp):
            st = state.clone()
            o, _ = fused_recurrent_kda(
                q=inp[0],
                k=inp[1],
                v=inp[2],
                g=inp[3],
                beta=inp[4],
                scale=K**-0.5,
                initial_state=st,
                inplace_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=None,
            )
            return o.float()

        S = torch.zeros(1, H, V, K, device=dev, dtype=torch.float32)
        pre = synth(512, 0)
        fused_recurrent_kda(
            q=pre[0],
            k=pre[1],
            v=pre[2],
            g=pre[3],
            beta=pre[4],
            scale=K**-0.5,
            initial_state=S,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )
        dec = synth(128, 1)
        o_ref = decode(S, dec)
        q, scale = Int8CheckpointStore.quantize(S)  # [1,H,V,K]
        S_int8 = Int8CheckpointStore.dequantize(q, scale, torch.float32)
        o_int8 = decode(S_int8, dec)
        rel = (o_int8 - o_ref).norm() / o_ref.norm()
        self.assertLess(rel.item(), 1.5e-2, f"int8 decode err {rel} too high")


class TestInt8CheckpointAdmission(CustomTestCase):
    """An unavailable checkpoint must not consume a request's tracking state."""

    def setUp(self):
        from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sglang.srt.mem_cache.static_kv_storage import StaticKVStorage
        from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
        from sglang.test.unified_allocator_fixtures import setup_allocator_context

        setup_allocator_context()

        self.checkpoints = MambaCheckpointPool(
            num_layers=2,
            num_slots=2,
            num_heads=2,
            head_v_dim=16,
            head_k_dim=16,
            conv_shapes=[(3, 8)],
            conv_dtype=torch.bfloat16,
            temporal_dtype=torch.bfloat16,
            device="cpu",
        )
        self.active = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.randn(2, 6, 2, 16, 16).bfloat16(),
                conv=[torch.randn(2, 6, 3, 8).bfloat16()],
            )
        )
        self.slots = MambaSlotAllocator(5, "cpu")
        self.slots.alloc(3)
        self.requests = object.__new__(HybridReqToTokenPool)
        self.requests.mamba_ping_pong_track_buffer_size = 2
        self.requests.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.tensor(
            [[2, 3]]
        )
        storage = StaticKVStorage(
            self.requests,
            SimpleNamespace(
                state_slots=self.slots,
                state=SimpleNamespace(pool=self.active, ckpt_pool=self.checkpoints),
            ),
        )
        self.req = SimpleNamespace(
            kv=SimpleNamespace(
                mamba_pool_idx=torch.tensor(1),
                mamba_ping_pong_track_buffer=torch.tensor([2, 3]),
                mamba_ping_pong_empty=[False, False],
                mamba_last_track_idx=0,
                mamba_last_track_seqlen=64,
                req_pool_idx=0,
            )
        )
        self.component = object.__new__(MambaComponent)
        self.component.host_only_checkpoint = False
        # Tree eviction is the boundary: a no-op means all checkpoints are pinned.
        self.component.cache = SimpleNamespace(
            kv_storage=lambda: storage,
            req_to_token_pool=self.requests,
            enable_mamba_extra_buffer=True,
            evict=lambda _: None,
            evict_for_alloc=lambda _: None,
        )
        self.params = SimpleNamespace(mamba_value=None, mamba_keep_idx=None)

    def tearDown(self):
        from sglang.srt.runtime_context import reset_context

        reset_context()

    def prepare(self, finished=False):
        return self.component.prepare_for_caching_req(
            self.req, self.params, 64, finished
        )

    def assert_tracking_unchanged(self):
        self.assertEqual(self.req.kv.mamba_ping_pong_track_buffer.tolist(), [2, 3])
        self.assertEqual(
            self.requests.req_index_to_mamba_ping_pong_track_buffer_mapping.tolist(),
            [[2, 3]],
        )

    def test_pinned_checkpoints_skip_unfinished_without_donation(self):
        self.checkpoints.alloc(2)
        self.assertEqual(self.prepare(), 0)
        self.assertIsNone(self.params.mamba_value)
        self.assertEqual(self.slots.available_size(), 2)
        self.assert_tracking_unchanged()

    def test_pinned_checkpoints_skip_finished_without_donation(self):
        self.checkpoints.alloc(2)
        self.assertEqual(self.prepare(finished=True), 0)
        self.assertIsNone(self.params.mamba_value)
        self.assertEqual(self.slots.available_size(), 2)
        self.assert_tracking_unchanged()

    def test_active_exhaustion_returns_reserved_checkpoint(self):
        self.slots.alloc(2)
        self.assertEqual(self.prepare(), 0)
        self.assertEqual(self.checkpoints.available_size(), 2)
        self.assert_tracking_unchanged()

    def test_store_error_returns_reservations_before_donation(self):
        # A real out-of-range source raises inside the eager codec. The request
        # and the two allocators must still own exactly their original slots.
        self.req.kv.mamba_ping_pong_track_buffer[0] = 8
        with self.assertRaises(IndexError):
            self.prepare()
        self.assertEqual(self.checkpoints.available_size(), 2)
        self.assertEqual(self.slots.available_size(), 2)
        self.assertEqual(self.req.kv.mamba_ping_pong_track_buffer.tolist(), [8, 3])

    def test_eviction_can_make_checkpoint_admission_succeed(self):
        held = self.checkpoints.alloc(2)
        self.component.cache.evict = lambda _: self.checkpoints.free(held[:1])
        source = self.active.mamba_cache.temporal[:, 2].clone()
        self.assertEqual(self.prepare(), 64)
        self.assertEqual(self.params.mamba_value.tolist(), [1])
        self.assertEqual(self.req.kv.mamba_ping_pong_track_buffer.tolist(), [4, 3])
        self.assertEqual(self.slots.available_size(), 2)
        expected_q, expected_scale = Int8CheckpointStore.quantize(source.unsqueeze(1))
        self.assertTrue(
            torch.equal(self.checkpoints.temporal.qdata[:, [1]], expected_q)
        )
        self.assertTrue(
            torch.equal(
                self.checkpoints.temporal.scale[:, [1]], expected_scale.bfloat16()
            )
        )


if __name__ == "__main__":
    unittest.main()
