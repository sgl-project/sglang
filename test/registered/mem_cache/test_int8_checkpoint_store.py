"""Tests for Int8CheckpointStore (int8-compressed cached linear-attn states).

CPU tests cover the codec error bound, store/load round-trip, and the active-pool
copy-on-write helpers. The CUDA test reproduces the validated decode-output error
(int8 checkpoint loaded once then decoded continues bf16) ~ 0.5%, far below the
bf16-baseline-relative threshold that GSM8K showed is quality-safe.

    python -m pytest test/registered/mem_cache/test_int8_checkpoint_store.py -v
"""

import types
import unittest
from unittest import mock

import torch

import sglang.srt.mem_cache.mamba_checkpoint_pool as mamba_checkpoint_pool
from sglang.srt.mem_cache.mamba_checkpoint_pool import (
    Int8CheckpointStore,
    MambaCheckpointPool,
)
from sglang.test.ci.ci_register import register_cpu_ci

# These are CPU codec / store / pool unit tests (the single CUDA decode-error
# case self-skips when no GPU is present), so they run on the cheapest CI tier.
# The int8 GPU decode path is covered end-to-end by
# test_int8_mamba_checkpoint_e2e (now in the extra stage).
register_cpu_ci(est_time=6, suite="base-a-test-cpu")

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


class TestMambaCheckpointConvTransfer(unittest.TestCase):
    """Regression for the NPU conv-layout transfer (review comments on #40140).

    ``store_from_active`` / ``load_to_active`` patch the module-level ``_is_npu``
    flag (no real NPU device is needed — the branch under test is pure tensor
    slice/transpose) and verify:

      * KDA keeps its [window, channels] layout (no flip) with a non-square conv;
      * GDN/Mamba flips [channels, window] <-> [window, channels];
      * the speculative-verify window extension (spec_num_draft_tokens - 1) is
        sliced off on store and re-zeroed on load.
    """

    W, C = 3, 5  # non-square, so an unconditional transpose would break KDA

    def _pool(self, conv_shapes, is_kda, spec_draft=None):
        return MambaCheckpointPool(
            num_layers=L,
            num_slots=8,
            num_heads=H,
            head_v_dim=V,
            head_k_dim=K,
            conv_shapes=conv_shapes,
            conv_dtype=torch.float32,
            device="cpu",
            temporal_dtype=torch.float32,
            is_kda=is_kda,
            spec_num_draft_tokens=spec_draft,
        )

    def _make_active(self, conv, temporal):
        cache = types.SimpleNamespace(conv=[conv], temporal=temporal)
        return types.SimpleNamespace(mamba_cache=cache)

    def _roundtrip(self, pool, active_conv, active_slots, ckpt_slots, dst_slots):
        """Store active_slots -> ckpt_slots, then load back into dst_slots. The
        active conv at dst_slots must match the source at active_slots."""
        temporal = torch.randn(L, 4, H, V, K) * 1e-2
        active = self._make_active(active_conv, temporal)
        pool.store_from_active(active, active_slots, ckpt_slots)
        expected = active_conv[:, active_slots].clone()
        pool.load_to_active(active, ckpt_slots, dst_slots)
        return expected, active_conv[:, dst_slots]

    def test_npu_kda_conv_no_transpose_non_square(self):
        with mock.patch.object(mamba_checkpoint_pool, "_is_npu", True):
            pool = self._pool([(self.W, self.C)], is_kda=True)
            active_conv = torch.randn(L, 4, self.W, self.C)

            expected, got = self._roundtrip(
                pool,
                active_conv,
                torch.tensor([1, 2]),
                torch.tensor([3, 4]),
                torch.tensor([0, 1]),
            )
            # KDA: checkpoint == active, no (2,3) flip
            self.assertTrue(torch.equal(got, expected))

    def test_npu_gdn_conv_transpose_non_square(self):
        with mock.patch.object(mamba_checkpoint_pool, "_is_npu", True):
            # GDN conv_shapes read (channels, window)
            pool = self._pool([(self.C, self.W)], is_kda=False)
            active_conv = torch.randn(L, 4, self.W, self.C)

            expected, got = self._roundtrip(
                pool,
                active_conv,
                torch.tensor([1, 2]),
                torch.tensor([3, 4]),
                torch.tensor([0, 1]),
            )
            # GDN: checkpoint [C, W] == active.transpose(2,3), and the load
            # flips back, so the round-trip is exact
            self.assertTrue(torch.equal(got, expected))

    def test_npu_spec_window_extension_sliced_and_zeroed(self):
        num_draft = 4  # extra_conv_len = 3
        extra = num_draft - 1
        for is_kda, conv_shapes in ((True, [(self.W, self.C)]), (False, [(self.C, self.W)])):
            with self.subTest(is_kda=is_kda):
                with mock.patch.object(mamba_checkpoint_pool, "_is_npu", True):
                    pool = self._pool(conv_shapes, is_kda, spec_draft=num_draft)
                    active_conv = torch.randn(L, 4, self.W + extra, self.C)
                    temporal = torch.randn(L, 4, H, V, K) * 1e-2
                    active = self._make_active(active_conv, temporal)

                    active_slots = torch.tensor([1])
                    ckpt_slots = torch.tensor([2])
                    pool.store_from_active(active, active_slots, ckpt_slots)
                    # checkpoint keeps only the first W window slots; GDN/Mamba
                    # flips them to [C, W], KDA keeps [W, C]
                    expected = active_conv[:, [1], : self.W]
                    if not is_kda:
                        expected = expected.transpose(2, 3)
                    self.assertTrue(torch.equal(pool.conv[0][:, [2]], expected))

                    dst_slots = torch.tensor([0])
                    active_conv[:, dst_slots] = torch.randn_like(
                        active_conv[:, dst_slots]
                    )
                    pool.load_to_active(active, ckpt_slots, dst_slots)
                    # restore the first W window, zero the extension
                    self.assertTrue(
                        torch.equal(
                            active_conv[:, [0], : self.W],
                            active_conv[:, [1], : self.W],
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            active_conv[:, [0], self.W :],
                            torch.zeros_like(active_conv[:, [0], self.W :]),
                        )
                    )


if __name__ == "__main__":
    unittest.main()
