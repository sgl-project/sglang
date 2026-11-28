"""Tests for Int8CheckpointStore (int8-compressed cached linear-attn states).

CPU tests cover the codec error bound for both scale axes, the per-value-channel
bound of the "v" axis, store/load round-trip, and the active-pool copy-on-write
helpers. The CUDA tests reproduce the validated decode-output error (int8
checkpoint loaded once then decoded continues bf16) ~ 0.5% on KDA, far below the
bf16-baseline-relative threshold that GSM8K showed is quality-safe, and check the
same error per value channel on a GDN (per-head scalar gate) recurrence.

    python -m pytest test/registered/mem_cache/test_int8_checkpoint_store.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.mamba_checkpoint_pool import (
    Int8CheckpointStore,
    MambaCheckpointPool,
    int8_scale_axis,
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

    # ---- scale_axis="v": one scale per (head, v-channel), for scalar-gate models ----

    def test_scale_axis_follows_gate(self):
        # KDA gates per k-channel; GDN / Mamba2 gate a head with one scalar
        self.assertEqual(int8_scale_axis(SimpleNamespace(is_kda=True)), "k")
        self.assertEqual(int8_scale_axis(SimpleNamespace(is_kda=False)), "v")
        with self.assertRaises(ValueError):
            Int8CheckpointStore.quantize(_rand_state(1), scale_axis="hv")

    def test_v_axis_error_bound(self):
        s = _rand_state(8)[..., : K // 2]  # d_v != d_k: the scale follows d_v
        q, scale = Int8CheckpointStore.quantize(s, scale_axis="v")
        self.assertEqual(q.dtype, torch.int8)
        self.assertEqual(scale.shape, (L, 8, H, V, 1))  # per (layer,slot,head,v-chan)
        deq = Int8CheckpointStore.dequantize(q, scale, torch.float32)
        rel = (deq - s).norm() / s.norm()
        self.assertLess(rel.item(), 1e-2, f"int8 codec rel err too high: {rel}")
        # every entry is within half a step of its own row's scale
        self.assertTrue(torch.all((deq - s).abs() <= 0.501 * scale))

    def test_v_axis_bounds_each_value_channel(self):
        # Per-head scalar gates (GDN / Mamba2): row v of S is written by value
        # channel v, and S is read contracted over k, so the error of row v is the
        # error of output channel v. A few large value channels must not set the
        # rounding step of the small ones.
        torch.manual_seed(0)
        s = _rand_state(2)
        s[..., :4, :] *= 30.0  # 4 of the 128 value channels dominate
        query = torch.nn.functional.normalize(torch.randn(K), dim=0)

        def readout_err(scale_axis):  # relative error per value channel, [V]
            q, scale = Int8CheckpointStore.quantize(s, scale_axis=scale_axis)
            deq = Int8CheckpointStore.dequantize(q, scale, torch.float32)
            o_ref, o = s @ query, deq @ query  # [L, n, H, V]
            num = torch.linalg.vector_norm(o - o_ref, dim=(0, 1, 2))
            return num / torch.linalg.vector_norm(o_ref, dim=(0, 1, 2))

        err_v, err_k = readout_err("v"), readout_err("k")
        self.assertLess(err_v.max().item(), 2e-2, f"v-axis readout err {err_v.max()}")
        # the k axis rounds the small rows with the large rows' step
        self.assertGreater(err_k[4:].median().item(), 5 * err_v[4:].median().item())

    def test_v_axis_pool_roundtrip_and_estimate(self):
        kw = dict(
            num_layers=L,
            num_slots=16,
            num_heads=H,
            head_v_dim=V,
            head_k_dim=K // 2,  # d_v != d_k: the two axes differ in scale bytes
            conv_shapes=[(4, K)],
            conv_dtype=torch.bfloat16,
            temporal_dtype=torch.float32,
            scale_axis="v",
        )
        est = MambaCheckpointPool.estimate_mem_usage_bytes(**kw)
        pool = MambaCheckpointPool(**kw, device="cpu")
        self.assertEqual(est["total"], pool.mem_usage_bytes())
        store = pool.temporal
        self.assertEqual(store.scale.shape[-2:], (V, 1))
        active = torch.zeros(L, 10, H, V, K // 2)
        active[:, 2] = _rand_state(1).squeeze(1)[..., : K // 2]
        # store active slot 2 -> ckpt slot 4 -> active slot 6 (cache-hit COW)
        store.store_from_pool(active, torch.tensor([2]), torch.tensor([4]))
        store.copy_to_pool(active, torch.tensor([4]), torch.tensor([6]))
        rel = (active[:, 6] - active[:, 2]).norm() / active[:, 2].norm()
        self.assertLess(rel.item(), 1e-2)


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

    def test_gdn_decode_error_per_value_channel(self):
        """GDN gates a head with one scalar, so the rows of S follow the magnitude
        of their value channel. Loading the int8 checkpoint with the "v" scale axis
        keeps every output channel close to the exact decode; the "k" axis loses
        the small value channels."""
        try:
            from sglang.kernels.ops.attention.fla.fused_recurrent import (
                fused_recurrent_gated_delta_rule,
            )
        except (ImportError, ModuleNotFoundError) as e:
            self.skipTest(f"fla kernels unavailable: {e}")

        dev = "cuda"
        # value-channel magnitudes spread over 1.5 decades
        channel = torch.logspace(-1.5, 0.0, V, device=dev)

        def synth(T, s):
            torch.manual_seed(s)
            q = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16) * 0.5
            k = torch.randn(1, T, H, K, device=dev, dtype=torch.bfloat16) * 0.5
            v = (torch.randn(1, T, H, V, device=dev) * 0.5 * channel).bfloat16()
            beta = torch.rand(1, T, H, device=dev, dtype=torch.bfloat16)
            g = -torch.rand(1, T, H, device=dev, dtype=torch.float32) * 0.1 - 0.005
            return q, k, v, g, beta

        def run(state, inp):
            return fused_recurrent_gated_delta_rule(
                q=inp[0],
                k=inp[1],
                v=inp[2],
                g=inp[3],
                beta=inp[4],
                scale=K**-0.5,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

        S0 = torch.zeros(1, H, V, K, device=dev, dtype=torch.float32)
        _, S = run(S0, synth(512, 0))
        dec = synth(16, 1)  # the tokens right after the cache hit
        o_ref = run(S, dec)[0].float()

        def decode_err(scale_axis):  # relative error per value channel, [V]
            q, scale = Int8CheckpointStore.quantize(S, scale_axis=scale_axis)
            S_int8 = Int8CheckpointStore.dequantize(q, scale, torch.float32)
            o = run(S_int8, dec)[0].float()
            num = torch.linalg.vector_norm(o - o_ref, dim=(0, 1, 2))
            return num / torch.linalg.vector_norm(o_ref, dim=(0, 1, 2))

        err_v, err_k = decode_err("v"), decode_err("k")
        self.assertLess(err_v.max().item(), 2e-2, f"v-axis decode err {err_v.max()}")
        self.assertGreater(
            err_k.max().item(),
            5 * err_v.max().item(),
            f"k-axis {err_k.max()} vs v-axis {err_v.max()}",
        )


if __name__ == "__main__":
    unittest.main()
