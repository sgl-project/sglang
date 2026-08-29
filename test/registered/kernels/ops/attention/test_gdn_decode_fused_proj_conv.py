import unittest

import torch

from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    can_use_fused_qkvzba_causal_conv1d_update_contiguous,
    fused_qkvzba_causal_conv1d_update_contiguous,
    fused_qkvzba_split_reshape_cat_contiguous,
)

# This is also the update implementation imported directly by GDNBackend on
# CUDA; the presence of the optional sgl_kernel AOT extension does not reroute
# GDN decode through srt.layers.attention.mamba.causal_conv1d.
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")


def _reference(
    qkvz,
    ba,
    state,
    weight,
    bias,
    indices,
    *,
    qkv_dim,
    v_dim,
    num_v_heads,
    head_v_dim,
    activation,
):
    qkv = qkvz[:, :qkv_dim]
    out = torch.empty_like(qkv)
    state_out = state.clone()
    width = weight.shape[1]
    for row, slot_tensor in enumerate(indices.cpu()):
        slot = int(slot_tensor)
        if slot < 0 or slot >= state.shape[0]:
            out[row].copy_(qkv[row])
            continue
        # The deployed direct-Triton decode wrapper uses an effective
        # state_len=width-1 even if the physical cache envelope is wider.
        history = state[slot, :, : width - 1].float()
        values = torch.cat((history, qkv[row, :, None].float()), dim=-1)
        acc = (values * weight.float()).sum(dim=-1)
        if bias is not None:
            acc = acc + bias.float()
        if activation in ("silu", "swish"):
            acc = torch.nn.functional.silu(acc)
        out[row].copy_(acc.to(qkv.dtype))
        if width > 2:
            state_out[slot, :, : width - 2].copy_(state[slot, :, 1 : width - 1])
        state_out[slot, :, width - 2].copy_(qkv[row])

    z = qkvz[:, qkv_dim:].reshape(-1, num_v_heads, head_v_dim).contiguous()
    b, a = ba.split([num_v_heads, num_v_heads], dim=-1)
    return out, z, b.contiguous(), a.contiguous(), state_out


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestGDNDecodeFusedProjectionConv1D(unittest.TestCase):
    def test_contiguous_unpack_ratio8_microbenchmark_baseline(self):
        batch = 9
        num_qk_heads = 1
        num_v_heads = 8
        head_dim = 128
        qkv_dim = (2 * num_qk_heads + num_v_heads) * head_dim
        v_dim = num_v_heads * head_dim
        qkvz = torch.randn(
            batch,
            qkv_dim + v_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        ba = torch.randn(
            batch,
            2 * num_v_heads,
            device="cuda",
            dtype=torch.bfloat16,
        )
        mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
            qkvz,
            ba,
            num_qk_heads,
            num_v_heads,
            head_dim,
            head_dim,
        )
        torch.testing.assert_close(mixed_qkv, qkvz[:, :qkv_dim])
        torch.testing.assert_close(
            z, qkvz[:, qkv_dim:].reshape(batch, num_v_heads, head_dim)
        )
        torch.testing.assert_close(b, ba[:, :num_v_heads])
        torch.testing.assert_close(a, ba[:, num_v_heads:])

    def _run_case(
        self,
        *,
        batch,
        q_dim,
        k_dim,
        v_dim,
        num_v_heads,
        head_v_dim,
        width,
        state_len,
        dtype,
        with_bias,
        activation,
        strided_state=False,
        with_padding=False,
    ):
        torch.manual_seed(17)
        device = "cuda"
        qkv_dim = q_dim + k_dim + v_dim
        qkvz = torch.randn(batch, qkv_dim + v_dim, device=device, dtype=dtype)
        ba = torch.randn(batch, 2 * num_v_heads, device=device, dtype=dtype)
        weight = torch.randn(qkv_dim, width, device=device, dtype=dtype) * 0.1
        bias = (
            torch.randn(qkv_dim, device=device, dtype=dtype) * 0.1
            if with_bias
            else None
        )
        slots = batch + 3
        if strided_state:
            backing = torch.randn(
                slots,
                state_len,
                qkv_dim * 2,
                device=device,
                dtype=dtype,
            )
            state = backing[:, :, ::2].transpose(1, 2)
            self.assertFalse(state.is_contiguous())
        else:
            state = torch.randn(slots, qkv_dim, state_len, device=device, dtype=dtype)
        indices = torch.randperm(slots, device=device, dtype=torch.int64)[:batch]
        if with_padding:
            indices[-1] = -1
        indices = indices.to(torch.int32)

        ref = _reference(
            qkvz,
            ba,
            state,
            weight,
            bias,
            indices,
            qkv_dim=qkv_dim,
            v_dim=v_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            activation=activation,
        )
        state_test = state.clone(memory_format=torch.preserve_format)
        out, z, b, a = fused_qkvzba_causal_conv1d_update_contiguous(
            qkvz,
            ba,
            state_test,
            weight,
            bias,
            indices,
            qkv_dim=qkv_dim,
            v_dim=v_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            activation=activation,
        )
        torch.cuda.synchronize()

        atol = 2e-2 if dtype == torch.bfloat16 else 3e-3
        output_max_diff = (out.float() - ref[0].float()).abs().max().item()
        state_max_diff = (state_test.float() - ref[4].float()).abs().max().item()
        print(
            "case "
            f"B={batch} QKV={qkv_dim} V={v_dim} W={width} "
            f"dtype={dtype} output_max_diff={output_max_diff:.8g} "
            f"state_max_diff={state_max_diff:.8g}",
            flush=True,
        )
        torch.testing.assert_close(out, ref[0], rtol=0, atol=atol)
        torch.testing.assert_close(z, ref[1], rtol=0, atol=0)
        torch.testing.assert_close(b, ref[2], rtol=0, atol=0)
        torch.testing.assert_close(a, ref[3], rtol=0, atol=0)
        torch.testing.assert_close(state_test, ref[4], rtol=0, atol=0)
        self.assertEqual(b.data_ptr() % 32, 0)
        self.assertEqual(a.data_ptr() % 32, 0)

    def test_random_shapes_widths_dtypes_and_state_updates(self):
        cases = (
            # Small boundary shapes.
            dict(
                batch=1,
                q_dim=16,
                k_dim=16,
                v_dim=32,
                num_v_heads=2,
                head_v_dim=16,
                width=2,
                state_len=1,
                dtype=torch.float16,
                with_bias=False,
                activation=None,
            ),
            dict(
                batch=17,
                q_dim=32,
                k_dim=32,
                v_dim=64,
                num_v_heads=4,
                head_v_dim=16,
                width=3,
                state_len=5,
                dtype=torch.bfloat16,
                with_bias=True,
                activation="silu",
                strided_state=True,
                with_padding=True,
            ),
            # Qwen3.5-35B TP16 local GDN dimensions.
            dict(
                batch=32,
                q_dim=128,
                k_dim=128,
                v_dim=256,
                num_v_heads=2,
                head_v_dim=128,
                width=4,
                state_len=3,
                dtype=torch.bfloat16,
                with_bias=False,
                activation="silu",
            ),
            # Large TP-local GDN dimensions with an 8:1 value/key head ratio.
            dict(
                batch=8,
                q_dim=128,
                k_dim=128,
                v_dim=1024,
                num_v_heads=8,
                head_v_dim=128,
                width=4,
                state_len=3,
                dtype=torch.bfloat16,
                with_bias=False,
                activation="silu",
            ),
        )
        for case in cases:
            with self.subTest(case=case):
                self._run_case(**case)

    def test_cuda_graph_replay(self):
        batch = 4
        qkv_dim, v_dim, num_v_heads, head_v_dim = 128, 64, 2, 32
        qkvz = torch.randn(batch, qkv_dim + v_dim, device="cuda", dtype=torch.bfloat16)
        ba = torch.randn(batch, 2 * num_v_heads, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(qkv_dim, 4, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(batch + 1, qkv_dim, 3, device="cuda", dtype=torch.bfloat16)
        initial_state = state.clone()
        indices = torch.arange(batch, device="cuda", dtype=torch.int32)
        # Compile before capture; Triton compilation itself is not graph-safe.
        fused_qkvzba_causal_conv1d_update_contiguous(
            qkvz,
            ba,
            state,
            weight,
            None,
            indices,
            qkv_dim=qkv_dim,
            v_dim=v_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            activation="silu",
        )
        state.copy_(initial_state)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fused_qkvzba_causal_conv1d_update_contiguous(
                qkvz,
                ba,
                state,
                weight,
                None,
                indices,
                qkv_dim=qkv_dim,
                v_dim=v_dim,
                num_v_heads=num_v_heads,
                head_v_dim=head_v_dim,
                activation="silu",
            )
        state.copy_(initial_state)
        graph.replay()
        ref_state = initial_state.clone()
        ref_qkv, ref_z, ref_b, ref_a = fused_qkvzba_split_reshape_cat_contiguous(
            qkvz,
            ba,
            1,
            num_v_heads,
            32,
            head_v_dim,
        )
        ref_qkv = causal_conv1d_update(
            ref_qkv,
            ref_state,
            weight,
            None,
            "silu",
            conv_state_indices=indices,
        )
        torch.testing.assert_close(captured[0], ref_qkv, rtol=0, atol=0)
        torch.testing.assert_close(captured[1], ref_z, rtol=0, atol=0)
        torch.testing.assert_close(captured[2], ref_b, rtol=0, atol=0)
        torch.testing.assert_close(captured[3], ref_a, rtol=0, atol=0)
        torch.testing.assert_close(state, ref_state, rtol=0, atol=0)

    def test_out_of_range_state_slots_are_safely_masked(self):
        torch.manual_seed(29)
        batch = 4
        qkv_dim, v_dim, num_v_heads, head_v_dim = 128, 64, 2, 32
        qkvz = torch.randn(batch, qkv_dim + v_dim, device="cuda", dtype=torch.bfloat16)
        ba = torch.randn(batch, 2 * num_v_heads, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(qkv_dim, 4, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(3, qkv_dim, 3, device="cuda", dtype=torch.bfloat16)
        # -1 is the expected padding sentinel; -2 and len(state) exercise the
        # hard lower/upper bounds.  Slot 1 remains a normal live update.
        indices = torch.tensor([-2, -1, state.shape[0], 1], device="cuda")
        indices = indices.to(torch.int32)

        ref = _reference(
            qkvz,
            ba,
            state,
            weight,
            None,
            indices,
            qkv_dim=qkv_dim,
            v_dim=v_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            activation="silu",
        )
        state_test = state.clone()
        out, z, b, a = fused_qkvzba_causal_conv1d_update_contiguous(
            qkvz,
            ba,
            state_test,
            weight,
            None,
            indices,
            qkv_dim=qkv_dim,
            v_dim=v_dim,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            activation="silu",
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(out, ref[0], rtol=0, atol=2e-2)
        torch.testing.assert_close(z, ref[1], rtol=0, atol=0)
        torch.testing.assert_close(b, ref[2], rtol=0, atol=0)
        torch.testing.assert_close(a, ref[3], rtol=0, atol=0)
        torch.testing.assert_close(state_test, ref[4], rtol=0, atol=0)

    def test_fp8_activation_is_an_explicit_fallback(self):
        if not hasattr(torch, "float8_e4m3fn"):
            self.skipTest("PyTorch has no FP8 dtype")
        qkvz = torch.empty(1, 96, device="cuda", dtype=torch.float8_e4m3fn)
        ba = torch.empty(1, 4, device="cuda", dtype=torch.bfloat16)
        state = torch.empty(2, 64, 3, device="cuda", dtype=torch.bfloat16)
        weight = torch.empty(64, 4, device="cuda", dtype=torch.bfloat16)
        indices = torch.zeros(1, device="cuda", dtype=torch.int32)
        eligible, reason = can_use_fused_qkvzba_causal_conv1d_update_contiguous(
            qkvz,
            ba,
            state,
            weight,
            None,
            indices,
            qkv_dim=64,
            v_dim=32,
            num_v_heads=2,
            activation="silu",
        )
        self.assertFalse(eligible)
        self.assertIn("FP16, BF16, or FP32", reason)


if __name__ == "__main__":
    unittest.main()
