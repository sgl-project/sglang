"""Tensor-descriptor load path in the Triton extend-attention kernel (XPU).

The descriptor path must be a pure performance change: for every shape it either
engages and produces the same values as the tensor-of-pointer path, or its gate
declines and the pointer path runs. Both halves are checked here --
``test_matches_pointer_path`` compares outputs with the path forced on and off,
``test_gate_engages_per_tile`` pins which tiles actually get descriptors,
``test_gate_declines_non_contiguous_heads`` covers the one gate branch no
end-to-end shape reaches, and ``test_descriptors_emit_2d_block_loads`` asserts the
generated code really uses hardware 2D block loads -- the mechanism the whole
change exists for, and the thing that silently regressed twice upstream
(intel/intel-xpu-backend-for-triton#7852, #7956) while outputs stayed correct.

Usage:
python3 -m unittest test_extend_attention_tensor_desc.TestExtendAttentionTensorDesc
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.extend_attention import (
    _flat_descriptor_fits,
    _flat_tile_descriptor,
    _fwd_kernel,
    extend_attention_fwd,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=90, suite="stage-b-test-1-gpu-xpu")


def _build_inputs(B, N_CTX, H_Q, H_KV, Lq, Lv, dtype, device, seed=0):
    """Mirror the input construction in test_triton_attention_kernels.py."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def randint(low, high, shape):
        return torch.randint(low, high, shape, generator=gen, device=device).to(
            torch.int32
        )

    def randn(shape):
        return torch.empty(shape, dtype=dtype, device=device).normal_(
            mean=0.1, std=0.2, generator=gen
        )

    b_seq_len_prefix = randint(1, N_CTX // 2, (B,))
    b_seq_len_extend = randint(1, N_CTX // 2, (B,))
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
    kv_indices = torch.zeros(
        (int(b_seq_len_prefix.sum().item()),), dtype=torch.int32, device=device
    )
    for i in range(B):
        kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i], device=device
        )

    total_token_num = int(torch.sum(b_seq_len).item())
    extend_token_num = int(torch.sum(b_seq_len_extend).item())
    k_buffer = randn((total_token_num, H_KV, Lq))
    v_buffer = randn((total_token_num, H_KV, Lv))

    k_extend = torch.empty((extend_token_num, H_KV, Lq), dtype=dtype, device=device)
    v_extend = torch.empty((extend_token_num, H_KV, Lv), dtype=dtype, device=device)
    q_extend = torch.empty((extend_token_num, H_Q, Lq), dtype=dtype, device=device)
    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        v_extend[extend_start:extend_end] = v_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        q_extend[extend_start:extend_end] = randn((int(b_seq_len_extend[i]), H_Q, Lq))

    qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

    return dict(
        q_extend=q_extend,
        k_extend=k_extend,
        v_extend=v_extend,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        max_len_extend=int(torch.max(b_seq_len_extend, 0)[0].item()),
        extend_token_num=extend_token_num,
        H_Q=H_Q,
        Lv=Lv,
        dtype=dtype,
        device=device,
    )


@unittest.skipUnless(is_xpu(), "tensor descriptors are only auto-enabled on XPU")
class TestExtendAttentionTensorDesc(CustomTestCase):
    # (B, N_CTX, H_Q, H_KV, Lq, Lv, label)
    #
    # The last three cover the gate's fallbacks: 192/128 keeps the rope sub-tile
    # descriptors (128 + 64 == 192), 128/96 gates V off while Q/K stay on
    # (BLOCK_DV=128 != 96), and 96/96 gates every tile off.
    CONFIGS = [
        (8, 1024, 16, 16, 128, 128, "MHA, D=128"),
        (8, 1024, 28, 4, 64, 64, "GQA, D=64"),
        (4, 512, 16, 16, 192, 128, "rope sub-tile, Lq=192"),
        (4, 512, 16, 16, 128, 96, "V tile gated off, Lv=96"),
        (4, 512, 16, 16, 96, 96, "all tiles gated off, D=96"),
    ]

    def _run(self, inp, use_desc: bool) -> torch.Tensor:
        o = torch.empty(
            (inp["extend_token_num"], inp["H_Q"], inp["Lv"]),
            dtype=inp["dtype"],
            device=inp["device"],
        )
        with envs.SGLANG_USE_TRITON_ATTN_TENSOR_DESC.override(use_desc):
            extend_attention_fwd(
                inp["q_extend"],
                inp["k_extend"],
                inp["v_extend"],
                o,
                inp["k_buffer"],
                inp["v_buffer"],
                inp["qo_indptr"],
                inp["kv_indptr"],
                inp["kv_indices"],
                None,  # custom_mask
                True,  # is_causal
                None,  # mask_indptr
                inp["max_len_extend"],
                1.0,  # k_scale
                1.0,  # v_scale
            )
        return o

    def test_matches_pointer_path(self):
        """The descriptor path must return exactly the pointer path's values.

        Both paths read the same tiles in the same order, so the accumulation is
        identical and the comparison can be exact -- a tolerance here would hide
        a descriptor reading the wrong head or the wrong rows.
        """
        for B, N_CTX, H_Q, H_KV, Lq, Lv, label in self.CONFIGS:
            with self.subTest(config=label):
                inp = _build_inputs(B, N_CTX, H_Q, H_KV, Lq, Lv, torch.bfloat16, "xpu")
                expected = self._run(inp, use_desc=False)
                got = self._run(inp, use_desc=True)
                torch.testing.assert_close(got, expected, atol=0, rtol=0)

    def _compile_and_inspect(self, inp, use_desc: bool):
        """Compile exactly this variant; return (block_loads, spills, reshapes).

        The per-variant cache is cleared first so the inspected entry is
        unambiguously this variant's -- both coexist in the kernel's cache
        otherwise and it is easy to read the wrong one.
        """
        for device_cache in _fwd_kernel.device_caches.values():
            device_cache[0].clear()
        self._run(inp, use_desc)
        torch.xpu.synchronize()
        compiled = [
            kernel
            for device_cache in _fwd_kernel.device_caches.values()
            for kernel in device_cache[0].values()
        ]
        self.assertEqual(len(compiled), 1, "expected exactly one compiled variant")
        kernel = compiled[0]
        return (
            kernel.asm["llir"].count("Subgroup2DBlockLoad"),
            kernel.n_spills,
            kernel.asm["ttgir"].count("tt.reshape"),
        )

    def test_descriptors_emit_2d_block_loads(self):
        """The descriptor path must lower to hardware 2D block loads.

        Timing is not assertable in CI, but the codegen difference behind it is:
        the pointer path emits no block loads, the descriptor path emits many, no
        ``tt.reshape`` survives (rank-2 descriptors need none), and the register
        spill count does not get worse. A backend or gate regression that quietly
        drops back to gather loads keeps every other test in this file green while
        costing ~4x per call in serving; this case is what catches it.
        """
        inp = _build_inputs(2, 512, 32, 8, 128, 128, torch.bfloat16, "xpu")
        ptr_loads, ptr_spills, ptr_reshapes = self._compile_and_inspect(inp, False)
        desc_loads, desc_spills, desc_reshapes = self._compile_and_inspect(inp, True)

        self.assertEqual(ptr_loads, 0, "pointer path should issue no block loads")
        self.assertGreater(desc_loads, 0, "descriptor path issued no block loads")
        self.assertEqual(ptr_reshapes, 0)
        self.assertEqual(desc_reshapes, 0, "rank-2 descriptors need no reshape")
        self.assertLessEqual(desc_spills, ptr_spills)

    def test_gate_engages_per_tile(self):
        """Pin how many tiles actually get descriptors, per shape.

        Without this, ``test_matches_pointer_path`` would also pass if the gate
        silently declined everywhere. Counts are Q + Qpe + K + Kpe + V, with the
        rope sub-tiles present only when BLOCK_DPE > 0.
        """
        cases = [
            (128, 128, 3, "Q, K, V"),
            (192, 128, 5, "Q, Qpe, K, Kpe, V"),
            (128, 96, 2, "Q, K -- V padded"),
            (96, 96, 0, "all padded"),
        ]
        for Lq, Lv, expected, label in cases:
            with self.subTest(Lq=Lq, Lv=Lv, tiles=label):
                inp = _build_inputs(2, 256, 8, 8, Lq, Lv, torch.bfloat16, "xpu")
                with patch(
                    "sglang.kernels.ops.attention.extend_attention._flat_tile_descriptor",
                    wraps=_flat_tile_descriptor,
                ) as spy:
                    self._run(inp, use_desc=True)
                self.assertEqual(spy.call_count, expected)

    def test_gate_declines_non_contiguous_heads(self):
        """The flattened view is only the same memory when heads are contiguous."""
        head_dim = 128
        wide = torch.zeros((64, 8, head_dim * 2), dtype=torch.bfloat16, device="xpu")
        sliced = wide[:, :, :head_dim]
        self.assertEqual(sliced.stride(-2), head_dim * 2)
        self.assertFalse(
            _flat_descriptor_fits(tensor=sliced, head_dim=head_dim, tile_width=head_dim)
        )


if __name__ == "__main__":
    unittest.main()
