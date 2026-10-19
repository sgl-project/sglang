"""Unit tests for the two fused MXFP4 scatters.

The W4A4 pre-permute has two implementations behind
`SGLANG_USE_DEEPGEMM_W4A4_FUSED_SCATTER`:

    fused     : ep_scatter_quant_mxfp4(recv_x, ...)                [compact]
                quant_scatter_mxfp4_masked(x, ...)                 [masked]
    two-step  : quant_mxfp4_group32_v2(recv_x) -> ep_scatter(q, sf, ...)
                           -> fill_gateup_input_triton_kernel(..., SCALE_MN_MAJOR)

Both fused kernels' contract is that they emit *exactly* what the two-step form
would (commit message: "Output is bit-identical to the two-step path ... m_indices,
q rows, scale words, exact-once coverage"), so this file pins that equality and
also exercises both wrappers' validation branches - the runner-side test
(`test_w4a4_deep_gemm.py`) mocks the compact one, so nothing else calls it.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from sglang.kernels.ops.moe.ep_moe_kernels import (
    ep_scatter,
    ep_scatter_quant_mxfp4,
    fill_gateup_input_triton_kernel,
    fused_moe_dispatch_index,
    quant_scatter_mxfp4_masked,
)
from sglang.kernels.ops.quantization.mxfp4_group_quant import quant_mxfp4_group32_v2
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


E, TOPK = 2, 2


def _topk_ids(T):
    """Deterministic routing: expert 0 gets rows (0,0) (0,1) (1,0) (2,0), expert 1 the rest."""
    ids = [[0, 1], [0, 1], [0, 0]]
    # Slice the tensor, not the list: `[][:0]` is 1D and would trip the wrapper's
    # "must be 2D" check in the T == 0 case.
    return torch.tensor(ids, dtype=torch.int32, device="cuda")[:T]


def _make_inputs(T=3, K=256):
    """Inputs in the compact-layout shape the runner produces.

    `num_recv_tokens_per_expert` is the 128-aligned count (sums to the padded
    buffer height), `num_valid_tokens_per_expert` the real one - the same
    distinction `_fwd_kernel_ep_scatter_1` relies on to write -1 padding rows.
    """
    recv_x = torch.randn(T, K, dtype=torch.bfloat16, device="cuda")
    recv_topk = _topk_ids(T)
    valid = torch.bincount(recv_topk.flatten().to(torch.int64), minlength=E).to(
        torch.int32
    )
    padded = (torch.div(valid + 127, 128, rounding_mode="floor") * 128).to(torch.int32)
    all_tokens = int(padded.sum().item())
    return dict(
        recv_x=recv_x,
        recv_topk=recv_topk,
        num_recv_tokens_per_expert=padded,
        num_valid_tokens_per_expert=valid,
        expert_start_loc=torch.zeros(E, dtype=torch.int32, device="cuda"),
        output_tensor=torch.empty(all_tokens, K // 2, dtype=torch.int8, device="cuda"),
        output_tensor_scale=torch.zeros(
            K // 128, all_tokens, dtype=torch.int32, device="cuda"
        ).transpose(0, 1),
        m_indices=torch.empty(all_tokens, dtype=torch.int32, device="cuda"),
        output_index=torch.empty(T, TOPK, dtype=torch.int32, device="cuda"),
        all_tokens=all_tokens,
    )


def _run_two_step(inp):
    """The two-step reference: quantize the whole tensor, then scatter it."""
    q, sf = quant_mxfp4_group32_v2(inp["recv_x"])
    out = torch.empty_like(inp["output_tensor"])
    scale_storage = torch.zeros(
        inp["output_tensor_scale"].shape[1],
        inp["all_tokens"],
        dtype=torch.int32,
        device="cuda",
    ).transpose(0, 1)
    m_indices = torch.empty_like(inp["m_indices"])
    output_index = torch.empty_like(inp["output_index"])
    ep_scatter(
        q,
        sf,
        inp["recv_topk"],
        inp["num_recv_tokens_per_expert"],
        inp["num_valid_tokens_per_expert"],
        torch.zeros(E, dtype=torch.int32, device="cuda"),
        out,
        scale_storage,
        m_indices,
        output_index,
        scale_ue8m0=False,
        # the fp4 activation is half as wide as fp8, hence half the block size:
        # (K // 2) // 64 == K // 128 keeps ep_scatter's scale copy unchanged.
        quant_block_size=64,
    )
    return out, scale_storage, m_indices, output_index


def _rows(buf, index, valid):
    """Rows of `buf` in (token, topk) order, for the pairs `valid` selects."""
    return torch.stack([buf[i] for i in index[valid]])


class TestEpScatterQuantMxfp4(CustomTestCase):
    def test_matches_the_two_step_path(self):
        """The fused kernel must emit byte-identical q / scale / m_indices.

        Rows are compared through `output_index` rather than by position: both
        implementations hand out rows with an atomic counter, so *which* row a
        (token, expert) pair lands on is free to differ; what must not differ is
        the content each pair ends up pointing at.
        """
        inp = _make_inputs()
        fused_inp = {
            k: (v.clone() if torch.is_tensor(v) else v) for k, v in inp.items()
        }
        fused_scale = torch.zeros(
            inp["output_tensor_scale"].shape[1],
            inp["all_tokens"],
            dtype=torch.int32,
            device="cuda",
        )
        fused_scale_view = fused_scale.transpose(0, 1)

        ep_scatter_quant_mxfp4(
            fused_inp["recv_x"],
            fused_inp["recv_topk"],
            fused_inp["num_recv_tokens_per_expert"],
            fused_inp["num_valid_tokens_per_expert"],
            fused_inp["expert_start_loc"],
            fused_inp["output_tensor"],
            fused_scale_view,
            fused_inp["m_indices"],
            fused_inp["output_index"],
        )
        ref_out, ref_scale, ref_m, ref_index = _run_two_step(inp)

        valid = inp["recv_topk"] >= 0
        # m_indices must agree exactly: -1 for the 128-aligned padding rows.
        self.assertTrue(torch.equal(fused_inp["m_indices"], ref_m))
        # Every valid (token, topk) pair must point at a row carrying that
        # token's quantized activation, and each row must be claimed once.
        self.assertTrue(
            torch.equal(
                _rows(fused_inp["output_tensor"], fused_inp["output_index"], valid),
                _rows(ref_out, ref_index, valid),
            )
        )
        self.assertTrue(
            torch.equal(
                _rows(fused_scale_view, fused_inp["output_index"], valid),
                _rows(ref_scale, ref_index, valid),
            )
        )
        self.assertEqual(
            int(fused_inp["output_index"][valid].numel()),
            len(set(fused_inp["output_index"][valid].tolist())),
        )
        # Padding rows stay zero: deep_gemm reads all ceil(M/4)*4 scale columns
        # and a garbage byte decodes to a NaN ue8m0 scale.
        padding = fused_inp["m_indices"] < 0
        self.assertEqual(int(fused_scale_view[padding].abs().sum().item()), 0)

    def test_honours_strided_topk_and_output_index(self):
        """recv_topk / output_index must be addressed through both strides.

        A column-sliced view has stride1 == 2. Reading it as if it were dense
        picks up the padding column as the second expert id and stores each row
        index on that padding column instead of the view's own element, so the
        scatter lands on the wrong slots and the columns the view skips get
        clobbered.
        """
        inp = _make_inputs()
        T = inp["recv_topk"].shape[0]
        topk_backing = torch.zeros(T, 2 * TOPK, dtype=torch.int32, device="cuda")
        recv_topk = topk_backing[:, ::2]
        self.assertEqual(recv_topk.stride(1), 2)
        recv_topk.copy_(inp["recv_topk"])
        index_backing = torch.full((T, 2 * TOPK), -7, dtype=torch.int32, device="cuda")
        output_index = index_backing[:, ::2]

        fused_scale = torch.zeros(
            inp["output_tensor_scale"].shape[1],
            inp["all_tokens"],
            dtype=torch.int32,
            device="cuda",
        )
        fused_scale_view = fused_scale.transpose(0, 1)
        ep_scatter_quant_mxfp4(
            inp["recv_x"],
            recv_topk,
            inp["num_recv_tokens_per_expert"],
            inp["num_valid_tokens_per_expert"],
            inp["expert_start_loc"],
            inp["output_tensor"],
            fused_scale_view,
            inp["m_indices"],
            output_index,
        )
        ref_out, ref_scale, _, ref_index = _run_two_step(inp)

        valid = inp["recv_topk"] >= 0
        self.assertTrue(
            torch.equal(
                _rows(inp["output_tensor"], output_index, valid),
                _rows(ref_out, ref_index, valid),
            )
        )
        self.assertTrue(
            torch.equal(
                _rows(fused_scale_view, output_index, valid),
                _rows(ref_scale, ref_index, valid),
            )
        )
        # The columns the views skip are not the kernel's to write.
        self.assertEqual(int(topk_backing[:, 1::2].abs().sum().item()), 0)
        self.assertTrue(bool((index_backing[:, 1::2] == -7).all().item()))

    def test_empty_batch_returns_without_launching(self):
        inp = _make_inputs(T=0)
        ep_scatter_quant_mxfp4(
            inp["recv_x"],
            inp["recv_topk"],
            inp["num_recv_tokens_per_expert"],
            inp["num_valid_tokens_per_expert"],
            inp["expert_start_loc"],
            inp["output_tensor"],
            inp["output_tensor_scale"],
            inp["m_indices"],
            inp["output_index"],
        )

    def _expect_value_error(self, **overrides):
        inp = _make_inputs()
        inp.update(overrides)
        with self.assertRaises(ValueError):
            ep_scatter_quant_mxfp4(
                inp["recv_x"],
                inp["recv_topk"],
                inp["num_recv_tokens_per_expert"],
                inp["num_valid_tokens_per_expert"],
                inp["expert_start_loc"],
                inp["output_tensor"],
                inp["output_tensor_scale"],
                inp["m_indices"],
                inp["output_index"],
            )

    def test_rejects_bad_operand_shapes_and_dtypes(self):
        for name, override in (
            (
                "recv_x must be 2D",
                {"recv_x": torch.randn(3, 256, 1, dtype=torch.bfloat16, device="cuda")},
            ),
            (
                "recv_x must be bf16",
                {"recv_x": torch.randn(3, 256, dtype=torch.float16, device="cuda")},
            ),
            (
                "output_tensor must be int8",
                {
                    "output_tensor": torch.empty(
                        256, 128, dtype=torch.float8_e4m3fn, device="cuda"
                    )
                },
            ),
            (
                "recv_x hidden must be a multiple of 128",
                {"recv_x": torch.randn(3, 192, dtype=torch.bfloat16, device="cuda")},
            ),
            (
                "token counts must match",
                {"recv_topk": torch.zeros(5, TOPK, dtype=torch.int32, device="cuda")},
            ),
            (
                "topk must be positive",
                {"recv_topk": torch.zeros(3, 0, dtype=torch.int32, device="cuda")},
            ),
            (
                "packed hidden size must match",
                {
                    "output_tensor": torch.empty(
                        256, 64, dtype=torch.int8, device="cuda"
                    )
                },
            ),
            (
                "scale shape must match",
                {
                    "output_tensor_scale": torch.zeros(
                        256, 1, dtype=torch.int32, device="cuda"
                    )
                },
            ),
            (
                "output_index must match topk",
                {"output_index": torch.empty(3, 1, dtype=torch.int32, device="cuda")},
            ),
            (
                "m_indices must be 128-aligned",
                {"m_indices": torch.empty(64, dtype=torch.int32, device="cuda")},
            ),
        ):
            with self.subTest(name):
                self._expect_value_error(**override)

    def test_requires_sm100(self):
        # The fused kernel uses the same hardware e2m1 converter as v2, so it has
        # to refuse pre-Blackwell instead of launching a kernel that cannot run.
        inp = _make_inputs()
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            with self.assertRaises(RuntimeError):
                ep_scatter_quant_mxfp4(
                    inp["recv_x"],
                    inp["recv_topk"],
                    inp["num_recv_tokens_per_expert"],
                    inp["num_valid_tokens_per_expert"],
                    inp["expert_start_loc"],
                    inp["output_tensor"],
                    inp["output_tensor_scale"],
                    inp["m_indices"],
                    inp["output_index"],
                )

    def test_rejects_non_cuda_operands(self):
        inp = _make_inputs()
        with self.assertRaises(ValueError):
            ep_scatter_quant_mxfp4(
                inp["recv_x"].cpu(),
                inp["recv_topk"].cpu(),
                inp["num_recv_tokens_per_expert"].cpu(),
                inp["num_valid_tokens_per_expert"].cpu(),
                inp["expert_start_loc"],
                inp["output_tensor"],
                inp["output_tensor_scale"],
                inp["m_indices"],
                inp["output_index"],
            )

    def test_rejects_non_dense_packed_rows(self):
        inp = _make_inputs()
        with self.assertRaises(ValueError):
            ep_scatter_quant_mxfp4(
                inp["recv_x"],
                inp["recv_topk"],
                inp["num_recv_tokens_per_expert"],
                inp["num_valid_tokens_per_expert"],
                inp["expert_start_loc"],
                inp["output_tensor"].t(),
                inp["output_tensor_scale"],
                inp["m_indices"],
                inp["output_index"],
            )


class TestQuantScatterMxfp4Masked(CustomTestCase):
    """The masked twin: same quantization, per-expert padded destinations.

    Unlike the compact path the row assignment is fully deterministic (it comes
    from `fused_moe_dispatch_index`, not an atomic counter), so the two
    implementations can be compared through `src2dst` directly.
    """

    T, K, E, TOPK = 7, 256, 2, 2
    # 13/2 routing with two dropped slots, so the padding paths are live.
    IDS = [[0, 1], [1, -1], [0, 0], [-1, 1], [1, 0], [0, -1], [1, 1]]

    def _inputs(self, tokens=None):
        tokens = self.T if tokens is None else tokens
        device = "cuda"
        torch.manual_seed(20260921)
        x = torch.randn(tokens, self.K, dtype=torch.bfloat16, device=device)
        topk_ids = torch.tensor(self.IDS, dtype=torch.int32, device=device)[:tokens]
        m_max = (tokens // 256 + 1) * 256
        _, src2dst = fused_moe_dispatch_index(topk_ids, self.E, m_max)
        gateup = torch.zeros(
            self.E, m_max, self.K // 2, dtype=torch.int8, device=device
        )
        scale = torch.zeros(
            self.E, self.K // 128, m_max, dtype=torch.int32, device=device
        )
        return x, topk_ids, src2dst, gateup, scale, m_max

    def _run_two_step(self, x, topk_ids, src2dst, m_max):
        packed, packed_scale = quant_mxfp4_group32_v2(x)
        gateup = torch.zeros(
            self.E, m_max, self.K // 2, dtype=torch.int8, device="cuda"
        )
        scale = torch.zeros(
            self.E, self.K // 128, m_max, dtype=torch.int32, device="cuda"
        )
        fill_gateup_input_triton_kernel[(x.shape[0],)](
            packed,
            packed_scale,
            gateup,
            scale,
            src2dst,
            topk_ids,
            self.TOPK,
            self.K // 2,
            self.K // 128,
            m_max,
            packed_scale.stride(0),
            packed_scale.stride(1),
            BLOCK_SIZE=1024,
            IS_FP8=True,
            SCALE_MN_MAJOR=True,
        )
        return gateup, scale, packed, packed_scale

    def test_matches_the_two_step_path(self):
        x, topk_ids, src2dst, gateup, scale, m_max = self._inputs()
        quant_scatter_mxfp4_masked(x, topk_ids, src2dst, gateup, scale, m_max)
        ref_gateup, ref_scale, _, _ = self._run_two_step(x, topk_ids, src2dst, m_max)

        self.assertTrue(torch.equal(gateup, ref_gateup))
        self.assertTrue(torch.equal(scale, ref_scale))

        # And the rows really do carry each token's own quantization, so the
        # equality above is not two kernels agreeing on a wrong answer.
        packed, packed_scale = quant_mxfp4_group32_v2(x)
        valid = topk_ids >= 0
        # src2dst is flat (T * topk), like fused_moe_dispatch_index emits.
        for token, slot in valid.nonzero().tolist():
            expert, row = divmod(int(src2dst[token * self.TOPK + slot]), m_max)
            self.assertTrue(
                torch.equal(
                    gateup[expert, row].view(torch.uint8),
                    packed[token].view(torch.uint8),
                )
            )
            # `scale` here is the (E, K // 128, m_max) MN-major *storage*, so a
            # token's word row lives on the last axis, not the middle one.
            self.assertTrue(torch.equal(scale[expert, :, row], packed_scale[token]))

    def test_only_the_dropped_slots_stay_untouched(self):
        # Slot -1 must not be written, which in the masked layout means the
        # destination buffers keep exactly the rows its tokens would have taken.
        x, topk_ids, src2dst, gateup, scale, m_max = self._inputs()
        quant_scatter_mxfp4_masked(x, topk_ids, src2dst, gateup, scale, m_max)
        written = set()
        for token, slot in (topk_ids >= 0).nonzero().tolist():
            written.add(int(src2dst[token * self.TOPK + slot]))
        unwritten = [row for row in range(self.E * m_max) if row not in written]
        self.assertTrue(unwritten)
        untouched = torch.tensor(unwritten, device="cuda")
        self.assertEqual(int(gateup.view(-1, self.K // 2)[untouched].abs().sum()), 0)

    def test_empty_batch_returns_without_launching(self):
        x, topk_ids, src2dst, gateup, scale, m_max = self._inputs(tokens=0)
        quant_scatter_mxfp4_masked(x, topk_ids, src2dst, gateup, scale, m_max)

    def test_requires_sm100(self):
        x, topk_ids, src2dst, gateup, scale, m_max = self._inputs()
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            with self.assertRaises(RuntimeError):
                quant_scatter_mxfp4_masked(x, topk_ids, src2dst, gateup, scale, m_max)

    def test_rejects_bad_operands(self):
        x, topk_ids, src2dst, gateup, scale, m_max = self._inputs()
        bad_inputs = {
            "x not 2D": (x.unsqueeze(0), topk_ids, src2dst, gateup, scale, m_max),
            "x not bf16": (x.float(), topk_ids, src2dst, gateup, scale, m_max),
            "topk not int32": (
                x,
                topk_ids.to(torch.int64),
                src2dst,
                gateup,
                scale,
                m_max,
            ),
            "src2dst not 1D": (
                x,
                topk_ids,
                src2dst.view(-1, self.TOPK),
                gateup,
                scale,
                m_max,
            ),
            "src2dst truncated": (
                x,
                topk_ids,
                src2dst[:-1],
                gateup,
                scale,
                m_max,
            ),
            # Same element count, wrong element stride: the kernel reads
            # src2dst as a flat buffer, so this must be rejected rather than
            # silently scattering every token to the wrong row.
            "src2dst not dense": (
                x,
                topk_ids,
                torch.zeros(2 * src2dst.numel(), dtype=torch.int32, device="cuda")[::2],
                gateup,
                scale,
                m_max,
            ),
            "gateup not int8": (
                x,
                topk_ids,
                src2dst,
                gateup.float(),
                scale,
                m_max,
            ),
            "scale not int32": (
                x,
                topk_ids,
                src2dst,
                gateup,
                scale.float(),
                m_max,
            ),
            "hidden not a multiple of 128": (
                x[:, :192],
                topk_ids,
                src2dst,
                gateup[:, :, :96],
                scale[:, :1],
                m_max,
            ),
            "topk_ids not dense": (
                x,
                torch.zeros(
                    x.shape[0], 2 * self.TOPK, dtype=torch.int32, device="cuda"
                )[:, ::2],
                src2dst,
                gateup,
                scale,
                m_max,
            ),
            "token count mismatch": (
                x,
                topk_ids[:3],
                src2dst[: 3 * self.TOPK],
                gateup,
                scale,
                m_max,
            ),
            "gateup rows strided": (
                x,
                topk_ids,
                src2dst,
                torch.zeros(
                    self.E, m_max, 2 * (self.K // 2), dtype=torch.int8, device="cuda"
                )[:, :, ::2],
                scale,
                m_max,
            ),
            "gateup slabs not dense": (
                x,
                topk_ids,
                src2dst,
                torch.zeros(
                    self.E, m_max, self.K // 2 + 8, dtype=torch.int8, device="cuda"
                )[:, :, : self.K // 2],
                scale,
                m_max,
            ),
            "packed hidden mismatch": (
                x,
                topk_ids,
                src2dst,
                gateup[:, :, :64],
                scale,
                m_max,
            ),
            "scale hidden mismatch": (
                x,
                topk_ids,
                src2dst,
                gateup,
                scale[:, :1],
                m_max,
            ),
        }
        for label, args in bad_inputs.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    quant_scatter_mxfp4_masked(*args)

        with self.subTest("non-CUDA operands"):
            with self.assertRaises(ValueError):
                quant_scatter_mxfp4_masked(
                    x.cpu(), topk_ids.cpu(), src2dst.cpu(), gateup, scale, m_max
                )


if __name__ == "__main__":
    unittest.main()
