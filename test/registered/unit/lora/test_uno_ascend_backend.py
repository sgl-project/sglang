"""CPU numerical coverage for the rank-128 Ascend UNO GEMM route."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.backend.uno_ascend_backend import UnoAscendLoRABackend
from sglang.srt.speculative.uno_lora import (
    UnoCudaGraphLoRAState,
    uno_lora_backend_for_device,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestUnoAscendLoRA(unittest.TestCase):
    def setUp(self):
        self.backend = UnoAscendLoRABackend.__new__(UnoAscendLoRABackend)
        self.backend.device = torch.device("cpu")
        self.backend.max_loras_per_batch = 2

    def prepare(self, bs, rank, *, reverse=False):
        base, adapter = (1, 0) if reverse else (0, 1)
        ranks, scales = [0, 0], [0.0, 0.0]
        ranks[adapter], scales[adapter] = rank, 16.0
        self.backend.prepare_lora_token_segments(
            segment_lens=[1, 7] * bs,
            weight_indices=[base, adapter] * bs,
            lora_ranks=ranks,
            scalings=scales,
        )
        return adapter

    def check_projection(self, bs, rank, widths, dtype, *, reverse=False):
        slot = self.prepare(bs, rank, reverse=reverse)
        generator = torch.Generator().manual_seed(37906)
        slices, rows, hidden = len(widths), bs * 8, 160
        x = torch.randn(rows, hidden, generator=generator).to(dtype)
        a = (torch.randn(2, rank * slices, hidden, generator=generator) * 0.02).to(
            dtype
        )
        b = (torch.randn(2, sum(widths), rank, generator=generator) * 0.02).to(dtype)
        base = (torch.randn(rows, sum(widths), generator=generator) * 0.1).to(dtype)
        offsets = [0]
        for width in widths:
            offsets.append(offsets[-1] + width)
        expected = base.float().clone()
        for request in range(bs):
            start, end = request * 8 + 1, (request + 1) * 8
            for i, (lo, hi) in enumerate(zip(offsets, offsets[1:])):
                # Independent dense reference; every Q/K/V or gate/up slice
                # consumes its own A weights and the entire adapter rank.
                intermediate = (
                    x[start:end].float() @ a[slot, i * rank : (i + 1) * rank].float().T
                )
                expected[start:end, lo:hi] += 16 * (
                    intermediate @ b[slot, lo:hi].float().T
                )
        x, a, b = (tensor.to(self.backend.device) for tensor in (x, a, b))
        base_device = base.to(self.backend.device)
        output = base_device.clone()
        offset_tensor = torch.tensor(offsets, dtype=torch.int32)

        def project():
            if slices == 1:
                middle = self.backend.run_lora_a_sgemm(x, a)
                return self.backend.run_lora_b_sgemm(middle, b, base_output=output)
            if slices == 3:
                return self.backend.run_qkv_lora(
                    x,
                    a,
                    b,
                    offset_tensor,
                    base_output=output,
                    output_offset_cpu=offset_tensor,
                )
            return self.backend.run_gate_up_lora(
                x,
                a,
                b,
                offset_tensor,
                base_output=output,
                output_offset_cpu=offset_tensor,
            )

        if getattr(self, "capture_graph", False):
            for _ in range(2):
                output.copy_(base_device)
                project()
            torch.npu.synchronize()
            output.copy_(base_device)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, auto_dispatch_capture=True):
                actual = project()
            # Replay must read current input storage rather than capture-time
            # values. Negating x negates the unscaled A/B delta in the reference.
            x.neg_()
            expected = 2 * base.float() - expected
            output.copy_(base_device)
            graph.replay()
            torch.npu.synchronize()
        else:
            actual = project()
        self.assertIs(actual, output)
        actual = actual.cpu()
        torch.testing.assert_close(actual[::8], base[::8], atol=0, rtol=0)
        tolerance = (
            dict(atol=0.01, rtol=0.02)
            if dtype == torch.bfloat16
            else dict(atol=1e-5, rtol=1e-4)
        )
        torch.testing.assert_close(
            actual.float(), expected.to(dtype).float(), **tolerance
        )

    @unittest.skipUnless(
        hasattr(torch, "npu") and torch.npu.is_available(), "requires an Ascend NPU"
    )
    def test_rank128_npu_graph(self):
        self.backend.device = torch.device("npu:0")
        self.capture_graph = True
        for bs in (1, 64):
            for widths in ([256], [256, 128, 128], [256, 256]):
                with self.subTest(bs=bs, widths=widths):
                    self.check_projection(bs, 128, widths, torch.bfloat16)

    def test_rank_128_all_projections_and_seed_isolation(self):
        for dtype in (torch.float32, torch.bfloat16):
            for bs in (1, 4, 64):
                for widths in ([48], [48, 24, 24], [48, 48]):
                    with self.subTest(dtype=dtype, bs=bs, widths=widths):
                        self.check_projection(bs, 128, widths, dtype)

    def test_other_ranks_and_reversed_adapter_slot(self):
        for rank in (8, 16, 32, 64, 256):
            with self.subTest(rank=rank):
                self.check_projection(
                    2, rank, [48, 24, 24], torch.float32, reverse=True
                )

    def test_route_survives_graph_bucket_switch_and_reset(self):
        def prepare_segments(*, lora_ids, segment_lens):
            self.prepare(len(segment_lens) // 2, 128)

        manager = SimpleNamespace(
            lora_backend=self.backend,
            prepare_lora_token_segments=prepare_segments,
            reset_lora_batch=self.backend.reset_batch_state,
        )
        state = UnoCudaGraphLoRAState(manager, "uno", 8)
        state.capture_draft(64)
        retained = self.backend.batch_info
        state.capture_draft(1)
        state.activate_draft(64)
        self.assertIs(self.backend.batch_info, retained)
        self.assertEqual(self.backend._route(512).rank, 128)
        state.reset()
        self.assertIsNone(self.backend.batch_info)
        with self.assertRaisesRegex(RuntimeError, "token route"):
            self.backend._route(512)

    def test_invalid_route_and_token_count_fail_before_matmul(self):
        with self.assertRaises(ValueError):
            self.backend.prepare_lora_token_segments(
                segment_lens=[2, 6],
                weight_indices=[0, 1],
                lora_ranks=[0, 128],
                scalings=[0, 16],
            )
        self.prepare(1, 128)
        with self.assertRaisesRegex(RuntimeError, "token route"):
            self.backend.run_lora_a_sgemm(torch.zeros(7, 160), torch.zeros(2, 128, 160))

    def test_device_selects_safe_backend(self):
        self.assertEqual(uno_lora_backend_for_device("npu:0"), "uno_ascend")
        self.assertEqual(uno_lora_backend_for_device("cuda:0"), "uno_cublas")


if __name__ == "__main__":
    unittest.main()
