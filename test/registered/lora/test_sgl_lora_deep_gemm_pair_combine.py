"""Focused correctness tests for the typed DeepGEMM LoRA combine seam."""

from __future__ import annotations

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.moe_lora_signal_gates import require_delta_close
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

from sglang.kernels.ops.moe.ep_moe_kernels import post_reorder_deepgemm
from sglang.srt.lora.sgl_lora.hooks import PairDomainLoRAContribution


class TestSglLoraDeepGemmPairCombine(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")
        cls.num_tokens = 4
        cls.topk = 3
        cls.hidden_size = 37
        cls.scale = 1.75

    def _make_case(self, output_dtype: torch.dtype, seed: int = 31):
        torch.manual_seed(seed)
        down_output = torch.randn(
            3, 5, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )
        pair_delta = torch.randn(
            self.num_tokens,
            self.topk,
            self.hidden_size,
            dtype=torch.bfloat16,
            device=self.device,
        )
        topk_ids = torch.tensor(
            [[0, 1, 2], [2, -1, 1], [0, 2, -1], [1, 0, 2]],
            dtype=torch.int32,
            device=self.device,
        )
        src2dst = torch.tensor(
            [[0, 5, 10], [11, -1, 6], [1, 12, -1], [7, 2, 13]],
            dtype=torch.int32,
            device=self.device,
        )
        topk_weights = torch.tensor(
            [
                [0.20, 0.30, 0.50],
                [0.65, 0.15, 0.20],
                [0.40, 0.35, 0.25],
                [0.10, 0.70, 0.20],
            ],
            dtype=torch.float32,
            device=self.device,
        )
        # A nonlocal pair must not read either its invalid row or its delta.
        pair_delta[topk_ids < 0] = 1000
        output = torch.empty(
            self.num_tokens,
            self.hidden_size,
            dtype=output_dtype,
            device=self.device,
        )
        return down_output, pair_delta, src2dst, topk_ids, topk_weights, output

    def _reference(
        self,
        down_output: torch.Tensor,
        pair_delta: torch.Tensor | None,
        src2dst: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        output_dtype: torch.dtype,
    ):
        down_rows = down_output.view(-1, self.hidden_size)
        result = torch.zeros(
            self.num_tokens,
            self.hidden_size,
            dtype=torch.float32,
            device=self.device,
        )
        for slot in range(self.topk):
            valid = topk_ids[:, slot] >= 0
            rows = src2dst[:, slot].clamp_min(0).to(torch.int64)
            pair = down_rows[rows].float()
            if pair_delta is not None:
                pair = pair + pair_delta[:, slot].float()
            pair = torch.where(valid[:, None], pair, torch.zeros_like(pair))
            result = result + pair * topk_weights[:, slot, None].float()
        return (result * self.scale).to(output_dtype)

    def _invoke(
        self,
        down_output,
        pair_delta,
        src2dst,
        topk_ids,
        topk_weights,
        output,
    ):
        post_reorder_deepgemm(
            down_output,
            output,
            src2dst,
            topk_ids,
            topk_weights,
            self.topk,
            self.num_tokens,
            self.hidden_size,
            self.scale,
            pair_delta=pair_delta,
        )

    def test_optional_pair_delta_matches_fixed_topk_fp32_reference(self):
        for output_dtype in (torch.bfloat16, torch.float32):
            for has_delta in (False, True):
                with self.subTest(output_dtype=output_dtype, has_delta=has_delta):
                    case = self._make_case(output_dtype)
                    down, delta, src, ids, weights, output = case
                    selected_delta = delta if has_delta else None
                    expected = self._reference(
                        down, selected_delta, src, ids, weights, output_dtype
                    )
                    self._invoke(down, selected_delta, src, ids, weights, output)
                    torch.cuda.synchronize()
                    # Same BF16 inputs on both sides: the FP32 destination
                    # genuinely compares at the FP32 gate; BF16 is
                    # storage-rounding bound.
                    require_delta_close(
                        output,
                        expected,
                        destination_dtype=output_dtype,
                        label=f"pair combine has_delta={has_delta}",
                    )

    def test_pair_delta_graph_replay_uses_updated_inputs(self):
        for output_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(output_dtype=output_dtype):
                case = self._make_case(output_dtype)
                down, delta, src, ids, weights, output = case

                # Warm compilation before capture.
                self._invoke(down, delta, src, ids, weights, output)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._invoke(down, delta, src, ids, weights, output)

                torch.manual_seed(53)
                down.copy_(torch.randn_like(down))
                delta.copy_(torch.randn_like(delta))
                delta[ids < 0] = 2000
                weights.copy_(torch.rand_like(weights))
                expected = self._reference(down, delta, src, ids, weights, output_dtype)
                graph.replay()
                torch.cuda.synchronize()
                require_delta_close(
                    output,
                    expected,
                    destination_dtype=output_dtype,
                    label="pair combine graph replay",
                )

    def test_typed_contribution_validates_pair_storage(self):
        _, delta, _, _, _, _ = self._make_case(torch.bfloat16)
        contribution = PairDomainLoRAContribution(delta)
        contribution.validate_for(
            expected_shape=(
                self.num_tokens,
                self.topk,
                self.hidden_size,
            ),
            expected_device=delta.device,
        )
        with self.assertRaisesRegex(ValueError, "must have shape"):
            contribution.validate_for(
                expected_shape=(
                    self.num_tokens,
                    self.topk + 1,
                    self.hidden_size,
                ),
                expected_device=delta.device,
            )


if __name__ == "__main__":
    unittest.main()
