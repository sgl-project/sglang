"""Independent-reference tests for the policy-free BF16 MoE-LoRA core."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

from sglang.srt.lora.sgl_lora.bf16 import (
    grouped_lora_a,
    stock_grouped_lora_b,
)
from sglang.srt.lora.sgl_lora.routing import build_virtual_expert_routing


class TestSglLoraBf16Core(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")
        cls.config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 16,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        }

    def setUp(self):
        torch.manual_seed(17)

    def _routing(
        self,
        topk_ids,
        adapters,
        *,
        factor_count,
        max_loras,
        factor_map,
    ):
        return build_virtual_expert_routing(
            torch.tensor(topk_ids, dtype=torch.int32, device=self.device),
            torch.tensor(adapters, dtype=torch.int32, device=self.device),
            factor_expert_count=factor_count,
            max_loras=max_loras,
            block_size=self.config["BLOCK_SIZE_M"],
            routed_expert_to_factor_id=torch.tensor(
                factor_map, dtype=torch.int32, device=self.device
            ),
        )

    @staticmethod
    def _reference_lora_a(
        input,
        weight,
        route,
        *,
        pair_input=False,
        input_row_map=None,
        initial=None,
    ):
        pairs = route.virtual_topk_ids.numel()
        top_k = route.virtual_topk_ids.shape[1]
        result = (
            torch.empty(
                (pairs, weight.shape[1]), dtype=torch.bfloat16, device=input.device
            )
            if initial is None
            else initial.clone()
        )
        virtual_ids = route.virtual_topk_ids.reshape(-1).cpu().tolist()
        host_row_map = None if input_row_map is None else input_row_map.cpu().tolist()
        for pair_id, factor_id in enumerate(virtual_ids):
            if factor_id < 0:
                continue
            if host_row_map is not None:
                input_row = host_row_map[pair_id]
            elif pair_input:
                input_row = pair_id
            else:
                input_row = pair_id // top_k
            if input_row < 0:
                result[pair_id].zero_()
                continue
            result[pair_id] = torch.mv(
                weight[factor_id].float(), input[input_row].float()
            ).to(torch.bfloat16)
        return result

    @staticmethod
    def _reference_lora_b(
        intermediate,
        weight,
        route,
        *,
        destination,
        destination_offsets,
    ):
        result = destination.clone()
        virtual_ids = route.virtual_topk_ids.reshape(-1).cpu().tolist()
        num_slices = len(destination_offsets)
        rank = weight.shape[2]
        width = weight.shape[1] // num_slices
        for pair_id, factor_id in enumerate(virtual_ids):
            for slice_id, destination_offset in enumerate(destination_offsets):
                output_slice = result[
                    pair_id, destination_offset : destination_offset + width
                ]
                if factor_id < 0:
                    output_slice.zero_()
                    continue
                a = intermediate[
                    pair_id, slice_id * rank : (slice_id + 1) * rank
                ].float()
                b = weight[factor_id, slice_id * width : (slice_id + 1) * width].float()
                output_slice.copy_(torch.mv(b, a).to(torch.bfloat16))
        return result

    def test_half_distinct_gate_up_a_and_b_with_provider_order(self):
        topk_ids = [[4, 5], [6, 9], [4, 7], [5, 6]]
        adapters = [0, 1, -1, 0]
        factor_map = [-1, -1, -1, -1, 0, 1, 2, 3, -1, -1]
        route = self._routing(
            topk_ids,
            adapters,
            factor_count=4,
            max_loras=2,
            factor_map=factor_map,
        )
        hidden = torch.randn(4, 32, dtype=torch.bfloat16, device=self.device)
        rank = 16
        a_weight = torch.randn(
            8, 2 * rank, 32, dtype=torch.bfloat16, device=self.device
        )
        a_weight[:, rank:].add_(0.75)
        a_output = torch.full(
            (8, 2 * rank), 13, dtype=torch.bfloat16, device=self.device
        )
        a_initial = a_output.clone()
        grouped_lora_a(
            hidden,
            a_weight,
            a_output,
            route,
            config=self.config,
        )
        a_reference = self._reference_lora_a(hidden, a_weight, route, initial=a_initial)
        valid_pairs = route.virtual_topk_ids.reshape(-1) >= 0
        torch.testing.assert_close(
            a_output[valid_pairs],
            a_reference[valid_pairs],
            rtol=2e-2,
            atol=2e-2,
        )
        self.assertFalse(
            torch.equal(
                a_output[valid_pairs, :rank],
                a_output[valid_pairs, rank:],
            )
        )

        width = 24
        b_weight = torch.randn(
            8, 2 * width, rank, dtype=torch.bfloat16, device=self.device
        )
        b_weight[:, width:].sub_(0.625)
        provider_up_gate_offsets = (width, 0)
        destination = torch.full(
            (8, 2 * width), 91, dtype=torch.bfloat16, device=self.device
        )
        b_reference = self._reference_lora_b(
            a_output,
            b_weight,
            route,
            destination=destination,
            destination_offsets=provider_up_gate_offsets,
        )
        stock_grouped_lora_b(
            a_output,
            b_weight,
            destination,
            route,
            destination_offsets=provider_up_gate_offsets,
            config=self.config,
        )
        torch.testing.assert_close(destination, b_reference, rtol=2e-2, atol=2e-2)
        invalid_pairs = ~valid_pairs
        self.assertTrue(
            torch.equal(
                destination[invalid_pairs],
                torch.zeros_like(destination[invalid_pairs]),
            )
        )

    def test_down_a_provider_row_map_and_invalid_row_zero(self):
        route = self._routing(
            [[4, 5], [6, 7], [5, 4]],
            [0, 1, 0],
            factor_count=4,
            max_loras=2,
            factor_map=[-1, -1, -1, -1, 0, 1, 2, 3],
        )
        provider_input = torch.randn(8, 32, dtype=torch.bfloat16, device=self.device)
        row_map = torch.tensor(
            [5, 1, -1, 3, 0, 4], dtype=torch.int32, device=self.device
        )
        weight = torch.randn(8, 16, 32, dtype=torch.bfloat16, device=self.device)
        output = torch.full((6, 16), 37, dtype=torch.bfloat16, device=self.device)
        reference = self._reference_lora_a(
            provider_input,
            weight,
            route,
            input_row_map=row_map,
            initial=output,
        )
        grouped_lora_a(
            provider_input,
            weight,
            output,
            route,
            config=self.config,
            input_row_map=row_map,
        )
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
        self.assertTrue(torch.equal(output[2], torch.zeros_like(output[2])))

    def test_stock_single_slice_overwrites_base_rows_only_in_target(self):
        route = self._routing(
            [[4, 5], [5, 4]],
            [0, -1],
            factor_count=2,
            max_loras=2,
            factor_map=[-1, -1, -1, -1, 0, 1],
        )
        rank, width = 16, 20
        intermediate = torch.randn(4, rank, dtype=torch.bfloat16, device=self.device)
        weight = torch.randn(4, width, rank, dtype=torch.bfloat16, device=self.device)
        destination = torch.full(
            (4, width + 7), 71, dtype=torch.bfloat16, device=self.device
        )
        reference = self._reference_lora_b(
            intermediate,
            weight,
            route,
            destination=destination,
            destination_offsets=(3,),
        )
        stock_grouped_lora_b(
            intermediate,
            weight,
            destination,
            route,
            destination_offsets=(3,),
            config=self.config,
        )
        torch.testing.assert_close(destination, reference, rtol=2e-2, atol=2e-2)
        self.assertTrue(
            torch.equal(
                destination[:, :3],
                torch.full_like(destination[:, :3], 71),
            )
        )
        self.assertTrue(
            torch.equal(
                destination[:, 3 + width :],
                torch.full_like(destination[:, 3 + width :], 71),
            )
        )
        invalid_pairs = route.virtual_topk_ids.reshape(-1) < 0
        self.assertTrue(
            torch.equal(
                destination[invalid_pairs, 3 : 3 + width],
                torch.zeros_like(destination[invalid_pairs, 3 : 3 + width]),
            )
        )

    def test_eager_and_cuda_graph_are_exact_across_128_replays(self):
        topk_ids = torch.tensor(
            [[4, 5], [6, 8], [5, 4], [7, 6]],
            dtype=torch.int32,
            device=self.device,
        )
        adapters = torch.tensor([0, -1, 1, 0], dtype=torch.int32, device=self.device)
        factor_map = torch.tensor(
            [-1, -1, -1, -1, 0, 1, 2, 3, -1],
            dtype=torch.int32,
            device=self.device,
        )
        route = build_virtual_expert_routing(
            topk_ids,
            adapters,
            factor_expert_count=4,
            max_loras=2,
            block_size=16,
            routed_expert_to_factor_id=factor_map,
        )
        hidden = torch.randn(4, 32, dtype=torch.bfloat16, device=self.device)
        a_weight = torch.randn(8, 32, 32, dtype=torch.bfloat16, device=self.device)
        a_output = torch.empty(8, 32, dtype=torch.bfloat16, device=self.device)
        b_weight = torch.randn(8, 48, 16, dtype=torch.bfloat16, device=self.device)
        b_output = torch.empty(8, 48, dtype=torch.bfloat16, device=self.device)

        def run_core():
            a_output.fill_(19)
            grouped_lora_a(
                hidden,
                a_weight,
                a_output,
                route,
                config=self.config,
            )
            b_output.fill_(23)
            stock_grouped_lora_b(
                a_output,
                b_weight,
                b_output,
                route,
                destination_offsets=(24, 0),
                config=self.config,
            )

        run_core()
        torch.cuda.synchronize()
        eager = (a_output.clone(), b_output.clone())
        for _ in range(128):
            run_core()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(a_output, eager[0]))
            self.assertTrue(torch.equal(b_output, eager[1]))

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                run_core()
        torch.cuda.current_stream().wait_stream(warmup_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_core()
        for _ in range(128):
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(a_output, eager[0]))
            self.assertTrue(torch.equal(b_output, eager[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
