"""Independent-reference tests for the Step-3 indexed LoRA-A candidate.

The indexed arm derives its (adapter, factor) group INSIDE the kernel from
route sources via ``virtual_expert_ids_inline`` — the R2 single definition.
What these cases guard:

* semantic agreement with an independent host reference at both sites
  (token-major gate/up, pair-major down), including the factor-map path a
  global-ID domain exercises — a divergence between the kernel's inline key
  use and the plan-based path produces wrong LoRA output, not an error;
* the PRESERVE contract: rows at invalid pairs (sentinel expert, base-row
  adapter, non-owned expert through the map) are never written — the lab's
  poison probes rely on it, and a masked-store regression would silently
  turn preserved rows into zeros or garbage;
* the determinism claim of the arm (serial K loop, FP32 accumulator, no
  atomics): bitwise replay stability and eager/CUDA-graph equality — the
  property a deterministic-split-K competitor must also meet, pinned here
  so the comparison stays apples-to-apples.

Triton only: no SM100 requirement, so this also runs on the H200 pod.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")

from benchmark.kernels.lora_moe.lora_a_candidates import (
    INDEXED_DEFAULT_CONFIG,
    SPLIT_K_DEFAULT_CONFIG,
    invoke_indexed_lora_a,
    invoke_split_k_lora_a,
    run_lora_a,
    split_k_heuristic,
)
from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from benchmark.kernels.lora_moe.signal_gates import require_delta_close
from sglang.srt.lora.sgl_lora.routing import build_virtual_expert_routing

CONFIG = {"BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32, "num_warps": 4, "num_stages": 2}


class TestIndexedLoraACandidate(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def setUp(self):
        torch.manual_seed(23)

    def _routing(
        self,
        topk_ids,
        adapters,
        *,
        lora_experts_per_adapter,
        max_loras,
        lora_expert_map,
    ):
        return build_virtual_expert_routing(
            torch.tensor(topk_ids, dtype=torch.int32, device=self.device),
            torch.tensor(adapters, dtype=torch.int32, device=self.device),
            lora_experts_per_adapter=lora_experts_per_adapter,
            max_loras=max_loras,
            block_size=16,
            lora_expert_map=(
                None
                if lora_expert_map is None
                else torch.tensor(
                    lora_expert_map, dtype=torch.int32, device=self.device
                )
            ),
        )

    @staticmethod
    def _reference(input, weight, route, *, pair_input=False, initial=None):
        """Host FP32 mv per valid pair; invalid pairs left untouched."""
        pairs = route.virtual_topk_ids.numel()
        top_k = route.virtual_topk_ids.shape[1]
        result = (
            torch.zeros(
                (pairs, weight.shape[1]), dtype=torch.bfloat16, device=input.device
            )
            if initial is None
            else initial.clone()
        )
        for pair_id, group in enumerate(route.virtual_topk_ids.reshape(-1).tolist()):
            if group < 0:
                continue
            row = pair_id if pair_input else pair_id // top_k
            result[pair_id] = torch.mv(weight[group].float(), input[row].float()).to(
                torch.bfloat16
            )
        return result

    def _tensors(self, *, tokens, hidden, groups, width):
        input = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=self.device)
        weight = (
            torch.randn(groups, width, hidden, dtype=torch.float32, device=self.device)
            * hidden**-0.5
        ).to(torch.bfloat16)
        return input, weight

    def test_gate_site_matches_reference_through_the_lora_expert_map(self):
        # Global-domain ids: 8 routable experts, this rank owns [4, 8) as
        # factors 0..3; id 9 is out of map range, -1 is a literal sentinel.
        lora_expert_map = [-1, -1, -1, -1, 0, 1, 2, 3]
        topk_ids = [[4, 1, 6], [5, 5, -1], [7, 4, 2], [6, 9, 5]]
        adapters = [0, 1, -1, 1]  # token 2 is a base row
        route = self._routing(
            topk_ids,
            adapters,
            lora_experts_per_adapter=4,
            max_loras=2,
            lora_expert_map=lora_expert_map,
        )
        input, weight = self._tensors(tokens=4, hidden=64, groups=8, width=24)
        reference = self._reference(input, weight, route)
        valid = (route.virtual_topk_ids.reshape(-1) >= 0).cpu()
        self.assertTrue(bool(valid.any()) and not bool(valid.all()))
        # Both the small test config and the TIMING default (BK64) — the
        # timed compile variant must be the tested one (second S3 review).
        for config in (CONFIG, INDEXED_DEFAULT_CONFIG):
            output = torch.full(
                (12, 24), float("nan"), dtype=torch.bfloat16, device=self.device
            )
            invoke_indexed_lora_a(input, weight, output, route, config=config)
            require_delta_close(
                output.cpu().float()[valid],
                reference.cpu().float()[valid],
                gate_dtype=torch.bfloat16,
                label=f"indexed gate/up-A vs host reference ({config})",
            )
            # PRESERVE contract: every invalid pair keeps its poison.
            self.assertTrue(bool(output[~valid.to(self.device)].isnan().all()))

    def test_down_site_pair_major_local_domain(self):
        topk_ids = [[0, 2], [1, -1], [3, 0]]
        adapters = [1, 0, 2]
        route = self._routing(
            topk_ids,
            adapters,
            lora_experts_per_adapter=4,
            max_loras=3,
            lora_expert_map=None,
        )
        pairs = 6
        input, weight = self._tensors(tokens=pairs, hidden=48, groups=12, width=16)
        output = torch.full(
            (pairs, 16), float("nan"), dtype=torch.bfloat16, device=self.device
        )
        invoke_indexed_lora_a(
            input, weight, output, route, config=CONFIG, pair_input=True
        )
        reference = self._reference(input, weight, route, pair_input=True)
        valid = (route.virtual_topk_ids.reshape(-1) >= 0).cpu()
        require_delta_close(
            output.cpu().float()[valid],
            reference.cpu().float()[valid],
            gate_dtype=torch.bfloat16,
            label="indexed down-A vs host reference",
        )
        self.assertTrue(bool(output[~valid.to(self.device)].isnan().all()))

    def test_eager_and_cuda_graph_are_bitwise_across_replays(self):
        topk_ids = torch.randint(0, 8, (32, 4), dtype=torch.int32)
        adapters = torch.tensor([i % 5 - 1 for i in range(32)], dtype=torch.int32)
        route = self._routing(
            topk_ids.tolist(),
            adapters.tolist(),
            lora_experts_per_adapter=8,
            max_loras=4,
            lora_expert_map=None,
        )
        input, weight = self._tensors(tokens=32, hidden=96, groups=32, width=32)
        output = torch.zeros(128, 32, dtype=torch.bfloat16, device=self.device)

        invoke_indexed_lora_a(input, weight, output, route, config=CONFIG)
        first = output.clone()
        for _ in range(64):
            invoke_indexed_lora_a(input, weight, output, route, config=CONFIG)
            self.assertTrue(torch.equal(output, first))

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke_indexed_lora_a(input, weight, output, route, config=CONFIG)
        for _ in range(64):
            output.zero_()
            # Valid rows must be rewritten identically by replay; zeroed
            # invalid rows stay zero (preserve = not written).
            graph.replay()
            torch.cuda.synchronize()
            valid = route.virtual_topk_ids.reshape(-1) >= 0
            self.assertTrue(torch.equal(output[valid], first[valid]))
            self.assertTrue(bool((output[~valid] == 0).all()))


class TestSplitKLoraACandidate(CustomTestCase):
    """Deterministic split-K: FP32 workspace + fixed-order reduce.

    Guards: (1) numerical agreement with the host reference at every split
    factor — a wrong strided-K partition or a missed tail tile is silent
    corruption at exactly one split count; (2) the determinism claim that
    NAMES this family: bitwise replay stability eagerly and under CUDA
    graph, the property the rejected atomic split-K cannot meet; (3) the
    occupancy heuristic keeps splitting the decode regime — the audited
    regression (HEAD's launcher collapsed the rank-tiered rule to
    ``128 // base_grid``) made SPLIT_K hit 1 far too early.
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def setUp(self):
        torch.manual_seed(29)

    def _route_and_tensors(self):
        topk_ids = torch.randint(0, 8, (24, 4), dtype=torch.int32)
        topk_ids[3, 1] = -1  # literal sentinel
        adapters = torch.tensor([i % 5 - 1 for i in range(24)], dtype=torch.int32)
        route = build_virtual_expert_routing(
            topk_ids.to(self.device),
            adapters.to(self.device),
            lora_experts_per_adapter=8,
            max_loras=4,
            block_size=16,
        )
        input = torch.randn(24, 512, dtype=torch.bfloat16, device=self.device)
        weight = (
            torch.randn(32, 24, 512, dtype=torch.float32, device=self.device)
            * 512**-0.5
        ).to(torch.bfloat16)
        return route, input, weight

    @staticmethod
    def _reference(input, weight, route, top_k):
        pairs = route.virtual_topk_ids.numel()
        result = torch.zeros(
            (pairs, weight.shape[1]), dtype=torch.bfloat16, device=input.device
        )
        for pair_id, group in enumerate(route.virtual_topk_ids.reshape(-1).tolist()):
            if group < 0:
                continue
            result[pair_id] = torch.mv(
                weight[group].float(), input[pair_id // top_k].float()
            ).to(torch.bfloat16)
        return result

    def test_every_split_factor_matches_the_reference(self):
        route, input, weight = self._route_and_tensors()
        reference = self._reference(input, weight, route, top_k=4)
        valid = (route.virtual_topk_ids.reshape(-1) >= 0).cpu()
        for split_k in (1, 2, 4, 8):
            config = {
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "num_warps": 4,
                "num_stages": 2,
                "SPLIT_K": split_k,
            }
            output = torch.zeros(96, 24, dtype=torch.bfloat16, device=self.device)
            invoke_split_k_lora_a(input, weight, output, route, config=config)
            require_delta_close(
                output.cpu().float()[valid],
                reference.cpu().float()[valid],
                gate_dtype=torch.bfloat16,
                label=f"split-K={split_k} vs host reference",
            )

    def test_bitwise_replay_and_graph_at_split_4(self):
        route, input, weight = self._route_and_tensors()
        config = {
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "num_warps": 4,
            "num_stages": 2,
            "SPLIT_K": 4,
        }
        workspace = torch.empty(4, 96, 24, dtype=torch.float32, device=self.device)
        output = torch.zeros(96, 24, dtype=torch.bfloat16, device=self.device)
        invoke_split_k_lora_a(
            input, weight, output, route, config=config, workspace=workspace
        )
        first = output.clone()
        for _ in range(64):
            invoke_split_k_lora_a(
                input, weight, output, route, config=config, workspace=workspace
            )
            self.assertTrue(torch.equal(output, first))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            invoke_split_k_lora_a(
                input, weight, output, route, config=config, workspace=workspace
            )
        for _ in range(64):
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(output, first))

    def test_down_site_pair_major_matches_the_reference(self):
        # The timed pair_input=True path was previously untested (second S3
        # review): a row-domain slip there is silent at the gate site.
        route, _, weight = self._route_and_tensors()
        pairs = 96
        input = torch.randn(pairs, 512, dtype=torch.bfloat16, device=self.device)
        reference = torch.zeros(pairs, 24, dtype=torch.bfloat16, device=self.device)
        for pair_id, group in enumerate(route.virtual_topk_ids.reshape(-1).tolist()):
            if group >= 0:
                reference[pair_id] = torch.mv(
                    weight[group].float(), input[pair_id].float()
                ).to(torch.bfloat16)
        valid = (route.virtual_topk_ids.reshape(-1) >= 0).cpu()
        for config in (
            {
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
                "num_warps": 4,
                "num_stages": 2,
                "SPLIT_K": 4,
            },
            SPLIT_K_DEFAULT_CONFIG,
        ):
            output = torch.zeros(pairs, 24, dtype=torch.bfloat16, device=self.device)
            invoke_split_k_lora_a(
                input, weight, output, route, config=config, pair_input=True
            )
            require_delta_close(
                output.cpu().float()[valid],
                reference.cpu().float()[valid],
                gate_dtype=torch.bfloat16,
                label=f"split-K down-A vs host reference ({config.get('SPLIT_K')})",
            )

    def test_heuristic_keeps_splitting_the_decode_regime(self):
        # Decode shape: P=128 capacity, skinny rank — the audited regression
        # collapsed exactly this to 1.
        split = split_k_heuristic(
            n=16, k_dim=2048, capacity=128, block_m=16, block_n=16, block_k=64
        )
        self.assertGreater(split, 1)
        self.assertLessEqual(split, 8)
        # Saturated prefill grid must not split.
        self.assertEqual(
            split_k_heuristic(
                n=256, k_dim=2048, capacity=65536, block_m=16, block_n=128, block_k=64
            ),
            1,
        )


class TestRunLoraASpecDispatch(CustomTestCase):
    """The spec IS the dispatch (second S3 review): what a record labels
    must be what executed, and unsupported combinations must raise instead
    of falling through to another family."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda")

    def setUp(self):
        torch.manual_seed(37)
        self.route = build_virtual_expert_routing(
            torch.randint(0, 8, (16, 4), dtype=torch.int32, device=self.device),
            torch.tensor(
                [i % 5 - 1 for i in range(16)], dtype=torch.int32, device=self.device
            ),
            lora_experts_per_adapter=8,
            max_loras=4,
            block_size=16,
        )
        self.input = torch.randn(16, 64, dtype=torch.bfloat16, device=self.device)
        self.weight = (
            torch.randn(32, 24, 64, dtype=torch.float32, device=self.device) * 64**-0.5
        ).to(torch.bfloat16)

    def test_spec_families_execute_their_own_kernels(self):
        from sglang.srt.lora.sgl_lora.bf16 import grouped_lora_a

        config = {
            "BLOCK_SIZE_N": 16,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 2,
        }
        via_spec = torch.zeros(64, 24, dtype=torch.bfloat16, device=self.device)
        run_lora_a(
            LoraAExecutionSpec(site="gate_up", ownership="grouped"),
            input=self.input,
            weight=self.weight,
            output=via_spec,
            routing=self.route,
            config=config,
        )
        direct = torch.zeros(64, 24, dtype=torch.bfloat16, device=self.device)
        grouped_lora_a(self.input, self.weight, direct, self.route, config=config)
        self.assertTrue(torch.equal(via_spec, direct))

        splitk_config = dict(config, SPLIT_K=2)
        via_spec.zero_()
        run_lora_a(
            LoraAExecutionSpec(
                site="gate_up",
                ownership="grouped",
                reduction="deterministic_split_k",
            ),
            input=self.input,
            weight=self.weight,
            output=via_spec,
            routing=self.route,
            config=splitk_config,
        )
        direct.zero_()
        invoke_split_k_lora_a(
            self.input, self.weight, direct, self.route, config=splitk_config
        )
        self.assertTrue(torch.equal(via_spec, direct))

    def test_unsupported_specs_raise_instead_of_falling_through(self):
        kwargs = dict(
            input=self.input,
            weight=self.weight,
            output=torch.zeros(64, 24, dtype=torch.bfloat16, device=self.device),
            routing=self.route,
            config=INDEXED_DEFAULT_CONFIG,
        )
        with self.assertRaises(NotImplementedError):
            run_lora_a(
                LoraAExecutionSpec(
                    site="gate_up",
                    ownership="indexed",
                    reduction="deterministic_split_k",
                ),
                **kwargs,
            )
        # P5: cutedsl is implemented (grouped whole-rank composite), so the
        # fail-closed contract moved — a DECLARED cutedsl spec without its
        # prepared plan must refuse to run, never fall through to Triton.
        with self.assertRaises(ValueError):
            run_lora_a(
                LoraAExecutionSpec(
                    site="gate_up", ownership="grouped", implementation="cutedsl"
                ),
                **kwargs,
            )
        with self.assertRaises(NotImplementedError):
            run_lora_a(
                LoraAExecutionSpec(
                    site="gate_up",
                    ownership="indexed",
                    implementation="cutedsl",
                ),
                **kwargs,
                cutedsl_plan=object(),
            )
        with self.assertRaises(NotImplementedError):
            run_lora_a(
                LoraAExecutionSpec(
                    site="gate_up",
                    ownership="grouped",
                    shared_handling="token_dedup",
                ),
                **kwargs,
            )


if __name__ == "__main__":
    unittest.main()
