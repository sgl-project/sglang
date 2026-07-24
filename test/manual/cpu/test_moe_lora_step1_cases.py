"""CPU tests for the Step-1 case schema, routes, reference, and signal gates.

GPU kernel validation lives in ``check_step1_correctness.py``; these tests
prove the harness itself: deterministic case resolution, route statistics,
reference semantics, and the §21.1 tolerance policy.
"""

import unittest

import msgspec
import torch

from benchmark.kernels.lora_moe.cases import (
    AdapterCell,
    MODEL_PRESETS,
    MoeLoraBenchCase,
    Topology,
    build_case,
    materialize_case_tensors,
)
from benchmark.kernels.lora_moe.reference import (
    reference_local_moe,
    reference_pair_stages,
)
from benchmark.kernels.lora_moe.routes import (
    generate_topk_ids,
    resolve_route_stats,
)
from sglang.test.moe_lora_signal_gates import (
    DegenerateSignalError,
    check_delta,
    nan_poison_,
    require_bitwise_equal,
    require_finite,
    require_signal_close,
    resolve_signal_gates,
)


def _tiny_case(**overrides) -> MoeLoraBenchCase:
    keywords = dict(
        device="cpu",
        model_preset="tiny_smoke",
        adapter_cell=AdapterCell(
            active_adapters=2, include_base_rows=True, slot_capacity=4
        ),
        route_generator="iid",
        num_tokens=16,
        active_rank=16,
        route_coeff_precision="bf16_rounded",
        source_revision="test",
        seed=7,
    )
    keywords.update(overrides)
    return build_case(**keywords)


class TestCaseSchema(unittest.TestCase):
    def test_case_id_is_stable_and_content_addressed(self):
        first = _tiny_case()
        second = _tiny_case()
        self.assertEqual(first.case_id, second.case_id)
        self.assertNotEqual(first.case_id, _tiny_case(seed=8).case_id)

    def test_case_serializes_to_json(self):
        case = _tiny_case()
        decoded = msgspec.json.decode(
            msgspec.json.encode(case), type=MoeLoraBenchCase
        )
        self.assertEqual(decoded, case)

    def test_topology_derivation(self):
        case = _tiny_case(
            model_preset="qwen35_35b",
            topology=Topology(tp_size=8, ep_size=8),
            num_tokens=4,
        )
        self.assertEqual(case.num_experts_local, 32)
        self.assertEqual(case.intermediate_size_local, 512)

    def test_route_stats_recorded(self):
        case = _tiny_case()
        self.assertGreater(case.p_valid, 0)
        self.assertGreaterEqual(case.p_aligned, case.p_valid)
        self.assertEqual(
            sum(size * count for size, count in case.group_size_histogram.items()),
            case.p_valid,
        )

    def test_no_local_route_has_zero_valid_pairs(self):
        case = _tiny_case(
            route_generator="no_local",
            expert_id_domain="global",
            topology=Topology(tp_size=2, ep_size=2),
        )
        self.assertEqual(case.p_valid, 0)

    def test_rank_tail_is_zero_by_loader_contract(self):
        case = _tiny_case(active_rank=8, max_rank=16, physical_rank=16)
        tensors = materialize_case_tensors(case)
        tail = tensors.lora_a_down[:, :, 8:16, :].float()
        self.assertEqual(float(tail.abs().max()), 0.0)


class TestRoutes(unittest.TestCase):
    def test_topk_ids_distinct_within_token(self):
        for generator_name in ("balanced", "iid", "hotset_80_20", "one_hot"):
            ids = generate_topk_ids(
                route_generator=generator_name,
                num_tokens=32,
                top_k=4,
                num_routable_experts=16,
                num_local_experts=16,
                seed=3,
            )
            for row in ids.tolist():
                self.assertEqual(len(set(row)), len(row), generator_name)

    def test_stats_match_manual_count(self):
        ids = torch.tensor([[0, 1], [1, 2], [3, 0]], dtype=torch.int32)
        mapping = torch.tensor([0, -1, 1])
        stats = resolve_route_stats(
            topk_ids=ids,
            token_lora_mapping=mapping,
            factor_expert_count=4,
            max_loras=2,
            block_size=4,
            routed_expert_to_factor_id=None,
        )
        # Tokens 0 and 2 carry adapters -> 4 valid pairs, 4 distinct
        # (adapter, expert) groups over 3 distinct experts {0, 1, 3}.
        self.assertEqual(stats.p_valid, 4)
        self.assertEqual(stats.group_count, 4)
        self.assertEqual(stats.e_hit, 3)
        self.assertEqual(stats.p_aligned, 16)


class TestReferenceSemantics(unittest.TestCase):
    def test_zero_lora_matches_base_only_bitwise(self):
        case = _tiny_case(
            adapter_cell=AdapterCell(
                active_adapters=0, include_base_rows=True, slot_capacity=4
            )
        )
        tensors = materialize_case_tensors(case)
        with_lora = reference_local_moe(case, tensors, include_lora=True)
        base_only = reference_local_moe(case, tensors, include_lora=False)
        require_bitwise_equal(with_lora, base_only, label="zero-lora parity")

    def test_lora_signal_is_present_and_scales_linearly(self):
        case = _tiny_case()
        tensors = materialize_case_tensors(case)
        base = reference_local_moe(case, tensors, include_lora=False)
        full = reference_local_moe(case, tensors, include_lora=True)
        delta = full - base
        self.assertGreater(float(delta.abs().max()), 0.0)

        # Down-only linearity is exact: with gate/up factors zeroed, the
        # activation is unchanged and doubling B_down doubles the delta
        # (scaling BF16 by 2 is exact).
        down_only = materialize_case_tensors(case)
        down_only.lora_a_gate_up = torch.zeros_like(down_only.lora_a_gate_up)
        base_ref = reference_local_moe(case, down_only, include_lora=False)
        delta_one = reference_local_moe(case, down_only) - base_ref
        down_only.lora_b_down = down_only.lora_b_down * 2
        delta_two = reference_local_moe(case, down_only) - base_ref
        torch.testing.assert_close(delta_two, delta_one * 2, rtol=1e-6, atol=1e-6)

    def test_gated_algebra_matches_direct_formula(self):
        case = _tiny_case(num_tokens=4)
        tensors = materialize_case_tensors(case)
        stages = reference_pair_stages(case, tensors)
        pair = int(torch.nonzero(stages.pair_adapter >= 0)[0])
        token, expert = pair // case.top_k, int(stages.pair_expert[pair])
        adapter = int(stages.pair_adapter[pair])
        x = tensors.hidden_states[token].float()
        i_local = case.intermediate_size_local
        r = case.physical_rank

        a = tensors.lora_a_gate_up[adapter, expert].float()
        b = tensors.lora_b_gate_up[adapter, expert].float()
        a_out = a @ x
        expected_gate = b[:i_local, :] @ a_out[:r]
        expected_up = b[i_local:, :] @ a_out[r:]
        observed = stages.gate_up_delta[pair]
        torch.testing.assert_close(
            observed[:i_local], expected_gate, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            observed[i_local:], expected_up, rtol=1e-5, atol=1e-5
        )

    def test_nongated_relu2_and_shared_outer_run(self):
        case = _tiny_case(model_preset="tiny_smoke_relu2")
        tensors = materialize_case_tensors(case)
        stages = reference_pair_stages(case, tensors)
        self.assertGreaterEqual(float(stages.activation.min()), 0.0)

        shared = _tiny_case(shared_factor_signature="shared_both")
        shared_tensors = materialize_case_tensors(shared)
        self.assertEqual(shared_tensors.lora_a_gate_up.shape[1], 1)
        self.assertEqual(shared_tensors.lora_b_down.shape[1], 1)
        delta = reference_local_moe(shared, shared_tensors) - reference_local_moe(
            shared, shared_tensors, include_lora=False
        )
        self.assertGreater(float(delta.abs().max()), 0.0)

    def test_slice_targeting_zeroes_untargeted_half(self):
        case = _tiny_case(slice_target="gate_only")
        tensors = materialize_case_tensors(case)
        r = case.physical_rank
        self.assertEqual(
            float(tensors.lora_a_gate_up[:, :, r:, :].float().abs().max()), 0.0
        )
        self.assertGreater(
            float(tensors.lora_a_gate_up[:, :, :r, :].float().abs().max()), 0.0
        )

    def test_routed_scaling_applied_exactly_once(self):
        unit = _tiny_case(routed_scaling_factor=1.0)
        scaled = _tiny_case(routed_scaling_factor=2.0)
        tensors = materialize_case_tensors(unit)
        torch.testing.assert_close(
            reference_local_moe(scaled, tensors),
            reference_local_moe(unit, tensors) * 2.0,
        )


class TestSignalGates(unittest.TestCase):
    def test_zero_signal_is_rejected(self):
        with self.assertRaises(DegenerateSignalError):
            resolve_signal_gates(
                torch.zeros(8), destination_dtype=torch.bfloat16
            )

    def test_signal_below_noise_floor_is_rejected(self):
        with self.assertRaises(DegenerateSignalError):
            resolve_signal_gates(
                torch.full((8,), 1e-4),
                destination_dtype=torch.bfloat16,
                base_reference=torch.full((8,), 100.0),
            )

    def test_dropped_delta_fails_gate(self):
        reference = torch.randn(64)
        gates = resolve_signal_gates(reference, destination_dtype=torch.bfloat16)
        record = check_delta(torch.zeros(64), reference, gates)
        self.assertFalse(record.passed)

    def test_bf16_rounding_passes_gate(self):
        reference = torch.randn(4096)
        gates = resolve_signal_gates(reference, destination_dtype=torch.bfloat16)
        rounded = reference.to(torch.bfloat16).float()
        record = check_delta(rounded, reference, gates)
        self.assertTrue(record.passed)

    def test_require_signal_close_end_to_end(self):
        base = torch.randn(32, 16) * 10
        delta = torch.randn(32, 16)
        reference_output = base + delta
        observed = base + delta.to(torch.bfloat16).float()
        record = require_signal_close(
            observed,
            reference_output,
            base_reference=base,
            destination_dtype=torch.bfloat16,
            label="unit",
        )
        self.assertTrue(record.passed)
        with self.assertRaises(AssertionError):
            require_signal_close(
                base,  # dropped delta
                reference_output,
                base_reference=base,
                destination_dtype=torch.bfloat16,
                label="dropped",
            )

    def test_poison_helpers(self):
        buffer = torch.ones(8)
        nan_poison_(buffer)
        with self.assertRaises(AssertionError):
            require_finite(buffer, label="poisoned")
        clean = torch.ones(8)
        require_finite(clean, label="clean")


if __name__ == "__main__":
    unittest.main()
