"""Execution and binding contracts of the CuTeDSL LoRA-A plan.

Eighth S3 review: no registered test executed a real ``CutedslLoraAPlan``
— the dispatcher test only pinned that a declared CuTe arm without a plan
refuses to run. These cases execute the real composite (dispatch, staging,
masked grouped GEMM, scatter) and pin the contracts that reviews seven and
eight established:

* both sites produce grouped-baseline-close bridges on valid pairs;
* a gate/up invocation handing the DOWN weight is refused (the seventh-
  review check accepted a weight matching EITHER site's owner);
* a different route with the SAME [T, K] shape is refused (the old check
  compared against a derived copy's pointer, so shape was the only test);
* rebuilding metadata leaves outputs bitwise unchanged (the atomics assign
  a fresh within-group order; a stale inverse map silently permuted);
* rebuilding after the route CONTENTS grew a group beyond the sized m_max
  is refused instead of spilling rows into the next group's storage.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

from benchmark.kernels.lora_moe.bench_lora_a import _LegFixture, site_baseline
from benchmark.kernels.lora_moe.cases import AdapterCell, Topology, build_case
from benchmark.kernels.lora_moe.lora_a_candidates import run_lora_a
from benchmark.kernels.lora_moe.lora_a_execution import LoraAExecutionSpec
from benchmark.kernels.lora_moe.signal_gates import require_delta_close
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import ROUTE_ALIGNED, ROUTE_FUSED_IDS

SPEC = {
    site: LoraAExecutionSpec(site=site, ownership="grouped", implementation="cutedsl")
    for site in ("gate_up", "down")
}


def _make_fixture(device, seed):
    case = build_case(
        device=str(device),
        model_preset="qwen35_35b",
        topology=Topology(tp_size=8, ep_size=8),
        adapter_cell=AdapterCell(
            active_adapters=4, include_base_rows=True, slot_capacity=8
        ),
        route_generator="iid",
        num_tokens=64,
        active_rank=16,
        seed=seed,
        source_revision="cutedsl-plan-test",
    )
    fixture = _LegFixture(case, device)
    return fixture, fixture.route(ROUTE_ALIGNED), fixture.route(ROUTE_FUSED_IDS)


class TestCutedslPlanExecutionAndBinding(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        if torch.cuda.get_device_capability() < (9, 0):
            raise unittest.SkipTest("the masked grouped GEMM needs SM90+")
        try:
            import cutlass  # noqa: F401

            from benchmark.kernels.lora_moe.lora_a_cutedsl import (  # noqa: F401
                CutedslAConfig,
            )
        except ImportError as error:
            raise unittest.SkipTest(f"CuTeDSL unavailable: {error}")
        cls.device = torch.device("cuda")

    def _plan(self, fixture, fused):
        from benchmark.kernels.lora_moe.lora_a_cutedsl import (
            CutedslAConfig,
            build_cutedsl_lora_a_plan,
            supported_token_widths,
        )

        return build_cutedsl_lora_a_plan(
            fused_route=fused,
            gate_up_weight=fixture.a_gate_up,
            down_weight=fixture.a_down,
            config=CutedslAConfig(token_width=supported_token_widths(self.device)[0]),
        )

    def _run(self, fixture, aligned, plan, site):
        inp, weight, out = fixture.site_buffers(site)
        run_lora_a(
            SPEC[site],
            input=inp,
            weight=weight,
            output=out,
            routing=aligned,
            config=PROVISIONAL_LAUNCH_CONFIG.lora_a,
            cutedsl_plan=plan,
        )
        return out

    def test_executes_both_sites_close_to_grouped(self):
        fixture, aligned, fused = _make_fixture(self.device, seed=11)
        plan = self._plan(fixture, fused)
        for site in ("gate_up", "down"):
            baseline = site_baseline(fixture, site, aligned).clone()
            _, _, out = fixture.site_buffers(site)
            out.fill_(float("nan"))
            self._run(fixture, aligned, plan, site)
            valid = fixture.valid_pairs
            require_delta_close(
                out[valid].float(),
                baseline[valid].float(),
                gate_dtype=torch.bfloat16,
                label=f"cutedsl plan vs grouped at {site}",
            )

    def test_cross_site_weight_is_refused(self):
        fixture, aligned, fused = _make_fixture(self.device, seed=11)
        plan = self._plan(fixture, fused)
        with self.assertRaisesRegex(ValueError, "per fixture and per site"):
            run_lora_a(
                SPEC["gate_up"],
                input=fixture.hidden_states,
                weight=fixture.a_down,  # the DOWN weight on a gate/up call
                output=fixture.gate_rank_out,
                routing=aligned,
                config=PROVISIONAL_LAUNCH_CONFIG.lora_a,
                cutedsl_plan=plan,
            )

    def test_same_shape_foreign_route_is_refused(self):
        fixture, _, fused = _make_fixture(self.device, seed=11)
        plan = self._plan(fixture, fused)
        other, other_aligned, _ = _make_fixture(self.device, seed=137)
        with self.assertRaisesRegex(ValueError, "different route"):
            run_lora_a(
                SPEC["gate_up"],
                input=other.hidden_states,
                weight=fixture.a_gate_up,  # right weight, wrong route
                output=other.gate_rank_out,
                routing=other_aligned,  # same [T, K] shape, different tensors
                config=PROVISIONAL_LAUNCH_CONFIG.lora_a,
                cutedsl_plan=plan,
            )

    def test_metadata_rebuild_is_output_invariant(self):
        # Ninth S3 review: rebuild coverage must include the DOWN site,
        # whose staging consumes the refreshed inverse mapping directly.
        fixture, aligned, fused = _make_fixture(self.device, seed=11)
        plan = self._plan(fixture, fused)
        firsts = {
            site: self._run(fixture, aligned, plan, site).clone()
            for site in ("gate_up", "down")
        }
        plan.build_metadata()
        valid = fixture.valid_pairs
        for site in ("gate_up", "down"):
            _, _, out = fixture.site_buffers(site)
            out.fill_(float("nan"))
            self._run(fixture, aligned, plan, site)
            self.assertTrue(
                torch.equal(firsts[site][valid], out[valid]),
                f"rebuild permuted the {site} bridge output",
            )

    def test_within_capacity_sentinel_change_is_refused(self):
        # Ninth S3 review: a content change that fits every group can
        # still flip sentinel validity, leaving the frozen valid-pair set
        # stale — the rebuild must refuse, not silently mis-scatter.
        fixture, _, fused = _make_fixture(self.device, seed=11)
        plan = self._plan(fixture, fused)
        flat = plan.virtual_topk_ids.view(-1)
        first_valid = int(plan.valid_pair_ids[0])
        flat[first_valid] = -1  # shrink groups: within capacity, new valid set
        with self.assertRaisesRegex(ValueError, "route contents changed"):
            plan.build_metadata()

    def test_missing_routing_and_alias_weight_are_refused(self):
        fixture, aligned, fused = _make_fixture(self.device, seed=11)
        plan = self._plan(fixture, fused)
        with self.assertRaisesRegex(ValueError, "cannot be skipped"):
            plan.require_binding("gate_up", fixture.a_gate_up, None)
        # Same storage, different view semantics (ninth S3 review).
        alias = fixture.a_gate_up.view(-1, fixture.a_gate_up.shape[2])
        with self.assertRaisesRegex(ValueError, "per fixture and per site"):
            plan.require_binding("gate_up", alias, aligned)

    def test_rebuild_with_wider_group_is_refused(self):
        fixture, _, fused = _make_fixture(self.device, seed=11)
        plan = self._plan(fixture, fused)
        # Skew the route contents in place: every pair lands in one group,
        # which exceeds the m_max sized for the balanced route. The guard
        # must refuse BEFORE any plan state or staging buffer is touched.
        plan.virtual_topk_ids.fill_(0)
        with self.assertRaisesRegex(ValueError, "exceeds the sized m_max"):
            plan.build_metadata()


if __name__ == "__main__":
    unittest.main()
