"""VL routing must dispatch and record physical experts, including graph replay.

Regression for #42188: ordinary CUDA VL routing skipped the standard TopK
finalizer, so non-identity EPLB placement dispatched the wrong experts and the
expert recorder stayed empty even for text-only requests.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.eplb.expert_distribution import _SelectExpertsSinglePassGatherer
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe.topk import StandardTopKOutputPacked, TopKConfig
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.multimodal.dsv41.vl_routing import vision_topk
from sglang.srt.runtime_context import get_flags, get_parallel, get_resources
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


class _CountingRecorder:
    """Use the real per-layer counter without starting a distributed engine."""

    def __init__(self, num_experts):
        self.gatherer = _SelectExpertsSinglePassGatherer.__new__(
            _SelectExpertsSinglePassGatherer
        )
        self.gatherer._expert_location_metadata = SimpleNamespace(
            num_physical_experts=num_experts
        )
        self.gatherer._data = torch.zeros(
            (1, num_experts), device="cuda", dtype=torch.int32
        )

    def on_select_experts(self, *, topk_ids):
        self.gatherer.on_select_experts(0, topk_ids)

    @property
    def counts(self):
        return self.gatherer._data[0]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA VL router required")
class TestVisionTopKFinalization(CustomTestCase):
    num_experts = 384
    routed_k = 6

    def setUp(self):
        super().setUp()
        topology = published_topology(enable_eplb=True)
        topology.__enter__()
        self.addCleanup(topology.__exit__, None, None, None)
        flags = get_flags().moe.override(
            a2a_backend=MoeA2ABackend.NONE,
            runner_backend=MoeRunnerBackend.FLASHINFER_MXFP4,
        )
        flags.__enter__()
        self.addCleanup(flags.__exit__, None, None, None)
        self.recorder = _CountingRecorder(self.num_experts * 2)
        recorder_patch = patch.object(
            get_resources(), "expert_distribution_recorder", self.recorder
        )
        recorder_patch.start()
        self.addCleanup(recorder_patch.stop)

    def _moe(self, shared=0, scale_in_topk=False):
        bias = torch.arange(self.num_experts, device="cuda", dtype=torch.float32)
        return SimpleNamespace(
            layer_id=0,
            config=SimpleNamespace(image_token_id=99),
            gate=SimpleNamespace(
                e_score_correction_bias=bias,
                e_score_correction_bias_vl=-bias,
            ),
            topk=SimpleNamespace(
                topk_config=TopKConfig(
                    top_k=self.routed_k + shared,
                    num_fused_shared_experts=shared,
                    renormalize=True,
                    routed_scaling_factor=2.5,
                    apply_routed_scaling_factor_on_output=scale_in_topk,
                    fused_shared_experts_scaling_factor=0.25 if shared else None,
                    scoring_func="sqrtsoftplus",
                )
            ),
        )

    def _dispatch(self, dynamic=False):
        permutation = torch.arange(
            self.num_experts, device="cuda", dtype=torch.int32
        ).roll(137)
        return ExpertLocationDispatchInfo(
            ep_dispatch_algorithm="dynamic" if dynamic else "static",
            partial_logical_to_rank_dispatch_physical_map=permutation,
            partial_logical_to_all_physical_map=torch.stack(
                [permutation, permutation + self.num_experts], dim=1
            ),
            partial_logical_to_all_physical_map_num_valid=torch.full(
                (self.num_experts,), 2, device="cuda", dtype=torch.int32
            ),
            num_physical_experts=self.num_experts * 2,
            rank_invariant=True,
        )

    def _expected_ids(self, input_ids, valid, info=None):
        text = torch.arange(
            self.num_experts - self.routed_k,
            self.num_experts,
            device="cuda",
            dtype=torch.int32,
        )
        image = torch.arange(self.routed_k, device="cuda", dtype=torch.int32)
        logical = torch.where((input_ids == 99)[:, None], image, text)
        if info is not None:
            logical = info.partial_logical_to_rank_dispatch_physical_map[logical]
            if info.ep_dispatch_algorithm == "dynamic":
                logical = (
                    logical
                    + (
                        torch.arange(
                            input_ids.numel(), device="cuda", dtype=torch.int32
                        )
                        % 2
                    )[:, None]
                    * self.num_experts
                )
        return logical[:valid].sort(dim=1).values

    def _check(self, output, input_ids, valid, info=None, shared=0, folded=False):
        actual = output.topk_ids[:, : self.routed_k]
        expected = self._expected_ids(input_ids, valid, info)
        torch.testing.assert_close(actual[:valid].sort(dim=1).values, expected)
        torch.testing.assert_close(actual[valid:], torch.full_like(actual[valid:], -1))
        expected_counts = torch.bincount(
            expected.flatten().long(), minlength=self.recorder.counts.numel()
        )
        torch.testing.assert_close(self.recorder.counts, expected_counts.int())
        routed_weight = (2.5 if folded else 1.0) / self.routed_k
        torch.testing.assert_close(
            output.topk_weights[:valid, : self.routed_k],
            torch.full_like(
                output.topk_weights[:valid, : self.routed_k], routed_weight
            ),
        )
        torch.testing.assert_close(
            output.topk_weights[valid:], torch.zeros_like(output.topk_weights[valid:])
        )
        if shared:
            torch.testing.assert_close(
                output.topk_ids[:valid, -1],
                torch.full_like(output.topk_ids[:valid, -1], self.num_experts),
            )
            torch.testing.assert_close(
                output.topk_weights[:valid, -1],
                torch.full_like(
                    output.topk_weights[:valid, -1], 0.25 if folded else 0.1
                ),
            )

    def test_text_and_image_rows_are_recorded(self):
        for ids in ([7, 8, 9, 10], [7, 99, 8, 99]):
            with self.subTest(ids=ids):
                self.recorder.counts.zero_()
                input_ids = torch.tensor(ids, device="cuda")
                output = vision_topk(
                    self._moe(),
                    torch.zeros((4, self.num_experts), device="cuda"),
                    input_ids,
                    num_token_non_padded=torch.tensor(
                        3, device="cuda", dtype=torch.int32
                    ),
                )
                self._check(output, input_ids, 3)
                self.assertIsInstance(output, StandardTopKOutputPacked)
                packed = output.packed_topk_ids.to(torch.int64) & 0xFFFFFFFF
                packed_ids = (packed >> 16).int()
                packed_weights = (packed & 0xFFFF).to(torch.int16).view(torch.bfloat16)
                torch.testing.assert_close(packed_ids[:3], output.topk_ids[:3])
                torch.testing.assert_close(
                    packed_weights.float(),
                    output.topk_weights.to(torch.bfloat16).float(),
                )

    def test_nonidentity_mapping_disables_stale_packed_ids(self):
        input_ids = torch.tensor([7, 99, 8, 99], device="cuda")
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                self.recorder.counts.zero_()
                info = self._dispatch(dynamic)
                output = vision_topk(
                    self._moe(),
                    torch.zeros((4, self.num_experts), device="cuda"),
                    input_ids,
                    num_token_non_padded=torch.tensor(
                        3, device="cuda", dtype=torch.int32
                    ),
                    expert_location_dispatch_info=info,
                )
                self._check(output, input_ids, 3, info)
                self.assertNotIsInstance(output, StandardTopKOutputPacked)

    def test_shared_scaling_and_padding_after_mapping(self):
        input_ids = torch.tensor([7, 99, 8, 99], device="cuda")
        for folded in (False, True):
            for mapped in (False, True):
                with self.subTest(folded=folded, mapped=mapped):
                    self.recorder.counts.zero_()
                    info = self._dispatch() if mapped else None
                    output = vision_topk(
                        self._moe(shared=1, scale_in_topk=folded),
                        torch.zeros((4, self.num_experts), device="cuda"),
                        input_ids,
                        num_token_non_padded=torch.tensor(
                            3, device="cuda", dtype=torch.int32
                        ),
                        expert_location_dispatch_info=info,
                    )
                    self._check(output, input_ids, 3, info, shared=1, folded=folded)

    def test_per_rank_shared_slot_keeps_routed_recording_and_masks_padding(self):
        input_ids = torch.tensor([7, 99, 8, 99], device="cuda")
        info = self._dispatch()
        # Four EP ranks split the physical experts plus one shared slot each.
        with (
            get_flags().moe.override(a2a_backend=MoeA2ABackend.DEEPEP),
            get_parallel().override(
                tp_size=4,
                attn_tp_size=4,
                moe_ep_size=4,
                moe_ep_rank=2,
                tp_rank=2,
                attn_tp_rank=2,
            ),
        ):
            output = vision_topk(
                self._moe(shared=1),
                torch.zeros((4, self.num_experts), device="cuda"),
                input_ids,
                num_token_non_padded=torch.tensor(3, device="cuda", dtype=torch.int32),
                expert_location_dispatch_info=info,
            )
        # The dispatch map has 768 physical routed experts, hence 192 per rank.
        physical = self._expected_ids(input_ids, 3, info)
        slots = physical + physical // 192
        torch.testing.assert_close(output.topk_ids[:3, :6].sort(dim=1).values, slots)
        torch.testing.assert_close(
            output.topk_ids[:3, -1],
            torch.full_like(output.topk_ids[:3, -1], 2 * 193 + 192),
        )
        torch.testing.assert_close(
            output.topk_weights[:3, -1],
            torch.full_like(output.topk_weights[:3, -1], 0.4),
        )
        torch.testing.assert_close(
            output.topk_ids[3:], torch.full_like(output.topk_ids[3:], -1)
        )
        torch.testing.assert_close(
            output.topk_weights[3:], torch.zeros_like(output.topk_weights[3:])
        )
        torch.testing.assert_close(
            self.recorder.counts,
            torch.bincount(
                physical.flatten().long(), minlength=self.recorder.counts.numel()
            ).int(),
        )

    def test_empty_input_has_no_routes(self):
        output = vision_topk(
            self._moe(),
            torch.empty((0, self.num_experts), device="cuda"),
            torch.empty(0, device="cuda", dtype=torch.int64),
        )
        self.assertEqual(output.topk_ids.shape, (0, self.routed_k))
        self.assertEqual(output.topk_weights.shape, (0, self.routed_k))
        self.assertEqual(int(self.recorder.counts.sum()), 0)

    def test_graph_replay_refreshes_modality_mapping_and_padding(self):
        moe = self._moe()
        logits = torch.zeros((4, self.num_experts), device="cuda")
        input_ids = torch.full((4,), 7, device="cuda", dtype=torch.int64)
        valid = torch.tensor(4, device="cuda", dtype=torch.int32)
        info = self._dispatch()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                vision_topk(moe, logits, input_ids, valid, info)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = vision_topk(moe, logits, input_ids, valid, info)
        for ids, count in (([7, 7, 7, 7], 4), ([99, 7, 99, 7], 3), ([7, 7, 7, 7], 0)):
            with self.subTest(ids=ids, valid=count):
                input_ids.copy_(torch.tensor(ids, device="cuda"))
                valid.fill_(count)
                self.recorder.counts.zero_()
                graph.replay()
                self._check(output, input_ids, count, info)

    def test_model_entry_points_dispatch_physical_experts(self):
        """Both MoE streams must pass placement to VL routing before dispatch."""
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        class Experts:
            quant_method = None
            moe_runner_config = SimpleNamespace(inplace=False)

            def __call__(self, hidden, output):
                # A small numerical expert boundary makes wrong dispatch visible
                # in the model's result, without loading checkpoint weights.
                factors = (output.topk_ids + 1).float()
                return hidden * (factors * output.topk_weights).sum(1, keepdim=True)

        input_ids = torch.tensor([7, 99, 8, 99], device="cuda")
        hidden = torch.ones((4, 3), device="cuda")
        valid = torch.tensor(3, device="cuda", dtype=torch.int32)
        info = self._dispatch()
        expected = hidden.clone()
        expected[:3] *= (
            (self._expected_ids(input_ids, 3, info) + 1).float().mean(1, keepdim=True)
        )
        expected[3:] = 0
        for dual_stream in (False, True):
            with self.subTest(dual_stream=dual_stream):
                self.recorder.counts.zero_()
                moe = self._moe()
                gate = lambda *_args: torch.zeros((4, self.num_experts), device="cuda")
                gate.e_score_correction_bias = moe.gate.e_score_correction_bias
                gate.e_score_correction_bias_vl = moe.gate.e_score_correction_bias_vl
                moe.gate = gate
                moe.experts = Experts()
                moe.is_nextn = False
                moe.num_fused_shared_experts = 0
                moe._shared_expert_tp1 = True
                moe._fuse_shared_experts_inside_sbo = False
                moe.alt_stream = torch.cuda.Stream()
                moe._maybe_quant_moe_input_once = lambda *_args: None
                moe._should_quant_routed_input_mxfp8 = lambda *_args: False
                moe._forward_shared_experts = lambda *_args, **_kwargs: None
                moe.routed_scaling_factor = 2.5
                forward = (
                    DeepseekV2MoE.forward_normal_dual_stream
                    if dual_stream
                    else DeepseekV2MoE.forward_normal
                )
                kwargs = {} if dual_stream else {"skip_shared_experts": True}
                with patch.object(
                    ExpertLocationDispatchInfo, "init_new", return_value=info
                ):
                    output = forward(
                        moe,
                        hidden,
                        input_ids_global=input_ids,
                        num_token_non_padded=valid,
                        use_vision_topk=True,
                        return_moe_output=True,
                        **kwargs,
                    )
                torch.testing.assert_close(output.routed, expected)
                self.assertEqual(int(self.recorder.counts.sum()), 3 * self.routed_k)


if __name__ == "__main__":
    unittest.main()
