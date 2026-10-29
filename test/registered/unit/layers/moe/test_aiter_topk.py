"""Tests for srt/layers/moe/aiter_topk.py, the AITER grouped-topk fusion that
writes routed columns into a persistent buffer with pre-filled shared columns.

``TestAiterSharedColumnsKernel`` runs the real AITER kernel (ROCm only).
``TestAiterSharedColumnsContract`` stubs the kernel so the host-side contract
(eligibility, buffer layout and lifetime, select_experts wiring) runs on every
GPU runner.
"""

import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.fused_moe_triton_kernels import fused_append_shared_experts
from sglang.srt.configs.shared_experts import SharedExpertsConfig
from sglang.srt.environ import envs
from sglang.srt.layers.moe import aiter_topk
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig, _use_aiter
from sglang.srt.layers.moe.utils import MoeA2ABackend
from sglang.srt.runtime_context import (
    get_context,
    get_flags,
    get_resources,
    get_schedule,
    reset_context,
)
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
# backend-specific: only ROCm runs the real AITER kernel through the strided
# shared-column buffer, including CUDA-graph replay.
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")


def _shared_column_buffers():
    return {
        name: buf
        for name, buf in get_resources().buffers.items()
        if name.startswith("aiter_topk_shared_columns:")
    }


def _publish(test):
    """Publish a default config with fresh resources for the test's lifetime."""
    reset_context()
    test.addCleanup(reset_context)
    get_context().override_server_args().install()


def _full_batch(num_tokens, device):
    # A non-padded count keeps select_experts on the ordinary routing + append
    # path while masking no row, which makes it the reference for the fusion.
    return torch.tensor([num_tokens], dtype=torch.int32, device=device)


@unittest.skipUnless(
    torch.cuda.is_available() and _use_aiter,
    "the real biased_grouped_topk kernel ships with AITER on ROCm",
)
class TestAiterSharedColumnsKernel(CustomTestCase):
    """The fused output must be bit-identical to routing followed by the append."""

    # (num_tokens, num_experts, num_expert_group, topk_group, topk_routed,
    #  n_shared, routed_scaling_factor, renormalize).
    CASES = [
        # Kimi K2.5: 384 routed experts, one group, top-8, one shared.
        (1, 384, 1, 1, 8, 1, 2.827, True),
        (37, 384, 1, 1, 8, 1, 2.827, True),
        (8192, 384, 1, 1, 8, 1, 2.827, True),
        # DeepSeek-V3 grouped routing: 8 groups, 4 of them kept.
        (37, 256, 8, 4, 8, 1, 2.5, True),
        (512, 256, 8, 4, 8, 1, 2.5, False),
        # A model without a routed scaling factor.
        (37, 256, 8, 4, 8, 1, None, True),
    ]

    def setUp(self):
        super().setUp()
        _publish(self)

    def _inputs(self, num_tokens, num_experts):
        device = get_device()
        g = torch.Generator(device="cpu").manual_seed(num_tokens * 31 + num_experts)
        hidden_states = torch.randn(
            (num_tokens, 16), generator=g, dtype=torch.bfloat16
        ).to(device)
        router_logits = torch.randn(
            (num_tokens, num_experts), generator=g, dtype=torch.bfloat16
        ).to(device)
        correction_bias = torch.randn(
            (num_experts,), generator=g, dtype=torch.float32
        ).to(device)
        return hidden_states, router_logits, correction_bias

    @staticmethod
    def _config(bias, groups, topk_group, topk_routed, s, rsf, renormalize, factor):
        return TopKConfig(
            top_k=topk_routed + s,
            use_grouped_topk=True,
            renormalize=renormalize,
            num_expert_group=groups,
            topk_group=topk_group,
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=s),
            correction_bias=bias,
            routed_scaling_factor=rsf,
            fused_shared_experts_scaling_factor=factor,
        )

    def test_fused_output_matches_routing_then_append(self):
        # None and 1 are the two spellings models use for an unscaled shared
        # expert; both must land on the same weight.
        for factor in (None, 1, 0.25):
            for case in self.CASES:
                num_tokens, num_experts, groups, topk_group = case[:4]
                with self.subTest(case=case, factor=factor):
                    hidden, logits, bias = self._inputs(num_tokens, num_experts)
                    config = self._config(bias, groups, topk_group, *case[4:], factor)
                    fused = aiter_topk.try_select_experts(hidden, logits, config)
                    self.assertIsNotNone(fused)
                    plain = topk_module.select_experts(
                        hidden,
                        logits,
                        config,
                        num_token_non_padded=_full_batch(num_tokens, logits.device),
                    )
                    self.assertEqual(fused.topk_ids.shape, plain.topk_ids.shape)
                    self.assertTrue(torch.equal(fused.topk_ids, plain.topk_ids))
                    self.assertTrue(torch.equal(fused.topk_weights, plain.topk_weights))

    def test_select_experts_skips_append_and_preserves_padding(self):
        hidden, logits, bias = self._inputs(37, 384)
        config = self._config(bias, 1, 1, 8, 1, 2.827, True, 0.25)
        append_path = "sglang.kernels.ops.moe.fused_moe_triton_kernels.fused_append_shared_experts"
        with (
            patch(append_path, wraps=fused_append_shared_experts) as append,
            patch.object(topk_module, "capture_routed_experts_if_allowed") as capture,
        ):
            fused = topk_module.select_experts(hidden, logits, config)
            append.assert_not_called()
            # Routed-expert capture sees the routed columns only.
            self.assertEqual(capture.call_args.args[2].shape[1], 8)
            expected_ids = fused.topk_ids.clone()
            expected_w = fused.topk_weights.clone()
            valid = torch.tensor([17], dtype=torch.int32, device=logits.device)
            padded = topk_module.select_experts(
                hidden, logits, config, num_token_non_padded=valid
            )
            append.assert_called_once()
            self.assertTrue(torch.equal(padded.topk_ids[:17], expected_ids[:17]))
            self.assertTrue(torch.equal(padded.topk_weights[:17], expected_w[:17]))
            self.assertEqual(torch.count_nonzero(padded.topk_weights[17:]).item(), 0)
            append.reset_mock()
            reused = topk_module.select_experts(hidden, logits, config)
            append.assert_not_called()
            self.assertTrue(torch.equal(reused.topk_ids, expected_ids))
            self.assertTrue(torch.equal(reused.topk_weights, expected_w))

    def test_graph_replay_tracks_new_logits(self):
        hidden, logits, bias = self._inputs(37, 384)
        config = self._config(bias, 1, 1, 8, 1, 2.827, True, 1)
        aiter_topk.try_select_experts(hidden, logits, config)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = aiter_topk.try_select_experts(hidden, logits, config)
        for _ in range(3):
            logits.normal_()
            graph.replay()
            ref = topk_module.select_experts(
                hidden,
                logits,
                config,
                num_token_non_padded=_full_batch(37, logits.device),
            )
            self.assertTrue(torch.equal(out.topk_ids, ref.topk_ids))
            self.assertTrue(torch.equal(out.topk_weights, ref.topk_weights))


class _FakeAiterBiasedGroupedTopk:
    """Stand-in for ``aiter.biased_grouped_topk`` with the same call contract.

    The fusion needs only in-place writes that honor the row stride of the
    views it is handed; ``copy_`` keeps that, so a caller that passed
    contiguous scratch instead of a view into the buffer would lose its
    shared columns.
    """

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        gating_output,
        correction_bias,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        renormalize,
        routed_scaling_factor,
    ):
        self.calls.append(
            {
                "weights_shape": tuple(topk_weights.shape),
                "weights_row_stride": topk_weights.stride(0),
                "ids_row_stride": topk_ids.stride(0),
                "weights_ptr": topk_weights.data_ptr(),
                "ids_ptr": topk_ids.data_ptr(),
            }
        )
        k = topk_weights.shape[1]
        scores = gating_output.float() + correction_bias.float()
        values, indices = torch.topk(scores, k, dim=-1)
        if renormalize:
            values = values / values.sum(dim=-1, keepdim=True)
        topk_ids.copy_(indices.to(topk_ids.dtype))
        topk_weights.copy_((values * routed_scaling_factor).to(topk_weights.dtype))


@unittest.skipUnless(
    torch.cuda.is_available(), "the fusion allocates its buffers on the accelerator"
)
class TestAiterSharedColumnsContract(CustomTestCase):
    """Host-side contract of the fusion, against a stubbed kernel."""

    NUM_EXPERTS = 384
    TOPK_ROUTED = 8
    N_SHARED = 1

    def setUp(self):
        super().setUp()
        _publish(self)
        self.fake_kernel = _FakeAiterBiasedGroupedTopk()
        fake_aiter = types.ModuleType("aiter")
        fake_aiter.biased_grouped_topk = self.fake_kernel
        # The fusion imports aiter lazily, so swapping the module entry covers
        # both "aiter absent" (NVIDIA) and "aiter present" (ROCm).
        patcher = patch.dict(sys.modules, {"aiter": fake_aiter})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _config(self, **overrides):
        kwargs = dict(
            top_k=self.TOPK_ROUTED + self.N_SHARED,
            use_grouped_topk=True,
            renormalize=True,
            num_expert_group=1,
            topk_group=1,
            shared_experts=SharedExpertsConfig(num_fused_shared_experts=self.N_SHARED),
            correction_bias=torch.randn(
                (self.NUM_EXPERTS,), dtype=torch.float32, device=get_device()
            ),
            routed_scaling_factor=2.827,
            fused_shared_experts_scaling_factor=1.0,
        )
        kwargs.update(overrides)
        return TopKConfig(**kwargs)

    def _inputs(self, num_tokens=37):
        device = get_device()
        hidden_states = torch.randn(
            (num_tokens, 16), dtype=torch.bfloat16, device=device
        )
        router_logits = torch.randn(
            (num_tokens, self.NUM_EXPERTS), dtype=torch.bfloat16, device=device
        )
        return hidden_states, router_logits

    def _fuse(self, config, num_tokens=37, **kwargs):
        hidden_states, router_logits = self._inputs(num_tokens)
        return aiter_topk.try_select_experts(
            hidden_states, router_logits, config, **kwargs
        )

    def test_ineligible_configs_fall_back_without_touching_aiter(self):
        """Every guard in the eligibility predicate returns None, allocates no
        buffer, and never reaches the kernel."""
        config_cases = {
            "not_grouped_topk": {"use_grouped_topk": False},
            "no_correction_bias": {"correction_bias": None},
            "no_shared_experts": {"shared_experts": SharedExpertsConfig(), "top_k": 8},
            "no_expert_group": {"num_expert_group": None},
            "no_topk_group": {"topk_group": None},
            "scaling_on_output": {"apply_routed_scaling_factor_on_output": True},
        }
        for name, overrides in config_cases.items():
            with self.subTest(case=name):
                self.assertIsNone(self._fuse(self._config(**overrides)))

        call_cases = {
            "padded_tokens": {
                "num_token_non_padded": torch.tensor([17], device=get_device())
            },
            "expert_location_dispatch": {"expert_location_dispatch_info": object()},
        }
        for name, kwargs in call_cases.items():
            with self.subTest(case=name):
                self.assertIsNone(self._fuse(self._config(), **kwargs))

        with self.subTest(case="expert_parallel"):
            with get_context().override_server_args(tp_size=2, ep_size=2):
                self.assertIsNone(self._fuse(self._config()))

        with self.subTest(case="per_rank_shared_slots"):
            with get_flags().moe.override(a2a_backend=MoeA2ABackend.DEEPEP):
                self.assertIsNone(self._fuse(self._config()))

        for setting in (
            envs.SGLANG_SIMULATE_UNIFORM_EXPERTS,
            envs.SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS,
            envs.SGLANG_K3_RADIX4_TOPK,
            envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK,
        ):
            with self.subTest(case=setting.name), setting.override(True):
                self.assertIsNone(self._fuse(self._config()))

        with self.subTest(case="config_unpublished"):
            reset_context()
            self.assertIsNone(self._fuse(self._config()))

        self.assertEqual(self.fake_kernel.calls, [])
        self.assertEqual(_shared_column_buffers(), {})

    def test_shared_columns_are_prefilled_and_survive_the_kernel(self):
        """The returned rows carry the routed kernel output plus the prefilled
        shared columns, and the kernel only ever sees the routed sub-view."""
        total = self.TOPK_ROUTED + self.N_SHARED
        # None and 1 are the two spellings of an unscaled shared expert.
        for factor in (None, 1, 0.25):
            with self.subTest(factor=factor):
                self.fake_kernel.calls.clear()
                output = self._fuse(
                    self._config(fused_shared_experts_scaling_factor=factor)
                )

                self.assertEqual(tuple(output.topk_ids.shape), (37, total))
                self.assertEqual(tuple(output.topk_weights.shape), (37, total))
                shared_ids = output.topk_ids[:, self.TOPK_ROUTED :]
                self.assertTrue(torch.all(shared_ids == self.NUM_EXPERTS))
                expected_weight = 1.0 if factor is None else float(factor)
                shared_w = output.topk_weights[:, self.TOPK_ROUTED :]
                self.assertTrue(torch.all(shared_w == expected_weight))
                routed_ids = output.topk_ids[:, : self.TOPK_ROUTED]
                self.assertTrue(torch.all(routed_ids < self.NUM_EXPERTS))

                # The kernel gets a strided view whose row stride spans the
                # shared columns; that stride is what removes the append kernel.
                (call,) = self.fake_kernel.calls
                self.assertEqual(call["weights_shape"], (37, self.TOPK_ROUTED))
                self.assertEqual(call["weights_row_stride"], total)
                self.assertEqual(call["ids_row_stride"], total)
                self.assertEqual(call["weights_ptr"], output.topk_weights.data_ptr())
                self.assertEqual(call["ids_ptr"], output.topk_ids.data_ptr())

    def test_buffer_is_persistent_per_config_and_isolated_across_layers(self):
        """Addresses stay fixed across calls (the CUDA-graph replay
        requirement) and two layers never share one buffer."""
        config = self._config()
        other_layer = self._config()
        first = self._fuse(config, num_tokens=37)
        second = self._fuse(config, num_tokens=64)
        third = self._fuse(other_layer, num_tokens=37)

        self.assertEqual(first.topk_ids.data_ptr(), second.topk_ids.data_ptr())
        self.assertEqual(first.topk_weights.data_ptr(), second.topk_weights.data_ptr())
        self.assertNotEqual(first.topk_ids.data_ptr(), third.topk_ids.data_ptr())
        self.assertNotEqual(
            first.topk_weights.data_ptr(), third.topk_weights.data_ptr()
        )
        self.assertEqual(len(_shared_column_buffers()), 2)

        # The larger batch reuses the same buffer, so the shared columns of the
        # rows it newly exposes must still hold the prefilled values.
        self.assertTrue(torch.all(second.topk_weights[:, self.TOPK_ROUTED :] == 1.0))
        self.assertTrue(
            torch.all(second.topk_ids[:, self.TOPK_ROUTED :] == self.NUM_EXPERTS)
        )

    def test_buffer_height_tracks_the_prefill_budget(self):
        """Height is the larger prefill budget, floored at 8192 and capped; a
        batch taller than the buffer falls back."""
        cap = aiter_topk._MAX_BUFFER_TOKENS
        cases = [
            # (chunked_prefill_size, max_prefill_tokens, expected height)
            (16384, 32768, 32768),
            (2048, 0, 8192),
            (-1, 0, cap),  # chunked prefill disabled, no prefill budget
            (cap * 4, cap * 4, cap),
        ]
        for chunked, max_prefill, height in cases:
            with (
                self.subTest(chunked=chunked, max_prefill=max_prefill),
                get_schedule().override(
                    chunked_prefill_size=chunked, max_prefill_tokens=max_prefill
                ),
            ):
                get_resources().buffers.clear()
                self.assertIsNotNone(self._fuse(self._config(), num_tokens=1))
                ((weights, ids),) = _shared_column_buffers().values()
                self.assertEqual(weights.shape[0], height)
                self.assertEqual(ids.shape[0], height)

        with get_schedule().override(chunked_prefill_size=2048, max_prefill_tokens=0):
            config = self._config()
            self.assertIsNotNone(self._fuse(config, num_tokens=8192))
            self.assertIsNone(self._fuse(config, num_tokens=8193))

    def test_graph_capture_before_first_call_keeps_append_path(self):
        """A capture that would allocate the buffer falls back instead, and the
        buffer is created by the next eager call."""
        config = self._config()
        hidden_states, router_logits = self._inputs(37)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = aiter_topk.try_select_experts(
                hidden_states, router_logits, config
            )
        self.assertIsNone(captured)
        self.assertEqual(_shared_column_buffers(), {})
        self.assertIsNotNone(
            aiter_topk.try_select_experts(hidden_states, router_logits, config)
        )
        self.assertEqual(len(_shared_column_buffers()), 1)

    def test_select_experts_prefers_the_fusion_on_aiter_builds(self):
        """``select_experts`` returns the fused output when the fusion applies,
        and otherwise falls through to the ordinary routing path."""
        hidden_states, router_logits = self._inputs(37)
        config = self._config()
        total = self.TOPK_ROUTED + self.N_SHARED
        with (
            patch.object(topk_module, "_is_hip", True),
            patch.object(topk_module, "_use_aiter", True),
        ):
            fused = topk_module.select_experts(hidden_states, router_logits, config)
            self.assertEqual(len(self.fake_kernel.calls), 1)
            self.assertEqual(tuple(fused.topk_ids.shape), (37, total))

            # Fall-through: the ordinary routing gate is stubbed because the
            # real one resolves to aiter symbols that a non-ROCm build never
            # imported; the shared-expert append after it is backend-neutral.
            routed = (
                torch.rand((37, self.TOPK_ROUTED), device=router_logits.device),
                torch.randint(
                    0,
                    self.NUM_EXPERTS,
                    (37, self.TOPK_ROUTED),
                    dtype=torch.int32,
                    device=router_logits.device,
                ),
            )
            with patch.object(topk_module, "biased_grouped_topk", return_value=routed):
                fallback = topk_module.select_experts(
                    hidden_states,
                    router_logits,
                    config,
                    num_token_non_padded=_full_batch(37, router_logits.device),
                )
            self.assertEqual(len(self.fake_kernel.calls), 1)
            self.assertEqual(tuple(fallback.topk_ids.shape), (37, total))
            self.assertTrue(
                torch.equal(fallback.topk_ids[:, : self.TOPK_ROUTED], routed[1])
            )


if __name__ == "__main__":
    unittest.main()
