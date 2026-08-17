"""Unit tests for the two ways shared experts reach the top-k tensors on GPU.

Covers ``fused_append_remap_shared_experts_deepep``, which collapses
``fused_append_shared_experts()`` followed by
``remap_topk_for_per_rank_shared_slots()`` into a
single Triton launch on the per-rank shared-slot path, and the aiter
grouped-topk fast path, which pre-populates the shared columns of a persistent
buffer instead of appending them per layer. Both are GPU-only, so these tests
are skipped when no accelerator is present.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    fused_append_remap_shared_experts_deepep,
    fused_append_shared_experts,
)
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import (
    TopKConfig,
    _use_aiter,
    biased_grouped_topk_gpu,
    remap_topk_for_per_rank_shared_slots,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")


def _reference_append_remap(
    topk_ids, topk_weights, s, scale_factor, shared_id_base, num_local_routed
):
    """Pure-torch golden reference mirroring the kernel's documented contract.

    Routed IDs:   e -> e + e // num_local_routed
    Shared IDs:   shared_id_base + arange(s)
    Routed wgt:   passthrough
    Shared wgt:   scale_factor
    """
    m, k = topk_ids.shape
    out_ids = torch.empty((m, k + s), dtype=topk_ids.dtype, device=topk_ids.device)
    out_w = torch.empty(
        (m, k + s), dtype=topk_weights.dtype, device=topk_weights.device
    )
    out_ids[:, :k] = topk_ids + topk_ids // num_local_routed
    out_w[:, :k] = topk_weights
    shared = shared_id_base + torch.arange(s, device=topk_ids.device)
    out_ids[:, k:] = shared.to(topk_ids.dtype)
    out_w[:, k:] = scale_factor
    return out_ids, out_w


@unittest.skipUnless(
    torch.cuda.is_available(), "fused append+remap kernel requires a GPU"
)
class TestFusedAppendRemapPerRankSharedSlots(CustomTestCase):
    # (m, k, num_physical_routed, ep_size, ep_rank, num_fused_shared_experts).
    # k and num_fused_shared_experts are kept powers of two (tl.arange constraint).
    CASES = [
        (1, 8, 256, 8, 0, 1),
        (4, 8, 256, 8, 7, 1),
        (17, 8, 264, 8, 3, 1),
        (128, 16, 128, 4, 2, 2),
    ]

    def _make_inputs(self, m, k, num_physical_routed, ids_dtype=torch.int64):
        device = get_device()
        g = torch.Generator(device="cpu").manual_seed(m * 1000 + k * 7 + 1)
        topk_ids = torch.randint(
            0, num_physical_routed, (m, k), generator=g, dtype=ids_dtype
        ).to(device)
        topk_weights = torch.rand((m, k), generator=g, dtype=torch.float32).to(device)
        return topk_ids, topk_weights

    @staticmethod
    def _shared_id_base(num_physical_routed, ep_size, ep_rank, s):
        num_local_routed = num_physical_routed // ep_size
        num_local_experts = num_local_routed + s
        return ep_rank * num_local_experts + num_local_routed, num_local_routed

    def test_matches_golden_reference(self):
        """Kernel output equals the documented routed-remap + shared-append math."""
        for m, k, npr, ep_size, ep_rank, s in self.CASES:
            with self.subTest(m=m, k=k, npr=npr, ep_rank=ep_rank, s=s):
                shared_id_base, num_local_routed = self._shared_id_base(
                    npr, ep_size, ep_rank, s
                )
                scale_factor = 1.0
                topk_ids, topk_weights = self._make_inputs(m, k, npr)

                got_ids, got_w = fused_append_remap_shared_experts_deepep(
                    topk_ids,
                    topk_weights,
                    s,
                    scale_factor,
                    shared_id_base,
                    num_local_routed,
                )
                exp_ids, exp_w = _reference_append_remap(
                    topk_ids,
                    topk_weights,
                    s,
                    scale_factor,
                    shared_id_base,
                    num_local_routed,
                )

                self.assertEqual(tuple(got_ids.shape), (m, k + s))
                self.assertTrue(torch.equal(got_ids, exp_ids))
                self.assertTrue(torch.allclose(got_w, exp_w))

    def test_equivalence_with_eager_append_then_remap(self):
        """Fused kernel == append shared experts + per-rank shared-slot remap.

        The eager remap overwrites the shared weight: 1.0 on the aiter/HIP path
        (routed_scaling_factor is pre-folded into the routed topk weights), else
        1/routed_scaling_factor. The fused kernel is invoked with that same value
        so the two paths stay bit-identical (ids match regardless of scaling).
        """
        rsf = 2.5
        scale_factor = 1.0 if _use_aiter else 1.0 / rsf
        for m, k, npr, ep_size, ep_rank, s in self.CASES:
            with self.subTest(m=m, k=k, npr=npr, ep_rank=ep_rank, s=s):
                shared_id_base, num_local_routed = self._shared_id_base(
                    npr, ep_size, ep_rank, s
                )
                topk_ids, topk_weights = self._make_inputs(m, k, npr)

                fused_ids, fused_w = fused_append_remap_shared_experts_deepep(
                    topk_ids.clone(),
                    topk_weights.clone(),
                    s,
                    scale_factor,
                    shared_id_base,
                    num_local_routed,
                )

                with get_parallel().override(moe_ep_size=ep_size, moe_ep_rank=ep_rank):
                    eager_ids, eager_w = fused_append_shared_experts(
                        topk_ids.clone(),
                        topk_weights.clone(),
                        s,
                        scale_factor,
                        npr,  # shared-expert base id (overwritten by the remap)
                    )
                    eager_ids, eager_w = remap_topk_for_per_rank_shared_slots(
                        eager_ids,
                        eager_w,
                        s,
                        npr,
                        TopKConfig(
                            top_k=k,
                            num_fused_shared_experts=s,
                            routed_scaling_factor=rsf,
                        ),
                    )

                self.assertTrue(torch.equal(fused_ids, eager_ids))
                self.assertTrue(torch.allclose(fused_w, eager_w))

    def test_shared_weight_is_one_on_aiter_path(self):
        """On the aiter path the always-on shared expert must contribute 1.0x."""
        m, k, npr, ep_size, ep_rank, s = 8, 8, 256, 8, 1, 1
        shared_id_base, num_local_routed = self._shared_id_base(
            npr, ep_size, ep_rank, s
        )
        topk_ids, topk_weights = self._make_inputs(m, k, npr)

        _, got_w = fused_append_remap_shared_experts_deepep(
            topk_ids, topk_weights, s, 1.0, shared_id_base, num_local_routed
        )
        self.assertTrue(torch.all(got_w[:, -s:] == 1.0))

    def test_no_shared_experts_is_noop(self):
        """s == 0 returns the inputs untouched (no kernel launch)."""
        topk_ids, topk_weights = self._make_inputs(4, 8, 256)
        got_ids, got_w = fused_append_remap_shared_experts_deepep(
            topk_ids, topk_weights, 0, 1.0, 0, 32
        )
        self.assertTrue(torch.equal(got_ids, topk_ids))
        self.assertTrue(torch.equal(got_w, topk_weights))


@unittest.skipUnless(
    torch.cuda.is_available() and _use_aiter,
    "the persistent shared-column buffer only exists on the aiter grouped-topk path",
)
class TestAiterGroupedTopkSharedFuse(CustomTestCase):
    """The pre-populated buffer must return what the plain append would have.

    ``biased_grouped_topk_gpu`` fills the shared columns of a persistent buffer
    once and lets aiter write only the routed columns, so no per-layer append
    kernel runs. When a batch is larger than the buffer it returns the routed
    columns alone and ``select_experts`` appends the shared experts itself. The
    two spellings are supposed to be bit-identical; that is what is asserted
    here, since a divergence would silently change routing weights.
    """

    # (num_tokens, num_experts, num_expert_group, topk_group, topk_routed,
    #  n_shared, routed_scaling_factor) on the DeepSeek-V3 / GLM-5 routing shape:
    # a decode step, a ragged prefill chunk, and two larger chunks. aiter folds
    # the routed scaling into the routed weights, so each size carries a
    # different one -- unset, unscaled, and the two values models configure.
    CASES = [
        (1, 256, 8, 4, 8, 1, 2.5),
        (37, 256, 8, 4, 8, 1, None),
        (128, 256, 8, 4, 8, 1, 1.0),
        (512, 256, 8, 4, 8, 1, 1.5),
    ]

    def setUp(self):
        super().setUp()
        # The buffer and the token budget are module-level caches; a stale entry
        # from another test would decide this one's path.
        topk_module._aiter_topk_fuse_shared_bufs.clear()
        topk_module._aiter_topk_fuse_shared_max_tokens_cache = None

    def _make_inputs(self, num_tokens, num_experts):
        device = get_device()
        g = torch.Generator(device="cpu").manual_seed(num_tokens * 31 + num_experts)
        hidden_states = torch.randn(
            (num_tokens, 16), generator=g, dtype=torch.bfloat16
        ).to(device)
        gating_output = torch.randn(
            (num_tokens, num_experts), generator=g, dtype=torch.bfloat16
        ).to(device)
        correction_bias = torch.randn(
            (num_experts,), generator=g, dtype=torch.float32
        ).to(device)
        return hidden_states, gating_output, correction_bias

    def _topk(
        self, inputs, num_expert_group, topk_group, topk_routed, n_shared, rsf, factor
    ):
        hidden_states, gating_output, correction_bias = inputs
        return biased_grouped_topk_gpu(
            hidden_states,
            gating_output,
            correction_bias,
            topk_routed,
            renormalize=True,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            num_fused_shared_experts=n_shared,
            routed_scaling_factor=rsf,
            apply_routed_scaling_factor_on_output=False,
            fused_shared_experts_scaling_factor=factor,
        )

    def test_prepopulated_buffer_matches_plain_append(self):
        # None and 1 are the two spellings models use for an unscaled shared
        # expert; both must land on the same weight.
        for factor in (None, 1):
            for case in self.CASES:
                num_tokens, num_experts, groups, topk_group, topk_routed, s, rsf = case
                with self.subTest(num_tokens=num_tokens, rsf=rsf, factor=factor):
                    inputs = self._make_inputs(num_tokens, num_experts)

                    with get_parallel().override(moe_ep_size=1):
                        fused_w, fused_ids = self._topk(
                            inputs, groups, topk_group, topk_routed, s, rsf, factor
                        )
                        # The return aliases the persistent buffer, which the
                        # next case overwrites.
                        fused_w, fused_ids = fused_w.clone(), fused_ids.clone()

                        # A zero budget puts every batch over the buffer, which
                        # is exactly the fallback select_experts appends to.
                        with patch.object(
                            topk_module,
                            "_get_aiter_topk_fuse_shared_max_tokens",
                            return_value=0,
                        ):
                            plain_w, plain_ids = self._topk(
                                inputs, groups, topk_group, topk_routed, s, rsf, factor
                            )

                    self.assertEqual(tuple(plain_ids.shape), (num_tokens, topk_routed))
                    plain_ids, plain_w = fused_append_shared_experts(
                        plain_ids,
                        plain_w,
                        s,
                        1.0 if factor is None else factor,
                        num_experts,  # shared-expert base id
                    )

                    self.assertEqual(
                        tuple(fused_ids.shape), (num_tokens, topk_routed + s)
                    )
                    self.assertTrue(torch.equal(fused_ids, plain_ids))
                    self.assertTrue(torch.equal(fused_w, plain_w))

    def test_expert_parallelism_takes_the_plain_path(self):
        """Shared columns are only local to a rank when EP is off, so ep_size > 1
        must not fuse -- select_experts mirrors this check to decide whether to
        append."""
        num_tokens, num_experts, groups, topk_group, topk_routed, s, rsf = self.CASES[1]
        inputs = self._make_inputs(num_tokens, num_experts)
        with get_parallel().override(moe_ep_size=2):
            _, topk_ids = self._topk(inputs, groups, topk_group, topk_routed, s, rsf, 1)
        self.assertEqual(tuple(topk_ids.shape), (num_tokens, topk_routed))

    def test_uninitialized_expert_parallel_group_falls_back(self):
        """Reaching this path before the MoE EP group exists must drop to the
        plain append rather than raise out of the parallel-state accessor."""
        num_tokens, num_experts, groups, topk_group, topk_routed, s, rsf = self.CASES[0]
        inputs = self._make_inputs(num_tokens, num_experts)
        with patch.object(
            topk_module,
            "get_parallel",
            side_effect=AssertionError(
                "expert model parallel group is not initialized"
            ),
        ):
            _, topk_ids = self._topk(inputs, groups, topk_group, topk_routed, s, rsf, 1)
        self.assertEqual(tuple(topk_ids.shape), (num_tokens, topk_routed))


if __name__ == "__main__":
    unittest.main()
