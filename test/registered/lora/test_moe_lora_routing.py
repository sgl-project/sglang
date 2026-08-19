"""Correctness tests for SGL-LoRA virtual-expert routing."""

import unittest

import torch

from sglang.srt.lora.moe.routing import (
    RouteViewKind,
    build_virtual_expert_routing,
)
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase


def _serial_materialized_reference():
    """The simplest correct pipeline: every stage standalone, nothing fused,
    no overlap window."""
    from sglang.srt.lora.moe.execution_plan import (
        ActivationFn,
        BridgeLayout,
        FinalizeFamily,
        FinalizeSpec,
        LoraAFamily,
        LoraASpec,
        LoraBFamily,
        LoraBSpec,
        MiddleFamily,
        MiddleSpec,
        MoeLoraExecutionPlan,
        Site,
    )

    return MoeLoraExecutionPlan(
        gate_up_a=LoraASpec(
            Site.GATE_UP, LoraAFamily.GROUPED, False, BridgeLayout.PAIR_MAJOR
        ),
        gate_up_b=LoraBSpec(
            Site.GATE_UP,
            LoraBFamily.ONE_LAUNCH_SLICED,
            False,
            BridgeLayout.PAIR_MAJOR,
        ),
        middle=MiddleSpec(MiddleFamily.MATERIALIZED, ActivationFn.SILU),
        down_a=LoraASpec(
            Site.DOWN, LoraAFamily.GROUPED, False, BridgeLayout.PAIR_MAJOR
        ),
        down_b=LoraBSpec(
            Site.DOWN, LoraBFamily.ONE_LAUNCH_SLICED, False, BridgeLayout.PAIR_MAJOR
        ),
        finalize=FinalizeSpec(FinalizeFamily.MATERIALIZED),
    )


register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-small")


class TestMoeLoraRouting(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda:0"

    def _build(
        self,
        topk_ids,
        adapters,
        *,
        lora_experts_per_adapter,
        max_loras=2,
        lora_expert_map=None,
        block_size=16,
        dtype=torch.int32,
        view=RouteViewKind.ALIGNED,
    ):
        # A real workspace, since the fused builder requires one.
        return build_virtual_expert_routing(
            torch.tensor(topk_ids, dtype=dtype, device=self.device),
            torch.tensor(adapters, dtype=dtype, device=self.device),
            lora_experts_per_adapter=lora_experts_per_adapter,
            max_loras=max_loras,
            block_size=block_size,
            lora_expert_map=(
                None
                if lora_expert_map is None
                else torch.tensor(lora_expert_map, dtype=dtype, device=self.device)
            ),
            view=view,
            workspace=MoeLoraWorkspace(),
            scratch_prefix="test:route",
        )

    def test_narrower_views_refuse_fields_they_did_not_build(self):
        """A view must not silently hand back a field it never computed.

        The three views exist so a schedule pays only for what it reads (plan
        section 29 R1). If an unbuilt field returned None instead of raising,
        a consumer that requested the wrong view would pass None into a Triton
        launch and fail somewhere unrelated -- or, worse, index a stale buffer.
        """
        ids, adapters = [[0, 1]], [0]
        aligned = self._build(
            ids, adapters, lora_experts_per_adapter=2, view=RouteViewKind.ALIGNED
        )
        self.assertEqual(aligned.virtual_topk_ids.numel(), 2)
        self.assertGreater(aligned.sorted_pair_ids.numel(), 0)

        fused = self._build(
            ids, adapters, lora_experts_per_adapter=2, view=RouteViewKind.FUSED_IDS
        )
        self.assertTrue(
            torch.equal(fused.virtual_topk_ids, aligned.virtual_topk_ids),
            "fused_ids must agree bitwise with the aligned view it is a prefix of",
        )
        for field in ("sorted_pair_ids", "block_virtual_expert_ids"):
            with self.assertRaisesRegex(ValueError, RouteViewKind.ALIGNED):
                getattr(fused, field)

        raw = self._build(
            ids, adapters, lora_experts_per_adapter=2, view=RouteViewKind.RAW
        )
        with self.assertRaisesRegex(ValueError, RouteViewKind.FUSED_IDS):
            raw.virtual_topk_ids
        with self.assertRaisesRegex(ValueError, RouteViewKind.ALIGNED):
            raw.sorted_pair_ids
        # A raw consumer fuses the key computation into its own kernel, so the
        # sources must survive on the view.
        self.assertEqual(raw.lora_experts_per_adapter, 2)
        self.assertEqual(raw.token_lora_mapping.numel(), 1)

    def test_fused_aligned_build_without_a_workspace_is_rejected(self):
        """The alternative is the builder's global cache, which clobbers."""
        from sglang.srt.lora.moe.routing import FUSED_ALIGN_MIN_VIRTUAL_EXPERTS

        experts = FUSED_ALIGN_MIN_VIRTUAL_EXPERTS // 32
        ids = torch.zeros((8, 8), dtype=torch.int32, device=self.device)
        slots = torch.zeros(8, dtype=torch.int32, device=self.device)
        for half in ({}, {"workspace": MoeLoraWorkspace()}, {"scratch_prefix": "x"}):
            with self.subTest(**{k: bool(v) for k, v in half.items()}):
                with self.assertRaisesRegex(ValueError, "needs workspace"):
                    build_virtual_expert_routing(
                        ids,
                        slots,
                        lora_experts_per_adapter=experts,
                        max_loras=32,
                        block_size=16,
                        **half,
                    )

    def test_unknown_view_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown route view"):
            self._build([[0, 1]], [0], lora_experts_per_adapter=2, view="grouped")

    def test_identity_and_explicit_lora_expert_maps(self):
        cases = (
            ("local_identity", [[0, 1], [2, 3]], None, 4, [[0, 1], [6, 7]]),
            (
                "global_owned",
                [[0, 1], [2, 3]],
                [0, -1, 2, 3],
                4,
                [[0, -1], [6, 7]],
            ),
            (
                "global_to_local",
                [[4, 5], [6, 7]],
                [-1, -1, -1, -1, 0, 1, 2, 3],
                4,
                [[0, 1], [6, 7]],
            ),
            (
                "local_to_offset_factor",
                [[0, 1], [2, 3]],
                [4, 5, 6, 7],
                8,
                [[4, 5], [14, 15]],
            ),
        )
        for (
            name,
            topk_ids,
            lora_expert_map,
            lora_experts_per_adapter,
            expected,
        ) in cases:
            with self.subTest(name=name):
                route = self._build(
                    topk_ids,
                    [0, 1],
                    lora_experts_per_adapter=lora_experts_per_adapter,
                    lora_expert_map=lora_expert_map,
                )
                self.assertEqual(route.virtual_topk_ids.cpu().tolist(), expected)

    def test_invalid_adapter_expert_and_map_ids_become_one_sentinel(self):
        route = self._build(
            [[-2], [-1], [3], [4], [99], [0], [0]],
            [0, 0, 0, 0, 0, 2, 3],
            lora_experts_per_adapter=4,
        )
        self.assertEqual(
            route.virtual_topk_ids.flatten().cpu().tolist(),
            [-1, -1, 3, -1, -1, -1, -1],
        )

        mapped = self._build(
            [[0, 1, 2, 3]],
            [0],
            lora_experts_per_adapter=3,
            lora_expert_map=[0, -1, 3, 99],
        )
        self.assertEqual(mapped.virtual_topk_ids.cpu().tolist(), [[0, -1, -1, -1]])

        live_blocks = route.num_pairs_post_padded.item() // route.block_size
        live_ids = route.block_virtual_expert_ids[:live_blocks]
        self.assertTrue(
            bool(
                (
                    (live_ids == -1)
                    | ((live_ids >= 0) & (live_ids < route.num_virtual_experts))
                )
                .all()
                .item()
            )
        )

    def test_int64_ids_preserve_dtype(self):
        route = self._build(
            [[0, 3], [4, -2]],
            [0, 1],
            lora_experts_per_adapter=4,
            lora_expert_map=[0, 1, 2, 3, -1],
            dtype=torch.int64,
        )
        self.assertEqual(route.virtual_topk_ids.cpu().tolist(), [[0, 3], [-1, -1]])
        self.assertEqual(route.virtual_topk_ids.dtype, torch.int64)

    def test_sentinel_bucket_is_included_in_capacity(self):
        route = self._build(
            [[0], [1], [2], [3], [0], [1], [2], [3], [-1]],
            [0, 0, 0, 0, 1, 1, 1, 1, 0],
            lora_experts_per_adapter=4,
            block_size=4,
        )
        self.assertEqual(route.num_pairs_post_padded.item(), 9 * 4)
        self.assertGreaterEqual(route.sorted_pair_ids.numel(), 9 * 4)
        self.assertGreaterEqual(route.block_virtual_expert_ids.numel(), 9)

    def test_aligned_view_policy_boundaries(self):
        """Pin the two-predicate align policy below, at, and above each edge.

        Sited by the section 40 redo (plan section 13 rule 5): fused when
        V >= 8192 (the JIT kernel's EPT 8->16 rung edge) OR P >= 16384 (fused
        wins that column at every V, both timing modes). The policy constants
        are load-bearing serving behavior; a silent change is a performance
        regression no correctness test would catch — that exact failure
        happened once, when a raised kernel ceiling never reached the
        dispatch — so the values AND the edge behavior are pinned together.
        """
        from sglang.srt.lora.moe import fused_align
        from sglang.srt.lora.moe.routing import (
            _JIT_ALIGN_MAX_VIRTUAL_EXPERTS,
            FUSED_ALIGN_MIN_PAIRS,
            FUSED_ALIGN_MIN_VIRTUAL_EXPERTS,
            RouteViewKind,
            build_virtual_expert_routing,
        )

        self.assertEqual(FUSED_ALIGN_MIN_VIRTUAL_EXPERTS, 8192)
        self.assertEqual(FUSED_ALIGN_MIN_PAIRS, 16384)
        # The shared JIT align primitive covers 8191 real buckets; the fused
        # builder takes over exactly at the 8192 dispatch edge, so the two
        # constants must stay adjacent with no gap.
        self.assertEqual(_JIT_ALIGN_MAX_VIRTUAL_EXPERTS, 8191)
        self.assertEqual(
            FUSED_ALIGN_MIN_VIRTUAL_EXPERTS, _JIT_ALIGN_MAX_VIRTUAL_EXPERTS + 1
        )

        # (V, T, expects_fused): straddles both edges; K = 8 so P = 8 * T.
        cases = (
            (8160, 8, False),  # below both edges -> ID pass + JIT
            (8192, 8, True),  # at the V edge (the EPT rung)
            (12288, 8, True),  # kimi EP1 x 32 slots, the realistic large case
            (1024, 2048, True),  # small V, P = 16384: the P edge
            (1024, 1024, False),  # small V, P = 8192: below the P edge
            (40960, 8, True),  # above the JIT ceiling: fused is the only path
        )
        original = fused_align.fused_align_block_size
        for num_virtual, num_tokens, expects_fused in cases:
            lora_experts_per_adapter = num_virtual // 32
            with self.subTest(V=num_virtual, P=num_tokens * 8):
                ids = torch.randint(
                    0,
                    lora_experts_per_adapter,
                    (num_tokens, 8),
                    dtype=torch.int32,
                    device=self.device,
                )
                slots = torch.randint(
                    0, 32, (num_tokens,), dtype=torch.int32, device=self.device
                )
                calls: list[bool] = []
                try:

                    def spy(*args, **kwargs):
                        calls.append(True)
                        return original(*args, **kwargs)

                    fused_align.fused_align_block_size = spy
                    route = build_virtual_expert_routing(
                        ids,
                        slots,
                        lora_experts_per_adapter=lora_experts_per_adapter,
                        max_loras=32,
                        block_size=16,
                        view=RouteViewKind.ALIGNED,
                        workspace=MoeLoraWorkspace(),
                        scratch_prefix="test:route",
                    )
                finally:
                    fused_align.fused_align_block_size = original
                self.assertEqual(
                    bool(calls),
                    expects_fused,
                    f"V={num_virtual}, P={num_tokens * 8} took the wrong path",
                )
                if expects_fused:
                    self.assertEqual(len(calls), 1)
                self.assertGreater(int(route.num_pairs_post_padded), 0)
                self.assertEqual(route.sorted_pair_ids.dtype, torch.int32)
                self.assertEqual(route.block_virtual_expert_ids.dtype, torch.int32)

    def test_sentinel_blocks_isolate_invalid_pairs_on_both_align_paths(self):
        """A5 pin-down (gate-1 packet): the sentinel-bucket contract, black-box.

        Invalid pairs (base tokens, non-owned experts) ride sentinel routes.
        The contract every LoRA kernel depends on: a block labelled ``-1``
        never contains a valid pair, a valid pair never sits in a ``-1``
        block, every slot inside the padded plan is a READABLE index (< P or
        the padding value P — the stock B kernel dereferences ``-1`` blocks'
        slots to zero-fill, which is how an uninitialized tail once caused an
        illegal memory access), and every valid pair appears exactly once
        under its own key's label. Checked through the dispatch on BOTH
        policy paths — V=256 (ID pass + JIT) and V=12288 (fused) — since the
        two implement the contract independently.
        """
        for lora_experts_per_adapter, max_loras in ((8, 32), (384, 32)):
            num_virtual = lora_experts_per_adapter * max_loras
            with self.subTest(V=num_virtual):
                generator = torch.Generator(device="cpu").manual_seed(23)
                num_tokens, top_k = 96, 8
                topk_ids = torch.randint(
                    -1,
                    lora_experts_per_adapter,
                    (num_tokens, top_k),
                    generator=generator,
                    dtype=torch.int32,
                ).to(self.device)
                token_lora_mapping = torch.randint(
                    -1,
                    max_loras,
                    (num_tokens,),
                    generator=generator,
                    dtype=torch.int32,
                ).to(self.device)
                route = build_virtual_expert_routing(
                    topk_ids,
                    token_lora_mapping,
                    lora_experts_per_adapter=lora_experts_per_adapter,
                    max_loras=max_loras,
                    block_size=16,
                    view=RouteViewKind.ALIGNED,
                    workspace=MoeLoraWorkspace(),
                    scratch_prefix="test:route",
                )
                num_pairs = num_tokens * top_k
                keys = (
                    torch.where(
                        (token_lora_mapping[:, None] >= 0) & (topk_ids >= 0),
                        token_lora_mapping[:, None].to(torch.int64)
                        * lora_experts_per_adapter
                        + topk_ids.to(torch.int64),
                        torch.tensor(-1, dtype=torch.int64, device=self.device),
                    )
                    .reshape(-1)
                    .cpu()
                )
                num_padded = int(route.num_pairs_post_padded)
                sorted_ids = route.sorted_pair_ids.cpu()
                block_ids = route.block_virtual_expert_ids.cpu()
                self.assertTrue(bool((keys == -1).any()), "case must have sentinels")

                seen: dict[int, int] = {}
                for block in range(num_padded // 16):
                    label = int(block_ids[block])
                    slots = sorted_ids[block * 16 : (block + 1) * 16]
                    self.assertTrue(
                        bool((slots <= num_pairs).all()),
                        f"block {block} holds an unreadable slot index",
                    )
                    real = slots[slots < num_pairs]
                    for pair in real.tolist():
                        self.assertNotIn(pair, seen, "pair appears twice in the plan")
                        seen[pair] = label
                        if label == -1:
                            self.assertEqual(
                                int(keys[pair]),
                                -1,
                                f"valid pair {pair} placed in a sentinel block",
                            )
                        else:
                            self.assertEqual(
                                int(keys[pair]),
                                label,
                                f"pair {pair} in block labelled {label}",
                            )
                valid = {i for i in range(num_pairs) if int(keys[i]) >= 0}
                self.assertEqual(
                    valid,
                    {p for p, l in seen.items() if l != -1},
                    "every valid pair must appear exactly once under its key",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
