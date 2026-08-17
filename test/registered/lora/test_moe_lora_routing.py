"""Correctness tests for canonical SGL-LoRA virtual-expert routing."""

import unittest
from unittest import mock

import torch

from sglang.srt.lora.moe.routing import (
    ROUTE_ALIGNED,
    ROUTE_FUSED_IDS,
    ROUTE_RAW,
    build_virtual_expert_routing,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase


def _serial_materialized_reference():
    """The simplest correct pipeline: every stage standalone, nothing fused,
    no overlap window."""
    from sglang.srt.lora.moe.execution_plan import (
        ActivationFamily,
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
        middle=MiddleSpec(MiddleFamily.MATERIALIZED, ActivationFamily.SWIGLU),
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
        view=ROUTE_ALIGNED,
        use_pdl=None,
    ):
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
            use_pdl=use_pdl,
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
            ids, adapters, lora_experts_per_adapter=2, view=ROUTE_ALIGNED
        )
        self.assertEqual(aligned.virtual_topk_ids.numel(), 2)
        self.assertGreater(aligned.sorted_pair_ids.numel(), 0)

        fused = self._build(
            ids, adapters, lora_experts_per_adapter=2, view=ROUTE_FUSED_IDS
        )
        self.assertTrue(
            torch.equal(fused.virtual_topk_ids, aligned.virtual_topk_ids),
            "fused_ids must agree bitwise with the aligned view it is a prefix of",
        )
        for field in ("sorted_pair_ids", "block_virtual_expert_ids"):
            with self.assertRaisesRegex(ValueError, ROUTE_ALIGNED):
                getattr(fused, field)

        raw = self._build(ids, adapters, lora_experts_per_adapter=2, view=ROUTE_RAW)
        with self.assertRaisesRegex(ValueError, ROUTE_FUSED_IDS):
            raw.virtual_topk_ids
        with self.assertRaisesRegex(ValueError, ROUTE_ALIGNED):
            raw.sorted_pair_ids
        # A raw consumer fuses the key computation into its own kernel, so the
        # sources must survive on the view.
        self.assertEqual(raw.lora_experts_per_adapter, 2)
        self.assertEqual(raw.token_slots.numel(), 1)

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
            build_virtual_expert_routing,
        )
        from sglang.srt.lora.moe.routing_shape import (
            FUSED_ALIGN_MIN_PAIRS,
            FUSED_ALIGN_MIN_VIRTUAL_EXPERTS,
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
                        calls.append(kwargs["use_pdl"])
                        return original(*args, **kwargs)

                    fused_align.fused_align_block_size = spy
                    route = build_virtual_expert_routing(
                        ids,
                        slots,
                        lora_experts_per_adapter=lora_experts_per_adapter,
                        max_loras=32,
                        block_size=16,
                        view=ROUTE_ALIGNED,
                    )
                finally:
                    fused_align.fused_align_block_size = original
                self.assertEqual(
                    bool(calls),
                    expects_fused,
                    f"V={num_virtual}, P={num_tokens * 8} took the wrong path",
                )
                if expects_fused:
                    self.assertEqual(
                        calls,
                        [False],
                        "standard fused alignment must default PDL off",
                    )
                self.assertGreater(int(route.num_pairs_post_padded), 0)
                self.assertEqual(route.sorted_pair_ids.dtype, torch.int32)
                self.assertEqual(route.block_virtual_expert_ids.dtype, torch.int32)

    def test_explicit_standard_pdl_reaches_only_the_fused_align_builder(self):
        from sglang.srt.lora.moe import fused_align

        original = fused_align.fused_align_block_size
        for num_virtual, expects_fused in ((8160, False), (8192, True)):
            lora_experts_per_adapter = num_virtual // 32
            ids = torch.randint(
                0,
                lora_experts_per_adapter,
                (8, 8),
                dtype=torch.int32,
                device=self.device,
            )
            slots = torch.randint(
                0,
                32,
                (8,),
                dtype=torch.int32,
                device=self.device,
            )
            calls = []
            try:

                def spy(*args, **kwargs):
                    calls.append(kwargs["use_pdl"])
                    return original(*args, **kwargs)

                fused_align.fused_align_block_size = spy
                build_virtual_expert_routing(
                    ids,
                    slots,
                    lora_experts_per_adapter=lora_experts_per_adapter,
                    max_loras=32,
                    block_size=16,
                    view=ROUTE_ALIGNED,
                    use_pdl=True,
                )
            finally:
                fused_align.fused_align_block_size = original
            self.assertEqual(calls, [True] if expects_fused else [])

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
        for lora_experts_per_adapter, slot_capacity in ((8, 32), (384, 32)):
            num_virtual = lora_experts_per_adapter * slot_capacity
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
                token_slots = torch.randint(
                    -1,
                    slot_capacity,
                    (num_tokens,),
                    generator=generator,
                    dtype=torch.int32,
                ).to(self.device)
                route = build_virtual_expert_routing(
                    topk_ids,
                    token_slots,
                    lora_experts_per_adapter=lora_experts_per_adapter,
                    max_loras=slot_capacity,
                    block_size=16,
                    view=ROUTE_ALIGNED,
                )
                num_pairs = num_tokens * top_k
                keys = (
                    torch.where(
                        (token_slots[:, None] >= 0) & (topk_ids >= 0),
                        token_slots[:, None].to(torch.int64) * lora_experts_per_adapter
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


class TestMoeLoraDualGranularityRoutes(CustomTestCase):
    """The single-pass dual-granularity builder against standalone builds.

    Identity is asserted plan-for-plan: the padded pair count and the FULL
    block-label table must be bitwise-identical to a standalone fused build
    at the same granularity, and the sorted-pair plan must be identical
    modulo intra-bucket order — the one degree of freedom the route contract
    explicitly leaves nondeterministic (atomic slot claiming; two standalone
    runs differ the same way).
    """

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda:0"

    def _scratch(self, num_buckets: int):
        from sglang.srt.lora.moe.routing import FusedAlignScratch

        return FusedAlignScratch(
            counts=torch.zeros(num_buckets, dtype=torch.int32, device=self.device),
            block_cumulative=torch.empty(
                num_buckets + 1, dtype=torch.int32, device=self.device
            ),
            cursor=torch.empty(num_buckets, dtype=torch.int32, device=self.device),
            bucket_end=torch.empty(num_buckets, dtype=torch.int32, device=self.device),
        )

    def _standalone_plan(
        self,
        topk_ids,
        token_slots,
        *,
        lora_experts_per_adapter,
        max_loras,
        block_size,
    ):
        """One standalone fused aligned build, exactly as `_aligned_pair_route`
        launches it on the fused dispatch path."""
        from sglang.srt.lora.moe.fused_align import fused_align_block_size
        from sglang.srt.lora.moe.routing import _routing_capacity

        num_virtual = lora_experts_per_adapter * max_loras
        return fused_align_block_size(
            topk_ids,
            token_slots,
            lora_experts_per_adapter=lora_experts_per_adapter,
            max_loras=max_loras,
            block_size=block_size,
            capacity=_routing_capacity(topk_ids.numel(), block_size, num_virtual),
            num_pairs_post_padded_out=torch.empty(
                1, dtype=torch.int32, device=self.device
            ),
            scratch=self._scratch(num_virtual + 1),
        )

    def _dual(
        self,
        topk_ids,
        token_slots,
        *,
        lora_experts_per_adapter,
        max_loras,
        block_sizes,
        use_pdl=None,
    ):
        from sglang.srt.lora.moe.routing import (
            build_dual_granularity_aligned_routes,
        )

        num_buckets = lora_experts_per_adapter * max_loras + 1
        scratches = (self._scratch(num_buckets), self._scratch(num_buckets))
        views = build_dual_granularity_aligned_routes(
            topk_ids,
            token_slots,
            lora_experts_per_adapter=lora_experts_per_adapter,
            max_loras=max_loras,
            block_sizes=block_sizes,
            num_pairs_post_padded_outs=(
                torch.empty(1, dtype=torch.int32, device=self.device),
                torch.empty(1, dtype=torch.int32, device=self.device),
            ),
            scratches=scratches,
            use_pdl=use_pdl,
        )
        return views, scratches

    @staticmethod
    def _canonical(sorted_pair_ids, block_ids, num_pairs_post_padded, block_size):
        """Plan identity modulo intra-bucket scatter order.

        Consecutive equal block labels form one bucket's region (labels are
        the strictly increasing virtual-expert ids plus the trailing -1
        sentinel run); sorting the slots inside each region canonicalizes the
        atomic-cursor order while keeping padding (= num_pairs) at the tail.
        """
        padded = int(num_pairs_post_padded.item())
        labels = block_ids.cpu().tolist()
        plan = sorted_pair_ids[:padded].cpu()
        blocks = padded // block_size
        segments = []
        start = 0
        for index in range(1, blocks + 1):
            if index == blocks or labels[index] != labels[start]:
                segment = plan[start * block_size : index * block_size]
                segments.append((labels[start], segment.sort().values.tolist()))
                start = index
        return padded, labels, segments

    def _assert_view_matches(self, view, reference, block_size):
        ref_sorted, ref_blocks, ref_padded = reference
        self.assertEqual(view.block_size, block_size)
        self.assertEqual(int(view.num_pairs_post_padded.item()), int(ref_padded.item()))
        self.assertEqual(
            view.block_virtual_expert_ids.cpu().tolist(),
            ref_blocks.cpu().tolist(),
            "block labels are deterministic and must match bitwise",
        )
        self.assertEqual(
            self._canonical(
                view.sorted_pair_ids,
                view.block_virtual_expert_ids,
                view.num_pairs_post_padded,
                block_size,
            ),
            self._canonical(ref_sorted, ref_blocks, ref_padded, block_size),
        )

    def _iid(self, num_tokens, top_k, expert_hi, slot_hi, *, seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        topk_ids = torch.randint(
            -1, expert_hi, (num_tokens, top_k), generator=generator, dtype=torch.int32
        ).to(self.device)
        token_slots = torch.randint(
            -1, slot_hi, (num_tokens,), generator=generator, dtype=torch.int32
        ).to(self.device)
        return topk_ids, token_slots

    def test_dual_views_match_standalone_fused_builds(self):
        lora_experts_per_adapter, max_loras = 8, 32  # V = 256
        cases = {
            # iid with -1 sentinels in both the expert and adapter columns.
            "iid_sentinels": self._iid(
                96, 8, lora_experts_per_adapter, max_loras, seed=7
            ),
            # Every pair on one bucket: all other experts empty, maximal skew.
            "hotset_empty_experts": (
                torch.zeros((96, 8), dtype=torch.int32, device=self.device),
                torch.zeros(96, dtype=torch.int32, device=self.device),
            ),
            # No valid pair anywhere: the whole batch rides the sentinel bucket.
            "all_sentinel": (
                torch.randint(-1, 8, (64, 8), dtype=torch.int32, device=self.device),
                torch.full((64,), -1, dtype=torch.int32, device=self.device),
            ),
            # Empty pair domain.
            "empty": (
                torch.empty((0, 8), dtype=torch.int32, device=self.device),
                torch.empty((0,), dtype=torch.int32, device=self.device),
            ),
            # The GB300 dispatch-edge pair count (P = 16384).
            "large_p": self._iid(2048, 8, lora_experts_per_adapter, max_loras, seed=11),
        }

        for name, (topk_ids, token_slots) in cases.items():
            for block_sizes in ((16, 64), (16, 16)):
                with self.subTest(case=name, block_sizes=block_sizes):
                    views, scratches = self._dual(
                        topk_ids,
                        token_slots,
                        lora_experts_per_adapter=lora_experts_per_adapter,
                        max_loras=max_loras,
                        block_sizes=block_sizes,
                    )
                    for view, block_size in zip(views, block_sizes):
                        self._assert_view_matches(
                            view,
                            self._standalone_plan(
                                topk_ids,
                                token_slots,
                                lora_experts_per_adapter=lora_experts_per_adapter,
                                max_loras=max_loras,
                                block_size=block_size,
                            ),
                            block_size,
                        )
                    for scratch in scratches:
                        self.assertEqual(
                            int(scratch.counts.abs().sum().item()),
                            0,
                            "the scan must restore the zero-counts invariant",
                        )

    def test_dual_views_match_above_the_jit_ceiling(self):
        """V = 12288: the regime where fused is the only CUDA-speed builder."""
        lora_experts_per_adapter, max_loras = 384, 32
        topk_ids, token_slots = self._iid(
            96, 8, lora_experts_per_adapter, max_loras, seed=23
        )
        views, _ = self._dual(
            topk_ids,
            token_slots,
            lora_experts_per_adapter=lora_experts_per_adapter,
            max_loras=max_loras,
            block_sizes=(16, 64),
        )
        for view, block_size in zip(views, (16, 64)):
            self._assert_view_matches(
                view,
                self._standalone_plan(
                    topk_ids,
                    token_slots,
                    lora_experts_per_adapter=lora_experts_per_adapter,
                    max_loras=max_loras,
                    block_size=block_size,
                ),
                block_size,
            )

    def test_dual_pdl_chain_matches_pdl_off(self):
        topk_ids, token_slots = self._iid(96, 8, 8, 32, seed=31)
        plans = {}
        for enabled in (False, True):
            views, _ = self._dual(
                topk_ids,
                token_slots,
                lora_experts_per_adapter=8,
                max_loras=32,
                block_sizes=(16, 64),
                use_pdl=enabled,
            )
            plans[enabled] = [
                self._canonical(
                    view.sorted_pair_ids,
                    view.block_virtual_expert_ids,
                    view.num_pairs_post_padded,
                    block_size,
                )
                for view, block_size in zip(views, (16, 64))
            ]
        self.assertEqual(plans[False], plans[True])

    def test_build_routes_takes_one_dual_pass_at_fused_shapes(self):
        """End-to-end wiring: one dual pass replaces both standalone builds."""
        from sglang.srt.lora.moe import route_factory
        from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

        lora_experts_per_adapter, max_loras = 8, 32
        topk_ids, token_slots = self._iid(
            2048, 8, lora_experts_per_adapter, max_loras, seed=43
        )
        original = route_factory.build_dual_granularity_aligned_routes
        with mock.patch.object(
            route_factory,
            "build_dual_granularity_aligned_routes",
            side_effect=original,
        ) as dual:
            routes = route_factory.build_routes(
                _serial_materialized_reference(),
                topk_ids=topk_ids,
                token_slots=token_slots,
                num_local_experts=lora_experts_per_adapter,
                max_loras=max_loras,
                block_size=16,
                gate_up_a_block_size=64,
                workspace=MoeLoraWorkspace(),
            )
        self.assertEqual(dual.call_count, 1)
        for view, block_size in (
            (routes.aligned_per_expert, 16),
            (routes.gate_up_a_aligned_per_expert, 64),
        ):
            self._assert_view_matches(
                view,
                self._standalone_plan(
                    topk_ids,
                    token_slots,
                    lora_experts_per_adapter=lora_experts_per_adapter,
                    max_loras=max_loras,
                    block_size=block_size,
                ),
                block_size,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
