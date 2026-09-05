"""Correctness tests for SGL-LoRA virtual-expert routing."""

import unittest

import pytest
import torch

from sglang.srt.lora.moe.route_view import RouteViewKind
from sglang.srt.lora.moe.routing import build_virtual_expert_routing
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase


def _serial_materialized_reference():
    """The simplest correct pipeline: every stage standalone, nothing fused,
    no overlap window."""
    from sglang.srt.lora.moe.execution_plan import (
        ActFamily,
        ActivationFn,
        ActSpec,
        BridgeLayout,
        FinalizeFamily,
        FinalizeSpec,
        LoraAFamily,
        LoraASpec,
        LoraBFamily,
        LoraBSpec,
        MoeLoraExecutionPlan,
        Site,
    )

    return MoeLoraExecutionPlan(
        gate_up_a=LoraASpec(
            Site.GATE_UP, LoraAFamily.GROUPED, False, BridgeLayout.PAIR_MAJOR
        ),
        gate_up_b=LoraBSpec(
            Site.GATE_UP,
            LoraBFamily.GROUPED,
            False,
            BridgeLayout.PAIR_MAJOR,
        ),
        act=ActSpec(ActFamily.MATERIALIZED, ActivationFn.SILU),
        down_a=LoraASpec(
            Site.DOWN, LoraAFamily.GROUPED, False, BridgeLayout.PAIR_MAJOR
        ),
        down_b=LoraBSpec(
            Site.DOWN, LoraBFamily.GROUPED, False, BridgeLayout.PAIR_MAJOR
        ),
        finalize=FinalizeSpec(FinalizeFamily.MATERIALIZED),
    )


register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-large")


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
        num_local_experts,
        max_loras=2,
        block_size=16,
        dtype=torch.int32,
        view=RouteViewKind.ALIGNED,
    ):
        # A real workspace, since the fused builder requires one.
        per_expert = build_virtual_expert_routing(
            torch.tensor(topk_ids, dtype=dtype, device=self.device),
            torch.tensor(adapters, dtype=dtype, device=self.device),
            num_local_experts=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            view=view,
            workspace=MoeLoraWorkspace(),
            tensor_prefix="test:route",
        )
        return per_expert

    def test_narrower_views_refuse_fields_they_did_not_build(self):
        """A view must not silently hand back a field it never computed.

        The two views exist so a schedule pays only for what it reads (plan
        section 29 R1). If an unbuilt field returned None instead of raising,
        a consumer that requested the wrong view would pass None into a Triton
        launch and fail far from the mistake.
        """
        ids, adapters = [[0, 1]], [0]
        aligned = self._build(
            ids, adapters, num_local_experts=2, view=RouteViewKind.ALIGNED
        )
        self.assertGreater(aligned.sorted_pair_ids.numel(), 0)
        self.assertGreater(aligned.block_virtual_expert_ids.numel(), 0)

        raw = self._build(ids, adapters, num_local_experts=2, view=RouteViewKind.RAW)
        for field in ("sorted_pair_ids", "block_virtual_expert_ids"):
            with self.assertRaisesRegex(ValueError, RouteViewKind.ALIGNED):
                getattr(raw, field)
        # A raw consumer fuses the key computation into its own kernel, so the
        # sources must survive on the view.
        self.assertEqual(raw.lora_experts_per_adapter, 2)
        self.assertEqual(raw.token_lora_mapping.numel(), 1)

    def test_unknown_view_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown route view"):
            self._build([[0, 1]], [0], num_local_experts=2, view="grouped")

    def test_invalid_adapter_and_expert_ids_become_one_sentinel(self):
        route = self._build(
            [[-2], [-1], [3], [4], [99], [0], [0]],
            [0, 0, 0, 0, 0, 2, 3],
            num_local_experts=4,
        )
        # Only pair 2 is valid (adapter 0, expert 3 -> key 3); every other
        # pair is invalid for a different reason -- negative expert, expert
        # past the local count, adapter past the capacity -- and they must all
        # collapse onto the ONE sentinel rather than distinct bad keys.
        live_blocks = route.num_pairs_post_padded.item() // route.block_size
        keys = route.block_virtual_expert_ids[:live_blocks].cpu().tolist()
        self.assertIn(3, keys)
        self.assertEqual(set(keys) - {-1}, {3})

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

    def test_int64_source_ids_build_an_int32_plan(self):
        """int64 sources are accepted; the plan itself stays int32."""
        route = self._build(
            [[0, 3], [4, -2]],
            [0, 1],
            num_local_experts=4,
            dtype=torch.int64,
        )
        # Adapter 0 owns experts 0 and 3 (keys 0 and 3); adapter 1's pairs are
        # both invalid -- expert 4 is past the local count, -2 is a sentinel.
        live_blocks = route.num_pairs_post_padded.item() // route.block_size
        keys = route.block_virtual_expert_ids[:live_blocks].cpu().tolist()
        self.assertEqual(set(keys) - {-1}, {0, 3})
        self.assertEqual(route.sorted_pair_ids.dtype, torch.int32)
        self.assertEqual(route.block_virtual_expert_ids.dtype, torch.int32)

    def test_sentinel_bucket_is_included_in_capacity(self):
        route = self._build(
            [[0], [1], [2], [3], [0], [1], [2], [3], [-1]],
            [0, 0, 0, 0, 1, 1, 1, 1, 0],
            num_local_experts=4,
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
        from sglang.srt.lora.moe import routing as routing_module
        from sglang.srt.lora.moe.route_view import RouteViewKind
        from sglang.srt.lora.moe.routing import (
            FUSED_ALIGN_MIN_PAIRS,
            FUSED_ALIGN_MIN_VIRTUAL_EXPERTS,
            build_virtual_expert_routing,
        )

        self.assertEqual(FUSED_ALIGN_MIN_VIRTUAL_EXPERTS, 8192)
        self.assertEqual(FUSED_ALIGN_MIN_PAIRS, 16384)
        # The JIT primitive's own ceiling (8191) is asserted where it lives, in
        # kernels/ops/moe/virtual_experts.py; restating it here could only ever
        # agree with itself. What guards the dispatch is the edge cases below:
        # 8192 has to REACH the fused builder, because the JIT path cannot
        # align it.

        # (V, T, expects_fused): straddles both edges; K = 8 so P = 8 * T.
        cases = (
            (8160, 8, False),  # below both edges -> ID pass + JIT
            (8192, 8, True),  # at the V edge (the EPT rung)
            (12288, 8, True),  # kimi EP1 x 32 slots, the realistic large case
            (1024, 2048, True),  # small V, P = 16384: the P edge
            (1024, 1024, False),  # small V, P = 8192: below the P edge
            (40960, 8, True),  # above the JIT ceiling: fused is the only path
        )
        original = routing_module._build_aligned
        for num_virtual, num_tokens, expects_fused in cases:
            num_local_experts = num_virtual // 32
            with self.subTest(V=num_virtual, P=num_tokens * 8):
                ids = torch.randint(
                    0,
                    num_local_experts,
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

                    routing_module._build_aligned = spy
                    route = build_virtual_expert_routing(
                        ids,
                        slots,
                        num_local_experts=num_local_experts,
                        max_loras=32,
                        block_size=16,
                        view=RouteViewKind.ALIGNED,
                        workspace=MoeLoraWorkspace(),
                        tensor_prefix="test:route",
                    )
                finally:
                    routing_module._build_aligned = original
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
        for num_local_experts, max_loras in ((8, 32), (384, 32)):
            num_virtual = num_local_experts * max_loras
            with self.subTest(V=num_virtual):
                generator = torch.Generator(device="cpu").manual_seed(23)
                num_tokens, top_k = 96, 8
                topk_ids = torch.randint(
                    -1,
                    num_local_experts,
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
                    num_local_experts=num_local_experts,
                    max_loras=max_loras,
                    block_size=16,
                    view=RouteViewKind.ALIGNED,
                    workspace=MoeLoraWorkspace(),
                    tensor_prefix="test:route",
                )
                num_pairs = num_tokens * top_k
                keys = (
                    torch.where(
                        (token_lora_mapping[:, None] >= 0) & (topk_ids >= 0),
                        token_lora_mapping[:, None].to(torch.int64) * num_local_experts
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


def test_segment_token_route_matches_reference() -> None:
    """The token route from request segments: each request's tokens in place,
    padded to whole blocks and keyed by its slot; requests without an adapter
    and empty requests add no block; leftover blocks read -1."""
    if not torch.cuda.is_available():
        pytest.skip("route kernels need CUDA")
    from sglang.srt.lora.moe.routing import _build_segment_token_route
    from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

    device = torch.device("cuda")
    block = 16
    seg_lens = [5, 0, 17, 3, 16, 2]
    slots = [0, 2, 1, -1, 1, 0]
    seg_indptr = torch.tensor(
        [0] + list(torch.tensor(seg_lens).cumsum(0)), dtype=torch.int32, device=device
    )
    num_tokens = int(seg_indptr[-1])
    mapping = torch.cat(
        [torch.full((n,), s, dtype=torch.int32) for n, s in zip(seg_lens, slots)]
    ).to(device)
    workspace = MoeLoraWorkspace()
    workspace.begin_forward(graph_mode=False)
    route = _build_segment_token_route(
        seg_indptr=seg_indptr,
        token_lora_mapping=mapping,
        num_tokens=num_tokens,
        num_local_experts=4,
        max_loras=3,
        block_size=block,
        workspace=workspace,
    )
    torch.cuda.synchronize()
    exp_sorted, exp_blocks, start = [], [], 0
    for n, s in zip(seg_lens, slots):
        if n > 0 and s >= 0:
            padded = -(-n // block) * block
            exp_sorted += [start + i if i < n else num_tokens for i in range(padded)]
            exp_blocks += [s] * (padded // block)
        start += n
    padded_total = len(exp_sorted)
    assert int(route.maybe_num_pairs_post_padded.item()) == padded_total
    assert route.maybe_sorted_pair_ids[:padded_total].tolist() == exp_sorted
    blocks = route.maybe_block_virtual_expert_ids.tolist()
    assert blocks[: len(exp_blocks)] == exp_blocks
    assert all(b == -1 for b in blocks[len(exp_blocks) :])
    assert route.topk_ids.shape == (num_tokens, 1) and route.is_shared_outer
