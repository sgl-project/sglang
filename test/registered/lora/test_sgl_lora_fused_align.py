"""Correctness of the fused ID+histogram+scan+scatter route plan."""

import unittest

import torch

from sglang.srt.lora.sgl_lora.bf16 import grouped_lora_a, stock_grouped_lora_b
from sglang.srt.lora.sgl_lora.fused_align import fused_align_block_size
from sglang.srt.lora.sgl_lora.moe_lora_runner import PROVISIONAL_LAUNCH_CONFIG
from sglang.srt.lora.sgl_lora.routing import (
    RouteView,
    _align_block_size_torch,
    _build_virtual_topk_ids,
    _routing_capacity,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

BLOCK_SIZE_M = PROVISIONAL_LAUNCH_CONFIG.routing_block_size


class TestSglLoraFusedAlign(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda:0"

    def _inputs(
        self, lora_experts_per_adapter, slot_capacity, num_tokens, top_k, seed=11
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        topk_ids = torch.randint(
            0,
            lora_experts_per_adapter,
            (num_tokens, top_k),
            generator=generator,
            dtype=torch.int32,
        )
        token_slots = torch.randint(
            -1, slot_capacity, (num_tokens,), generator=generator, dtype=torch.int32
        )
        return topk_ids.to(self.device), token_slots.to(self.device)

    def _plan_meaning(self, sorted_ids, block_ids, num_padded, virtual_ids, num_pairs):
        """``{virtual expert -> set(pair ids)}``, the only contractual content.

        Intra-bucket pair order and the position of the sentinel group are NOT
        contractual — the incumbent CUDA and torch paths already disagree on
        both — so the torch reference is a semantic oracle here, never a bitwise
        one.
        """
        flat = virtual_ids.reshape(-1)
        grouping: dict[int, set[int]] = {}
        for block in range(int(num_padded) // BLOCK_SIZE_M):
            expert = int(block_ids[block])
            if expert < 0:
                continue
            slot = sorted_ids[block * BLOCK_SIZE_M : (block + 1) * BLOCK_SIZE_M]
            real = slot[slot < num_pairs]
            if real.numel() == 0:
                continue
            owners = flat[real.long()]
            self.assertTrue(
                bool((owners == expert).all()),
                f"block {block} is labelled expert {expert} but holds pairs "
                f"belonging to {sorted(set(owners.tolist()))}",
            )
            grouping.setdefault(expert, set()).update(real.tolist())
        return grouping

    def test_matches_torch_oracle_including_above_the_incumbent_ceiling(self):
        """Same grouping as the reference, including above the JIT ceiling.

        The JIT align kernel tops out at 32767 virtual experts; this kernel is
        the only CUDA path beyond that, and under the section 40 policy it also
        serves V >= 8192 and P >= 16384. The V=32768 cells exercise the
        beyond-ceiling regime no other kernel can reach.
        """
        for lora_experts_per_adapter, slot_capacity in (
            (32, 8),
            (32, 32),
            (384, 32),
            (1024, 32),
        ):
            for num_tokens in (1, 64, 512):
                with self.subTest(
                    V=lora_experts_per_adapter * slot_capacity, T=num_tokens
                ):
                    topk_ids, token_slots = self._inputs(
                        lora_experts_per_adapter, slot_capacity, num_tokens, top_k=8
                    )
                    num_pairs = topk_ids.numel()
                    num_virtual = lora_experts_per_adapter * slot_capacity
                    virtual_ids = _build_virtual_topk_ids(
                        topk_ids,
                        token_slots,
                        lora_experts_per_adapter,
                        slot_capacity,
                        None,
                    )
                    capacity = _routing_capacity(num_pairs, BLOCK_SIZE_M, num_virtual)
                    fused = fused_align_block_size(
                        topk_ids,
                        token_slots,
                        lora_experts_per_adapter=lora_experts_per_adapter,
                        max_loras=slot_capacity,
                        block_size=BLOCK_SIZE_M,
                        capacity=capacity,
                    )
                    reference = _align_block_size_torch(
                        virtual_ids, BLOCK_SIZE_M, num_virtual
                    )
                    self.assertEqual(
                        int(fused[2]),
                        int(reference[2]),
                        "padded length must match the reference",
                    )
                    self.assertEqual(
                        self._plan_meaning(*fused, virtual_ids, num_pairs),
                        self._plan_meaning(*reference, virtual_ids, num_pairs),
                        "grouping of pairs by virtual expert must match",
                    )

    def test_fused_path_honors_a_global_to_local_lora_expert_map(self):
        """The fused align's map branch, at a size that FORCES the fused path.

        Eleventh S3 review: every fused-align case passed
        ``lora_expert_map=None``, so ``USE_LORA_EXPERT_MAP`` was never
        compiled True in this kernel — the map branch was exercised only by
        the id-builder and the JIT align. This cell sits above the section-40
        thresholds (V >= 8192 AND P >= 16384) — the region the routing policy
        selects this kernel for — and uses a non-contiguous global->local map so a branch
        that ignored the map (or dropped its -1 rows) could not still agree
        with the oracle.
        """
        slot_capacity, local_experts, global_experts = 32, 256, 1024
        num_tokens, top_k = 4096, 8
        num_virtual = local_experts * slot_capacity
        self.assertGreaterEqual(num_virtual, 8192)
        self.assertGreaterEqual(num_tokens * top_k, 16384)

        generator = torch.Generator(device="cpu").manual_seed(83)
        # Owned experts are a strided, NON-contiguous subset of the global
        # domain — the case an identity mapping cannot serve.
        owned = torch.arange(1, global_experts, 4)[:local_experts]
        lora_expert_map = torch.full((global_experts,), -1, dtype=torch.int32)
        lora_expert_map[owned] = torch.arange(local_experts, dtype=torch.int32)
        topk_ids = torch.randint(
            0,
            global_experts,
            (num_tokens, top_k),
            generator=generator,
            dtype=torch.int32,
        )
        token_slots = torch.randint(
            -1, slot_capacity, (num_tokens,), generator=generator, dtype=torch.int32
        )
        topk_ids = topk_ids.to(self.device)
        token_slots = token_slots.to(self.device)
        lora_expert_map = lora_expert_map.to(self.device)

        virtual_ids = _build_virtual_topk_ids(
            topk_ids, token_slots, local_experts, slot_capacity, lora_expert_map
        )
        # The map must actually bite: some pairs owned, some not.
        self.assertTrue(bool((virtual_ids >= 0).any()))
        self.assertTrue(bool((virtual_ids < 0).any()))

        num_pairs = topk_ids.numel()
        capacity = _routing_capacity(num_pairs, BLOCK_SIZE_M, num_virtual)
        fused = fused_align_block_size(
            topk_ids,
            token_slots,
            lora_experts_per_adapter=local_experts,
            max_loras=slot_capacity,
            block_size=BLOCK_SIZE_M,
            capacity=capacity,
            lora_expert_map=lora_expert_map,
        )
        reference = _align_block_size_torch(virtual_ids, BLOCK_SIZE_M, num_virtual)
        self.assertEqual(int(fused[2]), int(reference[2]))
        self.assertEqual(
            self._plan_meaning(*fused, virtual_ids, num_pairs),
            self._plan_meaning(*reference, virtual_ids, num_pairs),
        )

    def test_fused_shared_outer_rejects_ids_outside_the_local_range(self):
        """The positive out-of-range bound, in the fused kernel.

        With one LoRA expert per adapter the generic validity test becomes
        ``0 < 1`` — always true — so ``shared_outer_local_expert_count`` is
        the only thing rejecting a routed id this rank does not own. The row
        pattern covers -1 (sentinel), 0 and E-1 (owned edges), and E and E+n
        (positive, NOT owned): the last two are what a missing bound would
        wrongly admit.
        """
        local_experts, slot_capacity = 8, 4
        rows = [
            [-1, 0, local_experts - 1, local_experts],
            [local_experts + 3, 0, -1, local_experts - 1],
            [local_experts, local_experts + 1, local_experts + 2, 0],
        ]
        topk_ids = torch.tensor(rows, dtype=torch.int32, device=self.device)
        token_slots = torch.tensor([0, 2, -1], dtype=torch.int32, device=self.device)
        num_pairs = topk_ids.numel()
        num_virtual = 1 * slot_capacity

        virtual_ids = _build_virtual_topk_ids(
            topk_ids,
            token_slots,
            1,
            slot_capacity,
            None,
            shared_outer_local_expert_count=local_experts,
        )
        # Only in-range ids on non-base tokens survive, and they all fold to
        # the adapter's single LoRA expert (key == adapter slot).
        expected = [
            [-1, 0, 0, -1],
            [-1, 2, -1, 2],
            [-1, -1, -1, -1],
        ]
        self.assertEqual(virtual_ids.cpu().tolist(), expected)

        capacity = _routing_capacity(num_pairs, BLOCK_SIZE_M, num_virtual)
        fused = fused_align_block_size(
            topk_ids,
            token_slots,
            lora_experts_per_adapter=1,
            max_loras=slot_capacity,
            block_size=BLOCK_SIZE_M,
            capacity=capacity,
            shared_outer_local_expert_count=local_experts,
        )
        reference = _align_block_size_torch(virtual_ids, BLOCK_SIZE_M, num_virtual)
        self.assertEqual(int(fused[2]), int(reference[2]))
        self.assertEqual(
            self._plan_meaning(*fused, virtual_ids, num_pairs),
            self._plan_meaning(*reference, virtual_ids, num_pairs),
        )

    def test_direct_entry_enforces_the_full_shared_outer_contract(self):
        """The DIRECT entry point must reject what the builder rejects.

        Bug regression (eleventh S3 review): ``fused_align_block_size`` is
        reachable without going through ``build_virtual_expert_routing``,
        and it used to validate only the map/shared mutual exclusion. A
        caller could therefore reach the kernels with
        ``lora_experts_per_adapter != 1`` while asking for the shared-outer
        key form — the keys would be built against the wrong bucket count
        with no error. Each case below RAISES on the fixed code and was
        accepted by the pre-fix implementation.
        """
        topk_ids = torch.zeros((4, 2), dtype=torch.int32, device=self.device)
        token_slots = torch.zeros(4, dtype=torch.int32, device=self.device)
        common = dict(
            topk_ids=topk_ids,
            token_slots=token_slots,
            max_loras=4,
            block_size=BLOCK_SIZE_M,
            capacity=_routing_capacity(topk_ids.numel(), BLOCK_SIZE_M, 4),
        )
        # The one the pre-fix code let through.
        with self.assertRaisesRegex(ValueError, "exactly one LoRA expert"):
            fused_align_block_size(
                lora_experts_per_adapter=2,
                shared_outer_local_expert_count=8,
                **common,
            )
        with self.assertRaisesRegex(ValueError, "must be positive"):
            fused_align_block_size(
                lora_experts_per_adapter=1,
                shared_outer_local_expert_count=0,
                **common,
            )
        # This one the pre-fix code already caught; kept so the shared
        # validator cannot regress on any single condition.
        with self.assertRaisesRegex(ValueError, "contradictory"):
            fused_align_block_size(
                lora_experts_per_adapter=1,
                shared_outer_local_expert_count=8,
                lora_expert_map=torch.zeros(8, dtype=torch.int32, device=self.device),
                **common,
            )

    def test_fused_map_masks_ids_at_and_beyond_the_map_extent(self):
        """The map branch's own out-of-range mask.

        ``in_map`` gates the gather on ``0 <= routed_id < map_extent``. This
        pins all three edges in one plan: ``-1`` (sentinel), ``map_extent``
        and ``map_extent + n`` (positive but past the table, which an
        unmasked gather would read out of bounds), alongside owned and
        explicitly non-owned in-range ids.
        """
        map_extent, slot_capacity, local_experts = 6, 4, 3
        # ids 0,2,4 owned -> 0,1,2 ; ids 1,3,5 in range but NOT owned.
        lora_expert_map = torch.tensor(
            [0, -1, 1, -1, 2, -1], dtype=torch.int32, device=self.device
        )
        rows = [
            [0, 1, map_extent, -1],
            [2, map_extent + 5, 4, 3],
            [4, 0, -1, map_extent + 1],
        ]
        topk_ids = torch.tensor(rows, dtype=torch.int32, device=self.device)
        token_slots = torch.tensor([1, 0, -1], dtype=torch.int32, device=self.device)
        num_virtual = local_experts * slot_capacity
        num_pairs = topk_ids.numel()

        virtual_ids = _build_virtual_topk_ids(
            topk_ids, token_slots, local_experts, slot_capacity, lora_expert_map
        )
        # adapter 1 -> keys 1*3 + lora_expert; adapter 0 -> keys 0*3 + ...;
        # base row (-1 slot) contributes nothing.
        expected = [
            [3, -1, -1, -1],
            [1, -1, 2, -1],
            [-1, -1, -1, -1],
        ]
        self.assertEqual(virtual_ids.cpu().tolist(), expected)

        fused = fused_align_block_size(
            topk_ids,
            token_slots,
            lora_experts_per_adapter=local_experts,
            max_loras=slot_capacity,
            block_size=BLOCK_SIZE_M,
            capacity=_routing_capacity(num_pairs, BLOCK_SIZE_M, num_virtual),
            lora_expert_map=lora_expert_map,
        )
        reference = _align_block_size_torch(virtual_ids, BLOCK_SIZE_M, num_virtual)
        self.assertEqual(int(fused[2]), int(reference[2]))
        self.assertEqual(
            self._plan_meaning(*fused, virtual_ids, num_pairs),
            self._plan_meaning(*reference, virtual_ids, num_pairs),
        )

    def test_varying_plans_still_produce_bitwise_identical_output(self):
        """The atomic slot claim must not leak into consumer output.

        The scatter claims slots with `tl.atomic_add`, so the plan differs run
        to run. That is only acceptable because consumers write `out[pair_id]`
        — indexed by pair, not by slot — and accumulate in fixed k order. If a
        future consumer indexed by slot, or a B kernel gained split-K with
        atomics, output would silently become nondeterministic; nothing else in
        the suite covers that, because every other test uses the deterministic
        incumbent align.

        The plan-varies assertion is load-bearing: without it this test would
        pass vacuously if the scatter ever became deterministic, and stop
        guarding anything.
        """
        lora_experts_per_adapter, slot_capacity, num_tokens, top_k = 32, 8, 64, 8
        hidden, rank = 256, 16
        topk_ids, token_slots = self._inputs(
            lora_experts_per_adapter, slot_capacity, num_tokens, top_k
        )
        num_pairs = topk_ids.numel()
        num_virtual = lora_experts_per_adapter * slot_capacity
        capacity = _routing_capacity(num_pairs, BLOCK_SIZE_M, num_virtual)

        generator = torch.Generator(device="cpu").manual_seed(5)
        hidden_states = torch.randn(
            num_tokens, hidden, generator=generator, dtype=torch.bfloat16
        ).to(self.device)
        lora_a = (
            torch.randn(
                slot_capacity,
                lora_experts_per_adapter,
                rank,
                hidden,
                generator=generator,
                dtype=torch.bfloat16,
            ).to(self.device)
            / hidden**0.5
        )
        lora_b = (
            torch.randn(
                slot_capacity,
                lora_experts_per_adapter,
                hidden,
                rank,
                generator=generator,
                dtype=torch.bfloat16,
            ).to(self.device)
            / rank**0.5
        )

        outputs, plans = [], []
        for _ in range(20):
            sorted_ids, block_ids, num_padded = fused_align_block_size(
                topk_ids,
                token_slots,
                lora_experts_per_adapter=lora_experts_per_adapter,
                max_loras=slot_capacity,
                block_size=BLOCK_SIZE_M,
                capacity=capacity,
            )
            route = RouteView(
                view="aligned",
                num_virtual_experts=num_virtual,
                block_size=BLOCK_SIZE_M,
                topk_ids=topk_ids,
                token_slots=token_slots,
                lora_experts_per_adapter=lora_experts_per_adapter,
                max_loras=slot_capacity,
                maybe_virtual_topk_ids=torch.empty_like(topk_ids),
                maybe_sorted_pair_ids=sorted_ids,
                maybe_block_virtual_expert_ids=block_ids,
                maybe_num_pairs_post_padded=num_padded,
            )
            rank_out = torch.empty(
                num_pairs, rank, dtype=torch.bfloat16, device=self.device
            )
            grouped_lora_a(
                hidden_states,
                lora_a.flatten(0, 1),
                rank_out,
                route,
                config=PROVISIONAL_LAUNCH_CONFIG.lora_a,
            )
            delta = torch.empty(
                num_pairs, hidden, dtype=torch.bfloat16, device=self.device
            )
            stock_grouped_lora_b(
                rank_out,
                lora_b.flatten(0, 1),
                delta,
                route,
                destination_offsets=(0,),
                config=PROVISIONAL_LAUNCH_CONFIG.lora_b,
            )
            torch.cuda.synchronize()
            outputs.append(delta.clone())
            plans.append(sorted_ids.clone())

        distinct_plans = len({plan.cpu().numpy().tobytes() for plan in plans})
        self.assertGreater(
            distinct_plans,
            1,
            "the scatter is expected to produce varying plans; if it became "
            "deterministic this test no longer guards anything and should be "
            "rewritten rather than left passing",
        )
        for index, output in enumerate(outputs[1:], start=1):
            self.assertTrue(
                torch.equal(outputs[0], output),
                f"replay {index} differs from replay 0 despite the plan being "
                f"semantically equivalent ({distinct_plans} distinct plans seen)",
            )


if __name__ == "__main__":
    unittest.main()
