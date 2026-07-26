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

    def _inputs(self, factor_experts, slot_capacity, num_tokens, top_k, seed=11):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        topk_ids = torch.randint(
            0,
            factor_experts,
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
        for factor_experts, slot_capacity in ((32, 8), (32, 32), (384, 32), (1024, 32)):
            for num_tokens in (1, 64, 512):
                with self.subTest(V=factor_experts * slot_capacity, T=num_tokens):
                    topk_ids, token_slots = self._inputs(
                        factor_experts, slot_capacity, num_tokens, top_k=8
                    )
                    num_pairs = topk_ids.numel()
                    num_virtual = factor_experts * slot_capacity
                    virtual_ids = _build_virtual_topk_ids(
                        topk_ids, token_slots, factor_experts, slot_capacity, None
                    )
                    capacity = _routing_capacity(num_pairs, BLOCK_SIZE_M, num_virtual)
                    fused = fused_align_block_size(
                        topk_ids,
                        token_slots,
                        factor_expert_count=factor_experts,
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
        factor_experts, slot_capacity, num_tokens, top_k = 32, 8, 64, 8
        hidden, rank = 256, 16
        topk_ids, token_slots = self._inputs(
            factor_experts, slot_capacity, num_tokens, top_k
        )
        num_pairs = topk_ids.numel()
        num_virtual = factor_experts * slot_capacity
        capacity = _routing_capacity(num_pairs, BLOCK_SIZE_M, num_virtual)

        generator = torch.Generator(device="cpu").manual_seed(5)
        hidden_states = torch.randn(
            num_tokens, hidden, generator=generator, dtype=torch.bfloat16
        ).to(self.device)
        lora_a = (
            torch.randn(
                slot_capacity,
                factor_experts,
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
                factor_experts,
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
                factor_expert_count=factor_experts,
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
                factor_expert_count=factor_experts,
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
