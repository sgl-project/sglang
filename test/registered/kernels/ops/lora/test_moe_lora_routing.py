"""Correctness tests for SGL-LoRA bucket routing."""

import unittest
from collections import defaultdict

import torch

from sglang.srt.lora.kernels import routing as routing_module
from sglang.srt.lora.kernels.routing import build_route
from sglang.srt.lora.route_view import RouteViewKind
from sglang.srt.lora.workspace import LoraWorkspace
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase


def _serial_materialized_reference():
    """Use standalone stages with no fusion or overlap."""
    from sglang.srt.lora.moe.plan import (
        ActFamily,
        ActivationFn,
        ActSpec,
        AFamily,
        ASpec,
        BFamily,
        BridgeLayout,
        BSpec,
        FinalizeFamily,
        FinalizeSpec,
        MoePlan,
        Site,
    )

    return MoePlan(
        gate_up_a=ASpec(Site.GATE_UP, AFamily.GROUPED, False, BridgeLayout.PAIR_MAJOR),
        gate_up_b=BSpec(
            Site.GATE_UP,
            BFamily.GROUPED,
            False,
            BridgeLayout.PAIR_MAJOR,
        ),
        act=ActSpec(ActFamily.MATERIALIZED, ActivationFn.SILU),
        down_a=ASpec(Site.DOWN, AFamily.GROUPED, False, BridgeLayout.PAIR_MAJOR),
        down_b=BSpec(Site.DOWN, BFamily.GROUPED, False, BridgeLayout.PAIR_MAJOR),
        finalize=FinalizeSpec(FinalizeFamily.MATERIALIZED),
    )


register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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
        per_expert = build_route(
            torch.tensor(adapters, dtype=dtype, device=self.device),
            group_ids=torch.tensor(topk_ids, dtype=dtype, device=self.device),
            groups_per_slot=num_local_experts,
            max_loras=max_loras,
            block_size=block_size,
            view=view,
            workspace=LoraWorkspace(),
            tensor_prefix="test:route",
        )
        return per_expert

    @staticmethod
    def _parity_inputs(rows, width, groups_per_slot, max_loras, dtype, shift=0):
        """CPU inputs include both invalid slots and invalid/folded group IDs."""
        slots = (torch.arange(rows // width, dtype=dtype) + shift) % max_loras
        slots[::7] = -1
        slots[1::11] = max_loras
        if width == 1:
            return slots, None
        groups = (
            (torch.arange(rows, dtype=dtype) * 37 + shift) % groups_per_slot
        ).reshape(-1, width)
        groups[shift % 11 :: 11, -1] = -2
        groups[2 + shift :: 13, 0] = 37 if groups_per_slot == 1 else groups_per_slot
        # Exercise the last valid bucket, including the high-NUM_BINS case.
        slots[3] = max_loras - 1
        groups[3, 0] = groups_per_slot - 1
        return slots, groups

    @staticmethod
    def _expected_bucket_rows(slots, groups, groups_per_slot, max_loras):
        """Independent host oracle; do not reuse either builder's bucket kernel."""
        expected = defaultdict(list)
        width = 1 if groups is None else groups.shape[1]
        group_rows = None if groups is None else groups.tolist()
        for token, slot in enumerate(slots.tolist()):
            for column in range(width):
                group = 0 if group_rows is None else group_rows[token][column]
                live = (
                    0 <= slot < max_loras
                    and group >= 0
                    and (groups_per_slot == 1 or group < groups_per_slot)
                )
                bucket = -1
                if live:
                    bucket = slot * groups_per_slot
                    if groups_per_slot != 1:
                        bucket += group
                expected[bucket].append(token * width + column)
        return dict(expected)

    def _assert_route_layout(self, result, expected, block_size):
        sorted_ids, block_ids, total = result
        for tensor in result:
            self.assertEqual(tensor.dtype, torch.int32)
        rows = sum(map(len, expected.values()))
        expected_labels = [
            bucket
            for bucket, pairs in sorted(expected.items())
            for _ in range((len(pairs) + block_size - 1) // block_size)
        ]
        padded = int(total.item())
        self.assertEqual(padded, len(expected_labels) * block_size)
        self.assertLessEqual(padded, sorted_ids.numel())
        # Small and CUDA alignment both put sentinel -1 first. Only the live
        # prefix is contractual; unused capacity need not contain the same data.
        labels = block_ids[: len(expected_labels)].cpu().tolist()
        self.assertEqual(labels, expected_labels)
        values = sorted_ids[:padded].cpu().tolist()
        self.assertTrue(all(0 <= pair <= rows for pair in values))
        actual = {}
        offset = 0
        for bucket, wanted in sorted(expected.items()):
            size = ((len(wanted) + block_size - 1) // block_size) * block_size
            bucket_rows = values[offset : offset + size]
            # Atomic claims may permute rows inside a bucket. Compare multisets,
            # not raw arrays or sets that could hide duplicate/missing rows.
            actual[bucket] = sorted(pair for pair in bucket_rows if pair != rows)
            self.assertEqual(actual[bucket], wanted)
            self.assertEqual(bucket_rows.count(rows), size - len(wanted))
            offset += size
        return padded, labels, actual

    def _parity_builders(self, slots, groups, groups_per_slot, max_loras, block_size):
        small_workspace, cuda_workspace = LoraWorkspace(), LoraWorkspace()
        for workspace in (small_workspace, cuda_workspace):
            workspace.begin_forward(graph_mode=True)
        shape = (slots.numel(), 1) if groups is None else groups.shape
        buckets = cuda_workspace.tensor(
            "parity:bucket_ids", shape, dtype=torch.int32, device=self.device
        )

        def small():
            route = routing_module._build_small_route(
                slots,
                groups,
                groups_per_slot=groups_per_slot,
                max_loras=max_loras,
                block_size=block_size,
                workspace=small_workspace,
                tensor_prefix="parity:small",
            )
            return (
                route.sorted_pair_ids,
                route.block_bucket_ids,
                route.num_pairs_post_padded,
            )

        def cuda():
            routing_module._build_route_bucket_ids(
                slots,
                groups,
                groups_per_slot=groups_per_slot,
                max_loras=max_loras,
                out=buckets,
            )
            return routing_module._align_block_size_jit(
                buckets,
                block_size,
                groups_per_slot * max_loras,
                scratch=lambda size: cuda_workspace.tensor(
                    "parity:scratch", (size,), dtype=torch.int32, device=self.device
                ),
            )

        return small, cuda

    def test_small_route_matches_cuda_layout(self):
        # A bounded set, not a Cartesian sweep: both capacity branches, native
        # 4/8 warps, dense/MoE/shared rows, and the low-bucket extension.
        cases = (
            (1, 1, 1, 4, 16, torch.int32),
            (512, 8, 64, 4, 32, torch.int32),
            (513, 1, 1, 4, 128, torch.int64),
            (768, 1, 1, 4, 128, torch.int32),
            (768, 8, 1, 4, 63, torch.int64),
            (512, 8, 1024, 4, 16, torch.int32),  # 4097 bins round to NUM_BINS=8192.
        )
        for rows, width, groups_per_slot, max_loras, block_size, dtype in cases:
            with self.subTest(
                rows=rows, width=width, groups=groups_per_slot, block=block_size
            ):
                host_slots, host_groups = self._parity_inputs(
                    rows, width, groups_per_slot, max_loras, dtype
                )
                expected = self._expected_bucket_rows(
                    host_slots, host_groups, groups_per_slot, max_loras
                )
                builders = self._parity_builders(
                    host_slots.to(self.device),
                    None if host_groups is None else host_groups.to(self.device),
                    groups_per_slot,
                    max_loras,
                    block_size,
                )
                layouts = [
                    self._assert_route_layout(build(), expected, block_size)
                    for build in builders
                ]
                self.assertEqual(layouts[0], layouts[1])

    def test_small_route_cuda_parity_replays_changed_inputs(self):
        rows, width, groups_per_slot, max_loras, block_size = 768, 8, 1, 4, 128
        host_slots, host_groups = self._parity_inputs(
            rows, width, groups_per_slot, max_loras, torch.int64
        )
        slots, groups = host_slots.to(self.device), host_groups.to(self.device)
        builders = self._parity_builders(
            slots, groups, groups_per_slot, max_loras, block_size
        )
        expected = self._expected_bucket_rows(
            host_slots, host_groups, groups_per_slot, max_loras
        )
        captures = []
        for build in builders:
            warm = build()
            self._assert_route_layout(warm, expected, block_size)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = build()
            self.assertEqual(
                [tensor.data_ptr() for tensor in result],
                [tensor.data_ptr() for tensor in warm],
            )
            captures.append((graph, result))

        for state in ("changed", "inactive", "restored"):
            with self.subTest(state=state):
                new_slots, new_groups = self._parity_inputs(
                    rows,
                    width,
                    groups_per_slot,
                    max_loras,
                    torch.int64,
                    shift=int(state == "changed"),
                )
                if state == "inactive":
                    new_slots.fill_(-1)
                slots.copy_(new_slots)
                groups.copy_(new_groups)
                expected = self._expected_bucket_rows(
                    new_slots, new_groups, groups_per_slot, max_loras
                )
                layouts = []
                for graph, result in captures:
                    for tensor in result:
                        tensor.fill_(-7)
                    graph.replay()
                    layouts.append(
                        self._assert_route_layout(result, expected, block_size)
                    )
                self.assertEqual(layouts[0], layouts[1])

    def test_narrower_views_refuse_fields_they_did_not_build(self):
        """Unbuilt view fields must raise at access, not pass None into a kernel launch."""
        ids, adapters = [[0, 1]], [0]
        aligned = self._build(
            ids, adapters, num_local_experts=2, view=RouteViewKind.ALIGNED
        )
        self.assertGreater(aligned.sorted_pair_ids.numel(), 0)
        self.assertGreater(aligned.block_bucket_ids.numel(), 0)

        raw = self._build(ids, adapters, num_local_experts=2, view=RouteViewKind.RAW)
        for field in ("sorted_pair_ids", "block_bucket_ids"):
            with self.assertRaisesRegex(ValueError, RouteViewKind.ALIGNED):
                getattr(raw, field)
        # A raw consumer fuses the key computation into its own kernel, so the
        # sources must survive on the view.
        self.assertEqual(raw.groups_per_slot, 2)
        self.assertEqual(raw.token_slots.numel(), 1)

    def test_unknown_view_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown route view"):
            self._build([[0, 1]], [0], num_local_experts=2, view="grouped")

    def test_invalid_adapter_and_expert_ids_become_one_sentinel(self):
        route = self._build(
            [[-2], [-1], [3], [4], [99], [0], [0]],
            [0, 0, 0, 0, 0, 2, 3],
            num_local_experts=4,
        )
        # Only pair 2 is valid. Invalid expert/adapter ids must share one sentinel.
        live_blocks = route.num_pairs_post_padded.item() // route.block_size
        keys = route.block_bucket_ids[:live_blocks].cpu().tolist()
        self.assertIn(3, keys)
        self.assertEqual(set(keys) - {-1}, {3})

        live_blocks = route.num_pairs_post_padded.item() // route.block_size
        live_ids = route.block_bucket_ids[:live_blocks]
        self.assertTrue(
            bool(
                ((live_ids == -1) | ((live_ids >= 0) & (live_ids < route.num_buckets)))
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
        keys = route.block_bucket_ids[:live_blocks].cpu().tolist()
        self.assertEqual(set(keys) - {-1}, {0, 3})
        self.assertEqual(route.sorted_pair_ids.dtype, torch.int32)
        self.assertEqual(route.block_bucket_ids.dtype, torch.int32)

    def test_sentinel_bucket_is_included_in_capacity(self):
        route = self._build(
            [[0], [1], [2], [3], [0], [1], [2], [3], [-1]],
            [0, 0, 0, 0, 1, 1, 1, 1, 0],
            num_local_experts=4,
            block_size=4,
        )
        self.assertEqual(route.num_pairs_post_padded.item(), 9 * 4)
        self.assertGreaterEqual(route.sorted_pair_ids.numel(), 9 * 4)
        self.assertGreaterEqual(route.block_bucket_ids.numel(), 9)

    def test_aligned_view_policy_boundaries(self):
        """Pin both sides of the fused-policy thresholds: G >= 8192 or P >= 16384.
        The G boundary exceeds the JIT builder's capacity; both constants affect dispatch.
        """
        from sglang.srt.lora.kernels import routing as routing_module
        from sglang.srt.lora.kernels.routing import (
            _LARGE_ROUTE_MIN_BUCKETS,
            _LARGE_ROUTE_MIN_PAIRS,
        )
        from sglang.srt.lora.route_view import RouteViewKind

        self.assertEqual(_LARGE_ROUTE_MIN_BUCKETS, 8192)
        self.assertEqual(_LARGE_ROUTE_MIN_PAIRS, 16384)
        # G=8192 must reach the fused builder; the JIT path cannot align it.

        # (G, T, expects_fused): straddles both edges; K = 8 so P = 8 * T.
        cases = (
            (8160, 8, False),  # below both edges -> ID pass + JIT
            (8192, 8, True),  # at the G edge (the EPT rung)
            (12288, 8, True),  # kimi EP1 x 32 slots, the realistic large case
            (1024, 2048, True),  # small G, P = 16384: the P edge
            (1024, 1024, False),  # small G, P = 8192: below the P edge
            (40960, 8, True),  # above the JIT ceiling: fused is the only path
        )
        original = routing_module._build_large_route
        for num_route_buckets, num_tokens, expects_fused in cases:
            num_local_experts = num_route_buckets // 32
            with self.subTest(G=num_route_buckets, P=num_tokens * 8):
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

                    routing_module._build_large_route = spy
                    route = build_route(
                        slots,
                        group_ids=ids,
                        groups_per_slot=num_local_experts,
                        max_loras=32,
                        block_size=16,
                        view=RouteViewKind.ALIGNED,
                        workspace=LoraWorkspace(),
                        tensor_prefix="test:route",
                    )
                finally:
                    routing_module._build_large_route = original
                self.assertEqual(
                    bool(calls),
                    expects_fused,
                    f"G={num_route_buckets}, P={num_tokens * 8} took the wrong path",
                )
                if expects_fused:
                    self.assertEqual(len(calls), 1)
                self.assertGreater(int(route.num_pairs_post_padded), 0)
                self.assertEqual(route.sorted_pair_ids.dtype, torch.int32)
                self.assertEqual(route.block_bucket_ids.dtype, torch.int32)

    def test_sentinel_blocks_isolate_invalid_pairs_on_both_align_paths(self):
        """Invalid pairs belong only to -1-labelled blocks; valid pairs appear once.
        Every padded slot must be readable (< P or sentinel P), since B dereferences
        even invalid blocks to zero-fill. Check both independent alignment paths.
        """
        for num_local_experts, max_loras in ((8, 32), (384, 32)):
            num_route_buckets = num_local_experts * max_loras
            with self.subTest(G=num_route_buckets):
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
                route = build_route(
                    token_lora_mapping,
                    group_ids=topk_ids,
                    groups_per_slot=num_local_experts,
                    max_loras=max_loras,
                    block_size=16,
                    view=RouteViewKind.ALIGNED,
                    workspace=LoraWorkspace(),
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
                block_bucket_ids = route.block_bucket_ids.cpu()
                self.assertTrue(bool((keys == -1).any()), "case must have sentinels")

                seen: dict[int, int] = {}
                for block in range(num_padded // 16):
                    label = int(block_bucket_ids[block])
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
