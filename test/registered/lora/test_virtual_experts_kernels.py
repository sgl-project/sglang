"""Unit tests for the LoRA virtual-experts kernels under post-EP-dispatch
sentinel `topk_ids` (-1) and out-of-range expert IDs.

Covers two regression bugs that surface only with `--lora-use-virtual-experts`
+ `ep_size > 1`:

- `_fused_virtual_topk_ids` must preserve negative sentinel topk_ids. After
  EP dispatch, non-local experts arrive as `-1`; the pre-fix kernel mapped
  them onto a real virtual-expert slot belonging to another adapter and
  triggered OOB loads in downstream LoRA kernels.

- `_align_block_size_torch` / `_align_block_size_jit` (the `>= 1024`-expert
  fallback paths) must route `-1` and `>= num_experts` IDs into a sentinel
  bucket so they don't OOB-index `padded_offsets[sorted_expert_ids]` (negative
  wrap, or past-end) and don't get assigned to a real expert in the
  consumer-block table.

Both kernels run on CUDA or XPU; only the CUDA-JIT align variant is CUDA-only.
The fallback is gated on `virtual_num_experts >= 1024` in production, but we
exercise it directly here at smaller sizes for cheaper iteration; one test
sticks to the >1024 regime to mirror the production trigger.

Usage:
    python -m pytest test/registered/lora/test_virtual_experts_kernels.py -v
"""

import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.utils import get_device, is_cuda, is_xpu
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=14, stage="extra-a", runner_config="1-gpu-small")
# backend-specific: XPU covers Triton sentinels and the torch alignment fallback.
register_xpu_ci(est_time=20, suite="stage-a-test-1-gpu-xpu")


def _require_accelerator_device():
    if not (is_cuda() or is_xpu()):
        raise unittest.SkipTest("CUDA or XPU required")
    return get_device(0)


from sglang.kernels.ops.lora.moe.virtual_experts import (
    _align_block_size_jit,
    _align_block_size_torch,
    _fused_virtual_topk_ids,
)


class TestFusedVirtualTopkIdsPreservesSentinels(CustomTestCase):
    """Item B regression: post-EP-dispatch -1 sentinels must NOT be remapped."""

    @classmethod
    def setUpClass(cls):
        cls.device = _require_accelerator_device()

    @unittest.skipIf(is_xpu(), "CUDA graph replay required")
    def test_capacity_tail_is_not_consumed_on_replay(self):
        """Unused routing capacity never contributes, including a zero used count."""
        import sglang.kernels.ops.lora.moe.trtllm_lora_temp.virtual_experts as temp
        import sglang.kernels.ops.lora.moe.virtual_experts as stock
        from sglang.srt.lora.trtllm_lora_temp.specialized_expand import (
            _moe_lora_expand_add_kernel,
        )

        block_m, block_n, capacity_blocks = 16, 16, 9
        routes, width, router_topk = block_m * capacity_blocks, 33, 2
        sorted_ids = torch.zeros(routes, dtype=torch.int32, device=self.device)
        expert_ids = torch.full(
            (capacity_blocks,), -3, dtype=torch.int32, device=self.device
        )
        count = torch.zeros(1, dtype=torch.int32, device=self.device)
        order = torch.arange(routes, dtype=torch.int32, device=self.device)
        route_values = (torch.arange(routes, device=self.device) % 7).half() / 8
        scores = torch.full((routes,), 0.5, device=self.device, dtype=torch.float16)

        def publish(blocks):
            count.fill_(blocks * block_m)
            sorted_ids.zero_()
            sorted_ids[: blocks * block_m].copy_(order[: blocks * block_m])
            expert_ids.fill_(-3)
            expert_ids[:blocks].fill_(0)
            if blocks:
                expert_ids[0] = -1

        variants = (
            [
                ("stock-shrink", stock._moe_lora_shrink_splitk_kernel, split_k)
                for split_k in (1, 2)
            ]
            + [
                ("temp-shrink", temp._moe_lora_shrink_splitk_kernel, split_k)
                for split_k in (1, 2)
            ]
            + [("expand", _moe_lora_expand_add_kernel, add) for add in (False, True)]
        )
        for name, kernel, mode in variants:
            with self.subTest(kernel=name, mode=mode):
                shrink = name != "expand"
                contraction = 64 if shrink else 32
                a = route_values[:, None].expand(routes, contraction).contiguous()
                # Negative tail IDs address a poisoned leading record if consumed.
                backing = torch.full(
                    (4, width, contraction),
                    float("nan"),
                    dtype=torch.float16,
                    device=self.device,
                )
                weight = backing[3:]
                weight.fill_(0.125)
                additive = not shrink and mode
                output_rows = routes // router_topk if additive else routes
                output = torch.empty(
                    (output_rows, width), dtype=torch.float16, device=self.device
                )
                initial = 0.0 if shrink and mode > 1 else 0.75
                baseline = torch.full_like(output, initial)
                grid = (capacity_blocks * ((width + block_n - 1) // block_n),)

                def launch():
                    if shrink:
                        kernel[(grid[0] * mode,)](
                            a,
                            weight,
                            output,
                            sorted_ids,
                            expert_ids,
                            count,
                            width,
                            contraction,
                            routes,
                            a.stride(0),
                            a.stride(1),
                            weight.stride(0),
                            weight.stride(1),
                            weight.stride(2),
                            output.stride(0),
                            output.stride(1),
                            top_k=1,
                            BLOCK_SIZE_M=block_m,
                            BLOCK_SIZE_N=block_n,
                            BLOCK_SIZE_K=32,
                            GROUP_SIZE_M=4,
                            SPLIT_K=mode,
                            num_warps=4,
                            num_stages=1,
                        )
                    else:
                        kernel[grid](
                            a,
                            weight,
                            output,
                            scores,
                            sorted_ids,
                            expert_ids,
                            count,
                            width,
                            contraction,
                            routes,
                            a.stride(0),
                            a.stride(1),
                            weight.stride(0),
                            weight.stride(1),
                            weight.stride(2),
                            output.stride(0),
                            output.stride(1),
                            router_topk=router_topk,
                            MUL_ROUTED_WEIGHT=True,
                            FUSE_SUM_ALL_REDUCE=mode,
                            BLOCK_SIZE_M=block_m,
                            BLOCK_SIZE_N=block_n,
                            BLOCK_SIZE_R=32,
                            GROUP_SIZE_M=4,
                            GATED_A_HALF=0,
                            BROADCAST_A=False,
                            num_warps=4,
                            num_stages=1,
                        )

                publish(capacity_blocks)
                output.copy_(baseline)
                launch()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    launch()
                for blocks in (0, 3, 5, capacity_blocks, 0, 3):
                    publish(blocks)
                    output.copy_(baseline)
                    graph.replay()
                    expected = baseline.clone()
                    active_end = blocks * block_m
                    projection = route_values * (contraction * 0.125)
                    if shrink:
                        expected[block_m:active_end] = projection[
                            block_m:active_end, None
                        ]
                    elif additive:
                        delta = torch.zeros(routes, device=self.device).half()
                        delta[block_m:active_end] = projection[block_m:active_end] * 0.5
                        expected += delta.view(-1, router_topk).sum(1)[:, None]
                    elif blocks:
                        expected[:block_m] = 0
                        expected[block_m:active_end] = (
                            projection[block_m:active_end, None] * 0.5
                        )
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    @unittest.skipIf(is_xpu(), "CUDA paired routing and graph replay required")
    def test_paired_routing_matches_staged_routes_and_replays(self):
        """Both per-expert stages share raw routes, including distinct tuned blocks."""
        import weakref

        import sglang.kernels.ops.lora.moe.trtllm_lora_temp.virtual_experts as temp
        from sglang.srt.layers.moe.moe_runner.triton_utils import (
            fused_moe_triton_config,
        )

        slots, experts, rank, hidden, width, topk = 3, 3, 8, 65, 33, 2
        # Strided payload metadata must not force flatten/copy during routing.
        a = torch.empty(
            (slots, experts, rank, hidden * 2), device=self.device, dtype=torch.float16
        )[..., ::2]
        b = torch.empty(
            (slots, experts, width * 2, rank), device=self.device, dtype=a.dtype
        )[:, :, ::2, :]
        shared_a = torch.empty(
            (slots, 1, rank, hidden), device=self.device, dtype=a.dtype
        )
        shared_b = torch.empty(
            (slots, 1, hidden, rank), device=self.device, dtype=a.dtype
        )

        def config(w1_shape, w2_shape, stage_topk, dtype, tokens):
            # Gate B and down shared B deliberately select different blocks.
            return {
                "BLOCK_SIZE_M": 32 if w2_shape[1] == width else 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 16,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 2,
            }

        for tokens in (6, 512):
            with (
                self.subTest(tokens=tokens),
                patch.object(
                    fused_moe_triton_config,
                    "try_get_optimal_moe_config",
                    side_effect=config,
                ),
                patch.object(
                    temp.lora_envs.SGLANG_OPT_LORA_SHRINK_TUNE,
                    "get",
                    return_value=False,
                ),
                patch.object(
                    temp.lora_envs.SGLANG_OPT_LORA_PREFILL_ROUTING_REUSE,
                    "get",
                    return_value=True,
                ),
            ):
                x = torch.empty((tokens, hidden), device=self.device, dtype=a.dtype)
                ids = torch.arange(tokens * topk, device=self.device).view(tokens, topk)
                ids.remainder_(experts)
                mapping = torch.arange(tokens, device=self.device).remainder_(slots)
                enabled = torch.ones(slots * 2, device=self.device, dtype=torch.int32)[
                    ::2
                ]
                weights = torch.ones((tokens, topk), device=self.device)
                output = torch.empty(
                    (tokens, topk, width), device=self.device, dtype=a.dtype
                )

                def prepare(cache):
                    self.assertTrue(
                        temp._prewarm_per_expert_lora_routes(
                            x,
                            a,
                            b,
                            ids,
                            mapping,
                            cache,
                            shrink_lora_b=shared_b,
                            adapter_enabled=enabled,
                            include_inactive_routes=True,
                            local_num_experts=experts,
                        )
                    )

                def check(cache):
                    valid_slot = (mapping >= 0) & (mapping < slots)
                    safe_slot = mapping.clamp(0, slots - 1)
                    mask = valid_slot & (enabled[safe_slot] != 0)
                    expected = torch.where(
                        mask[:, None] & (ids >= 0) & (ids < experts),
                        ids + mapping[:, None] * experts,
                        -1,
                    ).flatten()
                    for key, (
                        sorted_ids,
                        expert_ids,
                        count,
                        token_mask,
                    ) in cache.items():
                        block = key[2]
                        used = int(count[0].item())
                        self.assertLessEqual(used, sorted_ids.numel())
                        route = sorted_ids[:used].long()
                        expert = expert_ids[: used // block].repeat_interleave(block)
                        real = route < ids.numel()
                        route, expert = route[real], expert[real]
                        self.assertEqual(route.numel(), ids.numel())
                        self.assertEqual(route.unique().numel(), ids.numel())
                        actual = torch.empty_like(expected)
                        actual[route] = expert.to(actual.dtype)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        torch.testing.assert_close(token_mask, mask, rtol=0, atol=0)

                raw_refs = []
                remap = temp._fused_virtual_topk_ids

                def observe_raw(*args, **kwargs):
                    result = remap(*args, **kwargs)
                    raw_refs.append(weakref.ref(result[0]))
                    return result

                cache = {}
                with patch.object(
                    temp, "_fused_virtual_topk_ids", side_effect=observe_raw
                ):
                    prepare(cache)
                self.assertEqual(len(raw_refs), 1)
                self.assertIsNone(raw_refs[0]())
                expected_blocks = {32, 64 if tokens >= 512 else 16}
                self.assertEqual({key[2] for key in cache}, expected_blocks)
                tables = list(cache.values())
                self.assertIs(tables[0][3], tables[1][3])
                check(cache)

                # The unchanged public staged route path is an independent cache oracle.
                reference = {}
                common = dict(
                    output=output,
                    hidden_states=x,
                    topk_ids=ids,
                    topk_weights=weights,
                    token_lora_mapping=mapping,
                    routing_cache=reference,
                    stage="routing",
                    include_inactive_routes=True,
                    adapter_enabled=enabled,
                    local_num_experts=experts,
                )
                temp.merged_experts_fused_moe_lora_add(
                    **common,
                    lora_a=shared_a,
                    lora_b=b,
                    mul_routed_weight=False,
                    experts_shared_outer_loras_a=True,
                    experts_shared_outer_loras_b=False,
                    prewarm_a_routing=False,
                )
                temp.merged_experts_fused_moe_lora_add(
                    **common,
                    lora_a=a,
                    lora_b=shared_b,
                    mul_routed_weight=True,
                    experts_shared_outer_loras_a=False,
                    experts_shared_outer_loras_b=True,
                    prewarm_b_routing=False,
                )
                self.assertEqual(set(cache), set(reference))
                check(reference)

                # Enabled identity and inactive-route mode retain their distinct keys.
                other_enabled = enabled.clone().zero_()
                self.assertTrue(
                    temp._prewarm_per_expert_lora_routes(
                        x,
                        a,
                        b,
                        ids,
                        mapping,
                        cache,
                        shrink_lora_b=shared_b,
                        adapter_enabled=other_enabled,
                        include_inactive_routes=True,
                    )
                )
                self.assertTrue(
                    temp._prewarm_per_expert_lora_routes(
                        x,
                        a,
                        b,
                        ids,
                        mapping,
                        cache,
                        shrink_lora_b=shared_b,
                        adapter_enabled=enabled,
                        include_inactive_routes=False,
                    )
                )
                self.assertEqual(len(cache), 6)
                for key, value in cache.items():
                    if key[4] == id(other_enabled):
                        self.assertFalse(bool(value[3].any()))
                # Global-ID EP keeps the existing staged/merged provider path.
                untouched = {}
                self.assertFalse(
                    temp._prewarm_per_expert_lora_routes(
                        x,
                        a,
                        b,
                        ids,
                        mapping,
                        untouched,
                        shrink_lora_b=shared_b,
                        local_num_experts=experts - 1,
                    )
                )
                self.assertEqual(untouched, {})
                self.assertTrue(
                    temp._prewarm_per_expert_lora_routes(
                        x[:0],
                        a,
                        b,
                        ids[:0],
                        mapping[:0],
                        untouched,
                        shrink_lora_b=shared_b,
                        adapter_enabled=enabled,
                    )
                )
                self.assertEqual(untouched, {})

                if tokens != 6:
                    continue
                # Graph replays mutate borrowed contents, not identities or cache keys.
                initial_ids, initial_mapping = ids.clone(), mapping.clone()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    replay_cache = {}
                    prepare(replay_cache)
                for state in ("partial", "disabled", "missing", "zero", "recovery"):
                    ids.copy_(initial_ids)
                    mapping.copy_(initial_mapping)
                    enabled.fill_(1)
                    if state == "partial":
                        ids[0, 0] = -2
                        ids[1, 1] = experts
                    elif state == "disabled":
                        enabled.zero_()
                    elif state == "missing":
                        mapping[0] = -1
                        mapping[1] = slots + 1
                    elif state == "zero":
                        ids.fill_(-1)
                    saved_ids, saved_mapping = ids.clone(), mapping.clone()
                    graph.replay()
                    check(replay_cache)
                    torch.testing.assert_close(ids, saved_ids, rtol=0, atol=0)
                    torch.testing.assert_close(mapping, saved_mapping, rtol=0, atol=0)

    @unittest.skipIf(is_xpu(), "CUDA staged virtual-expert helpers required")
    def test_empty_virtual_calls_preserve_stage_results(self):
        """Empty route sets preserve add outputs and the staged shrink return contract."""
        import sglang.kernels.ops.lora.moe.trtllm_lora_temp.virtual_experts as temp
        import sglang.kernels.ops.lora.moe.virtual_experts as stock
        from sglang.srt.layers.moe.moe_runner.triton_utils import (
            fused_moe_triton_config,
        )

        hidden, width, rank = 32, 16, 16
        a = torch.empty((1, 2, rank, hidden), device=self.device, dtype=torch.float16)
        b = torch.empty((1, 2, width, rank), device=self.device, dtype=torch.float16)
        with patch.object(
            fused_moe_triton_config,
            "try_get_optimal_moe_config",
            side_effect=AssertionError(
                "empty routes must return before tuning or routing"
            ),
        ):
            for tokens, topk in ((0, 2), (3, 0)):
                with self.subTest(tokens=tokens, topk=topk):
                    backing = torch.full(
                        (3, 2, width), 0.75, device=self.device, dtype=torch.float16
                    )
                    saved = backing.clone()
                    output = backing[:tokens, :topk]
                    cache_marker = object()
                    cache = {"untouched": cache_marker}
                    kwargs = dict(
                        output=output,
                        hidden_states=torch.empty(
                            (tokens, hidden), device=self.device, dtype=a.dtype
                        ),
                        lora_a=a,
                        lora_b=b,
                        topk_ids=torch.empty(
                            (tokens, topk), device=self.device, dtype=torch.int32
                        ),
                        topk_weights=torch.empty(
                            (tokens, topk), device=self.device, dtype=torch.float32
                        ),
                        token_lora_mapping=torch.empty(
                            tokens, device=self.device, dtype=torch.int32
                        ),
                        mul_routed_weight=True,
                        experts_shared_outer_loras_a=False,
                        experts_shared_outer_loras_b=False,
                        routing_cache=cache,
                    )
                    self.assertIsNone(stock.merged_experts_fused_moe_lora_add(**kwargs))
                    for stage in ("routing", "all", "shrink", "expand"):
                        for provide_buffer in (False, True):
                            with self.subTest(
                                stage=stage, provide_buffer=provide_buffer
                            ):
                                buffer = (
                                    torch.empty(
                                        (tokens, topk, rank),
                                        device=self.device,
                                        dtype=a.dtype,
                                    )
                                    if provide_buffer
                                    else None
                                )
                                call_kwargs = dict(
                                    kwargs, stage=stage, intermediate_buffer=buffer
                                )
                                if stage == "expand" and buffer is None:
                                    with self.assertRaisesRegex(
                                        AssertionError, "requires intermediate_buffer"
                                    ):
                                        temp.merged_experts_fused_moe_lora_add(
                                            **call_kwargs
                                        )
                                    continue
                                result = temp.merged_experts_fused_moe_lora_add(
                                    **call_kwargs
                                )
                                if stage == "shrink":
                                    self.assertEqual(result.shape, (tokens, topk, rank))
                                    self.assertEqual(result.dtype, a.dtype)
                                    self.assertEqual(result.device, a.device)
                                    if buffer is not None:
                                        self.assertIs(result, buffer)
                                else:
                                    self.assertIsNone(result)
                    if tokens:
                        reduced = backing[:, 0]
                        self.assertIsNone(
                            temp.merged_experts_fused_moe_lora_add(
                                **dict(kwargs, output=reduced, fuse_sum_all_reduce=True)
                            )
                        )
                    with self.assertRaisesRegex(AssertionError, "invalid stage"):
                        temp.merged_experts_fused_moe_lora_add(
                            **dict(kwargs, stage="invalid")
                        )
                    self.assertEqual(cache, {"untouched": cache_marker})
                    torch.testing.assert_close(backing, saved, rtol=0, atol=0)

    def test_negative_sentinels_preserved(self):
        """Virtual scratch IDs stay bounded without changing borrowed route metadata."""
        for dtype, max_loras in product((torch.int32, torch.int64), (1, 4)):
            with self.subTest(dtype=dtype, max_loras=max_loras):
                self._check_remap_domain(dtype, max_loras)

    def _check_remap_domain(self, dtype, max_loras):
        num_experts = 16
        wide = 2**40 if dtype == torch.int64 else 2**30
        topk_ids = torch.tensor(
            [
                [3, -1],
                [-3, 5],
                [0, 7],
                [-1, -1],
                [2, 9],
                [11, -2],
                [-1, 4],
                [6, -1],
                [16, 17],
                [wide, -wide],
                [torch.iinfo(dtype).min, torch.iinfo(dtype).max],
                [0, 15],
            ],
            dtype=dtype,
            device=self.device,
        )
        token_lora_mapping = torch.tensor(
            [0, 1, 0, 2, -1, 1, 0, 1, 0, 1, 4, 5],
            dtype=torch.int32,
            device=self.device,
        )
        original_ids = topk_ids.clone()
        original_mapping = token_lora_mapping.clone()
        providers = {"base": _fused_virtual_topk_ids}
        if is_cuda():
            from sglang.kernels.ops.lora.moe.trtllm_lora_temp.virtual_experts import (
                _fused_virtual_topk_ids as trtllm_virtual_ids,
            )

            providers["trtllm"] = trtllm_virtual_ids
        for (provider, remap), shared_outer, use_enabled in product(
            providers.items(), (False, True), (False, True)
        ):
            with self.subTest(
                provider=provider, shared_outer=shared_outer, enabled=use_enabled
            ):
                enabled = (
                    torch.ones(max_loras * 2, device=self.device, dtype=torch.int32)[
                        ::2
                    ]
                    if use_enabled
                    else None
                )
                if enabled is not None and max_loras > 1:
                    enabled[1] = 0

                def run():
                    return remap(
                        topk_ids,
                        token_lora_mapping,
                        num_experts,
                        shared_outer=shared_outer,
                        max_loras=max_loras,
                        adapter_enabled=enabled,
                    )

                virtual_ids, mask, virtual_experts = run()
                graph = None
                if is_cuda():
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        virtual_ids, mask, virtual_experts = run()
                for state in ("mixed", "disabled", "inactive", "recovered"):
                    topk_ids.copy_(original_ids)
                    token_lora_mapping.copy_(original_mapping)
                    if enabled is not None:
                        enabled.fill_(1)
                        if max_loras > 1:
                            enabled[1] = 0
                    if state == "disabled":
                        if enabled is not None:
                            token_lora_mapping.zero_()
                            enabled.zero_()
                        else:
                            token_lora_mapping.fill_(-1)
                    elif state == "inactive":
                        topk_ids.fill_(-1)
                    saved_ids, saved_mapping = (
                        topk_ids.clone(),
                        token_lora_mapping.clone(),
                    )
                    if graph is None:
                        virtual_ids, mask, virtual_experts = run()
                    else:
                        graph.replay()
                    expected = torch.full_like(topk_ids, -1)
                    for row, slot in enumerate(saved_mapping.cpu().tolist()):
                        active = 0 <= slot < max_loras and (
                            enabled is None or bool(enabled[slot])
                        )
                        self.assertEqual(bool(mask[row]), active)
                        for route, expert in enumerate(saved_ids[row].cpu().tolist()):
                            if (
                                active
                                and expert >= 0
                                and (shared_outer or expert < num_experts)
                            ):
                                expected[row, route] = (
                                    slot
                                    if shared_outer
                                    else expert + slot * num_experts
                                )
                    torch.testing.assert_close(virtual_ids, expected, rtol=0, atol=0)
                    self.assertTrue(
                        bool(
                            (
                                (virtual_ids == -1)
                                | ((virtual_ids >= 0) & (virtual_ids < virtual_experts))
                            ).all()
                        )
                    )
                    torch.testing.assert_close(topk_ids, saved_ids, rtol=0, atol=0)
                    torch.testing.assert_close(
                        token_lora_mapping, saved_mapping, rtol=0, atol=0
                    )

    @unittest.skipUnless(is_cuda(), "Merged LoRA alignment requires CUDA")
    def test_shared_outer_merged_alignment_skips_original_negative_routes(self):
        """Shared-outer blocks contain exactly live original slots after remapping."""
        from sglang.kernels.ops.lora.moe.trtllm_lora_temp.moe_lora_merged_align import (
            moe_lora_merged_align,
        )

        topk_ids = torch.tensor(
            [[-1, 2, 0], [-1, -1, -1], [1, -3, 3]],
            device=self.device,
            dtype=torch.int32,
        )
        live_ids = topk_ids.clone()
        mapping = torch.tensor([0, 1, 2], device=self.device, dtype=torch.int32)
        block_size = 16
        for fuse_scatter in (False, True):
            for state in ("mixed", "idle", "remapped"):
                with self.subTest(fuse_scatter=fuse_scatter, state=state):
                    topk_ids.copy_(live_ids)
                    mapping.copy_(
                        torch.tensor(
                            [2, 0, 1] if state == "remapped" else [0, 1, 2],
                            device=self.device,
                            dtype=torch.int32,
                        )
                    )
                    if state == "idle":
                        topk_ids.fill_(-1)
                    original_ids = topk_ids.clone()
                    original_mapping = mapping.clone()
                    sorted_ids, experts, count, _, _ = moe_lora_merged_align(
                        topk_ids,
                        mapping,
                        num_experts=4,
                        shared_outer=True,
                        max_loras=3,
                        block_size=block_size,
                        do_skip=True,
                        fuse_scatter=fuse_scatter,
                        return_inactive_prefix_count=True,
                    )
                    self.assertEqual(count.shape, (2,))
                    self.assertEqual(count.data_ptr() % 8, 0)
                    self.assertEqual(int(count[1]), 0)
                    used = int(count[0])
                    self.assertGreaterEqual(used, 0)
                    self.assertEqual(used % block_size, 0)
                    self.assertLessEqual(used, sorted_ids.numel())
                    blocks = sorted_ids[:used].view(-1, block_size)
                    self.assertTrue(
                        bool(((blocks >= 0) & (blocks <= topk_ids.numel())).all())
                    )
                    emitted = blocks[blocks < topk_ids.numel()]
                    expected = torch.nonzero(topk_ids.flatten() >= 0).flatten()
                    self.assertEqual(
                        sorted(emitted.cpu().tolist()), expected.cpu().tolist()
                    )
                    for index, block in enumerate(blocks):
                        flat_ids = block[block < topk_ids.numel()].long()
                        tokens = flat_ids // topk_ids.shape[1]
                        self.assertTrue(bool((topk_ids.flatten()[flat_ids] >= 0).all()))
                        self.assertTrue(
                            bool((original_mapping[tokens] == experts[index]).all())
                        )
                    torch.testing.assert_close(topk_ids, original_ids, rtol=0, atol=0)
                    torch.testing.assert_close(
                        mapping, original_mapping, rtol=0, atol=0
                    )
            topk_ids.copy_(live_ids)
            _, experts, count, _, _ = moe_lora_merged_align(
                topk_ids,
                mapping,
                num_experts=4,
                shared_outer=True,
                max_loras=3,
                block_size=block_size,
                do_skip=False,
                fuse_scatter=fuse_scatter,
                return_inactive_prefix_count=True,
            )
            expected_prefix = (int((topk_ids < 0).sum()) + block_size - 1) // block_size
            self.assertEqual(int(count[1]), expected_prefix)
            self.assertTrue(bool((experts[:expected_prefix] == -1).all()))
            ep_ids = torch.tensor(
                [[-1, 4, 5], [-3, -1, 4]], device=self.device, dtype=torch.int32
            )
            ep_mapping = torch.zeros(2, device=self.device, dtype=torch.int32)
            sorted_ep, ep_experts, ep_count, _, _ = moe_lora_merged_align(
                ep_ids,
                ep_mapping,
                num_experts=8,
                shared_outer=False,
                max_loras=1,
                block_size=block_size,
                local_expert_offset=4,
                local_num_experts=2,
                do_skip=False,
                compact=True,
                fuse_scatter=fuse_scatter,
                return_inactive_prefix_count=True,
            )
            self.assertEqual(int(ep_count[1]), 1)
            used_blocks = sorted_ep[: int(ep_count[0])].view(-1, block_size)
            for index, block in enumerate(used_blocks):
                live = block[block < ep_ids.numel()].long()
                actual = ep_ids.flatten()[live]
                expected = torch.where(actual < 0, -1, actual)
                self.assertTrue(bool((expected == ep_experts[index]).all()))

    @unittest.skipUnless(is_cuda(), "Merged LoRA alignment requires CUDA")
    def test_merged_alignment_raw_domains_on_replay(self):
        """Raw integer extremes and missing adapters never become real weight IDs."""
        from sglang.kernels.ops.lora.moe.trtllm_lora_temp.moe_lora_merged_align import (
            moe_lora_merged_align,
        )

        num_experts, slots, block_size = 4, 3, 8
        for dtype, shared_outer, use_enabled, fuse_scatter in product(
            (torch.int32, torch.int64), (False, True), (False, True), (False, True)
        ):
            with self.subTest(
                dtype=dtype,
                shared=shared_outer,
                enabled=use_enabled,
                fused=fuse_scatter,
            ):
                wide = 2**40 if dtype == torch.int64 else 2**30
                ids = torch.tensor(
                    [
                        [0, 1],
                        [-3, 2],
                        [4, 5],
                        [wide, 0],
                        [0, 1],
                        [2, -1],
                        [torch.iinfo(dtype).min, torch.iinfo(dtype).max],
                    ],
                    device=self.device,
                    dtype=dtype,
                )
                mapping = torch.tensor(
                    [0, 1, 2, 0, -1, slots, 1], device=self.device, dtype=torch.int32
                )
                original_ids, original_mapping = ids.clone(), mapping.clone()
                enabled = (
                    torch.ones(slots * 2, device=self.device, dtype=torch.int32)[::2]
                    if use_enabled
                    else None
                )

                def run():
                    return moe_lora_merged_align(
                        ids,
                        mapping,
                        num_experts,
                        shared_outer,
                        slots,
                        block_size,
                        do_skip=False,
                        fuse_scatter=fuse_scatter,
                        adapter_enabled=enabled,
                        return_inactive_prefix_count=True,
                    )

                run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    sorted_ids, experts, count, adapter_mask, virtual_experts = run()
                for state in ("mixed", "disabled", "inactive", "recovered"):
                    ids.copy_(original_ids)
                    mapping.copy_(original_mapping)
                    if enabled is not None:
                        enabled.fill_(1)
                        enabled[1] = 0
                    if state == "disabled":
                        mapping.fill_(-1)
                        if enabled is not None:
                            enabled.zero_()
                    elif state == "inactive":
                        ids.fill_(-1)
                    saved_ids, saved_mapping = ids.clone(), mapping.clone()
                    graph.replay()
                    groups = {}
                    for row, slot in enumerate(saved_mapping.cpu().tolist()):
                        active = 0 <= slot < slots and (
                            enabled is None or bool(enabled[slot])
                        )
                        self.assertEqual(bool(adapter_mask[row]), active)
                        for route, expert in enumerate(saved_ids[row].cpu().tolist()):
                            virtual = -1
                            if (
                                active
                                and expert >= 0
                                and (shared_outer or expert < num_experts)
                            ):
                                virtual = (
                                    slot
                                    if shared_outer
                                    else expert + slot * num_experts
                                )
                            groups.setdefault(virtual, []).append(
                                row * ids.shape[1] + route
                            )
                    expected_total = sum(
                        ((len(group) + block_size - 1) // block_size) * block_size
                        for group in groups.values()
                    )
                    expected_prefix = (
                        len(groups.get(-1, [])) + block_size - 1
                    ) // block_size
                    self.assertEqual(
                        count.cpu().tolist(), [expected_total, expected_prefix]
                    )
                    self.assertLessEqual(expected_total, sorted_ids.numel())
                    expected_experts, offset = [], 0
                    for expert, indices in sorted(groups.items()):
                        padded = (
                            (len(indices) + block_size - 1) // block_size
                        ) * block_size
                        actual = sorted_ids[offset : offset + padded].cpu()
                        self.assertEqual(
                            sorted(actual[actual < ids.numel()].tolist()), indices
                        )
                        self.assertTrue(
                            bool((actual[actual >= ids.numel()] == ids.numel()).all())
                        )
                        expected_experts += [expert] * (padded // block_size)
                        offset += padded
                    self.assertEqual(
                        experts[: expected_total // block_size].cpu().tolist(),
                        expected_experts,
                    )
                    self.assertTrue(
                        all(
                            expert == -1 or 0 <= expert < virtual_experts
                            for expert in expected_experts
                        )
                    )
                    torch.testing.assert_close(ids, saved_ids, rtol=0, atol=0)
                    torch.testing.assert_close(mapping, saved_mapping, rtol=0, atol=0)

    @unittest.skipUnless(is_cuda(), "Stock virtual LoRA hook fixture requires CUDA")
    def test_gate_up_hook_uses_each_stacked_a_projection(self):
        """One shrink/expand pairs independent gate/up ranks, including ragged halves."""
        from sglang.kernels.ops.lora.moe import virtual_experts
        from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
        from sglang.srt.lora.lora_moe_runners import build_lora_hooks

        tokens, topk, experts, rank, hidden = 3, 2, 2, 16, 64
        dtype = torch.float16
        x = torch.zeros((tokens, hidden), device=self.device, dtype=dtype)
        x[:, 0] = torch.arange(1, tokens + 1, device=self.device, dtype=dtype)
        ids = torch.tensor(
            [[-1, 1], [0, 1], [-1, -1]], device=self.device, dtype=torch.int32
        )
        weights = torch.tensor(
            [[float("nan"), 0.5], [0.25, 0.75], [float("nan"), float("nan")]],
            device=self.device,
        )
        mapping = torch.tensor([0, 1, 0], device=self.device, dtype=torch.int32)
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 2,
        }
        real_add = virtual_experts.merged_experts_fused_moe_lora_add
        # Split-A, common-A and non-gated layouts retain one full-width call.
        for (a_parts, b_parts), intermediate in product(
            ((2, 2), (1, 2), (1, 1)), (17, 32, 48)
        ):
            for shared_outer in (False, True):
                with self.subTest(
                    a_parts=a_parts,
                    b_parts=b_parts,
                    intermediate=intermediate,
                    shared_outer=shared_outer,
                ):
                    a = torch.zeros(
                        (2, 1 if shared_outer else experts, a_parts * rank, hidden),
                        device=self.device,
                        dtype=dtype,
                    )
                    b = torch.zeros(
                        (2, experts, b_parts * intermediate, rank),
                        device=self.device,
                        dtype=dtype,
                    )
                    for slot in range(2):
                        for expert in range(a.shape[1]):
                            a[slot, expert, 0, 0] = 2 + slot + expert
                            if a_parts == 2:
                                a[slot, expert, rank, 0] = 4 + slot + expert
                        for expert in range(experts):
                            b[slot, expert, :intermediate, 0] = 5 + slot + expert
                            if b_parts == 2:
                                b[slot, expert, intermediate:, 0] = 7 + slot + expert
                    output = torch.full(
                        (tokens, topk, b_parts * intermediate),
                        0.5,
                        device=self.device,
                        dtype=dtype,
                    )
                    safe_ids = ids.clamp_min(0)
                    a_ids = torch.zeros_like(ids) if shared_outer else safe_ids
                    selected_a = a[mapping[:, None], a_ids]
                    selected_b = b[mapping[:, None], safe_ids]
                    shrunk = torch.einsum(
                        "mh,mkrh->mkr", x.float(), selected_a.float()
                    ).to(dtype)
                    if a_parts == 2:
                        delta = torch.einsum(
                            "mkpr,mkpir->mkpi",
                            shrunk.view(tokens, topk, 2, rank).float(),
                            selected_b.view(
                                tokens, topk, 2, intermediate, rank
                            ).float(),
                        ).flatten(2)
                    else:
                        delta = torch.einsum(
                            "mkr,mkir->mki", shrunk.float(), selected_b.float()
                        )
                    delta = delta.to(dtype).masked_fill((ids < 0).unsqueeze(-1), 0)
                    expected = output + delta
                    info = SimpleNamespace(
                        max_lora_rank=rank,
                        has_active_lora=True,
                        lora_use_virtual_experts=True,
                        experts_shared_outer_loras=shared_outer,
                        gate_up_lora_a_weights=a,
                        gate_up_lora_b_weights=b,
                        token_lora_mapping=mapping,
                        adapter_enabled=torch.ones(
                            2, device=self.device, dtype=torch.int32
                        ),
                    )
                    calls = []

                    def observed_add(**kwargs):
                        projected = kwargs["output"]
                        self.assertEqual(projected.stride(-2), output.shape[-1])
                        self.assertEqual(
                            projected.untyped_storage().data_ptr(),
                            output.untyped_storage().data_ptr(),
                        )
                        calls.append(
                            (
                                projected.data_ptr(),
                                kwargs["lora_a"].shape[2],
                                kwargs.get("a_output_slices", 1),
                            )
                        )
                        return real_add(**kwargs)

                    with (
                        override_config(config),
                        patch(
                            "sglang.srt.lora.lora_moe_runners.get_is_capture_mode",
                            return_value=False,
                        ),
                        patch.object(
                            virtual_experts,
                            "merged_experts_fused_moe_lora_add",
                            observed_add,
                        ),
                    ):
                        hooks = build_lora_hooks(x, info, ids)
                        hooks.after_gate_up(x, output, weights, ids)
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    self.assertEqual(
                        calls, [(output.data_ptr(), a_parts * rank, a_parts)]
                    )

    def test_positive_topk_remapped_correctly(self):
        """Sanity: valid (non-negative) IDs follow the
        `base + safe_lora * num_experts` rule."""
        topk_ids = torch.tensor(
            [[3, 1], [0, 7], [2, 9]], dtype=torch.int32, device=self.device
        )
        token_lora_mapping = torch.tensor(
            [0, 1, 2], dtype=torch.int32, device=self.device
        )
        num_experts = 16
        max_loras = 4

        virtual_ids, _, _ = _fused_virtual_topk_ids(
            topk_ids, token_lora_mapping, num_experts, False, max_loras
        )

        for m in range(topk_ids.shape[0]):
            lora = token_lora_mapping[m].item()
            for k in range(topk_ids.shape[1]):
                base = topk_ids[m, k].item()
                expected = base + max(lora, 0) * num_experts
                self.assertEqual(virtual_ids[m, k].item(), expected)

    def test_no_lora_token_has_no_virtual_expert(self):
        """A missing adapter never occupies a virtual expert, including slot zero."""
        topk_ids = torch.tensor([[3, 5]], dtype=torch.int32, device=self.device)
        token_lora_mapping = torch.tensor([-1], dtype=torch.int32, device=self.device)
        num_experts = 16

        virtual_ids, mask, _ = _fused_virtual_topk_ids(
            topk_ids, token_lora_mapping, num_experts, False, max_loras=4
        )
        torch.testing.assert_close(virtual_ids, torch.full_like(virtual_ids, -1))
        torch.testing.assert_close(
            topk_ids, torch.tensor([[3, 5]], device=self.device, dtype=torch.int32)
        )
        self.assertFalse(bool(mask[0].item()))


class _AlignBlockSizeSentinelBucketBase(CustomTestCase):
    """Shared tests for both the torch.compile and JIT align_block_size paths.

    Subclasses override ``_align`` to select the concrete implementation.
    """

    @classmethod
    def setUpClass(cls):
        if cls is _AlignBlockSizeSentinelBucketBase:
            raise unittest.SkipTest("Base class")
        cls.device = _require_accelerator_device()

    def _align(self, topk_ids, block_size, num_experts):
        raise NotImplementedError

    def _align_used(self, topk_ids, block_size, num_experts):
        sorted_ids, experts, count = self._align(topk_ids, block_size, num_experts)
        used = int(count[0])
        self.assertGreaterEqual(used, 0)
        self.assertEqual(used % block_size, 0)
        self.assertLessEqual(used, sorted_ids.numel())
        self.assertLessEqual(used // block_size, experts.numel())
        return sorted_ids, experts[: used // block_size], count

    @staticmethod
    def _assigned_experts(expert_ids: torch.Tensor) -> list:
        """Return the list of real expert ids assigned to blocks (filtering
        out -1 sentinels for padding/exclusion)."""
        return expert_ids[expert_ids != -1].cpu().tolist()

    def _assert_only_real_or_sentinel(self, expert_ids: torch.Tensor, num_experts: int):
        for eid in expert_ids.cpu().tolist():
            self.assertTrue(
                eid == -1 or 0 <= eid < num_experts,
                f"expert_ids contains invalid value {eid}",
            )

    def test_all_valid_baseline(self):
        """Sanity: with no invalid IDs, every present real expert appears
        in the assignment, and no junk values leak through."""
        num_experts = 8
        block_size = 16
        topk_ids = torch.tensor(
            [[0, 3], [4, 7], [1, 2], [5, 6]],
            dtype=torch.int32,
            device=self.device,
        )

        _, expert_ids, _ = self._align_used(topk_ids, block_size, num_experts)

        self._assert_only_real_or_sentinel(expert_ids, num_experts)
        assigned = set(self._assigned_experts(expert_ids))
        self.assertEqual(assigned, set(range(num_experts)))

    def test_negative_ids_routed_to_sentinel(self):
        """`-1` tokens must not appear as real expert assignments and must
        not corrupt the assignment of real IDs."""
        num_experts = 8
        block_size = 16
        topk_ids = torch.tensor(
            [[0, -1], [-1, 7], [1, -1], [-1, -1]],
            dtype=torch.int32,
            device=self.device,
        )

        _, expert_ids, _ = self._align_used(topk_ids, block_size, num_experts)

        self._assert_only_real_or_sentinel(expert_ids, num_experts)
        assigned = self._assigned_experts(expert_ids)
        for valid_eid in (0, 1, 7):
            self.assertIn(valid_eid, assigned)

    def test_oor_ids_routed_to_sentinel(self):
        """IDs `>= num_experts` (e.g. virtual-experts remap when combined
        with non-local sentinels) must not break cumsum/searchsorted and
        must not show up as real assignments."""
        num_experts = 8
        block_size = 16
        topk_ids = torch.tensor(
            [[0, 100], [50, 7], [1, 200]],
            dtype=torch.int32,
            device=self.device,
        )

        _, expert_ids, _ = self._align_used(topk_ids, block_size, num_experts)

        self._assert_only_real_or_sentinel(expert_ids, num_experts)
        assigned = self._assigned_experts(expert_ids)
        for valid_eid in (0, 1, 7, 50):
            if valid_eid >= num_experts:
                self.assertNotIn(valid_eid, assigned)
            else:
                self.assertIn(valid_eid, assigned)

    def test_mixed_invalid_at_production_size(self):
        """Mirror the production trigger: `num_experts >= 1024` (only path
        where the large-expert fallback is invoked instead of the native
        align kernel)."""
        num_experts = 1500
        block_size = 16
        topk_ids = torch.tensor(
            [
                [-1, 500],
                [num_experts + 7, 1000],
                [num_experts * 2, 100],
                [-1, 0],
            ],
            dtype=torch.int32,
            device=self.device,
        )

        _, expert_ids, _ = self._align_used(topk_ids, block_size, num_experts)

        self._assert_only_real_or_sentinel(expert_ids, num_experts)
        assigned = self._assigned_experts(expert_ids)
        for valid_eid in (0, 100, 500, 1000):
            self.assertIn(valid_eid, assigned)

    def test_empty_topk_ids_does_not_crash(self):
        """Edge: empty input. Should return empty/zero outputs without
        OOB indexing on the sentinel bucket."""
        num_experts = 8
        block_size = 16
        topk_ids = torch.empty((0, 2), dtype=torch.int32, device=self.device)

        sorted_token_ids, expert_ids, num_post_padded = self._align_used(
            topk_ids, block_size, num_experts
        )

        self.assertEqual(num_post_padded.item(), 0)
        self.assertEqual(self._assigned_experts(expert_ids), [])


class TestAlignBlockSizeTorchSentinelBucket(_AlignBlockSizeSentinelBucketBase):
    """Test the pure-PyTorch torch.compile fallback path (AMD/ROCm compatible)."""

    def _align(self, topk_ids, block_size, num_experts):
        return _align_block_size_torch(topk_ids, block_size, num_experts)


@unittest.skipIf(
    is_xpu(),
    "_align_block_size_jit builds a CUDA JIT kernel via tvm_ffi.load_inline "
    "(requires a CUDA/nvcc install); it cannot run on XPU. The torch variant "
    "(TestAlignBlockSizeTorchSentinelBucket) covers the same alignment logic "
    "on that platform.",
)
class TestAlignBlockSizeJitSentinelBucket(_AlignBlockSizeSentinelBucketBase):
    """Test raw-ID CUDA alignment without a post-hoc expert-ID sanitizer."""

    def _align(self, topk_ids, block_size, num_experts):
        return _align_block_size_jit(topk_ids, block_size, num_experts)


if __name__ == "__main__":
    unittest.main(verbosity=2)
