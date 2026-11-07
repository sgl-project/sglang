"""Which window id space an attention backend's window reads map through.

A fused draft whose window length differs from every target window keeps its
window layers in a band of its own, so its window storage is not the target's
window. `resolve_window_id_space` must name the same storage the runner's ids
resolve the sliding-window role to, and the backends that keep a window space
of their own must not read a unified pool through it untranslated.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.base_attn_backend import (
    resolve_window_id_space,
    resolve_window_kv_pool,
)
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MHAGeometry, WindowLayers, kv_layer_decls
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.layout.fused_draft import (
    LAYER_FULL,
    LAYER_WINDOW,
    DraftKVGeometry,
    DraftKVProfile,
    DraftLayerSet,
    PlacementContext,
    place_fused_draft,
)
from sglang.srt.mem_cache.unified_draft_pool import routed_draft_pool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cuda_ci(est_time=6, stage="base-b", runner_config="1-gpu-small")

_DTYPE = torch.bfloat16


def _geometry():
    return MHAGeometry(head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=_DTYPE)


def _unified(target_window, placement):
    decls = kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=_geometry(),
        windows=[
            WindowLayers(layer_ids=(1,), geometry=_geometry(), window=target_window)
        ],
    )
    plan = plan_kv_storage(decls, page_size=1, fused_draft=placement)
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=256,
            window_tokens=256,
            state_slots=0,
            max_running_requests=2,
            context_len=256,
            alignment=1,
        ),
    )
    ctx = UnifiedBuildContext(
        device="cuda",
        start_layer=0,
        end_layer=2,
        enable_memory_saver=False,
        need_sort=False,
        max_num_reqs=2,
        model_context_len=256,
        lazy_compaction=True,
    )
    return build_kv_storage(plan, sized, ctx), plan


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestWindowIdSpaceResolution(CustomTestCase):
    def _draft_runner(self, draft_window, target_window):
        setup_allocator_context()
        geometry = DraftKVGeometry(head_num=1, head_dim=8, v_head_dim=8)
        placement = place_fused_draft(
            profile=DraftKVProfile(
                num_layers=2,
                num_depths=1,
                kinds={
                    LAYER_FULL: DraftLayerSet(layer_ids=(0,), geometry=geometry),
                    LAYER_WINDOW: DraftLayerSet(
                        layer_ids=(1,), geometry=geometry, window=draft_window
                    ),
                },
            ),
            num_runners=1,
            ctx=PlacementContext(
                host_names=("full", f"window_{target_window}"),
                target_windows=(target_window,),
            ),
            store_dtype=_DTYPE,
        ).placement
        storage, plan = _unified(target_window, placement)
        allocator = storage.allocator
        hosts = list(placement.hosts())
        layers_by_host = {
            host: [1] if getattr(plan.sub_pool(host).lifetime, "window", None) else [0]
            for host in hosts
        }
        pool = routed_draft_pool(
            arena=allocator.arena,
            plan=plan,
            owner=plan.owner().name,
            page_size=1,
            lanes_by_host={
                host: dict(
                    zip(layers_by_host[host], list(placement.lanes_for(0, host)))
                )
                for host in hosts
            },
        )
        runner_ids = RunnerIds(
            req_to_token=storage.requests.req_to_token,
            token_to_kv_pool_allocator=allocator,
            token_to_kv_pool=pool,
            page_size=1,
            device="cuda",
            is_draft_worker=True,
        )
        runner = SimpleNamespace(
            token_to_kv_pool=pool,
            token_to_kv_pool_allocator=allocator,
            runner_ids=runner_ids,
            is_draft_worker=True,
            spec_algorithm=SimpleNamespace(is_frozen_kv_mtp=lambda: False),
        )
        return runner, allocator

    def test_a_draft_window_band_resolves_to_its_own_storage(self):
        """A draft window of another length than the target's lives in its own
        band; the window space a backend resolves for the draft runner is that
        band's, the one the runner's ids resolve the role to, not the target's
        window."""
        for draft_window, target_window in ((8, 16), (32, 16), (16, 16)):
            with self.subTest(draft_window=draft_window, target_window=target_window):
                runner, allocator = self._draft_runner(draft_window, target_window)
                pool = resolve_window_kv_pool(runner)
                self.assertIs(pool, runner.token_to_kv_pool)
                resolved = resolve_window_id_space(runner, pool)
                expected = runner.runner_ids.space(IdSpaceKind.SLIDING_WINDOW)
                self.assertIsNotNone(expected)
                self.assertEqual(
                    runner.runner_ids.sub_pool_name(IdSpaceKind.SLIDING_WINDOW),
                    f"window_{draft_window}",
                )
                self.assertEqual(resolved.key, expected.key)
                target = allocator.id_spaces()[allocator.roles.window]
                if draft_window != target_window:
                    self.assertNotEqual(resolved.key, target.key)


if __name__ == "__main__":
    unittest.main()
