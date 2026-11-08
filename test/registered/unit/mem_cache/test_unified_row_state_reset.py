"""A request row goes back the moment its request ends, while the forward that
ran the request last may still advance the row's speculative ring cursors. The
row's next owner must start from clean cursors all the same: its request memory
is reset in its first forward, on the forward stream, after the old forward's
last access, never on the scheduling stream when the row is handed out."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.kv.alloc.forward_steps import forward_steps_of
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import (
    MHAGeometry,
    StateGeometry,
    StateReplay,
    kv_layer_decls,
    with_state_replay,
)
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.runtime_context import mamba_cache_chunk_size
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cuda_ci(est_time=10, stage="extra-a", runner_config="1-gpu-small")

_CURSORS = ("replayssm_spec_write_pos", "replayssm_cache_base", "replayssm_is_flush")
# Long enough for the scheduling side of the test to run while it sleeps.
_HOLD_CYCLES = 2_000_000_000


def _storage():
    setup_allocator_context()
    dense = MHAGeometry(
        head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16
    )
    geometry = with_state_replay(
        StateGeometry(
            conv_state_shapes=((3, 8),),
            conv_dtype=torch.bfloat16,
            temporal_state_shape=(1, 4, 8),
            temporal_dtype=torch.float32,
        ),
        StateReplay(
            decode=False, speculative=True, is_kda=False, cache_len=8, key_heads=1
        ),
        layer_ids=(1, 2),
        speculative_num_draft_tokens=3,
    )
    plan = plan_kv_storage(
        kv_layer_decls(
            full_layer_ids=(0,),
            full_geometry=dense,
            window_layer_ids=(),
            window_geometry=dense,
            window=16,
            state_layer_ids=(1, 2),
            state_geometry=geometry,
            checkpoint_interval=mamba_cache_chunk_size(),
        ),
        page_size=4,
        speculative_num_draft_tokens=3,
    )
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=256,
            window_tokens=128,
            state_slots=16,
            max_running_requests=3,
            context_len=64,
            alignment=1,
        ),
    )
    ctx = UnifiedBuildContext(
        device="cuda",
        start_layer=0,
        end_layer=3,
        enable_memory_saver=False,
        need_sort=False,
        max_num_reqs=3,
        model_context_len=64,
        lazy_compaction=True,
        enable_mamba_extra_buffer=False,
        state_cache_params=SimpleNamespace(layers=(1, 2)),
        speculative_num_draft_tokens=3,
    )
    return build_kv_storage(plan, sized, ctx)


def _request(rid):
    return SimpleNamespace(rid=rid, kv=ReqKvInfo())


def _leave_free_slots(storage):
    """Two requests run and end with nothing in flight, a third stays: their
    rows and the state band's free list hold two quiet slots, so a request
    taking a slot gets a quiet one rather than one freed under a forward in
    flight. This also loads the kernels of a row's lifecycle: a first launch
    loads its module, which waits for the whole device."""
    reqs = [_request(f"warmup{i}") for i in range(3)]
    storage.acquire(reqs)
    storage.reset_new_rows()
    for req in reqs[:2]:
        storage.release_state("mamba", req)
        storage.release_row(req)
    torch.cuda.synchronize()
    return reqs[2]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRowStateResetOnReuse(CustomTestCase):
    def test_reused_row_starts_clean_after_old_forward(self):
        storage = _storage()
        steps = forward_steps_of(storage.allocator)
        self.assertIsNotNone(steps)
        memory = storage.request_memory
        parts = {p.name.removeprefix("mamba."): p for p in memory.parts}
        for name in (*_CURSORS, "replayssm_d"):
            self.assertIn(name, parts)
        tensors = {name: memory.tensors[f"mamba.{name}"] for name in parts}
        schedule = torch.cuda.current_stream()
        forward = torch.cuda.Stream()
        _leave_free_slots(storage)

        old = _request("old")
        (row,) = storage.acquire([old])
        old_slot = storage.allocator.state.slots.translate(old.kv.mamba_pool_idx)
        with torch.cuda.stream(forward):
            storage.reset_new_rows()

        # The new request's first forward is a graph that reads the cursors
        # and ring through persistent row ids, as a decode replay does, then
        # advances its cursor.
        graph_rows = torch.tensor([row], device="cuda")
        seen = {
            name: torch.empty_like(
                tensor.index_select(parts[name].row_axis, graph_rows)
            )
            for name, tensor in tensors.items()
        }
        graph = torch.cuda.CUDAGraph()
        capture = torch.cuda.Stream()
        capture.wait_stream(schedule)
        with torch.cuda.stream(capture):
            with torch.cuda.graph(graph, stream=capture):
                for name, tensor in tensors.items():
                    part = parts[name]
                    seen[name].copy_(tensor.index_select(part.row_axis, graph_rows))
                tensors["replayssm_spec_write_pos"].index_fill_(0, graph_rows, 4)
        torch.cuda.synchronize()

        # The old request's last forward: its cursor commit (and ring write)
        # held back on the forward stream.
        steps.forward_launched()
        forward.wait_stream(schedule)
        with torch.cuda.stream(forward):
            rows = torch.tensor([row], device="cuda")
            torch.cuda._sleep(_HOLD_CYCLES)
            for name, tensor in tensors.items():
                tensor.index_fill_(parts[name].row_axis, rows, 9)
            old_done = torch.cuda.Event()
            old_done.record()
        steps.forward_recorded(old_done)

        # The old request ends while that forward runs: its row and state
        # slot go back at once, and the new request takes the row.
        storage.release_state("mamba", old)
        storage.release_row(old)
        new = _request("new")
        self.assertEqual(storage.acquire([new]), [row])
        # Only the row is shared: the new request's state slot sits on other
        # physical pages than the old one, whose pages wait for the forward.
        new_slot = storage.allocator.state.slots.translate(new.kv.mamba_pool_idx)
        self.assertNotEqual(int(new_slot), int(old_slot))
        schedule.synchronize()
        self.assertFalse(old_done.query(), "the old forward ended too early")

        # The new request's first forward, on the forward stream.
        forward.wait_stream(schedule)
        with torch.cuda.stream(forward):
            storage.reset_new_rows()
            graph.replay()
        torch.cuda.synchronize()

        for name, tensor in tensors.items():
            with self.subTest(part=name):
                # The first forward sees a clean cursor and ring.
                self.assertTrue(torch.all(seen[name] == parts[name].reset_value))
        # The old forward's late write did not land on the new cursor.
        self.assertEqual(int(tensors["replayssm_spec_write_pos"][row]), 4)
        for name in ("replayssm_cache_base", "replayssm_is_flush"):
            self.assertEqual(int(tensors[name][row]), 0)

    def test_row_reset_waits_for_the_next_forward(self):
        storage = _storage()
        memory = storage.request_memory
        write_pos = memory.tensors["mamba.replayssm_spec_write_pos"]
        old = _request("old")
        (row,) = storage.acquire([old])
        storage.reset_new_rows()
        write_pos[row] = 6
        storage.release_state("mamba", old)
        storage.release_row(old)
        new = _request("new")
        self.assertEqual(storage.acquire([new]), [row])
        torch.cuda.synchronize()
        # Handing the row out writes nothing on the scheduling stream.
        self.assertEqual(int(write_pos[row]), 6)
        storage.reset_new_rows()
        torch.cuda.synchronize()
        self.assertEqual(int(write_pos[row]), 0)
        # A row is reset once: later forwards leave its new values alone.
        write_pos[row] = 2
        storage.reset_new_rows()
        torch.cuda.synchronize()
        self.assertEqual(int(write_pos[row]), 2)


if __name__ == "__main__":
    unittest.main()
