"""A reused request row's request-scoped state starts clean and is never
reached by the previous request's late writes: under unified memory a row binds
a zeroed unit of the request sub-pool that no forward in flight touches; on
static pools the row is reset in its new request's first forward, on the
forward stream, after the previous request's last forward."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
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
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
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
            decode=False, speculative=True, is_kda=False, cache_len=16, key_heads=1
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


class _ModelLaunched(Exception):
    pass


def _forward(req_to_token_pool, token_to_kv_pool_allocator):
    """`ModelRunner.forward` up to the point it would launch the model."""
    runner = object.__new__(ModelRunner)
    runner.forward_pass_id = 0
    runner.req_to_token_pool = req_to_token_pool
    runner.token_to_kv_pool_allocator = token_to_kv_pool_allocator
    runner.msprobe_debugger = None
    runner.is_draft_worker = False
    runner.canary_manager = None

    def launch(*args):
        raise _ModelLaunched

    runner._forward_raw = launch
    batch = SimpleNamespace(
        apply_deprecated_skip_attn_backend_init=lambda _: None,
        forward_mode=ForwardMode.DECODE,
        batch_size=1,
    )
    try:
        runner.forward(batch)
    except _ModelLaunched:
        return
    raise AssertionError("the forward did not reach the model")


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


def _ring(storage):
    """The GDN speculative ring and cursors, by name, as views over the request
    sub-pool's units, and how a request row names its unit."""
    view = storage.view("mamba")
    tensors = {name: getattr(view, name) for name in _CURSORS}
    tensors["replayssm_d"] = view.mamba_cache.replayssm_d
    ids = storage.allocator.request_id_spaces()["request"]
    return tensors, lambda row: int(ids.ids(torch.tensor([row], device="cuda"))[0])


def _fill(tensors, unit, value):
    _fill_units(tensors, torch.tensor([unit], device="cuda"), value)


def _fill_units(tensors, units, value):
    for name, tensor in tensors.items():
        tensor.index_fill_(0 if name in _CURSORS else 1, units, value)


def _at(tensors, name, unit):
    tensor = tensors[name]
    return tensor[unit] if name in _CURSORS else tensor[:, unit]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRowStateResetOnReuse(CustomTestCase):
    def test_reused_row_starts_clean_after_old_forward(self):
        """A reused row's new request gets a zeroed unit of its own while the
        previous request's last forward still runs, and that forward's late
        write does not reach the new request's cursors or ring."""
        storage = _storage()
        steps = forward_steps_of(storage.allocator)
        self.assertIsNotNone(steps)
        tensors, unit_of = _ring(storage)
        schedule = torch.cuda.current_stream()
        forward = torch.cuda.Stream()
        _leave_free_slots(storage)

        old = _request("old")
        (row,) = storage.acquire([old])
        old_unit = unit_of(row)
        # Load the fill kernels now: a first launch waits for the whole device.
        # The ids are on the device before the hold: a copy from the host
        # behind it would wait for it.
        old_units = torch.tensor([old_unit], device="cuda")
        _fill_units(tensors, old_units, 0)
        torch.cuda.synchronize()

        # The old request's last forward: its cursor commit (and ring write)
        # held back on the forward stream.
        steps.forward_launched()
        forward.wait_stream(schedule)
        with torch.cuda.stream(forward):
            torch.cuda._sleep(_HOLD_CYCLES)
            _fill_units(tensors, old_units, 9)
            old_done = torch.cuda.Event()
            old_done.record()
        steps.forward_recorded(old_done)

        # The old request ends while that forward runs: its row goes back at
        # once, and the new request takes the row with another unit.
        storage.release_state("mamba", old)
        storage.release_row(old)
        new = _request("new")
        self.assertEqual(storage.acquire([new]), [row])
        new_unit = unit_of(row)
        self.assertNotEqual(new_unit, old_unit)
        schedule.synchronize()
        self.assertFalse(old_done.query(), "the old forward ended too early")
        for name in tensors:
            with self.subTest(part=name):
                self.assertTrue(torch.all(_at(tensors, name, new_unit) == 0))
        torch.cuda.synchronize()
        # The old forward's late write landed on its own unit only.
        for name in tensors:
            with self.subTest(part=name):
                self.assertTrue(torch.all(_at(tensors, name, new_unit) == 0))
                self.assertTrue(torch.all(_at(tensors, name, old_unit) == 9))

    def test_backup_before_first_forward_drops_previous_request_state(self):
        """A retraction backup taken before the request's first forward holds
        zeroed request state, not the row's previous request's; one taken after
        that forward keeps what the forward wrote."""
        for own_forward in (False, True):
            with self.subTest(own_forward=own_forward):
                storage = _storage()
                allocator = storage.allocator
                tensors, unit_of = _ring(storage)

                old = _request("old")
                (row,) = storage.acquire([old])
                _forward(storage.requests, allocator)
                _fill(tensors, unit_of(row), 7)
                storage.release_state("mamba", old)
                storage.release_row(old)

                req = _request("req")
                self.assertEqual(storage.acquire([req]), [row])
                if own_forward:
                    _forward(storage.requests, allocator)
                    _fill(tensors, unit_of(row), 3)
                saved = allocator.get_cpu_copy(
                    allocator.alloc(4),
                    mamba_indices=req.kv.mamba_pool_idx,
                    req_pool_index=row,
                )
                storage.release_state("mamba", req)
                storage.release_row(req)

                (new_row,) = storage.acquire([req])
                allocator.load_cpu_copy(
                    saved,
                    allocator.alloc(4),
                    mamba_indices=req.kv.mamba_pool_idx,
                    req_pool_index=new_row,
                )
                _forward(storage.requests, allocator)
                torch.cuda.synchronize()
                unit = unit_of(new_row)
                for name in tensors:
                    with self.subTest(part=name):
                        expected = 3 if own_forward else 0
                        self.assertTrue(torch.all(_at(tensors, name, unit) == expected))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestStaticRowStateReset(CustomTestCase):
    def test_reused_row_cursors_reset_in_next_forward(self):
        """A static pool's reused row keeps its ReplaySSM cursors when handed
        out; the next `ModelRunner.forward` resets them."""
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=64,
            n_groups=1,
            num_heads=2,
            head_dim=32,
            state_size=16,
            conv_kernel=4,
        )
        table = HybridReqToTokenPool(
            size=4,
            mamba_size=4,
            mamba_spec_state_size=4,
            max_context_len=64,
            device="cuda",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(shape=shape, layers=[0]),
            mamba_layer_ids=[0],
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
            enable_linear_replayssm_spec=True,
        )
        allocator = SimpleNamespace(storage=None, state=table.state)
        storage = kv_storage_of(table, allocator)
        cursors = [getattr(table.state.pool, name) for name in _CURSORS]

        (row,) = storage.take_rows(1)
        _forward(table, allocator)
        for cursor in cursors:
            cursor[row] = 5
        storage.release_rows([row])
        self.assertEqual(storage.take_rows(1), [row])
        torch.cuda.synchronize()
        # Handing the row out writes nothing on the scheduling stream.
        self.assertTrue(all(int(cursor[row]) == 5 for cursor in cursors))
        _forward(table, allocator)
        torch.cuda.synchronize()
        self.assertTrue(all(int(cursor[row]) == 0 for cursor in cursors))


if __name__ == "__main__":
    unittest.main()
