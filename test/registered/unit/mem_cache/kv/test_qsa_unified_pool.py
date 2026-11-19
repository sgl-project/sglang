"""QSA compressed keys follow FULL pages and pending keys follow request rows."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.qsa.config import QSAProfile
from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MHAGeometry, StateGeometry, kv_layer_decls
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.qsa import qsa_storage_parts
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 64
FULL_LAYERS = (7, 10)
STATE_LAYERS = (8, 9)
PROFILE = QSAProfile(n_heads=8, kv_heads=1, head_dim=64, budget=4096, compress_ratio=8)


@pytest.fixture
def storage():
    setup_allocator_context()
    try:
        layers = kv_layer_decls(
            full_layer_ids=FULL_LAYERS,
            full_geometry=MHAGeometry(
                head_num=1, head_dim=64, v_head_dim=64, kv_cache_dtype=torch.bfloat16
            ),
            state_layer_ids=STATE_LAYERS,
            state_geometry=StateGeometry(
                conv_state_shapes=((3, 128),),
                conv_dtype=torch.bfloat16,
                temporal_state_shape=(2, 16, 32),
                temporal_dtype=torch.float32,
            ),
            checkpoint_interval=PAGE_SIZE,
        )
        plan = plan_kv_storage(
            layers,
            page_size=PAGE_SIZE,
            **qsa_storage_parts(layers, PROFILE, PAGE_SIZE, torch.bfloat16),
        )
        sized = size_plan(
            plan,
            KVSizingInputs(
                full_tokens=2048,
                state_slots=32,
                max_running_requests=1,
                context_len=256,
                alignment=1,
                radix_cache=False,
            ),
        )
        yield build_kv_storage(
            plan,
            sized,
            UnifiedBuildContext(
                device="cpu",
                start_layer=FULL_LAYERS[0],
                end_layer=FULL_LAYERS[-1] + 1,
                enable_memory_saver=False,
                need_sort=False,
                max_num_reqs=1,
                model_context_len=256,
                state_cache_params=SimpleNamespace(layers=STATE_LAYERS),
                qsa_profile=PROFILE,
            ),
        )
    finally:
        reset_context()


def test_qsa_index_keys_follow_relocated_full_kv(storage):
    allocator = storage.allocator
    owner = allocator.full_attn_allocator
    cache = allocator.get_kvcache()
    ratio = PROFILE.compress_ratio
    logical_pages = [allocator.alloc(PAGE_SIZE) for _ in range(3)]
    old_physical_pages = [owner.ids.translate(ids) for ids in logical_pages]

    for page, slots in enumerate(old_physical_pages):
        compressed_slots = slots[::ratio] // ratio
        for layer in FULL_LAYERS:
            tag = 10 * page + layer
            cache.get_key_buffer(layer)[slots] = tag
            cache.get_value_buffer(layer)[slots] = -tag
            keys = torch.arange(compressed_slots.numel()) + tag
            cache.set_qsa_compressed_k_buffer(
                layer,
                compressed_slots,
                keys[:, None, None].expand(-1, 1, PROFILE.head_dim),
            )

    # Free a live page from the middle, leaving a hole for lazy compaction.
    allocator.free(logical_pages[1])
    owner.flush_for_allocation()
    assert any(
        not torch.equal(
            old_physical_pages[page], owner.ids.translate(logical_pages[page])
        )
        for page in (0, 2)
    )
    for page in (0, 2):
        slots = owner.ids.translate(logical_pages[page])
        compressed_slots = slots[::ratio] // ratio
        for layer in FULL_LAYERS:
            tag = 10 * page + layer
            assert (cache.get_key_buffer(layer)[slots] == tag).all()
            assert (cache.get_value_buffer(layer)[slots] == -tag).all()
            expected = torch.arange(compressed_slots.numel()) + tag
            actual = cache.get_qsa_compressed_k_buffer(layer)[compressed_slots]
            assert (actual == expected[:, None, None]).all()


def test_qsa_pending_ring_resets_when_request_row_is_reused(storage):
    (row,) = storage.take_rows(1)
    cache = storage.allocator.get_kvcache()
    ratio = PROFILE.compress_ratio
    ring = row * ratio + torch.arange(ratio)
    for layer in FULL_LAYERS:
        cache.set_qsa_key_state_buffer(
            layer, ring, torch.ones(ratio, PROFILE.kv_heads, PROFILE.head_dim)
        )
        assert cache.get_qsa_key_state_buffer(layer)[ring].all()
    cache.set_qsa_rope_position_buffer(ring, torch.arange(ratio) + 100)
    assert cache.get_qsa_rope_position_buffer(ring).all()

    storage.release_rows([row])
    assert storage.take_rows(1) == [row]
    for layer in FULL_LAYERS:
        assert not cache.get_qsa_key_state_buffer(layer)[ring].any()
    assert not cache.get_qsa_rope_position_buffer(ring).any()
