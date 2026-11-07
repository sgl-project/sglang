"""Preserve native packed-factory dimensions without changing legacy compaction."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from sglang_simulator.simulation.sglang.model_runner import C_KVCacheConfiguratorHook


def model_config(index_head_dim=128):
    return SimpleNamespace(
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        index_head_dim=index_head_dim,
        head_dim=512,
        v_head_dim=128,
        kv_lora_rank=512,
    )


def dimensions(config):
    return tuple(
        getattr(config, name)
        for name in ("qk_nope_head_dim", "qk_rope_head_dim", "index_head_dim")
    )


def packed_configurator_type():
    # The actual native configurator is slotted. Observations here test only
    # the hook call boundary; native pool controls are separate execution tests.
    @dataclass(slots=True)
    class Configurator:
        model_config: object
        events: list = field(default_factory=list)
        callback: object = None
        before_callback: object = None
        fail: bool = False
        fail_before_factory: bool = False

        def configure(self):
            pass

        def _init_pools(self, marker, *, request_slots):
            self.events.append(("before", dimensions(self.model_config)))
            if self.fail_before_factory:
                raise RuntimeError("outer initialization failure")
            if self.before_callback is not None:
                callback, self.before_callback = self.before_callback, None
                callback()
            try:
                pool = self._build_dsv4_kv_pool(marker, request_slots=request_slots)
            finally:
                self.events.append(("after", dimensions(self.model_config)))
            return SimpleNamespace(token_to_kv_pool=pool)

        def _build_dsv4_kv_pool(self, marker, *, request_slots):
            self.events.append(("factory", dimensions(self.model_config)))
            if self.callback is not None:
                self.callback()
            if self.fail:
                raise ValueError("native factory failure")
            return SimpleNamespace(
                marker=marker,
                request_slots=request_slots,
                qk_nope_head_dim=self.model_config.qk_nope_head_dim,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
            )

    C_KVCacheConfiguratorHook.hook(Configurator)
    return Configurator


def test_slotted_packed_factory_sees_original_dimensions_and_arguments():
    configurator = packed_configurator_type()(model_config())
    assert not hasattr(configurator, "__dict__")
    marker, request_slots = object(), object()
    pools = configurator._init_pools(marker, request_slots=request_slots)
    assert configurator.events == [
        ("before", (1, 1, 1)),
        ("factory", (448, 64, 128)),
        ("after", (1, 1, 1)),
    ]
    assert dimensions(configurator.model_config) == (448, 64, 128)
    assert pools.token_to_kv_pool.marker is marker
    assert pools.token_to_kv_pool.request_slots is request_slots


def test_factory_exception_restores_compacted_then_original_dimensions():
    configurator = packed_configurator_type()(model_config(), fail=True)
    with pytest.raises(ValueError, match="native factory failure"):
        configurator._init_pools(object(), request_slots=3)
    assert configurator.events[-1] == ("after", (1, 1, 1))
    assert dimensions(configurator.model_config) == (448, 64, 128)

    # A later direct call must not inherit stale context from the failed init.
    configurator.fail = False
    configurator.model_config.index_head_dim = 256
    configurator._build_dsv4_kv_pool(object(), request_slots=4)
    assert configurator.events[-1] == ("factory", (448, 64, 256))


def test_nested_configurators_keep_geometry_bound_to_their_own_identity():
    configurator_type = packed_configurator_type()
    inner = configurator_type(model_config(index_head_dim=256))
    outer = configurator_type(model_config())
    outer.callback = lambda: inner._init_pools(object(), request_slots=4)
    outer._init_pools(object(), request_slots=3)
    assert outer.events == [
        ("before", (1, 1, 1)),
        ("factory", (448, 64, 128)),
        ("after", (1, 1, 1)),
    ]
    assert inner.events == [
        ("before", (1, 1, 1)),
        ("factory", (448, 64, 256)),
        ("after", (1, 1, 1)),
    ]
    assert dimensions(outer.model_config) == (448, 64, 128)
    assert dimensions(inner.model_config) == (448, 64, 256)


def test_outer_initialization_failure_also_resets_geometry_context():
    configurator = packed_configurator_type()(model_config(), fail_before_factory=True)
    with pytest.raises(RuntimeError, match="outer initialization failure"):
        configurator._init_pools(object(), request_slots=3)
    assert dimensions(configurator.model_config) == (448, 64, 128)
    configurator.model_config.index_head_dim = 256
    configurator._build_dsv4_kv_pool(object(), request_slots=4)
    assert configurator.events[-1] == ("factory", (448, 64, 256))


def test_reentrant_packed_factory_restores_each_enclosing_scope():
    configurator = packed_configurator_type()(model_config())

    def recurse_once():
        configurator.callback = None
        configurator._build_dsv4_kv_pool(object(), request_slots=4)
        assert dimensions(configurator.model_config) == (448, 64, 128)

    configurator.callback = recurse_once
    configurator._init_pools(object(), request_slots=3)
    assert [event for event in configurator.events if event[0] == "factory"] == [
        ("factory", (448, 64, 128)),
        ("factory", (448, 64, 128)),
    ]
    assert configurator.events[-1] == ("after", (1, 1, 1))
    assert dimensions(configurator.model_config) == (448, 64, 128)


def test_nested_initialization_on_same_owner_restores_compacted_caller_state():
    configurator = packed_configurator_type()(model_config())
    inner_results = []

    def initialize_inner():
        assert dimensions(configurator.model_config) == (1, 1, 1)
        inner_results.append(configurator._init_pools(object(), request_slots=4))
        assert dimensions(configurator.model_config) == (1, 1, 1)

    configurator.before_callback = initialize_inner
    configurator._init_pools(object(), request_slots=3)
    assert configurator.events == [
        ("before", (1, 1, 1)),
        ("before", (1, 1, 1)),
        ("factory", (448, 64, 128)),
        ("after", (1, 1, 1)),
        ("factory", (448, 64, 128)),
        ("after", (1, 1, 1)),
    ]
    assert inner_results[0].token_to_kv_pool.qk_nope_head_dim == 448
    assert dimensions(configurator.model_config) == (448, 64, 128)


@pytest.mark.parametrize("pool_kind", ["mha", "mla"])
def test_legacy_pool_compaction_and_logical_restoration_are_unchanged(pool_kind):
    @dataclass(slots=True)
    class Configurator:
        model_config: object
        observed: object = None

        def configure(self):
            pass

        def _init_pools(self):
            self.observed = (
                self.model_config.head_dim,
                self.model_config.kv_lora_rank,
                self.model_config.qk_rope_head_dim,
            )
            pool = (
                SimpleNamespace(head_num=2, head_dim=1, row_dim=2)
                if pool_kind == "mha"
                else SimpleNamespace(kv_lora_rank=1, qk_rope_head_dim=1, kv_cache_dim=2)
            )
            return SimpleNamespace(token_to_kv_pool=pool)

    C_KVCacheConfiguratorHook.hook(Configurator)
    configurator = Configurator(model_config())
    pool = configurator._init_pools().token_to_kv_pool
    assert configurator.observed == (1, 1, 1)
    assert dimensions(configurator.model_config) == (448, 64, 128)
    if pool_kind == "mha":
        assert (pool.head_dim, pool.row_dim) == (512, 1024)
    else:
        assert (pool.kv_lora_rank, pool.qk_rope_head_dim, pool.kv_cache_dim) == (
            512,
            64,
            576,
        )
