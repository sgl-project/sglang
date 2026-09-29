"""CPU-only coverage for the simulator's native KV/Mamba sizing hand-off."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from sglang_simulator.simulation.sglang import model_runner

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.model_executor import pool_configurator
from sglang.srt.runtime_context import get_context, get_schedule


@pytest.fixture
def configurator(monkeypatch):
    # Hook an isolated subclass so other tests retain the native implementation.
    class SimulatorConfigurator(KVCacheConfigurator):
        pass

    model_runner.C_KVCacheConfiguratorHook.hook(SimulatorConfigurator)
    target = SimulatorConfigurator.__new__(SimulatorConfigurator)
    target.server_args = SimpleNamespace(max_total_tokens=None)
    target.model_config = SimpleNamespace(context_len=8192)
    target.mambaish_config = SimpleNamespace(
        mamba2_cache_params=SimpleNamespace(
            layers=[0], mamba_cache_per_req=16 * (1 << 20)
        )
    )
    target.hybrid_gdn_config = None
    target.hybrid_kda_config = None
    target.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    target.pp_size = 1
    target.attn_dp_size = 1
    target.page_size = 256

    # Use the real byte-to-token calculation without constructing a model or pools.
    pool = pool_configurator.DefaultPoolConfigurator.__new__(
        pool_configurator.DefaultPoolConfigurator
    )
    pool._cell_size = 1 << 18
    monkeypatch.setattr(
        pool_configurator, "create_memory_pool_configurator", lambda _: pool
    )
    monkeypatch.setattr(model_runner.ConfigManager, "get_model_info", lambda: object())
    monkeypatch.setattr(
        model_runner.ConfigManager, "get_accelerator_info", lambda: object()
    )
    monkeypatch.setattr(model_runner, "resolve_scheduler_config", lambda **_: object())
    monkeypatch.setattr(
        model_runner,
        "profile_device_available_bytes",
        Mock(return_value=1536 * (1 << 20)),
    )
    return target


@pytest.mark.parametrize("max_total_tokens", [None, 4096])
@pytest.mark.parametrize("max_mamba_cache_size", [None, 63])
@pytest.mark.parametrize("disable_radix_cache", [False, True])
@pytest.mark.parametrize("attn_dp_size", [1, 2])
@pytest.mark.parametrize("max_running_requests", [8, 64])
def test_hybrid_cache_sizing(
    configurator,
    max_total_tokens,
    max_mamba_cache_size,
    disable_radix_cache,
    attn_dp_size,
    max_running_requests,
):
    configurator.server_args.max_total_tokens = max_total_tokens
    configurator.attn_dp_size = attn_dp_size
    with get_context().override_server_args(
        max_total_tokens=max_total_tokens,
        max_mamba_cache_size=max_mamba_cache_size,
        max_running_requests=max_running_requests,
        mamba_full_memory_ratio=3.0,
        page_size=256,
        pp_size=1,
        disable_radix_cache=disable_radix_cache,
        disable_overlap_schedule=False,
        enable_linear_replayssm_spec=False,
        enable_unified_memory=False,
        mem_fraction_static=0.9,
    ):
        config = configurator._resolve_memory_pool_config(0)

        if max_mamba_cache_size is not None:
            expected_slots = max_mamba_cache_size // attn_dp_size
        elif disable_radix_cache:
            expected_slots = max_running_requests // attn_dp_size
        else:
            # 75% of the 1 GiB explicit / 1.5 GiB profiled budget, using
            # 16 MiB per state slot and reserving one padding slot.
            expected_slots = 47 if max_total_tokens is not None else 71
        assert get_schedule().max_mamba_cache_size == expected_slots
        slots_per_request = 1 if disable_radix_cache else 3
        assert config.max_running_requests == min(
            max_running_requests // attn_dp_size, expected_slots // slots_per_request
        )

        if max_total_tokens is not None:
            # Mamba sizing must not shrink the explicitly requested KV capacity.
            assert config.max_total_num_tokens == max_total_tokens
            model_runner.profile_device_available_bytes.assert_not_called()
        else:
            remaining_bytes = (1536 - (expected_slots + 1) * 16) * (1 << 20)
            expected_tokens = remaining_bytes // (1 << 18) // 256 * 256
            assert config.max_total_num_tokens == expected_tokens
            model_runner.profile_device_available_bytes.assert_called_once()


def test_non_hybrid_explicit_capacity_does_not_size_mamba(configurator):
    configurator.mambaish_config = None
    configurator.server_args.max_total_tokens = 4096
    configurator._handle_max_mamba_cache = Mock(
        side_effect=AssertionError("non-hybrid models have no Mamba state")
    )
    assert configurator._profile_available_bytes(0) == 1 << 30
    configurator._handle_max_mamba_cache.assert_not_called()
    model_runner.profile_device_available_bytes.assert_not_called()
