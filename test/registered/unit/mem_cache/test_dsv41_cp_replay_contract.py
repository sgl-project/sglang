"""CPU checks for the newly admitted CP encoder-replay envelope."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "cp_replay_contract",
    ROOT / "python/sglang/srt/mem_cache/dsv41_replay_contract.py",
)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def config(**changes):
    values = dict(
        enable_encoder_swa_bounded_replay=True,
        attn_cp_size=8,
        tp_size=8,
        pp_size=1,
        dp_size=1,
        dcp_size=1,
        attn_dp_size=1,
        enable_prefill_cp=True,
        cp_strategy="interleave",
        cuda_graph_config=SimpleNamespace(prefill=SimpleNamespace(backend="disabled")),
        disaggregation_mode="prefill",
        disaggregation_transfer_backend="mooncake",
        speculative_algorithm="DSPARK",
        moe_a2a_backend="none",
        enable_kv_cache_sharding=False,
        enable_dsa_cache_layer_split=False,
        enable_cp_tp_group_sharing=False,
        enable_hierarchical_cache=True,
        hicache_host_memory_mode="cache",
        hicache_storage_backend="mooncake",
    )
    return SimpleNamespace(**(values | changes))


def test_cp1_does_not_inspect_cp_only_options():
    cfg = SimpleNamespace(enable_encoder_swa_bounded_replay=True, attn_cp_size=1)
    contract.validate_cp_encoder_replay(cfg, is_cuda=False, is_sm90=False)
    assert not contract.is_cp_encoder_replay(cfg)


def test_disabled_replay_does_not_change_generic_cp():
    cfg = SimpleNamespace(enable_encoder_swa_bounded_replay=False)
    contract.validate_cp_encoder_replay(cfg, is_cuda=False, is_sm90=False)


def test_supported_topology():
    contract.validate_cp_encoder_replay(config(), is_cuda=True, is_sm90=True)


@pytest.mark.parametrize(
    "change",
    [
        {"pp_size": 2},
        {"dp_size": 2},
        {"dcp_size": 2},
        {"attn_dp_size": 2},
        {"enable_prefill_cp": False},
        {"cp_strategy": "zigzag"},
        {"attn_cp_size": 4},
        {"disaggregation_mode": "decode"},
        {"disaggregation_transfer_backend": "nixl"},
        {"speculative_algorithm": "EAGLE"},
        {"moe_a2a_backend": "megamoe"},
        {"enable_kv_cache_sharding": True},
        {"enable_dsa_cache_layer_split": True},
        {"enable_cp_tp_group_sharing": True},
        {"hicache_host_memory_mode": "buffer_only"},
        {"hicache_storage_backend": "file"},
    ],
)
def test_unsupported_topology_fails_explicitly(change):
    with pytest.raises(ValueError):
        contract.validate_cp_encoder_replay(
            config(**change), is_cuda=True, is_sm90=True
        )


@pytest.mark.parametrize("cuda,sm90", [(False, False), (True, False)])
def test_hardware_gate(cuda, sm90):
    with pytest.raises(ValueError):
        contract.validate_cp_encoder_replay(config(), is_cuda=cuda, is_sm90=sm90)
