import json
from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache.sparsity import parse_hisparse_config


def _server_args(hisparse_config=None):
    return SimpleNamespace(hisparse_config=hisparse_config)


def test_hisparse_config_defaults_swap_in_block_size():
    config = parse_hisparse_config(_server_args())

    assert config.swap_in_block_size == 512


def test_hisparse_config_parses_swap_in_block_size():
    config = parse_hisparse_config(
        _server_args(
            '{"top_k": 2048, "device_buffer_size": 6144, '
            '"host_to_device_ratio": 5, "swap_in_block_size": 768}'
        )
    )

    assert config.top_k == 2048
    assert config.device_buffer_size == 6144
    assert config.host_to_device_ratio == 5
    assert config.swap_in_block_size == 768


@pytest.mark.parametrize("value", [0, -1, 1025, True, "512"])
def test_hisparse_config_rejects_invalid_swap_in_block_size(value):
    with pytest.raises(ValueError, match="swap_in_block_size"):
        parse_hisparse_config(_server_args(json.dumps({"swap_in_block_size": value})))
