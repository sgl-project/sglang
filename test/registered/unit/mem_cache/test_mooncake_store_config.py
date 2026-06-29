import json
import sys

import pytest

from sglang.srt.environ import envs, temp_set_env
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    DEFAULT_LOCAL_BUFFER_SIZE,
    MooncakeStoreConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_mooncake_local_buffer_size_loads_from_env():
    with temp_set_env(
        MOONCAKE_MASTER="127.0.0.1:50051",
        MOONCAKE_LOCAL_BUFFER_SIZE=str(4 * 1024 * 1024 * 1024),
    ):
        config = MooncakeStoreConfig.load_from_env()

    assert config.local_buffer_size == 4 * 1024 * 1024 * 1024


def test_mooncake_local_buffer_size_defaults_to_16mb():
    with temp_set_env(
        MOONCAKE_MASTER="127.0.0.1:50051",
        MOONCAKE_LOCAL_BUFFER_SIZE=None,
    ):
        config = MooncakeStoreConfig.load_from_env()

    assert config.local_buffer_size == DEFAULT_LOCAL_BUFFER_SIZE


def test_mooncake_local_buffer_size_loads_from_file(tmp_path):
    config_path = tmp_path / "mooncake.json"
    config_path.write_text(
        json.dumps(
            {
                "master_server_address": "127.0.0.1:50051",
                "local_buffer_size": "8gb",
            }
        ),
        encoding="utf-8",
    )

    with envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(str(config_path)):
        config = MooncakeStoreConfig.from_file()

    assert config.local_buffer_size == 8 * 1024 * 1024 * 1024


def test_mooncake_local_buffer_size_file_none_uses_default(tmp_path):
    config_path = tmp_path / "mooncake.json"
    config_path.write_text(
        json.dumps(
            {
                "master_server_address": "127.0.0.1:50051",
                "local_buffer_size": None,
            }
        ),
        encoding="utf-8",
    )

    with envs.SGLANG_HICACHE_MOONCAKE_CONFIG_PATH.override(str(config_path)):
        config = MooncakeStoreConfig.from_file()

    assert config.local_buffer_size == DEFAULT_LOCAL_BUFFER_SIZE


def test_mooncake_local_buffer_size_accepts_mb_suffix():
    config = MooncakeStoreConfig.load_from_extra_config(
        {
            "master_server_address": "127.0.0.1:50051",
            "local_buffer_size": "32mb",
        }
    )

    assert config.local_buffer_size == 32 * 1024 * 1024


def test_mooncake_local_buffer_size_accepts_kb_suffix():
    config = MooncakeStoreConfig.load_from_extra_config(
        {
            "master_server_address": "127.0.0.1:50051",
            "local_buffer_size": "512kb",
        }
    )

    assert config.local_buffer_size == 512 * 1024


def test_mooncake_local_buffer_size_loads_from_extra_config():
    config = MooncakeStoreConfig.load_from_extra_config(
        {
            "master_server_address": "127.0.0.1:50051",
            "local_buffer_size": 0,
        }
    )

    assert config.local_buffer_size == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
