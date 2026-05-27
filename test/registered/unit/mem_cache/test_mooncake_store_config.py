import json

from sglang.srt.environ import envs, temp_set_env
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    DEFAULT_LOCAL_BUFFER_SIZE,
    MooncakeStoreConfig,
)


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


def test_mooncake_local_buffer_size_loads_from_extra_config():
    config = MooncakeStoreConfig.load_from_extra_config(
        {
            "master_server_address": "127.0.0.1:50051",
            "local_buffer_size": 0,
        }
    )

    assert config.local_buffer_size == 0
