# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan import (
    FastHunyuanConfig,
    HunyuanConfig,
)


def test_hunyuan_configs_default_to_parallel_tiled_vae_decode():
    for config_cls in (HunyuanConfig, FastHunyuanConfig):
        assert config_cls().vae_config.parallel_decode_mode == "tiled"


def test_hunyuan_parallel_decode_mode_can_be_overridden():
    for config_cls in (HunyuanConfig, FastHunyuanConfig):
        config = config_cls()

        config.update_config_from_dict(
            {"vae_config.parallel_decode_mode": "spatial_shard"}
        )

        assert config.vae_config.parallel_decode_mode == "spatial_shard"
