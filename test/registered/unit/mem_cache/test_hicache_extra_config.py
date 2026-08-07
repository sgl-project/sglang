import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.mem_cache.hicache_storage import (
    load_hicache_storage_backend_extra_config,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHiCacheStorageBackendExtraConfig(unittest.TestCase):
    def test_loads_inline_json(self):
        config = load_hicache_storage_backend_extra_config(
            '{"canonical_page_size": 64, "canonical_dcp_size": 8}'
        )
        self.assertEqual(config["canonical_page_size"], 64)
        self.assertEqual(config["canonical_dcp_size"], 8)

    def test_loads_supported_config_files(self):
        contents = {
            ".json": json.dumps(
                {"canonical_page_size": 64, "canonical_dcp_size": 8}
            ),
            ".yaml": "canonical_page_size: 64\ncanonical_dcp_size: 8\n",
            ".toml": "canonical_page_size = 64\ncanonical_dcp_size = 8\n",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            for suffix, content in contents.items():
                with self.subTest(suffix=suffix):
                    path = Path(tmpdir) / f"hicache{suffix}"
                    path.write_text(content)
                    config = load_hicache_storage_backend_extra_config(f"@{path}")
                    self.assertEqual(config["canonical_page_size"], 64)
                    self.assertEqual(config["canonical_dcp_size"], 8)

    def test_server_args_resolver_accepts_file_config(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            json.dump(
                {"canonical_page_size": 64, "canonical_dcp_size": 8},
                config_file,
            )
            config_file.flush()
            server_args = SimpleNamespace(
                hicache_storage_backend="mooncake",
                hicache_storage_backend_extra_config=f"@{config_file.name}",
                mamba_track_interval=64,
                disaggregation_mode="prefill",
                disaggregation_decode_enable_offload_kvcache=False,
                get_model_config=lambda: SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=["KimiK3ForConditionalGeneration"]
                    )
                ),
            )
            server_args.hicache_storage_backend_extra_config_dict = (
                load_hicache_storage_backend_extra_config(
                    server_args.hicache_storage_backend_extra_config
                )
            )

            ServerArgs._resolve_canonical_hybrid_checkpointing(server_args)

        self.assertEqual(server_args._canonical_hybrid_checkpoint_interval, 512)
        self.assertEqual(server_args.mamba_track_interval, 512)

    def test_dcp_l3_rejects_non_mooncake_backend(self):
        server_args = SimpleNamespace(
            dcp_size=8,
            disaggregation_decode_enable_offload_kvcache=True,
            hicache_storage_backend="file",
        )

        with self.assertRaisesRegex(NotImplementedError, "Mooncake"):
            ServerArgs._resolve_hicache_dcp_compatibility(server_args)

    def test_dcp_l3_requires_canonical_layout(self):
        server_args = SimpleNamespace(
            dcp_size=8,
            disaggregation_decode_enable_offload_kvcache=True,
            hicache_storage_backend="mooncake",
            hicache_storage_backend_extra_config_dict={},
        )

        with self.assertRaisesRegex(ValueError, "canonical_dcp_size"):
            ServerArgs._resolve_hicache_dcp_compatibility(server_args)


if __name__ == "__main__":
    unittest.main()
