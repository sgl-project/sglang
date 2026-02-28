"""Test dumper.apply_source_patches() integration with source_patcher."""

from pathlib import Path
from types import ModuleType

import yaml

from sglang.srt.debug_utils.dumper import DumperConfig, _Dumper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

SAMPLE_MODULE_NAME = "_source_patcher_test_fixtures.sample_module"


class TestDumperApplySourcePatches:
    def test_no_config_is_noop(self) -> None:
        config = DumperConfig(source_patcher_config=None)
        d = _Dumper(config=config)
        d.apply_source_patches()

    def test_patches_applied_from_yaml(
        self, sample_module: ModuleType, tmp_path: Path
    ) -> None:
        cls = sample_module.SampleClass
        obj = cls()
        assert obj.greet("world") == "hello world"

        original_code = cls.greet.__code__

        patch_config = {
            "patches": [
                {
                    "target": f"{SAMPLE_MODULE_NAME}.SampleClass.greet",
                    "edits": [
                        {
                            "match": 'greeting = f"hello {name}"',
                            "replacement": 'greeting = f"dumper_patched {name}"',
                        }
                    ],
                }
            ]
        }

        config_path = tmp_path / "patch_config.yaml"
        config_path.write_text(yaml.dump(patch_config))

        config = DumperConfig(source_patcher_config=str(config_path))
        d = _Dumper(config=config)

        try:
            d.apply_source_patches()
            assert obj.greet("world") == "dumper_patched world"
        finally:
            cls.greet.__code__ = original_code

        assert obj.greet("world") == "hello world"
