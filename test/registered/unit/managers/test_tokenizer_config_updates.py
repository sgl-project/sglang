"""Control-plane config updates stay on the tokenizer manager.

Regression: runtime updates (weight version, model path, HiCache attach) were
written onto the manager's ServerArgs instance so that the readback endpoints
would show them. They are per-engine — several Engines can share a tokenizer
process — so they live on the manager and the endpoints overlay them.
"""

import re
import unittest
from pathlib import Path

import sglang
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _manager(**fields):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = ServerArgs(model_path="dummy", **fields)
    manager._config_updates = {}
    return manager


class TestTokenizerConfigUpdates(CustomTestCase):
    def test_startup_config_shows_through_until_something_updates_it(self):
        manager = _manager(weight_version="v1")
        self.assertEqual(manager.config_value("weight_version"), "v1")

        manager.record_config_updates(weight_version="v2")
        self.assertEqual(manager.config_value("weight_version"), "v2")

    def test_the_serverargs_instance_is_not_written(self):
        manager = _manager(weight_version="v1")
        manager.record_config_updates(weight_version="v2")
        self.assertEqual(manager.server_args.weight_version, "v1")

    def test_two_engines_keep_their_own_updates(self):
        first, second = _manager(weight_version="v1"), _manager(weight_version="v1")
        first.record_config_updates(weight_version="v2")
        self.assertEqual(second.config_value("weight_version"), "v1")

    def test_the_readback_dict_carries_the_updates(self):
        manager = _manager(hicache_storage_backend=None)
        manager.record_config_updates(
            hicache_storage_backend="file", hicache_write_policy="write_through"
        )
        resolved = manager.resolved_config_dict(
            {"hicache_storage_backend": None, "model_path": "dummy"}
        )
        self.assertEqual(resolved["hicache_storage_backend"], "file")
        self.assertEqual(resolved["hicache_write_policy"], "write_through")
        self.assertEqual(resolved["model_path"], "dummy")

    def test_detach_reports_the_backend_as_gone(self):
        manager = _manager(hicache_storage_backend="file")
        manager.record_config_updates(
            hicache_storage_backend=None, hicache_storage_backend_extra_config=None
        )
        self.assertIsNone(manager.config_value("hicache_storage_backend"))


CONTROL_PLANE_FIELDS = (
    "weight_version",
    "model_path",
    "load_format",
    "hicache_storage_backend",
    "hicache_storage_backend_extra_config",
    "hicache_storage_prefetch_policy",
    "hicache_write_policy",
)

# Modules that answer readbacks or fill responses; the tokenizer manager's own
# __init__ seeds attributes from the constructor argument, which is not a
# readback and not matched by the patterns below.
READBACK_MODULES = (
    "srt/managers/tokenizer_manager.py",
    "srt/managers/tokenizer_control_mixin.py",
    "srt/managers/multi_tokenizer_mixin.py",
    "srt/entrypoints/http_server.py",
    "srt/entrypoints/grpc_bridge.py",
    "srt/entrypoints/engine.py",
    "srt/entrypoints/openai",
)


class TestControlPlaneFieldsAreNotReadFromTheInstance(CustomTestCase):
    def test_readbacks_go_through_the_manager(self):
        root = Path(next(iter(sglang.__path__)))
        patterns = [
            re.compile(
                rf"self\.server_args\.{f}\b|tokenizer_manager\.server_args\.{f}\b"
            )
            for f in CONTROL_PLANE_FIELDS
        ]
        stale = []
        for rel in READBACK_MODULES:
            paths = (
                sorted((root / rel).rglob("*.py"))
                if (root / rel).is_dir()
                else [root / rel]
            )
            for path in paths:
                for number, line in enumerate(path.read_text().split("\n"), 1):
                    if any(p.search(line) for p in patterns):
                        stale.append(
                            f"{path.relative_to(root)}:{number}: {line.strip()}"
                        )
        self.assertEqual(
            stale,
            [],
            "control-plane fields change at runtime and the update lives on the "
            "TokenizerManager; read them with config_value() / "
            "resolved_config_dict() so the readback reflects the change:\n"
            + "\n".join(stale),
        )


if __name__ == "__main__":
    unittest.main()
