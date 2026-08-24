"""Control-plane config updates go into the process log, not onto the record.

Regression: runtime updates (weight version, model path, HiCache attach) were
written onto the manager's ServerArgs instance so that the readback endpoints
would show them. The record stays pristine; the update lands in the runtime
context, which is where a reader of any field asks for the value in effect.
"""

import re
import unittest
from pathlib import Path

import sglang
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _manager(case, **fields):
    """A manager over a published config: the updates it records go to the bags."""
    server_args = ServerArgs(model_path="dummy", **fields)
    publish(server_args, role="tokenizer")
    case.addCleanup(reset_context)
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = server_args
    return manager


class TestTokenizerConfigUpdates(CustomTestCase):
    def test_startup_config_shows_through_until_something_updates_it(self):
        manager = _manager(self, weight_version="v1")
        self.assertEqual(manager.config_value("weight_version"), "v1")

        manager.record_config_updates("test", weight_version="v2")
        self.assertEqual(manager.config_value("weight_version"), "v2")

    def test_the_serverargs_instance_is_not_written(self):
        manager = _manager(self, weight_version="v1")
        manager.record_config_updates("test", weight_version="v2")
        self.assertEqual(manager.server_args.weight_version, "v1")

    def test_the_readback_dict_carries_the_updates(self):
        manager = _manager(self, hicache_storage_backend=None)
        manager.record_config_updates(
            "test", hicache_storage_backend="file", hicache_write_policy="write_through"
        )
        manager.model_path = "dummy"
        manager.served_model_name = "dummy"
        resolved = manager.resolved_config_dict(
            {"hicache_storage_backend": None, "model_path": "dummy"}
        )
        self.assertEqual(resolved["hicache_storage_backend"], "file")
        self.assertEqual(resolved["hicache_write_policy"], "write_through")
        self.assertEqual(resolved["model_path"], "dummy")

    def test_detach_reports_the_backend_as_gone(self):
        manager = _manager(self, hicache_storage_backend="file")
        manager.record_config_updates(
            "test",
            hicache_storage_backend=None,
            hicache_storage_backend_extra_config=None,
        )
        self.assertIsNone(manager.config_value("hicache_storage_backend"))

    def test_an_unknown_field_is_refused(self):
        manager = _manager(self)
        with self.assertRaisesRegex(ValueError, "not a resolved config leaf"):
            manager.record_config_updates("test", waight_version="v2")

    def test_a_name_that_is_not_a_config_leaf_is_refused(self):
        manager = _manager(self)
        with self.assertRaisesRegex(ValueError, "not a config leaf"):
            manager.config_value("waight_version")

    def test_the_source_is_kept_for_provenance(self):
        manager = _manager(self, weight_version="v1")
        manager.record_config_updates("http.update_weight_version", weight_version="v2")
        self.assertEqual(
            get_context().overrides_log(),
            [("http.update_weight_version", {"weight_version": "v2"})],
        )

    def test_the_dump_snapshot_identifies_the_running_checkpoint(self):
        import dataclasses

        manager = _manager(self, load_format="auto")
        manager.model_path = "at-startup"
        manager.served_model_name = "at-startup"
        manager._update_model_path_info("after-reload", "dummy")

        snapshot = manager.resolved_config_dict(dataclasses.asdict(manager.server_args))
        self.assertEqual(snapshot["model_path"], "after-reload")
        self.assertEqual(snapshot["served_model_name"], "after-reload")
        self.assertEqual(snapshot["load_format"], "dummy")
        self.assertEqual(
            get_context().overrides_log(),
            [("tokenizer.update_weights", {"load_format": "dummy"})],
        )
        self.assertEqual(manager.server_args.model_path, "dummy")

    def test_an_unsnapshotable_config_does_not_lose_the_dump(self):
        class Hostile:
            def __deepcopy__(self, memo):
                raise RuntimeError("refuses to be copied")

        # Through the constructor: the field is raw input, and a resolved
        # record refuses to be written.
        manager = _manager(self, custom_sigquit_handler=Hostile())
        manager.model_path = "dummy"
        manager.served_model_name = "dummy"

        self.assertIsNone(manager._dump_config_snapshot())

    def test_an_unpickleable_field_does_not_lose_the_dump(self):
        import dataclasses
        import pickle

        # What --custom-sigquit-handler leaves on a real ServerArgs.
        manager = _manager(self, custom_sigquit_handler=lambda *_: None)
        manager.model_path = "dummy"
        manager.served_model_name = "dummy"

        payload = {
            "server_args": manager.server_args,
            "config_updates": get_context().overrides_log(),
            "resolved_config": manager.resolved_config_dict(
                dataclasses.asdict(manager.server_args)
            ),
            "requests": [],
        }
        with self.assertRaises(Exception):
            pickle.dumps(payload)

        # The fallback drops both copies of the offending object, not just one.
        payload["server_args"] = None
        payload["resolved_config"] = None
        self.assertTrue(pickle.dumps(payload))

    def test_the_model_path_readback_follows_the_manager(self):
        manager = _manager(self)
        manager.model_path = "after-update"
        manager.served_model_name = "after-update"
        resolved = manager.resolved_config_dict({"model_path": "at-startup"})
        self.assertEqual(resolved["model_path"], "after-update")
        self.assertEqual(resolved["served_model_name"], "after-update")


CONTROL_PLANE_FIELDS = (
    "weight_version",
    "model_path",
    "served_model_name",
    "load_format",
    "hicache_storage_backend",
    "hicache_storage_backend_extra_config",
    "hicache_storage_prefetch_policy",
    "hicache_write_policy",
)

# Modules that answer readbacks or fill responses; the tokenizer manager's own
# __init__ seeds attributes from the constructor argument, which is not a
# readback and not matched by the patterns below. Prometheus label sets are
# exempt: a label must stay fixed for the lifetime of the series, so the metrics
# collector keeps the name the server started with.
EXEMPT_LINES = (
    (
        "srt/managers/tokenizer_manager.py",
        '"model_name": self.server_args.served_model_name',
    ),
)
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
                    if any(
                        rel_exempt == path.relative_to(root).as_posix()
                        and needle in line
                        for rel_exempt, needle in EXEMPT_LINES
                    ):
                        continue
                    if any(p.search(line) for p in patterns):
                        stale.append(
                            f"{path.relative_to(root)}:{number}: {line.strip()}"
                        )
        self.assertEqual(
            stale,
            [],
            "control-plane fields change at runtime and the update lands in "
            "the process bags (model_path / served_model_name stay manager "
            "attributes); read them with config_value() / "
            "resolved_config_dict() so the readback reflects the change:\n"
            + "\n".join(stale),
        )


if __name__ == "__main__":
    unittest.main()
