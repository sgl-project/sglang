import types
import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.loader.component_loaders.scheduler_loader import (
    SchedulerLoader,
    _filter_supported_scheduler_init_kwargs,
)


class _SchedulerWithoutKwargs:
    def __init__(self, foo, bar=1):
        self.foo = foo
        self.bar = bar
        self.shift = None

    def set_shift(self, shift):
        self.shift = shift


class _SchedulerWithKwargs:
    def __init__(self, foo, **kwargs):
        self.foo = foo
        self.kwargs = kwargs


class TestSchedulerLoader(unittest.TestCase):
    def test_filter_supported_scheduler_init_kwargs_drops_unknown_fields(self):
        filtered, ignored = _filter_supported_scheduler_init_kwargs(
            _SchedulerWithoutKwargs,
            {"foo": 1, "bar": 2, "shift_terminal": True},
        )

        self.assertEqual(filtered, {"foo": 1, "bar": 2})
        self.assertEqual(ignored, ["shift_terminal"])

    def test_filter_supported_scheduler_init_kwargs_keeps_kwargs_scheduler(self):
        filtered, ignored = _filter_supported_scheduler_init_kwargs(
            _SchedulerWithKwargs,
            {"foo": 1, "shift_terminal": True},
        )

        self.assertEqual(filtered, {"foo": 1, "shift_terminal": True})
        self.assertEqual(ignored, [])

    @patch(
        "sglang.multimodal_gen.runtime.loader.component_loaders.scheduler_loader.ModelRegistry.resolve_model_cls",
        return_value=(_SchedulerWithoutKwargs, None),
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.component_loaders.scheduler_loader.get_diffusers_component_config",
        return_value={
            "_class_name": "UniPCMultistepScheduler",
            "foo": 7,
            "shift_terminal": True,
        },
    )
    def test_load_customized_ignores_unknown_scheduler_config_keys(
        self, _mock_config, _mock_resolve
    ):
        loader = SchedulerLoader()
        server_args = types.SimpleNamespace(
            pipeline_config=types.SimpleNamespace(flow_shift=1.25)
        )

        with self.assertLogs(
            "sglang.multimodal_gen.runtime.loader.component_loaders.scheduler_loader",
            level="WARNING",
        ) as logs:
            scheduler = loader.load_customized("/unused", server_args)

        self.assertIsInstance(scheduler, _SchedulerWithoutKwargs)
        self.assertEqual(scheduler.foo, 7)
        self.assertEqual(scheduler.shift, 1.25)
        self.assertTrue(
            any("shift_terminal" in message for message in logs.output),
            logs.output,
        )

