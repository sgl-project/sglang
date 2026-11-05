"""Exercise weight-placement hooks through the real ModelRunner.load_model."""

import gc
import unittest
import weakref
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.model_executor import model_runner as runner_module
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.load_model_utils import (
    LoadedModel,
)
from sglang.srt.utils import offloader as offloader_module
from sglang.srt.utils.offloader import BaseOffloader, NoopOffloader

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Model(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name


class _RecordingOffloader(BaseOffloader):
    def __init__(self, events, transform, fail_init=False):
        self.events = events
        self.transform = transform
        self.fail_init = fail_init

    def post_init(self):
        self.events.append("post_init")
        if self.fail_init:
            raise RuntimeError("placement initialization failed")

    def post_load_model(self, model):
        self.events.append("post_load:" + model.name)
        return self.transform(model)


class TestModelRunnerOffloader(CustomTestCase):
    def _load(self, *, transform=None, draft=False, fail_init=False, noop=False):
        events = []
        references = {}
        config = SimpleNamespace(
            dtype=torch.float32, get_quantization_config_log_str=lambda: ""
        )
        offloader = (
            NoopOffloader()
            if noop
            else _RecordingOffloader(
                events, transform or (lambda model: _Model("replacement")), fail_init
            )
        )
        runner = SimpleNamespace(
            device="cpu",
            gpu_id=0,
            server_args=object(),
            model_config=config,
            draft_load_format=None,
            draft_model_idx=None,
            load_group=object(),
            memory_saver_adapter=object(),
            is_draft_worker=draft,
            spec_algorithm=None,
            offloader=offloader,
            remote_instance_weight_transporter=SimpleNamespace(
                engine=None, session_id=None
            ),
            _load_format_scope=lambda _: nullcontext(),
        )
        runner.maybe_precompile_model_kernels_after_loading = lambda: (
            ModelRunner.maybe_precompile_model_kernels_after_loading(runner)
        )
        self.runner = runner
        self.events = events
        self.references = references

        def load(**_):
            events.append("load")
            model = _Model("original")
            references["original"] = weakref.ref(model)
            # Use the real result type, without a mock retaining the old model.
            return LoadedModel(
                loader=object(), model=model, remote_instance_weight_info=None
            )

        memory_calls = 0

        def memory(*_):
            nonlocal memory_calls
            memory_calls += 1
            if memory_calls == 3:
                gc.collect()
                references["old_alive_at_memory"] = references["original"]() is not None
                events.append("memory:" + runner.model.name)
                return 8.0
            return 10.0

        def record(name):
            def callback(model, *args, **kwargs):
                events.append(name + ":" + model.name)

            return callback

        def sliding(model, _):
            events.append("sliding:" + model.name)
            return 128

        def noop_call(*args, **kwargs):
            pass

        replacements = {
            "get_available_gpu_memory": memory,
            "set_cuda_arch": noop_call,
            "build_load_config": lambda **_: object(),
            "maybe_enable_ipc_weight_cache": noop_call,
            "adjust_config_with_unaligned_cpu_tp": lambda config, *_: config,
            "maybe_trigger_remote_instance_nccl_send_group": noop_call,
            "load_model_with_memory_saver": load,
            "current_platform": SimpleNamespace(post_load_model=record("platform")),
            "get_parallel": lambda: SimpleNamespace(tp_rank=0, tp_size=1, pp_rank=0),
            "get_model": lambda: SimpleNamespace(
                weight_cache_mode="off", weight_cache_socket=None, kv_cache_dtype="auto"
            ),
            "get_exec": lambda: SimpleNamespace(
                comm=SimpleNamespace(enable_layerwise_nvtx_marker=True),
                moe=SimpleNamespace(elastic_ep_backend=None, is_ep_joiner=False),
            ),
            "maybe_precompile_model_kernels_after_loading": record("precompile"),
            "PytHooks": lambda: SimpleNamespace(register_hooks=record("hooks")),
            "load_kv_cache_scales": record("scales"),
            "resolve_sliding_window_size": sliding,
            "report_online_quantization": record("quantization"),
            "maybe_register_debug_tensor_dump_hook": record("debug"),
            "dumper": SimpleNamespace(may_enable=False),
            "reserve_rope_cache_for_long_sequences": record("rope"),
            "dist_barrier_after_load": lambda **_: events.append("barrier"),
        }
        with ExitStack() as stack:
            for name, value in replacements.items():
                stack.enter_context(patch.object(runner_module, name, value))
            # The runner owns its offloader even after another runner is created.
            stack.enter_context(
                patch.object(
                    offloader_module,
                    "_instance",
                    _RecordingOffloader([], lambda model: model, fail_init=True),
                )
            )
            ModelRunner.load_model(runner)
        return runner

    def test_replacement_precedes_every_post_load_consumer(self):
        runner = self._load()
        self.assertEqual(
            self.events,
            [
                "load",
                "platform:original",
                "post_init",
                "post_load:original",
                "precompile:replacement",
                "hooks:replacement",
                "scales:replacement",
                "sliding:replacement",
                "memory:replacement",
                "quantization:replacement",
                "debug:replacement",
                "rope:replacement",
                "barrier",
            ],
        )
        self.assertEqual(runner.weight_load_mem_usage, 2.0)
        self.assertEqual(runner.sliding_window_size, 128)

    def test_discarded_model_is_released_before_memory_measurement(self):
        self._load()
        self.assertFalse(self.references["old_alive_at_memory"])
        gc.collect()
        self.assertIsNone(self.references["original"]())

    def test_wrapper_may_intentionally_retain_original(self):
        def wrap(model):
            wrapper = _Model("wrapper")
            wrapper.original = model
            return wrapper

        runner = self._load(transform=wrap)
        self.assertTrue(self.references["old_alive_at_memory"])
        self.assertIs(runner.model.original, self.references["original"]())
        self.assertIn("hooks:wrapper", self.events)

    def test_default_offloader_keeps_model_and_downstream_steps(self):
        runner = self._load(noop=True)
        self.assertIs(runner.model, self.references["original"]())
        self.assertIn("scales:original", self.events)
        self.assertEqual(self.events[-1], "barrier")

    def test_draft_skips_both_placement_hooks(self):
        runner = self._load(draft=True, fail_init=True)
        self.assertIs(runner.model, self.references["original"]())
        self.assertNotIn("post_init", self.events)
        self.assertNotIn("post_load:original", self.events)
        self.assertIn("precompile:original", self.events)

    def test_invalid_result_stops_before_downstream_consumers(self):
        for invalid in (None, "not a module"):
            with self.subTest(result=invalid):
                with self.assertRaises(TypeError):
                    self._load(transform=lambda _, invalid=invalid: invalid)
                self.assertEqual(
                    self.events,
                    ["load", "platform:original", "post_init", "post_load:original"],
                )
                self.assertIs(self.runner.model, self.references["original"]())

    def test_post_init_failure_stops_before_placement_and_followups(self):
        with self.assertRaisesRegex(RuntimeError, "initialization failed"):
            self._load(fail_init=True)
        self.assertEqual(self.events, ["load", "platform:original", "post_init"])

    def test_placement_failure_stops_before_followups(self):
        def fail(_):
            raise RuntimeError("placement failed")

        with self.assertRaisesRegex(RuntimeError, "placement failed"):
            self._load(transform=fail)
        self.assertEqual(
            self.events,
            ["load", "platform:original", "post_init", "post_load:original"],
        )


if __name__ == "__main__":
    unittest.main()
