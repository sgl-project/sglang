# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def _load_runtime_module():
    numpy_stub = ModuleType("numpy")
    torch_stub = ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})

    server_args = SimpleNamespace(enable_cfg_parallel=False)
    server_args_module = ModuleType("sglang.multimodal_gen.runtime.server_args")
    server_args_module.get_global_server_args = lambda: server_args

    forward_context = SimpleNamespace(current_timestep=0, forward_batch=None)
    forward_context_module = ModuleType(
        "sglang.multimodal_gen.runtime.managers.forward_context"
    )
    forward_context_module.get_forward_context = lambda: forward_context

    stubs = {
        "numpy": numpy_stub,
        "torch": torch_stub,
        server_args_module.__name__: server_args_module,
        forward_context_module.__name__: forward_context_module,
    }
    module_path = (
        Path(__file__).resolve().parents[2] / "runtime" / "cache" / "teacache.py"
    )
    return stubs, module_path, server_args, forward_context


def _load_params_module():
    sampling_params_module = ModuleType(
        "sglang.multimodal_gen.configs.sample.sampling_params"
    )
    sampling_params_module.CacheParams = type("CacheParams", (), {})
    module_path = (
        Path(__file__).resolve().parents[2] / "configs" / "sample" / "teacache.py"
    )
    return {sampling_params_module.__name__: sampling_params_module}, module_path


def _import_from_path(name, module_path):
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _make_batch():
    params = SimpleNamespace(
        teacache_thresh=0.1,
        get_coefficients=lambda: [1.0],
    )
    return SimpleNamespace(
        enable_teacache=True,
        teacache_params=params,
        num_inference_steps=50,
        do_classifier_free_guidance=True,
        is_cfg_negative=False,
    )


class TestTeaCacheLifecycle(unittest.TestCase):
    def test_serial_cfg_resets_only_before_positive_branch(self):
        stubs, module_path, server_args, forward_context = _load_runtime_module()
        with patch.dict(sys.modules, stubs):
            module = _import_from_path("test_teacache_runtime_serial", module_path)

            class DummyTeaCache(module.TeaCacheMixin):
                prefix = "wan"

            cache = DummyTeaCache()
            cache._init_teacache_state()
            batch = _make_batch()
            forward_context.forward_batch = batch
            server_args.enable_cfg_parallel = False

            cache._get_teacache_context()
            cache.cnt = 1
            cache.previous_modulated_input = object()
            batch.is_cfg_negative = True

            ctx = cache._get_teacache_context()

            self.assertEqual(cache.cnt, 1)
            self.assertIsNotNone(cache.previous_modulated_input)
            self.assertFalse(ctx.is_cfg_parallel)
            self.assertTrue(ctx.is_cfg_negative)

    def test_cfg_parallel_negative_branch_resets_at_request_start(self):
        stubs, module_path, server_args, forward_context = _load_runtime_module()
        with patch.dict(sys.modules, stubs):
            module = _import_from_path("test_teacache_runtime_parallel", module_path)

            class DummyTeaCache(module.TeaCacheMixin):
                prefix = "wan"

            cache = DummyTeaCache()
            cache._init_teacache_state()
            cache.cnt = 50
            cache.is_cfg_negative = True
            cache.previous_modulated_input_negative = object()
            cache.previous_residual_negative = object()
            cache.accumulated_rel_l1_distance_negative = 7.0

            batch = _make_batch()
            batch.is_cfg_negative = True
            forward_context.forward_batch = batch
            server_args.enable_cfg_parallel = True

            ctx = cache._get_teacache_context()

            self.assertEqual(cache.cnt, 0)
            self.assertIsNone(cache.previous_modulated_input_negative)
            self.assertIsNone(cache.previous_residual_negative)
            self.assertEqual(cache.accumulated_rel_l1_distance_negative, 0.0)
            self.assertTrue(ctx.is_cfg_parallel)
            self.assertTrue(ctx.is_cfg_negative)


class TestTeaCacheBoundaries(unittest.TestCase):
    def test_cfg_parallel_uses_one_local_forward_per_timestep(self):
        stubs, module_path = _load_params_module()
        with patch.dict(sys.modules, stubs):
            module = _import_from_path("test_teacache_params", module_path)
            params = module.TeaCacheParams(start_skipping=5, end_skipping=-1)

            self.assertEqual(params.get_skip_boundaries(50, True), (10, 98))
            self.assertEqual(
                params.get_skip_boundaries(50, True, cfg_parallel=True),
                (5, 49),
            )


if __name__ == "__main__":
    unittest.main()
