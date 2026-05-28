import importlib
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _FakeDBCacheConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def reset(self, **kwargs):
        return kwargs


def _install_cache_dit_stub():
    cache_dit = types.ModuleType("cache_dit")
    cache_dit.refresh_calls = []
    cache_dit.steps_mask_calls = []
    cache_dit.enable_cache_calls = []

    def refresh_context(transformer, cache_config, verbose=False):
        cache_dit.refresh_calls.append(
            {
                "transformer": transformer,
                "cache_config": cache_config,
                "verbose": verbose,
            }
        )

    def steps_mask(*, mask_policy, total_steps):
        cache_dit.steps_mask_calls.append(
            {"mask_policy": mask_policy, "total_steps": total_steps}
        )
        return [1] * total_steps

    def enable_cache(
        transformer,
        cache_config=None,
        calibrator_config=None,
        parallelism_config=None,
        attention_backend=None,
        **kwargs,
    ):
        cache_dit.enable_cache_calls.append(
            {
                "transformer": transformer,
                "cache_config": cache_config,
                "calibrator_config": calibrator_config,
                "parallelism_config": parallelism_config,
                "attention_backend": attention_backend,
                "kwargs": kwargs,
            }
        )

    cache_dit.refresh_context = refresh_context
    cache_dit.steps_mask = steps_mask
    cache_dit.enable_cache = enable_cache
    cache_dit.BlockAdapter = object
    cache_dit.DBCacheConfig = _FakeDBCacheConfig
    cache_dit.ForwardPattern = object
    cache_dit.ParamsModifier = object
    cache_dit.TaylorSeerCalibratorConfig = object

    block_adapters = types.ModuleType("cache_dit.caching.block_adapters")

    class _FakeBlockAdapterRegister:
        @staticmethod
        def is_supported(_transformer):
            return True

    block_adapters.BlockAdapterRegister = _FakeBlockAdapterRegister

    parallelism = types.ModuleType("cache_dit.parallelism")
    parallelism.ParallelismBackend = object
    parallelism.ParallelismConfig = object

    return {
        "cache_dit": cache_dit,
        "cache_dit.caching.block_adapters": block_adapters,
        "cache_dit.parallelism": parallelism,
    }


def _install_sglang_dependency_stubs():
    sglang = types.ModuleType("sglang")
    multimodal_gen = types.ModuleType("sglang.multimodal_gen")
    envs = types.ModuleType("sglang.multimodal_gen.envs")
    runtime = types.ModuleType("sglang.multimodal_gen.runtime")
    distributed = types.ModuleType("sglang.multimodal_gen.runtime.distributed")
    parallel_state = types.ModuleType(
        "sglang.multimodal_gen.runtime.distributed.parallel_state"
    )
    utils = types.ModuleType("sglang.multimodal_gen.runtime.utils")
    logging_utils = types.ModuleType(
        "sglang.multimodal_gen.runtime.utils.logging_utils"
    )

    parallel_state.get_ring_parallel_world_size = lambda: 1
    parallel_state.get_tp_world_size = lambda: 1
    parallel_state.get_ulysses_parallel_world_size = lambda: 1
    parallel_state.get_dit_group = lambda: None

    envs.SGLANG_CACHE_DIT_ATTN_BACKEND = os.environ.get("SGLANG_CACHE_DIT_ATTN_BACKEND")
    envs.SGLANG_CACHE_DIT_MINDIESD_COMPILE = (
        os.environ.get("SGLANG_CACHE_DIT_MINDIESD_COMPILE", "").lower() == "true"
    )

    class _FakeLogger:
        def debug(self, *_args, **_kwargs):
            pass

        def info(self, *_args, **_kwargs):
            pass

    logging_utils.init_logger = lambda _name: _FakeLogger()

    return {
        "sglang": sglang,
        "sglang.multimodal_gen": multimodal_gen,
        "sglang.multimodal_gen.envs": envs,
        "sglang.multimodal_gen.runtime": runtime,
        "sglang.multimodal_gen.runtime.distributed": distributed,
        "sglang.multimodal_gen.runtime.distributed.parallel_state": parallel_state,
        "sglang.multimodal_gen.runtime.utils": utils,
        "sglang.multimodal_gen.runtime.utils.logging_utils": logging_utils,
    }


def _install_torch_stub():
    torch = types.ModuleType("torch")
    torch_nn = types.ModuleType("torch.nn")
    torch_dist = types.ModuleType("torch.distributed")

    class _FakeModule:
        pass

    class _FakeProcessGroup:
        pass

    class _FakeReduceOp:
        AVG = "AVG"

    torch_nn.Module = _FakeModule
    torch_dist.ProcessGroup = _FakeProcessGroup
    torch_dist.ReduceOp = _FakeReduceOp
    torch.distributed = torch_dist
    torch.nn = torch_nn

    return {
        "torch": torch,
        "torch.nn": torch_nn,
        "torch.distributed": torch_dist,
    }


class TestCacheDitRefreshContext(unittest.TestCase):
    def _import_module_with_stub(self):
        stub_modules = _install_cache_dit_stub()
        stub_modules.update(_install_sglang_dependency_stubs())
        stub_modules.update(_install_torch_stub())
        module_path = (
            Path(__file__).resolve().parents[2]
            / "runtime"
            / "cache"
            / "cache_dit_integration.py"
        )
        with patch.dict(sys.modules, stub_modules):
            spec = importlib.util.spec_from_file_location(
                "test_cache_dit_integration_target", module_path
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
        return module

    def test_refresh_context_without_scm_preset_skips_steps_mask(self):
        module = self._import_module_with_stub()
        module.refresh_context_on_transformer(
            transformer="transformer",
            num_inference_steps=50,
            scm_preset=None,
            verbose=True,
        )

        self.assertEqual(module.cache_dit.steps_mask_calls, [])
        self.assertEqual(len(module.cache_dit.refresh_calls), 1)
        self.assertEqual(
            module.cache_dit.refresh_calls[0]["cache_config"],
            {
                "num_inference_steps": 50,
                "steps_computation_mask": None,
                "steps_computation_policy": None,
            },
        )

    def test_refresh_context_with_scm_preset_uses_steps_mask(self):
        module = self._import_module_with_stub()
        module.refresh_context_on_transformer(
            transformer="transformer",
            num_inference_steps=8,
            scm_preset="fast",
        )

        self.assertEqual(
            module.cache_dit.steps_mask_calls,
            [{"mask_policy": "fast", "total_steps": 8}],
        )
        self.assertEqual(
            module.cache_dit.refresh_calls[0]["cache_config"],
            {
                "num_inference_steps": 8,
                "steps_computation_mask": [1] * 8,
                "steps_computation_policy": "fast",
            },
        )

    def test_dual_refresh_without_scm_preset_skips_steps_mask(self):
        module = self._import_module_with_stub()
        module.refresh_context_on_dual_transformer(
            transformer="transformer",
            transformer_2="transformer_2",
            num_high_noise_steps=12,
            num_low_noise_steps=6,
            scm_preset=None,
        )

        self.assertEqual(module.cache_dit.steps_mask_calls, [])
        self.assertEqual(len(module.cache_dit.refresh_calls), 2)
        self.assertEqual(
            module.cache_dit.refresh_calls[0]["cache_config"],
            {
                "num_inference_steps": 12,
                "steps_computation_mask": None,
                "steps_computation_policy": None,
            },
        )
        self.assertEqual(
            module.cache_dit.refresh_calls[1]["cache_config"],
            {
                "num_inference_steps": 6,
                "steps_computation_mask": None,
                "steps_computation_policy": None,
            },
        )

    def _import_with_attn_env(self, attn_backend):
        saved = os.environ.get("SGLANG_CACHE_DIT_ATTN_BACKEND")
        if attn_backend is not None:
            os.environ["SGLANG_CACHE_DIT_ATTN_BACKEND"] = attn_backend
        else:
            os.environ.pop("SGLANG_CACHE_DIT_ATTN_BACKEND", None)
        try:
            stub_modules = _install_cache_dit_stub()
            stub_modules.update(_install_sglang_dependency_stubs())
            stub_modules.update(_install_torch_stub())
            module_path = (
                Path(__file__).resolve().parents[2]
                / "runtime"
                / "cache"
                / "cache_dit_integration.py"
            )
            with patch.dict(sys.modules, stub_modules):
                spec = importlib.util.spec_from_file_location(
                    "test_cache_dit_integration_target", module_path
                )
                module = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(module)
            return module
        finally:
            if saved is not None:
                os.environ["SGLANG_CACHE_DIT_ATTN_BACKEND"] = saved
            else:
                os.environ.pop("SGLANG_CACHE_DIT_ATTN_BACKEND", None)

    def test_enable_cache_passes_attention_backend_when_env_set(self):
        module = self._import_with_attn_env("_mindiesd_laser")
        transformer = type(
            "FakeTransformer",
            (),
            {"__class__": type("FakeCls", (), {"__name__": "FluxTransformer2DModel"})},
        )()
        module.enable_cache_on_transformer(
            transformer=transformer,
            config=module.CacheDitConfig(
                enabled=True,
                num_inference_steps=20,
            ),
            model_name="transformer",
            sp_group=None,
            tp_group=None,
        )
        self.assertEqual(len(module.cache_dit.enable_cache_calls), 1)
        self.assertEqual(
            module.cache_dit.enable_cache_calls[0]["attention_backend"],
            "_mindiesd_laser",
        )

    def test_enable_cache_skips_attention_backend_when_env_unset(self):
        module = self._import_with_attn_env(None)
        transformer = type(
            "FakeTransformer",
            (),
            {"__class__": type("FakeCls", (), {"__name__": "FluxTransformer2DModel"})},
        )()
        module.enable_cache_on_transformer(
            transformer=transformer,
            config=module.CacheDitConfig(
                enabled=True,
                num_inference_steps=20,
            ),
            model_name="transformer",
            sp_group=None,
            tp_group=None,
        )
        self.assertEqual(len(module.cache_dit.enable_cache_calls), 1)
        self.assertIsNone(module.cache_dit.enable_cache_calls[0]["attention_backend"])


if __name__ == "__main__":
    unittest.main()
