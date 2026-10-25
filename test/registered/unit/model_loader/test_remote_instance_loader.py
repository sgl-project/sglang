"""Unit tests for RemoteInstanceModelLoader construction - no server, no weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.model_loader.loader as loader_mod
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

EXTRA_CONFIG = {"enable_multithread_load": True, "num_threads": 64}

# ServerArgs forwards the backend as a plain str, so every case covers both that
# and the enum member; the loader's comparison holds only via the str mixin.
MODELEXPRESS_BACKENDS = (
    RemoteInstanceWeightLoaderBackend.MODELEXPRESS,
    "modelexpress",
)
OTHER_BACKENDS = (
    RemoteInstanceWeightLoaderBackend.NCCL,
    "nccl",
    RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE,
    "transfer_engine",
)
ALL_BACKENDS = MODELEXPRESS_BACKENDS + OTHER_BACKENDS


def _load_config(backend, extra_config=None):
    return LoadConfig(
        load_format=LoadFormat.REMOTE_INSTANCE,
        model_loader_extra_config=extra_config or {},
        remote_instance_weight_loader_backend=backend,
    )


class TestRemoteInstanceModelLoaderExtraConfig(CustomTestCase):
    def test_modelexpress_backend_accepts_extra_config(self):
        for backend in MODELEXPRESS_BACKENDS:
            with self.subTest(backend=backend):
                loader = loader_mod.RemoteInstanceModelLoader(
                    _load_config(backend, EXTRA_CONFIG)
                )
                # ModelExpress hands the extra config to the DefaultModelLoader
                # it falls back to, so it must survive construction unmodified.
                self.assertEqual(
                    loader.load_config.model_loader_extra_config, EXTRA_CONFIG
                )

    def test_modelexpress_backend_rejects_unknown_extra_config_key(self):
        for backend in MODELEXPRESS_BACKENDS:
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    loader_mod.RemoteInstanceModelLoader(
                        _load_config(backend, {"num_thread": 64})
                    )

    def test_other_backends_reject_extra_config(self):
        for backend in OTHER_BACKENDS:
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    loader_mod.RemoteInstanceModelLoader(
                        _load_config(backend, EXTRA_CONFIG)
                    )

    def test_all_backends_construct_without_extra_config(self):
        for backend in ALL_BACKENDS:
            with self.subTest(backend=backend):
                loader = loader_mod.RemoteInstanceModelLoader(_load_config(backend))
                self.assertFalse(loader.load_config.model_loader_extra_config)


class _FakeMxLoader:
    def __init__(self):
        self.calls = []

    def before_weights_release(self, **kwargs):
        self.calls.append(("before_weights_release", kwargs))

    def after_weights_resume(self, **kwargs):
        self.calls.append(("after_weights_resume", kwargs))

    def reload_model(self, **kwargs):
        self.calls.append(("reload_model", kwargs))


class _FakeMxLoaderWithoutHooks:
    """A modelexpress install that predates the release/resume/reload hooks."""


def _modelexpress_loader(mx_loader):
    loader = loader_mod.RemoteInstanceModelLoader(_load_config("modelexpress"))
    loader._mx_loader = mx_loader
    return loader


class TestRemoteInstanceModelLoaderWeightHooks(CustomTestCase):
    def test_hooks_forward_to_modelexpress(self):
        mx_loader = _FakeMxLoader()
        loader = _modelexpress_loader(mx_loader)
        model, model_config, device_config = object(), object(), object()

        loader.before_weights_release(content_preserved=False)
        loader.after_weights_resume(content_preserved=False)
        reloaded = loader.reload_weights_inplace(
            model=model, model_config=model_config, device_config=device_config
        )

        self.assertTrue(reloaded)
        self.assertEqual(
            mx_loader.calls,
            [
                ("before_weights_release", {"content_preserved": False}),
                ("after_weights_resume", {"content_preserved": False}),
                (
                    "reload_model",
                    {
                        "model": model,
                        "model_config": model_config,
                        "device_config": device_config,
                    },
                ),
            ],
        )

    def test_hooks_tolerate_modelexpress_without_them(self):
        loader = _modelexpress_loader(_FakeMxLoaderWithoutHooks())

        loader.before_weights_release(content_preserved=True)
        loader.after_weights_resume(content_preserved=True)
        self.assertFalse(
            loader.reload_weights_inplace(
                model=object(), model_config=object(), device_config=object()
            )
        )

    def test_hooks_are_inert_without_modelexpress(self):
        for backend in OTHER_BACKENDS:
            with self.subTest(backend=backend):
                loader = loader_mod.RemoteInstanceModelLoader(_load_config(backend))
                loader.before_weights_release(content_preserved=False)
                loader.after_weights_resume(content_preserved=False)
                self.assertFalse(
                    loader.reload_weights_inplace(
                        model=object(), model_config=object(), device_config=object()
                    )
                )


class TestWeightUpdaterRemoteInstanceReload(CustomTestCase):
    def setUp(self):
        module = "sglang.srt.model_executor.model_runner_components.weight_updater"
        for target, kwargs in (
            (
                f"{module}.get_model",
                {"return_value": SimpleNamespace(weight_cache_mode="off")},
            ),
            (f"{module}.get_available_gpu_memory", {"return_value": 0.0}),
        ):
            patcher = patch(target, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.model = torch.nn.Linear(2, 2)
        self.model_config = SimpleNamespace(model_path="served")
        self.mx_loader = _FakeMxLoader()
        self.recapture_cuda_graph = Mock()
        # cuda, so a recapture the reload should skip would actually run.
        self.updater = WeightUpdater(
            tp_rank=0,
            device="cuda",
            gpu_id=0,
            model_config=self.model_config,
            custom_weight_loaders={},
            get_model=lambda: self.model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=self.recapture_cuda_graph,
            get_model_runner=lambda: SimpleNamespace(
                loader=_modelexpress_loader(self.mx_loader)
            ),
        )

    def test_reloads_the_served_model_through_its_own_loader(self):
        success, _ = self.updater.update_weights_from_disk(
            "served", LoadFormat.REMOTE_INSTANCE, recapture_cuda_graph=True
        )

        self.assertTrue(success)
        [(name, kwargs)] = self.mx_loader.calls
        self.assertEqual(name, "reload_model")
        self.assertIs(kwargs["model"], self.model)
        self.assertEqual(self.model_config.model_path, "served")
        self.recapture_cuda_graph.assert_not_called()

    def test_loader_without_in_place_reload_is_reported(self):
        self.mx_loader = _FakeMxLoaderWithoutHooks()

        success, message = self.updater.update_weights_from_disk(
            "served", LoadFormat.REMOTE_INSTANCE
        )

        self.assertFalse(success)
        self.assertIn("cannot reload weights in place", message)

    def test_rejects_another_model_path_without_changing_the_served_one(self):
        """The default disk path rewrites model_config.model_path before loading;
        a remote_instance reload must reject a different path before that."""
        success, _ = self.updater.update_weights_from_disk(
            "other", LoadFormat.REMOTE_INSTANCE
        )

        self.assertFalse(success)
        self.assertEqual(self.model_config.model_path, "served")
        self.assertEqual(self.mx_loader.calls, [])

    def test_failed_reload_is_reported_not_raised(self):
        """A reload failure (e.g. an unreachable ModelExpress server) must fail the
        request instead of stopping the scheduler, so the caller can retry."""
        self.mx_loader.reload_model = Mock(side_effect=RuntimeError("unreachable"))

        success, message = self.updater.update_weights_from_disk(
            "served", LoadFormat.REMOTE_INSTANCE
        )

        self.assertFalse(success)
        self.assertIn("unreachable", message)


if __name__ == "__main__":
    unittest.main()
