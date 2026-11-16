import gc
import inspect
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers import offloader  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOffloaderFactory(CustomTestCase):
    def setUp(self):
        self.enterContext(patch.object(offloader, "_offloader_factory", None))
        self.offload_config = SimpleNamespace(
            cpu_offload_gb=0,
            offload_group_size=0,
            offload_num_in_group=1,
            offload_prefetch_step=2,
            offload_mode="cpu",
        )
        self.model_config = SimpleNamespace(is_startup_weight_load_overlap=False)
        self.enterContext(
            patch.object(
                offloader,
                "get_exec",
                return_value=SimpleNamespace(offload=self.offload_config),
            )
        )
        self.enterContext(
            patch.object(offloader, "get_model", return_value=self.model_config)
        )
        self.context = offloader.OffloaderContext(
            device="cpu",
            gpu_id=0,
            tp_rank=0,
            model_config=object(),
            is_draft_worker=False,
        )

    def test_no_factory_preserves_native_selection_without_context(self):
        self.assertIsInstance(offloader.create_offloader(), offloader.NoopOffloader)
        self.offload_config.cpu_offload_gb = 2
        selected = offloader.create_offloader()
        self.assertIsInstance(selected, offloader.OffloaderV1)
        self.assertEqual(selected._cpu_offload_max_bytes, 2 * 1024**3)

        self.offload_config.cpu_offload_gb = 0
        self.offload_config.offload_group_size = 3
        with patch.object(offloader, "OffloaderV2") as native:
            self.assertIs(offloader.create_offloader(), native.return_value)
            native.assert_called_once_with(
                group_size=3, num_in_group=1, prefetch_step=2, mode="cpu"
            )

    def test_factory_receives_context_and_creates_per_runner_instances(self):
        contexts = []

        def factory(context):
            contexts.append(context)
            return offloader.BaseOffloader()

        offloader.register_offloader_factory(factory)
        target = offloader.create_offloader(self.context)
        draft_context = offloader.OffloaderContext(
            device="cpu",
            gpu_id=0,
            tp_rank=0,
            model_config=object(),
            is_draft_worker=True,
        )
        draft = offloader.create_offloader(draft_context)
        self.assertIs(contexts[0], self.context)
        self.assertIs(contexts[1], draft_context)
        self.assertIsNot(target, draft)
        with self.assertRaises(AttributeError):
            self.context.is_draft_worker = True

    def test_registration_is_idempotent_but_rejects_another_provider(self):
        def factory(context):
            return None

        offloader.register_offloader_factory(factory)
        offloader.register_offloader_factory(factory)
        with self.assertRaisesRegex(RuntimeError, "already registered"):
            offloader.register_offloader_factory(lambda context: None)
        self.assertIs(offloader._offloader_factory, factory)

    def test_invalid_factory_and_result_are_rejected(self):
        with self.assertRaisesRegex(TypeError, "must be callable"):
            offloader.register_offloader_factory(None)
        offloader.register_offloader_factory(lambda context: object())
        with self.assertRaisesRegex(TypeError, "BaseOffloader or None"):
            offloader.create_offloader(self.context)

    def test_registered_factory_requires_runner_context(self):
        factory = Mock(return_value=None)
        offloader.register_offloader_factory(factory)
        with self.assertRaisesRegex(ValueError, "requires OffloaderContext"):
            offloader.create_offloader()
        factory.assert_not_called()

    def test_declining_provider_keeps_native_selection_and_overlap(self):
        offloader.register_offloader_factory(lambda context: None)
        self.model_config.is_startup_weight_load_overlap = True
        self.assertIsInstance(
            offloader.create_offloader(self.context), offloader.NoopOffloader
        )
        self.offload_config.cpu_offload_gb = 1
        self.assertIsInstance(
            offloader.create_offloader(self.context), offloader.OffloaderV1
        )
        self.offload_config.cpu_offload_gb = 0
        self.offload_config.offload_group_size = 3
        with patch.object(offloader, "OffloaderV2") as native:
            self.assertIs(offloader.create_offloader(self.context), native.return_value)

    def test_selected_provider_rejects_native_offload_flags(self):
        offloader.register_offloader_factory(lambda context: offloader.BaseOffloader())
        for name in ("cpu_offload_gb", "offload_group_size"):
            with self.subTest(flag=name), patch.object(self.offload_config, name, 1):
                with self.assertRaisesRegex(ValueError, "cannot be combined"):
                    offloader.create_offloader(self.context)

    def test_selected_provider_rejects_startup_overlap(self):
        offloader.register_offloader_factory(lambda context: offloader.BaseOffloader())
        self.model_config.is_startup_weight_load_overlap = True
        with self.assertRaisesRegex(ValueError, "startup-weight-load-mode=overlap"):
            offloader.create_offloader(self.context)

    def test_factory_failure_propagates(self):
        factory = Mock(side_effect=RuntimeError("placement setup failed"))
        offloader.register_offloader_factory(factory)
        with self.assertRaisesRegex(RuntimeError, "placement setup failed"):
            offloader.create_offloader(self.context)
        factory.assert_called_once_with(self.context)


class TestOffloaderContract(CustomTestCase):
    def test_documented_hook_signatures(self):
        expected = {
            offloader.BaseOffloader.wrap_modules: (
                "self",
                "all_modules_generator",
                "submodule_accessor",
                "whitelist_param_names_creator",
            ),
            offloader.BaseOffloader.post_init: ("self",),
            offloader.BaseOffloader.post_load_model: ("self", "model"),
        }
        for method, parameters in expected.items():
            with self.subTest(method=method.__name__):
                signature = inspect.signature(method)
                self.assertEqual(tuple(signature.parameters), parameters)
                signature.bind(*([None] * len(parameters)))


class TestLiveOffloaderRestrictions(CustomTestCase):
    def setUp(self):
        self.enterContext(
            patch.object(offloader, "_instance", offloader.NoopOffloader())
        )
        self.enterContext(
            patch.object(offloader, "_live_offloaders", weakref.WeakValueDictionary())
        )

    def test_draft_does_not_clear_live_target_copy_engine_restriction(self):
        class AsyncOffloader(offloader.BaseOffloader):
            # User implementations need not be hashable.
            __hash__ = None

            @property
            def forbid_copy_engine_usage(self):
                return True

        self.assertFalse(offloader.forbid_copy_engine_usage())
        target = AsyncOffloader()
        offloader.set_offloader(target)
        offloader.set_offloader(offloader.NoopOffloader())
        self.assertTrue(offloader.forbid_copy_engine_usage())
        reference = weakref.ref(target)
        del target
        gc.collect()
        self.assertIsNone(reference())
        self.assertFalse(offloader.forbid_copy_engine_usage())


if __name__ == "__main__":
    unittest.main()
