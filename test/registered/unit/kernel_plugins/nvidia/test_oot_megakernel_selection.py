"""CPU tests for selecting an out-of-tree megakernel without a vendor SDK."""

import pickle
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from msgspec.structs import replace

from sglang.kernels.plugin_kernels.megakernel import spec as registry
from sglang.srt import platforms
from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.kernel_plugins import megakernel
from sglang.srt.kernel_plugins.nvidia import binding
from sglang.srt.platforms.interface import (
    PlatformCapabilities,
    PlatformEnum,
    SRTPlatform,
)
from sglang.test.ci.ci_register import register_cpu_ci

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import ARCH, SPEC, server_args  # noqa: E402

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")

TARGET = "fake_accelerator"
PATH = "fake_oot_megakernel:SPEC"


class FakePlatform(SRTPlatform):
    _enum = PlatformEnum.OOT
    device_name = "fake"
    capabilities = PlatformCapabilities(supports_megakernel=True)
    target = TARGET
    required = False

    def get_megakernel_target(self):
        return self.target

    def requires_megakernel(self, server_args, hf_config):
        return self.required


class TestOutOfTreeMegakernelSelection(unittest.TestCase):
    def setUp(self):
        self.platform = FakePlatform()
        self.model = SimpleNamespace(
            hf_config=SimpleNamespace(architectures=[ARCH]), dtype="bfloat16"
        )
        self.provider = SimpleNamespace(
            SPEC=replace(
                SPEC, requires=replace(SPEC.requires, isas=frozenset({TARGET}))
            )
        )
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.object(platforms, "_current_platform", self.platform))
        stack.enter_context(
            patch.object(megakernel, "model_config_of", return_value=self.model)
        )
        self.load_plugins = stack.enter_context(
            patch.object(megakernel, "load_plugins")
        )
        stack.enter_context(patch.dict(registry._REGISTRY, clear=True))
        stack.enter_context(
            patch.dict(sys.modules, {"fake_oot_megakernel": self.provider})
        )
        for name in (
            "is_available",
            "current_device",
            "get_device_properties",
            "mem_get_info",
        ):
            stack.enter_context(
                patch.object(
                    binding.torch.cuda,
                    name,
                    side_effect=AssertionError(f"unexpected CUDA probe: {name}"),
                )
            )
        stack.enter_context(
            patch.object(
                binding,
                "plugin_reasons",
                side_effect=AssertionError("unexpected NVIDIA entry point check"),
            )
        )

    def register_provider(self, path=PATH):
        registry.register(architecture=ARCH, target=TARGET, path=path)

    def select(self, **options):
        args = server_args(**options)
        megakernel.select_oot_megakernel(args)
        return args

    def test_automatic_selection_records_a_lazy_picklable_identity(self):
        registry.register(architecture=ARCH, path="unused_cuda_provider:SPEC")
        self.register_provider()
        with patch.object(registry.importlib, "import_module") as import_module:
            args = self.select()
            import_module.assert_not_called()
        self.assertEqual(
            megakernel.get_oot_megakernel_selection(args),
            megakernel.MegakernelSelection(
                ARCH, TARGET, PATH, registry.MEGAKERNEL_API_VERSION
            ),
        )
        megakernel.validate_oot_megakernel(args)
        child_args = pickle.loads(pickle.dumps(args))
        self.assertIs(
            megakernel.load_selected_oot_megakernel(child_args), self.provider.SPEC
        )

    def test_architecture_default_and_foreign_target_keep_the_regular_path(self):
        registry.register(architecture=ARCH, path="unused_cuda_provider:SPEC")
        registry.register(
            architecture=ARCH, target="different", path="unused_other:SPEC"
        )
        with patch.object(registry.importlib, "import_module") as import_module:
            args = self.select()
            megakernel.validate_oot_megakernel(args)
            import_module.assert_not_called()

    def test_required_launch_refuses_when_the_plugin_did_not_register(self):
        self.platform.required = True
        with self.assertRaisesRegex(
            ValueError, f"no provider is registered.*{ARCH}.*{TARGET}"
        ):
            self.select()

    def test_launcher_loads_plugins_before_looking_up_registrations(self):
        self.load_plugins.side_effect = self.register_provider
        self.assertEqual(
            megakernel.get_oot_megakernel_selection(self.select()).path, PATH
        )

    def test_legacy_flag_is_a_warned_explicit_request_on_opted_in_platforms(self):
        self.platform.capabilities = PlatformCapabilities(
            supports_megakernel=True, accepts_legacy_megakernel_flag=True
        )
        with self.assertRaisesRegex(ValueError, f"no provider is registered.*{TARGET}"):
            self.select(enable_nvidia_megakernel=True)
        self.register_provider()
        with self.assertLogs(megakernel.logger, "WARNING") as logs:
            args = self.select(enable_nvidia_megakernel=True)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("deprecated", logs.output[0])
        self.assertTrue(args.enable_nvidia_megakernel)
        self.assertFalse(resolving_view(args).enable_nvidia_megakernel)

    def test_platform_without_legacy_alias_rejects_explicit_flag(self):
        self.register_provider()
        with self.assertRaisesRegex(ValueError, "unsupported on fake"):
            self.select(enable_nvidia_megakernel=True)

    def test_in_tree_selection_leaves_the_nvidia_flag_alone(self):
        self.platform._enum = PlatformEnum.CUDA
        self.register_provider()
        args = self.select(enable_nvidia_megakernel=True)
        self.assertIsNone(megakernel.get_oot_megakernel_selection(args))
        self.assertTrue(resolving_view(args).enable_nvidia_megakernel)

    def test_invalid_match_refuses_with_the_registry_reason(self):
        self.register_provider("provider_does_not_exist:SPEC")
        with self.assertRaisesRegex(
            ValueError, f"provider_does_not_exist.*{TARGET}.*does not import"
        ):
            megakernel.validate_oot_megakernel(self.select())

    def test_launch_validation_reports_incompatible_config(self):
        self.register_provider()
        with self.assertRaisesRegex(ValueError, "tp_size 8"):
            megakernel.validate_oot_megakernel(self.select(tp_size=8))

    def test_invalid_provider_api_refuses_before_worker_startup(self):
        self.register_provider()
        self.provider.SPEC = replace(self.provider.SPEC, api_version=1)
        with self.assertRaisesRegex(ValueError, "uses megakernel API version 1"):
            megakernel.validate_oot_megakernel(self.select())

    def test_child_refuses_missing_registration_changed_path_and_changed_target(self):
        self.register_provider()
        args = self.select()
        registry._REGISTRY.clear()
        with self.assertRaisesRegex(ValueError, "the launcher selected"):
            megakernel.load_selected_oot_megakernel(args)
        self.register_provider("different_provider:SPEC")
        with self.assertRaisesRegex(
            ValueError, f"different_provider.*the launcher selected"
        ):
            megakernel.load_selected_oot_megakernel(args)
        self.platform.target = "different"
        with self.assertRaisesRegex(ValueError, "this process cannot serve"):
            megakernel.load_selected_oot_megakernel(args)

    def test_install_and_prepare_platform_delegate_once(self):
        self.register_provider()
        args = self.select()
        model = SimpleNamespace()
        runner = SimpleNamespace(
            server_args=args,
            model=model,
            model_config=self.model,
            is_draft_worker=False,
            decode_num_tokens_per_req=lambda: 1,
        )
        calls = []
        built = SimpleNamespace(widths=(1, 2))

        def dispatch(*args):
            pass

        dispatch.prepare = lambda mr: calls.append(mr)
        self.provider.SPEC = replace(self.provider.SPEC, build=lambda *args: built)
        self.platform.install_megakernel_dispatch = lambda **kwargs: dispatch
        with (
            patch.object(megakernel, "_gather_tp", side_effect=lambda obj: [obj]),
            patch.object(
                megakernel,
                "get_device",
                return_value=SimpleNamespace(watchdog_timeout=None),
            ),
        ):
            megakernel.install_oot_megakernel(runner)
            megakernel.prepare_oot_megakernel(runner)
            megakernel.prepare_oot_megakernel(runner)
        self.assertIs(megakernel.oot_built_megakernel(model), built)
        self.assertIs(megakernel.oot_megakernel_dispatch(model), dispatch)
        self.assertEqual(calls, [runner])

    def test_local_phase_reports_remote_failure_before_the_next_phase(self):
        with patch.object(
            megakernel, "_gather_tp", return_value=[None, "ValueError: invalid layout"]
        ):
            with self.assertRaisesRegex(
                ValueError, "rank 1: ValueError: invalid layout"
            ):
                megakernel._local_phase("build", lambda: None)


if __name__ == "__main__":
    unittest.main()
