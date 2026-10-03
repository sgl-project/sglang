import importlib
import sys
import types
from contextlib import ExitStack
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.test.test_utils import CustomTestCase

_DEEPEP_MODULE = "sglang.srt.layers.moe.token_dispatcher.deepep"
_HOOK_MODULE = "sglang.srt.hardware_backend.ppu.moe.ppu_deepep_hooks"


class TestPPUDeepEPHooks(CustomTestCase):
    def setUp(self):
        HookRegistry.reset()
        self._original_hook_module = sys.modules.pop(_HOOK_MODULE, None)

    def tearDown(self):
        HookRegistry.reset()
        if self._original_hook_module is None:
            sys.modules.pop(_HOOK_MODULE, None)
        else:
            sys.modules[_HOOK_MODULE] = self._original_hook_module

    def test_mnnvl_fabric_support_hook_replaces_deepep_probe(self):
        original_probe = Mock(return_value=None)
        deepep_module = types.ModuleType(_DEEPEP_MODULE)
        deepep_module._is_mnnvl_fabric_supported = original_probe

        with ExitStack() as stack:
            parent = importlib.import_module("sglang.srt")
            module_name = parent.__name__
            for child_name in ("layers", "moe", "token_dispatcher", "deepep"):
                module_name = f"{module_name}.{child_name}"
                child = (
                    deepep_module
                    if module_name == _DEEPEP_MODULE
                    else sys.modules.get(module_name, types.ModuleType(module_name))
                )
                stack.enter_context(patch.dict(sys.modules, {module_name: child}))
                stack.enter_context(
                    patch.object(parent, child_name, child, create=True)
                )
                parent = child

            importlib.import_module(_HOOK_MODULE)
            HookRegistry.apply_hooks()

            with envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.override(False):
                self.assertFalse(deepep_module._is_mnnvl_fabric_supported())
            with envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.override(True):
                self.assertTrue(deepep_module._is_mnnvl_fabric_supported())
            original_probe.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
