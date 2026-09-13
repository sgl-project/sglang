# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
from torch import nn

from sglang.srt.model_loader.loader import device_loading_context
from sglang.srt.model_loader.post_load import stage_module_for_post_load
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")


def _process_device() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return None


PROCESS_DEVICE = _process_device()


@unittest.skipUnless(PROCESS_DEVICE is not None, "requires CUDA")
class TestModulePostLoadStaging(unittest.TestCase):
    def test_preserves_unchanged_parameter_and_buffer_storage(self):
        module = nn.Module()
        module.weight = nn.Parameter(torch.arange(4.0).reshape(2, 2))
        module.register_buffer("scale", torch.ones(2))
        weight = module.weight
        scale = module.scale
        weight_ptr = weight.data_ptr()
        scale_ptr = scale.data_ptr()

        with stage_module_for_post_load(module, PROCESS_DEVICE):
            self.assertEqual(module.weight.device, PROCESS_DEVICE)
            self.assertEqual(module.scale.device, PROCESS_DEVICE)
            module.weight.data.add_(1)
            module.scale.add_(2)

        self.assertIs(module.weight, weight)
        self.assertIs(module.scale, scale)
        self.assertEqual(module.weight.data_ptr(), weight_ptr)
        self.assertEqual(module.scale.data_ptr(), scale_ptr)
        torch.testing.assert_close(module.weight, torch.arange(4.0).reshape(2, 2) + 1)
        torch.testing.assert_close(module.scale, torch.full((2,), 3.0))

    def test_restores_replacements_and_new_state_with_mixed_residency(self):
        module = nn.Module()
        module.weight = nn.Parameter(torch.ones(2, 2))
        module.register_buffer("resident", torch.ones(1, device=PROCESS_DEVICE))
        module.child = nn.Module()
        module.child.scale = nn.Parameter(torch.ones(2))

        with stage_module_for_post_load(module, PROCESS_DEVICE):
            for tensor in (*module.parameters(), *module.buffers()):
                self.assertEqual(tensor.device, PROCESS_DEVICE)
            module.weight = nn.Parameter(torch.ones(3, 2, device=PROCESS_DEVICE))
            del module.child._parameters["scale"]
            module.child.register_buffer(
                "scale", torch.ones(3, device=PROCESS_DEVICE), persistent=False
            )
            module.new_parameter = nn.Parameter(torch.ones(1, device=PROCESS_DEVICE))
            module.child.register_buffer(
                "new_buffer", torch.ones(1, device=PROCESS_DEVICE)
            )
            module.child.register_buffer(
                "runtime_buffer", torch.ones(1, device=PROCESS_DEVICE), persistent=False
            )

        # ordinary replacements inherit origin; nonpersistent buffers stay executable
        self.assertEqual(module.weight.device.type, "cpu")
        self.assertEqual(module.weight.shape, (3, 2))
        self.assertEqual(module.child.scale.device, PROCESS_DEVICE)
        self.assertIn("scale", module.child._buffers)
        # the child's unique origin wins over the mixed module residency
        self.assertEqual(module.child.new_buffer.device.type, "cpu")
        # non-persistent runtime state is absent from offloader state_dicts
        self.assertEqual(module.child.runtime_buffer.device, PROCESS_DEVICE)
        # a new tensor owned by the mixed root remains on the process device
        self.assertEqual(module.new_parameter.device, PROCESS_DEVICE)
        self.assertEqual(module.resident.device, PROCESS_DEVICE)

    def test_restores_existing_and_new_state_after_hook_error(self):
        module = nn.Module()
        module.weight = nn.Parameter(torch.ones(2, 2))
        module.register_buffer("scale", torch.ones(1))

        with self.assertRaisesRegex(RuntimeError, "hook failed"):
            with stage_module_for_post_load(module, PROCESS_DEVICE):
                module.weight.data.add_(1)
                module.register_buffer(
                    "workspace", torch.ones(1, device=PROCESS_DEVICE)
                )
                raise RuntimeError("hook failed")

        self.assertEqual(module.weight.device.type, "cpu")
        self.assertEqual(module.scale.device.type, "cpu")
        self.assertEqual(module.workspace.device.type, "cpu")

    def test_preserves_registered_alias(self):
        module = nn.Module()
        shared = nn.Parameter(torch.ones(2))
        module.left = shared
        module.right = shared

        with stage_module_for_post_load(module, PROCESS_DEVICE):
            self.assertIs(module.left, module.right)
            module.left.data.mul_(2)

        self.assertIs(module.left, shared)
        self.assertIs(module.right, shared)
        self.assertEqual(module.left.device.type, "cpu")
        torch.testing.assert_close(module.left, torch.full((2,), 2.0))

    def test_rejects_alias_with_conflicting_restore_devices_before_restoring(self):
        module = nn.Module()
        module.left = nn.Parameter(torch.ones(1))
        module.child = nn.Module()
        module.child.right = nn.Parameter(torch.ones(1, device=PROCESS_DEVICE))

        with self.assertRaisesRegex(RuntimeError, "conflicting restore devices"):
            with stage_module_for_post_load(module, PROCESS_DEVICE):
                shared = nn.Parameter(torch.ones(1, device=PROCESS_DEVICE))
                module.left = shared
                module.child.right = shared

        self.assertIs(module.left, module.child.right)
        self.assertEqual(module.left.device, PROCESS_DEVICE)


class TestModulePostLoadValidation(unittest.TestCase):
    @unittest.skipUnless(PROCESS_DEVICE is not None, "requires CUDA")
    def test_srt_cpu_loading_context_remains_a_noop(self):
        module = nn.Module()
        module.weight = nn.Parameter(torch.ones(1, device=PROCESS_DEVICE))

        with device_loading_context(module, torch.device("cpu")):
            self.assertEqual(module.weight.device, PROCESS_DEVICE)

        self.assertEqual(module.weight.device, PROCESS_DEVICE)

    def test_rejects_meta_before_moving_other_state(self):
        module = nn.Module()
        module.meta_weight = nn.Parameter(torch.empty(1, device="meta"))
        module.register_buffer("scale", torch.ones(1))
        scale_ptr = module.scale.data_ptr()

        with self.assertRaisesRegex(RuntimeError, "meta tensor"):
            with stage_module_for_post_load(module, torch.device("cpu")):
                self.fail("context should not be entered")

        self.assertEqual(module.scale.device.type, "cpu")
        self.assertEqual(module.scale.data_ptr(), scale_ptr)


if __name__ == "__main__":
    unittest.main()
