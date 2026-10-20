import unittest
from unittest import mock

import torch

from sglang.srt.utils import offloader
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class _Scale(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, device=device))
        self.error = None

    def forward(self, value, *, bias=0):
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        return value * self.weight + bias


class TestOffloaderForward(CustomTestCase):
    def test_successful_calls_keep_using_device_tensors(self):
        module = _Scale()
        end = mock.Mock()
        offloader._hook_module_forward_raw(
            module, end, lambda: {"weight": torch.tensor(3.0)}
        )
        wrapped = module.forward

        for _ in range(2):
            self.assertEqual(module(torch.tensor(4.0), bias=1).item(), 13.0)
            self.assertIs(module.forward, wrapped)
            self.assertEqual(module.weight.item(), 2.0)
        self.assertEqual(end.call_count, 2)

    def test_exceptions_preserve_wrapper_and_original_error(self):
        for phase in ("prepare", "forward", "end"):
            with self.subTest(phase=phase):
                module = _Scale()
                error = RuntimeError(f"{phase} failed")
                state = {"weight": torch.tensor(3.0)}
                prepare = mock.Mock(return_value=state)
                end = mock.Mock()
                if phase == "prepare":
                    prepare.side_effect = [error, state]
                elif phase == "forward":
                    module.error = error
                else:
                    end.side_effect = [error, None]
                offloader._hook_module_forward_raw(module, end, prepare)
                wrapped = module.forward

                with self.assertRaises(RuntimeError) as raised:
                    module(torch.tensor(4.0), bias=1)
                self.assertIs(raised.exception, error)
                self.assertIs(module.forward, wrapped)
                self.assertEqual(module.weight.item(), 2.0)
                self.assertEqual(end.call_count, int(phase == "end"))

                self.assertEqual(module(torch.tensor(4.0), bias=1).item(), 13.0)
                self.assertIs(module.forward, wrapped)
                self.assertEqual(module.weight.item(), 2.0)
                self.assertEqual(prepare.call_count, 2)
                self.assertEqual(end.call_count, 1 + int(phase == "end"))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestOffloaderV1Forward(CustomTestCase):
    def test_exceptions_preserve_cpu_offload(self):
        for phase in ("prepare", "forward"):
            with self.subTest(phase=phase):
                module = _Scale(device="cuda")
                offloader.OffloaderV1(cpu_offload_max_bytes=1024).maybe_offload_to_cpu(
                    module
                )
                wrapped = module.forward
                self.assertEqual(module.weight.device.type, "cpu")
                error = RuntimeError(f"{phase} failed")
                if phase == "forward":
                    module.error = error
                    with self.assertRaises(RuntimeError) as raised:
                        module(torch.tensor(4.0, device="cuda"), bias=1)
                else:
                    with mock.patch.object(
                        offloader, "_get_offloaded_device_state", side_effect=error
                    ):
                        with self.assertRaises(RuntimeError) as raised:
                            module(torch.tensor(4.0, device="cuda"), bias=1)
                self.assertIs(raised.exception, error)
                self.assertIs(module.forward, wrapped)
                self.assertEqual(module.weight.device.type, "cpu")

                self.assertEqual(
                    module(torch.tensor(4.0, device="cuda"), bias=1).item(), 9.0
                )
                self.assertIs(module.forward, wrapped)
                self.assertEqual(module.weight.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
