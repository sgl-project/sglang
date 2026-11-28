"""Dynamic grad contexts follow policy at entry and restore the caller."""

import copy
import unittest

import torch
from torch import nn

from sglang.srt.utils import common
from sglang.srt.utils.common import DynamicGradMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDynamicGradModePolicy(unittest.TestCase):
    def setUp(self):
        self.original_policy = common._ENABLE_TORCH_INFERENCE_MODE
        self.inference_guard = torch.inference_mode(False)
        self.inference_guard.__enter__()
        self.grad_guard = torch.enable_grad()
        self.grad_guard.__enter__()

    def tearDown(self):
        DynamicGradMode.set_inference_mode(self.original_policy)
        self.grad_guard.__exit__(None, None, None)
        self.inference_guard.__exit__(None, None, None)

    def assert_policy(self, inference):
        self.assertFalse(torch.is_grad_enabled())
        self.assertEqual(torch.is_inference_mode_enabled(), inference)

    def assert_caller_restored(self):
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_decorator_created_before_enabling_inference(self):
        for bare in (False, True):
            with self.subTest(bare=bare):
                DynamicGradMode.set_inference_mode(False)

                def create():
                    self.assert_policy(True)
                    return torch.ones(2)

                decorated = (
                    DynamicGradMode(create) if bare else DynamicGradMode()(create)
                )
                DynamicGradMode.set_inference_mode(True)
                output = decorated()
                self.assertTrue(output.is_inference())
                self.assert_caller_restored()

    def test_decorator_created_before_disabling_inference(self):
        for bare in (False, True):
            with self.subTest(bare=bare):
                DynamicGradMode.set_inference_mode(True)

                def create():
                    self.assert_policy(False)
                    return torch.ones(2)

                decorated = (
                    DynamicGradMode(create) if bare else DynamicGradMode()(create)
                )
                DynamicGradMode.set_inference_mode(False)
                output = decorated()
                self.assertFalse(output.is_inference())
                self.assert_caller_restored()

    def test_context_constructed_before_policy_change(self):
        for initial in (False, True):
            with self.subTest(initial=initial):
                DynamicGradMode.set_inference_mode(initial)
                context = DynamicGradMode()
                DynamicGradMode.set_inference_mode(not initial)
                with context:
                    self.assert_policy(not initial)
                self.assert_caller_restored()

    def test_exit_uses_policy_selected_on_entry(self):
        for initial in (False, True):
            with self.subTest(initial=initial):
                with torch.inference_mode(False), torch.enable_grad():
                    DynamicGradMode.set_inference_mode(initial)
                    with DynamicGradMode():
                        self.assert_policy(initial)
                        DynamicGradMode.set_inference_mode(not initial)
                        self.assert_policy(initial)
                    self.assert_caller_restored()

    def test_exception_exit_preserves_original_error_and_grad_state(self):
        for initial in (False, True):
            with self.subTest(initial=initial):
                with torch.inference_mode(False), torch.enable_grad():
                    DynamicGradMode.set_inference_mode(initial)
                    error = RuntimeError("model forward failed")
                    with self.assertRaises(RuntimeError) as caught:
                        with DynamicGradMode():
                            DynamicGradMode.set_inference_mode(not initial)
                            raise error
                    self.assertIs(caught.exception, error)
                    self.assert_caller_restored()

    def test_explicit_false_mode_survives_cloning_and_policy_change(self):
        for initial in (False, True):
            with self.subTest(initial=initial):
                DynamicGradMode.set_inference_mode(initial)
                context = DynamicGradMode(False)
                clone = context.clone()
                decorated = context(lambda: (torch.is_grad_enabled(), torch.ones(2)))
                DynamicGradMode.set_inference_mode(True)
                with clone:
                    self.assert_caller_restored()
                enabled, output = decorated()
                self.assertTrue(enabled)
                self.assertFalse(output.is_inference())
                self.assert_caller_restored()

    def test_one_decorator_supports_repeated_policy_changes(self):
        DynamicGradMode.set_inference_mode(False)

        @DynamicGradMode()
        def create():
            return torch.is_grad_enabled(), torch.ones(2)

        for inference in (False, True, False, True, True, False):
            with self.subTest(inference=inference):
                DynamicGradMode.set_inference_mode(inference)
                enabled, output = create()
                self.assertFalse(enabled)
                self.assertEqual(output.is_inference(), inference)
                self.assert_caller_restored()

    def test_native_model_inference_then_training_updates(self):
        DynamicGradMode.set_inference_mode(False)
        model = nn.Linear(3, 2, dtype=torch.float64)
        with torch.no_grad():
            model.weight.copy_(torch.arange(6, dtype=torch.float64).reshape(2, 3) / 8)
            model.bias.copy_(torch.tensor([0.2, -0.3], dtype=torch.float64))
        reference = copy.deepcopy(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
        inputs = torch.tensor([[1.0, -2.0, 0.5], [-1.5, 0.2, 2.0]], dtype=torch.float64)
        target = torch.tensor([[0.1, -0.4], [0.5, 0.7]], dtype=torch.float64)

        @DynamicGradMode()
        def predict():
            return model(inputs)

        for inference in (True, False, True):
            DynamicGradMode.set_inference_mode(inference)
            actual = predict()
            with torch.inference_mode(inference), torch.no_grad():
                expected = reference(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertFalse(actual.requires_grad)
            self.assertEqual(actual.is_inference(), inference)
            self.assertTrue(model.training)
            self.assert_caller_restored()
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            (model(inputs) - target).square().mean().backward()
            (reference(inputs) - target).square().mean().backward()
            for left, right in zip(model.parameters(), reference.parameters()):
                self.assertTrue(left.requires_grad)
                torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
            optimizer.step()
            reference_optimizer.step()
            for left, right in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_nested_context_preserves_outer_grad_and_inference_states(self):
        for outer in ("enabled", "no_grad", "inference"):
            for selected in (False, True):
                with self.subTest(outer=outer, selected=selected):
                    outer_context = {
                        "enabled": torch.enable_grad,
                        "no_grad": torch.no_grad,
                        "inference": torch.inference_mode,
                    }[outer]()
                    with outer_context:
                        before = (
                            torch.is_grad_enabled(),
                            torch.is_inference_mode_enabled(),
                        )
                        DynamicGradMode.set_inference_mode(selected)
                        with DynamicGradMode():
                            self.assertFalse(torch.is_grad_enabled())
                            self.assertEqual(
                                torch.is_inference_mode_enabled(), selected or before[1]
                            )
                        self.assertEqual(
                            (
                                torch.is_grad_enabled(),
                                torch.is_inference_mode_enabled(),
                            ),
                            before,
                        )
                    self.assert_caller_restored()

    def test_false_mode_keeps_existing_no_grad_fallback(self):
        DynamicGradMode.set_inference_mode(False)
        with DynamicGradMode(False):
            self.assert_policy(False)
        self.assert_caller_restored()

    def test_invalid_policy_value_retains_existing_selection(self):
        DynamicGradMode.set_inference_mode(True)
        with self.assertLogs(common.logger, level="WARNING"):
            DynamicGradMode.set_inference_mode("false")
        with DynamicGradMode():
            self.assert_policy(True)
        self.assert_caller_restored()


if __name__ == "__main__":
    unittest.main()
