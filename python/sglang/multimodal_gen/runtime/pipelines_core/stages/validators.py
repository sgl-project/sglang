# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Common validators for pipeline stage verification.

This module provides reusable validation functions that can be used across
all pipeline stages for input/output verification.
"""

from collections.abc import Callable
from typing import Any

import torch


class StageValidators:
    """Common validators for pipeline stages."""

    @staticmethod
    def not_none(value: Any) -> bool:
        """Check if value is not None."""
        return value is not None

    @staticmethod
    def positive_int(value: Any) -> bool:
        """Check if value is a positive integer."""
        return isinstance(value, int) and value > 0

    @staticmethod
    def non_negative_int(value: Any) -> bool:
        """Check if value is a non-negative float."""
        return isinstance(value, int | float) and value >= 0

    @staticmethod
    def positive_float(value: Any) -> bool:
        """Check if value is a positive float."""
        return isinstance(value, int | float) and value > 0

    @staticmethod
    def non_negative_float(value: Any) -> bool:
        """Check if value is a non-negative float."""
        return isinstance(value, int | float) and value >= 0

    @staticmethod
    def divisible_by(value: Any, divisor: int) -> bool:
        """Check if value is divisible by divisor."""
        return value is not None and isinstance(value, int) and value % divisor == 0

    @staticmethod
    def is_tensor(value: Any) -> bool:
        """Check if value is a torch tensor and doesn't contain NaN values."""
        if not isinstance(value, torch.Tensor):
            return False
        return not torch.isnan(value).any().item()

    @staticmethod
    def tensor_with_dims(value: Any, dims: int) -> bool:
        """Check if value is a tensor with specific dimensions and no NaN values."""
        if not isinstance(value, torch.Tensor):
            return False
        if value.dim() != dims:
            return False
        return not torch.isnan(value).any().item()

    @staticmethod
    def tensor_min_dims(value: Any, min_dims: int) -> bool:
        """Check if value is a tensor with at least min_dims dimensions and no NaN values."""
        if not isinstance(value, torch.Tensor):
            return False
        if value.dim() < min_dims:
            return False
        return not torch.isnan(value).any().item()

    @staticmethod
    def tensor_shape_matches(value: Any, expected_shape: tuple) -> bool:
        """Check if tensor shape matches expected shape (None for any size) and no NaN values."""
        if not isinstance(value, torch.Tensor):
            return False
        if len(value.shape) != len(expected_shape):
            return False
        for actual, expected in zip(value.shape, expected_shape, strict=True):
            if expected is not None and actual != expected:
                return False
        return not torch.isnan(value).any().item()

    @staticmethod
    def list_not_empty(value: Any) -> bool:
        """Check if value is a non-empty list."""
        return isinstance(value, list) and len(value) > 0

    @staticmethod
    def list_length(value: Any, length: int) -> bool:
        """Check if list has specific length."""
        return isinstance(value, list) and len(value) == length

    @staticmethod
    def list_min_length(value: Any, min_length: int) -> bool:
        """Check if list has at least min_length items."""
        return isinstance(value, list) and len(value) >= min_length

    @staticmethod
    def string_not_empty(value: Any) -> bool:
        """Check if value is a non-empty string."""
        return isinstance(value, str) and len(value.strip()) > 0

    @staticmethod
    def string_not_none(value: Any) -> bool:
        """Check if value is a non-empty string."""
        return isinstance(value, str) and len(value) > 0

    @staticmethod
    def string_or_list_strings(value: Any) -> bool:
        """Check if value is a string or list of strings."""
        if isinstance(value, str):
            return True
        if isinstance(value, list):
            return all(isinstance(item, str) for item in value)
        return False

    @staticmethod
    def bool_value(value: Any) -> bool:
        """Check if value is a boolean."""
        return isinstance(value, bool)

    @staticmethod
    def generator_or_list_generators(value: Any) -> bool:
        """Check if value is a Generator or list of Generators."""
        if isinstance(value, torch.Generator):
            return True
        if isinstance(value, list):
            return all(isinstance(item, torch.Generator) for item in value)
        return False

    @staticmethod
    def is_list(value: Any) -> bool:
        """Check if value is a list (can be empty)."""
        return isinstance(value, list)

    @staticmethod
    def is_tuple(value: Any) -> bool:
        """Check if value is a tuple."""
        return isinstance(value, tuple)

    @staticmethod
    def none_or_tensor(value: Any) -> bool:
        """Check if value is None or a tensor without NaN values."""
        if value is None:
            return True
        if not isinstance(value, torch.Tensor):
            return False
        return not torch.isnan(value).any().item()

    @staticmethod
    def list_of_tensors_with_dims(value: Any, dims: int) -> bool:
        """Check if value is a non-empty list where all items are tensors with specific dimensions and no NaN values."""
        if not isinstance(value, list) or len(value) == 0:
            return False
        for item in value:
            if not isinstance(item, torch.Tensor):
                return False
            if item.dim() != dims:
                return False
            if torch.isnan(item).any().item():
                return False
        return True

    @staticmethod
    def list_of_tensors(value: Any) -> bool:
        """Check if value is a non-empty list where all items are tensors without NaN values."""
        if not isinstance(value, list) or len(value) == 0:
            return False
        for item in value:
            if not isinstance(item, torch.Tensor):
                return False
            if torch.isnan(item).any().item():
                return False
        return True

    @staticmethod
    def list_of_tensors_with_min_dims(value: Any, min_dims: int) -> bool:
        """Check if value is a non-empty list where all items are tensors with at least min_dims dimensions and no NaN values."""
        if not isinstance(value, list) or len(value) == 0:
            return False
        for item in value:
            if not isinstance(item, torch.Tensor):
                return False
            if item.dim() < min_dims:
                return False
            if torch.isnan(item).any().item():
                return False
        return True

    @staticmethod
    def none_or_tensor_with_dims(dims: int) -> Callable[[Any], bool]:
        """Return a validator that checks if value is None or a tensor with specific dimensions and no NaN values."""

        def validator(value: Any) -> bool:
            if value is None:
                return True
            if not isinstance(value, torch.Tensor):
                return False
            if value.dim() != dims:
                return False
            return not torch.isnan(value).any().item()

        return validator

    @staticmethod
    def none_or_list(value: Any) -> bool:
        """Check if value is None or a list."""
        return value is None or isinstance(value, list)

    @staticmethod
    def none_or_positive_int(value: Any) -> bool:
        """Check if value is None or a positive integer."""
        return value is None or (isinstance(value, int) and value > 0)

    # Helper methods that return functions for common patterns
    @staticmethod
    def with_dims(dims: int) -> Callable[[Any], bool]:
        """Return a validator that checks if tensor has specific dimensions and no NaN values."""

        def validator(value: Any) -> bool:
            return StageValidators.tensor_with_dims(value, dims)

        return validator

    @staticmethod
    def min_dims(min_dims: int) -> Callable[[Any], bool]:
        """Return a validator that checks if tensor has at least min_dims dimensions and no NaN values."""

        def validator(value: Any) -> bool:
            return StageValidators.tensor_min_dims(value, min_dims)

        return validator

    @staticmethod
    def divisible(divisor: int) -> Callable[[Any], bool]:
        """Return a validator that checks if value is divisible by divisor."""

        def validator(value: Any) -> bool:
            return StageValidators.divisible_by(value, divisor)

        return validator

    @staticmethod
    def positive_int_divisible(divisor: int) -> Callable[[Any], bool]:
        """Return a validator that checks if value is a positive integer divisible by divisor."""

        def validator(value: Any) -> bool:
            return (
                isinstance(value, int)
                and value > 0
                and StageValidators.divisible_by(value, divisor)
            )

        return validator

    @staticmethod
    def list_of_tensors_dims(dims: int) -> Callable[[Any], bool]:
        """Return a validator that checks if value is a list of tensors with specific dimensions and no NaN values."""

        def validator(value: Any) -> bool:
            return StageValidators.list_of_tensors_with_dims(value, dims)

        return validator

    @staticmethod
    def list_of_tensors_min_dims(min_dims: int) -> Callable[[Any], bool]:
        """Return a validator that checks if value is a list of tensors with at least min_dims dimensions and no NaN values."""

        def validator(value: Any) -> bool:
            return StageValidators.list_of_tensors_with_min_dims(value, min_dims)

        return validator


class ValidationFailure:
    """Details about a specific validation failure."""

    def __init__(
        self,
        validator_name: str,
        actual_value: Any,
        expected: str | None = None,
        error_msg: str | None = None,
    ):
        self.validator_name = validator_name
        self.actual_value = actual_value
        self.expected = expected
        self.error_msg = error_msg

    def __str__(self) -> str:
        parts = [f"Validator '{self.validator_name}' failed"]

        if self.error_msg:
            parts.append(f"Error: {self.error_msg}")

        # Add actual value info (but limit very long representations)
        actual_str = self._format_value(self.actual_value)
        parts.append(f"Actual: {actual_str}")

        if self.expected:
            parts.append(f"Expected: {self.expected}")

        return ". ".join(parts)

    def _format_value(self, value: Any) -> str:
        """Format a value for display in error messages."""
        if value is None:
            return "None"
        elif isinstance(value, torch.Tensor):
            return f"tensor(shape={list(value.shape)}, dtype={value.dtype})"
        elif isinstance(value, list):
            if len(value) == 0:
                return "[]"
            elif len(value) <= 3:
                item_strs = [self._format_value(item) for item in value]
                return f"[{', '.join(item_strs)}]"
            else:
                return f"list(length={len(value)}, first_item={self._format_value(value[0])})"
        elif isinstance(value, str):
            if len(value) > 50:
                return f"'{value[:47]}...'"
            else:
                return f"'{value}'"
        else:
            return f"{type(value).__name__}({value})"


class VerificationResult:
    """Wrapper class for stage verification results."""

    def __init__(self) -> None:
        self._checks: dict[str, bool] = {}
        self._failures: dict[str, list[ValidationFailure]] = {}

    def add_check(
        self,
        field_name: str,
        value: Any,
        validators: Callable[[Any], bool] | list[Callable[[Any], bool]],
    ) -> "VerificationResult":
        """
        Add a validation check for a field.

        Args:
            field_name: Name of the field being checked
            value: The actual value to validate
            validators: Single validation function or list of validation functions.
                       Each function will be called with the value as its first argument.

        Returns:
            Self for method chaining

        Examples:
            # Single validator
            result.add_check("tensor", my_tensor, V.is_tensor)

            # Multiple validators (all must pass)
            result.add_check("latents", batch.latents, [V.is_tensor, V.with_dims(5)])

            # Using partial functions for parameters
            result.add_check("height", batch.height, [V.not_none, V.divisible(8)])
        """
        if not isinstance(validators, list):
            validators = [validators]

        failures = []
        all_passed = True

        # Apply all validators and collect detailed failure info
        for validator in validators:
            try:
                passed = validator(value)
                if not passed:
                    all_passed = False
                    failure = self._create_validation_failure(validator, value)
                    failures.append(failure)
            except Exception as e:
                # If any validator raises an exception, consider the check failed
                all_passed = False
                validator_name = getattr(validator, "__name__", str(validator))
                failure = ValidationFailure(
                    validator_name=validator_name,
                    actual_value=value,
                    error_msg=f"Exception during validation: {str(e)}",
                )
                failures.append(failure)

        self._checks[field_name] = all_passed
        if not all_passed:
            self._failures[field_name] = failures

        return self

    def _create_validation_failure(
        self, validator: Callable, value: Any
    ) -> ValidationFailure:
        """Create a ValidationFailure with detailed information."""
        validator_name = getattr(validator, "__name__", str(validator))

        # Try to extract meaningful expected value info based on validator type
        expected = None
        error_msg = None

        # Handle common validator patterns
        if hasattr(validator, "__closure__") and validator.__closure__:
            # This is likely a closure (like our helper functions)
            if "dims" in validator_name or "with_dims" in str(validator):
                if isinstance(value, torch.Tensor):
                    expected = f"tensor with {validator.__closure__[0].cell_contents} dimensions"
                else:
                    expected = "tensor with specific dimensions"
            elif "divisible" in str(validator):
                expected = (
                    f"integer divisible by {validator.__closure__[0].cell_contents}"
                )

        # Handle specific validator types and check for NaN values
        if validator_name == "is_tensor":
            expected = "torch.Tensor without NaN values"
            if isinstance(value, torch.Tensor) and torch.isnan(value).any().item():
                error_msg = (
                    f"tensor contains {torch.isnan(value).sum().item()} NaN values"
                )
        elif validator_name == "positive_int":
            expected = "positive integer"
        elif validator_name == "not_none":
            expected = "non-None value"
        elif validator_name == "list_not_empty":
            expected = "non-empty list"
        elif validator_name == "bool_value":
            expected = "boolean value"
        elif (
            "tensor_with_dims" in validator_name or "tensor_min_dims" in validator_name
        ):
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any().item():
                    error_msg = f"tensor has {value.dim()} dimensions but contains {torch.isnan(value).sum().item()} NaN values"
                else:
                    error_msg = f"tensor has {value.dim()} dimensions"
        elif validator_name == "is_list":
            expected = "list"
        elif validator_name == "none_or_tensor":
            expected = "None or tensor without NaN values"
            if isinstance(value, torch.Tensor) and torch.isnan(value).any().item():
                error_msg = (
                    f"tensor contains {torch.isnan(value).sum().item()} NaN values"
                )
        elif validator_name == "list_of_tensors":
            expected = "non-empty list of tensors without NaN values"
            if isinstance(value, list) and len(value) > 0:
                nan_count = 0
                for item in value:
                    if (
                        isinstance(item, torch.Tensor)
                        and torch.isnan(item).any().item()
                    ):
                        nan_count += torch.isnan(item).sum().item()
                if nan_count > 0:
                    error_msg = (
                        f"list contains tensors with total {nan_count} NaN values"
                    )
        elif "list_of_tensors_with_dims" in validator_name:
            expected = (
                "non-empty list of tensors with specific dimensions and no NaN values"
            )
            if isinstance(value, list) and len(value) > 0:
                nan_count = 0
                for item in value:
                    if (
                        isinstance(item, torch.Tensor)
                        and torch.isnan(item).any().item()
                    ):
                        nan_count += torch.isnan(item).sum().item()
                if nan_count > 0:
                    error_msg = (
                        f"list contains tensors with total {nan_count} NaN values"
                    )

        return ValidationFailure(
            validator_name=validator_name,
            actual_value=value,
            expected=expected,
            error_msg=error_msg,
        )

    def is_valid(self) -> bool:
        """Check if all validations passed."""
        return all(self._checks.values())

    def get_failed_fields(self) -> list[str]:
        """Get list of fields that failed validation."""
        return [field for field, passed in self._checks.items() if not passed]

    def get_detailed_failures(self) -> dict[str, list[ValidationFailure]]:
        """Get detailed failure information for each failed field."""
        return self._failures.copy()

    def get_failure_summary(self) -> str:
        """Get a comprehensive summary of all validation failures."""
        if self.is_valid():
            return "All validations passed"

        summary_parts = []
        for field_name, failures in self._failures.items():
            field_summary = f"\n  Field '{field_name}':"
            for i, failure in enumerate(failures, 1):
                field_summary += f"\n    {i}. {failure}"
            summary_parts.append(field_summary)

        return "Validation failures:" + "".join(summary_parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility."""
        return self._checks.copy()


# Alias for convenience
V = StageValidators
