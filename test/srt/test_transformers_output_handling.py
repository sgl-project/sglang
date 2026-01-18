# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for transformers model output handling.

This tests the fix for issue #15339 where custom model outputs like
GuardLogitsOutputWithPast don't have to_tuple() method.
"""

import unittest
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class MockModelOutputWithLastHiddenState:
    """Mock output with last_hidden_state attribute."""

    last_hidden_state: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None


@dataclass
class MockModelOutputWithHiddenStates:
    """Mock output with hidden_states attribute only."""

    hidden_states: Tuple[torch.Tensor, ...]
    last_hidden_state: Optional[torch.Tensor] = None


class MockModelOutputTupleAccess:
    """Mock output that can be accessed like a tuple."""

    def __init__(self, hidden_states: torch.Tensor):
        self._hidden_states = hidden_states

    def __getitem__(self, idx):
        if idx == 0:
            return self._hidden_states
        raise IndexError("Index out of range")


class TestTransformersOutputHandling(unittest.TestCase):
    """Test the logic for extracting hidden states from various output formats."""

    def _extract_hidden_states(self, output):
        """Simulate the logic in TransformersForCausalLM.forward()."""
        if (
            hasattr(output, "last_hidden_state")
            and output.last_hidden_state is not None
        ):
            return output.last_hidden_state[0, ...]
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
            return output.hidden_states[-1][0, ...]
        else:
            # Fallback: try to access output as a tuple/list
            return output[0][0, ...]

    def test_output_with_last_hidden_state(self):
        """Test extraction from output with last_hidden_state attribute."""
        hidden = torch.randn(1, 10, 768)  # batch=1, seq=10, hidden=768
        output = MockModelOutputWithLastHiddenState(last_hidden_state=hidden)

        result = self._extract_hidden_states(output)

        self.assertEqual(result.shape, (10, 768))
        torch.testing.assert_close(result, hidden[0, ...])

    def test_output_with_hidden_states(self):
        """Test extraction from output with hidden_states tuple."""
        hidden_states = tuple(torch.randn(1, 10, 768) for _ in range(12))  # 12 layers
        output = MockModelOutputWithHiddenStates(hidden_states=hidden_states)

        result = self._extract_hidden_states(output)

        self.assertEqual(result.shape, (10, 768))
        # Should extract from the last hidden state in the tuple
        torch.testing.assert_close(result, hidden_states[-1][0, ...])

    def test_output_tuple_access(self):
        """Test extraction from output that supports tuple-like access."""
        hidden = torch.randn(1, 10, 768)
        output = MockModelOutputTupleAccess(hidden)

        result = self._extract_hidden_states(output)

        self.assertEqual(result.shape, (10, 768))
        torch.testing.assert_close(result, hidden[0, ...])

    def test_output_priority_last_hidden_state(self):
        """Test that last_hidden_state takes priority over hidden_states."""
        last_hidden = torch.randn(1, 10, 768)
        hidden_states = tuple(torch.randn(1, 10, 768) for _ in range(12))

        # Output with both attributes set
        output = MockModelOutputWithLastHiddenState(
            last_hidden_state=last_hidden, hidden_states=hidden_states
        )

        result = self._extract_hidden_states(output)

        # Should use last_hidden_state, not hidden_states[-1]
        torch.testing.assert_close(result, last_hidden[0, ...])


if __name__ == "__main__":
    unittest.main()
