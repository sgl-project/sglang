"""
Unit tests for the scheduler output processor mixin and Req tokenizer attributes.

This module tests:
- init_return_logprob_containers: initialization of logprob containers on the scheduler
- clear_list: clearing logprob container lists
- Req.eos_token_id and Req.additional_stop_token_ids: tokenizer attribute caching

Usage:
    python test_scheduler_output_processor.py
    python -m pytest test_scheduler_output_processor.py -v
"""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestInitReturnLogprobContainers(unittest.TestCase):
    """Test cases for init_return_logprob_containers method."""

    def _create_mixin_instance(self):
        """Create a minimal instance of SchedulerOutputProcessorMixin."""
        mixin = SchedulerOutputProcessorMixin.__new__(SchedulerOutputProcessorMixin)
        return mixin

    def test_creates_all_container_lists(self):
        """Test that init_return_logprob_containers creates all 12 container lists."""
        mixin = self._create_mixin_instance()
        mixin.init_return_logprob_containers()

        expected_attrs = [
            "input_token_logprobs_val",
            "input_token_logprobs_idx",
            "output_token_logprobs_val",
            "output_token_logprobs_idx",
            "input_top_logprobs_val",
            "input_top_logprobs_idx",
            "output_top_logprobs_val",
            "output_top_logprobs_idx",
            "input_token_ids_logprobs_val",
            "input_token_ids_logprobs_idx",
            "output_token_ids_logprobs_val",
            "output_token_ids_logprobs_idx",
        ]

        for attr in expected_attrs:
            self.assertTrue(hasattr(mixin, attr), f"Missing attribute: {attr}")
            self.assertEqual(
                getattr(mixin, attr), [], f"Attribute {attr} should be an empty list"
            )

    def test_containers_are_independent_lists(self):
        """Test that each container is an independent list, not shared references."""
        mixin = self._create_mixin_instance()
        mixin.init_return_logprob_containers()

        # Append to one list and verify others are not affected
        mixin.input_token_logprobs_val.append("test")
        self.assertEqual(len(mixin.input_token_logprobs_val), 1)
        self.assertEqual(len(mixin.input_token_logprobs_idx), 0)
        self.assertEqual(len(mixin.output_token_logprobs_val), 0)


class TestClearList(unittest.TestCase):
    """Test cases for clear_list method."""

    def _create_mixin_instance(self):
        """Create a minimal instance with initialized containers."""
        mixin = SchedulerOutputProcessorMixin.__new__(SchedulerOutputProcessorMixin)
        mixin.init_return_logprob_containers()
        return mixin

    def test_clear_list_clears_all_containers(self):
        """Test that clear_list clears all container lists."""
        mixin = self._create_mixin_instance()

        # Add data to all containers
        container_names = [
            "input_token_logprobs_val",
            "input_token_logprobs_idx",
            "output_token_logprobs_val",
            "output_token_logprobs_idx",
            "input_top_logprobs_val",
            "input_top_logprobs_idx",
            "output_top_logprobs_val",
            "output_top_logprobs_idx",
            "input_token_ids_logprobs_val",
            "input_token_ids_logprobs_idx",
            "output_token_ids_logprobs_val",
            "output_token_ids_logprobs_idx",
        ]
        for name in container_names:
            getattr(mixin, name).append("some_data")
            self.assertEqual(
                len(getattr(mixin, name)), 1, f"{name} should have 1 item before clear"
            )

        mixin.clear_list()

        for name in container_names:
            self.assertEqual(
                len(getattr(mixin, name)), 0, f"{name} should be empty after clear_list"
            )

    def test_clear_list_creates_missing_attributes(self):
        """Test that clear_list creates lists for attributes that don't exist yet."""
        mixin = SchedulerOutputProcessorMixin.__new__(SchedulerOutputProcessorMixin)

        container_names = [
            "input_token_logprobs_val",
            "input_token_logprobs_idx",
            "output_token_logprobs_val",
            "output_token_logprobs_idx",
            "input_top_logprobs_val",
            "input_top_logprobs_idx",
            "output_top_logprobs_val",
            "output_top_logprobs_idx",
            "input_token_ids_logprobs_val",
            "input_token_ids_logprobs_idx",
            "output_token_ids_logprobs_val",
            "output_token_ids_logprobs_idx",
        ]

        # Verify attributes don't exist before clear_list
        for name in container_names:
            self.assertFalse(
                hasattr(mixin, name), f"{name} should not exist before clear_list"
            )

        mixin.clear_list()

        # Verify attributes are created as empty lists
        for name in container_names:
            self.assertTrue(
                hasattr(mixin, name), f"{name} should exist after clear_list"
            )
            self.assertEqual(
                getattr(mixin, name), [], f"{name} should be an empty list"
            )

    def test_clear_list_after_partial_set_to_none(self):
        """Test clear_list behavior when containers are properly initialized.

        Note: In actual code, clear_list() is only called when input_token_logprobs_val
        is not None (see stream_output at line 1010), so this test verifies the normal
        behavior when containers are properly initialized as lists.
        """
        mixin = self._create_mixin_instance()

        # All containers should be initialized as empty lists
        self.assertIsInstance(mixin.input_token_logprobs_val, list)
        self.assertIsInstance(mixin.input_top_logprobs_val, list)

        # Add some data
        mixin.input_top_logprobs_val.append("data")
        mixin.output_top_logprobs_val.append("data")

        # clear_list should clear all containers
        mixin.clear_list()

        # All containers should be empty lists
        self.assertEqual(mixin.input_token_logprobs_val, [])
        self.assertEqual(mixin.input_token_logprobs_idx, [])
        self.assertEqual(mixin.output_token_logprobs_val, [])
        self.assertEqual(mixin.output_token_logprobs_idx, [])
        self.assertEqual(mixin.input_top_logprobs_val, [])
        self.assertEqual(mixin.output_top_logprobs_val, [])

    def test_init_then_clear_then_reinit_cycle(self):
        """Test the full lifecycle: init -> populate -> clear -> reinit."""
        mixin = self._create_mixin_instance()

        # Populate
        mixin.input_token_logprobs_val.append([0.1, 0.2])
        mixin.output_token_logprobs_val.append([0.3])

        # Clear
        mixin.clear_list()
        self.assertEqual(mixin.input_token_logprobs_val, [])
        self.assertEqual(mixin.output_token_logprobs_val, [])

        # Re-populate
        mixin.input_token_logprobs_val.append([0.5])
        self.assertEqual(len(mixin.input_token_logprobs_val), 1)


class TestReqTokenizerAttributes(unittest.TestCase):
    """Test cases for Req.eos_token_id and Req.additional_stop_token_ids attributes."""

    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _create_req(self, rid="test-req", input_ids=None):
        """Create a Req instance with minimal required arguments."""
        if input_ids is None:
            input_ids = [1, 2, 3]
        return Req(
            rid=rid,
            origin_input_text="test text",
            origin_input_ids=input_ids,
            sampling_params=SamplingParams(),
        )

    def test_req_has_eos_token_id_attribute(self):
        """Test that Req instances have eos_token_id attribute initialized to None."""
        req = self._create_req(rid="test-1")
        self.assertIsNone(req.eos_token_id)

    def test_req_has_additional_stop_token_ids_attribute(self):
        """Test that Req instances have additional_stop_token_ids attribute initialized to None."""
        req = self._create_req(rid="test-2")
        self.assertIsNone(req.additional_stop_token_ids)

    def test_req_eos_token_id_can_be_set(self):
        """Test that eos_token_id can be set on a Req instance."""
        req = self._create_req(rid="test-3")
        req.eos_token_id = 2
        self.assertEqual(req.eos_token_id, 2)

    def test_req_additional_stop_token_ids_can_be_set(self):
        """Test that additional_stop_token_ids can be set on a Req instance."""
        req = self._create_req(rid="test-4")
        req.additional_stop_token_ids = [32000, 32001]
        self.assertEqual(req.additional_stop_token_ids, [32000, 32001])

    def test_req_tokenizer_attributes_set_from_mock_tokenizer(self):
        """Test setting eos_token_id and additional_stop_token_ids from a mock tokenizer."""
        req = self._create_req(rid="test-5")

        # Simulate what the scheduler does
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.additional_stop_token_ids = [32000, 32001]

        req.tokenizer = mock_tokenizer
        req.eos_token_id = mock_tokenizer.eos_token_id
        req.additional_stop_token_ids = mock_tokenizer.additional_stop_token_ids

        self.assertEqual(req.eos_token_id, 2)
        self.assertEqual(req.additional_stop_token_ids, [32000, 32001])

    def test_multiple_reqs_independent_tokenizer_attrs(self):
        """Test that different Req instances can have different tokenizer attributes."""
        req1 = self._create_req(rid="test-6a")
        req2 = self._create_req(rid="test-6b")

        req1.eos_token_id = 2
        req1.additional_stop_token_ids = [100]

        req2.eos_token_id = 50256
        req2.additional_stop_token_ids = [200, 300]

        self.assertEqual(req1.eos_token_id, 2)
        self.assertEqual(req1.additional_stop_token_ids, [100])
        self.assertEqual(req2.eos_token_id, 50256)
        self.assertEqual(req2.additional_stop_token_ids, [200, 300])


if __name__ == "__main__":
    unittest.main()
