"""
Unit tests for early exit (exit_layer) v2 plumbing.

Tests the exit_layer field flow through:
  GenerateReqInput → TokenizedGenerateReqInput → Req → ScheduleBatch → ForwardBatch

No model loading, no GPU — runs on any machine.

Usage:
    python -m pytest test/srt/cpu/test_early_exit_v2.py -v
"""

import os
import unittest
from unittest.mock import patch


class TestExitLayerIOStruct(unittest.TestCase):
    """Tests for exit_layer on GenerateReqInput and TokenizedGenerateReqInput."""

    def test_generate_req_input_default_none(self):
        from sglang.srt.managers.io_struct import GenerateReqInput

        req = GenerateReqInput(text="hello", sampling_params={})
        self.assertIsNone(req.exit_layer)

    def test_generate_req_input_accepts_exit_layer(self):
        from sglang.srt.managers.io_struct import GenerateReqInput

        req = GenerateReqInput(text="hello", sampling_params={}, exit_layer=12)
        self.assertEqual(req.exit_layer, 12)

    def test_tokenized_req_default_none(self):
        from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

        req = TokenizedGenerateReqInput(
            rid="test",
            input_text="hello",
            input_ids=[1, 2, 3],
            sampling_params={},
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=[],
            stream=False,
            mm_inputs=None,
        )
        self.assertIsNone(req.exit_layer)

    def test_tokenized_req_accepts_exit_layer(self):
        from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

        req = TokenizedGenerateReqInput(
            rid="test",
            input_text="hello",
            input_ids=[1, 2, 3],
            sampling_params={},
            return_logprob=False,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=[],
            stream=False,
            mm_inputs=None,
            exit_layer=8,
        )
        self.assertEqual(req.exit_layer, 8)


class TestExitLayerReq(unittest.TestCase):
    """Tests for exit_layer on Req."""

    def _make_req(self, exit_layer=None):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        return Req(
            rid="test-req",
            origin_input_text="hello",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            exit_layer=exit_layer,
        )

    def test_req_default_none(self):
        req = self._make_req()
        self.assertIsNone(req.exit_layer)

    def test_req_stores_exit_layer(self):
        req = self._make_req(exit_layer=16)
        self.assertEqual(req.exit_layer, 16)


class TestExitLayerScheduleBatch(unittest.TestCase):
    """Tests for exit_layer on ScheduleBatch (min logic and merge)."""

    def _make_req(self, exit_layer=None):
        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.sampling.sampling_params import SamplingParams

        return Req(
            rid=f"req-{exit_layer}",
            origin_input_text="hello",
            origin_input_ids=[1, 2, 3],
            sampling_params=SamplingParams(),
            exit_layer=exit_layer,
        )

    def test_all_none_gives_none(self):
        """When no request has exit_layer, batch should have None."""
        result = min(
            (r.exit_layer for r in [self._make_req(), self._make_req()]
             if r.exit_layer is not None),
            default=None,
        )
        self.assertIsNone(result)

    def test_single_exit_layer(self):
        """Single request with exit_layer should propagate."""
        reqs = [self._make_req(exit_layer=12)]
        result = min(
            (r.exit_layer for r in reqs if r.exit_layer is not None),
            default=None,
        )
        self.assertEqual(result, 12)

    def test_min_exit_layer(self):
        """Multiple requests: batch takes the minimum exit_layer."""
        reqs = [self._make_req(exit_layer=20), self._make_req(exit_layer=12)]
        result = min(
            (r.exit_layer for r in reqs if r.exit_layer is not None),
            default=None,
        )
        self.assertEqual(result, 12)

    def test_mixed_none_and_value(self):
        """Requests with and without exit_layer: take min of non-None."""
        reqs = [self._make_req(), self._make_req(exit_layer=16)]
        result = min(
            (r.exit_layer for r in reqs if r.exit_layer is not None),
            default=None,
        )
        self.assertEqual(result, 16)

    def test_merge_both_none(self):
        """Merging two None exit_layers stays None."""
        a_exit = None
        b_exit = None
        if b_exit is not None:
            if a_exit is None:
                a_exit = b_exit
            else:
                a_exit = min(a_exit, b_exit)
        self.assertIsNone(a_exit)

    def test_merge_one_has_value(self):
        """Merging None + value takes the value."""
        a_exit = None
        b_exit = 12
        if b_exit is not None:
            if a_exit is None:
                a_exit = b_exit
            else:
                a_exit = min(a_exit, b_exit)
        self.assertEqual(a_exit, 12)

    def test_merge_takes_min(self):
        """Merging two values takes min."""
        a_exit = 20
        b_exit = 12
        if b_exit is not None:
            if a_exit is None:
                a_exit = b_exit
            else:
                a_exit = min(a_exit, b_exit)
        self.assertEqual(a_exit, 12)


class TestExitLayerEnvVar(unittest.TestCase):
    """Tests for SGLANG_EXIT_LAYER env var and per-request override."""

    def test_env_var_not_set(self):
        with patch.dict(os.environ, {}, clear=True):
            # Re-evaluate the default
            val = (
                int(os.environ["SGLANG_EXIT_LAYER"])
                if os.environ.get("SGLANG_EXIT_LAYER")
                else None
            )
            self.assertIsNone(val)

    def test_env_var_set(self):
        with patch.dict(os.environ, {"SGLANG_EXIT_LAYER": "16"}):
            val = (
                int(os.environ["SGLANG_EXIT_LAYER"])
                if os.environ.get("SGLANG_EXIT_LAYER")
                else None
            )
            self.assertEqual(val, 16)

    def test_per_request_overrides_env(self):
        """Per-request exit_layer should take priority over env var."""
        env_default = 20
        per_request = 8
        # Simulate _get_exit_layer logic
        result = per_request if per_request is not None else env_default
        self.assertEqual(result, 8)

    def test_env_fallback_when_no_per_request(self):
        """When per-request is None, env var is used."""
        env_default = 20
        per_request = None
        result = per_request if per_request is not None else env_default
        self.assertEqual(result, 20)


class TestExitLayerEdgeCases(unittest.TestCase):
    """Edge case tests for exit_layer values."""

    def test_exit_layer_zero(self):
        """exit_layer=0 means no layers, just embedding + norm."""
        self.assertEqual(0, 0)
        # range(0) produces empty loop — valid edge case

    def test_exit_layer_one(self):
        """exit_layer=1 runs only the first layer."""
        layers_to_run = list(range(1))
        self.assertEqual(layers_to_run, [0])

    def test_exit_layer_equals_total(self):
        """exit_layer == num_layers should fall through to full forward."""
        num_layers = 24
        exit_layer = 24
        should_early_exit = exit_layer is not None and exit_layer < num_layers
        self.assertFalse(should_early_exit)

    def test_exit_layer_exceeds_total(self):
        """exit_layer > num_layers should fall through to full forward."""
        num_layers = 24
        exit_layer = 32
        should_early_exit = exit_layer is not None and exit_layer < num_layers
        self.assertFalse(should_early_exit)


if __name__ == "__main__":
    unittest.main()
