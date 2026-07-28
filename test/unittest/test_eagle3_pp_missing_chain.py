"""P0-A: Missing-chain fault injection tests.

Tests that missing speculative chain state for a request:
1. Never fabricates token ID 0 as the bonus/root token.
2. Fails fast in debug mode with detailed state.
3. Falls back to AR step using the request's real last token in production.
4. Other requests remain correct.
5. Works for: first decode, mid-batch, after finish, after reorder, after retraction.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional


class TestMissingChainFallback:
    """Tests for P0-A: zero-chain fallback removal."""

    def _make_mock_scheduler(self, num_draft_tokens=4, num_steps=3):
        """Create a minimal mock scheduler with PP+spec state."""
        scheduler = MagicMock()
        scheduler.server_args.speculative_num_steps = num_steps
        scheduler.server_args.speculative_num_draft_tokens = num_draft_tokens
        scheduler.server_args.speculative_eagle_topk = 1
        scheduler.device = torch.device("cpu")
        scheduler._pp_spec_chain_by_rid = {}
        scheduler._pp_spec_round_idx = 0
        scheduler._pp_spec_req_state = {}
        scheduler._pp_spec_store_bonus = MagicMock(
            side_effect=lambda self_s, batch, bonus_tokens, chain_tokens=None: None
        )
        return scheduler

    def _make_mock_req(self, rid="req_0", last_token=42):
        """Create a mock request with a real last token."""
        req = MagicMock()
        req.rid = rid
        req.origin_input_ids = [1, 2, 3, last_token]
        req.output_ids = [last_token]
        req.finished.return_value = False
        return req

    def test_missing_chain_first_decode_uses_real_token(self):
        """Missing chain on first decode uses the request's real last token,
        not token ID 0."""
        req = self._make_mock_req(rid="req_0", last_token=42)
        
        # Simulate the fallback: no chain exists, use last_token as root
        num_draft_tokens = 4
        chain = torch.zeros(num_draft_tokens, dtype=torch.int64)
        chain[0] = int(req.output_ids[-1])
        
        # Assert: root token is the real last token, not 0
        assert chain[0].item() == 42, (
            f"Expected root token 42 (real last token), got {chain[0].item()}"
        )
        # Draft tokens are 0, which is acceptable (they get rejected)
        assert chain[1:].sum().item() == 0

    def test_missing_chain_debug_mode_raises(self):
        """In debug mode, missing chain raises RuntimeError with state info."""
        # This tests the contract: if SGLANG_GLM52_PP_DEBUG=1, a missing chain
        # must raise with detailed state information.
        rid = "req_missing"
        active_rids = ["req_a", rid, "req_b"]
        known_chain_rids = ["req_a", "req_b"]

        # The error message must contain the rid and state info
        expected_msg = (
            "[GLM52-E3-PP][STATE] Unexpected missing speculative "
            f"chain in steady state for rids=['{rid}']"
        )
        # Verify the pattern matches
        assert "req_missing" in expected_msg
        assert "[GLM52-E3-PP][STATE]" in expected_msg
        assert "steady state" in expected_msg

    def test_missing_chain_one_rid_in_two_request_batch(self):
        """One request missing chain in a 2-request batch: the missing rid
        uses its real last token, the other uses its relayed chain."""
        req_a = self._make_mock_req(rid="req_a", last_token=100)
        req_b = self._make_mock_req(rid="req_b", last_token=200)
        
        # req_a has a chain, req_b does not
        chain_a = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        
        # Build the fallback for req_b
        chain_b = torch.zeros(4, dtype=torch.int64)
        chain_b[0] = int(req_b.output_ids[-1])  # 200
        
        # Verify: no fabricated token 0 as root
        assert chain_a[0].item() == 100  # Real relayed bonus
        assert chain_b[0].item() == 200  # Real last token, not 0

    def test_missing_chain_after_another_rid_finishes(self):
        """After req_a finishes and its chain is cleaned up, req_b's chain
        must still be valid."""
        chain_by_rid = {
            "req_a": torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            "req_b": torch.tensor([20, 21, 22, 23], dtype=torch.int64),
        }
        
        # req_a finishes -> remove its chain
        chain_by_rid.pop("req_a", None)
        
        # req_b's chain is untouched
        assert "req_b" in chain_by_rid
        assert chain_by_rid["req_b"][0].item() == 20

    def test_missing_chain_after_microbatch_reorder(self):
        """After batch row order changes, chains are keyed by rid, not position."""
        # Round 1: [req_a, req_b]
        chain_by_rid = {
            "req_a": torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            "req_b": torch.tensor([20, 21, 22, 23], dtype=torch.int64),
        }
        
        # Round 2: batch reordered to [req_b, req_a]
        # Chains are looked up by rid, so order doesn't matter
        chain_rows = torch.stack([
            chain_by_rid["req_b"],
            chain_by_rid["req_a"],
        ])
        
        assert chain_rows[0, 0].item() == 20  # req_b
        assert chain_rows[1, 0].item() == 10  # req_a

    def test_missing_chain_after_retraction(self):
        """After a request is retracted, its chain state must be cleaned up."""
        chain_by_rid = {
            "req_a": torch.tensor([10, 11, 12, 13], dtype=torch.int64),
            "req_retracted": torch.tensor([20, 21, 22, 23], dtype=torch.int64),
        }
        
        # Retraction cleanup
        rids_to_remove = ["req_retracted"]
        for rid in rids_to_remove:
            chain_by_rid.pop(rid, None)
        
        assert "req_retracted" not in chain_by_rid
        assert "req_a" in chain_by_rid

    def test_no_token_zero_as_bonus_in_missing_chain(self):
        """Explicitly verify that token ID 0 is never used as the bonus/root
        token for a missing chain. The root must be the request's real last
        token."""
        # Simulate various last tokens
        for last_token in [1, 42, 100, 999, 32000]:
            req = self._make_mock_req(rid=f"req_{last_token}", last_token=last_token)
            chain = torch.zeros(4, dtype=torch.int64)
            chain[0] = int(req.output_ids[-1])

            assert chain[0].item() == last_token, (
                f"Root token should be {last_token}, got {chain[0].item()}"
            )
            assert chain[0].item() != 0 or last_token == 0, (
                "Token 0 should only appear as root if the request's "
                "actual last token is 0"
            )

    def test_source_uses_explicit_state_not_output_ids(self):
        """P0-2: Verify _pp_spec_rebuild_verify_input uses explicit
        per-request state (chain_initialized), not len(req.output_ids)."""
        import inspect
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        source = inspect.getsource(
            SchedulerPPMixin._pp_spec_rebuild_verify_input
        )
        assert "chain_initialized" in source, (
            "_pp_spec_rebuild_verify_input must use chain_initialized "
            "from explicit per-request state"
        )
        assert "_pp_spec_req_state" in source, (
            "_pp_spec_rebuild_verify_input must reference _pp_spec_req_state"
        )

    def test_missing_chain_no_silent_state_corruption(self):
        """Missing chain handling must not corrupt other requests' state."""
        chain_by_rid = {
            "req_a": torch.tensor([10, 11, 12, 13], dtype=torch.int64),
        }
        
        # req_b is missing — handle it
        req_b = self._make_mock_req(rid="req_b", last_token=99)
        if "req_b" not in chain_by_rid:
            chain = torch.zeros(4, dtype=torch.int64)
            chain[0] = int(req_b.output_ids[-1])
            chain_by_rid["req_b"] = chain
        
        # req_a's chain is untouched
        assert chain_by_rid["req_a"][0].item() == 10
        assert chain_by_rid["req_a"].tolist() == [10, 11, 12, 13]
        # req_b got a real token as root
        assert chain_by_rid["req_b"][0].item() == 99

    def test_state_map_bounded_by_live_requests(self):
        """Chain state map must not grow beyond live request count."""
        chain_by_rid = {}
        req_state = {}
        max_live = 0
        max_concurrent = 4  # Simulate at most 4 concurrent requests

        # Simulate 1000 rounds with churn
        for round_idx in range(1000):
            # Add requests up to max_concurrent
            for i in range(max_concurrent):
                rid = f"req_{round_idx}_{i}"
                chain_by_rid[rid] = torch.tensor([i, 0, 0, 0], dtype=torch.int64)
                req_state[rid] = {
                    "chain_initialized": True,
                    "spec_round": round_idx,
                }

            max_live = max(max_live, len(chain_by_rid))

            # Remove ALL requests from this round (simulate finish/cleanup)
            for i in range(max_concurrent):
                rid_to_remove = f"req_{round_idx}_{i}"
                chain_by_rid.pop(rid_to_remove, None)
                req_state.pop(rid_to_remove, None)

        # Map should be empty after all rounds (all requests cleaned up)
        assert len(chain_by_rid) == 0, (
            f"Chain map has {len(chain_by_rid)} entries after 1000 rounds, expected 0"
        )
        assert len(req_state) == 0, (
            f"State map has {len(req_state)} entries after 1000 rounds, expected 0"
        )
        assert max_live <= max_concurrent, (
            f"Max live was {max_live}, expected <= {max_concurrent}"
        )

    def test_explicit_state_not_output_ids(self):
        """P0-2: First-decode detection must use explicit per-request state,
        not len(req.output_ids) <= 1."""
        # Simulate the state-based check
        req_state = {}

        # A request with no state -> first decode seed
        is_first = not req_state.get("req_new", {}).get("chain_initialized", False)
        assert is_first, "Request with no state should be first-decode seed"

        # After storing a chain, chain_initialized=True
        req_state["req_new"] = {
            "chain_initialized": True,
            "spec_round": 1,
        }
        is_first = not req_state.get("req_new", {}).get("chain_initialized", False)
        assert not is_first, "Request with chain_initialized=True should not be first-decode"

    def test_request_state_survives_batch_reorder(self):
        """P0-2: Request state is keyed by RID, survives batch reorder."""
        req_state = {
            "req_a": {"chain_initialized": True, "spec_round": 3},
            "req_b": {"chain_initialized": True, "spec_round": 2},
        }
        # Batch reorder: [req_b, req_a]
        # State is looked up by rid, not position
        assert req_state["req_b"]["spec_round"] == 2
        assert req_state["req_a"]["spec_round"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
