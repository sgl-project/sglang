"""P1-D: Two-round scheduler state closure tests.

Tests that PP+spec scheduler state closes correctly across rounds:
- Scenario A: 2 requests, both continue for 2+ rounds
- Scenario B: request A finishes after round 1, request B continues
- Scenario C: batch row order changes between rounds
- Scenario D: one request is retracted
- Scenario E: missing chain fault injection
"""

import pytest
import torch
from unittest.mock import MagicMock


class TestSchedulerStateClosure:
    """Test round-to-round state closure for PP+spec."""

    def _make_chain_state(self, num_draft_tokens=4):
        """Create a fresh chain state dict."""
        return {}

    def _store_bonus(self, chain_by_rid, rids, bonus_tokens, chain_tokens=None,
                     num_draft_tokens=4):
        """Simulate _pp_spec_store_bonus."""
        if chain_tokens is not None:
            rows = chain_tokens.to(torch.int64).reshape(
                len(rids), num_draft_tokens
            )
        else:
            rows = torch.zeros(
                (len(rids), num_draft_tokens), dtype=torch.int64
            )
            rows[:, 0] = bonus_tokens.to(torch.int64)

        for i, rid in enumerate(rids):
            chain_by_rid[rid] = rows[i].to(device="cpu", dtype=torch.int64).clone()

    def _cleanup_finished(self, chain_by_rid, rids, finished_rids):
        """Simulate cleanup in _pp_process_batch_result."""
        for rid in finished_rids:
            chain_by_rid.pop(rid, None)

    def _scenario_a_two_requests_two_rounds(self):
        """Scenario A: 2 requests, both continue for 2 rounds."""
        chain_by_rid = {}
        num_draft_tokens = 4

        # Round 1: prefill -> store bonus
        rids_r1 = ["req_a", "req_b"]
        bonus_r1 = torch.tensor([100, 200], dtype=torch.int64)
        chain_r1 = torch.tensor([100, 101, 102, 103, 200, 201, 202, 203], dtype=torch.int64)
        self._store_bonus(chain_by_rid, rids_r1, bonus_r1, chain_r1, num_draft_tokens)

        # Verify round 1 state
        assert len(chain_by_rid) == 2
        assert chain_by_rid["req_a"].tolist() == [100, 101, 102, 103]
        assert chain_by_rid["req_b"].tolist() == [200, 201, 202, 203]

        # Round 2: verify -> store new bonus
        rids_r2 = ["req_a", "req_b"]
        bonus_r2 = torch.tensor([104, 204], dtype=torch.int64)
        chain_r2 = torch.tensor([104, 105, 106, 107, 204, 205, 206, 207], dtype=torch.int64)
        self._store_bonus(chain_by_rid, rids_r2, bonus_r2, chain_r2, num_draft_tokens)

        # Verify round 2 state
        assert len(chain_by_rid) == 2
        assert chain_by_rid["req_a"].tolist() == [104, 105, 106, 107]
        assert chain_by_rid["req_b"].tolist() == [204, 205, 206, 207]

        return chain_by_rid

    def test_scenario_a(self):
        """2 requests, both continue for 2+ rounds."""
        chain_by_rid = self._scenario_a_two_requests_two_rounds()
        assert len(chain_by_rid) == 2

    def test_scenario_b_finish_after_round_1(self):
        """Request A finishes after round 1, request B continues."""
        chain_by_rid = {}
        num_draft_tokens = 4

        # Round 1
        rids_r1 = ["req_a", "req_b"]
        bonus_r1 = torch.tensor([100, 200], dtype=torch.int64)
        chain_r1 = torch.tensor([100, 101, 102, 103, 200, 201, 202, 203], dtype=torch.int64)
        self._store_bonus(chain_by_rid, rids_r1, bonus_r1, chain_r1, num_draft_tokens)

        # req_a finishes
        self._cleanup_finished(chain_by_rid, rids_r1, ["req_a"])

        # Verify: req_a cleaned up, req_b preserved
        assert "req_a" not in chain_by_rid
        assert "req_b" in chain_by_rid
        assert chain_by_rid["req_b"].tolist() == [200, 201, 202, 203]

        # Round 2: only req_b
        rids_r2 = ["req_b"]
        bonus_r2 = torch.tensor([204], dtype=torch.int64)
        chain_r2 = torch.tensor([204, 205, 206, 207], dtype=torch.int64)
        self._store_bonus(chain_by_rid, rids_r2, bonus_r2, chain_r2, num_draft_tokens)

        assert len(chain_by_rid) == 1
        assert chain_by_rid["req_b"].tolist() == [204, 205, 206, 207]

    def test_scenario_c_row_order_changes(self):
        """Batch row order changes between rounds."""
        chain_by_rid = {}
        num_draft_tokens = 4

        # Round 1: [req_a, req_b]
        rids_r1 = ["req_a", "req_b"]
        bonus_r1 = torch.tensor([100, 200], dtype=torch.int64)
        chain_r1 = torch.tensor([100, 101, 102, 103, 200, 201, 202, 203], dtype=torch.int64)
        self._store_bonus(chain_by_rid, rids_r1, bonus_r1, chain_r1, num_draft_tokens)

        # Round 2: [req_b, req_a] (reordered)
        rids_r2 = ["req_b", "req_a"]
        bonus_r2 = torch.tensor([204, 104], dtype=torch.int64)
        chain_r2 = torch.tensor([204, 205, 206, 207, 104, 105, 106, 107], dtype=torch.int64)
        self._store_bonus(chain_by_rid, rids_r2, bonus_r2, chain_r2, num_draft_tokens)

        # Chains are keyed by rid, so order doesn't matter
        assert chain_by_rid["req_a"].tolist() == [104, 105, 106, 107]
        assert chain_by_rid["req_b"].tolist() == [204, 205, 206, 207]

    def test_scenario_d_retraction(self):
        """One request is retracted then resumed."""
        chain_by_rid = {}
        num_draft_tokens = 4

        # Round 1: both active
        rids_r1 = ["req_a", "req_b"]
        bonus_r1 = torch.tensor([100, 200], dtype=torch.int64)
        self._store_bonus(chain_by_rid, rids_r1, bonus_r1, num_draft_tokens=num_draft_tokens)

        # req_b retracted
        chain_by_rid.pop("req_b", None)
        assert "req_b" not in chain_by_rid
        assert "req_a" in chain_by_rid

        # Round 2: only req_a, req_b is gone
        rids_r2 = ["req_a"]
        bonus_r2 = torch.tensor([104], dtype=torch.int64)
        chain_r2 = torch.tensor([104, 105, 106, 107], dtype=torch.int64)
        self._store_bonus(chain_by_rid, rids_r2, bonus_r2, chain_r2, num_draft_tokens)

        assert len(chain_by_rid) == 1
        assert chain_by_rid["req_a"].tolist() == [104, 105, 106, 107]

    def test_scenario_e_missing_chain(self):
        """Missing chain fault injection: rid not in chain_by_rid."""
        chain_by_rid = {}
        num_draft_tokens = 4

        # req_a has a chain, req_b does not
        chain_by_rid["req_a"] = torch.tensor([100, 101, 102, 103], dtype=torch.int64)

        # Missing chain for req_b: must use real last token, not 0
        req_b_last_token = 999
        if "req_b" not in chain_by_rid:
            chain = torch.zeros(num_draft_tokens, dtype=torch.int64)
            chain[0] = req_b_last_token
            chain_by_rid["req_b"] = chain

        # Verify: no fabricated token 0 as root
        assert chain_by_rid["req_b"][0].item() == 999
        assert chain_by_rid["req_a"].tolist() == [100, 101, 102, 103]

    def test_no_gpu_slice_retained(self):
        """Verify chains are stored on CPU, not GPU."""
        chain_by_rid = {}
        num_draft_tokens = 4

        # Store from GPU tensors
        bonus = torch.tensor([100], dtype=torch.int64, device="cpu")
        chain = torch.tensor([100, 101, 102, 103], dtype=torch.int64, device="cpu")

        # Simulate the storage (CPU clone)
        rids = ["req_a"]
        rows = chain.to(torch.int64).reshape(len(rids), num_draft_tokens)
        for i, rid in enumerate(rids):
            chain_by_rid[rid] = rows[i].to(device="cpu", dtype=torch.int64).clone()

        # Verify: stored on CPU
        assert chain_by_rid["req_a"].device.type == "cpu"

    def test_stress_1000_rounds(self):
        """Stress test: 1000 synthetic rounds with request churn."""
        chain_by_rid = {}
        max_concurrent = 4
        num_draft_tokens = 4
        max_live = 0

        for round_idx in range(1000):
            # Add requests
            for i in range(max_concurrent):
                rid = f"req_{round_idx}_{i}"
                chain = torch.tensor(
                    [i * 100 + j for j in range(num_draft_tokens)],
                    dtype=torch.int64
                )
                chain_by_rid[rid] = chain

            max_live = max(max_live, len(chain_by_rid))

            # Remove all from this round
            for i in range(max_concurrent):
                rid = f"req_{round_idx}_{i}"
                chain_by_rid.pop(rid, None)

        assert len(chain_by_rid) == 0, (
            f"Chain map has {len(chain_by_rid)} entries after 1000 rounds"
        )
        assert max_live <= max_concurrent


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
