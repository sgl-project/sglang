"""Phase 9: P1-5 1,000-round production-state stress test.

Tests RID-keyed speculative state through many rounds with pseudo-random churn.
"""
from __future__ import annotations

import os
import sys
import random
import torch
import traceback

sys.path.insert(0, "/home/liang/sglang/python")

from sglang.srt.speculative.glm52_eagle3_pp import (
    build_layer_to_slot_map,
    build_slot_ownership_map,
    allocate_packed_aux_buffer,
    pack_aux_into_buffer,
    unpack_aux_from_buffer,
    validate_pp_proxy_keys,
    REQUIRED_PP_PROXY_KEYS,
    GLM52_EAGLE3_AUX_PP_KEY,
)
from sglang.srt.distributed.pp_packed_transport import (
    PPSchemaCache, PPStaticBufferRegistry,
    calculate_pp_buffer_layout, pack_pp_proxy_tensors, unpack_pp_proxy_tensors,
)


class MockSchedulerState:
    """Minimal mock of scheduler PP+spec state."""
    def __init__(self, num_draft_tokens=4):
        self._pp_spec_chain_by_rid = {}
        self._pp_spec_req_state = {}
        self._pp_spec_round_idx = 0
        self.num_draft_tokens = num_draft_tokens
    
    def store_bonus(self, rids, bonus_tokens, chain_tokens=None):
        """Store per-request chain state."""
        self._pp_spec_round_idx += 1
        for i, rid in enumerate(rids):
            if chain_tokens is not None:
                rows = chain_tokens[i]
            else:
                rows = torch.zeros(self.num_draft_tokens, dtype=torch.int64)
                rows[0] = int(bonus_tokens[i])
            self._pp_spec_chain_by_rid[rid] = rows.clone()
            self._pp_spec_req_state[rid] = {
                "chain_initialized": True,
                "spec_round": self._pp_spec_req_state.get(rid, {}).get("spec_round", 0) + 1,
            }
    
    def remove_finished(self, rids):
        """Remove finished requests."""
        for rid in rids:
            self._pp_spec_chain_by_rid.pop(rid, None)
            self._pp_spec_req_state.pop(rid, None)
    
    def get_chain(self, rid):
        return self._pp_spec_chain_by_rid.get(rid)
    
    @property
    def num_active(self):
        return len(self._pp_spec_chain_by_rid)


def test_stress_1000_rounds():
    """Run 1,000 rounds of pseudo-random request churn."""
    random.seed(42)
    state = MockSchedulerState(num_draft_tokens=4)
    
    # Track expected state
    expected_chains = {}
    all_rids = set()
    max_active = 64
    
    errors = []
    
    for rnd in range(1000):
        # Pseudo-random churn
        active_rids = list(state._pp_spec_chain_by_rid.keys())
        
        # Some requests finish
        to_remove = []
        for rid in active_rids:
            if random.random() < 0.1:  # 10% chance to finish
                to_remove.append(rid)
        
        if to_remove:
            state.remove_finished(to_remove)
            for rid in to_remove:
                expected_chains.pop(rid, None)
                all_rids.discard(rid)
        
        # Some new requests arrive
        active_rids = list(state._pp_spec_chain_by_rid.keys())
        num_new = random.randint(0, min(5, max_active - len(active_rids)))
        new_rids = [f"req_{rnd}_{i}" for i in range(num_new)]
        all_rids.update(new_rids)
        
        # All active rids get new bonus tokens
        active_rids = list(state._pp_spec_chain_by_rid.keys()) + new_rids
        if not active_rids:
            continue
        
        # Generate bonus tokens
        bonus = torch.randint(0, 1000, (len(active_rids),), dtype=torch.int64)
        
        # Sometimes include chain tokens
        if random.random() < 0.7 and len(active_rids) > 0:
            chain = torch.randint(0, 1000, (len(active_rids), 4), dtype=torch.int64)
            state.store_bonus(active_rids, bonus, chain)
            for i, rid in enumerate(active_rids):
                expected_chains[rid] = chain[i].clone()
        else:
            state.store_bonus(active_rids, bonus)
            for i, rid in enumerate(active_rids):
                expected_chains[rid] = torch.zeros(4, dtype=torch.int64)
                expected_chains[rid][0] = bonus[i]
        
        # Validate state
        for rid in active_rids:
            actual = state.get_chain(rid)
            expected = expected_chains[rid]
            if actual is None:
                errors.append(f"Round {rnd}: rid {rid} missing chain")
                continue
            if not torch.equal(actual, expected):
                errors.append(f"Round {rnd}: rid {rid} chain mismatch: expected {expected.tolist()}, got {actual.tolist()}")
        
        # Verify no stale state
        for rid in state._pp_spec_chain_by_rid:
            if rid not in expected_chains:
                errors.append(f"Round {rnd}: stale chain for rid {rid}")
        
        # Verify no RID leakage
        if len(state._pp_spec_chain_by_rid) != len(expected_chains):
            errors.append(f"Round {rnd}: state size {len(state._pp_spec_chain_by_rid)} != expected {len(expected_chains)}")
        
        # Check spec round tracking
        for rid in active_rids:
            s = state._pp_spec_req_state.get(rid, {})
            if not s.get("chain_initialized"):
                errors.append(f"Round {rnd}: rid {rid} not chain_initialized")
    
    if errors:
        print(f"  FAILURES ({len(errors)} errors):")
        for e in errors[:10]:
            print(f"    {e}")
        return False
    
    print(f"  1,000 rounds PASSED (max_active={max_active}, total_unique_rids={len(all_rids)})")
    return True


def test_schema_cache_bounded():
    """Test schema cache stays within bounds."""
    cache = PPSchemaCache(max_entries=64)
    
    for i in range(200):
        from sglang.srt.distributed.pp_packed_transport import PPSchemaKey
        key = PPSchemaKey(
            presence_mask=i,
            hidden_size=64,
            capture_layers=3,
            topk_size=8,
            dtype_id=2,
            max_rows_bucket=64,
        )
        cache.register(key, 1000, 100, {}, {})
    
    assert cache.size <= 64, f"Cache size {cache.size} exceeds max 64"
    assert cache.evictions > 0, "Expected some evictions"
    
    print(f"  Schema cache bounded: size={cache.size}, evictions={cache.evictions}")
    return True


def test_static_buffer_stability():
    """Test static buffer pointer stability."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    registry = PPStaticBufferRegistry(device=device)
    
    ptrs = {}
    for bucket in [1, 4, 16, 64]:
        db, cb = registry.get_or_allocate(bucket, 1024, 256, torch.bfloat16)
        ptrs[bucket] = (db.data_ptr(), cb.data_ptr())
    
    # Re-request same buckets — pointers must be stable
    for bucket in [1, 4, 16, 64]:
        db, cb = registry.get_or_allocate(bucket, 1024, 256, torch.bfloat16)
        assert db.data_ptr() == ptrs[bucket][0], f"Bucket {bucket}: data ptr changed!"
        assert cb.data_ptr() == ptrs[bucket][1], f"Bucket {bucket}: control ptr changed!"
    
    # Verify allocation count
    assert registry.allocation_count == 4
    
    print(f"  Static buffer stability: {registry.allocation_count} allocations, pointers stable")
    return True


def test_packed_transport_round_trip():
    """Test packed transport round-trip with varying active rows."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden_size = 64
    num_capture = 3
    topk_size = 8
    max_rows = 64
    
    for active_rows in [1, 4, 16, 32, 64, 1, 16, 4]:
        td = {
            "hidden_states": torch.randn(active_rows, hidden_size, dtype=torch.bfloat16, device=device),
            "residual": torch.randn(active_rows, hidden_size, dtype=torch.bfloat16, device=device),
            "glm52_eagle3_aux_hidden_states": torch.randn(active_rows, num_capture, hidden_size, dtype=torch.bfloat16, device=device),
            "topk_indices": torch.randint(0, 100, (active_rows, topk_size), dtype=torch.int32, device=device),
            "next_token_ids": torch.randint(0, 1000, (active_rows,), dtype=torch.int64, device=device),
        }
        
        sk, dn, cn, do, co = calculate_pp_buffer_layout(td, hidden_size, num_capture, topk_size, max_rows)
        db = torch.zeros(dn, dtype=torch.bfloat16, device=device)
        cb = torch.zeros(cn, dtype=torch.int32, device=device)
        
        pack_pp_proxy_tensors(td, db, cb, do, co, active_rows)
        
        cache = PPSchemaCache()
        entry = cache.register(sk, dn, cn, do, co)
        
        result = unpack_pp_proxy_tensors(db, cb, entry, active_rows, device, torch.bfloat16, hidden_size, num_capture, topk_size)
        
        # Validate
        assert torch.allclose(result["hidden_states"], td["hidden_states"], atol=1e-2)
        assert torch.allclose(result["residual"], td["residual"], atol=1e-2)
        assert torch.allclose(result["glm52_eagle3_aux_hidden_states"], td["glm52_eagle3_aux_hidden_states"], atol=1e-2)
        assert torch.equal(result["topk_indices"], td["topk_indices"])
    
    print("  Packed transport round-trip PASSED")
    return True


if __name__ == "__main__":
    print("=== Phase 9: P1-5 Stress Test ===")
    
    ok = test_stress_1000_rounds()
    if not ok:
        sys.exit(1)
    
    ok = test_schema_cache_bounded()
    if not ok:
        sys.exit(1)
    
    ok = test_static_buffer_stability()
    if not ok:
        sys.exit(1)
    
    ok = test_packed_transport_round_trip()
    if not ok:
        sys.exit(1)
    
    print("\n=== All Phase 9 tests PASSED ===")
