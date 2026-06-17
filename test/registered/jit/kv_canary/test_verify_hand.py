from __future__ import annotations

import random
import struct
from dataclasses import dataclass
from typing import Callable

import pytest
import torch

from sglang.jit_kernel.kv_canary import consts
from sglang.jit_kernel.kv_canary.consts import splitmix64, splitmix64_mix3
from sglang.jit_kernel.kv_canary.verify import (
    CanaryLaunchTag,
    RealKvSource,
    VerifyOrWriteContext,
    VerifyPlan,
    launch_canary_verify_kernel,
)
from sglang.jit_kernel.kv_canary.verify_ref import (
    _compute_real_kv_hash_scalar,
    launch_canary_verify_kernel_torch_reference,
)
from sglang.jit_kernel.kv_canary.write_ref import (
    launch_canary_write_kernel_torch_reference,
)
from sglang.jit_kernel.tests.kv_canary._canary_helpers import (
    FakeViolationLog,
    assert_only_bits_set,
    chain_anchor_signed,
    make_canary_buf,
    make_canary_buf_pair,
    make_log_pair,
    make_real_kv_source,
    make_real_kv_sources,
    make_verify_plan,
    make_verify_plan_pair,
    make_write_plan,
    read_slot_fields,
    stamp_clean_chain,
    stamp_pair,
    to_signed_int64,
    write_slot_fields,
)
from sglang.jit_kernel.tests.kv_canary._differential import (
    _run_both_verify,
    run_verify_diff,
)
from sglang.jit_kernel.tests.kv_canary._fixtures import clone_real_kv_sources
from sglang.jit_kernel.tests.kv_canary._hand_oracle import (
    _hand_fold_all,
    _hand_fold_partial,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")
register_amd_ci(est_time=30, suite="jit-kernel-unit-test-amd")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Test requires CUDA"
)


_DEVICE = torch.device("cuda")


# ---------------------------------------------------------------------------
# Shared per-test scaffolding helpers.
# ---------------------------------------------------------------------------


def _buf_pair(
    num_slots: int = 16, slot_stride_bytes: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    return make_canary_buf_pair(
        num_slots=num_slots, slot_stride_bytes=slot_stride_bytes, device=_DEVICE
    )


def _stamp_head(
    buf_pair: tuple[torch.Tensor, torch.Tensor],
    *,
    slot_idx: int,
    token: int = 42,
    position: int = 0,
    prev_hash: int | None = None,
    real_kv_hash: int = 0,
) -> None:
    """``stamp_pair`` with ``prev_hash`` defaulting to ``chain_anchor_signed()`` (the chain-head value)."""
    stamp_pair(
        buf_pair,
        slot_idx=slot_idx,
        token=token,
        position=position,
        prev_hash=chain_anchor_signed() if prev_hash is None else prev_hash,
        real_kv_hash=real_kv_hash,
    )


def _plan_pair_single(
    *,
    slot_idx: int,
    position: int,
    prev_slot_idx: int = -1,
    expected_input_id: int | None = None,
    capacity: int | None = None,
) -> tuple[VerifyPlan, VerifyPlan]:
    """Single-entry verify plan pair, with optional expected_input_id (None → default sentinel)."""
    expected = None if expected_input_id is None else [expected_input_id]
    return make_verify_plan_pair(
        slot_indices=[slot_idx],
        positions=[position],
        prev_slot_indices=[prev_slot_idx],
        expected_input_ids=expected,
        capacity=capacity,
        device=_DEVICE,
    )


def _n_violations(log: FakeViolationLog) -> int:
    return int(log.write_index[0].item())


def _fail_bits(log: FakeViolationLog, row: int = 0) -> int:
    return int(log.ring[row, consts.VIOLATION_FIELD_FAIL_REASON_BITS].item())


def _run_both_verify_no_rkv(
    *,
    buf_pair: tuple[torch.Tensor, torch.Tensor],
    plan_pair: tuple[VerifyPlan, VerifyPlan],
    cuda_log: FakeViolationLog,
    ref_log: FakeViolationLog,
    assert_equal: bool = True,
    kernel_kind: CanaryLaunchTag = CanaryLaunchTag.HEAD_K_FULL,
) -> None:
    """``_run_both_verify`` with empty real_kv sources / NONE mode — the most common in-place verify run."""
    _run_both_verify(
        cuda_canary_buf=buf_pair[0],
        ref_canary_buf=buf_pair[1],
        plan_cuda=plan_pair[0],
        plan_ref=plan_pair[1],
        cuda_log=cuda_log,
        ref_log=ref_log,
        real_kv_sources_cuda=(),
        real_kv_sources_ref=(),
        real_kv_hash_mode=consts.RealKvHashMode.NONE,
        kernel_kind=kernel_kind,
        assert_equal=assert_equal,
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class _VerifySingleSlotInput:
    token: int = 42
    position: int = 0
    stored_prev_hash_signed: int
    stored_real_kv_hash_signed: int = 0
    real_kv_sources: tuple[RealKvSource, ...] = ()
    real_kv_hash_mode: consts.RealKvHashMode = consts.RealKvHashMode.NONE


def _run_verify_single_slot_byte_equal(case: _VerifySingleSlotInput) -> None:
    buf_pair = _buf_pair()
    stamp_pair(
        buf_pair,
        slot_idx=1,
        token=case.token,
        position=case.position,
        prev_hash=case.stored_prev_hash_signed,
        real_kv_hash=case.stored_real_kv_hash_signed,
    )
    sources_cuda = case.real_kv_sources
    sources_ref = clone_real_kv_sources(sources_cuda)
    plan_pair = _plan_pair_single(slot_idx=1, position=case.position)
    run_verify_diff(
        buf_pair=buf_pair,
        plan_pair=plan_pair,
        real_kv_sources_pair=(sources_cuda, sources_ref),
        real_kv_hash_mode=case.real_kv_hash_mode,
    )


def _stamp_clean_kv_chain(
    *,
    buf_pair: tuple[torch.Tensor, torch.Tensor],
    sources_cuda: tuple[RealKvSource, ...],
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    real_kv_hash_mode: consts.RealKvHashMode,
) -> None:
    """Use the Python write ref impl to populate the canary buf for a fresh chain.

    Lets verify tests start from a known-good chain without re-implementing splitmix64 by hand.
    """
    n = int(input_ids.shape[0])
    cuda_buf, ref_buf = buf_pair
    write_plan = make_write_plan(
        write_offsets=[0, n],
        seed_slot_indices=[-1],
        num_valid_reqs=1,
        device=_DEVICE,
    )
    log = FakeViolationLog.allocate(device=_DEVICE)
    # enable_write_input_assert=False is hard-wired here, so the kernel API requires the
    # expected_* tensors be None (otherwise it raises ValueError).
    launch_canary_write_kernel_torch_reference(
        context=VerifyOrWriteContext(
            canary_buf=cuda_buf,
            kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
            violation_ring=log.ring,
            violation_write_index=log.write_index,
            slot_run_counter=log.slot_run_counter,
            kernel_run_counter=log.kernel_run_counter,
            enable_chain_position_assert=log.enable_chain_position_assert,
            real_kv_sources=sources_cuda,
            real_kv_hash_mode=real_kv_hash_mode,
        ),
        plan=write_plan,
        input_ids=input_ids,
        positions=positions,
        out_cache_loc=out_cache_loc,
        enable_write_input_assert=False,
        expected_input_tokens=None,
        expected_input_positions=None,
    )
    ref_buf.copy_(cuda_buf)


# ---------------------------------------------------------------------------
# Kernel-contract invariants.
# ---------------------------------------------------------------------------


class TestChain:
    def test_chain_head_anchor(self) -> None:
        """``prev_slot_idx == -1`` → kernel uses ``splitmix64(consts.CANARY_CHAIN_ANCHOR)`` as the expected prev_hash."""
        # Step 1: stamp slot 5 such that stored.prev_hash already equals splitmix64(consts.CANARY_CHAIN_ANCHOR).
        buf_pair = _buf_pair()
        _stamp_head(buf_pair, slot_idx=5)

        # Step 2: a single-entry plan with prev_slot_idx = -1 should record no violation.
        plan_pair = _plan_pair_single(slot_idx=5, position=0)
        cuda_log, _ = run_verify_diff(buf_pair=buf_pair, plan_pair=plan_pair)
        assert _n_violations(cuda_log) == 0

    def test_chain_link_byte_equal_5_step(self) -> None:
        """5-step chain, CUDA vs ref byte-equal across ring / counters / canary_buf (read-only)."""
        cuda_buf, ref_buf = _buf_pair()
        slot_indices = [1, 2, 3, 4, 5]
        tokens = [11, 22, 33, 44, 55]
        positions = [0, 1, 2, 3, 4]
        stamp_clean_chain(
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            tokens=tokens,
            positions=positions,
            slot_indices=slot_indices,
        )
        plan_pair = make_verify_plan_pair(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=[-1, 1, 2, 3, 4],
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(buf_pair=(cuda_buf, ref_buf), plan_pair=plan_pair)
        assert _n_violations(cuda_log) == 0

    def test_chain_link_byte_equal_5_step_hardcoded(self) -> None:
        """5-step chain with hand-computed splitmix64 expected sequence; defends against ref + CUDA co-drift."""
        tokens = [101, 202, 303, 404, 505]
        positions = [0, 1, 2, 3, 4]
        slot_indices = [1, 2, 3, 4, 5]
        real_kv_hashes = [0, 0, 0, 0, 0]

        # Step 1: compute the expected stored prev_hash sequence in pure Python via splitmix64.
        expected_prev_hashes_u64: list[int] = []
        running = splitmix64(consts.CANARY_CHAIN_ANCHOR)
        for token, position, real_kv_hash in zip(tokens, positions, real_kv_hashes):
            expected_prev_hashes_u64.append(running)
            running = splitmix64_mix3(running, token, position)
        expected_prev_hashes_signed = [
            to_signed_int64(h) for h in expected_prev_hashes_u64
        ]

        # Step 2: stamp each slot manually with the hardcoded expected prev_hash.
        buf_pair = _buf_pair()
        cuda_buf, _ = buf_pair
        for slot_idx, token, position, prev_hash in zip(
            slot_indices, tokens, positions, expected_prev_hashes_signed
        ):
            stamp_pair(
                buf_pair,
                slot_idx=slot_idx,
                token=token,
                position=position,
                prev_hash=prev_hash,
            )

        # Step 3: verify the 5-step chain — no violation expected and the ref vs CUDA state byte-equal.
        plan_pair = make_verify_plan_pair(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=[-1, 1, 2, 3, 4],
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(buf_pair=buf_pair, plan_pair=plan_pair)

        assert _n_violations(cuda_log) == 0

        # Step 4: also independently confirm the *stored* prev_hash at each slot matches the hardcoded sequence.
        for slot_idx, expected_signed in zip(slot_indices, expected_prev_hashes_signed):
            _, _, stored_prev_hash, _ = read_slot_fields(
                canary_buf=cuda_buf, slot_idx=slot_idx
            )
            assert stored_prev_hash == expected_signed

    def test_chain_advance_formula_matches_spec(self) -> None:
        """Ref impl agrees with the chained splitmix64 chain-step formula.

        The chain step folds each of the 3 inputs into the accumulator sequentially via
        ``acc = splitmix64(acc ^ next)``, starting from ``splitmix64(prev_hash)``. ``splitmix64_mix3``
        must produce the same result as the explicit chain. ``real_kv_hash`` is intentionally NOT
        part of the chain hash — see ``compute_slot_hash`` for the radix-folding rationale.
        """
        cases = [
            (consts.CANARY_CHAIN_ANCHOR, 0, 0),
            (0x1234567890ABCDEF, 100, 5),
            (0, 0xFFFF, 0x7FFFFFFF),
            (0x123, 1, 1),
            (0xFFFFFFFFFFFFFFFF, 0xFFFF, 0xFFFF),
        ]
        for prev_hash, token, position in cases:
            u64_mask = (1 << 64) - 1
            h = splitmix64(prev_hash & u64_mask)
            h = splitmix64(h ^ (token & u64_mask))
            expected = splitmix64(h ^ (position & u64_mask))

            actual = (
                splitmix64_mix3(
                    prev_hash & u64_mask, token & u64_mask, position & u64_mask
                )
                & u64_mask
            )
            assert actual == expected, (
                f"chain advance mismatch: prev={prev_hash:#x} token={token:#x} pos={position:#x} "
                f"expected={expected:#x} actual={actual:#x}"
            )

    def test_chain_head_anchored_on_constant(self) -> None:
        """prev_slot==-1 + stored prev_hash != splitmix64(ANCHOR) → CHAIN_HASH bit set."""
        buf_pair = _buf_pair()
        slot_idx = 5
        _stamp_head(buf_pair, slot_idx=slot_idx, prev_hash=to_signed_int64(0xDEADBEEF))

        plan_pair = _plan_pair_single(slot_idx=slot_idx, position=0)
        cuda_log, _ = run_verify_diff(buf_pair=buf_pair, plan_pair=plan_pair)
        assert _n_violations(cuda_log) == 1
        assert _fail_bits(cuda_log) & consts.FailReason.VERIFY_CHAIN_HASH_MISMATCH

    def test_chain_head_prev_hash_equals_splitmix64_anchor_random_50(self) -> None:
        random.seed(0)
        expected_prev_hash_signed = to_signed_int64(
            splitmix64(consts.CANARY_CHAIN_ANCHOR)
        )

        for _ in range(50):
            token = random.randint(0, 0x7FFFFFFF)
            position = random.randint(0, 0x7FFFFFFF)
            slot_idx = random.randint(0, 15)

            buf_pair = _buf_pair()
            _stamp_head(
                buf_pair,
                slot_idx=slot_idx,
                token=token,
                position=position,
                prev_hash=expected_prev_hash_signed,
            )

            plan_pair = _plan_pair_single(slot_idx=slot_idx, position=position)
            cuda_log, _ = run_verify_diff(
                buf_pair=buf_pair, plan_pair=plan_pair, assert_equal=False
            )

            assert (
                _n_violations(cuda_log) == 0
            ), f"unexpected violation at iteration token={token} position={position} slot={slot_idx}"

    def test_prev_slot_padding_skips_chain_check_arbitrary_stored_hash(self) -> None:
        """prev_slot_idx == TOKEN_TO_KV_SLOT_PADDING → chain check is skipped, regardless of stored chain hash."""
        buf_pair = _buf_pair()
        # Stamp slot 5 with an arbitrary (non-anchor, non-derivable) prev_hash. Without the skip,
        # the kernel would compute expected_chain_hash from slot 0's canary (all zeros = 0) and
        # flag VERIFY_CHAIN_HASH_MISMATCH on every such row.
        stamp_pair(
            buf_pair,
            slot_idx=5,
            token=42,
            position=0,
            prev_hash=to_signed_int64(0xDEADBEEFCAFEBABE),
        )

        plan_pair = _plan_pair_single(
            slot_idx=5, position=0, prev_slot_idx=consts.TOKEN_TO_KV_SLOT_PADDING
        )
        cuda_log, _ = run_verify_diff(buf_pair=buf_pair, plan_pair=plan_pair)
        assert _n_violations(cuda_log) == 0

    def test_prev_slot_padding_does_not_mask_position_check(self) -> None:
        """prev_slot == padding skips ONLY the chain check; position mismatch still fires."""
        buf_pair = _buf_pair()
        stamp_pair(
            buf_pair,
            slot_idx=7,
            token=11,
            position=0,
            prev_hash=to_signed_int64(0x12345678),
        )

        # Plan claims position 99 — chain check is skipped (prev=padding) but position must still fire.
        plan_pair = _plan_pair_single(
            slot_idx=7, position=99, prev_slot_idx=consts.TOKEN_TO_KV_SLOT_PADDING
        )
        cuda_log, _ = run_verify_diff(buf_pair=buf_pair, plan_pair=plan_pair)

        assert _n_violations(cuda_log) == 1
        assert_only_bits_set(
            _fail_bits(cuda_log), consts.FailReason.VERIFY_POSITION_MISMATCH
        )


class TestViolationField:
    def test_violation_token_mismatch(self) -> None:
        """Stored token differs from a fresh write at the same slot → TOKEN-side accounting via the chain bit."""
        # Verify kernel doesn't have a TOKEN fail bit per se — token mismatch propagates into next-slot
        # CHAIN_HASH mismatch. Inject token corruption at slot 2 and verify slot 3 sees CHAIN_HASH bit set.
        cuda_buf, ref_buf = _buf_pair()
        slot_indices = [1, 2, 3]
        tokens = [100, 200, 300]
        positions = [0, 1, 2]
        stamp_clean_chain(
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            tokens=tokens,
            positions=positions,
            slot_indices=slot_indices,
        )

        # Step: corrupt the stored token at slot 2 in both buffers — chain hash propagates downstream.
        stamp_pair(
            (cuda_buf, ref_buf),
            slot_idx=2,
            token=999,
            position=1,
            prev_hash=0,
        )

        plan_pair = make_verify_plan_pair(
            slot_indices=[3], positions=[2], prev_slot_indices=[2], device=_DEVICE
        )
        cuda_log, _ = run_verify_diff(buf_pair=(cuda_buf, ref_buf), plan_pair=plan_pair)

        assert _n_violations(cuda_log) == 1
        assert_only_bits_set(
            _fail_bits(cuda_log), consts.FailReason.VERIFY_CHAIN_HASH_MISMATCH
        )

    def test_violation_position_mismatch(self) -> None:
        """Stored position differs from what the slot's chain reconstruction would yield → POSITION bit."""
        # Stamp slot 7 with a valid head chain but stored position = 0; ask verify to expect position 5.
        buf_pair = _buf_pair()
        _stamp_head(buf_pair, slot_idx=7)

        plan_pair = _plan_pair_single(slot_idx=7, position=5)
        cuda_log, _ = run_verify_diff(buf_pair=buf_pair, plan_pair=plan_pair)

        assert _n_violations(cuda_log) == 1
        assert_only_bits_set(
            _fail_bits(cuda_log), consts.FailReason.VERIFY_POSITION_MISMATCH
        )

    def test_violation_position_diverges_from_plan(self) -> None:
        """Plan-supplied position contradicts stored position → POSITION bit (verify trusts plan, not +1)."""
        # Step: a clean chain head with stored position 0; plan claims position 99 — kernel must flag POSITION.
        buf_pair = _buf_pair()
        _stamp_head(buf_pair, slot_idx=3, token=11)

        plan_pair = _plan_pair_single(slot_idx=3, position=99)
        cuda_log, _ = run_verify_diff(buf_pair=buf_pair, plan_pair=plan_pair)

        assert_only_bits_set(
            _fail_bits(cuda_log), consts.FailReason.VERIFY_POSITION_MISMATCH
        )

    def test_violation_prev_hash_mismatch(self) -> None:
        """Stored prev_hash differs from predecessor-derived expectation → CHAIN_HASH bit."""
        cuda_buf, ref_buf = _buf_pair()
        slot_indices = [1, 2]
        tokens = [10, 20]
        positions = [0, 1]
        stamp_clean_chain(
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            tokens=tokens,
            positions=positions,
            slot_indices=slot_indices,
        )

        # Step: corrupt slot 2's stored prev_hash with a bogus signed int64.
        stamp_pair(
            (cuda_buf, ref_buf),
            slot_idx=2,
            token=20,
            position=1,
            prev_hash=0x1234567812345678,
        )

        plan_pair = make_verify_plan_pair(
            slot_indices=[2], positions=[1], prev_slot_indices=[1], device=_DEVICE
        )
        cuda_log, _ = run_verify_diff(buf_pair=(cuda_buf, ref_buf), plan_pair=plan_pair)

        assert_only_bits_set(
            _fail_bits(cuda_log), consts.FailReason.VERIFY_CHAIN_HASH_MISMATCH
        )

    def test_violation_real_kv_hash_mismatch(self) -> None:
        """Mutate one byte of a RealKvSource tensor after writing the chain → REAL_KV_HASH bit on verify."""
        buf_pair = _buf_pair()
        sources_cuda = make_real_kv_sources(count=1, device=_DEVICE)

        # Step: write a chain with real_kv_hash mixin, then mutate one byte in the source tensors so the next
        # verify reconstructs a hash that differs from the stored one.
        _stamp_clean_kv_chain(
            buf_pair=buf_pair,
            sources_cuda=sources_cuda,
            input_ids=torch.tensor([7, 8, 9], dtype=torch.int64, device=_DEVICE),
            positions=torch.tensor([0, 1, 2], dtype=torch.int64, device=_DEVICE),
            out_cache_loc=torch.tensor([1, 2, 3], dtype=torch.int64, device=_DEVICE),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )

        # Mutate one byte in BOTH copies so the verify recomputed hash diverges from stored.
        sources_ref = clone_real_kv_sources(sources_cuda)
        sources_cuda[0].tensor[1, 0] ^= 0xFF
        sources_ref[0].tensor.copy_(sources_cuda[0].tensor)

        plan_pair = _plan_pair_single(slot_idx=1, position=0)
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            real_kv_sources_pair=(sources_cuda, sources_ref),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )

        assert_only_bits_set(
            _fail_bits(cuda_log), consts.FailReason.VERIFY_REAL_KV_HASH_MISMATCH
        )

    @pytest.mark.parametrize("bit_to_trigger", ["POSITION", "PREV_HASH", "REAL_KV"])
    @pytest.mark.parametrize("injection_position", ["head", "mid", "last"])
    @pytest.mark.parametrize("ring_state", ["open", "full"])
    def test_violation_bit_injection_position_ring_state_matrix(
        self,
        bit_to_trigger: str,
        injection_position: str,
        ring_state: str,
    ) -> None:
        """Sweep injection_position x bit_to_trigger x ring_state for verify-kernel fail-reason coverage."""
        _RING_CAPACITY = 4
        slot_indices = [1, 2, 3, 4, 5]
        tokens = [11, 22, 33, 44, 55]
        positions = [0, 1, 2, 3, 4]

        corruption_index = {"head": 0, "mid": 2, "last": 4}[injection_position]
        corrupt_slot = slot_indices[corruption_index]

        expected_bit = {
            "POSITION": consts.FailReason.VERIFY_POSITION_MISMATCH,
            "PREV_HASH": consts.FailReason.VERIFY_CHAIN_HASH_MISMATCH,
            "REAL_KV": consts.FailReason.VERIFY_REAL_KV_HASH_MISMATCH,
        }[bit_to_trigger]

        if bit_to_trigger == "REAL_KV":
            buf_pair = _buf_pair()
            sources_cuda = make_real_kv_sources(count=1, device=_DEVICE)
            _stamp_clean_kv_chain(
                buf_pair=buf_pair,
                sources_cuda=sources_cuda,
                input_ids=torch.tensor(tokens, dtype=torch.int64, device=_DEVICE),
                positions=torch.tensor(positions, dtype=torch.int64, device=_DEVICE),
                out_cache_loc=torch.tensor(
                    slot_indices, dtype=torch.int64, device=_DEVICE
                ),
                real_kv_hash_mode=consts.RealKvHashMode.ALL,
            )
            sources_ref = clone_real_kv_sources(sources_cuda)
            sources_cuda[0].tensor[corrupt_slot, 0] ^= 0xFF
            sources_ref[0].tensor.copy_(sources_cuda[0].tensor)
            real_kv_hash_mode = consts.RealKvHashMode.ALL
            real_kv_sources_cuda = sources_cuda
            real_kv_sources_ref = sources_ref
        else:
            cuda_buf, ref_buf = _buf_pair()
            buf_pair = (cuda_buf, ref_buf)
            stamp_clean_chain(
                cuda_buf=cuda_buf,
                ref_buf=ref_buf,
                tokens=tokens,
                positions=positions,
                slot_indices=slot_indices,
            )
            real_kv_hash_mode = consts.RealKvHashMode.NONE
            real_kv_sources_cuda = ()
            real_kv_sources_ref = ()

            if bit_to_trigger == "POSITION":
                stored_token, stored_pos, stored_prev, stored_rkv = read_slot_fields(
                    canary_buf=cuda_buf, slot_idx=corrupt_slot
                )
                stamp_pair(
                    buf_pair,
                    slot_idx=corrupt_slot,
                    token=stored_token,
                    position=stored_pos + 99,
                    prev_hash=stored_prev,
                    real_kv_hash=stored_rkv,
                )
            else:
                stored_token, stored_pos, stored_prev, stored_rkv = read_slot_fields(
                    canary_buf=cuda_buf, slot_idx=corrupt_slot
                )
                flipped_prev = stored_prev ^ 1
                stamp_pair(
                    buf_pair,
                    slot_idx=corrupt_slot,
                    token=stored_token,
                    position=stored_pos,
                    prev_hash=flipped_prev,
                    real_kv_hash=stored_rkv,
                )

        ring_capacity = _RING_CAPACITY
        cuda_log, ref_log = make_log_pair(capacity=ring_capacity, device=_DEVICE)
        if ring_state == "full":
            prefill_slots = list(range(8, 8 + ring_capacity))
            for slot_idx in prefill_slots:
                _stamp_head(buf_pair, slot_idx=slot_idx, token=1)
            prefill_plan_pair = make_verify_plan_pair(
                slot_indices=prefill_slots,
                positions=[99] * ring_capacity,
                prev_slot_indices=[-1] * ring_capacity,
                device=_DEVICE,
            )
            _run_both_verify_no_rkv(
                buf_pair=buf_pair,
                plan_pair=prefill_plan_pair,
                cuda_log=cuda_log,
                ref_log=ref_log,
                assert_equal=False,
            )
            assert _n_violations(cuda_log) == ring_capacity

        prev_slot_indices = [-1, 1, 2, 3, 4]
        plan_cuda, plan_ref = make_verify_plan_pair(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=prev_slot_indices,
            device=_DEVICE,
        )

        _run_both_verify(
            cuda_canary_buf=buf_pair[0],
            ref_canary_buf=buf_pair[1],
            plan_cuda=plan_cuda,
            plan_ref=plan_ref,
            cuda_log=cuda_log,
            ref_log=ref_log,
            real_kv_sources_cuda=real_kv_sources_cuda,
            real_kv_sources_ref=real_kv_sources_ref,
            real_kv_hash_mode=real_kv_hash_mode,
            assert_equal=False,
        )

        if ring_state == "open":
            write_index = _n_violations(cuda_log)
            rows_stored = min(write_index, ring_capacity)
            found = any(
                _fail_bits(cuda_log, row_idx) & expected_bit
                for row_idx in range(rows_stored)
            )
            assert found, (
                f"expected bit {expected_bit:#x} not found in any ring row "
                f"(bit_to_trigger={bit_to_trigger} injection_position={injection_position})"
            )
        else:
            assert (
                _n_violations(cuda_log) > ring_capacity
            ), "write_index did not advance beyond ring_capacity after overflow"

    def test_position_mismatch_sets_position_bit_only(self) -> None:
        """Plan.position != stored.position with chain hash correct → only POSITION bit set."""
        buf_pair = _buf_pair()
        slot_idx = 5
        _stamp_head(buf_pair, slot_idx=slot_idx, position=10)

        plan_pair = _plan_pair_single(slot_idx=slot_idx, position=99)
        cuda_log, _ = run_verify_diff(buf_pair=buf_pair, plan_pair=plan_pair)
        assert _n_violations(cuda_log) == 1
        bits = _fail_bits(cuda_log)
        assert (
            bits & consts.FailReason.VERIFY_POSITION_MISMATCH
        ), f"expected POSITION bit, got {bits:#b}"
        assert (
            bits & consts.FailReason.VERIFY_CHAIN_HASH_MISMATCH
        ) == 0, f"chain hash bit unexpectedly set: {bits:#b}"


class TestRealKvHash:
    def test_real_kv_mode_off_yields_zero(self) -> None:
        """OFF mode → stored real_kv_hash field stays zero post-write; verify with OFF agrees byte-equal."""
        buf_pair = _buf_pair()
        sources = make_real_kv_sources(count=2, device=_DEVICE)

        plan_pair = _plan_pair_single(slot_idx=1, position=0)
        _stamp_head(buf_pair, slot_idx=1, token=1)

        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            real_kv_sources_pair=(sources, sources),
        )

        assert _n_violations(cuda_log) == 0

    @pytest.mark.parametrize(
        "mode",
        [
            pytest.param(consts.RealKvHashMode.PARTIAL, id="partial"),
            pytest.param(consts.RealKvHashMode.ALL, id="all"),
        ],
    )
    def test_real_kv_mode_byte_equal(self, mode: consts.RealKvHashMode) -> None:
        """PARTIAL / ALL modes both produce CUDA-vs-ref byte-equal state on a clean 3-step chain."""
        buf_pair = _buf_pair()
        sources_cuda = make_real_kv_sources(count=2, device=_DEVICE)
        sources_ref = clone_real_kv_sources(sources_cuda)

        # Write a chain through the ref so both buffers are byte-equal post-write.
        _stamp_clean_kv_chain(
            buf_pair=buf_pair,
            sources_cuda=sources_cuda,
            input_ids=torch.tensor([10, 20, 30], dtype=torch.int64, device=_DEVICE),
            positions=torch.tensor([0, 1, 2], dtype=torch.int64, device=_DEVICE),
            out_cache_loc=torch.tensor([1, 2, 3], dtype=torch.int64, device=_DEVICE),
            real_kv_hash_mode=mode,
        )

        plan_pair = make_verify_plan_pair(
            slot_indices=[1, 2, 3],
            positions=[0, 1, 2],
            prev_slot_indices=[-1, 1, 2],
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            real_kv_sources_pair=(sources_cuda, sources_ref),
            real_kv_hash_mode=mode,
        )

        assert _n_violations(cuda_log) == 0

    @pytest.mark.parametrize("count", [1, 2, 3, 4])
    def test_real_kv_sources_fold_1_to_4(self, count: int) -> None:
        """Fold ``count`` sources sequentially → CUDA matches ref for every count in {1..4}."""
        buf_pair = _buf_pair()
        sources_cuda = make_real_kv_sources(count=count, device=_DEVICE)
        sources_ref = clone_real_kv_sources(sources_cuda)

        _stamp_clean_kv_chain(
            buf_pair=buf_pair,
            sources_cuda=sources_cuda,
            input_ids=torch.tensor([1, 2], dtype=torch.int64, device=_DEVICE),
            positions=torch.tensor([0, 1], dtype=torch.int64, device=_DEVICE),
            out_cache_loc=torch.tensor([1, 2], dtype=torch.int64, device=_DEVICE),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )

        plan_pair = make_verify_plan_pair(
            slot_indices=[1, 2],
            positions=[0, 1],
            prev_slot_indices=[-1, 1],
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            real_kv_sources_pair=(sources_cuda, sources_ref),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )

        assert _n_violations(cuda_log) == 0

    @pytest.mark.parametrize(
        "mode,fold_fn,expected_hash",
        [
            pytest.param(
                consts.RealKvHashMode.PARTIAL,
                _hand_fold_partial,
                0x6041580849E6407D,
                id="partial",
            ),
            pytest.param(
                consts.RealKvHashMode.ALL,
                _hand_fold_all,
                0x6041580849E6407D,
                id="all",
            ),
        ],
    )
    def test_real_kv_hash_fold_mode_hardcoded(
        self,
        mode: consts.RealKvHashMode,
        fold_fn: Callable[[bytes], int],
        expected_hash: int,
    ) -> None:
        # Step 1: build one RealKvSource with read_bytes=16 and a fixed byte pattern at slot 1.
        _PATTERN = bytes(
            [
                0x01,
                0x02,
                0x04,
                0x08,
                0x10,
                0x20,
                0x40,
                0x80,
                0x81,
                0x82,
                0x84,
                0x88,
                0x90,
                0xA0,
                0xC0,
                0xFF,
            ]
        )

        buf_pair = _buf_pair()
        source_cuda = make_real_kv_source(
            num_slots=16,
            num_bytes_per_token=16,
            page_size=1,
            read_bytes=16,
            device=_DEVICE,
        )
        source_cuda.tensor[1, :16] = torch.tensor(list(_PATTERN), dtype=torch.uint8)
        source_ref = RealKvSource(
            tensor=source_cuda.tensor.clone(),
            page_size=source_cuda.page_size,
            num_bytes_per_token=source_cuda.num_bytes_per_token,
            read_bytes=source_cuda.read_bytes,
        )

        # Step 2: verify hand-computed fold matches the hex literal.
        assert fold_fn(_PATTERN) == expected_hash

        # Step 3: stamp slot 1 with a chain-head entry whose real_kv_hash equals the expected value.
        _stamp_head(
            buf_pair,
            slot_idx=1,
            token=7,
            real_kv_hash=to_signed_int64(expected_hash),
        )

        # Step 4: 1-entry verify plan; no violation because stored matches recomputed.
        plan_pair = _plan_pair_single(slot_idx=1, position=0)
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            real_kv_sources_pair=((source_cuda,), (source_ref,)),
            real_kv_hash_mode=mode,
            assert_equal=False,
        )

        assert _n_violations(cuda_log) == 0

        # Step 5: mutate one byte in the source so the recomputed hash diverges from stored.
        source_cuda.tensor[1, 0] ^= 0xFF
        source_ref.tensor.copy_(source_cuda.tensor)

        plan_pair2 = _plan_pair_single(slot_idx=1, position=0)
        cuda_log2, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair2,
            real_kv_sources_pair=((source_cuda,), (source_ref,)),
            real_kv_hash_mode=mode,
        )

        assert_only_bits_set(
            _fail_bits(cuda_log2), consts.FailReason.VERIFY_REAL_KV_HASH_MISMATCH
        )

    def test_real_kv_hash_all_mode_with_multiple_sources(self) -> None:
        """ALL mode with count=2 page=16 bytes=128 sources: chain still verifies clean."""
        buf_pair = _buf_pair(num_slots=32)
        sources_cuda = make_real_kv_sources(
            count=2,
            num_bytes_per_token=128,
            page_size=16,
            num_slots=32,
            device=_DEVICE,
        )
        sources_ref = clone_real_kv_sources(sources_cuda)

        slot_indices = [1, 2, 3]
        tokens = [100, 200, 300]
        positions = [0, 1, 2]

        running = splitmix64(consts.CANARY_CHAIN_ANCHOR)
        real_kv_hashes: list[int] = []
        for slot_idx in slot_indices:
            real_kv_hashes.append(
                _compute_real_kv_hash_scalar(
                    real_kv_sources=sources_cuda,
                    real_kv_hash_mode=consts.RealKvHashMode.ALL,
                    slot_idx=slot_idx,
                    work_device=torch.device("cpu"),
                )
            )

        for slot_idx, token, position, rkv in zip(
            slot_indices, tokens, positions, real_kv_hashes
        ):
            signed_prev = to_signed_int64(running)
            stamp_pair(
                buf_pair,
                slot_idx=slot_idx,
                token=token,
                position=position,
                prev_hash=signed_prev,
                real_kv_hash=to_signed_int64(rkv),
            )
            running = splitmix64_mix3(running, token, position)

        plan_pair = make_verify_plan_pair(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=[-1, 1, 2],
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            real_kv_sources_pair=(sources_cuda, sources_ref),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )
        assert _n_violations(cuda_log) == 0

    def test_real_kv_hash_partial_mode_detects_single_bit_flip(self) -> None:
        """PARTIAL mode + 1-bit flip in source tensor → REAL_KV_HASH bit set in violation row."""
        buf_pair = _buf_pair()
        sources_cuda = make_real_kv_sources(
            count=1, num_bytes_per_token=16, device=_DEVICE
        )
        slot_idx = 3
        row_bytes = (
            sources_cuda[0]
            .tensor[slot_idx, : sources_cuda[0].read_bytes]
            .detach()
            .cpu()
            .tolist()
        )
        rkv_clean = _hand_fold_partial(bytes(row_bytes))
        _stamp_head(
            buf_pair,
            slot_idx=slot_idx,
            real_kv_hash=to_signed_int64(rkv_clean),
        )

        sources_cuda[0].tensor[slot_idx, 0] ^= 1
        sources_ref = clone_real_kv_sources(sources_cuda)

        plan_pair = _plan_pair_single(slot_idx=slot_idx, position=0)
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            real_kv_sources_pair=(sources_cuda, sources_ref),
            real_kv_hash_mode=consts.RealKvHashMode.PARTIAL,
        )

        assert _n_violations(cuda_log) >= 1
        bits = _fail_bits(cuda_log)
        assert (
            bits & consts.FailReason.VERIFY_REAL_KV_HASH_MISMATCH
        ), f"expected REAL_KV_HASH bit, got {bits:#b}"

    def test_real_kv_off_does_not_deref_real_kv_sources(self) -> None:
        buf_pair = _buf_pair(num_slots=8)
        _stamp_head(buf_pair, slot_idx=1, token=1)

        garbage_source = make_real_kv_source(
            num_slots=8,
            num_bytes_per_token=16,
            page_size=1,
            read_bytes=16,
            device=_DEVICE,
            fill=0xDE,
        )

        plan_pair = _plan_pair_single(slot_idx=1, position=0)
        cuda_log, ref_log = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            real_kv_sources_pair=((garbage_source,), (garbage_source,)),
            assert_equal=False,
        )

        assert _n_violations(cuda_log) == 0
        assert _n_violations(ref_log) == 0


class TestRealKvSource:
    def test_real_kv_source_rejects_zero_read_bytes(self) -> None:
        """RealKvSource has no \"skip me\" sentinel — read_bytes=0 must raise rather than silently pass."""
        with pytest.raises(ValueError, match="read_bytes"):
            RealKvSource(
                tensor=torch.zeros((1, 16), dtype=torch.uint8, device=_DEVICE),
                page_size=1,
                num_bytes_per_token=16,
                read_bytes=0,
            )

    def test_real_kv_source_padding_below_4(self) -> None:
        """Host wrapper pads to 4 slots when fewer sources are supplied; dummy slots are never dereferenced."""
        buf_pair = _buf_pair()
        sources = make_real_kv_sources(count=2, device=_DEVICE)

        plan_pair = _plan_pair_single(slot_idx=1, position=0)
        _stamp_head(buf_pair, slot_idx=1, token=1)
        run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            real_kv_sources_pair=(sources, sources),
        )

    def test_real_kv_source_above_4_raises(self) -> None:
        """``len(real_kv_sources) > 4`` → host wrapper raises ValueError before launching."""
        canary_buf = make_canary_buf(device=_DEVICE)
        plan = make_verify_plan(
            slot_indices=[1], positions=[0], prev_slot_indices=[-1], device=_DEVICE
        )
        log = FakeViolationLog.allocate(device=_DEVICE)
        sources = make_real_kv_sources(count=4, device=_DEVICE)
        extra = make_real_kv_source(device=_DEVICE)
        too_many = sources + (extra,)

        with pytest.raises(ValueError, match="at most 4 RealKvSource"):
            launch_canary_verify_kernel(
                context=VerifyOrWriteContext(
                    canary_buf=canary_buf,
                    kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
                    violation_ring=log.ring,
                    violation_write_index=log.write_index,
                    slot_run_counter=log.slot_run_counter,
                    kernel_run_counter=log.kernel_run_counter,
                    enable_chain_position_assert=log.enable_chain_position_assert,
                    real_kv_sources=too_many,
                    real_kv_hash_mode=consts.RealKvHashMode.NONE,
                ),
                plan=plan,
                check_verify_expected_token=True,
            )

    def test_real_kv_source_holey_dim1(self) -> None:
        """``tensor.shape[1] > page_size * num_bytes_per_token`` → trailing bytes are skipped."""
        buf_pair = _buf_pair()
        holey_source = make_real_kv_source(
            num_slots=16,
            num_bytes_per_token=16,
            page_size=1,
            read_bytes=16,
            pad_dim1=16,  # 16 trailing pad bytes per row; must be skipped.
            device=_DEVICE,
        )
        trailing_start = holey_source.page_size * holey_source.num_bytes_per_token
        # Fill those skipped trailing bytes with garbage; CUDA must not read them.
        holey_source.tensor[:, trailing_start:].fill_(0xAA)
        sources = (holey_source,)
        sources_ref = clone_real_kv_sources(sources)

        _stamp_clean_kv_chain(
            buf_pair=buf_pair,
            sources_cuda=sources,
            input_ids=torch.tensor([1, 2], dtype=torch.int64, device=_DEVICE),
            positions=torch.tensor([0, 1], dtype=torch.int64, device=_DEVICE),
            out_cache_loc=torch.tensor([1, 2], dtype=torch.int64, device=_DEVICE),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )

        plan_pair = make_verify_plan_pair(
            slot_indices=[1, 2],
            positions=[0, 1],
            prev_slot_indices=[-1, 1],
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            real_kv_sources_pair=(sources, sources_ref),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )

        assert _n_violations(cuda_log) == 0


class TestLayoutAndScheduling:
    def test_page_size_gt_1_access_pattern(self) -> None:
        """``page_size > 1`` → byte access follows ``(row=slot//page, col=(slot%page)*bpt:)``."""
        buf_pair = _buf_pair(num_slots=8)
        src = make_real_kv_source(
            num_slots=8,
            num_bytes_per_token=16,
            page_size=4,  # 2 rows × 4 slots/page × 16 bytes/slot.
            read_bytes=16,
            device=_DEVICE,
        )
        # Each slot's 16 bytes get a slot-specific signature so kernel mis-indexing would shift the hash.
        flat = src.tensor.view(-1)
        for slot_idx in range(8):
            row = slot_idx // src.page_size
            col = (slot_idx % src.page_size) * src.num_bytes_per_token
            for k in range(src.num_bytes_per_token):
                flat_index = row * (src.page_size * src.num_bytes_per_token) + col + k
                flat[flat_index] = (slot_idx * 13 + k) & 0xFF
        sources = (src,)
        sources_ref = clone_real_kv_sources(sources)

        _stamp_clean_kv_chain(
            buf_pair=buf_pair,
            sources_cuda=sources,
            input_ids=torch.tensor([1, 2], dtype=torch.int64, device=_DEVICE),
            positions=torch.tensor([0, 1], dtype=torch.int64, device=_DEVICE),
            out_cache_loc=torch.tensor([1, 5], dtype=torch.int64, device=_DEVICE),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )

        plan_pair = make_verify_plan_pair(
            slot_indices=[1, 5],
            positions=[0, 1],
            prev_slot_indices=[-1, 1],
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            real_kv_sources_pair=(sources, sources_ref),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )

        assert _n_violations(cuda_log) == 0

    def test_swa_translated_slot_indices(self) -> None:
        """SWA-translated slots already passed in plan; verify kernel does no further translation."""
        # SWA verify plans carry pre-translated slot indices — the verify kernel never sees the FULL slot
        # index again. We pre-stamp the SWA-side slot and feed it directly into the verify plan to assert no
        # extra translation happens kernel-side.
        buf_pair = _buf_pair()
        _stamp_head(buf_pair, slot_idx=2, token=99)

        plan_pair = _plan_pair_single(slot_idx=2, position=0)
        cuda_log, _ = run_verify_diff(buf_pair=buf_pair, plan_pair=plan_pair)
        assert _n_violations(cuda_log) == 0

    def test_empty_plan_no_op(self) -> None:
        """``verify_num_valid = 0`` → no ring write, no slot_run_counter bump, only kernel_run_counter += 1."""
        buf_pair = _buf_pair()
        plan_pair = make_verify_plan_pair(
            slot_indices=[],
            positions=[],
            prev_slot_indices=[],
            capacity=4,
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(buf_pair=buf_pair, plan_pair=plan_pair)

        assert _n_violations(cuda_log) == 0
        assert int(cuda_log.slot_run_counter[0].item()) == 0
        assert int(cuda_log.kernel_run_counter[0].item()) == 1

    def test_slot_zero_plan_entry_is_skipped(self) -> None:
        """slot 0 is reserved padding: verify skips loads/violations but still counts the submitted entry."""
        canary_buf = make_canary_buf(num_slots=16, slot_stride_bytes=32, device=_DEVICE)
        write_slot_fields(
            canary_buf=canary_buf,
            slot_idx=0,
            token=999,
            position=123,
            prev_hash=to_signed_int64(0xDEADBEEF),
            real_kv_hash=0,
        )
        plan = make_verify_plan(
            slot_indices=[0],
            positions=[0],
            prev_slot_indices=[-1],
            device=_DEVICE,
        )
        log = FakeViolationLog.allocate(capacity=8, device=_DEVICE)

        launch_canary_verify_kernel(
            context=VerifyOrWriteContext(
                canary_buf=canary_buf,
                kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
                violation_ring=log.ring,
                violation_write_index=log.write_index,
                slot_run_counter=log.slot_run_counter,
                kernel_run_counter=log.kernel_run_counter,
                enable_chain_position_assert=log.enable_chain_position_assert,
                real_kv_sources=(),
                real_kv_hash_mode=consts.RealKvHashMode.NONE,
            ),
            plan=plan,
            check_verify_expected_token=True,
        )
        torch.cuda.synchronize()

        assert _n_violations(log) == 0
        assert int(log.slot_run_counter[0].item()) == 1
        assert int(log.kernel_run_counter[0].item()) == 1

    @pytest.mark.parametrize(
        "runner",
        [launch_canary_verify_kernel, launch_canary_verify_kernel_torch_reference],
    )
    def test_disabled_plan_skips_slots_but_counts_kernel(
        self, runner: Callable[..., None]
    ) -> None:
        """``VerifyPlan.enable = 0`` skips active entries while still marking the verify launch as run."""
        canary_buf = make_canary_buf(num_slots=16, slot_stride_bytes=32, device=_DEVICE)
        slot_idx = 5
        write_slot_fields(
            canary_buf=canary_buf,
            slot_idx=slot_idx,
            token=42,
            position=1,
            prev_hash=to_signed_int64(0x1234),
            real_kv_hash=0,
        )
        plan = make_verify_plan(
            slot_indices=[slot_idx],
            positions=[0],
            prev_slot_indices=[-1],
            device=_DEVICE,
        )
        plan.enable[0] = 0

        log = FakeViolationLog.allocate(capacity=8, device=_DEVICE)
        log.ring.fill_(-777)
        log.write_index[0] = 3
        log.slot_run_counter[0] = 11
        log.kernel_run_counter[0] = 13
        ring_before = log.ring.clone()
        write_index_before = log.write_index.clone()
        slot_run_before = log.slot_run_counter.clone()
        kernel_run_before = log.kernel_run_counter.clone()

        runner(
            context=VerifyOrWriteContext(
                canary_buf=canary_buf,
                kernel_kind=CanaryLaunchTag.HEAD_K_FULL,
                violation_ring=log.ring,
                violation_write_index=log.write_index,
                slot_run_counter=log.slot_run_counter,
                kernel_run_counter=log.kernel_run_counter,
                enable_chain_position_assert=log.enable_chain_position_assert,
                real_kv_sources=(),
                real_kv_hash_mode=consts.RealKvHashMode.NONE,
            ),
            plan=plan,
            check_verify_expected_token=True,
        )
        if runner is launch_canary_verify_kernel:
            torch.cuda.synchronize()

        assert torch.equal(log.ring, ring_before)
        assert torch.equal(log.write_index, write_index_before)
        assert torch.equal(log.slot_run_counter, slot_run_before)
        assert (
            int(log.kernel_run_counter[0].item())
            == int(kernel_run_before[0].item()) + 1
        )

    def test_paged_layout_page_size_16(self) -> None:
        """page_size=16: slot→page mapping doesn't change verify chain semantics on a clean chain."""
        buf_pair = _buf_pair(num_slots=64)
        sources_cuda = make_real_kv_sources(
            count=1,
            num_bytes_per_token=16,
            page_size=16,
            num_slots=64,
            device=_DEVICE,
        )
        sources_ref = clone_real_kv_sources(sources_cuda)

        # Step: cross a page boundary by writing slots [15, 16] which straddle pages 0 and 1.
        slot_indices = [15, 16]
        tokens = [77, 88]
        positions = [0, 1]
        running = splitmix64(consts.CANARY_CHAIN_ANCHOR)
        # Use the reference fold (8-byte little-endian word pack + splitmix64), not a
        # byte-by-byte loop, so the stamped real_kv_hash matches what the kernel /
        # verify reference will recompute. A byte-by-byte fold was the previous bug
        # here and triggered REAL_KV_HASH violations on otherwise clean chains.
        rkv_values = [
            _compute_real_kv_hash_scalar(
                slot_idx=slot_idx,
                real_kv_sources=sources_cuda,
                real_kv_hash_mode=consts.RealKvHashMode.ALL,
                work_device=_DEVICE,
            )
            for slot_idx in slot_indices
        ]

        for slot_idx, token, position, rkv in zip(
            slot_indices, tokens, positions, rkv_values
        ):
            signed_prev = to_signed_int64(running)
            stamp_pair(
                buf_pair,
                slot_idx=slot_idx,
                token=token,
                position=position,
                prev_hash=signed_prev,
                real_kv_hash=to_signed_int64(rkv),
            )
            running = splitmix64_mix3(running, token, position)

        plan_pair = make_verify_plan_pair(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=[-1, 15],
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            real_kv_sources_pair=(sources_cuda, sources_ref),
            real_kv_hash_mode=consts.RealKvHashMode.ALL,
        )
        assert _n_violations(cuda_log) == 0

    def test_empty_plan_keeps_slot_counter_unchanged(self) -> None:
        buf_pair = _buf_pair(num_slots=8)
        _stamp_head(buf_pair, slot_idx=1, token=7)

        nonempty_plan_pair = _plan_pair_single(slot_idx=1, position=0)
        empty_plan_pair = make_verify_plan_pair(
            slot_indices=[],
            positions=[],
            prev_slot_indices=[],
            capacity=4,
            device=_DEVICE,
        )
        cuda_log, ref_log = make_log_pair(device=_DEVICE)
        for _ in range(30):
            slot_before = int(cuda_log.slot_run_counter[0].item())
            kernel_before = int(cuda_log.kernel_run_counter[0].item())

            _run_both_verify_no_rkv(
                buf_pair=buf_pair,
                plan_pair=empty_plan_pair,
                cuda_log=cuda_log,
                ref_log=ref_log,
                assert_equal=False,
            )

            assert int(cuda_log.slot_run_counter[0].item()) == slot_before
            assert int(cuda_log.kernel_run_counter[0].item()) == kernel_before + 1

            _run_both_verify_no_rkv(
                buf_pair=buf_pair,
                plan_pair=nonempty_plan_pair,
                cuda_log=cuda_log,
                ref_log=ref_log,
                assert_equal=False,
            )


class TestRunCounter:
    def test_kernel_run_counter_per_call(self) -> None:
        """``kernel_run_counter`` increments by 1 per call, even when ``verify_num_valid == 0``."""
        buf_pair = _buf_pair()
        plan_pair = make_verify_plan_pair(
            slot_indices=[],
            positions=[],
            prev_slot_indices=[],
            capacity=4,
            device=_DEVICE,
        )
        # Use the low-level wrapper with explicit logs so we can observe the cross-call counter accumulation.
        cuda_log, ref_log = make_log_pair(device=_DEVICE)
        for _ in range(3):
            _run_both_verify_no_rkv(
                buf_pair=buf_pair,
                plan_pair=plan_pair,
                cuda_log=cuda_log,
                ref_log=ref_log,
            )

        assert int(cuda_log.kernel_run_counter[0].item()) == 3

    def test_slot_run_counter_per_entry(self) -> None:
        """``slot_run_counter`` accumulates ``verify_num_valid`` entries per call."""
        cuda_buf, ref_buf = _buf_pair()
        slot_indices = [1, 2, 3, 4]
        tokens = [10, 11, 12, 13]
        positions = [0, 1, 2, 3]
        stamp_clean_chain(
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            tokens=tokens,
            positions=positions,
            slot_indices=slot_indices,
        )

        plan_pair = make_verify_plan_pair(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=[-1, 1, 2, 3],
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(buf_pair=(cuda_buf, ref_buf), plan_pair=plan_pair)

        assert int(cuda_log.slot_run_counter[0].item()) == 4

    def test_grid_stride_processes_entries_beyond_grid_size(self) -> None:
        """``verify_num_valid`` exceeding the persistent grid thread count is fully processed via grid-stride."""
        n_active = 40000
        cuda_buf, ref_buf = make_canary_buf_pair(
            num_slots=n_active + 1, slot_stride_bytes=32, device=_DEVICE
        )
        slot_indices = list(range(1, n_active + 1))
        tokens = [i + 10 for i in range(n_active)]
        positions = list(range(n_active))
        stamp_clean_chain(
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            tokens=tokens,
            positions=positions,
            slot_indices=slot_indices,
        )

        plan_pair = make_verify_plan_pair(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=[-1] + slot_indices[:-1],
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(buf_pair=(cuda_buf, ref_buf), plan_pair=plan_pair)

        assert int(cuda_log.slot_run_counter[0].item()) == n_active
        assert int(cuda_log.kernel_run_counter[0].item()) == 1

    def test_replay_does_not_double_count_run_counters(self) -> None:
        """Two consecutive runs on same plan: slot_run_counter += 2N, kernel_run_counter += 2."""
        cuda_buf, ref_buf = _buf_pair()
        slot_indices = [1, 2, 3]
        tokens = [10, 20, 30]
        positions = [0, 1, 2]
        stamp_clean_chain(
            cuda_buf=cuda_buf,
            ref_buf=ref_buf,
            tokens=tokens,
            positions=positions,
            slot_indices=slot_indices,
        )
        plan_pair = make_verify_plan_pair(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=[-1, 1, 2],
            device=_DEVICE,
        )
        cuda_log, ref_log = make_log_pair(device=_DEVICE)
        for _ in range(2):
            _run_both_verify_no_rkv(
                buf_pair=(cuda_buf, ref_buf),
                plan_pair=plan_pair,
                cuda_log=cuda_log,
                ref_log=ref_log,
            )

        assert int(cuda_log.slot_run_counter[0].item()) == 2 * len(slot_indices)
        assert int(cuda_log.kernel_run_counter[0].item()) == 2

    def test_slot_run_counter_delta_equals_active_entries_across_random_plans(
        self,
    ) -> None:
        random.seed(0)
        num_slots = 32
        cuda_buf = make_canary_buf(
            num_slots=num_slots, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        buf_pair = (cuda_buf, ref_buf)

        for slot_idx in range(num_slots):
            _stamp_head(
                buf_pair,
                slot_idx=slot_idx,
                token=slot_idx + 10,
                position=slot_idx,
            )

        cuda_log, ref_log = make_log_pair(device=_DEVICE)
        for _ in range(50):
            bs = random.randint(1, 16)
            entries_per_req = random.randint(1, 8)
            n_entries = bs * entries_per_req
            if n_entries > num_slots:
                n_entries = num_slots
            slot_indices = random.sample(range(num_slots), n_entries)
            positions = [random.randint(0, 99) for _ in range(n_entries)]
            prev_slot_indices = [-1] * n_entries

            plan_pair = make_verify_plan_pair(
                slot_indices=slot_indices,
                positions=positions,
                prev_slot_indices=prev_slot_indices,
                device=_DEVICE,
            )

            before = int(cuda_log.slot_run_counter[0].item())
            _run_both_verify_no_rkv(
                buf_pair=buf_pair,
                plan_pair=plan_pair,
                cuda_log=cuda_log,
                ref_log=ref_log,
                assert_equal=False,
            )
            after = int(cuda_log.slot_run_counter[0].item())
            assert after - before == n_entries

    def test_kernel_run_counter_per_call_invariant_50_calls(self) -> None:
        buf_pair = _buf_pair(num_slots=8)
        _stamp_head(buf_pair, slot_idx=1)

        plan_pair = _plan_pair_single(slot_idx=1, position=0)
        cuda_log, ref_log = make_log_pair(device=_DEVICE)
        for n in range(1, 51):
            _run_both_verify_no_rkv(
                buf_pair=buf_pair,
                plan_pair=plan_pair,
                cuda_log=cuda_log,
                ref_log=ref_log,
                assert_equal=False,
            )
            assert int(cuda_log.kernel_run_counter[0].item()) == n


class TestViolationRing:
    def test_violation_ring_fill_once_first_row(self) -> None:
        """First violation lands at ring[0]; subsequent violations advance ``violation_write_index``."""
        buf_pair = _buf_pair()
        # 3 chain-head entries with stored values that all yield POSITION mismatch (positions all 99).
        for slot_idx in (1, 2, 3):
            _stamp_head(buf_pair, slot_idx=slot_idx, token=1)

        plan_pair = make_verify_plan_pair(
            slot_indices=[1, 2, 3],
            positions=[99, 99, 99],
            prev_slot_indices=[-1, -1, -1],
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair, plan_pair=plan_pair, assert_equal=False
        )

        assert _n_violations(cuda_log) == 3
        # row 0 is filled (slot 1 → POSITION bit).
        assert _fail_bits(cuda_log) & consts.FailReason.VERIFY_POSITION_MISMATCH

    def test_violation_ring_overflow_counter_still_increments(self) -> None:
        """Ring capacity exceeded → rows beyond are dropped but ``write_index`` still grows."""
        buf_pair = _buf_pair()
        n_violations = 10
        slot_indices = list(range(1, n_violations + 1))
        for slot_idx in slot_indices:
            _stamp_head(buf_pair, slot_idx=slot_idx, token=1)

        plan_pair = make_verify_plan_pair(
            slot_indices=slot_indices,
            positions=[99] * n_violations,
            prev_slot_indices=[-1] * n_violations,
            device=_DEVICE,
        )
        cuda_log, ref_log = make_log_pair(capacity=4, device=_DEVICE)
        _run_both_verify_no_rkv(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            cuda_log=cuda_log,
            ref_log=ref_log,
            assert_equal=False,
        )

        assert _n_violations(cuda_log) == n_violations
        assert _n_violations(ref_log) == n_violations
        # Atomic-order may permute ring contents under overflow; only the write_index counter is
        # byte-equal — we relax the ring-contents check here.
        assert torch.equal(cuda_log.write_index, ref_log.write_index)

    def test_kernel_kind_stamped_into_row(self) -> None:
        """Different ``CanaryLaunchTag`` values → violation row.kernel_kind reflects each."""
        buf_pair = _buf_pair()
        _stamp_head(buf_pair, slot_idx=1, token=1)

        for tag in (CanaryLaunchTag.HEAD_K_FULL, CanaryLaunchTag.SWEEP_V_SWA):
            plan_pair = _plan_pair_single(slot_idx=1, position=99)
            cuda_log, _ = run_verify_diff(
                buf_pair=buf_pair, plan_pair=plan_pair, kernel_kind=tag
            )
            kk = int(cuda_log.ring[0, consts.VIOLATION_FIELD_KERNEL_KIND].item())
            assert kk == int(tag)

    def test_violation_ring_row_byte_layout_hardcoded(self) -> None:
        buf_pair = _buf_pair()
        anchor_hash_signed = chain_anchor_signed()
        _stamp_head(buf_pair, slot_idx=5, token=33)

        plan_pair = _plan_pair_single(slot_idx=5, position=99)
        cuda_log, ref_log = make_log_pair(capacity=4, device=_DEVICE)
        _run_both_verify_no_rkv(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            cuda_log=cuda_log,
            ref_log=ref_log,
        )

        assert _n_violations(cuda_log) == 1

        # splitmix64(consts.CANARY_CHAIN_ANCHOR) = 0xde7fae23a9a1b716; signed = -2414019407054260458.
        # Slot 5 was stamped with position=0 (stored); plan claims position=99 → POSITION mismatch.
        # stored_chain_hash == expected_chain_hash (both splitmix64(ANCHOR)) → no CHAIN_HASH bit.
        # plan has no populated verify_expected_tokens → entries read -1 sentinel, the
        # verify kernel skips the token check (no TOKEN_MISMATCH bit) but the row's
        # expected_token field reflects the gathered sentinel value.
        # expected_aux = expected_chain_hash = splitmix64(ANCHOR) signed (same value as stored_chain_hash).
        kernel_kind_val = int(CanaryLaunchTag.HEAD_K_FULL)
        slot_idx_val = 5
        position_val = 0
        stored_token_val = 33
        expected_token_val = -1
        stored_chain_hash_val = anchor_hash_signed
        expected_aux_val = anchor_hash_signed
        fail_reason_bits_val = consts.FailReason.VERIFY_POSITION_MISMATCH

        expected_bytes = struct.pack(
            "<8q",
            kernel_kind_val,
            slot_idx_val,
            position_val,
            stored_token_val,
            expected_token_val,
            stored_chain_hash_val,
            expected_aux_val,
            fail_reason_bits_val,
        )

        actual_bytes = cuda_log.ring[0].cpu().numpy().tobytes()

        if actual_bytes != expected_bytes:
            expected_fields = struct.unpack("<8q", expected_bytes)
            actual_fields = struct.unpack("<8q", actual_bytes)
            field_names = [
                "kernel_kind",
                "slot_idx",
                "position",
                "stored_token",
                "expected_token",
                "stored_chain_hash",
                "expected_aux",
                "fail_reason_bits",
            ]
            mismatches = [
                f"  [{i}] {name}: expected {e} ({e:#x}) got {a} ({a:#x})"
                for i, (name, e, a) in enumerate(
                    zip(field_names, expected_fields, actual_fields)
                )
                if e != a
            ]
            raise AssertionError(
                "violation_ring row binary layout mismatch:\n" + "\n".join(mismatches)
            )

    def test_violation_ring_atomic_with_many_violations(self) -> None:
        """50 simultaneously-corrupted entries → write_index == 50 (no atomicity loss)."""
        n = 50
        cuda_buf = make_canary_buf(
            num_slots=n + 4, slot_stride_bytes=32, device=_DEVICE
        )
        ref_buf = cuda_buf.clone()
        slot_indices = list(range(1, n + 1))
        positions = [0] * n

        plan_pair = make_verify_plan_pair(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=[-1] * n,
            capacity=n,
            device=_DEVICE,
        )
        cuda_log, ref_log = make_log_pair(capacity=128, device=_DEVICE)
        _run_both_verify_no_rkv(
            buf_pair=(cuda_buf, ref_buf),
            plan_pair=plan_pair,
            cuda_log=cuda_log,
            ref_log=ref_log,
            assert_equal=False,
        )

        assert _n_violations(cuda_log) == n
        assert _n_violations(ref_log) == n

    def test_violation_rows_have_valid_kernel_kind_and_slot(self) -> None:
        """Each violation row's kernel_kind matches the launch tag; slot_idx is one of the plan slots."""
        buf_pair = _buf_pair()
        slot_indices = [1, 2, 3, 4]
        positions = [0, 1, 2, 3]
        plan_pair = make_verify_plan_pair(
            slot_indices=slot_indices,
            positions=positions,
            prev_slot_indices=[-1] * 4,
            device=_DEVICE,
        )
        launch_tag = CanaryLaunchTag.HEAD_V_SWA
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair, plan_pair=plan_pair, kernel_kind=launch_tag
        )
        n_violations = _n_violations(cuda_log)
        plan_slot_set = set(slot_indices)
        for row in range(n_violations):
            kind = int(cuda_log.ring[row, consts.VIOLATION_FIELD_KERNEL_KIND].item())
            assert kind == int(
                launch_tag
            ), f"row {row} kind {kind} != {int(launch_tag)}"
            slot = int(cuda_log.ring[row, 1].item())
            assert slot in plan_slot_set, f"row {row} slot {slot} not in plan"

    def test_clear_resets_ring_and_write_index_zero(self) -> None:
        buf_pair = _buf_pair(num_slots=8)
        for slot_idx in range(1, 4):
            _stamp_head(buf_pair, slot_idx=slot_idx, token=1, position=99)

        plan_pair = make_verify_plan_pair(
            slot_indices=[1, 2, 3],
            positions=[0, 0, 0],
            prev_slot_indices=[-1, -1, -1],
            device=_DEVICE,
        )
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair, plan_pair=plan_pair, assert_equal=False
        )

        assert _n_violations(cuda_log) > 0

        cuda_log_fresh = FakeViolationLog.allocate(device=_DEVICE)
        assert _n_violations(cuda_log_fresh) == 0
        assert torch.all(cuda_log_fresh.ring == 0).item()


class TestBoundarySweep:
    @pytest.mark.parametrize(
        "token_val",
        [0, 1, 0xFFFFFFFF, -1, 0x80000000, 0x7FFFFFFF],
    )
    def test_token_boundary_byte_equal_sweep(self, token_val: int) -> None:
        """Sweep token boundary values; assert CUDA vs ref state byte-equal."""
        _run_verify_single_slot_byte_equal(
            _VerifySingleSlotInput(
                token=token_val, stored_prev_hash_signed=chain_anchor_signed()
            )
        )

    @pytest.mark.parametrize(
        "position_val",
        [0, 1, 127, 128, 129, 0x7FFFFFFF],
    )
    def test_position_boundary_byte_equal_sweep(self, position_val: int) -> None:
        """Sweep position boundary values; assert CUDA vs ref state byte-equal."""
        _run_verify_single_slot_byte_equal(
            _VerifySingleSlotInput(
                position=position_val, stored_prev_hash_signed=chain_anchor_signed()
            )
        )

    @pytest.mark.parametrize(
        "prev_hash_val",
        [
            pytest.param(0, id="zero"),
            pytest.param(None, id="splitmix64_of_zero"),
            pytest.param(0xFFFFFFFFFFFFFFFF, id="all_ones"),
            pytest.param(0x8000000000000000, id="sign_bit"),
        ],
    )
    def test_prev_hash_boundary_byte_equal_sweep(
        self, prev_hash_val: int | None
    ) -> None:
        """Sweep prev_hash boundary values; assert CUDA vs ref state byte-equal."""
        if prev_hash_val is None:
            prev_hash_val = splitmix64(0)
        _run_verify_single_slot_byte_equal(
            _VerifySingleSlotInput(
                stored_prev_hash_signed=to_signed_int64(prev_hash_val)
            )
        )

    @pytest.mark.parametrize(
        "stored_rkv_val",
        [0, 1, 0xFFFFFFFFFFFFFFFF, 0x8000000000000000],
    )
    def test_real_kv_hash_boundary_byte_equal_sweep(self, stored_rkv_val: int) -> None:
        """Sweep real_kv_hash boundary values; assert CUDA vs ref state byte-equal."""
        sources_cuda = make_real_kv_sources(count=1, device=_DEVICE)
        _run_verify_single_slot_byte_equal(
            _VerifySingleSlotInput(
                stored_prev_hash_signed=chain_anchor_signed(),
                stored_real_kv_hash_signed=to_signed_int64(stored_rkv_val),
                real_kv_sources=sources_cuda,
                real_kv_hash_mode=consts.RealKvHashMode.PARTIAL,
            )
        )


class TestVerifyExpectedInputIds:
    """Cover the new verify-time token-id check via VerifyPlan.verify_expected_tokens."""

    def _run(
        self,
        *,
        stored_token: int,
        expected_input_id: int,
        check_verify_expected_token: bool = True,
    ) -> FakeViolationLog:
        buf_pair = _buf_pair()
        _stamp_head(buf_pair, slot_idx=1, token=stored_token)
        plan_pair = _plan_pair_single(
            slot_idx=1, position=0, expected_input_id=expected_input_id
        )
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            check_verify_expected_token=check_verify_expected_token,
        )
        return cuda_log

    def test_sentinel_skips_token_check(self) -> None:
        """``expected_input_id == -1`` must not fire even if stored token differs."""
        log = self._run(stored_token=42, expected_input_id=-1)
        assert _n_violations(log) == 0

    def test_match_records_no_violation(self) -> None:
        """Matching stored vs expected token: zero violations."""
        log = self._run(stored_token=42, expected_input_id=42)
        assert _n_violations(log) == 0

    def test_mismatch_fires_verify_token_bit(self) -> None:
        """Mismatch: kVerifyTokenMismatch bit set; expected_token field populated."""
        log = self._run(stored_token=42, expected_input_id=99)
        assert _n_violations(log) == 1
        row = log.ring[0].tolist()
        assert_only_bits_set(
            int(row[consts.VIOLATION_FIELD_FAIL_REASON_BITS]),
            int(consts.FailReason.VERIFY_TOKEN_MISMATCH),
        )
        assert int(row[consts.VIOLATION_FIELD_STORED_TOKEN]) == 42
        assert int(row[consts.VIOLATION_FIELD_EXPECTED_TOKEN]) == 99

    def test_check_disabled_never_raises_token_mismatch(self) -> None:
        """check_verify_expected_token=False: token mismatch is silently ignored byte-equal CUDA vs ref."""
        log = self._run(
            stored_token=42,
            expected_input_id=99,
            check_verify_expected_token=False,
        )
        assert _n_violations(log) == 0

    def test_check_disabled_does_not_read_expected_tokens_tensor(self) -> None:
        """check_verify_expected_token=False with garbage expected_input_id: kernel must not deref the tensor."""
        buf_pair = _buf_pair()
        _stamp_head(buf_pair, slot_idx=1)
        garbage_value = (1 << 63) - 1
        plan_pair = _plan_pair_single(
            slot_idx=1, position=0, expected_input_id=garbage_value
        )
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            check_verify_expected_token=False,
        )
        assert _n_violations(cuda_log) == 0

    def test_token_and_position_both_mismatch_set_both_bits(self) -> None:
        """token mismatch + position mismatch: both fail bits set; expected_token row carries the gathered id."""
        buf_pair = _buf_pair()
        slot_idx = 5
        _stamp_head(buf_pair, slot_idx=slot_idx, position=10)
        plan_pair = _plan_pair_single(
            slot_idx=slot_idx, position=99, expected_input_id=123
        )
        cuda_log, _ = run_verify_diff(
            buf_pair=buf_pair,
            plan_pair=plan_pair,
            check_verify_expected_token=True,
        )
        assert _n_violations(cuda_log) == 1
        row = cuda_log.ring[0].tolist()
        expected_bits = int(consts.FailReason.VERIFY_TOKEN_MISMATCH) | int(
            consts.FailReason.VERIFY_POSITION_MISMATCH
        )
        assert_only_bits_set(
            int(row[consts.VIOLATION_FIELD_FAIL_REASON_BITS]), expected_bits
        )
        assert int(row[consts.VIOLATION_FIELD_STORED_TOKEN]) == 42
        assert int(row[consts.VIOLATION_FIELD_EXPECTED_TOKEN]) == 123


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
