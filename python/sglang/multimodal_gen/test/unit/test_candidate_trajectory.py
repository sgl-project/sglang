# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the phase-1 candidate-trajectory contract
(:mod:`sglang.multimodal_gen.runtime.pipelines_core.candidates`).

Exercises real ``torch`` tensor ops -- including on the Apple Silicon MPS
backend when available -- since, unlike the step-reuse contract, candidate
reduction and RNG-stream independence are inherently tensor operations and
are worth validating against a real accelerator rather than only plain
Python values.
"""

import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.candidates import (
    CandidateContractError,
    CandidateGroup,
    CandidateTrajectorySpec,
    build_candidate_group,
    derive_candidate_generators,
    get_candidate_reducer,
    reduce_candidates,
    register_candidate_reducer,
    validate_candidate_admission,
)


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _device()


class TestCandidateTrajectorySpecValidation:
    def test_default_is_count_one_reducer_none(self):
        spec = CandidateTrajectorySpec()
        assert spec.count == 1
        assert spec.reducer == "none"

    def test_rejects_count_below_one(self):
        with pytest.raises(ValueError):
            CandidateTrajectorySpec(count=0)

    def test_count_one_must_use_reducer_none(self):
        with pytest.raises(ValueError):
            CandidateTrajectorySpec(count=1, reducer="mean")

    def test_count_above_one_requires_real_reducer(self):
        with pytest.raises(ValueError):
            CandidateTrajectorySpec(count=4, reducer="none")

    def test_rejects_unknown_seed_policy(self):
        with pytest.raises(ValueError):
            CandidateTrajectorySpec(count=2, reducer="mean", seed_policy="bogus")

    def test_rejects_unregistered_reducer_eagerly(self):
        with pytest.raises(CandidateContractError):
            CandidateTrajectorySpec(count=2, reducer="does_not_exist")

    def test_valid_multi_candidate_spec(self):
        spec = CandidateTrajectorySpec(count=4, reducer="mean")
        assert spec.count == 4


class TestCandidateGroupIdentity:
    def test_build_candidate_group_from_spec(self):
        spec = CandidateTrajectorySpec(count=3, reducer="mean")
        group = build_candidate_group("req-1", spec)
        assert group.request_id == "req-1"
        assert group.candidate_ids == (0, 1, 2)
        assert group.public_output == "action"

    def test_rejects_empty_candidate_ids(self):
        with pytest.raises(ValueError):
            CandidateGroup(request_id="req-1", candidate_ids=())

    def test_rejects_duplicate_candidate_ids(self):
        with pytest.raises(ValueError):
            CandidateGroup(request_id="req-1", candidate_ids=(0, 0, 1))

    def test_rejects_unsorted_candidate_ids(self):
        with pytest.raises(ValueError):
            CandidateGroup(request_id="req-1", candidate_ids=(1, 0, 2))

    def test_unrelated_requests_have_independent_id_spaces(self):
        # Requirement 4: candidate IDs remain stable when unrelated requests
        # are co-batched -- i.e. two requests' candidate_ids never need to
        # be disambiguated by anything other than (request_id, candidate_id).
        spec = CandidateTrajectorySpec(count=2, reducer="mean")
        group_a = build_candidate_group("req-a", spec)
        group_b = build_candidate_group("req-b", spec)
        assert group_a.candidate_ids == group_b.candidate_ids == (0, 1)
        assert group_a.request_id != group_b.request_id


class TestCountOneIsIdentityPath:
    def test_count_one_reduction_returns_the_single_candidate_unchanged(self):
        # Requirement 1: count=1 must be identical to the existing path.
        spec = CandidateTrajectorySpec()  # count=1, reducer="none"
        candidate = torch.randn(4, 8, device=DEVICE)
        batch = candidate.unsqueeze(0)  # shape (1, 4, 8)
        result = reduce_candidates(batch, spec)
        assert result.shape == candidate.shape
        assert torch.equal(result, candidate)

    def test_reducer_none_rejects_more_than_one_candidate(self):
        fn = get_candidate_reducer("none")
        with pytest.raises(CandidateContractError):
            fn(torch.randn(2, 3, device=DEVICE))


class TestReducerRegistry:
    def test_mean_reducer_matches_standalone_mean(self):
        # Requirement 3: the reduced action matches a standalone reducer
        # applied to the same captured candidate tensors.
        spec = CandidateTrajectorySpec(count=5, reducer="mean")
        candidates = torch.randn(5, 16, 3, device=DEVICE)
        reduced = reduce_candidates(candidates, spec)
        expected = candidates.mean(dim=0)
        assert torch.allclose(reduced, expected)
        assert reduced.device.type == DEVICE.type

    def test_register_custom_reducer(self):
        @register_candidate_reducer("test_max_for_candidate_trajectory")
        def _max_reducer(candidates: torch.Tensor) -> torch.Tensor:
            return candidates.max(dim=0).values

        spec = CandidateTrajectorySpec(
            count=3, reducer="test_max_for_candidate_trajectory"
        )
        candidates = torch.tensor([[1.0, 5.0], [3.0, 2.0], [0.0, 9.0]], device=DEVICE)
        result = reduce_candidates(candidates, spec)
        assert torch.equal(result, torch.tensor([3.0, 9.0], device=DEVICE))

    def test_cannot_register_duplicate_reducer_name(self):
        @register_candidate_reducer("test_dup_for_candidate_trajectory")
        def _first(candidates: torch.Tensor) -> torch.Tensor:
            return candidates.sum(dim=0)

        with pytest.raises(CandidateContractError):

            @register_candidate_reducer("test_dup_for_candidate_trajectory")
            def _second(candidates: torch.Tensor) -> torch.Tensor:
                return candidates.sum(dim=0)

    def test_unknown_reducer_lookup_raises(self):
        with pytest.raises(CandidateContractError):
            get_candidate_reducer("nonexistent_reducer_xyz")


class TestPartialGroupNeverReduces:
    def test_reduce_candidates_rejects_shape_mismatch(self):
        # Requirement 7: a cancelled/failed candidate must not silently
        # produce a partial aggregate.
        spec = CandidateTrajectorySpec(count=4, reducer="mean")
        partial = torch.randn(3, 8, device=DEVICE)  # one candidate missing
        with pytest.raises(CandidateContractError):
            reduce_candidates(partial, spec)


class TestAdmission:
    def test_group_fitting_batch_is_admitted(self):
        spec = CandidateTrajectorySpec(count=4, reducer="mean")
        validate_candidate_admission(spec, max_execution_batch=8)  # no raise

    def test_group_exceeding_batch_is_rejected_atomically(self):
        spec = CandidateTrajectorySpec(count=10, reducer="mean")
        with pytest.raises(CandidateContractError):
            validate_candidate_admission(spec, max_execution_batch=8)


class TestRngStreamIndependenceAndEquivalence:
    def test_per_candidate_streams_are_independent(self):
        spec = CandidateTrajectorySpec(count=4, reducer="mean")
        generators = derive_candidate_generators(
            base_seed=1234, spec=spec, device=DEVICE
        )
        draws = [torch.randn(16, generator=g, device=DEVICE) for g in generators]
        for i in range(len(draws)):
            for j in range(i + 1, len(draws)):
                assert not torch.equal(draws[i], draws[j])

    def test_seed_derivation_is_deterministic(self):
        spec = CandidateTrajectorySpec(count=4, reducer="mean")
        gens_a = derive_candidate_generators(base_seed=99, spec=spec, device=DEVICE)
        gens_b = derive_candidate_generators(base_seed=99, spec=spec, device=DEVICE)
        draws_a = [torch.randn(8, generator=g, device=DEVICE) for g in gens_a]
        draws_b = [torch.randn(8, generator=g, device=DEVICE) for g in gens_b]
        for a, b in zip(draws_a, draws_b):
            assert torch.equal(a, b)

    def test_sequential_and_batched_execution_produce_identical_candidates(self):
        # Requirement 2: fixed per-candidate seeds produce the same ordered
        # candidate tensors in sequential and batched modes.
        spec = CandidateTrajectorySpec(count=4, reducer="mean")
        base_seed = 777

        # "Sequential" mode: draw each candidate's tensor one at a time.
        seq_generators = derive_candidate_generators(base_seed, spec, device=DEVICE)
        sequential = torch.stack(
            [torch.randn(32, generator=g, device=DEVICE) for g in seq_generators]
        )

        # "Batched" mode: same derivation, but the candidate axis is filled
        # into one pre-allocated batch tensor row by row (as a batched
        # denoising path would), still consuming one generator per row.
        batch_generators = derive_candidate_generators(base_seed, spec, device=DEVICE)
        batched = torch.empty(spec.count, 32, device=DEVICE)
        for i, g in enumerate(batch_generators):
            batched[i] = torch.randn(32, generator=g, device=DEVICE)

        assert torch.equal(sequential, batched)

    def test_shared_seed_policy_gives_identical_streams(self):
        spec = CandidateTrajectorySpec(count=3, reducer="mean", seed_policy="shared")
        generators = derive_candidate_generators(base_seed=42, spec=spec, device=DEVICE)
        draws = [torch.randn(8, generator=g, device=DEVICE) for g in generators]
        assert torch.equal(draws[0], draws[1])
        assert torch.equal(draws[1], draws[2])

    def test_different_base_seeds_do_not_collide(self):
        spec = CandidateTrajectorySpec(count=2, reducer="mean")
        gens_1 = derive_candidate_generators(base_seed=1, spec=spec, device=DEVICE)
        gens_2 = derive_candidate_generators(base_seed=2, spec=spec, device=DEVICE)
        d1 = torch.randn(8, generator=gens_1[0], device=DEVICE)
        d2 = torch.randn(8, generator=gens_2[0], device=DEVICE)
        assert not torch.equal(d1, d2)


class TestEndToEndCandidateFlow:
    def test_full_flow_matches_manual_reference(self):
        # Requirements 1-3 combined into one end-to-end check, run on the
        # active DEVICE (MPS when available).
        spec = CandidateTrajectorySpec(count=8, reducer="mean")
        group = build_candidate_group("req-e2e", spec)
        validate_candidate_admission(spec, max_execution_batch=16)

        generators = derive_candidate_generators(
            base_seed=2024, spec=spec, device=DEVICE
        )
        candidate_shape = (4, 6)
        candidates = torch.stack(
            [
                torch.randn(*candidate_shape, generator=g, device=DEVICE)
                for g in generators
            ]
        )
        assert candidates.shape[0] == len(group.candidate_ids)

        reduced = reduce_candidates(candidates, spec)
        assert torch.allclose(reduced, candidates.mean(dim=0))
        assert reduced.shape == candidate_shape


class TestMultiPromptGeneratorAggregation:
    """Covers the exact aggregation pattern
    InputValidationStage._generate_seeds uses when batch.candidate_spec is
    set: for each prompt's independently-derived base_seed, flatten that
    prompt's derive_candidate_generators(...) output into one per-request
    generator list, prompt-major / candidate-minor order.
    """

    @staticmethod
    def _aggregate(base_seeds, spec, device):
        # Mirrors InputValidationStage._generate_seeds' candidate_spec branch.
        return [
            gen
            for base_seed in base_seeds
            for gen in derive_candidate_generators(base_seed, spec, device=device)
        ]

    def test_flattened_order_is_prompt_major_candidate_minor(self):
        spec = CandidateTrajectorySpec(count=3, reducer="mean")
        base_seeds = [100, 200]
        generators = self._aggregate(base_seeds, spec, DEVICE)
        assert len(generators) == 6  # 2 prompts * 3 candidates

        draws = [torch.randn(4, generator=g, device=DEVICE) for g in generators]
        # Same-prompt candidates come from consecutive derive_candidate_generators
        # calls; re-deriving independently for each prompt must reproduce them.
        expected_prompt0 = [
            torch.randn(4, generator=g, device=DEVICE)
            for g in derive_candidate_generators(100, spec, device=DEVICE)
        ]
        expected_prompt1 = [
            torch.randn(4, generator=g, device=DEVICE)
            for g in derive_candidate_generators(200, spec, device=DEVICE)
        ]
        for a, b in zip(draws[0:3], expected_prompt0):
            assert torch.equal(a, b)
        for a, b in zip(draws[3:6], expected_prompt1):
            assert torch.equal(a, b)

    def test_different_prompts_never_collide(self):
        spec = CandidateTrajectorySpec(count=2, reducer="mean")
        generators = self._aggregate([1, 2, 3], spec, DEVICE)
        draws = [torch.randn(4, generator=g, device=DEVICE) for g in generators]
        for i in range(len(draws)):
            for j in range(i + 1, len(draws)):
                assert not torch.equal(draws[i], draws[j])

    def test_single_prompt_matches_bare_derive_candidate_generators(self):
        spec = CandidateTrajectorySpec(count=4, reducer="mean")
        generators = self._aggregate([777], spec, DEVICE)
        expected = derive_candidate_generators(777, spec, device=DEVICE)
        assert len(generators) == len(expected)
        for g, e in zip(generators, expected):
            d1 = torch.randn(4, generator=g, device=DEVICE)
            d2 = torch.randn(4, generator=e, device=DEVICE)
            assert torch.equal(d1, d2)
