"""Unit tests for the DSpark ASD acceptance adapter.

Covers the --speculative-dspark-asd-config-path contract (strict-by-default
resolution, config-file parsing, optional-package degradation) and the
acceptance-accounting math of accept_or_native with the ASD rule replaced by
deterministic stubs, so CI verifies the charging/cap/bonus logic without the
optional research package installed.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.dspark_components import asd_dspark as asd_dspark_module
from sglang.srt.speculative.dspark_components.asd_dspark import (
    DSparkASDAdapter,
    DSparkASDSettings,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _server_args(**overrides):
    fields = {"speculative_dspark_asd_config_path": None}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _config_file(payload) -> str:
    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    with handle:
        handle.write(json.dumps(payload))
    return handle.name


# Canonical ASD decode configuration (see the ASD repository's
# asd/reproduce/dspark/asd_config.py for the strict key set).
_VALID_CONFIG = {
    "B": 8.0,
    "g": 2.0,
    "m": 2,
    "value_scheme": "normalized_suffix",
    "block_size": 16,
}


class _StubConfig:
    """Duck-typed DSparkASDConfig stand-in (risk_budget + hooks)."""

    risk_budget: float = 8.0

    def validate_block_size(self, observed: int) -> None:
        pass

    def fingerprint(self) -> str:
        return "stub-fingerprint"


class _StubDecision:
    """Duck-typed choose_prefix_batch result: accepted/mismatched/regrets."""

    def __init__(self, accepted, mismatched, regrets):
        self.accepted = accepted
        self.mismatched = mismatched
        self.regrets = regrets


class TestStrictByDefault(CustomTestCase):
    def test_unset_path_resolves_strict(self):
        settings = DSparkASDSettings.from_server_args(_server_args())
        self.assertFalse(settings.active)
        self.assertIsNone(settings.config)
        self.assertIsNone(settings.fingerprint)


class TestConfigParsing(CustomTestCase):
    def test_invalid_json_raises_value_error(self):
        path = Path(tempfile.mkstemp(suffix=".json")[1])
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            DSparkASDSettings.from_server_args(
                _server_args(speculative_dspark_asd_config_path=str(path))
            )

    def test_non_object_json_raises_value_error(self):
        with self.assertRaises(ValueError):
            DSparkASDSettings.from_server_args(
                _server_args(speculative_dspark_asd_config_path=_config_file([1, 2]))
            )


class TestOptionalPackageDegradation(CustomTestCase):
    def test_missing_package_raises_clear_runtime_error(self):
        if asd_dspark_module.DSparkASDConfig is not None:
            self.skipTest("ASD package installed; degradation path not exercised")
        with self.assertRaises(RuntimeError) as ctx:
            DSparkASDSettings.from_server_args(
                _server_args(
                    speculative_dspark_asd_config_path=_config_file(_VALID_CONFIG)
                )
            )
        self.assertIn("ASD package", str(ctx.exception))


class TestSettingsWithPackage(CustomTestCase):
    def test_valid_config_resolves_active(self):
        if asd_dspark_module.DSparkASDConfig is None:
            self.skipTest("ASD package not installed")
        settings = DSparkASDSettings.from_server_args(
            _server_args(speculative_dspark_asd_config_path=_config_file(_VALID_CONFIG))
        )
        self.assertTrue(settings.active)
        self.assertIsNotNone(settings.fingerprint)


class TestAdapterInvariants(CustomTestCase):
    """ASD invariants are enforced only when ASD acceptance is active."""

    def test_nonpositive_gamma_raises_when_active(self):
        with self.assertRaisesRegex(ValueError, "gamma must be positive"):
            DSparkASDAdapter(
                settings=DSparkASDSettings(config=_StubConfig()),
                gamma=0,
                verify_num_draft_tokens=1,
                device="cpu",
            )

    def test_verify_width_mismatch_raises_when_active(self):
        with self.assertRaisesRegex(ValueError, "gamma \\+ 1"):
            DSparkASDAdapter(
                settings=DSparkASDSettings(config=_StubConfig()),
                gamma=4,
                verify_num_draft_tokens=4,
                device="cpu",
            )

    def test_inactive_adapter_skips_asd_invariants(self):
        # A native (strict) DSpark start must not be coupled to ASD
        # assumptions: bogus gamma/width are tolerated when inactive.
        DSparkASDAdapter(
            settings=DSparkASDSettings(),
            gamma=0,
            verify_num_draft_tokens=7,
            device="cpu",
        )

    def test_inactive_adapter_skips_runtime_guard(self):
        adapter = DSparkASDAdapter(
            settings=DSparkASDSettings(),
            gamma=4,
            verify_num_draft_tokens=5,
            device="cpu",
        )
        # Strict settings never restrict the runtime path.
        adapter.require_supported_runtime(
            disable_cuda_graph=False,
            disable_overlap_schedule=False,
            ragged_verify_mode="compact",
            simulate_accept_length=0.0,
        )
        adapter.require_unfolded_accept()  # no-op when inactive


class TestAcceptanceAccounting(CustomTestCase):
    """Verify accept_or_native's charging/cap/bonus math with the rule stubbed.

    choose_prefix_batch and DSparkASDConfig come from the optional asd
    package; deterministic stand-ins let CI exercise the code that consumes
    the rule's decision (budget charging, cap trimming, bonus selection)
    without the package.
    """

    def _make_bound_adapter(self) -> DSparkASDAdapter:
        adapter = DSparkASDAdapter(
            settings=DSparkASDSettings(config=_StubConfig()),
            gamma=2,
            verify_num_draft_tokens=3,
            device="cpu",
        )
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid="r0"), SimpleNamespace(rid="r1")],
            req_to_token_pool=SimpleNamespace(_alloc_size=4),
            req_pool_indices_cpu=[1, 2],
        )
        adapter.bind_batch(batch)
        return adapter

    def _logits_with_tops(self, tops) -> torch.Tensor:
        # [bs*width, vocab] logits whose per-row argmax is given by tops.
        logits = torch.zeros(len(tops), 5, dtype=torch.float32)
        for row, top in enumerate(tops):
            logits[row, top] = 1.0
        return logits

    def test_budget_charging_cap_trim_and_bonus(self):
        adapter = self._make_bound_adapter()
        # candidates [bs, width]: anchor token + gamma drafts.
        candidates = torch.tensor([[0, 1, 2], [0, 3, 4]], dtype=torch.int64)
        # r0 target tops = [2, 1, 3]; r1 target tops = [4, 0, 3].
        logits = self._logits_with_tops([2, 1, 3, 4, 0, 3])
        # The rule relaxes r0's position 0 (regret 1.5) and proposes two
        # accepted drafts for both requests; r1's verify budget (cutoff
        # verify_len 2 -> max 1 verifiable draft) trims its second draft.
        decision = _StubDecision(
            accepted=torch.tensor([2, 2], dtype=torch.int32),
            mismatched=torch.tensor([[True, False], [False, False]]),
            regrets=torch.tensor([[1.5, 2.0], [0.0, 0.0]]),
        )
        with patch.object(
            asd_dspark_module, "choose_prefix_batch", lambda **kwargs: decision
        ):
            accepted, bonus, cap_trim = adapter.accept_or_native(
                candidates=candidates,
                target_logits=logits,
                cutoff_verify_lens=torch.tensor([3, 2], dtype=torch.int32),
                req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
                all_greedy=True,
                native_accept=self.fail,
            )
        self.assertEqual(accepted.tolist(), [2, 1])
        self.assertEqual(cap_trim.tolist(), [0, 1])
        # bonus = target top token at the accepted length.
        self.assertEqual(bonus.tolist(), [3, 0])
        # r0 paid the relaxed token's regret (8 -> 6.5); r1's relaxed token
        # was cap-trimmed away, so its budget is untouched.
        budgets = adapter._remaining_budget.tolist()
        self.assertAlmostEqual(budgets[1], 6.5)
        self.assertAlmostEqual(budgets[2], 8.0)

    def test_budget_clamped_at_zero(self):
        adapter = self._make_bound_adapter()
        candidates = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        logits = self._logits_with_tops([2, 1, 3])
        decision = _StubDecision(
            accepted=torch.tensor([2], dtype=torch.int32),
            mismatched=torch.tensor([[True, True]]),
            regrets=torch.tensor([[100.0, 100.0]]),
        )
        with patch.object(
            asd_dspark_module, "choose_prefix_batch", lambda **kwargs: decision
        ):
            _, _, _ = adapter.accept_or_native(
                candidates=candidates,
                target_logits=logits,
                cutoff_verify_lens=None,
                req_pool_indices=torch.tensor([1], dtype=torch.int64),
                all_greedy=True,
                native_accept=self.fail,
            )
        self.assertEqual(adapter._remaining_budget.tolist()[1], 0.0)

    def test_inactive_adapter_falls_back_to_native(self):
        adapter = DSparkASDAdapter(
            settings=DSparkASDSettings(),
            gamma=2,
            verify_num_draft_tokens=3,
            device="cpu",
        )
        sentinel = (torch.tensor([1]), torch.tensor([2]), torch.tensor([0]))
        accepted, bonus, cap_trim = adapter.accept_or_native(
            candidates=torch.zeros(1, 3, dtype=torch.int64),
            target_logits=None,
            cutoff_verify_lens=None,
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            all_greedy=False,
            native_accept=lambda: sentinel,
        )
        self.assertIs(accepted, sentinel[0])
        self.assertIs(bonus, sentinel[1])
        self.assertIs(cap_trim, sentinel[2])

    def test_active_requires_greedy(self):
        adapter = self._make_bound_adapter()
        with self.assertRaisesRegex(ValueError, "greedy"):
            adapter.accept_or_native(
                candidates=torch.zeros(2, 3, dtype=torch.int64),
                target_logits=torch.zeros(6, 5),
                cutoff_verify_lens=None,
                req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
                all_greedy=False,
                native_accept=self.fail,
            )

    def test_active_requires_target_logits(self):
        adapter = self._make_bound_adapter()
        with self.assertRaisesRegex(RuntimeError, "target logits"):
            adapter.accept_or_native(
                candidates=torch.zeros(2, 3, dtype=torch.int64),
                target_logits=None,
                cutoff_verify_lens=None,
                req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
                all_greedy=True,
                native_accept=self.fail,
            )

    def test_candidate_width_mismatch_raises(self):
        adapter = self._make_bound_adapter()
        with self.assertRaisesRegex(ValueError, "candidate width"):
            adapter.accept_or_native(
                candidates=torch.zeros(2, 4, dtype=torch.int64),
                target_logits=torch.zeros(8, 5),
                cutoff_verify_lens=None,
                req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
                all_greedy=True,
                native_accept=self.fail,
            )

    def test_target_logits_row_count_mismatch_raises(self):
        adapter = self._make_bound_adapter()
        with self.assertRaisesRegex(ValueError, "target logits row count"):
            adapter.accept_or_native(
                candidates=torch.zeros(2, 3, dtype=torch.int64),
                target_logits=torch.zeros(5, 5),
                cutoff_verify_lens=None,
                req_pool_indices=torch.tensor([1, 2], dtype=torch.int64),
                all_greedy=True,
                native_accept=self.fail,
            )

    def test_unbound_decode_raises(self):
        adapter = DSparkASDAdapter(
            settings=DSparkASDSettings(config=_StubConfig()),
            gamma=2,
            verify_num_draft_tokens=3,
            device="cpu",
        )
        with self.assertRaisesRegex(RuntimeError, "before prefill"):
            adapter.accept_or_native(
                candidates=torch.zeros(1, 3, dtype=torch.int64),
                target_logits=torch.zeros(3, 5),
                cutoff_verify_lens=None,
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                all_greedy=True,
                native_accept=self.fail,
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
