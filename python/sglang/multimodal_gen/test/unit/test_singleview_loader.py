# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for native/singleview_loader.py architecture normalization (Phase 7)."""

from __future__ import annotations

import os

import pytest

from sglang.multimodal_gen.native.singleview_loader import (
    _normalize_blackwell_arch,
    _resolved_cuda_arch_list,
    _effective_cuda_arch_list,
)


# ---------------------------------------------------------------------------
# Phase 7.1: _normalize_blackwell_arch
# ---------------------------------------------------------------------------

class TestNormalizeBlackwellArch:
    """Unit tests for architecture list normalization (12.0→12.0a etc.)."""

    # -- smoke / passthrough --------------------------------------------------

    def test_passthrough_empty(self):
        """Empty or whitespace-only arch lists are normalized to ''."""
        assert _normalize_blackwell_arch("") == ""
        assert _normalize_blackwell_arch("   ") == ""

    def test_passthrough_single_non_blackwell(self):
        """Non-Blackwell tokens are left untouched."""
        for tok in ("8.9", "9.0", "9.0a", "10.0", "10.0a", "7.5", "8.0", "10.1"):
            assert _normalize_blackwell_arch(tok) == tok

    def test_passthrough_blackwell_with_a_suffix(self):
        """12.0a and 12.1a are already correct → preserved as-is."""
        assert _normalize_blackwell_arch("12.0a") == "12.0a"
        assert _normalize_blackwell_arch("12.1a") == "12.1a"

    def test_passthrough_non_sensical(self):
        """Tokens that aren't valid arch strings pass through unchanged."""
        for tok in ("unsupported", "cpu", "rocm", "12.99"):
            assert _normalize_blackwell_arch(tok) == tok

    # -- bare 12.0 / 12.1 → 12.0a / 12.1a -----------------------------------

    def test_bare_12_0_gets_a_suffix(self):
        """12.0 → 12.0a (the core normalization case)."""
        assert _normalize_blackwell_arch("12.0") == "12.0a"

    def test_bare_12_1_gets_a_suffix(self):
        """12.1 → 12.1a (DGX Spark / sm_121)."""
        assert _normalize_blackwell_arch("12.1") == "12.1a"

    # -- multi-arch (semicolons) ----------------------------------------------

    def test_multi_arch_hopper_and_blackwell(self):
        """8.9;12.0 → 8.9;12.0a."""
        assert _normalize_blackwell_arch("8.9;12.0") == "8.9;12.0a"

    def test_multi_arch_blackwell_a_and_bare(self):
        """12.0a;12.1 → 12.0a;12.1a."""
        assert _normalize_blackwell_arch("12.0a;12.1") == "12.0a;12.1a"

    def test_multi_arch_three_plus_tokens(self):
        """8.9;12.0;12.0a → 8.9;12.0a;12.0a."""
        assert _normalize_blackwell_arch("8.9;12.0;12.0a") == "8.9;12.0a;12.0a"

    def test_multi_arch_all_already_correct(self):
        """12.0a;12.1a → 12.0a;12.1a (idempotent)."""
        assert _normalize_blackwell_arch("12.0a;12.1a") == "12.0a;12.1a"

    def test_multi_arch_9_0a_and_12_1(self):
        """9.0a;12.1 → 9.0a;12.1a."""
        assert _normalize_blackwell_arch("9.0a;12.1") == "9.0a;12.1a"

    # -- comma separator coercion ---------------------------------------------

    def test_comma_separator_coerced_to_semicolon(self):
        """Commas are replaced with semicolons."""
        assert _normalize_blackwell_arch("12.0,8.9") == "12.0a;8.9"

    def test_mixed_comma_and_semicolon(self):
        """Mixed separator input works."""
        assert _normalize_blackwell_arch("12.0;8.9,10.0a") == "12.0a;8.9;10.0a"

    # -- PTX suffix -----------------------------------------------------------

    def test_ptx_suffix_preserved(self):
        """12.0+PTX → 12.0a+PTX."""
        assert _normalize_blackwell_arch("12.0+PTX") == "12.0a+PTX"

    def test_ptx_already_a_suffix(self):
        """12.0a+PTX → 12.0a+PTX (no double-append)."""
        assert _normalize_blackwell_arch("12.0a+PTX") == "12.0a+PTX"

    def test_ptx_multi_arch(self):
        """8.9+PTX;12.0 → 8.9+PTX;12.0a."""
        assert _normalize_blackwell_arch("8.9+PTX;12.0") == "8.9+PTX;12.0a"

    # -- synthetic / edge -----------------------------------------------------

    def test_double_bare_12_0(self):
        """12.0;12.0 → 12.0a;12.0a (both normalized)."""
        assert _normalize_blackwell_arch("12.0;12.0") == "12.0a;12.0a"

    def test_mixed_real_and_fake(self):
        """7.5;12.0 → 7.5;12.0a (mixed real/fake tokens)."""
        assert _normalize_blackwell_arch("7.5;12.0") == "7.5;12.0a"

    def test_12_0a_already_present_no_double_append(self):
        """12.0a is preserved (no 12.0aa)."""
        assert _normalize_blackwell_arch("12.0a") == "12.0a"


# ---------------------------------------------------------------------------
# Phase 7.2: _resolved_cuda_arch_list and _effective_cuda_arch_list
# ---------------------------------------------------------------------------

_ARCH_LIST_ENV = "TORCH_CUDA_ARCH_LIST"
_OMNI_ARCH_ENV = "OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST"


def _setenv(key: str, value: str | None) -> None:
    """Set or remove an env var for the duration of a test."""
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


class TestResolvedCudaArchList:
    """Tests for _resolved_cuda_arch_list (env-var aware arch resolver)."""

    def test_default_returns_12_0a(self, monkeypatch):
        """When no env vars are set, _resolved_cuda_arch_list returns 12.0a."""
        monkeypatch.delenv(_ARCH_LIST_ENV, raising=False)
        monkeypatch.delenv(_OMNI_ARCH_ENV, raising=False)
        result = _resolved_cuda_arch_list()
        assert result == "12.0a"

    def test_torch_12_0_normalized(self, monkeypatch):
        """TORCH_CUDA_ARCH_LIST=12.0 → normalized to 12.0a."""
        monkeypatch.setenv(_ARCH_LIST_ENV, "12.0")
        monkeypatch.delenv(_OMNI_ARCH_ENV, raising=False)
        result = _resolved_cuda_arch_list()
        assert result == "12.0a"

    def test_torch_12_0a_returns_none(self, monkeypatch):
        """TORCH_CUDA_ARCH_LIST=12.0a → returns None (no override needed)."""
        monkeypatch.setenv(_ARCH_LIST_ENV, "12.0a")
        monkeypatch.delenv(_OMNI_ARCH_ENV, raising=False)
        result = _resolved_cuda_arch_list()
        assert result is None

    def test_omni_12_1_normalized(self, monkeypatch):
        """OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST=12.1 → normalized to 12.1a."""
        monkeypatch.delenv(_ARCH_LIST_ENV, raising=False)
        monkeypatch.setenv(_OMNI_ARCH_ENV, "12.1")
        result = _resolved_cuda_arch_list()
        assert result == "12.1a"

    def test_omni_multi_arch_normalized(self, monkeypatch):
        """OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST=8.9;12.0 → 8.9;12.0a."""
        monkeypatch.delenv(_ARCH_LIST_ENV, raising=False)
        monkeypatch.setenv(_OMNI_ARCH_ENV, "8.9;12.0")
        result = _resolved_cuda_arch_list()
        assert result == "8.9;12.0a"

    def test_torch_takes_precedence(self, monkeypatch):
        """TORCH_CUDA_ARCH_LIST takes priority over OMNI-specific env."""
        monkeypatch.setenv(_ARCH_LIST_ENV, "9.0;12.0")
        monkeypatch.setenv(_OMNI_ARCH_ENV, "12.0a")
        # TORCH_CUDA_ARCH_LIST=9.0;12.0 → normalized to 9.0;12.0a
        result = _resolved_cuda_arch_list()
        assert result == "9.0;12.0a"

    def test_torch_12_0a_already_correct_returns_none(self, monkeypatch):
        """When TORCH_CUDA_ARCH_LIST is already correct, return None."""
        monkeypatch.setenv(_ARCH_LIST_ENV, "12.0a")
        monkeypatch.delenv(_OMNI_ARCH_ENV, raising=False)
        result = _resolved_cuda_arch_list()
        assert result is None

    def test_torch_12_0a_multi(self, monkeypatch):
        """TORCH_CUDA_ARCH_LIST=12.0a;9.0 → already correct, return None."""
        monkeypatch.setenv(_ARCH_LIST_ENV, "12.0a;9.0")
        monkeypatch.delenv(_OMNI_ARCH_ENV, raising=False)
        result = _resolved_cuda_arch_list()
        assert result is None


class TestEffectiveCudaArchList:
    """Tests for _effective_cuda_arch_list (always returns a string)."""

    def test_default_returns_12_0a(self, monkeypatch):
        """When no env vars set, returns the normalized default 12.0a."""
        monkeypatch.delenv(_ARCH_LIST_ENV, raising=False)
        monkeypatch.delenv(_OMNI_ARCH_ENV, raising=False)
        result = _effective_cuda_arch_list()
        assert result == "12.0a"

    def test_torch_12_0_normalized(self, monkeypatch):
        """TORCH_CUDA_ARCH_LIST=12.0 → 12.0a."""
        monkeypatch.setenv(_ARCH_LIST_ENV, "12.0")
        monkeypatch.delenv(_OMNI_ARCH_ENV, raising=False)
        result = _effective_cuda_arch_list()
        assert result == "12.0a"

    def test_torch_12_0a_unmodified(self, monkeypatch):
        """TORCH_CUDA_ARCH_LIST=12.0a → 12.0a."""
        monkeypatch.setenv(_ARCH_LIST_ENV, "12.0a")
        monkeypatch.delenv(_OMNI_ARCH_ENV, raising=False)
        result = _effective_cuda_arch_list()
        assert result == "12.0a"

    def test_omni_12_1_normalized(self, monkeypatch):
        """OMNIDREAMS_SINGLEVIEW_CUDA_ARCH_LIST=12.1 → 12.1a."""
        monkeypatch.delenv(_ARCH_LIST_ENV, raising=False)
        monkeypatch.setenv(_OMNI_ARCH_ENV, "12.1")
        result = _effective_cuda_arch_list()
        assert result == "12.1a"

    def test_torch_takes_precedence(self, monkeypatch):
        """TORCH_CUDA_ARCH_LIST wins over OMNI-specific env."""
        monkeypatch.setenv(_ARCH_LIST_ENV, "9.0;12.0")
        monkeypatch.setenv(_OMNI_ARCH_ENV, "12.0a")
        result = _effective_cuda_arch_list()
        assert result == "9.0;12.0a"
