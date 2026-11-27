"""PD disaggregation combination unit tests (issue #1105).

Manual-only (not registered for CI). Exercises module-level handlers and
``resolution_result`` the same way as
``test/registered/unit/server_args/test_server_args.py``.
"""

import os
import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.cuda_graph_hook import (
    apply_cuda_graph_disaggregation_roles,
    parse_cuda_graph_config,
)
from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.arg_groups.pd_disaggregation_hook import (
    handle_pd_disaggregation,
    logger as pd_logger,
)
from sglang.srt.arg_groups.serving_hook import handle_load_balance_method
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.runtime_context import get_context, get_server_args, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pd_args(**overrides):
    """Build ServerArgs and run PD disaggregation validation."""
    args = ServerArgs(model_path="dummy", **overrides)
    handle_pd_disaggregation(args)
    return args


def _make_spec_pd_args(**overrides):
    """PD validation + speculative hook (radix+spec rules live there)."""
    args = ServerArgs(model_path="dummy", **overrides)
    handle_pd_disaggregation(args)
    handle_speculative_decoding(args)
    return args


def _make_load_balance_args(**overrides):
    """Build ServerArgs with PD + load-balance resolution."""
    args = ServerArgs(model_path="dummy", **overrides)
    handle_pd_disaggregation(args)
    handle_load_balance_method(args)
    return args


def _make_server_checks_args(**overrides):
    """Run product ``check_server_args`` (chunked-prefill divisibility)."""
    # served_model_name must be non-None: check_server_args does ``":" not in name``.
    overrides.setdefault("served_model_name", "dummy")
    args = ServerArgs(model_path="dummy", **overrides)
    args.check_server_args()
    return args


def _make_cuda_graph_args(**overrides):
    """Build ServerArgs with PD role CUDA-graph wiring."""
    args = ServerArgs(model_path="dummy", **overrides)
    handle_pd_disaggregation(args)
    parse_cuda_graph_config(args)
    apply_cuda_graph_disaggregation_roles(args)
    return args


def _make_unified_memory_args(**overrides):
    args = ServerArgs(model_path="dummy", **overrides)
    handle_unified_memory_pool(args)
    return args


def _install_override(**fields):
    """Install a scoped ServerArgs override on the runtime context."""
    reset_context()
    override = get_context().override_server_args(**fields)
    override.install()
    return override


# ---------------------------------------------------------------------------
# 1. PD + speculative (MTP) combinations
# ---------------------------------------------------------------------------


class TestPDSpeculativeCombos(CustomTestCase):
    """Validate PD disaggregation + speculative / MTP server-args rules."""

    def test_pd_decode_radix_cache_rejects_speculative(self):
        """PD decode + radix cache rejects non-allowlisted speculative algos."""
        # EAGLE/EAGLE3/DFLASH/DSPARK are allowlisted; NGRAM/UNO still raise.
        with self.assertRaises(ValueError) as ctx:
            _make_spec_pd_args(
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_transfer_backend="nixl",
                speculative_algorithm="NGRAM",
            )
        self.assertIn("speculative decoding", str(ctx.exception))

    def test_pd_decode_radix_cache_rejects_mtp(self):
        """PD decode + radix cache + UNO must also raise."""
        with self.assertRaises(ValueError) as ctx:
            _make_spec_pd_args(
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_transfer_backend="mooncake",
                speculative_algorithm="UNO",
            )
        self.assertIn("speculative decoding", str(ctx.exception))

    def test_pd_decode_radix_cache_allows_no_spec(self):
        """PD decode + radix cache without speculative is accepted."""
        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="mooncake",
            speculative_algorithm=None,
        )
        self.assertFalse(resolution_result(args, "disable_radix_cache"))
        self.assertIsNone(args.speculative_algorithm)

    def test_pd_prefill_rejects_fake_transfer_backend(self):
        """PD prefill does not support the fake transfer backend."""
        with self.assertRaises(AssertionError):
            _make_pd_args(
                disaggregation_mode="prefill",
                disaggregation_transfer_backend="fake",
            )

    def test_pd_prefill_accepts_nixl(self):
        """PD prefill + nixl is a valid combination."""
        args = _make_pd_args(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
        )
        self.assertEqual(args.disaggregation_mode, "prefill")
        self.assertEqual(args.disaggregation_transfer_backend, "nixl")

    def test_pd_decode_rejects_radix_cache_with_fake_backend(self):
        """PD decode + radix cache + fake transfer backend must raise."""
        with self.assertRaises(ValueError) as ctx:
            _make_pd_args(
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_transfer_backend="fake",
            )
        self.assertIn("fake", str(ctx.exception))

    def test_pd_decode_radix_cache_rejects_hisparse(self):
        """PD decode + radix cache + HiSparse must raise."""
        with self.assertRaises(ValueError) as ctx:
            _make_pd_args(
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_transfer_backend="nixl",
                enable_hisparse=True,
            )
        self.assertIn("--enable-hisparse", str(ctx.exception))


# ---------------------------------------------------------------------------
# 2. PD + prefix cache combinations
# ---------------------------------------------------------------------------


class TestPDPrefixCacheCombos(CustomTestCase):
    """Validate PD disaggregation + prefix-cache / radix-cache rules."""

    def test_pd_decode_forces_chunk_cache_by_default(self):
        """PD decode without explicit radix cache defaults to chunk cache."""
        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
        )
        self.assertTrue(resolution_result(args, "disable_radix_cache"))

    def test_pd_decode_radix_cache_sets_disable_radix_cache_false(self):
        """PD decode + radix cache must set disable_radix_cache=False."""
        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="mooncake",
        )
        self.assertFalse(resolution_result(args, "disable_radix_cache"))

    def test_pd_decode_radix_cache_with_dp_attention_warns(self):
        """PD decode + radix cache + DP attention logs an experimental warning."""
        with self.assertLogs(pd_logger, level="WARNING") as logs:
            args = _make_pd_args(
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_transfer_backend="mooncake",
                # attn_dp_enabled_of() keys off attn_dp_size > 1
                attn_dp_size=2,
            )
        joined = "\n".join(logs.output)
        self.assertIn("EXPERIMENTAL", joined)
        self.assertIn("DP attention", joined)
        self.assertFalse(resolution_result(args, "disable_radix_cache"))


# ---------------------------------------------------------------------------
# 3. PD + chunked-prefill combinations
# ---------------------------------------------------------------------------


class TestPDChunkedPrefillCombos(CustomTestCase):
    """Validate PD disaggregation + chunked-prefill rules via check_server_args."""

    def test_pd_decode_skips_chunked_prefill_size_validation(self):
        """PD decode skips the chunked_prefill_size % page_size check."""
        args = _make_server_checks_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            chunked_prefill_size=1000,
            page_size=16,
        )
        self.assertEqual(args.chunked_prefill_size, 1000)

    def test_pd_prefill_accepts_chunked_prefill(self):
        """PD prefill + valid chunked_prefill_size is accepted."""
        args = _make_server_checks_args(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
            chunked_prefill_size=4096,
            page_size=16,
        )
        self.assertEqual(args.chunked_prefill_size, 4096)

    def test_pd_prefill_chunked_prefill_must_be_divisible_by_page_size(self):
        """PD prefill enforces chunked_prefill_size % page_size == 0."""
        with self.assertRaises(AssertionError):
            _make_server_checks_args(
                disaggregation_mode="prefill",
                disaggregation_transfer_backend="nixl",
                chunked_prefill_size=1000,
                page_size=16,
            )

    def test_non_pd_chunked_prefill_must_be_divisible_by_page_size(self):
        """Non-PD mode enforces chunked_prefill_size % page_size == 0."""
        with self.assertRaises(AssertionError):
            _make_server_checks_args(
                disaggregation_mode="null",
                chunked_prefill_size=1000,
                page_size=16,
            )


# ---------------------------------------------------------------------------
# 4. PD + CUDA graph role assignment
# ---------------------------------------------------------------------------


class TestPDCudaGraphRoles(CustomTestCase):
    """Validate that PD mode disables the irrelevant CUDA graph phase."""

    def test_pd_prefill_disables_decode_cuda_graph(self):
        """PD prefill disables the decode-phase CUDA graph."""
        args = _make_cuda_graph_args(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
        )
        cfg = resolution_result(args, "cuda_graph_config")
        self.assertEqual(cfg.decode.backend, Backend.DISABLED)

    def test_pd_decode_disables_prefill_cuda_graph(self):
        """PD decode disables the prefill-phase CUDA graph."""
        args = _make_cuda_graph_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
        )
        cfg = resolution_result(args, "cuda_graph_config")
        self.assertEqual(cfg.prefill.backend, Backend.DISABLED)

    def test_non_pd_keeps_both_cuda_graphs(self):
        """Non-PD mode does not force-disable either phase when both are set."""
        # Prefill defaults to disabled; pin both backends so the role hook
        # is the only thing that could clear them.
        args = _make_cuda_graph_args(
            disaggregation_mode="null",
            cuda_graph_backend_prefill="full",
            cuda_graph_backend_decode="full",
        )
        cfg = resolution_result(args, "cuda_graph_config", default=args.cuda_graph_config)
        self.assertNotEqual(cfg.decode.backend, Backend.DISABLED)
        self.assertNotEqual(cfg.prefill.backend, Backend.DISABLED)


# ---------------------------------------------------------------------------
# 5. PD + DCP (decode context parallel) combinations
# ---------------------------------------------------------------------------


class TestPDDcpCombos(CustomTestCase):
    """Validate PD disaggregation + DCP rules."""

    def test_pd_decode_dcp_requires_supported_backend(self):
        """PD decode DCP rejects unsupported transfer backends."""
        # mooncake/nixl/mori/ascend/fake are allowlisted; anything else raises.
        with self.assertRaises(ValueError) as ctx:
            _make_pd_args(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="libfabric",
                dcp_size=4,
            )
        self.assertIn("mooncake, nixl, mori, ascend, or fake", str(ctx.exception))

    def test_pd_decode_dcp_disables_radix_cache_by_default(self):
        """PD decode DCP without radix flag resolves disable_radix_cache=True."""
        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            dcp_size=4,
        )
        self.assertEqual(args.dcp_size, 4)
        self.assertTrue(resolution_result(args, "disable_radix_cache"))

    def test_pd_decode_dcp_accepts_nixl(self):
        """PD decode DCP + nixl is a valid combination."""
        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            dcp_size=4,
        )
        self.assertEqual(args.dcp_size, 4)
        self.assertTrue(resolution_result(args, "disable_radix_cache"))

    def test_pd_decode_dcp_accepts_fake(self):
        """PD decode DCP + fake (synthetic benchmark) is accepted."""
        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="fake",
            dcp_size=4,
        )
        self.assertEqual(args.dcp_size, 4)


# ---------------------------------------------------------------------------
# 6. PD + load-balance method
# ---------------------------------------------------------------------------


class TestPDLoadBalanceMethod(CustomTestCase):
    """Validate PD load-balance method defaults and overrides."""

    def test_non_pd_defaults_to_round_robin(self):
        args = _make_load_balance_args(disaggregation_mode="null")
        self.assertEqual(
            resolution_result(args, "load_balance_method"), "round_robin"
        )

    def test_pd_prefill_defaults_to_follow_bootstrap_room(self):
        args = _make_load_balance_args(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
        )
        self.assertEqual(
            resolution_result(args, "load_balance_method"),
            "follow_bootstrap_room",
        )

    def test_pd_decode_defaults_to_round_robin(self):
        args = _make_load_balance_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
        )
        self.assertEqual(
            resolution_result(args, "load_balance_method"), "round_robin"
        )

    def test_pd_prefill_explicit_override(self):
        args = _make_load_balance_args(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
            load_balance_method="round_robin",
        )
        self.assertEqual(
            resolution_result(args, "load_balance_method"), "round_robin"
        )

    def test_invalid_disaggregation_mode_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _make_load_balance_args(disaggregation_mode="invalid_mode")
        self.assertIn("Invalid disaggregation_mode", str(ctx.exception))


# ---------------------------------------------------------------------------
# 7. PD + unified memory pool
# ---------------------------------------------------------------------------


class TestPDUnifiedMemoryCombos(CustomTestCase):
    """Validate PD disaggregation + unified memory pool rules."""

    def test_unified_memory_pd_requires_mooncake(self):
        """Unified memory + PD disaggregation requires mooncake backend."""
        with self.assertRaises(AssertionError) as ctx:
            _make_unified_memory_args(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="nixl",
                enable_unified_memory=True,
            )
        self.assertIn("mooncake", str(ctx.exception))

    def test_unified_memory_pd_rejects_pp(self):
        """Unified memory + PD disaggregation does not support PP."""
        with self.assertRaises(AssertionError) as ctx:
            _make_unified_memory_args(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="mooncake",
                enable_unified_memory=True,
                pp_size=2,
            )
        self.assertIn("pipeline parallelism", str(ctx.exception))

    def test_unified_memory_pd_rejects_hisparse(self):
        """Unified memory + PD disaggregation is not compatible with HiSparse."""
        with self.assertRaises(AssertionError) as ctx:
            _make_unified_memory_args(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="mooncake",
                enable_unified_memory=True,
                enable_hisparse=True,
            )
        self.assertIn("--enable-hisparse", str(ctx.exception))


# ---------------------------------------------------------------------------
# 8. PD + transfer backend mooncake_tcp
# ---------------------------------------------------------------------------


class TestPDMooncakeTcpCombo(CustomTestCase):
    """Validate PD + mooncake_tcp transport backend rewriting."""

    def test_mooncake_tcp_rewrites_to_mooncake(self):
        """mooncake_tcp is rewritten to mooncake with MC_FORCE_TCP=1."""
        prev = os.environ.get("MC_FORCE_TCP")
        self.addCleanup(
            lambda: (
                os.environ.__setitem__("MC_FORCE_TCP", prev)
                if prev is not None
                else os.environ.pop("MC_FORCE_TCP", None)
            )
        )
        os.environ.pop("MC_FORCE_TCP", None)

        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake_tcp",
        )
        self.assertEqual(
            resolution_result(args, "disaggregation_transfer_backend"), "mooncake"
        )
        self.assertEqual(os.environ.get("MC_FORCE_TCP"), "1")

    def test_mooncake_tcp_prefill_rewrites(self):
        """mooncake_tcp on prefill is also rewritten."""
        prev = os.environ.get("MC_FORCE_TCP")
        self.addCleanup(
            lambda: (
                os.environ.__setitem__("MC_FORCE_TCP", prev)
                if prev is not None
                else os.environ.pop("MC_FORCE_TCP", None)
            )
        )
        os.environ.pop("MC_FORCE_TCP", None)

        args = _make_pd_args(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake_tcp",
        )
        self.assertEqual(
            resolution_result(args, "disaggregation_transfer_backend"), "mooncake"
        )
        self.assertEqual(os.environ.get("MC_FORCE_TCP"), "1")


# ---------------------------------------------------------------------------
# 9. PD + extra decode slots
# ---------------------------------------------------------------------------


class TestPDDecodeExtraSlots(CustomTestCase):
    """Validate PD decode extra-slots auto-sizing."""

    def test_pd_decode_small_batch_reserves_extra_slots(self):
        """Small per-worker batch reserves 2x extra decode slots."""
        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            max_running_requests=16,
            dp_size=1,
        )
        self.assertEqual(
            resolution_result(args, "disaggregation_decode_extra_slots"), 32
        )

    def test_pd_decode_large_batch_no_extra_slots(self):
        """Large per-worker batch reserves no extra decode slots."""
        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            max_running_requests=128,
            dp_size=1,
        )
        self.assertEqual(
            resolution_result(args, "disaggregation_decode_extra_slots"), 0
        )


# ---------------------------------------------------------------------------
# 10. Runtime context override smoke
# ---------------------------------------------------------------------------


class TestPDComboRuntimeContextOverride(CustomTestCase):
    """Smoke-test that PD combo overrides propagate to the runtime context."""

    def test_override_disaggregation_mode_propagates(self):
        override = _install_override(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
        )
        self.addCleanup(override.restore)

        args = get_server_args()
        self.assertEqual(resolution_result(args, "disaggregation_mode"), "decode")
        self.assertEqual(
            resolution_result(args, "disaggregation_transfer_backend"), "nixl"
        )

    def test_override_speculative_algorithm_propagates(self):
        override = _install_override(
            speculative_algorithm="EAGLE",
        )
        self.addCleanup(override.restore)

        self.assertEqual(
            resolution_result(get_server_args(), "speculative_algorithm"), "EAGLE"
        )

    def test_override_chunked_prefill_size_propagates(self):
        override = _install_override(
            chunked_prefill_size=8192,
        )
        self.addCleanup(override.restore)

        self.assertEqual(
            resolution_result(get_server_args(), "chunked_prefill_size"), 8192
        )


if __name__ == "__main__":
    unittest.main()
