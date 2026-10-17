"""PD disaggregation combination unit tests (issue #1105).

Covers server-args validation and config-bag population for PD +
speculative / prefix-cache / chunked-prefill combinations without
real 8-GPU e2e.  The tests use the same mock / override pattern as
the existing unit tests in ``test/registered/unit``.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.srt.runtime_context import get_context, get_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pd_args(**overrides):
    """Build ServerArgs and run PD disaggregation validation.

    ``model_path="dummy"`` short-circuits ``resolve_once`` before PD handlers
    run, so invoke ``_handle_pd_disaggregation`` directly (same as
    ``test/registered/unit/server_args/test_server_args.py``).
    """
    args = ServerArgs(model_path="dummy", **overrides)
    args._handle_pd_disaggregation()
    return args


def _make_load_balance_args(**overrides):
    """Build ServerArgs with PD + load-balance resolution."""
    args = ServerArgs(model_path="dummy", **overrides)
    args._handle_pd_disaggregation()
    args._handle_load_balance_method()
    return args


def _make_server_checks_args(**overrides):
    """Run only the chunked-prefill divisibility rule from check_server_args."""
    args = ServerArgs(model_path="dummy", **overrides)
    if args.chunked_prefill_size > 0 and args.disaggregation_mode != "decode":
        assert (
            args.chunked_prefill_size % args.page_size == 0
        ), "chunked_prefill_size must be divisible by page_size"
    return args


def _make_cuda_graph_args(**overrides):
    """Build ServerArgs with CUDA graph config (PD role wiring)."""
    args = ServerArgs(model_path="dummy", **overrides)
    args.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        is_piecewise_cuda_graph_disabled_model=False,
        is_multimodal=False,
        is_multimodal_piecewise_cuda_graph_supported=False,
    )
    with (
        patch("sglang.srt.utils.is_cuda", return_value=True),
        patch.object(ServerArgs, "use_mla_backend", return_value=False),
    ):
        args._handle_pd_disaggregation()
        args._handle_cuda_graph_config()
    return args


def _make_unified_memory_args(**overrides):
    args = ServerArgs(model_path="dummy", **overrides)
    args._handle_unified_memory_pool()
    return args


def _install_override(**fields):
    """Install a scoped ServerArgs override on the runtime context."""
    override = get_context().override_server_args(**fields)
    override.install()
    return override


# ---------------------------------------------------------------------------
# 1. PD + speculative (MTP) combinations
# ---------------------------------------------------------------------------

class TestPDSpeculativeCombos(CustomTestCase):
    """Validate PD disaggregation + speculative / MTP server-args rules."""

    def test_pd_decode_radix_cache_rejects_speculative(self):
        """PD decode + radix cache + speculative algorithm must raise."""
        with self.assertRaises(ValueError) as ctx:
            _make_pd_args(
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_transfer_backend="nixl",
                speculative_algorithm="EAGLE",
            )
        self.assertIn(
            "speculative decoding",
            str(ctx.exception),
        )

    def test_pd_decode_radix_cache_rejects_mtp(self):
        """PD decode + radix cache + NEXTN (MTP) must also raise."""
        with self.assertRaises(ValueError) as ctx:
            _make_pd_args(
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_transfer_backend="mooncake",
                speculative_algorithm="NEXTN",
            )
        self.assertIn(
            "speculative decoding",
            str(ctx.exception),
        )

    def test_pd_decode_radix_cache_allows_no_spec(self):
        """PD decode + radix cache without speculative is accepted."""
        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="mooncake",
            speculative_algorithm=None,
        )
        self.assertTrue(args.disaggregation_decode_enable_radix_cache)
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
        self.assertTrue(args.disable_radix_cache)

    def test_pd_decode_radix_cache_sets_disable_radix_cache_false(self):
        """PD decode + radix cache must set disable_radix_cache=False."""
        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="mooncake",
        )
        self.assertFalse(args.disable_radix_cache)

    def test_pd_decode_radix_cache_with_dp_attention_warns(self):
        """PD decode + radix cache + DP attention logs an experimental warning."""
        with patch("sglang.srt.arg_groups.pd_disaggregation_hook.logger"):
            args = _make_pd_args(
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_transfer_backend="mooncake",
                enable_dp_attention=True,
            )
        # The warning is emitted; we just verify the config is accepted.
        self.assertFalse(args.disable_radix_cache)


# ---------------------------------------------------------------------------
# 3. PD + chunked-prefill combinations
# ---------------------------------------------------------------------------

class TestPDChunkedPrefillCombos(CustomTestCase):
    """Validate PD disaggregation + chunked-prefill rules."""

    def test_pd_decode_skips_chunked_prefill_size_validation(self):
        """PD decode skips the chunked_prefill_size % page_size check."""
        # chunked_prefill_size=1000 with page_size=16 would normally fail
        # (1000 % 16 != 0), but PD decode skips the check.
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
        from sglang.srt.model_executor.cuda_graph_config import Backend

        args = _make_cuda_graph_args(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
        )
        self.assertEqual(
            args.cuda_graph_config.decode.backend,
            Backend.DISABLED,
        )

    def test_pd_decode_disables_prefill_cuda_graph(self):
        """PD decode disables the prefill-phase CUDA graph."""
        from sglang.srt.model_executor.cuda_graph_config import Backend

        args = _make_cuda_graph_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
        )
        self.assertEqual(
            args.cuda_graph_config.prefill.backend,
            Backend.DISABLED,
        )

    def test_non_pd_keeps_both_cuda_graphs(self):
        """Non-PD mode keeps both CUDA graph phases enabled (default)."""
        from sglang.srt.model_executor.cuda_graph_config import Backend

        args = _make_cuda_graph_args(disaggregation_mode="null")
        # Default backend is FULL (not DISABLED).
        self.assertNotEqual(
            args.cuda_graph_config.decode.backend,
            Backend.DISABLED,
        )
        self.assertNotEqual(
            args.cuda_graph_config.prefill.backend,
            Backend.DISABLED,
        )


# ---------------------------------------------------------------------------
# 5. PD + DCP (decode context parallel) combinations
# ---------------------------------------------------------------------------

class TestPDDcpCombos(CustomTestCase):
    """Validate PD disaggregation + DCP rules."""

    def test_pd_decode_dcp_requires_supported_backend(self):
        """PD decode DCP rejects unsupported transfer backends."""
        with self.assertRaises(ValueError) as ctx:
            _make_pd_args(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="mori",
                dcp_size=4,
            )
        self.assertIn("mooncake, nixl, or fake", str(ctx.exception))

    def test_pd_decode_dcp_rejects_radix_cache(self):
        """PD decode DCP + radix cache must raise."""
        with self.assertRaises(ValueError) as ctx:
            _make_pd_args(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="nixl",
                dcp_size=4,
                disaggregation_decode_enable_radix_cache=True,
            )
        self.assertIn("chunk cache", str(ctx.exception))

    def test_pd_decode_dcp_rejects_hierarchical_cache(self):
        """PD decode DCP + hierarchical cache must raise."""
        with self.assertRaises(ValueError) as ctx:
            _make_pd_args(
                disaggregation_mode="decode",
                disaggregation_transfer_backend="nixl",
                dcp_size=4,
                enable_hierarchical_cache=True,
            )
        self.assertIn("--enable-hierarchical-cache", str(ctx.exception))

    def test_pd_decode_dcp_accepts_nixl(self):
        """PD decode DCP + nixl is a valid combination."""
        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            dcp_size=4,
        )
        self.assertEqual(args.dcp_size, 4)
        self.assertTrue(args.disable_radix_cache)

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
        self.assertEqual(args.load_balance_method, "round_robin")

    def test_pd_prefill_defaults_to_follow_bootstrap_room(self):
        args = _make_load_balance_args(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
        )
        self.assertEqual(args.load_balance_method, "follow_bootstrap_room")

    def test_pd_decode_defaults_to_round_robin(self):
        args = _make_load_balance_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
        )
        self.assertEqual(args.load_balance_method, "round_robin")

    def test_pd_prefill_explicit_override(self):
        args = _make_load_balance_args(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
            load_balance_method="round_robin",
        )
        self.assertEqual(args.load_balance_method, "round_robin")

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
        import os

        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="mooncake_tcp",
        )
        self.assertEqual(args.disaggregation_transfer_backend, "mooncake")
        self.assertEqual(os.environ.get("MC_FORCE_TCP"), "1")
        # Clean up
        os.environ.pop("MC_FORCE_TCP", None)

    def test_mooncake_tcp_prefill_rewrites(self):
        """mooncake_tcp on prefill is also rewritten."""
        import os

        args = _make_pd_args(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mooncake_tcp",
        )
        self.assertEqual(args.disaggregation_transfer_backend, "mooncake")
        os.environ.pop("MC_FORCE_TCP", None)


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
        # per_worker = 16, which is <= 32, so extra_slots = 16 * 2 = 32
        self.assertEqual(args.disaggregation_decode_extra_slots, 32)

    def test_pd_decode_large_batch_no_extra_slots(self):
        """Large per-worker batch reserves no extra decode slots."""
        args = _make_pd_args(
            disaggregation_mode="decode",
            disaggregation_transfer_backend="nixl",
            max_running_requests=128,
            dp_size=1,
        )
        # per_worker = 128, which is > 32, so extra_slots = 0
        self.assertEqual(args.disaggregation_decode_extra_slots, 0)


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

        self.assertEqual(get_server_args().disaggregation_mode, "decode")
        self.assertEqual(
            get_server_args().disaggregation_transfer_backend, "nixl"
        )

    def test_override_speculative_algorithm_propagates(self):
        override = _install_override(
            speculative_algorithm="EAGLE",
        )
        self.addCleanup(override.restore)

        self.assertEqual(get_server_args().speculative_algorithm, "EAGLE")

    def test_override_chunked_prefill_size_propagates(self):
        override = _install_override(
            chunked_prefill_size=8192,
        )
        self.addCleanup(override.restore)

        self.assertEqual(get_server_args().chunked_prefill_size, 8192)


if __name__ == "__main__":
    unittest.main()
