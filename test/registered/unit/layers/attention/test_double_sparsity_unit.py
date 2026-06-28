"""Unit tests for standalone Double Sparsity (placeholder scaffolding).

Covers the runtime backbone: config parsing surface (absence of
``selection_mode`` / ``top_p``), selector ABI shape, validator
fail-fast behaviour for missing-config and HiSparse mutual-exclusion
(), and the ``_select_topk_indices`` config-gated branch on
``DeepseekV2AttentionMLA`` (hook).

Real selection kernels, FP8 page-signature projection, CUDA-graph
capture, NIAH / MMLU quality runs, and the upstream-shaped ship-gate are
exercised by later milestones.
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.attention.double_sparsity import (
    DoubleSparsityConfig,
    DoubleSparsitySelector,
    parse_double_sparsity_config,
    validate_double_sparsity,
)
from sglang.srt.layers.attention.double_sparsity.selector import (
    assert_real_selector_or_placeholder_allowed,
)
from sglang.test.ci.ci_register import register_cuda_ci

# CUDA-only: the Double Sparsity feature is validated on the FlashMLA/H200 path;
# AMD CI is intentionally not registered until the path is verified on ROCm.
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


def _valid_payload(path: str = "/tmp/cm.safetensors") -> str:
    return (
        '{"top_k": 2048, "page_size": 64, '
        f'"channel_mask_path": "{path}", "device_buffer_size": 4096}}'
    )


class TestDoubleSparsityConfigParser(unittest.TestCase):
    def test_minimal_required_fields(self):
        cfg = parse_double_sparsity_config(_valid_payload())
        self.assertEqual(cfg.top_k, 2048)
        self.assertEqual(cfg.page_size, 64)
        self.assertEqual(cfg.device_buffer_size, 4096)
        self.assertEqual(cfg.extra, {})
        self.assertIsInstance(cfg, DoubleSparsityConfig)

    def test_extra_dict_is_accepted(self):
        payload = (
            '{"top_k": 2048, "page_size": 64, '
            '"channel_mask_path": "/tmp/cm.safetensors", '
            '"device_buffer_size": 4096, "extra": {"experiment": "x"}}'
        )
        cfg = parse_double_sparsity_config(payload)
        self.assertEqual(cfg.extra, {"experiment": "x"})

    def test_selection_mode_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_double_sparsity_config('{"selection_mode": "TOPP"}')
        self.assertIn("selection_mode", str(ctx.exception))

    def test_top_p_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            parse_double_sparsity_config('{"top_p": 0.9}')
        self.assertIn("top_p", str(ctx.exception))

    def test_missing_channel_mask_path(self):
        payload = '{"top_k": 2048, "page_size": 64, "device_buffer_size": 4096}'
        with self.assertRaises(ValueError) as ctx:
            parse_double_sparsity_config(payload)
        self.assertIn("channel_mask_path", str(ctx.exception))

    def test_invalid_json(self):
        with self.assertRaises(ValueError):
            parse_double_sparsity_config("not json")

    def test_invalid_top_k(self):
        payload = (
            '{"top_k": 0, "page_size": 64, '
            '"channel_mask_path": "/tmp/cm.safetensors", "device_buffer_size": 4096}'
        )
        with self.assertRaises(ValueError):
            parse_double_sparsity_config(payload)

    def test_selector_width_overflow_policy_default_is_full_fallback(self):
        cfg = parse_double_sparsity_config(_valid_payload())
        self.assertEqual(cfg.selector_width_overflow_policy, "full_fallback")

    def test_selector_width_overflow_policy_fail_closed_parses(self):
        payload = (
            '{"channel_mask_path": "/tmp/cm.safetensors", '
            '"selector_width_buckets": [4608], '
            '"selector_width_overflow_policy": "fail_closed"}'
        )
        cfg = parse_double_sparsity_config(payload)
        self.assertEqual(cfg.selector_width_overflow_policy, "fail_closed")
        self.assertEqual(cfg.selector_width_buckets, [4608])

    def test_fail_closed_requires_a_compact_bucket(self):
        payload = (
            '{"channel_mask_path": "/tmp/cm.safetensors", '
            '"selector_width_buckets": [], '
            '"selector_width_overflow_policy": "fail_closed"}'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_double_sparsity_config(payload)
        self.assertIn("fail_closed", str(ctx.exception))

    def test_invalid_overflow_policy_rejected(self):
        payload = (
            '{"channel_mask_path": "/tmp/cm.safetensors", '
            '"selector_width_overflow_policy": "bogus"}'
        )
        with self.assertRaises(ValueError) as ctx:
            parse_double_sparsity_config(payload)
        self.assertIn("selector_width_overflow_policy", str(ctx.exception))


class TestDoubleSparsitySelectorABI(unittest.TestCase):
    def setUp(self):
        cfg = parse_double_sparsity_config(_valid_payload())
        self.selector = DoubleSparsitySelector(
            config=cfg,
            num_local_heads=16,
            head_dim=128,
            device=torch.device("cpu"),
        )

    def test_shapes_and_dtypes(self):
        queries = torch.zeros(3, 16, 128)
        req_pool = torch.tensor([0, 1, 2], dtype=torch.int32)
        seq_lens = torch.tensor([100, 200, 300], dtype=torch.int32)
        sparse_mask = torch.zeros(3, 10, dtype=torch.int32)

        selected_indices, valid_lengths = self.selector.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool,
            sparse_mask=sparse_mask,
            seq_lens=seq_lens,
        )

        self.assertEqual(selected_indices.dtype, torch.int32)
        self.assertEqual(valid_lengths.dtype, torch.int32)
        self.assertEqual(tuple(selected_indices.shape), (3, 2048))
        self.assertEqual(tuple(valid_lengths.shape), (3,))

    def test_sequence_ascending_invariant(self):
        queries = torch.zeros(2, 16, 128)
        req_pool = torch.tensor([0, 1], dtype=torch.int32)
        seq_lens = torch.tensor([100, 4096], dtype=torch.int32)
        sparse_mask = torch.zeros(2, 70, dtype=torch.int32)

        selected_indices, valid_lengths = self.selector.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool,
            sparse_mask=sparse_mask,
            seq_lens=seq_lens,
        )

        for row in range(selected_indices.shape[0]):
            length = int(valid_lengths[row])
            unpadded = selected_indices[row, :length].tolist()
            padding = selected_indices[row, length:].tolist()
            self.assertTrue(
                all(unpadded[i] < unpadded[i + 1] for i in range(len(unpadded) - 1)),
                f"row {row} not strictly ascending: {unpadded}",
            )
            self.assertTrue(all(v == -1 for v in padding), f"row {row} padding")

    def test_valid_lengths_clipped_to_max_top_k(self):
        queries = torch.zeros(1, 16, 128)
        req_pool = torch.tensor([0], dtype=torch.int32)
        seq_lens = torch.tensor([10_000_000], dtype=torch.int32)
        sparse_mask = torch.zeros(1, 1, dtype=torch.int32)
        _, valid_lengths = self.selector.retrieve_topk(
            queries=queries,
            layer_id=0,
            req_pool_indices=req_pool,
            sparse_mask=sparse_mask,
            seq_lens=seq_lens,
        )
        self.assertEqual(int(valid_lengths[0]), 2048)


class TestValidator(unittest.TestCase):
    def _args(self, **kwargs):
        defaults = dict(
            enable_double_sparsity=False,
            enable_hisparse=False,
            enable_hierarchical_cache=False,
            disaggregation_mode=None,
            double_sparsity_config=None,
            page_size=64,
        )
        defaults.update(kwargs)
        return SimpleNamespace(**defaults)

    def test_disabled_is_no_op(self):
        validate_double_sparsity(self._args(enable_double_sparsity=False))

    def test_mutual_exclusion_with_hisparse(self):
        args = self._args(enable_double_sparsity=True, enable_hisparse=True)
        with self.assertRaises(ValueError) as ctx:
            validate_double_sparsity(args)
        self.assertIn("mutually exclusive", str(ctx.exception).lower())

    def test_missing_config(self):
        args = self._args(enable_double_sparsity=True, double_sparsity_config=None)
        os.environ["SGLANG_DS_ALLOW_NO_ADAPTER"] = "1"
        try:
            with self.assertRaises(ValueError) as ctx:
                validate_double_sparsity(args)
            self.assertIn("channel_mask_path", str(ctx.exception))
        finally:
            os.environ.pop("SGLANG_DS_ALLOW_NO_ADAPTER", None)

    def test_disaggregation_rejected(self):
        args = self._args(
            enable_double_sparsity=True,
            disaggregation_mode="decode",
            double_sparsity_config=_valid_payload(),
        )
        os.environ["SGLANG_DS_ALLOW_NO_ADAPTER"] = "1"
        try:
            with self.assertRaises(ValueError) as ctx:
                validate_double_sparsity(args)
            self.assertIn("disaggregation", str(ctx.exception).lower())
        finally:
            os.environ.pop("SGLANG_DS_ALLOW_NO_ADAPTER", None)

    def test_hierarchical_cache_rejected(self):
        # The hierarchical-cache check fires before the payload check, so no
        # adapter/payload is needed to reach it.
        args = self._args(
            enable_double_sparsity=True,
            enable_hierarchical_cache=True,
        )
        with self.assertRaises(ValueError) as ctx:
            validate_double_sparsity(args)
        self.assertIn("hierarchical", str(ctx.exception).lower())

    def test_hierarchical_cache_allowed_without_double_sparsity(self):
        # DSA-native (Double Sparsity off) + hierarchical cache must NOT be
        # rejected by the Double Sparsity validator (early-return no-op).
        validate_double_sparsity(
            self._args(enable_double_sparsity=False, enable_hierarchical_cache=True)
        )

    def test_page_size_mismatch(self):
        args = self._args(
            enable_double_sparsity=True,
            double_sparsity_config=_valid_payload(),
            page_size=32,
        )
        os.environ["SGLANG_DS_ALLOW_NO_ADAPTER"] = "1"
        try:
            with self.assertRaises(ValueError) as ctx:
                validate_double_sparsity(args)
            self.assertIn("page_size", str(ctx.exception))
        finally:
            os.environ.pop("SGLANG_DS_ALLOW_NO_ADAPTER", None)

    def test_valid_path(self):
        import os as _os
        import tempfile

        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            save_channel_mask,
        )

        # head_dim=128 below, so channel indices must be in [0, 128).
        sel_t = torch.randint(0, 128, (2, 4, 16), dtype=torch.int32)
        w_t = torch.randn(2, 4, 16, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = _os.path.join(tmp, "cm.safetensors")
            save_channel_mask(
                path,
                sel_t,
                w_t,
                dtype="fp8_e4m3",
                head_dim=128,
                page_size=64,
                label_dim=16,
                created_at="2026-05-20T00:00:00Z",
            )
            args = self._args(
                enable_double_sparsity=True,
                double_sparsity_config=_valid_payload(path),
                page_size=64,
                kv_cache_dtype="fp8_e4m3",
                dsa_prefill_backend="flashmla_kv",
                dsa_decode_backend="flashmla_kv",
                disable_radix_cache=True,
            )
            os.environ["SGLANG_DS_ALLOW_PLACEHOLDER"] = "1"
            os.environ["SGLANG_DS_ALLOW_NO_ADAPTER"] = "1"
            try:
                validate_double_sparsity(args)
                self.assertIsInstance(
                    args._double_sparsity_parsed_config, DoubleSparsityConfig
                )
            finally:
                os.environ.pop("SGLANG_DS_ALLOW_PLACEHOLDER", None)
                os.environ.pop("SGLANG_DS_ALLOW_NO_ADAPTER", None)

    def test_capability_check_uses_existing_model_config_symbol(self):
        """Regression: the DS validator's capability check imported a stale
        `is_deepseek_nsa` after model_config renamed it to `is_deepseek_dsa`,
        raising ImportError at server startup (DS boot crashed before model
        load). Lock that the validator references the existing symbol and that
        the symbol classifies DeepSeek-V3.2 as a DSA model."""
        import inspect

        from sglang.srt.configs.model_config import is_deepseek_dsa
        from sglang.srt.layers.attention.double_sparsity import validator as _v

        src = inspect.getsource(_v)
        self.assertNotIn("is_deepseek_nsa", src)
        self.assertIn("is_deepseek_dsa", src)
        self.assertTrue(
            is_deepseek_dsa(
                SimpleNamespace(
                    architectures=["DeepseekV32ForCausalLM"], index_topk=2048
                )
            )
        )
        self.assertFalse(
            is_deepseek_dsa(
                SimpleNamespace(architectures=["LlamaForCausalLM"], index_topk=None)
            )
        )

    def test_marks_channel_mask_valid_on_success(self):
        """a healthy validator pass must set the ``sglang_double_sparsity_channel_mask_valid`` gauge to 1."""

        try:
            import prometheus_client  # noqa: F401
        except ImportError:
            self.skipTest("prometheus_client not installed")
        import os as _os
        import tempfile

        from sglang.srt.layers.attention.double_sparsity import metrics as m
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            save_channel_mask,
        )

        m.reset_for_testing()
        sel_t = torch.randint(0, 128, (2, 4, 16), dtype=torch.int32)
        w_t = torch.randn(2, 4, 16, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = _os.path.join(tmp, "cm.safetensors")
            save_channel_mask(
                path,
                sel_t,
                w_t,
                dtype="fp8_e4m3",
                head_dim=128,
                page_size=64,
                label_dim=16,
                created_at="2026-05-20T00:00:00Z",
            )
            args = self._args(
                enable_double_sparsity=True,
                double_sparsity_config=_valid_payload(path),
                page_size=64,
                kv_cache_dtype="fp8_e4m3",
                dsa_prefill_backend="flashmla_kv",
                dsa_decode_backend="flashmla_kv",
                disable_radix_cache=True,
            )
            os.environ["SGLANG_DS_ALLOW_PLACEHOLDER"] = "1"
            os.environ["SGLANG_DS_ALLOW_NO_ADAPTER"] = "1"
            try:
                validate_double_sparsity(args)
            finally:
                os.environ.pop("SGLANG_DS_ALLOW_PLACEHOLDER", None)
                os.environ.pop("SGLANG_DS_ALLOW_NO_ADAPTER", None)
        gauge = m._metric_objs.get("channel_mask_valid")
        self.assertIsNotNone(gauge, "channel_mask_valid gauge should be registered")
        self.assertEqual(
            gauge._value.get(), 1, "gauge must read 1 after a successful validation"
        )
        m.reset_for_testing()


class TestPlaceholderGuard(unittest.TestCase):
    def setUp(self):
        cfg = parse_double_sparsity_config(_valid_payload())
        self.selector = DoubleSparsitySelector(
            config=cfg,
            num_local_heads=16,
            head_dim=128,
            device=torch.device("cpu"),
        )

    def test_placeholder_refuses_serving(self):
        with self.assertRaises(RuntimeError) as ctx:
            assert_real_selector_or_placeholder_allowed(self.selector)
        self.assertIn("placeholder", str(ctx.exception).lower())

    def test_real_selector_passes(self):
        class _Real:
            IS_PLACEHOLDER = False

        assert_real_selector_or_placeholder_allowed(_Real())

    def test_real_selector_after_direct_toggle(self):
        # Tests can flip a placeholder selector to real mode by setting
        # IS_PLACEHOLDER = False directly when they need the guard to
        # pass without going through bind_runtime_data.
        self.selector.IS_PLACEHOLDER = False
        assert_real_selector_or_placeholder_allowed(self.selector)


class TestSelectTopkIndicesHookBranch(unittest.TestCase):
    """Exercise the ``_select_topk_indices`` config-gated branch directly.

    Builds an instance through ``object.__new__`` so we can wire only the
    fields the branch reads, avoiding the full DeepseekV2AttentionMLA
    constructor (which depends on distributed init).
    """

    def _make_attn(self, *, use_ds: bool):
        from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

        attn = object.__new__(DeepseekV2AttentionMLA)
        attn.use_double_sparsity = use_ds
        attn.double_sparsity_selector = None
        attn.indexer = MagicMock(
            return_value=torch.tensor([7, 8, 9], dtype=torch.int32)
        )
        if use_ds:
            cfg = parse_double_sparsity_config(_valid_payload())
            attn.double_sparsity_selector = DoubleSparsitySelector(
                config=cfg,
                num_local_heads=16,
                head_dim=128,
                device=torch.device("cpu"),
            )
        return attn

    def _make_attn_real(self):
        """Build a hook fixture whose DS selector is in real mode (not
        placeholder), via direct IS_PLACEHOLDER toggle. No env vars."""
        attn = self._make_attn(use_ds=True)
        attn.double_sparsity_selector.IS_PLACEHOLDER = False
        return attn

    def test_native_branch_calls_indexer(self):
        attn = self._make_attn(use_ds=False)
        result = attn._select_topk_indices(
            x=torch.zeros(2, 16, 128),
            q_lora=torch.zeros(2, 16, 128),
            positions=torch.zeros(2, dtype=torch.int32),
            forward_batch=SimpleNamespace(),
            layer_id=0,
        )
        attn.indexer.assert_called_once()
        self.assertTrue(torch.equal(result, torch.tensor([7, 8, 9], dtype=torch.int32)))


class TestPageTableAdapter(unittest.TestCase):
    """Verify ``logical_to_physical`` correctly maps logical token positions to
    physical KV-cache slot indices via req_to_token gather (token-level adapter).
    """

    def _adapter(self):
        from sglang.srt.layers.attention.double_sparsity.page_table_adapter import (
            logical_to_physical,
        )

        return logical_to_physical

    def test_basic_req_to_token_gather(self):
        """Logical positions are gathered from req_to_token; -1 padding preserved."""
        adapter = self._adapter()
        req_to_token = torch.tensor(
            [[10, 20, 30, 40, 50, 60, 70, 80]], dtype=torch.int32
        )
        selected = torch.tensor([[0, 2, 4, -1, -1, -1]], dtype=torch.int32)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)
        out = torch.full_like(selected, -1)
        error_count = adapter(selected, req_pool_indices, req_to_token, out)
        self.assertEqual(out[0, 0].item(), 10)  # req_to_token[0, 0]
        self.assertEqual(out[0, 1].item(), 30)  # req_to_token[0, 2]
        self.assertEqual(out[0, 2].item(), 50)  # req_to_token[0, 4]
        self.assertEqual(out[0, 3].item(), -1)  # padding preserved
        self.assertEqual(error_count, 0)

    def test_padding_minus_one_preserved(self):
        """Positions equal to -1 must remain -1 in the output."""
        adapter = self._adapter()
        req_to_token = torch.arange(100, dtype=torch.int32).unsqueeze(0)
        selected = torch.tensor([[-1, -1, -1]], dtype=torch.int32)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)
        out = torch.zeros_like(selected)
        error_count = adapter(selected, req_pool_indices, req_to_token, out)
        self.assertTrue(torch.all(out == -1).item())
        self.assertEqual(error_count, 0)

    def test_bad_pool_index_row_gets_minus_one(self):
        """Rows where req_pool_indices is out of range for req_to_token get all -1."""
        adapter = self._adapter()
        req_to_token = torch.tensor([[10, 20, 30]], dtype=torch.int32)  # 1 pool row
        selected = torch.tensor([[0, 1, -1]], dtype=torch.int32)
        req_pool_indices = torch.tensor([5], dtype=torch.int32)  # bad: only 1 pool row
        out = torch.zeros_like(selected)
        error_count = adapter(selected, req_pool_indices, req_to_token, out)
        self.assertTrue(torch.all(out == -1).item())
        self.assertEqual(error_count, 1)

    def test_error_count_matches_bad_pool_rows(self):
        """error_count equals the number of out-of-range req_pool_indices rows."""
        adapter = self._adapter()
        req_to_token = torch.arange(20, dtype=torch.int32).reshape(2, 10)
        selected = torch.tensor([[0, 1, -1], [0, 2, -1]], dtype=torch.int32)
        req_pool_indices = torch.tensor([0, 99], dtype=torch.int32)  # row 1 bad
        out = torch.full_like(selected, -1)
        error_count = adapter(selected, req_pool_indices, req_to_token, out)
        self.assertEqual(error_count, 1)

    def test_empty_batch_returns_zero(self):
        """bs=0 gives error_count=0 and out remains -1."""
        adapter = self._adapter()
        req_to_token = torch.zeros((1, 10), dtype=torch.int32)
        selected = torch.zeros((0, 4), dtype=torch.int32)
        req_pool_indices = torch.zeros((0,), dtype=torch.int32)
        out = torch.full((0, 4), -1, dtype=torch.int32)
        error_count = adapter(selected, req_pool_indices, req_to_token, out)
        self.assertEqual(error_count, 0)
        self.assertEqual(out.shape[0], 0)

    def test_out_tensor_modified_in_place(self):
        """The pre-allocated out tensor is written in-place."""
        adapter = self._adapter()
        req_to_token = torch.tensor([[100, 200, 300]], dtype=torch.int32)
        selected = torch.tensor([[0, 2, -1]], dtype=torch.int32)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)
        out = torch.full((1, 3), -99, dtype=torch.int32)
        original_data_ptr = out.data_ptr()
        error_count = adapter(selected, req_pool_indices, req_to_token, out)
        self.assertEqual(out.data_ptr(), original_data_ptr)  # same storage
        self.assertEqual(out[0, 0].item(), 100)
        self.assertEqual(out[0, 1].item(), 300)
        self.assertEqual(out[0, 2].item(), -1)
        self.assertEqual(error_count, 0)

    def test_all_bad_pool_gives_all_minus_one(self):
        """When all rows have bad pool indices, output is all -1."""
        adapter = self._adapter()
        req_to_token = torch.zeros((1, 10), dtype=torch.int32)
        selected = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
        req_pool_indices = torch.tensor([99, 100], dtype=torch.int32)  # all bad
        out = torch.zeros((2, 3), dtype=torch.int32)
        error_count = adapter(selected, req_pool_indices, req_to_token, out)
        self.assertTrue(torch.all(out == -1).item())
        self.assertEqual(error_count, 2)

    def test_mixed_valid_invalid_pool(self):
        """Valid pool rows get correct physical slots; invalid rows get -1."""
        adapter = self._adapter()
        req_to_token = torch.tensor([[5, 10, 15, 20]], dtype=torch.int32)
        selected = torch.tensor([[0, 2, -1], [1, 3, -1]], dtype=torch.int32)
        req_pool_indices = torch.tensor([0, 99], dtype=torch.int32)  # row 1 bad
        out = torch.full((2, 3), -99, dtype=torch.int32)
        error_count = adapter(selected, req_pool_indices, req_to_token, out)
        self.assertEqual(out[0, 0].item(), 5)  # req_to_token[0, 0]
        self.assertEqual(out[0, 1].item(), 15)  # req_to_token[0, 2]
        self.assertEqual(out[0, 2].item(), -1)  # padding
        self.assertTrue(torch.all(out[1] == -1).item())  # bad pool → all -1
        self.assertEqual(error_count, 1)

    def test_physical_slots_from_req_to_token(self):
        """Physical slot values are exactly req_to_token[pool, position]."""
        adapter = self._adapter()
        torch.manual_seed(42)
        req_to_token = torch.randint(0, 65536, (4, 32), dtype=torch.int32)
        selected = torch.tensor([[0, 5, 10, 15, -1]], dtype=torch.int32)
        req_pool_indices = torch.tensor([2], dtype=torch.int32)
        out = torch.full((1, 5), -1, dtype=torch.int32)
        error_count = adapter(selected, req_pool_indices, req_to_token, out)
        self.assertEqual(out[0, 0].item(), req_to_token[2, 0].item())
        self.assertEqual(out[0, 1].item(), req_to_token[2, 5].item())
        self.assertEqual(out[0, 2].item(), req_to_token[2, 10].item())
        self.assertEqual(out[0, 3].item(), req_to_token[2, 15].item())
        self.assertEqual(out[0, 4].item(), -1)
        self.assertEqual(error_count, 0)

    def test_multi_pool_rows_each_use_own_pool(self):
        """Each batch row uses its own pool row from req_to_token."""
        adapter = self._adapter()
        req_to_token = torch.tensor([[1, 2, 3, 4], [10, 20, 30, 40]], dtype=torch.int32)
        selected = torch.tensor([[0, 3, -1], [1, 2, -1]], dtype=torch.int32)
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int32)
        out = torch.full((2, 3), -1, dtype=torch.int32)
        error_count = adapter(selected, req_pool_indices, req_to_token, out)
        self.assertEqual(out[0, 0].item(), 1)  # req_to_token[0, 0]
        self.assertEqual(out[0, 1].item(), 4)  # req_to_token[0, 3]
        self.assertEqual(out[1, 0].item(), 20)  # req_to_token[1, 1]
        self.assertEqual(out[1, 1].item(), 30)  # req_to_token[1, 2]
        self.assertEqual(error_count, 0)

    def test_negative_pool_index_treated_as_error(self):
        """Negative req_pool_indices are out-of-range → those rows get -1."""
        adapter = self._adapter()
        req_to_token = torch.arange(10, dtype=torch.int32).unsqueeze(0)
        selected = torch.tensor([[0, 1, -1]], dtype=torch.int32)
        req_pool_indices = torch.tensor([-1], dtype=torch.int32)
        out = torch.zeros((1, 3), dtype=torch.int32)
        error_count = adapter(selected, req_pool_indices, req_to_token, out)
        self.assertTrue(torch.all(out == -1).item())
        self.assertEqual(error_count, 1)

    def test_empty_selection_all_padding(self):
        """When all positions are -1, output is all -1 with no errors."""
        adapter = self._adapter()
        req_to_token = torch.arange(20, dtype=torch.int32).unsqueeze(0)
        selected = torch.full((1, 5), -1, dtype=torch.int32)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)
        out = torch.zeros((1, 5), dtype=torch.int32)
        error_count = adapter(selected, req_pool_indices, req_to_token, out)
        self.assertTrue(torch.all(out == -1).item())
        self.assertEqual(error_count, 0)

    def test_output_dtype_is_int32(self):
        """Output tensor retains int32 dtype (type-stable adapter)."""
        adapter = self._adapter()
        req_to_token = torch.arange(10, dtype=torch.int32).unsqueeze(0)
        selected = torch.tensor([[0, 2, -1]], dtype=torch.int32)
        req_pool_indices = torch.tensor([0], dtype=torch.int32)
        out = torch.full((1, 3), -1, dtype=torch.int32)
        adapter(selected, req_pool_indices, req_to_token, out)
        self.assertEqual(out.dtype, torch.int32)


class TestChannelMaskLoader(unittest.TestCase):
    def _make_payload(self, *, L=4, H=4, label_dim=16, head_dim=128):
        sel = torch.randint(0, head_dim, (L, H, label_dim), dtype=torch.int32)
        w = torch.randn(L, H, label_dim, dtype=torch.float32)
        return sel, w

    def test_roundtrip(self):
        import os
        import tempfile

        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            load_channel_mask,
            save_channel_mask,
        )

        sel, w = self._make_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cm.safetensors")
            h = save_channel_mask(
                path,
                sel,
                w,
                dtype="fp8_e4m3",
                head_dim=128,
                page_size=64,
                label_dim=16,
                created_at="2026-05-20T00:00:00Z",
            )
            cm = load_channel_mask(path)
        self.assertEqual(cm.content_sha256, h)
        self.assertEqual(cm.dtype, "fp8_e4m3")
        self.assertEqual(cm.head_dim, 128)
        self.assertEqual(cm.page_size, 64)
        self.assertEqual(cm.label_dim, 16)
        self.assertTrue(torch.equal(cm.channel_selection, sel))

    def test_content_hash_mismatch(self):
        import os
        import tempfile

        from safetensors import safe_open
        from safetensors.torch import save_file

        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            compute_content_sha256,
            load_channel_mask,
            save_channel_mask,
        )

        sel, w = self._make_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cm.safetensors")
            save_channel_mask(
                path,
                sel,
                w,
                dtype="fp8_e4m3",
                head_dim=128,
                page_size=64,
                label_dim=16,
                created_at="2026-05-20T00:00:00Z",
            )
            # Tamper: rewrite with a metadata content_sha256 from a different payload.
            tampered_sel = sel.clone()
            tampered_sel[0, 0, 0] = 999
            tampered_hash = compute_content_sha256(tampered_sel, w)
            # Read original metadata then resave with bogus hash
            with safe_open(path, framework="pt") as f:
                tensors = {k: f.get_tensor(k) for k in f.keys()}
                md = dict(f.metadata() or {})
            md["content_sha256"] = tampered_hash
            save_file(tensors, path, metadata=md)
            with self.assertRaises(ValueError) as ctx:
                load_channel_mask(path)
            self.assertIn("hash mismatch", str(ctx.exception))

    def test_missing_file(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            load_channel_mask,
        )

        with self.assertRaises(FileNotFoundError):
            load_channel_mask("/nonexistent/path.safetensors")

    def test_load_rejects_out_of_range_channel_indices(self):
        """a content-hash-valid file whose
        channel_selection has values >= head_dim must be rejected at load.
        """

        import os
        import tempfile

        from safetensors.torch import save_file

        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            compute_content_sha256,
            load_channel_mask,
        )

        sel = torch.zeros(1, 2, 4, dtype=torch.int32)
        # Plant an out-of-range index: head_dim=128 in the metadata below.
        sel[0, 0, 0] = 200
        w = torch.zeros(1, 2, 4, dtype=torch.float32)
        content = compute_content_sha256(sel, w)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.safetensors")
            save_file(
                {"channel_selection": sel, "channel_weights": w},
                path,
                metadata={
                    "schema_version": "1",
                    "dtype": "fp8_e4m3",
                    "head_dim": "128",
                    "page_size": "64",
                    "label_dim": "4",
                    "created_at": "2026-05-20T00:00:00Z",
                    "content_sha256": content,
                },
            )
            with self.assertRaises(ValueError) as ctx:
                load_channel_mask(path)
            msg = str(ctx.exception)
            self.assertIn("head_dim", msg)
            self.assertIn("out of range", msg)

    def test_validate_runtime_mismatches(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask,
            validate_against_runtime,
        )

        mask = ChannelMask(
            channel_selection=torch.zeros(2, 2, 4, dtype=torch.int32),
            channel_weights=torch.zeros(2, 2, 4, dtype=torch.float32),
            schema_version="1",
            dtype="fp8_e4m3",
            head_dim=128,
            page_size=64,
            label_dim=4,
            content_sha256="x",
        )
        validate_against_runtime(
            mask,
            server_kv_cache_dtype="fp8_e4m3",
            server_page_size=64,
            server_label_dim=4,
            model_head_dim=128,
        )
        with self.assertRaises(ValueError):
            validate_against_runtime(
                mask,
                server_kv_cache_dtype="bfloat16",
                server_page_size=64,
                server_label_dim=4,
                model_head_dim=128,
            )


class TestChannelMaskSlicePerRank(unittest.TestCase):
    """TP head sharding helper."""

    def test_slice_per_rank_returns_local_block(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask,
            slice_per_rank,
        )

        sel = torch.arange(2 * 16 * 8, dtype=torch.int32).reshape(2, 16, 8)
        wts = torch.arange(2 * 16 * 8, dtype=torch.float32).reshape(2, 16, 8)
        mask = ChannelMask(
            channel_selection=sel,
            channel_weights=wts,
            schema_version="1",
            dtype="fp8_e4m3",
            head_dim=128,
            page_size=64,
            label_dim=8,
            content_sha256="abc",
        )
        # TP=4 → num_local_heads=4; rank 2 owns heads [8, 12).
        sliced = slice_per_rank(mask, num_local_heads=4, rank=2, tp_size=4)
        self.assertEqual(tuple(sliced.channel_selection.shape), (2, 4, 8))
        self.assertTrue(torch.equal(sliced.channel_selection, sel[:, 8:12, :]))
        self.assertTrue(torch.equal(sliced.channel_weights, wts[:, 8:12, :]))
        # Metadata is carried forward unchanged.
        self.assertEqual(sliced.content_sha256, "abc")
        self.assertEqual(sliced.head_dim, 128)
        self.assertEqual(sliced.label_dim, 8)

    def test_slice_per_rank_rejects_uneven_split(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask,
            slice_per_rank,
        )

        mask = ChannelMask(
            channel_selection=torch.zeros(1, 10, 4, dtype=torch.int32),
            channel_weights=torch.zeros(1, 10, 4, dtype=torch.float32),
            schema_version="1",
            dtype="fp8_e4m3",
            head_dim=128,
            page_size=64,
            label_dim=4,
            content_sha256="x",
        )
        with self.assertRaises(ValueError):
            slice_per_rank(mask, num_local_heads=4, rank=0, tp_size=2)

    def test_bind_rejects_unsliced_full_mask(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask,
        )

        cfg = parse_double_sparsity_config(_valid_payload())
        sel = DoubleSparsitySelector(
            config=cfg,
            num_local_heads=4,
            head_dim=128,
            device=torch.device("cpu"),
        )
        sel.absorbed_w_sel = torch.zeros(4, 8, 16)
        # Mask is still at H_full=32 (un-sliced) — must be rejected.
        full_mask = ChannelMask(
            channel_selection=torch.zeros(2, 32, 8, dtype=torch.int32),
            channel_weights=torch.zeros(2, 32, 8, dtype=torch.float32),
            schema_version="1",
            dtype="fp8_e4m3",
            head_dim=128,
            page_size=64,
            label_dim=8,
            content_sha256="x",
        )
        with self.assertRaises(ValueError) as ctx:
            sel.bind_runtime_data(full_mask)
        self.assertIn("slice_per_rank", str(ctx.exception))


class TestVerifyBindShapes(unittest.TestCase):
    """Bind-time shape gate: a calibrated mask must match the running model's
    no-PE head width / head count / layer count, or DS hard-errors naming the
    field instead of silently selecting the wrong channels.

    Parameterized across the narrower (128/128) and wider (192/256) MLA shapes
    so a change that hardens one shape cannot silently drop the other.
    """

    # (nope_head_dim, v_head_dim, num_heads, num_layers, label_dim)
    SHAPES = {
        "narrow_128": (128, 128, 128, 61, 16),
        "wide_192": (192, 256, 96, 78, 32),
    }

    def _make_mask(
        self,
        *,
        head_dim: int,
        label_dim: int,
        num_layers: int,
        num_heads: int,
        max_index: int = None,
        weights_shape=None,
    ):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask,
        )

        hi = head_dim if max_index is None else max_index
        sel = torch.randint(
            0, max(hi, 1), (num_layers, num_heads, label_dim), dtype=torch.int32
        )
        w = torch.ones(
            weights_shape or (num_layers, num_heads, label_dim), dtype=torch.float32
        )
        return ChannelMask(
            channel_selection=sel,
            channel_weights=w,
            schema_version="ds_channel_mask_v1",
            dtype="fp8_e4m3",
            head_dim=head_dim,
            page_size=64,
            label_dim=label_dim,
            content_sha256="0" * 64,
            created_at="2026-06-07T00:00:00Z",
        )

    def _verify(self, mask, *, nope, num_heads, num_layers, label_dim, tp_size=1):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            verify_bind_shapes,
        )

        verify_bind_shapes(
            mask,
            model_nope_head_dim=nope,
            num_local_heads=num_heads // tp_size,
            tp_size=tp_size,
            num_hidden_layers=num_layers,
            server_page_size=64,
            server_label_dim=label_dim,
            server_kv_cache_dtype="fp8_e4m3",
        )

    def test_matching_mask_passes_both_shapes(self):
        for name, (nope, _v, h, L, ld) in self.SHAPES.items():
            with self.subTest(shape=name):
                mask = self._make_mask(
                    head_dim=nope, label_dim=ld, num_layers=L, num_heads=h
                )
                # Must not raise.
                self._verify(mask, nope=nope, num_heads=h, num_layers=L, label_dim=ld)

    def test_matching_mask_passes_with_tp_split(self):
        nope, _v, h, L, ld = self.SHAPES["wide_192"]
        mask = self._make_mask(head_dim=nope, label_dim=ld, num_layers=L, num_heads=h)
        self._verify(
            mask, nope=nope, num_heads=h, num_layers=L, label_dim=ld, tp_size=8
        )

    def test_narrow_mask_on_wide_model_hard_errors_naming_head_dim(self):
        # A mask calibrated for the 128 no-PE width loaded against a 192 model:
        # indices stay in range (no crash) but the head_dim equality must fail.
        _nope, _v, h, L, ld = self.SHAPES["wide_192"]
        mask = self._make_mask(
            head_dim=128, label_dim=ld, num_layers=L, num_heads=h, max_index=128
        )
        with self.assertRaises(ValueError) as cm:
            self._verify(mask, nope=192, num_heads=h, num_layers=L, label_dim=ld)
        self.assertIn("head_dim", str(cm.exception))

    def test_index_out_of_nope_range_hard_errors(self):
        nope, _v, h, L, ld = self.SHAPES["wide_192"]
        mask = self._make_mask(head_dim=nope, label_dim=ld, num_layers=L, num_heads=h)
        # Force a selection index past the no-PE width.
        mask.channel_selection[0, 0, 0] = nope + 5
        with self.assertRaises(ValueError) as cm:
            self._verify(mask, nope=nope, num_heads=h, num_layers=L, label_dim=ld)
        self.assertIn("max index", str(cm.exception))

    def test_layer_count_mismatch_hard_errors(self):
        nope, _v, h, L, ld = self.SHAPES["wide_192"]
        mask = self._make_mask(
            head_dim=nope, label_dim=ld, num_layers=L - 1, num_heads=h
        )
        with self.assertRaises(ValueError) as cm:
            self._verify(mask, nope=nope, num_heads=h, num_layers=L, label_dim=ld)
        self.assertIn("layers", str(cm.exception))

    def test_head_count_mismatch_hard_errors(self):
        nope, _v, h, L, ld = self.SHAPES["wide_192"]
        mask = self._make_mask(head_dim=nope, label_dim=ld, num_layers=L, num_heads=h)
        with self.assertRaises(ValueError) as cm:
            self._verify(mask, nope=nope, num_heads=h + 8, num_layers=L, label_dim=ld)
        self.assertIn("num_heads", str(cm.exception))

    def test_label_dim_mismatch_hard_errors(self):
        nope, _v, h, L, ld = self.SHAPES["wide_192"]
        mask = self._make_mask(head_dim=nope, label_dim=ld, num_layers=L, num_heads=h)
        with self.assertRaises(ValueError) as cm:
            self._verify(mask, nope=nope, num_heads=h, num_layers=L, label_dim=ld + 1)
        self.assertIn("label_dim", str(cm.exception))

    def test_weights_shape_mismatch_hard_errors(self):
        nope, _v, h, L, ld = self.SHAPES["wide_192"]
        mask = self._make_mask(
            head_dim=nope,
            label_dim=ld,
            num_layers=L,
            num_heads=h,
            weights_shape=(L, h, ld + 2),
        )
        with self.assertRaises(ValueError) as cm:
            self._verify(mask, nope=nope, num_heads=h, num_layers=L, label_dim=ld)
        self.assertIn("channel_weights", str(cm.exception))

    def test_auto_kv_dtype_skips_dtype_leg(self):
        # When the server dtype is still "auto", the dtype mismatch leg is a
        # no-op (head_dim is the real check); a matching mask still passes.
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            verify_bind_shapes,
        )

        nope, _v, h, L, ld = self.SHAPES["wide_192"]
        mask = self._make_mask(head_dim=nope, label_dim=ld, num_layers=L, num_heads=h)
        verify_bind_shapes(
            mask,
            model_nope_head_dim=nope,
            num_local_heads=h,
            tp_size=1,
            num_hidden_layers=L,
            server_page_size=64,
            server_label_dim=ld,
            server_kv_cache_dtype="auto",
        )


class TestDSIndexerCacheGate(unittest.TestCase):
    """DS-mode indexer index-k sidecar gate on DSATokenToKVPool, the matching cell-size
    drop in the configurator, and the hierarchical-cache host-sidecar guard. The gate is
    DS-only; DSA-native and HiSparse keep the buffer."""

    def _pool(self, gate: bool):
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        return DSATokenToKVPool(
            size=256,
            page_size=64,
            kv_lora_rank=512,
            dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=64,
            layer_num=2,
            device="cpu",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=656,
            start_layer=0,
            end_layer=2,
            gate_index_k_cache=gate,
        )

    def test_gated_pool_skips_index_k_allocation(self):
        p = self._pool(gate=True)
        self.assertTrue(p.gate_index_k_cache)
        self.assertIsNone(p.index_k_with_scale_buffer)

    def test_ungated_pool_allocates_index_k(self):
        p = self._pool(gate=False)
        self.assertFalse(p.gate_index_k_cache)
        self.assertIsNotNone(p.index_k_with_scale_buffer)
        self.assertEqual(len(p.index_k_with_scale_buffer), 2)

    def test_gated_data_accessors_fail_loudly(self):
        p = self._pool(gate=True)
        idx = torch.zeros(1, dtype=torch.int64)
        for call in (
            lambda: p.get_index_k_with_scale_buffer(0),
            lambda: p.get_index_k_continuous(0, 1, idx),
            lambda: p.get_index_k_scale_continuous(0, 1, idx),
            lambda: p.set_index_k_scale_buffer(0, idx, idx, idx),
        ):
            with self.assertRaises(RuntimeError):
                call()

    def test_gated_management_methods_are_none_safe(self):
        p = self._pool(gate=True)
        # size accounting omits the (absent) index-k sidecar; no crash.
        self.assertGreater(p.get_kv_size_bytes(), 0)
        # state transfer reports no index-k buffers.
        self.assertEqual(p.get_state_buf_infos(), ([], [], []))
        # offload round-trip carries index_k=None and restores without touching it.
        idx = torch.arange(0, 128, dtype=torch.int64)
        cpu = p.get_cpu_copy(idx)
        self.assertIn("index_k", cpu)
        self.assertIsNone(cpu["index_k"])
        p.load_cpu_copy(cpu, idx)  # must not raise

    def test_indexer_host_rejects_gated_pool(self):
        # Hierarchical cache builds a DSA indexer host sidecar from the device
        # pool; a gated pool has no index-k buffer, so construction must fail
        # loudly (defense-in-depth behind the server-args validator) rather than
        # hit a NoneType iteration deep inside init_kv_buffer.
        from sglang.srt.mem_cache.memory_pool_host import DSAIndexerPoolHost

        p = self._pool(gate=True)
        with self.assertRaises(RuntimeError) as ctx:
            DSAIndexerPoolHost(p, None, "layer_first")
        self.assertIn("gate_index_k_cache", str(ctx.exception))

    def test_indexer_host_does_not_guard_ungated_pool(self):
        # The gate guard is the first statement in __init__; for an ungated
        # (DSA-native) pool it must NOT fire. Construction then fails later on
        # the stub anchor_host, proving execution passed the guard untouched.
        from sglang.srt.mem_cache.memory_pool_host import DSAIndexerPoolHost

        p = self._pool(gate=False)
        with self.assertRaises(AttributeError):
            DSAIndexerPoolHost(p, None, "layer_first")

    def _cell_size(self, *, ds_on: bool, hisparse: bool = False):
        from types import SimpleNamespace
        from unittest import mock

        from sglang.srt.model_executor import pool_configurator as pc

        cfg = pc.DefaultPoolConfigurator.__new__(pc.DefaultPoolConfigurator)
        mr = SimpleNamespace(
            model_config=SimpleNamespace(
                kv_lora_rank=512, qk_rope_head_dim=64, hf_config=object()
            ),
            kv_cache_dtype=torch.float8_e4m3fn,
            use_mla_backend=True,
            server_args=SimpleNamespace(enable_double_sparsity=ds_on),
            enable_hisparse=hisparse,
        )
        with mock.patch.object(
            pc, "get_attention_tp_size", return_value=1
        ), mock.patch.object(
            pc, "is_deepseek_dsa", return_value=True
        ), mock.patch.object(
            pc, "get_dsa_index_head_dim", return_value=128
        ), mock.patch.object(
            pc, "is_float4_e2m1fn_x2", return_value=False
        ):
            return cfg._compute_cell_size(mr, num_layers=2)

    def test_cell_size_drops_indexer_term_when_ds_gated(self):
        # The indexer term is 128 + 128//128*4 = 132 bytes/token/layer (uint8),
        # so for 2 layers the DS-gated cell is 264 bytes smaller than DSA-native.
        ds = self._cell_size(ds_on=True)
        dsa = self._cell_size(ds_on=False)
        self.assertEqual(dsa - ds, 132 * 2)

    def test_cell_size_keeps_indexer_term_for_hisparse(self):
        # HiSparse keeps the index-k buffer, so even with DS on the term stays.
        hi = self._cell_size(ds_on=True, hisparse=True)
        dsa = self._cell_size(ds_on=False)
        self.assertEqual(hi, dsa)


class TestDoubleSparsityRequestSummary(unittest.TestCase):
    """anchor: meta_info[\"double_sparsity\"] is a per-request summary
    dict (not a list of per-token dicts) for any N > 1 generated tokens.
    """

    def test_ds_meta_info_request_summary(self):
        # The transport contract is: BatchTokenIDOutput.per_request_summary
        # holds {key: List[dict]} where the list is per-request (length=bs),
        # NOT per-output-token. tokenizer_manager unpacks summary[i] into
        # meta_info[key] as one dict per request.
        from sglang.srt.managers.io_struct import BatchTokenIDOutput

        # The dataclass should accept the new field. The simulation is the
        # observable surface: pack two requests, each with N>1 tokens.
        bs = 2
        per_request_summary = {
            "double_sparsity": [
                {"sparsity_rate": 0.7, "selected_tokens": 12, "dense_fallback": 0},
                {"sparsity_rate": 0.5, "selected_tokens": 8, "dense_fallback": 1},
            ],
        }
        # Verify the field exists on BatchTokenIDOutput (msgspec.Struct upstream,
        # dataclass on older bases — accept either).
        fields = (
            set(BatchTokenIDOutput.__struct_fields__)
            if hasattr(BatchTokenIDOutput, "__struct_fields__")
            else {f.name for f in BatchTokenIDOutput.__dataclass_fields__.values()}
        )
        self.assertIn(
            "per_request_summary",
            fields,
            "BatchTokenIDOutput must carry per_request_summary for .",
        )
        # Each entry in the list is a per-request dict (not a list-of-dicts):
        for entry in per_request_summary["double_sparsity"]:
            self.assertIsInstance(entry, dict)
            self.assertIn("sparsity_rate", entry)


class TestTableFreeConfigAndValidation(unittest.TestCase):
    """Config contract for the absorbed-latent selection path: the served default
    is cosine + current-slot inclusion (the two restored fixes), scorer_norm is
    restricted to ('off', 'cosine'), and the removed table-substrate fields are
    rejected as unknown."""

    def test_served_default_is_cosine_and_current_include(self):
        cfg = parse_double_sparsity_config(_valid_payload())
        self.assertEqual(cfg.scorer_norm, "cosine")
        self.assertIs(cfg.include_current_slot, True)

    def test_accepts_cosine_config(self):
        payload = (
            '{"channel_mask_path": "/tmp/cm.safetensors", "page_size": 64, '
            '"scorer_norm": "cosine"}'
        )
        cfg = parse_double_sparsity_config(payload)
        self.assertEqual(cfg.scorer_norm, "cosine")

    def test_explicit_rawdot_current_excluded_control_parses(self):
        # The raw-dot bisection control must stay reachable by EXPLICIT config now
        # that the default is flipped — pinned by its expected (off, false) values.
        payload = (
            '{"channel_mask_path": "/tmp/cm.safetensors", "page_size": 64, '
            '"scorer_norm": "off", "include_current_slot": false}'
        )
        cfg = parse_double_sparsity_config(payload)
        self.assertEqual(cfg.scorer_norm, "off")
        self.assertIs(cfg.include_current_slot, False)

    def test_rejects_non_bool_include_current_slot(self):
        payload = (
            '{"channel_mask_path": "/tmp/cm.safetensors", "page_size": 64, '
            '"include_current_slot": "yes"}'
        )
        with self.assertRaises(ValueError):
            parse_double_sparsity_config(payload)

    def test_rejects_hybrid(self):
        payload = (
            '{"channel_mask_path": "/tmp/cm.safetensors", "page_size": 64, '
            '"scorer_norm": "hybrid"}'
        )
        with self.assertRaises(ValueError):
            parse_double_sparsity_config(payload)

    def test_rejects_unknown_table_free_field(self):
        payload = (
            '{"channel_mask_path": "/tmp/cm.safetensors", "page_size": 64, '
            '"table_free": true}'
        )
        with self.assertRaises(ValueError):
            parse_double_sparsity_config(payload)

    def test_rejects_unknown_signature_dtype_field(self):
        payload = (
            '{"channel_mask_path": "/tmp/cm.safetensors", "page_size": 64, '
            '"signature_dtype": "int8"}'
        )
        with self.assertRaises(ValueError):
            parse_double_sparsity_config(payload)


class TestCosineKeyNorm(unittest.TestCase):
    """Cosine scorer building blocks (Fix B): resident-fp8 layout assertion,
    full-coverage projection map (fail-closed), the resident key-norm helper, and
    the cosine oracle's materialized-raw == raw-dot identity (so normalization is
    the only variable between the raw-dot and cosine arms)."""

    def test_resident_fp8_layout_assertion(self):
        from sglang.srt.layers.attention.double_sparsity.absorbed_latent import (
            assert_resident_fp8_layout,
        )

        # Real 0.4.4 layout: lora=512, rope=64 bf16 = 128B -> 512 + 4*4 + 128 = 656.
        self.assertEqual(
            assert_resident_fp8_layout(
                kv_lora_rank=512, kv_cache_dim=656, rope_bytes=128
            ),
            4,
        )
        for kw in (
            dict(kv_lora_rank=500, kv_cache_dim=656, rope_bytes=128),  # lora % 128
            dict(kv_lora_rank=512, kv_cache_dim=640, rope_bytes=128),  # wrong total
            dict(kv_lora_rank=512, kv_cache_dim=656, rope_bytes=64),  # wrong rope
        ):
            with self.assertRaises(ValueError):
                assert_resident_fp8_layout(**kw)

    def test_cosine_projection_coverage_fail_closed(self):
        import torch

        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend

        H, label_dim, lora = 4, 8, 256
        mk = lambda: torch.randn(H, label_dim, lora)  # noqa: E731
        fn = DeepseekSparseAttnBackend._prepare_cosine_projections
        cpu = torch.device("cpu")
        m = fn({0: mk(), 1: mk()}, n_layers=2, exp_shape=(H, label_dim, lora), device=cpu)
        self.assertEqual(set(m), {0, 1})
        self.assertEqual(tuple(m[0].shape), (H * label_dim, lora))
        with self.assertRaises(RuntimeError):  # missing layer 1
            fn({0: mk()}, n_layers=2, exp_shape=(H, label_dim, lora), device=cpu)
        with self.assertRaises(RuntimeError):  # extra layer 2
            fn(
                {0: mk(), 1: mk(), 2: mk()},
                n_layers=2,
                exp_shape=(H, label_dim, lora),
                device=cpu,
            )
        with self.assertRaises(RuntimeError):  # shape mismatch
            fn(
                {0: mk(), 1: torch.randn(H, label_dim, lora + 1)},
                n_layers=2,
                exp_shape=(H, label_dim, lora),
                device=cpu,
            )

    def test_cosine_oracle_materialized_raw_equals_rawdot(self):
        import torch

        from sglang.srt.layers.attention.double_sparsity.absorbed_latent import (
            absorbed_latent_cosine_logical,
            absorbed_latent_score_logical,
            key_norms_from_latent,
        )

        torch.manual_seed(0)
        H, label_dim, lora, qk = 3, 4, 8, 6
        bs, T, S = 2, 7, 5
        queries = torch.randn(bs, H, qk)
        c_kv = torch.randn(T, lora)
        w_sel = torch.randn(H, label_dim, lora)
        cs = torch.stack([torch.randperm(qk)[:label_dim] for _ in range(H)]).to(
            torch.int64
        )
        cw = torch.randn(H, label_dim)
        rtt = torch.randint(0, T, (bs, S), dtype=torch.int64)
        rpi = torch.arange(bs, dtype=torch.int64)
        sl = torch.tensor([S, S - 1], dtype=torch.int64)
        # normalize=False routes the raw dot through the materialized-signature path
        # — it must equal the absorbed raw-dot oracle (normalization is the ONLY
        # variable between raw-dot and cosine).
        raw = absorbed_latent_score_logical(queries, c_kv, w_sel, cs, cw, rpi, rtt, sl, S)
        mat = absorbed_latent_cosine_logical(
            queries, c_kv, w_sel, cs, cw, rpi, rtt, sl, S, normalize=False
        )
        self.assertTrue(torch.allclose(raw, mat, atol=1e-5, equal_nan=True))
        # the cosine arm must actually differ from raw-dot (normalization active)
        cos = absorbed_latent_cosine_logical(
            queries, c_kv, w_sel, cs, cw, rpi, rtt, sl, S, normalize=True
        )
        self.assertFalse(torch.allclose(cos, mat, equal_nan=True))
        # key-norm helper matches a manual ||w_sel[h] @ c_kv[t]||
        kn = key_norms_from_latent(w_sel, c_kv)
        manual = torch.stack(
            [
                torch.stack([(w_sel[h] @ c_kv[t]).norm() for h in range(H)])
                for t in range(T)
            ]
        )
        self.assertTrue(torch.allclose(kn, manual, atol=1e-5))


class TestMetrics(unittest.TestCase):
    def test_meta_info_shape(self):
        from sglang.srt.layers.attention.double_sparsity import metrics as m

        stats = m.DoubleSparsityRequestStats(
            sparsity_rate=0.0625,
            selected_tokens=128,
            total_tokens=2048,
            dense_fallback=0,
        )
        info = m.meta_info_for_request(stats)
        self.assertEqual(
            set(info.keys()),
            {"sparsity_rate", "selected_tokens", "total_tokens", "dense_fallback"},
        )
        self.assertAlmostEqual(info["sparsity_rate"], 0.0625)
        self.assertEqual(info["selected_tokens"], 128)
        self.assertEqual(info["total_tokens"], 2048)
        self.assertEqual(info["dense_fallback"], 0)


class TestMlaNopeExtractionDualShape(unittest.TestCase):
    """`_extract_mla_nope_prefix` must pick the per-head no-PE prefix at the real
    MLA widths of BOTH the narrower (qk_nope/v = 128/128, rope 64) and the wider
    (qk_nope/v = 192/256, rope 64) shapes. Sentinel poison values prove the
    reshape-before-slice picks K_noPE / Q_noPE and never the V (K-side, suffix =
    v_head_dim) or RoPE (Q-side, suffix = qk_rope_head_dim) columns of an earlier
    head. Locks the GLM K-side suffix = v_head_dim (256), NOT rope (64).
    """

    # (qk_nope_head_dim, v_head_dim, qk_rope_head_dim)
    SHAPES = {
        "narrow_128": (128, 128, 64),
        "wide_192": (192, 256, 64),
    }

    def _extract(self):
        from sglang.srt.layers.attention.double_sparsity.calibrate import (
            _extract_mla_nope_prefix,
        )

        return _extract_mla_nope_prefix

    def test_k_side_extracts_nope_not_v(self):
        extract = self._extract()
        T, H = 3, 8
        for name, (nope, v, _rope) in self.SHAPES.items():
            with self.subTest(shape=name):
                # Per-head K layout: [K_nope (nope) | V (v)]; K_nope=1.0, V=100.0.
                per_head = nope + v
                t = torch.ones(T, H * per_head)
                blk = t.view(T, H, per_head)
                blk[:, :, nope:] = 100.0  # poison every head's V columns
                out = extract(t, H, nope, v)  # suffix_dim = v_head_dim
                self.assertEqual(tuple(out.shape), (T, H, nope))
                self.assertLess(
                    out.max().item(),
                    10.0,
                    f"{name}: K extraction leaked V columns (max={out.max():.1f})",
                )
                self.assertTrue(torch.allclose(out, torch.ones(T, H, nope)))

    def test_q_side_extracts_nope_not_rope(self):
        extract = self._extract()
        T, H = 2, 8
        for name, (nope, _v, rope) in self.SHAPES.items():
            with self.subTest(shape=name):
                # Per-head Q layout: [Q_nope (nope) | Q_rope (rope)]; nope=1.0, rope=100.0.
                per_head = nope + rope
                t = torch.ones(T, H * per_head)
                blk = t.view(T, H, per_head)
                blk[:, :, nope:] = 100.0
                out = extract(t, H, nope, rope)  # suffix_dim = qk_rope_head_dim
                self.assertEqual(tuple(out.shape), (T, H, nope))
                self.assertLess(
                    out.max().item(), 10.0, f"{name}: Q leaked RoPE columns"
                )
                self.assertTrue(torch.allclose(out, torch.ones(T, H, nope)))


class TestCalibrateCorpusEmpty(unittest.TestCase):
    """empty corpus must raise a clear ValueError."""

    def test_empty_file_raises_value_error(self):
        import tempfile

        from sglang.srt.layers.attention.double_sparsity.calibrate import (
            _read_corpus_file,
        )

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n  \n\t\n")  # whitespace only
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                _read_corpus_file(path, num_samples=4)
            self.assertIn("no non-empty lines", str(ctx.exception))
        finally:
            os.unlink(path)


class TestCalibrateHooksFireRequirement(unittest.TestCase):
    """real-path calibration must raise when one or more
    layers' K-projection hooks never fire — otherwise zero-importance rows
    silently land in the channel mask.
    """

    def test_missing_hooks_raises_runtime_error(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import MagicMock, patch

        from sglang.srt.layers.attention.double_sparsity.calibrate import (
            _collect_channel_importance,
        )

        # Fake config: 2 layers, 4 heads, head_dim=16, no MLA split.
        cfg = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=4,
            head_dim=16,
            hidden_size=64,
        )
        # Fake layer object with a self_attn that exposes NONE of the
        # probed K-projection attribute names.
        bare_attn = SimpleNamespace()  # no k_proj, no kv_b_proj, no wk
        fake_layer = SimpleNamespace(self_attn=bare_attn)
        fake_inner = SimpleNamespace(layers=[fake_layer, fake_layer])
        fake_model = MagicMock()
        fake_model.model = fake_inner
        fake_model.eval = lambda: None
        fake_model.device = torch.device("cpu")

        # Tokenizer returns a tensor we can pass to the model call.
        fake_tok = MagicMock(
            return_value=MagicMock(
                to=lambda *_a, **_k: {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
            )
        )

        with patch("transformers.AutoConfig") as mock_cfg_cls, patch(
            "transformers.AutoModelForCausalLM"
        ) as mock_model_cls, patch(
            "transformers.AutoTokenizer"
        ) as mock_tok_cls, tempfile.TemporaryDirectory() as tmp:
            mock_cfg_cls.from_pretrained.return_value = cfg
            mock_model_cls.from_pretrained.return_value = fake_model
            mock_tok_cls.from_pretrained.return_value = fake_tok
            with self.assertRaises(RuntimeError) as ctx:
                _collect_channel_importance(
                    model_path=tmp,
                    dtype="bfloat16",
                    tp=1,
                    num_layers_hint=None,
                    num_heads_hint=None,
                    head_dim_hint=None,
                    prompts=["hello"],
                    allow_synthetic=False,
                )
        msg = str(ctx.exception)
        self.assertIn("hooks did not fire", msg)
        self.assertIn("allow-synthetic", msg)


class TestCalibrateMethod1(unittest.TestCase):
    """Method 1 Q+K joint importance in _collect_channel_importance.

    Verifies that the calibrator computes mean(abs(Q_nope * K_nope)) rather
    than K-only L2, falls back gracefully when Q is absent, and that
    load_channel_mask rejects 512-d channel indices calibrated against a
    128-d model.
    """

    def _make_fake_model(
        self,
        *,
        num_layers=1,
        num_heads=2,
        k_head_dim=4,
        v_head_dim=4,
        has_q_proj=True,
        is_mla=True,
    ):
        """Return (config, model, expected_importance, fake_layer) stubs wired for
        _collect_channel_importance.  Uses real nn.Module so PyTorch forward-hooks
        fire when model(**inputs) is called."""
        import torch.nn as nn

        if is_mla:
            cfg = SimpleNamespace(
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                qk_nope_head_dim=k_head_dim,
                v_head_dim=v_head_dim,
                head_dim=k_head_dim + 64,
                hidden_size=num_heads * (k_head_dim + 64),
            )
        else:
            cfg = SimpleNamespace(
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                head_dim=k_head_dim,
                hidden_size=num_heads * k_head_dim,
            )

        k_full = num_heads * (k_head_dim + v_head_dim)
        q_full = num_heads * (k_head_dim + 64)
        T = 3
        rng = torch.Generator().manual_seed(42)

        class _FixedOutLinear(nn.Module):
            """Returns a fixed tensor (tuple-wrapped) from forward; PyTorch hooks fire."""

            def __init__(self, out_tensor):
                super().__init__()
                self._out = out_tensor

            def forward(self, x):
                return (self._out,)

        class _FakeAttn(nn.Module):
            def __init__(self, **named_projs):
                super().__init__()
                for name, mod in named_projs.items():
                    self.add_module(name, mod)

            def forward(self, x):
                for mod in self.children():
                    mod(x)

        class _FakeLayer(nn.Module):
            def __init__(self, attn):
                super().__init__()
                self.self_attn = attn

            def forward(self, x):
                self.self_attn(x)

        class _FakeInner(nn.Module):
            def __init__(self, layer_list):
                super().__init__()
                self.layers = nn.ModuleList(layer_list)

            def forward(self, x):
                for layer in self.layers:
                    layer(x)

        class _FakeTopModel(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.model = inner

            def forward(self, **_kwargs):
                self.model(torch.zeros(1))

            @property
            def device(self):
                return torch.device("cpu")

        if is_mla:
            k_out_full = torch.rand(T, k_full, generator=rng)
            q_out_full = torch.rand(T, q_full, generator=rng)
            named_projs = {"kv_b_proj": _FixedOutLinear(k_out_full)}
            if has_q_proj:
                named_projs["q_b_proj"] = _FixedOutLinear(q_out_full)
            # Correct extraction: reshape per-head first, then slice noPE prefix.
            # head_dim = k_head_dim + 64 (rope), so qk_rope_head_dim = 64.
            qk_rope_head_dim = 64
            k_nope_ref = (
                k_out_full.float()
                .reshape(T, num_heads, k_head_dim + v_head_dim)[..., :k_head_dim]
                .contiguous()
            )
            q_nope_ref = (
                q_out_full.float()
                .reshape(T, num_heads, k_head_dim + qk_rope_head_dim)[..., :k_head_dim]
                .contiguous()
            )
        else:
            k_out = torch.rand(T, num_heads * k_head_dim, generator=rng)
            q_out = torch.rand(T, num_heads * k_head_dim, generator=rng)
            named_projs = {"k_proj": _FixedOutLinear(k_out)}
            if has_q_proj:
                named_projs["q_proj"] = _FixedOutLinear(q_out)
            k_nope_ref = k_out.float().reshape(T, num_heads, k_head_dim)
            q_nope_ref = q_out.float().reshape(T, num_heads, k_head_dim)

        if has_q_proj:
            expected_importance = (q_nope_ref * k_nope_ref).abs().mean(dim=0)
        else:
            expected_importance = k_nope_ref.pow(2).mean(dim=0)

        attn = _FakeAttn(**named_projs)
        fake_layer = _FakeLayer(attn)
        fake_model = _FakeTopModel(_FakeInner([fake_layer]))

        return cfg, fake_model, expected_importance, fake_layer

    def _run_calibration(self, cfg, fake_model, tmpdir):
        """Patch transformers and invoke _collect_channel_importance."""
        from unittest.mock import patch

        from sglang.srt.layers.attention.double_sparsity.calibrate import (
            _collect_channel_importance,
        )

        fake_tok = MagicMock(
            return_value=MagicMock(
                to=lambda *_a, **_k: {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
            )
        )

        with patch("transformers.AutoConfig") as mc, patch(
            "transformers.AutoModelForCausalLM"
        ) as mm, patch("transformers.AutoTokenizer") as mt:
            mc.from_pretrained.return_value = cfg
            mm.from_pretrained.return_value = fake_model
            mt.from_pretrained.return_value = fake_tok

            importance, weights = _collect_channel_importance(
                model_path=tmpdir,
                dtype="bfloat16",
                tp=1,
                num_layers_hint=None,
                num_heads_hint=None,
                head_dim_hint=None,
                prompts=["hello world"],
                allow_synthetic=False,
            )
        return importance, weights

    def test_qk_pairing_uses_method1_formula(self):
        """Method 1: importance = mean(abs(Q_nope * K_nope)) not sum(K^2)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg, model, expected_imp, _ = self._make_fake_model(
                num_layers=1,
                num_heads=2,
                k_head_dim=4,
                v_head_dim=4,
                has_q_proj=True,
                is_mla=True,
            )
            importance, _ = self._run_calibration(cfg, model, tmpdir)

        # importance[0] should match mean(abs(Q*K)) for layer 0
        actual = importance[0].cpu()
        self.assertEqual(tuple(actual.shape), (2, 4), "importance shape must be [H, D]")
        self.assertTrue(
            torch.allclose(actual, expected_imp, atol=1e-5),
            f"Method 1 importance mismatch.\nExpected:\n{expected_imp}\nGot:\n{actual}",
        )
        # Also verify it does NOT match K-only sum(K^2): these are different tensors
        # (the test fixture uses random Q ≠ K, so Q*K ≠ K^2).

    def test_k_only_fallback_when_q_missing(self):
        """When no Q projection is found, fall back to K-only L2 with a warning."""
        import logging
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg, model, expected_k_only, _ = self._make_fake_model(
                num_layers=1,
                num_heads=2,
                k_head_dim=4,
                v_head_dim=4,
                has_q_proj=False,
                is_mla=True,
            )
            with self.assertLogs(
                "sglang.srt.layers.attention.double_sparsity.calibrate",
                level=logging.WARNING,
            ) as log_ctx:
                importance, _ = self._run_calibration(cfg, model, tmpdir)

        self.assertTrue(
            any("no Q projection" in msg for msg in log_ctx.output),
            "Expected warning about missing Q projection",
        )
        actual = importance[0].cpu()
        self.assertTrue(
            torch.allclose(actual, expected_k_only, atol=1e-5),
            f"K-only fallback importance mismatch.\nExpected:\n{expected_k_only}\nGot:\n{actual}",
        )

    def test_mla_k_extraction_ignores_v_columns(self):
        """K hook must reshape per-head before slicing; V columns must not pollute K_nope."""
        import tempfile

        import torch.nn as nn

        num_heads, k_head_dim, v_head_dim = 2, 4, 4
        T = 3
        k_full = num_heads * (k_head_dim + v_head_dim)  # 16
        q_full = num_heads * (k_head_dim + 64)  # 136

        # K output: K_nope = 1.0, V = 100.0 (sentinel poison value).
        # Layout per-head: [K_nope_h0(0:4), V_h0(4:8), K_nope_h1(8:12), V_h1(12:16)]
        k_out = torch.ones(T, k_full)
        k_out[:, 4:8] = 100.0  # V for head 0
        k_out[:, 12:16] = 100.0  # V for head 1

        # Q output: all 1.0 (isolates K extraction as the variable under test)
        q_out = torch.ones(T, q_full)

        cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=num_heads,
            qk_nope_head_dim=k_head_dim,
            v_head_dim=v_head_dim,
            head_dim=k_head_dim + 64,
            hidden_size=num_heads * (k_head_dim + 64),
        )

        class _Fixed(nn.Module):
            def __init__(self, out):
                super().__init__()
                self._out = out

            def forward(self, x):
                return (self._out,)

        class _Attn(nn.Module):
            def __init__(self, **p):
                super().__init__()
                for n, m in p.items():
                    self.add_module(n, m)

            def forward(self, x):
                for m in self.children():
                    m(x)

        class _Layer(nn.Module):
            def __init__(self, a):
                super().__init__()
                self.self_attn = a

            def forward(self, x):
                self.self_attn(x)

        class _Inner(nn.Module):
            def __init__(self, ls):
                super().__init__()
                import torch.nn as nn2

                self.layers = nn2.ModuleList(ls)

            def forward(self, x):
                for l in self.layers:
                    l(x)

        class _Top(nn.Module):
            def __init__(self, i):
                super().__init__()
                self.model = i

            def forward(self, **_kw):
                self.model(torch.zeros(1))

            @property
            def device(self):
                return torch.device("cpu")

        attn = _Attn(kv_b_proj=_Fixed(k_out), q_b_proj=_Fixed(q_out))
        fake_model = _Top(_Inner([_Layer(attn)]))

        importance, _ = self._run_calibration(cfg, fake_model, tempfile.mkdtemp())

        # Under correct extraction: both heads see K_nope = 1.0, Q = 1.0 → importance = 1.0
        # Under wrong flat-slice: head 1 sees V_h0 = 100.0 → importance ≈ 100.0
        actual = importance[0].cpu()
        self.assertLess(
            actual.max().item(),
            10.0,
            f"K extraction appears to include V columns (max={actual.max():.1f}). "
            f"Expected all values near 1.0 (K_nope=1.0 × Q=1.0).\nActual:\n{actual}",
        )
        self.assertTrue(
            torch.allclose(actual, torch.ones(num_heads, k_head_dim), atol=1e-5),
            f"K importance must be 1.0 for all heads/channels.\nActual:\n{actual}",
        )

    def test_mla_q_extraction_ignores_rope_columns(self):
        """Q hook must reshape per-head before slicing; RoPE columns must not pollute Q_nope."""
        import tempfile

        import torch.nn as nn

        num_heads, k_head_dim, v_head_dim, qk_rope_head_dim = 2, 4, 4, 64
        T = 3
        k_full = num_heads * (k_head_dim + v_head_dim)  # 16
        q_full = num_heads * (k_head_dim + qk_rope_head_dim)  # 136

        # Q output: Q_nope = 1.0, Q_rope = 100.0 (sentinel poison value).
        # Per-head layout: [Q_nope_h0(0:4), Q_rope_h0(4:68), Q_nope_h1(68:72), Q_rope_h1(72:136)]
        q_out = torch.ones(T, q_full)
        q_out[:, 4:68] = 100.0  # Q_rope for head 0
        q_out[:, 72:136] = 100.0  # Q_rope for head 1

        # K output: K_nope = 1.0, V = 0.0 (V excluded by correct extraction)
        k_out = torch.zeros(T, k_full)
        k_out[:, 0:4] = 1.0  # K_nope head 0
        k_out[:, 8:12] = 1.0  # K_nope head 1

        cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=num_heads,
            qk_nope_head_dim=k_head_dim,
            v_head_dim=v_head_dim,
            head_dim=k_head_dim + qk_rope_head_dim,
            hidden_size=num_heads * (k_head_dim + qk_rope_head_dim),
        )

        class _Fixed(nn.Module):
            def __init__(self, out):
                super().__init__()
                self._out = out

            def forward(self, x):
                return (self._out,)

        class _Attn(nn.Module):
            def __init__(self, **p):
                super().__init__()
                for n, m in p.items():
                    self.add_module(n, m)

            def forward(self, x):
                for m in self.children():
                    m(x)

        class _Layer(nn.Module):
            def __init__(self, a):
                super().__init__()
                self.self_attn = a

            def forward(self, x):
                self.self_attn(x)

        class _Inner(nn.Module):
            def __init__(self, ls):
                super().__init__()
                import torch.nn as nn2

                self.layers = nn2.ModuleList(ls)

            def forward(self, x):
                for l in self.layers:
                    l(x)

        class _Top(nn.Module):
            def __init__(self, i):
                super().__init__()
                self.model = i

            def forward(self, **_kw):
                self.model(torch.zeros(1))

            @property
            def device(self):
                return torch.device("cpu")

        attn = _Attn(kv_b_proj=_Fixed(k_out), q_b_proj=_Fixed(q_out))
        fake_model = _Top(_Inner([_Layer(attn)]))

        importance, _ = self._run_calibration(cfg, fake_model, tempfile.mkdtemp())

        # Under correct extraction: both heads see Q_nope=1.0 × K_nope=1.0 → importance = 1.0
        # Under wrong flat-slice: head 1 gets Q_rope_h0 (100.0) → importance ≈ 100.0
        actual = importance[0].cpu()
        self.assertLess(
            actual.max().item(),
            10.0,
            f"Q extraction appears to include RoPE columns (max={actual.max():.1f}). "
            f"Expected all values near 1.0.\nActual:\n{actual}",
        )
        self.assertTrue(
            torch.allclose(actual, torch.ones(num_heads, k_head_dim), atol=1e-5),
            f"Q importance must be 1.0 for all heads/channels.\nActual:\n{actual}",
        )

    def test_3d_hook_output_handled(self):
        """Hook outputs of shape [1, T, W] (batch dim) must yield identical importance to [T, W].

        _extract_mla_nope_prefix flattens all leading dims with
        ``tensor.reshape(-1, tensor.shape[-1])`` before the per-head reshape,
        so adding a batch dimension must not change the computed values.
        """
        import tempfile

        import torch.nn as nn

        num_layers, num_heads, k_head_dim, v_head_dim = 1, 2, 4, 4
        T = 3

        # 2-D reference: _make_fake_model uses seed=42, T=3
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_2d, model_2d, _, _ = self._make_fake_model(
                num_layers=num_layers,
                num_heads=num_heads,
                k_head_dim=k_head_dim,
                v_head_dim=v_head_dim,
                has_q_proj=True,
                is_mla=True,
            )
            importance_2d, _ = self._run_calibration(cfg_2d, model_2d, tmpdir)

        # 3-D variant: same random values but outputs are [1, T, W] instead of [T, W].
        # Regenerate with the same seed so tensors match _make_fake_model exactly.
        k_full = num_heads * (k_head_dim + v_head_dim)
        q_full = num_heads * (k_head_dim + 64)
        rng = torch.Generator().manual_seed(42)
        k_out_3d = torch.rand(T, k_full, generator=rng).unsqueeze(0)  # [1, T, W_k]
        q_out_3d = torch.rand(T, q_full, generator=rng).unsqueeze(0)  # [1, T, W_q]

        class _3DLinear(nn.Module):
            def __init__(self, out_3d):
                super().__init__()
                self._out = out_3d

            def forward(self, x):
                return (self._out,)

        class _FakeAttn3D(nn.Module):
            def __init__(self):
                super().__init__()
                self.kv_b_proj = _3DLinear(k_out_3d)
                self.q_b_proj = _3DLinear(q_out_3d)

            def forward(self, x):
                self.kv_b_proj(x)
                self.q_b_proj(x)

        class _FakeLayer3D(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = _FakeAttn3D()

            def forward(self, x):
                self.self_attn(x)

        class _FakeInner3D(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_FakeLayer3D()])

            def forward(self, x):
                for layer in self.layers:
                    layer(x)

        class _FakeTopModel3D(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _FakeInner3D()

            def forward(self, **_kwargs):
                self.model(torch.zeros(1))

            @property
            def device(self):
                return torch.device("cpu")

        cfg_3d = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            qk_nope_head_dim=k_head_dim,
            v_head_dim=v_head_dim,
            head_dim=k_head_dim + 64,
            hidden_size=num_heads * (k_head_dim + 64),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            importance_3d, _ = self._run_calibration(cfg_3d, _FakeTopModel3D(), tmpdir)

        actual_2d = importance_2d[0].cpu()
        actual_3d = importance_3d[0].cpu()

        self.assertTrue(
            actual_3d.isfinite().all(),
            f"3-D hook outputs produced non-finite importance:\n{actual_3d}",
        )
        self.assertTrue(
            torch.allclose(actual_3d, actual_2d, atol=1e-5),
            f"3-D and 2-D hook outputs must produce identical importance.\n"
            f"2D:\n{actual_2d}\n3D:\n{actual_3d}",
        )

    def test_pile_val_blocks_concatenate_across_docs(self):
        """_build_pile_val_token_blocks concatenates across document boundaries.

        Three short docs of 200 tokens each (600 total) with block_size=512:
        the single output block must span all three documents — not just truncate
        the first document.
        """
        from unittest.mock import patch

        from sglang.srt.layers.attention.double_sparsity.calibrate import (
            _build_pile_val_token_blocks,
        )

        # Doc i yields token IDs [i*200 .. i*200+199]
        # Concatenated stream: [0..199][200..399][400..599] = 600 tokens total
        # A block_size=512 block must include tokens from all 3 docs.
        doc_texts = ["doc0_text", "doc1_text", "doc2_text"]
        fake_examples = [{"text": t} for t in doc_texts]

        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(fake_examples))
        mock_ds.shuffle.return_value = mock_ds

        def fake_tokenize(text, add_special_tokens=False, return_attention_mask=False):
            if "doc0" in text:
                return {"input_ids": list(range(0, 200))}
            elif "doc1" in text:
                return {"input_ids": list(range(200, 400))}
            else:
                return {"input_ids": list(range(400, 600))}

        fake_tok = MagicMock(side_effect=fake_tokenize)

        mock_datasets_module = MagicMock()
        mock_datasets_module.load_dataset.return_value = mock_ds

        with patch.dict(sys.modules, {"datasets": mock_datasets_module}):
            blocks = _build_pile_val_token_blocks(
                fake_tok,
                num_blocks=1,
                block_size=512,
                seed=42,
            )

        self.assertEqual(len(blocks), 1, "Must return exactly 1 block")
        self.assertEqual(
            tuple(blocks[0].shape), (1, 512), "Block shape must be [1, 512]"
        )

        block_ids = blocks[0][0].tolist()
        # Doc 0 occupies positions 0..199 → token IDs 0..199
        self.assertEqual(block_ids[0], 0)
        self.assertEqual(block_ids[199], 199)
        # Doc 1 occupies positions 200..399 → token IDs 200..399
        self.assertEqual(block_ids[200], 200)
        # Position 511 is in doc 2 range (400..599); token ID equals position
        # since each doc's IDs equal their position in the concatenated stream.
        self.assertEqual(
            block_ids[511],
            511,
            f"Token at index 511 must come from doc 2 (cross-document boundary). "
            f"Got {block_ids[511]}; docs were merely truncated if this fails.",
        )

    def test_dsv32_real_config_shape_q_hook_fires(self):
        """V3.2 config has qk_rope_head_dim=4 but no head_dim field.

        hidden_size // num_heads = 32 // 4 = 8, not qk_nope + qk_rope = 12.
        The old code derived qk_rope_head_dim = 8 - 8 = 0 (or negative in prod),
        setting full_mla_q_width=None and silently skipping every Q hook.
        The fix reads config.qk_rope_head_dim directly; this test proves Method 1
        Q/K importance is accumulated correctly for this config shape.
        """
        import tempfile
        from unittest.mock import patch as _patch

        import torch.nn as nn

        from sglang.srt.layers.attention.double_sparsity.calibrate import (
            _collect_channel_importance,
        )

        num_heads = 4
        qk_nope = 8
        qk_rope = 4
        v_head_dim_val = 4
        T = 3

        # Config with explicit qk_rope_head_dim, no head_dim.
        # hidden_size // num_heads = 32 // 4 = 8 ≠ qk_nope + qk_rope = 12.
        cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_attention_heads=num_heads,
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=qk_rope,
            v_head_dim=v_head_dim_val,
            hidden_size=32,
            # intentionally no head_dim attribute
        )

        k_full = num_heads * (qk_nope + v_head_dim_val)  # 4*(8+4)=48
        q_full = num_heads * (qk_nope + qk_rope)  # 4*(8+4)=48
        rng = torch.Generator().manual_seed(42)
        k_out = torch.rand(T, k_full, generator=rng)
        q_out = torch.rand(T, q_full, generator=rng)

        k_nope_ref = (
            k_out.float()
            .reshape(T, num_heads, qk_nope + v_head_dim_val)[..., :qk_nope]
            .contiguous()
        )
        q_nope_ref = (
            q_out.float()
            .reshape(T, num_heads, qk_nope + qk_rope)[..., :qk_nope]
            .contiguous()
        )
        expected_imp = (q_nope_ref * k_nope_ref).abs().mean(dim=0)

        class _FixedOut(nn.Module):
            def __init__(self, out):
                super().__init__()
                self._out = out

            def forward(self, x):
                return (self._out,)

        class _FakeAttn(nn.Module):
            def __init__(self):
                super().__init__()
                self.kv_b_proj = _FixedOut(k_out)
                self.q_b_proj = _FixedOut(q_out)

            def forward(self, x):
                self.kv_b_proj(x)
                self.q_b_proj(x)

        class _FakeLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = _FakeAttn()

            def forward(self, x):
                self.self_attn(x)

        class _FakeInner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_FakeLayer()])

            def forward(self, x):
                for layer in self.layers:
                    layer(x)

        class _FakeTopModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = _FakeInner()

            def forward(self, **_kwargs):
                self.model(torch.zeros(1))

            @property
            def device(self):
                return torch.device("cpu")

        fake_tok = MagicMock(
            return_value=MagicMock(
                to=lambda *_a, **_k: {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with _patch("transformers.AutoConfig") as mc, _patch(
                "transformers.AutoModelForCausalLM"
            ) as mm, _patch("transformers.AutoTokenizer") as mt:
                mc.from_pretrained.return_value = cfg
                mm.from_pretrained.return_value = _FakeTopModel()
                mt.from_pretrained.return_value = fake_tok

                importance, _ = _collect_channel_importance(
                    model_path=tmpdir,
                    dtype="bfloat16",
                    tp=1,
                    num_layers_hint=None,
                    num_heads_hint=None,
                    head_dim_hint=None,
                    prompts=["hello world"],
                    allow_synthetic=False,
                )

        actual = importance[0].cpu()
        self.assertEqual(
            tuple(actual.shape),
            (num_heads, qk_nope),
            "importance shape must be [H, qk_nope_head_dim]",
        )
        self.assertTrue(
            actual.isfinite().all(),
            f"V3.2 config shape produced non-finite importance:\n{actual}",
        )
        self.assertTrue(
            torch.allclose(actual, expected_imp, atol=1e-5),
            f"Method 1 importance mismatch with V3.2 config shape (no head_dim field).\n"
            f"Expected:\n{expected_imp}\nGot:\n{actual}",
        )

    def test_512d_channel_index_rejected(self):
        """load_channel_mask must reject channel indices >= head_dim=128."""
        import tempfile

        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            DoubleSparsityChannelMaskCorrupt,
            load_channel_mask,
            save_channel_mask,
        )

        L, H, label_dim = 2, 4, 8
        # channel_selection contains index 512 (out of range for head_dim=128)
        channel_selection = torch.zeros(L, H, label_dim, dtype=torch.int32)
        channel_selection[0, 0, 0] = 512  # 512-d index — invalid for 128-d model
        channel_weights = torch.ones(L, H, label_dim, dtype=torch.float32)

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        try:
            save_channel_mask(
                path,
                channel_selection,
                channel_weights,
                dtype="bfloat16",
                head_dim=128,
                page_size=64,
                label_dim=label_dim,
                created_at="2026-01-01T00:00:00Z",
            )
            with self.assertRaises(
                (DoubleSparsityChannelMaskCorrupt, ValueError)
            ) as ctx:
                load_channel_mask(path)
            self.assertIn("out of range", str(ctx.exception))
        finally:
            import os as _os

            _os.unlink(path)

    def test_label_dim_exceeds_k_head_dim_raises(self):
        """calibrate() must raise ValueError when label_dim > head_dim."""
        import argparse

        from sglang.srt.layers.attention.double_sparsity.calibrate import calibrate

        args = argparse.Namespace(
            model="/nonexistent",
            dtype="bfloat16",
            tp=1,
            output="/tmp/test_calib_out.safetensors",
            label_dim=256,  # > head_dim which would be derived as 128
            page_size=64,
            num_samples=4,
            ctx_len=64,
            block_size=512,
            seed=42,
            dataset=None,
            num_layers=1,
            num_heads=2,
            head_dim=128,
            allow_synthetic=True,
        )
        with self.assertRaises(ValueError) as ctx:
            calibrate(args)
        self.assertIn("label-dim", str(ctx.exception))


@unittest.skipUnless(torch.cuda.is_available(), "bf16 resident score is a Triton kernel")
class TestBf16ResidentCosine(unittest.TestCase):
    """AC-8: the absorbed paged score reads a BF16 resident latent (k_nope)
    directly — no fp8 per-128-block scale — and matches a torch reference. The
    fp8 path is unaffected (separate BF16_LATENT=False constexpr branch)."""

    def test_bf16_resident_raw_dot_matches_reference(self):
        from sglang.srt.layers.attention.double_sparsity.absorbed_latent_kernel import (
            absorbed_score_paged_fp8,
        )

        dev = torch.device("cuda")
        torch.manual_seed(0)
        bs, H, lora, max_seq_len, max_tokens = 2, 2, 256, 4, 8
        v = (torch.randn(bs, H, lora, device=dev) * 0.1).to(torch.float32)
        latent_bf16 = (torch.randn(max_tokens, lora, device=dev) * 0.1).to(
            torch.bfloat16
        )
        req_pool = torch.tensor([0, 0], dtype=torch.int32, device=dev)
        # logical t -> physical slot (a non-identity permutation to exercise paging)
        rtt = torch.tensor(
            [[5, 2, 7, 1, 0, 0, 0, 0]], dtype=torch.int32, device=dev
        )
        seq_lens = torch.tensor([max_seq_len, max_seq_len], dtype=torch.int32, device=dev)

        scores = absorbed_score_paged_fp8(
            v,
            latent_bf16,  # bf16 latent -> BF16_LATENT path (auto-detected)
            None,  # no scales
            req_pool,
            rtt,
            seq_lens,
            max_seq_len,
            written=None,
            head_agg="max",
            cosine=False,
        )

        # Reference: score[b,t] = max_h Σ_l v[b,h,l] * latent_fp32[phys(t), l]
        lat_f32 = latent_bf16.to(torch.float32)
        ref = torch.full((bs, max_seq_len), float("-inf"), device=dev)
        for b in range(bs):
            pool = int(req_pool[b])
            for t in range(max_seq_len):
                phys = int(rtt[pool, t])
                dots = (v[b] * lat_f32[phys].unsqueeze(0)).sum(dim=1)  # [H]
                ref[b, t] = dots.max()
        # bf16 latent + tf32 MMA: loose tolerance on small-magnitude scores.
        torch.testing.assert_close(
            scores.float(), ref.float(), rtol=2e-2, atol=2e-2
        )


@unittest.skipUnless(torch.cuda.is_available(), "current-slot force-include is a Triton kernel")
class TestIncludeCurrentSlotForceInclude(unittest.TestCase):
    """AC-7/AC-5/AC-4: the current-slot +inf force-include fails closed on a
    selector-width miss (no clamp to a different token) and, for cosine, only
    fires when the row q-norm and the current physical-slot k-norm are finite.
    Only the current slot (seq_len-1) is ever touched (H3 preserved)."""

    def _run(
        self,
        seq_len,
        max_seq_len,
        *,
        cosine=False,
        qnorm_row=None,
        cur_knorm_row=None,
        H=2,
    ):
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            _force_include_current_slot,
        )

        dev = torch.device("cuda")
        bs = 1
        scores = torch.zeros((bs, max_seq_len), dtype=torch.bfloat16, device=dev)
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=dev)
        scratch_qnorm = key_norm_cache = req_pool = rtt = None
        layer_id = 0
        if cosine:
            scratch_qnorm = torch.tensor(
                [qnorm_row if qnorm_row is not None else [1.0] * H],
                dtype=torch.float32,
                device=dev,
            )
            req_pool = torch.tensor([0], dtype=torch.int32, device=dev)
            max_ctx = max_seq_len + 4
            # current physical slot for logical seq_len-1 is phys=7 (arbitrary).
            phys = 7
            rtt = torch.zeros((1, max_ctx), dtype=torch.int32, device=dev)
            if 0 <= seq_len - 1 < max_ctx:
                rtt[0, seq_len - 1] = phys
            max_tokens = 16
            key_norm_cache = torch.ones(
                (1, max_tokens, H), dtype=torch.float32, device=dev
            )
            if cur_knorm_row is not None:
                key_norm_cache[0, phys] = torch.tensor(
                    cur_knorm_row, dtype=torch.float32, device=dev
                )
        _force_include_current_slot(
            scores,
            seq_lens,
            max_seq_len,
            bs,
            cosine=cosine,
            scratch_qnorm=scratch_qnorm,
            key_norm_cache=key_norm_cache,
            layer_id=layer_id,
            req_pool_indices=req_pool,
            req_to_token=rtt,
        )
        return scores

    def test_rawdot_in_range_forces_current_only(self):
        scores = self._run(5, 8)  # cur = 4
        self.assertTrue(torch.isinf(scores[0, 4]).item())
        # H3: nothing else touched.
        others = torch.cat([scores[0, :4], scores[0, 5:]])
        self.assertFalse(torch.isinf(others).any().item())

    def test_width_miss_fails_closed(self):
        # seq_len-1 = 9 >= max_seq_len 8 -> no force-include (no clamp to col 7).
        scores = self._run(10, 8)
        self.assertFalse(torch.isinf(scores).any().item())

    def test_cosine_finite_forces(self):
        scores = self._run(5, 8, cosine=True, qnorm_row=[1.0, 2.0], cur_knorm_row=[1.0, 1.0])
        self.assertTrue(torch.isinf(scores[0, 4]).item())

    def test_cosine_nan_qnorm_no_force(self):
        scores = self._run(
            5, 8, cosine=True, qnorm_row=[float("nan"), 1.0], cur_knorm_row=[1.0, 1.0]
        )
        self.assertFalse(torch.isinf(scores).any().item())

    def test_cosine_inf_qnorm_no_force(self):
        scores = self._run(
            5, 8, cosine=True, qnorm_row=[float("inf"), 1.0], cur_knorm_row=[1.0, 1.0]
        )
        self.assertFalse(torch.isinf(scores).any().item())

    def test_cosine_nan_current_knorm_still_forces(self):
        # The current decode token's KV is valid by construction, so a NaN in its
        # (lagging) key-norm CACHE entry must NOT drop it — only the row q-norm
        # and width gate the current-slot force-include.
        scores = self._run(
            5, 8, cosine=True, qnorm_row=[1.0, 1.0], cur_knorm_row=[float("nan"), 1.0]
        )
        self.assertTrue(torch.isinf(scores[0, 4]).item())

    def test_cosine_width_miss_fails_closed(self):
        scores = self._run(
            12, 8, cosine=True, qnorm_row=[1.0, 1.0], cur_knorm_row=[1.0, 1.0]
        )
        self.assertFalse(torch.isinf(scores).any().item())


if __name__ == "__main__":
    unittest.main()
