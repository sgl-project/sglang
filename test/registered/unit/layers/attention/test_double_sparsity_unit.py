"""Unit tests for standalone Double Sparsity (placeholder scaffolding).

Covers the round-0 backbone: config parsing surface (AC-11 absence of
``selection_mode`` / ``top_p``), selector ABI shape (AC-2), validator
fail-fast behaviour for missing-config and HiSparse mutual-exclusion
(AC-1 + DEC-8), and the ``_select_topk_indices`` config-gated branch on
``DeepseekV2AttentionMLA`` (AC-2 hook).

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
        payload = (
            '{"top_k": 2048, "page_size": 64, "device_buffer_size": 4096}'
        )
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
        args = self._args(
            enable_double_sparsity=True, double_sparsity_config=None
        )
        with self.assertRaises(ValueError) as ctx:
            validate_double_sparsity(args)
        self.assertIn("channel_mask_path", str(ctx.exception))

    def test_disaggregation_rejected(self):
        args = self._args(
            enable_double_sparsity=True,
            disaggregation_mode="decode",
            double_sparsity_config=_valid_payload(),
        )
        with self.assertRaises(ValueError) as ctx:
            validate_double_sparsity(args)
        self.assertIn("disaggregation", str(ctx.exception).lower())

    def test_page_size_mismatch(self):
        args = self._args(
            enable_double_sparsity=True,
            double_sparsity_config=_valid_payload(),
            page_size=32,
        )
        with self.assertRaises(ValueError) as ctx:
            validate_double_sparsity(args)
        self.assertIn("page_size", str(ctx.exception))

    def test_valid_path(self):
        args = self._args(
            enable_double_sparsity=True,
            double_sparsity_config=_valid_payload(),
            page_size=64,
        )
        os.environ["SGLANG_DS_ALLOW_PLACEHOLDER"] = "1"
        try:
            validate_double_sparsity(args)
            self.assertIsInstance(
                args._double_sparsity_parsed_config, DoubleSparsityConfig
            )
        finally:
            os.environ.pop("SGLANG_DS_ALLOW_PLACEHOLDER", None)


class TestPlaceholderGuard(unittest.TestCase):
    def setUp(self):
        cfg = parse_double_sparsity_config(_valid_payload())
        self.selector = DoubleSparsitySelector(
            config=cfg,
            num_local_heads=16,
            head_dim=128,
            device=torch.device("cpu"),
        )

    def test_refuses_without_env(self):
        os.environ.pop("SGLANG_DS_ALLOW_PLACEHOLDER", None)
        with self.assertRaises(RuntimeError) as ctx:
            assert_real_selector_or_placeholder_allowed(self.selector)
        self.assertIn("placeholder", str(ctx.exception).lower())

    def test_allowed_with_env(self):
        os.environ["SGLANG_DS_ALLOW_PLACEHOLDER"] = "1"
        try:
            assert_real_selector_or_placeholder_allowed(self.selector)
        finally:
            os.environ.pop("SGLANG_DS_ALLOW_PLACEHOLDER", None)

    def test_real_selector_passes(self):
        class _Real:
            IS_PLACEHOLDER = False

        assert_real_selector_or_placeholder_allowed(_Real())


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
        attn.indexer = MagicMock(return_value=torch.tensor([7, 8, 9], dtype=torch.int32))
        if use_ds:
            cfg = parse_double_sparsity_config(_valid_payload())
            attn.double_sparsity_selector = DoubleSparsitySelector(
                config=cfg,
                num_local_heads=16,
                head_dim=128,
                device=torch.device("cpu"),
            )
        return attn

    def test_ds_branch_returns_tuple(self):
        os.environ["SGLANG_DS_ALLOW_PLACEHOLDER"] = "1"
        try:
            attn = self._make_attn(use_ds=True)
            forward_batch = SimpleNamespace(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
                seq_lens=torch.tensor([100, 200], dtype=torch.int32),
                sparse_mask=None,
            )
            result = attn._select_topk_indices(
                x=torch.zeros(2, 16, 128),
                q_lora=torch.zeros(2, 16, 128),
                positions=torch.zeros(2, dtype=torch.int32),
                forward_batch=forward_batch,
                layer_id=0,
            )
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 2)
            selected_indices, valid_lengths = result
            self.assertEqual(selected_indices.dtype, torch.int32)
            self.assertEqual(valid_lengths.dtype, torch.int32)
            self.assertEqual(tuple(selected_indices.shape), (2, 2048))
            self.assertEqual(tuple(valid_lengths.shape), (2,))
            attn.indexer.assert_not_called()
        finally:
            os.environ.pop("SGLANG_DS_ALLOW_PLACEHOLDER", None)

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

    def test_ds_branch_refuses_without_env(self):
        os.environ.pop("SGLANG_DS_ALLOW_PLACEHOLDER", None)
        attn = self._make_attn(use_ds=True)
        forward_batch = SimpleNamespace(
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([100], dtype=torch.int32),
            sparse_mask=None,
        )
        with self.assertRaises(RuntimeError):
            attn._select_topk_indices(
                x=torch.zeros(1, 16, 128),
                q_lora=torch.zeros(1, 16, 128),
                positions=torch.zeros(1, dtype=torch.int32),
                forward_batch=forward_batch,
                layer_id=0,
            )


if __name__ == "__main__":
    unittest.main()
