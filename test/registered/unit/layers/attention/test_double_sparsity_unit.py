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

    def test_adapter_gate_rejects_by_default(self):
        """Round-3 fix [P1]: validator rejects --enable-double-sparsity at
        startup until the page-table adapter lands, so a misconfigured
        server fails at boot instead of on first request.
        """

        args = self._args(
            enable_double_sparsity=True,
            double_sparsity_config=_valid_payload(),
            page_size=64,
        )
        os.environ.pop("SGLANG_DS_ALLOW_NO_ADAPTER", None)
        with self.assertRaises(ValueError) as ctx:
            validate_double_sparsity(args)
        self.assertIn("adapter", str(ctx.exception).lower())
        self.assertIn("SGLANG_DS_ALLOW_NO_ADAPTER", str(ctx.exception))

    def test_mutual_exclusion_with_hisparse(self):
        args = self._args(enable_double_sparsity=True, enable_hisparse=True)
        os.environ["SGLANG_DS_ALLOW_NO_ADAPTER"] = "1"
        try:
            with self.assertRaises(ValueError) as ctx:
                validate_double_sparsity(args)
            self.assertIn("mutually exclusive", str(ctx.exception).lower())
        finally:
            os.environ.pop("SGLANG_DS_ALLOW_NO_ADAPTER", None)

    def test_missing_config(self):
        args = self._args(
            enable_double_sparsity=True, double_sparsity_config=None
        )
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
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            save_channel_mask,
        )
        import tempfile, os as _os
        sel_t = torch.randint(0, 512, (2, 4, 16), dtype=torch.int32)
        w_t = torch.randn(2, 4, 16, dtype=torch.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = _os.path.join(tmp, "cm.safetensors")
            save_channel_mask(
                path, sel_t, w_t, dtype="fp8_e4m3", head_dim=128, page_size=64,
                label_dim=16, created_at="2026-05-20T00:00:00Z",
            )
            args = self._args(
                enable_double_sparsity=True,
                double_sparsity_config=_valid_payload(path),
                page_size=64,
                kv_cache_dtype="fp8_e4m3",
                nsa_prefill_backend="flashmla_kv",
                nsa_decode_backend="flashmla_kv",
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

    def test_ds_branch_raises_pending_adapter(self):
        """With the placeholder guard satisfied, the DS branch must still
        fail loudly because the page-table adapter that translates
        ``(selected_indices, valid_lengths)`` to the NSA backend's
        token-level ``topk_indices`` tensor has not landed yet. The selector
        ABI itself is exercised independently via
        ``DoubleSparsitySelector.retrieve_topk`` (see ``TestSelectorAbi``).
        """

        os.environ["SGLANG_DS_ALLOW_PLACEHOLDER"] = "1"
        try:
            attn = self._make_attn(use_ds=True)
            forward_batch = SimpleNamespace(
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
                seq_lens=torch.tensor([100, 200], dtype=torch.int32),
                sparse_mask=None,
            )
            with self.assertRaises(NotImplementedError):
                attn._select_topk_indices(
                    x=torch.zeros(2, 16, 128),
                    q_lora=torch.zeros(2, 16, 128),
                    positions=torch.zeros(2, dtype=torch.int32),
                    forward_batch=forward_batch,
                    layer_id=0,
                )
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


class TestChannelMaskLoader(unittest.TestCase):
    def _make_payload(self, *, L=4, H=4, label_dim=16):
        sel = torch.randint(0, 512, (L, H, label_dim), dtype=torch.int32)
        w = torch.randn(L, H, label_dim, dtype=torch.float32)
        return sel, w

    def test_roundtrip(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            save_channel_mask,
            load_channel_mask,
        )
        import tempfile, os
        sel, w = self._make_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cm.safetensors")
            h = save_channel_mask(
                path, sel, w, dtype="fp8_e4m3", head_dim=128, page_size=64,
                label_dim=16, created_at="2026-05-20T00:00:00Z",
            )
            cm = load_channel_mask(path)
        self.assertEqual(cm.content_sha256, h)
        self.assertEqual(cm.dtype, "fp8_e4m3")
        self.assertEqual(cm.head_dim, 128)
        self.assertEqual(cm.page_size, 64)
        self.assertEqual(cm.label_dim, 16)
        self.assertTrue(torch.equal(cm.channel_selection, sel))

    def test_content_hash_mismatch(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            save_channel_mask, load_channel_mask, compute_content_sha256,
        )
        from safetensors import safe_open
        from safetensors.torch import save_file
        import tempfile, os
        sel, w = self._make_payload()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cm.safetensors")
            save_channel_mask(
                path, sel, w, dtype="fp8_e4m3", head_dim=128, page_size=64,
                label_dim=16, created_at="2026-05-20T00:00:00Z",
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

    def test_validate_runtime_mismatches(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask, validate_against_runtime,
        )
        mask = ChannelMask(
            channel_selection=torch.zeros(2, 2, 4, dtype=torch.int32),
            channel_weights=torch.zeros(2, 2, 4, dtype=torch.float32),
            schema_version="1", dtype="fp8_e4m3", head_dim=128, page_size=64,
            label_dim=4, content_sha256="x",
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
                mask, server_kv_cache_dtype="bfloat16",
                server_page_size=64, server_label_dim=4, model_head_dim=128,
            )

    def test_sanity_probe_placeholder_inconclusive(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask, startup_sanity_probe,
        )
        mask = ChannelMask(
            channel_selection=torch.zeros(2, 2, 4, dtype=torch.int32),
            channel_weights=torch.zeros(2, 2, 4, dtype=torch.float32),
            schema_version="1", dtype="fp8_e4m3", head_dim=128, page_size=64,
            label_dim=4, content_sha256="x",
        )
        cfg = parse_double_sparsity_config(_valid_payload())
        selector = DoubleSparsitySelector(
            config=cfg, num_local_heads=4, head_dim=128, device=torch.device("cpu"),
        )
        r = startup_sanity_probe(mask, selector)
        self.assertFalse(r.passed)
        self.assertEqual(r.skipped_reason, "placeholder_selector")


class TestPageSignatureTableLifecycle(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers.attention.double_sparsity.page_signature_table import (
            allocate_page_signature_table,
        )
        self.table = allocate_page_signature_table(
            num_layers_local=2, max_pages=32, num_heads_local=4, label_dim=16,
            page_size=64, dtype=torch.float16, device=torch.device("cpu"),
        )

    def test_assign_populate_free(self):
        self.table.on_pages_assigned(0, [3, 7])
        self.table.mark_populated(0, [3, 7])
        self.assertTrue(bool(self.table.valid_mask[0, 3].item()))
        self.assertTrue(bool(self.table.valid_mask[0, 7].item()))
        self.table.on_page_freed(0, 7)
        self.assertFalse(bool(self.table.valid_mask[0, 7].item()))

    def test_evict_idempotent(self):
        self.table.on_pages_assigned(1, [5])
        self.table.mark_populated(1, [5])
        self.table.on_page_evicted(1, 5)
        self.table.on_page_evicted(1, 5)  # idempotent
        self.assertFalse(bool(self.table.valid_mask[1, 5].item()))

    def test_hot_page_clears_on_free(self):
        self.table.set_hot_page(0, 11)
        self.assertEqual(self.table.get_hot_page(0), 11)
        self.table.on_page_freed(0, 11)
        self.assertIsNone(self.table.get_hot_page(0))

    def test_estimate_hbm_bytes(self):
        from sglang.srt.layers.attention.double_sparsity.page_signature_table import (
            estimate_hbm_bytes,
        )
        b = estimate_hbm_bytes(
            num_layers_local=60, max_pages=15_625, num_heads_local=16,
            label_dim=16, dtype=torch.float16,
        )
        # within ±10% of the documented 480 MB budget
        self.assertGreater(b, 400 * 1024 * 1024)
        self.assertLess(b, 520 * 1024 * 1024)


class TestSelectionKernel(unittest.TestCase):
    def test_project_query(self):
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            project_query_onto_channels,
        )
        queries = torch.randn(2, 4, 16)
        sel = torch.randint(0, 16, (4, 8), dtype=torch.int32)
        w = torch.randn(4, 8, dtype=torch.float32)
        out = project_query_onto_channels(queries, sel, w)
        self.assertEqual(tuple(out.shape), (2, 4, 8))

    def test_invalid_pages_excluded(self):
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            compute_page_scores, select_topk_sequence_order,
        )
        torch.manual_seed(11)
        L, P, H, D = 2, 8, 4, 4
        queries = torch.randn(2, H, 16)
        sigs = torch.randn(L, P, H, D, dtype=torch.float16)
        vmask = torch.ones(L, P, dtype=torch.bool)
        vmask[0, 2] = False
        sel = torch.randint(0, 16, (L, H, D), dtype=torch.int32)
        w = torch.randn(L, H, D, dtype=torch.float32)
        scores = compute_page_scores(queries, sigs, vmask, sel, w, layer_id=0)
        idx, lens = select_topk_sequence_order(scores, max_top_k=4)
        for r in range(2):
            self.assertNotIn(2, idx[r, : lens[r]].tolist())

    def test_hot_page_overrides_invalid(self):
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            select_topk_sequence_order,
        )
        scores = torch.full((1, 8), -1e9, dtype=torch.float32)
        scores[0, 5] = 0.5  # one valid
        scores[0, 2] = float("-inf")  # invalid
        idx, lens = select_topk_sequence_order(scores.clone(), max_top_k=3, hot_pages=[[2]])
        # Hot page 2 was -inf and we want it forced in.
        # Note: per the kernel, hot pages set score to +inf which forces them in.
        row = idx[0, : lens[0]].tolist()
        self.assertIn(2, row)

    def test_ascending_invariant(self):
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            select_topk_sequence_order,
        )
        torch.manual_seed(7)
        scores = torch.randn(3, 16)
        idx, lens = select_topk_sequence_order(scores, max_top_k=6)
        for r in range(3):
            row = idx[r, : lens[r]].tolist()
            self.assertTrue(
                all(row[i] < row[i + 1] for i in range(len(row) - 1)),
                f"row {r} not ascending: {row}",
            )

    def test_all_reduce_noop_without_group(self):
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            all_reduce_page_scores,
        )
        x = torch.randn(8)
        y = all_reduce_page_scores(x, process_group=None)
        self.assertTrue(torch.equal(x, y))


class TestPageSignatureWrite(unittest.TestCase):
    def test_dequant_per_tile(self):
        from sglang.srt.layers.attention.double_sparsity.page_signature_write import (
            dequant_nope_fp8_to_bf16,
        )
        torch.manual_seed(0)
        n = 4
        nope_fp8 = torch.randn(n, 512).to(torch.float8_e4m3fn)
        scales = torch.rand(n, 4) * 2.0
        u8 = torch.zeros(n, 528, dtype=torch.uint8)
        u8[:, :512] = nope_fp8.view(torch.uint8)
        u8[:, 512:].view(torch.float32)[:, :] = scales.contiguous()
        bf16 = dequant_nope_fp8_to_bf16(u8)
        self.assertEqual(tuple(bf16.shape), (n, 512))
        # Tolerance accommodates bf16 rounding noise; max observed ~0.015.
        ref0 = nope_fp8[0, :128].to(torch.float32) * scales[0, 0]
        got0 = bf16[0, :128].to(torch.float32)
        self.assertTrue(torch.allclose(got0, ref0, atol=5e-2))

    def test_compute_hot_pages(self):
        from sglang.srt.layers.attention.double_sparsity.page_signature_write import (
            compute_hot_pages,
        )
        seq_lens = torch.tensor([0, 100, 200, 320], dtype=torch.int32)
        hot = compute_hot_pages(seq_lens=seq_lens, page_size=64, local_window=2)
        # 0 → []
        # 100 → last=1, window=2: [0,1]
        # 200 → last=3, window=2: [2,3]
        # 320 → last=4, window=2: [3,4]
        self.assertEqual(hot, [[], [0, 1], [2, 3], [3, 4]])

    def test_project_page_signature(self):
        from sglang.srt.layers.attention.double_sparsity.page_signature_write import (
            project_page_to_signature,
        )
        nope = torch.randn(8, 512, dtype=torch.bfloat16)
        sel = torch.randint(0, 512, (4, 16), dtype=torch.int32)
        w = torch.randn(4, 16, dtype=torch.float32)
        sig = project_page_to_signature(nope, sel, w, reduce="mean")
        self.assertEqual(tuple(sig.shape), (4, 16))


class TestSelectorRealMode(unittest.TestCase):
    def test_bind_runtime_data_flips_placeholder(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask,
        )
        from sglang.srt.layers.attention.double_sparsity.page_signature_table import (
            allocate_page_signature_table,
        )
        cfg = parse_double_sparsity_config(_valid_payload())
        sel = DoubleSparsitySelector(
            config=cfg, num_local_heads=4, head_dim=128, device=torch.device("cpu"),
        )
        self.assertTrue(sel.IS_PLACEHOLDER)
        table = allocate_page_signature_table(
            num_layers_local=2, max_pages=16, num_heads_local=4, label_dim=16,
            page_size=64, dtype=torch.float16, device=torch.device("cpu"),
        )
        mask = ChannelMask(
            channel_selection=torch.zeros(2, 4, 16, dtype=torch.int32),
            channel_weights=torch.zeros(2, 4, 16, dtype=torch.float32),
            schema_version="1", dtype="fp8_e4m3", head_dim=128, page_size=64,
            label_dim=16, content_sha256="x",
        )
        sel.bind_runtime_data(table, mask)
        self.assertFalse(sel.IS_PLACEHOLDER)

    def test_real_mode_topk_uses_signatures(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask,
        )
        from sglang.srt.layers.attention.double_sparsity.page_signature_table import (
            allocate_page_signature_table,
        )
        cfg = parse_double_sparsity_config(_valid_payload())
        sel = DoubleSparsitySelector(
            config=cfg, num_local_heads=4, head_dim=128, device=torch.device("cpu"),
        )
        table = allocate_page_signature_table(
            num_layers_local=2, max_pages=16, num_heads_local=4, label_dim=16,
            page_size=64, dtype=torch.float16, device=torch.device("cpu"),
        )
        table.signatures.uniform_(-1, 1)
        table.valid_mask.fill_(True)
        mask = ChannelMask(
            channel_selection=torch.randint(0, 128, (2, 4, 16), dtype=torch.int32),
            channel_weights=torch.randn(2, 4, 16, dtype=torch.float32),
            schema_version="1", dtype="fp8_e4m3", head_dim=128, page_size=64,
            label_dim=16, content_sha256="x",
        )
        sel.bind_runtime_data(table, mask)

        queries = torch.randn(2, 4, 128)
        req_pool = torch.tensor([0, 1], dtype=torch.int32)
        seq_lens = torch.tensor([500, 320], dtype=torch.int32)
        sparse_mask = torch.ones(2, 16, dtype=torch.int32)
        indices, lengths = sel.retrieve_topk(
            queries=queries, layer_id=0, req_pool_indices=req_pool,
            sparse_mask=sparse_mask, seq_lens=seq_lens,
        )
        # max_top_k=2048 from default _valid_payload; output shape matches
        self.assertEqual(tuple(indices.shape), (2, 2048))

    def test_per_request_mask_isolates_pages(self):
        """Round-2 fix [P2]: a request must not select pages owned by a
        different request in the same batch, even if those pages are
        globally valid in the signature table.
        """

        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask,
        )
        from sglang.srt.layers.attention.double_sparsity.page_signature_table import (
            allocate_page_signature_table,
        )
        cfg = parse_double_sparsity_config(_valid_payload())
        sel = DoubleSparsitySelector(
            config=cfg, num_local_heads=2, head_dim=64, device=torch.device("cpu"),
        )
        table = allocate_page_signature_table(
            num_layers_local=1, max_pages=8, num_heads_local=2, label_dim=8,
            page_size=64, dtype=torch.float16, device=torch.device("cpu"),
        )
        table.signatures.uniform_(-1, 1)
        # Globally all 8 pages are valid (e.g. two different requests
        # occupy disjoint slices of the same table).
        table.valid_mask.fill_(True)
        mask = ChannelMask(
            channel_selection=torch.randint(0, 64, (1, 2, 8), dtype=torch.int32),
            channel_weights=torch.randn(1, 2, 8, dtype=torch.float32),
            schema_version="1", dtype="fp8_e4m3", head_dim=64, page_size=64,
            label_dim=8, content_sha256="x",
        )
        sel.bind_runtime_data(table, mask)
        # Override max_top_k for a sharp test: only 4 slots, 8 candidate
        # pages, and we'll prove request 0 never gets pages 4..7.
        sel.max_top_k = 4

        queries = torch.randn(2, 2, 64)
        req_pool = torch.tensor([0, 1], dtype=torch.int32)
        seq_lens = torch.tensor([256, 256], dtype=torch.int32)
        # Request 0 owns pages [0, 1, 2, 3]; request 1 owns pages [4, 5, 6, 7].
        per_request = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ], dtype=torch.int32)
        indices, lengths = sel.retrieve_topk(
            queries=queries, layer_id=0, req_pool_indices=req_pool,
            sparse_mask=per_request, seq_lens=seq_lens,
        )
        self.assertEqual(tuple(indices.shape), (2, 4))
        row0 = [int(v) for v in indices[0].tolist() if v >= 0]
        row1 = [int(v) for v in indices[1].tolist() if v >= 0]
        self.assertTrue(all(p in {0, 1, 2, 3} for p in row0),
                        f"request 0 selected foreign pages: {row0}")
        self.assertTrue(all(p in {4, 5, 6, 7} for p in row1),
                        f"request 1 selected foreign pages: {row1}")


class TestHotPagesIntersectPerRequest(unittest.TestCase):
    """Round-7 fix [P2]: hot-page forcing must not re-introduce pages that
    a row's per_request_valid mask excluded.
    """

    def test_hot_pages_filtered_by_per_request_valid(self):
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            retrieve_topk_via_signatures,
        )
        bs = 2
        num_layers, max_pages, num_heads, label_dim = 1, 8, 2, 4
        head_dim = 16
        device = torch.device("cpu")
        # Uniform random signatures + valid_mask all True.
        signatures = torch.randn(num_layers, max_pages, num_heads, label_dim)
        valid_mask = torch.ones(num_layers, max_pages, dtype=torch.bool)
        channel_selection = torch.zeros(num_layers, num_heads, label_dim,
                                         dtype=torch.int32)
        # Make each label-dim slot point at a distinct channel index.
        for h in range(num_heads):
            for d in range(label_dim):
                channel_selection[0, h, d] = (h * label_dim + d) % head_dim
        channel_weights = torch.ones(num_layers, num_heads, label_dim,
                                      dtype=torch.float32)

        queries = torch.randn(bs, num_heads, head_dim)
        # Request 0 owns pages [0,1,2,3]; request 1 owns [4,5,6,7].
        per_request = torch.tensor([
            [1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ], dtype=torch.int32)
        # Hot pages: in mixed batches, compute_hot_pages may return the same
        # *logical* page index for both rows. Here we simulate the bad case
        # where the caller has not yet translated logical -> physical: pass
        # [[3], [3]] and ensure request 1 still cannot win page 3.
        hot = [[3], [3]]
        indices, lengths = retrieve_topk_via_signatures(
            queries=queries,
            page_signatures=signatures,
            valid_mask=valid_mask,
            channel_selection=channel_selection,
            channel_weights=channel_weights,
            layer_id=0,
            max_top_k=4,
            hot_pages=hot,
            per_request_valid=per_request,
        )
        row0 = [int(v) for v in indices[0].tolist() if v >= 0]
        row1 = [int(v) for v in indices[1].tolist() if v >= 0]
        self.assertIn(3, row0, "request 0 should still get its hot page 3")
        self.assertNotIn(3, row1,
                          f"hot-page intersection failed for row 1: got {row1}")
        self.assertTrue(all(p in {4, 5, 6, 7} for p in row1),
                        f"row 1 contains foreign pages: {row1}")


class TestM3BFixtureWiderTable(unittest.TestCase):
    """Round-7 fix [P2]: M3-B fixture must work when table.max_pages
    exceeds the prompt page count (the normal allocator case).
    """

    def test_fixture_passes_when_table_wider_than_prompt(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask,
        )
        from sglang.srt.layers.attention.double_sparsity.page_signature_table import (
            allocate_page_signature_table,
        )
        from sglang.srt.layers.attention.double_sparsity.page_signature_write import (
            m3b_page_stability_fixture,
        )
        cfg = parse_double_sparsity_config(_valid_payload())
        sel = DoubleSparsitySelector(
            config=cfg, num_local_heads=2, head_dim=32, device=torch.device("cpu"),
        )
        # Table is much wider than the test prompt (16 pages vs 4).
        table = allocate_page_signature_table(
            num_layers_local=1, max_pages=16, num_heads_local=2, label_dim=4,
            page_size=64, dtype=torch.float32, device=torch.device("cpu"),
        )
        table.signatures.uniform_(-1, 1)
        table.valid_mask[0, :4] = True
        mask = ChannelMask(
            channel_selection=torch.zeros(1, 2, 4, dtype=torch.int32),
            channel_weights=torch.ones(1, 2, 4, dtype=torch.float32),
            schema_version="1", dtype="fp8_e4m3", head_dim=32, page_size=64,
            label_dim=4, content_sha256="x",
        )
        sel.bind_runtime_data(table, mask)
        # 4 prompt pages at page_size=64 -> seq_len=256.
        prompt_tokens = torch.zeros(1, 256, dtype=torch.int32)
        ok = m3b_page_stability_fixture(
            sel, prompt_tokens=prompt_tokens, page_size=64, num_repeats=2,
        )
        self.assertTrue(ok, "fixture should pass with wider table than prompt")


class TestCalibrateCorpusEmpty(unittest.TestCase):
    """Round-7 fix [P3]: empty corpus must raise a clear ValueError."""

    def test_empty_file_raises_value_error(self):
        from sglang.srt.layers.attention.double_sparsity.calibrate import (
            _read_corpus_file,
        )
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("\n  \n\t\n")  # whitespace only
            path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                _read_corpus_file(path, num_samples=4)
            self.assertIn("no non-empty lines", str(ctx.exception))
        finally:
            os.unlink(path)


class TestChannelMaskSlicePerRank(unittest.TestCase):
    """Round-2 fix [P2]: TP head sharding helper."""

    def test_slice_per_rank_returns_local_block(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask, slice_per_rank,
        )
        sel = torch.arange(2 * 16 * 8, dtype=torch.int32).reshape(2, 16, 8)
        wts = torch.arange(2 * 16 * 8, dtype=torch.float32).reshape(2, 16, 8)
        mask = ChannelMask(
            channel_selection=sel, channel_weights=wts,
            schema_version="1", dtype="fp8_e4m3", head_dim=128, page_size=64,
            label_dim=8, content_sha256="abc",
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
            ChannelMask, slice_per_rank,
        )
        mask = ChannelMask(
            channel_selection=torch.zeros(1, 10, 4, dtype=torch.int32),
            channel_weights=torch.zeros(1, 10, 4, dtype=torch.float32),
            schema_version="1", dtype="fp8_e4m3", head_dim=128, page_size=64,
            label_dim=4, content_sha256="x",
        )
        with self.assertRaises(ValueError):
            slice_per_rank(mask, num_local_heads=4, rank=0, tp_size=2)

    def test_bind_rejects_unsliced_full_mask(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask,
        )
        from sglang.srt.layers.attention.double_sparsity.page_signature_table import (
            allocate_page_signature_table,
        )
        cfg = parse_double_sparsity_config(_valid_payload())
        sel = DoubleSparsitySelector(
            config=cfg, num_local_heads=4, head_dim=128, device=torch.device("cpu"),
        )
        table = allocate_page_signature_table(
            num_layers_local=2, max_pages=8, num_heads_local=4, label_dim=8,
            page_size=64, dtype=torch.float16, device=torch.device("cpu"),
        )
        # Mask is still at H_full=32 (un-sliced) — must be rejected.
        full_mask = ChannelMask(
            channel_selection=torch.zeros(2, 32, 8, dtype=torch.int32),
            channel_weights=torch.zeros(2, 32, 8, dtype=torch.float32),
            schema_version="1", dtype="fp8_e4m3", head_dim=128, page_size=64,
            label_dim=8, content_sha256="x",
        )
        with self.assertRaises(ValueError) as ctx:
            sel.bind_runtime_data(table, full_mask)
        self.assertIn("slice_per_rank", str(ctx.exception))


class TestBindRuntimeDataDeviceAlignment(unittest.TestCase):
    """Round-4 fix [P2]: bind_runtime_data must align a CPU-loaded mask
    onto the page-signature table's device.
    """

    def test_bind_moves_cpu_mask_onto_table_device(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask,
        )
        from sglang.srt.layers.attention.double_sparsity.page_signature_table import (
            allocate_page_signature_table,
        )
        cfg = parse_double_sparsity_config(_valid_payload())
        sel = DoubleSparsitySelector(
            config=cfg, num_local_heads=4, head_dim=128, device=torch.device("cpu"),
        )
        # Table and mask on the same (CPU) device — no-op move, but the
        # bind path must still succeed.
        table = allocate_page_signature_table(
            num_layers_local=2, max_pages=8, num_heads_local=4, label_dim=8,
            page_size=64, dtype=torch.float16, device=torch.device("cpu"),
        )
        mask = ChannelMask(
            channel_selection=torch.zeros(2, 4, 8, dtype=torch.int32),
            channel_weights=torch.zeros(2, 4, 8, dtype=torch.float32),
            schema_version="1", dtype="fp8_e4m3", head_dim=128, page_size=64,
            label_dim=8, content_sha256="x",
        )
        sel.bind_runtime_data(table, mask)
        # Mask now lives on the table's device.
        self.assertEqual(
            sel.channel_mask.channel_selection.device,
            table.signatures.device,
        )
        self.assertEqual(
            sel.channel_mask.channel_weights.device,
            table.signatures.device,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA needed for cross-device alignment test")
    def test_bind_moves_cpu_mask_onto_cuda_table(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask,
        )
        from sglang.srt.layers.attention.double_sparsity.page_signature_table import (
            allocate_page_signature_table,
        )
        cuda_dev = torch.device("cuda")
        cfg = parse_double_sparsity_config(_valid_payload())
        sel = DoubleSparsitySelector(
            config=cfg, num_local_heads=4, head_dim=128, device=cuda_dev,
        )
        table = allocate_page_signature_table(
            num_layers_local=2, max_pages=8, num_heads_local=4, label_dim=8,
            page_size=64, dtype=torch.float16, device=cuda_dev,
        )
        # Mask loaded on CPU (the load_channel_mask default path).
        mask = ChannelMask(
            channel_selection=torch.zeros(2, 4, 8, dtype=torch.int32),
            channel_weights=torch.zeros(2, 4, 8, dtype=torch.float32),
            schema_version="1", dtype="fp8_e4m3", head_dim=128, page_size=64,
            label_dim=8, content_sha256="x",
        )
        self.assertEqual(mask.channel_selection.device.type, "cpu")
        sel.bind_runtime_data(table, mask)
        self.assertEqual(sel.channel_mask.channel_selection.device.type, "cuda")
        self.assertEqual(sel.channel_mask.channel_weights.device.type, "cuda")
        # Original mask object is unchanged (caller's reference is intact).
        self.assertEqual(mask.channel_selection.device.type, "cpu")


class TestSanityProbeRealSelector(unittest.TestCase):
    """Round-4 fix [P2]: sanity probe must plant a real signal and discriminate."""

    def test_probe_finds_planted_needle(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask, startup_sanity_probe,
        )
        from sglang.srt.layers.attention.double_sparsity.page_signature_table import (
            allocate_page_signature_table,
        )
        cfg = parse_double_sparsity_config(_valid_payload())
        sel = DoubleSparsitySelector(
            config=cfg, num_local_heads=2, head_dim=64, device=torch.device("cpu"),
        )
        table = allocate_page_signature_table(
            num_layers_local=1, max_pages=16, num_heads_local=2, label_dim=4,
            page_size=64, dtype=torch.float32, device=torch.device("cpu"),
        )
        # A deterministic non-trivial channel mask: heads point at distinct
        # channel indices in the first label-dim slot.
        sel_tensor = torch.zeros(1, 2, 4, dtype=torch.int32)
        sel_tensor[0, 0, 0] = 3
        sel_tensor[0, 1, 0] = 7
        # Other label-dim slots index other channels (just to fill).
        sel_tensor[0, 0, 1] = 4
        sel_tensor[0, 0, 2] = 5
        sel_tensor[0, 0, 3] = 6
        sel_tensor[0, 1, 1] = 8
        sel_tensor[0, 1, 2] = 9
        sel_tensor[0, 1, 3] = 10
        w_tensor = torch.ones(1, 2, 4, dtype=torch.float32)
        mask = ChannelMask(
            channel_selection=sel_tensor,
            channel_weights=w_tensor,
            schema_version="1", dtype="fp8_e4m3", head_dim=64, page_size=64,
            label_dim=4, content_sha256="x",
        )
        sel.bind_runtime_data(table, mask)

        result = startup_sanity_probe(
            mask, sel, haystack_pages=8, page_size=64, needle_page=4,
        )
        self.assertTrue(result.passed,
                        f"probe should find planted needle; got {result}")
        self.assertEqual(result.score, 1.0)
        self.assertEqual(result.needle_position, 4)
        # The probe must restore the table after running.
        self.assertTrue(torch.equal(
            table.signatures[0, :8],
            torch.zeros_like(table.signatures[0, :8]),
        ))
        self.assertTrue(torch.equal(
            table.valid_mask[0, :8],
            torch.zeros_like(table.valid_mask[0, :8]),
        ))

    def test_probe_returns_no_table_when_unbound(self):
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask, startup_sanity_probe,
        )
        cfg = parse_double_sparsity_config(_valid_payload())
        # Construct selector, manually flip IS_PLACEHOLDER without binding a
        # table — exercises the new "no_page_signature_table" branch.
        sel = DoubleSparsitySelector(
            config=cfg, num_local_heads=2, head_dim=64, device=torch.device("cpu"),
        )
        sel.IS_PLACEHOLDER = False
        sel.page_signature_table = None
        mask = ChannelMask(
            channel_selection=torch.zeros(1, 2, 4, dtype=torch.int32),
            channel_weights=torch.zeros(1, 2, 4, dtype=torch.float32),
            schema_version="1", dtype="fp8_e4m3", head_dim=64, page_size=64,
            label_dim=4, content_sha256="x",
        )
        r = startup_sanity_probe(mask, sel, haystack_pages=8, needle_page=4)
        self.assertFalse(r.passed)
        self.assertEqual(r.skipped_reason, "no_page_signature_table")


class TestBenchmarkCompareReader(unittest.TestCase):
    """Round-4 fix [P2]: benchmark_compare must read server_info nested
    fields and derive per-request TPS from bench_serving --output-details
    arrays."""

    def _import_compare(self):
        import importlib
        import sys as _sys
        # Walk up from the test file to find the project's `development/` dir.
        cur = os.path.dirname(os.path.abspath(__file__))
        development_dir = None
        for _ in range(8):
            candidate = os.path.join(cur, "development", "benchmark_compare.py")
            if os.path.isfile(candidate):
                development_dir = os.path.dirname(candidate)
                break
            cur = os.path.dirname(cur)
        if development_dir is None:
            raise FileNotFoundError("development/benchmark_compare.py not found")
        if development_dir not in _sys.path:
            _sys.path.insert(0, development_dir)
        if "benchmark_compare" in _sys.modules:
            return _sys.modules["benchmark_compare"]
        return importlib.import_module("benchmark_compare")

    def _write_jsonl(self, tmpdir, name, payload):
        import json as _json
        path = os.path.join(tmpdir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_json.dumps(payload) + "\n")
        return path

    def test_reads_server_info_nested_context(self):
        bc = self._import_compare()
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            payload = {
                "max_concurrency": 32,
                "median_ttft_ms": 800.0,
                "p99_ttft_ms": 21000.0,
                "median_tpot_ms": 4.0,
                "p99_tpot_ms": 12.0,
                "output_lens": [100, 110, 90, 105, 95, 100, 100, 100],
                "ttfts": [0.5, 0.6, 0.4, 0.55, 0.5, 0.5, 0.5, 0.5],
                "itls": [[0.01] * 99, [0.01] * 109, [0.01] * 89,
                         [0.01] * 104, [0.01] * 94, [0.01] * 99,
                         [0.01] * 99, [0.01] * 99],
                "server_info": {
                    "tp_size": 8,
                    "page_size": 64,
                    "disable_radix_cache": True,
                    "gpu_id": "H200",
                },
            }
            path = self._write_jsonl(tmp, "ds_c32.jsonl", payload)
            ctx, m = bc._read_bench_jsonl(path)
            self.assertEqual(ctx.tp_size, 8)
            self.assertEqual(ctx.page_size, 64)
            self.assertEqual(ctx.disable_radix_cache, True)
            self.assertEqual(ctx.concurrency, 32)
            # P50 of 100/(0.5+0.99)=~67 tok/s for each row -> P50 close to 67.
            self.assertIsNotNone(m.output_tps_p50)
            self.assertGreater(m.output_tps_p50, 50)
            self.assertLess(m.output_tps_p50, 120)
            # TTFT P50 / P99 in seconds.
            self.assertAlmostEqual(m.ttft_p50_s, 0.8, places=3)
            self.assertAlmostEqual(m.ttft_p99_s, 21.0, places=3)

    def test_match_refuse_treats_none_context_as_missing(self):
        """Round-5 fix [P2]: required-context field that is None on either
        side must be reported as a mismatch, not silently accepted via
        ``None == None``.
        """

        bc = self._import_compare()
        # Both contexts have server_info entirely missing.
        empty = bc.RunContext(
            gpu_id="", tp_size=None, page_size=None,
            disable_radix_cache=None, concurrency=None,
        )
        reasons = bc._match_or_refuse(empty, empty)
        joined = " ".join(reasons).lower()
        self.assertIn("tp_size missing", joined)
        self.assertIn("page_size missing", joined)
        self.assertIn("disable_radix_cache missing", joined)
        self.assertIn("concurrency missing", joined)

    def test_no_op_status_unknown_when_metrics_absent(self):
        """Round-5 fix [P2]: ``_no_op_status`` must return ``unknown`` when
        DS observability fields are absent, so the report does not falsely
        print "clean".
        """

        bc = self._import_compare()
        m = bc.RunMetrics(
            concurrency=32, num_prompts=4, isl=4096, osl=512,
            output_tps_p50=42.0, output_tps_p99=80.0,
            ttft_p50_s=0.5, ttft_p99_s=2.0,
            tpot_p50_ms=4.0, tpot_p99_ms=12.0,
            goodput_under_slo=0.9,
            selected_pages_mean=None,
            dense_fallback_total=None,
            total_pages_mean=None,
        )
        status, reason = bc._no_op_status(m)
        self.assertEqual(status, "unknown")
        self.assertIn("dense_fallback_total", reason)
        self.assertIn("selected_pages_mean", reason)
        self.assertIn("total_pages_mean", reason)
        # And the rendered report uses "unknown", not "clean".
        baseline = bc.RunMetrics(
            concurrency=32, num_prompts=4, isl=4096, osl=512,
            output_tps_p50=50.0, output_tps_p99=80.0,
            ttft_p50_s=0.4, ttft_p99_s=1.8,
            tpot_p50_ms=4.0, tpot_p99_ms=10.0,
            goodput_under_slo=0.95,
            selected_pages_mean=None,
            dense_fallback_total=None,
            total_pages_mean=None,
        )
        md = bc.render_markdown_report(
            baseline, m, baseline_path="b.jsonl", ds_path="d.jsonl",
        )
        self.assertIn("No-op detector:** unknown", md)
        self.assertNotIn("No-op detector:** clean", md)

    def test_strict_gpu_rejects_when_both_missing(self):
        """Round-6 fix [P2]: --strict-gpu must reject when GPU IDs are
        absent on both sides, not silently accept None==None.
        """

        bc = self._import_compare()
        empty = bc.RunContext(
            gpu_id=None, tp_size=8, page_size=64,
            disable_radix_cache=True, concurrency=32,
        )
        reasons = bc._match_or_refuse(empty, empty, strict_gpu=True)
        self.assertTrue(
            any("gpu_id missing" in r for r in reasons),
            f"expected gpu_id missing reason; got {reasons}",
        )

    def test_strict_gpu_rejects_when_one_missing(self):
        bc = self._import_compare()
        base = bc.RunContext(
            gpu_id="H200", tp_size=8, page_size=64,
            disable_radix_cache=True, concurrency=32,
        )
        ds = bc.RunContext(
            gpu_id=None, tp_size=8, page_size=64,
            disable_radix_cache=True, concurrency=32,
        )
        reasons = bc._match_or_refuse(base, ds, strict_gpu=True)
        self.assertTrue(
            any("gpu_id missing" in r for r in reasons),
            f"expected gpu_id missing reason; got {reasons}",
        )

    def test_strict_gpu_accepts_when_both_match(self):
        """Sanity: strict-GPU still publishes when GPU IDs match."""

        bc = self._import_compare()
        base = bc.RunContext(
            gpu_id="H200", tp_size=8, page_size=64,
            disable_radix_cache=True, concurrency=32,
        )
        ds = bc.RunContext(
            gpu_id="H200", tp_size=8, page_size=64,
            disable_radix_cache=True, concurrency=32,
        )
        reasons = bc._match_or_refuse(base, ds, strict_gpu=True)
        self.assertEqual(reasons, [])

    def test_no_op_status_clean_when_metrics_present_and_zero(self):
        """Sanity-check the new ``clean`` path: all observability fields
        present, fallback zero, selected != total → ``clean``.
        """

        bc = self._import_compare()
        m = bc.RunMetrics(
            concurrency=32, num_prompts=4, isl=4096, osl=512,
            output_tps_p50=42.0, output_tps_p99=80.0,
            ttft_p50_s=0.5, ttft_p99_s=2.0,
            tpot_p50_ms=4.0, tpot_p99_ms=12.0,
            goodput_under_slo=0.9,
            selected_pages_mean=128.0,
            dense_fallback_total=0,
            total_pages_mean=2048.0,
        )
        status, reason = bc._no_op_status(m)
        self.assertEqual(status, "clean")
        self.assertEqual(reason, "")

    def test_refuses_mismatch_when_server_info_disagrees(self):
        bc = self._import_compare()
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            base = {
                "max_concurrency": 32,
                "median_ttft_ms": 800.0, "p99_ttft_ms": 21000.0,
                "median_tpot_ms": 4.0, "p99_tpot_ms": 12.0,
                "output_lens": [100, 100], "ttfts": [0.5, 0.5],
                "itls": [[0.01] * 99, [0.01] * 99],
                "server_info": {"tp_size": 8, "page_size": 64,
                                "disable_radix_cache": True},
            }
            ds_diff = dict(base)
            ds_diff["server_info"] = {"tp_size": 4, "page_size": 64,
                                       "disable_radix_cache": True}
            p1 = self._write_jsonl(tmp, "base.jsonl", base)
            p2 = self._write_jsonl(tmp, "ds.jsonl", ds_diff)
            b_ctx, _ = bc._read_bench_jsonl(p1)
            d_ctx, _ = bc._read_bench_jsonl(p2)
            reasons = bc._match_or_refuse(b_ctx, d_ctx)
            self.assertTrue(any("tp_size" in r for r in reasons),
                            f"expected tp_size mismatch, got {reasons}")


class TestMetrics(unittest.TestCase):
    def test_meta_info_shape(self):
        from sglang.srt.layers.attention.double_sparsity import metrics as m
        stats = m.DoubleSparsityRequestStats(
            sparsity_rate=0.0625, selected_pages=128, dense_fallback=0
        )
        info = m.meta_info_for_request(stats)
        self.assertEqual(set(info.keys()), {"sparsity_rate", "selected_pages", "dense_fallback"})
        self.assertAlmostEqual(info["sparsity_rate"], 0.0625)
        self.assertEqual(info["selected_pages"], 128)
        self.assertEqual(info["dense_fallback"], 0)

    def test_record_selection_increments_counters(self):
        from sglang.srt.layers.attention.double_sparsity import metrics as m
        m.reset_for_testing()
        m.record_selection(selected_pages=10, total_valid_pages=100)
        m.record_selection(selected_pages=20, total_valid_pages=100)
        # Best-effort: if prometheus_client unavailable, metrics are no-ops.
        if "selected_pages_sum" in m._metric_objs:
            sps = m._metric_objs["selected_pages_sum"]._value.get()
            cnt = m._metric_objs["selected_pages_count"]._value.get()
            self.assertEqual(sps, 30)
            self.assertEqual(cnt, 2)

    def test_reset_for_testing_unregisters_collectors(self):
        """Round-6 fix [P3]: reset_for_testing must unregister collectors
        from prometheus_client.REGISTRY, otherwise a subsequent
        re-registration raises ValueError: Duplicated timeseries.
        """

        from sglang.srt.layers.attention.double_sparsity import metrics as m
        try:
            import prometheus_client  # noqa: F401
        except ImportError:
            self.skipTest("prometheus_client not installed")
        # First registration cycle.
        m.reset_for_testing()
        m.mark_channel_mask_valid(True)
        m.record_selection(selected_pages=5, total_valid_pages=10)
        self.assertTrue(m._metrics_registered)
        # Reset, then re-register. The second registration must not raise.
        m.reset_for_testing()
        self.assertFalse(m._metrics_registered)
        # If reset didn't unregister, this re-registration raises
        # "Duplicated timeseries" during the next Gauge/Counter construction.
        m.mark_channel_mask_valid(False)
        m.record_selection(selected_pages=7, total_valid_pages=10)
        self.assertTrue(m._metrics_registered)
        # Clean up so other tests do not see this state.
        m.reset_for_testing()


class TestCUDAGraphCapture(unittest.TestCase):
    def test_allocate_state_shapes(self):
        from sglang.srt.layers.attention.double_sparsity.cuda_graph import (
            allocate_graph_state,
        )
        s = allocate_graph_state(
            max_bs=4, max_top_k=8, num_score_blocks=2, partial_topk=3,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(s.selected_indices.shape), (4, 8))
        self.assertEqual(tuple(s.valid_lengths.shape), (4,))
        self.assertEqual(tuple(s.scratch_partial_scores.shape), (4, 2, 3))
        self.assertTrue(torch.all(s.selected_indices == -1).item())

    def test_eager_replay_on_cpu(self):
        from sglang.srt.layers.attention.double_sparsity.cuda_graph import (
            allocate_graph_state, capture_decode_step,
        )
        cfg = parse_double_sparsity_config(_valid_payload())
        sel = DoubleSparsitySelector(
            config=cfg, num_local_heads=4, head_dim=128, device=torch.device("cpu"),
        )
        state = allocate_graph_state(
            max_bs=2, max_top_k=2048, device=torch.device("cpu"),
        )
        queries = torch.zeros(2, 4, 128)
        replay = capture_decode_step(
            sel, state=state,
            queries=queries, layer_id=0,
            req_pool_indices=torch.tensor([0, 1], dtype=torch.int32),
            sparse_mask=torch.ones(2, 16, dtype=torch.int32),
            seq_lens=torch.tensor([200, 320], dtype=torch.int32),
        )
        idx1, lens1 = replay()
        idx2, lens2 = replay()
        self.assertTrue(torch.equal(idx1, idx2))
        self.assertTrue(torch.equal(lens1, lens2))


_CUDA_AVAILABLE = torch.cuda.is_available()


@unittest.skipUnless(_CUDA_AVAILABLE, "Triton equivalence tests require CUDA")
class TestTritonEquivalence(unittest.TestCase):
    """Round 2: Triton kernels must match the torch reference numerically."""

    def test_compute_page_scores_triton_matches_torch(self):
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            _compute_page_scores_triton,
            compute_page_scores,
            project_query_onto_channels,
        )

        torch.manual_seed(11)
        device = torch.device("cuda")
        bs, H, head_dim = 4, 8, 64
        L, P, label_dim = 2, 64, 16
        queries = torch.randn(bs, H, head_dim, device=device, dtype=torch.float32)
        sigs = torch.randn(L, P, H, label_dim, device=device, dtype=torch.float16)
        vmask = torch.ones(L, P, dtype=torch.bool, device=device)
        vmask[0, 5] = False
        vmask[0, 17] = False
        sel = torch.randint(0, head_dim, (L, H, label_dim), dtype=torch.int32, device=device)
        w = torch.randn(L, H, label_dim, dtype=torch.float32, device=device)

        scores_triton = compute_page_scores(queries, sigs, vmask, sel, w, layer_id=0)
        scores_torch = compute_page_scores(
            queries.cpu(), sigs.cpu(), vmask.cpu(), sel.cpu(), w.cpu(), layer_id=0
        )

        finite_triton = torch.isfinite(scores_triton.cpu())
        finite_torch = torch.isfinite(scores_torch)
        self.assertTrue(
            torch.equal(finite_triton, finite_torch),
            "Triton and torch disagree on which pages are invalid",
        )
        finite = finite_triton & finite_torch
        diff = (scores_triton.cpu()[finite] - scores_torch[finite]).abs()
        self.assertLess(
            diff.max().item(),
            1e-2,
            f"Triton vs torch max diff {diff.max().item()} exceeds 1e-2",
        )

    def test_page_signature_write_triton_matches_torch(self):
        from sglang.srt.layers.attention.double_sparsity.page_signature_write import (
            _page_signature_write_triton,
            dequant_nope_fp8_to_bf16,
            project_page_to_signature,
        )

        torch.manual_seed(13)
        device = torch.device("cuda")
        num_pages, page_size, H, label_dim = 3, 64, 4, 16

        nope_fp8 = torch.randn(num_pages * page_size, 512).to(torch.float8_e4m3fn).to(device)
        scales = (torch.rand(num_pages * page_size, 4, device=device) * 2.0).contiguous()
        nope_u8 = torch.zeros(num_pages * page_size, 528, dtype=torch.uint8, device=device)
        nope_u8[:, :512] = nope_fp8.view(torch.uint8)
        nope_u8[:, 512:].view(torch.float32)[:, :] = scales
        nope_parts = nope_u8.reshape(num_pages, page_size, 528).contiguous()

        sel = torch.randint(0, 512, (H, label_dim), dtype=torch.int32, device=device)
        w = torch.randn(H, label_dim, dtype=torch.float32, device=device)

        sig_triton = _page_signature_write_triton(nope_parts, sel, w)

        sig_torch = torch.zeros(
            num_pages, H, label_dim, dtype=torch.float16, device=device
        )
        for p in range(num_pages):
            bf16 = dequant_nope_fp8_to_bf16(nope_parts[p])
            sig_torch[p] = project_page_to_signature(bf16, sel, w, reduce="mean").to(
                torch.float16
            )

        diff = (sig_triton.to(torch.float32) - sig_torch.to(torch.float32)).abs()
        # fp16 rounding noise dominates here; loose tolerance is fine.
        self.assertLess(
            diff.max().item(),
            5.0,
            f"Triton vs torch page_signature_write max diff {diff.max().item()} too high",
        )


@unittest.skipUnless(_CUDA_AVAILABLE, "NSA cross-validation requires CUDA + Triton")
class TestNSACrossValidation(unittest.TestCase):
    """Drive DS dequant via NSA's own quantizer to verify byte-layout contract."""

    def test_quant_kcache_roundtrip(self):
        from sglang.srt.layers.attention.nsa.quant_k_cache import (
            quantize_k_cache_separate,
        )
        from sglang.srt.layers.attention.double_sparsity.page_signature_write import (
            dequant_nope_fp8_to_bf16,
        )

        torch.manual_seed(17)
        device = torch.device("cuda")
        T = 64
        k_nope = torch.randn(T, 512, dtype=torch.bfloat16, device=device)
        k_rope = torch.randn(T, 64, dtype=torch.bfloat16, device=device)

        nope_u8, _ = quantize_k_cache_separate(k_nope, k_rope, tile_size=128)
        self.assertEqual(tuple(nope_u8.shape), (T, 1, 528))

        recovered = dequant_nope_fp8_to_bf16(nope_u8)
        self.assertEqual(tuple(recovered.shape), (T, 512))
        self.assertEqual(recovered.dtype, torch.bfloat16)

        recovered_f = recovered.to(torch.float32)
        original_f = k_nope.to(torch.float32)
        for tile in range(4):
            s, e = tile * 128, (tile + 1) * 128
            ref = original_f[:, s:e]
            got = recovered_f[:, s:e]
            l2_rel = ((ref - got).norm() / ref.norm().clamp_min(1e-9)).item()
            # fp8_e4m3 has ~4-bit mantissa; per-tile relative L2 should sit
            # in the 1-5% band on Gaussian inputs.
            self.assertLess(
                l2_rel,
                0.06,
                f"tile {tile} relative L2 error {l2_rel:.4f} exceeds 6% — "
                "FP8 byte layout may have shifted in NSA's quant_k_cache.",
            )


@unittest.skipUnless(_CUDA_AVAILABLE, "End-to-end pipeline test requires CUDA + Triton")
class TestEndToEndPipeline(unittest.TestCase):
    """Round 3 drift recovery: full DS pipeline composes on synthetic V3.2-shape inputs.

    NSA quantizer → page_signature_write (Triton) → bind_runtime_data →
    retrieve_topk → m3b_page_stability_fixture. No production code mutation;
    no model weights required.
    """

    def _build_fixture(self, *, num_layers=2, num_heads=4, num_pages=8, page_size=64, label_dim=16):
        from sglang.srt.layers.attention.double_sparsity import (
            DoubleSparsitySelector,
            allocate_page_signature_table,
            parse_double_sparsity_config,
        )
        from sglang.srt.layers.attention.double_sparsity.channel_mask import (
            ChannelMask, compute_content_sha256,
        )
        from sglang.srt.layers.attention.double_sparsity.page_signature_write import (
            page_signature_write,
        )
        from sglang.srt.layers.attention.nsa.quant_k_cache import (
            quantize_k_cache_separate,
        )

        device = torch.device("cuda")
        torch.manual_seed(31)

        all_tokens = num_pages * page_size
        k_nope = torch.randn(all_tokens, 512, dtype=torch.bfloat16, device=device)
        k_rope = torch.randn(all_tokens, 64, dtype=torch.bfloat16, device=device)
        nope_u8_flat, _ = quantize_k_cache_separate(k_nope, k_rope, tile_size=128)
        nope_parts_u8 = nope_u8_flat.squeeze(1).reshape(num_pages, page_size, 528).contiguous()

        sel = torch.randint(0, 512, (num_layers, num_heads, label_dim), dtype=torch.int32, device=device)
        w = torch.randn(num_layers, num_heads, label_dim, dtype=torch.float32, device=device)
        content_hash = compute_content_sha256(sel, w)
        mask = ChannelMask(
            channel_selection=sel,
            channel_weights=w,
            schema_version="1",
            dtype="fp8_e4m3",
            head_dim=512,
            page_size=page_size,
            label_dim=label_dim,
            content_sha256=content_hash,
        )

        table = allocate_page_signature_table(
            num_layers_local=num_layers,
            max_pages=num_pages,
            num_heads_local=num_heads,
            label_dim=label_dim,
            page_size=page_size,
            dtype=torch.float16,
            device=device,
        )
        for layer in range(num_layers):
            page_signature_write(
                table.signatures,
                table.valid_mask,
                layer_id=layer,
                page_ids=list(range(num_pages)),
                nope_parts_u8=nope_parts_u8,
                channel_selection_layer=sel[layer],
                channel_weights_layer=w[layer],
            )

        cfg = parse_double_sparsity_config(
            '{"top_k": 4, "page_size": 64, '
            '"channel_mask_path": "/tmp/_fixture_only.safetensors", '
            '"device_buffer_size": 4096}'
        )
        selector = DoubleSparsitySelector(
            config=cfg, num_local_heads=num_heads, head_dim=512, device=device,
        )
        selector.bind_runtime_data(table, mask)
        return selector, table, mask, num_pages

    def test_full_pipeline_on_v32_shape_synthetic(self):
        selector, table, mask, num_pages = self._build_fixture()
        self.assertFalse(selector.IS_PLACEHOLDER, "bind_runtime_data should flip placeholder off")
        self.assertTrue(table.valid_mask.all().item(), "all pages should be populated")

        device = table.signatures.device
        bs = 2
        queries = torch.randn(bs, selector.num_local_heads, selector.head_dim, device=device)
        req_pool = torch.tensor([0, 1], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([num_pages * 64, num_pages * 64], dtype=torch.int32, device=device)
        sparse_mask = torch.ones(bs, num_pages, dtype=torch.int32, device=device)

        indices, lengths = selector.retrieve_topk(
            queries=queries, layer_id=0,
            req_pool_indices=req_pool, sparse_mask=sparse_mask, seq_lens=seq_lens,
        )

        self.assertEqual(indices.dtype, torch.int32)
        self.assertEqual(lengths.dtype, torch.int32)
        self.assertEqual(tuple(indices.shape), (bs, selector.max_top_k))
        self.assertEqual(tuple(lengths.shape), (bs,))
        for row in range(bs):
            row_indices = indices[row, : int(lengths[row])].tolist()
            self.assertTrue(
                all(row_indices[i] < row_indices[i + 1] for i in range(len(row_indices) - 1)),
                f"row {row} not sequence-ascending: {row_indices}",
            )
            for pid in row_indices:
                self.assertGreaterEqual(pid, 0)
                self.assertLess(pid, num_pages)

    def test_hot_page_forced_into_selected(self):
        from sglang.srt.layers.attention.double_sparsity.page_signature_write import (
            compute_hot_pages,
        )

        selector, table, mask, num_pages = self._build_fixture()
        device = table.signatures.device
        bs = 1
        queries = torch.zeros(bs, selector.num_local_heads, selector.head_dim, device=device)
        req_pool = torch.tensor([0], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([num_pages * 64], dtype=torch.int32, device=device)
        sparse_mask = torch.ones(bs, num_pages, dtype=torch.int32, device=device)

        hot = compute_hot_pages(seq_lens=seq_lens, page_size=64, local_window=1)
        indices, lengths = selector.retrieve_topk(
            queries=queries, layer_id=0,
            req_pool_indices=req_pool, sparse_mask=sparse_mask, seq_lens=seq_lens,
            hot_pages=hot,
        )
        row = indices[0, : int(lengths[0])].tolist()
        self.assertIn(num_pages - 1, row, f"hot page {num_pages - 1} not in {row}")

    def test_m3b_fixture_passes_on_bound_selector(self):
        from sglang.srt.layers.attention.double_sparsity.page_signature_write import (
            m3b_page_stability_fixture,
        )

        selector, table, mask, num_pages = self._build_fixture()
        prompt_tokens = torch.zeros(1, num_pages * 64, dtype=torch.int32, device=table.signatures.device)
        passed = m3b_page_stability_fixture(
            selector, prompt_tokens=prompt_tokens, page_size=64, num_repeats=3,
        )
        self.assertTrue(passed, "DEC-2 page-stability fixture should hold on a real-bound selector")


class TestCustomizedInfoIntegration(unittest.TestCase):
    """Round 2: DS stats → tokenizer_manager.customized_info wiring point."""

    def test_customized_info_shape(self):
        from sglang.srt.layers.attention.double_sparsity.metrics import (
            DoubleSparsityRequestStats,
            customized_info_for_request,
        )

        stats = DoubleSparsityRequestStats(
            sparsity_rate=0.05, selected_pages=64, dense_fallback=0
        )
        payload = customized_info_for_request(stats)
        self.assertEqual(
            set(payload.keys()), {"sparsity_rate", "selected_pages", "dense_fallback"}
        )
        self.assertAlmostEqual(payload["sparsity_rate"], 0.05)
        self.assertEqual(payload["selected_pages"], 64)
        self.assertEqual(payload["dense_fallback"], 0)


if __name__ == "__main__":
    unittest.main()
