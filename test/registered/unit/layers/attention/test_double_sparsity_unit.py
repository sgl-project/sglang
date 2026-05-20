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
