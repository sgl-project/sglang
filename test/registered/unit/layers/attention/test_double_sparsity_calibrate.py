"""Unit tests for the Double Sparsity offline channel-mask calibrator
(``sglang.srt.layers.attention.double_sparsity.calibrate``).

Split out of ``test_double_sparsity_unit.py`` to keep each test file under the
2,000-line limit (developer_guide/contribution_guide.md).
"""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch  # noqa: F401  # used by class-level skip guards / local test imports

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


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


if __name__ == "__main__":
    unittest.main()
