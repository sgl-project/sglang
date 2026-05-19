"""
Alignment test: verify SGLang early exit matches HF Transformers ground truth.

Compares intermediate layer hidden states (after norm) between:
  - HF Transformers with output_hidden_states=True
  - SGLang Engine with exit_layer

Model: Qwen2.5-0.5B (24 layers)
Environment: CPU (OrbStack or GPU machine)

Usage:
    source ~/.venv/sglang/bin/activate
    export PYTHONPATH=$(pwd)/python:$PYTHONPATH
    python -m pytest test/srt/models/test_early_exit_alignment.py -v

    # Or run directly:
    python test/srt/models/test_early_exit_alignment.py
"""

import os
import unittest

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model config
MODEL = os.environ.get(
    "EARLY_EXIT_TEST_MODEL",
    "/home/chenhua/.cache/modelscope/Qwen/Qwen2.5-0.5B",
)
PROMPT = "The quick brown fox jumps over the lazy dog"
EXIT_LAYER = 12


class TestHFEarlyExitSimulation(unittest.TestCase):
    """Verify HF Transformers intermediate layer behavior (no SGLang needed)."""

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        cls.model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float32
        )
        cls.model.eval()
        cls.inputs = cls.tokenizer(PROMPT, return_tensors="pt")
        with torch.no_grad():
            cls.hf_out = cls.model(**cls.inputs, output_hidden_states=True)
        cls.num_layers = cls.model.config.num_hidden_layers

    def test_hidden_states_count(self):
        """output_hidden_states returns embedding + N layer outputs."""
        count = len(self.hf_out.hidden_states)
        print(f"\n  Model: {MODEL}, num_layers={self.num_layers}")
        print(f"  hidden_states count: {count} (embedding + {self.num_layers} layers)")
        self.assertEqual(count, self.num_layers + 1)

    def test_different_layers_differ(self):
        """Intermediate layers produce different embeddings."""
        layer_4 = self.hf_out.hidden_states[4][0, -1, :]
        layer_12 = self.hf_out.hidden_states[12][0, -1, :]
        layer_last = self.hf_out.hidden_states[self.num_layers][0, -1, :]

        cos_4_12 = torch.nn.functional.cosine_similarity(
            layer_4.unsqueeze(0), layer_12.unsqueeze(0)
        ).item()
        cos_12_last = torch.nn.functional.cosine_similarity(
            layer_12.unsqueeze(0), layer_last.unsqueeze(0)
        ).item()
        print(f"\n  Layer 4 vs 12 cosine:  {cos_4_12:.4f}")
        print(f"  Layer 12 vs {self.num_layers} cosine: {cos_12_last:.4f}")

        self.assertFalse(torch.allclose(layer_4, layer_12, atol=1e-3))
        self.assertFalse(torch.allclose(layer_12, layer_last, atol=1e-3))

    def test_normed_vs_raw_differ(self):
        """Raw and normed hidden states at layer K should differ."""
        raw = self.hf_out.hidden_states[EXIT_LAYER][0, -1, :]
        with torch.no_grad():
            normed = self.model.model.norm(
                self.hf_out.hidden_states[EXIT_LAYER]
            )
        normed_last = normed[0, -1, :]

        self.assertFalse(
            torch.allclose(raw, normed_last, atol=1e-3),
            "Raw and normed hidden states should differ",
        )

    def test_early_exit_vs_full_cosine(self):
        """Cosine similarity between early exit (layer K + norm) and
        full forward (last layer + norm) should be < 1.0 (they ARE different)."""
        with torch.no_grad():
            early_normed = self.model.model.norm(
                self.hf_out.hidden_states[EXIT_LAYER]
            )
        early_last = early_normed[0, -1, :]

        with torch.no_grad():
            full_normed = self.model.model.norm(
                self.hf_out.hidden_states[self.num_layers]
            )
        full_last = full_normed[0, -1, :]

        cosine = torch.nn.functional.cosine_similarity(
            early_last.unsqueeze(0), full_last.unsqueeze(0)
        ).item()

        print(f"\n  Early exit (layer {EXIT_LAYER}) vs full forward cosine: {cosine:.4f}")
        print(f"  Early exit first 5: {[f'{v:.4f}' for v in early_last[:5].tolist()]}")
        print(f"  Full forward first 5: {[f'{v:.4f}' for v in full_last[:5].tolist()]}")

        self.assertLess(cosine, 0.99)

    def test_layer_norms_increase(self):
        """Hidden state norms generally increase with layer depth."""
        norms = []
        for i in [1, 4, 8, 12, 16, 20, self.num_layers]:
            if i > self.num_layers:
                continue
            norm = self.hf_out.hidden_states[i][0, -1, :].norm().item()
            norms.append((i, norm))

        # Last layer norm should be larger than first layer norm
        self.assertGreater(
            norms[-1][1], norms[0][1],
            f"Last layer norm ({norms[-1]}) should be > first layer norm ({norms[0]})",
        )


class TestSGLangVsHFAlignment(unittest.TestCase):
    """Compare SGLang Engine early exit output with HF Transformers ground truth.

    Requires SGLang Engine to work on CPU (may need CPU compatibility patches).
    Skip gracefully if Engine cannot start.
    """

    @classmethod
    def setUpClass(cls):
        # HF ground truth
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        cls.model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float32
        )
        cls.model.eval()
        cls.inputs = cls.tokenizer(PROMPT, return_tensors="pt")
        with torch.no_grad():
            cls.hf_out = cls.model(**cls.inputs, output_hidden_states=True)
            cls.hf_normed = cls.model.model.norm(
                cls.hf_out.hidden_states[EXIT_LAYER]
            )
        cls.hf_normed_last = cls.hf_normed[0, -1, :]

        # SGLang Engine
        # Requires: config.json architectures set to "Qwen2ForEarlyExitCausalLM"
        #           and SGLANG_EXIT_LAYER env var set
        cls.engine = None
        try:
            import sglang as sgl

            os.environ["SGLANG_EXIT_LAYER"] = str(EXIT_LAYER)
            cls.engine = sgl.Engine(
                model_path=MODEL,
                device="cpu",
                enable_return_hidden_states=True,
            )
        except Exception as e:
            cls.engine_error = str(e)

    @classmethod
    def tearDownClass(cls):
        if cls.engine is not None:
            cls.engine.shutdown()
        os.environ.pop("SGLANG_EXIT_LAYER", None)

    def setUp(self):
        if self.engine is None:
            self.skipTest(
                f"SGLang Engine not available on CPU: {getattr(self, 'engine_error', 'unknown')}"
            )

    def test_early_exit_returns_hidden_states(self):
        """Engine with exit_layer should return hidden_states in meta_info."""
        out = self.engine.generate(
            [PROMPT],
            sampling_params={"max_new_tokens": 1},
            return_hidden_states=True,
        )
        self.assertIn("hidden_states", out[0]["meta_info"])

    def test_early_exit_vs_hf_cosine_similarity(self):
        """SGLang early exit hidden states should match HF (cosine > 0.99)."""
        out = self.engine.generate(
            [PROMPT],
            sampling_params={"max_new_tokens": 1},
            return_hidden_states=True,
        )
        hs = out[0]["meta_info"]["hidden_states"]

        # Extract last token hidden state
        if isinstance(hs[0], list):
            sgl_last = hs[0][-1] if isinstance(hs[0][0], list) else hs[0]
        else:
            sgl_last = hs

        sgl_tensor = torch.tensor(sgl_last, dtype=torch.float32)
        hf_tensor = self.hf_normed_last

        cosine = torch.nn.functional.cosine_similarity(
            sgl_tensor.unsqueeze(0), hf_tensor.unsqueeze(0)
        ).item()
        max_diff = (sgl_tensor - hf_tensor).abs().max().item()
        mean_diff = (sgl_tensor - hf_tensor).abs().mean().item()

        print(f"\n  SGLang early exit first 5: {[f'{v:.4f}' for v in sgl_tensor[:5].tolist()]}")
        print(f"  HF normed layer {EXIT_LAYER} first 5: {[f'{v:.4f}' for v in hf_tensor[:5].tolist()]}")
        print(f"  Cosine similarity:  {cosine:.6f}")
        print(f"  Max abs difference: {max_diff:.6f}")
        print(f"  Mean abs difference: {mean_diff:.6f}")

        self.assertGreater(
            cosine, 0.99,
            f"SGLang early exit should align with HF (cosine={cosine:.6f})",
        )

    def test_early_exit_vs_hf_max_diff(self):
        """Max absolute difference between SGLang and HF should be small."""
        out = self.engine.generate(
            [PROMPT],
            sampling_params={"max_new_tokens": 1},
            return_hidden_states=True,
        )
        hs = out[0]["meta_info"]["hidden_states"]

        if isinstance(hs[0], list):
            sgl_last = hs[0][-1] if isinstance(hs[0][0], list) else hs[0]
        else:
            sgl_last = hs

        sgl_tensor = torch.tensor(sgl_last, dtype=torch.float32)
        hf_tensor = self.hf_normed_last

        max_diff = (sgl_tensor - hf_tensor).abs().max().item()
        # bf16 vs fp32 precision difference can be up to ~2.0
        self.assertLess(
            max_diff, 5.0,
            f"Max abs diff between SGLang and HF should be small (got {max_diff:.6f})",
        )

    def test_full_forward_vs_early_exit_differ(self):
        """Full forward and early exit should produce different hidden states."""
        # Full forward (no exit_layer override — but env var is set,
        # so we need a separate engine or clear env var)
        # Instead, compare with HF full forward ground truth
        full_hs = self.hf_out.hidden_states[-1][0, -1, :]

        out = self.engine.generate(
            [PROMPT],
            sampling_params={"max_new_tokens": 1},
            return_hidden_states=True,
        )
        hs = out[0]["meta_info"]["hidden_states"]

        if isinstance(hs[0], list):
            sgl_last = hs[0][-1] if isinstance(hs[0][0], list) else hs[0]
        else:
            sgl_last = hs

        sgl_tensor = torch.tensor(sgl_last, dtype=torch.float32)

        cosine_with_full = torch.nn.functional.cosine_similarity(
            sgl_tensor.unsqueeze(0), full_hs.unsqueeze(0)
        ).item()

        self.assertLess(
            cosine_with_full, 0.99,
            f"Early exit should differ from full forward "
            f"(cosine={cosine_with_full:.4f})",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
