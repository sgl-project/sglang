import json
import os
import shutil
import tempfile
import unittest

import requests
import torch
from parameterized import parameterized_class
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sglang.srt.environ import envs
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

MODEL = "deepseek-ai/DeepSeek-V3"
MODEL_OVERRIDE_ARGS = {
    "num_hidden_layers": 2,
    "first_k_dense_replace": 0,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_group": 1,
    "topk_group": 1,
}

# Default values for all config dimensions
DEFAULTS = {
    "tp_size": 8,
    "ep_size": 1,
    "precision": "fp8",
    "attention_backend": None,
    "chunked_prefill": False,
    "cuda_graph": False,
}

# Configs extended on top of defaults
CONFIGS = [
    # --- Initial baseline ---
    {},
    # --- Future configs ---
    # {"tp_size": 2},
    # {"tp_size": 2, "ep_size": 2},
    # {"precision": "fp8"},
    # {"tp_size": 2, "precision": "fp8"},
    # {"attention_backend": "triton"},
    # {"attention_backend": "fa3"},
    # {"cuda_graph": True},
    # {"chunked_prefill": False},
]


def _merge_config(overrides):
    """Merge overrides with DEFAULTS."""
    return {**DEFAULTS, **overrides}


ALL_CONFIGS = [_merge_config(c) for c in CONFIGS]


def _config_suffix(cfg):
    """Generate a readable suffix from a config dict, showing only non-default values."""
    parts = []
    for key, val in cfg.items():
        if val != DEFAULTS.get(key):
            parts.append(f"{key}={val}")
    return "_".join(parts) if parts else "default"


def _class_name(cls, _num, params):
    return f"{cls.__name__}_{_config_suffix(params)}"


def build_server_args(cfg, dummy_weights=True):
    """Build server launch args from a config dict."""
    args = [
        "--json-model-override-args",
        json.dumps(MODEL_OVERRIDE_ARGS),
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.5",
    ]

    if dummy_weights:
        args.extend(["--load-format", "dummy"])

    # Precision / quantization
    precision = cfg["precision"]
    if precision == "bf16":
        args.extend(["--dtype", "bfloat16"])
    elif precision == "fp8":
        args.extend(["--quantization", "fp8"])
    elif precision == "fp4":
        args.extend(["--quantization", "modelopt_fp4"])
    else:
        args.extend(["--dtype", precision])

    # Parallelism
    if cfg["tp_size"] > 1:
        args.extend(["--tp-size", str(cfg["tp_size"])])
    if cfg["ep_size"] > 1:
        args.extend(["--ep-size", str(cfg["ep_size"])])

    # Attention backend
    if cfg["attention_backend"] is not None:
        args.extend(["--attention-backend", cfg["attention_backend"]])

    # Chunked prefill
    if not cfg["chunked_prefill"]:
        args.extend(["--chunked-prefill-size", "-1"])

    # CUDA graph
    if not cfg["cuda_graph"]:
        args.append("--disable-cuda-graph")

    return args


def _create_tiny_hf_model(output_dir):
    """Create a tiny DeepSeek V3 model with random weights and save to output_dir.

    Also copies custom modeling code and tokenizer files from the HF cache into
    the output dir, since save_pretrained only saves weights/config.
    """
    from huggingface_hub import snapshot_download

    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    for key, val in MODEL_OVERRIDE_ARGS.items():
        setattr(config, key, val)
    # Strip quantization config so HF loads plain bf16 weights
    if hasattr(config, "quantization_config"):
        del config.quantization_config

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.save_pretrained(output_dir)
    del model
    torch.cuda.empty_cache()

    # Copy custom code and tokenizer files from the HF cache.
    # save_pretrained only saves weights + config, but we also need modeling
    # code (*.py) for trust_remote_code and tokenizer files.
    cached_dir = snapshot_download(
        MODEL,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf"],
    )
    for fname in os.listdir(cached_dir):
        src = os.path.join(cached_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# test_sanity: Verify forward pass completes without errors (dummy weights)
# ---------------------------------------------------------------------------


@parameterized_class(
    ALL_CONFIGS,
    class_name_func=_class_name,
)
class TestDeepSeekV3Sanity(DefaultServerBase):
    """Modeling-level sanity tests for DeepSeek V3.

    parameterized_class creates a separate test class per config, so
    DefaultServerBase's setUpClass/tearDownClass handle server lifecycle
    naturally for each configuration.
    """

    model = MODEL

    def __repr__(self):
        cfg = {k: getattr(self, k) for k in DEFAULTS}
        return _config_suffix(cfg)

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        cfg = {k: getattr(cls, k) for k in DEFAULTS}
        cls.other_args = build_server_args(cfg, dummy_weights=True)
        super().setUpClass()

    def test_sanity(self):
        """Verify model forward pass completes and returns valid responses."""
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Hello world",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("text", result)


# ---------------------------------------------------------------------------
# test_allclose_to_hf: Numerical equivalence against HuggingFace reference
# ---------------------------------------------------------------------------

PREFILL_TOLERANCE = 5e-2
DECODE_TOLERANCE = 5e-2
TEST_INPUT = "The capital of France is"


@parameterized_class(
    ALL_CONFIGS,
    class_name_func=_class_name,
)
class TestDeepSeekV3AllcloseToHF(DefaultServerBase):
    """Compare SGLang outputs against HuggingFace Transformers reference.

    For each config:
    1. Create a tiny HF model (reduced layers/experts) with random weights
    2. Save to a temp directory
    3. Launch SGLang server loading from that checkpoint
    4. Run HF forward pass directly, query SGLang server
    5. Compare logprobs
    """

    model = None

    def __repr__(self):
        cfg = {k: getattr(self, k) for k in DEFAULTS}
        return _config_suffix(cfg)

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)

        # Create tiny model with random weights and save to temp dir
        cls._temp_dir = tempfile.mkdtemp(prefix="sglang_test_dsv3_")
        _create_tiny_hf_model(cls._temp_dir)

        # Launch SGLang server loading from the temp dir (shared weights)
        cls.model = cls._temp_dir
        cfg = {k: getattr(cls, k) for k in DEFAULTS}
        cls.other_args = build_server_args(cfg, dummy_weights=False)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if hasattr(cls, "_temp_dir") and os.path.exists(cls._temp_dir):
            shutil.rmtree(cls._temp_dir, ignore_errors=True)

    def test_allclose_to_hf(self):
        """Verify SGLang produces numerically equivalent logprobs to HF."""
        # --- HF reference forward pass ---
        tokenizer = AutoTokenizer.from_pretrained(
            self._temp_dir, trust_remote_code=True
        )
        hf_model = AutoModelForCausalLM.from_pretrained(
            self._temp_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda()

        input_ids = tokenizer(TEST_INPUT, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            hf_logits = hf_model(input_ids).logits[0]  # (seq_len, vocab)

        del hf_model
        torch.cuda.empty_cache()

        # --- SGLang forward pass via server ---
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": TEST_INPUT,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 1,
                },
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 5,
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()

        # Compare prefill logprobs: for each token position, the top logprob
        # from SGLang should be close to the corresponding HF logprob.
        sg_input_logprobs = result["meta_info"]["input_token_logprobs"]
        # First token has no logprob, skip it
        for i, sg_logprob in enumerate(sg_input_logprobs[1:], start=1):
            # HF logprob at position i-1 predicts token at position i
            hf_token_logprob = torch.log_softmax(hf_logits[i - 1], dim=-1)[
                input_ids[0, i]
            ].item()
            diff = abs(sg_logprob - hf_token_logprob)
            self.assertLess(
                diff,
                PREFILL_TOLERANCE,
                f"Prefill logprob mismatch at position {i}: "
                f"SGLang={sg_logprob:.6f}, HF={hf_token_logprob:.6f}, diff={diff:.6f}",
            )


if __name__ == "__main__":
    unittest.main()
