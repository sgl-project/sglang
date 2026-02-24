import json
import unittest

import requests
from parameterized import parameterized_class

from sglang.srt.environ import envs
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

MODEL = "deepseek-ai/DeepSeek-V3"
MODEL_OVERRIDE_ARGS = json.dumps(
    {
        "num_hidden_layers": 2,
        "first_k_dense_replace": 0,
        "n_routed_experts": 8,
    }
)

# Default values for all config dimensions
DEFAULTS = {
    "tp_size": 1,
    "ep_size": 1,
    "precision": "bf16",
    "attention_backend": None,
    "chunked_prefill": True,
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


def build_server_args(cfg):
    """Build server launch args from a config dict."""
    args = [
        "--load-format",
        "dummy",
        "--json-model-override-args",
        MODEL_OVERRIDE_ARGS,
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.5",
    ]

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
        cls.other_args = build_server_args(cfg)
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


if __name__ == "__main__":
    unittest.main()
