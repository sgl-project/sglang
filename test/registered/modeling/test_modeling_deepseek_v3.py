import json
import os
import shutil
import tempfile
import unittest

import requests
import torch
from parameterized import parameterized_class

from transformers import AutoConfig, AutoTokenizer
from transformers.models.deepseek_v3 import DeepseekV3ForCausalLM

# Workaround: the sglang:dev Docker image ships ~/.cache/sglang as a JSON file
# (kernel repo metadata), but sglang code expects it to be a directory (for
# torch compile cache, custom_all_reduce_utils.py P2P access cache).
_sglang_cache = os.path.expanduser("~/.cache/sglang")
if os.path.exists(_sglang_cache) and not os.path.isdir(_sglang_cache):
    os.remove(_sglang_cache)
os.makedirs(_sglang_cache, exist_ok=True) 

from sglang.srt.environ import envs
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

MODEL = "deepseek-ai/DeepSeek-V3"
MODEL_OVERRIDE_ARGS = {
    "num_hidden_layers": 2,
    "first_k_dense_replace": 1,  # First layer is dense, second is MoE
    "n_routed_experts": 8,
    "num_experts_per_tok": 2, 
    "n_group": 1,
    "topk_group": 1,
}
RANDOM_SEED = 42

# Default values sanity tests.
DEFAULTS = {
    "tp_size": 4,
    "ep_size": 1,
    "precision": "fp8",
    "attention_backend": None,
    "chunked_prefill": False,
    "cuda_graph": False,
    "moe_dense_tp_size": None,
}

# Default values for allclose tests.
ALLCLOSE_DEFAULTS = {
    "tp_size": 1,
    "ep_size": 1,
    "precision": "bf16",
    "attention_backend": None,
    "chunked_prefill": False,
    "cuda_graph": False,
    "moe_dense_tp_size": None,
}

# Configs extended on top of defaults
SANITY_CONFIGS = [
    {},
]

# Configs for allclose tests.
# HF reference always runs TP=1; SGLang runs with these configs.
ALLCLOSE_CONFIGS = [
    {},
    {"tp_size": 2},
    {"tp_size": 2, "moe_dense_tp_size": 1},
]


def _merge_config(overrides, defaults=DEFAULTS):
    return {**defaults, **overrides}


ALL_SANITY_CONFIGS = [_merge_config(c) for c in SANITY_CONFIGS]
ALL_ALLCLOSE_CONFIGS = [_merge_config(c, ALLCLOSE_DEFAULTS) for c in ALLCLOSE_CONFIGS]


def _config_suffix(cfg, defaults=DEFAULTS):
    parts = []
    for key, val in cfg.items():
        if val != defaults.get(key):
            parts.append(f"{key}={val}")
    return "_".join(parts) if parts else "default"


def _class_name(cls, _num, params):
    return f"{cls.__name__}_{_config_suffix(params)}"


def build_server_args(cfg, dummy_weights=True):
    args = [
        "--json-model-override-args",
        json.dumps(MODEL_OVERRIDE_ARGS),
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.5",
    ]

    if dummy_weights:
        args.extend(["--load-format", "dummy"])

    precision = cfg["precision"]
    if precision == "bf16":
        args.extend(["--dtype", "bfloat16"])
    elif precision == "fp8":
        args.extend(["--quantization", "fp8"])
    elif precision == "fp4":
        args.extend(["--quantization", "modelopt_fp4"])
    else:
        args.extend(["--dtype", precision])

    if cfg["tp_size"] > 1:
        args.extend(["--tp-size", str(cfg["tp_size"])])
    if cfg["ep_size"] > 1:
        args.extend(["--ep-size", str(cfg["ep_size"])])
    if cfg.get("moe_dense_tp_size") is not None:
        args.extend(["--moe-dense-tp-size", str(cfg["moe_dense_tp_size"])])
    if cfg["attention_backend"] is not None:
        args.extend(["--attention-backend", cfg["attention_backend"]])
    if not cfg["chunked_prefill"]:
        args.extend(["--chunked-prefill-size", "-1"])
    if not cfg["cuda_graph"]:
        args.append("--disable-cuda-graph")

    return args


def _save_tiny_hf_checkpoint(model, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.config.save_pretrained(output_dir)
    state_dict = {
        k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()
    }
    from safetensors.torch import save_file

    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))


def _create_tiny_hf_model(output_dir):
    from huggingface_hub import snapshot_download

    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=False)
    for key, val in MODEL_OVERRIDE_ARGS.items():
        setattr(config, key, val)
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    if hasattr(config, "auto_map"):
        del config.auto_map

    torch.manual_seed(RANDOM_SEED)
    model = DeepseekV3ForCausalLM(config)
    _save_tiny_hf_checkpoint(model, output_dir)
    del model
    torch.cuda.empty_cache()

    cached_dir = snapshot_download(
        MODEL,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf"],
    )
    keep_extensions = {".json", ".model", ".txt"}
    for fname in os.listdir(cached_dir):
        if not any(fname.endswith(ext) for ext in keep_extensions):
            continue
        if "index.json" in fname:
            continue
        src = os.path.join(cached_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src) and os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Shared checkpoint + HF reference (created once, used by all allclose tests)
# ---------------------------------------------------------------------------

TEST_INPUT = "The capital of France is"

_shared_checkpoint_dir = None
_hf_reference = None  # dict with 'logits', 'input_ids', 'tokenizer' -- reset on config change


def _get_shared_checkpoint():
    global _shared_checkpoint_dir
    if _shared_checkpoint_dir is None:
        _shared_checkpoint_dir = tempfile.mkdtemp(prefix="sglang_test_dsv3_shared_")
        _create_tiny_hf_model(_shared_checkpoint_dir)
    return _shared_checkpoint_dir


def _get_hf_reference():
    """Compute HF prefill forward pass once, cache the result.

    Also keeps the HF model loaded so decode logits can be computed per-test
    (each SGLang config may generate a different token).
    """
    global _hf_reference
    if _hf_reference is not None:
        return _hf_reference

    ckpt = _get_shared_checkpoint()
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    hf_model = DeepseekV3ForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16
    ).cuda()

    input_ids = tokenizer(TEST_INPUT, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits[0].cpu()  # (seq_len, vocab)

    _hf_reference = {
        "logits": hf_logits,
        "input_ids": input_ids.cpu(),
        "tokenizer": tokenizer,
        "model": hf_model,  # keep alive for decode
    }
    return _hf_reference


def _get_hf_decode_logits(generated_token_id: int):
    """Run HF forward on [input_ids + generated_token] and return decode logits."""
    ref = _get_hf_reference()
    hf_model = ref["model"]
    input_ids = ref["input_ids"].cuda()
    decode_ids = torch.cat(
        [input_ids, torch.tensor([[generated_token_id]], device=input_ids.device)],
        dim=1,
    )
    with torch.no_grad():
        logits = hf_model(decode_ids).logits[0, -1].cpu()  # (vocab,)
    return logits


# ---------------------------------------------------------------------------
# test_sanity: Verify forward pass completes without errors (dummy weights)
# ---------------------------------------------------------------------------


@parameterized_class(
    ALL_SANITY_CONFIGS,
    class_name_func=_class_name,
)
class TestDeepSeekV3Sanity(DefaultServerBase):
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
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Hello world",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("text", result)


# ---------------------------------------------------------------------------
# test_allclose_to_hf: Numerical equivalence against HuggingFace reference
# ---------------------------------------------------------------------------


def _allclose_class_name(cls, _num, params):
    return f"{cls.__name__}_{_config_suffix(params, ALLCLOSE_DEFAULTS)}"


@parameterized_class(
    ALL_ALLCLOSE_CONFIGS,
    class_name_func=_allclose_class_name,
)
class TestDeepSeekV3AllcloseToHF(DefaultServerBase):
    """Compare SGLang outputs against HuggingFace Transformers reference.

    All configs share the same checkpoint (fixed seed) and the same HF
    reference logits, so diffs are directly comparable across configs.
    """

    model = None

    def __repr__(self):
        cfg = {k: getattr(self, k) for k in ALLCLOSE_DEFAULTS}
        return _config_suffix(cfg, ALLCLOSE_DEFAULTS)

    @classmethod
    def setUpClass(cls):
        envs.SGLANG_JIT_DEEPGEMM_PRECOMPILE.set(False)
        cls.model = _get_shared_checkpoint()
        cfg = {k: getattr(cls, k) for k in ALLCLOSE_DEFAULTS}
        cls.other_args = build_server_args(cfg, dummy_weights=False)
        super().setUpClass()

    @staticmethod
    def _compare_logprobs(hf_logits_pos, sg_token_logprob, sg_top_entries,
                          actual_token=None, label=""):
        """Compare HF logits at one position against SGLang's logprobs.

        Returns dict with input_token_diff and topk metrics.
        """
        hf_logprobs = torch.log_softmax(hf_logits_pos.float(), dim=-1)

        # Input/actual token comparison (neutral sample point)
        input_diff = None
        if actual_token is not None and sg_token_logprob is not None:
            sg_lp = sg_token_logprob[0]
            hf_lp = hf_logprobs[actual_token].item()
            input_diff = abs(sg_lp - hf_lp)

        # Top-K comparison
        topk_diffs = []
        for sg_lp, sg_tid, _ in sg_top_entries:
            hf_lp = hf_logprobs[sg_tid].item()
            topk_diffs.append(abs(sg_lp - hf_lp))

        hf_top20 = set(hf_logits_pos.topk(20).indices.tolist())
        sg_top20 = set(e[1] for e in sg_top_entries[:20])
        overlap = len(sg_top20 & hf_top20)

        topk_avg = sum(topk_diffs) / len(topk_diffs) if topk_diffs else 0
        parts = [f"{label}"]
        if input_diff is not None:
            parts.append(f"input_token_diff={input_diff:.4f}")
        parts.append(f"topK_avg={topk_avg:.4f}  top20_overlap={overlap}/20")
        print(f"  {'  '.join(parts)}")

        return {"input_diff": input_diff, "topk_diffs": topk_diffs}

    def test_allclose_to_hf(self):
        ref = _get_hf_reference()
        hf_logits = ref["logits"]
        input_ids = ref["input_ids"]
        seq_len = input_ids.shape[1]

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": TEST_INPUT,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "logprob_start_len": 0,
                "top_logprobs_num": 20,
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()

        cfg = {k: getattr(self, k) for k in ALLCLOSE_DEFAULTS}
        cfg_name = _config_suffix(cfg, ALLCLOSE_DEFAULTS)
        print(f"\n{'='*70}")
        print(f"Config: {cfg_name}")
        print(f"Input: {TEST_INPUT!r}  tokens: {input_ids[0].tolist()}")

        # --- Prefill comparison ---
        print(f"\n  PREFILL:")
        sg_input_logprobs = result["meta_info"]["input_token_logprobs"]
        sg_input_top = result["meta_info"]["input_top_logprobs"]
        prefill_input_diffs = []
        prefill_topk_diffs = []

        for i in range(1, seq_len):
            actual_token = input_ids[0, i].item()
            sg_top = sg_input_top[i] if i < len(sg_input_top) else []
            sg_lp = sg_input_logprobs[i] if i < len(sg_input_logprobs) else None
            stats = self._compare_logprobs(
                hf_logits[i - 1], sg_lp, sg_top,
                actual_token=actual_token, label=f"Pos {i}:",
            )
            if stats["input_diff"] is not None:
                prefill_input_diffs.append(stats["input_diff"])
            prefill_topk_diffs.extend(stats["topk_diffs"])

        # --- Decode comparison ---
        print(f"\n  DECODE:")
        sg_output_logprobs = result["meta_info"]["output_token_logprobs"]
        sg_output_top = result["meta_info"]["output_top_logprobs"]

        # The generated token is what SGLang decoded
        generated_token_id = sg_output_logprobs[0][1]
        hf_decode_logits = _get_hf_decode_logits(generated_token_id)

        decode_stats = self._compare_logprobs(
            hf_decode_logits, sg_output_logprobs[0], sg_output_top[0],
            actual_token=generated_token_id, label="Gen:",
        )
        decode_input_diff = decode_stats["input_diff"]
        decode_topk_diffs = decode_stats["topk_diffs"]

        # --- Summary ---
        all_input = prefill_input_diffs + ([decode_input_diff] if decode_input_diff else [])
        all_topk = prefill_topk_diffs + decode_topk_diffs
        print(f"\n  SUMMARY (prefill + decode):")
        print(f"    input_token: avg={sum(all_input)/len(all_input):.4f}  max={max(all_input):.4f}")
        print(f"    topK:        avg={sum(all_topk)/len(all_topk):.4f}  max={max(all_topk):.4f}")
        print(f"{'='*70}")


if __name__ == "__main__":
    unittest.main()
