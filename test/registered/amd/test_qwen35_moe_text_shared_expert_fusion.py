"""Qwen3.5-MoE text-only checkpoints keep their shared expert under ROCm fusion.

With SGLANG_USE_AITER=1 the Qwen3.5 MoE block serves the shared expert as one
more fused MoE slot (index ``num_experts``), so ``load_weights`` has to move the
checkpoint's ``mlp.shared_expert.*`` onto that slot. The text-only entry class
``Qwen3_5MoeForCausalLM`` -- the architecture Qwen3.8-2.4T-A95B reports -- once
skipped that remap (#40754): every shared-expert tensor was dropped with a
"not found in params_dict" warning, the slot was never loaded, and GSM8K
collapsed while the server still started and served.

Qwen3.8 is the only text-only checkpoint of this family and its FP8 weights do
not fit on one node, so this serves a four-layer random-weight checkpoint of the
same architecture on one GPU. The config keys are Qwen3.8's; the layer widths
are Qwen3.5-35B-A3B's, which the ROCm GDN, gated-attention and fused-MoE kernels
already serve.

The oracle is the same checkpoint served with ``--disable-shared-experts-fusion``,
which loads the shared expert into its own MLP: fusing it must not move the
prompt logprobs beyond kernel rounding. Both routed-expert layouts are served
because the loader maps the shared expert on a separate branch for each: stacked
3-D tensors, as the BF16 Qwen3.8 checkpoint stores them, and one tensor per
expert, as its FP8 checkpoint does. The shared expert is scaled to dominate
every MoE block, and a copy with that expert zeroed -- the model the bug ended
up serving -- checks that the comparison notices it missing. Each launch also
confirms it took the path it asked for, so neither side of the comparison can
quietly fall back to the other.

Registry: nightly-amd-1-gpu suite
"""

import io
import json
import os
import shutil
import tempfile
import unittest

import requests
import torch
from safetensors.torch import save_file

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_amd_ci(est_time=600, suite="nightly-amd-1-gpu", nightly=True)

HIDDEN_SIZE = 2048
NUM_LAYERS = 4
FULL_ATTENTION_INTERVAL = 4
NUM_ATTENTION_HEADS = 16
NUM_KEY_VALUE_HEADS = 2
HEAD_DIM = 256
LINEAR_NUM_KEY_HEADS = 16
LINEAR_NUM_VALUE_HEADS = 32
LINEAR_HEAD_DIM = 128
LINEAR_CONV_KERNEL_DIM = 4
# Fusion needs the shared expert exactly as wide as a routed one.
MOE_INTERMEDIATE_SIZE = 512
NUM_EXPERTS = 32
NUM_EXPERTS_PER_TOK = 8
VOCAB_SIZE = 16384

INIT_STD = 0.02
# Makes the shared expert the largest term of every MoE block's output.
SHARED_EXPERT_DOWN_STD = 0.1

NUM_PROMPTS = 8
PROMPT_LEN = 128
# Mean |logprob difference| per prompt token against the unfused server. On
# these weights BF16 rounding alone accounts for ~0.02 of it; zeroing the shared
# expert moves it by ~1.
MAX_FUSION_DRIFT = 0.1
MIN_MISSING_SHARED_EXPERT_DRIFT = 0.5

SERVER_ARGS = [
    "--attention-backend",
    "aiter",
    "--skip-tokenizer-init",
    "--disable-radix-cache",
    "--disable-cuda-graph",
]
SERVER_ENV = {"SGLANG_USE_AITER": "1"}
FUSION_AUTO_DISABLED_LOG = "Shared experts fusion optimization is disabled"


def _config() -> dict:
    return {
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe_text",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "dtype": "bfloat16",
        "full_attention_interval": FULL_ATTENTION_INTERVAL,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": HIDDEN_SIZE,
        "initializer_range": INIT_STD,
        "layer_types": [
            (
                "full_attention"
                if (layer + 1) % FULL_ATTENTION_INTERVAL == 0
                else "linear_attention"
            )
            for layer in range(NUM_LAYERS)
        ],
        "linear_conv_kernel_dim": LINEAR_CONV_KERNEL_DIM,
        "linear_key_head_dim": LINEAR_HEAD_DIM,
        "linear_num_key_heads": LINEAR_NUM_KEY_HEADS,
        "linear_num_value_heads": LINEAR_NUM_VALUE_HEADS,
        "linear_value_head_dim": LINEAR_HEAD_DIM,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 4096,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "num_attention_heads": NUM_ATTENTION_HEADS,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
        "num_hidden_layers": NUM_LAYERS,
        "num_key_value_heads": NUM_KEY_VALUE_HEADS,
        "output_gate_type": "swish",
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "shared_expert_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "tie_word_embeddings": False,
        "vocab_size": VOCAB_SIZE,
    }


def _random_weights() -> dict[str, torch.Tensor]:
    """Seeded weights named as the BF16 Qwen3.8 checkpoint names them."""
    gen = torch.Generator().manual_seed(0)

    def normal(*shape, std=INIT_STD):
        return (torch.randn(shape, generator=gen) * std).to(torch.bfloat16)

    def zeros(size):
        # Qwen3.5 RMSNorms scale by (1 + weight); only the gated GDN norm
        # scales by the weight itself.
        return torch.zeros(size, dtype=torch.bfloat16)

    key_dim = LINEAR_NUM_KEY_HEADS * LINEAR_HEAD_DIM
    value_dim = LINEAR_NUM_VALUE_HEADS * LINEAR_HEAD_DIM
    weights = {
        "model.embed_tokens.weight": normal(VOCAB_SIZE, HIDDEN_SIZE),
        "model.norm.weight": zeros(HIDDEN_SIZE),
        "lm_head.weight": normal(VOCAB_SIZE, HIDDEN_SIZE),
    }
    for layer in range(NUM_LAYERS):
        prefix = f"model.layers.{layer}."
        weights[prefix + "input_layernorm.weight"] = zeros(HIDDEN_SIZE)
        weights[prefix + "post_attention_layernorm.weight"] = zeros(HIDDEN_SIZE)
        if (layer + 1) % FULL_ATTENTION_INTERVAL == 0:
            attn = prefix + "self_attn."
            q_dim = NUM_ATTENTION_HEADS * HEAD_DIM
            kv_dim = NUM_KEY_VALUE_HEADS * HEAD_DIM
            # attn_output_gate: q_proj also emits the output gate.
            weights[attn + "q_proj.weight"] = normal(2 * q_dim, HIDDEN_SIZE)
            weights[attn + "k_proj.weight"] = normal(kv_dim, HIDDEN_SIZE)
            weights[attn + "v_proj.weight"] = normal(kv_dim, HIDDEN_SIZE)
            weights[attn + "o_proj.weight"] = normal(HIDDEN_SIZE, q_dim)
            weights[attn + "q_norm.weight"] = zeros(HEAD_DIM)
            weights[attn + "k_norm.weight"] = zeros(HEAD_DIM)
        else:
            gdn = prefix + "linear_attn."
            conv_dim = 2 * key_dim + value_dim
            weights[gdn + "in_proj_qkv.weight"] = normal(conv_dim, HIDDEN_SIZE)
            weights[gdn + "in_proj_z.weight"] = normal(value_dim, HIDDEN_SIZE)
            weights[gdn + "in_proj_b.weight"] = normal(
                LINEAR_NUM_VALUE_HEADS, HIDDEN_SIZE
            )
            weights[gdn + "in_proj_a.weight"] = normal(
                LINEAR_NUM_VALUE_HEADS, HIDDEN_SIZE
            )
            weights[gdn + "conv1d.weight"] = normal(conv_dim, 1, LINEAR_CONV_KERNEL_DIM)
            weights[gdn + "A_log"] = (
                torch.empty(LINEAR_NUM_VALUE_HEADS).uniform_(0, 16, generator=gen).log()
            )
            weights[gdn + "dt_bias"] = torch.ones(
                LINEAR_NUM_VALUE_HEADS, dtype=torch.bfloat16
            )
            weights[gdn + "norm.weight"] = torch.ones(LINEAR_HEAD_DIM)
            weights[gdn + "out_proj.weight"] = normal(HIDDEN_SIZE, value_dim)
        mlp = prefix + "mlp."
        weights[mlp + "gate.weight"] = normal(NUM_EXPERTS, HIDDEN_SIZE)
        weights[mlp + "experts.gate_up_proj"] = normal(
            NUM_EXPERTS, 2 * MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE
        )
        weights[mlp + "experts.down_proj"] = normal(
            NUM_EXPERTS, HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE
        )
        weights[mlp + "shared_expert.gate_proj.weight"] = normal(
            MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE
        )
        weights[mlp + "shared_expert.up_proj.weight"] = normal(
            MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE
        )
        weights[mlp + "shared_expert.down_proj.weight"] = normal(
            HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, std=SHARED_EXPERT_DOWN_STD
        )
        weights[mlp + "shared_expert_gate.weight"] = normal(1, HIDDEN_SIZE)
    return weights


def _per_expert_layout(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """The same weights with one tensor per routed expert, as Qwen3.8-FP8 stores them."""
    out = {}
    for name, tensor in weights.items():
        if name.endswith(".experts.gate_up_proj"):
            experts = name.removesuffix("gate_up_proj")
            for expert_id, gate_up in enumerate(tensor):
                gate, up = gate_up.chunk(2)
                out[f"{experts}{expert_id}.gate_proj.weight"] = gate.clone()
                out[f"{experts}{expert_id}.up_proj.weight"] = up.clone()
        elif name.endswith(".experts.down_proj"):
            experts = name.removesuffix("down_proj")
            for expert_id, down in enumerate(tensor):
                out[f"{experts}{expert_id}.down_proj.weight"] = down.clone()
        else:
            out[name] = tensor
    return out


def _without_shared_expert(
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: (
            torch.zeros_like(tensor)
            if name.endswith(".shared_expert.down_proj.weight")
            else tensor
        )
        for name, tensor in weights.items()
    }


def _write_checkpoint(path: str, weights: dict[str, torch.Tensor]) -> str:
    os.makedirs(path)
    save_file(weights, os.path.join(path, "model.safetensors"), {"format": "pt"})
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(_config(), f, indent=2)
    return path


def _prompt_logprobs(input_ids: list[int]) -> list[float]:
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {"temperature": 0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": 0,
        },
        timeout=300,
    )
    response.raise_for_status()
    # The first prompt token has no logprob.
    return [
        logprob
        for logprob, _, _ in response.json()["meta_info"]["input_token_logprobs"][1:]
    ]


def _serve_and_score(
    model_path: str, prompts: list[list[int]], fuse_shared_expert: bool
) -> list[list[float]]:
    stdout, stderr = io.StringIO(), io.StringIO()
    process = popen_launch_server(
        model_path,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=SERVER_ARGS
        + ([] if fuse_shared_expert else ["--disable-shared-experts-fusion"]),
        env=SERVER_ENV,
        return_stdout_stderr=(stdout, stderr),
    )
    try:
        server_info = requests.get(
            DEFAULT_URL_FOR_TEST + "/server_info", timeout=60
        ).json()
        logprobs = [_prompt_logprobs(prompt) for prompt in prompts]
    finally:
        terminate_and_kill_process_tree(process)

    # The flag covers argument overrides; the log line covers the Qwen3.5 gate,
    # which disables fusion per checkpoint without touching the flag.
    fused = (
        not server_info["disable_shared_experts_fusion"]
        and FUSION_AUTO_DISABLED_LOG not in stdout.getvalue() + stderr.getvalue()
    )
    if fused != fuse_shared_expert:
        raise AssertionError(
            f"{os.path.basename(model_path)} was served with shared-expert fusion "
            f"{'on' if fused else 'off'}, so the fused/unfused comparison is void"
        )
    return logprobs


def _mean_abs_diff(a: list[list[float]], b: list[list[float]]) -> float:
    diffs = [
        abs(x - y)
        for row_a, row_b in zip(a, b, strict=True)
        for x, y in zip(row_a, row_b, strict=True)
    ]
    return sum(diffs) / len(diffs)


@unittest.skipUnless(torch.version.hip, "shared-expert fusion is the ROCm aiter path")
class TestQwen35MoeTextSharedExpertFusion(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.workdir = tempfile.mkdtemp(prefix="qwen35_moe_text_")
        weights = _random_weights()
        cls.stacked_path = _write_checkpoint(
            os.path.join(cls.workdir, "stacked_experts"), weights
        )
        cls.per_expert_path = _write_checkpoint(
            os.path.join(cls.workdir, "per_expert"), _per_expert_layout(weights)
        )
        cls.no_shared_expert_path = _write_checkpoint(
            os.path.join(cls.workdir, "no_shared_expert"),
            _without_shared_expert(weights),
        )
        del weights

        gen = torch.Generator().manual_seed(1)
        cls.prompts = [
            torch.randint(3, VOCAB_SIZE, (PROMPT_LEN,), generator=gen).tolist()
            for _ in range(NUM_PROMPTS)
        ]
        cls.reference = _serve_and_score(
            cls.stacked_path, cls.prompts, fuse_shared_expert=False
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "workdir"):
            shutil.rmtree(cls.workdir, ignore_errors=True)

    def _assert_fused_matches_unfused(self, model_path: str):
        logprobs = _serve_and_score(model_path, self.prompts, fuse_shared_expert=True)
        drift = _mean_abs_diff(logprobs, self.reference)
        print(f"{os.path.basename(model_path)}: fused vs unfused drift={drift:.4f}")
        self.assertLess(
            drift,
            MAX_FUSION_DRIFT,
            "fusing the shared expert moved the prompt logprobs, so the fused MoE "
            "slot does not compute the checkpoint's mlp.shared_expert.*",
        )

    def test_fused_per_expert_checkpoint_matches_unfused(self):
        self._assert_fused_matches_unfused(self.per_expert_path)

    def test_fused_stacked_expert_checkpoint_matches_unfused(self):
        self._assert_fused_matches_unfused(self.stacked_path)

    def test_missing_shared_expert_is_detected(self):
        logprobs = _serve_and_score(
            self.no_shared_expert_path, self.prompts, fuse_shared_expert=False
        )
        drift = _mean_abs_diff(logprobs, self.reference)
        print(f"zeroed shared expert vs unfused drift={drift:.4f}")
        self.assertGreater(
            drift,
            MIN_MISSING_SHARED_EXPERT_DRIFT,
            "zeroing the shared expert barely moves the logprobs, so a fused slot "
            "that failed to load would pass unnoticed",
        )


if __name__ == "__main__":
    unittest.main()
