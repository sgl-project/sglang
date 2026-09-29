"""A local, deterministic Beam checkpoint and its eager BF16 oracle for tests.

The oracle is limited to this fixture and the serving precision policy: BF16
gates, residuals and head projection, FP32 SSMax and output processing. It uses
split checkpoint tensors and ordinary PyTorch operations, without another model
implementation or a downloaded checkpoint. It bounds integration regressions on
the fixture; it is not a trained-model quality or bitwise-equivalence claim.
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.srt.configs.beam import BeamConfig

# Per-token KL bound for engine-vs-oracle and cached-vs-fresh comparisons on
# this fixture. Measured worst cases at the 6-token prompt: 1.02e-3 .. 1.17e-3
# on B300 (2026-09-14) and 1.68e-3 on an earlier B300 run; every longer prompt
# stays below 5e-4. A wrong sliding window or a dropped routing scale moves the
# fixture by well over 1e-2, so 5e-3 keeps the guard while absorbing kernel and
# batch-shape rounding differences that the architecture amplifies.
BEAM_FIXTURE_MAX_KL = 5e-3


def make_beam_fixture(path):
    config = BeamConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        # Match the release GQA ratio (48 query heads / 8 KV heads) at both TP sizes.
        num_attention_heads=12,
        num_key_value_heads=2,
        head_dim=64,
        # Room for the KL consistency helpers: LongBench prompts up to 4500
        # tokens plus two 512-token turns. Weights and oracle are unaffected.
        max_position_embeddings=8192,
        sliding_window=512,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        moe_intermediate_size=128,
        num_experts=16,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        architectures=["BeamForCausalLM"],
        torch_dtype="bfloat16",
    )
    generator = torch.Generator().manual_seed(2026)
    state = {}
    attention_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim

    def matrix(name, rows, columns, scale=0.05, dtype=torch.bfloat16):
        state[name] = (torch.randn(rows, columns, generator=generator) * scale).to(
            dtype
        )

    def norm(name, width):
        state[name] = (0.9 + 0.2 * torch.rand(width, generator=generator)).bfloat16()

    matrix("model.embed_tokens.weight", 128, 128, scale=0.5)
    # The BF16 projection output is promoted before FP32 softcap and logprobs.
    matrix("lm_head.weight", 128, 128, scale=0.2)
    norm("model.norm.weight", 128)
    for layer in range(4):
        prefix = f"model.layers.{layer}"
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            norm(f"{prefix}.{name}.weight", 128)
        for name, width in (
            ("q_proj", attention_width),
            ("k_proj", kv_width),
            ("v_proj", kv_width),
        ):
            matrix(f"{prefix}.self_attn.{name}.weight", width, 128)
        matrix(f"{prefix}.self_attn.gate_proj.weight", attention_width, 128)
        matrix(f"{prefix}.self_attn.o_proj.weight", 128, attention_width)
        norm(f"{prefix}.self_attn.q_norm.weight", 64)
        norm(f"{prefix}.self_attn.k_norm.weight", 64)
        state[f"{prefix}.self_attn.gate_bias"] = torch.tensor(
            [-1.125, -0.375, 0.625, 1.375], dtype=torch.bfloat16
        ).repeat(config.num_attention_heads // 4)
        if config.mlp_layer_types[layer] == "dense":
            mlps = [(f"{prefix}.mlp", 256)]
        else:
            matrix(f"{prefix}.mlp.gate.weight", 16, 128, dtype=torch.float32)
            # Keep a proven group-selection margin: sigmoid scores are in
            # [0, 1], so these biases separate retained/rejected group sums by
            # at least 2. BF16 attention rounding must not change the selected
            # branches in a numerical differential. Rotate groups across layers
            # to exercise every expert; combine weights remain input-dependent.
            # Score-dependent expert selection is covered by the router unit test.
            groups = (torch.arange(8) - 2 * (layer - 1)) % 8
            state[f"{prefix}.mlp.gate.e_score_correction_bias"] = torch.where(
                groups < 4, 1.0, -1.0
            ).repeat_interleave(2)
            mlps = [(f"{prefix}.mlp.experts.{expert}", 128) for expert in range(16)]
        for mlp, width in mlps:
            matrix(f"{mlp}.gate_proj.weight", width, 128)
            matrix(f"{mlp}.up_proj.weight", width, 128)
            matrix(f"{mlp}.down_proj.weight", 128, width)

    config.save_pretrained(path)
    save_file(state, str(Path(path) / "model.safetensors"))
    write_char_tokenizer(path)
    return config, state


def write_char_tokenizer(path):
    """Save a character-level tokenizer whose ids fit the fixture's 128-entry vocab.

    Printable ASCII plus newline and tab map to ids 4..100; anything else is
    ``<unk>``. Server launches without ``skip_tokenizer_init`` and the shared
    logprob-consistency helpers can tokenize text against the fixture.
    """
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    specials = ["<unk>", "<s>", "</s>", "<pad>"]
    chars = [chr(c) for c in range(32, 127)] + ["\n", "\t"]
    vocab = {token: i for i, token in enumerate(specials + chars)}
    assert len(vocab) <= 128
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    tokenizer.decoder = decoders.Fuse()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=8192,
    ).save_pretrained(path)


@torch.inference_mode()
@sdpa_kernel(SDPBackend.MATH)
def eager_beam_logits(config, state, token_ids):
    """BF16 activations; FP32 attention intermediates, SSMax and output processing.

    Pin attention to the math backend so the oracle does not select a fused
    kernel whose intermediate rounding varies with the device and input shape.
    """
    device = state["model.embed_tokens.weight"].device
    ids = torch.tensor(token_ids, dtype=torch.long, device=device)
    positions = torch.arange(ids.numel(), device=device)
    hidden = F.embedding(ids, state["model.embed_tokens.weight"])
    head_dim = config.head_dim
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads

    def norm(value, name):
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + config.rms_norm_eps
        )
        return (normalized * state[name]).to(value.dtype)

    def mlp(value, prefix, *, route_weight=None):
        gate = F.linear(value, state[f"{prefix}.gate_proj.weight"])
        up = F.linear(value, state[f"{prefix}.up_proj.weight"])
        activated = (F.silu(gate.float()) * up.float()).to(gate.dtype)
        down_weight = state[f"{prefix}.down_proj.weight"]
        if route_weight is not None:
            # The fused down projection applies routing weights to its FP32
            # accumulator before storing each weighted contribution in BF16.
            accumulator = F.linear(activated.float(), down_weight.float())
            return (accumulator * route_weight.float().unsqueeze(-1)).to(value.dtype)
        return F.linear(activated, down_weight)

    def residual_add(branch, residual, depth):
        return branch * (1 / math.sqrt(depth)) + residual * (
            math.sqrt(depth - 1) / math.sqrt(depth)
        )

    causal_mask = positions[:, None] >= positions[None, :]
    rope_theta = config.rope_parameters["sliding_attention"]["rope_theta"]
    inverse_frequencies = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    frequencies = positions.float()[:, None] * inverse_frequencies
    angles = torch.cat((frequencies, frequencies), dim=-1)
    cos, sin = angles.cos()[:, None], angles.sin()[:, None]
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        residual = hidden
        hidden = norm(hidden, f"{prefix}.input_layernorm.weight")
        attn = f"{prefix}.self_attn"
        q = F.linear(hidden, state[f"{attn}.q_proj.weight"]).view(
            -1, num_heads, head_dim
        )
        k = F.linear(hidden, state[f"{attn}.k_proj.weight"]).view(
            -1, num_kv_heads, head_dim
        )
        v = F.linear(hidden, state[f"{attn}.v_proj.weight"]).view(
            -1, num_kv_heads, head_dim
        )
        gate = F.linear(hidden, state[f"{attn}.gate_proj.weight"]).view(
            -1, num_heads, head_dim
        )
        q = norm(q, f"{attn}.q_norm.weight")
        k = norm(k, f"{attn}.k_norm.weight")
        if config.layer_types[layer] == "sliding_attention":
            q, k = q.float(), k.float()
            q1, q2 = q.chunk(2, dim=-1)
            k1, k2 = k.chunk(2, dim=-1)
            q = (q * cos + torch.cat((-q2, q1), dim=-1) * sin).bfloat16()
            k = (k * cos + torch.cat((-k2, k1), dim=-1) * sin).bfloat16()
        else:
            scale = torch.exp(0.1 * torch.log1p((positions.float() + 1) / 2048))
            q = (q.float() * scale[:, None, None]).bfloat16()
        mask = causal_mask
        if config.layer_types[layer] == "sliding_attention":
            mask = mask & (positions[:, None] - positions[None, :] < 512)
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.repeat_interleave(num_heads // num_kv_heads, dim=1)
            .transpose(0, 1)
            .unsqueeze(0),
            v.repeat_interleave(num_heads // num_kv_heads, dim=1)
            .transpose(0, 1)
            .unsqueeze(0),
            attn_mask=mask,
            scale=head_dim**-0.5,
        )[0].transpose(0, 1)
        output = output * torch.sigmoid(
            gate + state[f"{attn}.gate_bias"][None, :, None]
        )
        output = F.linear(
            output.reshape(-1, num_heads * head_dim), state[f"{attn}.o_proj.weight"]
        )
        output = norm(output, f"{prefix}.post_attention_layernorm.weight")
        hidden = residual_add(output, residual, 2 * (layer + 1))
        residual = hidden
        hidden = norm(hidden, f"{prefix}.pre_feedforward_layernorm.weight")
        if config.mlp_layer_types[layer] == "dense":
            output = mlp(hidden, f"{prefix}.mlp")
        else:
            router = f"{prefix}.mlp.gate"
            scores = F.linear(hidden.float(), state[f"{router}.weight"]).sigmoid()
            corrected = scores + state[f"{router}.e_score_correction_bias"]
            groups = corrected.view(-1, 8, 2).topk(2, dim=-1).values.sum(-1)
            retained = torch.zeros_like(groups, dtype=torch.bool).scatter_(
                1, groups.topk(4, dim=-1).indices, True
            )
            selected = (
                corrected.masked_fill(~retained.repeat_interleave(2, dim=1), -torch.inf)
                .topk(8, dim=-1)
                .indices
            )
            weights = scores.gather(1, selected)
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
            contributions = hidden.new_empty((*selected.shape, hidden.shape[-1]))
            for expert in range(config.num_experts):
                tokens, slots = torch.where(selected == expert)
                contributions[tokens, slots] = mlp(
                    hidden[tokens],
                    f"{prefix}.mlp.experts.{expert}",
                    route_weight=weights[tokens, slots],
                )
            # Weight each expert before its BF16 store; sum and scale in FP32.
            output = (
                contributions.float().sum(dim=1) * config.routed_scaling_factor
            ).to(hidden.dtype)
        output = norm(output, f"{prefix}.post_feedforward_layernorm.weight")
        hidden = residual_add(output, residual, 2 * (layer + 1) + 1)
    hidden = norm(hidden, "model.norm.weight")
    logits = F.linear(hidden, state["lm_head.weight"]).float()
    return 30 * torch.tanh(logits / 30)


def engine_token_logprobs(engine, token_ids, count, vocab_size):
    """Greedy-generate ``count`` tokens; return their ids and full-vocab logprobs."""
    result = engine.generate(
        input_ids=token_ids,
        sampling_params={
            "temperature": 0,
            "max_new_tokens": count,
            "ignore_eos": True,
        },
        return_logprob=True,
        token_ids_logprob=list(range(vocab_size)),
    )
    return result["output_ids"], torch.tensor(
        [
            [entry[0] for entry in row]
            for row in result["meta_info"]["output_token_ids_logprobs"]
        ],
        device="cuda",
    )


def assert_logprobs_close(test, actual, expected, *, tag, max_kl=BEAM_FIXTURE_MAX_KL):
    delta = (actual - expected).abs()
    kl = (expected.exp() * (expected - actual)).sum(-1).max().item()
    print(
        f"{tag}: max_abs={delta.max().item():.8f} mean_abs={delta.mean().item():.8f} "
        f"max_kl={kl:.8f}",
        flush=True,
    )
    # Optimized BF16 attention and TP partial sums have different rounding
    # orders; the elementwise envelope bounds every token, the KL bound each
    # distribution. Primitive precision and window membership have strict tests.
    torch.testing.assert_close(actual, expected, atol=0.15, rtol=0.002)
    test.assertLessEqual(kl, max_kl, msg=f"{tag}: per-token KL")
