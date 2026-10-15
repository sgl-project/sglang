"""DFlash drafts with MLA attention (SpecForge ``dflash_config.attention_mode: mla``),
served as expanded MHA."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3Config

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.models.dflash import DFlashDraftModel, DFlashMLAAttention
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_utils import (
    dflash_draft_cell_size_per_token,
    expand_dflash_mla_kv_shapes,
    get_dflash_attention_modes,
    is_dflash_mla_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

HIDDEN, HEADS, KV_LORA, NOPE, ROPE, VDIM, THETA, EPS = 16, 2, 8, 8, 4, 6, 1e4, 1e-6
# SpecForge's Qwen3DFlashMLAAttention parameter names (external checkpoint layout).
SPECFORGE_KV_KEYS = {
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
}
SPECFORGE_Q_KEYS = {
    None: {"q_proj.weight"},
    12: {"q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight"},
}


def _export_fields(q_lora_rank=None, rope_interleave=True, **dflash_overrides):
    """A SpecForge MLA export: head_dim / num_key_value_heads keep the GQA
    template's values, which the MLA layers do not use."""
    dflash_config = {"attention_mode": "mla", "block_size": 4, "target_layer_ids": [0]}
    dflash_config.update(dflash_overrides)
    return dict(
        architectures=["DFlashDraftModel"],
        hidden_size=HIDDEN,
        intermediate_size=32,
        num_hidden_layers=1,
        num_target_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=64,
        rms_norm_eps=EPS,
        rope_theta=THETA,
        max_position_embeddings=4096,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=VDIM,
        rope_interleave=rope_interleave,
        dflash_config=dflash_config,
    )


def _served_config(**kwargs) -> Qwen3Config:
    config = Qwen3Config(**_export_fields(**kwargs))
    expand_dflash_mla_kv_shapes(config)
    return config


def _randomize(module: nn.Module) -> None:
    torch.manual_seed(0)
    with torch.no_grad():
        for param in module.parameters():
            param.copy_(torch.randn_like(param) * 0.2 + (param == 1).float())


# ---- SpecForge reference math (Qwen3DFlashMLAAttention._compute_qkv) ---------
def _rmsnorm(x, weight):
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)).to(
        x.dtype
    ) * weight


def _rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rotate_half_interleaved(x):
    first, second = x.reshape(*x.shape[:-1], -1, 2).unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def _mla_rope(x, positions, interleaved):
    """x [n, heads, ROPE]; Qwen3RotaryEmbedding(head_dim=ROPE) + apply_mla_rope."""
    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROPE, 2).float() / ROPE))
    freqs = positions.float()[:, None] * inv_freq.to(positions.device)[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().to(x.dtype), emb.sin().to(x.dtype)
    if interleaved:
        cos = cos[..., : ROPE // 2].repeat_interleave(2, dim=-1)
        sin = sin[..., : ROPE // 2].repeat_interleave(2, dim=-1)
        rotated = _rotate_half_interleaved(x)
    else:
        rotated = _rotate_half(x)
    return x * cos[:, None] + rotated * sin[:, None]


def _reference_kv(w, hidden, positions, interleaved):
    latent = hidden @ w["kv_a_proj_with_mqa.weight"].T
    kv_a, k_pe = latent.split([KV_LORA, ROPE], dim=-1)
    kv = _rmsnorm(kv_a, w["kv_a_layernorm.weight"]) @ w["kv_b_proj.weight"].T
    k_nope, v = kv.view(-1, HEADS, NOPE + VDIM).split([NOPE, VDIM], dim=-1)
    k_pe = _mla_rope(k_pe[:, None], positions, interleaved).expand(-1, HEADS, -1)
    return torch.cat((k_nope, k_pe), dim=-1), v


def _reference_output(attn, ctx_hidden, ctx_positions, hidden, positions, interleaved):
    """Computed in fp32 from the module's (possibly bf16) weights."""
    w = {name: param.float() for name, param in attn.named_parameters()}
    ctx_hidden, hidden = ctx_hidden.float(), hidden.float()
    if attn.q_lora_rank is None:
        q = hidden @ w["q_proj.weight"].T
    else:
        q_a = _rmsnorm(hidden @ w["q_a_proj.weight"].T, w["q_a_layernorm.weight"])
        q = q_a @ w["q_b_proj.weight"].T
    q_nope, q_pe = q.view(-1, HEADS, NOPE + ROPE).split([NOPE, ROPE], dim=-1)
    q = torch.cat((q_nope, _mla_rope(q_pe, positions, interleaved)), dim=-1)
    k_ctx, v_ctx = _reference_kv(w, ctx_hidden, ctx_positions, interleaved)
    k_blk, v_blk = _reference_kv(w, hidden, positions, interleaved)
    k, v = torch.cat((k_ctx, k_blk)), torch.cat((v_ctx, v_blk))
    probs = (torch.einsum("qhd,khd->hqk", q, k) * (NOPE + ROPE) ** -0.5).softmax(-1)
    out = torch.einsum("hqk,khd->qhd", probs, v).reshape(-1, HEADS * VDIM)
    return out @ w["o_proj.weight"].T


class _BlockOverCachedContext(nn.Module):
    """Stands in for RadixAttention: the draft block attends non-causally to
    the cached context rows plus its own K/V."""

    def __init__(self, radix, k_ctx, v_ctx):
        super().__init__()
        self.radix, self.k_ctx, self.v_ctx = radix, k_ctx, v_ctx

    def forward(self, q, k, v, forward_batch):
        heads, qk_dim, v_dim = (
            self.radix.tp_q_head_num,
            self.radix.qk_head_dim,
            self.radix.v_head_dim,
        )
        keys = torch.cat((self.k_ctx, k)).view(-1, heads, qk_dim)
        values = torch.cat((self.v_ctx, v)).view(-1, heads, v_dim)
        scores = torch.einsum("qhd,khd->hqk", q.view(-1, heads, qk_dim), keys)
        probs = (scores * self.radix.scaling).softmax(-1)
        return torch.einsum("hqk,khd->qhd", probs, values).reshape(-1, heads * v_dim)


def _serve_block(attn, ctx_hidden, ctx_positions, hidden, positions):
    """The worker's context-KV hook order, then the draft block forward."""
    k_ctx, v_ctx = attn.kv_proj_only(ctx_hidden)
    k_ctx = attn.apply_k_rope(ctx_positions, attn.apply_k_norm(k_ctx))
    attn.attn = _BlockOverCachedContext(attn.attn, k_ctx, v_ctx)
    return attn(positions, hidden, None)


class _SingleRankTestCase(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy", device="cpu", tp_size=1), role="test")


class TestDFlashMLAConfig(_SingleRankTestCase):
    def test_scalar_and_per_layer_mode_forms(self):
        """SpecForge exports the scalar; the Qwen3.8 A-MLA checkpoint carries a list."""
        layers = 3
        for dflash_config in (
            {"attention_mode": "MLA"},
            {"attention_modes": ["mla"] * layers},
        ):
            with self.subTest(dflash_config=dflash_config):
                config = {"num_hidden_layers": layers, "dflash_config": dflash_config}
                self.assertEqual(get_dflash_attention_modes(config), ("mla",) * layers)
                self.assertTrue(is_dflash_mla_draft_config(config))
        gqa = {"num_hidden_layers": layers, "dflash_config": {"block_size": 4}}
        self.assertFalse(is_dflash_mla_draft_config(gqa))

    def test_unservable_mode_layouts_fail_fast(self):
        cases = {
            "only one": {"attention_mode": "mla", "attention_modes": ["mla"] * 2},
            "one mode per draft layer": {"attention_modes": ["mla"]},
            "not supported": {"attention_mode": "kda"},
            "mix mla": {"attention_modes": ["mla", "gqa"]},
        }
        for message, dflash_config in cases.items():
            with self.subTest(message=message):
                config = {"num_hidden_layers": 2, "dflash_config": dflash_config}
                with self.assertRaisesRegex(ValueError, message):
                    is_dflash_mla_draft_config(config)

    def test_draft_model_config_sizes_expanded_kv_pool(self):
        """A SpecForge export must serve without hand-editing head_dim and
        num_key_value_heads: the draft KV pool holds one [k_nope | k_pe] row
        and one v row per attention head."""
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(
                json.dumps({"model_type": "qwen3", **_export_fields()})
            )
            draft = ModelConfig(path, is_draft_model=True)

        self.assertEqual((draft.head_dim, draft.v_head_dim), (NOPE + ROPE, VDIM))
        self.assertEqual(draft.get_num_kv_heads(1), HEADS)
        self.assertEqual(
            dflash_draft_cell_size_per_token(
                draft_model_config=draft,
                draft_num_layers=1,
                draft_kv_cache_dtype=torch.bfloat16,
                tp_size=1,
            ),
            HEADS * (NOPE + ROPE + VDIM) * 2,
        )

    def test_draft_attention_rejects_unexpanded_kv_layout(self):
        """Guards the pool/attention contract when a config skips ModelConfig."""
        with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
            DFlashMLAAttention(Qwen3Config(**_export_fields()), layer_id=0)


class TestDFlashMLAWeights(_SingleRankTestCase):
    def test_specforge_checkpoint_loads_for_both_q_parameterizations(self):
        for q_lora_rank, q_keys in SPECFORGE_Q_KEYS.items():
            with self.subTest(q_lora_rank=q_lora_rank):
                model = DFlashDraftModel(_served_config(q_lora_rank=q_lora_rank))
                self.assertFalse(model.supports_fused_context_kv)
                attn_keys = {
                    name.removeprefix("layers.0.self_attn.")
                    for name, _ in model.named_parameters()
                    if name.startswith("layers.0.self_attn.")
                }
                self.assertEqual(attn_keys, SPECFORGE_KV_KEYS | q_keys)

                checkpoint = {}
                for name, param in model.named_parameters():
                    if name.endswith("gate_up_proj.weight"):
                        gate, up = torch.randn_like(param).chunk(2)
                        prefix = name.removesuffix("gate_up_proj.weight")
                        checkpoint[prefix + "gate_proj.weight"] = gate
                        checkpoint[prefix + "up_proj.weight"] = up
                    else:
                        checkpoint[name] = torch.randn_like(param)
                model.load_weights(checkpoint.items())

                for name, param in model.named_parameters():
                    if name.endswith("gate_up_proj.weight"):
                        prefix = name.removesuffix("gate_up_proj.weight")
                        expected = torch.cat(
                            (
                                checkpoint[prefix + "gate_proj.weight"],
                                checkpoint[prefix + "up_proj.weight"],
                            )
                        )
                    else:
                        expected = checkpoint[name]
                    torch.testing.assert_close(param, expected, rtol=0, atol=0)


class TestDFlashMLANumerics(_SingleRankTestCase):
    """Context KV materialization and the draft block forward must reproduce the
    attention SpecForge trained: partial RoPE at the right positions, the right
    pair convention, and K/V rows that differ in width."""

    def _check(self, *, device, dtype, use_native_ops, atol):
        for q_lora_rank, interleaved in ((None, True), (12, False)):
            with self.subTest(q_lora_rank=q_lora_rank, rope_interleave=interleaved):
                attn = DFlashMLAAttention(
                    _served_config(
                        q_lora_rank=q_lora_rank, rope_interleave=interleaved
                    ),
                    layer_id=0,
                )
                _randomize(attn)
                # get_rope() shares one cached instance; keep .to() from leaking out.
                attn.rotary_emb = copy.deepcopy(attn.rotary_emb)
                attn.to(device=device, dtype=dtype)
                if use_native_ops:
                    norms = [attn.kv_a_layernorm]
                    if q_lora_rank is not None:
                        norms.append(attn.q_a_layernorm)
                    for module in (attn.rotary_emb, *norms):
                        module.forward = module.forward_native

                torch.manual_seed(1)
                ctx_hidden = torch.randn(5, HIDDEN, device=device, dtype=dtype)
                hidden = torch.randn(4, HIDDEN, device=device, dtype=dtype)
                ctx_positions = torch.tensor([0, 1, 2, 7, 100], device=device)
                positions = torch.arange(1000, 1004, device=device)

                expected = _reference_output(
                    attn, ctx_hidden, ctx_positions, hidden, positions, interleaved
                )
                actual = _serve_block(
                    attn, ctx_hidden, ctx_positions, hidden, positions
                )
                torch.testing.assert_close(
                    actual.float(), expected, atol=atol, rtol=atol
                )

    def test_matches_specforge_reference_cpu(self):
        self._check(device="cpu", dtype=torch.float32, use_native_ops=True, atol=1e-5)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_kernels_match_specforge_reference(self):
        self._check(
            device="cuda", dtype=torch.bfloat16, use_native_ops=False, atol=5e-2
        )


if __name__ == "__main__":
    unittest.main()
