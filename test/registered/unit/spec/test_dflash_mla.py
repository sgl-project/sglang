"""DFlash and DSpark MLA drafts on the target MLA layer against SpecForge's
expanded math."""

import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from transformers import Qwen3Config

from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_loader.loader import post_load_weights
from sglang.srt.models.dflash import DFlashDraftModel, DFlashMLAAttention
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_utils import (
    dflash_draft_cell_size_per_token,
    parse_dflash_draft_config,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.utils import BumpAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN, HEADS, KV_LORA, NOPE, ROPE, VDIM, THETA, EPS = 16, 2, 8, 8, 8, 6, 1e4, 1e-6
Q_LORA = 16
# SpecForge's attention parameter names (external checkpoint layout).
SPECFORGE_KEYS = {
    None: {"q_proj.weight"},
    Q_LORA: {"q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight"},
}
SPECFORGE_KV_KEYS = {
    "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight",
    "kv_b_proj.weight",
    "o_proj.weight",
}


def _export_fields(q_lora_rank=None, rope_interleave=True):
    """A SpecForge MLA export; head_dim / num_key_value_heads are GQA template
    values the MLA layers do not use."""
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
        dflash_config={
            "attention_mode": "mla",
            "block_size": 4,
            "target_layer_ids": [0],
        },
    )


def _specforge_weights(attn) -> dict:
    """The served layer's weights under SpecForge's names (split down projection)."""
    weights = dict(attn.named_parameters())
    if attn.q_lora_rank is not None:
        fused = weights.pop("fused_qkv_a_proj_with_mqa.weight")
        weights["q_a_proj.weight"], weights["kv_a_proj_with_mqa.weight"] = fused.split(
            [attn.q_lora_rank, KV_LORA + ROPE]
        )
    return weights


# ---- SpecForge reference math (Qwen3DFlashMLAAttention._compute_qkv) ---------
def _rmsnorm(x, weight):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight


def _mla_rope(x, positions, interleaved):
    """x [n, heads, ROPE]; Qwen3RotaryEmbedding(head_dim=ROPE) + apply_mla_rope."""
    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROPE, 2).float() / ROPE))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    if interleaved:
        cos, sin = (
            freqs.cos().repeat_interleave(2, -1),
            freqs.sin().repeat_interleave(2, -1),
        )
        first, second = x.unflatten(-1, (-1, 2)).unbind(-1)
        rotated = torch.stack((-second, first), -1).flatten(-2)
    else:
        cos, sin = freqs.cos().repeat(1, 2), freqs.sin().repeat(1, 2)
        first, second = x.chunk(2, -1)
        rotated = torch.cat((-second, first), -1)
    return x * cos[:, None] + rotated * sin[:, None]


def _reference_output(attn, ctx_hidden, ctx_positions, hidden, positions, interleaved):
    w = _specforge_weights(attn)

    def kv(x, pos):
        kv_a, k_pe = (x @ w["kv_a_proj_with_mqa.weight"].T).split([KV_LORA, ROPE], -1)
        kv = _rmsnorm(kv_a, w["kv_a_layernorm.weight"]) @ w["kv_b_proj.weight"].T
        k_nope, v = kv.view(-1, HEADS, NOPE + VDIM).split([NOPE, VDIM], -1)
        k_pe = _mla_rope(k_pe[:, None], pos, interleaved).expand(-1, HEADS, -1)
        return torch.cat((k_nope, k_pe), -1), v

    if attn.q_lora_rank is None:
        q = hidden @ w["q_proj.weight"].T
    else:
        q_a = _rmsnorm(hidden @ w["q_a_proj.weight"].T, w["q_a_layernorm.weight"])
        q = q_a @ w["q_b_proj.weight"].T
    q_nope, q_pe = q.view(-1, HEADS, NOPE + ROPE).split([NOPE, ROPE], -1)
    q = torch.cat((q_nope, _mla_rope(q_pe, positions, interleaved)), -1)
    (k_ctx, v_ctx), (k_blk, v_blk) = (
        kv(ctx_hidden, ctx_positions),
        kv(hidden, positions),
    )
    k, v = torch.cat((k_ctx, k_blk)), torch.cat((v_ctx, v_blk))
    probs = (torch.einsum("qhd,khd->hqk", q, k) * (NOPE + ROPE) ** -0.5).softmax(-1)
    out = torch.einsum("hqk,khd->qhd", probs, v).reshape(-1, HEADS * VDIM)
    return out @ w["o_proj.weight"].T


class _BlockOverCachedContext(torch.nn.Module):
    """Bidirectional attention of the block over the cached latent context, in the
    triton call convention: concatenated [latent | rope] Q/K rows, latent V."""

    def __init__(self, radix, k_ctx, v_ctx):
        super().__init__()
        self.radix, self.k_ctx, self.v_ctx = radix, k_ctx, v_ctx

    def forward(self, q, k, v, forward_batch, **kwargs):
        assert self.radix.attn_type == AttentionType.ENCODER_ONLY
        heads = self.radix.tp_q_head_num
        keys = torch.cat((self.k_ctx, k)).expand(-1, heads, -1)
        values = torch.cat((self.v_ctx, v)).expand(-1, heads, -1)
        q = q.view(-1, heads, self.radix.qk_head_dim)
        probs = (torch.einsum("qhd,khd->hqk", q, keys) * self.radix.scaling).softmax(-1)
        return torch.einsum("hqk,khd->qhd", probs, values).flatten(1)


def _serve_block(attn, ctx_hidden, ctx_positions, hidden, positions):
    """The worker's context-KV write, then the draft block forward."""
    k_nope, k_pe = attn.project_context_kv(ctx_positions, ctx_hidden)
    attn.attn_mqa = _BlockOverCachedContext(
        attn.attn_mqa, torch.cat((k_nope, k_pe), -1), k_nope
    )
    backend = SimpleNamespace(
        prefill_attention_backend_str="triton", decode_attention_backend_str="triton"
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        extend_prefix_lens_cpu=None,
        attn_cp_metadata=None,
    )
    with (
        torch.no_grad(),
        forward_context(ForwardContext(attn_backend=backend)),
        patch.object(
            attn.kv_b_proj, "forward", side_effect=AssertionError("expanded K/V")
        ),
    ):
        allocator = BumpAllocator(buffer_size=2, dtype=torch.float32, device="cpu")
        return attn(positions, hidden, batch, zero_allocator=allocator)


class TestDFlashMLA(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        # CPU CI does not enable the CPU engine; keep RoPE on the native path.
        self.enterContext(
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        )
        # The CPU RoPE factory serves only DeepSeek YaRN; the tensors stay on CPU.
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")

    def test_attention_mode_forms(self):
        """SpecForge exports the scalar; the Qwen3.8 A-MLA checkpoint carries a list."""
        for dflash_config in (
            {"attention_mode": "MLA"},
            {"attention_modes": ["mla"] * 2},
        ):
            config = {"num_hidden_layers": 2, "dflash_config": dflash_config}
            self.assertTrue(parse_dflash_draft_config(draft_hf_config=config).is_mla)
            # The DSpark launch hook reads it from DSpark's own parse.
            self.assertTrue(parse_dspark_draft_config(draft_hf_config=config).is_mla)
        config = {"num_hidden_layers": 2}
        self.assertFalse(parse_dflash_draft_config(draft_hf_config=config).is_mla)
        with self.assertRaisesRegex(ValueError, "mix mla"):
            parse_dflash_draft_config(
                draft_hf_config={
                    "num_hidden_layers": 2,
                    "dflash_config": {"attention_modes": ["mla", "gqa"]},
                }
            )

    def test_draft_model_config_sizes_compressed_kv_pool(self):
        with tempfile.TemporaryDirectory() as path:
            Path(path, "config.json").write_text(
                json.dumps({"model_type": "qwen3", **_export_fields()})
            )
            draft = ModelConfig(path, is_draft_model=True)
        self.assertEqual(draft.attention_arch, AttentionArch.MLA)
        # The template's GQA head_dim must not size a separate SWA buffer.
        self.assertEqual(draft.swa_v_head_dim, draft.v_head_dim)
        # One latent + RoPE row per token, replicated across TP ranks.
        for tp_size in (1, 2):
            self.assertEqual(
                dflash_draft_cell_size_per_token(
                    draft_model_config=draft,
                    draft_num_layers=1,
                    draft_kv_cache_dtype=torch.bfloat16,
                    tp_size=tp_size,
                ),
                (KV_LORA + ROPE) * 2,
            )

    def test_every_load_path_absorbs_kv_b_proj(self):
        """Split and fused checkpoints, and loaders that bypass load_weights,
        all leave the w_kc / w_vc the absorbed forward reads."""

        def assert_absorbed(attn):
            w_kc, w_vc = attn.kv_b_proj.weight.view(HEADS, NOPE + VDIM, KV_LORA).split(
                [NOPE, VDIM], dim=1
            )
            torch.testing.assert_close(attn.w_kc, w_kc, rtol=0, atol=0)
            torch.testing.assert_close(attn.w_vc, w_vc.transpose(1, 2), rtol=0, atol=0)

        for q_lora_rank, q_keys in SPECFORGE_KEYS.items():
            with self.subTest(q_lora_rank=q_lora_rank):
                model = DFlashDraftModel(Qwen3Config(**_export_fields(q_lora_rank)))
                attn = model.layers[0].self_attn
                checkpoint = {
                    f"layers.0.self_attn.{name}": torch.randn_like(param)
                    for name, param in _specforge_weights(attn).items()
                }
                self.assertEqual(
                    {name.rsplit("self_attn.", 1)[1] for name in checkpoint},
                    SPECFORGE_KV_KEYS | q_keys,
                )
                model.load_weights(checkpoint.items())
                for name, param in _specforge_weights(attn).items():
                    torch.testing.assert_close(
                        param, checkpoint[f"layers.0.self_attn.{name}"], rtol=0, atol=0
                    )
                assert_absorbed(attn)

                fused = {
                    f"layers.0.self_attn.{name}": torch.randn_like(param)
                    for name, param in attn.named_parameters()
                }
                model.load_weights(fused.items())
                for name, param in attn.named_parameters():
                    torch.testing.assert_close(
                        param, fused[f"layers.0.self_attn.{name}"], rtol=0, atol=0
                    )
                assert_absorbed(attn)

                with torch.no_grad():
                    attn.kv_b_proj.weight.normal_()
                post_load_weights(model)
                assert_absorbed(attn)

    def test_sliding_layers_require_triton(self):
        """Only triton applies an MLA layer's sliding window; the other MLA
        backends would silently attend over the full context."""
        config = Qwen3Config(
            **_export_fields(),
            layer_types=["sliding_attention"],
            use_sliding_window=True,
            sliding_window=4,
        )
        resolve = "sglang.srt.speculative.draft_worker_common.resolve_draft_worker_attention_backend"
        with patch(resolve, return_value="flashinfer"):
            with self.assertRaisesRegex(ValueError, "require .* triton"):
                DFlashMLAAttention(config, layer_id=0)
        with patch(resolve, return_value="triton"):
            attn = DFlashMLAAttention(config, layer_id=0)
        self.assertEqual(attn.attn_mqa.sliding_window_size, 3)

    def test_matches_specforge_reference(self):
        """Context-KV materialization and the block forward reproduce SpecForge's
        expanded attention: partial RoPE, pair convention, K/V of different width."""
        for q_lora_rank, interleaved in itertools.product(
            (None, Q_LORA), (True, False)
        ):
            with self.subTest(q_lora_rank=q_lora_rank, rope_interleave=interleaved):
                attn = DFlashMLAAttention(
                    Qwen3Config(**_export_fields(q_lora_rank, interleaved)), layer_id=0
                )
                # get_rope() shares one cached instance across layers.
                attn.rotary_emb = copy.deepcopy(attn.rotary_emb)
                torch.manual_seed(0)
                with torch.no_grad():
                    for param in attn.parameters():
                        param.copy_(
                            torch.randn_like(param) * 0.2 + (param == 1).float()
                        )
                attn.absorb_kv_b_proj()
                norms = [attn.kv_a_layernorm] + (
                    [attn.q_a_layernorm] if q_lora_rank else []
                )
                for module in (attn.rotary_emb, *norms):
                    module.forward = module.forward_native

                ctx_hidden, hidden = torch.randn(5, HIDDEN), torch.randn(4, HIDDEN)
                ctx_positions = torch.tensor([0, 1, 2, 7, 100])
                positions = torch.arange(1000, 1004)
                with torch.no_grad():
                    expected = _reference_output(
                        attn, ctx_hidden, ctx_positions, hidden, positions, interleaved
                    )
                actual = _serve_block(
                    attn, ctx_hidden, ctx_positions, hidden, positions
                )
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_prefix_valid_write_skips_rows_past_commit_lens(self):
        """Rows past commit_lens are not written anywhere, slot 0 included,
        like the MHA pool's prefix-valid write."""
        # CUDA runs the Triton write kernel, other platforms its naive twin.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pool = MLATokenToKVPool(
            size=15,
            page_size=1,
            dtype=torch.bfloat16,
            kv_lora_rank=KV_LORA,
            qk_rope_head_dim=ROPE,
            layer_num=1,
            device=device,
            enable_memory_saver=False,
        )
        pool.kv_buffer[0].fill_(7)
        latent = torch.randn(6, 1, KV_LORA, dtype=torch.bfloat16, device=device)
        rope = torch.randn(6, 1, ROPE, dtype=torch.bfloat16, device=device)
        pool.set_mla_kv_buffer_prefix_valid(
            SimpleNamespace(layer_id=0),
            torch.tensor([[3, 4, 5], [8, 9, 10]], device=device),
            torch.tensor([2, 1], dtype=torch.int32, device=device),
            latent,
            rope,
        )
        expected = torch.full_like(pool.kv_buffer[0], 7)
        expected[[3, 4, 8]] = torch.cat((latent, rope), dim=-1)[[0, 1, 3]]
        torch.testing.assert_close(pool.kv_buffer[0], expected, rtol=0, atol=0)

    def test_dspark_commits_verify_rows_through_mla_layer(self):
        """DSpark's fused context-KV paths slice a qkv_proj; an MLA draft must commit
        the layer's latent rows through the masked MLA writer instead."""
        model = DSparkDraftModel(
            Qwen3Config(**_export_fields(), markov_rank=4, markov_head_type="vanilla")
        )
        attn = model.layers[0].self_attn
        attn.rotary_emb = copy.deepcopy(attn.rotary_emb)
        for module in (attn.rotary_emb, attn.kv_a_layernorm):
            module.forward = module.forward_native
        # A real MLA pool also exposes the K/V buffer getters the GQA bundle probes.
        pool = Mock(spec=MLATokenToKVPool)
        ctx_hidden, positions = torch.randn(4, HIDDEN), torch.tensor([0, 1, 5, 6])
        loc_2d = torch.tensor([[3, 4], [8, 9]])
        commit_lens = torch.tensor([1, 2], dtype=torch.int32)
        with torch.no_grad():
            expected = attn.project_context_kv(positions, ctx_hidden)
            model.write_target_hidden_kv(
                target_hidden=ctx_hidden,
                pool=pool,
                positions=positions,
                cache_loc=loc_2d.flatten(),
                cache_loc_2d=loc_2d,
                commit_lens=commit_lens,
                target_hidden_is_projected=True,
            )
        layer, got_loc, got_lens, *rows = (
            pool.set_mla_kv_buffer_prefix_valid.call_args.args
        )
        self.assertIs(layer, attn.attn_mqa)
        self.assertIs(got_loc, loc_2d)
        self.assertIs(got_lens, commit_lens)
        for got, want in zip(rows, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
