"""CPU unit tests for AngelSpec DFly2 support on DeepSeek-V4.1-Flash."""

from __future__ import annotations

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.deepseek_v41 import DeepseekV41Config
from sglang.srt.configs.model_config import is_deepseek_v4, is_hybrid_swa_model
from sglang.srt.models.deepseek_v4_dfly import (
    DFlyDeepseekV41DraftModel,
    DFlyDSV41AttentiveCorrection,
    _hc_mixes,
    _hc_post,
    _hc_pre,
    apply_dsv41_rope,
    compute_dsv41_rope_cos_sin,
    compute_dsv41_yarn_inv_freq,
    parse_dsv41_dfly_config,
)
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import commit_accept, validate_dflash_request
from sglang.srt.speculative.dfly_utils import (
    DFlyRequestState,
    build_dfly_extra_topk,
    build_dfly_swa_indices,
)
from sglang.srt.speculative.dfly_worker_v2 import DFlyWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import (
    CustomSpecAlgo,
    _assert_custom_spec_algo_conforms,
)
from sglang.srt.speculative.spec_tp_sync import _RNG, SpecTpSyncSite

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


# ---------------------------------------------------------------------------
# Reference config matching AngelSlim/DS-V4.1-Flash-DFly2-B6
# ---------------------------------------------------------------------------


def _make_tiny_dsv41_dfly_hf_config(
    *,
    hidden_size: int = 64,
    num_heads: int = 4,
    head_dim: int = 32,
    rope_dim: int = 8,
    q_lora_rank: int = 16,
    o_lora_rank: int = 16,
    o_groups: int = 2,
    n_routed_experts: int = 8,
    num_experts_per_tok: int = 2,
    block_size: int = 6,
    causal_block: bool = False,
    target_kv_layers=(28, 32, 36),
    target_index_layers=(28, 32, 36),
):
    return DeepseekV41Config(
        architectures=["DFlyDeepseekV41DraftModel"],
        vocab_size=256,
        hidden_size=hidden_size,
        num_hidden_layers=40,
        num_attention_heads=num_heads,
        num_key_value_heads=1,
        head_dim=head_dim,
        qk_rope_head_dim=rope_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        o_groups=o_groups,
        rms_norm_eps=1e-20,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        sliding_window=128,
        index_topk=512,
        n_routed_experts=n_routed_experts,
        n_shared_experts=1,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=32,
        compress_rope_theta=160000,
        dfly_config={
            "block_size": block_size,
            "causal_block": causal_block,
            "target_kv_layers": list(target_kv_layers),
            "target_index_layers": list(target_index_layers),
            "index_topk": 512,
            "mask_token_id": 255,
            "rope": {
                "theta": 160000.0,
                "original_seq_len": 65536,
                "factor": 16.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            },
            "mtp_fusion": "slot0",
            "hidden_correction": {"intermediate_size": hidden_size},
            "attentive_correction": {
                "source_layer": -1,
                "block_keys": True,
                "block_mode": "gated",
                "block_kv_in": "hidden_embed",
                "max_block_slots": 16,
            },
            "dense_dtype": "bfloat16",
            "n_routed_experts": n_routed_experts,
            "num_experts_per_tok": num_experts_per_tok,
        },
    )


def _single_rank_parallel():
    return SimpleNamespace(
        tp_size=1,
        tp_rank=0,
        attn_tp_size=1,
        attn_tp_rank=0,
        tp_group=SimpleNamespace(world_size=1),
    )


# ---------------------------------------------------------------------------
# 1. SpeculativeAlgorithm, registry, tp_sync, request validation
# ---------------------------------------------------------------------------


def test_dfly_algorithm_predicates_and_registry():
    algo = SpeculativeAlgorithm.from_string("DFLY")
    assert algo is SpeculativeAlgorithm.DFLY
    assert algo.is_dfly()
    assert algo.uses_block_verify()
    assert not algo.is_dflash_family()
    assert not algo.is_dflash()
    assert not algo.is_dspark()
    assert not algo.has_draft_kv()
    assert algo.supports_target_verify_for_draft()
    assert algo.supports_prefill_shared_read_done()
    assert not algo.supports_grammar_overlap()
    assert algo.is_speculative()

    # CustomSpecAlgo duck-typing contract must remain satisfied.
    _assert_custom_spec_algo_conforms(CustomSpecAlgo)
    assert SpecTpSyncSite.DFLY_ACCEPT_GREEDY not in _RNG
    assert SpecTpSyncSite.DFLY_ACCEPT_SAMPLE in _RNG

    # ModelConfig recognizes DFlyDeepseekV41DraftModel.
    cfg = SimpleNamespace(architectures=["DFlyDeepseekV41DraftModel"])
    assert is_deepseek_v4(cfg)
    assert is_hybrid_swa_model(["DFlyDeepseekV41DraftModel"])


def test_dfly_validate_request_rejects_grammar():
    req_plain = SimpleNamespace(
        return_hidden_states=False,
        sampling_params=SimpleNamespace(
            json_schema=None, regex=None, ebnf=None, structural_tag=None
        ),
    )
    assert (
        validate_dflash_request(
            req_plain, enable_overlap=True, bound_runtime=False, supports_grammar=False
        )
        is None
    )

    req_grammar = SimpleNamespace(
        return_hidden_states=False,
        sampling_params=SimpleNamespace(
            json_schema='{"type": "object"}',
            regex=None,
            ebnf=None,
            structural_tag=None,
        ),
    )
    err = validate_dflash_request(
        req_grammar, enable_overlap=True, bound_runtime=False, supports_grammar=False
    )
    assert err is not None and "constrained decoding" in err


def test_dfly_server_args_resolution():
    from sglang.srt.arg_groups.model_override_base import resolving_view
    from sglang.srt.arg_groups.speculative_hook import _handle_dfly
    from sglang.srt.server_args import ServerArgs

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "architectures": ["DFlyDeepseekV41DraftModel"],
                    "model_type": "deepseek_v41",
                    "dfly_config": {"block_size": 6},
                },
                f,
            )
        sa = ServerArgs(
            model_path="dummy-target",
            speculative_algorithm="DFLY",
            speculative_draft_model_path=tmpdir,
            device="cuda",
        )
        _handle_dfly(sa)
        rv = resolving_view(sa)
        assert rv.speculative_num_draft_tokens == 6
        assert rv.speculative_num_steps == 1
        assert rv.speculative_eagle_topk == 1
        assert rv.max_running_requests == 48

        # Explicit --speculative-dfly-block-size overrides checkpoint block_size.
        sa2 = ServerArgs(
            model_path="dummy-target",
            speculative_algorithm="DFLY",
            speculative_draft_model_path=tmpdir,
            speculative_dfly_block_size=8,
            device="cuda",
        )
        _handle_dfly(sa2)
        assert resolving_view(sa2).speculative_num_draft_tokens == 8

        # Incompatible options are rejected.
        sa_pp = ServerArgs(
            model_path="dummy-target",
            speculative_algorithm="DFLY",
            pp_size=2,
            device="cuda",
        )
        with pytest.raises(ValueError, match="PP"):
            _handle_dfly(sa_pp)


# ---------------------------------------------------------------------------
# 2. Config parsing (exact AngelSlim/DS-V4.1-Flash-DFly2-B6 schema)
# ---------------------------------------------------------------------------


def test_parse_dsv41_dfly_config_matches_hf_release():
    hf_cfg = DeepseekV41Config(
        architectures=["DFlyDeepseekV41DraftModel"],
        text_config={
            "vocab_size": 129280,
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "num_attention_heads": 64,
            "head_dim": 512,
            "qk_rope_head_dim": 64,
            "q_lora_rank": 1280,
            "o_lora_rank": 1024,
            "o_groups": 8,
            "n_routed_experts": 384,
            "num_experts_per_tok": 6,
            "sliding_window": 128,
            "index_topk": 512,
        },
        dfly_config={
            "block_size": 6,
            "causal_block": False,
            "target_kv_layers": [28, 32, 36],
            "target_index_layers": [28, 32, 36],
            "index_topk": 512,
            "mask_token_id": 128799,
            "rope": {
                "theta": 160000.0,
                "original_seq_len": 65536,
                "factor": 16.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
            },
            "mtp_fusion": "slot0",
            "hidden_correction": {"intermediate_size": 5120},
            "attentive_correction": {
                "source_layer": -1,
                "block_keys": True,
                "block_mode": "gated",
                "block_kv_in": "hidden_embed",
                "max_block_slots": 16,
            },
        },
    )

    # Simulate probing 128 experts from safetensors header (exact HF layout).
    with patch(
        "sglang.srt.models.deepseek_v4_dfly._probe_checkpoint_num_experts",
        return_value=128,
    ):
        cfg = parse_dsv41_dfly_config(hf_cfg)

    assert cfg.num_layers == 3
    assert cfg.block_size == 6
    assert cfg.causal_block is False
    assert cfg.mask_token_id == 128799
    assert cfg.target_kv_layers == (28, 32, 36)
    assert cfg.target_index_layers == (28, 32, 36)
    assert cfg.target_index_topk == 512
    assert cfg.sliding_window == 128
    assert cfg.n_routed_experts == 128
    assert cfg.num_experts_per_tok == 3
    assert cfg.attentive_correction is not None
    assert cfg.attentive_correction.source_layer == 2  # -1 % 3
    assert cfg.attentive_correction.block_mode == "gated"


# ---------------------------------------------------------------------------
# 3. DFly utilities: state, extra topk packing, SWA index building
# ---------------------------------------------------------------------------


def test_build_dfly_extra_topk_packs_and_repeats():
    state = DFlyRequestState(
        pool_size=8,
        hidden_size=16,
        num_layers=3,
        target_topk=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    req_ids = torch.tensor([2, 5], dtype=torch.int64)
    # Mix valid entries and -1 holes in unsorted positions.
    topk = torch.tensor(
        [
            [[-1, 10, -1, 20, 30, -1, -1, -1], [-1, -1, -1, -1, -1, -1, -1, -1]],
            [[7, 8, 9, -1, -1, -1, -1, -1], [42, -1, 99, -1, -1, -1, -1, -1]],
            [[1, 2, 3, 4, 5, 6, 7, 8], [100, -1, -1, -1, -1, -1, -1, -1]],
        ],
        dtype=torch.int32,
    )
    state.write(
        req_pool_indices=req_ids,
        hidden=torch.ones(2, 16),
        topk=topk,
    )

    out_rows = torch.full((3, 12, 8), -1, dtype=torch.int32)
    out_lengths = torch.zeros((3, 12), dtype=torch.int32)
    block_rows, block_lengths, ctx_rows, ctx_counts = build_dfly_extra_topk(
        state_topk=state.topk,
        req_pool_indices=req_ids,
        block_size=4,
        out_rows=out_rows,
        out_lengths=out_lengths,
        align=8,
    )

    # Layer 0, req 0 (pool slot 2): 3 valid entries packed to front [10, 20, 30].
    assert ctx_counts[0, 0].item() == 3
    assert ctx_rows[0, 0, :4].tolist() == [10, 20, 30, -1]
    assert block_lengths[0, :4].tolist() == [3, 3, 3, 3]
    for j in range(4):
        assert block_rows[0, j, :4].tolist() == [10, 20, 30, -1]

    # Layer 0, req 1 (pool slot 5): 0 valid entries -> ctx_counts=0, block_lengths clamped to 1.
    assert ctx_counts[0, 1].item() == 0
    assert block_lengths[0, 4:8].tolist() == [1, 1, 1, 1]


def test_build_dfly_swa_indices_paged_and_request_window():
    # Paged SWA path: bs=2, prefix_lens=[3, 0], block_size=4, swa_window=8, align=16.
    req_to_token = torch.arange(100, dtype=torch.int32).view(5, 20)
    full_to_swa = torch.arange(100, dtype=torch.int32) + 1000
    swa_out = torch.tensor([500, 501, 502, 503, 600, 601, 602, 603], dtype=torch.int32)

    block_idx, block_len, ctx_idx, ctx_cnt = build_dfly_swa_indices(
        req_pool_indices=torch.tensor([1, 3], dtype=torch.int64),
        prefix_lens=torch.tensor([3, 0], dtype=torch.int64),
        block_size=4,
        swa_window=8,
        causal_block=False,
        req_to_token=req_to_token,
        full_to_swa_mapping=full_to_swa,
        swa_out_cache_loc=swa_out,
        align=16,
    )
    assert block_idx.shape == (8, 16)
    assert ctx_idx.shape == (2, 16)
    # Req 0 (pool 1, prefix_len=3): slots for pos 0,1,2 -> full [20,21,22] -> swa [1020,1021,1022]
    assert ctx_cnt.tolist() == [3, 0]
    assert ctx_idx[0, :4].tolist() == [1020, 1021, 1022, -1]
    # Bidirectional block: every row of req 0 sees [1020, 1021, 1022, 500, 501, 502, 503]
    for j in range(4):
        assert block_len[j].item() == 7
        assert block_idx[j, :8].tolist() == [1020, 1021, 1022, 500, 501, 502, 503, -1]
    # Req 1 (prefix_len=0): sees only its 4 block slots [600, 601, 602, 603]
    for j in range(4, 8):
        assert block_len[j].item() == 4
        assert block_idx[j, :5].tolist() == [600, 601, 602, 603, -1]

    # RequestWindow path with causal_block=True.
    layout = SimpleNamespace(
        history_valid=torch.tensor(
            [False, False, True, True, True, True, True, True]
            + [False, False, False, False, False, False, True, True]
        ),
        history_loc=torch.arange(16, dtype=torch.int32),
        write_loc=torch.arange(16, 24, dtype=torch.int32),
    )
    c_block_idx, c_block_len, c_ctx_idx, c_ctx_cnt = build_dfly_swa_indices(
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        prefix_lens=torch.tensor([6, 2], dtype=torch.int64),
        block_size=4,
        swa_window=8,
        causal_block=True,
        req_to_token=req_to_token,
        full_to_swa_mapping=None,
        swa_out_cache_loc=None,
        request_window_layout=layout,
        align=16,
    )
    assert c_ctx_cnt.tolist() == [6, 2]
    assert c_ctx_idx[1, :3].tolist() == [14, 15, -1]
    assert c_block_len[:4].tolist() == [7, 8, 9, 10]
    assert c_block_idx[0, :8].tolist() == [2, 3, 4, 5, 6, 7, 16, -1]
    assert c_block_idx[2, :10].tolist() == [2, 3, 4, 5, 6, 7, 16, 17, 18, -1]


# ---------------------------------------------------------------------------
# 4. Numerical parity with AngelSpec math (RoPE, mHC, MTPFusion, AttentiveCorrection)
# ---------------------------------------------------------------------------


def test_rope_roundtrip_and_yarn_matches_reference():
    cfg = _make_tiny_dsv41_dfly_hf_config()
    dfly_cfg = parse_dsv41_dfly_config(cfg)
    inv_freq = compute_dsv41_yarn_inv_freq(cfg.qk_rope_head_dim, dfly_cfg.rope)
    positions = torch.tensor([0, 1, 128, 70000], dtype=torch.int64)
    cos, sin = compute_dsv41_rope_cos_sin(positions, inv_freq)

    x = torch.randn(4, 32, dtype=torch.float32)
    rot = apply_dsv41_rope(x, cos, sin, inverse=False)
    back = apply_dsv41_rope(rot, cos, sin, inverse=True)
    # Non-RoPE channels untouched, RoPE channels invert exactly.
    assert torch.equal(rot[:, :-8], x[:, :-8])
    assert torch.allclose(back, x, atol=1e-5, rtol=1e-5)


def test_mhc_doubly_stochastic_mixing():
    torch.manual_seed(0)
    hc_mult, dim = 4, 16
    mix = (2 + hc_mult) * hc_mult
    h = torch.randn(3, hc_mult, dim)
    fn = torch.randn(mix, hc_mult * dim) * 0.1
    scale = torch.tensor([0.5, 0.5, 0.5])
    base = torch.zeros(mix)

    pre, post, comb = _hc_mixes(h, fn, scale, base, iters=20, eps=1e-6, norm_eps=1e-20)
    # Sinkhorn produces approximately doubly-stochastic comb matrices.
    assert torch.allclose(comb.sum(dim=-1), torch.ones(3, hc_mult), atol=1e-3)
    assert torch.allclose(comb.sum(dim=-2), torch.ones(3, hc_mult), atol=1e-3)
    collapsed = _hc_pre(h, pre)
    assert collapsed.shape == (3, dim)
    updated = _hc_post(torch.randn(3, dim), h, post, comb)
    assert updated.shape == (3, hc_mult, dim)


def test_attentive_correction_slot_by_slot_matches_full_block():
    """Verify slot-by-slot streaming inference in DFlyDSV41AttentiveCorrection
    matches full-block forward (AngelSpec dsv41_attentive_correction.py)."""
    torch.manual_seed(42)
    hf_cfg = _make_tiny_dsv41_dfly_hf_config(block_size=5)
    dfly_cfg = parse_dsv41_dfly_config(hf_cfg)
    bs, width, ctx_len = 2, 5, 10
    dim = hf_cfg.hidden_size
    head_dim = hf_cfg.head_dim

    with (
        patch(
            "sglang.srt.models.deepseek_v4_dfly.get_parallel",
            return_value=_single_rank_parallel(),
        ),
        patch(
            "sglang.srt.layers.linear.get_parallel",
            return_value=_single_rank_parallel(),
        ),
        patch(
            "sglang.srt.layers.linear.get_group_rank_size",
            return_value=(0, 1),
        ),
    ):
        ac = DFlyDSV41AttentiveCorrection(
            config=hf_cfg,
            dfly_config=dfly_cfg,
            ac_config=dfly_cfg.attentive_correction,
            source_target_layer=36,
        )
        # Randomize all weights (including zero-initialized gates/wo_b) to test every term.
        with torch.no_grad():
            for p in ac.parameters():
                p.normal_(std=0.2)

        hidden_norm = torch.randn(bs, width, dim)
        embed_norm = torch.randn(bs, width, dim)
        positions = torch.tensor([[10, 11, 12, 13, 14], [20, 21, 22, 23, 24]])
        ctx_latent = torch.randn(bs, ctx_len, head_dim)
        # Request 0 has 6 context keys; Request 1 has 0 context keys (empty context edge case).
        ctx_indices = torch.tensor(
            [[0, 2, 4, 5, 8, 9], [-1, -1, -1, -1, -1, -1]], dtype=torch.int32
        )

        cos, sin = compute_dsv41_rope_cos_sin(positions, ac.inv_freq)

        # --- Full-block reference (AngelSpec equation) ---
        inputs_full = torch.cat([hidden_norm, embed_norm], dim=-1)  # [bs, width, 2D]
        q_full = ac.wq_b(ac.q_norm(ac.q_in(inputs_full)[0]))[0].view(
            bs, width, ac.n_heads, head_dim
        )
        q_full = apply_dsv41_rope(q_full, cos.unsqueeze(2), sin.unsqueeze(2))
        block_full = apply_dsv41_rope(ac.kv_norm(ac.kv_in(inputs_full)[0]), cos, sin)

        # Dense context attention for every slot
        valid_ctx = ctx_indices >= 0  # [bs, K]
        safe_ctx = ctx_indices.clamp(min=0).long()
        k_ctx = torch.gather(
            ctx_latent, 1, safe_ctx.unsqueeze(-1).expand(-1, -1, head_dim)
        ).float()  # [bs, K, head_dim]
        ctx_logits = torch.einsum("bshm,bkm->bhsk", q_full.float(), k_ctx) * ac.scaling
        ctx_logits = ctx_logits.masked_fill(~valid_ctx[:, None, None, :], float("-inf"))
        ctx_total = torch.logaddexp(
            torch.logsumexp(ctx_logits, dim=-1, keepdim=True),
            ac.attn_sink.float().view(1, -1, 1, 1),
        )
        ctx_probs = torch.where(
            valid_ctx[:, None, None, :],
            torch.exp(ctx_logits - ctx_total),
            torch.zeros_like(ctx_logits),
        )
        ctx_read_full = torch.einsum("bhsk,bkm->bshm", ctx_probs, k_ctx)
        alive = valid_ctx.any(dim=-1)  # [bs]
        ctx_read_full = torch.where(
            alive[:, None, None, None], ctx_read_full, torch.zeros_like(ctx_read_full)
        )

        # Gated causal block attention across all slots
        blk_logits = (
            torch.einsum("bshm,bkm->bhsk", q_full.float(), block_full.float())
            * ac.scaling
        )
        steps = torch.arange(width)
        back = steps.view(-1, 1) - steps.view(1, -1)
        blk_logits = blk_logits + ac.block_rel_bias.float()[:, back.clamp(min=0)]
        blk_logits = blk_logits.masked_fill(back < 0, float("-inf"))
        blk_total = torch.logaddexp(
            torch.logsumexp(blk_logits, dim=-1, keepdim=True),
            ac.block_sink.float().view(1, -1, 1, 1),
        )
        blk_read_full = torch.einsum(
            "bhsk,bkm->bshm", torch.exp(blk_logits - blk_total), block_full.float()
        )
        gate_full = torch.tanh(
            ac.block_gate(inputs_full)[0].float()
            + ac.block_gate_bias.float()[:, steps].transpose(0, 1)
        )
        combined_full = (ctx_read_full + blk_read_full * gate_full.unsqueeze(-1)).to(
            q_full.dtype
        )
        combined_full = apply_dsv41_rope(
            combined_full, cos.unsqueeze(2), sin.unsqueeze(2), inverse=True
        )
        wo_a_w = ac.wo_a.weight.view(ac.n_groups, ac.o_lora_rank, -1)
        ref_out = ac.wo_b(
            torch.einsum(
                "bsgd,grd->bsgr",
                combined_full.view(bs, width, ac.n_groups, -1).float(),
                wo_a_w.float(),
            )
            .flatten(2)
            .to(q_full.dtype)
        )[0]

        # --- Slot-by-slot streaming forward (DFlyDSV41AttentiveCorrection.forward_step) ---
        def fake_ctx_read_fn(q_in: torch.Tensor, attn_sink: torch.Tensor):
            q_real = q_in[:, : ac.n_local_heads, :].float()
            logits = torch.einsum("bhm,bkm->bhk", q_real, k_ctx) * ac.scaling
            logits = logits.masked_fill(~valid_ctx[:, None, :], float("-inf"))
            tot = torch.logaddexp(
                torch.logsumexp(logits, dim=-1, keepdim=True),
                attn_sink[: ac.n_local_heads].float().view(1, -1, 1),
            )
            probs = torch.where(
                valid_ctx[:, None, :],
                torch.exp(logits - tot),
                torch.zeros_like(logits),
            )
            return torch.einsum("bhk,bkm->bhm", probs, k_ctx).to(q_in.dtype)

        block_keys_buf = torch.empty(bs, width, head_dim)
        inputs_0 = torch.cat([hidden_norm[:, 0], embed_norm[:, 0]], dim=-1)
        block_keys_buf[:, 0] = ac.compute_block_key(
            inputs_0, hidden_norm[:, 0], cos[:, 0], sin[:, 0]
        )

        for step in range(1, width):
            inputs_s = torch.cat([hidden_norm[:, step], embed_norm[:, step]], dim=-1)
            step_out = ac.forward_step(
                inputs=inputs_s,
                hidden_norm=hidden_norm[:, step],
                cos=cos[:, step],
                sin=sin[:, step],
                step=step,
                block_keys_buf=block_keys_buf,
                context_read_fn=fake_ctx_read_fn,
                has_context=alive,
            )
            assert torch.allclose(step_out, ref_out[:, step], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. Full DFlyDeepseekV41DraftModel weight remapping & loading
# ---------------------------------------------------------------------------


class _DummyMoE(nn.Module):
    """Lightweight stub matching DeepseekV2MoE's parameter names for CPU tests."""

    def __init__(self, hidden_size: int, inter_size: int, n_experts: int):
        super().__init__()
        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(torch.empty(n_experts, hidden_size))
        self.gate.e_score_correction_bias = nn.Parameter(torch.empty(n_experts))

        self.shared_experts = nn.Module()
        self.shared_experts.gate_up_proj = nn.Linear(
            hidden_size, 2 * inter_size, bias=False
        )
        self.shared_experts.gate_up_proj.weight.weight_loader = self._stacked_loader
        self.shared_experts.down_proj = nn.Linear(inter_size, hidden_size, bias=False)

        self.experts = nn.Module()
        self.experts.w13_weight = nn.Parameter(
            torch.empty(n_experts, 2 * inter_size, hidden_size)
        )
        self.experts.w13_weight.weight_loader = self._expert_loader
        self.experts.w2_weight = nn.Parameter(
            torch.empty(n_experts, hidden_size, inter_size)
        )
        self.experts.w2_weight.weight_loader = self._expert_loader

    @staticmethod
    def _stacked_loader(param, loaded, shard_id):
        half = param.shape[0] // 2
        param.data[shard_id * half : (shard_id + 1) * half].copy_(loaded)

    @staticmethod
    def _expert_loader(param, loaded, name, shard_id, expert_id):
        if shard_id == "w1":
            half = param.shape[1] // 2
            param.data[expert_id, :half].copy_(loaded)
        elif shard_id == "w3":
            half = param.shape[1] // 2
            param.data[expert_id, half:].copy_(loaded)
        elif shard_id == "w2":
            param.data[expert_id].copy_(loaded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_dfly_draft_model_load_weights_complete_key_coverage():
    hf_cfg = _make_tiny_dsv41_dfly_hf_config()
    d = hf_cfg.hidden_size
    inter = hf_cfg.moe_intermediate_size
    n_exp = hf_cfg.n_routed_experts
    q_lora = hf_cfg.q_lora_rank
    n_heads = hf_cfg.num_attention_heads
    head_dim = hf_cfg.head_dim
    o_groups = hf_cfg.o_groups
    o_lora = hf_cfg.o_lora_rank
    heads_per_group = n_heads // o_groups

    with (
        patch(
            "sglang.srt.models.deepseek_v4_dfly.get_parallel",
            return_value=_single_rank_parallel(),
        ),
        patch(
            "sglang.srt.models.deepseek_v4_dfly.get_num_fused_shared_experts",
            return_value=0,
        ),
        patch(
            "sglang.srt.layers.linear.get_parallel",
            return_value=_single_rank_parallel(),
        ),
        patch(
            "sglang.srt.layers.linear.get_group_rank_size",
            return_value=(0, 1),
        ),
    ):
        model = DFlyDeepseekV41DraftModel(
            config=hf_cfg,
            quant_config=None,
            mlp_factory=lambda _i: _DummyMoE(d, inter, n_exp),
        )

        # Build exact HuggingFace checkpoint keys from AngelSlim/DS-V4.1-Flash-DFly2-B6.
        weights = [
            ("model.final_norm.weight", torch.ones(d)),
            ("model.mtp_fusion.enorm.weight", torch.ones(d)),
            ("model.mtp_fusion.hnorm.weight", torch.ones(d)),
            ("model.mtp_fusion.eh_proj.weight", torch.randn(d, 2 * d)),
            ("model.mtp_fusion.gate", torch.zeros(d)),
            ("model.hidden_correction.hidden_norm.weight", torch.ones(d)),
            ("model.hidden_correction.embed_norm.weight", torch.ones(d)),
            ("model.hidden_correction.gate_proj.weight", torch.randn(d, 2 * d)),
            ("model.hidden_correction.up_proj.weight", torch.randn(d, 2 * d)),
            ("model.hidden_correction.down_proj.weight", torch.randn(d, d)),
            (
                "model.hidden_correction.attention.q_in.weight",
                torch.randn(q_lora, 2 * d),
            ),
            ("model.hidden_correction.attention.q_norm.weight", torch.ones(q_lora)),
            (
                "model.hidden_correction.attention.wq_b.weight",
                torch.randn(n_heads * head_dim, q_lora),
            ),
            ("model.hidden_correction.attention.attn_sink", torch.zeros(n_heads)),
            (
                "model.hidden_correction.attention.wo_a.weight",
                torch.randn(o_groups * o_lora, heads_per_group * head_dim),
            ),
            (
                "model.hidden_correction.attention.wo_b.weight",
                torch.randn(d, o_groups * o_lora),
            ),
            (
                "model.hidden_correction.attention.kv_in.weight",
                torch.randn(head_dim, 2 * d),
            ),
            ("model.hidden_correction.attention.kv_norm.weight", torch.ones(head_dim)),
            ("model.hidden_correction.attention.block_sink", torch.zeros(n_heads)),
            (
                "model.hidden_correction.attention.block_rel_bias",
                torch.zeros(n_heads, 16),
            ),
            (
                "model.hidden_correction.attention.block_gate.weight",
                torch.zeros(n_heads, 2 * d),
            ),
            (
                "model.hidden_correction.attention.block_gate_bias",
                torch.zeros(n_heads, 16),
            ),
            # Ignored optional keys must not fail loading.
            ("model.confidence_head.proj.weight", torch.zeros(1, d)),
        ]
        for i in range(3):
            p = f"model.layers.{i}"
            weights.extend(
                [
                    (f"{p}.attn_norm.weight", torch.ones(d)),
                    (f"{p}.ffn_norm.weight", torch.ones(d)),
                    (f"{p}.hc_attn_base", torch.zeros(24)),
                    (f"{p}.hc_attn_fn", torch.zeros(24, 4 * d)),
                    (f"{p}.hc_attn_scale", torch.zeros(3)),
                    (f"{p}.hc_ffn_base", torch.zeros(24)),
                    (f"{p}.hc_ffn_fn", torch.zeros(24, 4 * d)),
                    (f"{p}.hc_ffn_scale", torch.zeros(3)),
                    (f"{p}.attn.attn_sink", torch.zeros(n_heads)),
                    (f"{p}.attn.wq_a.weight", torch.randn(q_lora, d)),
                    (f"{p}.attn.q_norm.weight", torch.ones(q_lora)),
                    (f"{p}.attn.wq_b.weight", torch.randn(n_heads * head_dim, q_lora)),
                    (f"{p}.attn.wkv.weight", torch.randn(head_dim, d)),
                    (f"{p}.attn.kv_norm.weight", torch.ones(head_dim)),
                    (
                        f"{p}.attn.wo_a.weight",
                        torch.randn(o_groups * o_lora, heads_per_group * head_dim),
                    ),
                    (f"{p}.attn.wo_b.weight", torch.randn(d, o_groups * o_lora)),
                    (f"{p}.ffn.gate.weight", torch.randn(n_exp, d)),
                    (f"{p}.ffn.gate.bias", torch.zeros(n_exp)),
                    (f"{p}.ffn.gate.bias_vl", torch.zeros(n_exp)),
                    (f"{p}.ffn.shared_experts.w1.weight", torch.randn(inter, d)),
                    (f"{p}.ffn.shared_experts.w2.weight", torch.randn(d, inter)),
                    (f"{p}.ffn.shared_experts.w3.weight", torch.randn(inter, d)),
                ]
            )
            for e in range(n_exp):
                weights.extend(
                    [
                        (f"{p}.ffn.experts.{e}.w1.weight", torch.randn(inter, d)),
                        (f"{p}.ffn.experts.{e}.w2.weight", torch.randn(d, inter)),
                        (f"{p}.ffn.experts.{e}.w3.weight", torch.randn(inter, d)),
                    ]
                )

        # Should load every parameter with zero missing weights.
        model.load_weights(weights)


# ---------------------------------------------------------------------------
# 6. commit_accept & DFlyWorkerV2 state commit
# ---------------------------------------------------------------------------


def test_commit_accept_places_bonus_at_boundary():
    candidates = torch.tensor(
        [
            [10, 11, 12, 13, 14, 15],
            [20, 21, 22, 23, 24, 25],
            [30, 31, 32, 33, 34, 35],
        ],
        dtype=torch.int64,
    )
    accept_len = torch.tensor([0, 2, 5], dtype=torch.int32)
    bonus = torch.tensor([100, 200, 300], dtype=torch.int64)
    out_tokens, commit_lens = commit_accept(candidates, accept_len, bonus)
    assert commit_lens.tolist() == [1, 3, 6]
    assert out_tokens[0, :1].tolist() == [100]
    assert out_tokens[1, :3].tolist() == [21, 22, 200]
    assert out_tokens[2, :6].tolist() == [31, 32, 33, 34, 35, 300]


def test_dfly_worker_prefill_and_commit_state():
    worker = DFlyWorkerV2.__new__(DFlyWorkerV2)
    worker.device = torch.device("cpu")
    worker.block_size = 6
    worker.state = DFlyRequestState(
        pool_size=8,
        hidden_size=4,
        num_layers=3,
        target_topk=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    worker._topk_capture = torch.arange(3 * 12 * 4, dtype=torch.int32).view(3, 12, 4)
    worker.model_runner = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(engram_hasher=None))
    )

    # Prefill writes hidden[0..bs-1] and topk[:, 0..bs-1] into state.
    logits_out = SimpleNamespace(
        hidden_states=torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32
        )
    )
    prefill_res = SimpleNamespace(
        logits_output=logits_out,
        next_token_ids=torch.tensor([42, 99], dtype=torch.int64),
    )
    worker._target_worker = SimpleNamespace(
        forward_batch_generation=MagicMock(return_value=prefill_res)
    )
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        seq_lens=torch.tensor([10, 20], dtype=torch.int64),
        req_pool_indices=torch.tensor([3, 6], dtype=torch.int64),
        batch_size=lambda: 2,
    )
    res = worker._forward_prefill(batch, on_publish=None)
    assert isinstance(res.next_draft_input, DFlashDraftInputV2)
    assert res.logits_output.hidden_states is None
    assert torch.equal(worker.state.hidden[3], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(worker.state.hidden[6], torch.tensor([5.0, 6.0, 7.0, 8.0]))
    assert torch.equal(worker.state.topk[:, 3], worker._topk_capture[:, 0])
    assert torch.equal(worker.state.topk[:, 6], worker._topk_capture[:, 1])

    # Commit state from verify source_rows (e.g. req 0 accepted 2 drafts -> row 2,
    # req 1 accepted 4 drafts -> row 6 + 4 = 10).
    verify_hidden = torch.arange(12 * 4, dtype=torch.float32).view(12, 4)
    source_rows = torch.tensor([2, 10], dtype=torch.int64)
    worker._commit_state(batch, verify_hidden, source_rows)
    assert torch.equal(worker.state.hidden[3], verify_hidden[2])
    assert torch.equal(worker.state.hidden[6], verify_hidden[10])
    assert torch.equal(worker.state.topk[:, 3], worker._topk_capture[:, 2])
    assert torch.equal(worker.state.topk[:, 6], worker._topk_capture[:, 10])
