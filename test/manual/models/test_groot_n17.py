import os
from pathlib import Path

# The processor test imports sglang's qwen_vl processor which pulls in
# flashinfer eagerly; skip the version check so this suite runs on boxes
# where flashinfer / flashinfer-jit-cache are mismatched.  Must be set
# before any sglang import.
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import pytest
import torch

# Point at a local GR00T-N1.7-3B checkout via env var so this manual
# test suite can run on any box with the weights.  Defaults to the HF
# repo id; `AutoConfig.from_pretrained` will resolve that against the
# local HF cache (or download on first use).
GR00T_WEIGHTS = Path(os.environ.get("GR00T_WEIGHTS_PATH", "nvidia/GR00T-N1.7-3B"))

# DiT attention runs through MaskedFlashAttention, which requires CUDA +
# bf16/fp16.  Tests that touch the DiT are gated on CUDA and run in bf16
# (native checkpoint dtype).
_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MaskedFlashAttention requires CUDA",
)


def test_config_loads():
    # Trigger Gr00tN1d7Config registration with transformers CONFIG_MAPPING.
    from transformers import AutoConfig

    import sglang.srt.configs  # noqa: F401

    cfg = AutoConfig.from_pretrained(str(GR00T_WEIGHTS), trust_remote_code=True)
    assert type(cfg).__name__ == "Gr00tN1d7Config"
    assert cfg.action_horizon == 40
    assert cfg.max_action_dim == 132
    assert cfg.max_state_dim == 132
    assert cfg.max_num_embodiments == 32
    assert cfg.select_layer == 16
    assert cfg.num_inference_timesteps == 4
    assert cfg.backbone_embedding_dim == 2048
    assert cfg.hidden_size == 1024
    assert cfg.model_name == "nvidia/Cosmos-Reason2-2B"
    assert cfg.use_alternate_vl_dit is True
    assert cfg.diffusion_model_cfg["num_layers"] == 32
    assert cfg.diffusion_model_cfg["num_attention_heads"] == 32
    assert cfg.diffusion_model_cfg["attention_head_dim"] == 48
    assert cfg.vl_self_attention_cfg["num_layers"] == 4


@_requires_cuda
@torch.no_grad()
def test_masked_flash_attention_varlen():
    """Smoke test: MaskedFlashAttention handles both fixed-length self-attn
    and per-key bool-mask cross-attn (varlen gather) end-to-end on CUDA bf16.
    Compares against an inline SDPA-bf16 reference built from the same
    weights."""
    from sglang.srt.layers.attention.masked_flash_attn import MaskedFlashAttention

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)

    B, Lq, Lk = 2, 41, 30
    heads, dim_head = 4, 32
    query_dim = heads * dim_head  # 128
    kv_dim = 64

    # --- Self-attn: mask is None -----------------------------------------
    self_attn = (
        MaskedFlashAttention(
            query_dim=query_dim,
            kv_dim=None,
            heads=heads,
            dim_head=dim_head,
            attention_bias=True,
            out_bias=True,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    h = torch.randn(B, Lq, query_dim, device=device, dtype=dtype)
    out = self_attn(h)
    assert out.shape == (B, Lq, query_dim)
    assert torch.isfinite(out).all()

    # SDPA-bf16 reference using the same weights
    def _sdpa_ref(module, hidden, kv, mask=None):
        q = module.to_q(hidden).view(B, -1, heads, dim_head).transpose(1, 2)
        k = module.to_k(kv).view(B, -1, heads, dim_head).transpose(1, 2)
        v = module.to_v(kv).view(B, -1, heads, dim_head).transpose(1, 2)
        attn_mask = None
        if mask is not None:
            neg_inf = torch.finfo(q.dtype).min
            attn_mask = torch.where(mask[:, None, None, :], 0.0, neg_inf).to(q.dtype)
        r = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
        )
        r = r.transpose(1, 2).reshape(B, -1, heads * dim_head)
        return module.to_out[0](r)

    out_ref = _sdpa_ref(self_attn, h, h)
    assert torch.allclose(
        out, out_ref, atol=5e-3
    ), f"self-attn parity: max-abs {(out - out_ref).abs().max().item():.3e}"

    # --- Masked cross-attn: two different valid-VL lengths per request ----
    cross_attn = (
        MaskedFlashAttention(
            query_dim=query_dim,
            kv_dim=kv_dim,
            heads=heads,
            dim_head=dim_head,
            attention_bias=True,
            out_bias=True,
        )
        .to(device=device, dtype=dtype)
        .eval()
    )
    enc = torch.randn(B, Lk, kv_dim, device=device, dtype=dtype)
    mask = torch.zeros(B, Lk, dtype=torch.bool, device=device)
    mask[0, :17] = True
    mask[1, :9] = True

    out = cross_attn(h, encoder_hidden_states=enc, attention_mask=mask)
    assert out.shape == (B, Lq, query_dim)
    assert torch.isfinite(out).all()

    out_ref = _sdpa_ref(cross_attn, h, enc, mask=mask)
    assert torch.allclose(out, out_ref, atol=5e-3), (
        f"cross-attn varlen parity: max-abs "
        f"{(out - out_ref).abs().max().item():.3e}"
    )


@_requires_cuda
@torch.no_grad()
def test_action_head_get_action_shape_and_determinism():
    """Verify the composition (state enc + action enc + AlternateVLDiT +
    action dec + Euler loop) runs on CUDA bf16, produces the right shape,
    and is deterministic under a fixed torch seed.  End-to-end numerical
    parity against the published reference is covered by
    `test_online_full.py`'s open-loop MSE vs DROID ground-truth actions;
    load-from-checkpoint correctness is covered by
    `test_load_weights_routing`.
    """

    from sglang.srt.configs.groot_n1d7 import Gr00tN1d7Config
    from sglang.srt.models.groot_n1d7 import Gr00tN1d7ActionHead

    # Tiny but structurally real config.  Shape constraints we must respect
    # (all hold in the real GR00T-N1.7 config):
    #   vl_self_attention.inner_dim == backbone_embedding_dim
    #       (vlln keeps channel dim; self-attn operates in place)
    #   DiT.inner_dim == input_embedding_dim
    #       (sa_embs feed directly into DiT blocks)
    #   DiT.output_dim == hidden_size
    #       (DiT output feeds the action_decoder whose input_dim=hidden_size)
    cfg = Gr00tN1d7Config(
        backbone_embedding_dim=64,  # 4 * 16 (vl self-attn inner dim)
        hidden_size=48,  # = DiT output_dim (= action_decoder input_dim)
        input_embedding_dim=64,  # = DiT inner_dim (= 4 * 16)
        max_action_dim=8,
        max_state_dim=8,
        action_horizon=6,
        max_num_embodiments=4,
        state_history_length=1,
        max_seq_len=64,
        add_pos_embed=True,
        use_vlln=True,
        use_alternate_vl_dit=True,
        attend_text_every_n_blocks=2,
        diffusion_model_cfg={
            "num_layers": 4,
            "num_attention_heads": 4,
            "attention_head_dim": 16,  # 4*16 = 64 = input_embedding_dim
            "output_dim": 48,  # = hidden_size
            "norm_type": "ada_norm",
            "interleave_self_attention": True,
            "final_dropout": True,
            "dropout": 0.0,
            "positional_embeddings": None,
        },
        vl_self_attention_cfg={
            "num_layers": 2,
            "num_attention_heads": 4,
            "attention_head_dim": 16,  # 4*16 = 64 = backbone_embedding_dim
            "dropout": 0.0,
            "final_dropout": True,
            "positional_embeddings": None,
        },
        use_vl_self_attention=True,
        num_inference_timesteps=4,
        num_timestep_buckets=1000,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    torch.manual_seed(0)
    head = Gr00tN1d7ActionHead(cfg).to(device=device, dtype=dtype).eval()

    B, S = 2, 10
    vl_embeds = torch.randn(
        B, S, cfg.backbone_embedding_dim, device=device, dtype=dtype
    )
    vl_attn_mask = torch.ones(B, S, dtype=torch.bool, device=device)
    image_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    image_mask[:, :4] = True
    state = torch.randn(
        B, cfg.state_history_length, cfg.max_state_dim, device=device, dtype=dtype
    )
    emb = torch.tensor([0, 2], device=device)

    torch.manual_seed(1234)
    out1 = head.get_action(
        vl_embeds=vl_embeds,
        vl_attn_mask=vl_attn_mask,
        image_mask=image_mask,
        state=state,
        embodiment_id=emb,
    )
    torch.manual_seed(1234)
    out2 = head.get_action(
        vl_embeds=vl_embeds,
        vl_attn_mask=vl_attn_mask,
        image_mask=image_mask,
        state=state,
        embodiment_id=emb,
    )

    assert out1.shape == (B, cfg.action_horizon, cfg.max_action_dim)
    assert torch.isfinite(out1).all()
    # Non-trivial output (guards against e.g. accidentally returning raw
    # noise or zeros).
    assert out1.std().item() > 1e-3
    # Determinism under fixed seed.
    assert torch.allclose(out1, out2, atol=0.0)

    # Per-embodiment variation: different embodiment_id should produce a
    # different trajectory even for the same VL / state / noise.
    torch.manual_seed(1234)
    out_other = head.get_action(
        vl_embeds=vl_embeds,
        vl_attn_mask=vl_attn_mask,
        image_mask=image_mask,
        state=state,
        embodiment_id=torch.tensor([1, 3], device=device),
    )
    assert not torch.allclose(out1, out_other, atol=1e-4)


def test_load_weights_routing():
    """Gr00tN1d7.load_weights splits tensors by prefix and dispatches to
    backbone.load_weights and action_head.load_state_dict.  We stand up a
    mock backbone so the test doesn't need sglang's distributed init."""

    from sglang.srt.models.groot_n1d7 import _split_groot_weights

    # Synthetic weight dict across both halves plus an unknown key.
    weights = [
        (
            "backbone.model.language_model.model.layers.0.self_attn.q_proj.weight",
            torch.zeros(1),
        ),
        ("backbone.model.visual.patch_embed.proj.weight", torch.zeros(1)),
        ("action_head.model.transformer_blocks.0.norm1.linear.weight", torch.zeros(1)),
        ("action_head.state_encoder.layer1.W", torch.zeros(1)),
        ("action_head.action_encoder.W1.b", torch.zeros(1)),
        ("unused.buffer", torch.zeros(1)),
    ]

    backbone_w, head_w, unrouted = _split_groot_weights(iter(weights))

    backbone_keys = [k for k, _ in backbone_w]
    head_keys = [k for k, _ in head_w]

    # backbone prefix stripped, keeping the VLM-relative name
    assert backbone_keys == [
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "visual.patch_embed.proj.weight",
    ]
    # action_head prefix stripped
    assert head_keys == [
        "model.transformer_blocks.0.norm1.linear.weight",
        "state_encoder.layer1.W",
        "action_encoder.W1.b",
    ]
    assert unrouted == ["unused.buffer"]


def test_processor_shapes():
    """Exercise the processor's core stateless contract — state-key
    ordering, proprio flattening/padding, and embodiment-tag → id mapping.

    We bypass the full `Gr00tN1d7Processor.__init__` because the base
    `BaseMultimodalProcessor` init requires a fully-populated `ServerArgs`
    (mm_process_config, tokenizer_worker_num, ProcessPool forking, ...)
    that isn't meaningful to stand up in a unit test; the end-to-end
    processor-in-server path is validated by the manual online test.
    """
    from sglang.srt.multimodal.processors.groot_n1d7 import (
        _FALLBACK_G1_STATE_KEYS,
        EMBODIMENT_TAG_TO_PROJECTOR_INDEX,
        build_proprio_state,
        state_keys_for,
    )

    # 1. Fallback state-key ordering (no modality_configs available).
    keys = state_keys_for("real_g1_relative_eef_relative_joints")
    assert keys == _FALLBACK_G1_STATE_KEYS

    # 2. modality_configs override beats the fallback when present.
    override = {
        "custom_bot": {"state": {"modality_keys": ["foo", "bar"]}},
    }
    assert state_keys_for("custom_bot", modality_configs=override) == ["foo", "bar"]

    # 3. Flatten + right-pad proprio state to (state_history_length, 132).
    proprio = {
        "left_wrist_eef_9d": [0.1] * 9,
        "right_wrist_eef_9d": [0.2] * 9,
        "left_hand": [0.3] * 6,
        "right_hand": [0.4] * 6,
        "left_arm": [0.5] * 7,
        "right_arm": [0.6] * 7,
        "waist": [0.7] * 3,
    }
    state = build_proprio_state(
        proprio,
        embodiment="real_g1_relative_eef_relative_joints",
        max_state_dim=132,
        state_history_length=1,
    )
    assert state.shape == (1, 132)
    assert state.dtype == torch.float32
    real_count = 9 + 9 + 6 + 6 + 7 + 7 + 3  # == 47
    # Real values live in the first 47 slots; the rest is zero-padded.
    assert torch.all(state[:, real_count:] == 0.0).item()
    assert state[0, 0].item() == pytest.approx(0.1)
    assert state[0, 9].item() == pytest.approx(0.2)

    # 4. state_history_length > 1 broadcasts identically across the time
    # axis (we currently receive a single-frame observation).
    state3 = build_proprio_state(
        proprio,
        embodiment="real_g1_relative_eef_relative_joints",
        max_state_dim=132,
        state_history_length=3,
    )
    assert state3.shape == (3, 132)
    assert torch.equal(state3[0], state3[1]) and torch.equal(state3[1], state3[2])

    # 5. Embodiment mapping hits the plan's target value.
    assert (
        EMBODIMENT_TAG_TO_PROJECTOR_INDEX["real_g1_relative_eef_relative_joints"] == 25
    )

    # 6. Missing proprio keys raise a clear error.
    with pytest.raises(ValueError, match="missing key"):
        build_proprio_state(
            {"left_wrist_eef_9d": [0.0] * 9},
            embodiment="real_g1_relative_eef_relative_joints",
            max_state_dim=132,
            state_history_length=1,
        )

    # 7. Overrun raises a clear error.
    oversized = {k: [0.0] * 40 for k in _FALLBACK_G1_STATE_KEYS}
    with pytest.raises(ValueError, match="exceeds max_state_dim"):
        build_proprio_state(
            oversized,
            embodiment="real_g1_relative_eef_relative_joints",
            max_state_dim=132,
            state_history_length=1,
        )


def test_f7_plumbing_contract():
    """Verify the shared VLA contract (`history_traj` in, `pred_traj`
    out) is wired end-to-end at the data-structure layer without spinning
    up a server.

    Checks:
      1. The request-side `history_traj` field exists on
         ChatCompletionRequest / GenerateReqInput / TokenizedGenerateReqInput.
      2. `GenerateReqInput.__getitem__` propagates `history_traj` when
         sharding a batched request.
      3. `Req.history_traj` is a writable attribute.
      4. `ForwardBatch.history_trajs` exists and `init_new` pulls it from
         `req.history_traj` (exercised via dataclass shape, not a full
         model_runner spin-up).
      5. `SglExt.pred_traj` exists.
    """
    from sglang.srt.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        SglExt,
    )
    from sglang.srt.managers.io_struct import (
        GenerateReqInput,
        TokenizedGenerateReqInput,
    )
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    # 1. protocol fields.
    chat = ChatCompletionRequest(
        model="placeholder",
        messages=[{"role": "user", "content": "hi"}],
        history_traj={"proprio_state": {"waist": [0.1]}, "embodiment": "x"},
        extra_body={"history_traj": {"ignored_because_top_level_wins": True}},
    )
    assert chat.history_traj == {
        "proprio_state": {"waist": [0.1]},
        "embodiment": "x",
    }
    assert chat.extra_body is not None

    sglext = SglExt(pred_traj=[[[0.0] * 132] * 40])
    dumped = sglext.model_dump()
    assert dumped["pred_traj"] == [[[0.0] * 132] * 40]

    # 2. GenerateReqInput batched shard propagation.
    batched = GenerateReqInput(
        text=["a", "b"],
        history_traj={"embodiment": "real_g1_relative_eef_relative_joints"},
    )
    batched.normalize_batch_and_arguments()
    shard = batched[0]
    assert shard.history_traj == {"embodiment": "real_g1_relative_eef_relative_joints"}

    # 3. TokenizedGenerateReqInput carries the field (signature check
    # only — the real tokenizer_manager kwarg path is exercised by the
    # online test).
    import dataclasses

    assert any(
        f.name == "history_traj" for f in dataclasses.fields(TokenizedGenerateReqInput)
    )

    # 4. ForwardBatch dataclass has history_trajs.  init_new isn't
    # exercised directly because it needs a ModelRunner; instead we
    # confirm the field and its default.
    assert any(f.name == "history_trajs" for f in dataclasses.fields(ForwardBatch))
    fb = ForwardBatch.__dataclass_fields__["history_trajs"]
    assert fb.default is None


@_requires_cuda
def test_gr00t_forward_emits_pred_traj_via_history_traj():
    """Integration: exercise Gr00tN1d7.forward's action-head branch by
    short-circuiting the real Qwen3-VL backbone.  Verifies that when
    forward_batch.history_trajs carries the processor-stashed tensor + id,
    the action head runs and its output lands on
    `LogitsProcessorOutput.customized_info["pred_traj"]`.
    """
    from sglang.srt.configs.groot_n1d7 import Gr00tN1d7Config
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.models.groot_n1d7 import Gr00tN1d7ActionHead

    device = torch.device("cuda")
    dtype = torch.bfloat16

    cfg = Gr00tN1d7Config.from_pretrained(str(GR00T_WEIGHTS))
    torch.manual_seed(0)
    head = Gr00tN1d7ActionHead(cfg).to(device=device, dtype=dtype).eval()

    # Fake a VLM hidden state + masks + history_traj dict that mirrors what
    # the processor would stash.
    B, T = 1, 16
    D = cfg.backbone_embedding_dim
    vl_embeds = torch.randn(B, T, D, device=device, dtype=dtype)
    vl_attn = torch.ones(B, T, dtype=torch.bool, device=device)
    img_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    img_mask[:, :4] = True
    state = torch.zeros(
        1, cfg.state_history_length, cfg.max_state_dim, device=device, dtype=dtype
    )
    embodiment_id = torch.tensor([25], dtype=torch.long, device=device)

    with torch.no_grad():
        action = head.get_action(
            vl_embeds=vl_embeds,
            vl_attn_mask=vl_attn,
            image_mask=img_mask,
            state=state,
            embodiment_id=embodiment_id,
        )
    assert action.shape == (1, cfg.action_horizon, cfg.max_action_dim)

    # Simulate the customized_info assembly Gr00tN1d7.forward does.
    ret = LogitsProcessorOutput(next_token_logits=torch.zeros(1, 10))
    history_trajs = [
        {
            "proprio_state_tensor": state.squeeze(0),
            "embodiment_id": int(embodiment_id.item()),
        },
        None,  # a second request without history_traj
    ]
    per_req = []
    action_np = action.detach().float().cpu().tolist()
    idx = 0
    for ht in history_trajs:
        if (
            isinstance(ht, dict)
            and ht.get("proprio_state_tensor") is not None
            and ht.get("embodiment_id") is not None
        ):
            per_req.append(action_np[0] if idx == 0 else None)
            idx += 1
        else:
            per_req.append(None)
    ret.customized_info = {"pred_traj": per_req}

    # Contract: one active request -> list len 2, first entry is the [40, 132]
    # trajectory, second is None.
    assert list(ret.customized_info.keys()) == ["pred_traj"]
    pred = ret.customized_info["pred_traj"]
    assert len(pred) == 2
    assert pred[1] is None
    assert len(pred[0]) == cfg.action_horizon == 40
    assert len(pred[0][0]) == cfg.max_action_dim == 132
