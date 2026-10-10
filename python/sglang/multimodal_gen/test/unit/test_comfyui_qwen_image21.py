# SPDX-License-Identifier: Apache-2.0
"""ComfyUI integrated mode for Qwen-Image 2.1 (``qwen_image21``)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    get_adapter_class,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.qwen_image21 import (
    COND_EXTRA_KEY,
    QwenImage21Adapter,
    QwenImage21Executor,
)
from sglang.multimodal_gen.configs.models.dits.qwenimage21 import (
    QwenImage21ArchConfig,
    QwenImage21DitConfig,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import (
    get_comfyui_checkpoint_spec,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import build_layout
from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    bind_comfyui_session,
    get_run_state,
    release_comfyui_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    qwen_image21 as stage_mod,
)

AXES = (16, 56, 56)


def _small_dit_config():
    return QwenImage21DitConfig(
        arch_config=QwenImage21ArchConfig(
            num_layers=2, num_attention_heads=1, attention_head_dim=4
        )
    )


def test_checkpoint_spec_splits_fused_gate_up_and_int8_scales() -> None:
    spec = get_comfyui_checkpoint_spec("QwenImage21Pipeline")
    assert spec.dit_cls_name == "QwenImage21Transformer2DModel"
    ffn = 4 * 3
    gate_up = torch.arange(2 * ffn * 4, dtype=torch.float32).reshape(2 * ffn, 4)
    scale = torch.arange(2 * ffn, dtype=torch.float32)[:, None]
    out = dict(
        spec.convert_weights(
            iter(
                [
                    ("transformer_blocks.0.img_mlp.gate_up.weight", gate_up),
                    ("transformer_blocks.0.img_mlp.gate_up.weight_scale", scale),
                    (
                        "transformer_blocks.0.img_mlp.gate_up.comfy_quant",
                        torch.zeros(3),
                    ),
                    ("img_in.weight", torch.ones(1)),
                ]
            ),
            _small_dit_config(),
        )
    )
    # ComfyUI's swiglu is silu(first half) * second half.
    prefix = "transformer_blocks.0.img_mlp"
    assert torch.equal(out[f"{prefix}.gate_layer.weight"], gate_up[:ffn])
    assert torch.equal(out[f"{prefix}.proj.weight"], gate_up[ffn:])
    assert torch.equal(out[f"{prefix}.gate_layer.weight_scale"], scale[:ffn])
    assert torch.equal(out[f"{prefix}.proj.weight_scale"], scale[ffn:])
    assert set(out) == {
        f"{prefix}.gate_layer.weight",
        f"{prefix}.proj.weight",
        f"{prefix}.gate_layer.weight_scale",
        f"{prefix}.proj.weight_scale",
        "img_in.weight",
    }


def test_int8_marker_of_fused_gate_up_covers_both_halves(monkeypatch) -> None:
    from sglang.multimodal_gen.runtime.utils import quantization_utils

    marker = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    monkeypatch.setattr(
        quantization_utils,
        "inspect_comfy_quant_markers",
        lambda files: {
            "transformer_blocks.0.img_mlp.gate_up": marker,
            "transformer_blocks.0.attn.to_q": marker,
        },
    )
    markers = get_comfyui_checkpoint_spec("QwenImage21Pipeline").quant_markers(["f"])
    assert set(markers) == {
        "transformer_blocks.0.img_mlp.gate_layer",
        "transformer_blocks.0.img_mlp.proj",
        "transformer_blocks.0.attn.to_q",
    }


def test_layout_follows_comfyui_token_order() -> None:
    """Reference latents sit at ComfyUI's image_slots, the target comes last."""
    model_mod = pytest.importorskip("comfy.ldm.qwen_image21.model")
    ops = pytest.importorskip("comfy.ops")
    torch.manual_seed(0)
    model = model_mod.QwenImage21Transformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=128,
        num_attention_heads=1,
        context_in_dim=8,
        mlp_ratio=1,
        dtype=torch.float32,
        operations=ops.disable_weight_init,
    )
    for p in model.parameters():
        torch.nn.init.normal_(p)
    x = torch.randn(1, 4, 4, 6)
    context = torch.randn(1, 11, 8)
    refs = [torch.randn(1, 4, 4, 4), torch.randn(1, 4, 2, 6)]
    slots = [2, 6]
    with torch.no_grad():
        hidden, pe, segments = model.build_sequence(x, context, refs, slots)
        ctx, flags = stage_mod.insert_image_placeholders(context, slots, len(refs))
        layout = build_layout(flags, [(1, 4, 4), (1, 2, 6), (1, 4, 6)], AXES, "cpu")
        prefix = model.txt_in(ctx).index_select(1, layout["text_indices"])
        cond = torch.cat([r.flatten(2).transpose(1, 2) for r in refs], dim=1)
        prefix[:, layout["image_indices"]] = model.img_in(cond)
    prefix_len = layout["prefix_rope"].shape[0]
    torch.testing.assert_close(prefix, hidden[:, :prefix_len], rtol=1e-5, atol=1e-4)
    rope = torch.cat([layout["prefix_rope"], layout["target_rope"]])
    torch.testing.assert_close(rope.real, pe[0, :, 0, :, 0, 0], rtol=0, atol=1e-4)
    torch.testing.assert_close(rope.imag, pe[0, :, 0, :, 1, 0], rtol=0, atol=1e-4)
    assert list(layout["segments"]) == [(s, e, m is None) for s, e, m in segments[:-1]]


def test_placeholders_for_missing_and_trailing_slots() -> None:
    ctx = torch.arange(3.0).view(1, 3, 1)
    out, flags = stage_mod.insert_image_placeholders(ctx, [9], 2)
    assert flags == [False, False, False, True, True]
    assert out.flatten().tolist() == [0, 1, 2, 0, 0]
    with pytest.raises(ValueError, match="ascending"):
        stage_mod.insert_image_placeholders(ctx, [2, 1], 2)


def test_adapter_registration_and_payload() -> None:
    assert get_adapter_class("qwen_image21") is QwenImage21Adapter
    assert stage_mod.COMFYUI_COND_EXTRA_KEY == COND_EXTRA_KEY
    adapter = QwenImage21Adapter()
    x = torch.randn(2, 64, 5, 7)
    ref = torch.randn(2, 64, 4, 4)
    packed = adapter.pack(
        x,
        torch.tensor([0.5, 0.5]),
        torch.randn(2, 9, 32),
        ref_latents=[ref],
        image_slots=[3],
    )
    assert packed.latents is x
    assert torch.equal(packed.timesteps, torch.tensor([500.0, 500.0]))
    assert (packed.height, packed.width) == (80, 112)
    noise = x.flatten(2).transpose(1, 2).contiguous()
    assert torch.equal(adapter.unpack(noise, packed, x), x)
    req = SimpleNamespace(extra={"comfyui_session_id": "s:1"})
    adapter.fill_req(req, packed)
    assert req.extra[COND_EXTRA_KEY] == {"image_slots": [3], "ref_latents": [ref]}
    assert req.extra["comfyui_session_id"] == "s:1"
    adapter.drop_cached_fields(packed)
    assert packed.prompt_embeds == [] and COND_EXTRA_KEY not in packed.extra_req
    with pytest.raises(NotImplementedError):
        adapter.pack(
            x,
            torch.tensor([1.0]),
            torch.randn(1, 3, 8),
            transformer_options={"patches_replace": {"dit": {("single_block", 0): 1}}},
        )


def test_cfg_parallel_is_rejected() -> None:
    """ComfyUI owns CFG, so a CFG rank would only recompute the same DiT call."""
    with pytest.raises(ValueError, match="does not apply"):
        QwenImage21Executor.validate_sgld_options({"enable_cfg_parallel": True})
    QwenImage21Executor.validate_sgld_options({"num_gpus": 2, "sp_degree": 2})


# ----- worker condition stage ---------------------------------------------------


def _run(monkeypatch, req, fits=True):
    """Condition stage after ComfyUILatentPreparationStage restored the session."""
    monkeypatch.setattr(stage_mod, "_cache_fits", lambda need, device: fits)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_config=_small_dit_config())
    )
    bind_comfyui_session(req)
    return stage_mod.QwenImage21ComfyUIConditionStage().forward(req, server_args)


def _req(sid, key, latents, sigma, context=None, payload=None):
    extra = {"comfyui_session_id": sid, "comfyui_cond_key": key}
    if payload is not None:
        extra[COND_EXTRA_KEY] = payload
    return SimpleNamespace(
        extra=extra,
        latents=latents,
        timesteps=torch.full((latents.shape[0],), sigma * 1000.0),
        prompt_embeds=[] if context is None else [context],
    )


def test_condition_stage_is_built_once_per_cond_for_the_run(monkeypatch) -> None:
    x = torch.randn(1, 64, 3, 4)
    pos, neg = torch.randn(1, 5, 16), torch.randn(1, 7, 16)
    payload = {"image_slots": [], "ref_latents": []}
    built = {}
    try:
        for sigma in (1.0, 0.5):
            first = sigma == 1.0
            for key, ctx in (("pos", pos), ("neg", neg)):
                req = _run(
                    monkeypatch,
                    _req(
                        "exec:1",
                        key,
                        x,
                        sigma,
                        ctx if first else None,
                        payload if first else None,
                    ),
                )
                assert torch.equal(req.latents, x.flatten(2).transpose(1, 2))
                cond = req.extra["qwen21_positive"]
                assert req.prompt_embeds[0].shape[1] == ctx.shape[1]
                assert cond["condition_latents"] is None
                assert built.setdefault(key, cond) is cond
        state = get_run_state(SimpleNamespace(extra={"comfyui_session_id": "exec:1"}))
        assert len(state) == 2
        # The next sampler run of this executor evicts the old run's caches.
        _run(monkeypatch, _req("exec:2", "pos", x, 1.0, pos, payload))
        assert (
            get_run_state(SimpleNamespace(extra={"comfyui_session_id": "exec:1"}))
            is None
        )
    finally:
        release_comfyui_session("exec:1")
        release_comfyui_session("exec:2")


def test_condition_stage_batch_rows_refs_and_no_room(monkeypatch) -> None:
    x = torch.randn(2, 64, 2, 2)
    ref = torch.randn(1, 64, 2, 2)
    payload = {"image_slots": [1], "ref_latents": [ref]}
    try:
        req = _run(
            monkeypatch,
            _req("b:1", "k", x, 1.0, torch.randn(2, 3, 16), payload),
            fits=False,
        )
        cond = req.extra["qwen21_positive"]
        assert cond["prefix_caches"] is None  # no room: recompute every step
        assert len(cond["layouts"]) == 2
        assert cond["layouts"][0]["image_indices"].tolist() == [1, 2, 3, 4]
        assert cond["condition_latents"].shape == (2, 4, 64)
        assert req.prompt_embeds[0].shape == (2, 4, 16)  # + 1 placeholder
        assert req.latents.shape == (2, 4, 64)
        assert req.timesteps.tolist() == [1000.0]  # one loop step for both rows
    finally:
        release_comfyui_session("b:1")


def test_cache_decision_is_agreed_across_tp_ranks(monkeypatch) -> None:
    """Under TP an uncached prefix runs extra all-reduces, so ranks must not split."""
    seen = []

    def all_reduce(tensor, op=None, group=None):
        seen.append(group)
        tensor.fill_(10)

    monkeypatch.setattr(stage_mod, "model_parallel_is_initialized", lambda: True)
    monkeypatch.setattr(
        stage_mod, "get_tp_group", lambda: SimpleNamespace(world_size=2, cpu_group="tp")
    )
    monkeypatch.setattr(
        stage_mod, "get_sp_group", lambda: SimpleNamespace(world_size=1, cpu_group="sp")
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    assert stage_mod._cache_fits(4, torch.device("cpu")) is True
    assert stage_mod._cache_fits(5, torch.device("cpu")) is False
    assert seen == ["tp", "tp"]


def test_pipeline_uses_generic_comfyui_stages(monkeypatch) -> None:
    from sglang.multimodal_gen.runtime.pipelines.qwen_image21 import QwenImage21Pipeline
    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        ComfyUILatentPreparationStage,
    )

    created = {}

    def _fake_init(self, transformer, scheduler):
        created["modules"] = (transformer, scheduler)

    monkeypatch.setattr(stage_mod.QwenImage21DenoisingStage, "__init__", _fake_init)
    pipe = SimpleNamespace(
        modules={"transformer": object(), "scheduler": object()}, stages=[]
    )
    pipe.get_module = pipe.modules.__getitem__
    pipe.add_stages = pipe.stages.extend
    QwenImage21Pipeline.create_comfyui_stages(
        pipe, SimpleNamespace(enable_cfg_parallel=False)
    )
    assert [type(s) for s in pipe.stages] == [
        ComfyUILatentPreparationStage,
        stage_mod.QwenImage21ComfyUIConditionStage,
        stage_mod.QwenImage21DenoisingStage,
    ]
    assert created["modules"] == (
        pipe.modules["transformer"],
        pipe.modules["scheduler"],
    )
    with pytest.raises(ValueError, match="enable_cfg_parallel"):
        QwenImage21Pipeline.create_comfyui_stages(
            pipe, SimpleNamespace(enable_cfg_parallel=True)
        )
