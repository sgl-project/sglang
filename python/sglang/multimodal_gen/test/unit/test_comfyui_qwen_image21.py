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
from sglang.multimodal_gen.runtime.pipelines_core.comfyui_mode import (
    bind_comfyui_session,
    get_run_state,
    release_comfyui_session,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    qwen_image21 as stage_mod,
)

# hidden_size 4, ffn 12.
_DIT_CONFIG = QwenImage21DitConfig(
    arch_config=QwenImage21ArchConfig(
        num_layers=2, num_attention_heads=1, attention_head_dim=4
    )
)


def test_checkpoint_spec_splits_fused_gate_up_and_int8_scales() -> None:
    mlp, ffn = "transformer_blocks.0.img_mlp", 12
    gate_up, scale = torch.randn(2 * ffn, 4), torch.randn(2 * ffn, 1)
    weights = [
        (f"{mlp}.gate_up.weight", gate_up),
        (f"{mlp}.gate_up.weight_scale", scale),
        (f"{mlp}.gate_up.comfy_quant", torch.zeros(3)),
        ("img_in.weight", torch.ones(1)),
    ]
    spec = get_comfyui_checkpoint_spec("QwenImage21Pipeline")
    out = dict(spec.convert_weights(iter(weights), _DIT_CONFIG))
    assert len(out) == 5 and "img_in.weight" in out
    # ComfyUI's swiglu is silu(first half) * second half.
    for param, fused in (("weight", gate_up), ("weight_scale", scale)):
        assert torch.equal(out[f"{mlp}.gate_layer.{param}"], fused[:ffn])
        assert torch.equal(out[f"{mlp}.proj.{param}"], fused[ffn:])


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
    gate = markers.pop("transformer_blocks.0.img_mlp.gate_layer")
    proj = markers.pop("transformer_blocks.0.img_mlp.proj")
    assert gate == proj == marker and gate is not proj
    assert set(markers) == {"transformer_blocks.0.attn.to_q"}


class _RecordingExecutor(QwenImage21Executor):
    """Real forward and cond-cache bookkeeping; the worker echoes the latents."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.adapter = QwenImage21Adapter()
        self._sent_conds, self._cond_key_cache, self.sent = set(), {}, []

    def _execute_packed(self, packed, x, timestep):
        self._mark_and_maybe_drop(packed)
        req = SimpleNamespace(extra={})
        self.adapter.fill_req(req, packed)
        self.sent.append((packed, req))
        return self.adapter.unpack(x.flatten(2).transpose(1, 2), packed, x)


def test_reference_latents_reach_the_worker_and_key_the_cond_cache() -> None:
    """Edit workflows pass ref_latents: the executor must forward them, and two
    conds that differ only in their reference must not share a cached cond."""
    assert get_adapter_class("qwen_image21") is QwenImage21Adapter
    assert stage_mod.COMFYUI_COND_EXTRA_KEY == COND_EXTRA_KEY
    ex = _RecordingExecutor()
    x, context = torch.randn(1, 64, 5, 7), torch.ones(1, 9, 32)
    options = {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0])}
    ref_a, ref_b = torch.zeros(1, 64, 4, 4), torch.ones(1, 64, 4, 4)
    kwargs = dict(image_slots=[3], transformer_options=options)
    for ref in (ref_a, ref_b, ref_a):
        out = ex(x, torch.tensor([0.5]), context, ref_latents=[ref], **kwargs)
        assert torch.equal(out, x)
    (packed, req), (_, req_b), (_, req_a) = ex.sent
    assert torch.equal(req.timesteps, torch.tensor([500.0]))
    assert (packed.height, packed.width) == (80, 112)
    payload = req.extra[COND_EXTRA_KEY]
    assert payload["image_slots"] == [3] and payload["final_sigma"] == 0.5
    assert payload["ref_latents"][0] is ref_a
    assert req_b.extra[COND_EXTRA_KEY]["ref_latents"][0] is ref_b
    # An identical cond is restored by the worker.
    assert COND_EXTRA_KEY not in req_a.extra and req_a.prompt_embeds == []


def _run(monkeypatch, sid, key, sigma, prompt_embeds, payload, fits=True):
    """Condition stage after ComfyUILatentPreparationStage restored the session."""
    monkeypatch.setattr(stage_mod, "_cache_fits", lambda need, device: fits)
    req = SimpleNamespace(
        extra={
            "comfyui_session_id": sid,
            "comfyui_cond_key": key,
            COND_EXTRA_KEY: payload,
        },
        latents=torch.ones(1, 64, 2, 3),
        timesteps=torch.tensor([sigma * 1000.0]),
        prompt_embeds=prompt_embeds,
    )
    bind_comfyui_session(req)
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_config=_DIT_CONFIG)
    )
    return stage_mod.QwenImage21ComfyUIConditionStage().forward(req, server_args)


def test_cond_is_built_once_per_run_and_freed_after_the_final_sigma(
    monkeypatch,
) -> None:
    """Each cond's prefix K/V cache is reused across steps, and the cache (GiBs)
    must not outlive the sampler's last call."""
    ref = torch.randn(1, 64, 2, 2)
    # Per cond: context, image_slots, ref_latents, room for the cache.
    conds = {
        "pos": (torch.randn(1, 5, 16), [], [], True),
        "neg": (torch.randn(1, 7, 16), [1], [ref], False),
    }
    built = {}
    try:
        for step, sigma in enumerate((1.0, 0.5, 0.25)):
            for key, (ctx, slots, refs, fits) in conds.items():
                payload = dict(image_slots=slots, ref_latents=refs, final_sigma=0.25)
                sent = ([ctx], payload) if step == 0 else ([], None)
                req = _run(monkeypatch, "r:1", key, sigma, *sent, fits=fits)
                assert req.latents.shape == (1, 6, 64)
                assert req.prompt_embeds[0].shape[1] == ctx.shape[1] + len(refs)
                cond = req.extra["qwen21_positive"]
                assert built.setdefault(key, cond) is cond
                # No room: the prefix is recomputed every step.
                assert (cond["prefix_caches"] is None) == (not fits)
            assert len(get_run_state(req)) == (0 if sigma == 0.25 else 2)
    finally:
        release_comfyui_session("r:1")
    assert built["pos"]["condition_latents"] is None
    expected = ref.flatten(2).transpose(1, 2).bfloat16()
    assert torch.equal(built["neg"]["condition_latents"], expected)
    assert built["neg"]["layouts"][0]["image_indices"].tolist() == [1, 2, 3, 4]


def test_cache_decision_is_agreed_across_tp_ranks(monkeypatch) -> None:
    """Under TP an uncached prefix runs extra all-reduces, so ranks must not split."""
    seen = []

    def all_reduce(tensor, op=None, group=None):
        seen.append(group)
        tensor.fill_(10)

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


def test_comfyui_stages_and_cfg_parallel_rejection(monkeypatch) -> None:
    from sglang.multimodal_gen.runtime.pipelines.qwen_image21 import QwenImage21Pipeline
    from sglang.multimodal_gen.runtime.pipelines_core.stages import (
        ComfyUILatentPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
        QwenImage21DenoisingStage,
    )

    monkeypatch.setattr(QwenImage21DenoisingStage, "__init__", lambda *args: None)
    pipe = SimpleNamespace(get_module=lambda name: object(), stages=[])
    pipe.add_stages = pipe.stages.extend
    create = QwenImage21Pipeline.create_comfyui_stages
    create(pipe, SimpleNamespace(enable_cfg_parallel=False))
    assert [type(s) for s in pipe.stages] == [
        ComfyUILatentPreparationStage,
        stage_mod.QwenImage21ComfyUIConditionStage,
        QwenImage21DenoisingStage,
    ]
    with pytest.raises(ValueError, match="enable_cfg_parallel"):
        create(pipe, SimpleNamespace(enable_cfg_parallel=True))
