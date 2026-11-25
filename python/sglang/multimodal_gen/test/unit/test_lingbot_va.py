# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_va import (
    LingBotVALiberoConfig,
)
from sglang.multimodal_gen.runtime.models.dits.lingbot_va import LingBotVATransformer
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_va import (
    LingBotVAStage,
    LingBotVAState,
    PreparedChunk,
)


@torch.inference_mode()
def test_grouped_flow_preserves_row_ownership_and_uses_the_prefix():
    config = LingBotVALiberoConfig(
        latent_shape=(2, 2),
        frame_chunk_size=2,
        action_per_frame=2,
        action_horizon=4,
        video_steps=1,
    )
    torch.manual_seed(42)
    transformer = LingBotVATransformer(
        num_attention_heads=1,
        attention_head_dim=12,
        text_dim=8,
        freq_dim=8,
        ffn_dim=24,
        num_layers=1,
    )
    stage = LingBotVAStage(
        {
            "transformer": transformer,
            "vae": Mock(),
            "tokenizer": None,
            "text_encoder": None,
        },
        config,
    )

    def chunk(seed, prefix=None):
        rng = torch.Generator().manual_seed(seed)

        def noise(*shape):
            return torch.randn(shape, generator=rng)

        request = SimpleNamespace(
            num_inference_steps=2, extra={"vla": {}, "action_realtime": {}}
        )
        if prefix is not None:
            request.extra["action_realtime"]["action_prefix"] = prefix.T.numpy()
        return PreparedChunk(
            request=request,
            state=LingBotVAState(),
            prompt=str(seed),
            frame=0,
            text=noise(2, 4, 8),
            anchor=noise(1, 48, 1, 2, 2),
            history=[],
            cache=[],
            visual=noise(1, 48, 2, 2, 2),
            actions=noise(1, 30, 2, 2, 1),
            prefix=prefix,
        )

    left, right = chunk(1), chunk(2)
    stage.denoise([left, right], check_deadline=False)

    def output(row):
        return row.request.extra["vla"]["actions"]

    for row, seed in ((left, 1), (right, 2)):
        single = chunk(seed)
        stage.denoise([single])
        torch.testing.assert_close(output(row), output(single), atol=1e-5, rtol=1e-5)
        assert row.request.extra["vla"]["extras"]["policy_batch_size"] == 2

    prefix = torch.ones(7, 1)
    conditioned = chunk(1, prefix)
    prefix_inputs = []
    noise = conditioned.actions[0, :7, 1, 0, 0].clone()

    def observe_action_input(module, args, kwargs):
        if kwargs.get("action"):
            latents, timestep = args[:2]
            prefix_inputs.append(
                (latents[0, :7, 1, 0, 0].clone(), timestep[0, 1] / 1000)
            )

    with transformer.register_forward_pre_hook(observe_action_input, with_kwargs=True):
        stage.denoise([conditioned])
    torch.testing.assert_close(output(conditioned)[0, :1], prefix.T)
    assert len(prefix_inputs) == 2
    midway, sigma = prefix_inputs[1]
    assert 0 < sigma < 1
    torch.testing.assert_close(midway, prefix[:, 0] * (1 - sigma) + noise * sigma)
