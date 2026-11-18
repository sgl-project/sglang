# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from sglang.multimodal_gen.configs.models.dits.praxis import (
    PraxisArchConfig,
    PraxisDiTConfig,
)
from sglang.multimodal_gen.runtime.models.dits import praxis as layout
from sglang.multimodal_gen.runtime.models.dits import praxis_layers as layers
from sglang.multimodal_gen.runtime.models.dits.praxis import PraxisTransformer

CPU = torch.device("cpu")


def _tiny_arch(**overrides) -> PraxisArchConfig:
    values = dict(
        dim=64,
        timestep_dim=16,
        num_registers=2,
        head_dim=16,
        num_kv_heads=2,
        dense_ffn_hidden=32,
        num_routed_experts=4,
        top_k=2,
        expert_hidden=8,
        rope_effective_dim=8,
        text_dim=8,
        latent_channels=8,
        encoder_depth=2,
        encoder_dense_blocks=[0, 1],
        tap_blocks=[0, 1],
        decoder_depth=2,
        decoder_dense_blocks=[0, 1],
        decoder_channels=2,
        expert_dim=32,
        expert_layers=4,
        expert_heads=2,
        expert_head_dim=16,
        expert_trunk_dim=64,
        expert_rope_dim=16,
        expert_rope_effective_dim=8,
        action_dim=4,
        proprio_dim=3,
        num_future_per_chunk=2,
        max_chunks=4,
    )
    values.update(overrides)
    return PraxisArchConfig(**values)


def _tiny_model(**overrides) -> PraxisTransformer:
    torch.manual_seed(0)
    model = PraxisTransformer(PraxisDiTConfig(arch_config=_tiny_arch(**overrides)))
    for param in model.parameters():
        nn.init.normal_(param, std=0.2)
    for name, buffer in model.named_buffers():
        if name.endswith(("freqs", "phases")):
            nn.init.normal_(buffer)
    return model.eval()


def test_window_mask_lets_a_chunk_read_only_its_past():
    """Noised chunk k reads itself and the clean copies before it; a leak of a later chunk breaks causality."""
    # [global x1 | noised: chunk 0, chunk 1 (1 token each) | clean: chunk 0 (1 token)]
    kinds, chunks = layout.token_layout(
        num_global=1, frames=2, tokens_per_frame=1, chunk_frames=1,
        num_clean_frames=1, device=CPU,
    )  # fmt: skip
    expected = torch.tensor(
        [
            [1, 0, 0, 0],  # global reads only global
            [1, 1, 0, 0],  # noised 0: itself, no clean before it
            [1, 0, 1, 1],  # noised 1: itself and clean 0
            [1, 0, 0, 1],  # clean 0: itself, never a noised token
        ],
        dtype=torch.bool,
    )
    assert torch.equal(layout.window_mask(kinds=kinds, chunks=chunks), expected)


def test_expert_cross_mask_reads_clean_frames_before_its_chunk():
    # 2 chunks of 2 frames, 1 token per frame, 1 global token: KV is [global | clean 0 1 | noised 0 1 2 3].
    query_chunks = torch.tensor([0, 1])
    mask = layers.expert_cross_mask(
        query_chunks=query_chunks, chunk_size=2, num_global=1,
        num_clean_frames=2, num_noised_frames=4, tokens_per_frame=1,
        include_noised=True,
    )  # fmt: skip
    assert mask[0].tolist() == [1, 0, 0, 1, 1, 0, 0]
    assert mask[1].tolist() == [1, 1, 1, 0, 0, 1, 1]
    clean_only = layers.expert_cross_mask(
        query_chunks=query_chunks, chunk_size=2, num_global=1,
        num_clean_frames=2, num_noised_frames=4, tokens_per_frame=1,
        include_noised=False,
    )  # fmt: skip
    assert clean_only[1].tolist() == [1, 1, 1, 0, 0, 0, 0]


def test_encoder_positions_put_text_before_video_and_clean_on_noised_ids():
    ids = layout.encoder_positions(
        num_registers=1, anchor_hw=(1, 1), num_text=2, frames=2, height=1, width=2,
        num_clean_frames=1, device=CPU,
    )[0].tolist()  # fmt: skip
    stride = layout.FRAME_STRIDE
    assert ids == [
        0,  # register
        1,  # anchor: frame block 0
        stride + 1, stride + 2,  # text: frame block 1
        2 * stride + 1, 2 * stride + 2,  # noised frame 0 (row 0, columns 0 and 1)
        3 * stride + 1, 3 * stride + 2,  # noised frame 1
        2 * stride + 1, 2 * stride + 2,  # the clean copy repeats the ids of noised frame 0
    ]  # fmt: skip


def test_state_dict_keys_are_the_checkpoint_keys():
    """The checkpoint loads without a name mapping; a renamed module would break the strict load."""
    keys = set(
        _tiny_model(encoder_dense_blocks=[0], decoder_dense_blocks=[0]).state_dict()
    )
    for key in (
        "backbone.encoder.registers",
        "backbone.encoder.t_embedding.freqs",
        "backbone.encoder.t_proj.VideoTensor.weight",
        "backbone.encoder.input_projections.TextEncodingTensor.proj.3.weight",
        "backbone.encoder.input_projections.VideoTensor.proj.bias",
        "backbone.encoder.core.blocks.0.s",
        "backbone.encoder.core.blocks.0.mha.in_proj.q_weight",
        "backbone.encoder.core.blocks.0.mha.gate_proj.weight",
        "backbone.encoder.core.blocks.0.ffn.0.linear.a_weight",
        "backbone.encoder.core.blocks.0.ffn.1.weight",
        "backbone.encoder.core.blocks.1.ffn.expert_bias",
        "backbone.encoder.core.blocks.1.ffn.router.weight",
        "backbone.encoder.core.blocks.1.ffn.routed.gate_up_weight",
        "backbone.encoder.core.blocks.1.ffn.shared.down",
        "backbone.encoder.output_projections.VideoTensor.proj.weight",
        "backbone.decoder.core.modulators.0.weight",
        "backbone.decoder.output_projections.VideoTensor.proj.1.bias",
        "action_expert.action_encoder.tau.mlp.2.weight",
        "action_expert.blocks.0.to_kv.weight",
        "action_expert.blocks.1.adaln_mlp.proj.bias",
        "action_expert.future_chunk_embed.weight",
        "action_expert.tap_norm_scale",
        "action_expert.head.bias",
    ):
        assert key in keys, key
    assert not any("rope" in key for key in keys)


def test_layerwise_offload_names_the_block_stacks():
    """Layerwise offload streams the stacks named in ``layer_names``; a rename would silently disable it."""
    model = _tiny_model()
    modules = dict(model.named_modules())
    depths = {
        "backbone.encoder.core.blocks": model.arch.encoder_depth,
        "backbone.decoder.core.blocks": model.arch.decoder_depth,
        "action_expert.blocks": model.arch.expert_layers,
    }
    assert sorted(model.layer_names) == sorted(depths)
    for name, depth in depths.items():
        assert isinstance(modules[name], nn.ModuleList) and len(modules[name]) == depth


def test_routing_bias_stays_float32_under_a_dtype_cast():
    """The checkpoint stores the routing bias in float32; a bf16 cast of the model must not round it."""
    model = _tiny_model(encoder_dense_blocks=[0]).to(torch.bfloat16)
    block = model.backbone.encoder.core.blocks[1]
    assert block.ffn.expert_bias.dtype == torch.float32
    assert block.ffn.router.weight.dtype == torch.bfloat16


def _window_inputs(*, chunks: int, seed: int = 0):
    """A window of ``chunks`` chunks of 2 latent frames on a 2 x 2 grid (decoder grid 4 x 4)."""
    g = torch.Generator().manual_seed(seed)
    frames = 2 * chunks
    return dict(
        hidden_states=torch.randn(1, 8, frames, 2, 2, generator=g),
        encoder_hidden_states=torch.randn(1, 3, 8, generator=g),
        timestep=torch.full((1, chunks), 0.5),
        anchor=torch.randn(1, 8, 1, 2, 2, generator=g),
        clean=torch.randn(1, 8, frames - 2, 2, 2, generator=g),
        noised_actions=torch.randn(1, frames, 4, generator=g),
        action_timestep=torch.full((1, chunks), 0.5),
        proprio=torch.randn(1, chunks, 3, generator=g),
    )


def test_window_velocity_of_a_chunk_ignores_later_noised_chunks():
    """Video and action velocity of chunk 0 must not read the noised state of chunk 1."""
    model = _tiny_model()
    inputs = _window_inputs(chunks=2)
    video, action = model(**inputs)
    assert video.shape == inputs["hidden_states"].shape
    assert action.shape == inputs["noised_actions"].shape
    perturbed = dict(inputs)
    perturbed["hidden_states"] = inputs["hidden_states"].clone()
    perturbed["hidden_states"][:, :, 2:] += 1.0
    perturbed["noised_actions"] = inputs["noised_actions"].clone()
    perturbed["noised_actions"][:, 2:] += 1.0
    video2, action2 = model(**perturbed)
    assert torch.equal(video[:, :, :2], video2[:, :, :2])
    assert torch.equal(action[:, :2], action2[:, :2])
    assert not torch.equal(video[:, :, 2:], video2[:, :, 2:])
    assert not torch.equal(action[:, 2:], action2[:, 2:])


@pytest.mark.parametrize("decode, act", [(False, True), (True, False)])
def test_a_skipped_part_is_none_and_the_other_is_unchanged(decode, act):
    """The unguided pass skips the video decoder or the expert; the part it keeps must not change."""
    model = _tiny_model()
    inputs = _window_inputs(chunks=2)
    full_video, full_action = model(**inputs)
    video, action = model(**inputs, decode=decode, act=act)
    assert (video is None) == (not decode) and (action is None) == (not act)
    kept, full = (action, full_action) if act else (video, full_video)
    assert torch.equal(kept, full)
