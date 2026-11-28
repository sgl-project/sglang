"""The srt-backed Qwen3-VL text model against the in-tree reference stack.

``test_two_ranks`` torchruns this file for the tensor-parallel case.
"""

import os
import subprocess
import sys

import pytest
import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_tp_group,
    maybe_init_distributed_environment_and_model_parallel,
    use_tensor_parallel_group,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.encoders import qwen3vl as reference
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl_text import (
    Qwen3VLTextModel,
)

_TWO_RANKS = int(os.environ.get("WORLD_SIZE", "1")) == 2
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
single_rank = pytest.mark.skipif(_TWO_RANKS, reason="single-rank case")
two_ranks = pytest.mark.skipif(not _TWO_RANKS, reason="launched by test_two_ranks")


def _text_config(num_layers=3):
    return Qwen3VLTextConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 5_000_000.0,
            "mrope_section": [4, 6, 6],
            "mrope_interleaved": True,
        },
    )


def _relative_error(actual, expected):
    actual, expected = actual.float(), expected.float()
    return ((actual - expected).norm() / expected.norm()).item()


def _models(config, deepstack_layers):
    """The replicated reference stack, and the srt-backed model built in the
    current TP group and loaded from the reference's checkpoint names."""
    torch.manual_seed(0)
    ref = reference.Qwen3VLTextModel(config).cuda().to(torch.bfloat16)
    for param in ref.parameters():
        param.data.normal_(0, 0.05)
    model = Qwen3VLTextModel(config, deepstack_visual_indexes=range(deepstack_layers))
    model.cuda().to(torch.bfloat16)
    for name, tensor in ref.state_dict().items():
        model.load_weight(name, tensor)
    return ref, model


def _inputs(config, batch=2, seq_len=37, deepstack_layers=2):
    gen = torch.Generator(device="cuda").manual_seed(1)
    embeds = torch.randn(
        batch, seq_len, config.hidden_size, generator=gen, device="cuda"
    ).to(torch.bfloat16)
    # text, a 2x3 image grid, text: M-RoPE positions as Qwen3-VL lays them out
    text = torch.arange(10).expand(3, -1)
    t = torch.zeros(6)
    h, w = torch.arange(2).repeat_interleave(3), torch.arange(3).repeat(2)
    image = torch.stack([t, h, w]) + 10
    tail = (torch.arange(seq_len - 16) + 13).expand(3, -1)
    row = torch.cat([text, image, tail], dim=1).long()
    position_ids = row[:, None].expand(3, batch, seq_len).cuda()
    mask = torch.zeros(batch, seq_len, dtype=torch.bool, device="cuda")
    mask[:, 10:16] = True
    deepstack = [
        torch.randn(
            int(mask.sum()), config.hidden_size, generator=gen, device="cuda"
        ).to(torch.bfloat16)
        for _ in range(deepstack_layers)
    ]
    return embeds, position_ids, mask, deepstack


def _check_against_reference(output_hidden_states):
    config = _text_config()
    ref, model = _models(config, deepstack_layers=2)
    embeds, position_ids, mask, deepstack = _inputs(config)
    kwargs = dict(
        inputs_embeds=embeds,
        position_ids=position_ids,
        visual_pos_masks=mask,
        deepstack_visual_embeds=deepstack,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        expected = ref(**{**kwargs, "inputs_embeds": embeds.clone()}, use_cache=False)
        actual = model(**kwargs)

    assert _relative_error(actual.last_hidden_state, expected.last_hidden_state) < 1e-2
    if output_hidden_states:
        assert len(actual.hidden_states) == len(expected.hidden_states)
        for got, want in zip(actual.hidden_states, expected.hidden_states):
            assert _relative_error(got, want) < 1e-2
    else:
        assert actual.hidden_states is None


@single_rank
@pytest.mark.usefixtures("srt_runtime")
@pytest.mark.parametrize("output_hidden_states", [False, True])
def test_srt_text_model_matches_reference(output_hidden_states):
    _check_against_reference(output_hidden_states)


@single_rank
@pytest.mark.usefixtures("srt_runtime")
def test_srt_text_model_without_final_norm_returns_the_residual_stream():
    config = _text_config()
    ref, model = _models(config, deepstack_layers=0)
    ref.norm = torch.nn.Identity()
    model.drop_final_norm()
    embeds, position_ids, _, _ = _inputs(config, deepstack_layers=0)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        expected = ref(
            inputs_embeds=embeds.clone(), position_ids=position_ids, use_cache=False
        ).last_hidden_state
        actual = model(inputs_embeds=embeds, position_ids=position_ids)
    assert _relative_error(actual.last_hidden_state, expected) < 1e-2


@single_rank
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_two_ranks():
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    cmd += ["--nproc-per-node=2", "-m", "pytest", "-q", __file__]
    subprocess.run(cmd, check=True, timeout=1200)


@pytest.fixture(scope="module")
def tensor_parallel():
    from sglang.srt.runtime_context import get_context, publish
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    maybe_init_distributed_environment_and_model_parallel(tp_size=2, sp_size=1)
    if get_context()._server_args is None:
        publish(
            SrtServerArgs(model_path="dummy", tp_size=2),
            role="diffusion_gpu_worker",
        )
    with use_tensor_parallel_group(get_tp_group()):
        yield


@two_ranks
@pytest.mark.usefixtures("tensor_parallel")
@pytest.mark.parametrize("output_hidden_states", [False, True])
def test_tensor_parallel_srt_text_model_matches_reference(output_hidden_states):
    _check_against_reference(output_hidden_states)
