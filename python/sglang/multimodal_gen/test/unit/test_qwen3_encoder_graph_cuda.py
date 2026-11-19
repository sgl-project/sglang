# SPDX-License-Identifier: Apache-2.0
"""Qwen3 text encoder CUDA graph replay; no checkpoint needed."""

from types import SimpleNamespace

import pytest
import torch

import sglang.multimodal_gen.runtime.models.encoders.qwen3 as qwen3
from sglang.multimodal_gen.configs.models.encoders.qwen3 import (
    Qwen3TextArchConfig,
    Qwen3TextConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2KleinPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.managers.memory_managers.memory_occupation_controller import (
    MemoryOccupationController,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    set_global_server_args,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    BreakableCUDAGraph,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform.is_cuda()),
    reason="requires CUDA",
)

SEQ_LEN = 40


@pytest.fixture(scope="module")
def encoder():
    set_global_server_args(
        ServerArgs(
            model_path="black-forest-labs/FLUX.2-klein-4B",
            num_gpus=1,
            pipeline_config=Flux2KleinPipelineConfig(),
            attention_backend="torch_sdpa",
        )
    )
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    config = Qwen3TextConfig(
        arch_config=Qwen3TextArchConfig(
            vocab_size=64,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
        )
    )
    # the loader fills this from the checkpoint's config.json
    config.arch_config.rope_parameters = {"rope_type": "default", "rope_theta": 1e6}
    torch.manual_seed(0)
    with set_default_torch_dtype(torch.bfloat16):
        model = qwen3.Qwen3ForCausalLM(config).cuda().eval()
    for name, param in model.named_parameters():
        if name.endswith("norm.weight"):
            torch.nn.init.ones_(param)
        else:
            torch.nn.init.normal_(param, std=0.05)
    return model


def _request(valid_len, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    input_ids = torch.randint(0, 64, (1, SEQ_LEN), device="cuda", generator=generator)
    attention_mask = torch.zeros(1, SEQ_LEN, dtype=torch.long, device="cuda")
    attention_mask[:, :valid_len] = 1
    return input_ids, attention_mask


def _encode(encoder, input_ids, attention_mask):
    with set_forward_context(None, None):
        out = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    # a replay overwrites the previous output
    return [hidden.clone() for hidden in out.hidden_states]


@torch.inference_mode()
def test_prompts_of_any_length_replay_one_graph_bitwise_like_eager(
    encoder, monkeypatch
):
    # Klein pads every prompt to one length; each prompt has its own valid length
    requests = [_request(n, seed) for seed, n in enumerate((23, 23, 31, 7, SEQ_LEN))]
    with monkeypatch.context() as eager_only:
        eager_only.setattr(encoder, "_cuda_graphs", None, raising=False)
        expected = [_encode(encoder, *request) for request in requests]

    replays = []
    replay = BreakableCUDAGraph.replay

    def counting_replay(graph):
        replays.append(graph)
        return replay(graph)

    monkeypatch.setattr(BreakableCUDAGraph, "replay", counting_replay)
    got = [_encode(encoder, *request) for request in requests]

    # the first request runs eagerly, the second captures, all later ones replay
    assert len(replays) == len(requests) - 1
    assert all(graph is replays[0] for graph in replays)
    for want, have in zip(expected, got):
        assert len(have) == len(want)
        for a, b in zip(have, want):
            assert torch.equal(a, b)


@torch.inference_mode()
def test_replaced_weights_drop_the_graph(encoder, monkeypatch):
    request = _request(17, 100)
    for _ in range(2):
        _encode(encoder, *request)
    graphs = encoder._cuda_graphs
    assert graphs.graphs

    layer = encoder.layers[0].mlp.down_proj
    layer.weight = torch.nn.Parameter(layer.weight.clone() + 0.01)
    got = [_encode(encoder, *request)]
    assert not graphs.graphs
    # the old graphs freed their pool, so the recapture needs a new one
    got.append(_encode(encoder, *request))
    assert graphs.graphs
    with monkeypatch.context() as eager_only:
        eager_only.setattr(encoder, "_cuda_graphs", None, raising=False)
        want = _encode(encoder, *request)
    for have in got:
        for a, b in zip(have, want):
            assert torch.equal(a, b)


@torch.inference_mode()
def test_module_hooks_keep_the_encoder_eager(encoder):
    # layerwise offload loads weights from hooks, which a replay would skip
    calls = []
    handle = encoder.layers[0].register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        for _ in range(3):
            _encode(encoder, *_request(11, 200))
    finally:
        handle.remove()
    assert len(calls) == 3


def _storage_bytes(tensors):
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage() for t in tensors}
    return sum(storage.nbytes() for storage in storages.values())


def _captured_bytes(graphs):
    tensors = []
    for _, static_ids, output in graphs.graphs.values():
        tensors += [static_ids, output.last_hidden_state, *output.hidden_states]
    return _storage_bytes([t for t in tensors if t is not None])


@pytest.mark.parametrize("offloaded_between_uses", [False, True])
@torch.inference_mode()
def test_release_memory_occupation_frees_the_graphs(
    encoder, monkeypatch, offloaded_between_uses
):
    request = _request(19, 300)
    for _ in range(2):
        _encode(encoder, *request)
    assert encoder._cuda_graphs.graphs
    if offloaded_between_uses:
        # component CPU offload moves the encoder off the GPU after each use
        encoder.to("cpu")

    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    weight_bytes = _storage_bytes(
        [t for t in [*encoder.parameters(), *encoder.buffers()] if t.is_cuda]
    )
    captured_bytes = _captured_bytes(encoder._cuda_graphs)
    controller = MemoryOccupationController(
        SimpleNamespace(modules={"text_encoder": encoder}),
        rank=0,
        use_fsdp_inference=False,
    )
    assert controller.release_memory_occupation()["success"]
    try:
        assert not encoder._cuda_graphs.graphs
        assert (
            torch.cuda.memory_allocated() <= allocated - weight_bytes - captured_bytes
        )
    finally:
        assert controller.resume_memory_occupation()["success"]
        encoder.cuda()

    got = [_encode(encoder, *request) for _ in range(2)]
    assert encoder._cuda_graphs.graphs
    with monkeypatch.context() as eager_only:
        eager_only.setattr(encoder, "_cuda_graphs", None, raising=False)
        want = _encode(encoder, *request)
    for have in got:
        for a, b in zip(have, want):
            assert torch.equal(a, b)
