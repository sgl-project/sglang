# SPDX-License-Identifier: Apache-2.0
"""Qwen3 text encoder CUDA graph replay; no checkpoint needed."""

import gc
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp

import sglang.multimodal_gen.runtime.models.encoders.qwen3 as qwen3
from sglang.multimodal_gen.configs.models.encoders.qwen3 import (
    Qwen3TextArchConfig,
    Qwen3TextConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2KleinPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_replica_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
    use_tensor_parallel_group,
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


def _server_args(tp_size):
    return ServerArgs(
        model_path="black-forest-labs/FLUX.2-klein-4B",
        num_gpus=tp_size,
        tp_size=tp_size,
        pipeline_config=Flux2KleinPipelineConfig(),
        attention_backend="torch_sdpa",
    )


@pytest.fixture(scope="module")
def encoder():
    set_global_server_args(_server_args(1))
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    return _build_encoder()


def _build_encoder():
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


def _run_tp2_replay(rank, rendezvous, fold):
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=2,
        rank=rank,
        local_rank=rank,
        distributed_init_method=rendezvous,
        backend="nccl",
        timeout=60,
    )
    initialize_model_parallel(tensor_parallel_degree=2, sequence_parallel_degree=1)
    try:
        set_global_server_args(_server_args(2))
        # as the loader binds it; the replica group (--encoder-parallel fold) has no
        # custom all-reduce, so its all-reduces are captured as NCCL kernels
        group = get_replica_group() if fold else get_tp_group()
        with use_tensor_parallel_group(group):
            encoder = _build_encoder()
        encoder.bind_encoder_tp_group(group)
        requests = [_request(n, seed) for seed, n in enumerate((23, 23, 31, SEQ_LEN))]
        # ids, not graphs: a live graph holding NCCL kernels hangs process-group teardown
        replays = []
        replay = BreakableCUDAGraph.replay

        def counting_replay(graph):
            replays.append(id(graph))
            return replay(graph)

        try:
            with torch.inference_mode():
                graphs, encoder._cuda_graphs = encoder._cuda_graphs, None
                expected = [_encode(encoder, *request) for request in requests]
                encoder._cuda_graphs = graphs
                with patch.object(BreakableCUDAGraph, "replay", counting_replay):
                    got = [_encode(encoder, *request) for request in requests]
            assert len(replays) == len(requests) - 1
            assert len(set(replays)) == 1
            for want, have in zip(expected, got):
                for a, b in zip(have, want):
                    assert torch.equal(a, b)
        finally:
            encoder.release_cuda_graphs()
            gc.collect()
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("fold", [False, True], ids=["tp", "fold"])
def test_tensor_parallel_encoder_replays_bitwise_like_eager(tmp_path, fold):
    # the row-parallel all-reduces are captured with the rest of the forward
    mp.spawn(
        _run_tp2_replay,
        args=(f"file://{tmp_path / 'rendezvous'}", fold),
        nprocs=2,
        join=True,
    )


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
