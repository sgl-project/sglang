import base64
import os
import pickle
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest
import requests
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from safetensors import safe_open

from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.test_utils import get_dynamic_server_port
from sglang.srt.utils import init_custom_process_group

GROUP_NAME = "test-update"
LORA_RANK = 4
LORA_ALPHA = 4


def post(base_url, endpoint, body):
    response = requests.post(f"{base_url}/{endpoint}", json=body, timeout=180)
    response.raise_for_status()
    return response.json()


def run_while_posting(sender_side, base_url, endpoint, body):
    with ThreadPoolExecutor(max_workers=1) as executor:
        response = executor.submit(post, base_url, endpoint, body)
        sender_result = sender_side()
        assert response.result()["success"]
    return sender_result


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def init_device_bound_default_group(device, rendezvous_file):
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous_file}",
        rank=0,
        world_size=1,
        device_id=device,
    )


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    num_gpus = int(os.environ.get("SGLANG_TEST_WEIGHT_ENGINE_GPUS", "1"))
    if torch.cuda.device_count() <= num_gpus:
        pytest.skip("Requires the engine GPUs plus a separate sender GPU")
    model = os.environ.get(
        "SGLANG_TEST_WEIGHT_MODEL", "black-forest-labs/FLUX.2-klein-base-4B"
    )
    visible_gpus = os.environ.get(
        "CUDA_VISIBLE_DEVICES",
        ",".join(str(i) for i in range(torch.cuda.device_count())),
    ).split(",")
    engine_gpus = visible_gpus[:num_gpus]
    port = get_dynamic_server_port()
    server = ServerManager(
        model=model,
        port=port,
        extra_args=f"--num-gpus {num_gpus} --enable-cfg-parallel false --warmup-mode off --dit-precision bf16 "
        + os.environ.get("SGLANG_TEST_WEIGHT_SERVER_ARGS", ""),
        env_vars={"CUDA_VISIBLE_DEVICES": ",".join(engine_gpus)},
    ).start()
    sender_device = torch.device("cuda", len(engine_gpus))
    torch.cuda.set_device(sender_device)
    init_device_bound_default_group(
        sender_device, tmp_path_factory.mktemp("weights") / "rendezvous"
    )
    try:
        yield f"http://127.0.0.1:{port}", model, num_gpus
    finally:
        dist.destroy_process_group()
        server.cleanup()


def init_update_group(base_url, num_gpus):
    port = free_port()
    world_size = 1 + num_gpus

    def join_as_sender():
        return init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=world_size,
            group_name=GROUP_NAME,
            timeout=timedelta(seconds=120),
        )

    return run_while_posting(
        join_as_sender,
        base_url,
        "init_weights_update_group",
        dict(
            master_address="127.0.0.1",
            master_port=port,
            rank_offset=1,
            world_size=world_size,
            group_name=GROUP_NAME,
        ),
    )


def destroy_update_group(base_url, group):
    run_while_posting(
        lambda: dist.destroy_process_group(group),
        base_url,
        "destroy_weights_update_group",
        {"group_name": GROUP_NAME},
    )


def update_options(mode):
    return dict(
        target_modules=["transformer"],
        weight_update_mode=mode,
        lora_alpha=LORA_ALPHA,
        lora_rank=LORA_RANK,
    )


def update_weights_from_tensor(base_url, weights, mode):
    payload = {"transformer": [(name, tensor.cpu()) for name, tensor in weights]}
    response = post(
        base_url,
        "update_weights_from_tensor",
        dict(
            serialized_named_tensors=[base64.b64encode(pickle.dumps(payload)).decode()],
            **update_options(mode),
        ),
    )
    assert response["success"]


def update_weights_from_distributed(base_url, group, weights, mode):
    def broadcast():
        tensors = [tensor.contiguous() for _, tensor in weights]
        handles = [
            dist.broadcast(tensor, src=0, group=group, async_op=True)
            for tensor in tensors
        ]
        for handle in handles:
            handle.wait()

    run_while_posting(
        broadcast,
        base_url,
        "update_weights_from_distributed",
        dict(
            names=[name for name, _ in weights],
            dtypes=[str(tensor.dtype).removeprefix("torch.") for _, tensor in weights],
            shapes=[list(tensor.shape) for _, tensor in weights],
            group_name=GROUP_NAME,
            **update_options(mode),
        ),
    )


def transformer_checksum(base_url):
    return post(base_url, "get_weights_checksum", {"module_names": ["transformer"]})


def load_attention_weight(model):
    model_dir = Path(model) if Path(model).is_dir() else Path(snapshot_download(model))
    for path in sorted((model_dir / "transformer").glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as reader:
            for name in reader.keys():
                if name.endswith(".attn.to_q.weight"):
                    return name, reader.get_tensor(name).cuda()
    pytest.fail("No attention projection found")


def make_weights(name, original, mode):
    if mode != "lora_merge":
        return [(name, original + 0.01)]
    prefix = name.removesuffix(".weight")
    out_features, in_features = original.shape
    lora_a = torch.randn(in_features, LORA_RANK, device="cuda").t() * 0.1
    lora_b = torch.randn(LORA_RANK, out_features, device="cuda").t() * 0.1
    return [(f"{prefix}.lora_A", lora_a), (f"{prefix}.lora_B", lora_b)]


@pytest.mark.parametrize("mode", [None, "lora_merge"])
def test_distributed_matches_tensor_updates_and_reconnects(engine, mode):
    base_url, model, num_gpus = engine
    name, original = load_attention_weight(model)
    torch.manual_seed(123)
    weights = make_weights(name, original, mode)
    other_weights = [(key, tensor + 0.02) for key, tensor in weights]

    update_weights_from_tensor(base_url, weights, mode)
    expected = transformer_checksum(base_url)

    group = init_update_group(base_url, num_gpus)
    durations = []
    try:
        update_weights_from_distributed(base_url, group, other_weights, mode)
        assert transformer_checksum(base_url) != expected
        for _ in range(2):
            start = time.perf_counter()
            update_weights_from_distributed(base_url, group, weights, mode)
            durations.append(time.perf_counter() - start)
            assert transformer_checksum(base_url) == expected
    finally:
        destroy_update_group(base_url, group)
    print(
        f"weight_update mode={mode} bytes={sum(t.numel() * t.element_size() for _, t in weights)} seconds={durations}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
