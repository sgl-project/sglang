import base64
import os
import pickle
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
import requests
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from safetensors import safe_open

from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.test_utils import get_dynamic_server_port
from sglang.srt.utils import init_custom_process_group


def post(base_url, endpoint, body):
    response = requests.post(f"{base_url}/{endpoint}", json=body, timeout=180)
    response.raise_for_status()
    return response.json()


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("Requires a sender GPU and a separate engine GPU")
    model = os.environ.get(
        "SGLANG_TEST_WEIGHT_MODEL", "black-forest-labs/FLUX.2-klein-base-4B"
    )
    num_gpus = int(os.environ.get("SGLANG_TEST_WEIGHT_ENGINE_GPUS", "1"))
    if torch.cuda.device_count() <= num_gpus:
        pytest.skip("Requires a separate sender GPU")
    port = get_dynamic_server_port()
    visible = os.environ.get(
        "CUDA_VISIBLE_DEVICES",
        ",".join(str(i) for i in range(torch.cuda.device_count())),
    ).split(",")
    server = ServerManager(
        model=model,
        port=port,
        extra_args=f"--num-gpus {num_gpus} --enable-cfg-parallel false --warmup-mode off --dit-precision bf16 "
        + os.environ.get("SGLANG_TEST_WEIGHT_SERVER_ARGS", ""),
        env_vars={"CUDA_VISIBLE_DEVICES": ",".join(visible[:num_gpus])},
    ).start()
    torch.cuda.set_device(num_gpus)
    rendezvous = tmp_path_factory.mktemp("weights") / "rendezvous"
    dist.init_process_group(
        "nccl",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
        device_id=torch.device("cuda", num_gpus),
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url, model, num_gpus
    finally:
        dist.destroy_process_group()
        server.cleanup()


def connect(base_url, num_gpus):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with ThreadPoolExecutor(max_workers=1) as executor:
        response = executor.submit(
            post,
            base_url,
            "init_weights_update_group",
            dict(
                master_address="127.0.0.1",
                master_port=port,
                rank_offset=1,
                world_size=num_gpus + 1,
                group_name="test-update",
            ),
        )
        options = dist.ProcessGroupNCCL.Options()
        group = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=num_gpus + 1,
            group_name="test-update",
            timeout=timedelta(seconds=120),
            pg_options=options,
        )
        options.split_from = None
        assert response.result()["success"]
    return group


def update_weights(base_url, weights, mode, group=None):
    kwargs = dict(
        target_modules=["transformer"],
        weight_update_mode=mode,
        lora_alpha=4,
        lora_rank=4,
    )
    if group is None:
        payload = {"transformer": [(name, tensor.cpu()) for name, tensor in weights]}
        kwargs["serialized_named_tensors"] = [
            base64.b64encode(pickle.dumps(payload)).decode()
        ]
        return post(base_url, "update_weights_from_tensor", kwargs)
    kwargs.update(
        names=[name for name, _ in weights],
        dtypes=[str(t.dtype).removeprefix("torch.") for _, t in weights],
        shapes=[list(t.shape) for _, t in weights],
        group_name="test-update",
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        response = executor.submit(
            post, base_url, "update_weights_from_distributed", kwargs
        )
        tensors = [tensor.contiguous() for _, tensor in weights]
        handles = [
            dist.broadcast(tensor, src=0, group=group, async_op=True)
            for tensor in tensors
        ]
        for handle in handles:
            handle.wait()
        return response.result()


@pytest.mark.parametrize("mode", [None, "lora_merge"])
def test_distributed_matches_tensor_updates_and_reconnects(engine, mode):
    from pathlib import Path

    base_url, model, num_gpus = engine
    model_dir = Path(model) if Path(model).is_dir() else Path(snapshot_download(model))
    for path in sorted((model_dir / "transformer").glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as reader:
            names = [
                name for name in reader.keys() if name.endswith(".attn.to_q.weight")
            ]
            if names:
                name = names[0]
                original = reader.get_tensor(name).cuda()
                break
    else:
        pytest.fail("No attention projection found")
    torch.manual_seed(123)
    if mode == "lora_merge":
        prefix = name.removesuffix(".weight")
        weights = [
            (
                f"{prefix}.lora_A",
                torch.randn(original.shape[1], 4, device="cuda").t() * 0.1,
            ),
            (
                f"{prefix}.lora_B",
                torch.randn(4, original.shape[0], device="cuda").t() * 0.1,
            ),
        ]
    else:
        weights = [(name, original + 0.01)]
    changed = [(name, tensor + 0.02) for name, tensor in weights]
    assert update_weights(base_url, weights, mode)["success"]
    expected = post(base_url, "get_weights_checksum", {"module_names": ["transformer"]})
    group = connect(base_url, num_gpus)
    durations = []
    try:
        assert update_weights(base_url, changed, mode, group)["success"]
        assert (
            post(base_url, "get_weights_checksum", {"module_names": ["transformer"]})
            != expected
        )
        for _ in range(2):
            start = time.perf_counter()
            assert update_weights(base_url, weights, mode, group)["success"]
            durations.append(time.perf_counter() - start)
            assert (
                post(
                    base_url, "get_weights_checksum", {"module_names": ["transformer"]}
                )
                == expected
            )
    finally:
        with ThreadPoolExecutor(max_workers=1) as executor:
            response = executor.submit(
                post,
                base_url,
                "destroy_weights_update_group",
                {"group_name": "test-update"},
            )
            dist.destroy_process_group(group)
            assert response.result()["success"]
    print(
        f"weight_update mode={mode} bytes={sum(t.numel() * t.element_size() for _, t in weights)} seconds={durations}"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
