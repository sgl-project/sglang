"""Real-checkpoint TP1 GPU comparison without loading the target model."""

import argparse
import json
import os
import socket
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers import AutoConfig

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.dp_attention import initialize_dp_attention
from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.srt.models.dflash_context import (
    initialize_context_encoder,
    load_context_encoder_weights,
)
from sglang.srt.runtime_context import SpawnRanks, publish, reset_context
from sglang.srt.server_args import ServerArgs


def weights(root):
    for path in sorted(root.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def run(root, output):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    reset_context()
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(local_rank)
    server = ServerArgs(model_path="dummy", device="cuda", tp_size=world)
    publish(server, role="test", ranks=SpawnRanks(world_rank=rank, gpu_id=local_rank))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://" if world > 1 else f"tcp://127.0.0.1:{port}",
    )
    initialize_model_parallel()
    initialize_dp_attention(server)
    config = AutoConfig.from_pretrained(
        str(root), trust_remote_code=True, local_files_only=True
    )
    with torch.device("cuda"):
        full = DFlash2DraftModel(config)
        slim = DFlash2DraftModel.__new__(DFlash2DraftModel)
        nn.Module.__init__(slim)
        initialize_context_encoder(slim, config, None)
    full.load_weights(weights(root))
    load_context_encoder_weights(slim, weights(root))
    assert len(slim.context_loaded_tensors) == 17
    torch.manual_seed(418406695)
    positions = torch.tensor(
        [
            0,
            1,
            63,
            64,
            127,
            128,
            2047,
            2048,
            2049,
            2175,
            2176,
            4095,
            4096,
            65535,
            131071,
            524287,
            1048575,
        ],
        dtype=torch.int64,
        device="cuda",
    )
    hidden = torch.randn(
        (positions.numel(), 5 * int(config.hidden_size)),
        dtype=torch.bfloat16,
        device="cuda",
    )
    rows = []
    with torch.inference_mode():
        a = full.project_target_hidden(hidden)
        b = slim.project_target_hidden(hidden)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        for i, (f, s) in enumerate(zip(full.layers, slim.layers)):
            ka, va = f.self_attn.kv_proj_only(a)
            kb, vb = s.self_attn.kv_proj_only(b)
            ka = f.self_attn.apply_k_rope(positions, f.self_attn.apply_k_norm(ka))
            kb = s.self_attn.apply_k_rope(positions, s.self_attn.apply_k_norm(kb))
            torch.testing.assert_close(ka, kb, rtol=0, atol=0)
            torch.testing.assert_close(va, vb, rtol=0, atol=0)
            rows.append(
                dict(
                    layer=i,
                    k_max_abs=float((ka.float() - kb.float()).abs().max()),
                    v_max_abs=float((va.float() - vb.float()).abs().max()),
                )
            )
    result = dict(
        state="PASS",
        model=str(root),
        tp=world,
        rank=rank,
        checkpoint_tensors_loaded=slim.context_loaded_tensors,
        full_parameter_bytes=sum(
            p.numel() * p.element_size() for p in full.parameters()
        ),
        slim_parameter_bytes=sum(
            p.numel() * p.element_size() for p in slim.parameters()
        ),
        positions=positions.cpu().tolist(),
        projection_max_abs=float((a.float() - b.float()).abs().max()),
        layers=rows,
        scope="Real checkpoint projection, Knorm and RoPE equivalence; not PD transport or a long-context inference test",
    )
    if world > 1:
        results = [None] * world
        torch.distributed.all_gather_object(results, result)
        result = dict(state="PASS", tp=world, ranks=results)
    if rank == 0:
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    destroy_model_parallel()
    destroy_distributed_environment()
    reset_context()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("/userdata/GLM-5.3-DFlash2"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.model, a.output)
