"""Checkpoint-backed KDA projection/shard check; not whole-model qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 22, 44])
    parser.add_argument("--fp32-diagnostic", action="store_true")
    args = parser.parse_args()
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(20260913)
    dist.init_process_group("nccl")
    from sglang.srt.runtime_context import get_parallel, publish
    from sglang.srt.server_args import ServerArgs

    publish(ServerArgs(model_path=args.model, device="cuda"), role="test")
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.models import glm5_next as model

    root = Path(args.model)
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    cfg = SimpleNamespace(**config["text_config"])
    cfg.dtype = torch.bfloat16
    quant = Fp8Config.from_config(config["quantization_config"])
    names = [
        "q_proj",
        "k_proj",
        "v_proj",
        "b_proj",
        "f_a_proj",
        "g_a_proj",
        "f_b_proj",
        "g_b_proj",
    ]
    with (
        get_parallel().override(
            tp_rank=rank, tp_size=world, attn_tp_rank=rank, attn_tp_size=world
        ),
        torch.device("cuda"),
        torch.inference_mode(),
    ):
        for layer in args.layers:
            prefix = f"model.layers.{layer}.self_attn"
            weights = []
            hashes = {}
            for name in names:
                key = f"model.language_model.layers.{layer}.self_attn.{name}.weight"
                with safe_open(root / index[key], framework="pt", device="cpu") as f:
                    weight = f.get_tensor(key)
                assert weight.dtype == torch.bfloat16, (key, weight.dtype)
                hashes[key] = hashlib.sha256(
                    weight.view(torch.uint8).numpy().tobytes()
                ).hexdigest()
                weights.append(weight.cuda())
            module = model.Glm5NextLinearAttention(
                layer, cfg.hidden_size, cfg, quant_config=quant, prefix=prefix
            )
            assert module.do_fuse_qkvbfg
            for i, w in enumerate(weights[:6]):
                module.fused_qkvbfg_a_proj.weight_loader(
                    module.fused_qkvbfg_a_proj.weight, w, i
                )
            for i, w in enumerate(weights[6:]):
                module.fused_fg_b_proj.weight_loader(
                    module.fused_fg_b_proj.weight, w, i
                )
            # Check rank slices independently of the production loader.
            local = [
                w.chunk(world, dim=0)[rank] if i < 4 or i >= 6 else w
                for i, w in enumerate(weights)
            ]
            torch.testing.assert_close(
                module.fused_qkvbfg_a_proj.weight, torch.cat(local[:6]), rtol=0, atol=0
            )
            torch.testing.assert_close(
                module.fused_fg_b_proj.weight, torch.stack(local[6:]), rtol=0, atol=0
            )
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "layer": layer,
                            "checkpoint_tensor_sha256": hashes,
                            "world_size": world,
                            "source": model.__file__,
                        }
                    ),
                    flush=True,
                )
            for count in (1, 8, 16, 257):
                x = torch.randn(count, cfg.hidden_size, device="cuda")
                dist.broadcast(x, 0)
                reference = (
                    torch.cat([x @ w.T for w in local[:3]], -1),
                    x @ local[3].T,
                    (x @ local[4].T) @ local[6].T,
                    (x @ local[5].T) @ local[7].T,
                )
                actual = module.forward_qkvbfg_fused(x, None)
                if args.fp32_diagnostic:
                    xf = x.float()
                    wf = [w.float() for w in local]
                    oracle = (
                        torch.cat([xf @ w.T for w in wf[:3]], -1),
                        xf @ wf[3].T,
                        (xf @ wf[4].T) @ wf[6].T,
                        (xf @ wf[5].T) @ wf[7].T,
                    )
                    for component, (a, b, gold) in enumerate(
                        zip(actual, reference, oracle)
                    ):
                        scale = gold.square().mean().sqrt().clamp_min(1e-8)
                        ae = ((a.float() - gold).square().mean().sqrt() / scale).item()
                        be = ((b.float() - gold).square().mean().sqrt() / scale).item()
                        mismatch = ((a - b).abs() > 0.003 + 0.02 * b.abs()).sum().item()
                        print(
                            json.dumps(
                                {
                                    "rank": rank,
                                    "layer": layer,
                                    "tokens": count,
                                    "component": component,
                                    "fused_nrmse": ae,
                                    "unfused_nrmse": be,
                                    "strict_mismatches": mismatch,
                                }
                            ),
                            flush=True,
                        )
                        assert torch.isfinite(a).all()
                        assert ae <= max(1.10 * be, be + 0.0001), (ae, be)
                else:
                    for a, b in zip(actual, reference):
                        torch.testing.assert_close(a, b, atol=0.003, rtol=0.02)
                if count <= 16:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            module.forward_qkvbfg_fused(x, None)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = module.forward_qkvbfg_fused(x, None)
                    for _ in range(5):
                        graph.replay()
                    torch.cuda.synchronize()
                    for a, b in zip(captured, actual):
                        torch.testing.assert_close(a, b, atol=0, rtol=0)
                dist.barrier()
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "layer": layer,
                            "tokens": count,
                            "checkpoint_shard_projection_graph": "pass",
                        }
                    ),
                    flush=True,
                )
            del module, weights, local
    dist.destroy_process_group()
    print(f"KDA_CHECKPOINT_TP{world}_PASS rank={rank}", flush=True)


if __name__ == "__main__":
    main()
