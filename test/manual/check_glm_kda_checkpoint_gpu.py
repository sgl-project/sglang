"""Checkpoint-backed KDA projection/shard check; not whole-model qualification."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 22, 44])
    parser.add_argument("--fp32-diagnostic", action="store_true")
    parser.add_argument("--paired-loader-control", action="store_true")
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
            control = None
            if args.paired_loader_control:
                # Recreate the pre-patch constructor decision, not a hand-written
                # approximation of its projection methods or checkpoint loaders.
                with patch.object(
                    model, "_kda_projections_are_unquantized", return_value=False
                ):
                    control = model.Glm5NextLinearAttention(
                        layer, cfg.hidden_size, cfg, quant_config=quant, prefix=prefix
                    )
                assert not control.do_fuse_qkvbfg
                for shard, w in zip(("q", "k", "v"), weights[:3]):
                    control.qkv_proj.weight.weight_loader(
                        control.qkv_proj.weight, w, shard
                    )
                for name, w in zip(names[3:], weights[3:]):
                    param = getattr(control, name).weight
                    param.weight_loader(param, w)
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
            if control is not None:
                torch.testing.assert_close(
                    control.qkv_proj.weight, torch.cat(local[:3]), rtol=0, atol=0
                )
                for name, w in zip(names[3:], local[3:]):
                    torch.testing.assert_close(
                        getattr(control, name).weight, w, rtol=0, atol=0
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
                if control is not None:
                    manual_reference = reference
                    reference = control.forward_qkvbfg(x, None)
                    for a, b in zip(reference, manual_reference):
                        torch.testing.assert_close(a, b, atol=0.003, rtol=0.02)
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
                    methods = [module.forward_qkvbfg_fused]
                    if control is not None:
                        methods.append(control.forward_qkvbfg)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            for method in methods:
                                method(x, None)
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = [method(x, None) for method in methods]
                    for replay in range(5):
                        if control is not None:
                            # Reuse the same graph addresses with new request data.
                            # This catches stale views that a constant-input replay misses.
                            x.copy_(torch.randn_like(x))
                            dist.broadcast(x, 0)
                        graph.replay()
                        torch.cuda.synchronize()
                        for outputs, method in zip(captured, methods):
                            expected = method(x, None)
                            for a, b in zip(outputs, expected):
                                torch.testing.assert_close(a, b, atol=0, rtol=0)
                dist.barrier()
                print(
                    json.dumps(
                        {
                            "rank": rank,
                            "layer": layer,
                            "tokens": count,
                            "checkpoint_shard_projection_graph": "pass",
                            "paired_loader_control": control is not None,
                        }
                    ),
                    flush=True,
                )
            del module, control, weights, local
    dist.destroy_process_group()
    print(f"KDA_CHECKPOINT_TP{world}_PASS rank={rank}", flush=True)


if __name__ == "__main__":
    main()
