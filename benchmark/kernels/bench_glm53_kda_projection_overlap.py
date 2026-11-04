"""Compare GLM-5.3-Flash's merged and overlapped KDA input projections.

Runs production projection modules, with distinct layer weights larger than
L2. TP denotes the per-rank head partition; no distributed group is launched.
Optionally load the actual BF16 projections from a local checkpoint.
"""

import argparse
import gc
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
from sglang.srt.models.glm5_next import Glm5NextLinearAttention
from sglang.srt.runtime_context import get_context, get_parallel


def load_projections(attention, model_path, weight_map, layer_id, tp):
    prefix = f"model.language_model.layers.{layer_id}.self_attn"
    if f"{prefix}.q_proj.weight" not in weight_map:
        prefix = f"model.layers.{layer_id}.self_attn"

    def load(name, replicated=False):
        key = f"{prefix}.{name}.weight"
        with safe_open(model_path / weight_map[key], framework="pt") as f:
            source = f.get_slice(key)
            rows = source.get_shape()[0]
            return source[:] if replicated else source[: rows // tp]

    offset = 0
    for name in ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj"):
        source = load(name, replicated=name in ("f_a_proj", "g_a_proj"))
        rows = source.shape[0]
        attention.fused_qkvbfg_a_proj.weight[offset : offset + rows].copy_(source)
        offset += rows
    for i, name in enumerate(("f_b_proj", "g_b_proj")):
        attention.fused_fg_b_proj.weight[i].copy_(load(name))


def make_layers(tp, count, model_path):
    weight_map = None
    if model_path is not None:
        weight_map = json.loads(
            (model_path / "model.safetensors.index.json").read_text()
        )["weight_map"]
    stream = torch.cuda.Stream()
    layer_ids = [i for i in range(45) if i % 4 != 3][:count]
    layers = []
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with (
            torch.device("cuda"),
            get_parallel().override(
                tp_size=tp,
                tp_rank=0,
                attn_tp_size=tp,
                attn_tp_rank=0,
                attn_dp_size=1,
                attn_cp_size=1,
                moe_tp_size=tp,
            ),
        ):
            for layer_id in layer_ids:
                layer = Glm5NextLinearAttention(
                    layer_id,
                    4096,
                    SimpleNamespace(
                        linear_attn_config={
                            "head_dim": 128,
                            "num_heads": 64,
                            "short_conv_kernel_size": 4,
                        }
                    ),
                    alt_stream=stream,
                )
                if layer._proj_stream is None:
                    raise RuntimeError("overlap requires SM90 and TP4/TP8")
                if model_path is None:
                    layer.fused_qkvbfg_a_proj.weight.normal_(std=4096**-0.5)
                    layer.fused_fg_b_proj.weight.normal_(std=128**-0.5)
                else:
                    load_projections(layer, model_path, weight_map, layer_id, tp)
                layers.append(layer)
    finally:
        torch.set_default_dtype(old_dtype)
    return layers


def run_case(layers, m, rounds, replays):
    x = torch.randn(m, 4096, device="cuda", dtype=torch.bfloat16)
    limits = [layer._proj_overlap_max_tokens for layer in layers]
    graphs, outputs = {}, {}
    kinds = ("merged", "overlap", "auto")
    for kind in kinds:
        for layer, limit in zip(layers, limits):
            layer._proj_overlap_max_tokens = (
                0 if kind == "merged" else m if kind == "overlap" else limit
            )
        graph = torch.cuda.CUDAGraph()
        with model_capture_mode():
            for _ in range(3):
                for layer in layers:
                    layer.forward_qkvbfg_fused(x, None)
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                outputs[kind] = [
                    layer.forward_qkvbfg_fused(x, None) for layer in layers
                ]
        graphs[kind] = graph
        graph.replay()
        torch.cuda.synchronize()

    samples = {kind: [] for kind in kinds}
    for i in range(rounds):
        for kind in kinds if i % 2 == 0 else tuple(reversed(kinds)):
            graph = graphs[kind]
            for _ in range(5):
                graph.replay()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            samples[kind].append(
                start.elapsed_time(end) * 1000 / (replays * len(layers))
            )

    max_abs_diff = max(
        (a.float() - b.float()).abs().max().item()
        for ref, actual in zip(outputs["merged"], outputs["overlap"])
        for a, b in zip(ref, actual)
    )
    for layer, limit in zip(layers, limits):
        layer._proj_overlap_max_tokens = limit
    result = {
        "tokens_per_rank": m,
        "us_per_layer": {k: statistics.median(v) for k, v in samples.items()},
        "samples_us_per_layer": samples,
        "merged_vs_overlap_max_abs_diff": max_abs_diff,
    }
    del graphs, outputs, graph
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[4, 8])
    parser.add_argument("--tokens", nargs="+", type=int, default=[1, 6, 8, 16, 32, 128])
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.layers <= 34 or min(args.tokens) < 1:
        parser.error("use 1..34 layers and positive token counts")
    torch.manual_seed(42)
    result = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "model_path": str(args.model_path) if args.model_path else None,
        "layers": args.layers,
        "cases": [],
    }
    with (
        get_context().override_server_args(device="cuda", bf16_gemm_backend="torch"),
        envs.SGLANG_GLM5_KDA_PROJ_OVERLAP.override(True),
    ):
        for tp in args.tp_sizes:
            layers = make_layers(tp, args.layers, args.model_path)
            for m in args.tokens:
                case = {"tp": tp, **run_case(layers, m, args.rounds, args.replays)}
                result["cases"].append(case)
                args.output.write_text(json.dumps(result, indent=2))
                print(json.dumps(case), flush=True)
            del layers
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
