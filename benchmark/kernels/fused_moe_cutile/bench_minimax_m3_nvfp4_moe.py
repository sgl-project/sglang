"""One MiniMax-M3.1 NVFP4 MoE layer: SGLang Marlin W4A16 vs FlashInfer cuTile W4A4.

Adapted from the reproducer in sgl-project/sglang#49123. The cuTile path goes
through SGLang's integrated runner (fused_experts_none_to_flashinfer_cutile), so
the timings include its host work; the Marlin path is called directly, because
normal dispatch never selects Marlin on SM12x.

For each path and token count: eager and CUDA-graph-replay latency (p50/p90/p99),
error against float32 references, bit-identity over repeated runs, and peak
memory above the resident weights. Results are also written as JSON.

Requirements: an SM120/SM121 GPU; cuda-tile with nvidia-cuda-tileiras,
nvidia-cuda-nvcc and nvidia-nvvm of one major.minor (the cuTile path needs no
other toolkit). The Marlin baseline JIT-builds SGLang kernels with
/usr/local/cuda/bin/nvcc and needs a full CUDA toolkit there: the pip nvcc wheel
lacks matching CCCL headers. Its first call compiles for several minutes.

Timing rotates over several input sets so a replay does not re-read the previous
call's activations from L2; the expert weights (about 28 MiB per expert) exceed
L2 on their own for every routed token.

usage: bench_minimax_m3_nvfp4_moe.py <model dir> [--layer 30] [--experts 32]
           [--tokens 1,64,412,4096,8192,8193,16384] [--out results.json]
"""

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import time
import types

import torch
from safetensors import safe_open

from sglang.srt.layers.moe.moe_runner import flashinfer_cutile as cutile
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput

E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)
NUM_INPUT_SETS = 4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model_dir")
    p.add_argument("--layer", type=int, default=30)
    # The checkpoint has 128 experts per layer; the issue measured the first 32.
    p.add_argument("--experts", type=int, default=32)
    p.add_argument("--tokens", default="1,64,412,4096,8192,8193,16384")
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--identity-runs", type=int, default=20)
    p.add_argument("--reference-max-tokens", type=int, default=4096)
    # 0.5 is the issue's input scale; at that scale no pre-activation reaches the
    # SwiGLU limit on this checkpoint, so use about 4 to exercise the clamp.
    p.add_argument("--input-std", type=float, default=0.5)
    p.add_argument("--skip-marlin", action="store_true")
    p.add_argument("--out", default="cutile_nvfp4_moe_bench.json")
    return p.parse_args()


def load_layer(model_dir, layer, num_experts, device):
    """Experts in the layout SGLang's loader builds: GEMM1 rows are [w1 gate; w3 up]."""
    weight_map = json.load(
        open(os.path.join(model_dir, "model.safetensors.index.json"))
    )["weight_map"]
    probe = f"layers.{layer}.block_sparse_moe.experts.0.w1.weight_packed"
    prefix = next(k for k in weight_map if probe in k).rsplit(".0.w1.", 1)[0]
    handles = {}

    def get(name):
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(os.path.join(model_dir, shard), "pt")
        return handles[shard].get_tensor(name)

    def stack(proj, suffix):
        return [get(f"{prefix}.{e}.{proj}.{suffix}") for e in range(num_experts)]

    def scalars(proj, suffix):
        return torch.stack([x.float().reshape(()) for x in stack(proj, suffix)])

    def cat13(suffix):
        return torch.stack(
            [
                torch.cat((g.view(torch.uint8), u.view(torch.uint8)), 0)
                for g, u in zip(stack("w1", suffix), stack("w3", suffix))
            ]
        )

    layer_tensors = dict(
        w13_packed=cat13("weight_packed"),
        w13_block_scale=cat13("weight_scale").view(torch.float8_e4m3fn),
        w13_weight_global_scale=torch.stack(
            (
                scalars("w1", "weight_global_scale"),
                scalars("w3", "weight_global_scale"),
            ),
            dim=1,
        ),
        w13_input_global_scale=torch.stack(
            (scalars("w1", "input_global_scale"), scalars("w3", "input_global_scale")),
            dim=1,
        ),
        w2_packed=torch.stack(stack("w2", "weight_packed")).view(torch.uint8),
        w2_block_scale=torch.stack(
            [s.view(torch.uint8) for s in stack("w2", "weight_scale")]
        ).view(torch.float8_e4m3fn),
        w2_weight_global_scale=scalars("w2", "weight_global_scale"),
        w2_input_global_scale=scalars("w2", "input_global_scale"),
    )
    return {k: v.to(device) for k, v in layer_tensors.items()}


class Reference:
    """Float32 math on dequantized weights; optionally NVFP4 activations."""

    def __init__(self, layer, alpha, beta, limit):
        self.layer, self.alpha, self.beta, self.limit = layer, alpha, beta, limit
        self.intermediate = layer["w13_packed"].shape[1] // 2
        self.cache = {}

    @staticmethod
    def dequantize(packed, block_scale, encode_scale):
        table = torch.tensor(E2M1, device=packed.device)
        values = torch.stack(
            (table[(packed & 15).long()], table[(packed >> 4).long()]), dim=-1
        ).reshape(packed.shape[0], -1)
        return values * (block_scale.float() / encode_scale).repeat_interleave(
            16, dim=1
        )

    @staticmethod
    def nvfp4(v):
        thresholds = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=v.device
        )
        levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=v.device)
        groups = v.reshape(v.shape[0], -1, 16)
        scale = (groups.abs().amax(-1) / 6.0).to(torch.float8_e4m3fn).float()
        q = groups / scale.clamp(min=2.0**-9).unsqueeze(-1)
        code = (q.abs().unsqueeze(-1) > thresholds).sum(-1)
        sign = torch.where(q < 0, -1.0, 1.0)
        return (levels[code] * sign * scale.unsqueeze(-1)).reshape(v.shape)

    def weights(self, e):
        if e not in self.cache:
            layer, mid = self.layer, self.intermediate
            gate_scale, up_scale = layer["w13_weight_global_scale"][e]
            self.cache[e] = (
                self.dequantize(
                    layer["w13_packed"][e, :mid],
                    layer["w13_block_scale"][e, :mid],
                    gate_scale,
                ),
                self.dequantize(
                    layer["w13_packed"][e, mid:],
                    layer["w13_block_scale"][e, mid:],
                    up_scale,
                ),
                self.dequantize(
                    layer["w2_packed"][e],
                    layer["w2_block_scale"][e],
                    layer["w2_weight_global_scale"][e],
                ),
            )
        return self.cache[e]

    def __call__(self, x, ids, w, *, nvfp4_activations):
        xf = self.nvfp4(x.float()) if nvfp4_activations else x.float()
        hidden = self.layer["w2_packed"].shape[1]
        out = torch.zeros(x.shape[0], hidden, dtype=torch.float32, device=x.device)
        for e in range(self.layer["w13_packed"].shape[0]):
            rows, slot = (ids == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            gate_w, up_w, down_w = self.weights(e)
            gate = (xf[rows] @ gate_w.T).clamp(max=self.limit)
            up = (xf[rows] @ up_w.T).clamp(-self.limit, self.limit)
            act = (
                gate * torch.sigmoid(self.alpha * gate) * (up + self.beta)
            ).bfloat16()
            act = self.nvfp4(act.float()) if nvfp4_activations else act.float()
            out.index_add_(0, rows, w[rows, slot].unsqueeze(1) * (act @ down_w.T))
        return out


def error_metrics(y, ref):
    y, d = y.float(), y.float() - ref
    return dict(
        rel_l2=float(d.norm() / ref.norm()),
        max_abs=float(d.abs().max()),
        mean_abs=float(d.abs().mean()),
        rms=float(d.pow(2).mean().sqrt()),
        cosine=float(
            torch.nn.functional.cosine_similarity(y.flatten(), ref.flatten(), dim=0)
        ),
        non_finite=int((~torch.isfinite(y)).sum()),
    )


def build_marlin(layer, alpha, limit, intermediate):
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_nvfp4_layer_for_marlin,
    )

    def param(t):
        return torch.nn.Parameter(t.clone(), requires_grad=False)

    num_experts = layer["w13_packed"].shape[0]
    # Marlin keeps one GEMM1 global scale per expert: the gate's.
    ns = types.SimpleNamespace(
        w13_weight=param(layer["w13_packed"]),
        w2_weight=param(layer["w2_packed"]),
        w13_weight_scale=param(layer["w13_block_scale"]),
        w2_weight_scale=param(layer["w2_block_scale"]),
        w13_weight_scale_2=param(1.0 / layer["w13_weight_global_scale"][:, 0]),
        w2_weight_scale_2=param(1.0 / layer["w2_weight_global_scale"]),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=intermediate,
        params_dtype=torch.bfloat16,
    )
    prepare_moe_nvfp4_layer_for_marlin(ns, group_size=16)

    def run(x, ids, w):
        # A workspace per call, as the SGLang Marlin runner allocates it.
        return fused_marlin_moe(
            x,
            ns.w13_weight,
            ns.w2_weight,
            ns.w13_weight_scale,
            ns.w2_weight_scale,
            torch.empty(
                (x.shape[0], num_experts), dtype=torch.float32, device=x.device
            ),
            w,
            ids,
            global_num_experts=num_experts,
            w1_global_scale=ns.w13_weight_scale_2,
            w2_global_scale=ns.w2_weight_scale_2,
            workspace=marlin_make_workspace(x.device, max_blocks_per_sm=4),
            num_bits=4,
            clamp_limit=limit,
            gemm1_alpha=alpha,
            activation="silu",
            is_gated=True,
        )

    return run


def build_cutile(layer, top_k, alpha, beta, limit):
    quant_info = cutile.build_cutile_nvfp4_quant_info(
        **layer, top_k=top_k, swiglu_alpha=alpha, swiglu_beta=beta, swiglu_limit=limit
    )
    num_experts = layer["w13_packed"].shape[0]
    config = MoeRunnerConfig(num_experts=num_experts, num_local_experts=num_experts)

    def run(x, ids, w):
        dispatch = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=w, topk_ids=ids, router_logits=None
            ),
        )
        return cutile.fused_experts_none_to_flashinfer_cutile(
            dispatch, quant_info, config
        ).hidden_states

    return run


def percentiles(samples):
    ordered = sorted(samples)

    def at(q):
        return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]

    return dict(
        p50=statistics.median(ordered), p90=at(0.9), p99=at(0.99), n=len(ordered)
    )


def timed(fn, repeats):
    samples = []
    for i in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(i)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return percentiles(samples)


def bench_path(fn, input_sets, repeats, identity_runs):
    x, ids, w = input_sets[0]
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    first = fn(x, ids, w).clone()
    torch.cuda.synchronize()
    first_call_s = time.perf_counter() - t0
    outputs = [fn(x, ids, w).clone() for _ in range(identity_runs - 1)]
    torch.cuda.synchronize()
    peak_mib = (torch.cuda.max_memory_allocated() - resident) / 2**20
    identical = all(
        torch.equal(first.view(torch.int16), o.view(torch.int16)) for o in outputs
    )
    del outputs
    for inputs in input_sets:
        fn(*inputs)
    eager = timed(lambda i: fn(*input_sets[i % len(input_sets)]), repeats)

    # One graph per input set, replayed in rotation: no host work between kernels.
    graphs = []
    for inputs in input_sets:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = fn(*inputs)
        graphs.append((graph, captured))
    for graph, _ in graphs:
        graph.replay()
    torch.cuda.synchronize()
    replay = timed(lambda i: graphs[i % len(graphs)][0].replay(), repeats)
    graph_identical = torch.equal(
        first.view(torch.int16), graphs[0][1].view(torch.int16)
    )
    return first, dict(
        eager_ms=eager,
        graph_ms=replay,
        first_call_s=first_call_s,
        peak_above_resident_mib=peak_mib,
        bit_identical_runs=identity_runs,
        bit_identical=identical,
        graph_replay_matches_eager=graph_identical,
    )


def environment(device):
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    def git_sha(path):
        try:
            return subprocess.check_output(
                ["git", "-C", path, "rev-parse", "HEAD"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    packages = (
        "sglang",
        "flashinfer-python",
        "cuda-tile",
        "nvidia-cuda-tileiras",
        "nvidia-cuda-nvcc",
        "nvidia-nvvm",
        "torch",
        "triton",
    )
    import flashinfer

    return dict(
        gpu=torch.cuda.get_device_name(device),
        compute_capability=torch.cuda.get_device_capability(device),
        torch_cuda=torch.version.cuda,
        driver=subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version,temperature.gpu,power.draw,clocks.gr,utilization.gpu",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip(),
        packages={name: version(name) for name in packages},
        sglang_sha=git_sha(os.path.dirname(os.path.abspath(__file__))),
        flashinfer_sha=git_sha(os.path.dirname(flashinfer.__file__)),
        flashinfer_path=os.path.dirname(flashinfer.__file__),
    )


def main():
    args = parse_args()
    device = torch.device("cuda")
    reason = cutile.cutile_nvfp4_platform_unsupported_reason()
    if reason is not None:
        raise SystemExit(f"cuTile NVFP4 MoE is unavailable here: {reason}")
    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    text = cfg.get("text_config", cfg)
    top_k, hidden, intermediate = (
        text["num_experts_per_tok"],
        text["hidden_size"],
        text["intermediate_size"],
    )
    alpha, beta, limit = float(text["swiglu_alpha"]), 1.0, float(text["swiglu_limit"])
    tokens = [int(t) for t in args.tokens.split(",")]

    t0 = time.perf_counter()
    layer = load_layer(args.model_dir, args.layer, args.experts, device)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    weights_mib = torch.cuda.memory_allocated() / 2**20

    paths, prepare = {}, {}
    if not args.skip_marlin:
        t0 = time.perf_counter()
        paths["marlin_w4a16"] = build_marlin(layer, alpha, limit, intermediate)
        torch.cuda.synchronize()
        prepare["marlin_w4a16"] = dict(
            seconds=time.perf_counter() - t0,
            resident_mib=torch.cuda.memory_allocated() / 2**20 - weights_mib,
        )
    before = torch.cuda.memory_allocated() / 2**20
    t0 = time.perf_counter()
    paths["cutile_w4a4"] = build_cutile(layer, top_k, alpha, beta, limit)
    torch.cuda.synchronize()
    prepare["cutile_w4a4"] = dict(
        seconds=time.perf_counter() - t0,
        resident_mib=torch.cuda.memory_allocated() / 2**20 - before,
    )

    reference = Reference(layer, alpha, beta, limit)
    report = dict(
        environment=environment(device),
        workload=dict(
            layer=args.layer,
            experts=args.experts,
            top_k=top_k,
            hidden=hidden,
            intermediate=intermediate,
            tp=1,
            ep=1,
            routing="uniform random, distinct experts per token",
            input_std=args.input_std,
            swiglu=dict(alpha=alpha, beta=beta, limit=limit),
            checkpoint_load_s=load_s,
            checkpoint_weights_mib=weights_mib,
        ),
        prepare=prepare,
        results=[],
    )
    print(json.dumps(report["environment"], indent=1), flush=True)
    for m in tokens:
        input_sets = []
        for copy in range(NUM_INPUT_SETS):
            g = torch.Generator().manual_seed(m * 1009 + copy)
            x = (
                (torch.randn(m, hidden, generator=g) * args.input_std)
                .bfloat16()
                .to(device)
            )
            ids = torch.stack(
                [torch.randperm(args.experts, generator=g)[:top_k] for _ in range(m)]
            )
            w = torch.rand(m, top_k, generator=g).to(device)
            input_sets.append((x, ids.int().to(device), w))
        x, ids, w = input_sets[0]
        refs = None
        if m <= args.reference_max_tokens:
            refs = dict(
                bf16_activation=reference(x, ids, w, nvfp4_activations=False),
                nvfp4_activation=reference(x, ids, w, nvfp4_activations=True),
            )
        for name, fn in paths.items():
            try:
                out, row = bench_path(fn, input_sets, args.repeats, args.identity_runs)
            except torch.OutOfMemoryError as exc:
                row, out = dict(error=f"out of memory: {exc}"), None
                torch.cuda.empty_cache()
            row.update(tokens=m, path=name)
            if out is not None and refs is not None:
                row["error_vs"] = {k: error_metrics(out, r) for k, r in refs.items()}
            report["results"].append(row)
            print(json.dumps(row), flush=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
