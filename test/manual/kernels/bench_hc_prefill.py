"""MI355X timing of one Qwen4Exp gated hyper-connection site at prefill sizes.

The site is built as the model builds it, at Qwen3.8-Flash-Next's geometry (HC 4, hidden
2560, low-rank 320, per-branch norm, BF16) with random weights. Its mix and its combine
run as the model runs them: each replayed as a CUDA graph, after a 512 MB write that
evicts L2 and the MALL as the attention or MoE block between them does. Arms:

- unfused: the unfused mix and combine (SGLANG_USE_HIP_HC_PREFILL_TRITON off), with the
  TorchInductor settings of the environment;
- control: unfused again, a second site with the same weights, which sets the noise floor;
- new: SGLANG_USE_HIP_HC_PREFILL_TRITON=1.

The arms are interleaved per repetition; each time is the median. TorchInductor reads
TORCHINDUCTOR_MAX_AUTOTUNE at import, so --also-without-autotune repeats the run in a
second process with it and TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE set to 0. The kernel
names of one call of every arm are recorded, and the run fails if a new arm lacks the
prefill kernels or an unfused arm runs them.

    python test/manual/kernels/bench_hc_prefill.py --also-without-autotune --out out.json
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

import torch

from sglang.srt.environ import envs

HC, HS, LOWRANK, EPS = 4, 2560, 320, 1e-6
K = HC * HS
FLUSH_BYTES = 512 << 20
PREFILL_KERNEL = "_hc_prefill_"


def make_site(enabled):
    from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig

    cfg = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HS,
        params_dtype=torch.bfloat16,
        hc_lowrank=LOWRANK,
        rms_norm_eps=EPS,
        hc_per_branch_norm=True,
    )
    with envs.SGLANG_USE_HIP_HC_PREFILL_TRITON.override(enabled):
        site = GatedResidual(cfg).cuda().bfloat16()
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for param in site.parameters():
            param.copy_(0.02 * torch.randn(param.shape, generator=gen, device="cuda"))
    return site


def reference(site, h, y):
    """The site in FP32: mixed [M, HS] and combined [M, K]."""
    rows = h.shape[0]
    hf = h.float().view(rows, HC, HS)
    rstd = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS)
    normed = (hf * rstd).reshape(rows, K) * (1.0 + site.hc_norm.weight.float())
    t = torch.nn.functional.silu(
        normed @ site.input_mix_weight_down.weight.float().T / HC
    )
    gate = torch.sigmoid(t @ site.input_mix_weight_up.weight.float().T)
    mixed = (gate * normed).view(rows, HC, HS).mean(1)
    inject = 2 * torch.sigmoid(normed @ site.block_inject_weight.weight.float().T / HC)
    combined = (hf + inject[..., None] * y.float()[:, None, :]).reshape(rows, K)
    return mixed, combined


def relative_error(actual, ref):
    return ((actual.float() - ref).abs().max() / ref.abs().max()).item()


class Arm:
    """One site with its mix and combine captured as CUDA graphs."""

    def __init__(self, name, site, h, y):
        self.name, self.site, self.h, self.y = name, site, h, y
        for _ in range(3):  # compile, and build the folded weights, outside the capture
            self.site.combine(self.y, self.site.mix(self.h)[1])
        torch.cuda.synchronize()
        self.mix_graph, self.combine_graph = (
            torch.cuda.CUDAGraph(),
            torch.cuda.CUDAGraph(),
        )
        with torch.cuda.graph(self.mix_graph):
            self.mixed, residuals = self.site.mix(self.h)
        with torch.cuda.graph(self.combine_graph):
            self.combined = self.site.combine(self.y, residuals)
        self.samples = {"mix_us": [], "combine_us": []}

    def kernel_names(self):
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            self.site.combine(self.y, self.site.mix(self.h)[1])
            torch.cuda.synchronize()
        return sorted(
            {
                event.name[:100]
                for event in prof.events()
                if event.device_type == torch.autograd.DeviceType.CUDA
            }
        )


def timed_replay(graph, flush_buffer):
    flush_buffer.fill_(1.0)
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1e3


def quantile(samples, q):
    return statistics.quantiles(samples, n=100)[q - 1]


def run_rows(rows, reps, flush_buffer):
    h = torch.randn((rows, K), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((rows, HS), device="cuda", dtype=torch.bfloat16)
    arms = [
        Arm("unfused", make_site(False), h, y),
        Arm("control", make_site(False), h, y),
        Arm("new", make_site(True), h, y),
    ]
    want_mixed, want_combined = reference(arms[0].site, h, y)
    names = {arm.name: arm.kernel_names() for arm in arms}
    for arm in arms:
        has_prefill = any(PREFILL_KERNEL in name for name in names[arm.name])
        if has_prefill != (arm.name == "new"):
            raise SystemExit(f"{arm.name} arm at {rows} rows ran {names[arm.name]}")
    for _ in range(reps):
        for arm in arms:
            arm.samples["mix_us"].append(timed_replay(arm.mix_graph, flush_buffer))
            arm.samples["combine_us"].append(
                timed_replay(arm.combine_graph, flush_buffer)
            )
    result = {}
    for arm in arms:
        mix, combine = arm.samples["mix_us"], arm.samples["combine_us"]
        site = [a + b for a, b in zip(mix, combine)]
        result[arm.name] = {
            "mix_us": statistics.median(mix),
            "combine_us": statistics.median(combine),
            "site_us": statistics.median(site),
            "site_us_p10": quantile(site, 10),
            "site_us_p90": quantile(site, 90),
            "mixed_error": relative_error(arm.mixed, want_mixed),
            "combined_error": relative_error(arm.combined, want_combined),
            "kernels": names[arm.name],
        }
    unfused = result["unfused"]["site_us"]
    for name in ("control", "new"):
        result[name]["site_vs_unfused_pct"] = 100 * (
            result[name]["site_us"] / unfused - 1
        )
    return result


def run(args):
    import torch._inductor.config as inductor

    torch.manual_seed(0)
    flush_buffer = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32)
    results = {
        "device": torch.cuda.get_device_properties(0).gcnArchName,
        "torch": torch.__version__,
        "max_autotune": bool(inductor.max_autotune),
        "max_autotune_pointwise": bool(inductor.max_autotune_pointwise),
        "rows": {},
    }
    for rows in args.rows:
        results["rows"][rows] = run_rows(rows, args.reps, flush_buffer)
        torch.cuda.empty_cache()
    return results


def print_table(results):
    print(
        f"max_autotune={results['max_autotune']} "
        f"max_autotune_pointwise={results['max_autotune_pointwise']} "
        f"device={results['device']}"
    )
    columns = ("rows", "arm", "mix us", "combine us", "site us", "vs unfused")
    print("".join(f"{name:>12}" for name in (*columns, "mixed err", "comb err")))
    for rows, arms in results["rows"].items():
        for name, row in arms.items():
            delta = row.get("site_vs_unfused_pct")
            print(
                f"{rows:>12}{name:>12}{row['mix_us']:>12.1f}{row['combine_us']:>12.1f}"
                f"{row['site_us']:>12.1f}"
                f"{'' if delta is None else format(delta, '+.1f') + '%':>12}"
                f"{row['mixed_error']:>12.5f}{row['combined_error']:>12.5f}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[129, 160, 176, 256, 1024, 4096, 16384],
    )
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--out", help="write the results as JSON")
    parser.add_argument(
        "--also-without-autotune",
        action="store_true",
        help="repeat in a second process with TorchInductor max-autotune off",
    )
    args = parser.parse_args()

    output = {"inherited_inductor": run(args)}
    print_table(output["inherited_inductor"])
    if args.also_without_autotune:
        with tempfile.NamedTemporaryFile(suffix=".json") as child_out:
            env = {
                **os.environ,
                "TORCHINDUCTOR_MAX_AUTOTUNE": "0",
                "TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE": "0",
            }
            command = [
                sys.executable,
                __file__,
                "--reps",
                str(args.reps),
                "--out",
                child_out.name,
            ]
            command += ["--rows", *map(str, args.rows)]
            subprocess.run(command, env=env, check=True)
            with open(child_out.name) as f:
                output["without_max_autotune"] = json.load(f)["inherited_inductor"]
    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=1)


if __name__ == "__main__":
    main()
