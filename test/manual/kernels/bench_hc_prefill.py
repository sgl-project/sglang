"""MI355X timing of Qwen4Exp gated hyper-connection sites at verify and prefill sizes.

A site is built as the model builds it, at Qwen3.8-Flash-Next's geometry (HC 4, hidden
2560, low-rank 320, per-branch norm, BF16) with random weights. Arms:

- unfused: the unfused mix and combine (SGLANG_USE_HIP_HC_PREFILL_TRITON off), with the
  TorchInductor settings of the environment;
- bmm: the flag on, with the down projection as torch.bmm at every row count;
- control: bmm again, a second site with the same weights, which sets the noise floor;
- split_k: the flag on as shipped: the split-K down kernel up to _SPLIT_K_MAX_ROWS rows,
  torch.bmm above.

Every replay follows a 512 MB write that evicts L2 and the MALL, as the attention or MoE
block between a site's mix and combine does. Two measurements, the arms interleaved per
repetition and each time the median:

- per site: one site's mix and its combine, each replayed as a CUDA graph;
- per verify: --chain-sites distinct sites (each one's mix, then its combine of the mix's
  output, then the next site) captured as one CUDA graph, up to --chain-max-rows rows; the
  bmm, control and split_k arms only. Qwen3.8-Flash-Next's 96 target-model sites are
  96 / --chain-sites replays.

Every arm's mixed input, inject logits and combined residual are compared with an FP32
reference built from the site's weights. An error is max |actual - ref| / max |ref|
(the max error) or RMS(actual - ref) / RMS(ref) (the RMS error), over all elements.

TorchInductor reads TORCHINDUCTOR_MAX_AUTOTUNE at import, so --also-without-autotune
repeats the run in a second process with it and TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE set
to 0. The kernel names of one call of every arm are recorded, and the run fails if an
arm runs other kernels than its own.

    python test/manual/kernels/bench_hc_prefill.py --also-without-autotune --out out.json
"""

import argparse
import contextlib
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
TARGET_MODEL_SITES = 96
PREFILL_KERNEL = "_hc_prefill_"
SPLIT_K_KERNEL = "_hc_prefill_down_kernel"
HIPBLASLT_GEMM_KERNEL = "Cijk_"
ARMS = ("unfused", "bmm", "control", "split_k")


def make_site(enabled, seed=0):
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
    gen = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        for param in site.parameters():
            param.copy_(0.02 * torch.randn(param.shape, generator=gen, device="cuda"))
    return site


@contextlib.contextmanager
def split_k_up_to(max_rows):
    """The prefill path's split-K row limit while an arm is built and captured; a
    captured graph does not read it again."""
    from sglang.kernels.ops.gemm import hc_prefill_triton

    shipped = hc_prefill_triton._SPLIT_K_MAX_ROWS
    hc_prefill_triton._SPLIT_K_MAX_ROWS = max_rows
    try:
        yield
    finally:
        hc_prefill_triton._SPLIT_K_MAX_ROWS = shipped


def arm_context(arm_name):
    """bmm and control run the path as it is above the split-K limit."""
    if arm_name in ("bmm", "control"):
        return split_k_up_to(0)
    return contextlib.nullcontext()


def expected_kernels(arm_name, rows):
    """(substring, must run) pairs for the kernel names of one call of an arm."""
    from sglang.kernels.ops.gemm import hc_prefill_triton

    if arm_name == "unfused":
        return [(PREFILL_KERNEL, False)]
    split_k = arm_name == "split_k" and rows <= hc_prefill_triton._SPLIT_K_MAX_ROWS
    return [
        (PREFILL_KERNEL, True),
        (SPLIT_K_KERNEL, split_k),
        (HIPBLASLT_GEMM_KERNEL, not split_k),
    ]


def reference_mix(site, h):
    """The mix of a site in FP32 from its parameters: mixed [M, HS] and inject logits
    [M, HC]. ``h`` may already be FP32."""
    rows = h.shape[0]
    hf = h.float().view(rows, HC, HS)
    rstd = torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + EPS)
    normed = (hf * rstd).reshape(rows, K) * (1.0 + site.hc_norm.weight.float())
    t = torch.nn.functional.silu(
        normed @ site.input_mix_weight_down.weight.float().T / HC
    )
    gate = torch.sigmoid(t @ site.input_mix_weight_up.weight.float().T)
    mixed = (gate * normed).view(rows, HC, HS).mean(1)
    return mixed, normed @ site.block_inject_weight.weight.float().T


def reference_combine(h, logits, y):
    """h + 2 sigmoid(logits / HC) * y per branch in FP32: [M, K]."""
    rows = h.shape[0]
    inject = 2 * torch.sigmoid(logits / HC)
    combined = h.float().view(rows, HC, HS) + inject[..., None] * y.float()[:, None, :]
    return combined.reshape(rows, K)


def errors(actual, ref):
    """(max error, RMS error) of ``actual`` against the FP32 ``ref``."""
    diff = actual.float() - ref
    return (
        (diff.abs().max() / ref.abs().max()).item(),
        (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item(),
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


def summarize(samples):
    return {
        "median": statistics.median(samples),
        "p10": quantile(samples, 10),
        "p90": quantile(samples, 90),
    }


def kernel_names(call):
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    return sorted(
        {
            event.name[:100]
            for event in prof.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        }
    )


def check_kernels(arm_name, rows, names):
    for substring, must_run in expected_kernels(arm_name, rows):
        if any(substring in name for name in names) != must_run:
            raise SystemExit(
                f"{arm_name} arm at {rows} rows: {substring!r} "
                f"{'did not run' if must_run else 'ran'}; kernels: {names}"
            )


class SiteArm:
    """One site with its mix and combine captured as CUDA graphs."""

    def __init__(self, name, site, h, y):
        self.name, self.site, self.h, self.y = name, site, h, y
        with arm_context(name):
            # Compile, and build the folded weights, outside the capture.
            for _ in range(3):
                self.site.combine(self.y, self.site.mix(self.h)[1])
            torch.cuda.synchronize()
            self.mix_graph, self.combine_graph = (
                torch.cuda.CUDAGraph(),
                torch.cuda.CUDAGraph(),
            )
            with torch.cuda.graph(self.mix_graph):
                self.mixed, self.residuals = self.site.mix(self.h)
            with torch.cuda.graph(self.combine_graph):
                self.combined = self.site.combine(self.y, self.residuals)
            self.kernels = kernel_names(
                lambda: self.site.combine(self.y, self.site.mix(self.h)[1])
            )
        self.samples = {"mix_us": [], "combine_us": []}

    def replay(self, flush_buffer):
        self.samples["mix_us"].append(timed_replay(self.mix_graph, flush_buffer))
        self.samples["combine_us"].append(
            timed_replay(self.combine_graph, flush_buffer)
        )

    def inject_logits(self):
        """The prefill path returns them; the unfused path's are what its combine forms
        from the normed residual in BF16."""
        if self.residuals[1] is None:
            return self.residuals[2]
        return torch.nn.functional.linear(
            self.residuals[1].float(), self.site.block_inject_weight.weight.float()
        )


class ChainArm:
    """A chain of distinct sites captured as one CUDA graph."""

    def __init__(self, name, sites, h):
        self.name, self.sites, self.h = name, sites, h
        with arm_context(name):
            for _ in range(3):
                self.run()
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.run()
        self.samples = []

    def run(self):
        h = self.h
        for site in self.sites:
            mixed, residuals = site.mix(h)
            h = site.combine(mixed, residuals)
        return h

    def replay(self, flush_buffer):
        self.samples.append(timed_replay(self.graph, flush_buffer))


def run_rows(rows, args, flush_buffer):
    h = torch.randn((rows, K), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((rows, HS), device="cuda", dtype=torch.bfloat16)
    site_arms = [
        SiteArm(name, make_site(name in ("bmm", "control", "split_k")), h, y)
        for name in ARMS
    ]
    want_mixed, want_logits = reference_mix(site_arms[0].site, h)
    want_combined = reference_combine(h, want_logits, y)
    for arm in site_arms:
        check_kernels(arm.name, rows, arm.kernels)
    chain_arms = []
    if rows <= args.chain_max_rows:
        chain_sites = [make_site(True, seed=seed) for seed in range(args.chain_sites)]
        chain_arms = [ChainArm(name, chain_sites, h) for name in ARMS[1:]]
    for arm in chain_arms:
        arm.replay(flush_buffer)  # the first replay of a graph is not timed
        arm.samples.clear()
    for _ in range(args.reps):
        for arm in site_arms:
            arm.replay(flush_buffer)
        for arm in chain_arms:
            arm.replay(flush_buffer)
    per_site, per_verify = {}, {}
    for arm in site_arms:
        mix, combine = arm.samples["mix_us"], arm.samples["combine_us"]
        site = [a + b for a, b in zip(mix, combine)]
        per_site[arm.name] = {
            "mix_us": statistics.median(mix),
            "combine_us": statistics.median(combine),
            "site_us": statistics.median(site),
            "site_us_p10": quantile(site, 10),
            "site_us_p90": quantile(site, 90),
            **{
                f"{quantity}_{kind}_err": value
                for quantity, actual, ref in (
                    ("mixed", arm.mixed, want_mixed),
                    ("logits", arm.inject_logits(), want_logits),
                    ("combined", arm.combined, want_combined),
                )
                for kind, value in zip(("max", "rms"), errors(actual, ref))
            },
            "kernels": arm.kernels,
        }
    for name in ("bmm", "control", "split_k"):
        base = per_site["unfused"]["site_us"]
        per_site[name]["site_vs_unfused_pct"] = 100 * (
            per_site[name]["site_us"] / base - 1
        )
    for name in ("control", "split_k"):
        per_site[name]["site_vs_bmm_pct"] = 100 * (
            per_site[name]["site_us"] / per_site["bmm"]["site_us"] - 1
        )
    for arm in chain_arms:
        stats = summarize(arm.samples)
        scale = TARGET_MODEL_SITES / args.chain_sites * 1e-3
        per_verify[arm.name] = {
            "chain_us": stats["median"],
            "site_us": stats["median"] / args.chain_sites,
            "verify_ms": stats["median"] * scale,
            "verify_ms_p10": stats["p10"] * scale,
            "verify_ms_p90": stats["p90"] * scale,
        }
    for name in per_verify.keys() - {"bmm"}:
        per_verify[name]["verify_ms_saved_vs_bmm"] = (
            per_verify["bmm"]["verify_ms"] - per_verify[name]["verify_ms"]
        )
    saved_per_site = per_site["bmm"]["site_us"] - per_site["split_k"]["site_us"]
    per_site["split_k"]["verify_ms_saved_vs_bmm"] = (
        saved_per_site * TARGET_MODEL_SITES * 1e-3
    )
    return {"per_site": per_site, "per_verify": per_verify}


def run(args):
    import torch._inductor.config as inductor

    torch.manual_seed(0)
    flush_buffer = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.float32)
    results = {
        "device": torch.cuda.get_device_properties(0).gcnArchName,
        "torch": torch.__version__,
        "max_autotune": bool(inductor.max_autotune),
        "max_autotune_pointwise": bool(inductor.max_autotune_pointwise),
        "chain_sites": args.chain_sites,
        "rows": {},
    }
    for rows in args.rows:
        results["rows"][rows] = run_rows(rows, args, flush_buffer)
        torch.cuda.empty_cache()
    return results


def print_table(results):
    print(
        f"max_autotune={results['max_autotune']} "
        f"max_autotune_pointwise={results['max_autotune_pointwise']} "
        f"device={results['device']} chain_sites={results['chain_sites']}"
    )
    columns = (
        "rows",
        "arm",
        "mix us",
        "combine us",
        "site us",
        "vs unfused",
        "vs bmm",
        "mixed max",
        "logits max",
        "comb max",
        "verify ms",
        "saved site",
        "saved chain",
    )
    print("".join(f"{name:>12}" for name in columns))

    def pct(value):
        return "" if value is None else format(value, "+.1f") + "%"

    def ms(value):
        return "" if value is None else format(value, ".3f")

    for rows, measured in results["rows"].items():
        for name, row in measured["per_site"].items():
            chain = measured["per_verify"].get(name, {})
            print(
                f"{rows:>12}{name:>12}{row['mix_us']:>12.1f}{row['combine_us']:>12.1f}"
                f"{row['site_us']:>12.1f}"
                f"{pct(row.get('site_vs_unfused_pct')):>12}"
                f"{pct(row.get('site_vs_bmm_pct')):>12}"
                f"{row['mixed_max_err']:>12.5f}{row['logits_max_err']:>12.5f}"
                f"{row['combined_max_err']:>12.5f}"
                f"{ms(chain.get('verify_ms')):>12}"
                f"{ms(row.get('verify_ms_saved_vs_bmm')):>12}"
                f"{ms(chain.get('verify_ms_saved_vs_bmm')):>12}"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=int,
        nargs="+",
        default=[
            129,
            144,
            160,
            176,
            192,
            208,
            224,
            256,
            288,
            320,
            384,
            448,
            512,
            513,
        ],
    )
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--chain-sites", type=int, default=48)
    parser.add_argument(
        "--chain-max-rows",
        type=int,
        default=1024,
        help="no chains above this row count (a verify batch has at most 512 rows)",
    )
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
                "--chain-sites",
                str(args.chain_sites),
                "--chain-max-rows",
                str(args.chain_max_rows),
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
