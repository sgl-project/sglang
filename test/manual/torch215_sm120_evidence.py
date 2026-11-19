"""torch 2.15 / CUDA 13.4 upgrade: A/B evidence for two sm_120 (RTX 5090) regressions.

Not a regular test; run it with /rerun-test on a 1-gpu-5090 runner. It always exits 0
and writes its findings to the job log and step summary.

1. FLUX 3 video QK-norm + RoPE (test_flux3_video_qknorm_rope) stopped matching eager
   bit for bit. Runs the Triton kernel and eager F.rms_norm on identical inputs under
     A: torch 2.14.1 + Triton 3.8   (current main)
     B: torch 2.15.0 + Triton 3.9   (the upgrade)
     C: torch 2.15.0 + Triton 3.8   (isolates Triton from torch)
2. Wan VAE decode got ~3.5x slower. Times bf16 Conv3d layers shaped like the Wan 2.1
   VAE decoder under cuDNN 9.26 (torch 2.15's) and 9.24 (torch 2.14's), both on torch 2.15.

Side environments are throwaway uv venvs under /tmp, removed at the end.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KERNEL = REPO / "python/sglang/kernels/ops/diffusion/rope/flux3_video_qknorm_rope_triton.py"
NUMERICS = REPO / "python/sglang/kernels/numerics.py"
CU130 = "https://download.pytorch.org/whl/cu130"
CU134 = "https://download.pytorch.org/whl/cu134"

FLUX3_WORKER = r'''
import sys, json, torch, torch.nn.functional as F, triton
sys.path.insert(0, sys.argv[1])
from flux3k import flux3_video_qknorm_rope
inputs = torch.load(sys.argv[2])
out = {"torch": torch.__version__, "triton": triton.__version__,
       "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name()}
res = {}
for key, (qkv, cos, sin) in inputs.items():
    qkv, cos, sin = qkv.cuda(), cos.cuda(), sin.cuda()
    q, k, v = qkv.unbind(4)
    nq = F.rms_norm(q, (64,), eps=1e-5)
    nk = F.rms_norm(k, (64,), eps=1e-5)
    def rope(x):
        a, b = x.chunk(2, -1)
        return x * cos + torch.cat([-b, a], -1) * sin
    gq, gk, _ = flux3_video_qknorm_rope(qkv, cos, sin, eps=1e-5)
    res[key] = {"eager_norm_q": nq.cpu(), "eager_q": rope(nq).cpu(), "eager_k": rope(nk).cpu(),
                "kernel_q": gq.cpu(), "kernel_k": gk.cpu()}
torch.save(res, sys.argv[3])
print(json.dumps(out))
'''

CONV_WORKER = r'''
import json, torch, torch.nn as nn
torch.backends.cudnn.benchmark = False
dev = "cuda"
# Wan 2.1 VAE decoder: 3x3x3 causal Conv3d at each upsampling stage (480x832 output),
# decoded a few frames at a time with a 2-frame causal cache.
shapes = [(384, 384, 60, 104), (384, 384, 120, 208), (192, 192, 240, 416), (96, 96, 480, 832)]
res = {"torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "layers": []}
total = 0.0
for cin, cout, h, w in shapes:
    conv = nn.Conv3d(cin, cout, 3, padding=(0, 1, 1)).to(dev, torch.bfloat16)
    x = torch.randn(1, cin, 4, h, w, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        for _ in range(3):
            conv(x)
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            conv(x)
        e.record()
        torch.cuda.synchronize()
    ms = s.elapsed_time(e) / 10
    total += ms
    res["layers"].append({"shape": [cin, cout, 4, h, w], "ms": round(ms, 3)})
res["total_ms"] = round(total, 3)
print(json.dumps(res))
'''


def sh(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kw)


def uv():
    return [shutil.which("uv")] if shutil.which("uv") else [sys.executable, "-m", "uv"]


def make_venv(root, name, packages, index, extra=()):
    path = root / name
    sh(uv() + ["venv", "-q", "--python", "3.12", str(path)])
    py = path / "bin" / "python"
    sh(uv() + ["pip", "install", "-q", "-p", str(py), *packages, "numpy",
               "--extra-index-url", index, "--index-strategy", "unsafe-best-match"])
    for pkgs in extra:
        sh(uv() + ["pip", "install", "-q", "-p", str(py), "--no-deps", *pkgs])
    return str(py)


def run_json(py, code, *args):
    out = sh([py, "-c", code, *args], capture_output=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def ulp_gap(a, b):
    import torch
    ai = a.view(torch.int16).int()
    bi = b.view(torch.int16).int()
    ai = torch.where(ai < 0, -32768 - ai, ai)
    bi = torch.where(bi < 0, -32768 - bi, bi)
    return int((ai - bi).abs().max())


def flux3_inputs():
    import torch

    def tables(t, h, w):
        chunk = 16
        axis = 1.0 / (256.0 ** (torch.arange(0, chunk, 2).float() / chunk))
        inv = torch.stack([axis, axis, axis, torch.zeros(chunk // 2)])
        g = torch.meshgrid(*[torch.arange(n, dtype=torch.float32) for n in (t, h, w)], indexing="ij")
        pos = torch.stack(g + (torch.zeros_like(g[0]),), -1)
        f = torch.einsum("...a,af->...af", pos, inv).reshape(1, t, h, w, 1, -1)
        f = torch.cat([f, f], -1)
        return f.cos().bfloat16().contiguous(), f.sin().bfloat16().contiguous()

    inputs = {}
    for shape in [(1, 1, 8, 8, 4), (1, 1, 17, 20, 32), (1, 2, 8, 10, 8)]:
        b, t, h, w, heads = shape
        g = torch.Generator().manual_seed(0)
        qkv = torch.randn(b, t, h, w, 3, heads, 64, generator=g).bfloat16()
        inputs[str(shape)] = (qkv, *tables(t, h, w))
    return inputs


def main():
    import torch

    lines = ["## torch 2.15 sm_120 evidence", ""]
    root = Path(tempfile.mkdtemp(prefix="sm120-evidence-"))
    try:
        pkg = root / "pkg"
        pkg.mkdir()
        shutil.copy(NUMERICS, pkg / "numerics.py")
        src = KERNEL.read_text()
        src = src.replace("from sglang.kernels.numerics import", "from numerics import")
        src = src.replace(
            "from sglang.srt.utils.custom_op import register_custom_op",
            "register_custom_op = lambda *a, **k: (lambda f: f)",
        )
        (pkg / "flux3k.py").write_text(src)
        inputs_path = root / "flux3_inputs.pt"
        torch.save(flux3_inputs(), inputs_path)

        envs = {
            "A torch2.14.1+triton3.8": make_venv(root, "a", ["torch==2.14.1"], CU130),
            "B torch2.15.0+triton3.9": sys.executable,
            "C torch2.15.0+triton3.8": make_venv(
                root, "c", ["torch==2.15.0"], CU134, extra=[["triton==3.8.0"]]
            ),
        }

        # 1. FLUX 3 QK-norm + RoPE
        results = {}
        for name, py in envs.items():
            out = root / f"flux3_{name.split()[0]}.pt"
            info = run_json(py, FLUX3_WORKER, str(pkg), str(inputs_path), str(out))
            results[name] = (info, torch.load(out))
        lines += ["### 1. FLUX 3 QK-norm + RoPE kernel vs eager (bf16, bit-exact test)", ""]
        lines += ["| env | versions | shape | kernel == eager (q, k) | max ULP gap |", "|---|---|---|---|---|"]
        for name, (info, res) in results.items():
            for shape, r in res.items():
                eq = (torch.equal(r["kernel_q"], r["eager_q"]), torch.equal(r["kernel_k"], r["eager_k"]))
                gap = max(ulp_gap(r["kernel_q"], r["eager_q"]), ulp_gap(r["kernel_k"], r["eager_k"]))
                lines.append(f"| {name.split()[0]} | torch {info['torch']}, triton {info['triton']} | {shape} | {eq} | {gap} |")
        a, b, c = (results[n][1] for n in envs)
        lines += ["", "Cross-env comparisons (all shapes):", ""]
        for label, x, y, field in [
            ("eager F.rms_norm, A vs B (torch 2.14 vs 2.15)", a, b, "eager_norm_q"),
            ("kernel, A vs C (Triton 3.8, torch 2.14 vs 2.15)", a, c, "kernel_q"),
            ("kernel, B vs C (Triton 3.9 vs 3.8, torch 2.15)", b, c, "kernel_q"),
        ]:
            same = all(torch.equal(x[s][field], y[s][field]) for s in x)
            lines.append(f"- {label}: {'identical' if same else 'DIFFERENT'}")

        # 2. Conv3d (Wan VAE decode) under cuDNN 9.26 vs 9.24
        conv_envs = {
            "torch 2.15 + cuDNN 9.26 (default)": sys.executable,
            "torch 2.15 + cuDNN 9.24": make_venv(
                root, "d", ["torch==2.15.0"], CU134, extra=[["nvidia-cudnn-cu13==9.24.0.43"]]
            ),
            "torch 2.14.1 + cuDNN 9.24 (main)": envs["A torch2.14.1+triton3.8"],
        }
        lines += ["", "### 2. Wan VAE decoder-shaped bf16 Conv3d (3x3x3, T=4), mean of 10", ""]
        lines += ["| env | cuDNN | 384ch 60x104 | 384ch 120x208 | 192ch 240x416 | 96ch 480x832 | total ms |"]
        lines += ["|---|---|---|---|---|---|---|"]
        for name, py in conv_envs.items():
            r = run_json(py, CONV_WORKER)
            lines.append(f"| {name} | {r['cudnn']} | " + " | ".join(str(l["ms"]) for l in r["layers"]) + f" | {r['total_ms']} |")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    report = "\n".join(lines)
    print(report, flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
