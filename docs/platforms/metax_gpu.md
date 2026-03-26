# Metax GPUs (MACA)

This document describes how to run SGLang on Metax GPUs with the MACA software stack and vendor-provided PyTorch. If you hit issues, please [open an issue](https://github.com/sgl-project/sglang/issues).

## Why a separate `pyproject`

The default `python/pyproject.toml` pins NVIDIA-oriented wheels (`sglang-kernel`, FlashInfer, official CUDA PyTorch, etc.). Installing it on Metax would replace your vendor PyTorch. Use `python/pyproject_other.toml` instead—the same approach as [Moore Threads (MUSA)](mthreads_gpu.md).

## Install from source

1. Install **Metax/MACA PyTorch** (and matching `torchvision` / `torchaudio` if needed) from your vendor instructions. Confirm `python -c "import torch; print(torch.cuda.is_available())"`.

2. Swap the project file (same layout as [Moore Threads](mthreads_gpu.md)):

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
rm -f python/pyproject.toml && cp python/pyproject_other.toml python/pyproject.toml
```

3. Install SGLang **without letting pip replace your `torch`**.

`outlines`, `transformers`, `timm`, etc. depend on `torch`. A plain `pip install -e "python[srt_metax]"` may download **NVIDIA CUDA** PyTorch from PyPI if pip decides your current build is “insufficient”, which breaks MACA.

**Recommended (keeps vendor PyTorch):** install the editable package without dependencies, then install **all** `srt_metax` requirements in **one** `pip install` so the resolver sees your existing `torch` (avoid `xargs -n1`, which reinstalls dependencies per package and can pull PyPI `torch`).

```bash
cd python
pip install -e . --no-deps
deps=$(python3 <<'PY'
import tomllib
from pathlib import Path

data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
optional = data["project"]["optional-dependencies"]
metax = optional["srt_metax"]
expanded: list[str] = []
for item in metax:
    if item.startswith("sglang[") and item.endswith("]"):
        extra = item[len("sglang["):-1]
        expanded.extend(optional[extra])
    else:
        expanded.append(item)
seen: set[str] = set()
uniq: list[str] = []
for req in expanded:
    if req not in seen:
        seen.add(req)
        uniq.append(req)
print(" ".join(uniq))
PY
)
pip install $deps
```

(After the `cp` step, `python/pyproject.toml` is the former `pyproject_other.toml`, so the snippet reads the correct optional-dependencies table.)

Then verify the PyTorch build string:

```bash
python -c "import torch; print(torch.__version__)"
```

You should still see your **Metax/MACA** build (for example a `+metax…` local label). If you see a `+cu12…` PyPI-style build instead, uninstall `torch` / `torchvision` / `torchaudio` and reinstall them from Metax, then repeat the steps above.

**Alternative (single command, higher risk):**

```bash
pip install -e "python[srt_metax]"
```

Use only in a **clean** environment where the only `torch` is the vendor wheel and pip resolution keeps it.

4. **Do not** run `pip install -e "python"` using the default `pyproject.toml` in the same environment if you need to keep vendor PyTorch.

## Notes

- `sglang-kernel`, FlashInfer, and FlashAttention wheels from PyPI target NVIDIA CUDA; they are **not** part of `srt_metax`. Attention backends that rely on pure PyTorch / Triton (where supported by MACA) are the practical starting point; expect to tune flags for your stack.
- `nvidia-ml-py` is listed for tooling compatibility; GPU telemetry may not match NVIDIA NVML on Metax—treat as optional if it causes problems (`pip uninstall nvidia-ml-py`).

## Optional: development extras

```bash
pip install -e "python[dev_metax]"
```
