#!/usr/bin/env python3
"""Check the external operator packages DeepSeek-V4.1 needs on Ascend (A5).

Scope -- only operators that must be installed from another operator repo:

* ops-transformer       -> ``torch.ops.cann_ops_transformer.*`` (+ ``ds41.*``)
* cannbot-dsl net wheel -> ``ops.*`` Python entry points (DSL host side)
* cann-recipes-infer    -> ``torch.ops.custom.*``

CANN / torch_npu built-ins (``torch.ops.npu.*``, ``torch_npu.*``) ship with the
image and are deliberately out of scope, and so is anything only reachable from
the V4 (C4 / ratio-4) path.

Run on the target machine after installing the wheels. A missing entry exits
non-zero. ``--smoke`` additionally invokes the cheap external ops; smoke
failures are reported but never change the exit code.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import importlib.metadata as md
import os
import sys

OPS_TRANSFORMER = "ops-transformer"
CANNBOT_DSL = "cannbot-dsl (net wheel)"
RECIPES = "cann-recipes-infer"

# (repo, torch namespace path, note)
TORCH_ENTRIES = [
    (
        OPS_TRANSFORMER,
        "cann_ops_transformer.ds41.compressor",
        "low-ratio KV compress write (A5)",
    ),
    (
        OPS_TRANSFORMER,
        "cann_ops_transformer.ds41.indexer_prologue_k",
        "indexer K write, fused quant + RoPE",
    ),
    (
        OPS_TRANSFORMER,
        "cann_ops_transformer.ds41.quant_lightning_indexer",
        "indexer top-k scoring",
    ),
    (
        OPS_TRANSFORMER,
        "cann_ops_transformer.ds41.quant_lightning_indexer_metadata",
        "indexer top-k metadata",
    ),
    (
        OPS_TRANSFORMER,
        "cann_ops_transformer.ds41.quant_sparse_lightning_indexer",
        "level-two top-k inside the candidate blocks",
    ),
    (
        OPS_TRANSFORMER,
        "cann_ops_transformer.ds41.quant_sparse_lightning_indexer_metadata",
        "level-two top-k metadata",
    ),
    (
        OPS_TRANSFORMER,
        "cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla",
        "sparse MLA attention, every ratio",
    ),
    (
        OPS_TRANSFORMER,
        "cann_ops_transformer.ds41.mixed_quant_sparse_flash_mla_metadata",
        "sparse MLA metadata",
    ),
    (
        OPS_TRANSFORMER,
        "cann_ops_transformer.mhc_post",
        "mhc combine (A5, pre_from_prev)",
    ),
    (RECIPES, "custom.kv_compress_epilog_v2", "pool row write, 320B FP4 / 544B FP8"),
    (RECIPES, "custom.inplace_partial_rotary_mul", "partial RoPE for the indexer K"),
    (RECIPES, "custom.npu_hc_pre_v2", "mhc pre-mix"),
]

# (repo, module, callable, note)
MODULE_ENTRIES = [
    (
        CANNBOT_DSL,
        "ops.mixed_quant_sparse_flash_mla",
        "mixed_quant_sparse_flash_mla",
        "same op as ds41.mixed_quant_sparse_flash_mla",
    ),
    (
        CANNBOT_DSL,
        "ops.quant_lightning_indexer_dsl",
        "quant_lightning_indexer",
        "same op as ds41.quant_lightning_indexer",
    ),
    (
        CANNBOT_DSL,
        "ops.quant_lightning_indexer_metadata_dsl",
        "quant_lightning_indexer_metadata",
        "same op as ds41.quant_lightning_indexer_metadata",
    ),
    (
        CANNBOT_DSL,
        "ops.indexer_prologue_k",
        "indexer_prologue_k",
        "same op as ds41.indexer_prologue_k",
    ),
    (
        CANNBOT_DSL,
        "ops.quant_sparse_lightning_indexer_dsl",
        "quant_sparse_lightning_indexer",
        "same op as ds41.quant_sparse_lightning_indexer",
    ),
]

# Informational only: the op checks above are the authoritative signal. Dist
# names are the wheel file names with ``_`` normalised to ``-``.
PACKAGES = [
    ("cannbot-dsl", CANNBOT_DSL, "DSL framework wheel (cannbot_dsl-*.whl)"),
    ("cannbotdsl", CANNBOT_DSL, "alternate spelling, in case it is unnormalised"),
    ("cannbot-arena-net", CANNBOT_DSL, "net-ops wheel (provides ops.*)"),
    (
        "cannbot-arena-net-ops",
        CANNBOT_DSL,
        "legacy net-ops wheel, must NOT be co-installed",
    ),
    ("cann-ops-transformer", OPS_TRANSFORMER, "torch_extension wheel"),
    ("custom-ops", RECIPES, "recipes torch_ops_extension wheel"),
]

# Modules whose import performs the torch namespace registration. The list is
# extended at run time with the top-level packages of the installed dists, so a
# renamed entry point is discovered instead of guessed.
ACTIVATION = [
    "torch_npu",
    "cann_ops_transformer",
    "cann_ops_transformer.ops",
    "cann_ops_transformer.ops.ds41",
    "custom_ops",
    "torch_ops_extension",
]


def _resolve_torch(path: str):
    """Walk a dotted ``torch.ops`` path; return (object, problem)."""
    try:
        import torch
    except ImportError as exc:
        return None, f"torch not importable: {exc}"

    obj = torch.ops
    for part in path.split("."):
        if not hasattr(obj, part):
            return _resolve_ds41_module(path, f"'{part}' not registered")
        obj = getattr(obj, part)
    return obj, None


def _resolve_ds41_module(path: str, problem: str):
    """Fallback lookup on the Python module ``_ds41_ops()`` actually returns.

    On the A5 install part of the ``ds41`` namespace is callable through
    ``cann_ops_transformer.ops.ds41`` but absent from the nested ``torch.ops``
    namespace (observed for ``indexer_prologue_k`` and
    ``quant_sparse_lightning_indexer``), so check the module before a miss.
    """
    prefix = "cann_ops_transformer.ds41."
    if not path.startswith(prefix):
        return None, problem
    try:
        from cann_ops_transformer.ops import ds41
    except ImportError as exc:
        return None, f"{problem} (no cann_ops_transformer.ops.ds41: {exc})"
    op = getattr(ds41, path[len(prefix) :], None)
    if op is None:
        return None, problem
    return op, None


def _schema_of(op) -> str:
    packet = getattr(op, "default", None)
    schema = getattr(packet, "_schema", None)
    if schema is None:
        return ""
    text = str(schema)
    return text if len(text) <= 130 else text[:130] + " ..."


def _check_env() -> None:
    print("== environment ==")
    print(f"python      : {sys.version.split()[0]}")
    try:
        import torch

        print(f"torch       : {torch.__version__}")
    except Exception as exc:  # pragma: no cover - env probe
        print(f"torch       : import failed: {exc}")
        return

    try:
        import torch_npu  # noqa: F401

        print(f"torch_npu   : {getattr(torch_npu, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"torch_npu   : import failed: {exc}")

    print(f"CANN        : {_cann_version()}")
    print(f"npu devices : {_device_summary()}")
    print(f"arch35      : {_is_arch35()}")

    for name in ("ASCEND_HOME_PATH", "ASCEND_CUSTOM_OPP_PATH", "PYTHONPATH"):
        value = os.environ.get(name)
        if value:
            print(f"{name:12}: {value}")
    for name in (
        "CANNBOTDSL_NATIVE_BINARY_MODE",
        "CANNBOTDSL_WHEEL",
        "CANNBOTDSL_ROOT",
        "CANNBOTDSL_DS41_PROFILES",
    ):
        print(f"{name:12}: {os.environ.get(name, '<unset>')}")
    print()


def _cann_version() -> str:
    candidates = [
        os.environ.get("ASCEND_HOME_PATH", "") + "/version.cfg",
        "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
    ]
    candidates += sorted(glob.glob("/usr/local/Ascend/cann-*/version.cfg"))
    for path in candidates:
        if path and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8", errors="replace") as handle:
                    lines = [ln.strip() for ln in handle if "version" in ln.lower()]
            except OSError:
                continue
            if lines:
                return f"{path} -> {' | '.join(lines[:3])}"
    return "version.cfg not found"


def _device_summary() -> str:
    try:
        import torch

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            return "none (torch.npu unavailable)"
        count = torch.npu.device_count()
        return f"{count} x {torch.npu.get_device_name(0)}"
    except Exception as exc:
        return f"probe failed: {exc}"


def _is_arch35() -> str:
    try:
        import acl

        return str(acl.rt.get_device_info(0, 601) == (3510, 0))
    except Exception as exc:
        return f"probe failed: {exc}"


def _check_packages() -> None:
    print("== packages (informational) ==")
    for dist, repo, note in PACKAGES:
        try:
            version = md.version(dist)
        except md.PackageNotFoundError:
            print(f"absent  {dist:<28} ({repo}) {note}")
            continue
        except Exception as exc:  # pragma: no cover - broken metadata
            print(f"BROKEN  {dist:<28} metadata error: {exc}")
            continue
        print(f"OK      {dist:<28} {version} ({repo})")
    print()


def _discovered_modules() -> list:
    """Top-level packages of the installed op dists, as extra candidates."""
    found = []
    for dist, _, _ in PACKAGES:
        try:
            files = md.files(dist) or []
        except md.PackageNotFoundError:
            continue
        for entry in files:
            parts = str(entry).split("/")
            if len(parts) == 2 and parts[1] == "__init__.py" and parts[0] not in found:
                found.append(parts[0])
    return found


def _activate() -> None:
    # Importing cann_ops_transformer triggers a JIT compile of every op under
    # ``$TORCH_EXTENSIONS_DIR/<op>`` on first use, so announce each import
    # before it starts; otherwise a slow build looks like a hang.
    print("== registration ==")
    for name in ACTIVATION + _discovered_modules():
        if name in sys.modules:
            continue
        print(f"        importing {name} ...", flush=True)
        try:
            importlib.import_module(name)
            print(f"OK      import {name}")
        except Exception as exc:
            print(f"skip    import {name}: {type(exc).__name__}: {exc}")
    print()


def _check_torch_entries() -> list:
    print("== external torch ops ==")
    missing = []
    for repo, path, note in TORCH_ENTRIES:
        op, problem = _resolve_torch(path)
        if op is None:
            missing.append(path)
            print(f"MISSING {path:<52} {problem} ({repo})")
            continue
        print(f"OK      {path:<52} {_schema_of(op) or '<no default overload>'}")
        print(f"                {note}")
    print()
    return missing


def _check_modules() -> list:
    print("== cannbot-dsl net modules ==")
    missing = []
    for repo, module, attr, note in MODULE_ENTRIES:
        try:
            mod = importlib.import_module(module)
        except Exception as exc:
            missing.append(module)
            print(f"MISSING {module:<46} import failed: {type(exc).__name__}: {exc}")
            continue
        func = getattr(mod, attr, None)
        if not callable(func):
            missing.append(module)
            print(f"MISSING {module:<46} '{attr}' not callable on the module")
            continue
        print(f"OK      {module:<46} {attr} -> {getattr(func, '__module__', '?')}")
        print(f"                {note}")
    print()
    return missing


def _smoke() -> None:
    try:
        import torch
    except ImportError as exc:
        print(f"smoke: skipped, torch not importable: {exc}")
        return

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("smoke: skipped, no usable NPU device")
        return
    device = torch.device("npu:0")
    print("== smoke (informational) ==")

    try:
        torch.manual_seed(0)
        tokens, heads, head_dim = 4, 2, 8
        q = torch.randn(tokens, heads, head_dim, dtype=torch.bfloat16, device=device)
        cos = torch.randn(tokens, 1, 1, head_dim, dtype=torch.float32, device=device)
        sin = torch.randn(tokens, 1, 1, head_dim, dtype=torch.float32, device=device)
        before = q.clone()
        torch.ops.custom.inplace_partial_rotary_mul(
            q.unsqueeze(1),
            cos,
            sin,
            rotary_mode="interleave",
            partial_slice=[0, head_dim],
        )
        torch.npu.synchronize()
        changed = not torch.equal(before, q)
        print(f"PASS    custom.inplace_partial_rotary_mul (output changed={changed})")
    except Exception as exc:
        print(f"FAIL    custom.inplace_partial_rotary_mul: {type(exc).__name__}: {exc}")

    try:
        seq_len, batch = 8, 1
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=device)
        metadata = torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata(
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=None,
            seqused_k=seqused_k,
            cmp_residual_k=None,
            batch_size=batch,
            num_heads_q=32,
            num_heads_k=1,
            head_dim=128,
            topk=2048,
            mask_mode=3,
            cmp_ratio=1,
            layout_q="TND",
            layout_k="PA_BBND",
        )
        torch.npu.synchronize()
        print(
            f"PASS    ds41.quant_lightning_indexer_metadata -> shape {tuple(metadata.shape)}, "
            f"dtype {metadata.dtype}"
        )
    except Exception as exc:
        print(
            f"FAIL    ds41.quant_lightning_indexer_metadata: {type(exc).__name__}: {exc}"
        )
    print("smoke results are informational; the exit code only reflects missing ops\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="also invoke the cheap external ops on device",
    )
    parser.add_argument(
        "--skip-activation",
        action="store_true",
        help="do not import the op packages (skip the JIT build)",
    )
    args = parser.parse_args()

    _check_env()
    _check_packages()
    if not args.skip_activation:
        _activate()
    missing = _check_torch_entries()
    missing += _check_modules()

    if args.smoke:
        _smoke()

    print("== summary ==")
    if missing:
        print(f"{len(missing)} entry/entries missing:")
        for path in missing:
            print(f"  - {path}")
        if any(path.startswith("cann_ops_transformer.ds41.") for path in missing):
            print(
                "hint: the installed cann_ops_transformer wheel was built without the"
            )
            print("      experimental ops that create the ds41 module. Rebuild with:")
            print("      bash build.sh --torch_extension --experimental \\")
            print(
                "        --ops=compressor_v2,indexer_prologue_k_dsl,"
                "quant_lightning_indexer_dsl,mixed_quant_sparse_flash_mla_dsl \\"
            )
            print("        --vendor_name=ds41")
        return 1
    print("all DeepSeek-V4.1 external ops are registered")
    return 0


if __name__ == "__main__":
    sys.exit(main())
