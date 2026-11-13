"""Write the all-reduce selection the tuner opens.

The selection is the map from a table name to the opener and its static
arguments. Process objects (the CPU group and the device) are not in the
file; the tuner adds those when it opens the communicators.

    python -m sglang.auto_tune.generate_selection_config \
        --output selection_config.json

HIP QuickReduce keeps the table name ``quick_allreduce_<codec>``, which is
the name the server already dispatches. FlyDSL INT4 is a second entry,
``quick_allreduce_flydsl_INT4``. Edit the file to change who competes, then
pass it to ``run_allreduce --selection-config``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

# Codecs QuickAllReduce reads from ROCM_QUICK_REDUCE_QUANTIZATION. NONE leaves
# the communicator disabled, so it is not a backend. One HIP codec is timed:
# INT4 is faster than INT8, so the tuner does not compare HIP codecs.
QR_QUANT_REGIMES = ("FP", "INT8", "INT6", "INT4")

# Opener name to the static keys the tuner passes through, besides the CPU
# group and the device. ``opener`` and ``impl`` stay out of that call.
_OPENER_KWARGS = {
    "custom_allreduce": ("kind", "lift_cap"),
    "custom_allreduce_v2": (),
    "nccl": (),
    "torch_symm_mem": (),
    "quick_allreduce": ("quant", "cast_bf16_to_fp16"),
    "quick_allreduce_flydsl": (),
}
OPENER_NAMES = tuple(_OPENER_KWARGS)


def build_selection(
    qr_quant: str = "INT4",
    ignore_acr_max_size: bool = False,
    ignore_scr_max_size: bool = False,
    qr_cast_bf16_to_fp16: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """The selection map, in the order the ranks open the communicators.

    HIP QuickReduce is one entry at ``qr_quant``. FlyDSL INT4 is always a
    second entry: it is a different kernel, not another HIP codec.
    """
    if qr_quant not in QR_QUANT_REGIMES:
        raise ValueError(
            f"qr_quant must be one of {QR_QUANT_REGIMES}, got {qr_quant!r}"
        )
    return {
        "sglang_custom_allreduce": {
            "opener": "custom_allreduce",
            "kind": "sglang",
            "lift_cap": bool(ignore_scr_max_size),
        },
        "aiter_custom_allreduce": {
            "opener": "custom_allreduce",
            "kind": "aiter",
            "lift_cap": bool(ignore_acr_max_size),
        },
        "custom_allreduce_v2": {"opener": "custom_allreduce_v2"},
        "nccl": {"opener": "nccl"},
        "torch_symm_mem": {"opener": "torch_symm_mem"},
        f"quick_allreduce_{qr_quant}": {
            "opener": "quick_allreduce",
            "impl": "hip",
            "quant": qr_quant,
            "cast_bf16_to_fp16": bool(qr_cast_bf16_to_fp16),
        },
        "quick_allreduce_flydsl_INT4": {
            "opener": "quick_allreduce_flydsl",
            "impl": "flydsl",
            "quant": "INT4",
        },
    }


def _as_bool(value: Any, where: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    raise ValueError(f"{where} must be a boolean, got {value!r}")


def validate_selection(selection: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Return the selection if every entry names a known opener and its fields."""
    if not isinstance(selection, dict) or not selection:
        raise ValueError("selection config must be a non-empty JSON object")
    checked: Dict[str, Dict[str, Any]] = {}
    for name, spec in selection.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"selection names must be strings, got {name!r}")
        if not isinstance(spec, dict):
            raise ValueError(f"{name}: entry must be an object")
        opener = spec.get("opener")
        if opener not in _OPENER_KWARGS:
            raise ValueError(
                f"{name}: opener must be one of {OPENER_NAMES}, got {opener!r}"
            )
        entry = dict(spec)
        if opener == "custom_allreduce":
            if entry.get("kind") not in ("sglang", "aiter"):
                raise ValueError(
                    f"{name}: kind must be 'sglang' or 'aiter', got {entry.get('kind')!r}"
                )
            entry["lift_cap"] = _as_bool(entry.get("lift_cap", False), f"{name}.lift_cap")
        elif opener == "quick_allreduce":
            quant = entry.get("quant")
            if quant not in QR_QUANT_REGIMES:
                raise ValueError(
                    f"{name}: quant must be one of {QR_QUANT_REGIMES}, got {quant!r}"
                )
            entry["cast_bf16_to_fp16"] = _as_bool(
                entry.get("cast_bf16_to_fp16", True), f"{name}.cast_bf16_to_fp16"
            )
        elif opener == "quick_allreduce_flydsl":
            # QuickAllReduceInt4 has no codec argument. A leftover algorithm
            # or codec key from an older file is ignored.
            quant = entry.get("quant", "INT4")
            if quant != "INT4":
                raise ValueError(
                    f"{name}: FlyDSL row is QuickAllReduceInt4, got quant {quant!r}"
                )
            entry["quant"] = "INT4"
        checked[name] = entry
    return checked


def selection_open_params(
    selection: Dict[str, Dict[str, Any]],
    cpu_group,
    device,
) -> Dict[str, Dict[str, Any]]:
    """Map each table name to the opener and the kwargs that open it.

    The selection holds the static arguments. ``cpu_group`` and ``device``
    exist only after torchrun has started, so they are added here. Every rank
    passes the same selection.
    """
    selection = validate_selection(selection)
    bound: Dict[str, Dict[str, Any]] = {}
    for name, spec in selection.items():
        opener = spec["opener"]
        params = {key: spec[key] for key in _OPENER_KWARGS[opener]}
        params["cpu_group"] = cpu_group
        params["device"] = device
        bound[name] = {"opener": opener, "params": params}
    return bound


def load_selection(path: str) -> Dict[str, Dict[str, Any]]:
    """Read a selection file written by this script."""
    with open(path) as fh:
        raw = json.load(fh)
    try:
        return validate_selection(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def write_selection(path: str, selection: Dict[str, Dict[str, Any]]) -> None:
    validate_selection(selection)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(selection, fh, indent=2)
        fh.write("\n")


def _bool01(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected 0 or 1")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write the all-reduce selection JSON the tuner opens.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
example:
  python -m sglang.auto_tune.generate_selection_config \\
      --output /tmp/ar-tune/selection_config.json \\
      --qr-quant INT4 --qr-cast-bf16-to-fp16 1 \\
      --ignore-acr-max-size 1 --ignore-scr-max-size 0
""",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path of the JSON file to write. Parent directories are created.",
    )
    parser.add_argument(
        "--qr-quant",
        default="INT4",
        choices=list(QR_QUANT_REGIMES),
        help="HIP QuickReduce codec. Default INT4. FlyDSL stays INT4.",
    )
    parser.add_argument(
        "--qr-cast-bf16-to-fp16",
        type=_bool01,
        default=True,
        help="1 (default): HIP QuickReduce casts bf16 to fp16 inside the kernel.",
    )
    parser.add_argument(
        "--ignore-acr-max-size",
        type=_bool01,
        default=False,
        help="1: the AITER entry may run past its max size, up to its buffer.",
    )
    parser.add_argument(
        "--ignore-scr-max-size",
        type=_bool01,
        default=False,
        help="1: the SGLang custom all-reduce entry may run past its max size.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    selection = build_selection(
        qr_quant=args.qr_quant,
        ignore_acr_max_size=args.ignore_acr_max_size,
        ignore_scr_max_size=args.ignore_scr_max_size,
        qr_cast_bf16_to_fp16=args.qr_cast_bf16_to_fp16,
    )
    write_selection(args.output, selection)
    print(f"Wrote {len(selection)} backends to {args.output}")


if __name__ == "__main__":
    main()
