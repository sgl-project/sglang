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
from typing import Any, Dict

# Codecs QuickAllReduce reads from ROCM_QUICK_REDUCE_QUANTIZATION. NONE leaves
# the communicator disabled, so it is not a backend. One HIP codec is timed:
# INT4 is faster than INT8, so the tuner does not compare HIP codecs.
QR_QUANT_REGIMES = ("FP", "INT8", "INT6", "INT4")

# FlyDSL mesh QuickReduce wire formats. The default schedule is single-codec
# INT4. INT5 is mesh-only; the ring schedule accepts INT4 and INT6.
_FLYDSL_CODECS = ("int4", "int5", "int6")
_FLYDSL_ALGORITHMS = ("mesh", "ring")
_OPENERS = (
    "custom_allreduce",
    "custom_allreduce_v2",
    "nccl",
    "torch_symm_mem",
    "quick_allreduce",
    "quick_allreduce_flydsl",
)


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
            "algorithm": "mesh",
            "codec": "int4",
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
        if opener not in _OPENERS:
            raise ValueError(
                f"{name}: opener must be one of {_OPENERS}, got {opener!r}"
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
            algorithm = entry.get("algorithm", "mesh")
            codec = entry.get("codec", "int4")
            if algorithm not in _FLYDSL_ALGORITHMS:
                raise ValueError(
                    f"{name}: algorithm must be one of {_FLYDSL_ALGORITHMS}, "
                    f"got {algorithm!r}"
                )
            if codec not in _FLYDSL_CODECS:
                raise ValueError(
                    f"{name}: codec must be one of {_FLYDSL_CODECS}, got {codec!r}"
                )
            entry["algorithm"] = algorithm
            entry["codec"] = codec
        checked[name] = entry
    return checked


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
        description="Write the all-reduce selection JSON the tuner opens."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path of the JSON file to write.",
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
