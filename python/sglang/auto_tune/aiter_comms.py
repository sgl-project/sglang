"""Read communicator classes out of the AITER and SGLang source trees.

The scans are the competitor list. A class is timed when the tuner can
construct it. QuickReduce and FlyDSL join that set only when the requested
quant is declared in AITER.
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypedDict

DEFAULT_AITER_DIR = os.environ.get("AITER_DIR", "/sgl-workspace/aiter")
# Directory that contains the ``sglang`` package. This file lives at
# ``<root>/sglang/auto_tune/aiter_comms.py``.
DEFAULT_SGLANG_DIR = os.environ.get(
    "SGLANG_DIR", str(Path(__file__).resolve().parents[2])
)

# Directories that define the collectives. The rest of each tree is kernels.
# FlyDSL all-reduce is whichever ``quick_allreduce*.py`` that revision ships:
# ``quick_allreduce.py`` on current AITER, ``quick_allreduce_int4.py`` on
# older trees such as 90596736. The scan reads the files that exist.
_AITER_COMM_DIR = Path("aiter/dist/device_communicators")
_AITER_FLYDSL_DIR = Path("aiter/ops/flydsl")
_SGLANG_RELATIVE = (Path("sglang/srt/distributed/device_communicators"),)


def _aiter_sources(root: Path) -> Tuple[Path, ...]:
    """Communicator sources under ``root``.

    The device-communicator directory is always named. FlyDSL all-reduce
    files are the ``quick_allreduce*.py`` modules present in that tree.
    """
    sources = [_AITER_COMM_DIR]
    flydsl = root / _AITER_FLYDSL_DIR
    if flydsl.is_dir():
        sources.extend(
            sorted(
                path.relative_to(root)
                for path in flydsl.glob("quick_allreduce*.py")
            )
        )
    return tuple(sources)

_COMM_METHODS = {
    "all_reduce",
    "allreduce",
    "outplace_all_reduce",
    "custom_all_reduce",
    "quick_all_reduce",
}
_COMM_NAME = re.compile(r"(AllReduce|Allreduce|Communicator|All2AllManager)$")
_QUANT_CLASS = re.compile(r"QuickAllReduce(Int4|INT4|INT8|INT6|INT3|FP8|FP)$")
# Class names the scan may find and this tuner must not time. Empty: a class
# stays in the competition unless a run passes it to ``--skip``. That flag is
# how a class is left out when it is obsolete on the AITER pin under test, or
# when its accuracy is not acceptable. QuickAllReduceInt4 therefore still
# competes, which keeps AITER trees that have no FlyQuickAllReduce usable.
_SKIP: frozenset[str] = frozenset()


@dataclass(frozen=True)
class FoundComm:
    """One communicator class the scan found, named by its full class name."""

    qualname: str
    methods: Tuple[str, ...]
    quant: str = ""


@dataclass(frozen=True)
class AiterScan:
    """Communicators and quantizations read from one AITER tree."""

    root: Optional[str]
    comms: Tuple[str, ...]
    qr_regimes: Tuple[str, ...]
    fly_quants: Tuple[str, ...]
    found: Tuple[FoundComm, ...] = ()
    error: str = ""


def scan_aiter(root: str = DEFAULT_AITER_DIR) -> AiterScan:
    """Parse ``root`` and return the communicator classes and quant names.

    Nothing is imported. A missing tree is an empty scan with ``error`` set,
    so the tuner can still run.
    """
    path = Path(root)
    if not path.is_dir():
        return AiterScan(None, (), (), (), error=f"{root} is not a directory")
    found, regimes, fly, found_any = _scan_files(path, _aiter_sources(path))
    if not found_any:
        return AiterScan(
            str(path),
            (),
            (),
            (),
            error=f"{path} has no aiter communicator sources",
        )
    return AiterScan(
        str(path),
        tuple(item.qualname for item in found),
        regimes,
        fly,
        found,
    )


@dataclass(frozen=True)
class SglangScan:
    """Communicator classes read from one SGLang tree."""

    root: Optional[str]
    comms: Tuple[str, ...]
    found: Tuple[FoundComm, ...] = ()
    error: str = ""


def scan_sglang(root: str = DEFAULT_SGLANG_DIR) -> SglangScan:
    """Parse SGLang's device communicators. Nothing is imported.

    ``root`` is the directory that contains the ``sglang`` package.
    """
    path = Path(root)
    if not path.is_dir():
        return SglangScan(None, (), error=f"{root} is not a directory")
    found, _regimes, _fly, found_any = _scan_files(path, _SGLANG_RELATIVE)
    if not found_any:
        return SglangScan(
            str(path),
            (),
            error=f"{path} has no sglang communicator sources",
        )
    return SglangScan(str(path), tuple(item.qualname for item in found), found)


def _scanned(scan: AiterScan, sglang: SglangScan) -> Tuple[FoundComm, ...]:
    """Classes both scans returned, AITER first, each class name once."""
    found: List[FoundComm] = []
    seen: set[str] = set()
    for item in (*scan.found, *sglang.found):
        if item.qualname in seen:
            continue
        seen.add(item.qualname)
        found.append(item)
    return tuple(found)


def _call_methods(item: FoundComm) -> bool:
    """Whether the class body defines a collective this tuner can call."""
    return bool(item.quant or set(item.methods) & _COMM_METHODS)


def _entry_matches(qualname: str, entry: str) -> bool:
    """Whether one skip entry names this class.

    A full class name must match exactly. A name with no dot matches the
    class name alone, so ``QuickAllReduceInt4`` and
    ``aiter.ops.flydsl.quick_allreduce_int4.QuickAllReduceInt4`` both leave
    that class out.
    """
    short = qualname.rsplit(".", 1)[-1]
    return entry == qualname or ("." not in entry and entry == short)


def _is_skipped(qualname: str, extra: Sequence[str] = ()) -> bool:
    """Whether ``qualname`` is in ``_SKIP`` or in a ``--skip`` list."""
    return any(_entry_matches(qualname, entry) for entry in (*_SKIP, *extra))


def unmatched_skips(
    found: Sequence[str], extra: Sequence[str]
) -> Tuple[str, ...]:
    """``--skip`` entries that matched no scanned class name."""
    return tuple(
        entry
        for entry in extra
        if not any(_entry_matches(qualname, entry) for qualname in found)
    )


class ScannedParams(TypedDict):
    """What the scan knows about one class, before a process group exists."""

    qualname: str
    methods: Tuple[str, ...]
    quant_suffix: str
    quant: str
    lift_cap: bool
    cast_bf16_to_fp16: bool


class OpenerSpec(TypedDict):
    """One class the tuner will try to construct."""

    opener: Literal["scanned"]
    params: ScannedParams


def backend_plan(
    scan: AiterScan,
    sglang: SglangScan,
    quant: str,
    ignore_acr_max_size: bool = False,
    ignore_scr_max_size: bool = False,
    qr_cast_bf16_to_fp16: bool = True,
    skip: Sequence[str] = (),
) -> Tuple[Dict[str, OpenerSpec], Tuple[str, ...], Tuple[str, ...]]:
    """One opener per scanned class this tuner can call.

    The key is the full class name the scan returned. ``params`` carries
    that name and the methods the scan saw. It does not include the process
    group or the device. The second item is classes that are not opened:
    no collective to call, or a name in ``_SKIP`` or ``skip``. The third item
    is classes left out because ``quant`` is not declared for them.
    """
    hip, fly = qr_fly_competing(scan, quant)
    aiter_names = set(scan.comms)
    plan: Dict[str, OpenerSpec] = {}
    quant_left_out: List[str] = []
    skipped: List[str] = []
    for item in _scanned(scan, sglang):
        if _is_skipped(item.qualname, skip):
            skipped.append(f"{item.qualname} (skipped)")
            continue
        if not _call_methods(item):
            skipped.append(item.qualname)
            continue
        if item.quant:
            if item.quant != quant or not fly:
                quant_left_out.append(
                    f"{item.qualname} ({quant} is not its scanned quant {item.quant})"
                )
                continue
        elif "quick_all_reduce" in item.methods and not hip:
            quant_left_out.append(
                f"{item.qualname} ({quant} is not a QuickReduce regime)"
            )
            continue
        lift_cap = (
            ignore_acr_max_size
            if item.qualname in aiter_names
            else ignore_scr_max_size
        )
        plan[item.qualname] = {
            "opener": "scanned",
            "params": {
                "qualname": item.qualname,
                "methods": item.methods,
                "quant_suffix": item.quant,
                "quant": quant,
                "lift_cap": bool(lift_cap),
                "cast_bf16_to_fp16": bool(qr_cast_bf16_to_fp16),
            },
        }
    return plan, tuple(skipped), tuple(quant_left_out)


def qr_fly_competing(scan: AiterScan, quant: str) -> Tuple[bool, bool]:
    """Whether HIP QuickReduce and FlyDSL at ``quant`` join the timed set.

    HIP QuickReduce joins when ``quant`` is a ``QuickReduceRegime``. FlyDSL
    INT4 is ``FlyQuickAllReduce``. That class has no quant suffix, so it
    competes at INT4 whether or not ``QuickAllReduceInt4`` is still in the
    tree. A missing tree keeps HIP QuickReduce and keeps
    FlyDSL only for INT4.
    """
    if scan.root is None or scan.error:
        return True, quant == "INT4"
    names = {item.qualname.rsplit(".", 1)[-1] for item in scan.found}
    fly = quant == "INT4" and (
        "FlyQuickAllReduce" in names or quant in scan.fly_quants
    )
    return quant in scan.qr_regimes, fly


def short_name(qualname: str) -> str:
    """Class name from ``module.Class``. The JSON keeps the full name."""
    return qualname.rsplit(".", 1)[-1]


def source_name(qualname: str) -> str:
    """``aiter`` or ``sglang``: the package a scanned class was imported from."""
    return qualname.split(".", 1)[0]


def format_scan(scan: AiterScan, sglang: SglangScan, quant: str) -> str:
    """The log block: both communicator lists, then whether the quant competes."""
    lines: List[str] = []
    lines.extend(_list_block("Aiter", scan.root, scan.comms, scan.error))
    if scan.root and not scan.error:
        lines.append(
            "QuickReduce regimes: " + (", ".join(scan.qr_regimes) or "(none)")
        )
        lines.append(
            "FlyDSL quantizations: " + (", ".join(scan.fly_quants) or "(none)")
        )
    lines.extend(_list_block("SGLang", sglang.root, sglang.comms, sglang.error))
    hip, fly = qr_fly_competing(scan, quant)
    lines.append(
        f"Quant {quant}: HIP QuickReduce "
        f"{'competes' if hip else 'left out'}, FlyDSL "
        f"{'competes' if fly else 'left out'}"
    )
    return "\n".join(lines)


def _list_block(
    title: str, root: Optional[str], comms: Sequence[str], error: str
) -> List[str]:
    if root is None or error:
        return [f"{title} source: {error or root}"]
    lines = [f"{title} communicators in {root}:"]
    lines.extend(f"  {name}" for name in comms)
    if not comms:
        lines.append("  (none)")
    return lines


def _scan_files(
    root: Path, relatives: Sequence[Path]
) -> Tuple[Tuple[FoundComm, ...], Tuple[str, ...], Tuple[str, ...], bool]:
    found: List[FoundComm] = []
    regimes: List[str] = []
    fly: List[str] = []
    found_any = False
    for relative in relatives:
        target = root / relative
        files: Iterable[Path]
        if target.is_file():
            files = (target,)
        elif target.is_dir():
            files = sorted(target.rglob("*.py"))
        else:
            continue
        found_any = True
        for source in files:
            _read_source(root, source, found, regimes, fly)
    unique: List[FoundComm] = []
    seen: set[str] = set()
    for item in found:
        if item.qualname in seen:
            continue
        seen.add(item.qualname)
        unique.append(item)
    return (
        tuple(unique),
        tuple(dict.fromkeys(regimes)),
        tuple(dict.fromkeys(fly)),
        found_any,
    )


def _read_source(
    root: Path,
    source: Path,
    found: List[FoundComm],
    regimes: List[str],
    fly: List[str],
) -> None:
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError):
        return
    module = _module_name(root, source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name == "QuickReduceRegime":
                regimes.extend(_enum_members(node))
            matched = _QUANT_CLASS.search(node.name)
            quant = matched.group(1).upper() if matched else ""
            if quant:
                fly.append(quant)
            if _is_communicator(node):
                found.append(
                    FoundComm(
                        f"{module}.{node.name}",
                        _comm_methods(node),
                        quant,
                    )
                )
        elif isinstance(node, ast.Assign):
            fly.extend(_fly_quantizations(node))


def _module_name(root: Path, source: Path) -> str:
    relative = source.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _comm_methods(node: ast.ClassDef) -> Tuple[str, ...]:
    names: List[str] = []
    for child in node.body:
        if (
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name in _COMM_METHODS
            and child.name not in names
        ):
            names.append(child.name)
    return tuple(names)


def _is_communicator(node: ast.ClassDef) -> bool:
    if node.name.startswith("_"):
        return False
    if _COMM_NAME.search(node.name) or _QUANT_CLASS.search(node.name):
        return True
    return any(
        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        and child.name in _COMM_METHODS
        for child in node.body
    )


def _enum_members(node: ast.ClassDef) -> Sequence[str]:
    names: List[str] = []
    for child in node.body:
        target = None
        if isinstance(child, ast.Assign) and len(child.targets) == 1:
            target = child.targets[0]
        elif isinstance(child, ast.AnnAssign):
            target = child.target
        if isinstance(target, ast.Name) and target.id != "NONE":
            names.append(target.id)
    return names


def _fly_quantizations(node: ast.Assign) -> Sequence[str]:
    names = [
        target.id
        for target in node.targets
        if isinstance(target, ast.Name) and target.id == "FLY_QUANTIZATIONS"
    ]
    if not names or not isinstance(node.value, ast.Dict):
        return ()
    found: List[str] = []
    for key in node.value.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            found.append(key.value)
    return found
