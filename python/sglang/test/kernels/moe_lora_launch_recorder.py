"""Test-only recorder of the kernel launches of a MoE LoRA forward.

Nothing in the serving path imports this module. Inside ``record_launches()``
it wraps, and on exit restores:

* ``triton.runtime.jit.JITFunction.run``: every Triton launch (the dispatch
  fills, the activation, LoRA and finalize kernels, the Triton row quantizers,
  the stock fused-MoE kernel);
* the call of a compiled CuTeDSL function (``CudaDialectJitCompiledFunction``):
  every CuTeDSL launch, the grouped base GEMMs among them;
* ``moe_wna16_marlin_gemm`` on its defining module: the Marlin GEMM;
* ``LoraWorkspace.tensor``: only to learn the name of each workspace buffer.

Launches that go another way are not recorded: the stock FP8 group quantizer
(a C++ JIT kernel) and plain torch operations. Under CUDA graph capture the
Python launch code runs once, so a capture is recorded and a replay is not.

A launch is compared in two ways.

Its *structure* is the kernel, the grid, the non-tensor arguments and, per
tensor operand, the view the kernel was handed: shape, dtype, strides, the
name of the buffer behind it when that buffer has one (a workspace buffer, or
one the test bound) and the byte offset of the view in its storage, so two
equal-shaped slices of one allocation are different operands. What a launch
kind does not expose is recorded as ``UNRECORDED``, never as an empty value:

* Triton: everything above, the grid included.
* Marlin: the operands; the grid is set inside the C++ kernel.
* CuTeDSL: the grid is set inside the compiled function. An operand is a
  runtime tensor that carries its address, shape, element type, strides and
  how it was specialized (leading dimension, dynamic shape and stride masks),
  not its storage: the offset is recorded when the address lies in a named
  buffer and is ``UNRECORDED`` otherwise (the weights, a provider's own
  scale buffers).

Its *boundary* is a digest of every buffer the test has declared defined at
that moment, taken before the kernel runs over the buffer's comparable region:
the whole of a pair-major activation, the rows of the valid pairs gathered in
pair order for a provider-ordered buffer (``rows_by_pair``), never padding and
never storage whose producer has not run. The dispatch may place the same pair
on different physical rows in two forwards on identical inputs; gathered by
logical pair the two still compare equal, while a forward that hands a kernel
different valid data does not. Boundaries are not taken during graph capture,
where reading a buffer back is not allowed. Only the buffers a test declares
are digested: a boundary says nothing about any other operand.

``LaunchRecorder.artifact`` turns one recorded case into a JSON record (the
calls, the boundaries, the digests of the outputs the test names, the fixture,
the revision, the device and the library versions); ``first_difference`` names
the first call, operand, declared buffer or output at which two records of
one case part. A case is ``numeric`` when its boundaries and outputs repeat
on one device and are compared, or ``structure`` when only its launch
structure is (a reused test that owns its buffers). Run as a module with two
artifact files, or two directories of them, to compare a tree with another::

    python -m sglang.test.kernels.moe_lora_launch_recorder before.json after.json

The comparison fails closed. Two records compare only if both hold evidence
(launches, and for a numeric case a boundary or output digest) and agree in
format, check kind, fixture, device and library versions; the revision is
free to differ, since comparing two revisions is the purpose. Anything else
is ``NOT-COMPARABLE`` and is never reported as the same: ``first_difference``
raises ``NotComparable``, and the module exits 0 only when every case of both
sides was compared and is the same, 1 when a case differs or exists on one
side only, 3 when a path holds no records or cannot be read or a case is not
comparable, 2 on a usage error.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
from typing import Any, Callable

import torch

_SCALARS = (bool, int, float, str, type(None))
UNRECORDED = "unrecorded"  # a field this launch kind does not expose
# Two records compare only where these agree; the revision is free to differ.
_SAME_CONTEXT = ("format", "check", "fixture", "device", "environment")
_RECORD_FIELDS = (*_SAME_CONTEXT, "revision", "calls", "outputs")


class NotComparable(ValueError):
    """Two launch records that cannot be compared (``not_comparable``)."""


# What the recorder sees of each launch kind, stated in every artifact.
COVERAGE = {
    "triton": "grid, constants; per tensor operand shape, dtype, strides, "
    "buffer name and storage byte offset",
    "marlin": "constants; per tensor operand shape, dtype, strides, buffer name "
    "and storage byte offset; grid unrecorded (set inside the kernel)",
    "cutedsl": "constants; per operand shape, element type, strides, leading "
    "dimension and dynamic masks; buffer name and byte offset only inside a "
    "named buffer; grid unrecorded (set inside the compiled function)",
    "boundaries": "only the buffers the test declares, over their comparable "
    "regions; none during graph capture",
    "not recorded": "the stock FP8 group quantizer (a C++ JIT kernel), plain "
    "torch operations, graph replays",
}


def digest(tensor: torch.Tensor) -> str:
    """Short hash of a tensor's bytes; the dtype is part of the bytes only."""
    data = tensor.detach().contiguous().reshape(-1).cpu()
    return hashlib.sha256(data.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def rows_by_pair(
    buffer: torch.Tensor, pair_to_row: torch.Tensor, valid_pairs: torch.Tensor
) -> torch.Tensor:
    """The rows of a provider-ordered buffer for the valid pairs, in pair
    order. ``buffer`` is [..., width] with its leading dims flattening to the
    provider's row numbering; ``valid_pairs`` marks the pairs that have a local
    expert (``topk_ids >= 0``, flattened): the map of any other pair is not
    defined and its row is not read."""
    rows = pair_to_row.reshape(-1)[valid_pairs.reshape(-1)].long()
    return buffer.reshape(-1, buffer.shape[-1])[rows]


def _span(tensor: torch.Tensor) -> tuple[int, int]:
    storage = tensor.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


class LaunchRecorder:
    def __init__(self) -> None:
        self.launches: list[dict[str, Any]] = []
        self._named: dict[int, tuple[int, str]] = {}  # storage start -> (bytes, name)
        self._bound: dict[str, tuple[torch.Tensor, Callable | None]] = {}
        self._defined: list[str] = []

    def bind(
        self, name: str, tensor: torch.Tensor, region: Callable | None = None
    ) -> None:
        """Name a buffer; ``region`` maps it to its comparable region (default:
        all of it). It enters the boundaries only between define and retire."""
        self._bound[name] = (tensor, region)
        start, size = _span(tensor)
        self._named[start] = (size, name)

    def _locate(self, address: int) -> tuple[str | None, int | None]:
        """The named buffer holding ``address`` and its first byte."""
        for start, (size, name) in self._named.items():
            if start <= address < start + max(size, 1):
                return name, start
        return None, None

    def define(self, *names: str) -> None:
        """The producers of these buffers have run: their comparable regions
        hold this forward's data from here on."""
        for name in names:
            if name not in self._defined:
                self._defined.append(name)

    def retire(self, *names: str) -> None:
        """Their last consumer has run; the storage may be reused."""
        self._defined = [name for name in self._defined if name not in names]

    def _record(self, kind: str, kernel: str, grid: Any, args, kwargs) -> None:
        """One operand is (argument, shape, dtype, strides, buffer name,
        byte offset of the view in its storage, specialization)."""
        constants, operands = [], []
        for key, value in [*enumerate(args), *sorted(kwargs.items())]:
            if isinstance(value, torch.Tensor):
                operands.append(
                    (
                        key,
                        tuple(value.shape),
                        str(value.dtype),
                        tuple(value.stride()),
                        self._locate(value.data_ptr())[0],
                        value.data_ptr() - value.untyped_storage().data_ptr(),
                        None,
                    )
                )
            elif isinstance(value, _SCALARS):
                constants.append((key, value))
            elif hasattr(value, "data_ptr") and hasattr(value, "shape"):
                # A CuTeDSL runtime tensor: a DLPack view of a torch tensor
                # that keeps the view's address but not its storage.
                address = int(value.data_ptr)
                name, start = self._locate(address)
                dtype = getattr(value, "element_type", None)
                strides = getattr(value, "stride", None)
                operands.append(
                    (
                        key,
                        tuple(value.shape),
                        getattr(dtype, "__name__", UNRECORDED),
                        UNRECORDED if strides is None else tuple(strides),
                        name,
                        UNRECORDED if start is None else address - start,
                        tuple(
                            getattr(value, field, UNRECORDED)
                            for field in (
                                "leading_dim",
                                "dynamic_shapes_mask",
                                "dynamic_strides_mask",
                            )
                        ),
                    )
                )
            else:  # a stream handle and the like: the type only
                operands.append((key, (), type(value).__name__, (), None, None, None))
        boundary = {}
        if not torch.cuda.is_current_stream_capturing():
            for name in self._defined:
                tensor, region = self._bound[name]
                boundary[name] = digest(tensor if region is None else region(tensor))
        self.launches.append(
            dict(
                kind=kind,
                kernel=kernel,
                grid=(
                    tuple(int(extent) for extent in grid)
                    if isinstance(grid, (tuple, list))
                    else UNRECORDED
                ),
                constants=tuple(constants),
                operands=tuple(operands),
                boundary=boundary,
            )
        )

    def structure(self) -> list[tuple]:
        return [
            (x["kind"], x["kernel"], x["grid"], x["constants"], x["operands"])
            for x in self.launches
        ]

    def structure_digest(self) -> str:
        """One short hash of the whole launch structure: a run's record of
        what it launched, comparable with another run of the same case."""
        return hashlib.sha256(repr(self.structure()).encode()).hexdigest()[:16]

    def boundaries(self) -> list[tuple[str, dict[str, str]]]:
        return [(x["kernel"], x["boundary"]) for x in self.launches]

    def kinds(self) -> set[str]:
        return {x["kind"] for x in self.launches}

    def artifact(
        self,
        case: str,
        *,
        check: str,
        fixture: dict[str, Any],
        outputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """This case as a JSON record another tree's record compares with.
        ``check`` is "numeric" (the boundaries and the ``outputs`` digests
        repeat on one device and are compared) or "structure" (only the
        launch structure is); ``outputs`` are already restricted to their
        comparable regions; ``fixture`` is what generated the inputs."""
        if check not in ("numeric", "structure"):
            raise ValueError(f"unknown check {check!r}")
        if not self.launches:
            raise ValueError(f"{case}: no launch was recorded, nothing to export")
        if check == "numeric" and not (
            outputs or any(x["boundary"] for x in self.launches)
        ):
            raise ValueError(
                f"{case}: a numeric record needs a declared buffer or a named output"
            )
        operands = [operand for x in self.launches for operand in x["operands"]]
        record = dict(
            format=1,
            case=case,
            check=check,
            fixture=fixture,
            **_context(),
            coverage=dict(
                COVERAGE,
                launches=len(self.launches),
                launches_without_grid=sum(
                    x["grid"] == UNRECORDED for x in self.launches
                ),
                tensor_operands=sum(x[5] is not None for x in operands),
                tensor_operands_without_offset=sum(
                    x[5] == UNRECORDED for x in operands
                ),
                declared_buffers=sorted(self._bound),
            ),
            calls=[
                {key: value for key, value in x.items() if key != "boundary"}
                | dict(boundary=x["boundary"] if check == "numeric" else {})
                for x in self.launches
            ],
            outputs={name: digest(tensor) for name, tensor in (outputs or {}).items()},
        )
        return json.loads(json.dumps(record))  # tuples as the lists a file holds


@contextlib.contextmanager
def record_launches():
    from triton.runtime.jit import JITFunction

    import sglang.kernels.ops.moe.moe_wna16_marlin as marlin
    from sglang.srt.lora.workspace import LoraWorkspace

    try:
        from cutlass.cutlass_dsl.cuda_jit_executor import (
            CudaDialectJitCompiledFunction as CuteFunction,
        )
    except Exception:  # no CuTeDSL on this machine
        CuteFunction = None

    recorder = LaunchRecorder()
    run = JITFunction.run
    workspace_tensor = LoraWorkspace.tensor
    marlin_gemm = marlin.moe_wna16_marlin_gemm
    cute_call = CuteFunction.__call__ if CuteFunction else None

    def traced_run(self, *args, grid, warmup, **kwargs):
        if not warmup:
            recorder._record("triton", self.__name__, grid, args, kwargs)
        return run(self, *args, grid=grid, warmup=warmup, **kwargs)

    def traced_workspace_tensor(self, name, shape, **kwargs):
        out = workspace_tensor(self, name, shape, **kwargs)
        start, size = _span(out)
        recorder._named.setdefault(start, (size, name))
        return out

    def traced_marlin_gemm(*args, **kwargs):
        recorder._record("marlin", "moe_wna16_marlin_gemm", None, args, kwargs)
        return marlin_gemm(*args, **kwargs)

    def traced_cute_call(self, *args, **kwargs):
        # The name embeds the kernel object's repr; keep the part before it.
        name = str(getattr(self, "function_name", "cutedsl")).split("objectat")[0]
        recorder._record("cutedsl", name, None, args, kwargs)
        return cute_call(self, *args, **kwargs)

    JITFunction.run = traced_run
    LoraWorkspace.tensor = traced_workspace_tensor
    marlin.moe_wna16_marlin_gemm = traced_marlin_gemm
    if CuteFunction:
        CuteFunction.__call__ = traced_cute_call
    try:
        yield recorder
    finally:
        JITFunction.run = run
        LoraWorkspace.tensor = workspace_tensor
        marlin.moe_wna16_marlin_gemm = marlin_gemm
        if CuteFunction:
            CuteFunction.__call__ = cute_call


def _context() -> dict[str, Any]:
    """The tree, the device and the libraries a record was taken on."""
    root = pathlib.Path(__file__).resolve().parents[4]
    try:
        git = ["git", "-C", str(root)]
        revision = subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        revision += "+modified" if dirty else ""
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    device = "cpu"
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        device = dict(
            name=properties.name,
            capability=list(torch.cuda.get_device_capability(0)),
            sms=properties.multi_processor_count,
        )
    environment = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
    )
    for package in ("triton", "nvidia-cutlass-dsl", "flashinfer-python"):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = "absent"
    return dict(revision=revision, device=device, environment=environment)


def save_artifact(path: str | pathlib.Path, artifact: dict[str, Any]) -> None:
    """Add one case's record to the artifact file at ``path`` (a JSON object
    of cases by name), replacing an earlier record of the same case."""
    path = pathlib.Path(path)
    cases = json.loads(path.read_text()) if path.exists() else {}
    cases[artifact["case"]] = artifact
    path.write_text(json.dumps(cases, indent=1, sort_keys=True) + "\n")


def first_difference(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any] | None:
    """Where two records of one case first part, or None: the call (its
    kernel, its grid, its first differing constant or operand view, then, for
    a numeric case, a declared buffer at that call's boundary), the launch
    count, an output. Raises ``NotComparable`` for records that
    ``not_comparable`` rejects: None always means "compared and the same"."""
    reasons = not_comparable(before, after)
    if reasons:
        raise NotComparable("; ".join(reasons))
    numeric = before["check"] == "numeric"
    for index, (a, b) in enumerate(zip(before["calls"], after["calls"])):
        where = dict(call=index, kernel=a["kernel"])
        for field in ("kind", "kernel", "grid"):
            if a[field] != b[field]:
                return dict(where, field=field, before=a[field], after=b[field])
        for field, name in (("constants", "constant"), ("operands", "operand")):
            if a[field] != b[field]:
                # The first entry that differs (None past the shorter list).
                x, y = a[field], b[field]
                pairs = [*zip(x, y)]
                at = next((i for i, (p, q) in enumerate(pairs) if p != q), len(pairs))
                return dict(
                    where,
                    field=name,
                    before=x[at] if at < len(x) else None,
                    after=y[at] if at < len(y) else None,
                )
        if numeric and a["boundary"] != b["boundary"]:
            names = sorted({*a["boundary"], *b["boundary"]})
            name = next(
                n for n in names if a["boundary"].get(n) != b["boundary"].get(n)
            )
            return dict(
                where,
                field="buffer",
                buffer=name,
                before=a["boundary"].get(name),
                after=b["boundary"].get(name),
            )
    if len(before["calls"]) != len(after["calls"]):
        return dict(
            field="launch count",
            call=min(len(before["calls"]), len(after["calls"])),
            before=len(before["calls"]),
            after=len(after["calls"]),
        )
    if numeric and before["outputs"] != after["outputs"]:
        names = sorted({*before["outputs"], *after["outputs"]})
        name = next(
            n for n in names if before["outputs"].get(n) != after["outputs"].get(n)
        )
        return dict(
            field="output",
            output=name,
            before=before["outputs"].get(name),
            after=after["outputs"].get(name),
        )
    return None


def not_comparable(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Why two records of one case cannot be compared; empty when they can.
    A record must be a launch record that holds evidence: launches, and for
    a numeric case at least one boundary or output digest. The two must agree
    in format, check kind, fixture, device and library versions: digests and
    tuned launch structures mean the same only there. The revision is not
    looked at."""
    reasons = []
    for side, record in (("before", before), ("after", after)):
        missing = [key for key in _RECORD_FIELDS if key not in record]
        if missing:
            reasons.append(f"{side}: not a launch record (no {', '.join(missing)})")
        elif not record["calls"]:
            reasons.append(f"{side}: no launches recorded")
        elif record["check"] == "numeric" and not (
            record["outputs"] or any(call["boundary"] for call in record["calls"])
        ):
            reasons.append(f"{side}: a numeric record without any digest")
    for key in _SAME_CONTEXT:
        if key in before and key in after and before[key] != after[key]:
            reasons.append(f"{key}: {before[key]} -> {after[key]}")
    return reasons


def _load(path: pathlib.Path) -> dict[str, Any]:
    """The cases of one artifact file, or of every artifact file of a
    directory. A path without records is an error, not an empty comparison."""
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    cases: dict[str, Any] = {}
    for file in files:
        loaded = json.loads(file.read_text())
        if not isinstance(loaded, dict) or not all(
            isinstance(record, dict) for record in loaded.values()
        ):
            raise ValueError(f"{file}: not a JSON object of launch records")
        cases.update(loaded)
    if not cases:
        raise ValueError(f"{path}: no launch records")
    return cases


def main(argv: list[str]) -> int:
    """Compare two artifact files (or directories of them), one line per
    case: SAME, DIFFERS with the first difference, MISSING for a case one side
    lacks, NOT-COMPARABLE with the reasons. Exit status 0 only when every
    case was compared and is the same; 1 when a case differs or is missing;
    3 when a path has no records or cannot be read, or a case is not
    comparable; 2 on a usage error."""
    if len(argv) != 2:
        print(
            "usage: python -m sglang.test.kernels.moe_lora_launch_recorder <before> <after>"
        )
        return 2
    try:
        before, after = (_load(pathlib.Path(arg)) for arg in argv)
    except (OSError, ValueError) as error:
        print(f"INVALID {error}")
        return 3
    differs = blocked = False
    for case in sorted({*before, *after}):
        if case not in before or case not in after:
            print(f"MISSING {case}: only in {'before' if case in before else 'after'}")
            differs = True
            continue
        a, b = before[case], after[case]
        reasons = not_comparable(a, b)
        if reasons:
            print(f"NOT-COMPARABLE {case}: {'; '.join(reasons)}")
            blocked = True
            continue
        difference = first_difference(a, b)
        differs |= difference is not None
        print(
            f"{'SAME' if difference is None else 'DIFFERS'} {case} "
            f"[{a['check']}, {len(a['calls'])} launches, "
            f"{a['revision'][:10]} -> {b['revision'][:10]}]"
            + ("" if difference is None else f": {json.dumps(difference)}")
        )
    return 3 if blocked else int(differs)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
