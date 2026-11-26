import builtins
import inspect
import itertools
from types import EllipsisType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
)

from .case import (
    BenchInfo,
    Case,
    CaseResult,
    Tolerance,
    canonical_value,
    check_canonical,
    make_bench_id,
    make_case_id,
)
from .config import CASE_FILTER, CI_MODE, WORLD_SIZE
from .reporter import UNIT_SCALE, make_reporters
from .result import BenchResult, BenchSkip

F = TypeVar("F", bound=Callable[..., "BenchResult"])
PATTERN: TypeAlias = "Literal['pow2']"


class _Axis(NamedTuple):
    names: Tuple[str, ...]
    vals: List[Tuple[Any, ...]]
    ci_vals: Optional[List[Tuple[Any, ...]]]


class Benchmark(Generic[F]):
    """A benchmark fn plus its axes. `run` plans every case first, executes them in
    order, and hands each result to the reporters."""

    def __init__(
        self,
        fn: F,
        line_arg: str,
        line_vals: List[Any],
        *,
        unit: str,
        gate: Sequence[Any] | EllipsisType,
        gate_reason: str,
        tolerance: Tolerance,
    ) -> None:
        assert unit in UNIT_SCALE, f"unit {unit!r} is not one of {list(UNIT_SCALE)}"
        assert len({str(v) for v in line_vals}) == len(line_vals) > 0, line_vals
        assert line_arg in inspect.signature(fn).parameters, (
            f"line_arg {line_arg!r} is not a parameter of {fn.__name__}; "
            f"available: {list(inspect.signature(fn).parameters)}"
        )
        if gate is ...:
            gate = line_vals
        assert isinstance(gate, (list, tuple)), (
            f"gate must be a list of line_vals, `...` (every line) or [] (none); got {gate!r}"
        )
        assert all(g in line_vals for g in gate), (
            f"gate {gate!r} must be a subset of line_vals {line_vals!r}"
        )
        check_canonical(line_vals)
        source_file = inspect.getsourcefile(fn)
        assert source_file is not None, f"no source file for {fn!r}"
        self._fn = fn
        self._fn_params = inspect.signature(fn).parameters
        self._line_arg = line_arg
        self._line_vals = line_vals
        self._unit = unit
        self._gated_lines = frozenset(str(v) for v in gate)
        self._gate_reason = gate_reason
        self._tolerance = tolerance
        self._source_file = source_file
        self._bench_id = make_bench_id(source_file=source_file, fn_name=fn.__qualname__)
        self._configs: List[_Axis] = []
        self._seen_args = {line_arg}

    def _add_config(
        self,
        *,
        names: Tuple[str, ...],
        vals: List[Tuple[Any, ...]],
        ci_vals: Optional[List[Tuple[Any, ...]]],
    ) -> None:
        """Prepend a parametrize axis. Validates that names are real parameters
        of the benchmark fn, and rejects duplicates / collisions with line_arg."""
        assert len(names) > 0, "parametrize: must provide at least one name"
        for name in names:
            assert name in self._fn_params, (
                f"parametrize name {name!r} is not a parameter of "
                f"{self._fn.__name__}; available: {list(self._fn_params)}"
            )
            assert name not in self._seen_args, (
                f"parametrize name {name!r} is already used"
            )
            self._seen_args.add(name)
        self._configs.insert(0, _Axis(names=names, vals=vals, ci_vals=ci_vals))

    def info(self) -> BenchInfo:
        return BenchInfo(
            bench_id=self._bench_id,
            source_file=self._source_file,
            axis_names=tuple(n for axis in self._configs for n in axis.names),
            line_vals=tuple(self._line_vals),
            unit=self._unit,
            gated_lines=self._gated_lines,
            gate_reason=self._gate_reason,
            tolerance=self._tolerance,
        )

    def _plan(self) -> List[Case]:
        """Every case this run executes, in execution order: lines outermost, then
        the parameter rows. CI mode picks each axis's `ci_vals`."""
        self._check_params_covered()
        axes_vals = [
            axis.ci_vals if (CI_MODE and axis.ci_vals is not None) else axis.vals
            for axis in self._configs
        ]
        cases = [
            self._make_case(line_val=line_val, row=row, combo=combo)
            for line_val in self._line_vals
            for row, combo in enumerate(itertools.product(*axes_vals))
        ]
        return [c for c in cases if not CASE_FILTER or c.case_id in CASE_FILTER]

    def run(
        self,
        *,
        print_prefix: Optional[str] = None,
        print_suffix: Optional[str] = None,
    ) -> None:
        cases = self._plan()
        reporters = make_reporters(print_prefix=print_prefix, print_suffix=print_suffix)
        info = self.info()
        for reporter in reporters:
            reporter.begin(info=info)
        for case in cases:
            for reporter in reporters:
                reporter.before_case(case)
            result = self._execute(case)
            for reporter in reporters:
                reporter.on_result(result)
        for reporter in reporters:
            reporter.finish()

    def _check_params_covered(self) -> None:
        kinds = (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        required = {
            n
            for n, p in self._fn_params.items()
            if p.default is inspect.Parameter.empty and p.kind in kinds
        }
        missing = required - self._seen_args
        assert not missing, (
            f"parameters not parametrized for {self._fn.__name__}: {sorted(missing)}"
        )

    def _make_case(
        self, *, line_val: Any, row: int, combo: Tuple[Tuple[Any, ...], ...]
    ) -> Case:
        kwargs: Dict[str, Any] = {}
        for axis, values in zip(self._configs, combo):
            kwargs.update(zip(axis.names, values))
        params = {name: canonical_value(v) for name, v in kwargs.items()}
        line = str(line_val)
        return Case(
            bench_id=self._bench_id,
            params=params,
            line=line,
            world_size=WORLD_SIZE,
            case_id=make_case_id(
                bench_id=self._bench_id,
                params=params,
                line=line,
                world_size=WORLD_SIZE,
            ),
            row=row,
            kwargs={self._line_arg: line_val, **kwargs},
        )

    def _execute(self, case: Case) -> CaseResult:
        try:
            result = self._fn(**case.kwargs)
        except BenchSkip as e:
            return CaseResult(case=case, result=None, skip_reason=str(e))
        except BaseException as e:
            e.add_note(f"benchmark {case.bench_id} failed at {case.kwargs}")
            raise
        return CaseResult(case=case, result=result, skip_reason=None)


def benchmark(
    line_arg: str,
    line_vals: List[Any],
    *,
    unit: str = "us",
    gate: Sequence[Any] | EllipsisType = ...,
    gate_reason: str = "",
    tolerance: Tolerance = Tolerance(),
):
    """`gate`: the line_vals a perf-regression check compares; `...` (default) is
    every line, `[]` none. List only this repo's own implementations when the
    benchmark also has reference lines (torch, flashinfer, ...), which are then
    only recorded. `gate_reason`: optional note. `tolerance`: the allowed
    slowdown of every gated case of this benchmark."""

    def decorator(fn: F) -> Benchmark[F]:
        return Benchmark(
            fn,
            line_arg,
            line_vals,
            unit=unit,
            gate=gate,
            gate_reason=gate_reason,
            tolerance=tolerance,
        )

    return decorator


def parametrize(names: str, vals: List[Any], ci_vals: Optional[List[Any]] = None):
    """Add a parametrize axis. Pytest-style:

    - Single name:   `parametrize("dim", [1024, 4096])`
    - Multiple names (correlated):
                     `parametrize("h,d", [(1, 64), (2, 128)])`

    For multi-name axes, each value must be a tuple/list of matching length.
    """
    name_tuple = tuple(n.strip() for n in names.split(","))
    assert all(name_tuple), f"parametrize: empty name in {names!r}"
    arity = len(name_tuple)

    def _normalize(vs: List[Any]) -> List[Tuple[Any, ...]]:
        if arity == 1:
            return [(v,) for v in vs]
        out: List[Tuple[Any, ...]] = []
        for v in vs:
            assert isinstance(v, (tuple, list)), (
                f"parametrize: multi-name values must be tuples, got {v!r}"
            )
            t = tuple(v)
            assert len(t) == arity, (
                f"parametrize: each value must have length {arity}, got {t!r}"
            )
            out.append(t)
        return out

    full_range = _normalize(vals)
    ci_range = None if ci_vals is None else _normalize(ci_vals)
    check_canonical(full_range)
    if ci_range is not None and not all(v in full_range for v in ci_range):
        missing = [v for v in ci_range if v not in full_range]
        raise ValueError(f"{missing = } not in {full_range = }")

    def decorator(bench: Benchmark[F]) -> Benchmark[F]:
        bench._add_config(names=name_tuple, vals=full_range, ci_vals=ci_range)
        return bench

    return decorator


def range(*args: int, pattern: PATTERN) -> List[int]:
    # TODO: support other patterns
    assert pattern == "pow2", f"unsupported pattern: {pattern}"
    return [2**i for i in builtins.range(*args)]
