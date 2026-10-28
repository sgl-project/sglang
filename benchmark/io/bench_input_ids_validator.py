"""Microbenchmark: cost of validating `input_ids` during FastAPI body binding.

Compares two validators on a single `input_ids` field:
- GenerateReqInputPydanticValidator: default pydantic walk (per-element type check)
- GenerateReqInputCustomValidator: C-loop validator (validate_optional_list_i64_1d_2d)

Usage:
    python benchmark/io/bench_input_ids_validator.py
"""

import time
from dataclasses import dataclass
from typing import Annotated, List, Optional, Union

from pydantic import PlainValidator, TypeAdapter

from sglang.srt.utils.field_validators import validate_optional_list_i64_1d_2d


@dataclass
class GenerateReqInputPydanticValidator:
    """Default pydantic — walks every element of input_ids to type-check."""

    input_ids: Optional[Union[List[List[int]], List[int]]] = None


@dataclass
class GenerateReqInputCustomValidator:
    """C-loop validator via PlainValidator."""

    input_ids: Annotated[
        Optional[Union[List[List[int]], List[int]]],
        PlainValidator(validate_optional_list_i64_1d_2d),
    ] = None


_ta_pydantic = TypeAdapter(GenerateReqInputPydanticValidator)
_ta_custom = TypeAdapter(GenerateReqInputCustomValidator)


def _time(fn, n_iter=30):
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    return (time.perf_counter() - t0) * 1000 / n_iter


def main():
    print(
        f"{'n_tokens':>9s} | {'default pydantic (ms)':>22s} | "
        f"{'rigid i64 validator (ms)':>26s}"
    )
    print("-" * 65)

    for n in [1_000, 10_000, 100_000, 1_000_000]:
        d = {"input_ids": list(range(1, n + 1))}

        p1 = _time(lambda: _ta_pydantic.validate_python(d))
        p2 = _time(lambda: _ta_custom.validate_python(d))

        print(f"{n:>9d} | {p1:>22.3f} | {p2:>26.3f}")

    print("\nLegend: mean over 30 iters, in ms.")


if __name__ == "__main__":
    main()
