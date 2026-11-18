from typing import List, Literal, NamedTuple, Optional, Tuple, TypeAlias

Metric: TypeAlias = "float | Literal['avg']"


class BenchSkip(Exception):
    pass


def skip(reason: str):
    raise BenchSkip(reason)


class BenchResult(NamedTuple):
    metrics: Tuple[Metric, ...]
    times: List[float]  # in seconds
    memory_footprint: Optional[int]
    flops: Optional[float] = None
    samples: Tuple[float, ...] = ()  # every timed iteration, seconds, sorted

    def gbps(self) -> Optional[float]:
        if self.memory_footprint is None:
            return None
        return self.memory_footprint / (1024**3) / self.times[0]

    def tflops(self) -> Optional[float]:
        if self.flops is None:
            return None
        return self.flops / 1e12 / self.times[0]
