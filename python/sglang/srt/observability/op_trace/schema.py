"""Records shared by the capture-time op map, the trace analyzer and the CLI.

Op identity is semantic: a module path (``model.layers.3.self_attn.qkv_proj``)
plus an op kind derived from it (``self_attn.qkv_proj``). The kernel symbol and
the CUDA graph node it ran as are attributes, never the identity.
"""

from __future__ import annotations

from typing import Optional

import msgspec

OP_MAP_VERSION = 1

# Node type strings for non-kernel graph nodes; kernels carry their symbol.
NODE_KERNEL = "kernel"


class OpScope(msgspec.Struct, frozen=True):
    path: str
    module_class: str


class CapturedGraph(msgspec.Struct, frozen=True):
    """One captured CUDA graph; ``node_*`` lists are indexed by node creation
    order, which is the low 32 bits of CUPTI's ``graph node id`` at replay."""

    runner: str
    shape: str
    forward_mode: str
    size: int
    # Order of this capture among this process's op-map captures. A collector
    # that sees exec creation (CUPTI GRAPHEXEC_CREATED, in creation order) pairs
    # execs with captures in this order, checking the handle; without it the
    # analyzer matches execs by node fingerprint.
    capture_seq: int
    # CUgraph handle of the graph under capture; torch frees it after
    # instantiation, so a later capture can reuse the address.
    graph_handle: int
    node_type: list[str]
    # Index into OpMapFile.symbols; -1 for non-kernel nodes.
    node_symbol: list[int]
    node_grid: list[tuple[int, int, int]]
    node_block: list[tuple[int, int, int]]
    # Index into OpMapFile.ops; -1 when the node was created outside every
    # registered module forward (graph-runner glue).
    node_op: list[int]


class OpMapFile(msgspec.Struct, frozen=True):
    version: int
    pid: int
    tp_rank: int
    symbols: list[str]
    ops: list[OpScope]
    graphs: list[CapturedGraph]


class StepRecord(msgspec.Struct, frozen=True):
    seq: int
    forward_mode: str
    # Captured graph size (padded) for graph steps; real batch size for steps
    # delimited by a CPU-side step annotation.
    batch_size: Optional[int]
    num_tokens: Optional[int]
    is_graph: bool
    start_ns: int
    end_ns: int
    kernel_sum_ns: int
    num_kernels: int
    # Kernels whose symbol disagrees with the op map / eager reference, plus
    # reference kernels missing from this step; nonzero means a bad join.
    mismatches: int = 0


class KernelRecord(msgspec.Struct, frozen=True):
    step_seq: int
    start_ns: int
    dur_ns: int
    symbol: str
    op_path: str
    op_kind: str
    layer: Optional[int]
    stream: int


class OpStat(msgspec.Struct, frozen=True):
    bucket: str
    op_kind: str
    module_class: str
    num_layers: int
    kernels_per_step: float
    us_per_step_p50: float
    share_of_step: float
    us_per_layer_p50: Optional[float]
    top_symbol: str
    # Per-step sum of this op's kernel time, one value per step of the bucket.
    step_samples_us: list[float] = []


class BucketStat(msgspec.Struct, frozen=True):
    bucket: str
    num_steps: int
    step_kernel_sum_us_p50: float
    step_span_us_p50: float
    attributed_share: float
    symbol_mismatches: int
    # How graph execs were matched to captures: "exact" (exec creation order
    # from the native collector), "fingerprint", "fingerprint-ambiguous", or
    # "eager" when the bucket has no graph kernels.
    graph_join: str = "eager"
    step_kernel_sum_samples_us: list[float] = []


class EagerReference(msgspec.Struct, frozen=True):
    """Kernel order of one eager step from an attribution pass (CPU+GPU with
    op spans); a GPU-only timing pass of the same step joins to it by ordinal."""

    symbols: list[str]
    op_paths: list[str]
    forward_mode: str
    batch_size: Optional[int]
    num_tokens: Optional[int]


class TraceSummary(msgspec.Struct, frozen=True):
    trace: str
    op_maps: list[str]
    buckets: list[BucketStat]
    ops: list[OpStat]
    steps: list[StepRecord] = []
    kernels: list[KernelRecord] = []
