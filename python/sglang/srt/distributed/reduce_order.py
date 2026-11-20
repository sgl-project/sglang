"""The order in which an all-reduce adds its ranks' values, found by asking it.

A floating-point sum depends on the order of its additions, and an all-reduce does
not say which order it uses. This module finds it out. Every rank fills a tensor
with a chosen constant and the ranks all-reduce it; the bits that come back
identify the additions. Two forms are recognized:

  - one FP32 accumulator over the ranks in rank order, rounded once at the end
    (``wide``), as a one-stage peer-read kernel does it;
  - an addition tree whose every step rounds to the tensor's dtype, which may
    differ from one stretch of the tensor to the next, as a ring does it.

:func:`fold` then adds values in that order. A caller that holds the ``world``
versions of only some elements can so compute exactly the bits the all-reduce
would have returned for them (see ``exact_reduce_scatter.py``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, Tuple

import torch

# Most ranks an addition tree is enumerated for: (2 * 8 - 3)!! = 135,135 trees.
MAX_WORLD = 8
# Stretches of one order start at multiples of a cell. Finer ones would need a
# table as large as the data.
_MIN_CELL_ELEMENTS = 64

# An addition tree as a program (a0, b0, a1, b1, ...): register r < MAX_WORLD is
# rank r's value, addition i writes register MAX_WORLD + i, the last is the sum.
Program = Tuple[int, ...]


@dataclass(frozen=True)
class ReduceOrder:
    """How an all-reduce of ``num_elements`` elements over ``world`` ranks adds.

    ``wide`` is the FP32 accumulator in rank order. Otherwise element ``e`` is
    added by ``programs[cell_order[e // cell_elements]]``.
    """

    world: int
    num_elements: int
    wide: bool = False
    cell_elements: int = 0
    cell_order: Optional[torch.Tensor] = None  # [cells] int64, on the CPU
    programs: Tuple[Program, ...] = ()


def wide_order(world: int, num_elements: int) -> ReduceOrder:
    return ReduceOrder(world=world, num_elements=num_elements, wide=True)


def _add_rounded(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # The model of one step of a tree: add in FP32, round to the tensor's dtype.
    # Whether an all-reduce follows it is checked on data by the caller.
    return (a.float() + b.float()).to(a.dtype)


def run_programs(values: torch.Tensor, programs: torch.Tensor) -> torch.Tensor:
    """``values`` ``[world, ...]`` added by each row of ``programs`` ``[n, 2 *
    (world - 1)]``: ``[n, ...]``."""
    world = values.shape[0]
    n = programs.shape[0]
    registers = values.new_zeros((n, MAX_WORLD + world - 1, *values.shape[1:]))
    registers[:, :world] = values
    rows = torch.arange(n)
    for i in range(world - 1):
        a = registers[rows, programs[:, 2 * i]]
        b = registers[rows, programs[:, 2 * i + 1]]
        registers[:, MAX_WORLD + i] = _add_rounded(a, b)
    return registers[:, MAX_WORLD + world - 2]


def run_program(values: torch.Tensor, program: Program) -> torch.Tensor:
    """``values`` ``[world, n]`` added by one program: ``[n]``."""
    registers = {rank: values[rank] for rank in range(values.shape[0])}
    for i in range(0, len(program), 2):
        registers[MAX_WORLD + i // 2] = _add_rounded(
            registers[program[i]], registers[program[i + 1]]
        )
    return registers[MAX_WORLD + len(program) // 2 - 1]


def fold(values: torch.Tensor, order: ReduceOrder, first_element: int) -> torch.Tensor:
    """Add ``values`` ``[world, n]`` the way the all-reduce adds elements
    ``first_element .. first_element + n`` of its tensor: ``[n]``."""
    world, n = values.shape
    assert world == order.world and first_element + n <= order.num_elements
    if order.wide:
        total = values[0].float()
        for rank in range(1, world):
            total = total + values[rank].float()
        return total.to(values.dtype)
    cell = order.cell_elements
    first_cell = first_element // cell
    last_cell = (first_element + n - 1) // cell
    programs = order.cell_order[first_cell : last_cell + 1]
    if bool((programs == programs[0]).all()):
        return run_program(values, order.programs[int(programs[0])])
    skip = first_element - first_cell * cell
    per_element = programs.repeat_interleave(cell)[skip : skip + n].to(values.device)
    out = torch.empty_like(values[0])
    for index in programs.unique().tolist():
        chosen = per_element == index
        out[chosen] = run_program(values[:, chosen], order.programs[index])
    return out


def _addition_trees(world: int):
    """Every way to add ``world`` values pairwise: nested pairs of ranks."""

    @lru_cache(maxsize=None)
    def trees(ranks):
        if len(ranks) == 1:
            return (ranks[0],)
        first, rest = ranks[0], ranks[1:]
        out = []
        for mask in range(1 << len(rest)):
            left = (first,) + tuple(r for i, r in enumerate(rest) if mask >> i & 1)
            right = tuple(r for i, r in enumerate(rest) if not mask >> i & 1)
            if right:
                out += [(a, b) for a in trees(left) for b in trees(right)]
        return tuple(out)

    return trees(tuple(range(world)))


def _program_of(tree) -> Program:
    program = []

    def emit(node):
        if isinstance(node, int):
            return node
        a, b = emit(node[0]), emit(node[1])
        program.extend((a, b))
        return MAX_WORLD + len(program) // 2 - 1

    emit(tree)
    return tuple(program)


# Probe inputs per rank, in units of 2**-10 (8-bit mantissas over six binades,
# both signs; a greedy pick among random vectors). Their bf16 sums differ
# between any two addition trees.
_PROBE_VALUES = {
    4: (
        (3280,-3984,-237,1624), (136,-2256,992,540), (3408,3504,-7552,390),
        (-2768,1064,2800,2992),
    ),
    8: (
        (664,-7424,976,290,1656,4224,-466,-340), (-900,4480,2288,-3936,322,151,-2240,133),
        (600,-1896,4448,556,161,540,-3952,-308), (272,6752,-5568,-192,-3904,296,-1792,7232),
        (-6208,398,237,1600,-1064,5344,207,414),
        (2400,-3664,-151,4016,1000,5248,-7520,-748),
        (3776,280,-7680,452,266,-592,1008,2544), (-600,636,-129,884,1048,-1048,-3616,1952),
        (1240,-484,776,-8128,7680,3184,552,-237), (-6304,300,1928,208,3568,240,768,-242),
        (680,-1352,-466,7392,-2464,494,246,2720), (-616,2080,2000,-500,2400,-5568,211,-249),
        (-352,218,2224,304,-2288,-1912,153,1020),
        (-3200,920,5952,-6880,1464,2592,-229,7008), (1200,392,338,5440,143,-5984,-1504,209),
        (390,240,422,210,940,-2832,964,129), (1952,1320,170,1032,-6080,-2128,1136,1096),
        (-360,370,173,1648,4704,7072,243,-7872), (-235,1744,1056,-2656,205,804,198,1304),
        (1384,1880,776,-169,1080,2288,128,-7104), (-1592,-326,183,3600,-490,490,215,704),
        (904,-1912,230,298,300,728,380,580), (-7392,-280,-772,4480,-4736,2256,976,-1232),
        (-320,-2000,-268,1288,2048,-2000,2096,-1528),
        (1072,888,-1840,8096,644,200,494,1624), (-484,-216,137,600,-2688,4160,243,-1680),
        (134,872,1984,-278,-1000,-2704,504,408), (-1616,212,784,184,500,1136,-198,-221),
        (3184,5600,-1272,-2368,466,176,-264,-2080),
        (2592,532,2544,-5856,336,3376,2656,-248), (2464,5120,-240,1184,340,756,137,1168),
        (-3248,-936,940,296,688,-2368,442,245),
    ),
}  # fmt: skip


@lru_cache(maxsize=None)
def _probe_design(world: int):
    """``(probes [P, world] bf16, programs [n, 2 * (world - 1)], {bits of a
    program's sums of the probes: its row})`` over every addition tree."""
    programs = torch.tensor(
        [_program_of(tree) for tree in _addition_trees(world)], dtype=torch.uint8
    )
    probes = (torch.tensor(_PROBE_VALUES[world]) / 1024).to(torch.bfloat16)
    values = probes.t().contiguous()
    decode = {}
    # In pieces: the registers of all 135,135 trees of eight ranks at once
    # would take several hundred megabytes.
    for start in range(0, len(programs), 4096):
        sums = run_programs(values, programs[start : start + 4096].long())
        for offset, row in enumerate(sums.view(torch.int16).numpy()):
            decode[row.tobytes()] = start + offset
    if len(decode) != len(programs):
        raise RuntimeError(f"the probes do not separate the sums of {world} ranks")
    return probes, programs, decode


def num_probes(world: int) -> int:
    """All-reduces :func:`discover_tree_order` runs, or 0 if it cannot."""
    return len(_PROBE_VALUES.get(world, ()))


def discover_tree_order(
    all_reduce: Callable[[torch.Tensor], torch.Tensor],
    world: int,
    rank: int,
    shape,
    device,
) -> Optional[ReduceOrder]:
    """The addition trees ``all_reduce`` uses on a bf16 tensor of ``shape``.

    Every rank must call this at the same time: it runs ``num_probes(world)``
    all-reduces. Returns None when the sums are not those of addition trees.
    """
    probes, table, decode = _probe_design(world)
    num_elements = math.prod(shape)
    starts, values = [], []
    for probe in probes:
        scratch = torch.full(
            shape, float(probe[rank]), dtype=torch.bfloat16, device=device
        )
        out = all_reduce(scratch).reshape(-1).view(torch.int16)
        change = (out[1:] != out[:-1]).nonzero().view(-1) + 1
        start = torch.cat([change.new_zeros(1), change]).cpu()
        starts.append(start)
        values.append(out.cpu()[start])
    bounds = torch.unique(torch.cat(starts))
    # One row per stretch between two bounds: its sum of every probe.
    keys = torch.stack(
        [
            value[torch.searchsorted(start, bounds, right=True) - 1]
            for start, value in zip(starts, values)
        ],
        1,
    ).contiguous()
    bounds = bounds.tolist()
    rows = [decode.get(key.tobytes()) for key in keys.numpy()]
    if any(row is None for row in rows):
        return None
    pieces = [tuple(table[row].tolist()) for row in rows]
    cell = math.gcd(num_elements, *bounds)
    if cell < min(_MIN_CELL_ELEMENTS, num_elements):
        return None
    programs = tuple(sorted(set(pieces)))
    lengths = [
        (end - start) // cell for start, end in zip(bounds, bounds[1:] + [num_elements])
    ]
    cell_order = torch.repeat_interleave(
        torch.tensor([programs.index(p) for p in pieces]), torch.tensor(lengths)
    )
    return ReduceOrder(
        world=world,
        num_elements=num_elements,
        cell_elements=cell,
        cell_order=cell_order,
        programs=programs,
    )
