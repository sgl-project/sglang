"""Exact reduce-scatter: the order an all-reduce adds in, and the collective.

The order logic runs on the CPU against all-reduces simulated from addition
programs we choose. The collective runs in real rank processes: with gloo on the
CPU, and with NCCL on as many GPUs as the machine has.
"""

import math
import os
import socket
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.distributed import reduce_order as O
from sglang.srt.distributed.exact_reduce_scatter import ExactReduceScatter
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")
register_cuda_ci(est_time=120, stage="base-c", runner_config="8-gpu-h200")

parallel_state = pytest.importorskip("sglang.srt.distributed.parallel_state")


def _left_fold(order):
    """The program of fl(..fl(x[order[0]] + x[order[1]]).. + x[order[-1]])."""
    program, acc = [], order[0]
    for i, rank in enumerate(order[1:]):
        program += [acc, rank]
        acc = O.MAX_WORLD + i
    return tuple(program)


def _unordered(program):
    """The program with each addition's two operands in a fixed order."""
    return tuple(tuple(sorted(program[i : i + 2])) for i in range(0, len(program), 2))


def _ring(world, start):
    return _left_fold([(start + i) % world for i in range(world)])


def _random_bf16(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    scale = 2.0 ** torch.randint(-6, 6, shape, generator=generator)
    return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)


def _bits(tensor):
    return tensor.contiguous().view(torch.int16)


# --- the order logic, one process ---------------------------------------------


def test_addition_tree_counts_are_double_factorials():
    for world, count in ((2, 1), (3, 3), (4, 15), (5, 105)):  # (2n - 3)!!
        trees = O._addition_trees(world)
        assert len(trees) == count
        assert len({O._program_of(tree) for tree in trees}) == count


def test_a_program_rounds_at_every_step():
    values = torch.tensor([1.0, 2.0**-9, 2.0**-9, -1.0]).to(torch.bfloat16)
    left = _left_fold((0, 1, 2, 3))  # ((1 + e) + e) - 1: each e is absorbed
    small_first = (1, 2, 0, 8, 9, 3)  # (1 + (e + e)) - 1: a tie that rounds to even
    apart = (0, 3, 1, 2, 8, 9)  # (1 - 1) + (e + e)
    out = O.run_programs(values, torch.tensor([left, small_first, apart]))
    assert out.float().tolist() == [0.0, 0.0, 2.0**-8]
    for index, program in enumerate((left, small_first, apart)):
        assert O.run_program(values.view(4, 1), program).float().item() == out[index]


@pytest.mark.parametrize("world,count", [(4, 15), (8, 135135)])
def test_probes_separate_every_tree(world, count):
    _, _, decode = O._probe_design(world)
    assert len(decode) == count


def _simulated_all_reduce(world, shape, cell, cell_programs):
    """An all_reduce for ``discover_tree_order`` on one rank: call k is probe k,
    and element e is added by ``cell_programs[e // cell]``."""
    probes, _, _ = O._probe_design(world)
    programs = sorted(set(cell_programs))
    per_cell = torch.tensor([programs.index(p) for p in cell_programs])
    calls = iter(range(len(probes)))

    def all_reduce(tensor):
        sums = O.run_programs(probes[next(calls)], torch.tensor(programs))
        n = math.prod(shape)
        return sums[per_cell].repeat_interleave(cell)[:n].reshape(shape).clone()

    return all_reduce


@pytest.mark.parametrize("world", [4, 8])
def test_discovery_returns_the_programs_of_each_stretch(world):
    shape, cell = (world * 6, 256), 128
    num_cells = math.prod(shape) // cell
    # A ring that starts at another rank in every stretch, and one balanced tree.
    rings = [_ring(world, start) for start in range(world)]
    balanced = O._program_of(O._addition_trees(world)[-1])
    cell_programs = [
        (rings + [balanced])[(c // 5) % (world + 1)] for c in range(num_cells)
    ]
    all_reduce = _simulated_all_reduce(world, shape, cell, cell_programs)
    order = O.discover_tree_order(all_reduce, world, 0, shape, "cpu")
    assert order is not None and not order.wide
    assert order.num_elements == math.prod(shape)
    found = [order.programs[i] for i in order.cell_order.tolist()]
    expected = cell_programs[:: order.cell_elements // cell]
    assert order.cell_elements % cell == 0
    assert [_unordered(p) for p in found] == [_unordered(p) for p in expected]

    # Folding any range of elements gives the bits of the full sum there.
    values = _random_bf16((world, math.prod(shape)), seed=world)
    per_element = torch.tensor(
        [sorted(set(cell_programs)).index(p) for p in cell_programs]
    ).repeat_interleave(cell)
    table = torch.tensor(sorted(set(cell_programs)))
    full = O.run_programs(values, table)[per_element, torch.arange(values.shape[1])]
    for first, n in ((0, values.shape[1]), (cell * 3 + 17, cell * 9 + 5), (5, 40)):
        got = O.fold(values[:, first : first + n], order, first)
        assert torch.equal(_bits(got), _bits(full[first : first + n]))


def test_discovery_rejects_sums_that_are_no_tree():
    world, shape = 4, (8, 64)
    probes, _, _ = O._probe_design(world)
    calls = iter(range(len(probes)))

    def all_reduce(tensor):
        # Twice the sum: no addition tree gives it on every probe.
        total = probes[next(calls)].float().sum() * 2
        return torch.full(shape, float(total)).to(torch.bfloat16)

    assert O.discover_tree_order(all_reduce, world, 0, shape, "cpu") is None


def test_wide_fold_rounds_once():
    values = (
        torch.tensor([1.0, 2.0**-9, 2.0**-9, 2.0**-9]).to(torch.bfloat16).view(4, 1)
    )
    wide = O.fold(values, O.wide_order(4, 1), 0)
    stepwise = O.run_program(values, _left_fold((0, 1, 2, 3)))
    assert wide.float().item() == 1.0 + 2.0**-7  # 3e is above the tie
    assert stepwise.float().item() == 1.0  # each e is absorbed


# --- the group coordinator's call site ---------------------------------------


def _coordinator(rank=1, world=4):
    coordinator = parallel_state.GroupCoordinator.__new__(
        parallel_state.GroupCoordinator
    )
    coordinator.rank_in_group = rank
    coordinator.world_size = world
    return coordinator


def test_flag_off_keeps_the_all_reduce(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "1")
    monkeypatch.setenv("SGLANG_ENABLE_EXACT_REDUCE_SCATTER", "0")
    coordinator = _coordinator()
    data = torch.arange(32, dtype=torch.float32).to(torch.bfloat16).view(8, 4)
    coordinator.all_reduce = Mock(side_effect=lambda tensor: tensor * 2)
    coordinator.all_to_all_single = Mock(side_effect=AssertionError)
    output = torch.empty(2, 4, dtype=torch.bfloat16)
    coordinator.reduce_scatter_tensor(output, data)
    assert torch.equal(output, (data * 2)[2:4])
    coordinator.all_reduce.assert_called_once()


def test_flag_on_uses_the_exact_path_or_falls_back(monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_DETERMINISTIC_INFERENCE", "1")
    monkeypatch.setenv("SGLANG_ENABLE_EXACT_REDUCE_SCATTER", "1")
    coordinator = _coordinator()
    data = torch.arange(32, dtype=torch.float32).to(torch.bfloat16).view(8, 4)
    coordinator.all_reduce = Mock(side_effect=lambda tensor: tensor * 2)
    output = torch.empty(2, 4, dtype=torch.bfloat16)

    def exact(out, tensor):
        out.fill_(7.0)
        return True

    coordinator._exact_reduce_scatter_op = Mock(side_effect=exact)
    coordinator.reduce_scatter_tensor(output, data)
    assert (output == 7.0).all()
    coordinator.all_reduce.assert_not_called()

    coordinator._exact_reduce_scatter_op = Mock(return_value=False)
    coordinator.reduce_scatter_tensor(output, data)
    assert torch.equal(output, (data * 2)[2:4])
    coordinator.all_reduce.assert_called_once()


# --- the collective, in rank processes -----------------------------------------


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _known_all_reduce(world, kind, device):
    """An all-reduce with an order we choose, built on an all-gather."""

    def all_reduce(tensor):
        gathered = torch.empty(
            (world, *tensor.shape), dtype=tensor.dtype, device=device
        )
        dist.all_gather_into_tensor(
            gathered.view(-1, *tensor.shape[1:]), tensor.contiguous()
        )
        flat = gathered.view(world, -1)
        if kind == "wide":
            total = flat[0].float()
            for rank in range(1, world):
                total = total + flat[rank].float()
            return total.to(tensor.dtype).view(tensor.shape)
        # A ring that starts at another rank in each quarter of the tensor.
        out = torch.empty_like(flat[0])
        quarter = flat.shape[1] // 4
        for part in range(4):
            piece = slice(
                part * quarter, flat.shape[1] if part == 3 else (part + 1) * quarter
            )
            out[piece] = O.run_program(flat[:, piece], _ring(world, part % world))
        return out.view(tensor.shape)

    return all_reduce


def _rank_main(rank, world, port, backend, kind, shapes, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    device = torch.device(f"cuda:{rank}") if backend == "nccl" else torch.device("cpu")
    dist.init_process_group(backend, rank=rank, world_size=world)

    def backend_all_reduce(tensor):
        dist.all_reduce(tensor)
        return tensor

    all_reduce = (
        backend_all_reduce
        if kind == "backend"
        else _known_all_reduce(world, kind, device)
    )
    exact = ExactReduceScatter(all_reduce, dist.all_to_all_single, world, rank)
    report = []
    for shape in shapes:
        chunk = math.prod(shape) // world
        taken, same = [], True
        for call in range(4):
            data = _random_bf16(shape, seed=1000 * rank + call).to(device)
            want = all_reduce(data.clone()).reshape(-1)[
                rank * chunk : (rank + 1) * chunk
            ]
            want = want.clone()
            # The last call writes into the input's own chunk, as a caller may.
            got = (
                data.view(world, chunk)[rank]
                if call == 3
                else torch.empty(chunk, dtype=torch.bfloat16, device=device)
            )
            taken.append(exact(got, data))
            if taken[-1]:
                same = same and torch.equal(_bits(got), _bits(want))
        report.append((shape, taken, same))
    results.put((rank, report))
    dist.destroy_process_group()


def _run_ranks(world, backend, kind, shapes):
    context = mp.get_context("spawn")
    results = context.Queue()
    port = _free_port()
    ranks = [
        context.Process(
            target=_rank_main, args=(rank, world, port, backend, kind, shapes, results)
        )
        for rank in range(world)
    ]
    for process in ranks:
        process.start()
    reports = dict(results.get(timeout=600) for _ in ranks)
    for process in ranks:
        process.join(60)
        assert process.exitcode == 0
    return reports


_SHAPES = [(64, 128), (8, 4096), (16, 1000)]


def _check(reports, world, must_take):
    for shape_index, shape in enumerate(_SHAPES):
        taken = {tuple(reports[rank][shape_index][1]) for rank in range(world)}
        # Every rank decides alike, and the first sight of a shape never probes.
        assert len(taken) == 1, (shape, taken)
        taken = taken.pop()
        assert taken[0] is False
        assert all(reports[rank][shape_index][2] for rank in range(world)), shape
        if must_take:
            assert taken[1:] == (True, True, True), (shape, taken)
    return [reports[0][i][1] for i in range(len(_SHAPES))]


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("kind", ["wide", "rings"])
def test_known_orders_are_found_and_reproduced_gloo(world, kind):
    if kind == "rings" and world == 2:
        pytest.skip("two ranks have one order")
    _check(_run_ranks(world, "gloo", kind, _SHAPES), world, must_take=True)


@pytest.mark.parametrize("world", [2, 4, 8])
def test_backend_all_reduce_gloo(world):
    # Whether gloo's own order is found is its business; a result that is
    # returned must be the all-reduce's.
    _check(_run_ranks(world, "gloo", "backend", _SHAPES), world, must_take=False)


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("kind", ["wide", "rings", "backend"])
def test_exact_reduce_scatter_nccl(world, kind):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        pytest.skip(f"needs {world} GPUs")
    if kind == "rings" and world == 2:
        pytest.skip("two ranks have one order")
    taken = _check(
        _run_ranks(world, "nccl", kind, _SHAPES), world, must_take=kind != "backend"
    )
    print(f"NCCL world={world} kind={kind}: exact path taken per shape: {taken}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
