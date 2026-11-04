"""SM107 small-M BF16 GEMM routes: every key bit-equal to the stock dispatch, eager and under graph replay."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
        reason="the routes are SM107 only",
    ),
]

_KINDS = ("randn", "outliers")
_GRAPH_REPLAYS = 20


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _poison(numel: int) -> None:
    """Leave a NaN-filled block in the caching allocator, so an unwritten output element shows as NaN."""
    torch.full((numel,), float("nan"), dtype=torch.bfloat16, device="cuda")


def _inputs(m: int, n: int, k: int, kind: str, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, generator=g, device="cuda")
    if kind == "outliers":
        cols = torch.randint(0, k, (16,), generator=g, device="cuda")
        x[:, cols] *= 50.0
    w = torch.randn(n, k, generator=g, device="cuda") * 0.02
    return x.to(torch.bfloat16), w.to(torch.bfloat16)


@pytest.fixture(scope="module")
def dispatch():
    from sglang.srt.layers.quantization import sm107_bf16_smallm, unquant

    exec_ctx = SimpleNamespace(
        kernel=SimpleNamespace(bf16_gemm_backend="cutedsl"),
        deterministic=SimpleNamespace(enable_deterministic_inference=False),
    )
    platform = SimpleNamespace(is_sm100=True, device_sm=107)
    with (
        mock.patch.object(unquant, "get_exec", lambda: exec_ctx),
        mock.patch.object(unquant, "get_platform", lambda: platform),
        envs.SGLANG_ENABLE_SM107_BF16_SMALLM_ROUTES.override(True),
    ):
        unquant.initialize_bf16_gemm_config()
    unquant.precompile_splitk_tactics()  # the load-time check: the routes serve only after it
    assert unquant._sm107_smallm_on and sm107_bf16_smallm._checked, (
        "routes not installed or failed the load check"
    )
    yield unquant, sm107_bf16_smallm
    unquant._sm107_smallm_on = False


def _stock(unquant, x, w):
    with mock.patch.object(unquant, "_sm107_smallm_on", False):
        return unquant._bf16_gemm_dispatch_impl(x, w, None)


def _kernel_names(fn) -> list:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]


def test_every_key_bitwise_eager_and_graph(dispatch):
    unquant, routes = dispatch
    specializations = {}
    for (m, n, k), (kind, tactic) in sorted(routes.ROUTES.items()):
        for case, data in enumerate(_KINDS):
            x, w = _inputs(m, n, k, data, seed=m * 7 + case)
            ref = _stock(unquant, x, w)
            _poison(m * n)
            got = unquant._bf16_gemm_dispatch_impl(x, w, None)
            assert torch.equal(_bits(ref), _bits(got)), (m, n, k, kind, tactic, data)

        out = {}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            out["y"] = unquant._bf16_gemm_dispatch_impl(x, w, None)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out["y"] = unquant._bf16_gemm_dispatch_impl(x, w, None)
        for _ in range(_GRAPH_REPLAYS):
            out["y"].fill_(float("nan"))
            graph.replay()
            assert torch.equal(_bits(ref), _bits(out["y"])), (m, n, k, "graph")
        del graph

        names = _kernel_names(lambda: unquant._bf16_gemm_dispatch_impl(x, w, None))
        assert len(names) == 1, (m, n, k, names)  # one kernel, no split-K reduce
        specializations[(m, n, k)] = (kind, tactic, names[0])
    print("specializations", specializations)


def test_controls_fire(dispatch):
    """The comparison sees a one-bit change, and a different K slicing of the same GEMM changes some outputs."""
    unquant, routes = dispatch
    x, w = _inputs(8, 512, 2560, "randn", seed=0)
    ref = _stock(unquant, x, w)
    flipped = ref.clone()
    flipped.view(-1).view(torch.int16)[123] ^= 1
    assert int((_bits(ref) != _bits(flipped)).sum()) == 1

    differing = 0
    for m in range(9, 17):
        for n in (512, 640):
            x, w = _inputs(m, n, 2560, "randn", seed=100 + m)
            ref = _stock(unquant, x, w)
            reordered = torch.empty_like(ref)
            routes._cluster_out(
                x, w, reordered, (64, 16, 8, 5, 64)
            )  # 8 slices where cuBLAS sums 4
            differing += int((_bits(ref) != _bits(reordered)).sum())
    assert differing > 0
