"""``diffusion.rope``: the LTX-2 QK-norm + split-RoPE CUDA kernel.

Split out of ``test_rope.py`` rather than merged with the other RoPE kernels:
the lossless-default path is validated on B200, while the explicitly
quality-gated SM90 path is also checked on the large-GPU lane. The
``fused_inplace_qknorm_rope`` cases there are held to the *split* baseline,
whose sgl_kernel / FlashInfer dispatch differs on Blackwell -- their bit-exact
assertions fail on B200.  One file cannot carry both lane sets.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import sglang.kernels.kda_kernels.ltx2_qknorm_split_rope_jit as ltx2_qknorm_jit
import sglang.kernels.ops.diffusion.rope.ltx2_qknorm_split_rope_vibecuda_jit as ltx2_vibecuda
from sglang.kernels.jit.utils import load_jit
from sglang.kernels.kda_kernels import _cuda_source
from sglang.kernels.ops.diffusion import (
    can_use_ltx2_qknorm_split_rope_cuda,
    ltx2_qknorm_split_rope_cuda,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")

DEVICE = "cuda"
BF16_FUSED_ATOL = 1.6e-1


def test_ltx2_qknorm_hopper_requires_explicit_quality_gate(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(ltx2_qknorm_jit, "_is_sm100_or_newer", lambda _x: False)
    monkeypatch.setattr(ltx2_qknorm_jit, "_is_sm90", lambda _x: True)
    monkeypatch.setattr(ltx2_qknorm_jit, "_supported_side", lambda *_a, **_k: True)

    args = (sentinel,) * 8
    assert not ltx2_qknorm_jit.can_use_ltx2_qknorm_split_rope_cuda(
        *args, num_heads=32, head_dim=128
    )
    assert ltx2_qknorm_jit.can_use_ltx2_qknorm_split_rope_cuda(
        *args, num_heads=32, head_dim=128, allow_sm90=True
    )


def _require_b200() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("LTX2 QKNorm split-RoPE CUDA path is validated on B200")


def _require_blackwell_backend(backend: str) -> None:
    _require_b200()
    if backend == "vibecuda" and torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("vibecuda LTX2 path requires SM100 or SM103")


def test_ltx2_qknorm_vibecuda_rejects_other_architectures(monkeypatch) -> None:
    monkeypatch.setattr(
        ltx2_vibecuda,
        "current_arch",
        lambda: SimpleNamespace(cuda=SimpleNamespace(capability=(9, 0))),
    )
    q = SimpleNamespace(is_cuda=True)
    sentinel = object()
    with pytest.raises(RuntimeError, match="requires SM100 or SM103"):
        ltx2_vibecuda.can_use_ltx2_qknorm_split_rope_vibecuda(
            q,
            sentinel,
            sentinel,
            sentinel,
            sentinel,
            sentinel,
            sentinel,
            sentinel,
            num_heads=32,
            head_dim=128,
        )


def test_ltx2_qknorm_vibecuda_registry_target() -> None:
    from sglang.kernels import KernelBackend, select_kernel

    spec = select_kernel(
        "diffusion.ltx2_qknorm_split_rope", backend=KernelBackend.VIBECUDA
    )
    assert spec.load() is ltx2_vibecuda.ltx2_qknorm_split_rope_vibecuda


def _require_sm90() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.version.hip is not None or torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("quality-gated LTX2 Hopper path requires SM90")


def _ltx2_make_cos_sin(
    batch: int, seq_len: int, num_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    half_dim = head_dim // 2
    cos = torch.randn(
        batch, seq_len, num_heads, half_dim, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    sin = torch.randn(
        batch, seq_len, num_heads, half_dim, device="cuda", dtype=torch.bfloat16
    ).transpose(1, 2)
    return cos, sin


def _apply_split_rotary_ref(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    x_dtype = x.dtype
    batch = x.shape[0]
    _, num_heads, seq_len, _ = cos.shape
    last = 2 * cos.shape[-1]
    x = x.reshape(batch, seq_len, num_heads, last).swapaxes(1, 2)
    half = last // 2

    split_x = x.reshape(*x.shape[:-1], 2, half)
    first_x = split_x[..., :1, :]
    second_x = split_x[..., 1:, :]
    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    out[..., :1, :].addcmul_(-sin_u, second_x)
    out[..., 1:, :].addcmul_(sin_u, first_x)
    out = out.reshape(*out.shape[:-2], last)
    return (
        out.swapaxes(1, 2).reshape(batch, seq_len, num_heads * last).to(dtype=x_dtype)
    )


def _ltx2_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # rms_norm isn't autocast fp32-preserving, so feed fp32 inputs directly
    # to keep the normalized value unrounded until the final RoPE output.
    q_norm = F.rms_norm(q.float(), (q.shape[-1],), q_weight.float(), eps)
    k_norm = F.rms_norm(k.float(), (k.shape[-1],), k_weight.float(), eps)
    q_ref = _apply_split_rotary_ref(q_norm, q_cos, q_sin)
    k_ref = _apply_split_rotary_ref(k_norm, k_cos, k_sin)
    return q_ref.to(dtype=torch.bfloat16), k_ref.to(dtype=torch.bfloat16)


def test_ltx2_qknorm_hopper_quality_path_matches_within_bf16() -> None:
    _require_sm90()
    torch.cuda.manual_seed(20260908)
    batch, q_seq, k_seq, num_heads, head_dim = 1, 17, 9, 32, 64
    hidden = num_heads * head_dim
    eps = 1e-6
    q = torch.randn(batch, q_seq, hidden, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, k_seq, hidden, device="cuda", dtype=torch.bfloat16)
    q_cos, q_sin = _ltx2_make_cos_sin(batch, q_seq, num_heads, head_dim)
    k_cos, k_sin = _ltx2_make_cos_sin(batch, k_seq, num_heads, head_dim)
    q_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    assert not can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    assert can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        num_heads=num_heads,
        head_dim=head_dim,
        allow_sm90=True,
    )

    q_norm = F.rms_norm(q, (hidden,), q_weight, eps)
    k_norm = F.rms_norm(k, (hidden,), k_weight, eps)
    q_ref = _apply_split_rotary_ref(q_norm, q_cos, q_sin)
    k_ref = _apply_split_rotary_ref(k_norm, k_cos, k_sin)
    q_out, k_out = ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        eps=eps,
        num_heads=num_heads,
        head_dim=head_dim,
        allow_sm90=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(q_out, q_ref, rtol=0, atol=BF16_FUSED_ATOL)
    torch.testing.assert_close(k_out, k_ref, rtol=0, atol=BF16_FUSED_ATOL)


@pytest.fixture(scope="module")
def ltx2_hopper_baseline():
    _require_sm90()
    # Keep the original scalar CUDA implementation as the exact rounding oracle.
    return load_jit(
        "test_ltx2_qknorm_hopper_baseline",
        cuda_files=[_cuda_source("diffusion/ltx2_qknorm_split_rope.cuh")],
        cuda_wrappers=[
            ("run", "ltx2_qknorm_split_rope::LTX2QKNormSplitRopeKernel::run<true>")
        ],
    )


@pytest.mark.parametrize(
    "batch,q_seq,k_seq,num_heads,head_dim,q_pad,q_offset,k_pad,k_offset",
    [
        (1, 7, 3, 32, 128, 0, 0, 0, 0),
        (2, 5, 2, 32, 64, 0, 0, 0, 0),
        (2, 5, 2, 32, 128, 1, 0, 0, 0),
        (1, 7, 3, 32, 128, 0, 1, 0, 2),
        (1, 7, 3, 32, 128, 0, 2, 0, 0),
        (2, 4, 3, 2, 6, 0, 0, 1, 1),
        (2, 4, 3, 4, 4, 0, 0, 0, 0),
        (1, 3, 2, 128, 8, 0, 0, 0, 0),
        (1, 0, 3, 32, 128, 0, 0, 0, 0),
        (1, 3, 0, 32, 128, 0, 0, 0, 0),
        (1, 0, 0, 32, 128, 0, 0, 0, 0),
    ],
)
def test_ltx2_qknorm_hopper_alignment_and_empty_sides(
    ltx2_hopper_baseline,
    batch,
    q_seq,
    k_seq,
    num_heads,
    head_dim,
    q_pad,
    q_offset,
    k_pad,
    k_offset,
) -> None:
    torch.cuda.manual_seed(20260927)
    hidden = num_heads * head_dim

    def packed(shape, offset):
        size = 1
        for dim in shape:
            size *= dim
        return torch.randn(size + offset, device=DEVICE, dtype=torch.bfloat16)[
            offset:
        ].view(shape)

    def side(seq_len, pad, offset):
        shape = (batch, num_heads, seq_len, head_dim // 2)
        width = head_dim // 2 + pad
        strides = (seq_len * num_heads * width, width, num_heads * width, 1)
        size = (
            offset
            + 1
            + sum(max(dim - 1, 0) * stride for dim, stride in zip(shape, strides))
        )
        rope = [
            torch.randn(size, device=DEVICE, dtype=torch.bfloat16).as_strided(
                shape, strides, offset
            )
            for _ in range(2)
        ]
        return [
            packed((batch, seq_len, hidden), offset),
            *rope,
            packed((hidden,), offset),
        ]

    args = side(q_seq, q_pad, q_offset) + side(k_seq, k_pad, k_offset)

    def baseline():
        outputs = (torch.empty_like(args[0]), torch.empty_like(args[4]))
        ltx2_hopper_baseline.run(*outputs, *args, 1e-6, num_heads, head_dim)
        return outputs

    def candidate():
        return ltx2_qknorm_split_rope_cuda(
            *args, eps=1e-6, num_heads=num_heads, head_dim=head_dim, allow_sm90=True
        )

    def assert_exact(actual, expected):
        for output, reference in zip(actual, expected):
            assert output.shape == reference.shape
            assert output.stride() == reference.stride()
            assert output.dtype == reference.dtype
            assert torch.equal(output.view(torch.uint8), reference.view(torch.uint8))

    assert_exact(candidate(), baseline())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        candidate()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate()
    # Replay must consume current values, including on mixed scalar/vector paths.
    for value in args:
        value.add_(0.125)
    graph.replay()
    assert_exact(captured, baseline())


@pytest.mark.parametrize("backend", ["kda", "vibecuda"])
@pytest.mark.parametrize(
    "batch,q_seq,k_seq,num_heads,head_dim",
    [
        (1, 3, 3, 32, 128),
        (1, 5, 2, 32, 64),
        (2, 4, 3, 32, 64),
    ],
)
def test_ltx2_qknorm_split_rope_matches_torch_exactly(
    batch: int, q_seq: int, k_seq: int, num_heads: int, head_dim: int, backend: str
) -> None:
    _require_blackwell_backend(backend)
    torch.cuda.manual_seed(20260630)
    hidden = num_heads * head_dim
    eps = 1e-6
    q = torch.randn(batch, q_seq, hidden, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, k_seq, hidden, device="cuda", dtype=torch.bfloat16)
    q_cos, q_sin = _ltx2_make_cos_sin(batch, q_seq, num_heads, head_dim)
    k_cos, k_sin = _ltx2_make_cos_sin(batch, k_seq, num_heads, head_dim)
    q_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    assert can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        num_heads=num_heads,
        head_dim=head_dim,
        backend=backend,
    )

    q_ref, k_ref = _ltx2_reference(
        q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight, eps
    )
    q_out, k_out = ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        eps=eps,
        num_heads=num_heads,
        head_dim=head_dim,
        backend=backend,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(q_out, q_ref, rtol=0, atol=BF16_FUSED_ATOL)
    torch.testing.assert_close(k_out, k_ref, rtol=0, atol=BF16_FUSED_ATOL)


@pytest.mark.parametrize("backend", ["kda", "vibecuda"])
@pytest.mark.parametrize(
    "batch,q_seq,k_seq,num_heads,head_dim",
    [
        (1, 7, 5, 32, 128),
        (2, 4, 9, 32, 64),
    ],
)
def test_ltx2_qknorm_split_rope_vectorized_matches_scalar(
    batch: int, q_seq: int, k_seq: int, num_heads: int, head_dim: int, backend: str
) -> None:
    _require_blackwell_backend(backend)
    torch.cuda.manual_seed(20261023)
    hidden = num_heads * head_dim
    q = torch.randn(batch, q_seq, hidden, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, k_seq, hidden, device="cuda", dtype=torch.bfloat16)
    q_cos, q_sin = _ltx2_make_cos_sin(batch, q_seq, num_heads, head_dim)
    k_cos, k_sin = _ltx2_make_cos_sin(batch, k_seq, num_heads, head_dim)
    q_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    def misaligned(table: torch.Tensor) -> torch.Tensor:
        # same values off a 16-byte boundary, which keeps the scalar loop
        storage = torch.empty(table.numel() + 1, device="cuda", dtype=table.dtype)
        copy = storage[1:].view(table.contiguous().shape)
        copy.copy_(table)
        return copy

    def run(tables):
        return ltx2_qknorm_split_rope_cuda(
            q,
            tables[0],
            tables[1],
            q_weight,
            k,
            tables[2],
            tables[3],
            k_weight,
            eps=1e-6,
            num_heads=num_heads,
            head_dim=head_dim,
            backend=backend,
        )

    tables = (q_cos, q_sin, k_cos, k_sin)
    vectorized = run(tables)
    scalar = run(tuple(misaligned(t) for t in tables))
    for actual, expected in zip(vectorized, scalar):
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize(
    "batch,q_seq,k_seq,num_heads,head_dim,q_pad,q_offset,k_pad,k_offset",
    [
        (1, 7, 3, 32, 128, 0, 0, 0, 0),
        (2, 5, 2, 32, 64, 0, 0, 0, 0),
        (2, 5, 2, 32, 128, 1, 0, 0, 0),
        (1, 7, 3, 32, 128, 0, 1, 0, 2),
        (1, 7, 3, 32, 128, 0, 2, 0, 0),
        (1, 3, 2, 128, 8, 0, 0, 0, 0),
        (1, 0, 3, 32, 128, 0, 0, 0, 0),
        (1, 3, 0, 32, 128, 0, 0, 0, 0),
        (1, 0, 0, 32, 128, 0, 0, 0, 0),
    ],
)
def test_ltx2_qknorm_vibecuda_alignment_empty_stream_and_graph(
    batch,
    q_seq,
    k_seq,
    num_heads,
    head_dim,
    q_pad,
    q_offset,
    k_pad,
    k_offset,
) -> None:
    _require_blackwell_backend("vibecuda")
    torch.cuda.manual_seed(20260927)
    hidden = num_heads * head_dim

    def packed(shape, offset):
        size = 1
        for dim in shape:
            size *= dim
        return torch.randn(size + offset, device=DEVICE, dtype=torch.bfloat16)[
            offset:
        ].view(shape)

    def side(seq_len, pad, offset):
        shape = (batch, num_heads, seq_len, head_dim // 2)
        width = head_dim // 2 + pad
        strides = (seq_len * num_heads * width, width, num_heads * width, 1)
        size = (
            offset
            + 1
            + sum(max(dim - 1, 0) * stride for dim, stride in zip(shape, strides))
        )
        rope = [
            torch.randn(size, device=DEVICE, dtype=torch.bfloat16).as_strided(
                shape, strides, offset
            )
            for _ in range(2)
        ]
        return [
            packed((batch, seq_len, hidden), offset),
            *rope,
            packed((hidden,), offset),
        ]

    args = side(q_seq, q_pad, q_offset) + side(k_seq, k_pad, k_offset)
    saved = [value.clone() for value in args]

    def candidate():
        return ltx2_qknorm_split_rope_cuda(
            *args, eps=1e-6, num_heads=num_heads, head_dim=head_dim, backend="vibecuda"
        )

    def reference():
        return _ltx2_reference(
            args[0],
            args[4],
            args[1],
            args[2],
            args[5],
            args[6],
            args[3],
            args[7],
            1e-6,
        )

    def assert_exact(actual, expected):
        for output, expected_output in zip(actual, expected):
            assert output.shape == expected_output.shape
            assert output.stride() == expected_output.stride()
            assert output.dtype == expected_output.dtype
            assert torch.equal(
                output.view(torch.uint8), expected_output.view(torch.uint8)
            )

    assert_exact(candidate(), reference())
    assert all(torch.equal(value, original) for value, original in zip(args, saved))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        candidate()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = candidate()
    for value in args:
        value.add_(0.125)
    for output in captured:
        output.fill_(float("nan"))
    graph.replay()
    assert_exact(captured, reference())


@pytest.mark.parametrize("backend", ["kda", "vibecuda"])
def test_ltx2_qknorm_split_rope_reuses_supplied_outputs(backend: str) -> None:
    _require_blackwell_backend(backend)
    torch.cuda.manual_seed(20261125)
    batch, q_seq, k_seq, num_heads, head_dim = 1, 7, 5, 32, 64
    hidden = num_heads * head_dim
    q = torch.randn(batch, q_seq, hidden, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, k_seq, hidden, device="cuda", dtype=torch.bfloat16)
    q_cos, q_sin = _ltx2_make_cos_sin(batch, q_seq, num_heads, head_dim)
    k_cos, k_sin = _ltx2_make_cos_sin(batch, k_seq, num_heads, head_dim)
    q_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    args = (q, q_cos, q_sin, q_weight, k, k_cos, k_sin, k_weight)
    expected = _ltx2_reference(
        q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight, 1e-6
    )
    outputs = (torch.full_like(q, float("nan")), torch.full_like(k, float("nan")))

    def call(out, call_args=args):
        return ltx2_qknorm_split_rope_cuda(
            *call_args,
            eps=1e-6,
            num_heads=num_heads,
            head_dim=head_dim,
            backend=backend,
            out=out,
        )

    actual = call(outputs)
    assert actual[0] is outputs[0] and actual[1] is outputs[1]
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call(outputs)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(outputs)
    for value in args:
        value.add_(0.125)
    expected = _ltx2_reference(
        q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight, 1e-6
    )
    for output in outputs:
        output.fill_(float("nan"))
    graph.replay()
    assert torch.equal(outputs[0], expected[0])
    assert torch.equal(outputs[1], expected[1])

    invalid = (torch.empty_like(q, dtype=torch.float32), torch.empty_like(k))
    with pytest.raises(RuntimeError):
        call(invalid)
    with pytest.raises(RuntimeError, match="must not alias"):
        call((q, k))

    aliased_storage = torch.empty(q.numel() + 1, device=q.device, dtype=q.dtype)
    aliased_q = aliased_storage[:-1].view_as(q)
    aliased_q.copy_(q)
    with pytest.raises(RuntimeError, match="must not alias"):
        call(
            (aliased_storage[1:].view_as(q), outputs[1]),
            (aliased_q, *args[1:]),
        )

    overlapping_storage = torch.empty(
        max(q.numel(), k.numel()) + 1, device=q.device, dtype=q.dtype
    )
    overlapping_outputs = (
        overlapping_storage[: q.numel()].view_as(q),
        overlapping_storage[1 : k.numel() + 1].view_as(k),
    )
    with pytest.raises(RuntimeError, match="must not alias"):
        call(overlapping_outputs)

    noncontiguous_q_out = torch.empty(
        (batch, q_seq, 2 * hidden), device=q.device, dtype=q.dtype
    )[..., ::2]
    with pytest.raises(RuntimeError, match="must be contiguous"):
        call((noncontiguous_q_out, outputs[1]))


@pytest.mark.parametrize("backend", ["kda", "vibecuda"])
def test_ltx2_qknorm_split_rope_rejects_unsupported_inputs(backend: str) -> None:
    _require_blackwell_backend(backend)
    torch.cuda.manual_seed(20260630)
    q = torch.randn((1, 3, 4096), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    q_cos, q_sin = _ltx2_make_cos_sin(1, 3, 32, 128)
    q_weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16)

    assert can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        q_cos,
        q_sin,
        k_weight,
        num_heads=32,
        head_dim=128,
        backend=backend,
    )
    strided_q_weight = q_weight.new_empty(2 * q_weight.numel())[::2]
    assert not can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        strided_q_weight,
        k,
        q_cos,
        q_sin,
        k_weight,
        num_heads=32,
        head_dim=128,
        backend=backend,
    )
    assert not can_use_ltx2_qknorm_split_rope_cuda(
        q.float(),
        q_cos,
        q_sin,
        q_weight,
        k,
        q_cos,
        q_sin,
        k_weight,
        num_heads=32,
        head_dim=128,
        backend=backend,
    )
    assert not can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        q_cos.transpose(-1, -2),
        q_sin,
        k_weight,
        num_heads=32,
        head_dim=128,
        backend=backend,
    )
    if backend == "vibecuda":
        with pytest.raises(RuntimeError, match="head_dim % 8"):
            ltx2_qknorm_split_rope_cuda(
                q,
                q_cos,
                q_sin,
                q_weight,
                k,
                q_cos,
                q_sin,
                k_weight,
                eps=1e-6,
                num_heads=32,
                head_dim=4,
                backend=backend,
            )
    with pytest.raises(ValueError, match="unknown LTX2"):
        ltx2_qknorm_split_rope_cuda(
            q,
            q_cos,
            q_sin,
            q_weight,
            k,
            q_cos,
            q_sin,
            k_weight,
            eps=1e-6,
            num_heads=32,
            head_dim=128,
            backend="invalid",
        )


@pytest.mark.parametrize("backend", ["kda", "vibecuda"])
def test_ltx2_qknorm_split_rope_custom_op_torch_compile_fullgraph(
    backend: str,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    hopper = torch.cuda.get_device_capability() == (9, 0)
    if hopper and backend == "vibecuda":
        pytest.skip("vibecuda LTX2 path requires SM100 or SM103")
    if not hopper:
        _require_blackwell_backend(backend)
    torch.cuda.manual_seed(20260630)
    batch, q_seq, k_seq, num_heads, head_dim = 1, 3, 2, 32, 64
    hidden = num_heads * head_dim
    q = torch.randn(batch, q_seq, hidden, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, k_seq, hidden, device="cuda", dtype=torch.bfloat16)
    q_cos, q_sin = _ltx2_make_cos_sin(batch, q_seq, num_heads, head_dim)
    k_cos, k_sin = _ltx2_make_cos_sin(batch, k_seq, num_heads, head_dim)
    q_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    k_weight = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)

    def fn(q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight):
        return ltx2_qknorm_split_rope_cuda(
            q,
            q_cos,
            q_sin,
            q_weight,
            k,
            k_cos,
            k_sin,
            k_weight,
            eps=1e-6,
            num_heads=num_heads,
            head_dim=head_dim,
            allow_sm90=hopper,
            backend=backend,
        )

    compiled = torch.compile(fn, fullgraph=True)
    q_out, k_out = compiled(q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight)
    if hopper:
        expected = fn(q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight)
        torch.testing.assert_close(q_out, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(k_out, expected[1], rtol=0, atol=0)
        return
    q_ref, k_ref = _ltx2_reference(
        q, k, q_cos, q_sin, k_cos, k_sin, q_weight, k_weight, 1e-6
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(q_out, q_ref, rtol=0, atol=BF16_FUSED_ATOL)
    torch.testing.assert_close(k_out, k_ref, rtol=0, atol=BF16_FUSED_ATOL)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
