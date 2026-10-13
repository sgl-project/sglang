"""``replay_launches``: a Triton wrapper replays its recorded launches per signature.

A replayed call must equal the wrapper's own result bit-for-bit without running the
wrapper. A signature whose wrapper does more than launch into fresh outputs must stay
on the normal path. Replay must work inside CUDA graph capture and stay out of
torch.compile tracing.
"""

import sys

import pytest
import torch
import triton
import triton.language as tl
from triton.runtime.jit import JITFunction

from sglang.kernels.kda_kernels.layernorm_modulate_triton import (
    fused_layernorm_modulate_raw,
)
from sglang.kernels.ops.diffusion.activation.silu_mul_bitexact import (
    fused_packed_silu_mul_bitexact,
    fused_silu_mul_bitexact,
)
from sglang.kernels.ops.diffusion.common import fast_launch
from sglang.kernels.ops.diffusion.common.fast_launch import replay_launches
from sglang.kernels.ops.diffusion.norm.native_bf16_rmsnorm_triton import (
    rmsnorm_scale,
    rmsnorm_tanh_residual,
)
from sglang.kernels.ops.diffusion.norm.zimage_qk_rmsnorm_triton import (
    zimage_qk_rmsnorm_native,
)
from sglang.kernels.ops.diffusion.rope.flux2_qknorm_rope_triton import (
    flux2_strided_qknorm_rope,
)
from sglang.kernels.ops.diffusion.rope.rope_rotate_half_bitexact import (
    fused_rope_rotate_half_bitexact,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

EPS = 1e-6


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setattr(fast_launch, "_ENABLED", True)
    torch.manual_seed(0)


def _bf16(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


def _qkv_slices(tokens, heads):
    """q and k of a fused qkv projection: strided tokens, packed heads."""
    qkv = _bf16(1, tokens, 3 * heads * 128)
    q = qkv[..., : heads * 128].unflatten(-1, (heads, 128))
    k = qkv[..., heads * 128 : 2 * heads * 128].unflatten(-1, (heads, 128))
    return q, k


WRAPPERS = {
    "rmsnorm_scale": lambda: (
        rmsnorm_scale,
        (_bf16(1, 1088, 3840), _bf16(3840), _bf16(1, 1, 3840), EPS),
    ),
    "rmsnorm_tanh_residual": lambda: (
        rmsnorm_tanh_residual,
        (_bf16(2, 77, 2560), _bf16(2, 1, 2560), _bf16(2, 77, 2560), _bf16(2560), EPS),
    ),
    "zimage_qk_rmsnorm": lambda: (
        zimage_qk_rmsnorm_native,
        (_qkv_slices(1088, 30)[0], _bf16(128), EPS),
    ),
    "layernorm_modulate": lambda: (
        fused_layernorm_modulate_raw,
        (_bf16(2, 256, 3072), _bf16(2, 3072), _bf16(2, 3072), EPS),
    ),
    "layernorm_modulate_scale_only": lambda: (
        fused_layernorm_modulate_raw,
        (_bf16(1, 256, 3072), _bf16(1, 3072), None, EPS),
    ),
    "silu_mul": lambda: (
        fused_silu_mul_bitexact,
        (_bf16(2, 256, 1024), _bf16(2, 256, 1024)),
    ),
    "packed_silu_mul": lambda: (fused_packed_silu_mul_bitexact, (_bf16(2, 256, 2048),)),
    "flux2_qknorm_rope": lambda: (
        flux2_strided_qknorm_rope,
        (
            *_qkv_slices(256, 24),
            _bf16(128),
            _bf16(128),
            torch.randn(256, 128, device="cuda"),
            EPS,
        ),
    ),
    "rope_rotate_half": lambda: (
        fused_rope_rotate_half_bitexact,
        (_bf16(2, 64, 8, 128), _bf16(128, 64), _bf16(128, 64)),
    ),
}


def _no_jit(self, *args, **kwargs):
    raise AssertionError("JIT launch")


def _equal(a, b):
    if isinstance(a, tuple):
        return all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    return torch.equal(a, b)


def _normal_path(monkeypatch, fn, args):
    monkeypatch.setattr(fast_launch, "_ENABLED", False)
    out = fn(*args)
    monkeypatch.setattr(fast_launch, "_ENABLED", True)
    return out


MODES = {"no_grad": torch.no_grad, "inference_mode": torch.inference_mode}


@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize("name", sorted(WRAPPERS))
def test_replay_matches_wrapper_without_running_it(monkeypatch, name, mode):
    with MODES[mode]():
        fn, args = WRAPPERS[name]()
        fn(*args)  # records this signature
        fresh = WRAPPERS[name]()[1]
        expected = _normal_path(monkeypatch, fn, fresh)

        monkeypatch.setattr(JITFunction, "run", _no_jit)
        assert _equal(fn(*fresh), expected)


@triton.jit
def _copy_kernel(dst, src, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(dst + offsets, tl.load(src + offsets, mask=mask), mask=mask)


def _copy(dst, src):
    _copy_kernel[(triton.cdiv(src.numel(), 1024),)](dst, src, src.numel(), BLOCK=1024)


class _Counted:
    """A decorated toy wrapper that counts how often its body runs."""

    def __init__(self, body):
        self.calls = 0

        def fn(*args):
            self.calls += 1
            return body(*args)

        self.fn = replay_launches(fn)


def _copy_out(x):
    out = torch.empty_like(x)
    _copy(out, x)
    return out


def _float(n=4096, offset=0):
    return torch.randn(n + offset, device="cuda")[offset:]


@torch.no_grad()
def test_toy_wrapper_replays():
    toy = _Counted(_copy_out)
    for _ in range(3):
        x = _float()
        assert torch.equal(toy.fn(x), x)
    assert toy.calls == 1


def _copy_out_through_view(x):
    out = torch.empty_like(x)
    _copy(out.reshape(-1), x.reshape(-1))
    return out


@pytest.mark.parametrize("mode", sorted(MODES))
def test_views_replay(mode):
    # Inference tensors skip the autograd decomposition: reshape reaches the
    # recording guard as aten.reshape rather than aten.view.
    toy = _Counted(_copy_out_through_view)
    with MODES[mode]():
        for _ in range(3):
            x = _float().view(64, 64)
            assert torch.equal(toy.fn(x), x)
    assert toy.calls == 1


@torch.no_grad()
def test_misaligned_operand_records_its_own_launch():
    toy = _Counted(_copy_out)
    toy.fn(_float())
    x = _float(offset=1)
    assert x.data_ptr() % 16 != 0
    assert torch.equal(toy.fn(x), x)
    assert toy.calls == 2


def _zeros_out(x):
    out = torch.zeros_like(x)
    _copy(out, x)
    return out


def _via_temporary(x):
    tmp = torch.empty_like(x)
    _copy(tmp, x)
    return _copy_out(tmp)


def _reads_value(x, s):
    out = torch.empty_like(x)
    if s.item() > 0:
        _copy(out, x)
    return out


@torch.no_grad()
@pytest.mark.parametrize(
    "body,extra",
    [
        (_zeros_out, tuple),
        (_via_temporary, tuple),
        (_reads_value, lambda: (torch.ones((), device="cuda"),)),
        (lambda x, sizes: _copy_out(x), lambda: ([1, 2],)),
    ],
    ids=["aten-compute", "temporary", "value-read", "list-argument"],
)
def test_more_than_launches_stays_on_normal_path(body, extra):
    toy = _Counted(body)
    for _ in range(3):
        x = _float()
        assert torch.equal(toy.fn(x, *extra()), x)
    assert toy.calls == 3


@torch.no_grad()
def test_none_and_in_place_results():
    unsupported = _Counted(lambda x: None)
    for _ in range(2):
        assert unsupported.fn(_float()) is None
    assert unsupported.calls == 1

    def copy_into(dst, src):
        _copy(dst, src)
        return dst

    in_place = _Counted(copy_into)
    for _ in range(2):
        dst, src = _float(), _float()
        assert in_place.fn(dst, src) is dst and torch.equal(dst, src)
    assert in_place.calls == 1

    def copy_only(dst, src):
        _copy(dst, src)

    no_result = _Counted(copy_only)
    for _ in range(2):
        dst, src = _float(), _float()
        assert no_result.fn(dst, src) is None and torch.equal(dst, src)
    assert no_result.calls == 1


@torch.no_grad()
def test_scalars_key_by_type():
    toy = _Counted(lambda x, flag: _copy_out(x))
    for flag in (True, 1, 1.0, True):
        toy.fn(_float(), flag)
    assert toy.calls == 3


@torch.no_grad()
def test_nested_wrappers_replay_as_one():
    inner = _Counted(_copy_out)

    def chain(x):
        a = inner.fn(x)
        return a, inner.fn(a)

    outer = _Counted(chain)
    for _ in range(3):
        x = _float()
        a, b = outer.fn(x)
        assert torch.equal(a, x) and torch.equal(b, x)
    assert outer.calls == 1 and inner.calls == 2


@torch.no_grad()
def test_capture_replays_and_records():
    toy = _Counted(_copy_out)
    seen, unseen = _float(), _float(2048)
    toy.fn(seen)
    _copy_out(unseen)  # compile the kernel for this signature outside the capture

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_seen = toy.fn(seen)
        out_unseen = toy.fn(unseen)  # recorded inside the capture
    for t in (seen, unseen):
        t.copy_(torch.randn_like(t))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_seen, seen) and torch.equal(out_unseen, unseen)

    calls = toy.calls
    assert torch.equal(toy.fn(unseen), unseen)
    assert toy.calls == calls


@torch.no_grad()
def test_compile_traces_the_wrapper(monkeypatch):
    fn, args = WRAPPERS["rmsnorm_scale"]()
    expected = _normal_path(monkeypatch, fn, args)
    fn(*args)  # recorded; tracing must still run the wrapper
    compiled = torch.compile(fn, fullgraph=True, backend="eager")
    assert torch.equal(compiled(*args), expected)


@torch.no_grad()
def test_disabled_runs_the_wrapper(monkeypatch):
    toy = _Counted(_copy_out)
    monkeypatch.setattr(fast_launch, "_ENABLED", False)
    for _ in range(2):
        toy.fn(_float())
    assert toy.calls == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
