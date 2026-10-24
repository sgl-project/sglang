import pytest
import torch

from sglang.kernels.jit.utils import override_jit_cuda_arch
from sglang.kernels.ops.diffusion.common.jit_arch import require_sm80
from sglang.kernels.ops.diffusion.rope import qknorm_rope_jit
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def test_pre_sm80_skips_jit_without_compiling(monkeypatch) -> None:
    """On pre-SM80 GPUs every diffusion JIT build fails, but only after tens of
    seconds of doomed nvcc attempts on the first denoise step (#38516). The
    capability gate must reject before the compiler is ever invoked."""
    compile_calls: list = []
    monkeypatch.setattr(
        qknorm_rope_jit, "load_jit", lambda *a, **k: compile_calls.append(a)
    )
    with override_jit_cuda_arch(7, 5):
        ok = qknorm_rope_jit._can_use_fused_qknorm_rope(
            head_dim=256,
            rope_dim=192,
            is_neox=False,
            dtype=torch.float16,
            cache_dtype=torch.float16,
            round_norm_before_rope=True,
            pack_kv=False,
            cache_has_full_width=False,
        )
    assert not ok
    assert not compile_calls


def test_sm80_floor_boundary() -> None:
    """The floor is exactly SM80: Ampere passes, the last pre-Ampere arch raises."""
    with override_jit_cuda_arch(8, 0):
        require_sm80()
    with override_jit_cuda_arch(7, 5):
        with pytest.raises(RuntimeError, match="SM80"):
            require_sm80()
