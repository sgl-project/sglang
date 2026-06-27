import pytest
import torch

from sglang.multimodal_gen.runtime.models.dits import ltx_2 as ltx2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")

DEVICE = "cuda"
DTYPE = torch.bfloat16
EPS = 1e-6


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)
    ltx2._LTX2_FUSED_QKNORM_SPLIT_ROPE = None
    ltx2._LTX2_FUSED_QKNORM_SPLIT_ROPE_UNAVAILABLE = False
    ltx2._LTX2_FUSED_QKNORM_SPLIT_ROPE_RUNTIME_DISABLED = False


def _make_norm(hidden_size: int) -> torch.nn.RMSNorm:
    norm = torch.nn.RMSNorm(hidden_size, eps=EPS, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden_size, device=DEVICE, dtype=DTYPE))
    return norm


@torch.no_grad()
@pytest.mark.parametrize(
    ("batch", "q_seq", "k_seq", "num_heads", "head_dim"),
    [
        (1, 17, 17, 32, 64),
        (1, 5, 7, 16, 128),
        (1, 4, 4, 32, 128),
    ],
)
def test_ltx2_fused_qknorm_split_rope_pair(batch, q_seq, k_seq, num_heads, head_dim):
    hidden_size = num_heads * head_dim
    half_dim = head_dim // 2
    q = torch.randn(batch, q_seq, hidden_size, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, k_seq, hidden_size, device=DEVICE, dtype=DTYPE)
    q_cos = torch.randn(batch, num_heads, q_seq, half_dim, device=DEVICE, dtype=DTYPE)
    q_sin = torch.randn(batch, num_heads, q_seq, half_dim, device=DEVICE, dtype=DTYPE)
    k_cos = torch.randn(batch, num_heads, k_seq, half_dim, device=DEVICE, dtype=DTYPE)
    k_sin = torch.randn(batch, num_heads, k_seq, half_dim, device=DEVICE, dtype=DTYPE)
    q_norm = _make_norm(hidden_size)
    k_norm = _make_norm(hidden_size)

    actual = ltx2._ltx2_try_fused_qknorm_split_rope(
        q.clone(),
        k.clone(),
        q_norm,
        k_norm,
        EPS,
        (q_cos, q_sin),
        (k_cos, k_sin),
    )
    assert actual is not None
    q_actual, k_actual = actual

    q_expected = ltx2.apply_split_rotary_emb(q_norm(q), (q_cos, q_sin))
    k_expected = ltx2.apply_split_rotary_emb(k_norm(k), (k_cos, k_sin))
    torch.testing.assert_close(q_actual, q_expected, atol=0, rtol=0)
    torch.testing.assert_close(k_actual, k_expected, atol=0, rtol=0)


@torch.no_grad()
def test_ltx2_fused_qknorm_split_rope_rejects_interleaved_rope():
    hidden_size = 4096
    q = torch.randn(1, 4, hidden_size, device=DEVICE, dtype=DTYPE)
    k = torch.randn(1, 4, hidden_size, device=DEVICE, dtype=DTYPE)
    cos = torch.randn(1, 4, hidden_size, device=DEVICE, dtype=DTYPE)
    sin = torch.randn(1, 4, hidden_size, device=DEVICE, dtype=DTYPE)
    q_norm = _make_norm(hidden_size)
    k_norm = _make_norm(hidden_size)

    actual = ltx2._ltx2_try_fused_qknorm_split_rope(
        q, k, q_norm, k_norm, EPS, (cos, sin), None
    )
    assert actual is None


@torch.no_grad()
def test_ltx2_fused_qknorm_split_rope_rejects_fp32_norm_weight():
    hidden_size = 4096
    q = torch.randn(1, 4, hidden_size, device=DEVICE, dtype=DTYPE)
    k = torch.randn(1, 4, hidden_size, device=DEVICE, dtype=DTYPE)
    cos = torch.randn(1, 4, hidden_size, device=DEVICE, dtype=DTYPE)
    sin = torch.randn(1, 4, hidden_size, device=DEVICE, dtype=DTYPE)
    q_norm = torch.nn.RMSNorm(hidden_size, eps=EPS, device=DEVICE, dtype=torch.float32)
    k_norm = torch.nn.RMSNorm(hidden_size, eps=EPS, device=DEVICE, dtype=torch.float32)

    actual = ltx2._ltx2_try_fused_qknorm_split_rope(
        q, k, q_norm, k_norm, EPS, (cos, sin), None
    )
    assert actual is None
