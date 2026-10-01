import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="nightly-amd-1-gpu-mi45x", nightly=True)


def _is_gfx1250() -> bool:
    if not torch.cuda.is_available() or not torch.version.hip:
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return "gfx1250" in arch


def test_wan_cross_launch_selection_cpu(monkeypatch):
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        select_fav3_launch_config,
    )

    monkeypatch.delenv("SGLANG_GLUON_FAV3_WAN_FIXED_SHIFT", raising=False)
    config = select_fav3_launch_config(
        2,
        40,
        176_400,
        seqlen_k=512,
        num_cus=1,
    )
    assert config.schedule == "wan_cross"
    assert (config.block_m, config.block_n, config.num_warps) == (128, 256, 4)
    assert config.tdm_output
    assert config.llvm_fn_attrs == "amdgpu-sched-strategy=iterative-ilp"
    nearby_shapes = (
        (1, 40, 176_400, 512),
        (2, 39, 176_400, 512),
        (2, 40, 176_399, 512),
        (2, 40, 176_400, 511),
    )
    for batch, num_heads, seqlen_q, seqlen_k in nearby_shapes:
        nearby = select_fav3_launch_config(
            batch,
            num_heads,
            seqlen_q,
            seqlen_k=seqlen_k,
            num_cus=1,
        )
        assert nearby.schedule != "wan_cross"

    stable_self_config = select_fav3_launch_config(
        2,
        40,
        176_400,
        seqlen_k=176_400,
        num_cus=256,
    )
    assert stable_self_config.schedule == "pingpong"

    monkeypatch.setenv("SGLANG_GLUON_FAV3_WAN_FIXED_SHIFT", "1")
    self_config = select_fav3_launch_config(
        2,
        40,
        176_400,
        seqlen_k=176_400,
        num_cus=256,
    )
    assert self_config.schedule == "wan_self"


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
@pytest.mark.parametrize(
    "batch,seqlen_q,seqlen_k,num_heads,expected_schedule",
    [
        (1, 129, 65, 8, "pipeline"),
        (1, 300, 513, 8, "pipeline"),
        (1, 2050, 512, 40, "pingpong"),
        (2, 176_400, 512, 40, "wan_cross"),
        (2, 176_400, 176_400, 40, "wan_self"),
    ],
    ids=[
        "short-k-tail",
        "partial-rectangular-pipeline",
        "wide-pingpong-tdm",
        "wan-cross-specialized-large-offset",
        "wan-self-fixed-shift",
    ],
)
def test_gluon_fav3_matches_aiter(
    batch, seqlen_q, seqlen_k, num_heads, expected_schedule, monkeypatch
):
    aiter = pytest.importorskip("aiter")
    if expected_schedule == "wan_self":
        monkeypatch.setenv("SGLANG_GLUON_FAV3_WAN_FIXED_SHIFT", "1")
    else:
        monkeypatch.delenv("SGLANG_GLUON_FAV3_WAN_FIXED_SHIFT", raising=False)
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        select_fav3_launch_config,
    )
    from sglang.multimodal_gen.runtime.layers.attention.layer import (
        LocalAttention,
    )
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

    torch.manual_seed(0)
    query = torch.randn(
        batch,
        seqlen_q,
        num_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        batch,
        seqlen_k,
        num_heads,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)

    expected = aiter.flash_attn_func(
        query,
        key,
        value,
        dropout_p=0.0,
        causal=False,
        return_attn_probs=False,
        return_lse=False,
    )
    attention = LocalAttention(
        num_heads=num_heads,
        num_kv_heads=num_heads,
        head_size=128,
        softmax_scale=128**-0.5,
        required_attention_backend=AttentionBackendEnum.GLUON_FAV3,
        compute_dtype=torch.bfloat16,
        is_cross_attention=seqlen_q != seqlen_k,
    )
    assert attention.backend is AttentionBackendEnum.GLUON_FAV3
    with set_forward_context(current_timestep=0, attn_metadata=None):
        actual = attention(query, key, value)
    torch.cuda.synchronize()

    config = select_fav3_launch_config(batch, num_heads, seqlen_q, seqlen_k=seqlen_k)
    assert config.schedule == expected_schedule
    assert config.tdm_output is (
        expected_schedule in ("pingpong", "wan_cross", "wan_self")
    )
    if expected_schedule == "wan_cross":
        assert config.llvm_fn_attrs == "amdgpu-sched-strategy=iterative-ilp"
    elif seqlen_k <= 512:
        assert config.llvm_fn_attrs == "amdgpu-sched-strategy=max-ilp"
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=0.04,
        atol=0.04,
    )


@pytest.mark.skipif(not _is_gfx1250(), reason="requires AMD gfx1250")
def test_gluon_fav3_rejects_unsafe_options():
    from sglang.kernels.ops.attention.gluon_fav3_gfx1250 import (
        FAv3LaunchConfig,
        gluon_fav3_attention,
    )

    query = torch.randn(1, 1, 1, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="softmax_scale"):
            gluon_fav3_attention(query, key, value, softmax_scale=scale)

    invalid_config = FAv3LaunchConfig("pipeline", 64, 64, 4, False, "")
    with pytest.raises(ValueError, match="requires.*block_m"):
        gluon_fav3_attention(
            query,
            key,
            value,
            launch_config=invalid_config,
        )

    unsafe_fixed_config = FAv3LaunchConfig("wan_self", 256, 64, 8, True, "")
    with pytest.raises(ValueError, match="wan_self"):
        gluon_fav3_attention(
            query,
            key,
            value,
            launch_config=unsafe_fixed_config,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
