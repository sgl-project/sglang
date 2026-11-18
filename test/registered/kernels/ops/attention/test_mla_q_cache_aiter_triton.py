# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

pytest.importorskip("aiter")
from aiter import dtypes  # noqa: E402

from sglang.kernels.ops.attention import mla_q_cache_aiter_triton  # noqa: E402
from sglang.test.ci.ci_register import register_amd_ci  # noqa: E402
from sglang.test.kernels.mla_q_cache import mla_q_cache_inputs  # noqa: E402

# MI300-class runner: off gfx950 this kernel is the K3 writer's only fused path.
register_amd_ci(est_time=10, stage="base-b", runner_config="1-gpu-small-amd")

pytestmark = pytest.mark.skipif(
    not mla_q_cache_aiter_triton.available(),
    reason="requires AITER's no-RoPE Triton MLA Q/cache kernel (not on gfx1250)",
)

_ARGS = ("q_nope", "q_pe", "k_nope", "k_pe", "kv_cache", "slot_mapping", "k_scale")


def _args(values):
    return {name: values[name] for name in _ARGS}


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, dtypes.fp8])
def test_matches_split_chain_bytes(cache_dtype):
    """With a unit scale the split chain plainly casts cat(k_nope, k_pe),
    so the cache must hold the same bytes and Q must be the exact BF16 concat."""
    values = mla_q_cache_inputs(33, cache_dtype=cache_dtype)
    values["kv_cache"] = torch.zeros(40, 1, 576, device="cuda", dtype=cache_dtype)
    values["slot_mapping"] = torch.randperm(40, device="cuda")[:33]
    q = mla_q_cache_aiter_triton.run(**_args(values))
    torch.cuda.synchronize()
    expected_q = torch.cat((values["q_nope"], values["q_pe"]), dim=-1)
    expected_k = torch.cat((values["k_nope"], values["k_pe"]), dim=-1).to(cache_dtype)
    assert torch.equal(q, expected_q)
    cache = values["kv_cache"].view(torch.uint8)
    assert torch.equal(cache[values["slot_mapping"]], expected_k.view(torch.uint8))


@pytest.mark.parametrize("broken", ["fp8_e5m2", "paged_4d", "strided_slots"])
def test_rejects_inputs_outside_the_contract(broken):
    """AITER asserts on these caches and reads slots at stride 1; run() must
    return None without launching so the caller keeps the split chain."""
    values = mla_q_cache_inputs(4)
    if broken == "fp8_e5m2":
        values["kv_cache"] = values["kv_cache"].to(torch.float8_e5m2)
    elif broken == "paged_4d":
        values["kv_cache"] = values["kv_cache"].view(-1, 1, 1, 576)
    else:
        values["slot_mapping"] = torch.arange(8, device="cuda")[::2]
    assert mla_q_cache_aiter_triton.run(**_args(values)) is None
    torch.cuda.synchronize()
    assert torch.count_nonzero(values["kv_cache"].view(torch.uint8)).item() == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
