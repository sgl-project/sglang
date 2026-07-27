"""ReplaySSM ring per-req byte accounting (BaseLinearStateParams.replayssm_ring_bytes_per_req).

The memory solver charges this on top of mamba_cache_per_req so num_slots is not
over-provisioned (the ring is allocated per slot but is NOT part of the state cache
cost). This pins the arithmetic against hand-computed byte counts, across both gate
layouts (KDA per-K vector g vs GDN scalar g) and both rings (spec-verify adds the
raw v / pre-norm k / beta rings; decode does not). If the MambaPool allocation
changes shape, update both the allocation and this expectation together.
"""

import pytest
import torch

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# temporal = (hv=4, v_dim=8, k_dim=8), num_k_heads_per_tp = 4, L = 8, 2 layers.
# conv bf16 (2B), ssm/temporal fp32 (4B). Ring tensors (per slot, per layer):
#   d    hv*L*v_dim,  k    h_k*L*k_dim   -> ring dtype (conv under spec, ssm under decode)
#   g    hv*L*k_dim (KDA) / hv*L (GDN)   -> fp32
#   rawv hv*L*v_dim,  rawk h_k*L*k_dim   -> conv dtype (spec only)
#   beta hv*L                           -> fp32 (spec only)
DTYPE = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)
L = 8
LAYERS = [0, 1]


def _kda_params():
    shape = KimiLinearStateShape.create(
        tp_world_size=1, num_heads=4, head_dim=8, num_k_heads=4, head_k_dim=8
    )
    return KimiLinearCacheParams(shape=shape, dtype=DTYPE, layers=LAYERS)


def _gdn_params():
    # Only shape.temporal and shape.num_k_heads_per_tp are read here; the rest are
    # dummy (the accounting does not depend on them).
    shape = Mamba2StateShape(
        conv=[(4, 3)],
        temporal=(4, 8, 8),
        intermediate_size=0,
        conv_dim=0,
        ssm_state_size=0,
        num_heads=0,
        head_dim=0,
        state_size=0,
        conv_kernel=0,
        num_k_heads_per_tp=4,
    )
    return Mamba2CacheParams(shape=shape, dtype=DTYPE, layers=LAYERS)


class TestReplaySSMRingAccounting(CustomTestCase):
    def test_kda_spec(self):
        # per layer: d 512 + k 512 + g(4*8*8*4) 1024 + rawv 512 + rawk 512 + beta 128 = 3200
        self.assertEqual(
            _kda_params().replayssm_ring_bytes_per_req(cache_len=L, enable_spec=True),
            3200 * len(LAYERS),
        )

    def test_kda_decode(self):
        # decode ring uses the ssm dtype (fp32): d 1024 + k 1024 + g 1024 = 3072; no raw rings
        self.assertEqual(
            _kda_params().replayssm_ring_bytes_per_req(cache_len=L, enable_spec=False),
            3072 * len(LAYERS),
        )

    def test_gdn_spec_scalar_gate(self):
        # GDN g is a per-head SCALAR (hv*L*1*4 = 128), not per-K:
        # d 512 + k 512 + g 128 + rawv 512 + rawk 512 + beta 128 = 2304
        self.assertEqual(
            _gdn_params().replayssm_ring_bytes_per_req(cache_len=L, enable_spec=True),
            2304 * len(LAYERS),
        )

    def test_zero_len_ring(self):
        self.assertEqual(
            _kda_params().replayssm_ring_bytes_per_req(cache_len=0, enable_spec=True), 0
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
