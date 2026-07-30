"""ReplaySSM ring per-slot byte accounting (BaseLinearStateParams.replayssm_ring_bytes_per_req).

The memory solver charges this on top of mamba_cache_per_req so num_slots is not
over-provisioned (the ring is allocated per slot but is NOT part of the state
cache cost). This pins the arithmetic against hand-computed byte counts for the
GDN scalar-g gate layout, across both protocols (fold-every-commit allocates
only the raw v / pre-norm k / g / beta window; the circular spec ring
additionally allocates the (d, k) reconstruction records). The accounting is
GDN-only: KDA params must be rejected, not silently under-counted (the KDA
per-K gate ring is bigger). If the MambaPool allocation changes shape, update
both the allocation and this expectation together.
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

# temporal = (hv=4, v_dim=8, k_dim=8), num_k_heads_per_tp = 4, record_len = 8,
# 2 layers. conv bf16 (2B), fp32 gate/beta (4B). Ring tensors (per slot, per
# layer):
#   rawv hv*RL*v_dim, rawk h_k*RL*k_dim -> conv dtype
#   g    hv*RL                          -> fp32
#   beta hv*RL                          -> fp32
#   d    hv*RL*v_dim, k    h_k*RL*k_dim -> conv dtype (circular ring only)
DTYPE = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)
RL = 8
LAYERS = [0, 1]


def _kda_params():
    shape = KimiLinearStateShape.create(
        tp_world_size=1, num_heads=4, head_dim=8, num_k_heads=4, head_k_dim=8
    )
    return KimiLinearCacheParams(shape=shape, dtype=DTYPE, layers=LAYERS)


def _gdn_params():
    # Only shape.temporal and shape.num_k_heads_per_tp are read here; the rest
    # are dummy (the accounting does not depend on them).
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
    def test_gdn_fold(self):
        # fold window: rawv 512 + rawk 512 + g(scalar, 4*8*4) 128 + beta 128 = 1280
        self.assertEqual(
            _gdn_params().replayssm_ring_bytes_per_req(record_len=RL, spec_fold=True),
            1280 * len(LAYERS),
        )

    def test_gdn_circular_ring(self):
        # circular ring adds the (d, k) records: 1280 + d 512 + k 512 = 2304
        self.assertEqual(
            _gdn_params().replayssm_ring_bytes_per_req(record_len=RL, spec_fold=False),
            2304 * len(LAYERS),
        )

    def test_kda_rejected(self):
        # GDN-only contract: a KDA caller must fail loud, not get a scalar-g
        # byte count that under-charges its per-K gate ring.
        with self.assertRaises(AssertionError):
            _kda_params().replayssm_ring_bytes_per_req(record_len=RL, spec_fold=True)

    def test_zero_len_ring(self):
        self.assertEqual(
            _gdn_params().replayssm_ring_bytes_per_req(record_len=0, spec_fold=True), 0
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
