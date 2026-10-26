"""ReplaySSM ring per-slot byte accounting (BaseLinearStateParams.replayssm_ring_bytes_per_req).

The memory solver charges this on top of mamba_cache_per_req so num_slots is not
over-provisioned (the ring is allocated per slot but is NOT part of the state
cache cost). Pins the arithmetic against hand-computed byte counts for the
fold window (raw v / pre-norm k / g / beta) across both gate layouts: GDN
per-head scalar g vs KDA per-K vector g (KDA also keeps the chunked d/k rings
under spec, see MambaPool). If the MambaPool allocation changes shape, update
both the allocation and this expectation together.
"""

import os
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.mamba_utils import (
    KimiLinearCacheParams,
    KimiLinearStateShape,
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
    mamba2_state_dtype,
)
from sglang.srt.mem_cache.kv_cache_configurator import (  # noqa: F401
    _pp_local_per_request_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

# temporal = (hv=4, v_dim=8, k_dim=8), num_k_heads_per_tp = 4, record_len = 8,
# 2 layers. GDN records follow conv dtype; KDA records follow SSM dtype.
# Gate/beta remain fp32 (4B). Ring tensors (per slot, per layer):
# GDN stores compact d/k plus scalar g. KDA stores raw v/k, vector g, beta,
# and its existing d/k rings.
DTYPE = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)
RL = 8
LAYERS = [0, 1]


def _kda_params(dtype=DTYPE):
    shape = KimiLinearStateShape.create(
        tp_world_size=1, num_heads=4, head_dim=8, num_k_heads=4, head_k_dim=8
    )
    return KimiLinearCacheParams(shape=shape, dtype=dtype, layers=LAYERS)


def _gdn_params(temporal_dtype=torch.float32):
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
    dtype = Mamba2StateDType(conv=torch.bfloat16, temporal=temporal_dtype)
    return Mamba2CacheParams(shape=shape, dtype=dtype, layers=LAYERS)


class TestReplaySSMRingAccounting(CustomTestCase):
    def test_gdn_fold(self):
        # d 512 + normalized k 512 + scalar g 128 + d/k low parts 1024 = 2176
        self.assertEqual(
            _gdn_params().replayssm_ring_bytes_per_req(record_len=RL),
            2176 * len(LAYERS),
        )
        self.assertEqual(
            _gdn_params(torch.bfloat16).replayssm_ring_bytes_per_req(record_len=RL),
            2176 * len(LAYERS),
        )

    def test_kda_fold(self):
        # Four rawv/rawk/d/k rings: 4 * 4 * 8 * 8 elements per layer.
        # Vector g and beta add 1024 + 128 bytes regardless of state dtype.
        for ssm, expected in [(torch.float32, 5248), (torch.bfloat16, 3200)]:
            for conv in (torch.float32, torch.bfloat16):
                with self.subTest(ssm=ssm, conv=conv):
                    dtype = Mamba2StateDType(conv=conv, temporal=ssm)
                    self.assertEqual(
                        _kda_params(dtype).replayssm_ring_bytes_per_req(record_len=RL),
                        expected * len(LAYERS),
                    )

    def test_zero_len_ring(self):
        self.assertEqual(_gdn_params().replayssm_ring_bytes_per_req(record_len=0), 0)
        self.assertEqual(_kda_params().replayssm_ring_bytes_per_req(record_len=0), 0)

    def test_state_precision_and_override(self):
        dtypes = [
            ("float32", torch.float32),
            ("bfloat16", torch.bfloat16),
            ("float16", torch.float16),
        ]
        for name, dtype in dtypes:
            with self.subTest(dtype=name), patch.dict(os.environ, clear=False):
                os.environ["SGLANG_MAMBA_SSM_DTYPE"] = name
                os.environ.pop("SGLANG_MAMBA_CONV_DTYPE", None)
                expected = Mamba2StateDType(conv=torch.bfloat16, temporal=dtype)
                params = KimiLinearCacheParams(shape=None, layers=[])
                self.assertEqual(params.dtype, expected)
                self.assertEqual(mamba2_state_dtype(), expected)
                for conv_name, conv_dtype in dtypes:
                    with self.subTest(conv=conv_name):
                        os.environ["SGLANG_MAMBA_CONV_DTYPE"] = conv_name
                        self.assertEqual(
                            KimiLinearCacheParams(shape=None, layers=[]).dtype,
                            Mamba2StateDType(conv=conv_dtype, temporal=dtype),
                        )

    def test_explicit_dtype_is_preserved(self):
        dtype = Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32)
        params = KimiLinearCacheParams(shape=None, layers=[], dtype=dtype)
        self.assertEqual(params.dtype, dtype)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
