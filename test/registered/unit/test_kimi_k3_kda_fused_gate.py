"""Fail-closed gate for the whole-layer fused KDA backend
(``KimiK3DeltaAttention._forward_kda_fused_layer``).

The fused path produces wrong results if it engages on a layer whose shapes /
reduce topology the aiter kernels do not cover, so every miss must return
``None`` and fall back to the native per-kernel KDA path. These cases guard two
failure modes with no other coverage: the backend defaulting to inert, and any
single layer-config precondition being silently dropped.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention
from sglang.test.test_utils import CustomTestCase

# A layer config that satisfies the full static contract. Each test flips one
# field to a covered->uncovered value and asserts the gate falls back.
_GATE = KimiK3DeltaAttention._forward_kda_fused_layer


def _attn(**over):
    cfg = dict(bias=None, lower_bound=-5.0, layer_id=0)
    cfg.update(over)
    return SimpleNamespace(**cfg)


def _layer(**over):
    attn = over.pop("attn", _attn())
    o_norm = over.pop("o_norm", SimpleNamespace(eps=1e-5))
    cfg = dict(
        _kda_fused_layer_aiter=True,
        do_fuse_qkvbfg=True,
        use_full_rank_gate=True,
        attn_tp_size=8,
        _bfa_w=object(),
        all_reduce_fusion=False,
        split_sizes=(4608, 1536),
        _bfa_fa_size=128,
        _bfa_b_size=12,
    )
    cfg.update(over)
    return SimpleNamespace(attn=attn, o_norm=o_norm, **cfg)


# (name, layer) pairs, each uncovered by exactly one field. Dropping the
# matching guard in _forward_kda_fused_layer turns that case red.
_CONTRACT_MISSES = [
    ("do_fuse_qkvbfg off", _layer(do_fuse_qkvbfg=False)),
    ("not full-rank gate", _layer(use_full_rank_gate=False)),
    ("attn_tp_size != 8", _layer(attn_tp_size=4)),
    ("no [f_a|b] weight", _layer(_bfa_w=None)),
    ("all-reduce fusion on", _layer(all_reduce_fusion=True)),
    ("wrong split sizes", _layer(split_sizes=(4608, 1024))),
    ("wrong f_a size", _layer(_bfa_fa_size=64)),
    ("wrong beta size", _layer(_bfa_b_size=8)),
    ("conv bias present", _layer(attn=_attn(bias=object()))),
    ("lower_bound != -5", _layer(attn=_attn(lower_bound=-4.0))),
    ("norm eps != 1e-5", _layer(o_norm=SimpleNamespace(eps=1e-6))),
]


class TestKimiK3KdaFusedGate(CustomTestCase):
    def test_inert_when_backend_not_aiter(self):
        # Default (backend="native") / non-HIP: the fast path must never engage,
        # so stock numbers are reproduced bit-for-bit.
        layer = _layer(_kda_fused_layer_aiter=False)
        self.assertIsNone(_GATE(layer, hidden_states=None, forward_batch=None))

    def test_fail_closed_on_contract_miss(self):
        # Each uncovered layer config falls back to native before any state pool
        # is touched; a baseline miss would otherwise feed the kernel a shape it
        # cannot handle.
        for name, layer in _CONTRACT_MISSES:
            with self.subTest(miss=name):
                self.assertIsNone(_GATE(layer, hidden_states=None, forward_batch=None))


if __name__ == "__main__":
    unittest.main()
