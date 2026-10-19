"""Layout contract between the Qwen4-Exp FlyDSL monokernel and sglang's AITER weight layouts.

The monokernel reads sglang's MXFP4 FusedMoE params and FP8 projection weights in place, so it hard-codes
AITER's ``shuffle_weight(w, (16, 16))`` / ``e8m0_shuffle`` storage order and FusedMoE's padded intermediate
dim. These tests pin that contract on the host: the kernel package's un-shuffle helpers must invert AITER's
shuffles exactly, and its layout constants must match sglang's AITER MoE weight sizes.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytest.importorskip("flydsl")
shuffle = pytest.importorskip("aiter.ops.shuffle")
fp4_utils = pytest.importorskip("aiter.utility.fp4_utils")

from sglang.kernels.ops.speculative.qwen4_exp_monokernel import (  # noqa: E402
    layout as LY,
)
from sglang.kernels.ops.speculative.qwen4_exp_monokernel import pack  # noqa: E402
from sglang.kernels.ops.speculative.qwen4_exp_monokernel.weights import (  # noqa: E402
    HIDDEN,
    MOE_INTER,
    N_EXPERTS,
)
from sglang.srt.layers.moe.utils import get_moe_weight_sizes  # noqa: E402


def _bytes(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)


def _e8m0_shuffle_3d(s):
    """Scale shuffle as quark_w4a4_mxfp4_moe.process_weights_after_loading does it (flattened [E * rows, groups])."""
    e, r, _ = s.shape
    return fp4_utils.e8m0_shuffle(s.reshape(e * r, -1).contiguous()).view(e, r, -1)


@pytest.mark.parametrize(
    "rows,kbytes", [(2560, 6144), (16384, 2560), (128, 2560), (640, 2560)]
)
def test_unshuffle16_inverts_aiter_shuffle(rows, kbytes):
    """FP8 projection weights: AITER (16, 16) shuffle of a 1-byte [N, K] matrix and the kernel's inverse."""
    w = _bytes(rows, kbytes, seed=rows)
    shuffled = shuffle.shuffle_weight(w.clone(), (16, 16))
    assert not torch.equal(shuffled.view(torch.uint8), w)
    assert torch.equal(pack.unshuffle16(shuffled), w)


def test_moe_padding_matches_fusedmoe():
    """FusedMoE pads the MXFP4 intermediate dim for AITER; the kernel's constants assume the same padding."""
    w13_dim, w2_dim = get_moe_weight_sizes(
        MOE_INTER, is_concat=True, is_packed=True, is_aiter_moe=True
    )[:2]
    assert w13_dim == 2 * LY.MOE_INTER_PAD
    assert w2_dim == LY.MOE_INTER_PAD // 2
    assert LY.UP_TILE0 * 16 == LY.MOE_INTER_PAD
    assert LY.GU_BYTES == 2 * LY.MOE_INTER_PAD * HIDDEN // 2
    assert LY.DN_BYTES == HIDDEN * LY.MOE_INTER_PAD // 2


def test_unshuffle_moe_expert_inverts_fusedmoe_layout():
    """MXFP4 experts: checkpoint (gate, up, down + E8M0 scales) -> FusedMoE padded, AITER-shuffled params -> the
    kernel's per-expert inverse recovers the checkpoint bytes."""
    E, I, IP = 2, MOE_INTER, LY.MOE_INTER_PAD
    gate, up = _bytes(E, I, HIDDEN // 2, seed=1), _bytes(E, I, HIDDEN // 2, seed=2)
    down = _bytes(E, HIDDEN, I // 2, seed=3)
    gate_s, up_s = (
        _bytes(E, I, HIDDEN // 32, seed=4),
        _bytes(E, I, HIDDEN // 32, seed=5),
    )
    down_s = _bytes(E, HIDDEN, I // 32, seed=6)

    # FusedMoE padded storage: w13 = [gate; 0; up; 0] rows, w2 / its scales zero-padded along K
    w13 = torch.zeros(E, 2 * IP, HIDDEN // 2, dtype=torch.uint8)
    w13[:, :I], w13[:, IP : IP + I] = gate, up
    s13 = torch.zeros(E, 2 * IP, LY.GU_SGROUPS, dtype=torch.uint8)
    s13[:, :I], s13[:, IP : IP + I] = gate_s, up_s
    w2 = torch.zeros(E, HIDDEN, IP // 2, dtype=torch.uint8)
    w2[..., : I // 2] = down
    s2 = torch.zeros(E, HIDDEN, LY.DN_SGROUPS, dtype=torch.uint8)
    s2[..., : I // 32] = down_s

    m = {
        "gu": shuffle.shuffle_weight(w13.contiguous(), (16, 16)),
        "gu_s": _e8m0_shuffle_3d(s13),
        "dn": shuffle.shuffle_weight(w2.contiguous(), (16, 16)),
        "dn_s": _e8m0_shuffle_3d(s2),
    }
    for e in range(E):
        got = pack.unshuffle_moe_expert(m, e)
        ref = (gate[e], up[e], down[e], gate_s[e], up_s[e], down_s[e])
        for name, g, r in zip(
            ("gate", "up", "down", "gate_s", "up_s", "down_s"), got, ref
        ):
            assert torch.equal(g, r), f"expert {e} {name}"


def test_check_moe_params_accepts_fusedmoe_shapes():
    """The kernel's MoE asserts accept exactly the FusedMoE param shapes (routed experts + fused shared expert)."""
    E = N_EXPERTS + 1
    m = {
        "gu": torch.empty(E, 2 * LY.MOE_INTER_PAD, HIDDEN // 2, dtype=torch.uint8),
        "gu_s": torch.empty(E, 2 * LY.MOE_INTER_PAD, LY.GU_SGROUPS, dtype=torch.uint8),
        "dn": torch.empty(E, HIDDEN, LY.MOE_INTER_PAD // 2, dtype=torch.uint8),
        "dn_s": torch.empty(E, HIDDEN, LY.DN_SGROUPS, dtype=torch.uint8),
    }
    pack.check_moe_params(m)
    m["gu"] = torch.empty(
        E - 1, 2 * LY.MOE_INTER_PAD, HIDDEN // 2, dtype=torch.uint8
    )  # no fused shared expert
    with pytest.raises(AssertionError):
        pack.check_moe_params(m)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
