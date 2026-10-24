"""Host-side gating and weight-update refusal for the fused wo_b all-reduce.

mori is stubbed here: what is under test is sglang's half -- which layers the
fused path may substitute for, which shapes are worth fusing, and that a layer
holding a re-laid-out copy refuses an online weight update.
``test_mori_gemm_ar_contract`` covers the mori side, and
``test_mori_gemm_ar_collective`` the cross-rank part.
"""

import sys
import types
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

import sglang.srt.layers.mori_gemm_ar as mg  # noqa: E402

N, K = 5120, 4096


@contextmanager
def mori_stub(supports=True):
    gemm_ar = types.ModuleType("mori.ops.gemm_ar")
    gemm_ar.supports = lambda *a, **k: supports
    mods = {
        "mori": types.ModuleType("mori"),
        "mori.ops": types.ModuleType("mori.ops"),
        "mori.ops.gemm_ar": gemm_ar,
    }
    with patch.dict(sys.modules, mods):
        yield


class _Group:
    def __init__(self, ranks):
        self.ranks = list(ranks)
        self.world_size = len(self.ranks)
        self.rank_in_group = 0


class _Layer:
    """The attributes ``_reduces_over_tp_group`` reads off a RowParallelLinear."""

    def __init__(
        self,
        ranks=range(8),
        reduce_results=True,
        use_decode_attn_tp=False,
        use_dp_attention_reduce=False,
    ):
        self.tp_group = _Group(ranks)
        self.reduce_results = reduce_results
        self.use_decode_attn_tp = use_decode_attn_tp
        self.use_dp_attention_reduce = use_dp_attention_reduce


@contextmanager
def tp_group(ranks=range(8)):
    group = _Group(ranks)
    with (
        get_parallel().override(tp_group=group, tp_size=group.world_size),
        patch(
            "sglang.srt.distributed.utils.get_group_rank_size",
            lambda g: (0, g.world_size),
        ),
    ):
        yield group


class TestReducesOverTpGroup(CustomTestCase):
    """Fusing performs the all-reduce, so it may only replace the same one.

    Each False below is a configuration where the split path reduces over some
    other group, or over nothing; fusing one of them sums rows belonging to
    unrelated ranks instead of raising.
    """

    def test_full_tp_group_reducing(self):
        with tp_group():
            self.assertTrue(mg._reduces_over_tp_group(_Layer()))

    def test_not_reducing(self):
        with tp_group():
            self.assertFalse(mg._reduces_over_tp_group(_Layer(reduce_results=False)))

    def test_tp_size_one(self):
        with tp_group(ranks=[0]):
            self.assertFalse(mg._reduces_over_tp_group(_Layer(ranks=[0])))

    def test_decode_attn_tp_reduces_over_another_group(self):
        with tp_group():
            self.assertFalse(mg._reduces_over_tp_group(_Layer(use_decode_attn_tp=True)))

    def test_dp_attention_reduce_reduces_over_another_group(self):
        with tp_group():
            self.assertFalse(
                mg._reduces_over_tp_group(_Layer(use_dp_attention_reduce=True))
            )

    def test_attn_tp_subgroup_is_not_the_tp_group(self):
        # --attn-dp-size 2 at TP8: wo_b reduces over 4 ranks, the op over 8.
        with tp_group(ranks=range(8)):
            self.assertFalse(mg._reduces_over_tp_group(_Layer(ranks=range(4))))

    def test_same_width_different_ranks(self):
        # Equal size would pass a size comparison; the ranks are what is compared.
        with tp_group(ranks=range(4)):
            self.assertFalse(mg._reduces_over_tp_group(_Layer(ranks=range(4, 8))))


class TestEligible(CustomTestCase):
    """Expressible is mori's answer; worth it is this module's.

    The thresholds are measured crossovers (see the module's constants), so a
    rewrite that reads as equivalent -- a fill ratio in place of the bf16 floor,
    one floor for both wires -- moves at least one of these.
    """

    def _eligible(self, m, m_pad, fp8_gather=False, supports=True):
        env = envs.SGLANG_OPT_FUSED_WO_B_AR_FP8_GATHER
        with mori_stub(supports=supports), env.override(fp8_gather):
            return mg._eligible(m, m_pad, N, K, 8)

    def test_inexpressible_shape_is_declined(self):
        self.assertFalse(self._eligible(16384, 16384, supports=False))

    def test_bf16_wire_floor(self):
        self.assertTrue(self._eligible(8192, 8192))
        self.assertFalse(self._eligible(4096, 4096))

    def test_fp8_wire_has_a_lower_floor(self):
        self.assertFalse(self._eligible(2048, 2048))
        self.assertTrue(self._eligible(2048, 2048, fp8_gather=True))
        # m_pad 1024 still loses: one row band per destination leaves the
        # scatter nothing to overlap with.
        self.assertFalse(self._eligible(1024, 1024, fp8_gather=True))

    def test_ragged_m_must_fill_its_padding(self):
        self.assertTrue(self._eligible(7000, 8192))  # 0.854
        self.assertFalse(self._eligible(6000, 8192))  # 0.732


class _WoB(torch.nn.Module):
    """A wo_b prepared by the gfx950 native mxfp8 route.

    Native rather than aiter so the layout conversion stays pure torch and this
    file needs no mori; the aiter route is covered in the contract test.
    """

    def __init__(self, n=N, k=K):
        super().__init__()
        w = torch.randint(0, 255, (n * k,), dtype=torch.uint8)
        self.weight = torch.nn.Parameter(
            w.view(torch.float8_e4m3fn), requires_grad=False
        )
        self.weight_scale_mx_e8m0 = torch.nn.Parameter(
            torch.randint(120, 135, (n // 32, k // 32), dtype=torch.uint8),
            requires_grad=False,
        )
        self.mxfp8_native_ready = True


def _update_error(model):
    from sglang.srt.model_executor.model_runner_components.weight_updater import (
        _unsupported_derived_weight_cache_error,
    )

    return _unsupported_derived_weight_cache_error(model)


class TestWeightUpdateRefusal(CustomTestCase):
    """A fused copy cannot survive an in-place reload, so the layer refuses one.

    Before this, a weight update left the fused path multiplying by the weight
    the server started with while decode and every declined call used the new
    one -- no error, just wrong output.
    """

    def setUp(self):
        mg._prepared.clear()
        self.layer = _WoB()
        self.model = torch.nn.Module()
        self.model.wo_b = self.layer

    def tearDown(self):
        mg._prepared.clear()

    def test_no_copy_no_constraint(self):
        self.assertIsNone(_update_error(self.model))

    def test_building_a_copy_publishes_the_constraint(self):
        mg.mori_weight(self.layer)
        self.assertEqual(_update_error(self.model), mg._UPDATE_UNSUPPORTED)

    def test_release_lifts_it(self):
        mg.mori_weight(self.layer)
        mg._prepared.append(self.layer)
        mg.release_all_prepared()
        self.assertIsNone(_update_error(self.model))
        self.assertEqual(mg._prepared, [])

    def test_in_place_load_moves_nothing_observable(self):
        """Why refusal is the mechanism and a cache key on the weight is not.

        The loaders write through ``param.data.copy_()``. A test that writes to
        the parameter directly bumps ``_version`` and a version-keyed cache then
        passes, while the real loader leaves it alone -- so the write here is
        the loader's. A torch that does start bumping something turns this red,
        which is when the constraint could be lifted.
        """
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        w = self.layer.weight
        before = (
            w._version,
            w.data._version,
            w.data_ptr(),
            w.untyped_storage().data_ptr(),
        )
        fresh = torch.randint(0, 255, tuple(w.shape), dtype=torch.uint8).view(
            torch.float8_e4m3fn
        )
        default_weight_loader(w, fresh)

        self.assertTrue(torch.equal(w.view(torch.uint8), fresh.view(torch.uint8)))
        self.assertEqual(
            before,
            (w._version, w.data._version, w.data_ptr(), w.untyped_storage().data_ptr()),
        )


if __name__ == "__main__":
    unittest.main()
