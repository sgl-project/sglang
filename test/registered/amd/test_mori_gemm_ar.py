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
from unittest.mock import mock_open, patch

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


class TestKernelConfig(CustomTestCase):
    """The running kernel's options decide, including a rebuilt 5.15 kernel."""

    def test_move_notify_enabled_on_an_old_kernel(self):
        config = mock_open(read_data="CONFIG_DMABUF_MOVE_NOTIFY=y\n")
        with (
            patch("platform.release", return_value="5.15.0-custom"),
            patch("builtins.open", config),
            patch("gzip.open") as compressed,
        ):
            self.assertIsNone(mg._kernel_config_error())
        self.assertEqual(config.call_args.args[0], "/boot/config-5.15.0-custom")
        compressed.assert_not_called()

    def test_disabled_move_notify_reports_the_fix(self):
        config = mock_open(read_data="# CONFIG_DMABUF_MOVE_NOTIFY is not set\n")
        with patch("builtins.open", config):
            error = mg._kernel_config_error()
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("CONFIG_DMABUF_MOVE_NOTIFY", str(error))
        self.assertIn("linux-generic-hwe-22.04", str(error))

    def test_proc_config_is_used_when_boot_is_unreadable(self):
        compressed = mock_open(read_data="CONFIG_DMABUF_MOVE_NOTIFY=y\n")
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch("gzip.open", compressed),
        ):
            self.assertIsNone(mg._kernel_config_error())
        self.assertEqual(compressed.call_args.args[0], "/proc/config.gz")

    def test_disabled_proc_config_is_rejected(self):
        with (
            patch("builtins.open", side_effect=PermissionError),
            patch("gzip.open", mock_open(read_data="CONFIG_OTHER=y\n")),
        ):
            self.assertIsInstance(mg._kernel_config_error(), RuntimeError)

    def test_unreadable_config_leaves_the_placement_check_to_decide(self):
        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch("gzip.open", side_effect=FileNotFoundError),
        ):
            self.assertIsNone(mg._kernel_config_error())


class TestVramCounter(CustomTestCase):
    def test_current_logical_device_is_resolved_to_its_pci_address(self):
        # Device 2 need not be PCI device 2, especially with HIP_VISIBLE_DEVICES.
        props = types.SimpleNamespace(pci_domain_id=1, pci_bus_id=0xAF, pci_device_id=3)
        counter = mock_open(read_data="123456\n")
        with (
            patch.object(torch.cuda, "current_device", return_value=2),
            patch.object(
                torch.cuda, "get_device_properties", return_value=props
            ) as get,
            patch("builtins.open", counter),
        ):
            self.assertEqual(mg._vram_used_bytes(), 123456)
        get.assert_called_once_with(2)
        self.assertEqual(
            counter.call_args.args[0],
            "/sys/bus/pci/devices/0001:af:03.0/mem_info_vram_used",
        )

    def test_missing_pci_properties_make_the_probe_unavailable(self):
        with (
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=types.SimpleNamespace(),
            ),
        ):
            self.assertIsNone(mg._vram_used_bytes())

    def test_unreadable_or_invalid_counter_is_unknown(self):
        props = types.SimpleNamespace(pci_domain_id=0, pci_bus_id=1, pci_device_id=0)
        for contents in ("not a counter\n", "-1\n", None):
            counter = mock_open(read_data=contents or "")
            if contents is None:
                counter.side_effect = FileNotFoundError
            with (
                self.subTest(contents=contents),
                patch.object(torch.cuda, "current_device", return_value=0),
                patch.object(torch.cuda, "get_device_properties", return_value=props),
                patch("builtins.open", counter),
            ):
                self.assertIsNone(mg._vram_used_bytes())


class TestSettledVramCounter(CustomTestCase):
    @contextmanager
    def _clock(self):
        """Advance probe time without delaying the test."""
        elapsed = [0.0]

        def sleep(seconds):
            elapsed[0] = round(elapsed[0] + seconds, 8)

        with (
            patch("time.monotonic", side_effect=lambda: elapsed[0]),
            patch("time.sleep", side_effect=sleep),
        ):
            yield elapsed

    def test_deferred_frees_finish_before_the_baseline_is_accepted(self):
        # amdgpu can account for old allocations in batches, after torch's
        # allocator already reports them released. Two equal reads, or even
        # 250 ms of stability, would accept one of these stale baselines.
        def vram_used():
            if elapsed[0] < 0.25:
                return 20 << 30
            if elapsed[0] < 0.50:
                return 12 << 30
            return 4 << 30

        with (
            self._clock() as elapsed,
            patch.object(mg, "_vram_used_bytes", side_effect=vram_used),
        ):
            self.assertEqual(mg._settled_vram_used_bytes(), 4 << 30)
        self.assertGreaterEqual(elapsed[0], 1.25)

    def test_two_equal_readings_do_not_hide_a_later_large_free(self):
        values = iter([20 << 30, 20 << 30, 2 << 30])
        with (
            self._clock() as elapsed,
            patch.object(
                mg, "_vram_used_bytes", side_effect=lambda: next(values, 2 << 30)
            ),
        ):
            self.assertEqual(mg._settled_vram_used_bytes(), 2 << 30)
        self.assertGreaterEqual(elapsed[0], 0.85)

    def test_a_continuously_changing_counter_times_out_as_unknown(self):
        with (
            self._clock() as elapsed,
            patch.object(
                mg, "_vram_used_bytes", side_effect=lambda: int(elapsed[0] * 1_000_000)
            ),
        ):
            self.assertIsNone(mg._settled_vram_used_bytes())
        self.assertGreaterEqual(elapsed[0], 3.0)
        self.assertLessEqual(elapsed[0], 3.05)

    def test_unreadable_counter_is_unknown_even_after_a_valid_reading(self):
        for values in ([None], [2 << 30, None]):
            with (
                self.subTest(values=values),
                self._clock() as elapsed,
                patch.object(mg, "_vram_used_bytes", side_effect=values),
            ):
                self.assertIsNone(mg._settled_vram_used_bytes())
            self.assertLessEqual(elapsed[0], 0.05)


class TestWindowPlacement(CustomTestCase):
    WINDOW = 64 << 20

    def _state(self, before):
        state = mg._FusedWoB.__new__(mg._FusedWoB)
        state.rank = 0
        state._vram_before = before
        state.op = types.SimpleNamespace(window_bytes=self.WINDOW)
        return state

    def test_half_a_window_or_more_in_vram_is_accepted(self):
        before = 4 << 30
        for growth in (self.WINDOW // 2, self.WINDOW):
            with (
                self.subTest(growth=growth),
                patch.object(torch.cuda, "synchronize"),
                patch.object(
                    mg, "_settled_vram_used_bytes", return_value=before + growth
                ),
            ):
                self._state(before).check_placement()

    def test_insufficient_vram_growth_reports_without_closing_the_op(self):
        before = 4 << 30
        for growth in (self.WINDOW // 2 - 1, 0, -self.WINDOW):
            state = self._state(before)
            with (
                self.subTest(growth=growth),
                patch.object(torch.cuda, "synchronize"),
                patch.object(
                    mg, "_settled_vram_used_bytes", return_value=before + growth
                ),
                self.assertRaises(RuntimeError) as raised,
            ):
                state.check_placement()
            self.assertIn("VRAM", str(raised.exception))
            self.assertIn("64", str(raised.exception))
            # Cleanup belongs to the collective caller, after every rank agrees.
            self.assertIsNotNone(state.op)

    def test_unknown_counter_warns_and_keeps_the_opt_in_path(self):
        for before, after in ((None, 100), (100, None), (None, None)):
            with (
                self.subTest(before=before, after=after),
                patch.object(torch.cuda, "synchronize"),
                patch.object(mg, "_settled_vram_used_bytes", return_value=after),
                self.assertLogs(mg.logger, level="WARNING"),
            ):
                self._state(before).check_placement()


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
