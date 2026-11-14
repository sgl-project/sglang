"""HiCache: a Mamba slot's registered side states must survive the host round trip.

`MambaPool.register_slot_state` attaches state that lives in the same physical
checkpoint slot as conv/temporal but is neither per-layer nor part of
`mamba_cache` (Qwen4-Exp registers a PLE short-conv window and a PLE N-gram
token history this way). Every other mover of a slot carries it -- `copy_from`,
`clear_slots`, `get_cpu_copy`/`load_cpu_copy`, and the RDMA registration in
`_iter_transfer_state_entries`. Before this test the HiCache host tier did not,
so a slot restored from host kept whatever the previous occupant left in those
rows and the model conditioned on another request's tokens.

The copy kernels are CUDA-only, so the two conv/temporal helpers are stubbed and
the side-state legs are exercised at their real call sites on CPU tensors.
"""

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
from sglang.srt.mem_cache.pool_host import slot_state_copy
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

NUM_LAYERS = 2
NUM_DEVICE_SLOTS = 4
NUM_HOST_SLOTS = 4
CONV_SHAPE = (3, 2)
TEMPORAL_SHAPE = (2, 2)
SIDE_CONV_SHAPE = (5, 3)
NGRAM_WIDTH = 2


class _RegisteredSideState:
    """Stands in for ple_state_pool.ShortConvPool / NGramPool: the only thing the
    host pool may assume is the transfer-entry protocol, whose tensors carry the
    slot on axis 0."""

    def __init__(self, name, tensor):
        self.name = name
        self.tensor = tensor

    def iter_transfer_state_entries(self):
        yield self.name, self.tensor, None, 0


def _make_pools(with_side_state: bool):
    conv = [
        torch.zeros((NUM_LAYERS, NUM_DEVICE_SLOTS) + CONV_SHAPE, dtype=torch.bfloat16)
    ]
    temporal = torch.zeros(
        (NUM_LAYERS, NUM_DEVICE_SLOTS) + TEMPORAL_SHAPE, dtype=torch.bfloat16
    )
    siblings = ()
    if with_side_state:
        siblings = (
            _RegisteredSideState(
                "ple_short_conv",
                torch.zeros(
                    (NUM_DEVICE_SLOTS,) + SIDE_CONV_SHAPE, dtype=torch.bfloat16
                ),
            ),
            _RegisteredSideState(
                "ple_ngram",
                torch.zeros((NUM_DEVICE_SLOTS, NGRAM_WIDTH), dtype=torch.int64),
            ),
        )
    device_pool = SimpleNamespace(
        mamba_cache=SimpleNamespace(conv=conv, temporal=temporal),
        num_mamba_layers=NUM_LAYERS,
        size=NUM_DEVICE_SLOTS,
        device="cpu",
        _slot_siblings=siblings,
    )

    host = MambaPoolHost.__new__(MambaPoolHost)
    host.device_pool = device_pool
    host.buffer_info = MambaStateBufferInfo(
        temporal=device_pool.mamba_cache.temporal,
        conv=tuple(device_pool.mamba_cache.conv),
    )
    host.page_size = 1
    host.layout = "page_first"
    host.device = "cpu"
    host.pin_memory = False
    host.num_mamba_layers = NUM_LAYERS
    host.size = NUM_HOST_SLOTS
    host.conv_state_shapes = [c.shape[2:] for c in conv]
    host.temporal_state_shape = temporal.shape[2:]
    host.temporal_state_elem_size = int(temporal[0, 0].numel())
    host.conv_state_elem_sizes = [int(c[0, 0].numel()) for c in conv]
    host.conv_dtype = conv[0].dtype
    host.temporal_dtype = temporal.dtype
    host.dtype = host.conv_dtype
    host.slot_state_device_tensors = [
        state
        for sibling in device_pool._slot_siblings
        for _, state, _, _ in sibling.iter_transfer_state_entries()
    ]
    host.slot_state_buffers = [
        torch.zeros((NUM_HOST_SLOTS,) + tuple(state.shape[1:]), dtype=state.dtype)
        for state in host.slot_state_device_tensors
    ]
    host.temporal_buffer = torch.zeros(
        (NUM_HOST_SLOTS, NUM_LAYERS, 1) + TEMPORAL_SHAPE, dtype=temporal.dtype
    )
    host.conv_buffer = [
        torch.zeros((NUM_HOST_SLOTS, NUM_LAYERS, 1) + CONV_SHAPE, dtype=conv[0].dtype)
    ]
    host._init_write_back_staging_buffers()
    # The kernel path takes raw device pointers; the stubbed copies never
    # dereference them, but the real methods read the attributes.
    host.temporal_device_ptrs = torch.zeros(NUM_LAYERS, dtype=torch.uint64)
    host.conv_device_ptrs = [torch.zeros(NUM_LAYERS, dtype=torch.uint64) for _ in conv]
    host.size_per_token = host.get_size_per_token()
    return host, device_pool


def _stub_conv_and_temporal_copies():
    """The conv/temporal legs are CUDA kernels; this test is about the side state."""
    return (
        mock.patch.object(
            MambaPoolHost, "_copy_tensor_pf_lf", staticmethod(lambda **kw: None)
        ),
        mock.patch.object(
            MambaPoolHost,
            "_copy_tensor_all_layers_lf_pf",
            staticmethod(lambda **kw: None),
        ),
    )


class TestHiCacheMambaSlotSideStates(CustomTestCase):
    def test_size_per_token_accounts_for_side_state(self):
        with_side, _ = _make_pools(True)
        without_side, _ = _make_pools(False)
        side_bytes = (
            SIDE_CONV_SHAPE[0] * SIDE_CONV_SHAPE[1] * 2  # bfloat16
            + NGRAM_WIDTH * 8  # int64
        )
        self.assertEqual(
            with_side.get_size_per_token(),
            without_side.get_size_per_token() + side_bytes,
        )

    def test_mapping_lengths_carry_each_side_state_buffer(self):
        # Each side state is its own mmap (init_kv_buffer), so the budget books
        # it the way it is mapped: rounding up to a whole hugepage on its own,
        # like the temporal and conv buffers.
        token_layer_slots = NUM_HOST_SLOTS * NUM_LAYERS
        with_side, _ = _make_pools(True)
        self.assertEqual(
            with_side.get_mapping_lengths(),
            [
                token_layer_slots * math.prod(TEMPORAL_SHAPE) * 2,
                token_layer_slots * math.prod(CONV_SHAPE) * 2,
                NUM_HOST_SLOTS * math.prod(SIDE_CONV_SHAPE) * 2,
                NUM_HOST_SLOTS * NGRAM_WIDTH * 8,
            ],
        )
        without_side, _ = _make_pools(False)
        self.assertEqual(len(without_side.get_mapping_lengths()), 2)

    def test_side_state_survives_the_host_round_trip(self):
        host, device_pool = _make_pools(True)
        short_conv, ngram = (s.tensor for s in device_pool._slot_siblings)

        checkpoint_slot = torch.tensor([1])
        host_slot = torch.tensor([2])
        other_slot = torch.tensor([3])

        short_conv[checkpoint_slot] = 1.5
        ngram[checkpoint_slot] = torch.tensor([[11, 321]])
        # Another request owned the destination slot before the restore.
        short_conv[other_slot] = -7.0
        ngram[other_slot] = torch.tensor([[310, 279]])

        pf_lf, lf_pf = _stub_conv_and_temporal_copies()
        with pf_lf, lf_pf:
            host.backup_from_device_all_layer(
                device_pool, host_slot, checkpoint_slot, "direct"
            )
            host.load_to_device_per_layer(
                device_pool, host_slot, other_slot, 0, "direct"
            )

        self.assertTrue(
            torch.equal(short_conv[other_slot], short_conv[checkpoint_slot]),
            "the restored slot kept the previous occupant's PLE conv window",
        )
        self.assertEqual(
            ngram[other_slot].tolist(),
            [[11, 321]],
            "the restored slot kept the previous occupant's N-gram history",
        )

    def test_data_page_carries_side_state(self):
        """The L3 page is the same payload: what RAM carries, disk must carry."""
        host, device_pool = _make_pools(True)
        short_conv_host, ngram_host = host.slot_state_buffers
        short_conv_host[1] = 2.25
        ngram_host[1] = torch.tensor([7, 9])

        page = host.get_data_page(1)
        self.assertEqual(page.numel(), host.page_size * host.size_per_token)

        short_conv_host[1] = 0
        ngram_host[1] = 0
        host.set_from_flat_data_page(1, page)
        self.assertTrue(torch.all(short_conv_host[1] == 2.25))
        self.assertEqual(ngram_host[1].tolist(), [7, 9])

    def test_short_page_is_refused_rather_than_half_applied(self):
        host, _ = _make_pools(True)
        short = torch.zeros(host.size_per_token - 1, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            host.set_from_flat_data_page(0, short)

    def test_model_without_side_state_is_unaffected(self):
        host, device_pool = _make_pools(False)
        self.assertEqual(host.slot_state_device_tensors, [])
        self.assertEqual(host.slot_state_buffers, [])
        page = host.get_data_page(0)
        self.assertEqual(page.numel(), host.page_size * host.size_per_token)
        pf_lf, lf_pf = _stub_conv_and_temporal_copies()
        with pf_lf, lf_pf:
            host.backup_from_device_all_layer(
                device_pool, torch.tensor([0]), torch.tensor([0]), "direct"
            )
            host.load_to_device_per_layer(
                device_pool, torch.tensor([0]), torch.tensor([1]), 0, "direct"
            )


class TestSlotStateAsyncCopy(CustomTestCase):
    """SGLANG_OPT_HICACHE_MAMBA_SLOT_STATE_ASYNC: the side-state legs as stream-ordered
    copies through the pinned host buffers, the load leg only under host-only
    checkpoints; anything the kernel cannot serve keeps the blocking copy."""

    def setUp(self):
        self._saved = dict(slot_state_copy._state)
        slot_state_copy._state["off_reason"] = None
        self.addCleanup(slot_state_copy._state.update, self._saved)

    def _switches(self, *, async_on=True, host_only=False):
        return (
            envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(True),
            envs.SGLANG_OPT_HICACHE_MAMBA_SLOT_STATE_ASYNC.override(async_on),
            envs.SGLANG_OPT_MAMBA_HOST_ONLY_CHECKPOINT.override(host_only),
        )

    def test_cpu_states_keep_the_blocking_copy(self):
        host, device_pool = _make_pools(True)
        short_conv, ngram = (s.tensor for s in device_pool._slot_siblings)
        short_conv[1], ngram[1] = 2.5, torch.tensor([7, 9])
        master, switch, host_only = self._switches()
        pf_lf, lf_pf = _stub_conv_and_temporal_copies()
        with master, switch, host_only, pf_lf, lf_pf:
            host.backup_from_device_all_layer(
                device_pool, torch.tensor([2]), torch.tensor([1]), "direct"
            )
            self.assertIn("CUDA", slot_state_copy._state["off_reason"])
            host.load_to_device_per_layer(
                device_pool, torch.tensor([2]), torch.tensor([3]), 0, "direct"
            )
        self.assertTrue(torch.equal(short_conv[3], short_conv[1]))
        self.assertEqual(ngram[3].tolist(), [7, 9])

    def test_legs_take_the_stream_ordered_copy(self):
        host, device_pool = _make_pools(True)
        short_conv = device_pool._slot_siblings[0].tensor
        short_conv[1] = 4.0
        for host_only in (False, True):
            master, switch, host_only_env = self._switches(host_only=host_only)
            with (
                self.subTest(host_only=host_only),
                master,
                switch,
                host_only_env,
                mock.patch.object(slot_state_copy, "applies", return_value=True),
                mock.patch.object(
                    slot_state_copy, "backup_rows", return_value=True
                ) as backup,
                mock.patch.object(
                    slot_state_copy, "load_rows", return_value=True
                ) as load,
            ):
                host._transfer_slot_states(
                    torch.tensor([2]), torch.tensor([1]), to_device=False
                )
                host._transfer_slot_states(
                    torch.tensor([2]), torch.tensor([3]), to_device=True
                )
            backup.assert_called_once()
            self.assertEqual(backup.call_args.kwargs["host_indices"].tolist(), [2])
            # the backup went to the kernel, so the host rows stay as they were
            self.assertFalse(bool(host.slot_state_buffers[0][2].any()))
            self.assertEqual(load.call_count, int(host_only))

    def test_a_failed_launch_falls_back_to_the_blocking_copy(self):
        host, device_pool = _make_pools(True)
        short_conv = device_pool._slot_siblings[0].tensor
        short_conv[1] = 3.0
        master, switch, host_only = self._switches()
        with (
            master,
            switch,
            host_only,
            mock.patch.object(slot_state_copy, "applies", return_value=True),
            mock.patch.object(
                slot_state_copy, "_launch", side_effect=RuntimeError("no UVA")
            ),
        ):
            host._transfer_slot_states(
                torch.tensor([2]), torch.tensor([1]), to_device=False
            )
        self.assertIn("no UVA", slot_state_copy._state["off_reason"])
        self.assertTrue(torch.equal(host.slot_state_buffers[0][2], short_conv[1]))

    def test_switch_off_or_master_off_never_applies(self):
        state = torch.zeros(4, 8)
        host = torch.zeros(4, 8)
        for master_on, async_on in ((True, False), (False, True)):
            with (
                self.subTest(master=master_on, switch=async_on),
                envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(master_on),
                envs.SGLANG_OPT_HICACHE_MAMBA_SLOT_STATE_ASYNC.override(async_on),
            ):
                self.assertFalse(slot_state_copy.applies([state], [host]))
        self.assertIsNone(slot_state_copy._state["off_reason"])

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_round_trip_matches_the_blocking_copy(self):
        states = [
            torch.randn(6, 5, 3, device="cuda").bfloat16(),
            torch.randint(0, 1 << 20, (6, 2), device="cuda"),
            torch.randint(0, 255, (6, 7), device="cuda", dtype=torch.uint8),
        ]
        hosts = [
            torch.zeros((8,) + tuple(s.shape[1:]), dtype=s.dtype).pin_memory()
            for s in states
        ]
        master, switch, host_only = self._switches(host_only=True)
        with master, switch, host_only:
            self.assertTrue(slot_state_copy.applies(states, hosts))
            dev, hst = torch.tensor([4, 1], device="cuda"), torch.tensor([7, 0])
            self.assertTrue(slot_state_copy.backup_rows(states, hosts, dev, hst))
            torch.cuda.synchronize()
            for state, host in zip(states, hosts):
                self.assertTrue(torch.equal(host[[7, 0]], state[[4, 1]].cpu()))
            targets = [torch.zeros_like(s) for s in states]
            self.assertTrue(
                slot_state_copy.load_rows(
                    targets, hosts, torch.tensor([2, 5], device="cuda"), hst
                )
            )
            torch.cuda.synchronize()
            for target, state in zip(targets, states):
                self.assertTrue(torch.equal(target[[2, 5]], state[[4, 1]]))


if __name__ == "__main__":
    unittest.main()
