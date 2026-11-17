import unittest
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.mem_cache.pool_host.mamba as mamba_pool_host
from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
from sglang.srt.mem_cache.pool_host.mamba import (
    MambaPoolHost,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def _make_pool(num_layers=2, num_slots=4, draft_infos=(), side_states=()):
    """Build a MambaPoolHost-like stub with CPU tensors standing in for NPU."""
    pool = MambaPoolHost.__new__(MambaPoolHost)
    pool.temporal_state_elem_size = 2 * 4
    pool.conv_state_shapes = [torch.Size([2, 4])]
    pool.num_mamba_layers = num_layers + len(draft_infos)
    pool.mtp_draft_buffer_infos = tuple(draft_infos)

    def dev_buf(layers):
        return torch.empty(layers, num_slots + 1, 2, 4)

    pool.buffer_info = MambaStateBufferInfo(
        temporal=dev_buf(num_layers),
        conv=(dev_buf(num_layers),),
        extra_checkpoint_buffers=tuple(side_states),
    )
    pool.slot_state_device_tensors = list(side_states)
    pool.temporal_buffer = torch.empty(num_slots, pool.num_mamba_layers, 1, 2, 4)
    pool.conv_buffer = [torch.empty(num_slots, pool.num_mamba_layers, 1, 2, 4)]
    pool.slot_state_buffers = [
        torch.empty(num_slots, s.shape[1]) for s in side_states
    ]
    pool._buffer_device = torch.device("cpu")
    return pool


def _make_draft_info(num_slots):
    return MambaStateBufferInfo(
        temporal=torch.empty(1, num_slots + 1, 2, 4),
        conv=(torch.empty(1, num_slots + 1, 2, 4),),
    )


def _npu_state_transfer_gates(backend=True):
    """Patch the module so _init_npu_state_transfer runs on CPU.

    The operator names only exist as module attributes on NPU, so the patches
    create them; missing operators are an import-time error there, not a
    runtime gate to test."""
    return (
        patch.object(mamba_pool_host, "_is_npu", True),
        patch.object(
            mamba_pool_host,
            "transfer_kv_per_layer_kv_split_pf_lf",
            object(),
            create=True,
        ),
        patch.object(
            mamba_pool_host,
            "transfer_kv_all_layer_kv_split_lf_pf",
            object(),
            create=True,
        ),
        patch(
            "sglang.srt.mem_cache.pool_host.npu_host_memory.ascendc_io_enabled",
            return_value=backend,
        ),
    )


def _init_npu_state_transfer(pool, backend=True):
    gates = _npu_state_transfer_gates(backend=backend)
    with gates[0], gates[1], gates[2], gates[3]:
        pool._init_npu_state_transfer()
    return pool


def _enable_npu_state_transfer(pool):
    """Build the real views on a stub pool (gates patched for CPU)."""
    return _init_npu_state_transfer(pool)


class TestNPUMambaStateTransferConfig(unittest.TestCase):
    @patch.object(mamba_pool_host, "_is_npu", False)
    def test_non_npu_platform_is_noop(self):
        pool = _make_pool()
        pool._init_npu_state_transfer()
        self.assertEqual(pool._npu_state_transfer_views, ())

    def test_unmapped_host_memory_raises(self):
        pool = _make_pool()
        with self.assertRaises(RuntimeError):
            _init_npu_state_transfer(pool, backend=False)

    def test_slot_strided_host_view_stays_valid(self):
        pool = _make_pool()
        n, l = pool.temporal_buffer.shape[:2]
        wide = torch.zeros(n, l * 2, 1, 2, 4)
        pool.temporal_buffer = wide[:, ::2]  # slot-to-slot stride, payload intact
        pool = _init_npu_state_transfer(pool)
        (_, _, dev_v, host_v), = (
            v for v in pool._npu_state_transfer_views if v[0] == "temporal"
        )
        self.assertEqual(
            host_v.untyped_storage().data_ptr(), wide.untyped_storage().data_ptr()
        )
        self.assertTrue(
            torch.equal(
                host_v[1, 0].reshape(-1),
                pool.temporal_buffer[1, 0].reshape(-1),
            )
        )


class TestNPUMambaStateTransferLoad(unittest.TestCase):
    def test_load_submits_per_layer_components(self):
        pool = _enable_npu_state_transfer(_make_pool())
        op = MagicMock()
        host_indices = torch.tensor([1, 3])
        device_indices = torch.tensor([0, 4])

        with patch.object(
            mamba_pool_host,
            "transfer_kv_per_layer_kv_split_pf_lf",
            op,
            create=True,
        ):
            pool.load_to_device_per_layer(
                device_pool=None,
                host_indices=host_indices,
                device_indices=device_indices,
                layer_id=1,
                io_backend="kernel_ascend",
            )

        # layer 1: temporal + conv_0; side states only ride layer 0.
        self.assertEqual(op.call_count, 2)
        for call in op.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(kwargs["page_size"], 1)
            self.assertEqual(kwargs["device_layer_id"], 1)
            self.assertEqual(kwargs["host_layer_id"], 1)
            self.assertEqual(kwargs["device_indices"].tolist(), [0, 4])
            self.assertEqual(kwargs["host_indices"].tolist(), [1, 3])
            self.assertTrue(kwargs["device_v"].numel() == 0)
            self.assertTrue(kwargs["host_v"].numel() == 0)

    def test_load_layer0_submits_side_states(self):
        pool = _enable_npu_state_transfer(_make_pool(side_states=(torch.empty(5, 6),)))
        op = MagicMock()

        with patch.object(
            mamba_pool_host,
            "transfer_kv_per_layer_kv_split_pf_lf",
            op,
            create=True,
        ):
            pool.load_to_device_per_layer(
                device_pool=None,
                host_indices=torch.tensor([1]),
                device_indices=torch.tensor([2]),
                layer_id=0,
                io_backend="kernel_ascend",
            )

        # temporal + conv_0 + side_0 at layer 0.
        self.assertEqual(op.call_count, 3)
        side_call = op.call_args_list[-1].kwargs
        self.assertEqual(side_call["device_layer_id"], 0)
        self.assertEqual(side_call["host_layer_id"], 0)

    def test_load_draft_component_addresses_layer_zero(self):
        pool = _enable_npu_state_transfer(
            _make_pool(num_layers=2, draft_infos=(_make_draft_info(4),))
        )
        op = MagicMock()

        with patch.object(
            mamba_pool_host,
            "transfer_kv_per_layer_kv_split_pf_lf",
            op,
            create=True,
        ):
            pool.load_to_device_per_layer(
                device_pool=None,
                host_indices=torch.tensor([1]),
                device_indices=torch.tensor([2]),
                layer_id=2,  # n_main + depth 0
                io_backend="kernel_ascend",
            )

        # Draft temporal + conv submit once, on the draft layer's own call;
        # both operands are single-layer views, so both address layer 0.
        self.assertEqual(op.call_count, 2)
        for call in op.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(kwargs["device_layer_id"], 0)
            self.assertEqual(kwargs["host_layer_id"], 0)

    def test_load_skips_transfer_slot_states(self):
        pool = _enable_npu_state_transfer(_make_pool(side_states=(torch.empty(5, 6),)))
        pool._transfer_slot_states = MagicMock()

        with patch.object(
            mamba_pool_host,
            "transfer_kv_per_layer_kv_split_pf_lf",
            MagicMock(),
            create=True,
        ):
            pool.load_to_device_per_layer(
                device_pool=None,
                host_indices=torch.tensor([1]),
                device_indices=torch.tensor([2]),
                layer_id=0,
                io_backend="kernel_ascend",
            )
        pool._transfer_slot_states.assert_not_called()


class TestNPUMambaStateTransferBackup(unittest.TestCase):
    def test_backup_submits_all_layer_components(self):
        pool = _enable_npu_state_transfer(_make_pool())
        op = MagicMock()
        host_indices = torch.tensor([1, 3])
        device_indices = torch.tensor([0, 4])

        with patch.object(
            mamba_pool_host,
            "transfer_kv_all_layer_kv_split_lf_pf",
            op,
            create=True,
        ):
            pool.backup_from_device_all_layer(
                device_pool=None,
                host_indices=host_indices,
                device_indices=device_indices,
                io_backend="kernel_ascend",
            )

        self.assertEqual(op.call_count, 2)  # temporal + conv_0
        for call in op.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(kwargs["page_size"], 1)
            self.assertEqual(kwargs["host_layer_start"], 0)
            self.assertEqual(kwargs["device_indices"].tolist(), [0, 4])
            self.assertEqual(kwargs["host_indices"].tolist(), [1, 3])

    def test_backup_draft_component_addresses_layer_zero(self):
        pool = _enable_npu_state_transfer(
            _make_pool(num_layers=2, draft_infos=(_make_draft_info(4),))
        )
        op = MagicMock()

        with patch.object(
            mamba_pool_host,
            "transfer_kv_all_layer_kv_split_lf_pf",
            op,
            create=True,
        ):
            pool.backup_from_device_all_layer(
                device_pool=None,
                host_indices=torch.tensor([1]),
                device_indices=torch.tensor([2]),
                io_backend="kernel_ascend",
            )

        # temporal + conv_0 + temporal_draft + conv_0_draft; the draft host
        # views are pre-narrowed to their layer, so every call starts at 0.
        self.assertEqual(op.call_count, 4)
        for call in op.call_args_list:
            self.assertEqual(call.kwargs["host_layer_start"], 0)

    def test_backup_skips_transfer_slot_states(self):
        pool = _enable_npu_state_transfer(_make_pool(side_states=(torch.empty(5, 6),)))
        pool._transfer_slot_states = MagicMock()
        pool._copy_tensor_all_layers_lf_pf = MagicMock()

        with patch.object(
            mamba_pool_host,
            "transfer_kv_all_layer_kv_split_lf_pf",
            MagicMock(),
            create=True,
        ):
            pool.backup_from_device_all_layer(
                device_pool=None,
                host_indices=torch.tensor([1]),
                device_indices=torch.tensor([2]),
                io_backend="kernel_ascend",
            )
        pool._transfer_slot_states.assert_not_called()
        pool._copy_tensor_all_layers_lf_pf.assert_not_called()


class TestNPUMambaStateTransferViews(unittest.TestCase):
    """Real-tensor checks on the view-building path (no view mocks)."""

    def _pool(self):
        return _init_npu_state_transfer(
            _make_pool(
                num_layers=2,
                draft_infos=(_make_draft_info(4),),
                side_states=(torch.empty(5, 6),),
            )
        )

    def test_views_are_true_views_with_contiguous_payload(self):
        pool = self._pool()
        self.assertEqual(
            {(v[0], v[1]) for v in pool._npu_state_transfer_views},
            {
                ("temporal", None),
                ("conv_0", None),
                ("temporal_draft", 0),
                ("conv_0_draft", 0),
                ("side_0", None),
            },
        )
        # Every view must alias its source buffer: view() never copies, and
        # the round trip only lands in the real caches when storage matches.
        dev_src = {
            ("temporal", None): pool.buffer_info.temporal,
            ("conv_0", None): pool.buffer_info.conv[0],
            ("temporal_draft", 0): pool.mtp_draft_buffer_infos[0].temporal,
            ("conv_0_draft", 0): pool.mtp_draft_buffer_infos[0].conv[0],
            ("side_0", None): pool.slot_state_device_tensors[0],
        }
        host_src = {
            ("temporal", None): pool.temporal_buffer,
            ("conv_0", None): pool.conv_buffer[0],
            ("temporal_draft", 0): pool.temporal_buffer,
            ("conv_0_draft", 0): pool.conv_buffer[0],
            ("side_0", None): pool.slot_state_buffers[0],
        }
        for name, key, dev, host in pool._npu_state_transfer_views:
            k = (name, key)
            self.assertEqual(
                dev.untyped_storage().data_ptr(),
                dev_src[k].untyped_storage().data_ptr(),
                k,
            )
            self.assertEqual(
                host.untyped_storage().data_ptr(),
                host_src[k].untyped_storage().data_ptr(),
                k,
            )
            # Slot-to-slot strides may be arbitrary, but the trailing
            # (page, layer) payload must be contiguous.
            for view in (dev, host):
                self.assertEqual(view.stride(4), 1, k)
                self.assertEqual(view.stride(3), view.shape[4], k)
                self.assertEqual(view.stride(2), view.shape[3] * view.shape[4], k)

    def test_view_layout_matches_slot_layout(self):
        pool = self._pool()
        pool.temporal_buffer.copy_(
            torch.arange(pool.temporal_buffer.numel())
            .reshape(pool.temporal_buffer.shape)
            .float()
        )
        pool.buffer_info.temporal.copy_(
            torch.arange(pool.buffer_info.temporal.numel())
            .reshape(pool.buffer_info.temporal.shape)
            .float()
        )
        side_state = pool.slot_state_device_tensors[0]
        side_state.copy_(
            torch.arange(side_state.numel()).reshape(side_state.shape).float()
        )
        views = {(v[0], v[1]): (v[2], v[3]) for v in pool._npu_state_transfer_views}

        dev, host = views[("temporal", None)]
        for layer in (0, 1):
            for slot in (0, 4):
                # Device views are layer-first and address the real buffer.
                self.assertTrue(
                    torch.equal(
                        dev[layer, slot].reshape(-1),
                        pool.buffer_info.temporal[layer, slot].reshape(-1),
                    )
                )
        for slot in (0, 3):
            for layer in (0, 2):
                # Host views are page-first and address the real buffer.
                self.assertTrue(
                    torch.equal(
                        host[slot, layer].reshape(-1),
                        pool.temporal_buffer[slot, layer].reshape(-1),
                    )
                )

        # Draft: single-layer device allocation, host view pre-narrowed to
        # the draft layer of the shared pool buffer.
        dev, host = views[("temporal_draft", 0)]
        draft_info = pool.mtp_draft_buffer_infos[0]
        self.assertTrue(
            torch.equal(dev[0, 2].reshape(-1), draft_info.temporal[0, 2].reshape(-1))
        )
        self.assertTrue(
            torch.equal(host[3, 0].reshape(-1), pool.temporal_buffer[3, 2].reshape(-1))
        )

        # Side state: no layer axis; both views address their single layer 0.
        dev, host = views[("side_0", None)]
        self.assertTrue(torch.equal(dev[0, 1].reshape(-1), side_state[1].reshape(-1)))
        self.assertTrue(
            torch.equal(
                host[2, 0].reshape(-1), pool.slot_state_buffers[0][2].reshape(-1)
            )
        )

    def test_h2d_d2h_roundtrip_through_views(self):
        pool = _init_npu_state_transfer(_make_pool(num_layers=2))
        host, dev = pool.temporal_buffer, pool.buffer_info.temporal
        host.copy_(torch.arange(host.numel()).reshape(host.shape).float())
        dev.fill_(0)
        (_, _, dev_v, host_v), = (
            v for v in pool._npu_state_transfer_views if v[0] == "temporal"
        )
        hi, di = (2, 0), (3, 4)

        # H2D: the operator moves host[hi, l] into device[l, di].
        for l in range(2):
            for t in range(2):
                dev_v[l, di[t]].copy_(host_v[hi[t], l])
        for l in range(2):
            for t in range(2):
                self.assertTrue(
                    torch.equal(dev[l, di[t]].reshape(-1), host[hi[t], l].reshape(-1))
                )
        # Slots outside the index pair stay untouched.
        self.assertEqual(dev[0, 0].abs().sum().item(), 0.0)

        # D2H: the operator moves device[l, di] into host[hi, l] and the
        # write must land in the original host buffer.
        dev.copy_(torch.arange(dev.numel()).reshape(dev.shape).float() + 100)
        host.fill_(0)
        for l in range(2):
            for t in range(2):
                host_v[hi[t], l].copy_(dev_v[l, di[t]])
        for l in range(2):
            for t in range(2):
                self.assertTrue(
                    torch.equal(host[hi[t], l].reshape(-1), dev[l, di[t]].reshape(-1))
                )


if __name__ == "__main__":
    unittest.main()
