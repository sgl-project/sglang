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
    infos = (pool.buffer_info, *draft_infos)
    pool.temporal_device_layers = [
        layer for info in infos for layer in info.temporal.unbind(0)
    ]
    pool.conv_device_layers = [
        [layer for info in infos for layer in info.conv[0].unbind(0)]
    ]
    pool.temporal_dtype = pool.buffer_info.temporal.dtype
    pool.conv_dtype = pool.buffer_info.conv[0].dtype
    pool.conv_state_elem_sizes = [8]
    pool.slot_state_device_tensors = list(side_states)
    pool.temporal_buffer = torch.empty(num_slots, pool.num_mamba_layers, 1, 2, 4)
    pool.conv_buffer = [torch.empty(num_slots, pool.num_mamba_layers, 1, 2, 4)]
    pool.slot_state_buffers = [
        torch.empty((num_slots, *s.shape[1:])) for s in side_states
    ]
    pool.temporal_staging_buffer = None
    pool.conv_staging_buffers = [None]
    pool._temporal_can_use_jit = False
    pool._conv_can_use_jit = [False]
    pool.temporal_device_ptrs = None
    pool.conv_device_ptrs = [None]
    pool._buffer_device = torch.device("cpu")
    return pool


def _make_draft_info(num_slots):
    return MambaStateBufferInfo(
        temporal=torch.empty(1, num_slots + 1, 2, 4),
        conv=(torch.empty(1, num_slots + 1, 2, 4),),
    )


class TestNPUMambaStateTransferLoad(unittest.TestCase):
    def test_load_submits_per_layer_components(self):
        pool = _make_pool()
        op = MagicMock()
        host_indices = torch.tensor([1, 3])
        device_indices = torch.tensor([0, 4])

        with patch.object(
            mamba_pool_host,
            "transfer_state_per_layer_direct_pf_lf",
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
            self.assertEqual(kwargs["layer_id"], 1)
            self.assertEqual(kwargs["dst_indices"].tolist(), [0, 4])
            self.assertEqual(kwargs["src_indices"].tolist(), [1, 3])
            self.assertEqual(kwargs["dst"].shape, (5, 2, 4))
            self.assertEqual(kwargs["src"].shape, (4, 2, 1, 2, 4))

    def test_load_layer0_copies_side_state_with_torch(self):
        pool = _make_pool(side_states=(torch.zeros(5, 6),))
        pool.slot_state_buffers[0].zero_()
        pool.slot_state_buffers[0][1].fill_(7)
        op = MagicMock()

        with patch.object(
            mamba_pool_host,
            "transfer_state_per_layer_direct_pf_lf",
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

        self.assertEqual(op.call_count, 2)  # temporal and conv only
        self.assertTrue(torch.all(pool.slot_state_device_tensors[0][2] == 7))
        self.assertTrue(torch.all(pool.slot_state_device_tensors[0][1] == 0))

    def test_load_scalar_side_state_uses_torch_copy(self):
        pool = _make_pool(side_states=(torch.zeros(5),))
        pool.slot_state_buffers[0].zero_()
        pool.slot_state_buffers[0][1] = 7
        op = MagicMock()
        with patch.object(
            mamba_pool_host, "transfer_state_per_layer_direct_pf_lf", op, create=True
        ):
            pool.load_to_device_per_layer(
                device_pool=None,
                host_indices=torch.tensor([1]),
                device_indices=torch.tensor([2]),
                layer_id=0,
                io_backend="kernel_ascend",
            )
        self.assertEqual(op.call_count, 2)
        self.assertEqual(pool.slot_state_device_tensors[0][2].item(), 7)

    def test_load_draft_uses_global_host_layer(self):
        pool = _make_pool(num_layers=2, draft_infos=(_make_draft_info(4),))
        op = MagicMock()

        with patch.object(
            mamba_pool_host,
            "transfer_state_per_layer_direct_pf_lf",
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

        # Draft device layers are separate, while host layers share one buffer.
        self.assertEqual(op.call_count, 2)
        for call in op.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(kwargs["layer_id"], 2)
            self.assertEqual(kwargs["dst"].shape, (5, 2, 4))
            self.assertEqual(kwargs["src"].shape, (4, 3, 1, 2, 4))
        self.assertEqual(
            op.call_args_list[0].kwargs["dst"].data_ptr(),
            pool.mtp_draft_buffer_infos[0].temporal[0].data_ptr(),
        )

    def test_load_strided_device_slots_pass_original_view(self):
        pool = _make_pool()
        wide = torch.zeros(2, 10, 2, 4)
        pool.buffer_info = MambaStateBufferInfo(
            temporal=wide[:, ::2], conv=pool.buffer_info.conv
        )
        op = MagicMock(
            side_effect=lambda **kwargs: kwargs["dst"].index_fill_(
                0, kwargs["dst_indices"], 7
            )
        )
        with patch.object(
            mamba_pool_host, "transfer_state_per_layer_direct_pf_lf", op, create=True
        ):
            pool.load_to_device_per_layer(
                device_pool=None,
                host_indices=torch.tensor([1, 3]),
                device_indices=torch.tensor([0, 4]),
                layer_id=1,
                io_backend="kernel_ascend",
            )
        transfer = op.call_args_list[0].kwargs
        self.assertEqual(transfer["dst_indices"].tolist(), [0, 4])
        self.assertEqual(transfer["dst"].data_ptr(), wide[1, 0].data_ptr())
        self.assertEqual(transfer["dst"].stride(0), 16)
        self.assertTrue(torch.all(wide[1, 0] == 7))
        self.assertTrue(torch.all(wide[1, 8] == 7))
        self.assertTrue(torch.all(wide[1, 2] == 0))


class TestNPUMambaStateTransferBackup(unittest.TestCase):
    def test_backup_submits_each_main_component(self):
        pool = _make_pool()
        op = MagicMock()
        host_indices = torch.tensor([1, 3])
        device_indices = torch.tensor([0, 4])

        with patch.object(
            mamba_pool_host,
            "transfer_state_all_layer_direct_lf_pf",
            op,
            create=True,
        ):
            pool.backup_from_device_all_layer(
                device_pool=None,
                host_indices=host_indices,
                device_indices=device_indices,
                io_backend="kernel_ascend",
            )

        self.assertEqual(op.call_count, 2)  # temporal, then conv_0
        for call in op.call_args_list:
            kwargs = call.kwargs
            self.assertEqual(len(kwargs["device_states"]), 1)
            self.assertEqual(len(kwargs["host_states"]), 1)
            self.assertEqual(kwargs["device_indices"].tolist(), [0, 4])
            self.assertEqual(kwargs["host_indices"].tolist(), [1, 3])
        self.assertEqual(
            op.call_args_list[0].kwargs["host_states"][0].data_ptr(),
            pool.temporal_buffer.data_ptr(),
        )

    def test_backup_draft_component_addresses_layer_zero(self):
        pool = _make_pool(num_layers=2, draft_infos=(_make_draft_info(4),))
        op = MagicMock()

        with patch.object(
            mamba_pool_host,
            "transfer_state_all_layer_direct_lf_pf",
            op,
            create=True,
        ):
            pool.backup_from_device_all_layer(
                device_pool=None,
                host_indices=torch.tensor([1]),
                device_indices=torch.tensor([2]),
                io_backend="kernel_ascend",
            )

        # With draft layers, the existing device layer list is sent in chunks.
        self.assertEqual(op.call_count, 2)
        self.assertEqual(
            [state.shape[0] for state in op.call_args_list[0].kwargs["device_states"]],
            [1, 1, 1],
        )
        self.assertEqual(
            [state.shape[1] for state in op.call_args_list[0].kwargs["host_states"]],
            [1, 1, 1],
        )
        self.assertEqual(
            op.call_args_list[0].kwargs["host_states"][2].data_ptr(),
            pool.temporal_buffer[0, 2].data_ptr(),
        )

    def test_backup_chunks_draft_layer_list_at_operator_limit(self):
        pool = _make_pool(num_layers=17, draft_infos=(_make_draft_info(4),))
        op = MagicMock()
        with patch.object(
            mamba_pool_host, "transfer_state_all_layer_direct_lf_pf", op, create=True
        ):
            pool.backup_from_device_all_layer(
                device_pool=None,
                host_indices=torch.tensor([1]),
                device_indices=torch.tensor([2]),
                io_backend="kernel_ascend",
            )
        self.assertEqual(
            [len(c.kwargs["device_states"]) for c in op.call_args_list],
            [16, 2, 16, 2],
        )

    def test_backup_keeps_strided_host_view(self):
        pool = _make_pool()
        wide = torch.empty(4, 4, 1, 2, 4)
        pool.temporal_buffer = wide[:, ::2]
        op = MagicMock()
        with patch.object(
            mamba_pool_host, "transfer_state_all_layer_direct_lf_pf", op, create=True
        ):
            pool.backup_from_device_all_layer(
                device_pool=None,
                host_indices=torch.tensor([1]),
                device_indices=torch.tensor([2]),
                io_backend="kernel_ascend",
            )
        host = op.call_args_list[0].kwargs["host_states"][0]
        self.assertEqual(host.stride(), pool.temporal_buffer.stride())
        self.assertEqual(host.data_ptr(), wide.data_ptr())

    def test_backup_passes_strided_device_view_without_staging(self):
        pool = _make_pool()
        wide = torch.empty(2, 10, 2, 4)
        pool.buffer_info = MambaStateBufferInfo(
            temporal=wide[:, ::2], conv=pool.buffer_info.conv
        )
        op = MagicMock()
        with patch.object(
            mamba_pool_host, "transfer_state_all_layer_direct_lf_pf", op, create=True
        ):
            pool.backup_from_device_all_layer(
                device_pool=None,
                host_indices=torch.tensor([1]),
                device_indices=torch.tensor([2]),
                io_backend="kernel_ascend",
            )
        source = op.call_args_list[0].kwargs["device_states"][0]
        self.assertIs(source, pool.buffer_info.temporal)
        self.assertEqual(source.stride(), wide[:, ::2].stride())

    def test_backup_scalar_side_state_uses_torch_copy(self):
        pool = _make_pool(side_states=(torch.zeros(5),))
        pool.slot_state_buffers[0].zero_()
        pool.slot_state_device_tensors[0][2] = 7
        op = MagicMock()
        with patch.object(
            mamba_pool_host, "transfer_state_all_layer_direct_lf_pf", op, create=True
        ):
            pool.backup_from_device_all_layer(
                device_pool=None,
                host_indices=torch.tensor([1]),
                device_indices=torch.tensor([2]),
                io_backend="kernel_ascend",
            )
        self.assertEqual(op.call_count, 2)
        self.assertEqual(pool.slot_state_buffers[0][1].item(), 7)
        self.assertEqual(pool.slot_state_buffers[0][0].item(), 0)

    def test_backup_side_state_uses_torch_copy(self):
        pool = _make_pool(side_states=(torch.zeros(5, 6),))
        pool.slot_state_buffers[0].zero_()
        pool.slot_state_device_tensors[0][2].fill_(7)
        op = MagicMock()

        with patch.object(
            mamba_pool_host,
            "transfer_state_all_layer_direct_lf_pf",
            op,
            create=True,
        ):
            pool.backup_from_device_all_layer(
                device_pool=None,
                host_indices=torch.tensor([1]),
                device_indices=torch.tensor([2]),
                io_backend="kernel_ascend",
            )
        self.assertEqual(op.call_count, 2)
        self.assertTrue(torch.all(pool.slot_state_buffers[0][1] == 7))


if __name__ == "__main__":
    unittest.main()
