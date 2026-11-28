import unittest
from unittest.mock import patch

import torch

import sglang.srt.mem_cache.pool_host.mamba as mamba
from sglang.srt.mem_cache.device_pool_info import MambaStateBufferInfo
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

HOST, DEVICE = torch.tensor([1]), torch.tensor([2])


def _pool(layers=2, draft=False, side_shape=None):
    pool = MambaPoolHost.__new__(MambaPoolHost)

    def buffers(n):
        return MambaStateBufferInfo(
            temporal=torch.zeros(n, 5, 2, 4),
            conv=(torch.zeros(n, 5, 2, 4),),
        )

    pool.buffer_info = buffers(layers)
    pool.mtp_draft_buffer_infos = (buffers(1),) if draft else ()
    infos = (pool.buffer_info, *pool.mtp_draft_buffer_infos)
    pool.num_mamba_layers = layers + int(draft)
    pool.temporal_device_layers = [x for info in infos for x in info.temporal.unbind()]
    pool.conv_device_layers = [[x for info in infos for x in info.conv[0].unbind()]]
    pool.temporal_buffer = torch.zeros(4, pool.num_mamba_layers, 1, 2, 4)
    pool.conv_buffer = [torch.zeros_like(pool.temporal_buffer)]
    pool.temporal_state_elem_size = 8
    pool.conv_state_shapes = [torch.Size([2, 4])]
    pool.slot_state_device_tensors = (
        [] if side_shape is None else [torch.zeros((5, *side_shape))]
    )
    pool.slot_state_buffers = (
        [] if side_shape is None else [torch.zeros((4, *side_shape))]
    )
    pool.temporal_staging_buffer = pool.temporal_device_ptrs = None
    pool.conv_staging_buffers = pool.conv_device_ptrs = [None]
    pool._temporal_can_use_jit = False
    pool._conv_can_use_jit = [False]
    return pool


def _load(pool, layer=0):
    pool.load_to_device_per_layer(
        None, HOST, DEVICE, layer, "kernel_ascend", is_draft=layer >= 2
    )


def _backup(pool):
    pool.backup_from_device_all_layer(None, HOST, DEVICE, "kernel_ascend")


class TestNPUMambaStateTransfer(unittest.TestCase):
    def test_load_main_and_draft_layers(self):
        for draft, layer in ((False, 1), (True, 2)):
            with self.subTest(draft=draft):
                pool = _pool(draft=draft)
                with patch.object(
                    mamba, "transfer_kv_per_layer_direct_pf_lf", create=True
                ) as op:
                    _load(pool, layer)
                self.assertEqual(op.call_count, 2)  # temporal and conv
                for call in op.call_args_list:
                    args = call.kwargs
                    self.assertEqual(args["layer_id"], layer)
                    self.assertEqual(args["src_indices"].tolist(), [1])
                    self.assertEqual(args["dst_indices"].tolist(), [2])
                    # Indices stay host-resident so the AIV backend can
                    # validate without an NPU-to-CPU sync.
                    self.assertEqual(args["dst_indices"].device.type, "cpu")
                    self.assertEqual(args["dst"].shape, (5, 2, 4))
                if draft:
                    self.assertEqual(
                        op.call_args.kwargs["dst"].data_ptr(),
                        pool.mtp_draft_buffer_infos[0].conv[0][0].data_ptr(),
                    )

    def test_backup_main_and_draft_layers(self):
        for draft in (False, True):
            with self.subTest(draft=draft):
                pool = _pool(draft=draft)
                with patch.object(
                    mamba, "transfer_kv_all_layer_direct_lf_pf", create=True
                ) as op:
                    _backup(pool)
                self.assertEqual(op.call_count, 2)
                for call in op.call_args_list:
                    args = call.kwargs
                    self.assertEqual(args["device_indices"].tolist(), [2])
                    self.assertEqual(args["host_indices"].tolist(), [1])
                    # One component per layer (main plus draft layers).
                    self.assertEqual(len(args["device_states"]), 3 if draft else 1)
                    self.assertEqual(len(args["host_states"]), 3 if draft else 1)
                if draft:
                    self.assertEqual(
                        op.call_args_list[0].kwargs["device_states"][0].shape,
                        (1, 5, 2, 4),
                    )
                    self.assertEqual(
                        op.call_args_list[0].kwargs["host_states"][2].data_ptr(),
                        pool.temporal_buffer[0, 2].data_ptr(),
                    )

    def test_side_state_uses_torch_in_both_directions(self):
        for shape in ((), (6,)):
            with self.subTest(shape=shape):
                pool = _pool(side_shape=shape)
                pool.slot_state_buffers[0][1].fill_(7)
                with (
                    patch.object(
                        mamba, "transfer_kv_per_layer_direct_pf_lf", create=True
                    ),
                    patch.object(
                        mamba, "transfer_kv_all_layer_direct_lf_pf", create=True
                    ),
                ):
                    _load(pool)
                    self.assertTrue(
                        torch.all(pool.slot_state_device_tensors[0][2] == 7)
                    )
                    pool.slot_state_device_tensors[0][2].fill_(9)
                    _backup(pool)
                self.assertTrue(torch.all(pool.slot_state_buffers[0][1] == 9))

    def test_strided_load_stages_then_scatters(self):
        pool = _pool()
        wide = torch.zeros(2, 10, 2, 4)
        pool.buffer_info = MambaStateBufferInfo(
            temporal=wide[:, ::2], conv=pool.buffer_info.conv
        )

        def copy(**kw):
            kw["dst"].index_fill_(0, kw["dst_indices"], 7)

        with patch.object(
            mamba, "transfer_kv_per_layer_direct_pf_lf", side_effect=copy, create=True
        ) as op:
            _load(pool, 1)
        self.assertEqual(op.call_args_list[0].kwargs["dst"].stride(0), 8)
        self.assertTrue(torch.all(wide[1, 4] == 7))
        self.assertTrue(torch.all(wide[1, 2] == 0))

    def test_strided_backup_stages_source(self):
        pool = _pool()
        wide = torch.zeros(2, 10, 2, 4)
        wide[0, 4].fill_(7)
        pool.buffer_info = MambaStateBufferInfo(
            temporal=wide[:, ::2], conv=pool.buffer_info.conv
        )
        with patch.object(
            mamba, "transfer_kv_all_layer_direct_lf_pf", create=True
        ) as op:
            _backup(pool)
        source = op.call_args_list[0].kwargs["device_states"][0]
        self.assertEqual(source.shape, (2, 1, 2, 4))
        self.assertTrue(source.is_contiguous())
        self.assertTrue(torch.all(source[0, 0] == 7))


if __name__ == "__main__":
    unittest.main()
