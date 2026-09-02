import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np


def _install_fake_mori_if_unavailable():
    try:
        from mori.cpp import TransferStatus  # noqa: F401
        from mori.io import MemoryDesc  # noqa: F401

        return None
    except ImportError:
        pass

    saved_modules = {
        name: sys.modules.get(name) for name in ("mori", "mori.cpp", "mori.io")
    }

    fake_mori = types.ModuleType("mori")
    fake_mori.__path__ = []
    fake_cpp = types.ModuleType("mori.cpp")
    fake_io = types.ModuleType("mori.io")

    class FakeTransferStatus:
        pass

    class FakeMemoryDesc:
        pass

    fake_cpp.TransferStatus = FakeTransferStatus
    for name in (
        "BackendType",
        "EngineDesc",
        "IOEngine",
        "IOEngineConfig",
        "MemoryDesc",
        "MemoryLocationType",
        "PollCqMode",
        "RdmaBackendConfig",
        "StatusCode",
    ):
        setattr(fake_io, name, FakeMemoryDesc if name == "MemoryDesc" else object)

    fake_mori.cpp = fake_cpp
    fake_mori.io = fake_io
    sys.modules["mori"] = fake_mori
    sys.modules["mori.cpp"] = fake_cpp
    sys.modules["mori.io"] = fake_io
    return saved_modules


_SAVED_MORI_MODULES = _install_fake_mori_if_unavailable()

from sglang.srt.disaggregation.base.conn import StateType  # noqa: E402
from sglang.srt.disaggregation.mori.conn import MoriKVManager  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

if _SAVED_MORI_MODULES is not None:
    for _name, _module in _SAVED_MORI_MODULES.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _desc(data, size=4096):
    return SimpleNamespace(data=data, size=size)


class TestMoriDSATailState(unittest.TestCase):
    def test_wraparound_uses_descriptor_relative_offsets_and_pp_slice(self):
        manager = object.__new__(MoriKVManager)
        manager.kv_args = SimpleNamespace(prefill_start_layer=1, prefill_end_layer=3)
        manager.engine = MagicMock()
        manager.engine.allocate_transfer_uid.side_effect = range(4)
        manager.engine.batch_write.side_effect = lambda *args: [args[-1][0]]

        src_descs = [_desc(base) for base in (1000, 2000, 3000, 4000)]
        all_dst_descs = [_desc(10000 + 1000 * i) for i in range(8)]
        src_indices = np.array([1, 3, 2, 0, 2, 5], dtype=np.int32)
        dst_indices = np.array([2, 1, 3, 0, 1, 7], dtype=np.int32)

        statuses = manager._send_dsa_tail_state(
            src_indices,
            dst_indices,
            src_descs,
            all_dst_descs,
            [50] * 4,
            [70] * 8,
        )

        self.assertEqual(statuses, [0, 1, 2, 3])
        self.assertEqual(manager.engine.batch_write.call_count, 4)
        selected_dst_descs = [all_dst_descs[i] for i in (1, 2, 5, 6)]
        for call, src_desc, dst_desc in zip(
            manager.engine.batch_write.call_args_list,
            src_descs,
            selected_dst_descs,
        ):
            self.assertEqual(
                call.args,
                (
                    [src_desc],
                    [[80, 50, 60]],
                    [dst_desc],
                    [[150, 170, 140]],
                    [[20, 10, 10]],
                    [statuses[src_descs.index(src_desc)]],
                ),
            )

    def test_validates_all_descriptors_before_submitting(self):
        manager = object.__new__(MoriKVManager)
        manager.kv_args = SimpleNamespace(prefill_start_layer=0, prefill_end_layer=2)
        manager.engine = MagicMock()

        with self.assertRaisesRegex(
            ValueError, "DSA tail transfer block exceeds registered memory"
        ):
            manager._send_dsa_tail_state(
                np.array([1, 3, 2, 0, 2, 5], dtype=np.int32),
                np.array([2, 1, 3, 0, 1, 7], dtype=np.int32),
                [_desc(1000), _desc(2000, size=55)],
                [_desc(3000), _desc(4000)],
                [50, 50],
                [70, 70],
            )

        manager.engine.batch_write.assert_not_called()

    def test_send_state_dispatches_dsa_tail(self):
        manager = object.__new__(MoriKVManager)
        src_descs = [_desc(1000)]
        dst_descs = [_desc(2000)]
        src_indices = np.array([0, 0, 1, 0, 0, 2], dtype=np.int32)
        dst_indices = np.array([0, 1, 1, 0, 0, 3], dtype=np.int32)
        manager.state_mem_descs = [src_descs]
        manager.kv_args = SimpleNamespace(
            state_types=[StateType.DSA_TAIL],
            state_item_lens=[[20]],
            state_dim_per_tensor=[[]],
        )
        manager._send_dsa_tail_state = MagicMock(return_value=["submitted"])
        peer_info = SimpleNamespace(
            dst_state_mem_descs=[dst_descs],
            dst_state_item_lens=[[30]],
            dst_state_dim_per_tensor=[[]],
        )

        self.assertEqual(
            manager.send_state(peer_info, [src_indices], [dst_indices]),
            ["submitted"],
        )
        manager._send_dsa_tail_state.assert_called_once_with(
            src_indices,
            dst_indices,
            src_descs,
            dst_descs,
            [20],
            [30],
        )


if __name__ == "__main__":
    unittest.main()
