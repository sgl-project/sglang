from unittest import mock

import psutil

from sglang.multimodal_gen.runtime.utils import process


def test_kill_process_tree_parent_exits_during_child_enumeration():
    parent_pid = 12345
    parent = mock.Mock()
    parent.children.side_effect = psutil.NoSuchProcess(parent_pid)

    with (
        mock.patch.object(process.psutil, "Process", return_value=parent),
        mock.patch.object(process.signal, "signal"),
    ):
        process.kill_process_tree(parent_pid)

    parent.children.assert_called_once_with(recursive=True)
    parent.kill.assert_not_called()
