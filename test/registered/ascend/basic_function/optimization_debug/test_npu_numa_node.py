import re
import subprocess
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=800, suite="nightly-2-npu-a3", nightly=True)


class BaseNumaBindingTest(CustomTestCase):
    """Testcase: Verify that the --numa-node parameter can bind different TP processes to specified NUMA nodes.

    [Test Category] Parameter
    [Test Target] --numa-node
    """

    TP_SIZE = 2
    # When binding to different NUMA nodes, the CPU ranges are different
    CONFIG_NUMA_LIST = ["0", "1"]

    @classmethod
    def _is_numactl_installed(cls):
        try:
            subprocess.run(
                ["numactl", "--hardware"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    @classmethod
    def setUpClass(cls):
        if not cls._is_numactl_installed():
            print("Installing numactl...")
            subprocess.run(
                ["apt", "update", "-y"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["apt", "install", "-y", "numactl"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("numactl installed successfully.")

        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.other_args = [
            "--trust-remote-code",
            "--tp-size",
            str(cls.TP_SIZE),
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--numa-node",
            *cls.CONFIG_NUMA_LIST,
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=3600,
            other_args=cls.other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _get_tp_pids(self):
        result = subprocess.run(["ps", "-ef"], stdout=subprocess.PIPE, text=True)
        tp_pids = {}
        for line in result.stdout.splitlines():
            match = re.search(r"sglang::scheduler_TP(\d+)", line)
            if match:
                tp_pids[f"TP{match.group(1)}"] = line.split()[1]
        return tp_pids

    def _get_taskset_cpu_range(self, pid):
        output = subprocess.run(
            ["taskset", "-cp", pid], stdout=subprocess.PIPE, text=True
        ).stdout.strip()
        return output.split(":")[-1].strip()

    def test_numa_binding(self):
        tp_pids = self._get_tp_pids()
        cpu0 = self._get_taskset_cpu_range(tp_pids["TP0"])
        cpu1 = self._get_taskset_cpu_range(tp_pids["TP1"])

        # When binding to the same NUMA node, the CPU ranges should be the same; when binding to different NUMA nodes, the CPU ranges should be different.
        if self.CONFIG_NUMA_LIST[0] == self.CONFIG_NUMA_LIST[1]:
            self.assertEqual(
                cpu0, cpu1, "The same NUMA node should bind to the same CPU range"
            )
        else:
            self.assertNotEqual(
                cpu0, cpu1, "Different NUMA nodes should bind to different CPU ranges"
            )


class TestNumaSame(BaseNumaBindingTest):
    # When binding to the same NUMA node, the CPU ranges are the same
    CONFIG_NUMA_LIST = ["1", "1"]


if __name__ == "__main__":
    unittest.main()
