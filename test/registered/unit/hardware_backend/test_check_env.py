"""Unit tests for sglang.check_env: backend dispatch and the XPU / CPU reporters."""

import io
import subprocess
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import torch

from sglang import check_env
from sglang.check_env import CPUEnv, GPUEnv, UnknownEnv, XPUEnv, select_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

DETECTORS = (
    "is_cuda_v2",
    "is_hip",
    "is_npu",
    "is_musa",
    "is_mps",
    "is_xpu_v2",
    "is_cpu",
)

LSCPU_XEON = """Architecture:                         x86_64
Vendor ID:                            GenuineIntel
  Model name:                         Intel(R) Xeon(R) Platinum 8480+
    CPU family:                       6
    Flags:                            fpu vme avx512f avx512_bf16 amx_bf16 amx_int8
NUMA node0 CPU(s):                    0-55
"""

LSCPU_NO_AMX = """Vendor ID:                            AuthenticAMD
  Model name:                         AMD EPYC 7763
    Flags:                            fpu vme sse4_2 avx2
"""


def _props(name, memory_gib, driver_version):
    return types.SimpleNamespace(
        name=name,
        total_memory=int(memory_gib * 1024**3),
        driver_version=driver_version,
    )


class TestSelectEnv(CustomTestCase):
    """`select_env` used to be an if/elif chain with no else, so a host with no
    recognized accelerator raised NameError instead of printing an env report."""

    def _patch_detectors(self, **enabled):
        for name in DETECTORS:
            patcher = patch(
                f"sglang.check_env.{name}", return_value=enabled.get(name, False)
            )
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_no_accelerator_falls_back_instead_of_raising(self):
        self._patch_detectors()
        self.assertIsInstance(select_env(), UnknownEnv)

    def test_fallback_env_prints_a_full_report(self):
        env = UnknownEnv()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            env.check_env()
        output = buffer.getvalue()

        self.assertIn("Accelerator: none detected", output)
        self.assertIn("PyTorch:", output)
        self.assertIn("sglang:", output)

    def test_xpu_build_wins_over_the_cpu_fallback(self):
        self._patch_detectors(is_xpu_v2=True, is_cpu=True)
        self.assertIsInstance(select_env(), XPUEnv)

    def test_cpu_engine_is_not_reported_as_no_accelerator(self):
        self._patch_detectors(is_cpu=True)
        self.assertIsInstance(select_env(), CPUEnv)

    def test_cuda_build_wins_over_every_later_probe(self):
        self._patch_detectors(is_cuda_v2=True, is_xpu_v2=True, is_cpu=True)
        self.assertIsInstance(select_env(), GPUEnv)


class TestPackageList(CustomTestCase):
    """Subclasses used to extend the module-global PACKAGE_LIST in place, leaking
    one backend's extra packages into every env constructed afterwards."""

    def test_extra_packages_do_not_mutate_the_global_list(self):
        before = list(check_env.PACKAGE_LIST)

        env = XPUEnv()
        second = XPUEnv()

        self.assertEqual(check_env.PACKAGE_LIST, before)
        self.assertEqual(env.package_list, second.package_list)
        for package in XPUEnv.EXTRA_PACKAGE_LIST:
            self.assertIn(package, env.package_list)


class TestXPUEnv(CustomTestCase):
    def test_unavailable_device_still_reports_without_probing_devices(self):
        """A torch XPU build whose device is not visible (driver mismatch, no
        /dev/dri access) must report `XPU available: False`, not crash."""
        with (
            patch.object(torch.xpu, "is_available", return_value=False),
            patch.object(
                torch.xpu, "device_count", side_effect=AssertionError("probed devices")
            ),
        ):
            info = XPUEnv().get_info()

        self.assertEqual(info, {"XPU available": False})

    def test_identical_devices_are_grouped_into_one_row(self):
        props = [_props("Intel(R) Arc(TM) Pro B60 Graphics", 23.9, "1.15.38308+1")] * 4
        with (
            patch.object(torch.xpu, "device_count", return_value=4),
            patch.object(
                torch.xpu, "get_device_properties", side_effect=lambda k: props[k]
            ),
        ):
            info = XPUEnv().get_device_info()

        self.assertEqual(
            info,
            {
                "XPU 0,1,2,3": "Intel(R) Arc(TM) Pro B60 Graphics (23.9 GiB)",
                "XPU 0,1,2,3 Driver Version": "1.15.38308+1",
            },
        )

    def test_mixed_devices_are_reported_per_group(self):
        props = [
            _props("Intel(R) Arc(TM) Pro B60 Graphics", 23.9, "1.15.38308+1"),
            _props("Intel(R) Data Center GPU Max 1550", 63.9, "1.15.38308+1"),
        ]
        with (
            patch.object(torch.xpu, "device_count", return_value=2),
            patch.object(
                torch.xpu, "get_device_properties", side_effect=lambda k: props[k]
            ),
        ):
            info = XPUEnv().get_device_info()

        self.assertEqual(
            info,
            {
                "XPU 0": "Intel(R) Arc(TM) Pro B60 Graphics (23.9 GiB)",
                "XPU 1": "Intel(R) Data Center GPU Max 1550 (63.9 GiB)",
                "XPU 0,1 Driver Version": "1.15.38308+1",
            },
        )

    def test_topology_survives_a_missing_or_hung_xpu_smi(self):
        """xpu-smi is absent outside the XPU image, and hangs on a wedged
        Level-Zero driver; neither may abort the whole report."""
        for error in (
            FileNotFoundError("xpu-smi"),
            subprocess.TimeoutExpired(cmd="xpu-smi", timeout=15),
            subprocess.CalledProcessError(returncode=1, cmd="xpu-smi"),
        ):
            with self.subTest(error=type(error).__name__):
                with patch("subprocess.run", side_effect=error):
                    self.assertEqual(XPUEnv().get_topology(), {})

    def test_topology_is_prefixed_with_a_newline(self):
        completed = subprocess.CompletedProcess(
            args=["xpu-smi"], returncode=0, stdout="GPU 0/0  S\n"
        )
        with patch("subprocess.run", return_value=completed) as run:
            self.assertEqual(
                XPUEnv().get_topology(), {"XPU Topology": "\nGPU 0/0  S\n"}
            )

        self.assertEqual(run.call_args.kwargs["timeout"], 15)


class TestCPUEnv(CustomTestCase):
    def test_reports_the_engine_flag_and_the_isa_lscpu_advertises(self):
        with (
            patch("subprocess.check_output", return_value=LSCPU_XEON),
            patch.dict("os.environ", {"SGLANG_USE_CPU_ENGINE": "1"}),
        ):
            info = CPUEnv().get_info()

        self.assertEqual(info["SGLANG_USE_CPU_ENGINE"], "1")
        self.assertEqual(info["CPU Model"], "Intel(R) Xeon(R) Platinum 8480+")
        self.assertEqual(info["CPU ISA"], "avx512f,avx512_bf16,amx_bf16,amx_int8")

    def test_reports_a_cpu_without_avx512_or_amx(self):
        with patch("subprocess.check_output", return_value=LSCPU_NO_AMX):
            info = CPUEnv().get_info()

        self.assertEqual(info["CPU Model"], "AMD EPYC 7763")
        self.assertEqual(info["CPU ISA"], "no avx512/amx")

    def test_missing_lscpu_drops_cpu_details_only(self):
        with patch("subprocess.check_output", side_effect=FileNotFoundError("lscpu")):
            info = CPUEnv().get_info()

        self.assertNotIn("CPU Model", info)
        self.assertIn("Machine", info)


if __name__ == "__main__":
    unittest.main()
