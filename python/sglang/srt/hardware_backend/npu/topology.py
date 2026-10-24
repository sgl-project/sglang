"""Ascend runtime device mapping and npu-smi CPU locality discovery."""

import ctypes
import re
import subprocess
from functools import cached_property
from pathlib import Path

from sglang.srt.utils.cpu_affinity import parse_cpu_list


class NPUTopology:
    name = "NPU"

    @cached_property
    def _physical_device(self):
        # Ask the runtime instead of parsing ASCEND_RT_VISIBLE_DEVICES: container
        # logical IDs can themselves differ from host physical IDs.
        lib = ctypes.CDLL("libascendcl.so")
        try:
            fn = lib.aclrtGetPhyDevIdByUserDevId
        except AttributeError:
            # CANN documents that the legacy name also takes a *user* device ID.
            # Do not translate through aclrtGetLogicDevIdByUserDevId a second time.
            try:
                fn = lib.aclrtGetPhyDevIdByLogicDevId
            except AttributeError as exc:
                raise RuntimeError("CANN physical device query is unavailable") from exc
        fn.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        fn.restype = ctypes.c_int
        return fn

    @cached_property
    def _cpu_topology(self):
        try:
            output = subprocess.run(
                ["npu-smi", "info", "-t", "topo"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"npu-smi topology query failed: {exc}") from exc
        topology = {}
        for line in output.splitlines():
            match = re.fullmatch(r"\s*NPU(\d+)\s+.*?\s+([\d,-]+)\s*", line)
            if match:
                physical_id = int(match[1])
                if physical_id in topology:
                    raise ValueError(f"Duplicate NPU topology row: {physical_id}")
                topology[physical_id] = parse_cpu_list(match[2])
        if not topology:
            raise ValueError("npu-smi did not report NPU CPU affinity")
        return topology

    def device(self, device_id: int) -> tuple[int, set[int]]:
        physical_id = ctypes.c_int32(-1)
        rc = self._physical_device(device_id, ctypes.byref(physical_id))
        if rc != 0 or physical_id.value < 0:
            raise RuntimeError(
                f"CANN physical device query failed for {device_id}: {rc}"
            )
        cpus = self._cpu_topology.get(physical_id.value)
        if not cpus:
            raise ValueError(f"No CPU topology for physical NPU {physical_id.value}")
        return physical_id.value, cpus

    @staticmethod
    def numa_nodes(cpus: set[int]) -> list[int]:
        nodes = []
        for path in Path("/sys/devices/system/node").glob("node[0-9]*/cpulist"):
            if cpus & parse_cpu_list(path.read_text()):
                nodes.append(int(path.parent.name[4:]))
        return sorted(nodes)
