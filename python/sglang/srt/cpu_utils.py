from __future__ import annotations

import subprocess
from collections import defaultdict


def parse_lscpu_topology():
    try:
        # Get CPU topology: CPU,Core,Socket,Node
        output = subprocess.check_output(
            ["lscpu", "-p=CPU,Core,Socket,Node"], text=True
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error running 'lscpu': {e}")

    # Parse only data lines (skip comments)
    cpu_info = []
    for line in output.splitlines():
        if not line.startswith("#"):
            cpu, core, socket, node = map(int, line.strip().split(","))
            cpu_info.append((cpu, core, socket, node))

    # [(0,0,0,0),(1,1,0,0),...,(43,43,0,1),...,(256,0,0,0),...]
    return cpu_info


def get_physical_cpus_by_numa():
    cpu_info = parse_lscpu_topology()

    # Map NUMA node -> set of (core_id, socket) to avoid duplicates
    # 0: {(0,0): 0, (1, 0): 1,...}
    # ...
    # 5: {(214,1): 214, (215,1): 215}
    physical_by_node = defaultdict(dict)  # node -> core_id -> cpu_id

    for cpu, core, socket, node in cpu_info:
        key = (core, socket)
        if key not in physical_by_node[node]:
            physical_by_node[node][
                key
            ] = cpu  # pick first CPU seen for that physical core

    # Convert to list of physical CPUs per node
    # 0: [0,1,2,...,42]
    # ...
    # 2: [86,87,...,127]
    # ...
    # 5: [214,215,...,255]
    node_to_cpus = {}
    for node, core_to_cpu in physical_by_node.items():
        cpus = sorted(core_to_cpu.values())
        node_to_cpus[node] = cpus

    return node_to_cpus


# Compress sorted list of integers into range strings like 0-2,3,4-6
def compress_ranges(cpu_list):
    if not cpu_list:
        return ""
    ranges = []
    start = prev = cpu_list[0]
    for cpu in cpu_list[1:]:
        if cpu == prev + 1:
            prev = cpu
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = cpu
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


# Only physical cores are used. Logical cores are excluded.
def get_cpu_ids_by_node():
    node_to_cpus = get_physical_cpus_by_numa()
    # Sort by NUMA node index
    cpu_ids = [
        compress_ranges(sorted(node_to_cpus[node])) for node in sorted(node_to_cpus)
    ]
    # ['0-42', '43-85', '86-127', '128-170', '171-213', '214-255']
    return cpu_ids
