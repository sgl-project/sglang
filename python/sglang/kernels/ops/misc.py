"""Device probes that a launch configuration depends on.

Not an operator group -- these answer "what will the hardware actually schedule",
which a host-side dispatch needs before it can size a grid. Kept out of
``ops/__init__``'s eager group import for that reason; import it directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

__all__ = ["get_max_active_clusters"]


@cache_once
def _jit_probe_module() -> Module:
    return load_jit(
        "misc_probe",
        cuda_files=["misc/probe.cuh"],
        cuda_wrappers=[("get_max_active_clusters", "get_max_active_clusters")],
    )


@cache_once
def _get_max_active_clusters(cluster_size: int, occupancy: int) -> int:
    return int(_jit_probe_module().get_max_active_clusters(cluster_size, occupancy))


def get_max_active_clusters(cluster_size: int, occupancy: int) -> int:
    """Clusters of ``cluster_size`` blocks that can be resident at once.

    Asks the driver (``cudaOccupancyMaxActiveClusters``) rather than dividing SM
    count by cluster size: a cluster's blocks must be co-scheduled within one
    GPC, so the answer falls short of ``num_sms * occupancy / cluster_size`` once
    the cluster stops dividing a GPC evenly. On B200 (148 SMs) at occupancy 2 the
    driver reports 33 clusters of 8 where the division says 37, and 14 of 16
    where it says 18.

    Probed with an empty kernel pinned to ``occupancy`` blocks per SM, so the
    answer is the *scheduling* limit at that occupancy and nothing else. Pass the
    occupancy the real kernel reaches (the second ``__launch_bounds__``
    argument), not the one it asks for.

    :param cluster_size: Blocks per cluster.
    :param occupancy: Blocks per SM (``num_waves`` in ``csrc/misc/probe.cuh``).
    :raises RuntimeError: On pre-sm90 devices, which have no clusters.
    :raises ValueError: If nothing is schedulable, which a real device should
                        never report for a cluster width it supports.
    """
    result = _get_max_active_clusters(cluster_size, occupancy)
    if result == 0:
        raise ValueError(
            f"no cluster of {cluster_size} fits at occupancy {occupancy}; "
            "the cluster width is likely beyond what this device supports"
        )
    return result
