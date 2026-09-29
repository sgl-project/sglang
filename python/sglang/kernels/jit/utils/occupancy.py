"""Occupancy probes a host-side dispatch needs before it can size a grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.kernels.jit.utils.common import cache_once
from sglang.kernels.jit.utils.compile import load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

__all__ = ["NoSchedulableClustersError", "get_max_active_clusters"]


class NoSchedulableClustersError(ValueError):
    """The occupancy query succeeded, but no cluster fits the requested shape."""


@cache_once
def _jit_probe_module() -> Module:
    return load_jit(
        "occupancy_cluster_probe",
        cuda_files=["occupancy/cluster_probe.cuh"],
        cuda_wrappers=[("get_max_active_clusters", "get_max_active_clusters")],
    )


@cache_once
def _get_max_active_clusters(cluster_size: int, occupancy: int) -> int:
    return int(_jit_probe_module().get_max_active_clusters(cluster_size, occupancy))


def get_max_active_clusters(cluster_size: int, occupancy: int) -> int:
    """Clusters of ``cluster_size`` blocks that can be resident at once.

    Asks the driver (``cudaOccupancyMaxActiveClusters``) rather than dividing SM
    count by cluster size: a cluster's blocks must be co-scheduled within one
    GPC, so the answer falls short of the division once the cluster stops
    dividing a GPC evenly. The probe kernel is pinned to ``occupancy`` blocks per
    SM, so pass the occupancy the real kernel reaches (its second
    ``__launch_bounds__`` argument). Raises ``RuntimeError`` before sm90, which
    has no clusters, and ``NoSchedulableClustersError`` (a ``ValueError``)
    when the query succeeds but nothing is schedulable. Other probe errors
    propagate to the caller.
    """
    result = _get_max_active_clusters(cluster_size, occupancy)
    if result == 0:
        raise NoSchedulableClustersError(
            f"no cluster of {cluster_size} fits at occupancy {occupancy}; "
            "the cluster width is likely beyond what this device supports"
        )
    return result
