import sys

import pytest
import torch.nn as nn

from sglang.kernels.ops.diffusion.quality_gate import QualityGatedFusion
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_quality_gate_mounts_and_unmounts_all_sites():
    fusion = QualityGatedFusion(
        name="test fusion",
        marker_attr="_test_fusion_site",
        enabled_attr="_test_fusion_enabled",
    )
    root = nn.ModuleList([nn.Module(), nn.Module()])
    for index, site in enumerate(root):
        fusion.mark(site, index)

    assert [fusion.metadata(site) for site in root] == [0, 1]
    assert fusion.mount(root)
    assert all(fusion.is_enabled(site) for site in root)
    fusion.unmount(root)
    assert not any(fusion.is_enabled(site) for site in root)


def test_quality_gate_rejection_is_all_or_nothing():
    fusion = QualityGatedFusion(
        name="test fusion",
        marker_attr="_test_fusion_site",
        enabled_attr="_test_fusion_enabled",
    )
    root = nn.ModuleList([nn.Module(), nn.Module()])
    for index, site in enumerate(root):
        fusion.mark(site, index)

    assert not fusion.mount(
        root, reject_reason=lambda site: "rejected" if fusion.metadata(site) else None
    )
    assert not any(fusion.is_enabled(site) for site in root)
    assert not fusion.mount(nn.Module())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
