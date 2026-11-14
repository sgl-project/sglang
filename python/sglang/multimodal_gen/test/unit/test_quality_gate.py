# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from torch import nn

from sglang.kernels.ops.diffusion.sites.quality_gate import QualityGatedFusion


def test_rejected_fusion_can_mount_after_weights_change():
    fusion = QualityGatedFusion(
        name="test fusion",
        marker_attr="_fusion_site",
        enabled_attr="_fusion_enabled",
        reject_reason=lambda site: "requires bias" if site.bias is None else None,
    )
    site = nn.Linear(2, 2, bias=False)
    root = nn.Sequential(site)
    fusion.mark(site)

    with patch("sglang.kernels.ops.diffusion.sites.quality_gate.logger") as logger:
        assert not fusion.mount(root)
        assert not fusion.mount(root)
        assert not fusion.is_enabled(site)
        logger.info.assert_called_once()

        site.bias = nn.Parameter(site.weight.new_zeros(2))
        assert fusion.mount(root)
        assert fusion.is_enabled(site)

        fusion.unmount(root)
        site.register_parameter("bias", None)
        assert not fusion.mount(root)
        assert not fusion.is_enabled(site)
        assert logger.info.call_count == 2
