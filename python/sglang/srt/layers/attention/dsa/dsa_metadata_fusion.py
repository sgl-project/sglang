"""Opt-in KPool metadata fusion and its validated geometry boundary."""

import logging
from functools import partial

from sglang.kernels.ops.attention.dsa_metadata import (
    fused_dsa_decode_metadata,
    fused_dsa_draft_extend_metadata,
    fused_dsa_target_verify_metadata,
)
from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda, is_hip

logger = logging.getLogger(__name__)


def kpool_metadata_fusion_supported(pool_size, page_size, topk):
    return (
        pool_size > 1
        and page_size == 64
        and page_size % pool_size == 0
        and topk % pool_size == 0
    )


class DSAKPoolMetadataFusionMixin:
    experimental_kpool_metadata_fusion = False

    def _init_kpool_metadata_fusion(self):
        requested = envs.SGLANG_EXPERIMENTAL_DSA_KPOOL_METADATA_FUSION.get()
        supported = kpool_metadata_fusion_supported(
            self.dsa_index_kpool, self.real_page_size, self.dsa_index_topk
        )
        self.experimental_kpool_metadata_fusion = (
            requested and supported and is_cuda() and not is_hip()
        )
        self._fused_decode_metadata = fused_dsa_decode_metadata
        self._fused_verify_metadata = fused_dsa_target_verify_metadata
        self._fused_draft_extend_metadata = fused_dsa_draft_extend_metadata
        if self.experimental_kpool_metadata_fusion:
            from sglang.kernels.ops.attention.dsa_kpool_metadata.decode import (
                fused_dsa_decode_metadata as decode,
            )
            from sglang.kernels.ops.attention.dsa_kpool_metadata.draft_extend import (
                fused_dsa_draft_extend_metadata as draft_extend,
            )
            from sglang.kernels.ops.attention.dsa_kpool_metadata.verify import (
                fused_dsa_target_verify_metadata as verify,
            )

            self._fused_decode_metadata = partial(
                decode, index_kpool=self.dsa_index_kpool
            )
            self._fused_verify_metadata = partial(
                verify, index_kpool=self.dsa_index_kpool
            )
            self._fused_draft_extend_metadata = partial(
                draft_extend, index_kpool=self.dsa_index_kpool
            )
            logger.info(
                "DSA KPool metadata fusion enabled (pool=%d)", self.dsa_index_kpool
            )
        elif requested and self.dsa_index_kpool > 1:
            logger.warning(
                "DSA KPool metadata fusion unsupported for this platform/geometry; retaining ordinary metadata"
            )
