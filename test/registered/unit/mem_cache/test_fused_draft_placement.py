# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""`place_fused_draft` and the target's fused-draft decision: which drafts
fuse into the host's entries, and where each runner's layers land.

A decline keeps a draft on a private pool (or refuses it at boot); admitting
one that cannot read the fused rows correctly gives silently wrong drafts.
Pinned:
  - a replicated head under multi-layer EAGLE gets one lane RANGE per runner
    (one shared region would let the runners clobber each other's KV);
  - a per-depth head serves one depth per runner and needs one runner per
    depth;
  - a draft with SWA or recurrent-state layers of its own declines, and so do
    asymmetric K/V rows unless every resolved backend, the draft's included,
    carries v_head_dim through to the kernel;
  - so does a draft whose attention backend is off the translated MHA rails
    (it would read the fused rows with virtual ids), a draft under
    --dcp-size > 1 (each rank's rows hold only its share of the tokens), an
    explicit draft KV dtype unlike the host's, and a draft under HiCache or
    host-pool decode retraction (they build the draft's host pool off a
    device pool of its own).

    python -m pytest test/registered/unit/mem_cache/test_fused_draft_placement.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.kv_cache_hook import ASYMMETRIC_KV_BACKENDS
from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache.layout.fused_draft import (
    DenseDraftRegion,
    DraftKVGeometry,
    DraftKVProfile,
    place_fused_draft,
)
from sglang.srt.runtime_context import get_parallel, override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_DTYPE = torch.bfloat16


def _profile(
    *,
    num_layers=1,
    swa_layer_ids=(),
    num_depths=1,
    num_state_layers=0,
    head_dim=64,
    v_head_dim=64,
):
    return DraftKVProfile(
        num_layers=num_layers,
        full=DraftKVGeometry(head_num=4, head_dim=head_dim, v_head_dim=v_head_dim),
        swa_layer_ids=swa_layer_ids,
        num_depths=num_depths,
        num_state_layers=num_state_layers,
    )


def _place(profile, num_runners=1, asymmetric_rows_ok=False):
    return place_fused_draft(
        profile=profile,
        num_runners=num_runners,
        store_dtype=_DTYPE,
        asymmetric_rows_ok=asymmetric_rows_ok,
    )


class TestPlaceFusedDraft(CustomTestCase):
    def test_replicated_head_gets_one_slot_range_per_runner(self):
        decision = _place(_profile(num_layers=2), num_runners=3)
        placement = decision.placement
        self.assertIsNotNone(placement, decision.declined)
        self.assertEqual(placement.region.lane_num, 6)
        for r in range(3):
            self.assertEqual(placement.lanes_for(r), range(2 * r, 2 * r + 2))

    def test_per_depth_head_serves_one_depth_per_runner(self):
        decision = _place(_profile(num_layers=8, num_depths=8), num_runners=2)
        placement = decision.placement
        self.assertIsNotNone(placement, decision.declined)
        self.assertEqual(placement.region.lane_num, 2)
        self.assertEqual(placement.lanes_for(1), range(1, 2))

    def test_per_depth_head_needs_one_runner_per_depth(self):
        self.assertIsNone(_place(_profile(num_layers=8, num_depths=8), 1).placement)
        self.assertIsNone(_place(_profile(num_layers=8, num_depths=8), 9).placement)

    def test_swa_state_asymmetric_and_misaligned_drafts_decline(self):
        for profile in (
            _profile(swa_layer_ids=(0,)),
            _profile(num_state_layers=1),
            _profile(head_dim=64, v_head_dim=32),
            # 4 heads x 5 dims x 2 B = 40 B: not a 16-B-aligned entry part.
            _profile(head_dim=5, v_head_dim=5),
        ):
            decision = _place(profile)
            self.assertIsNone(decision.placement)
            self.assertIsNotNone(decision.declined)

    def test_asymmetric_rows_fuse_with_their_v_width_when_backends_allow(self):
        placement = _place(
            _profile(head_dim=64, v_head_dim=32), asymmetric_rows_ok=True
        ).placement
        self.assertIsNotNone(placement)
        self.assertEqual(placement.region.resolved_v_head_dim(), 32)
        self.assertEqual(placement.region.entry_bytes(), 4 * (64 + 32) * 2)
        # Symmetric rows never consult the rule.
        self.assertIsNotNone(_place(_profile(), asymmetric_rows_ok=False).placement)

    def test_a_misaligned_v_row_declines_even_when_backends_allow(self):
        # K rows are 4 x 64 x 2 B = 512 B; V rows 4 x 5 x 2 B = 40 B are not a
        # 16-B-aligned entry part.
        decision = _place(_profile(head_dim=64, v_head_dim=5), asymmetric_rows_ok=True)
        self.assertIsNone(decision.placement)
        self.assertIn("40 B", decision.declined)

    def test_region_carries_the_profile_geometry(self):
        placement = _place(_profile()).placement
        self.assertEqual(
            placement.region,
            DenseDraftRegion(
                lane_num=1, head_num=4, head_dim=64, v_head_dim=64, store_dtype=_DTYPE
            ),
        )


class TestAsymmetricBackendRule(CustomTestCase):
    """The rule reads the TARGET's resolved backends and the draft's explicit
    one (unset inherits the target's); one shared-head_dim backend anywhere
    disqualifies asymmetric rows."""

    def _carries(self, *, backends, draft_backend=None):
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        with (
            patch.object(kcc, "attention_backends", return_value=tuple(backends)),
            patch.object(
                kcc,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_draft_attention_backend=draft_backend
                ),
            ),
        ):
            return cfg._draft_backends_carry_v_head_dim()

    def test_every_split_stride_backend_qualifies(self):
        for backend in sorted(ASYMMETRIC_KV_BACKENDS):
            self.assertTrue(self._carries(backends=(backend,)), backend)

    def test_a_shared_head_dim_backend_anywhere_disqualifies(self):
        self.assertNotIn("flashinfer", ASYMMETRIC_KV_BACKENDS)
        self.assertFalse(self._carries(backends=("flashinfer",)))
        self.assertFalse(
            self._carries(backends=("triton",), draft_backend="flashinfer")
        )


class TestUnifiedSWAHeadGeometry(CustomTestCase):
    """BUG REGRESSION. The hybrid-SWA factories hard-coded a symmetric shape
    on GPU hosts, so an asymmetric model (MiMo-V2-Flash, 192/128) allocated
    192-wide V rows in both sub-pools while the boot solve priced 128."""

    def _configurator(self):
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        cfg.is_hybrid_swa = True
        cfg.is_hybrid_swa_compress = False
        cfg.use_mla_backend = False
        cfg.kv_cache_dtype = _DTYPE
        cfg.layer_info = SimpleNamespace(full_attention_layer_ids=[0, 3])
        cfg.model_config = SimpleNamespace(
            get_num_kv_heads=lambda tp, dcp: 4,
            head_dim=192,
            v_head_dim=128,
            get_swa_num_kv_heads=lambda tp: 8,
            swa_head_dim=192,
            swa_v_head_dim=128,
        )
        return cfg

    def test_both_sub_pools_take_the_model_config_geometry(self):
        with get_parallel().override(attn_tp_size=1, attn_dcp_size=1):
            g = self._configurator()._unified_swa_head_geometry()
        self.assertEqual((g.head_num, g.head_dim, g.v_head_dim), (4, 192, 128))
        self.assertEqual(
            (g.swa_head_num, g.swa_head_dim, g.swa_v_head_dim), (8, 192, 128)
        )

    def test_the_fused_price_reads_the_same_geometry(self):
        region = DenseDraftRegion(
            lane_num=1, head_num=4, head_dim=64, store_dtype=_DTYPE
        )
        with get_parallel().override(attn_tp_size=1, attn_dcp_size=1):
            spec = self._configurator()._full_host_spec(region)
        self.assertEqual(
            (spec.layer_num, spec.head_dim, spec.v_head_dim), (2, 192, 128)
        )


class TestFusedDraftDecision(CustomTestCase):
    """The target's boot decision over a host whose full sub-pool fuses."""

    def _decide(
        self,
        *,
        algorithm="EAGLE",
        draft_kv_dtype=None,
        host_kv_dtype=_DTYPE,
        kv_cache_dtype_flag="auto",
        draft_backend=None,
        target_backends=("triton", "triton"),
        dcp_size=1,
        hicache=False,
        retraction_backup=None,
    ):
        from sglang.srt.configs.model_config import AttentionArch
        from sglang.srt.mem_cache import kv_cache_configurator as kvc
        from sglang.srt.speculative import draft_worker_common

        cfg = kvc.KVCacheConfigurator.__new__(kvc.KVCacheConfigurator)
        cfg.is_hybrid_swa = True
        cfg.mambaish_config = None
        cfg.use_mla_backend = False
        cfg.is_draft_worker = False
        cfg.spec_algorithm = SimpleNamespace(
            is_eagle=lambda: algorithm == "EAGLE",
            is_dflash_family=lambda: algorithm in ("DFLASH", "DSPARK"),
        )
        cfg.model_config = SimpleNamespace(is_multi_layer_eagle=False)
        cfg.kv_cache_dtype = host_kv_dtype
        cfg.spec_aux_config = SimpleNamespace(
            eagle_draft_num_layers=1,
            draft_kv_num_layers=1,
            dflash_draft_num_layers=1,
            draft_model_config=SimpleNamespace(
                is_hybrid_swa=False,
                is_deepseek_v4_arch=False,
                num_nextn_predict_layers=None,
                get_num_kv_heads=lambda tp: 4,
                head_dim=64,
                v_head_dim=64,
                dtype=torch.bfloat16,
                attention_arch=AttentionArch.MHA,
            ),
        )
        memory = SimpleNamespace(
            enable_unified_memory=True, enable_hierarchical_cache=hicache
        )
        spec = SimpleNamespace(
            speculative_num_steps=1,
            speculative_draft_kv_cache_dtype=draft_kv_dtype,
            speculative_draft_attention_backend=draft_backend,
        )
        with (
            patch.object(kvc, "get_memory", return_value=memory),
            patch.object(
                kvc,
                "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_retraction_backup=retraction_backup
                ),
            ),
            patch.object(kvc, "get_spec", return_value=spec),
            patch.object(
                kvc,
                "get_model",
                return_value=SimpleNamespace(kv_cache_dtype=kv_cache_dtype_flag),
            ),
            patch.object(kvc, "attention_backends", return_value=target_backends),
            patch.object(draft_worker_common, "get_spec", return_value=spec),
            patch.object(
                draft_worker_common, "attention_backends", return_value=target_backends
            ),
            patch("sglang.srt.configs.hybrid_arch.mambaish_config", return_value=None),
            get_parallel().override(attn_tp_size=1, attn_dcp_size=dcp_size),
            override_platform(is_xpu=False, is_hip=False),
        ):
            return cfg._fused_draft_decision()

    def test_a_stateless_draft_is_placed(self):
        self.assertIsNotNone(self._decide().placement)

    def test_a_draft_kv_dtype_unlike_the_host_keeps_the_private_pool(self):
        """A fused draft stores its rows in the host's KV dtype; an explicit
        draft dtype that differs declines instead of mis-typing those rows,
        and one that matches the host's fuses."""
        self.assertIsNotNone(self._decide(draft_kv_dtype="auto").placement)
        declined = self._decide(draft_kv_dtype="fp8_e4m3")
        self.assertIsNone(declined.placement)
        self.assertIn("KV cache dtype", declined.declined)
        for algorithm in ("EAGLE", "DFLASH"):
            placed = self._decide(
                algorithm=algorithm,
                draft_kv_dtype="fp8_e4m3",
                host_kv_dtype=torch.float8_e4m3fn,
                kv_cache_dtype_flag="fp8_e4m3",
            )
            self.assertIsNotNone(placed.placement, algorithm)

    def test_a_draft_off_the_translated_rails_keeps_the_private_pool(self):
        """A fused draft reads its rows through the KV-index translator, which
        only triton, flashinfer, fa3 and trtllm_mha carry on every draft path.
        The draft's backend is the published one, a model hook's declaration
        included; otherwise it is what the draft runner inherits from the
        target."""
        declined = self._decide(algorithm="DSPARK", draft_backend="fa4")
        self.assertIsNone(declined.placement)
        self.assertIn("fa4", declined.declined)
        # A DFLASH-family draft runs the target's prefill backend, or the
        # platform default when that is not a draft backend.
        for target_backends, fuses in (
            (("fa4", "fa4"), False),
            (("fa3", "flashmla"), True),
            (("flashmla", "flashmla"), True),
        ):
            decision = self._decide(algorithm="DSPARK", target_backends=target_backends)
            self.assertEqual(decision.placement is not None, fuses, target_backends)
        # An EAGLE draft with no backend of its own runs the target's pair.
        self.assertIsNone(self._decide(target_backends=("fa3", "flashmla")).placement)

    def test_a_trtllm_mha_draft_fuses(self):
        """trtllm_mha refills its graph page table to what each replay reads
        and widens its eager table by the draft block, so its draft fuses."""
        # Kimi-Linear + DSPARK on SM100, where a model hook declares trtllm_mha.
        for algorithm in ("EAGLE", "DFLASH", "DSPARK"):
            decision = self._decide(algorithm=algorithm, draft_backend="trtllm_mha")
            self.assertIsNotNone(decision.placement, algorithm)
        # A draft with no backend of its own inherits the target's trtllm_mha.
        decision = self._decide(target_backends=("trtllm_mha", "trtllm_mha"))
        self.assertIsNotNone(decision.placement)

    def test_dcp_keeps_the_private_pool(self):
        """Under --dcp-size > 1 each rank's host rows hold only its share of
        the widened id space, while the draft, outside the DCP group, reads
        every token."""
        for algorithm in ("DSPARK", "EAGLE"):
            declined = self._decide(algorithm=algorithm, dcp_size=2)
            self.assertIsNone(declined.placement, algorithm)
            self.assertIn("--dcp-size 2", declined.declined)

    def test_hierarchical_cache_keeps_the_private_pool(self):
        """HiCache builds the draft's host pool off its own device pool, which
        a fused draft does not have; DSPARK + HiCache keeps working on the
        private pool."""
        declined = self._decide(algorithm="DSPARK", hicache=True)
        self.assertIsNone(declined.placement)
        self.assertIn("--enable-hierarchical-cache", declined.declined)

    def test_host_pool_retraction_keeps_the_private_pool(self):
        """Host-pool decode retraction builds the draft's host pool off its
        own device pool too; DSPARK with it keeps the private pool."""
        declined = self._decide(algorithm="DSPARK", retraction_backup="host_pool")
        self.assertIsNone(declined.placement)
        self.assertIn(
            "--disaggregation-decode-retraction-backup=host_pool", declined.declined
        )


if __name__ == "__main__":
    unittest.main()
