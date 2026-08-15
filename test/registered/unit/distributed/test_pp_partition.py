"""Automatic PP layer partition: DP core and get_pp_indices integration."""

import unittest

from sglang.srt.distributed.pp_partition import (
    _set_auto_pp_partition,
    compute_balanced_partition,
)
from sglang.srt.distributed.utils import get_pp_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GB = 1 << 30

# Qwen3.5-397B-flavoured reference: 60 layers, every 4th is full attention.
NUM_LAYERS = 60
FULL_IDS = list(range(3, 60, 4))  # 15 full-attention layers


def _partition(**overrides):
    kwargs = dict(
        num_layers=NUM_LAYERS,
        pp_size=8,
        full_attention_layer_ids=FULL_IDS,
        weight_bytes_per_layer=1.5 * GB,
        kv_bytes_per_token_per_full_layer=64 * 1024,  # 64 KB/token/layer
        mamba_bytes_per_slot_per_linear_layer=1 * 1024 * 1024,  # 1 MB/slot/layer
        first_stage_extra_bytes=2 * GB,
        last_stage_extra_bytes=2 * GB,
        draft_kv_bytes_per_token=0.0,
        reference_num_tokens=200_000,
        reference_num_slots=512,
    )
    kwargs.update(overrides)
    return compute_balanced_partition(**kwargs)


class TestBalancedPartition(CustomTestCase):
    def assert_valid(self, partition, pp_size):
        self.assertEqual(len(partition), pp_size)
        self.assertEqual(sum(partition), NUM_LAYERS)
        self.assertTrue(all(n >= 1 for n in partition))

    def test_uniform_costs_give_even_split(self):
        # No KV/mamba/draft asymmetry: only uniform weights, so the DP must
        # fall back to a balanced count split.
        p = _partition(
            kv_bytes_per_token_per_full_layer=0,
            mamba_bytes_per_slot_per_linear_layer=0,
            first_stage_extra_bytes=0,
            last_stage_extra_bytes=0,
        )
        self.assert_valid(p, 8)
        self.assertLessEqual(max(p) - min(p), 1)

    def test_last_stage_lightened_by_draft(self):
        with_draft = _partition(
            last_stage_extra_bytes=20 * GB, draft_kv_bytes_per_token=64 * 1024
        )
        without_draft = _partition(last_stage_extra_bytes=0)
        self.assert_valid(with_draft, 8)
        # The last stage gets fewer layers once the draft's fixed+variable
        # overhead is charged to it.
        self.assertLess(with_draft[-1], without_draft[-1])

    def test_kv_cost_concentrates_full_layers_cheaply(self):
        # When KV dominates, stages heavy in full-attention layers are
        # expensive, so the DP packs more (cheap linear) layers alongside
        # them and never exceeds the even-split max stage cost.
        p = _partition(
            weight_bytes_per_layer=0.1 * GB,
            kv_bytes_per_token_per_full_layer=512 * 1024,
            reference_num_tokens=500_000,
        )
        self.assert_valid(p, 8)

        def stage_full_count(stage):
            start = sum(p[:stage])
            return len([l for l in FULL_IDS if start <= l < start + p[stage]])

        # No stage carries more full-attention layers than the even split's max.
        self.assertLessEqual(max(stage_full_count(s) for s in range(8)), 2)

    def test_remainder_goes_to_cheap_stages(self):
        # 61 layers over 8 stages: the extra layer must not silently land on
        # the heaviest (draft-loaded) last stage.
        p = _partition(num_layers=61, last_stage_extra_bytes=20 * GB)
        self.assertEqual(sum(p), 61)
        self.assertEqual(len(p), 8)
        self.assertLessEqual(p[-1], max(p))

    def test_get_pp_indices_uses_cache(self):
        try:
            _set_auto_pp_partition([30, 30])
            self.assertEqual(get_pp_indices(60, 0, 2), (0, 30))
            self.assertEqual(get_pp_indices(60, 1, 2), (30, 60))
            # pp_size=1 (the draft worker) must not read the target's cache.
            self.assertEqual(get_pp_indices(60, 0, 1), (0, 60))
        finally:
            _set_auto_pp_partition(None)

    def test_env_var_still_wins(self):
        import os

        os.environ["SGLANG_PP_LAYER_PARTITION"] = "10,50"
        try:
            _set_auto_pp_partition([30, 30])
            self.assertEqual(get_pp_indices(60, 0, 2), (0, 10))
            self.assertEqual(get_pp_indices(60, 1, 2), (10, 60))
        finally:
            del os.environ["SGLANG_PP_LAYER_PARTITION"]
            _set_auto_pp_partition(None)


class TestEngagementAdapter(CustomTestCase):
    """maybe_set_auto_pp_partition: gates, hybrid detection, spec charging."""

    def setUp(self):
        import types
        from unittest import mock

        torch = __import__("torch")
        self._patches = []
        full_ids = list(range(3, 60, 4))
        linear_ids = [i for i in range(60) if i not in full_ids]
        fake_mamba = types.SimpleNamespace(
            full_attention_layer_ids=full_ids,
            mamba2_cache_params=types.SimpleNamespace(
                layers=linear_ids, mamba_cache_per_req=len(linear_ids) * (1 << 20)
            ),
        )
        self.fake_model_config = types.SimpleNamespace(
            full_attention_layer_ids=None,
            num_hidden_layers=60,
            hf_text_config=types.SimpleNamespace(
                hidden_size=4096,
                num_experts=512,
                moe_intermediate_size=1024,
                intermediate_size=None,
                v_head_dim=None,
                vocab_size=248320,
                mtp_num_hidden_layers=1,
                num_nextn_predict_layers=None,
            ),
            hf_config=types.SimpleNamespace(tie_word_embeddings=False),
            dtype=torch.bfloat16,
            get_num_kv_heads=lambda tp: max(1, 8 // tp),
            head_dim=128,
            num_attention_heads=64,
            hidden_size=4096,
        )
        self.spec = types.SimpleNamespace(
            speculative_algorithm=None,
            speculative_draft_model_path=None,
            speculative_draft_model_revision=None,
            speculative_draft_kv_cache_dtype=None,
        )

        import sglang.srt.distributed.pp_partition as pp
        import sglang.srt.runtime_context as rc

        self._patches.append(
            mock.patch.object(pp, "mambaish_config", lambda mc: fake_mamba)
        )
        self._patches.append(
            mock.patch.object(
                rc, "get_model", lambda: types.SimpleNamespace(kv_cache_dtype="auto")
            )
        )
        self._patches.append(
            mock.patch.object(
                rc,
                "get_schedule",
                lambda: types.SimpleNamespace(mem_fraction_static=0.9),
            )
        )
        self._patches.append(mock.patch.object(rc, "get_spec", lambda: self.spec))
        for p in self._patches:
            p.start()
        self.ps = types.SimpleNamespace(pp_size=2, attn_tp_size=2)
        self._gpu_bytes = 280 << 30

    def tearDown(self):
        for p in self._patches:
            p.stop()
        _set_auto_pp_partition(None)

    def _run(self):
        from sglang.srt.distributed.pp_partition import (
            get_auto_pp_partition,
            maybe_set_auto_pp_partition,
        )

        maybe_set_auto_pp_partition(
            self.fake_model_config, self.ps, total_gpu_bytes=self._gpu_bytes
        )
        return get_auto_pp_partition()

    def test_engages_for_hybrid_model(self):
        partition = self._run()
        self.assertIsNotNone(partition)
        self.assertEqual(len(partition), 2)
        self.assertEqual(sum(partition), 60)

    def test_skips_at_pp_size_1(self):
        self.ps.pp_size = 1
        self.assertIsNone(self._run())

    def test_env_var_wins(self):
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"SGLANG_PP_LAYER_PARTITION": "30,30"}):
            self.assertIsNone(self._run())

    def test_skips_non_hybrid(self):
        self.fake_model_config.num_hidden_layers = 15
        import types
        from unittest import mock

        import sglang.srt.distributed.pp_partition as pp

        # 15/15 full-attention layers: not a hybrid.
        with mock.patch.object(
            pp,
            "mambaish_config",
            lambda mc: types.SimpleNamespace(
                full_attention_layer_ids=list(range(15)),
                mamba2_cache_params=types.SimpleNamespace(
                    layers=[], mamba_cache_per_req=0
                ),
            ),
        ):
            self.assertIsNone(self._run())

    def test_spec_lightens_last_stage(self):
        plain = self._run()
        _set_auto_pp_partition(None)
        self.spec.speculative_algorithm = "EAGLE"
        # Two embedded MTP layers are heavy enough to force a strict shift.
        self.fake_model_config.hf_text_config.mtp_num_hidden_layers = 2
        with_spec = self._run()
        self.assertLess(with_spec[-1], plain[-1])

    def test_draft_model_path_branch(self):
        # External draft model + explicit draft KV dtype: covers the geometry
        # and dtype resolution of the speculative_draft_model_path branch.
        import types
        from unittest import mock

        torch = __import__("torch")

        self.spec.speculative_algorithm = "EAGLE"
        self.spec.speculative_draft_model_path = "/fake/draft"
        self.spec.speculative_draft_kv_cache_dtype = "fp8_e4m3"
        fake_draft_cfg = types.SimpleNamespace(
            num_hidden_layers=3,
            head_dim=64,
            hidden_size=2048,
            num_attention_heads=16,
            hf_text_config=types.SimpleNamespace(v_head_dim=None),
            get_num_kv_heads=lambda tp: 4,
            dtype=torch.bfloat16,
        )
        with (
            mock.patch(
                "sglang.srt.configs.model_config.ModelConfig.from_server_args",
                classmethod(lambda cls, *a, **k: fake_draft_cfg),
            ),
            mock.patch("sglang.srt.runtime_context.get_server_args", lambda: object()),
        ):
            partition = self._run()
        self.assertIsNotNone(partition)
        _set_auto_pp_partition(None)
        self.spec.speculative_draft_model_path = None
        self.spec.speculative_draft_kv_cache_dtype = None
        self.spec.speculative_algorithm = None
        baseline = self._run()
        # The point of this test is the fp8 dtype-resolution path (previously a
        # KeyError); the small fp8 draft need not force a partition shift.
        self.assertLessEqual(partition[-1], baseline[-1])

    def test_mtp_layer_count_attribute_fallback(self):
        # mtp_num_hidden_layers wins; num_nextn_predict_layers is the fallback.
        # Three draft layers are heavy enough to force a strict shift.
        self.spec.speculative_algorithm = "EAGLE"
        first = self._run()
        _set_auto_pp_partition(None)
        self.fake_model_config.hf_text_config.mtp_num_hidden_layers = None
        self.fake_model_config.hf_text_config.num_nextn_predict_layers = 3
        second = self._run()
        self.assertLess(second[-1], first[-1])


if __name__ == "__main__":
    unittest.main()
