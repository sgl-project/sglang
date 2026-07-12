"""Unit tests for the compact DSA indexer K-cache layout.

Hermetic (no GPU, no server). Guards:

  1. ``dsa_layer_skips_topk`` layer-placement resolution for the three config
     forms (``index_topk_pattern`` > ``index_topk_freq``/
     ``index_skip_topk_offset`` > dense), including the GLM-5.2 shipped
     config (freq=4, offset=3 -> 21 full / 57 shared of 78).
  2. ``get_dsa_compact_indexer_layer_ids`` gating: dense models, draft
     workers, hisparse, hierarchical cache, and DSA cache layer split all
     keep the dense one-slot-per-layer layout (returning None).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.model_config import dsa_layer_skips_topk
from sglang.srt.model_executor.pool_configurator import (
    get_dsa_compact_indexer_layer_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GLM52_NUM_LAYERS = 78
# indexer_types from zai-org/GLM-5.2 config.json: full on 0,1,2 then every
# 4th layer starting at 6; the checkpoint carries indexer weights only there.
GLM52_FULL_LAYERS = [0, 1, 2] + list(range(6, GLM52_NUM_LAYERS, 4))


def _glm52_config():
    return SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=2048,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        index_topk_pattern=None,
    )


def _dense_dsa_config():
    # DeepSeek-V3.2 style: no freq/pattern fields -> indexer on every layer.
    return SimpleNamespace(
        architectures=["DeepseekV32ForCausalLM"],
        index_topk=2048,
    )


def _mock_runner(hf_config, num_layers, **overrides):
    mr = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config),
        start_layer=0,
        end_layer=num_layers,
        is_draft_worker=False,
        enable_hisparse=False,
        server_args=SimpleNamespace(enable_hierarchical_cache=False),
    )
    for k, v in overrides.items():
        setattr(mr, k, v)
    return mr


def _patch_dsa_env(compact=True):
    return patch(
        "sglang.srt.environ.envs.SGLANG_DSA_COMPACT_INDEXER_CACHE.get",
        return_value=compact,
    )


def _patch_no_layer_split():
    return patch(
        "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
        return_value=(None, None),
    )


class TestDsaLayerSkipsTopk(CustomTestCase):
    def test_glm52_freq_offset_matches_shipped_indexer_types(self):
        cfg = _glm52_config()
        full = [l for l in range(GLM52_NUM_LAYERS) if not dsa_layer_skips_topk(cfg, l)]
        self.assertEqual(full, GLM52_FULL_LAYERS)
        self.assertEqual(len(full), 21)

    def test_pattern_takes_precedence_over_freq(self):
        cfg = _glm52_config()
        cfg.index_topk_pattern = "FSFS"
        self.assertFalse(dsa_layer_skips_topk(cfg, 0))
        self.assertTrue(dsa_layer_skips_topk(cfg, 1))
        self.assertFalse(dsa_layer_skips_topk(cfg, 2))
        self.assertTrue(dsa_layer_skips_topk(cfg, 3))

    def test_dense_config_never_skips(self):
        cfg = _dense_dsa_config()
        self.assertFalse(
            any(dsa_layer_skips_topk(cfg, l) for l in range(GLM52_NUM_LAYERS))
        )


class TestCompactIndexerLayerIds(CustomTestCase):
    def test_glm52_compacts_to_full_layers(self):
        mr = _mock_runner(_glm52_config(), GLM52_NUM_LAYERS)
        with _patch_dsa_env(), _patch_no_layer_split():
            ids = get_dsa_compact_indexer_layer_ids(mr)
        self.assertEqual(ids, GLM52_FULL_LAYERS)

    def test_pp_range_is_respected(self):
        mr = _mock_runner(
            _glm52_config(), GLM52_NUM_LAYERS, start_layer=39, end_layer=78
        )
        with _patch_dsa_env(), _patch_no_layer_split():
            ids = get_dsa_compact_indexer_layer_ids(mr)
        self.assertEqual(ids, [l for l in GLM52_FULL_LAYERS if 39 <= l < 78])

    def test_dense_model_returns_none(self):
        mr = _mock_runner(_dense_dsa_config(), 61)
        with _patch_dsa_env(), _patch_no_layer_split():
            self.assertIsNone(get_dsa_compact_indexer_layer_ids(mr))

    def test_gates_return_none(self):
        cases = [
            dict(is_draft_worker=True),
            dict(enable_hisparse=True),
            dict(server_args=SimpleNamespace(enable_hierarchical_cache=True)),
        ]
        for overrides in cases:
            mr = _mock_runner(_glm52_config(), GLM52_NUM_LAYERS, **overrides)
            with _patch_dsa_env(), _patch_no_layer_split():
                self.assertIsNone(
                    get_dsa_compact_indexer_layer_ids(mr), msg=str(overrides)
                )

    def test_layer_split_returns_none(self):
        mr = _mock_runner(_glm52_config(), GLM52_NUM_LAYERS)
        with _patch_dsa_env(), patch(
            "sglang.srt.layers.cp.utils.get_glm_dsa_cp_layer_shard_info",
            return_value=(0, 2),
        ):
            self.assertIsNone(get_dsa_compact_indexer_layer_ids(mr))

    def test_env_escape_hatch(self):
        mr = _mock_runner(_glm52_config(), GLM52_NUM_LAYERS)
        with _patch_dsa_env(compact=False), _patch_no_layer_split():
            self.assertIsNone(get_dsa_compact_indexer_layer_ids(mr))


if __name__ == "__main__":
    unittest.main()
