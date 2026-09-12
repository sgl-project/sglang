"""Which layer ids the request pool holds mamba (conv + SSM) state for.

Pins ``resolve_req_pool_mamba_layer_ids`` (``mem_cache/mamba_layer_ids.py``),
the torch-free decision behind ``KVCacheConfigurator._get_mamba_layer_ids_for_req_pool``:

* GLM-5.3-Flash (45 decoder layers, 34 KDA + 11 DSA, one DSA NextN draft layer
  at id 45): the NextN id is NOT appended, so MambaPool / MambaPoolHost size
  their layer axis to 34, not 35.
* A config without a layer classifier (``is_kda_layer``) keeps the upstream
  behaviour: with speculative decoding on, every NextN id is appended.
* A config whose classifier calls the NextN layer linear keeps the append.
* Speculative decoding off never appends; the pipeline ``[start, end)`` slice
  applies to the model's mamba layers as before.

Runs without torch: the module under test is loaded by file path (``import
sglang`` pulls torch), as ``unit/tools/test_docker_build_metadata_args.py`` does.

    python test/registered/unit/mem_cache/test_mamba_layer_ids.py
"""

import ast
import importlib.util
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
CI_REGISTER_PATH = REPO_ROOT / "python" / "sglang" / "test" / "ci" / "ci_register.py"
HELPER_PATH = (
    REPO_ROOT / "python" / "sglang" / "srt" / "mem_cache" / "mamba_layer_ids.py"
)
CONFIGURATOR_PATH = (
    REPO_ROOT / "python" / "sglang" / "srt" / "mem_cache" / "kv_cache_configurator.py"
)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


register_cpu_ci = _load_module("ci_register", CI_REGISTER_PATH).register_cpu_ci
register_cpu_ci(est_time=1, suite="base-a-test-cpu")

resolve_req_pool_mamba_layer_ids = _load_module(
    "mamba_layer_ids", HELPER_PATH
).resolve_req_pool_mamba_layer_ids


# GLM-5.3-Flash text config shape (configs/glm5_next.py): 45 decoder layers,
# every layer with ``i % 4 != 3`` is KDA (34 layers), the rest DSA (11 layers);
# ``num_nextn_predict_layers=1`` puts the draft block at id num_hidden_layers.
GLM_NUM_HIDDEN_LAYERS = 45
GLM_KDA_LAYERS = [i for i in range(GLM_NUM_HIDDEN_LAYERS) if i % 4 != 3]
GLM_NEXTN_LAYER_IDS = [GLM_NUM_HIDDEN_LAYERS]


def _glm_is_kda_layer(layer_idx: int) -> bool:
    # Mirrors Glm5NextTextConfig.is_kda_layer: membership in
    # linear_attn_config["kda_layers"], a subset of range(num_hidden_layers).
    return layer_idx in GLM_KDA_LAYERS


class TestResolveReqPoolMambaLayerIds(unittest.TestCase):
    def test_glm53_flash_shape_drops_dsa_nextn_layer(self):
        self.assertEqual(len(GLM_KDA_LAYERS), 34)
        got = resolve_req_pool_mamba_layer_ids(
            mamba_layers=GLM_KDA_LAYERS,
            start_layer=0,
            end_layer=GLM_NUM_HIDDEN_LAYERS,
            nextn_layer_ids=GLM_NEXTN_LAYER_IDS,
            is_linear_layer=_glm_is_kda_layer,
            spec_enabled=True,
        )
        self.assertEqual(got, GLM_KDA_LAYERS)
        self.assertNotIn(GLM_NUM_HIDDEN_LAYERS, got)
        self.assertEqual(len(got), 34)

    def test_glm53_flash_spec_off_identical_to_spec_on(self):
        spec_on = resolve_req_pool_mamba_layer_ids(
            GLM_KDA_LAYERS,
            0,
            GLM_NUM_HIDDEN_LAYERS,
            GLM_NEXTN_LAYER_IDS,
            _glm_is_kda_layer,
            spec_enabled=True,
        )
        spec_off = resolve_req_pool_mamba_layer_ids(
            GLM_KDA_LAYERS,
            0,
            GLM_NUM_HIDDEN_LAYERS,
            GLM_NEXTN_LAYER_IDS,
            _glm_is_kda_layer,
            spec_enabled=False,
        )
        self.assertEqual(spec_on, spec_off)
        self.assertEqual(spec_off, GLM_KDA_LAYERS)

    def test_config_without_classifier_keeps_upstream_append(self):
        # Qwen3-Next style: mamba layers, NextN ids, no is_kda_layer.
        mamba_layers = [0, 1, 2, 4, 5, 6]
        got = resolve_req_pool_mamba_layer_ids(
            mamba_layers=mamba_layers,
            start_layer=0,
            end_layer=8,
            nextn_layer_ids=[8],
            is_linear_layer=None,
            spec_enabled=True,
        )
        self.assertEqual(got, [0, 1, 2, 4, 5, 6, 8])

    def test_config_without_classifier_spec_off_does_not_append(self):
        got = resolve_req_pool_mamba_layer_ids(
            mamba_layers=[0, 1, 2],
            start_layer=0,
            end_layer=4,
            nextn_layer_ids=[4],
            is_linear_layer=None,
            spec_enabled=False,
        )
        self.assertEqual(got, [0, 1, 2])

    def test_linear_nextn_layer_is_appended(self):
        # A config whose classifier calls the draft layer linear keeps its slot.
        got = resolve_req_pool_mamba_layer_ids(
            mamba_layers=[0, 1, 2],
            start_layer=0,
            end_layer=4,
            nextn_layer_ids=[4],
            is_linear_layer=lambda i: i in (0, 1, 2, 4),
            spec_enabled=True,
        )
        self.assertEqual(got, [0, 1, 2, 4])

    def test_mixed_nextn_layers_keep_only_linear_ones(self):
        got = resolve_req_pool_mamba_layer_ids(
            mamba_layers=[0, 1],
            start_layer=0,
            end_layer=2,
            nextn_layer_ids=[2, 3],
            is_linear_layer=lambda i: i in (0, 1, 3),
            spec_enabled=True,
        )
        self.assertEqual(got, [0, 1, 3])

    def test_nextn_id_already_present_is_not_duplicated(self):
        got = resolve_req_pool_mamba_layer_ids(
            mamba_layers=[0, 1, 4],
            start_layer=0,
            end_layer=8,
            nextn_layer_ids=[4],
            is_linear_layer=None,
            spec_enabled=True,
        )
        self.assertEqual(got, [0, 1, 4])

    def test_pipeline_slice_filters_model_layers_only(self):
        # PP rank owning [16, 32): only its mamba layers survive; the NextN id
        # is appended irrespective of the slice, as upstream does.
        mamba_layers = [i for i in range(48) if i % 4 != 3]
        got = resolve_req_pool_mamba_layer_ids(
            mamba_layers=mamba_layers,
            start_layer=16,
            end_layer=32,
            nextn_layer_ids=[48],
            is_linear_layer=None,
            spec_enabled=True,
        )
        self.assertEqual(got, [i for i in range(16, 32) if i % 4 != 3] + [48])

        got_glm = resolve_req_pool_mamba_layer_ids(
            mamba_layers=mamba_layers,
            start_layer=16,
            end_layer=32,
            nextn_layer_ids=[48],
            is_linear_layer=lambda i: i in mamba_layers,
            spec_enabled=True,
        )
        self.assertEqual(got_glm, [i for i in range(16, 32) if i % 4 != 3])

    def test_returns_a_fresh_list(self):
        mamba_layers = [0, 1, 2]
        got = resolve_req_pool_mamba_layer_ids(
            mamba_layers, 0, 4, [4], None, spec_enabled=True
        )
        self.assertEqual(mamba_layers, [0, 1, 2])
        self.assertIsNot(got, mamba_layers)


class TestConfiguratorWiring(unittest.TestCase):
    """The configurator routes through the helper with the config's classifier.

    Source-level pin (the configurator imports torch): keeps a rebase from
    silently restoring the unconditional append.
    """

    @classmethod
    def setUpClass(cls):
        tree = ast.parse(CONFIGURATOR_PATH.read_text(encoding="utf-8"))
        cls.method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_get_mamba_layer_ids_for_req_pool"
        )
        cls.calls = [
            node
            for node in ast.walk(cls.method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_req_pool_mamba_layer_ids"
        ]

    def test_method_calls_helper_once(self):
        self.assertEqual(len(self.calls), 1)

    def test_helper_receives_config_classifier_and_nextn_ids(self):
        kwargs = {kw.arg: ast.unparse(kw.value) for kw in self.calls[0].keywords}
        self.assertEqual(
            set(kwargs),
            {
                "mamba_layers",
                "start_layer",
                "end_layer",
                "nextn_layer_ids",
                "is_linear_layer",
                "spec_enabled",
            },
        )
        self.assertEqual(
            kwargs["nextn_layer_ids"], "getattr(cfg, 'nextn_layer_ids', [])"
        )
        self.assertEqual(
            kwargs["is_linear_layer"], "getattr(cfg, 'is_kda_layer', None)"
        )
        self.assertEqual(kwargs["mamba_layers"], "cfg.mamba2_cache_params.layers")
        self.assertIn("max_speculative_num_draft_tokens()", kwargs["spec_enabled"])


if __name__ == "__main__":
    unittest.main()
