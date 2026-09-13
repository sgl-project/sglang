"""Exercise the opt-in QuaRot draft loader with a real, small BF16 checkpoint.

Only the DSpark activation predicate is patched to NPU. Tensor allocation,
checkpoint loading, FC and normalization run on CPU; this is not an NPU
arithmetic, full-model correctness or acceptance-rate test.
"""

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from test_dspark_weight_loading import _checkpoint

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.dspark_quarot import (
    GlmDSparkQuaRotConfig,
    glm_dspark_quarot_scope,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestDSparkNpuQuaRotLoading(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.group_dir = tempfile.TemporaryDirectory()
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{cls.group_dir.name}/gloo",
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")

    @classmethod
    def tearDownClass(cls):
        destroy_model_parallel()
        destroy_distributed_environment()
        cls.group_dir.cleanup()
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.enterContext(envs.SGLANG_RAGGED_VERIFY_MODE.override("static"))
        self.enterContext(
            get_context().override_server_args(
                speculative_algorithm="DSpark", device="cpu"
            )
        )
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.model_dir = self.root / "draft"
        self.model_dir.mkdir()
        self.config, self.weights = _checkpoint()
        self.config["transformer_layer_config"]["rms_norm_eps"] = 1e-5
        # Three different feature blocks prevent a block-order error from
        # disappearing behind repeated or constant input weights.
        values = (torch.arange(16 * 48).reshape(16, 48) * 7) % 61 - 30
        self.weights["fc.weight"] = (values.float() / 64).to(torch.bfloat16)
        self.weights["hidden_norm.weight"] = (torch.arange(16).float() / 16 + 0.5).to(
            torch.bfloat16
        )
        self.q = torch.zeros(16, 16, dtype=torch.float32)
        # Binary-exact orthogonal coefficients permit exact weight assertions
        # despite FP32 GEMM versus the independent FP64 scalar oracle. A column
        # permutation makes Q non-symmetric, so transposing it changes results.
        block = (
            torch.tensor(
                [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]],
                dtype=torch.float32,
            )[:, [1, 2, 3, 0]]
            / 2
        )
        for index in range(0, 16, 4):
            self.q[index : index + 4, index : index + 4] = block
        self.q_path = self.root / "quarot.safetensors"
        save_file({"global_rotation": self.q}, str(self.q_path))
        self.quarot_config = GlmDSparkQuaRotConfig(
            rotation_path=str(self.q_path),
            hidden_size=16,
            target_model_path=str(self.root / "target"),
        )

    def _load(self, *, mode="original", npu=True, scope=True, weights=None):
        (self.model_dir / "config.json").write_text(json.dumps(self.config))
        save_file(
            self.weights if weights is None else weights,
            str(self.model_dir / "model.safetensors"),
        )
        config = ModelConfig(
            str(self.model_dir),
            dtype="bfloat16",
            is_draft_model=True,
            speculative_algorithm="DSpark",
        )
        self.assertIsNone(config.quantization)
        loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_NPU_GLM_DSPARK_QUAROT.override(mode))
            stack.enter_context(
                patch("sglang.srt.models.dspark.is_npu", return_value=npu)
            )
            if scope:
                stack.enter_context(glm_dspark_quarot_scope(self.quarot_config))
            return loader.load_model(
                model_config=config, device_config=DeviceConfig("cpu")
            )

    def _expected_fc(self, original):
        # Independent scalar dot products use the actual serialized Q values.
        # They avoid calling the production converter or sharing its reshape.
        expected = torch.empty_like(original, dtype=torch.float64)
        source = original.double()
        q = self.q.double()
        for row in range(16):
            for feature in range(3):
                for column in range(16):
                    expected[row, feature * 16 + column] = sum(
                        source[row, feature * 16 + index] * q[index, column]
                        for index in range(16)
                    )
        return expected.to(torch.bfloat16)

    def _assert_own_weights(self, model, original_fc):
        self.assertIsInstance(model, DSparkDraftModel)
        self.assertTrue(model.uses_own_vocab_modules)
        self.assertEqual(model._glm_dspark_quarot_config, self.quarot_config)
        for module, key in (
            (model.embed_tokens, "embed_tokens.weight"),
            (model.lm_head, "lm_head.weight"),
        ):
            self.assertEqual(module.weight.dtype, torch.bfloat16)
            # VocabParallelEmbedding can pad its allocation beyond vocab_size.
            torch.testing.assert_close(
                module.weight[:32], self.weights[key], rtol=0, atol=0
            )
        torch.testing.assert_close(
            model.fc.weight, self._expected_fc(original_fc), rtol=0, atol=0
        )
        torch.testing.assert_close(
            model.hidden_norm.weight,
            self.weights["hidden_norm.weight"],
            rtol=0,
            atol=0,
        )
        self.assertEqual(model.fc.weight.dtype, torch.bfloat16)
        self.assertIsNone(model.confidence_head)

    def test_normal_loader_loads_own_vocab_and_rotates_each_original_fc_block(self):
        before = {name: value.clone() for name, value in self.weights.items()}
        model = self._load()
        self._assert_own_weights(model, before["fc.weight"])
        self.assertFalse(torch.equal(model.fc.weight, before["fc.weight"]))
        for name, value in before.items():
            torch.testing.assert_close(self.weights[name], value, rtol=0, atol=0)
        token_ids = torch.tensor([0, 7, 31])
        torch.testing.assert_close(
            model.forward_embed(token_ids),
            before["embed_tokens.weight"][token_ids],
            rtol=0,
            atol=0,
        )
        hidden = (torch.arange(32).reshape(2, 16).float() / 32).to(torch.bfloat16)
        logits, _ = model.compute_base_logits(hidden)
        torch.testing.assert_close(
            logits, hidden @ before["lm_head.weight"].T, rtol=0, atol=0
        )

    def test_loaded_fc_and_original_norm_execute_the_cpu_reference(self):
        model = self._load()
        original = (torch.arange(3 * 3 * 16).reshape(3, 3, 16) * 5) % 37 - 18
        original = original.double() / 32
        original *= torch.tensor([1.0, 0.01, 0.0001]).reshape(3, 1, 1)
        rotated = (original @ self.q.double()).reshape(3, 48).to(torch.bfloat16)
        expected_fc = self._expected_fc(self.weights["fc.weight"])
        pre_norm = (rotated.double() @ expected_fc.double().T).to(torch.bfloat16)
        work = pre_norm.double()
        expected = (
            work
            / (work.square().mean(dim=-1, keepdim=True) + 1e-5).sqrt()
            * self.weights["hidden_norm.weight"].double()
        ).to(torch.bfloat16)
        with torch.no_grad():
            actual = model.project_target_hidden(rotated)
        # Use torch's dtype-specific tolerances for this synthetic CPU oracle;
        # they are not a threshold for real NPU/model acceptance.
        torch.testing.assert_close(actual, expected)
        self.assertEqual(model.hidden_norm.variance_epsilon, 1e-5)

    def test_active_loading_does_not_replace_or_modify_shared_target_modules(self):
        shared_model = self._load(mode="", scope=False)
        with set_default_torch_dtype(torch.bfloat16), torch.device("cpu"):
            target_embedding = VocabParallelEmbedding(32, 16)
            target_head = ParallelLMHead(32, 16)
        with torch.no_grad():
            target_embedding.weight.fill_(0.125)
            target_head.weight.fill_(-0.25)
        embedding_before = target_embedding.weight.detach().clone()
        head_before = target_head.weight.detach().clone()
        shared_model.attach_shared_modules(
            embed_tokens=target_embedding, lm_head=target_head
        )
        active = self._load()
        active.load_weights(self.weights.items())
        self.assertIsNot(active.embed_tokens, target_embedding)
        self.assertIsNot(active.lm_head, target_head)
        self.assertIs(shared_model.embed_tokens, target_embedding)
        self.assertIs(shared_model.lm_head, target_head)
        torch.testing.assert_close(
            target_embedding.weight, embedding_before, rtol=0, atol=0
        )
        torch.testing.assert_close(target_head.weight, head_before, rtol=0, atol=0)
        # A subsequent draft has no lingering constructor scope or own vocab.
        subsequent = self._load(scope=False)
        self.assertIsNone(subsequent.embed_tokens)
        self.assertIsNone(subsequent.lm_head)

    def test_partial_original_fc_reload_converts_once_and_retains_parameter_identity(
        self,
    ):
        model = self._load()
        parameter = model.fc.weight
        embedding = model.embed_tokens.weight.detach().clone()
        head = model.lm_head.weight.detach().clone()
        changed = (-self.weights["fc.weight"] + 0.125).to(torch.bfloat16)
        for source in (self.weights["fc.weight"], self.weights["fc.weight"], changed):
            with self.subTest(changed=source is changed):
                before = source.clone()
                model.load_weights([("fc.weight", source)])
                self.assertIs(model.fc.weight, parameter)
                torch.testing.assert_close(
                    parameter, self._expected_fc(before), rtol=0, atol=0
                )
                torch.testing.assert_close(source, before, rtol=0, atol=0)
        torch.testing.assert_close(model.embed_tokens.weight, embedding, rtol=0, atol=0)
        torch.testing.assert_close(model.lm_head.weight, head, rtol=0, atol=0)

    def test_initial_load_requires_own_embedding_head_and_fc(self):
        for missing in ("embed_tokens.weight", "lm_head.weight", "fc.weight"):
            with self.subTest(missing=missing):
                incomplete = {
                    name: value
                    for name, value in self.weights.items()
                    if name != missing
                }
                with self.assertRaisesRegex(
                    ValueError, "(?i)(missing|required)"
                ) as error:
                    self._load(weights=incomplete)
                self.assertIn(missing, str(error.exception))

    def test_model_prefix_and_encoder_alias_load_the_same_original_parameters(self):
        for fc_name in ("model.fc.weight", "encoder.fc.weight"):
            with self.subTest(fc_name=fc_name):
                aliases = dict(self.weights)
                aliases[fc_name] = aliases.pop("fc.weight")
                aliases["model.embed_tokens.weight"] = aliases.pop(
                    "embed_tokens.weight"
                )
                aliases["model.lm_head.weight"] = aliases.pop("lm_head.weight")
                aliases["encoder.output_norm_enc.weight"] = aliases.pop(
                    "hidden_norm.weight"
                )
                model = self._load(weights=aliases)
                self._assert_own_weights(model, self.weights["fc.weight"])

    def test_inactive_paths_keep_the_shared_vocab_contract_without_reading_q(self):
        self.q_path.unlink()
        for options in (
            {"mode": ""},
            {"npu": False},
            {"scope": False},
        ):
            with self.subTest(options=options):
                model = self._load(**options)
                self.assertIsNone(model._glm_dspark_quarot_config)
                self.assertFalse(getattr(model, "uses_own_vocab_modules", False))
                self.assertIsNone(model.embed_tokens)
                self.assertIsNone(model.lm_head)
                torch.testing.assert_close(
                    model.fc.weight, self.weights["fc.weight"], rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
