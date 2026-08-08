"""Resolution is a pure function of the raw input plus this node's environment.

The end state for the configuration tier keeps ``ServerArgs`` at the user's raw
input and lets every process that publishes derive the resolved values itself
(bags do not cross a process boundary — a child projects its own from the record
it is handed). That is only sound if resolving the same raw input twice gives
the same answer, so this pins it:

- twice in this process, from equal raw inputs, every field agrees;
- the resolution is not order-dependent on a shared registry (a second config
  resolved after the first does not inherit its declarations);
- the raw record the two started from is itself unchanged by resolving a
  sibling.

A failure here means some resolution step reads state it also writes, and the
"re-derive in the child" contract would silently diverge between the launcher
and its schedulers.
"""

import dataclasses
import json
import os
import shutil
import tempfile
import unittest

from sglang.srt.environ import EnvField, envs
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}

_DEEPSEEK_MINI_CONFIG = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "hidden_size": 16,
    "intermediate_size": 32,
    "moe_intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "first_k_dense_replace": 1,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
    "kv_lora_rank": 8,
    "q_lora_rank": 8,
    "qk_nope_head_dim": 8,
    "qk_rope_head_dim": 8,
    "v_head_dim": 8,
    "topk_method": "greedy",
    "scoring_func": "softmax",
}

# The shapes the step-12 audit calls out as the ones whose resolution branches
# touch process state: a plain text model, a speculative launch, and a MoE/MLA
# architecture whose handlers fan out the most.
_SHAPES = (
    ("plain", _MINI_CONFIG, {}),
    (
        "speculative",
        _MINI_CONFIG,
        dict(
            speculative_algorithm="EAGLE",
            speculative_num_steps=2,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=3,
        ),
    ),
    ("deepseek", _DEEPSEEK_MINI_CONFIG, {}),
)

# `random_seed` is pinned by `_resolved` so it would compare equal anyway; it
# stays listed because a case that stops pinning it must not silently start
# comparing a value resolution randomizes.
_NOT_COMPARABLE = frozenset({"random_seed"})


class TestResolutionIsReproducible(CustomTestCase):
    def _config_dir(self, config: dict = None) -> str:
        config_dir = tempfile.mkdtemp(prefix="resolution_repro_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as handle:
            json.dump(config or _MINI_CONFIG, handle)
        return config_dir

    def setUp(self):
        # Resolution writes process state on the way through --
        # `_handle_multimodal_feature_transport` sets SGLANG_USE_CUDA_IPC_TRANSPORT
        # so tokenizer workers inherit the decision, and the same handler reads
        # `is_set()` on the way in. One resolution is therefore visible to the
        # next one in this process. These cases restore what they touched, and
        # it is a standing caveat on the determinism pinned here: the guarantee
        # holds per raw input *and* the process state a previous resolution left.
        saved_environ = dict(os.environ)
        # EnvField.set() also flips a descriptor-level flag that os.environ does
        # not carry, so restore that too.
        saved_none_flags = {
            name: field._set_to_none
            for name, field in vars(type(envs)).items()
            if isinstance(field, EnvField)
        }

        def restore():
            os.environ.clear()
            os.environ.update(saved_environ)
            for name, was_none in saved_none_flags.items():
                getattr(type(envs), name)._set_to_none = was_none

        self.addCleanup(restore)

    def _callTestMethod(self, method):
        # No retry here. CustomTestCase retries once in CI, but `addCleanup`
        # runs after the last attempt, so a second attempt would start from the
        # state the first one leaked -- exactly the regression these cases exist
        # to catch, turned into a pass.
        unittest.TestCase._callTestMethod(self, method)

    def _resolved(self, model_path: str, **kwargs) -> ServerArgs:
        # device="cuda" keeps the golden path host-independent: an
        # accelerator-less runner resolves only the base platform, where
        # get_device() raises.
        kwargs.setdefault("device", "cuda")
        kwargs.setdefault("random_seed", 42)
        return ServerArgs(model_path=model_path, **kwargs)

    def _comparable(self, server_args: ServerArgs) -> dict:
        out = {}
        for field in dataclasses.fields(server_args):
            if field.name in _NOT_COMPARABLE:
                continue
            value = getattr(server_args, field.name)
            # Nested dataclasses (cuda_graph_config) compare structurally.
            out[field.name] = (
                dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value
            )
        return out

    def test_two_resolutions_of_the_same_input_agree(self):
        for label, config, kwargs in _SHAPES:
            with self.subTest(shape=label):
                model_path = self._config_dir(config)
                first = self._resolved(model_path, **kwargs)
                second = self._resolved(model_path, **kwargs)
                self.assertEqual(self._comparable(first), self._comparable(second))

    def test_a_resolution_does_not_leak_into_the_next(self):
        # A config resolved with an explicit, non-default backend must not shift
        # what the next one picks. Residual process state (env, caches) is the
        # hazard, not the declaration registry, whose providers are import-time.
        model_path = self._config_dir()
        # The control has to be taken *before* the explicit resolution: if that
        # one contaminated the process, a control read afterwards would inherit
        # the same contamination and the assertion would pass vacuously.
        default_before = self._resolved(model_path)
        # torch_native is not the default on either CPU or CUDA, so the probe
        # really diverges on the suite that runs this.
        explicit = self._resolved(model_path, attention_backend="torch_native")
        self.assertEqual(explicit.attention_backend, "torch_native")
        self.assertNotEqual(
            self._comparable(explicit),
            self._comparable(default_before),
            "the probe resolved to the same config as the default, so this case "
            "would pass without exercising order dependence",
        )
        default_after = self._resolved(model_path)
        # Every field, not just the backend: a declaration registry takes
        # arbitrary field dicts, so a leak can land anywhere.
        self.assertEqual(
            self._comparable(default_after), self._comparable(default_before)
        )

    def test_resolving_a_sibling_leaves_the_first_alone(self):
        for label, config, kwargs in _SHAPES:
            with self.subTest(shape=label):
                model_path = self._config_dir(config)
                first = self._resolved(model_path, **kwargs)
                snapshot = self._comparable(first)
                self._resolved(
                    model_path, tp_size=2, chunked_prefill_size=1024, **kwargs
                )
                self.assertEqual(self._comparable(first), snapshot)

    def test_the_declaration_provenance_is_reproducible(self):
        model_path = self._config_dir()
        first = self._resolved(model_path)
        second = self._resolved(model_path)
        self.assertEqual(
            getattr(first, "_resolved_overrides", None),
            getattr(second, "_resolved_overrides", None),
        )


if __name__ == "__main__":
    unittest.main()
