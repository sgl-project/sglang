"""A draft model keeps the quant config its class declares before it is built.

Under unified memory a fused draft's KV region is sized before the draft loads,
in the dtype an ``auto`` draft KV cache resolves from the quant config the
loader predicts the draft keeps (``built_quant_config``). Once loaded, the
draft runner resolves the same dtype from the model's ``quant_config``
attribute. The two must agree, or startup refuses the region.
"""

import ast
import inspect
import re
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import DeepseekV3Config

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import (
    UnquantizedEmbeddingMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_loader import loader
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.models.kimi_k25_eagle3 import Eagle3DeepseekV2ForCausalLM
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.models.utils import keeps_no_quant_config
from sglang.srt.runtime_context import get_parallel, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Fp8KvCheckpoint(QuantizationConfig):
    """A checkpoint declaring an FP8 KV cache; its layers build unquantized."""

    kv_cache_quant_algo = "FP8"

    def __init__(self, name="fp8_kv_checkpoint"):
        super().__init__()
        self._name = name

    def get_name(self):
        return self._name

    def get_supported_act_dtypes(self):
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 0

    @staticmethod
    def get_config_filenames():
        return []

    @classmethod
    def from_config(cls, config):
        return cls()

    def get_scaled_act_names(self):
        return []

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase):
            return UnquantizedLinearMethod()
        if isinstance(layer, VocabParallelEmbedding):
            return UnquantizedEmbeddingMethod()
        return None


_SMALL = dict(
    hidden_size=16,
    intermediate_size=32,
    num_attention_heads=2,
    num_key_value_heads=1,
    vocab_size=64,
    rms_norm_eps=1e-6,
    max_position_embeddings=64,
    rope_theta=1e4,
)


def _kimi_eagle3():
    return DeepseekV3Config(
        **_SMALL,
        num_hidden_layers=1,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        draft_vocab_size=64,
    )


def _predict(model_class, hf_config, loader_quant_config):
    """``built_quant_config`` for ``model_class`` over a loader config."""
    with (
        patch.object(
            loader, "_get_quantization_config", return_value=loader_quant_config
        ),
        patch.object(
            loader,
            "get_model_architecture",
            return_value=(model_class, model_class.__name__),
        ),
    ):
        return loader.built_quant_config(SimpleNamespace(hf_config=hf_config), None)


class _ContextTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        # CPU CI does not enable the CPU engine; keep RoPE on the native path.
        self.enterContext(
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True)
        )
        publish(ServerArgs(model_path="dummy", device="cuda", tp_size=1), role="test")
        get_parallel().override_permanently(
            pp_group=SimpleNamespace(
                is_first_rank=True, is_last_rank=True, rank_in_group=0, world_size=1
            )
        )


class TestBuiltDraftKeepsPredictedQuantConfig(_ContextTestCase):
    def test_a_draft_narrowing_the_checkpoint_keeps_what_it_declares(self):
        """A target-only NVFP4 checkpoint: the Kimi EAGLE3 draft drops it."""
        hf_config = _kimi_eagle3()
        checkpoint = _Fp8KvCheckpoint("modelopt_fp4")
        model = Eagle3DeepseekV2ForCausalLM(hf_config, quant_config=checkpoint)
        self.assertIsNone(_predict(Eagle3DeepseekV2ForCausalLM, hf_config, checkpoint))
        self.assertIsNone(model.quant_config)


# Entry classes a speculative draft resolves to. DFLASH-family drafts are
# matched by their base class, whatever the checkpoint names them.
_DRAFT_NAME = re.compile(r"(NextN|MTP|Eagle3?|DFlash|DSpark|Draft|Assistant)")


def _draft_entry_classes():
    classes = {}
    for arch in sorted(ModelRegistry.get_supported_archs()):
        try:
            cls = ModelRegistry.resolve_model_cls(arch)[0]
        except Exception:
            continue  # an arch this environment cannot import
        if (
            _DRAFT_NAME.search(arch)
            or _DRAFT_NAME.search(cls.__name__)
            or issubclass(cls, DFlashDraftModel)
        ):
            classes[arch] = cls
    return classes


def _is_hook_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "own_quant_config"
    )


def _is_param(node):
    return isinstance(node, ast.Name) and node.id == "quant_config"


def _quant_config_writes(cls):
    """What the sglang classes of ``cls``'s MRO write: the values assigned to
    ``self.quant_config``, and the departures from the constructor's own
    ``quant_config`` that do not go through ``own_quant_config``."""
    kept, departures = [], []
    for klass in cls.__mro__:
        if not klass.__module__.startswith("sglang."):
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(klass)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and target.attr == "quant_config"
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "self"
                    ):
                        kept.append(klass.__name__)
                        if not (_is_param(node.value) or _is_hook_call(node.value)):
                            departures.append(
                                f"{klass.__name__}: self.quant_config = "
                                f"{ast.unparse(node.value)}"
                            )
            if not (isinstance(node, ast.FunctionDef) and node.name == "__init__"):
                continue
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Assign)
                    and any(_is_param(target) for target in sub.targets)
                    and not _is_hook_call(sub.value)
                ):
                    departures.append(
                        f"{klass.__name__}.__init__: quant_config = "
                        f"{ast.unparse(sub.value)}"
                    )
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "__init__"
                ):
                    for keyword in sub.keywords:
                        if keyword.arg == "quant_config" and not _is_param(
                            keyword.value
                        ):
                            departures.append(
                                f"{klass.__name__}.__init__: {ast.unparse(sub)}"
                            )
    return kept, departures


class TestEveryDraftDeclaresTheQuantConfigItKeeps(_ContextTestCase):
    """Source-level check over every registered draft entry class: building
    them all is not practical on CPU, so this reads what their constructors
    assign to ``self.quant_config``."""

    def test_every_draft_keeps_what_built_quant_config_predicts(self):
        drafts = _draft_entry_classes()
        self.assertGreater(len(drafts), 20, "the draft entry classes were not found")
        sentinel = _Fp8KvCheckpoint()
        offenders = []
        for arch, cls in drafts.items():
            kept, departures = _quant_config_writes(cls)
            offenders += [f"{arch}: {departure}" for departure in departures]
            keeps_nothing = getattr(cls, "own_quant_config", None) is (
                keeps_no_quant_config
            )
            if not kept and _predict(cls, SimpleNamespace(), sentinel) is not None:
                offenders.append(
                    f"{arch}: keeps no quant_config, but does not declare "
                    "own_quant_config = staticmethod(keeps_no_quant_config)"
                )
            if kept and keeps_nothing:
                offenders.append(
                    f"{arch}: declares keeps_no_quant_config, but "
                    f"{', '.join(sorted(set(kept)))} assigns self.quant_config"
                )
        self.assertEqual(
            [],
            offenders,
            "a draft's `auto` KV cache dtype is sized before it is built from "
            "the quant config built_quant_config predicts it keeps: the "
            "loader's, as the class's own_quant_config narrows it. Narrow "
            "through own_quant_config, or declare what the class keeps:\n  "
            + "\n  ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
