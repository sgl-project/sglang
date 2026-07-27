import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

sys.modules["sgl_kernel"] = MagicMock()
for _submodule in (
    "elementwise",
    "flash_attn",
    "flash_mla",
    "kvcacheio",
    "mamba",
    "quantization",
    "scalar_type",
    "sparse_flash_attn",
    "speculative",
    "utils",
):
    sys.modules[f"sgl_kernel.{_submodule}"] = MagicMock()

from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=12, suite="base-b-test-cpu")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.models import granitemoe  # noqa: E402
from sglang.srt.models.gemma2 import Gemma2ForCausalLM  # noqa: E402
from sglang.srt.models.gemma2_reward import (  # noqa: E402
    Gemma2ForSequenceClassification,
)
from sglang.srt.models.internlm2 import InternLM2ForCausalLM  # noqa: E402
from sglang.srt.models.internlm2_reward import InternLM2ForRewardModel  # noqa: E402
from sglang.srt.models.llama import LlamaForCausalLM  # noqa: E402
from sglang.srt.models.llama_classification import LlamaForClassification  # noqa: E402
from sglang.srt.models.llama_eagle import LlamaForCausalLMEagle  # noqa: E402
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3  # noqa: E402
from sglang.srt.models.llama_embedding import LlamaEmbeddingModel  # noqa: E402
from sglang.srt.models.llama_reward import (  # noqa: E402
    LlamaForSequenceClassification,
)
from sglang.srt.models.mistral import MistralForCausalLMMistralFormat  # noqa: E402
from sglang.srt.models.mistral_eagle import MistralForCausalLMEagle  # noqa: E402
from sglang.srt.models.mixtral import MixtralForCausalLM  # noqa: E402
from sglang.srt.models.qwen2 import Qwen2ForCausalLM  # noqa: E402
from sglang.srt.models.qwen2_classification import (  # noqa: E402
    Qwen2ForSequenceClassification,
)
from sglang.srt.models.qwen2_eagle import Qwen2ForCausalLMEagle  # noqa: E402
from sglang.srt.models.qwen2_rm import Qwen2ForRewardModel  # noqa: E402
from sglang.srt.models.qwen3_classification import (  # noqa: E402
    Qwen3ForSequenceClassification,
)
from sglang.srt.models.qwen3_embedding import Qwen3Model  # noqa: E402
from sglang.srt.models.qwen3_moe import Qwen3MoeForCausalLM  # noqa: E402
from sglang.srt.models.qwen3_moe_mtp import Qwen3MoeForCausalLMMTP  # noqa: E402


def _bare_module(cls):
    module = cls.__new__(cls)
    torch.nn.Module.__init__(module)
    return module


def _set_v2(monkeypatch, enabled):
    monkeypatch.setattr(
        envs,
        "SGLANG_ENABLE_WEIGHT_LOADER_V2",
        SimpleNamespace(get=lambda: enabled),
    )


@pytest.mark.parametrize(
    ("wrapper_class", "base_class", "weights", "expected_names"),
    [
        (
            Gemma2ForSequenceClassification,
            Gemma2ForCausalLM,
            [("model.weight", torch.ones(1))],
            ["model.weight"],
        ),
        (
            InternLM2ForRewardModel,
            InternLM2ForCausalLM,
            [("model.weight", torch.ones(1))],
            ["model.weight"],
        ),
        (
            LlamaForSequenceClassification,
            LlamaForCausalLM,
            [("model.weight", torch.ones(1))],
            ["model.weight"],
        ),
        (
            Qwen2ForSequenceClassification,
            Qwen2ForCausalLM,
            [("model.weight", torch.ones(1)), ("lm_head.weight", torch.ones(1))],
            ["model.weight"],
        ),
        (
            Qwen2ForRewardModel,
            Qwen2ForCausalLM,
            [("model.weight", torch.ones(1)), ("lm_head.weight", torch.ones(1))],
            ["model.weight"],
        ),
        (
            LlamaForCausalLMEagle,
            LlamaForCausalLM,
            [
                ("layers.0.weight", torch.ones(1)),
                ("norm.weight", torch.ones(1)),
                ("lm_head.weight", torch.ones(1)),
            ],
            ["model.layers.0.weight", "model.norm.weight"],
        ),
        (
            Qwen2ForCausalLMEagle,
            Qwen2ForCausalLM,
            [
                ("layers.0.weight", torch.ones(1)),
                ("norm.weight", torch.ones(1)),
                ("lm_head.weight", torch.ones(1)),
            ],
            ["model.layers.0.weight", "model.norm.weight"],
        ),
        (
            MistralForCausalLMMistralFormat,
            LlamaForCausalLM,
            [("norm.weight", torch.ones(1))],
            ["model.norm.weight"],
        ),
        (
            MistralForCausalLMEagle,
            LlamaForCausalLM,
            [("norm.weight", torch.ones(1))],
            ["model.norm.weight"],
        ),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_wrappers_select_direct_base_helper_once(
    monkeypatch, wrapper_class, base_class, weights, expected_names, enabled
):
    calls = []

    def capture(kind):
        def inner(module, incoming):
            calls.append((kind, module, list(incoming)))
            return {kind}

        return inner

    monkeypatch.setattr(base_class, "_legacy_load_weights", capture("legacy"))
    monkeypatch.setattr(base_class, "_load_weights_v2", capture("v2"))
    monkeypatch.setattr(
        base_class,
        "load_weights",
        lambda *_args, **_kwargs: pytest.fail("unbound dispatcher was called"),
    )
    _set_v2(monkeypatch, enabled)
    wrapper = _bare_module(wrapper_class)

    result = wrapper.load_weights(iter(weights))

    if enabled:
        assert result == {"v2"}
        assert len(calls) == 1
        kind, module, prepared = calls[0]
        assert kind == "v2"
        assert module is wrapper
        assert [name for name, _ in prepared] == expected_names
    elif wrapper_class in (LlamaForCausalLMEagle, Qwen2ForCausalLMEagle):
        assert result is None
        assert [
            (kind, module, [name for name, _ in prepared])
            for kind, module, prepared in calls
        ] == [
            ("legacy", wrapper, [expected_name])
            for expected_name in expected_names
        ]
    else:
        expected_result = (
            None if wrapper_class is Gemma2ForSequenceClassification else {"legacy"}
        )
        assert result == expected_result
        assert len(calls) == 1
        kind, module, prepared = calls[0]
        assert kind == "legacy"
        assert module is wrapper
        assert [name for name, _ in prepared] == expected_names


@pytest.mark.parametrize(
    "wrapper_class",
    [
        Qwen3ForSequenceClassification,
        Qwen3Model,
        LlamaForClassification,
        LlamaEmbeddingModel,
        LlamaForCausalLMEagle3,
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_model_local_wrappers_select_local_helper_once(
    monkeypatch, wrapper_class, enabled
):
    calls = []

    def capture(kind):
        def inner(module, incoming):
            calls.append((kind, module, list(incoming)))
            return {kind}

        return inner

    monkeypatch.setattr(wrapper_class, "_legacy_load_weights", capture("legacy"))
    monkeypatch.setattr(wrapper_class, "_load_weights_v2", capture("v2"))
    _set_v2(monkeypatch, enabled)
    wrapper = _bare_module(wrapper_class)
    weights = [("model.weight", torch.ones(1))]

    result = wrapper.load_weights(iter(weights))

    expected_kind = "v2" if enabled else "legacy"
    assert result == {expected_kind}
    assert calls == [(expected_kind, wrapper, weights)]


def test_llama_classification_legacy_preserves_per_weight_parent_fanout(
    monkeypatch,
):
    calls = []

    def capture_legacy(module, incoming):
        calls.append((module, list(incoming)))

    monkeypatch.setattr(LlamaForCausalLM, "_legacy_load_weights", capture_legacy)
    wrapper = _bare_module(LlamaForClassification)
    weights = [
        ("model.layers.0.weight", torch.ones(1)),
        ("model.layers.1.weight", torch.ones(1)),
        ("lm_head.weight", torch.ones(1)),
    ]

    result = wrapper._legacy_load_weights(iter(weights))

    assert result is None
    assert calls == [
        (wrapper, [weights[0]]),
        (wrapper, [weights[1]]),
    ]


def test_llama_embedding_legacy_preserves_early_return():
    wrapper = _bare_module(LlamaEmbeddingModel)
    wrapper.model = torch.nn.Linear(1, 1, bias=False)
    wrapper.model.weight.data.zero_()

    result = wrapper._legacy_load_weights(
        iter(
            [
                ("projector.weight", torch.ones(1)),
                ("weight", torch.ones_like(wrapper.model.weight)),
            ]
        )
    )

    assert result is None
    torch.testing.assert_close(
        wrapper.model.weight, torch.zeros_like(wrapper.model.weight)
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_qwen3_mtp_selects_helper_and_preserves_mtp_remap(monkeypatch, enabled):
    calls = []

    def capture_legacy(module, incoming, *, is_mtp):
        calls.append(("legacy", module, list(incoming), is_mtp))
        return {"legacy"}

    def capture_v2(module, incoming):
        calls.append(("v2", module, list(incoming)))
        return {"v2"}

    monkeypatch.setattr(Qwen3MoeForCausalLM, "_legacy_load_weights", capture_legacy)
    monkeypatch.setattr(Qwen3MoeForCausalLM, "_load_weights_v2", capture_v2)
    monkeypatch.setattr(
        Qwen3MoeForCausalLM,
        "load_weights",
        lambda *_args, **_kwargs: pytest.fail("parent dispatcher was called"),
    )
    _set_v2(monkeypatch, enabled)
    wrapper = _bare_module(Qwen3MoeForCausalLMMTP)
    weights = [
        ("mtp.fc.weight", torch.ones(1)),
        ("mtp.layers.0.weight", torch.ones(1)),
        ("model.layers.1.weight", torch.ones(1)),
    ]

    result = wrapper.load_weights(iter(weights))

    if enabled:
        assert result == {"v2"}
        assert [name for name, _ in calls[0][2]] == [
            "fc.weight",
            "model.layers.0.weight",
        ]
    else:
        assert result == {"legacy"}
        assert calls == [("legacy", wrapper, weights, True)]


@pytest.mark.parametrize("enabled", [False, True])
def test_granitemoe_preserves_expert_fanout_before_selected_loader(
    monkeypatch, enabled
):
    calls = []

    def capture_legacy(module, incoming):
        calls.append(("legacy", module, dict(incoming)))
        return {"legacy"}

    def capture_v2(module, incoming):
        calls.append(("v2", module, dict(incoming)))
        return {"v2"}

    monkeypatch.setattr(MixtralForCausalLM, "_legacy_load_weights", capture_legacy)
    monkeypatch.setattr(
        granitemoe.GraniteMoeForCausalLM, "_load_weights_v2", capture_v2
    )
    monkeypatch.setattr(
        MixtralForCausalLM,
        "load_weights",
        lambda *_args, **_kwargs: pytest.fail("unbound dispatcher was called"),
    )
    _set_v2(monkeypatch, enabled)
    wrapper = _bare_module(granitemoe.GraniteMoeForCausalLM)
    input_weight = torch.arange(16).reshape(2, 4, 2)
    output_weight = torch.arange(8).reshape(2, 2, 2)
    router_weight = torch.ones(2, 2)
    prefix = "model.layers.0.block_sparse_moe"

    result = wrapper.load_weights(
        iter(
            [
                (f"{prefix}.input_linear.weight", input_weight),
                (f"{prefix}.output_linear.weight", output_weight),
                (f"{prefix}.router.layer.weight", router_weight),
            ]
        )
    )

    expected_kind = "v2" if enabled else "legacy"
    assert result == {expected_kind}
    assert len(calls) == 1
    kind, module, remapped = calls[0]
    assert kind == expected_kind
    assert module is wrapper
    for expert_id in range(2):
        expected_w1, expected_w3 = input_weight[expert_id].chunk(2, dim=0)
        torch.testing.assert_close(
            remapped[f"{prefix}.experts.{expert_id}.w1.weight"], expected_w1
        )
        torch.testing.assert_close(
            remapped[f"{prefix}.experts.{expert_id}.w3.weight"], expected_w3
        )
        torch.testing.assert_close(
            remapped[f"{prefix}.experts.{expert_id}.w2.weight"],
            output_weight[expert_id],
        )
    torch.testing.assert_close(remapped[f"{prefix}.gate.weight"], router_weight)


@pytest.mark.parametrize(
    "module_name",
    ["qwen3_rm", "mellum", "teleflm", "llama4", "ministral3"],
)
def test_adjacent_model_import_smoke(module_name):
    try:
        importlib.import_module(f"sglang.srt.models.{module_name}")
    except ImportError as exc:
        pytest.skip(f"optional import unavailable: {exc}")
