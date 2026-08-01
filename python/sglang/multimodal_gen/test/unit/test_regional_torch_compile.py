from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2ArchConfig
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    CompiledModuleRegistry,
    compile_matching_submodules,
)


class _CompilableModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.compile_calls = []

    def compile(self, **kwargs):
        self.compile_calls.append(kwargs)


class _RegionalModel(_CompilableModule):
    _compile_conditions = [
        lambda name, _module: name.startswith("transformer_blocks.")
        and name.count(".") == 1
    ]

    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_CompilableModule(), _CompilableModule()]
        )
        self.transformer_blocks[0].inner = _CompilableModule()
        self.proj_out = _CompilableModule()


def test_ltx2_compile_conditions_match_only_direct_blocks():
    conditions = LTX2ArchConfig()._compile_conditions

    assert conditions
    assert any(condition("transformer_blocks.0", object()) for condition in conditions)
    assert not any(
        condition("transformer_blocks.0.attn1", object()) for condition in conditions
    )
    assert not any(
        condition("transformer_blocks", object()) for condition in conditions
    )


def test_compile_matching_submodules_matches_only_declared_regions():
    model = _RegionalModel()

    count = compile_matching_submodules(
        model,
        compile_kwargs={"mode": "default", "fullgraph": False},
    )

    assert count == 2
    assert [len(block.compile_calls) for block in model.transformer_blocks] == [1, 1]
    assert not model.transformer_blocks[0].inner.compile_calls
    assert not model.proj_out.compile_calls
    assert not model.compile_calls


def test_compile_matching_submodules_fails_when_no_region_matches():
    model = _RegionalModel()
    model._compile_conditions = [lambda _name, _module: False]

    with pytest.raises(ValueError, match="no matching submodules"):
        compile_matching_submodules(model, compile_kwargs={"mode": "default"})


def test_compiled_module_registry_installs_regions_once():
    model = _RegionalModel()
    registry = CompiledModuleRegistry()

    assert (
        registry.compile_regions_once(
            model,
            compile_kwargs={"mode": "default"},
        )
        == 2
    )
    assert (
        registry.compile_regions_once(
            model,
            compile_kwargs={"mode": "default"},
        )
        == 0
    )
    assert [len(block.compile_calls) for block in model.transformer_blocks] == [1, 1]


def test_denoising_stage_selects_regional_compile():
    model = _RegionalModel()
    stage = DenoisingStage.__new__(DenoisingStage)
    stage.server_args = SimpleNamespace(
        enable_breakable_cuda_graph=False,
        enable_torch_compile=True,
        regional_compile=True,
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(torch_compile_mode="default")
        ),
    )
    stage._cache_dit_enabled = False
    stage._torch_compile_registry = CompiledModuleRegistry()

    with (
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
            "current_platform.is_npu",
            return_value=False,
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
            "maybe_enable_inductor_compute_comm_overlap"
        ),
    ):
        stage._maybe_torch_compile(model)

    assert [len(block.compile_calls) for block in model.transformer_blocks] == [1, 1]
    assert not model.compile_calls
