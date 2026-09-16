from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    LTX2VideoTransformer3DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    CompiledModuleRegistry,
    build_torch_compile_kwargs,
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
        lambda name, _module: (
            name.startswith("transformer_blocks.") and name.count(".") == 1
        )
    ]

    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(
            [_CompilableModule(), _CompilableModule()]
        )
        self.transformer_blocks[0].inner = _CompilableModule()
        self.proj_out = _CompilableModule()


@pytest.mark.parametrize(
    ("backend", "options", "expected"),
    [
        (
            "custom_backend",
            {"pass_manager_config": {"persistent_buffers": ["weight"]}},
            {
                "backend": "custom_backend",
                "options": {"pass_manager_config": {"persistent_buffers": ["weight"]}},
            },
        ),
        (
            "inductor",
            {"max_autotune": True},
            {"backend": "inductor", "options": {"max_autotune": True}},
        ),
        (
            "inductor",
            None,
            {"backend": "inductor", "mode": "max-autotune-no-cudagraphs"},
        ),
    ],
)
def test_out_of_tree_platform_controls_compile_kwargs(backend, options, expected):
    """Out-of-tree hooks select valid backend, mode, and option combinations."""
    module = _CompilableModule()
    with (
        patch(
            "sglang.multimodal_gen.runtime.utils.torch_compile."
            "current_platform.is_out_of_tree",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.utils.torch_compile."
            "current_platform.get_compile_backend",
            return_value=backend,
        ) as get_compile_backend,
        patch(
            "sglang.multimodal_gen.runtime.utils.torch_compile."
            "current_platform.get_compile_options",
            return_value=options,
        ) as get_compile_options,
    ):
        compile_kwargs = build_torch_compile_kwargs(
            mode="max-autotune-no-cudagraphs",
            module=module,
        )

    assert compile_kwargs == {
        "dynamic": None,
        "fullgraph": False,
        **expected,
    }
    get_compile_backend.assert_called_once_with("max-autotune-no-cudagraphs")
    get_compile_options.assert_called_once_with(module)


def test_ltx2_compile_conditions_match_only_direct_blocks():
    conditions = LTX2VideoTransformer3DModel._compile_conditions

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
    compile_kwargs = {"backend": "custom_backend"}
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
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.denoising."
            "build_torch_compile_kwargs",
            return_value=compile_kwargs,
        ) as build_compile_kwargs,
    ):
        stage._maybe_torch_compile(model)

    build_compile_kwargs.assert_called_once_with(mode="default", module=model)
    assert [len(block.compile_calls) for block in model.transformer_blocks] == [1, 1]
    assert [block.compile_calls for block in model.transformer_blocks] == [
        [compile_kwargs],
        [compile_kwargs],
    ]
    assert not model.compile_calls
