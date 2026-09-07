from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.runtime.managers.memory_managers.component_manager import (
    ComponentResidencyManager,
    ComponentUse,
    ResidencyState,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    ComponentResidencyError,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency_strategies import (
    ComponentOffloadStrategy,
    ResidentStrategy,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.text_encoding import (
    RealtimeTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def test_component_offload_releases_preferred_component_after_request():
    strategy = ComponentOffloadStrategy()
    strategy.finish_use = Mock()
    module = torch.nn.Linear(2, 2)
    use = ComponentUse(
        stage_name="TextEncodingStage",
        component_name="text_encoder",
        preferred_ready_after_request=True,
    )
    state = ResidencyState(batch_is_warmup=False)

    strategy.finish_request(module, use, state, preferred=True)

    strategy.finish_use.assert_called_once_with(module, use, state)


def test_component_offload_keeps_preferred_component_after_warmup():
    strategy = ComponentOffloadStrategy()
    strategy.prepare_for_use = Mock()
    strategy.wait_for_use = Mock()
    strategy.finish_use = Mock()
    module = torch.nn.Linear(2, 2)
    use = ComponentUse(
        stage_name="TextEncodingStage",
        component_name="text_encoder",
        preferred_ready_after_request=True,
    )
    state = ResidencyState(batch_is_warmup=True)

    strategy.finish_request(module, use, state, preferred=True)

    strategy.prepare_for_use.assert_called_once_with(module, use, state)
    strategy.wait_for_use.assert_called_once_with(module, use, state)
    strategy.finish_use.assert_not_called()


def test_request_tail_uses_dynamic_component_instance():
    pipeline = SimpleNamespace(
        modules={},
        _stage_name_mapping={},
        component_residency_strategies={},
    )
    manager = ComponentResidencyManager(
        pipeline,
        SimpleNamespace(enable_layerwise_nvtx_marker=False),
    )
    strategy = Mock()
    strategy.prefetch_for_use.return_value = False
    manager.strategy_for = Mock(return_value=strategy)
    module = torch.nn.Linear(2, 2)
    use = ComponentUse("DynamicStage", "dynamic_encoder")

    manager.ensure_ready(use, module=module)
    manager.finish_request()

    strategy.finish_request.assert_called_once_with(
        module, use, manager.state, preferred=False
    )


def test_strategy_cache_replaces_stale_component_instance():
    server_args = SimpleNamespace(
        enable_layerwise_nvtx_marker=False,
        residency_mode=lambda _component_name: "resident",
    )
    pipeline = SimpleNamespace(
        modules={},
        _stage_name_mapping={},
        component_residency_strategies={},
    )
    manager = ComponentResidencyManager(pipeline, server_args)
    first_module = torch.nn.Linear(2, 2)
    second_module = torch.nn.Linear(2, 2)

    first_strategy = manager.strategy_for("transformer", first_module)
    second_strategy = manager.strategy_for("transformer", second_module)

    assert isinstance(first_strategy, ResidentStrategy)
    assert isinstance(second_strategy, ResidentStrategy)
    assert first_strategy is not second_strategy
    assert manager._strategy_cache["transformer"][0] is second_module


def test_forget_module_clears_active_manager_references():
    pipeline = SimpleNamespace(
        modules={},
        _stage_name_mapping={},
        component_residency_strategies={},
    )
    manager = ComponentResidencyManager(
        pipeline,
        SimpleNamespace(enable_layerwise_nvtx_marker=False),
    )
    module = torch.nn.Linear(2, 2)
    use = ComponentUse("stage", "transformer")
    manager._active_use = use
    manager._active_use_module = module
    manager.state.current_use = use
    manager._uses_seen["transformer"] = use
    manager._modules_seen["transformer"] = module
    manager._prefetched_use_keys.add(("stage", "transformer", None))

    manager.forget_module(module)

    assert manager._active_use is None
    assert manager._active_use_module is None
    assert manager.state.current_use is None
    assert "transformer" not in manager._uses_seen
    assert "transformer" not in manager._modules_seen
    assert manager._prefetched_use_keys == set()


def test_group_warmup_state_requires_every_batch_to_be_warmup():
    pipeline = SimpleNamespace(
        modules={},
        _stage_name_mapping={},
        component_residency_strategies={},
    )
    server_args = SimpleNamespace(enable_layerwise_nvtx_marker=False)
    manager = ComponentResidencyManager(pipeline, server_args)

    manager.begin_request(
        [],
        [
            SimpleNamespace(is_warmup=True),
            SimpleNamespace(is_warmup=False),
        ],
        server_args,
    )

    assert manager.state.batch_is_warmup is False


class _Stage:
    def __init__(self, *uses: ComponentUse):
        self.uses = list(uses)

    def component_uses(self, server_args, stage_name=None):
        return self.uses


def _manager_for_stage(stage, modules):
    pipeline = SimpleNamespace(
        modules=modules,
        _stage_name_mapping={"stage": stage},
        component_residency_strategies={},
    )
    server_args = SimpleNamespace(enable_layerwise_nvtx_marker=False)
    manager = ComponentResidencyManager(pipeline, server_args)
    manager.refresh_pipeline(pipeline)
    manager.begin_request([stage], SimpleNamespace(is_warmup=False), server_args)
    return manager, server_args


def _server_args_with_component_offload(component_name):
    server_args = ServerArgs.__new__(ServerArgs)
    server_args.component_residency = {component_name: "component-offload"}
    return server_args


def test_explicit_component_offload_requires_a_declared_request_use():
    stage = _Stage()
    pipeline = SimpleNamespace(
        modules={"auxiliary": torch.nn.Linear(2, 2)},
        _stage_name_mapping={"stage": stage},
        component_residency_strategies={},
    )
    server_args = _server_args_with_component_offload("auxiliary")
    manager = ComponentResidencyManager(pipeline, server_args)
    manager.refresh_pipeline(pipeline)

    with pytest.raises(
        ComponentResidencyError,
        match="'auxiliary'.*ComponentUse declaration",
    ):
        manager.begin_request([stage], SimpleNamespace(is_warmup=False), server_args)


def test_declared_component_use_admits_explicit_component_offload():
    stage = _Stage(ComponentUse("stage", "auxiliary"))
    pipeline = SimpleNamespace(
        modules={"auxiliary": torch.nn.Linear(2, 2)},
        _stage_name_mapping={"stage": stage},
        component_residency_strategies={},
    )
    server_args = _server_args_with_component_offload("auxiliary")
    manager = ComponentResidencyManager(pipeline, server_args)
    manager.refresh_pipeline(pipeline)

    manager.begin_request([stage], SimpleNamespace(is_warmup=False), server_args)


def test_single_component_stage_is_prepared_at_stage_entry():
    module = torch.nn.Linear(2, 2)
    use = ComponentUse("stage", "text_encoder")
    stage = _Stage(use)
    manager, server_args = _manager_for_stage(stage, {"text_encoder": module})
    strategy = Mock()
    manager.strategy_for = Mock(return_value=strategy)

    manager.before_stage(stage, 0, SimpleNamespace(is_warmup=False), server_args)
    manager.begin_stage()

    strategy.prepare_for_use.assert_called_once_with(module, use, manager.state)
    strategy.wait_for_use.assert_called_once_with(module, use, manager.state)


def test_explicit_component_use_is_prepared_only_at_call_site():
    module = torch.nn.Linear(2, 2)
    use = ComponentUse("stage", "text_encoder", start_at_stage_entry=False)
    stage = _Stage(use)
    manager, server_args = _manager_for_stage(stage, {"text_encoder": module})
    strategy = Mock()
    manager.strategy_for = Mock(return_value=strategy)

    manager.before_stage(stage, 0, SimpleNamespace(is_warmup=False), server_args)
    manager.begin_stage()

    strategy.prepare_for_use.assert_not_called()

    manager.begin_use(use)

    strategy.prepare_for_use.assert_called_once_with(module, use, manager.state)
    strategy.wait_for_use.assert_called_once_with(module, use, manager.state)


def test_realtime_text_encoder_use_starts_at_call_site():
    stage = RealtimeTextEncodingStage.__new__(RealtimeTextEncodingStage)
    stage.text_encoders = [None]
    stage._registered_stage_name = None

    uses = stage.component_uses(
        SimpleNamespace(component_precisions={}, pipeline_config=None),
        "RealtimeTextEncodingStage",
    )

    assert len(uses) == 1
    assert uses[0].component_name == "text_encoder"
    assert uses[0].start_at_stage_entry is False


def test_image_encoder_use_has_exact_precision():
    stage = ImageEncodingStage.__new__(ImageEncodingStage)
    stage.image_encoder = object()
    stage.text_encoder = None
    stage._registered_stage_name = None

    uses = stage.component_uses(
        SimpleNamespace(component_precisions={"image_encoder": "fp16"}),
        "ImageEncodingStage",
    )

    assert [(use.component_name, use.target_dtype) for use in uses] == [
        ("image_encoder", torch.float16)
    ]


def test_image_encoder_use_preserves_loaded_dtype_without_override():
    stage = ImageEncodingStage.__new__(ImageEncodingStage)
    stage.image_encoder = object()
    stage.text_encoder = None
    stage._registered_stage_name = None

    uses = stage.component_uses(
        SimpleNamespace(component_precisions={}),
        "ImageEncodingStage",
    )

    assert [(use.component_name, use.target_dtype) for use in uses] == [
        ("image_encoder", None)
    ]


def test_qwen_layered_uses_loaded_text_encoder(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines import qwen_image

    text_encoder = object()
    stage = SimpleNamespace()
    stage_kwargs = {}
    pipeline = qwen_image.QwenImageLayeredPipeline.__new__(
        qwen_image.QwenImageLayeredPipeline
    )
    pipeline.model_path = "model"
    pipeline.modules = {
        name: object()
        for name in (
            "text_encoder",
            "vae",
            "tokenizer",
            "processor",
            "transformer",
            "scheduler",
        )
    }
    pipeline.modules["text_encoder"] = text_encoder
    pipeline.add_stage_factory = lambda _role, factory, _name: factory()
    pipeline.add_standard_timestep_preparation_stage = lambda **_kwargs: None
    pipeline.add_standard_denoising_stage = lambda: None
    pipeline.add_standard_decoding_stage = lambda: None

    def create_stage(**kwargs):
        stage_kwargs.update(kwargs)
        return stage

    monkeypatch.setattr(
        qwen_image, "QwenImageLayeredBeforeDenoisingStage", create_stage
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_precision="bf16",
            text_encoder_precisions=("bf16",),
        )
    )

    pipeline.create_pipeline_stages(server_args)

    assert stage_kwargs["text_encoder"] is text_encoder


def test_single_component_stage_is_finished_at_stage_exit():
    module = torch.nn.Linear(2, 2)
    use = ComponentUse("stage", "text_encoder")
    stage = _Stage(use)
    manager, server_args = _manager_for_stage(stage, {"text_encoder": module})
    strategy = Mock()
    manager.strategy_for = Mock(return_value=strategy)

    manager.before_stage(stage, 0, SimpleNamespace(is_warmup=False), server_args)
    manager.begin_stage()
    manager.end_stage()

    strategy.finish_use.assert_called_once_with(module, use, manager.state)


def test_adjacent_stages_reuse_the_same_component_interval():
    module = torch.nn.Linear(2, 2)
    first_use = ComponentUse("first", "text_encoder")
    second_use = ComponentUse("second", "text_encoder")
    first_stage = _Stage(first_use)
    second_stage = _Stage(second_use)
    pipeline = SimpleNamespace(
        modules={"text_encoder": module},
        _stage_name_mapping={"first": first_stage, "second": second_stage},
        component_residency_strategies={},
    )
    server_args = SimpleNamespace(enable_layerwise_nvtx_marker=False)
    manager = ComponentResidencyManager(pipeline, server_args)
    manager.refresh_pipeline(pipeline)
    manager.begin_request(
        [first_stage, second_stage], SimpleNamespace(is_warmup=False), server_args
    )
    strategy = Mock()
    manager.strategy_for = Mock(return_value=strategy)

    manager.before_stage(first_stage, 0, SimpleNamespace(is_warmup=False), server_args)
    manager.begin_stage()
    manager.end_stage()
    manager.before_stage(second_stage, 1, SimpleNamespace(is_warmup=False), server_args)
    manager.begin_stage()
    manager.end_stage()

    strategy.prepare_for_use.assert_called_once_with(module, first_use, manager.state)
    strategy.finish_use.assert_called_once_with(module, second_use, manager.state)


def test_adjacent_same_component_replacement_finishes_old_instance():
    first_module = torch.nn.Linear(2, 2)
    second_module = torch.nn.Linear(2, 2)
    first_use = ComponentUse("first", "text_encoder")
    second_use = ComponentUse("second", "text_encoder")
    first_stage = _Stage(first_use)
    second_stage = _Stage(second_use)
    pipeline = SimpleNamespace(
        modules={"text_encoder": first_module},
        _stage_name_mapping={"first": first_stage, "second": second_stage},
        component_residency_strategies={},
    )
    server_args = SimpleNamespace(enable_layerwise_nvtx_marker=False)
    manager = ComponentResidencyManager(pipeline, server_args)
    manager.refresh_pipeline(pipeline)
    manager.begin_request(
        [first_stage, second_stage],
        SimpleNamespace(is_warmup=False),
        server_args,
    )
    first_strategy = Mock()
    second_strategy = Mock()
    manager.strategy_for = Mock(
        side_effect=lambda _component_name, module: (
            first_strategy if module is first_module else second_strategy
        )
    )

    manager.begin_use(first_use, module=first_module)
    manager.begin_use(second_use, module=second_module)

    first_strategy.finish_use.assert_called_once_with(
        first_module, first_use, manager.state
    )
    second_strategy.prepare_for_use.assert_called_once_with(
        second_module, second_use, manager.state
    )


def test_multi_component_stage_controls_its_use_intervals():
    uses = (
        ComponentUse("stage", "text_encoder"),
        ComponentUse("stage", "vae"),
    )
    stage = _Stage(*uses)
    manager, server_args = _manager_for_stage(stage, {})
    manager.begin_use = Mock()

    manager.before_stage(stage, 0, SimpleNamespace(is_warmup=False), server_args)
    manager.begin_stage()

    manager.begin_use.assert_not_called()


def test_dynamic_component_is_prepared_when_stage_supplies_its_module():
    use = ComponentUse("stage", "dynamic_encoder")
    stage = _Stage(use)
    manager, server_args = _manager_for_stage(stage, {})
    strategy = Mock()
    manager.strategy_for = Mock(return_value=strategy)

    manager.before_stage(stage, 0, SimpleNamespace(is_warmup=False), server_args)
    manager.begin_stage()
    module = torch.nn.Linear(2, 2)
    manager.begin_use(use, module=module)

    strategy.prepare_for_use.assert_called_once_with(module, use, manager.state)
    strategy.wait_for_use.assert_called_once_with(module, use, manager.state)


def test_component_is_not_kept_across_another_component_use():
    text_use = ComponentUse("stage", "text_encoder")
    stage = _Stage(
        text_use,
        ComponentUse("stage", "transformer"),
        ComponentUse("stage", "text_encoder", phase="second"),
    )
    module = torch.nn.Linear(2, 2)
    manager, server_args = _manager_for_stage(stage, {"text_encoder": module})
    strategy = Mock()
    manager.strategy_for = Mock(return_value=strategy)

    manager.before_stage(stage, 0, SimpleNamespace(is_warmup=False), server_args)
    manager.begin_use(text_use)
    manager.end_use(text_use)

    strategy.finish_use.assert_called_once_with(module, text_use, manager.state)
