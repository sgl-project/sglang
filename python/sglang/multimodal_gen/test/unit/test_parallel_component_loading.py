"""CPU unit tests for parallel heavy-component loading (#19092)."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from sglang.multimodal_gen.runtime.managers.memory_managers.component_loading_order import (
    ComponentLoadSpec,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)


def _spec(name: str, index: int) -> ComponentLoadSpec:
    return ComponentLoadSpec(
        module_name=name,
        load_module_name=name,
        component_model_path=f"/tmp/{name}",
        transformers_or_diffusers="diffusers",
        architecture=None,
        index=index,
    )


class _FakePipeline(ComposedPipelineBase):
    """Minimal concrete pipeline for exercising the load helpers."""

    _required_config_modules = ["transformer", "text_encoder", "vae"]

    def __init__(self) -> None:
        # Bypass ComposedPipelineBase.__init__ (downloads / full server setup).
        self.component_loaders = {}
        self.memory_usages = {}

    def create_pipeline_stages(self, server_args):  # pragma: no cover - unused
        return


def test_load_component_specs_overlaps_heavy_loads():
    pipeline = _FakePipeline()
    server_args = SimpleNamespace(
        resolve_component_attention_backend=lambda *a, **k: (None, None),
        resolve_component_backend_by_role=lambda *a, **k: None,
    )
    active = 0
    max_active = 0
    lock = threading.Lock()
    load_started = {name: threading.Event() for name in ("transformer", "text_encoder")}

    def fake_load_component(**kwargs):
        nonlocal active, max_active
        name = kwargs["component_name"]
        with lock:
            active += 1
            max_active = max(max_active, active)
        load_started[name].set()
        # Wait until the peer heavy load has also entered the loader so the
        # test proves overlap rather than lucky scheduling.
        peer = "text_encoder" if name == "transformer" else "transformer"
        assert load_started[peer].wait(timeout=2.0)
        time.sleep(0.05)
        with lock:
            active -= 1
        return SimpleNamespace(name=name), 1.0

    loaded: dict[str, Any] = {}
    with (
        patch.object(
            ComposedPipelineBase,
            "_pin_loader_thread_device",
            return_value=None,
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base."
            "PipelineComponentLoader.load_component",
            side_effect=fake_load_component,
        ),
    ):
        pipeline._load_component_specs(
            [_spec("transformer", 0), _spec("text_encoder", 1)],
            server_args,
            loaded,
            parallel=True,
            desc="Loading heavy components",
        )

    assert set(loaded) == {"transformer", "text_encoder"}
    assert max_active >= 2
    assert pipeline.memory_usages == {"transformer": 1.0, "text_encoder": 1.0}


def test_load_component_specs_stays_serial_for_single_heavy_module():
    pipeline = _FakePipeline()
    server_args = SimpleNamespace(
        resolve_component_attention_backend=lambda *a, **k: (None, None),
        resolve_component_backend_by_role=lambda *a, **k: None,
    )
    threads: list[int] = []

    def fake_load_component(**kwargs):
        threads.append(threading.get_ident())
        return SimpleNamespace(name=kwargs["component_name"]), 0.5

    loaded: dict[str, Any] = {}
    with (
        patch.object(
            ComposedPipelineBase,
            "_pin_loader_thread_device",
            return_value=None,
        ),
        patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base."
            "PipelineComponentLoader.load_component",
            side_effect=fake_load_component,
        ),
    ):
        pipeline._load_component_specs(
            [_spec("transformer", 0)],
            server_args,
            loaded,
            parallel=True,
            desc="Loading heavy components",
        )

    assert list(loaded) == ["transformer"]
    assert threads == [threading.get_ident()]
