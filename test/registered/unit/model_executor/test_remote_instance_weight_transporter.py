from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.model_runner_components import (
    remote_instance_weight_transporter as transporter_module,
)
from sglang.srt.model_executor.model_runner_components.remote_instance_weight_transporter import (
    RemoteInstanceWeightTransporter,
)
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
)


@pytest.mark.parametrize("initial_weight_info", [None, {"model.weight": (1, 2, 3)}])
def test_publishes_weight_info_when_already_populated(monkeypatch, initial_weight_info):
    registered = []
    published = []
    generated_weight_info = {"generated.weight": (4, 5, 6)}
    server_args = SimpleNamespace(
        remote_instance_weight_loader_use_transfer_engine=lambda: True,
        remote_instance_weight_loader_backend=(
            RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE
        ),
    )
    transporter = RemoteInstanceWeightTransporter(
        server_args=server_args,
        get_model=lambda: object(),
        tp_rank=0,
        gpu_id=0,
        engine=object(),
        weight_info=initial_weight_info,
    )

    def register_memory_region(model, engine):
        registered.append((model, engine))
        return generated_weight_info

    monkeypatch.setattr(
        transporter_module, "register_memory_region", register_memory_region
    )
    monkeypatch.setattr(
        RemoteInstanceWeightTransporter,
        "_register_to_engine_info_bootstrap",
        lambda self: published.append(self.weight_info),
    )

    transporter.maybe_register_and_publish_weight_info()

    assert len(registered) == (1 if initial_weight_info is None else 0)
    assert published == [initial_weight_info or generated_weight_info]
