from types import SimpleNamespace
from unittest.mock import patch

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
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_transporter(weight_info):
    return RemoteInstanceWeightTransporter(
        get_model=lambda: object(),
        tp_rank=0,
        gpu_id=0,
        engine=object(),
        weight_info=weight_info,
    )


def _runtime_model(*, backend, start_seed_via_transfer_engine):
    return SimpleNamespace(
        remote_instance_weight_loader_backend=backend,
        remote_instance_weight_loader_start_seed_via_transfer_engine=(
            start_seed_via_transfer_engine
        ),
    )


@pytest.mark.parametrize(
    ("backend", "start_seed_via_transfer_engine", "should_publish"),
    [
        pytest.param(
            RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE,
            True,
            True,
            id="start-seed-publishes-existing-registration",
        ),
        pytest.param(
            RemoteInstanceWeightLoaderBackend.TRANSFER_ENGINE,
            False,
            False,
            id="non-start-seed-does-not-publish",
        ),
        pytest.param(
            RemoteInstanceWeightLoaderBackend.MODELEXPRESS,
            True,
            False,
            id="modelexpress-is-a-no-op",
        ),
    ],
)
def test_pre_registered_weight_info_publish_contract(
    backend, start_seed_via_transfer_engine, should_publish
):
    weight_info = {"weight": (1, 2, 3)}
    transporter = _make_transporter(weight_info)
    runtime_model = _runtime_model(
        backend=backend,
        start_seed_via_transfer_engine=start_seed_via_transfer_engine,
    )

    with (
        patch.object(transporter_module, "get_model", return_value=runtime_model),
        patch.object(
            transporter_module,
            "remote_instance_transfer_engine_enabled",
            return_value=True,
        ),
        patch.object(transporter_module, "register_memory_region") as register,
        patch.object(
            RemoteInstanceWeightTransporter,
            "_register_to_engine_info_bootstrap",
        ) as publish,
    ):
        transporter.maybe_register_and_publish_weight_info()

    register.assert_not_called()
    if should_publish:
        publish.assert_called_once_with()
    else:
        publish.assert_not_called()
    assert transporter.weight_info is weight_info
