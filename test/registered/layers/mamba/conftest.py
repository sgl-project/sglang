import pytest

from sglang.srt.layers.attention.mamba.ops import ssu_dispatch
from sglang.srt.layers.attention.mamba.ops.ssu_dispatch import (
    initialize_mamba_selective_state_update_backend,
)
from sglang.srt.server_args import ServerArgs


@pytest.fixture(scope="session", autouse=True)
def _init_mamba_ssu_backend():
    """Initialize the Mamba SSU dispatch backend for the test session.

    In production this happens in Scheduler.init_mamba_backend(). Tests have no
    scheduler, so we do it here via the same public API.
    """
    initialize_mamba_selective_state_update_backend(ServerArgs(model_path="dummy"))
    yield
    ssu_dispatch._mamba_ssu_backend = None
