import pytest
import torch

from sglang.srt.utils import get_device


@pytest.fixture(scope="session", autouse=True)
def setup_session():
    torch.set_default_device(get_device())
    yield
