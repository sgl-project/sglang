import pytest
import torch

from sglang.srt.utils import is_musa

if is_musa():
    import torchada  # noqa: F401


# This fixture ensures the torch defaults don't get left in modified states between
# tests (e.g., when a test fails before restoring the original value), which
# can cause subsequent tests to fail.
@pytest.fixture(autouse=True)
def reset_torch_defaults():
    orig_default_device = torch.get_default_device()
    orig_default_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(orig_default_dtype)
    torch.set_default_device(orig_default_device)
