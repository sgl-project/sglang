import pytest
import torch


def test_change_torch_defaults():
    torch.set_default_device("cpu:0")
    torch.set_default_dtype(torch.float16)


def test_check_torch_defaults():
    assert torch.get_default_device() == torch.device("cpu")
    assert torch.get_default_dtype() == torch.float32


if __name__ == "__main__":
    pytest.main([__file__])
