import torch

from sglang.srt.models.mistral import Mistral3ForConditionalGeneration
from sglang.srt.models.utils import TowerAwareMixin


class _DummyInnerModel(torch.nn.Module, TowerAwareMixin):
    tower_names = ("vision_tower",)

    def __init__(self):
        super().__init__()
        self._dummy_param = torch.nn.Parameter(torch.zeros(1))


def _build_wrapper() -> Mistral3ForConditionalGeneration:
    wrapper = object.__new__(Mistral3ForConditionalGeneration)
    wrapper.inner = _DummyInnerModel()
    return wrapper


def test_mistral3_get_mm_tower_names_delegates_to_inner():
    wrapper = _build_wrapper()

    # The wrapper should expose tower names via the inner model.
    assert wrapper.get_mm_tower_names() == ["vision_tower"]

    # Ensure attribute access falls through to the inner nn.Module instance.
    assert list(wrapper.parameters())[0] is wrapper.inner._dummy_param
