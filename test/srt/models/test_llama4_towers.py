import pytest

from sglang.srt.models.mllama4 import Llama4ForConditionalGeneration


def _build_stub() -> Llama4ForConditionalGeneration:
    # Bypass heavy initialization while keeping class-level tower metadata accessible.
    return object.__new__(Llama4ForConditionalGeneration)


def test_llama4_tower_names_exposed():
    model = _build_stub()

    assert model.get_mm_tower_names() == ["vision_model"]
    assert model.get_tower_name() == "vision_model"


def test_llama4_get_tower_name_out_of_range():
    model = _build_stub()

    with pytest.raises(IndexError):
        model.get_tower_name(1)
