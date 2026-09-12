from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.models import moss_vl
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _model():
    model = object.__new__(moss_vl.MossVLForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.spatial_merge_size = 1
    model.is_mrope_enabled = True
    model._get_vision_features = Mock(side_effect=lambda pixels, grid: pixels)
    model._insert_separator_tokens = Mock(side_effect=lambda features, grid: features)
    model.language_model = Mock(side_effect=lambda **kwargs: kwargs["positions"])
    model.logits_processor = Mock(side_effect=lambda ids, hidden, *args: hidden)
    return model


def _mm_input(device="cpu"):
    item = SimpleNamespace(
        feature=torch.arange(8, dtype=torch.float32, device=device).reshape(4, 2),
        grid_thw=torch.tensor([[2, 1, 1]]),
        acknowledge_deferred_cuda_ipc_feature=lambda: None,
    )
    return moss_vl.MultimodalInputs(
        mm_items=[item],
        vision_position_ids=torch.arange(4, device=device).repeat(3, 1),
        visible_frame_counts=torch.tensor([0, 1, 2]),
        mrope_positions=torch.tensor([[4, 5, 6]]).repeat(3, 1),
        mrope_position_delta=torch.tensor([4]),
    )


def _batch(mm, length=3, prefix=0, device="cpu"):
    return SimpleNamespace(
        batch_size=1,
        forward_mode=SimpleNamespace(is_decode=lambda: False, is_extend=lambda: True),
        encoder_cached=[False],
        encoder_lens=torch.tensor([4], device=device),
        encoder_lens_cpu=[4],
        seq_lens=torch.tensor([prefix + length], device=device),
        extend_seq_lens=torch.tensor([length], device=device),
        extend_seq_lens_cpu=[length],
        extend_prefix_lens_cpu=[prefix],
        mm_inputs=[mm],
        mrope_positions=mm.mrope_positions[:, prefix : prefix + length].to(device),
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_repeated_prefill_preserves_vision_inputs_and_prompt_visibility(
    monkeypatch, device
):
    monkeypatch.setattr(moss_vl, "get_is_capture_mode", lambda: False)
    model, mm = _model(), _mm_input(device)
    original_feature = mm.mm_items[0].feature.cpu().clone()
    original_positions = mm.vision_position_ids.cpu().clone()

    # Repeat encoder computation as a retracted request accumulates output tokens.
    for length in (3, 5, 7):
        batch = _batch(mm, length=length, device=device)
        model.prepare_forward_batch(batch)
        expected = torch.ones(length, 4, dtype=torch.uint8, device=device)
        expected[0] = 0
        expected[1, 2:] = 0
        torch.testing.assert_close(
            batch.cross_attention_custom_mask.reshape(length, 4), expected
        )

        result = model.forward(
            torch.arange(length, device=device), batch.mrope_positions, batch
        )
        torch.testing.assert_close(
            result.cpu(), torch.arange(4, 4 + length).repeat(3, 1)
        )
        assert mm.mm_items[0].feature.device.type == "cpu"
        torch.testing.assert_close(mm.mm_items[0].feature, original_feature)
        torch.testing.assert_close(mm.vision_position_ids, original_positions)
        torch.testing.assert_close(mm.visible_frame_counts, torch.tensor([0, 1, 2]))
        assert model._get_vision_features.call_args.args[0].device.type == device

    # Normal request cleanup must still be able to release the retained inputs.
    mm.release_features()
    assert mm.mm_items[0].feature is None


@pytest.mark.parametrize("prefix,length", [(0, 5), (1, 4), (3, 2), (5, 2)])
def test_reprefill_mask_and_positions_with_cached_text_prefix(prefix, length):
    model, mm = _model(), _mm_input()
    batch = _batch(mm, length=length, prefix=prefix)
    model.prepare_forward_batch(batch)
    expected_counts = torch.tensor([0, 1, 2, 2, 2, 2, 2])[prefix : prefix + length]
    expected = torch.arange(4)[None, :] // 2 < expected_counts[:, None]
    torch.testing.assert_close(
        batch.cross_attention_custom_mask.reshape(length, 4).bool(), expected
    )
    row_mask = model.get_full_text_row_masked_out_mask(batch)
    torch.testing.assert_close(row_mask[:, 0], expected_counts > 0)
    positions = model._repair_mrope_positions(batch)
    torch.testing.assert_close(
        positions, torch.arange(4 + prefix, 4 + prefix + length).repeat(3, 1)
    )


def test_mixed_reprefill_preserves_per_request_mask_offsets():
    model = _model()
    first, second = _mm_input(), _mm_input()
    batch = _batch(first, length=2, prefix=3)
    batch.batch_size = 3
    batch.mm_inputs = [first, None, second]
    batch.encoder_lens_cpu = [4, 0, 4]
    batch.extend_seq_lens_cpu = [2, 2, 2]
    batch.extend_prefix_lens_cpu = [3, 5, 1]
    mask = model._build_cross_attention_custom_mask(batch)
    expected = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.uint8
    )
    torch.testing.assert_close(mask, expected.flatten())
    positions = model._repair_mrope_positions(batch)
    torch.testing.assert_close(
        positions, torch.tensor([[7, 8, 5, 6, 5, 6]]).repeat(3, 1)
    )


def test_cached_encoder_does_not_require_released_features():
    model, mm = _model(), _mm_input()
    mm.release_features()
    batch = _batch(mm)
    batch.encoder_cached = [True]
    assert model._collect_mm_data(batch) == (None, None, None)
    batch.encoder_cached = [False]
    with pytest.raises(RuntimeError, match="without its vision features"):
        model._collect_mm_data(batch)


@pytest.mark.parametrize("is_extend,lengths", [(False, [3]), (True, None), (True, [3])])
def test_mrope_fast_path_preserves_existing_tensor(is_extend, lengths):
    model, mm = _model(), _mm_input()
    batch = _batch(mm)
    batch.forward_mode.is_extend = lambda: is_extend
    batch.extend_seq_lens_cpu = lengths
    assert model._repair_mrope_positions(batch) is batch.mrope_positions


def test_short_mrope_table_requires_position_delta():
    model, mm = _model(), _mm_input()
    mm.mrope_position_delta = None
    with pytest.raises(ValueError, match="mrope_position_delta"):
        model._repair_mrope_positions(_batch(mm, length=5))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
