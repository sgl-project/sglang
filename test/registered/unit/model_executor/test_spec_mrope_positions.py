"""mRoPE positions of a speculative forward when a ragged target verify packs a
different number of tokens per request."""

import types

import torch

from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    spec_mrope_token_deltas,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CPU = types.SimpleNamespace(device=torch.device("cpu"))


def _mrope_positions(*, positions, mm_inputs, verify_lens=None):
    layout = (
        None
        if verify_lens is None
        else RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=verify_lens,
            device=torch.device("cpu"),
            grid=[positions.numel()],
        )
    )
    forward_batch = types.SimpleNamespace(seq_lens=torch.zeros(len(mm_inputs)))
    batch = types.SimpleNamespace(
        multimodal_inputs=mm_inputs,
        spec_info=types.SimpleNamespace(
            positions=positions, ragged_verify_layout=layout
        ),
    )
    ForwardBatch.compute_spec_mrope_positions(forward_batch, _CPU, batch)
    return forward_batch.mrope_positions


def _image(delta):
    return types.SimpleNamespace(mrope_position_delta=torch.tensor([[delta]]))


def test_token_deltas_split_evenly_or_by_verify_len():
    deltas = torch.tensor([100, -5, 7])
    uniform = spec_mrope_token_deltas(deltas=deltas, num_tokens=12)
    assert uniform.tolist() == [100] * 4 + [-5] * 4 + [7] * 4

    # 8 packed tokens of 3 requests (8 % 3 != 0) in a 10-token graph tier.
    ragged = spec_mrope_token_deltas(
        deltas=deltas, num_tokens=10, verify_lens=torch.tensor([4, 1, 3])
    )
    assert ragged.tolist() == [100] * 4 + [-5] + [7] * 3 + [0, 0]


def test_text_only_ragged_verify():
    positions = torch.tensor([10, 11, 12, 20, 30, 31, 32])
    mrope = _mrope_positions(
        positions=positions, mm_inputs=[None, None, None], verify_lens=[3, 1, 3]
    )
    assert mrope.shape == (3, 7)
    assert (mrope == positions).all()


def test_multimodal_ragged_verify_adds_each_requests_delta():
    positions = torch.tensor([10, 11, 20, 30, 31, 32, 0, 0])
    mrope = _mrope_positions(
        positions=positions,
        mm_inputs=[_image(-4), None, _image(9)],
        verify_lens=[2, 1, 3],
    )
    expected = torch.tensor([6, 7, 20, 39, 40, 41, 0, 0])
    assert (mrope == expected).all()


def test_multimodal_uniform_block_is_unchanged():
    positions = torch.arange(6) + 100
    mrope = _mrope_positions(positions=positions, mm_inputs=[_image(3), None])
    assert mrope[0].tolist() == [103, 104, 105, 103, 104, 105]
