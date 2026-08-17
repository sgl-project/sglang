# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    _copy_preplanned_decode_inputs,
    _refresh_decode_mrope_positions,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_preplanned_decode_refreshes_mrope_positions() -> None:
    buffers = SimpleNamespace(
        input_ids=torch.zeros(4, dtype=torch.long),
        positions=torch.zeros(4, dtype=torch.long),
        mrope_positions=torch.full((3, 4), -1, dtype=torch.long),
    )
    forward_batch = SimpleNamespace(
        batch_size=2,
        input_ids=torch.tensor([11, 12]),
        positions=torch.tensor([21, 22]),
        mrope_positions=torch.tensor(
            [
                [31, 32],
                [41, 42],
                [51, 52],
            ]
        ),
        mm_inputs=[object(), object()],
    )

    _copy_preplanned_decode_inputs(buffers, forward_batch, raw_num_token=2)

    assert buffers.input_ids.tolist() == [11, 12, 0, 0]
    assert buffers.positions.tolist() == [21, 22, 0, 0]
    assert buffers.mrope_positions.tolist() == [
        [31, 32, -1, -1],
        [41, 42, -1, -1],
        [51, 52, -1, -1],
    ]


def test_decode_synthesizes_text_mrope_from_positions() -> None:
    buffers = SimpleNamespace(
        mrope_positions=torch.full((3, 4), -1, dtype=torch.long),
    )
    forward_batch = SimpleNamespace(
        batch_size=2,
        positions=torch.tensor([21, 22]),
        mrope_positions=None,
        mm_inputs=[None, None],
    )

    _refresh_decode_mrope_positions(
        buffers,
        forward_batch,
        raw_num_token=2,
        temporal_only_text_mrope=True,
    )

    assert buffers.mrope_positions.tolist() == [
        [21, 22, -1, -1],
        [0, 0, -1, -1],
        [0, 0, -1, -1],
    ]


def test_decode_zeros_only_text_request_spatial_mrope() -> None:
    buffers = SimpleNamespace(
        mrope_positions=torch.full((3, 4), -1, dtype=torch.long),
    )
    forward_batch = SimpleNamespace(
        batch_size=2,
        positions=torch.tensor([21, 22]),
        mrope_positions=torch.tensor(
            [
                [31, 32],
                [41, 42],
                [51, 52],
            ]
        ),
        mm_inputs=[None, object()],
    )

    _refresh_decode_mrope_positions(
        buffers,
        forward_batch,
        raw_num_token=2,
        temporal_only_text_mrope=True,
    )

    assert buffers.mrope_positions.tolist() == [
        [31, 32, -1, -1],
        [0, 42, -1, -1],
        [0, 52, -1, -1],
    ]


def test_decode_bulk_zeros_all_text_spatial_mrope() -> None:
    buffers = SimpleNamespace(
        mrope_positions=torch.full((3, 4), -1, dtype=torch.long),
    )
    forward_batch = SimpleNamespace(
        batch_size=2,
        positions=torch.tensor([21, 22]),
        mrope_positions=torch.tensor(
            [
                [31, 32],
                [41, 42],
                [51, 52],
            ]
        ),
        mm_inputs=[None, None],
    )

    _refresh_decode_mrope_positions(
        buffers,
        forward_batch,
        raw_num_token=2,
        temporal_only_text_mrope=True,
    )

    assert buffers.mrope_positions.tolist() == [
        [31, 32, -1, -1],
        [0, 0, -1, -1],
        [0, 0, -1, -1],
    ]
