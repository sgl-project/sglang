# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import torch
from safetensors.torch import safe_open, save_file

from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)


def test_safetensors_iterator_filters_before_yield(tmp_path) -> None:
    checkpoint = tmp_path / "weights.safetensors"
    save_file(
        {
            "keep.weight": torch.tensor([1.0]),
            "skip.weight": torch.tensor([2.0]),
        },
        checkpoint,
    )

    materialized: list[str] = []

    class TrackingSafeOpen:
        def __init__(self, path: str, framework: str, device: str) -> None:
            self._context = safe_open(path, framework=framework, device=device)

        def __enter__(self):
            self._file = self._context.__enter__()
            return self

        def __exit__(self, *args):
            return self._context.__exit__(*args)

        def keys(self):
            return self._file.keys()

        def get_tensor(self, name: str):
            materialized.append(name)
            return self._file.get_tensor(name)

    with (
        patch(
            "sglang.multimodal_gen.runtime.loader.weight_utils._validate_safetensors_file",
            return_value=True,
        ),
        patch(
            "sglang.multimodal_gen.runtime.loader.weight_utils."
            "_raise_if_duplicate_safetensors_keys"
        ),
        patch(
            "sglang.multimodal_gen.runtime.loader.weight_utils.safe_open",
            side_effect=TrackingSafeOpen,
        ),
    ):
        weights = list(
            safetensors_weights_iterator(
                [str(checkpoint)], key_filter=lambda name: name.startswith("keep.")
            )
        )

    assert [name for name, _ in weights] == ["keep.weight"]
    assert materialized == ["keep.weight"]
    torch.testing.assert_close(weights[0][1], torch.tensor([1.0]))


def test_filter_disables_streamer_to_skip_before_materialization(tmp_path) -> None:
    checkpoint = tmp_path / "weights.safetensors"
    save_file({"keep": torch.ones(1), "skip": torch.zeros(1)}, checkpoint)

    with patch(
        "sglang.multimodal_gen.runtime.loader.weight_utils.SafetensorsStreamer",
        side_effect=AssertionError("filtered iteration must not use the streamer"),
        create=True,
    ):
        weights = list(
            safetensors_weights_iterator(
                [str(checkpoint)],
                use_runai_model_streamer=True,
                key_filter=lambda name: name == "keep",
            )
        )

    assert [name for name, _ in weights] == ["keep"]
