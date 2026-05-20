from types import SimpleNamespace

import torch

from sglang.srt.configs.janus_pro import VLChatProcessor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_add_image_token_accepts_python_indices_with_special_token():
    processor = SimpleNamespace(
        add_special_token=True,
        image_start_id=101,
        image_id=99,
        image_end_id=102,
        num_image_tokens=2,
    )

    input_ids, num_image_tokens = VLChatProcessor.add_image_token(
        processor, [1, 3], torch.tensor([1, 99, 2, 99, 3], dtype=torch.long)
    )

    assert input_ids.tolist() == [1, 99, 101, 99, 99, 102, 2, 99, 101, 99, 99, 102, 3]
    assert num_image_tokens.tolist() == [2, 2]


def test_add_image_token_accepts_python_indices_without_special_token():
    processor = SimpleNamespace(
        add_special_token=False,
        image_start_id=101,
        image_id=99,
        image_end_id=102,
        num_image_tokens=2,
    )

    input_ids, num_image_tokens = VLChatProcessor.add_image_token(
        processor, [1, 3], torch.tensor([1, 99, 2, 99, 3], dtype=torch.long)
    )

    assert input_ids.tolist() == [1, 101, 99, 99, 102, 2, 101, 99, 99, 102, 3]
    assert num_image_tokens.tolist() == [2, 2]
