from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

from sglang.srt.multimodal.processors.glm4v import _collapse_consecutive_token_ids


def test_collapse_consecutive_token_ids_preserves_distinct_placeholders():
    image_token_id = 99
    input_ids = [1, 99, 99, 99, 2, 99, 99, 3]

    assert _collapse_consecutive_token_ids(input_ids, image_token_id) == [
        1,
        99,
        2,
        99,
        3,
    ]


def test_collapse_consecutive_token_ids_handles_boundaries_and_empty_input():
    assert _collapse_consecutive_token_ids([99, 99, 1, 99, 99], 99) == [99, 1, 99]
    assert _collapse_consecutive_token_ids([], 99) == []
