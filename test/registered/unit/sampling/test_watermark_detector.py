import ast
import math
import sys
from pathlib import Path

import pytest
import torch

import sglang.srt.sampling.watermarking.config as config_module
import sglang.srt.sampling.watermarking.detector as detector_module
from sglang.srt.sampling.watermarking import (
    WatermarkDetector,
    WatermarkStatistics,
    detect,
    parse_watermark_key,
)
from sglang.srt.sampling.watermarking.core import (
    _dual_key_a_mask_torch,
    _hash_contexts,
    _watermark_hash32_torch,
)
from sglang.srt.sampling.watermarking.detector import (
    _gamma_log_survival,
    hash_context,
    position_coin,
    watermark_hash,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_KEY_A = "0123456789abcdef"
_KEY_B = "fedcba9876543210"
# Known-answer vectors shared with the CUDA selector test and the documented contract.
_CONTEXTS = [[1, 2, 3, 4], [4, 3, 2, 1]]
_VECTOR_KEYS = [_KEY_A, _KEY_B]
_VECTOR_KEYS_B = ["1111222233334444", "9999aaaabbbbcccc"]
_TOKEN_IDS = [0, 1, 17, 8191, 8192, 16396]
_EXPECTED_CONTEXT_HASHES = [1145416960, 47748951]
_EXPECTED_TOKEN_HASHES = [
    [1293512163, 858402549, 2305555132, 2450309311, 227333036, 2684237202],
    [221641642, 4015047244, 843143906, 1076944989, 3882127500, 2413234263],
]
_EXPECTED_KEY_A_MASK = [False, True]


def test_hash_matches_server_vectors():
    for index, context in enumerate(_CONTEXTS):
        key = parse_watermark_key(_VECTOR_KEYS[index])
        context_hash = hash_context(context)
        assert context_hash == _EXPECTED_CONTEXT_HASHES[index]
        assert [
            watermark_hash(key, context_hash, token_id) for token_id in _TOKEN_IDS
        ] == _EXPECTED_TOKEN_HASHES[index]
        coin = position_coin(
            key, parse_watermark_key(_VECTOR_KEYS_B[index]), context_hash
        )
        assert (coin < (1 << 31)) == _EXPECTED_KEY_A_MASK[index]


def test_hash_matches_torch_reference_for_variable_length_contexts():
    generator = torch.Generator().manual_seed(37577)
    num_rows = 64
    contexts = torch.randint(0, 2**31, (num_rows, 8), generator=generator)
    lengths = torch.randint(1, 9, (num_rows,), generator=generator, dtype=torch.int32)
    keys = torch.randint(-(2**63), 2**63 - 1, (num_rows,), generator=generator)
    keys_b = torch.randint(-(2**63), 2**63 - 1, (num_rows,), generator=generator)
    token_ids = torch.tensor([0, 1, 8191, 2**31 - 1])

    context_hashes = _hash_contexts(contexts, lengths)
    token_hashes = _watermark_hash32_torch(keys, context_hashes, token_ids)
    key_a_mask = _dual_key_a_mask_torch(
        keys, keys_b, context_hashes, torch.full((num_rows,), 1 << 31)
    )
    for row in range(num_rows):
        context_hash = hash_context(contexts[row, : lengths[row]].tolist())
        key, key_b = keys[row].item(), keys_b[row].item()
        assert context_hash == context_hashes[row].item()
        assert [
            watermark_hash(key, context_hash, token_id)
            for token_id in token_ids.tolist()
        ] == token_hashes[row].tolist()
        coin = position_coin(key, key_b, context_hash)
        assert (coin < (1 << 31)) == key_a_mask[row].item()


def test_prompt_alignment_and_midpoint_score():
    result = detect([17], key=_KEY_A, prompt_token_ids=[9, 1, 2, 3, 4])
    expected = -math.log1p(-(2305555132 + 0.5) / (1 << 32))
    assert result.combined.num_contexts == 1
    assert result.combined.score == expected
    assert result.combined.p_value == pytest.approx(math.exp(-expected))

    detector = WatermarkDetector(_KEY_A)
    no_prompt = detector.detect_tokens([1, 2, 3, 4, 17])
    assert no_prompt.combined == result.combined
    assert no_prompt.skipped_initial_tokens == 4
    short_prompt = detector.detect_tokens([2, 3], prompt_token_ids=[1])
    assert short_prompt.combined.num_contexts == 2
    assert short_prompt.skipped_initial_tokens == 0


def test_repeats_do_not_add_evidence_or_spend_prefix_budget():
    detector = WatermarkDetector(_KEY_A, context_window=1, max_contexts=3)
    first = detector.detect_tokens([1, 2, 1, 3, 4])
    repeated = detector.detect_tokens([1, 2, 1, 2, 3, 4, 999, 0])
    assert repeated.combined == first.combined
    assert repeated.repeated_contexts == 2
    assert repeated.tokens_examined == 6
    assert repeated.prefix_limit_reached


def test_prefix_cap_applies_before_dual_partition_and_ignores_tail():
    detector = WatermarkDetector(_KEY_A, key_b=_KEY_B)
    prefix = list(range(4100))
    result = detector.detect_tokens(prefix)
    assert result.combined.num_contexts == 4096
    assert (
        result.key_a_partition.num_contexts + result.key_b_partition.num_contexts
        == 4096
    )
    assert result == detector.detect_tokens(prefix + list(range(10000, 11000)))


def test_exact_tuples_remain_distinct_on_hash_collision():
    contexts = [[55826, 167495], [62387, 187178]]
    assert hash_context(contexts[0]) == hash_context(contexts[1])
    result = WatermarkDetector(_KEY_A, context_window=2).detect_tokens(
        [*contexts[0], 17, *contexts[1], 18]
    )
    assert result.combined.num_contexts == 4


def test_gamma_tail_and_log_p_underflow():
    assert math.exp(_gamma_log_survival(2.0, 3)) == pytest.approx(5 * math.exp(-2))
    assert math.exp(_gamma_log_survival(4096.0, 4096)) == pytest.approx(
        0.4979221728062154, abs=1e-10
    )
    underflow = WatermarkStatistics.from_scores([1000.0])
    assert underflow.p_value == 0.0
    assert underflow.log_p_value == -1000.0


def _generate_watermarked_tokens(key_a, key_b=None, mixing_probability=0.5):
    key_a = parse_watermark_key(key_a)
    key_b = parse_watermark_key(key_b) if key_b is not None else None
    threshold = int(mixing_probability * (1 << 32))
    tokens = [100, 200, 300, 400]
    for _ in range(256):
        context_hash = hash_context(tokens[-4:])
        key = key_a
        if key_b is not None and position_coin(key_a, key_b, context_hash) >= threshold:
            key = key_b
        tokens.append(
            max(range(64), key=lambda token: watermark_hash(key, context_hash, token))
        )
    return tokens


def test_dual_partition_and_cross_key_isolation():
    tokens = _generate_watermarked_tokens(_KEY_A, _KEY_B, mixing_probability=0.3)
    result = WatermarkDetector(
        _KEY_A, key_b=_KEY_B, mixing_probability=0.3
    ).detect_tokens(tokens)
    assert result.key_a_partition.p_value < 1e-4
    assert result.key_b_partition.p_value < 1e-4
    assert result.combined.p_value < 1e-4
    assert (
        result.key_a_all_positions
        == WatermarkDetector(_KEY_A).detect_tokens(tokens).combined
    )
    assert result.key_a_all_positions.z_score < result.key_a_partition.z_score
    assert WatermarkDetector("deadbeef").detect_tokens(tokens).combined.p_value > 1e-4

    single = _generate_watermarked_tokens(_KEY_A)
    assert WatermarkDetector(_KEY_A).detect_tokens(single).combined.p_value < 1e-4
    assert WatermarkDetector(_KEY_B).detect_tokens(single).combined.p_value > 1e-4


def test_coin_threshold_is_strict_and_empty_partition_is_neutral():
    key_a, key_b = _VECTOR_KEYS[0], _VECTOR_KEYS_B[0]
    coin = position_coin(
        parse_watermark_key(key_a),
        parse_watermark_key(key_b),
        hash_context(_CONTEXTS[0]),
    )
    at_boundary = WatermarkDetector(
        key_a, key_b=key_b, mixing_probability=coin / (1 << 32)
    ).detect_tokens([17], prompt_token_ids=_CONTEXTS[0])
    above_boundary = WatermarkDetector(
        key_a, key_b=key_b, mixing_probability=(coin + 1) / (1 << 32)
    ).detect_tokens([17], prompt_token_ids=_CONTEXTS[0])
    assert at_boundary.key_a_partition.num_contexts == 0
    assert at_boundary.key_a_partition.p_value == 1.0
    assert at_boundary.combined == at_boundary.key_b_all_positions
    assert above_boundary.key_b_partition.num_contexts == 0
    assert above_boundary.combined == above_boundary.key_a_all_positions


@pytest.mark.parametrize(
    "kwargs",
    [
        {"key": "1" * 17},
        {"key": "0x"},
        {"key": "+1"},
        {"context_window": 0},
        {"context_window": 65},
        {"context_window": True},
        {"max_contexts": 0},
        {"max_contexts": 4097},
        {"mixing_probability": 0},
        {"mixing_probability": 1},
        {"mixing_probability": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        WatermarkDetector(**({"key": _KEY_A} | kwargs))


@pytest.mark.parametrize("token_ids", [[-1], [1.5], [True], [1 << 32]])
def test_invalid_token_ids_are_rejected(token_ids):
    with pytest.raises(ValueError, match="token IDs"):
        WatermarkDetector(_KEY_A).detect_tokens(token_ids)


@pytest.mark.parametrize("module", [detector_module, config_module])
def test_detector_dependencies_do_not_import_torch(module):
    tree = ast.parse(Path(module.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "torch" not in imported


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
