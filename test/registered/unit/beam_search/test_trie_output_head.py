import sys

import pytest
import torch

from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_output_head import BeamTrieOutputHead
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config():
    return TrieOutputHeadConfig(
        tensor_path="/unused/trie_output_head.safetensors",
        token_start=8,
        codebook_size=4,
        num_codebooks=2,
    )


def test_full_projection_preserves_scores():
    torch.manual_seed(0)
    hidden_states = torch.randn(3, 5)
    weight = torch.randn(16, 5)
    depths = torch.tensor([0, 1, 0])

    def project(states, selected_weight):
        return torch.matmul(states, selected_weight.T)

    def transform(logits):
        return logits

    chunked = BeamTrieOutputHead(_config(), vocab_size=16, chunk_size=3)
    full = BeamTrieOutputHead(_config(), vocab_size=16, chunk_size=16)

    chunked_output = chunked.project_mixed(
        hidden_states,
        weight,
        depths,
        project,
        transform,
    )
    full_output = full.project_mixed(
        hidden_states,
        weight,
        depths,
        project,
        transform,
    )

    assert torch.equal(full_output.logits, chunked_output.logits)
    assert torch.allclose(
        full_output.normalizer,
        chunked_output.normalizer,
        rtol=1e-6,
        atol=1e-6,
    )


def _project(states, selected_weight):
    return torch.matmul(states, selected_weight.T)


def _identity(logits):
    return logits


def _normalizer_logsumexp(normalizer):
    return normalizer[:, 0] + normalizer[:, 1]


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 16, 100])
def test_streamed_normalizer_matches_full_vocab_logsumexp(chunk_size):
    torch.manual_seed(1)
    hidden_states = torch.randn(4, 5)
    weight = torch.randn(16, 5) * 4
    head = BeamTrieOutputHead(_config(), vocab_size=16, chunk_size=chunk_size)

    output = head.project_mixed(
        hidden_states, weight, torch.tensor([0, 1, 1, 0]), _project, _identity
    )

    expected = torch.logsumexp(hidden_states @ weight.T, dim=-1)
    torch.testing.assert_close(_normalizer_logsumexp(output.normalizer), expected)


def test_streamed_normalizer_is_stable_for_large_logits():
    hidden_states = torch.tensor([[1.0], [-1.0]])
    weight = torch.tensor([[500.0], [-700.0], [900.0], [100.0], [-900.0]] * 4)
    head = BeamTrieOutputHead(_config(), vocab_size=20, chunk_size=3)

    output = head.project_mixed(
        hidden_states, weight, torch.tensor([0, 1]), _project, _identity
    )

    expected = torch.logsumexp(hidden_states @ weight.T, dim=-1)
    assert torch.isfinite(output.normalizer).all()
    torch.testing.assert_close(_normalizer_logsumexp(output.normalizer), expected)


def test_normalizer_covers_only_the_model_vocabulary_and_applies_transform():
    torch.manual_seed(2)
    hidden_states = torch.randn(3, 5)
    padded_weight = torch.randn(24, 5)
    vocab_size = 18

    def transform(logits):
        return 3.0 * torch.tanh(logits / 3.0)

    head = BeamTrieOutputHead(_config(), vocab_size=vocab_size, chunk_size=5)
    output = head.project_mixed(
        hidden_states, padded_weight, torch.tensor([1, 0, 1]), _project, transform
    )

    expected = torch.logsumexp(
        transform(hidden_states @ padded_weight[:vocab_size].T), dim=-1
    )
    torch.testing.assert_close(_normalizer_logsumexp(output.normalizer), expected)


def test_normalizer_rejects_vocabulary_larger_than_lm_head():
    head = BeamTrieOutputHead(_config(), vocab_size=17, chunk_size=4)
    with pytest.raises(RuntimeError, match="exceeds LM-head rows"):
        head.project_mixed(
            torch.randn(1, 5),
            torch.randn(16, 5),
            torch.tensor([0]),
            _project,
            _identity,
        )


def test_projected_logits_select_the_row_depth_codebook():
    torch.manual_seed(3)
    config = _config()
    hidden_states = torch.randn(3, 5)
    weight = torch.randn(16, 5)
    depths = torch.tensor([1, 0, 1])
    head = BeamTrieOutputHead(config, vocab_size=16, chunk_size=6)

    output = head.project_mixed(hidden_states, weight, depths, _project, _identity)

    dense = hidden_states @ weight.T
    for row, depth in enumerate(depths.tolist()):
        start = config.token_start_for_depth(depth)
        torch.testing.assert_close(
            output.logits[row], dense[row, start : start + config.codebook_size]
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
