import sys

import pytest
import torch

from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_output_head import BeamTrieOutputHead
from sglang.srt.environ import envs
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


def test_topk_logprob_skips_full_vocabulary_normalizer():
    torch.manual_seed(0)
    hidden_states = torch.randn(3, 5)
    weight = torch.randn(16, 5)
    depths = torch.tensor([0, 1, 0])
    projected_widths = []

    def project(states, selected_weight):
        projected_widths.append(selected_weight.shape[0])
        return torch.matmul(states, selected_weight.T)

    with envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB.override(True):
        output = BeamTrieOutputHead(_config(), vocab_size=16).project_mixed(
            hidden_states,
            weight,
            depths,
            project,
            lambda logits: logits,
        )

    assert output.normalizer is None
    assert projected_widths == [8]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
