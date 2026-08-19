"""Unit tests for sglang.srt.lora.utils — CPU-only, no server, no GPU.

Covers the pure-Python / tensor helper functions that can be exercised
without instantiating model weights or launching a distributed environment:

- get_normalized_target_modules
- get_stacked_multiply
- get_target_module_name
- merge_and_chunk_segments
- build_lm_head_pass_segments

Usage:
    python -m pytest test/registered/unit/lora/test_lora_utils.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import unittest

from sglang.srt.lora.utils import (
    build_lm_head_pass_segments,
    get_normalized_target_modules,
    get_stacked_multiply,
    get_target_module_name,
    merge_and_chunk_segments,
)


class TestGetNormalizedTargetModules(unittest.TestCase):
    def test_fused_qkv(self):
        """q_proj, k_proj, v_proj all map to qkv_proj."""
        result = get_normalized_target_modules(["q_proj", "k_proj", "v_proj"])
        self.assertEqual(result, {"qkv_proj"})

    def test_gate_up_proj(self):
        result = get_normalized_target_modules(["gate_proj", "up_proj"])
        self.assertEqual(result, {"gate_up_proj"})

    def test_o_proj_passthrough(self):
        result = get_normalized_target_modules(["o_proj"])
        self.assertEqual(result, {"o_proj"})

    def test_lm_head_aliases(self):
        for alias in ["lm_head", "output", "unembed_tokens"]:
            with self.subTest(alias=alias):
                self.assertEqual(get_normalized_target_modules([alias]), {"lm_head"})

    def test_embed_tokens_aliases(self):
        for alias in ["embed_tokens", "vocab_emb", "embeddings", "word_embeddings"]:
            with self.subTest(alias=alias):
                self.assertEqual(
                    get_normalized_target_modules([alias]), {"embed_tokens"}
                )

    def test_prefixed_module_names_stripped(self):
        """Names like 'feed_forward.gate_proj' should still normalize correctly."""
        result = get_normalized_target_modules(["feed_forward.gate_proj"])
        self.assertEqual(result, {"gate_up_proj"})

    def test_fused_qkv_a_proj_aliases(self):
        for alias in ["q_a_proj", "kv_a_proj_with_mqa"]:
            with self.subTest(alias=alias):
                self.assertEqual(
                    get_normalized_target_modules([alias]),
                    {"fused_qkv_a_proj_with_mqa"},
                )

    def test_all_sentinel_string(self):
        self.assertEqual(get_normalized_target_modules("all"), {"all"})

    def test_all_linear_sentinel_string(self):
        self.assertEqual(get_normalized_target_modules("all-linear"), {"all"})

    def test_invalid_string_raises(self):
        with self.assertRaises(ValueError):
            get_normalized_target_modules("q_proj")

    def test_mixed_modules(self):
        result = get_normalized_target_modules(["q_proj", "o_proj", "gate_proj"])
        self.assertEqual(result, {"qkv_proj", "o_proj", "gate_up_proj"})


class TestGetStackedMultiply(unittest.TestCase):
    def test_qkv_proj(self):
        self.assertEqual(get_stacked_multiply("qkv_proj"), 3)

    def test_gate_up_proj(self):
        self.assertEqual(get_stacked_multiply("gate_up_proj"), 2)

    def test_gate_up_proj_moe(self):
        self.assertEqual(get_stacked_multiply("gate_up_proj_moe"), 2)

    def test_in_proj(self):
        self.assertEqual(get_stacked_multiply("in_proj"), 2)

    def test_fused_qkv_a_proj(self):
        self.assertEqual(get_stacked_multiply("fused_qkv_a_proj_with_mqa"), 2)

    def test_non_stacked_returns_one(self):
        for name in ["o_proj", "down_proj", "lm_head", "embed_tokens"]:
            with self.subTest(name=name):
                self.assertEqual(get_stacked_multiply(name), 1)

    def test_in_proj_qkvz(self):
        self.assertEqual(get_stacked_multiply("in_proj_qkvz"), 4)

    def test_base_model_override(self):
        """Model can override the multiplier via get_stacked_multiply method."""

        class MockModel:
            def get_stacked_multiply(self, name):
                return 10

        self.assertEqual(get_stacked_multiply("qkv_proj", base_model=MockModel()), 10)


class TestGetTargetModuleName(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(
            get_target_module_name("qkv_proj", {"qkv_proj", "o_proj"}), "qkv_proj"
        )

    def test_substring_match(self):
        # "gate_up_proj" is substring of "model.layers.0.gate_up_proj"
        result = get_target_module_name(
            "model.layers.0.gate_up_proj", {"gate_up_proj", "o_proj"}
        )
        self.assertEqual(result, "gate_up_proj")

    def test_longest_match_wins(self):
        # Both "up_proj" and "gate_up_proj" are substrings — longer wins
        result = get_target_module_name("gate_up_proj", {"up_proj", "gate_up_proj"})
        self.assertEqual(result, "gate_up_proj")

    def test_no_match_raises(self):
        with self.assertRaises(ValueError):
            get_target_module_name("unknown_layer", {"qkv_proj", "o_proj"})


class TestMergeAndChunkSegments(unittest.TestCase):
    def test_no_merging_different_adapters(self):
        wi = [0, 1, 2]
        pl = [3, 3, 3]
        seg_wi, seg_lens = merge_and_chunk_segments(wi, pl, chunk_size=100)
        self.assertEqual(seg_wi, [0, 1, 2])
        self.assertEqual(seg_lens, [3, 3, 3])

    def test_consecutive_same_adapter_merged(self):
        wi = [0, 0, 1]
        pl = [4, 2, 3]
        seg_wi, seg_lens = merge_and_chunk_segments(wi, pl, chunk_size=100)
        self.assertEqual(seg_wi, [0, 1])
        self.assertEqual(seg_lens, [6, 3])

    def test_chunk_splits_large_segment(self):
        wi = [0]
        pl = [10]
        seg_wi, seg_lens = merge_and_chunk_segments(wi, pl, chunk_size=4)
        # 10 tokens → chunks of 4, 4, 2
        self.assertEqual(seg_wi, [0, 0, 0])
        self.assertEqual(seg_lens, [4, 4, 2])

    def test_merge_then_chunk(self):
        # Two consecutive same-adapter seqs totalling 7 tokens, chunk=4
        wi = [1, 1]
        pl = [3, 4]
        seg_wi, seg_lens = merge_and_chunk_segments(wi, pl, chunk_size=4)
        self.assertEqual(seg_wi, [1, 1])
        self.assertEqual(seg_lens, [4, 3])

    def test_exact_chunk_boundary_no_split(self):
        wi = [2]
        pl = [4]
        seg_wi, seg_lens = merge_and_chunk_segments(wi, pl, chunk_size=4)
        self.assertEqual(seg_wi, [2])
        self.assertEqual(seg_lens, [4])

    def test_empty_input(self):
        seg_wi, seg_lens = merge_and_chunk_segments([], [], chunk_size=8)
        self.assertEqual(seg_wi, [])
        self.assertEqual(seg_lens, [])

    def test_single_token_sequences(self):
        wi = [0, 0, 1, 0]
        pl = [1, 1, 1, 1]
        seg_wi, seg_lens = merge_and_chunk_segments(wi, pl, chunk_size=100)
        self.assertEqual(seg_wi, [0, 1, 0])
        self.assertEqual(seg_lens, [2, 1, 1])


class TestBuildLmHeadPassSegments(unittest.TestCase):
    def test_single_pass(self):
        """Total tokens ≤ chunk_size → one pass."""
        wi = [0, 1]
        pl = [3, 2]  # 5 tokens total
        passes = build_lm_head_pass_segments(wi, pl, logprobs_chunk_size=8)
        self.assertEqual(len(passes), 1)
        seg_wi, seg_lens = passes[0]
        self.assertEqual(seg_wi, [0, 1])
        self.assertEqual(seg_lens, [3, 2])

    def test_two_passes_same_adapter(self):
        """8 tokens from adapter 0, chunk=4 → 2 passes each with 1 segment."""
        wi = [0]
        pl = [8]
        passes = build_lm_head_pass_segments(wi, pl, logprobs_chunk_size=4)
        self.assertEqual(len(passes), 2)
        for seg_wi, seg_lens in passes:
            self.assertEqual(seg_wi, [0])
            self.assertEqual(seg_lens, [4])

    def test_pass_boundary_splits_adapter_segment(self):
        """Adapter 0: 3 tokens; Adapter 1: 3 tokens; chunk=4.
        Pass 0: tokens [0,1,2,3] → adapter 0 (3) + adapter 1 (1)
        Pass 1: tokens [4,5]     → adapter 1 (2)
        """
        wi = [0, 1]
        pl = [3, 3]
        passes = build_lm_head_pass_segments(wi, pl, logprobs_chunk_size=4)
        self.assertEqual(len(passes), 2)
        # Pass 0
        seg_wi0, seg_lens0 = passes[0]
        self.assertEqual(seg_wi0, [0, 1])
        self.assertEqual(seg_lens0, [3, 1])
        # Pass 1
        seg_wi1, seg_lens1 = passes[1]
        self.assertEqual(seg_wi1, [1])
        self.assertEqual(seg_lens1, [2])

    def test_exact_chunk_boundary(self):
        wi = [0, 1]
        pl = [2, 2]  # 4 tokens, chunk=4 → 1 pass
        passes = build_lm_head_pass_segments(wi, pl, logprobs_chunk_size=4)
        self.assertEqual(len(passes), 1)

    def test_empty_input(self):
        passes = build_lm_head_pass_segments([], [], logprobs_chunk_size=4)
        self.assertEqual(passes, [])

    def test_single_token_per_sequence(self):
        wi = [0, 1, 0, 1]
        pl = [1, 1, 1, 1]
        passes = build_lm_head_pass_segments(wi, pl, logprobs_chunk_size=2)
        self.assertEqual(len(passes), 2)
        # Pass 0: tokens [0→adapter0, 1→adapter1]
        seg_wi0, seg_lens0 = passes[0]
        self.assertEqual(seg_wi0, [0, 1])
        self.assertEqual(seg_lens0, [1, 1])
        # Pass 1: tokens [2→adapter0, 3→adapter1]
        seg_wi1, seg_lens1 = passes[1]
        self.assertEqual(seg_wi1, [0, 1])
        self.assertEqual(seg_lens1, [1, 1])


if __name__ == "__main__":
    unittest.main()
