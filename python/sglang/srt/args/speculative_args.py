# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CLI argument definitions for speculative decoding."""

import argparse


class SpeculativeArgs:
    """CLI argument definitions for speculative decoding."""

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.args.constants import (
            LOAD_FORMAT_CHOICES,
            SPECULATIVE_DRAFT_MODEL_QUANTIZATION_CHOICES,
            MOE_RUNNER_BACKEND_CHOICES,
            MOE_A2A_BACKEND_CHOICES,
        )

        # Speculative decoding
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM"],
            help="Speculative algorithm.",
        )
        parser.add_argument(
            "--speculative-draft-model-path",
            "--speculative-draft-model",
            type=str,
            help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--speculative-draft-model-revision",
            type=str,
            default=None,
            help="The specific draft model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        parser.add_argument(
            "--speculative-draft-load-format",
            type=str,
            default=ServerArgs.speculative_draft_load_format,
            choices=LOAD_FORMAT_CHOICES,
            help="The format of the draft model weights to load. "
            "If not specified, will use the same format as --load-format. "
            "Use 'dummy' to initialize draft model weights with random values for profiling.",
        )
        parser.add_argument(
            "--speculative-num-steps",
            type=int,
            help="The number of steps sampled from draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_steps,
        )
        parser.add_argument(
            "--speculative-eagle-topk",
            type=int,
            help="The number of tokens sampled from the draft model in eagle2 each step.",
            default=ServerArgs.speculative_eagle_topk,
        )
        parser.add_argument(
            "--speculative-num-draft-tokens",
            type=int,
            help="The number of tokens sampled from the draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_draft_tokens,
        )
        parser.add_argument(
            "--speculative-accept-threshold-single",
            type=float,
            help="Accept a draft token if its probability in the target model is greater than this threshold.",
            default=ServerArgs.speculative_accept_threshold_single,
        )
        parser.add_argument(
            "--speculative-accept-threshold-acc",
            type=float,
            help="The accept probability of a draft token is raised from its target probability p to min(1, p / threshold_acc).",
            default=ServerArgs.speculative_accept_threshold_acc,
        )
        parser.add_argument(
            "--speculative-token-map",
            type=str,
            help="The path of the draft model's small vocab table.",
            default=ServerArgs.speculative_token_map,
        )
        parser.add_argument(
            "--speculative-attention-mode",
            type=str,
            choices=["prefill", "decode"],
            help="Attention backend for speculative decoding operations (both target verify and draft extend). Can be one of 'prefill' (default) or 'decode'.",
            default=ServerArgs.speculative_attention_mode,
        )
        parser.add_argument(
            "--speculative-draft-attention-backend",
            type=str,
            help="Attention backend for speculative decoding drafting.",
            default=ServerArgs.speculative_draft_attention_backend,
        )
        parser.add_argument(
            "--speculative-moe-runner-backend",
            type=str,
            choices=MOE_RUNNER_BACKEND_CHOICES,
            default=ServerArgs.speculative_moe_runner_backend,
            help="Choose the runner backend for MoE in speculative decoding.",
        )
        parser.add_argument(
            "--speculative-moe-a2a-backend",
            type=str,
            choices=MOE_A2A_BACKEND_CHOICES,
            default=ServerArgs.speculative_moe_a2a_backend,
            help="Choose the backend for MoE A2A in speculative decoding",
        )
        parser.add_argument(
            "--speculative-draft-model-quantization",
            type=str,
            choices=SPECULATIVE_DRAFT_MODEL_QUANTIZATION_CHOICES,
            default=ServerArgs.speculative_draft_model_quantization,
            help="The quantization method for speculative model.",
        )

        # Speculative decoding (ngram)
        parser.add_argument(
            "--speculative-ngram-min-bfs-breadth",
            type=int,
            default=ServerArgs.speculative_ngram_min_bfs_breadth,
            help="The minimum breadth for BFS (Breadth-First Search) in ngram speculative decoding.",
        )
        parser.add_argument(
            "--speculative-ngram-max-bfs-breadth",
            type=int,
            default=ServerArgs.speculative_ngram_max_bfs_breadth,
            help="The maximum breadth for BFS (Breadth-First Search) in ngram speculative decoding.",
        )
        parser.add_argument(
            "--speculative-ngram-match-type",
            type=str,
            choices=["BFS", "PROB"],
            default=ServerArgs.speculative_ngram_match_type,
            help="The match type for cache tree.",
        )
        parser.add_argument(
            "--speculative-ngram-max-trie-depth",
            type=int,
            default=ServerArgs.speculative_ngram_max_trie_depth,
            help="The max trie depth for ngram speculative decoding.",
        )
        parser.add_argument(
            "--speculative-ngram-capacity",
            type=int,
            default=ServerArgs.speculative_ngram_capacity,
            help="The cache capacity for ngram speculative decoding.",
        )
        parser.add_argument(
            "--speculative-ngram-external-corpus-path",
            type=str,
            default=ServerArgs.speculative_ngram_external_corpus_path,
            help="Optional path to an external corpus used to build a read-only SAM for ngram speculative decoding.",
        )
        parser.add_argument(
            "--speculative-ngram-external-sam-budget",
            type=int,
            default=ServerArgs.speculative_ngram_external_sam_budget,
            help="Number of draft nodes reserved for the external SAM subtree in ngram speculative decoding.",
        )
        parser.add_argument(
            "--speculative-ngram-external-corpus-max-tokens",
            type=int,
            default=ServerArgs.speculative_ngram_external_corpus_max_tokens,
            help="Fail startup if the tokenized external ngram corpus exceeds this many tokens. Tune this based on your CPU memory budget.",
        )

        # Multi-layer Eagle speculative decoding
        parser.add_argument(
            "--enable-multi-layer-eagle",
            action="store_true",
            help="Enable multi-layer Eagle speculative decoding.",
        )
