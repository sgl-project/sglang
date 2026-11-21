"""EAGLE and NGRAM verify apply a request's custom logit processor on each row's
own history.

Speculative verify samples the target tokens itself instead of going through the
Sampler, so a custom logit processor reached only the prefill token. Applied at
verify, a processor that reads the output must see each row's draft prefix after
the committed tokens; one that does not declare it sees the first row only, and
its request commits one token per round."""

import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.sampling.custom_logit_processor import (
    ROW_PREFIX_KEY,
    CustomLogitProcessor,
)
from sglang.srt.sampling.sampling_batch_info import ProcessorEntry, SamplingBatchInfo
from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_utils import verify_row_prefixes
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

VOCAB = 32
ROOT = 1


class _CountUp(CustomLogitProcessor):
    """Forces the token after the last one the row follows, so every verify row
    of a draft path 2, 3 agrees only if it sees its own draft prefix."""

    supports_row_prefix = True

    def __call__(self, logits, custom_param_list):
        for i, params in enumerate(custom_param_list):
            req = params["__req__"]
            last = [*req.output_ids, *params.get(ROW_PREFIX_KEY, ())][-1]
            logits[i, :] = -float("inf")
            logits[i, last + 1] = 0.0
        return logits


class _CountUpWithoutRowPrefix(_CountUp):
    supports_row_prefix = False


def _chain():
    # root -> 2 -> 3; the target logits favor 2, 3, then the bonus 4
    return dict(
        tokens=[ROOT, 2, 3],
        next_token=[1, 2, -1],
        next_sibling=[-1, -1, -1],
        argmax=[2, 3, 4],
        row_prefixes=[[], [2], [2, 3]],
    )


def _tree():
    # root -> {2 -> 3, 5}; the target logits favor 2, 3, then the bonus 4
    return dict(
        tokens=[ROOT, 2, 5, 3],
        next_token=[1, 3, -1, -1],
        next_sibling=[-1, 2, -1, -1],
        argmax=[2, 3, 9, 4],
        row_prefixes=[[], [2], [5], [2, 3]],
    )


def _verify_input(layout, *, kind, bs):
    n = len(layout["tokens"])
    tensors = dict(
        draft_token=torch.tensor(layout["tokens"] * bs, device="cuda"),
        retrieve_index=torch.arange(bs * n, device="cuda").view(bs, n),
        retrieve_next_token=torch.tensor([layout["next_token"]] * bs, device="cuda"),
        retrieve_next_sibling=torch.tensor(
            [layout["next_sibling"]] * bs, device="cuda"
        ),
        custom_mask=torch.ones(bs * n * n, dtype=torch.bool, device="cuda"),
        positions=None,
    )
    if kind == "ngram":
        return NgramVerifyInput(draft_token_num=n, **tensors)
    return SimpleNamespace(
        draft_token_num=n,
        max_tree_depth=3,
        tree_topk=1 if kind == "eagle-chain" else 2,
        draft_probs=None,
        **tensors,
    )


def _logits(layout, bs, *, peak):
    n = len(layout["tokens"])
    logits = torch.zeros((bs * n, VOCAB), device="cuda")
    for b in range(bs):
        for i, token in enumerate(layout["argmax"]):
            logits[b * n + i, token] = peak
    return logits


def _sampling_info(processor, *, greedy):
    # Request 0 runs the processor after its committed output; request 1 has none.
    req = SimpleNamespace(output_ids=[5, ROOT])
    return SamplingBatchInfo(
        temperatures=torch.ones((2, 1), device="cuda"),
        top_ps=torch.ones(2, device="cuda"),
        top_ks=torch.full((2,), VOCAB, dtype=torch.int32, device="cuda"),
        min_ps=torch.zeros(2, device="cuda"),
        is_all_greedy=greedy,
        is_any_greedy=greedy,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=VOCAB,
        has_custom_logit_processor=True,
        custom_params=[{"__req__": req}, None],
        custom_logit_processor={
            0: ProcessorEntry(
                processor=processor,
                rows=[0],
                indices=torch.tensor([0], device="cuda"),
            )
        },
        device="cuda",
    )


def _accept_tokens(predict, accept_lens, accept_index):
    return [
        predict[accept_index[b, : accept_lens[b]].long()].tolist()
        for b in range(len(accept_lens))
    ]


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestVerifyCustomLogitProcessor(CustomTestCase):
    def test_processor_sees_each_row_prefix_or_commits_one_token(self):
        group = SimpleNamespace(world_size=1)
        parallel = SimpleNamespace(tp_group=group, attn_tp_group=group)
        spec = SimpleNamespace(
            speculative_use_rejection_sampling=False,
            speculative_use_block_verification=False,
            speculative_accept_threshold_single=1.0,
            speculative_accept_threshold_acc=1.0,
        )
        cases = [
            ("eagle-chain", _chain(), False),
            # SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION fuses the chain acceptance
            ("eagle-chain", _chain(), True),
            ("eagle-tree", _tree(), False),
            ("ngram", _tree(), False),
        ]
        # A row-prefix processor accepts the whole path; without the declaration,
        # its request keeps the first token only.
        processors = [(_CountUp(), [2, 3, 4]), (_CountUpWithoutRowPrefix(), [2])]
        for kind, layout, fused in cases:
            n = len(layout["tokens"])
            row_prefixes = verify_row_prefixes(
                torch.tensor([layout["next_token"]] * 2),
                torch.tensor([layout["next_sibling"]] * 2),
                torch.tensor([layout["tokens"]] * 2),
            )
            self.assertEqual(row_prefixes[:n], layout["row_prefixes"])
            for (processor, accept), greedy in itertools.product(
                processors, (True, False)
            ):
                with (
                    self.subTest(
                        kind=kind,
                        fused=fused,
                        greedy=greedy,
                        processor=type(processor).__name__,
                    ),
                    envs.SGLANG_ENABLE_NV_MEGA_RUNTIME_OPTIMIZATION.override(fused),
                    patch.object(eagle_utils, "get_spec", return_value=spec),
                    patch(
                        "sglang.srt.runtime_context.get_parallel",
                        return_value=parallel,
                    ),
                    patch(
                        "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                        return_value=False,
                    ),
                ):
                    batch = SimpleNamespace(
                        device="cuda",
                        forward_mode=SimpleNamespace(is_idle=lambda: False),
                        seq_lens=torch.tensor([7, 11], device="cuda"),
                        sampling_info=_sampling_info(processor, greedy=greedy),
                        req_pool_indices=torch.tensor([3, 5], device="cuda"),
                        return_logprob=False,
                    )
                    logits_output = SimpleNamespace(
                        # peaked so that temperature sampling is deterministic
                        next_token_logits=_logits(layout, 2, peak=50.0),
                        next_token_logits_vocab_shard=None,
                    )
                    outputs = eagle_utils.eagle_sample(
                        _verify_input(layout, kind=kind, bs=2),
                        batch,
                        logits_output,
                        row_prefixes=row_prefixes,
                    )
                    self.assertEqual(_accept_tokens(*outputs), [accept, [2, 3, 4]])


if __name__ == "__main__":
    unittest.main()
