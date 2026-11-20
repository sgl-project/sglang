"""EAGLE and NGRAM verify apply a request's custom logit processor.

Speculative verify samples the target tokens itself instead of going through the
Sampler, so a custom logit processor reached only the prefill token: a processor
that forces one token was ignored on every verify round, and the drafts the raw
target logits favor were accepted."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor
from sglang.srt.sampling.sampling_batch_info import ProcessorEntry, SamplingBatchInfo
from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

VOCAB = 32
FORCED = 7
ROOT = 1


class _ForceToken(CustomLogitProcessor):
    def __call__(self, logits, custom_param_list):
        for i, params in enumerate(custom_param_list):
            logits[i, :] = -float("inf")
            logits[i, params["token_id"]] = 0.0
        return logits


def _chain():
    # root -> 2 -> 3; the target logits favor 2, 3, then the bonus 4
    return dict(
        tokens=[ROOT, 2, 3],
        next_token=[1, 2, -1],
        next_sibling=[-1, -1, -1],
        argmax=[2, 3, 4],
        accept=[2, 3, 4],
    )


def _tree():
    # root -> {2 -> 3, 5}; the target logits favor 2, 3, then the bonus 4
    return dict(
        tokens=[ROOT, 2, 5, 3],
        next_token=[1, 3, -1, -1],
        next_sibling=[-1, 2, -1, -1],
        argmax=[2, 3, 9, 4],
        accept=[2, 3, 4],
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


def _sampling_info(*, greedy):
    # Request 0 forces FORCED; request 1 has no processor.
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
        custom_params=[{"token_id": FORCED}, None],
        custom_logit_processor={
            0: ProcessorEntry(
                processor=_ForceToken(),
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
    def test_processor_forces_every_verify_row_of_its_request(self):
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
        for kind, layout, fused in cases:
            for greedy in (True, False):
                with (
                    self.subTest(kind=kind, fused=fused, greedy=greedy),
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
                        sampling_info=_sampling_info(greedy=greedy),
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
                    )
                    self.assertEqual(
                        _accept_tokens(*outputs), [[FORCED], layout["accept"]]
                    )


if __name__ == "__main__":
    unittest.main()
