from types import SimpleNamespace as NS
from unittest import TestCase, main
from unittest.mock import patch

import torch
from sglang.srt.managers import overlap_utils as overlap
from sglang.srt.speculative import eagle_disaggregation as disagg
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci


def gather_on_cpu(indices, topk_p, topk_index, tokens, hidden_states):
    return (
        topk_p[indices],
        topk_index[indices],
        tokens[indices],
        hidden_states[indices] if hidden_states is not None else None,
    )


register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class EagleDisaggTest(TestCase):
    def test_bootstrap_filter_merge_and_reused_slots(self):
        with (
            patch.object(overlap, "_is_cuda", False),
            patch.object(overlap, "_DEBUG_ASSERT", False),
            patch.object(overlap, "gather_spec_extras", gather_on_cpu),
            patch.object(overlap.FutureMap, "publish"),
            patch.object(torch.Tensor, "record_stream"),
            patch.object(
                torch, "get_device_module", return_value=NS(current_stream=lambda: None)
            ),
            patch(
                "sglang.srt.speculative.spec_utils.spec_need_hidden_states",
                return_value=True,
            ),
        ):
            for multi_layer in (False, True):
                for enable_overlap in (False, True):
                    with self.subTest(multi_layer=multi_layer, overlap=enable_overlap):
                        spec = NS(
                            speculative_eagle_topk=1,
                            speculative_num_steps=3,
                            enable_multi_layer_eagle=multi_layer,
                            speculative_use_rejection_sampling=True,
                        )
                        relay = overlap.FutureMap(
                            "cpu",
                            SpeculativeAlgorithm.EAGLE,
                            NS(req_to_token=torch.empty((8, 1))),
                        )

                        def bootstrap(
                            slots,
                            candidates,
                            spec=spec,
                            relay=relay,
                            enable_overlap=enable_overlap,
                        ):
                            batch = NS(
                                reqs=[
                                    NS(
                                        output_topk_p=[0.2] * len(tokens),
                                        output_topk_index=tokens,
                                        hidden_states_tensor=torch.ones(4),
                                        output_dsa_topk_indices=None,
                                    )
                                    for tokens in candidates
                                ],
                                req_pool_indices=torch.tensor(slots),
                                seq_lens=torch.full((len(slots),), 10),
                                device="cpu",
                                enable_overlap=enable_overlap,
                                model_config=NS(
                                    vocab_size=7, hf_config=NS(model_type="llama")
                                ),
                            )
                            with patch.object(disagg, "get_spec", return_value=spec):
                                return disagg.build_eagle_disagg_draft_input(
                                    batch,
                                    torch.ones(len(slots), dtype=torch.int64),
                                    relay,
                                )

                        def resolve(draft, relay=relay, enable_overlap=enable_overlap):
                            if enable_overlap:
                                relay._resolve_spec_extras(NS(spec_info=draft))

                        candidates = (
                            [[2, 3, 4], [4, 5, 6]] if multi_layer else [[2], [4]]
                        )
                        draft = bootstrap([1, 5], candidates)
                        expected = torch.nn.functional.one_hot(
                            torch.tensor(candidates), num_classes=7
                        ).float()
                        if not multi_layer:
                            expected = expected.squeeze(1)
                        torch.testing.assert_close(draft.draft_probs, expected)
                        torch.testing.assert_close(
                            draft.topk_p, torch.ones_like(draft.topk_p)
                        )

                        # Normal draft-extend replaces bootstrap delta q with full q.
                        probs = (
                            torch.arange(1, expected.numel() + 1)
                            .reshape(expected.shape)
                            .float()
                        )
                        probs /= probs.sum(dim=-1, keepdim=True)
                        draft.draft_probs = probs
                        if enable_overlap:
                            relay.stash(
                                draft.future_indices,
                                overlap.RelayPayload.from_draft_input(draft),
                            )
                        draft.filter_batch(torch.tensor([1]))
                        resolve(draft)
                        torch.testing.assert_close(draft.draft_probs, probs[1:])

                        # Mix an active request with a fresh slot and a recycled slot.
                        for slot in (3, 1):
                            incoming = bootstrap(
                                [slot], [[6, 2, 1]] if multi_layer else [[6]]
                            )
                            expected = torch.cat(
                                [draft.draft_probs, incoming.draft_probs]
                            )
                            draft.merge_batch(incoming)
                            resolve(draft)
                            torch.testing.assert_close(draft.draft_probs, expected)

                        spec.speculative_use_rejection_sampling = False
                        disabled = bootstrap([2], [[1, 2, 3]] if multi_layer else [[1]])
                        self.assertIsNone(disabled.draft_probs)
                        torch.testing.assert_close(
                            disabled.topk_p, torch.full_like(disabled.topk_p, 0.2)
                        )


if __name__ == "__main__":
    main()
