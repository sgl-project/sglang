"""Unit tests for deterministic EAGLE draft proposals and speculative softmax."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative import eagle_utils
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_utils import renorm_draft_probs, sample_draft_proposal
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEagleDraftSamplingPosition(CustomTestCase):
    def test_draft_extend_uses_post_verify_position(self):
        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.speculative_num_draft_tokens = 4
        worker.cuda_graph_runner_for_draft_extend = None
        worker.plan_stream = None
        worker.plan_stream_ctx = nullcontext()
        worker.seed_dsa_topk_from_draft_extend = False
        worker.topk = 1
        worker.device = torch.device("cpu")

        logits_output = SimpleNamespace(
            next_token_logits=torch.zeros(8, 16), hidden_states=None
        )
        worker.draft_runner = SimpleNamespace(
            canary_manager=None,
            forward=lambda _: SimpleNamespace(logits_output=logits_output),
        )
        sampling_info = SimpleNamespace(
            temperatures=torch.ones(2, 1),
            top_ks=torch.ones(2, dtype=torch.int32),
            sampling_seed=torch.tensor([11, 22], dtype=torch.int64),
        )
        batch = SimpleNamespace(
            seq_lens=torch.tensor([10, 20], dtype=torch.int64),
            sampling_info=sampling_info,
        )
        batch_result = SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=None),
            next_token_ids=torch.tensor([1, 2], dtype=torch.int64),
            accept_lens=torch.tensor([2, 4], dtype=torch.int64),
            new_seq_lens=torch.tensor([12, 24], dtype=torch.int64),
            next_draft_input=SimpleNamespace(),
            prepared_draft_extend_inputs=None,
        )
        forward_batch = SimpleNamespace(sampling_info=sampling_info)
        sample_result = (
            torch.zeros(2, 16),
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int64),
        )
        resolved_spec = SimpleNamespace(speculative_use_rejection_sampling=True)

        with (
            patch(
                "sglang.srt.speculative.eagle_worker_v2.prepare_for_draft_extend",
                return_value=forward_batch,
            ),
            patch(
                "sglang.srt.speculative.eagle_worker_v2.sample_draft_proposal",
                return_value=sample_result,
            ) as sample,
            patch(
                "sglang.srt.speculative.eagle_worker_v2.get_spec",
                return_value=resolved_spec,
            ),
        ):
            worker._draft_extend_for_decode(batch, batch_result)

        sample.assert_called_once()
        torch.testing.assert_close(
            sample.call_args.kwargs["positions"], batch_result.new_seq_lens
        )


def _verify_target_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Run the eager verify of one request and return its target distribution."""
    sampled = {}

    def capture_target_probs(**kwargs):
        sampled["target_probs"] = kwargs["target_probs"]

    num_draft_tokens = logits.shape[0]
    tree_arrays = torch.zeros((1, num_draft_tokens), dtype=torch.int32)
    verify_input = SimpleNamespace(
        draft_token_num=num_draft_tokens,
        draft_token=tree_arrays.flatten(),
        draft_probs=None,
        max_tree_depth=num_draft_tokens,
        tree_topk=1,
        retrieve_index=tree_arrays,
        retrieve_next_token=tree_arrays,
        retrieve_next_sibling=tree_arrays,
    )
    batch = SimpleNamespace(
        device="cpu",
        seq_lens=torch.tensor([4], dtype=torch.int32),
        forward_mode=SimpleNamespace(is_idle=lambda: False),
        sampling_info=SimpleNamespace(
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
            is_all_greedy=False,
            need_top_k_sampling=False,
            need_top_p_sampling=False,
            sampling_seed=None,
            temperatures=torch.tensor([[temperature]]),
        ),
    )
    with (
        patch.object(eagle_utils, "_is_cpu", False),
        patch.object(
            eagle_utils,
            "tree_speculative_sampling_target_only",
            side_effect=capture_target_probs,
        ),
        get_parallel().override(tp_group=SimpleNamespace(world_size=1)),
    ):
        eagle_utils.eagle_sample(
            verify_input, batch, SimpleNamespace(next_token_logits=logits)
        )
    return sampled["target_probs"].squeeze(0)


class TestSpeculativeSoftmaxRoute(CustomTestCase):
    def test_deterministic_inference_skips_the_fast_softmax_route(self):
        """The fast route depends on the batch shape, so one row can get different
        bits in different batches. Deterministic inference must use torch.softmax
        at each speculative softmax, as the sampler does, and other runs keep the
        fast route."""
        logits = torch.randn(2, 8)
        temperature = 0.5
        temperatures = torch.full((2, 1), temperature)
        plain = torch.softmax(logits, dim=-1)
        scaled = torch.softmax(logits / temperatures, dim=-1)
        fast_route = torch.full_like(logits, 0.125)
        sites = {
            "draft probs": (lambda: renorm_draft_probs(logits, None, False), plain),
            "rejection-sampling draft probs": (
                lambda: renorm_draft_probs(
                    logits, SimpleNamespace(temperatures=temperatures), True
                ),
                scaled,
            ),
            "draft proposal": (
                lambda: sample_draft_proposal(logits, temperatures)[0],
                scaled,
            ),
            "verify target probs": (
                lambda: _verify_target_probs(logits, temperature),
                scaled,
            ),
        }
        with (
            # The Gumbel draw uses a GPU kernel. This test checks q, not X.
            envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.override(False),
            patch(
                "sglang.kernels.cake_kernels.sampling.supports_softmax",
                return_value=True,
            ),
            patch("sglang.kernels.ops.sampling.cake_softmax", return_value=fast_route),
        ):
            for name, (site, torch_probs) in sites.items():
                for deterministic, expected in (
                    (True, torch_probs),
                    (False, fast_route),
                ):
                    with (
                        self.subTest(site=name, deterministic=deterministic),
                        get_context().override_server_args(
                            enable_deterministic_inference=deterministic
                        ),
                    ):
                        self.assertTrue(torch.equal(site(), expected))


if __name__ == "__main__":
    unittest.main()
