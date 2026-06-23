"""
End-to-end test for Decode-to-Prefill (D2P) KV cache replication.

Launches a full PD disaggregation cluster with D2P replication enabled,
runs multi-turn requests, and verifies that Round 1's cached_tokens
includes decode-generated output from Round 0 (proving D2P replicated
the KV back to prefill).
"""

import asyncio
import time
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.cache_hit_kit import (
    _send_round,
    async_request_sglang_generate,
    gen_payload,
)
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    is_in_ci,
    try_cached_model,
)

register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")


def _has_mooncake():
    try:
        import mooncake.engine  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(
    is_in_ci() or _has_mooncake(),
    "Mooncake is required for D2P replication test.",
)
class TestD2PMultiTurnCacheHit(PDDisaggregationServerBase):
    """Multi-turn test verifying D2P KV replication via cache hits.

    Round 0 is a cold start. After a D2P wait, Round 1 re-sends the full
    history (prompt + output from Round 0) plus a new suffix. If D2P
    worked, the prefill's radix cache already has the decode-generated
    output, so cached_tokens >= prompt_len + output_len from Round 0.
    """

    extra_prefill_args = [
        "--disaggregation-enable-d2p-kv-replication",
        "--disaggregation-ib-device",
        "mlx5_2"
    ]
    extra_decode_args = [
        "--disaggregation-decode-enable-radix-cache",
        "--disaggregation-enable-d2p-kv-replication",
        "--disaggregation-ib-device",
        "mlx5_10"
    ]
    transfer_backend_name = "mooncake"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)
        cls.transfer_backend = [
            "--disaggregation-transfer-backend",
            cls.transfer_backend_name,
        ]
        cls.launch_all()

    def _generate(self, input_ids, output_len, url=None):
        if url is None:
            url = self.base_url
        output = asyncio.run(
            async_request_sglang_generate(
                gen_payload(input_ids, output_len),
                f"{url}/generate",
            )
        )
        self.assertTrue(output.success, output.error)
        return output

    def _flush_cache(self):
        requests.post(f"{self.prefill_url}/flush_cache?timeout=30", timeout=60)
        requests.post(f"{self.decode_url}/flush_cache?timeout=30", timeout=60)

    def test_d2p_improves_prefill_cache_hits(self):
        from sglang.benchmark.datasets.random import sample_random_requests
        from sglang.benchmark.utils import get_tokenizer

        self._flush_cache()
        time.sleep(1)

        tokenizer = get_tokenizer(self.model)
        prompts = sample_random_requests(
            input_len=256,
            output_len=64,
            num_prompts=1,
            range_ratio=1.0,
            tokenizer=tokenizer,
            dataset_path="/workdir/ShareGPT_V3_unfiltered_cleaned_split.json",
            return_text=False,
        )
        suffixes = sample_random_requests(
            input_len=64,
            output_len=64,
            num_prompts=1,
            range_ratio=1.0,
            tokenizer=tokenizer,
            dataset_path="/workdir/ShareGPT_V3_unfiltered_cleaned_split.json",
            return_text=False,
        )

        output_len = 64
        history = list(prompts[0].prompt)

        # --- Round 0: cold start ---
        output0 = self._generate(history, output_len)
        print(
            f"\n[Round 0] prompt_len={output0.prompt_len}, "
            f"disagg_prefill_prefix_len={output0.disagg_prefill_prefix_len}, "
            f"disagg_decode_prefix_len={output0.cached_tokens}, "
            f"output_len={len(output0.output_ids)}"
        )
        # Round 0: cold start — no prefix hits anywhere
        self.assertEqual(
            output0.disagg_prefill_prefix_len, 0,
            "Round 0: prefill prefix should be 0",
        )
        self.assertEqual(output0.cached_tokens, 0, "Round 0: decode prefix should be 0")

        # Wait for D2P replication to complete
        time.sleep(3)

        # --- Round 1: multi-turn, should get cache hits from D2P ---
        history.extend(output0.output_ids)
        history.extend(list(suffixes[0].prompt))

        prev_total = output0.prompt_len + len(output0.output_ids)

        output1 = self._generate(history, output_len)
        print(
            f"[Round 1] prompt_len={output1.prompt_len}, "
            f"disagg_prefill_prefix_len={output1.disagg_prefill_prefix_len}, "
            f"disagg_decode_prefix_len={output1.cached_tokens}, "
            f"output_len={len(output1.output_ids)}"
        )

        # With D2P, prefill's radix cache has prompt + output from Round 0,
        # so disagg_prefill_prefix_len should cover the full previous sequence.
        expected_min_cached = prev_total - 2
        self.assertGreaterEqual(
            output1.disagg_prefill_prefix_len,
            expected_min_cached,
            f"Round 1: disagg_prefill_prefix_len should be >= {expected_min_cached} "
            f"(prompt+output from round 0), but got {output1.disagg_prefill_prefix_len}. "
            f"This means D2P didn't replicate output KV to prefill.",
        )

        print(
            f"  => D2P SUCCESS: disagg_prefill_prefix_len={output1.disagg_prefill_prefix_len}, "
            f"disagg_decode_prefix_len={output1.cached_tokens} "
            f"(without D2P prefill_prefix would be ~{output0.prompt_len})"
        )

        # Verify servers are still healthy
        for name, url in [
            ("prefill", self.prefill_url),
            ("decode", self.decode_url),
            ("lb", self.lb_url),
        ]:
            r = requests.get(f"{url}/health", timeout=10)
            r.raise_for_status()
            print(f"\n{name} server healthy after test")

    def test_d2p_generated_shared_prefix(self):
        """Multi-turn dialog benchmark showing D2P benefit.

        N clients each run R rounds of dialog. Each round appends a new
        user question, sends the full history, and appends the generated
        reply. D2P replicates generated tokens back to prefill between
        rounds, so each subsequent round's prefill prefix match covers
        the full prior conversation — not just the original prompt.

        Without D2P: disagg_prefill_prefix_len ≈ original prompt length
                     (generated tokens are only on decode side).
        With D2P:    disagg_prefill_prefix_len ≈ full prior history
                     (generated tokens replicated back to prefill).

        Prints a per-round table of disagg_prefill_prefix_len and TTFT.
        """
        from sglang.benchmark.datasets.random import sample_random_requests
        from sglang.benchmark.utils import get_tokenizer

        self._flush_cache()
        time.sleep(1)

        num_clients = 4
        num_rounds = 3
        initial_prompt_len = 256
        question_len = 64
        output_len = 64
        d2p_wait = 3

        tokenizer = get_tokenizer(self.model)

        initial_prompts = sample_random_requests(
            input_len=initial_prompt_len,
            output_len=output_len,
            num_prompts=num_clients,
            range_ratio=1.0,
            tokenizer=tokenizer,
            dataset_path="/workdir/ShareGPT_V3_unfiltered_cleaned_split.json",
            return_text=False,
        )
        questions = sample_random_requests(
            input_len=question_len,
            output_len=output_len,
            num_prompts=num_clients * max(num_rounds - 1, 1),
            range_ratio=1.0,
            tokenizer=tokenizer,
            dataset_path="/workdir/ShareGPT_V3_unfiltered_cleaned_split.json",
            return_text=False,
        )

        histories = [list(p.prompt) for p in initial_prompts]
        q_idx = 0

        print(f"\n{'='*70}")
        print(f"  Multi-turn D2P benchmark: {num_clients} clients × {num_rounds} rounds")
        print(f"  initial_prompt={initial_prompt_len}, question={question_len}, "
              f"output={output_len}")
        print(f"{'='*70}")

        for rnd in range(num_rounds):
            if rnd > 0:
                for i in range(num_clients):
                    histories[i].extend(list(questions[q_idx].prompt))
                    q_idx += 1

            payloads = [gen_payload(h, output_len) for h in histories]
            url = f"{self.base_url}/generate"
            outputs = asyncio.run(
                _send_round(payloads, url, max_parallel=num_clients)
            )

            print(f"\n[Round {rnd}]")
            print(f"  {'client':>6}  {'prompt_len':>10}  {'prefill_prefix':>14}  "
                  f"{'decode_prefix':>13}  {'ttft':>8}")
            print(f"  {'-'*6}  {'-'*10}  {'-'*14}  {'-'*13}  {'-'*8}")

            for i, out in enumerate(outputs):
                self.assertTrue(out.success, f"Round {rnd}, client {i}: {out.error}")

                print(
                    f"  {i:>6}  {out.prompt_len:>10}  "
                    f"{out.disagg_prefill_prefix_len:>14}  "
                    f"{out.cached_tokens:>13}  "
                    f"{out.ttft:>8.4f}s"
                )

                if rnd == 0:
                    self.assertEqual(
                        out.disagg_prefill_prefix_len, 0,
                        f"Round 0 client {i}: cold start, prefill prefix should be 0",
                    )
                else:
                    prev_history_len = out.prompt_len - question_len
                    expected_min = prev_history_len - 2
                    self.assertGreaterEqual(
                        out.disagg_prefill_prefix_len,
                        expected_min,
                        f"Round {rnd} client {i}: disagg_prefill_prefix_len="
                        f"{out.disagg_prefill_prefix_len} < {expected_min}. "
                        f"D2P should have replicated prior conversation to prefill.",
                    )

                histories[i].extend(out.output_ids)

            if rnd < num_rounds - 1:
                time.sleep(d2p_wait)

        print(f"\n{'='*70}")
        print(f"  D2P multi-turn benchmark complete")
        print(f"{'='*70}")

        for name, url in [
            ("prefill", self.prefill_url),
            ("decode", self.decode_url),
            ("lb", self.lb_url),
        ]:
            r = requests.get(f"{url}/health", timeout=10)
            r.raise_for_status()


if __name__ == "__main__":
    unittest.main()
