"""The Mooncake dataset keeps the trace's 512-token blocks: a block id expands
to the same 512 token ids for every request, so shared ids are shared prefixes
and prompts have the trace's lengths."""

import asyncio
import unittest
from types import SimpleNamespace

from sglang.benchmark.datasets.mooncake import (
    MOONCAKE_BLOCK_TOKENS,
    MooncakeBlocks,
    get_mooncake_request_over_time,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

VOCAB_SIZE = 32000


def _record(hash_ids, input_length, output_length=16, timestamp=0):
    return {
        "timestamp": timestamp,
        "hash_ids": hash_ids,
        "input_length": input_length,
        "output_length": output_length,
    }


class TestMooncakeBlocks(CustomTestCase):
    def test_shared_block_ids_share_prefix_tokens(self):
        blocks = MooncakeBlocks(VOCAB_SIZE)
        first = blocks.prompt(_record([1, 2, 3], 3 * MOONCAKE_BLOCK_TOKENS))
        second = blocks.prompt(_record([1, 2, 9], 3 * MOONCAKE_BLOCK_TOKENS))
        shared = 2 * MOONCAKE_BLOCK_TOKENS
        self.assertEqual(len(first), 3 * MOONCAKE_BLOCK_TOKENS)
        self.assertEqual(first[:shared], second[:shared])
        self.assertNotEqual(first[shared:], second[shared:])
        # A second builder, as the warmup uses, expands the ids the same way.
        self.assertEqual(
            MooncakeBlocks(VOCAB_SIZE).prompt(_record([1, 2, 3], 1536)), first
        )
        self.assertTrue(all(0 <= t < VOCAB_SIZE for t in first))

    def test_prompt_is_cut_to_the_trace_input_length(self):
        blocks = MooncakeBlocks(VOCAB_SIZE)
        self.assertEqual(len(blocks.prompt(_record([4, 5, 6], 1300))), 1300)

    def test_rows_carry_token_ids_and_their_length(self):
        rows = []

        async def collect():
            async for row in get_mooncake_request_over_time(
                [_record([7, 8], 900, output_length=10)],
                SimpleNamespace(vocab_size=VOCAB_SIZE),
                slowdown_factor=0.0,
                num_rounds=2,
            ):
                rows.append(row)

        asyncio.run(collect())
        self.assertEqual(len(rows), 2)
        first, second = rows
        self.assertEqual(first.prompt_len, 900)
        self.assertEqual(len(first.prompt), 900)
        self.assertEqual(first.output_len, 10)
        # Round two carries round one, its placeholder answer and the query again.
        self.assertEqual(second.prompt_len, 900 + 10 + 900)
        self.assertEqual(second.prompt[:900], first.prompt)
        self.assertEqual(second.prompt[910:], first.prompt)


if __name__ == "__main__":
    unittest.main()
