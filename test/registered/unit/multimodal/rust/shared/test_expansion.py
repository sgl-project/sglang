import random
import sys
import unittest
from pathlib import Path

from sglang.srt.multimodal.modality import Modality
from sglang.srt.multimodal.token_space_mm_process.expansion import (
    expand_token_placeholders,
)
from sglang.srt.multimodal.token_space_mm_process.process_strategy import (
    TokenSpaceProcessStrategy,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _mm_rust_utils import load_core  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CORE = load_core()
IMAGE = 100
VIDEO = 101
MODALITIES = {IMAGE: Modality.IMAGE, VIDEO: Modality.VIDEO}


def random_prompt(rng, media_tokens):
    slots = media_tokens + [rng.randint(1, 5) for _ in range(rng.randint(0, 8))]
    rng.shuffle(slots)
    return slots


def random_expansion(rng, token_id):
    body = [rng.choice([token_id, 7, 8]) for _ in range(rng.randint(0, 5))]
    body.insert(rng.randint(0, len(body)), token_id)
    return body


def random_case(rng):
    counts = {IMAGE: rng.randint(0, 3), VIDEO: rng.randint(0, 3)}
    prefix_counts = {token: rng.randint(0, count) for token, count in counts.items()}
    spec = {
        token: [random_expansion(rng, token) for _ in range(count)]
        for token, count in counts.items()
        if count
    }
    prefix_spec = {token: spec[token][: prefix_counts[token]] for token in spec}
    prefix_media = [token for token in spec for _ in range(prefix_counts[token])]
    prefix = expand_token_placeholders(random_prompt(rng, prefix_media), prefix_spec)
    new_media = [
        token for token in spec for _ in range(counts[token] - prefix_counts[token])
    ]
    return prefix + random_prompt(rng, new_media), spec, len(prefix)


def outcome(function, *args):
    try:
        return function(*args)
    except ValueError as error:
        return ("ValueError", str(error))


@unittest.skipUnless(CORE, "sglang-mm extension not built")
class TestRustTokenExpansion(CustomTestCase):
    def test_expansion_matches_token_space_processor(self):
        rng = random.Random(0)
        for case in range(300):
            input_ids, spec, start_len = random_case(rng)
            for start in (start_len, 0, -1, len(input_ids) + 1):
                with self.subTest(case=case, start=start):
                    self.assertEqual(
                        outcome(
                            CORE.common.expand_token_placeholders,
                            input_ids,
                            spec,
                            start,
                        ),
                        outcome(expand_token_placeholders, input_ids, spec, start),
                    )

    def test_media_spans_match_token_space_strategy(self):
        rng = random.Random(1)
        strategy = TokenSpaceProcessStrategy(None, None)
        strategy.mm_token_expansion_params = {
            "image_token_id": IMAGE,
            "video_token_id": VIDEO,
        }
        for case in range(300):
            input_ids, spec, start_len = random_case(rng)
            expanded = expand_token_placeholders(input_ids, spec, start_len)
            media_items = [(MODALITIES[token],) for token in spec for _ in spec[token]]
            expected = iter(strategy.locate_media(expanded, media_items, spec))
            for token, expansions in spec.items():
                with self.subTest(case=case, token=token):
                    self.assertEqual(
                        CORE.common.locate_media(expanded, token, expansions),
                        [next(expected) for _ in expansions],
                    )


if __name__ == "__main__":
    unittest.main()
