"""Where the triton beam walk stops agreeing with the torch reference.

The CI test (`test/registered/unit/spec/test_dflash_logits.py`) asserts elementwise
equality on well-separated scores, which is the only regime where equality is a
legitimate claim: the two implementations normalize with different reduction orders,
so once the gaps in the flattened pool approach fp32 noise either one may pick a
different node, and one different pick makes every deeper depth diverge.

This script measures where that boundary sits. Run it after touching either
implementation; it asserts nothing, it prints a table.

    python3 test/manual/spec/bench_dflash_beam_walk_agreement.py
"""

import torch

from sglang.kernels.ops.speculative.dflash import selector_beam_walk_triton
from sglang.srt.models.dflash import _beam_walk_torch

SLOTS = 7
TOP_K = 16
BATCH = 128
SEEDS = 10
WIDTHS = (2, 4, 8)
# randn * 3 is the order of magnitude a trained selector's lattice actually spans;
# the smaller scales exist to locate the failure boundary, not to model anything.
SCALES = (3.0, 1e-2, 1e-4, 1e-6)


def diverging_rows(*, scale: float, seed: int, beam_width: int) -> int:
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(BATCH, SLOTS, TOP_K, TOP_K, generator=generator) * scale
    candidate_ids = torch.randint(0, 5000, (BATCH, SLOTS, TOP_K), generator=generator)
    anchor = torch.randint(0, 5000, (BATCH,), generator=generator)

    expected = _beam_walk_torch(
        candidate_ids=candidate_ids,
        scores=scores,
        anchor_token_ids=anchor,
        beam_width=beam_width,
    )
    actual = selector_beam_walk_triton(
        candidate_ids=candidate_ids.cuda(),
        scores=scores.cuda(),
        anchor_token_ids=anchor.cuda(),
        beam_width=beam_width,
    )
    differs = (actual[0].cpu() != expected[0]).any(dim=1)
    differs |= (actual[1].cpu() != expected[1]).any(dim=1)
    return int(differs.sum())


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")
    print(f"{'score scale':>12}  {'rows':>6}  {'diverging':>9}  {'share':>7}")
    for scale in SCALES:
        rows = diverged = 0
        for seed in range(SEEDS):
            for beam_width in WIDTHS:
                diverged += diverging_rows(
                    scale=scale, seed=seed, beam_width=beam_width
                )
                rows += BATCH
        print(f"{scale:>12g}  {rows:>6}  {diverged:>9}  {100 * diverged / rows:>6.2f}%")


if __name__ == "__main__":
    main()
