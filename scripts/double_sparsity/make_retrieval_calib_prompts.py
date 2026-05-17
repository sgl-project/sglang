"""Generate synthetic NIAH-shaped calibration prompts for DS calibration.

The goal is to bias the K-channel calibration toward channels that
activate on retrieval-pattern tokens (specific facts surrounded by
filler context) rather than next-token-prediction channels learned
from raw wikitext. Empirically, wikitext calibration at heavy_channels=32
fails NIAH below token_budget=8192; this corpus is the cheapest way
to test whether the calibration shape (not the underlying mechanism)
is the constraint.

Each line in the output file is one prompt, formatted as

    <haystack prefix>  <needle>  <haystack suffix>  Question: <question>

with random needles placed at random positions inside generic
haystack filler. Calibrate.py truncates each line to ``--seq-len``
tokens; aim for roughly 4 * seq_len characters per line so the
truncation lands inside the haystack suffix, not before the needle.

Usage:
    python scripts/double_sparsity/make_retrieval_calib_prompts.py \\
        --output /workspace/ds_retrieval_calib_prompts.txt \\
        --n-prompts 128 --target-chars 20000 --seed 0

Then feed it to calibrate.py:
    python scripts/double_sparsity/calibrate.py \\
        --model meta-llama/Llama-3.1-70B-Instruct \\
        --output /workspace/calib_llama_3_1_70b_retrieval_s32.json \\
        --heavy-channels 32 --n-samples 64 --seq-len 4096 \\
        --prompts-file /workspace/ds_retrieval_calib_prompts.txt \\
        --device-map auto
"""

from __future__ import annotations

import argparse
import random
import string

# Generic filler sentences. Plausible-looking but no actual proper nouns
# or numerical facts that could collide with the random needles below.
_FILLER_SENTENCES = [
    "The streets of the city stretched on for miles in every direction.",
    "Afternoon light fell on the windows of the small bookshop.",
    "Travellers passed through the station every hour throughout the day.",
    "Old letters were kept inside the cabinet beside the desk.",
    "Notes on the corkboard reminded everyone of the next meeting.",
    "Rain had been steady all morning, slowing the traffic outside.",
    "Walkers along the river greeted one another with a polite nod.",
    "The library was filled with the soft sound of pages turning.",
    "Lamps along the street flickered as the evening grew darker.",
    "Children played in the courtyard until their parents called them in.",
    "A long shadow stretched across the floor of the empty hallway.",
    "Stones near the path had been polished smooth over decades.",
    "Workers carried boxes from the truck to the loading dock.",
    "The clock above the door had not been wound in many weeks.",
    "Music drifted out from the open window of the second floor.",
    "Tea was served in small cups around the wooden table.",
    "Boats rocked gently against the dock in the morning breeze.",
    "Trees lined the avenue in even, deliberate rows.",
    "Plans for the upcoming festival had been pinned to the wall.",
    "Glass jars on the shelf held an assortment of small items.",
    "Visitors paused to look at the painting near the entrance.",
    "Coats hung on the rack by the front door of the cottage.",
    "Footsteps echoed down the corridor late at night.",
    "Mountains rose in the distance, their peaks hidden in cloud.",
    "Lanterns along the path flickered as the wind picked up.",
    "Quiet voices could be heard from the room at the end of the hall.",
    "Steam rose from the cups arranged on the breakfast tray.",
    "Banners outside the hall announced the arrival of the new exhibition.",
    "The garden had been left untended for an entire season.",
    "Doors opened onto a small terrace overlooking the harbour.",
    "Letters were stamped, sealed, and placed in the wooden box.",
    "Cyclists rode past the open market on their way home.",
    "The professor reviewed his lecture notes one final time.",
    "Vines climbed the stone wall up to the second-floor balcony.",
    "Old maps were spread out across the long table by the window.",
    "Visitors waited patiently in the corridor outside the office.",
    "Bells from the tower rang the hour without fail each evening.",
    "Snow had begun to settle on the windowsills overnight.",
    "Boats passed beneath the bridge in slow, deliberate procession.",
    "The hallway smelled faintly of varnish and old books.",
]


def _random_needle_key(rng: random.Random) -> str:
    """Random plausible 'fact identifier' — a short capitalised noun phrase."""
    nouns = (
        "package",
        "passport",
        "ledger",
        "manuscript",
        "container",
        "carrier",
        "envelope",
        "bookcase",
        "satchel",
        "footlocker",
        "briefcase",
        "logbook",
        "register",
        "casket",
        "tankard",
        "thermos",
        "decanter",
        "kettle",
        "lamp",
        "scroll",
    )
    qualifier = (
        "old",
        "lost",
        "missing",
        "broken",
        "abandoned",
        "forgotten",
        "hidden",
        "stolen",
        "borrowed",
        "rusted",
    )
    return f"the {rng.choice(qualifier)} {rng.choice(nouns)}"


def _random_needle_value(rng: random.Random) -> str:
    """Random 'fact value' — a short string with letters + digits."""
    if rng.random() < 0.5:
        # numeric value
        return str(rng.randint(1000, 999_999))
    # alphanumeric token
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(8))


def _build_prompt(rng: random.Random, target_chars: int) -> str:
    """Build one prompt:
    <filler haystack>  Note: <key> is <value>.  <filler haystack>  Question: What is <key>?
    """
    needle_key = _random_needle_key(rng)
    needle_value = _random_needle_value(rng)
    needle = f"Note: {needle_key} is {needle_value}."
    question = f"Question: What is {needle_key}?"

    pieces = []
    # Insert the needle at a random ~30-70% position in the haystack.
    insert_at = rng.uniform(0.3, 0.7)
    needle_chars = int(target_chars * insert_at)

    chars = 0
    while chars < needle_chars:
        s = rng.choice(_FILLER_SENTENCES)
        pieces.append(s)
        chars += len(s) + 1
    pieces.append(needle)
    chars += len(needle) + 1
    while chars < target_chars - len(question) - 2:
        s = rng.choice(_FILLER_SENTENCES)
        pieces.append(s)
        chars += len(s) + 1
    pieces.append(question)
    return " ".join(pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--output", required=True, help="Output text file path")
    parser.add_argument("--n-prompts", type=int, default=128)
    parser.add_argument(
        "--target-chars",
        type=int,
        default=20000,
        help="Target characters per line (~4x target token count).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    with open(args.output, "w", encoding="utf-8") as f:
        for _ in range(args.n_prompts):
            prompt = _build_prompt(rng, args.target_chars)
            # Sanitize: replace any embedded newlines with spaces; one prompt
            # per line is the calibrate.py invariant.
            prompt = prompt.replace("\n", " ").replace("\r", " ")
            f.write(prompt + "\n")
    print(f"wrote {args.n_prompts} NIAH-shaped prompts to {args.output}")


if __name__ == "__main__":
    main()
