# Adapted from https://github.com/openai/simple-evals/

"""
AIME 2026 - American Invitational Mathematics Examination 2026
Dataset: MathArena/aime_2026
https://huggingface.co/datasets/MathArena/aime_2026

The American Invitational Mathematics Examination (AIME) is a challenging
competition math exam. All answers are integers from 000 to 999.
"""

from sglang.test.simple_eval_aime25 import AIME25Eval
from sglang.test.simple_eval_common import import_load_dataset


class AIME26Eval(AIME25Eval):
    def _load_examples(self) -> list[dict]:
        load_dataset = import_load_dataset()

        dataset = load_dataset("MathArena/aime_2026", split="train")
        return [
            {"question": row["problem"], "answer": str(row["answer"])}
            for row in dataset
        ]
