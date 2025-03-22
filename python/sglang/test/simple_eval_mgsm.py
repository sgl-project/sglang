# Adapted from https://github.com/openai/simple-evals/

"""
MGSM: Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems.
Language Models are Multilingual Chain-of-Thought Reasoners
Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, Jason Wei
https://arxiv.org/abs/2210.03057 reference: https://github.com/google-research/url-nlp
"""

import re
import urllib
from typing import Optional

from sglang.test import simple_eval_common as common
from sglang.test.simple_eval_common import (
    HTML_JINJA,
    Eval,
    EvalResult,
    SamplerBase,
    SingleEvalResult,
)

ALL_LANGUAGES = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
LATIN_LANGUAGES = ["de", "en", "es", "fr", "sw"]
NON_LATIN_LANGUAGES = ["bn", "ja", "ru", "te", "th", "zh"]

LANG_TO_FPATH = {
    "bn": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_bn.tsv",
    "de": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_de.tsv",
    "en": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_en.tsv",
    "es": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_es.tsv",
    "fr": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_fr.tsv",
    "ja": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_ja.tsv",
    "ru": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_ru.tsv",
    "sw": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_sw.tsv",
    "te": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_te.tsv",
    "th": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_th.tsv",
    "zh": "https://openaipublic.blob.core.windows.net/simple-evals/mgsm_zh.tsv",
}
LANG_TO_INSTRUCTIONS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{input}""",
    "bn": """�ই গণিতের সমস�যাটি সমাধান কর�ন। চূড়ান�ত উত�তর দেওয়ার আগে য�ক�তিসম�পন�ন পদক�ষেপ প�রদান কর�ন। চূড়ান�ত উত�তরটি �কক সংখ�যা হিসাবে "উত�তর:" �র পরে শেষ লাইনে দিন। "উত�তর:" �র পরে অন�য কিছ� য�ক�ত করবেন না।.

{input}""",
    "de": """Löse dieses Mathematikproblem. Gib die Schritte zur Begründung an, bevor du die endgültige Antwort in der letzten Zeile alleine im Format "Antwort:" gibst. Füge nichts anderes als die ganzzahlige Antwort nach "Antwort:" hinzu.

{input}""",
    "es": """Resuelve este problema matemático. Proporciona los pasos de razonamiento antes de dar la respuesta final en la última línea por sí misma en el formato de "Respuesta:". No añadas nada más que la respuesta entera después de "Respuesta:".

{input}""",
    "fr": """Résolvez ce problème de mathématiques. Donnez les étapes de raisonnement avant de fournir la réponse finale sur la dernière ligne elle-même dans le format de "Réponse:". N'ajoutez rien d'autre que la réponse entière après "Réponse:".

{input}""",
    "ja": """�数学��題を解������。最終的�答�を出����解答�推論�程を記述������。���最後�行�� "答�:" �形��答�を記述����後��整数�答�以外何も追加��������。

{input}""",
    "ru": """Решите �ту математиче�кую задачу. Объ��ните шаги ра��уждени� перед тем, как дать окончательный ответ в по�ледней �троке �ам по �ебе в формате "Ответ:". �е добавл�йте ничего, кроме целочи�ленного ответа по�ле "Ответ:".

{input}""",
    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".

{input}""",
    "te": """ఈ గణిత సమస�యన� పరిష�కరించండి. చివరి సమాధానాన�ని ఇవ�వదానికి మ�ంద� తర�కాత�మక అద�గ�లన� ఇవ�వండి. చివరి పంక�తిలో మాత�రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద�ని ఇవ�వండి సమాధానం: తర�వాత పూర�ణాంక సమాధానానికి తప�పించి ఎదేనా చేర�చవద�ద�.

{input}""",
    "th": """��้ปั�หาคณิตศาสตร์นี้ ให้ให้ขั้นตอน�ารใช้เหตุผล�่อนที่จะให้คำตอบสุดท้ายในบรรทัดสุดท้ายโดยอยู่ในรูป�บบ "คำตอบ:" ไม่ควรเพิ่มอะไรนอ�จา�คำตอบที่เป็นจำนวนเต็มหลังจา� "คำตอบ:"

{input}""",
    "zh": """解决这个数学问题。在最�一行给出答案�，请�供推�步骤。最�一行应该以 "答案: " 的形�独立给出答案。在 "答案：" ���添加除整数答案之外的任何内容。

{input}""",
}

LANG_TO_ANSWER_PREFIX = {
    "en": "Answer",
    "bn": "উত�তর",
    "de": "Antwort",
    "es": "Respuesta",
    "fr": "Réponse",
    "ja": "答�",
    "ru": "Ответ",
    "sw": "Jibu",
    "te": "సమాధానం",
    "th": "คำตอบ",
    "zh": "答案",
}


def parse_answer(answer: str, answer_prefix: str) -> str:
    if answer_prefix not in answer:
        return ""

    answer_text = answer.split(answer_prefix)[-1].strip()

    # find all the numbers (including decimals) in the string
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))

    # return the first number (removing trailing decimal point if present),
    # or an empty string if there were no numbers
    return numbers[-1].rstrip(".") if numbers else ""


def score_mgsm(target: str, prediction: str) -> bool:
    if "." in prediction:
        prediction = prediction.rstrip("0").rstrip(".")

    target = target.replace(",", "")
    prediction = prediction.replace(",", "")

    return target == prediction


def get_lang_examples(lang: str) -> list[dict[str, str]]:
    fpath = LANG_TO_FPATH[lang]
    examples = []
    with urllib.request.urlopen(fpath) as f:
        for line in f.read().decode("utf-8").splitlines():
            inputs, targets = line.strip().split("\t")
            if "." in targets:
                raise ValueError(f"targets {targets} contains a decimal point.")
            # targets = int(targets.replace(",", ""))
            examples.append({"inputs": inputs, "targets": targets, "lang": lang})
    return examples


def get_all_examples() -> list[dict[str, str]]:
    examples = []
    for lang in ALL_LANGUAGES:
        if lang != "en":
            continue
        examples += get_lang_examples(lang)
    return examples


class MGSMEval(Eval):
    def __init__(
        self,
        num_examples_per_lang: int = 250,  # restrict to a subset of the data for debugging
        num_threads: int = 64,
        languages: Optional[list[str]] = ALL_LANGUAGES,
    ):
        if languages is None:
            languages = ALL_LANGUAGES
        else:
            for language in languages:
                if language not in ALL_LANGUAGES:
                    raise ValueError(
                        f"language {language} is not a valid language. "
                        f"It should be one in {ALL_LANGUAGES}"
                    )
        self._languages = languages
        self._num_examples_per_lang = num_examples_per_lang
        self._num_threads = num_threads

        examples = []
        for lang in self._languages:
            lang_examples = get_lang_examples(lang)
            examples.extend(lang_examples[: self._num_examples_per_lang])
        self.examples = examples

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """TODO: Add docstring."""
def fn(example: dict[str, str]):
            language = example["lang"]
            latin_language = (
                "group_latin" if language in LATIN_LANGUAGES else "group_non_latin"
            )
            correct_answer = example["targets"]
            instructoin = LANG_TO_INSTRUCTIONS[language]
            prompt_messages = [
                sampler._pack_message(
                    content=instructoin.format(input=example["inputs"]), role="user"
                )
            ]
            try:
                response_text = sampler(prompt_messages)
            except Exception as e:
                response_text = ""

            answer_prefix = LANG_TO_ANSWER_PREFIX[language]
            extracted_answer = parse_answer(response_text, answer_prefix)

            score = score_mgsm(correct_answer, extracted_answer)
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={language: score, latin_language: score},
            )

        results = common.map_with_progress(
            fn, self.examples, num_threads=self._num_threads
        )
        return common.aggregate_results(results, default_stats=("mean", "std"))
