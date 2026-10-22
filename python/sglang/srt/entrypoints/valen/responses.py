# SPDX-License-Identifier: Apache-2.0
"""Valen answer fields and confidence formulas."""

import math

import torch


def valen_answers(questions, logits, temperature=1.0):
    """The released Valen answer() formulas, evaluated in FP32."""
    values = torch.as_tensor(logits, dtype=torch.float32)
    if values.ndim != 1 or len(values) != sum(len(q.keys) for q in questions):
        raise ValueError("Valen head did not return one logit per candidate")
    if not torch.isfinite(values).all():
        raise ValueError("Valen head returned nonfinite logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Valen temperature must be finite and positive")
    answers, offset = {}, 0
    for question in questions:
        count = len(question.keys)
        probabilities = (
            (values[offset : offset + count] / temperature).softmax(-1).tolist()
        )
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        offset += count
        result = {"type": question.kind}
        if question.kind == "noul":
            result["noul"] = probabilities[question.keys.index("true")]
        else:
            result["probabilities"] = dict(zip(question.keys, probabilities))
            mode = max(range(count), key=probabilities.__getitem__)
            if question.kind == "choice":
                result["choice"] = question.keys[mode]
                result["confidence"] = (
                    1.0
                    if count == 1
                    else max(0.0, (max(probabilities) - 1 / count) / (1 - 1 / count))
                )
            else:
                result["score"] = sum(i * p for i, p in enumerate(probabilities))
                result["legend"] = dict(zip(question.keys, question.descriptions))
                distance = sum(p * abs(i - mode) for i, p in enumerate(probabilities))
                uniform_mad = (
                    sum(abs(i - (count - 1) / 2) for i in range(count)) / count
                )
                result["confidence"] = max(0.0, 1.0 - distance / uniform_mad)
        answers[question.qid] = result
    return answers
