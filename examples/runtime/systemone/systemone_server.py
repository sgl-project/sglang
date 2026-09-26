"""Serve Jev's ``/v1/systemone`` typed-decision API from an SGLang ``/v1/score`` server.

TypeSafe's Jev API takes a ``state`` and a map of typed ``questions`` (``choice``,
``score``, ``noul``) and returns one typed answer per question: a chosen option with a
probability per option and a confidence, a probability-weighted score over ordered
levels, or the probability that a yes/no statement is true. It writes no text.

Any instruction-tuned model served by SGLang can answer the same shape of question
with a single forward pass per question: build a prompt that ends where the answer
label would start, and read the next-token distribution over the label tokens. That
is exactly what ``/v1/score`` does with ``label_token_ids`` + ``apply_softmax``, and
one ``/v1/score`` call scores every question of a request as one batch, so all
questions are evaluated in parallel and in isolation against the same state, as in
Jev.

Usage::

    # Terminal 1: any chat model
    python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct --port 30000

    # Terminal 2: the proxy
    python systemone_server.py --upstream http://127.0.0.1:30000 \
        --model Qwen/Qwen2.5-0.5B-Instruct --port 8300

    # Terminal 3: a Jev-shaped request
    curl -s localhost:8300/v1/systemone -H 'content-type: application/json' -d '{
      "state": "Help! My payouts have been failing for 3 days.",
      "questions": {
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Payments, invoicing, refunds",
                                    "technical": "Bugs, outages, integrations",
                                    "sales": "Pricing, upgrades, new accounts"}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm", "Frustrated", "Very angry"]},
        "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
      }
    }'

The request and response bodies follow https://docs.typesafe.ai/api. ``model`` in the
request is accepted and ignored; the served model answers. Differences from the hosted
model are listed in README.md (in short: the probabilities come from a generative
model's next-token distribution, not from a model trained to be calibrated, so fit
``--temperature`` on labelled data before trusting thresholds).
"""

import argparse
import json
import logging
import string
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests
from transformers import AutoTokenizer

logger = logging.getLogger("systemone")

MAX_CHOICE_OPTIONS = 26  # one letter label per option
MAX_SCORE_LEVELS = 10  # Jev's own limit
NOUL_LABELS = ("Yes", "No")


class SchemaError(ValueError):
    """The request body does not follow the /v1/systemone schema."""


def as_text(value) -> str:
    """Render a state / instructions / criteria value for the prompt."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def parse_questions(body: dict) -> list[dict]:
    if not isinstance(body, dict):
        raise SchemaError("request body must be a JSON object")
    if "state" not in body:
        raise SchemaError("'state' is required")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise SchemaError("'questions' must be a non-empty object")
    parsed = []
    for qid, q in questions.items():
        if not isinstance(q, dict):
            raise SchemaError(f"question '{qid}' must be an object")
        qtype = q.get("type")
        if qtype not in ("choice", "score", "noul"):
            raise SchemaError(f"question '{qid}': type must be choice, score or noul")
        if "instructions" not in q:
            raise SchemaError(f"question '{qid}': 'instructions' is required")
        criteria = q.get("criteria")
        if qtype == "choice":
            if not isinstance(criteria, dict) or not criteria:
                raise SchemaError(
                    f"question '{qid}': choice criteria must be a non-empty object"
                )
            if len(criteria) > MAX_CHOICE_OPTIONS:
                raise SchemaError(
                    f"question '{qid}': this server supports at most {MAX_CHOICE_OPTIONS} options"
                )
        elif qtype == "score":
            if (
                not isinstance(criteria, list)
                or not 2 <= len(criteria) <= MAX_SCORE_LEVELS
            ):
                raise SchemaError(
                    f"question '{qid}': score criteria must be a list of 2 to {MAX_SCORE_LEVELS} levels"
                )
        elif criteria is not None and not isinstance(criteria, dict):
            raise SchemaError(f"question '{qid}': noul criteria must be an object")
        parsed.append(
            {
                "id": qid,
                "type": qtype,
                "instructions": q["instructions"],
                "criteria": criteria,
            }
        )
    return parsed


def question_prompt(state, q: dict) -> tuple[str, list[str]]:
    """User-turn text for one question and the answer labels, in option order."""
    lines = ["State:", as_text(state), "", "Question:", as_text(q["instructions"])]
    if q["type"] == "choice":
        labels = list(string.ascii_uppercase[: len(q["criteria"])])
        lines.append("Options:")
        for label, (option, description) in zip(labels, q["criteria"].items()):
            desc = f" - {as_text(description)}" if description is not None else ""
            lines.append(f"{label}. {option}{desc}")
        lines.append("Answer with the letter of the single best option.")
    elif q["type"] == "score":
        labels = list(string.ascii_uppercase[: len(q["criteria"])])
        lines.append("Levels, from lowest to highest:")
        for label, level in zip(labels, q["criteria"]):
            lines.append(f"{label}. {as_text(level)}")
        lines.append("Answer with the letter of the level that fits best.")
    else:
        labels = list(NOUL_LABELS)
        criteria = q["criteria"] or {}
        if "true" in criteria:
            lines.append(f"Yes means: {as_text(criteria['true'])}")
        if "false" in criteria:
            lines.append(f"No means: {as_text(criteria['false'])}")
        lines.append("Answer Yes or No.")
    return "\n".join(lines), labels


def confidence(probs: list[float]) -> float:
    """Jev's published confidence: 0 for a uniform distribution, 1 for a certain one."""
    n = len(probs)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (n * max(probs) - 1) / (n - 1)))


class SystemOne:
    def __init__(
        self,
        upstream: str,
        model: str,
        tokenizer: str,
        temperature: float,
        system_prompt: str,
    ):
        self.upstream = upstream.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.session = requests.Session()
        self._label_ids: dict[str, int] = {}

    def label_token(self, label: str) -> int:
        """First token of the label, as it appears at the start of the assistant turn."""
        if label not in self._label_ids:
            ids = self.tokenizer.encode(label, add_special_tokens=False)
            if not ids:
                raise SchemaError(f"label {label!r} has no tokens")
            self._label_ids[label] = ids[0]
        return self._label_ids[label]

    def encode(self, user_text: str) -> list[int]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        return list(
            self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False
            )
        )

    def decide(self, body: dict) -> dict:
        questions = parse_questions(body)
        items, label_ids, labels_per_q = [], [], []
        for q in questions:
            text, labels = question_prompt(body["state"], q)
            ids = [self.label_token(label) for label in labels]
            if len(set(ids)) != len(ids):
                raise SchemaError(
                    f"question '{q['id']}': answer labels do not map to distinct tokens"
                )
            items.append(self.encode(text))
            label_ids.append(ids)
            labels_per_q.append(labels)

        resp = self.session.post(
            f"{self.upstream}/v1/score",
            json={
                "model": self.model,
                "query": [],
                "items": items,
                "label_token_ids": label_ids,
                "apply_softmax": True,
                "temperature": self.temperature,
            },
            timeout=600,
        )
        resp.raise_for_status()
        scores = resp.json()["scores"]

        answers = {}
        for q, probs in zip(questions, scores):
            if q["type"] == "choice":
                options = list(q["criteria"])
                answers[q["id"]] = {
                    "type": "choice",
                    "choice": options[max(range(len(probs)), key=probs.__getitem__)],
                    "probabilities": {o: round(p, 6) for o, p in zip(options, probs)},
                    "confidence": round(confidence(probs), 6),
                }
            elif q["type"] == "score":
                answers[q["id"]] = {
                    "type": "score",
                    "score": round(sum(i * p for i, p in enumerate(probs)), 6),
                    "legend": {
                        str(i): as_text(level) for i, level in enumerate(q["criteria"])
                    },
                    "probabilities": {str(i): round(p, 6) for i, p in enumerate(probs)},
                    "confidence": round(confidence(probs), 6),
                }
            else:
                answers[q["id"]] = {"type": "noul", "noul": round(probs[0], 6)}
        return {
            "model": self.model,
            "answers": answers,
            "usage": {
                "input_tokens": sum(len(ids) for ids in items),
                "output_tokens": 0,
            },
        }


def make_handler(engine: SystemOne):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

        def _json(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self._json(404, {"error": {"message": "not found"}})
            try:
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", 0)))
                    or b"null"
                )
                self._json(200, engine.decide(body))
            except (SchemaError, json.JSONDecodeError) as e:
                self._json(
                    400, {"error": {"type": "invalid_request", "message": str(e)}}
                )
            except requests.RequestException as e:
                self._json(
                    502, {"error": {"type": "upstream_error", "message": str(e)}}
                )

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--upstream", default="http://127.0.0.1:30000", help="SGLang server URL"
    )
    ap.add_argument("--model", required=True, help="model name as served by SGLang")
    ap.add_argument(
        "--tokenizer", default=None, help="tokenizer id or path (default: --model)"
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="softmax temperature over the label logits; fit on labelled data to calibrate",
    )
    ap.add_argument(
        "--system-prompt",
        default="You are a decision engine. Read the state, then answer the question with exactly one label and nothing else.",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8300)
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    engine = SystemOne(
        args.upstream,
        args.model,
        args.tokenizer or args.model,
        args.temperature,
        args.system_prompt,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    logger.info(
        "serving /v1/systemone on http://%s:%d -> %s",
        args.host,
        args.port,
        args.upstream,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
