"""Export the rawsystemone OpenAPI contract without importing the GPU runtime.

Requires Python 3.10+ and Pydantic 2. Run from any working directory.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Import the standalone protocol without executing sglang.__init__.
sys.path.insert(0, str(ROOT / "python/sglang/srt/entrypoints"))
from rawsystemone.protocol import (  # noqa: E402
    RawSystemOneRequest,
    RawSystemOneResponse,
)

OUTPUT = ROOT / "examples/runtime/rawsystemone.openapi.json"

FIELD_DESCRIPTIONS = {
    "RawSystemOneRequest": {
        "prefix": "Prompt text tokenized once using the native raw-text policy, "
        "without inserted whitespace or a chat template. The resulting prefix "
        "must contain at least one token; an empty string is valid only when "
        "the native tokenizer supplies a token such as BOS.",
        "suffixes": "Ordered candidate suffixes. Duplicates are allowed. Each suffix "
        "must tokenize to at least one token without tokenizer-added special tokens. "
        "Each complete prefix + suffix must have at least two native tokens. The "
        "server-configured option limit defaults to 128; it is not a fixed schema limit.",
        "return_token_logprobs": "Include complete native token records for every "
        "candidate. When false, token_logprobs is omitted from every result.",
    },
    "RawSystemOneResponse": {
        "id": "Server-generated request identifier.",
        "model": "The already-loaded default model's served name.",
        "scoring": "For each option, divide log P(suffix | prefix) by option_token_count, "
        "then apply a numerically stable softmax across all original options.",
        "tokenization": "Tokenize the prefix once with native raw-text special-token "
        "behavior, tokenize each suffix with add_special_tokens=false, and append "
        "the token IDs. This fixes the prefix boundary and can differ from "
        "tokenizing the concatenated text.",
        "data": "One result per original suffix, in input order, including duplicates.",
        "best_index": "Zero-based original index with the highest score. Exact ties "
        "select the first index; scores are not rounded before selection.",
        "usage": "Logical candidate totals, including repeated prefixes and duplicate "
        "candidates. These do not measure physical GPU work or cache savings.",
    },
    "CandidateScore": {
        "index": "Zero-based index in the request's suffixes array.",
        "score": "Normalized weight: softmax(logprob_sum / option_token_count) "
        "across all original options, including duplicates. Weights sum to one "
        "within floating-point precision. This is not calibrated correctness confidence.",
        "logprob_sum": "Natural log P(suffix | prefix): sum of suffix-token "
        "log-probabilities. Equivalent to the complete sequence log-probability "
        "minus the fixed prefix log-probability; prefix token scores are excluded.",
        "option_token_count": "Token count of this suffix encoded alone with "
        "add_special_tokens=false, preserving its exact whitespace. This is the "
        "score denominator, independent of prefix length and shared cache span.",
        "token_logprobs": "Present only when return_token_logprobs is true. Contains "
        "every token in the complete prefix + suffix in absolute position order, "
        "including position zero. Its length is not the score denominator.",
    },
    "TokenLogprob": {
        "position": "Zero-based absolute position in the complete native token sequence.",
        "token_id": "Native tokenizer token ID at this position.",
        "logprob": "Natural log-probability after full-vocabulary log-softmax. Null "
        "only at position zero; every later position has a finite value, including "
        "a possible 0.0. This is not a raw logit.",
    },
    "Usage": {
        "input_tokens": "Total native tokens in all complete prefix + suffix sequences, "
        "including repeated prefixes and duplicate candidates.",
        "scored_tokens": "Total scoreable tokens in all complete sequences: input_tokens "
        "minus the number of candidates. This is not the sum of option_token_count.",
        "generated_tokens": "Always zero; the endpoint does not generate tokens.",
    },
}

ERROR_RESPONSES = {
    "400": "Invalid fields, types, JSON, or Content-Type (invalid_request); unsupported "
    "serving configuration (unsupported_mode); or candidate limits "
    "(too_many_options, no_prefix_tokens, no_option_tokens, no_scoreable_tokens, context_length_exceeded, "
    "token_budget_exceeded). No candidate is silently truncated. Validation uses "
    "HTTP 400, not 422.",
    "409": "Model weights changed during scoring (model_changed). Retry the whole request.",
    "429": "The admission queue or native scheduler is overloaded (overloaded).",
    "500": "Incomplete native score coverage (incomplete_scores), non-finite scores "
    "(non_finite_scores), or native inference failure (inference_error).",
    "503": "The native tokenizer is unavailable (tokenizer_unavailable), or native "
    "inference is overloaded (overloaded).",
    "504": "The server-configured scoring deadline expired (timeout); default 300 seconds.",
    "default": "Other HTTP failure propagated from native inference (inference_error).",
}


def schema_ref(name):
    return {"$ref": f"#/components/schemas/{name}"}


def build_schema():
    schemas = {}
    for model, mode in (
        (RawSystemOneRequest, "validation"),
        (RawSystemOneResponse, "serialization"),
    ):
        schema = model.model_json_schema(
            ref_template="#/components/schemas/{model}", mode=mode
        )
        schemas.update(schema.pop("$defs", {}))
        schemas[model.__name__] = schema

    for name, descriptions in FIELD_DESCRIPTIONS.items():
        for field, description in descriptions.items():
            schemas[name]["properties"][field]["description"] = description

    # The route emits defaults and omits unrequested token details. Describe the
    # actual wire response rather than the looser model construction defaults.
    for name in ("RawSystemOneResponse", "Usage"):
        schemas[name]["required"] = list(schemas[name]["properties"])
    token_details = schemas["CandidateScore"]["properties"]["token_logprobs"]
    token_details.update(
        next(s for s in token_details.pop("anyOf") if s["type"] == "array")
    )
    token_details.pop("default", None)
    token_details["minItems"] = 2
    schemas["RawSystemOneResponse"]["properties"]["data"]["minItems"] = 1
    for name, fields in {
        "RawSystemOneResponse": {"best_index": 0},
        "CandidateScore": {"index": 0, "option_token_count": 1},
        "TokenLogprob": {"position": 0, "token_id": 0},
        "Usage": {"input_tokens": 2, "scored_tokens": 1},
    }.items():
        for field, minimum in fields.items():
            schemas[name]["properties"][field]["minimum"] = minimum
    for field in ("score", "logprob_sum"):
        schemas["CandidateScore"]["properties"][field]["format"] = "double"
    schemas["CandidateScore"]["properties"]["score"].update(minimum=0, maximum=1)
    for branch in schemas["TokenLogprob"]["properties"]["logprob"]["anyOf"]:
        if branch["type"] == "number":
            branch["format"] = "double"
    schemas["TokenLogprob"]["examples"] = [
        {"position": 0, "token_id": 1, "logprob": None},
        {"position": 1, "token_id": 42, "logprob": -1.5},
    ]

    schemas["RawSystemOneError"] = {
        "type": "object",
        "required": ["object", "message", "type", "param", "code"],
        "properties": {
            "object": {"type": "string", "const": "error"},
            "message": {
                "type": "string",
                "description": "Human-readable error explanation.",
            },
            "type": {
                "type": "string",
                "description": "Machine-readable error type; see response descriptions.",
            },
            "param": {"type": "null"},
            "code": {"type": "integer", "description": "HTTP status code."},
        },
        "example": {
            "object": "error",
            "message": "Invalid rawsystemone fields or field types.",
            "type": "invalid_request",
            "param": None,
            "code": 400,
        },
    }
    schemas["AuthenticationError"] = {
        "type": "object",
        "required": ["error"],
        "properties": {"error": {"type": "string"}},
        "example": {"error": "Unauthorized"},
    }

    example_sums = (-4.0, -3.0, -2.0)
    example_weights = [math.exp(total / 2 + 1) for total in example_sums]
    example_total = math.fsum(example_weights)
    success = RawSystemOneResponse(
        id="rawsystemone-example",
        model="served-model-name",
        data=[
            {
                "index": index,
                "score": example_weights[index] / example_total,
                "logprob_sum": total,
                "option_token_count": 2,
            }
            for index, total in enumerate(example_sums)
        ],
        best_index=2,
        usage={"input_tokens": 63, "scored_tokens": 60},
    ).model_dump(exclude_none=True)
    responses = {
        "200": {
            "description": "Complete scores for every candidate. All scores are finite; "
            "failures return an error instead of a partial result. Example values "
            "are illustrative, not measured model outputs.",
            "content": {
                "application/json": {
                    "schema": schema_ref("RawSystemOneResponse"),
                    "example": success,
                }
            },
        },
        "401": {
            "description": "Missing or invalid bearer API key when server authentication is enabled.",
            "content": {
                "application/json": {"schema": schema_ref("AuthenticationError")}
            },
        },
    }
    responses.update(
        {
            status: {
                "description": description,
                "content": {
                    "application/json": {"schema": schema_ref("RawSystemOneError")}
                },
            }
            for status, description in ERROR_RESPONSES.items()
        }
    )
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "SGLang rawsystemone API",
            "version": "1.2",
            "description": "Standalone contract for the native scoring endpoint from "
            "sglang_rawsystemone_spec_v1_2.md. Generated from the implemented Pydantic "
            "models with wire-format and semantic annotations. Regenerate with "
            "python scripts/export_rawsystemone_openapi.py.",
        },
        "servers": [
            {
                "url": "http://localhost:30000",
                "description": "Example local SGLang server; replace with your deployment URL.",
            }
        ],
        "paths": {
            "/v1/rawsystemone": {
                "post": {
                    "operationId": "scoreRawSystemOne",
                    "summary": "Normalize conditional suffix likelihoods",
                    "description": "Tokenize the prefix once and append each option's "
                    "tokens encoded with add_special_tokens=false. Prefill the fixed "
                    "prefix before parallel option evaluation when cache reuse is useful. "
                    "Compute each suffix's conditional log-probability sum, divide by "
                    "its option token count, and apply softmax across the original "
                    "options. Prefix token scores contribute to neither numerator nor "
                    "denominator. Uses the already-loaded default "
                    "model; no generation, streaming, model selection, or sampling "
                    "parameters. Server limits apply to the number of options, each "
                    "candidate's native context length, and total logical candidate "
                    "tokens (default 1,048,576, including duplicates). Internal "
                    "batching and concurrency are configured by the server.",
                    "security": [{}, {"BearerAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": schema_ref("RawSystemOneRequest"),
                                "example": {
                                    "prefix": "Customer: Please move my appointment to Friday.\n\nThe requested operation is",
                                    "suffixes": [
                                        " booking.",
                                        " cancellation.",
                                        " rescheduling.",
                                    ],
                                    "return_token_logprobs": False,
                                },
                            }
                        },
                    },
                    "responses": responses,
                }
            }
        },
        "components": {
            "schemas": schemas,
            "securitySchemes": {
                "BearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": "Use Authorization: Bearer <api-key>. Required when "
                    "the server is configured with --api-key; otherwise anonymous "
                    "requests are accepted. This is an API key, not necessarily a JWT.",
                }
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--check", action="store_true", help="Fail if the artifact is stale."
    )
    args = parser.parse_args()
    rendered = (
        json.dumps(build_schema(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    if args.check:
        if (
            not args.output.exists()
            or args.output.read_text(encoding="utf-8") != rendered
        ):
            parser.exit(1, f"Schema is stale: {args.output}\n")
        print(f"Schema is up to date: {args.output}")
    else:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
