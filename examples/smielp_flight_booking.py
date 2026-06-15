"""
SMIELP Flight Booking — End-to-End Usage Example
=================================================
Demonstrates the complete workflow for single-forward-pass NER/NLU extraction
using the Simultaneous Multi-Intent & Entity Logit Processor (SMIELP).

What this script does
---------------------
1. Defines a NER schema (intents + entity slots) for a flight booking domain.
2. Anchors the schema to the Qwen2.5-0.5B-Instruct tokenizer vocabulary.
3. Registers the SMIELP custom logit processor with a running sglang server.
4. Sends extraction requests and prints structured results from meta_info["smielp"].
5. Compares Phase A (logit slicing) vs Phase B (embedding-space cosine similarity).

Setup
-----
    # Terminal 1 — start server with both required flags
    python -m sglang.launch_server \\
        --model-path Qwen/Qwen2.5-0.5B-Instruct \\
        --enable-custom-logit-processor \\
        --enable-return-hidden-states \\
        --port 30000

    # Terminal 2 — run this script
    pip install openai transformers torch
    python examples/smielp_flight_booking.py

Performance note
----------------
XGrammar JSON extraction for the same 4-field schema needs ~20-40 decode tokens.
SMIELP uses exactly 1 decode token for any number of output fields.
Expected speedup: 5-15x on a single A100 with batch_size=1.
"""

import json
import sys
import pathlib

# ── Add project source to path ─────────────────────────────────────────────────
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from sglang.srt.logit_slicing import (
    NERSchema,
    IntentSchema,
    SlotSchema,
    VocabAnchor,
    build_anchor_config,
    build_phase_b_config,
    SimultaneousMultiIntentEntityLogitProcessor,
    SMIELPWithHiddenStates,
)
from sglang.srt.logit_slicing.vocab_anchor import STRATEGY_FIRST_TOKEN

# ── 1. Define the NER schema ───────────────────────────────────────────────────
#
# Labels must be single tokens in the model vocabulary (or use strategy='explicit'
# to hand-pick token IDs).  VocabAnchor warns when a label tokenises to multiple
# sub-words and uses the first token automatically.

schema = NERSchema(
    intents=IntentSchema(
        labels=["book", "cancel", "status"],   # all single tokens in Qwen2.5
    ),
    slots=[
        SlotSchema(name="city",  labels=["O", "B", "I"]),   # BIO tags
        SlotSchema(name="date",  labels=["O", "B", "I"]),
        SlotSchema(name="hotel", labels=["none", "present"]),   # binary presence
    ],
)

# ── 2. Anchor labels to vocabulary token IDs ──────────────────────────────────
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    print(f"[setup] Tokenizer loaded.  vocab_size={tokenizer.vocab_size}")
except ImportError:
    print("ERROR: 'transformers' not installed.  pip install transformers", file=sys.stderr)
    sys.exit(1)

# Phase A config — pure logit slicing (no --enable-return-hidden-states needed)
custom_params_phase_a = build_anchor_config(
    schema, tokenizer, strategy=STRATEGY_FIRST_TOKEN
)
print("[setup] Phase A config anchored:")
print(json.dumps(custom_params_phase_a["schema"], indent=2))

# Phase B config — embedding-space cosine similarity
# Requires the model's embedding matrix.  Here we load it from a HuggingFace model.
try:
    from transformers import AutoModelForCausalLM
    import torch

    print("[setup] Loading model for embedding matrix (this may take ~30s) …")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    embedding_matrix = model.model.embed_tokens.weight.detach().float()
    del model   # free memory; we only need the embedding weights

    custom_params_phase_b = build_phase_b_config(
        schema, tokenizer, embedding_matrix, strategy=STRATEGY_FIRST_TOKEN
    )
    print("[setup] Phase B config built.")
    print(f"[setup] label_embeddings shapes: "
          + ", ".join(f"{k}={v.shape}" for k, v in custom_params_phase_b["label_embeddings"].items()))
    _phase_b_available = True
except Exception as e:
    print(f"[setup] Phase B unavailable (model load failed: {e}); will skip Phase B.")
    _phase_b_available = False

# ── 3. Serialize the processors ───────────────────────────────────────────────
proc_a_str = SimultaneousMultiIntentEntityLogitProcessor.to_str()
proc_b_str = SMIELPWithHiddenStates.to_str()
print(f"[setup] Phase A processor serialized ({len(proc_a_str)} bytes).")
print(f"[setup] Phase B processor serialized ({len(proc_b_str)} bytes).")

# ── 4. Connect to sglang server ───────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: 'openai' not installed.  pip install openai", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url="http://localhost:30000/v1", api_key="none")

SYSTEM_PROMPT = (
    "You are a flight booking assistant. "
    "Extract the intent and entity slots from the user message."
)

TEST_PROMPTS = [
    "I want to book a flight to Paris on Friday and need a hotel.",
    "Can you cancel my upcoming London flight?",
    "What's the status of my Tokyo reservation?",
    "Book a ticket to Berlin, arriving next Monday. No hotel.",
]

# ── 5. Run Phase A extraction ─────────────────────────────────────────────────
print("\n" + "="*65)
print("  Phase A — Logit Slicing (max_new_tokens=1, no hidden states)")
print("="*65)

for prompt in TEST_PROMPTS:
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=1,
        temperature=0.0,
        extra_body={
            "custom_logit_processor": proc_a_str,
            "custom_params": custom_params_phase_a,
        },
    )

    # Results arrive in meta_info["smielp"][0]
    meta = response.choices[0].message.model_extra.get("meta_info", {})
    result = meta.get("smielp", [{}])[0]

    intent = result.get("intent", {})
    entities = result.get("entities", [])
    print(f"\nPrompt : {prompt}")
    print(f"  Intent  : {intent.get('label', '?')} (conf={intent.get('confidence', 0):.2f})")
    for ent in entities:
        print(f"  {ent['slot']:10}: {ent['tag']} (conf={ent['confidence']:.2f})")

# ── 6. Run Phase B extraction (if available) ──────────────────────────────────
if _phase_b_available:
    print("\n" + "="*65)
    print("  Phase B — Embedding Cosine Similarity (requires hidden states)")
    print("="*65)

    for prompt in TEST_PROMPTS:
        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=1,
            temperature=0.0,
            extra_body={
                "custom_logit_processor": proc_b_str,
                "custom_params": custom_params_phase_b,
                "return_hidden_states": True,      # requires --enable-return-hidden-states
            },
        )

        meta   = response.choices[0].message.model_extra.get("meta_info", {})
        result = meta.get("smielp", [{}])[0]
        intent = result.get("intent", {})

        print(f"\nPrompt : {prompt}")
        print(f"  Intent  : {intent.get('label', '?')} (conf={intent.get('confidence', 0):.2f})")
        print(f"  Mode    : {result.get('mode', '?')}")
        print(f"  HS L2   : {result.get('hidden_state_l2', 'N/A'):.3f}")
        for ent in result.get("entities", []):
            print(f"  {ent['slot']:10}: {ent['tag']} (conf={ent['confidence']:.2f})")

# ── 7. Comparison: XGrammar JSON extraction for the same schema ───────────────
print("\n" + "="*65)
print("  XGrammar Baseline — JSON schema constrained decoding")
print("="*65)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent":   {"type": "string", "enum": ["book", "cancel", "status"]},
        "city_tag": {"type": "string", "enum": ["O", "B", "I"]},
        "date_tag": {"type": "string", "enum": ["O", "B", "I"]},
        "hotel":    {"type": "string", "enum": ["none", "present"]},
    },
    "required": ["intent", "city_tag", "date_tag", "hotel"],
    "additionalProperties": False,
}

for prompt in TEST_PROMPTS[:2]:   # only 2 for brevity — each takes ~20 decode steps
    response = client.chat.completions.create(
        model="default",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=64,
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "ner", "schema": JSON_SCHEMA},
        },
    )
    content = response.choices[0].message.content
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {"raw": content}

    print(f"\nPrompt : {prompt}")
    print(f"  Result : {result}")

print("\n[done] SMIELP example complete.")
