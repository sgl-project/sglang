"""MoL auto-routing helpers for the route_decode KV-reuse path.

Ports the router-prompt / route-signal / route-map builders from the old
oai_api_connectivity_test client (test_lora_adapter_oai_api.py) so the SERVER
can synthesize the lora_router custom_params when a client calls the public
router model_id (e.g. mindlab-research/Macaron-V1-Preview-749B) with just a
prompt. This keeps the routing knowledge (per-LoRA descriptions / rules /
signals) on the server, parsed from the L*.md LoRA library.

No sglang imports here on purpose: pure parsing + string building so it can be
unit-tested standalone.

Adapter-name note: the L*.md library uses canonical names
(l1_living_vita_tau3, l2_swe_tb2, ...). OUR sglang server advertises shorter
served names (l1_living_vita_tau3, l2_swe_tb2, ...) loaded from the same shadow_loras weights.
route_to_adapter is therefore remapped through MOL_ROUTE_NAME_MAP / the
name_override table to OUR served names.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

ENTRY_ROUTE_ID = "L0"

# Default L<route> -> OUR served adapter name. Override via MOL_ROUTE_NAME_MAP
# (JSON) or the name_override arg. Must match LORA_PATHS_ARGS keys in the launcher.
DEFAULT_NAME_OVERRIDE: Dict[str, str] = {
    "L0": "l0_chat",
    "L1": "l1_living_vita_tau3",
    "L2": "l2_swe_tb2",
    "L3": "l3_a2ui",
    "L4": "l4_openclaw_pinch",
}


def _default_library_dir() -> Path:
    """Snapshot shipped inside the patch; override with MOL_LORA_LIBRARY_DIR."""
    env = os.environ.get("MOL_LORA_LIBRARY_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "mol_lora_library"


def parse_lora_markdown(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    meta: Dict[str, str] = {}
    body = text
    if text.startswith("---\n"):
        _, raw_meta, body = text.split("---", 2)
        for line in raw_meta.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()

    sections: Dict[str, List[str]] = {}
    current = None
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        if line.startswith("## "):
            current = line[3:].strip().lower().replace(" ", "_")
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)

    def section_text(name: str) -> str:
        return "\n".join(sections.get(name, [])).strip()

    def bullets(name: str) -> List[str]:
        out: List[str] = []
        for line in sections.get(name, []):
            stripped = line.strip()
            if stripped.startswith("- "):
                out.append(stripped[2:].strip())
        return out

    def comma_list(name: str) -> List[str]:
        raw = section_text(name)
        return [item.strip() for item in raw.split(",") if item.strip()]

    def section_items(name: str) -> List[str]:
        out: List[str] = []
        for line in sections.get(name, []):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("- "):
                stripped = stripped[2:].strip()
            out.append(stripped)
        return out

    task_id = meta.get("id") or path.stem
    return {
        "id": task_id,
        "task": meta.get("task", task_id),
        "level": meta.get("level", ""),
        "adapter_name": meta.get("adapter_name", ""),
        "priority": int(meta.get("priority", "0") or 0),
        "description": section_text("description"),
        "routing_rules": bullets("routing_rules"),
        "strong_signals": comma_list("strong_signals"),
        "positive_signals": comma_list("positive_signals"),
        "negative_signals": comma_list("negative_signals"),
        "examples": section_items("examples"),
        "library_path": str(path),
    }


def load_lora_library(library_dir: Path | str | None = None) -> Dict[str, Dict[str, Any]]:
    library_dir = Path(library_dir) if library_dir else _default_library_dir()
    if not library_dir.exists():
        raise FileNotFoundError(f"Missing LoRA Library directory: {library_dir}")
    tasks: Dict[str, Dict[str, Any]] = {}
    for path in sorted(library_dir.glob("*.md")):
        item = parse_lora_markdown(path)
        tasks[item["id"]] = item
    if not tasks:
        raise FileNotFoundError(f"No .md task files found in LoRA Library: {library_dir}")
    if ENTRY_ROUTE_ID not in tasks:
        raise FileNotFoundError(f"LoRA Library must include {ENTRY_ROUTE_ID}.md: {library_dir}")
    return tasks


def sorted_tasks(tasks: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    return sorted(
        tasks.items(),
        key=lambda item: (item[0] != ENTRY_ROUTE_ID, -int(item[1].get("priority", 0) or 0), item[0]),
    )


# --- router instruction builders (verbatim port) ---

def build_model_id_list(tasks: Dict[str, Dict[str, Any]]) -> str:
    return "\n".join(
        f"- {task_id}: {task.get('description', '')}"
        for task_id, task in sorted_tasks(tasks)
    )


def build_routing_rules(tasks: Dict[str, Dict[str, Any]]) -> str:
    rules: List[str] = []
    for task_id, task in sorted_tasks(tasks):
        for rule in task.get("routing_rules", []):
            rules.append(f"{task_id}: {rule}")
    return "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1))


def build_routing_examples(tasks: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    for task_id, task in sorted_tasks(tasks):
        for example in task.get("examples", []):
            if "=>" in example:
                user, route = example.split("=>", 1)
                route = route.strip() or task_id
            else:
                user, route = example, task_id
            lines.append(f"User: {user.strip()}\nmodel_id={route}")
    return "\n".join(lines)


def build_router_instruction(tasks: Dict[str, Dict[str, Any]]) -> str:
    examples = build_routing_examples(tasks)
    examples_block = f"Examples:\n{examples}\n\n" if examples else ""
    return (
        "Router instruction:\n"
        "You are running inside L0, the entry chat LoRA for a Mixture-of-LoRA service. "
        "Choose exactly one model_id for the next decoding phase. "
        "Classify by the user's goal and execution environment, not by keyword overlap. "
        "Return L0 for ordinary chat, general reasoning, language work, or ambiguous requests. "
        "Route to specialist LoRAs only when one task family clearly matches. "
        "Slash-separated labels mean one adapter covers all listed task families.\n\n"
        "Available model ids:\n"
        f"{build_model_id_list(tasks)}\n\n"
        "Routing rules:\n"
        f"{build_routing_rules(tasks)}\n\n"
        f"{examples_block}"
        "Return exactly one line in this format and nothing else:\n"
        f"model_id=<{'|'.join(tasks)}>\n\n"
        "model_id="
    )


def build_route_signals(tasks: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    route_signals: Dict[str, List[str]] = {}
    for task_id, task in tasks.items():
        if task_id == ENTRY_ROUTE_ID:
            continue
        signals: List[str] = []
        for field in ("strong_signals", "positive_signals"):
            for signal in task.get(field, []):
                if isinstance(signal, str) and signal.strip():
                    signals.append(signal.strip())
        route_signals[task_id] = list(dict.fromkeys(signals))
    return route_signals


def build_route_library(tasks: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Per-route library block for the weighted metadata guardrail (alpha parity).

    Unlike build_route_signals (strong+positive merged, used by the legacy
    single-hit override) this carries strong / positive / NEGATIVE signals plus
    priority separately so the decision side can run alpha's weighted scoring
    (priority + strong*100 + positive*10 - negative*250). Entry route (L0) is
    excluded: it is the default, not a scored specialist.
    """
    def _clean(items: Any) -> List[str]:
        out: List[str] = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, str) and it.strip():
                    out.append(it.strip())
        return list(dict.fromkeys(out))

    library: Dict[str, Dict[str, Any]] = {}
    for task_id, task in tasks.items():
        if task_id == ENTRY_ROUTE_ID:
            continue
        library[task_id] = {
            "priority": int(task.get("priority", 0) or 0),
            "strong_signals": _clean(task.get("strong_signals")),
            "positive_signals": _clean(task.get("positive_signals")),
            "negative_signals": _clean(task.get("negative_signals")),
        }
    return library


# --- weighted metadata guardrail (verbatim port of alpha RouterHarness) ---

GENERAL_L0_PREFIXES = (
    "translate ",
    "rewrite ",
    "proofread ",
    "explain ",
    "summarize ",
    "compare ",
    "what is ",
    "why does ",
    "give me an overview",
)


def is_general_l0_request(user_text: str) -> bool:
    return user_text.strip().lower().startswith(GENERAL_L0_PREFIXES)


def signal_matches(text_lower: str, signal: str) -> bool:
    """Word-boundary match for ascii-token signals; substring for phrases/CJK."""
    signal = signal.strip().lower()
    if not signal:
        return False
    if re.fullmatch(r"[a-z0-9_+-]+", signal):
        return re.search(
            rf"(?<![a-z0-9_+-]){re.escape(signal)}(?![a-z0-9_+-])", text_lower
        ) is not None
    return signal in text_lower


def signal_hits(user_text: str, signals: List[str]) -> List[str]:
    lower = user_text.lower()
    return [s for s in signals if signal_matches(lower, s)]


def signal_weight(signal: str) -> int:
    signal = signal.strip()
    if not signal:
        return 0
    if any(ord(ch) > 127 for ch in signal):
        if len(signal) >= 6:
            return 4
        if len(signal) >= 4:
            return 3
    if len(signal) >= 40:
        return 6
    if len(signal) >= 20:
        return 4
    if any(ch in signal for ch in ("/", "_", "-", ":", "`", "*")):
        return 4
    if " " in signal:
        return 3
    return 1


def weighted_signal_score(hits: List[str]) -> int:
    return sum(signal_weight(hit) for hit in hits)


def route_by_library(
    user_text: str, route_library: Dict[str, Dict[str, Any]]
) -> Tuple[str, str]:
    """Alpha weighted-scoring router. Returns (route_id, decision).

    route_id == ENTRY_ROUTE_ID means "stay on L0". Mirrors
    RouterHarness.route_by_library: weighted strong/positive minus heavy negative,
    priority tie-break, general-prefix guard, and ambiguity -> L0.
    """
    if not isinstance(route_library, dict) or not route_library:
        return ENTRY_ROUTE_ID, "no_library"
    candidates: List[Dict[str, Any]] = []
    for route_id, entry in route_library.items():
        if not isinstance(entry, dict):
            continue
        strong_hits = signal_hits(user_text, entry.get("strong_signals") or [])
        positive_hits = signal_hits(user_text, entry.get("positive_signals") or [])
        if not strong_hits and not positive_hits:
            continue
        strong_score = weighted_signal_score(strong_hits)
        positive_score = weighted_signal_score(positive_hits)
        if strong_score == 0 and positive_score < 2:
            continue
        negative_hits = signal_hits(user_text, entry.get("negative_signals") or [])
        negative_score = weighted_signal_score(negative_hits)
        priority = int(entry.get("priority", 0) or 0)
        score = priority + strong_score * 100 + positive_score * 10 - negative_score * 250
        if score <= 0:
            continue
        candidates.append({
            "route_id": route_id,
            "score": score,
            "strong_count": len(strong_hits),
            "strong_score": strong_score,
            "positive_score": positive_score,
        })

    if not candidates:
        return ENTRY_ROUTE_ID, "default_entry_lora"
    candidates.sort(key=lambda c: (c["score"], c["strong_count"]), reverse=True)
    top = candidates[0]
    if is_general_l0_request(user_text) and top["strong_score"] <= 1 and top["positive_score"] == 0:
        return ENTRY_ROUTE_ID, "general_entry_lora_guard"
    if len(candidates) == 1 or top["score"] > candidates[1]["score"]:
        return top["route_id"], "specialist_signal"
    return ENTRY_ROUTE_ID, "ambiguous_specialist_signal_default_entry_lora"


def answer_prompt(user_text: str) -> str:
    return f"User request:\n{user_text.strip()}\n\nAnswer:"


def resolve_name_override() -> Dict[str, str]:
    raw = os.environ.get("MOL_ROUTE_NAME_MAP")
    if not raw:
        return dict(DEFAULT_NAME_OVERRIDE)
    try:
        m = json.loads(raw)
        if isinstance(m, dict):
            return {str(k): str(v) for k, v in m.items()}
    except Exception:
        pass
    return dict(DEFAULT_NAME_OVERRIDE)


def build_route_to_adapter(
    tasks: Dict[str, Dict[str, Any]], name_override: Dict[str, str] | None = None
) -> Dict[str, str]:
    """Map route id (L0..L4) -> OUR served adapter name.

    Prefer the explicit override (our served names); fall back to the library's
    canonical adapter_name only if a route is missing from the override.
    """
    override = name_override if name_override is not None else resolve_name_override()
    out: Dict[str, str] = {}
    for task_id, task in tasks.items():
        out[task_id] = override.get(task_id) or str(task.get("adapter_name") or task_id)
    return out


def build_lora_router_params(
    user_text: str,
    tasks: Dict[str, Dict[str, Any]],
    *,
    keep_prefix_token_count: int,
    route_to_adapter: Dict[str, str],
    router_max_tokens: int = 16,
    decode_tokens: int = 32,
    enable_kv_reuse: bool = True,
) -> Dict[str, Any]:
    """Full custom_params['lora_router'] block for route_decode + KV-reuse."""
    entry_adapter = route_to_adapter.get(ENTRY_ROUTE_ID, "l0_chat")
    return {
        "mode": "route_decode",
        "entry_route_id": ENTRY_ROUTE_ID,
        "base_route_id": ENTRY_ROUTE_ID,
        "reset_to_route_id": ENTRY_ROUTE_ID,
        "base_route_adapter": entry_adapter,
        "route_to_adapter": route_to_adapter,
        "route_signals": build_route_signals(tasks),
        "route_library": build_route_library(tasks),
        "router_max_tokens": int(router_max_tokens),
        "decode_tokens": int(decode_tokens),
        "enable_kv_reuse": bool(enable_kv_reuse),
        "keep_prefix_token_count": int(keep_prefix_token_count),
        "query_prefix_token_count": int(keep_prefix_token_count),
        "specialist_context_token_count": int(keep_prefix_token_count),
        "query_cache_reused_token_count": int(keep_prefix_token_count) if enable_kv_reuse else 0,
        "task_reprefill_token_count": 0 if enable_kv_reuse else int(keep_prefix_token_count),
        "task_reprefill_required": False,
        "user_text": user_text,
    }


def build_router_prompt(user_text: str, tasks: Dict[str, Dict[str, Any]]) -> str:
    return answer_prompt(user_text) + "\n\n" + build_router_instruction(tasks)
