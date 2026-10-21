"""Compile the shared-state prompt used to train Valen decision checkpoints."""

from copy import deepcopy

from sglang.srt.layers.valen_head import ValenQuestion, pack_valen_layout
from sglang.srt.utils import ImageData

_NOUL = [
    ("true", "True / 是：满足问题中的条件。"),
    ("false", "False / 否：不满足问题中的条件。"),
]


def valen_options(question):
    if question.type == "noul":
        if question.criteria is not None:
            raise ValueError(
                "Valen noul questions use fixed true/false descriptions; omit criteria"
            )
        return _NOUL
    if question.type == "choice":
        return list(question.criteria.items())
    if not 2 <= len(question.criteria) <= 255:
        raise ValueError("Valen score questions require 2 to 255 levels")
    return [(str(i), description) for i, description in enumerate(question.criteria)]


def _text(value, tokenizer, *, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError("Valen instructions and criteria must be nonempty strings")
    special = set(tokenizer.all_special_tokens)
    special.update(
        str(token) for token in tokenizer.added_tokens_decoder.values() if token.special
    )
    if any(token in value for token in special):
        raise ValueError("Reserved tokenizer control token in Valen input")
    return value


def valen_messages(request, tokenizer):
    """Render native message state and collect images in placeholder order."""
    if isinstance(request.state, str):
        messages = [{"role": "user", "content": request.state}]
    elif isinstance(request.state, dict) and set(request.state) == {"messages"}:
        messages = deepcopy(request.state["messages"])
    else:
        raise ValueError("Valen state must be text or an object containing messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Valen state.messages must be a nonempty list")
    images = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "user",
            "assistant",
            "system",
        }:
            raise ValueError("Unsupported Valen state message role")
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list) or not content:
            raise ValueError("Valen message content must be text or content blocks")
        normalized = []
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Invalid Valen content block")
            if item.get("type") == "text":
                normalized.append(
                    {
                        "type": "text",
                        "text": _text(item.get("text"), tokenizer, allow_empty=True),
                    }
                )
            elif item.get("type") == "image_url":
                source = item.get("image_url")
                if not isinstance(source, dict) or not isinstance(
                    source.get("url"), str
                ):
                    raise ValueError("Valen image_url requires a url string")
                images.append(
                    ImageData(url=source["url"], detail=source.get("detail", "auto"))
                )
                normalized.append({"type": "image"})
            else:
                raise ValueError(
                    "Valen currently supports text and image_url content blocks"
                )
        message["content"] = normalized
    if request.images:
        if not isinstance(request.state, str):
            raise ValueError(
                "Use image_url blocks for images in a Valen messages state"
            )
        images.extend(
            ImageData(url=i.url, detail=i.detail or "auto") for i in request.images
        )
        messages[0]["content"] = [{"type": "image"} for _ in request.images] + messages[
            0
        ]["content"]
    return messages, images


def encode_valen(tokenizer, request, messages, *, max_length, image_token_counts):
    # Tokenizing only the template leaves one placeholder per image. The native
    # multimodal processor expands them later, and the layout shifts accordingly.
    ids = list(
        tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=False
        )
    )
    context_end = len(ids)

    def append(text, span=None):
        start = len(ids)
        if span is None:
            ids.extend(tokenizer.encode(text, add_special_tokens=False))
            return None
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids.extend(encoded["input_ids"])
        indices = [
            i
            for i, (a, b) in enumerate(encoded["offset_mapping"])
            if a < span[1] and b > span[0]
        ]
        if not indices:
            raise ValueError("Empty token span for Valen decision role")
        return start + indices[0], start + indices[-1] + 1

    append("<|im_start|>user\n")
    questions = []
    for i, question in enumerate(request.questions.values()):
        instruction = _text(question.instructions, tokenizer)
        prefix = f"Question {i + 1}\nTask: {question.type}\nQuestion: "
        instruction_span = append(
            prefix + instruction + "\nCandidates:\n",
            (len(prefix), len(prefix) + len(instruction)),
        )
        spans = []
        for key, description in valen_options(question):
            _text(key, tokenizer)
            _text(description, tokenizer)
            prefix = key + ": " if question.type != "score" else ""
            spans.append(
                append(
                    prefix + description, (len(prefix), len(prefix) + len(description))
                )
            )
            append("\n")
        questions.append((instruction_span, tuple(spans)))
    append("<|im_end|>\n<|im_start|>assistant\n")
    layout_questions = []
    for i, (instruction, candidates) in enumerate(questions):
        append(f"Question {i + 1} Decision:")
        layout_questions.append(ValenQuestion(instruction, candidates, len(ids) - 1))
        append("\n")
    length = len(ids) + sum(count - 1 for count in image_token_counts)
    if length > max_length:
        raise ValueError(
            f"Valen prompt has {length} tokens, exceeding max_length={max_length}; no truncation"
        )
    return ids, pack_valen_layout(len(ids), context_end, layout_questions)
