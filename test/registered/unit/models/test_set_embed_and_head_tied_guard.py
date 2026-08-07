"""Guard `set_embed_and_head` against tied word embeddings.

When `config.tie_word_embeddings` is set, model `__init__` aliases the head
onto the embedding table:

    if self.config.tie_word_embeddings:
        self.lm_head = self.model.embed_tokens

`self.lm_head` and `self.model.embed_tokens` are then the *same module*, so an
unconditional

    del self.model.embed_tokens.weight
    del self.lm_head.weight

deletes the same attribute twice and the second statement raises
`AttributeError: 'VocabParallelEmbedding' object has no attribute 'weight'`.
`set_embed_and_head` is on the EAGLE path (`EAGLEWorker.__init__` calls it on
the draft model), so this fires whenever a draft checkpoint ties its
embeddings -- e.g. `linborui/EAGLE-Llama-3.2-3B-Instruct`, which reports
`tie_word_embeddings: true`.

This is a source-level ratchet, in the style of the other `test_*_ratchet.py`
files: it walks `python/sglang/srt/models/` and fails if any class that
performs the aliasing also deletes both weights unconditionally. That keeps
newly added models from reintroducing the pattern.
"""

import ast
import pathlib
import re
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_MODELS_DIR = _REPO_ROOT / "python" / "sglang" / "srt" / "models"

# `self.lm_head = self.model.embed_tokens` under an `if ... tie_word_embeddings:`
_ALIAS_RE = re.compile(
    r"if\s+[^\n]*tie_word_embeddings\s*:\s*\n\s*self\.lm_head\s*=\s*self\.model\.embed_tokens"
)


class TestSetEmbedAndHeadTiedGuard(CustomTestCase):
    def test_tied_embedding_delete_is_guarded(self):
        offenders = []
        for path in sorted(_MODELS_DIR.glob("*.py")):
            offenders.extend(_find_unguarded(path))

        self.assertFalse(
            offenders,
            msg=(
                "`set_embed_and_head` deletes both `model.embed_tokens.weight` "
                "and `lm_head.weight` unconditionally in a class that aliases "
                "the two together when `tie_word_embeddings` is set. The second "
                "`del` raises AttributeError for tied checkpoints. Guard it, as "
                "`qwen3.py` and `qwen3_5_mtp.py` do:\n  " + "\n  ".join(offenders)
            ),
        )


def _find_unguarded(path: pathlib.Path):
    """Yield `<rel_path>:<lineno> <ClassName>` for each offending class."""
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    # Cheap pre-filter: most model files never define this method.
    if "set_embed_and_head" not in source:
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    found = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        class_src = ast.get_source_segment(source, cls) or ""
        if not _ALIAS_RE.search(class_src):
            continue
        for fn in cls.body:
            if not (
                isinstance(fn, ast.FunctionDef) and fn.name == "set_embed_and_head"
            ):
                continue
            body = ast.get_source_segment(source, fn) or ""
            if "del self.model.embed_tokens.weight" not in body:
                continue
            if "del self.lm_head.weight" not in body:
                continue
            # Any of the in-tree guards is accepted: a `hasattr` check, a
            # `tie_word_embeddings` branch, or a try/except around the deletes.
            if any(tok in body for tok in ("hasattr", "tie_word_embeddings", "try:")):
                continue
            rel = path.relative_to(_REPO_ROOT)
            found.append(f"{rel}:{fn.lineno} {cls.name}")
    return found


if __name__ == "__main__":
    unittest.main()
