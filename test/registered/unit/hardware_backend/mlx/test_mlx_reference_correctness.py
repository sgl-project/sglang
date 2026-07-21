"""Reference-equivalence correctness for the MLX MoE execution path.

The existing ``test/registered/models/test_qwen{2,3}_moe_mlx_correctness.py`` are
black-box smoke tests: they only check that a served model emits *plausible* text
("Paris" appears, "4" appears). A subtly broken KV cache, slot allocator, or
prefill-chunking change can still pass those.

This is the strong guard. The SGLang MLX backend executes models by wrapping
``mlx_lm`` (``mlx_lm.load`` + the model's own forward, with attention patched for
SGLang's cache) and decodes greedily (``mx.argmax``). So the ground truth for "did
SGLang corrupt the output?" is raw, *unpatched* ``mlx_lm`` greedy generation on the
same prompt. We record those reference tokens, then drive ``MlxModelRunner``
(prefill + decode_batch) and assert the tokens match exactly, up to and including
EOS. Empirically the agreement is exact (not approximate), so any divergence here
is a real regression.

A second test pins batching isolation: a prompt decoded inside a multi-request
``decode_batch`` must yield the same tokens as when decoded alone -- guarding the
slot/cache bookkeeping that single-request black-box tests never touch.

Memory safety: the runner patches the model in-place, so the reference needs its
own *separate* ``mlx_lm.load``. Loading both copies at once doubles resident weight
memory (~2x model) and can trigger an unrecoverable Metal command-buffer OOM that
hard-reboots a memory-constrained Mac. So we load the reference, record tokens,
fully release it (``del`` + ``gc`` + ``mx.clear_cache``), and only then build the
runner -- keeping peak at a single model copy. A pre-flight free-memory check skips
(never crashes) when there isn't headroom for even one copy.

MLX-gated like its siblings: registered on the CPU suite but skipped wherever the
``mlx`` package is absent (all current CI runners), so it runs for real only on
Apple Silicon. Override the model with ``SGLANG_MLX_TEST_MODEL`` (e.g. a local path
or ``mlx-community/Qwen3-30B-A3B-4bit`` for qwen3_moe); bump
``SGLANG_MLX_TEST_MIN_FREE_GB`` accordingly for larger models.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import unittest

from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-b-e2e-mlx")

_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)
_SKIP_REASON = "requires mlx + mlx_lm (Apple Silicon only)"

# Default to the portable mlx-community 4-bit qwen2_moe repo; override to a local
# dir or another MoE arch (e.g. Qwen3-30B-A3B-4bit) via the env var.
MODEL_PATH = os.environ.get(
    "SGLANG_MLX_TEST_MODEL", "mlx-community/Qwen1.5-MoE-A2.7B-Chat-4bit"
)
MEM_FRACTION_STATIC = float(os.environ.get("SGLANG_MLX_TEST_MEM_FRACTION", "0.7"))
# Skip (do NOT crash) unless this much system memory is free: enough for ONE
# resident model copy plus activations. An MLX Metal OOM is uncatchable and can
# reboot the machine, so this guard must fail safe. ~12 GB suits the default
# 14.3B-param 4-bit qwen2_moe; raise it for qwen3_moe (~18+).
MIN_FREE_GB = float(os.environ.get("SGLANG_MLX_TEST_MIN_FREE_GB", "12"))

# Short prompts with deterministic, quickly-terminating greedy answers.
PROMPTS = [
    "List the first 10 prime numbers, comma separated.",
    "What is the capital of France? Answer in one word.",
    "Write one short sentence about the ocean.",
]
MAX_NEW_TOKENS = 64  # hard cap; most prompts hit EOS well before this
BATCH_HORIZON = 24  # fixed step count for the batching-isolation test


def _available_gb():
    try:
        import psutil

        return psutil.virtual_memory().available / 1024**3
    except Exception:
        return None  # psutil absent -> skip the pre-flight check


@unittest.skipUnless(_HAS_MLX, _SKIP_REASON)
class TestMlxReferenceCorrectness(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        from mlx_lm import load

        avail = _available_gb()
        if avail is not None and avail < MIN_FREE_GB:
            raise unittest.SkipTest(
                f"insufficient free memory: {avail:.1f} GB < {MIN_FREE_GB} GB needed "
                f"to safely load {MODEL_PATH} (override SGLANG_MLX_TEST_MIN_FREE_GB)"
            )

        # --- Phase 1: reference tokens from UNPATCHED mlx_lm (one copy resident) ---
        try:
            ref_model, cls.tokenizer = load(
                MODEL_PATH, tokenizer_config={"trust_remote_code": True}
            )
        except Exception as exc:  # not cached / offline / bad path
            raise unittest.SkipTest(f"could not load {MODEL_PATH}: {exc}")

        eos = getattr(cls.tokenizer, "eos_token_ids", None) or {
            cls.tokenizer.eos_token_id
        }
        cls.eos_ids = set(eos)

        cls.cases = []  # (prompt, prompt_ids, reference_token_ids)
        for prompt in PROMPTS:
            prompt_ids = list(
                cls.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True
                )
            )
            ref_ids = cls._reference_greedy(
                ref_model, cls.tokenizer, prompt_ids, MAX_NEW_TOKENS
            )
            cls.cases.append((prompt, prompt_ids, ref_ids))

        # --- Release the reference BEFORE building the runner (cap peak at 1x) ---
        del ref_model
        gc.collect()
        mx.clear_cache()
        active_gb = mx.get_active_memory() / 1024**3
        if active_gb > 2.0:
            # Weights were not released; loading the runner now would double resident
            # memory and risk an OOM. Bail safely instead.
            raise unittest.SkipTest(
                f"reference model not released (active={active_gb:.1f} GB); "
                "skipping to avoid a double-resident OOM"
            )

        # --- Phase 2: SGLang runner (one copy resident) ---
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        cls.runner = MlxModelRunner(
            model_path=MODEL_PATH,
            trust_remote_code=True,
            disable_radix_cache=True,  # per-request contiguous caches; no big pool
            mem_fraction_static=MEM_FRACTION_STATIC,
        )
        cls.runner.init_cache_pools(req_to_token_pool=None)

    @classmethod
    def tearDownClass(cls):
        runner = getattr(cls, "runner", None)
        if runner is not None:
            runner.clear()
        cls.runner = None
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _reference_greedy(model, tokenizer, prompt_ids, max_new):
        """Ground-truth token ids from raw, unpatched mlx_lm greedy generation."""
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=0.0)  # greedy / argmax
        out = []
        for resp in stream_generate(
            model, tokenizer, mx.array(prompt_ids), max_tokens=max_new, sampler=sampler
        ):
            out.append(int(resp.token))
        return out

    def _prefill(self, rid, prompt_ids):
        return int(
            self.runner.prefill(
                req_id=rid,
                new_token_ids=list(prompt_ids),
                full_token_ids=list(prompt_ids),
                prefix_slot_ids=[],
                new_slot_ids=[],
                req_pool_idx=0,
            )
        )

    def _decode(self, rids):
        return [int(t) for t in self.runner.decode_batch(rids)]

    def _sglang_greedy(self, rid, prompt_ids, max_new):
        """SGLang MLX greedy generation, stopping at EOS like the reference."""
        tok = self._prefill(rid, prompt_ids)
        out = [tok]
        while len(out) < max_new and tok not in self.eos_ids:
            tok = self._decode([rid])[0]
            out.append(tok)
        self.runner.remove_request(rid)
        return out

    def _diff_msg(self, prompt, ref, sgl):
        horizon = min(len(ref), len(sgl))
        first = next((j for j in range(horizon) if ref[j] != sgl[j]), horizon)
        return (
            f"\nprompt: {prompt!r}"
            f"\n  first divergence @ index {first} (len ref={len(ref)} sgl={len(sgl)})"
            f"\n  ref text: {self.tokenizer.decode(ref)!r}"
            f"\n  sgl text: {self.tokenizer.decode(sgl)!r}"
        )

    # --- tests ------------------------------------------------------------

    def test_greedy_matches_reference_exact(self):
        """SGLang MLX greedy output == unpatched mlx_lm greedy output, token-for-token."""
        for i, (prompt, prompt_ids, ref) in enumerate(self.cases):
            sgl = self._sglang_greedy(f"ref-{i}", prompt_ids, MAX_NEW_TOKENS)
            self.assertEqual(sgl, ref, self._diff_msg(prompt, ref, sgl))

    def test_batched_decode_matches_solo(self):
        """A request's tokens are identical whether decoded alone or in a batch.

        Pins slot/cache isolation: ``decode_batch`` over several concurrent
        requests must not let one request's state bleed into another's.
        """
        ids_list = [prompt_ids for (_, prompt_ids, _) in self.cases]

        # Solo: prefill, decode a fixed horizon, remove -- one request at a time.
        solo = []
        for i, ids in enumerate(ids_list):
            seq = [self._prefill(f"solo-{i}", ids)]
            for _ in range(BATCH_HORIZON - 1):
                seq.append(self._decode([f"solo-{i}"])[0])
            self.runner.remove_request(f"solo-{i}")
            solo.append(seq)

        # Batched: prefill all, then advance them together in one decode_batch.
        rids = [f"batch-{i}" for i in range(len(ids_list))]
        batched = [[self._prefill(rid, ids)] for rid, ids in zip(rids, ids_list)]
        for _ in range(BATCH_HORIZON - 1):
            for j, t in enumerate(self._decode(rids)):
                batched[j].append(t)
        for rid in rids:
            self.runner.remove_request(rid)

        for i, (prompt, _, _) in enumerate(self.cases):
            self.assertEqual(
                batched[i], solo[i], self._diff_msg(prompt, solo[i], batched[i])
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
