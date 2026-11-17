"""SWA MIS must match independent per-item scoring, including mixed batches."""

import asyncio
import gc
import tempfile
import unittest

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    Cohere2Config,
    Cohere2ForCausalLM,
    Gemma2Config,
    Gemma2ForCausalLM,
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="extra-a", runner_config="1-gpu-small")

VOCAB_SIZE = 12000  # covers MIS_DELIMITER_TOKEN_ID = 9999
LABELS = [10, 11, 12, 13]
W_SWA = 32


def _save_tokenizer(path: str) -> None:
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2}
    for idx in range(3, VOCAB_SIZE):
        vocab[f"t{idx}"] = idx
    tk = Tokenizer(WordLevel(vocab=vocab, unk_token="<pad>"))
    tk.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    ).save_pretrained(path)


def _init_and_save(model: torch.nn.Module, path: str) -> None:
    for param in model.parameters():
        if param.ndim >= 2:
            torch.nn.init.normal_(param, mean=0.0, std=0.12)
    model.to(torch.float16).save_pretrained(path)


def _build_gemma2(path: str, window: int) -> None:
    torch.manual_seed(42)
    _init_and_save(
        Gemma2ForCausalLM(
            Gemma2Config(
                vocab_size=VOCAB_SIZE,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
                sliding_window=window,
                max_position_embeddings=2048,
                hidden_act="gelu_pytorch_tanh",
                bos_token_id=2,
                eos_token_id=1,
                pad_token_id=0,
            )
        ),
        path,
    )


def _build_gemma3(path: str, window: int) -> None:
    torch.manual_seed(42)
    _init_and_save(
        Gemma3ForCausalLM(
            Gemma3TextConfig(
                vocab_size=VOCAB_SIZE,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=64,
                query_pre_attn_scalar=64,
                sliding_window=window,
                max_position_embeddings=2048,
                layer_types=["sliding_attention"] * 4,
                hidden_activation="gelu_pytorch_tanh",
                bos_token_id=2,
                eos_token_id=1,
                pad_token_id=0,
            )
        ),
        path,
    )


def _build_cohere2(path: str, window: int) -> None:
    torch.manual_seed(42)
    _init_and_save(
        Cohere2ForCausalLM(
            Cohere2Config(
                vocab_size=VOCAB_SIZE,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                sliding_window=window,
                max_position_embeddings=2048,
                layer_types=["sliding_attention"] * 4,
                bos_token_id=2,
                eos_token_id=1,
                pad_token_id=0,
            )
        ),
        path,
    )


REGIMES = {
    "A_packed_le_W": (
        [20 + (i % 50) for i in range(12)],
        [[100 + j * 10 + k for k in range(4)] for j in range(2)],
    ),
    "B_qitem_le_W_lt_packed": (
        [20 + (i % 50) for i in range(16)],
        [[100 + j * 10 + k for k in range(8)] for j in range(6)],
    ),
    "C_qitem_gt_W": (
        [20 + (i % 50) for i in range(64)],
        [[100 + j * 10 + k for k in range(8)] for j in range(4)],
    ),
}


def qwen(path, window):
    torch.manual_seed(42)
    _init_and_save(
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=VOCAB_SIZE,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=4096,
                bos_token_id=2,
                eos_token_id=1,
                pad_token_id=0,
            )
        ),
        path,
    )


def tokens(n, offset):
    return [offset + i % 50 for i in range(n)]


CASES = dict(REGIMES)
for p, lengths in [
    (0, [1, 7, 33]),
    (1, [1, 31, 32, 33]),
    (30, [1, 2, 3]),
    (31, [1, 2]),
    (32, [1, 2]),
    (33, [1, 2]),
    (64, [1, 8, 33, 71]),
    (127, [1, 31, 3]),
    (8, [129, 3, 65]),
]:
    CASES[f"p{p}_items" + "_".join(map(str, lengths))] = (
        tokens(p, 20),
        [tokens(length, 100 + i * 100) for i, length in enumerate(lengths)],
    )
CASES["p128_items32x48"] = (
    tokens(128, 20),
    [tokens(32, 100 + i * 50) for i in range(48)],
)
base = CASES["C_qitem_gt_W"]
CASES["isolation_original"] = base
CASES["isolation_mutated"] = (
    base[0],
    [tokens(len(it), 900) if i == 1 else it for i, it in enumerate(base[1])],
)


LOGPROB_ATOL = 0.02
SCORE_RTOL = 0.02
SCORE_ATOL = 1e-9


def _collect(path, enable_mis, extra):
    engine = sgl.Engine(
        model_path=path,
        dtype="float16",
        mem_fraction_static=0.08,
        attention_backend="flashinfer",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        enable_mis=enable_mis,
        log_level="error",
        **extra,
    )
    kwargs = dict(
        label_token_ids=LABELS, apply_softmax=False, return_token_logprobs=True
    )
    try:
        results = {}
        for name, (query, items) in CASES.items():
            if enable_mis:
                r = engine.score(query=query, items=items, **kwargs)
                results[name] = (r.token_logprobs, r.scores)
            else:
                rows = [
                    engine.score(query=query, items=[item], **kwargs) for item in items
                ]
                results[name] = (
                    [r.token_logprobs[0] for r in rows],
                    [r.scores[0] for r in rows],
                )
        if enable_mis:

            async def concurrent():
                return await asyncio.gather(
                    *[
                        engine.async_score(query=q, items=items, **kwargs)
                        for q, items in CASES.values()
                    ]
                )

            batch = engine.loop.run_until_complete(concurrent())
            for name, result in zip(CASES, batch):
                results["batch_" + name] = (result.token_logprobs, result.scores)
        return results
    finally:
        engine.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


def _mixed_cohere(path, window):
    torch.manual_seed(42)
    _init_and_save(
        Cohere2ForCausalLM(
            Cohere2Config(
                vocab_size=VOCAB_SIZE,
                hidden_size=256,
                intermediate_size=512,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                sliding_window=window,
                max_position_embeddings=2048,
                layer_types=["sliding_attention", "full_attention"] * 2,
                bos_token_id=2,
                eos_token_id=1,
                pad_token_id=0,
            )
        ),
        path,
    )


class TestMultiItemScoringSWA(unittest.TestCase):
    def _check_model(self, builder, **extra):
        with tempfile.TemporaryDirectory() as path:
            builder(path, W_SWA)
            _save_tokenizer(path)
            ref = _collect(path, False, extra)
            actual = _collect(path, True, extra)
        for name in CASES:
            for key in (name, "batch_" + name):
                with self.subTest(case=key):
                    torch.testing.assert_close(
                        torch.tensor(actual[key][0]),
                        torch.tensor(ref[name][0]),
                        atol=LOGPROB_ATOL,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        torch.tensor(actual[key][1]),
                        torch.tensor(ref[name][1]),
                        atol=SCORE_ATOL,
                        rtol=SCORE_RTOL,
                    )
        # Changing item 1 must not affect items 0, 2 or 3 at all.
        for field in (0, 1):
            torch.testing.assert_close(
                torch.tensor(actual["isolation_original"][field])[[0, 2, 3]],
                torch.tensor(actual["isolation_mutated"][field])[[0, 2, 3]],
                atol=0,
                rtol=0,
            )

    def test_gemma2_hybrid(self):
        self._check_model(_build_gemma2)

    def test_gemma2_nonhybrid(self):
        self._check_model(_build_gemma2, disable_hybrid_swa_memory=True)

    def test_gemma3_all_sliding(self):
        self._check_model(_build_gemma3)

    def test_cohere2_layer_only(self):
        self._check_model(_build_cohere2)

    def test_cohere2_mixed_layers(self):
        self._check_model(_mixed_cohere)

    def test_ordinary_swa_cache_and_decode(self):
        # Layer-only planning also affects ordinary prefill/decode. Compare an
        # uncached, unchunked run with chunking and repeated prefix-cache hits.
        for builder in (_build_gemma2, _build_gemma3, _build_cohere2):
            with (
                self.subTest(model=builder.__name__),
                tempfile.TemporaryDirectory() as path,
            ):
                builder(path, W_SWA)
                _save_tokenizer(path)
                runs = []
                for cached in (False, True):
                    engine = sgl.Engine(
                        model_path=path,
                        dtype="float16",
                        mem_fraction_static=0.08,
                        attention_backend="flashinfer",
                        disable_cuda_graph=True,
                        disable_radix_cache=not cached,
                        chunked_prefill_size=16 if cached else -1,
                        enable_mis=False,
                        log_level="error",
                    )
                    try:
                        rows = []
                        for _ in range(2):
                            rows.append(
                                engine.generate(
                                    input_ids=[tokens(64, 20), tokens(97, 100)],
                                    sampling_params={
                                        "temperature": 0,
                                        "max_new_tokens": 6,
                                    },
                                    return_logprob=True,
                                )
                            )
                        runs.append(rows)
                    finally:
                        engine.shutdown()
                        gc.collect()
                        torch.cuda.empty_cache()
                for repeated in runs:
                    for batch in repeated:
                        for reference, actual in zip(runs[0][0], batch):
                            self.assertEqual(
                                reference["output_ids"], actual["output_ids"]
                            )
                            ref_lp = [
                                row[0]
                                for row in reference["meta_info"][
                                    "output_token_logprobs"
                                ]
                            ]
                            act_lp = [
                                row[0]
                                for row in actual["meta_info"]["output_token_logprobs"]
                            ]
                            torch.testing.assert_close(
                                torch.tensor(act_lp),
                                torch.tensor(ref_lp),
                                atol=LOGPROB_ATOL,
                                rtol=0,
                            )

    def test_qwen2_full_attention(self):
        self._check_model(qwen)


if __name__ == "__main__":
    unittest.main()
