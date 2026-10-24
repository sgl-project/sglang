"""Diagnostic only (do not merge): on the 2x H100 lane, measure the TP-overlap
row against a float32 HF reference and across SGLang configurations.

Prints `[TPDIAG2] {json}` lines; asserts nothing. Row 1 of DEFAULT_PROMPTS with
the RuterNorway adapter, the CI-generated ids, and the five tokens of decode
position 11 that the formal test compares.
"""

import json
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import DEFAULT_PROMPTS
from sglang.test.runners import HFRunner, SRTRunner

register_cuda_ci(est_time=1200, stage="extra-a", runner_config="2-gpu-large")

BASE = "meta-llama/Llama-2-7b-hf"
ADAPTERS = [
    "winddude/wizardLM-LlaMA-LoRA-7B",
    "RuterNorway/Llama-2-7b-chat-norwegian-LoRa",
]
WATCH = [384, 4684, 12535, 7077, 4937]
CI_GEN = [
    29896,
    29889,
    365,
    5288,
    294,
    526,
    2919,
    29892,
    1472,
    29899,
    484,
    384,
    287,
    15006,
    411,
    263,
    281,
    1507,
    368,
    24296,
    29889,
    13,
    268,
    29906,
    29889,
    2688,
    505,
    1023,
    304,
    267,
    373,
    1269,
]
POS = 11


def _emit(**kw):
    print("[TPDIAG2] " + json.dumps(kw), flush=True)


def _env():
    import importlib.metadata as md
    import os
    import socket

    def ver(n):
        try:
            return md.version(n)
        except Exception:
            return "?"

    return dict(
        host=socket.gethostname(),
        runner=os.environ.get("RUNNER_NAME"),
        sha=os.environ.get("GITHUB_SHA"),
        gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        transformers=ver("transformers"),
        peft=ver("peft"),
        sglang_kernel=ver("sglang-kernel"),
        flashinfer=ver("flashinfer-python"),
    )


def _watch(out, k):
    rows = out.token_ids_output_logprobs[k]
    return [{str(t): lp for lp, t, _ in pos} for pos in rows]


class TestLoRATPOverlapDiag(unittest.TestCase):
    def test_measure(self):
        _emit(kind="env", **_env())
        prompts = [DEFAULT_PROMPTS[0], DEFAULT_PROMPTS[1]]

        # HF references through the test's own HFRunner: fp16 (the test's reference) and fp32.
        for dtype in (torch.float16, torch.float32):
            try:
                with HFRunner(BASE, torch_dtype=dtype, model_type="generation") as hf:
                    out = hf.forward(
                        prompts,
                        max_new_tokens=32,
                        lora_paths=ADAPTERS,
                        token_ids_logprob=WATCH,
                    )
                d = getattr(out, "diag_hf", {}) or {}
                _emit(
                    kind="hf",
                    dtype=str(dtype),
                    gen_ids=d.get("gen_ids", [None, None])[1],
                    gen_eq_ci=d.get("gen_ids", [None, None])[1] == CI_GEN,
                    top5_pos11=[list(zip(d["top_ids"][1][POS], d["top_vals"][1][POS]))]
                    if d
                    else None,
                    watch_pos11=_watch(out, 1)[POS],
                    watch_all=_watch(out, 1),
                    score_dtype=d.get("score_dtype"),
                )
            except Exception as e:  # keep measuring
                _emit(kind="hf", dtype=str(dtype), error=repr(e))

        configs = [
            dict(name="tp2_fp16_overlap_on", tp=2, dtype=torch.float16, overlap=True),
            dict(
                name="tp2_fp16_overlap_on_again",
                tp=2,
                dtype=torch.float16,
                overlap=True,
            ),
            dict(name="tp2_fp16_overlap_off", tp=2, dtype=torch.float16, overlap=False),
            dict(name="tp1_fp16_overlap_on", tp=1, dtype=torch.float16, overlap=True),
            dict(
                name="tp2_fp16_nograph",
                tp=2,
                dtype=torch.float16,
                overlap=True,
                disable_cuda_graph=True,
            ),
            dict(
                name="tp2_fp16_triton_lora",
                tp=2,
                dtype=torch.float16,
                overlap=True,
                lora_backend="triton",
            ),
            dict(
                name="tp2_fp32_overlap_on",
                tp=2,
                dtype=torch.float32,
                overlap=True,
                mem=0.80,
            ),
            dict(
                name="tp1_fp32_overlap_on",
                tp=1,
                dtype=torch.float32,
                overlap=True,
                mem=0.80,
            ),
        ]
        for c in configs:
            try:
                with SRTRunner(
                    BASE,
                    torch_dtype=c["dtype"],
                    model_type="generation",
                    tp_size=c["tp"],
                    lora_paths=ADAPTERS,
                    enable_lora_overlap_loading=c["overlap"],
                    max_loras_per_batch=2,
                    max_loaded_loras=4,
                    lora_backend=c.get("lora_backend", "csgmv"),
                    disable_cuda_graph=c.get("disable_cuda_graph", False),
                    mem_fraction_static=c.get("mem", 0.88),
                    attention_backend="fa3",
                ) as srt:
                    out = srt.forward(
                        prompts,
                        max_new_tokens=32,
                        lora_paths=ADAPTERS,
                        token_ids_logprob=WATCH,
                    )
                    gen = [t[1] for t in out.output_token_logprobs_lst[1]]
                    rec = dict(
                        kind="sgl",
                        **{k: (str(v) if k == "dtype" else v) for k, v in c.items()},
                        gen_ids=gen,
                        gen_eq_ci=gen == CI_GEN,
                        top5_pos11=list(
                            zip(
                                out.top_output_logprob_idx[1][POS],
                                out.top_output_logprobs[1][POS],
                            )
                        ),
                        watch_pos11=_watch(out, 1)[POS],
                        watch_all=_watch(out, 1),
                    )
                    # Same engine, prefill path over the CI ids (cache flushed first).
                    try:
                        srt.engine.flush_cache()
                        p_ids = (
                            srt.tokenizer.encode(prompts[1])
                            if hasattr(srt, "tokenizer")
                            else None
                        )
                        if p_ids is None:
                            from transformers import AutoTokenizer

                            p_ids = AutoTokenizer.from_pretrained(BASE).encode(
                                prompts[1]
                            )
                        resp = srt.engine.generate(
                            input_ids=p_ids + CI_GEN,
                            lora_path=ADAPTERS[1],
                            sampling_params={"max_new_tokens": 1, "temperature": 0},
                            return_logprob=True,
                            logprob_start_len=len(p_ids) - 1,
                            top_logprobs_num=5,
                            token_ids_logprob=WATCH,
                        )
                        ent = resp["meta_info"]["input_token_ids_logprobs"][POS + 1]
                        rec["teacher_forced_pos11"] = {str(t): lp for lp, t, _ in ent}
                    except Exception as e:
                        rec["teacher_forced_error"] = repr(e)
                _emit(**rec)
            except Exception as e:
                _emit(kind="sgl", name=c["name"], error=repr(e)[:400])


if __name__ == "__main__":
    unittest.main(verbosity=2)
