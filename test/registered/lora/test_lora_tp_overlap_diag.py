"""Diagnostic only (do not merge): on the 2x H100 lane, measure the TP-overlap
row against a float32 HF reference and across SGLang configurations.

Prints `[TPDIAG2] {json}` lines; asserts nothing. Row 1 of DEFAULT_PROMPTS with
the RuterNorway adapter, the CI-generated ids, and the five tokens of decode
position 11 that the formal test compares.
"""

import json
import subprocess
import sys
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.lora_utils import DEFAULT_PROMPTS
from sglang.test.runners import HFRunner, SRTRunner

register_cuda_ci(est_time=1500, stage="extra-a", runner_config="2-gpu-large")

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

# Teacher-forced HF forward in a child process (frees the GPU fully afterwards).
HF_TF = r"""
import json, sys, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
dtype = getattr(torch, sys.argv[1]); adapter = sys.argv[2] or None
base, prompt, gen, watch = json.loads(sys.argv[3]), sys.argv[4], json.loads(sys.argv[5]), json.loads(sys.argv[6])
tok = AutoTokenizer.from_pretrained(base)
ids = tok.encode(prompt)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).cuda()
if adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter, torch_dtype=dtype, is_trainable=False)
model.eval()
with torch.no_grad():
    res = model(torch.tensor([ids + gen], device="cuda"), output_hidden_states=True)
out = []
n = len(ids)
for pos in range(len(gen)):
    k = n - 1 + pos
    raw = res.logits[0, k].float(); lp = F.log_softmax(raw, dim=-1)
    v, i = torch.topk(lp, k=5)
    out.append({"target": gen[pos], "argmax": int(lp.argmax()), "top5": list(zip(i.tolist(), v.tolist())),
                "watch": {str(t): lp[t].item() for t in watch},
                "top_logit": raw.max().item(), "final_hidden_absmax": res.hidden_states[-1][0, k].float().abs().max().item()})
print("HFTF " + json.dumps({"dtype": sys.argv[1], "adapter": adapter, "prompt_tokens": n, "per_pos": out}))
"""


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
    # Runners return plain logprob floats in the order of the requested ids.
    return [
        {str(t): lp for t, lp in zip(WATCH, pos)}
        for pos in out.token_ids_output_logprobs[k]
    ]


def _tf_entry(ent):
    if ent and isinstance(ent[0], (int, float)):
        return {str(t): lp for t, lp in zip(WATCH, ent)}
    return {str(e[1]): e[0] for e in ent}


def _hf_teacher_forced(dtype, adapter):
    cmd = [
        sys.executable,
        "-c",
        HF_TF,
        dtype,
        adapter or "",
        json.dumps(BASE),
        DEFAULT_PROMPTS[1],
        json.dumps(CI_GEN),
        json.dumps(WATCH),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    for line in r.stdout.splitlines():
        if line.startswith("HFTF "):
            d = json.loads(line[5:])
            _emit(
                kind="hf_tf",
                dtype=dtype,
                adapter=adapter,
                prompt_tokens=d["prompt_tokens"],
                pos11=d["per_pos"][POS],
                watch_all=[p["watch"] for p in d["per_pos"]],
                top_logit_all=[p["top_logit"] for p in d["per_pos"]],
                final_hidden_absmax_all=[
                    p["final_hidden_absmax"] for p in d["per_pos"]
                ],
                argmax_matches=sum(p["argmax"] == p["target"] for p in d["per_pos"]),
            )
            return
    _emit(
        kind="hf_tf", dtype=dtype, adapter=adapter, error=(r.stderr or r.stdout)[-800:]
    )


class TestLoRATPOverlapDiag(unittest.TestCase):
    def test_measure(self):
        _emit(kind="env", **_env())
        prompts = [DEFAULT_PROMPTS[0], DEFAULT_PROMPTS[1]]

        # HF references through the test's own HFRunner (generate), fp16 and fp32.
        for dtype in (torch.float16, torch.float32):
            try:
                with HFRunner(BASE, torch_dtype=dtype, model_type="generation") as hf:
                    out = hf.forward(
                        prompts,
                        max_new_tokens=32,
                        lora_paths=ADAPTERS,
                        token_ids_logprob=WATCH,
                    )
                d = getattr(out, "diag_hf", None) or {}
                gen = d.get("gen_ids", [None, None])[1] if d else None
                _emit(
                    kind="hf_gen",
                    dtype=str(dtype),
                    gen_ids=gen,
                    gen_eq_ci=gen == CI_GEN,
                    top5_pos11=(
                        list(zip(d["top_ids"][1][POS], d["top_vals"][1][POS]))
                        if d
                        else None
                    ),
                    watch_pos11=_watch(out, 1)[POS],
                    watch_all=_watch(out, 1),
                    score_dtype=d.get("score_dtype"),
                )
            except Exception as e:
                _emit(kind="hf_gen", dtype=str(dtype), error=repr(e)[:600])

        # Teacher-forced HF: fp32 and fp16 with the adapter, fp32 and fp16 without.
        for dtype, adapter in (
            ("float32", ADAPTERS[1]),
            ("float16", ADAPTERS[1]),
            ("float32", None),
            ("float16", None),
        ):
            try:
                _hf_teacher_forced(dtype, adapter)
            except Exception as e:
                _emit(kind="hf_tf", dtype=dtype, adapter=adapter, error=repr(e)[:600])

        configs = [
            dict(name="tp2_fp16_overlap_on", tp=2, overlap=True),
            dict(name="tp2_fp16_overlap_on_again", tp=2, overlap=True),
            dict(name="tp2_fp16_overlap_off", tp=2, overlap=False),
            dict(name="tp1_fp16_overlap_on", tp=1, overlap=True),
            dict(name="tp2_fp16_nograph", tp=2, overlap=True, disable_cuda_graph=True),
            dict(
                name="tp2_fp16_triton_lora", tp=2, overlap=True, lora_backend="triton"
            ),
            dict(name="tp2_fp16_no_lora_engine", tp=2, no_lora=True),
        ]
        for c in configs:
            try:
                lora_kw = (
                    {}
                    if c.get("no_lora")
                    else dict(
                        lora_paths=ADAPTERS,
                        enable_lora_overlap_loading=c["overlap"],
                        max_loras_per_batch=2,
                        max_loaded_loras=4,
                        lora_backend=c.get("lora_backend", "csgmv"),
                    )
                )
                with SRTRunner(
                    BASE,
                    torch_dtype=torch.float16,
                    model_type="generation",
                    tp_size=c["tp"],
                    disable_cuda_graph=c.get("disable_cuda_graph", False),
                    mem_fraction_static=0.88,
                    attention_backend="fa3",
                    **lora_kw,
                ) as srt:
                    rec = dict(kind="sgl", **c)
                    if not c.get("no_lora"):
                        out = srt.forward(
                            prompts,
                            max_new_tokens=32,
                            lora_paths=ADAPTERS,
                            token_ids_logprob=WATCH,
                        )
                        gen = [t[1] for t in out.output_token_logprobs_lst[1]]
                        rec.update(
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
                        srt.engine.flush_cache()
                    # Prefill path over the CI ids, with and without the adapter.
                    p_ids = srt.tokenizer.encode(prompts[1])
                    for tag, lp in (
                        ("tf_adapter", ADAPTERS[1]),
                        ("tf_no_adapter", None),
                    ):
                        if c.get("no_lora") and lp is not None:
                            continue
                        try:
                            resp = srt.engine.generate(
                                input_ids=p_ids + CI_GEN,
                                lora_path=lp,
                                sampling_params={"max_new_tokens": 1, "temperature": 0},
                                return_logprob=True,
                                logprob_start_len=len(p_ids) - 1,
                                top_logprobs_num=5,
                                token_ids_logprob=WATCH,
                            )
                            ents = resp["meta_info"]["input_token_ids_logprobs"]
                            rec[tag + "_pos11"] = _tf_entry(ents[POS + 1])
                            rec[tag + "_all"] = [
                                _tf_entry(ents[j + 1]) for j in range(len(CI_GEN))
                            ]
                            srt.engine.flush_cache()
                        except Exception as e:
                            rec[tag + "_error"] = repr(e)[:300]
                _emit(**rec)
            except Exception as e:
                _emit(kind="sgl", name=c["name"], error=repr(e)[:600])


if __name__ == "__main__":
    unittest.main(verbosity=2)
