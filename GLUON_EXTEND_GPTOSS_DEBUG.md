# Gluon Extend Attention GPT-OSS Debug Log

This note records the runtime findings from updating the vendored Gluon
extend-attention backend to the current `gluon-kernels` implementation.
It is intended as a scratch/debug reference, not as PR-facing documentation.

## Kernel Sync

- Synced the vendored gfx950 extend-attention runtime from
  `/home/tussingh/gluon-kernels-1/kernels/cdna4/fa/extend/`.
- The active runtime is now the self-contained
  `extend_attention_gfx950.py` plus `_common.py`.
- Updated `gluon_extend_attention.py` so the wrapper gate matches current
  kernel support more closely:
  - non-causal symmetric-head extend can route to Gluon;
  - FP8 FNUZ KV is rejected before dispatch because gfx950 expects OCP
    `torch.float8_e4m3fn`.

## Kernel Validation

The updated backend passed the registered Gluon attention test file:

```bash
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/tussingh/sglang-extend/python \
/home/tussingh/venv/bin/python -m pytest -q \
  test/registered/attention/test_gluon_extend_attention.py
```

Result:

```text
17 passed
```

Additional focused GPT-OSS-style parity was checked with D=64, Hq=64,
Hkv=8, BF16 KV, sliding-window attention, and attention sinks across
small, ragged, and long-prefix shapes. These matched Triton within the
registered tolerances.

## Failed / Incoherent Runtime Paths

The original TP=1 GPT-OSS launches booted and served, but generated
incoherent text. Logs showed that the failure was not isolated to Gluon
extend attention:

- KV cache was already BF16.
- Gluon extend attention was enabled and did not report fallback lines.
- The runtime selected AITER MXFP4 MoE and AITER-related kernels.
- AITER RoPE was also selected unless explicitly disabled.

Representative log signals:

```text
Detected ROCm and MXFP4 quantization format for GPT-OSS model, enabling aiter MXFP4 MOE kernel.
Aiter backend is selected for fused RoPE. This has lower precision.
Using KV cache dtype: torch.bfloat16
Gluon extend attention enabled on gfx950 (--enable-gluon-extend-attention).
```

The TP=2 legacy-container coherence path that used `SGLANG_USE_AITER=0`
failed in that container with the known `triton_kernels` API drift:

```text
TypeError: upcast_from_mxfp() got an unexpected keyword argument 'target_dtype'
```

Parser flags were also version-sensitive. On the newer SGLang tree,
`--tool-call-parser harmony` is no longer valid; `gpt-oss` is the valid
parser name. Parser changes affected response extraction, but did not
explain the incoherent raw model output on the AITER MXFP4 path.

## Coherent Runtime Path

The coherent backend uses Gluon only for extend attention and avoids the
suspect external AITER MXFP4/RoPE path:

```bash
SGLANG_USE_AITER=0 \
USE_ROCM_AITER_ROPE_BACKEND=0 \
PYTHONPATH=/home/tussingh/sglang-extend/python \
HIP_VISIBLE_DEVICES=0,1 \
python3 -m sglang.launch_server \
  --model-path /data/dev/morhuang/models/gpt-oss-120b \
  --tp 2 \
  --host 0.0.0.0 \
  --port 34074 \
  --attention-backend triton \
  --enable-gluon-extend-attention \
  --moe-runner-backend triton \
  --kv-cache-dtype bf16 \
  --trust-remote-code \
  --mem-fraction-static 0.72
```

Important log signals from the coherent run:

```text
Using the native apex kernel for RoPE.
Using KV cache dtype: torch.bfloat16
Gluon extend attention enabled on gfx950 (--enable-gluon-extend-attention).
Using default MoE kernel config. Performance might be sub-optimal!
The server is fired up and ready to roll!
```

The "default MoE kernel config" warning is expected for this fallback path:
it uses SGLang's in-tree Triton MoE runner after MXFP4 dequantization rather
than AITER MXFP4 MoE or the external `triton_kernels` package.

## Coherence Smoke

The old coherence prompt returned coherent raw harmony content:

```text
<|channel|>analysis<|message|>User asks: "What is the capital of France? Reply with a single sentence." Provide answer in one sentence.<|end|><|start|>assistant<|channel|>final<|message|>Paris is the capital of France.
```

## GSM8K Smoke

One GSM8K example was run through the normal SGLang eval harness against
the coherent backend:

```bash
PYTHONPATH=/home/tussingh/sglang-extend/python \
OPENAI_API_KEY=EMPTY \
python3 -m sglang.test.run_eval \
  --host 127.0.0.1 \
  --port 34074 \
  --model /data/dev/morhuang/models/gpt-oss-120b \
  --eval-name gsm8k \
  --num-examples 1 \
  --num-threads 1 \
  --max-tokens 512 \
  --reasoning-effort low
```

Result:

```text
Score: 1.000
Output throughput: 31.042 token/s
```

This verifies that GSM8K can run through the Gluon extend-attention backend
on GPT-OSS with coherent output. A full GSM8K run should use the coherent
TP=2 launch above and enough reserved machine time.
