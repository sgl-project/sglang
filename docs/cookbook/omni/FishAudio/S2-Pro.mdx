---
title: FishAudio S2 Pro
metatags:
    description: "Deploy FishAudio S2 Pro with SGLang - state-of-the-art text-to-speech with dual-autoregressive architecture, voice cloning, prosody control, and 80+ language support."
tag: NEW
---

## 1. Model Introduction

[FishAudio S2 Pro](https://huggingface.co/fishaudio/s2-pro) is a state-of-the-art text-to-speech model developed by [FishAudio](https://fish.audio), featuring fine-grained prosody and emotion control. Built on a Dual-Autoregressive (Dual-AR) transformer architecture with RVQ-based audio codec, S2 Pro achieves state-of-the-art quality across multiple TTS benchmarks.

S2 Pro tops the Audio Turing Test (0.515 posterior mean) and EmergentTTS-Eval (81.88% win rate against gpt-4o-mini-tts) while achieving the lowest WER on Seed-TTS Eval among all evaluated models including closed-source systems. Trained on over 10 million hours of audio across approximately 100 languages and aligned with GRPO-based reinforcement learning, it supports voice cloning and fine-grained inline control of prosody and emotion through natural-language tags.

**Key Features:**

- **Dual-AR Architecture**: 5B parameter model (4B Slow AR + 400M Fast AR) with RVQ-based audio codec at 10 codebooks (~21 Hz frame rate)
- **Voice Cloning**: High-quality voice cloning from a short reference audio clip
- **Prosody & Emotion Control**: Fine-grained inline control of prosody and emotion through natural-language tags
- **Multilingual**: 80+ language support (Tier 1: Japanese, English, Chinese; Tier 2: Korean, Spanish, Portuguese, Arabic, Russian, French, German)
- **SGLang Integration**: Inherits LLM-native serving optimizations (paged KV cache, radix prefix caching)

**License:** [FISH AUDIO RESEARCH LICENSE AGREEMENT](https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md)

This work is a collaboration between the SGLang Omni Team and [FishAudio Team](https://fish.audio). For more details on S2 Pro's model design and training, see FishAudio's [S2 release blog post](https://fish.audio/blog/fish-audio-open-sources-s2/).

## 2. Installation

S2 Pro uses `sglang-omni`, an ecosystem project for SGLang. Start with the Docker image, then install the `sglang-omni` package inside the container.

### 2.1 Docker

```bash Command
docker pull frankleeeee/sglang-omni:dev

docker run -it --shm-size 32g --gpus all frankleeeee/sglang-omni:dev /bin/zsh
```

### 2.2 Install sglang-omni (inside Docker)

```bash Command
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v ".[s2pro]"
huggingface-cli download fishaudio/s2-pro
```

## 3. Model Deployment

S2 Pro can be served via an OpenAI-compatible HTTP server or explored interactively through a Gradio playground.

### 3.1 Server

```bash Command
python -m sglang_omni.cli.cli serve \
    --model-path fishaudio/s2-pro \
    --config examples/configs/s2pro_tts.yaml \
    --port 8000
```

### 3.2 Interactive Playground

We provide a Gradio-based interactive playground. We highly recommend using the playground since audio data is hard to interact with by CLI.

```bash Command
./playground/tts/start.sh
```

## 4. Model Invocation

### 4.1 Text-to-Speech

Generate speech from text using the OpenAI-compatible `/v1/audio/speech` endpoint.

<Note>
Without a reference audio clip, the generated voice will use a default voice. Provide a reference audio for voice cloning.
</Note>

```bash Command
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, how are you?"}' \
    --output output.wav
```

### 4.2 Voice Cloning

Provide a reference audio file and its transcript for high-quality voice cloning:

```bash Command
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "references": [{"audio_path": "ref.wav", "text": "Transcript of ref audio."}]
    }' \
    --output output.wav
```

## 5. Architecture

S2 Pro uses a 3-stage pipeline:

```text Example
Text input ──► Preprocessing ──► SGLang AR Engine ──► DAC Vocoder ──► Audio output
                 (CPU)              (GPU)               (GPU)
```

**Stage 1 — Preprocessing:** Tokenizes the input text into a Qwen3-style chat prompt. For voice cloning, encodes the reference audio into VQ codes via the DAC codec and prepends them to the prompt as a system message.

**Stage 2 — Dual-AR Generation:** The Slow AR runs inside SGLang along the time axis. At each decode step, it predicts a semantic token, then the Fast AR (4-layer transformer) generates the remaining 9 residual codebook tokens conditioned on the hidden state. VQ embeddings are injected into the input embedding at masked positions, allowing the model to attend over both text and audio context through SGLang's KV cache.

**Stage 3 — Vocoder:** The accumulated codebook indices are decoded into a waveform by a DAC codec, producing the final audio output.

## 6. Performance

Evaluated on the full seed-tts-eval EN testset (1,088 samples) on a single H200 GPU.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Metric</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>BS=1</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>BS=2</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>BS=4</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>BS=8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Tok/s (mean)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>63.3</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>45.8</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>31.9</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>19.6</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>RTF (mean)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>0.340</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>0.473</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>0.676</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>1.097</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Latency (mean)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>1.33s</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>1.80s</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>2.69s</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>4.36s</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>TTFT (mean)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>19.6 ms</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>22.0 ms</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>31.6 ms</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>50.7 ms</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>TTFB (mean)</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>172.8 ms</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>249.9 ms</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>319.1 ms</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>509.6 ms</td>
    </tr>
  </tbody>
</table>

## 7. SGLang Omni Optimizations

By integrating S2 Pro's Dual-AR backbone into SGLang's paged-attention engine, we inherit LLM-native optimizations:

- **Paged KV cache** — SGLang manages KV cache for the Slow AR path, enabling efficient memory usage and high concurrency.
- **Radix prefix caching** — Shared system prompt and reference audio prefixes are cached across requests, keeping TTFT consistently low (~18ms).
- **torch.compile on Fast AR** — The 9-step codebook loop is compiled with torch.compile, achieving 5x speedup over eager mode.
- **FlashAttention 3** — Forced FA3 backend to match training-time attention numerics, avoiding early-EOS divergence from flashinfer.

## 8. Future Optimizations

To further improve throughput and latency in the future:

- **CUDA Graphs while torch.compile enabled.** The current implementation uses torch.compile on the Fast AR codebook loop (achieving 5x over eager), but does not capture CUDA graphs for the Slow AR path. Enabling CUDA graphs requires resolving numerical divergence from deterministic-mode constraints and adapting SGLang's graph capture to S2 Pro's interleaved VQ embedding injection, involving significant engineering that we leave for a future release.

- **Batched Fast AR head processing.** Currently, the Fast AR codebook decoding loop runs sequentially per request. Batching these steps across concurrent requests would improve GPU utilization at higher batch sizes, potentially improving throughput.

## 9. Engineering Appendix

<Accordion title="Engineering Appendix">

### BF16 RoPE Precision Mismatch

SGLang's default RoPE implementation precomputes `cos_sin_cache` in float32, but S2 Pro's model was trained entirely in bfloat16 including the RoPE frequencies. The precision difference caused logit divergence producing garbled audio with abnormally long sequences of tokens.

It's worth attention for any future engineering for fish audio inference infrastructure, since it's uncommon and hard to debug when accuracy of inference engine is higher than the precision of the model. Below is a simple fix once problem identified.

```python Example
def _truncate_rope_to_bf16(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "cos_sin_cache"):
            module.cos_sin_cache.data = module.cos_sin_cache.data.to(torch.bfloat16).to(
                torch.float32
            )
```

### Attention Backend Divergence Causing Early Stopping

SGLang defaults to flashinfer for attention, but S2 Pro was trained with FlashAttention. When future engineering meet early EOS token issue, this could suggest the fix.

</Accordion>
