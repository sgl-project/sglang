---
title: EmbeddingGemma 2
description: Serve Google's multimodal EmbeddingGemma 2 model (text, image, video, audio) with SGLang.
tag: NEW
---

## Overview

[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) is Google's multimodal embedding model. It maps text (including code), images, video, audio, and interleaved combinations of them into one shared, L2-normalized 768-dimensional vector space.

| Property | Value |
| :-- | :-- |
| Parameters | 740M total: 270M text backbone, 170M vision encoder, 300M audio encoder |
| Architecture | 24-layer bidirectional Gemma 4 encoder, 5:1 local:global attention, local window radius 512 |
| Context | 8,192 tokens shared by all modalities in a request |
| Output | 768 dimensions; Matryoshka truncation to 512, 256, or 128 |
| Pooling | Mean pooling over all tokens, then L2 normalization |

SGLang detects the `EmbeddingGemma2Model` architecture and configures it automatically:

- enables embedding mode, so `--is-embedding` is not needed;
- selects the Triton attention backend, which implements the bidirectional sliding window;
- disables RadixAttention prefix caching and chunked prefill, because bidirectional attention makes every token depend on the whole input;
- caps the context length at 8,192 tokens.

## Prerequisites

- An NVIDIA CUDA GPU. The model weighs about 1.5 GB in BF16, so any data-center GPU works.
- A Transformers build that includes EmbeddingGemma 2 (`model_type: embedding_gemma2`). Older releases fail at startup with `Transformers does not recognize this architecture`.
- An SGLang build that includes EmbeddingGemma 2 support:

```bash
pip install 'git+https://github.com/sgl-project/sglang.git#subdirectory=python'
```

Run the model in BF16 (the checkpoint default) or FP32. Do not use FP16: the model's activations exceed its range and produce NaN or degraded embeddings.

## Start the server

```bash
sglang serve \
  --model-path google/embeddinggemma-2 \
  --host 0.0.0.0
```

### Load only the encoders you need

The vision and audio encoders load only when their modality is allowed. Set a modality's per-request limit to `0` to skip loading its encoder. Images and video share the vision encoder.

| Serving | `--limit-mm-data-per-request` | Parameters loaded |
| :-- | :-- | :-- |
| Text only | `'{"image": 0, "video": 0, "audio": 0}'` | 270M |
| Text and images/video | `'{"audio": 0}'` | 440M |
| Text and audio | `'{"image": 0, "video": 0}'` | 570M |
| All modalities | (default) | 740M |

```bash
sglang serve \
  --model-path google/embeddinggemma-2 \
  --limit-mm-data-per-request '{"image": 0, "video": 0, "audio": 0}' \
  --host 0.0.0.0
```

A request for a disabled modality fails with HTTP 400, for example `Image count 1 exceeds limit 0 per request.`

### Enable Matryoshka dimensions

To let clients request shorter vectors through the `dimensions` field, declare the supported sizes when starting the server:

```bash
sglang serve \
  --model-path google/embeddinggemma-2 \
  --json-model-override-args '{"matryoshka_dimensions": [128, 256, 512, 768]}' \
  --host 0.0.0.0
```

SGLang truncates the pooled vector and then re-normalizes it, as the model card requires. Other sizes are rejected with HTTP 400. Queries and documents must use the same dimension.

## Task prompts

EmbeddingGemma 2 is trained with short task prefixes on text inputs. Prepend them yourself; images, video, and audio take no prefix. For documents without a title, use `title: none`.

| Use case | Query | Document |
| :-- | :-- | :-- |
| Retrieval | `task: search result \| query: {content}` | `title: {title} \| text: {content}` |
| Question answering | `task: question answering \| query: {content}` | `title: {title} \| text: {content}` |
| Fact checking | `task: fact checking \| query: {content}` | `title: {title} \| text: {content}` |
| Code retrieval | `task: code retrieval \| query: {content}` | `title: {title} \| text: {content}` |
| Classification | `task: classification \| query: {content}` | |
| Clustering | `task: clustering \| query: {content}` | |
| Semantic similarity | `task: sentence similarity \| query: {content}` | |

## Create embeddings

### Native `/encode` API

`/encode` supports every modality, including several media items in one embedding.

Text, one embedding per string:

```bash
curl http://127.0.0.1:30000/encode \
  -H 'Content-Type: application/json' \
  -d '{
    "text": [
      "task: search result | query: What causes the aurora borealis?",
      "title: none | text: The aurora is caused by solar wind particles colliding with the upper atmosphere."
    ]
  }'
```

An image, video, or audio clip on its own. SGLang inserts the placeholder for you:

```bash
curl http://127.0.0.1:30000/encode \
  -H 'Content-Type: application/json' \
  -d '{"image_data": "https://example.com/cat.jpg"}'
```

Use `video_data` or `audio_data` the same way. Audio should be mono at 16 kHz.

Interleaved input, one embedding for the whole sequence. Mark each item's position with `<|image|>`, `<|video|>`, or `<|audio|>`, and pass the items in the same order:

```bash
curl http://127.0.0.1:30000/encode \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Waterproof running shoes. <|image|> Breathable mesh upper. <|image|> Narration: <|audio|>",
    "image_data": ["https://example.com/shoe.jpg", "https://example.com/mesh.jpg"],
    "audio_data": "https://example.com/narration.wav"
  }'
```

#### How requests map to embeddings

| Request shape | Result |
| :-- | :-- |
| `text` is a string, media is a list | 1 embedding containing all the media |
| `text` is a list of N strings, media is a list of N entries | N embeddings, paired by position; an entry may be `null` or a list of several items |
| `text` is a list, media is a single item (not a list) | the item is attached to every request, so every string needs its placeholder |
| No `text`, media is a flat list `[a, b]` | 1 embedding of `a` and `b` together |
| No `text`, media is a list of lists `[[a], [b]]` | 2 embeddings |

Requests are rejected with HTTP 400 instead of being silently changed when:

- the number of `<|image|>`, `<|video|>`, or `<|audio|>` placeholders differs from the number of items of that modality (text without any placeholder plus an image is also a mismatch);
- a batched media list has a different length than the `text` list.

### OpenAI-compatible `/v1/embeddings`

```bash
curl http://127.0.0.1:30000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/embeddinggemma-2",
    "input": [
      {"text": "a product photo: <|image|>", "image": "https://example.com/shoe.jpg"},
      {"image": "https://example.com/mesh.jpg"},
      {"text": "task: search result | query: waterproof trail shoes"}
    ]
  }'
```

Each input item produces one embedding. In this API, an item carries at most one `image` or `video` and no audio. For several media items in one embedding, or for audio, use `/encode`.

## Input limits

All modalities share the 8,192-token context. Requests that exceed it fail with HTTP 400 (`The input (N tokens) is longer than the model's context length (8192 tokens).`).

| Modality | Cost | Max, single modality |
| :-- | :-- | :-- |
| Text | 1 token per subword | 8,192 tokens |
| Image | up to 280 tokens per image | about 29 images |
| Video | up to 140 tokens per sampled frame | about 58 frames |
| Audio | 25 tokens per second | about 327 seconds |

Video is sampled at 1 frame per second and capped at 32 frames by default, spread uniformly over longer clips, so a video of any length fits by default. A 2-minute 720p clip, for example, uses 3,906 tokens. Only the sampled frames are decoded.

To change the sampling rate or frame cap for every request, use `--mm-process-config`:

```bash
sglang serve \
  --model-path google/embeddinggemma-2 \
  --mm-process-config '{"video": {"fps": 2, "max_frames": 48}}' \
  --host 0.0.0.0
```

The number of frames that fit depends on resolution. At 720p, 64 frames fit; 70 frames exceed the context.
