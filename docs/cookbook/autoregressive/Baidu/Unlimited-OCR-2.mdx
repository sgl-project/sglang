---
title: Unlimited-OCR-2
description: "Deploy Baidu Unlimited-OCR-2 with SGLang for multi-page and book-length document OCR: DASH sparse attention, page-window splitting with KV reuse, one request per document."
tag: NEW
---

## Deployment

<a id="install" />

<Accordion title="Install SGLang">

Unlimited-OCR-2 support is not in a tagged SGLang release yet. Until it is, install SGLang from a source build that contains it. For all methods and hardware platforms, see the [official SGLang installation guide](/docs/get-started/install).

<Tabs>

<Tab title="Python (pip / uv)">

```bash Command
pip install -U uv
uv venv --python 3.12 && source .venv/bin/activate

git clone https://github.com/sgl-project/sglang.git
cd sglang
uv pip install --prerelease=allow -e python
```

Then run the **Python** output of the command panel below in that environment.

</Tab>

<Tab title="Docker">

```bash Command
docker pull lmsysorg/sglang:dev
```

For how to launch the image, see [Install → Method 3: Using Docker](/docs/get-started/install#method-3-using-docker). Substitute the inner `sglang serve ...` with what the command generator below produces.

</Tab>

</Tabs>

</Accordion>

Pick your hardware to generate the launch command. SGLang selects the `unlimited_ocr_v2` attention backend, the page size the checkpoint requires and a disabled radix cache on its own, so the recipe only carries the serving options: the `unlimited-ocr` chat template, explicit page-window processing with `--uocr-doc-split`, a per-page output budget, and decode CUDA graphs, which are off by default for this model. The built-in page guard runs automatically; `--enable-custom-logit-processor` is only needed for an external custom processor.

import { Deployment } from "/src/snippets/_deployment.jsx";
import { config } from "/src/snippets/configs/baidu/unlimited-ocr-2.jsx";

<Deployment config={config} />

## Playground

Use the Playground to adjust tensor parallelism on top of the selected deployment cell.

import { Playground } from "/src/snippets/_playground.jsx";

<Playground config={config} />

## 1. Model Introduction

[Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) is Baidu's document parsing model for long and very long documents: all pages of a document go into one request, and the answer carries one `<PAGE>` section per input page, in page order. It replaces the sliding window of [Unlimited-OCR](/cookbook/autoregressive/Baidu/Unlimited-OCR) with DASH sparse attention, where a learned router picks the KV blocks each query reads and every page keeps its own 2D position frame, so decode cost stays nearly flat as the document grows.

The checkpoint has about 3.3B parameters in BF16 (a DeepSeek-style MoE language backbone with SAM and CLIP vision encoders) and is released under the MIT license.

**Recommended generation:** `temperature=0`, with `skip_special_tokens=false` so the `<PAGE>` markers stay in the output.

**Resources:** [Hugging Face](https://huggingface.co/baidu/Unlimited-OCR-2) · [GitHub](https://github.com/baidu/Unlimited-OCR) · [MarathonDocBench](https://huggingface.co/datasets/baidu/MarathonDocBench)

## 2. Configuration Tips

- **Prompt layout**: list the page images first, then one text part holding `<PAGE><image>` once per page followed by the instruction, for example `"<PAGE><image>" * num_pages + "Multi page parsing."`.
- **Whole-document processing is the default.** Omit `--uocr-doc-split` to process the document together, within the configured context length.
- **Page-window processing is opt-in.** The deployment recipe enables `--uocr-doc-split` for long documents. `--uocr-doc-page-window N` sets the maximum pages each window transcribes in every image mode (default 1). Later windows reuse the preceding window's KV tail. This changes the available context and is approximate; it is not a lossless replacement for whole-document inference. The client still sends one request and receives one response or stream, with `metadata.doc_split_*` describing the processing.
- **Per-page output budget:** `--uocr-max-page-tokens N` limits how many tokens one output page may spend before the guard advances to the next page or ends the document. The default 0 leaves the page length uncapped. A nonzero cap also sizes the default context of a split server. It changes stopping behavior and needs no custom-processor flag. The page budget excludes page markers and termination tokens; overlap scheduling may leave one already-computed token beyond the budget. Set the request's total `max_tokens` to cover the document.
- **Per-request options:** `doc_split_options.max_batch_pages` may lower the server's window size; `max_page_tokens` may lower a nonzero server page cap. `lookahead_pages` defaults to 1 and `overlap_pages` to 0. Set `lookahead_pages: 0` to omit the following page from the window's input. Lookahead and overlap change the input context and must fit the server's context budget. Window options require `--uocr-doc-split`; a per-page output cap can also be used without splitting.
- **Data parallelism:** page-window processing requires a single DP rank. Omit `--uocr-doc-split` with `--dp-size` or `--attn-dp-size` above 1.
- **Not supported**: speculative decoding, pipeline parallelism and PD disaggregation are rejected at launch.

## 3. Advanced Usage

### 3.1 Multi-page document

The example renders a three-page document and prints each page's section. It uses the page-window deployment recipe above.

<Accordion title="Multi-page Example (Python)">

```python Example
import base64
import io

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")


def page_image(number):
    img = Image.new("RGB", (768, 1024), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=64)
    draw.text((80, 120), "Quarterly Report", fill="black", font=font)
    draw.text((80, 320), f"Account {number}", fill="black", font=font)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


pages = [page_image(number) for number in ("31415", "27182", "16180")]
prompt = "<PAGE><image>" * len(pages) + "Multi page parsing."

response = client.chat.completions.create(
    model="baidu/Unlimited-OCR-2",
    messages=[{"role": "user", "content": pages + [{"type": "text", "text": prompt}]}],
    max_tokens=8000 * len(pages),
    extra_body={
        "images_config": {"image_mode": "base"},
        "skip_special_tokens": False,
        "doc_split_options": {"max_page_tokens": 4000},
    },
)

if not response.metadata["doc_split_complete"]:
    raise RuntimeError("OCR document is incomplete; inspect response.metadata")

for i, body in enumerate(response.choices[0].message.content.split("<PAGE>")[1:]):
    print(f"page {i + 1}: {body.strip()!r}")
print("windows:", response.metadata["doc_split_batches"])
```

</Accordion>

Check `metadata.doc_split_complete` before treating a non-streaming split response as a complete document. If a window exceeds the context limit while other windows succeed, the response contains those successful windows, reports `finish_reason: "length"`, and lists the rejected window in `metadata.doc_split_incomplete_batches`.

<Accordion title="Example Output">

```text Output
page 1: '<BLOCK>title<LEVEL2>[99, 126, 723, 196]<FORK>Quarterly Report\n<BLOCK>title<LEVEL3>[99, 321, 687, 384]<FORK>Account 31415'
page 2: '<BLOCK>title<LEVEL3>[99, 126, 723, 203]<FORK>Quarterly Report\n<BLOCK>title<LEVEL4>[99, 321, 687, 383]<FORK>Account 27182'
page 3: '<BLOCK>title<LEVEL4>[101, 126, 721, 201]<FORK>Quarterly Report\n<BLOCK>title<LEVEL5>[101, 321, 687, 383]<FORK>Account 16180'
windows: 3
```

</Accordion>

### 3.2 Choosing an image mode

Pass `images_config.image_mode` per request; when it is omitted SGLang uses `gundam`. vLLM's `mm_processor_kwargs={"crop_mode": true}` selects `gundam` and `false` selects `base`. Lower modes reduce prefill cost per page, which also lets a window hold more pages.

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Mode</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Use</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Window policy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px"}}><code>tiny</code>, <code>small</code></td>
      <td style={{padding: "9px 12px"}}>Lowest prefill cost for simple pages.</td>
      <td style={{padding: "9px 12px"}}><code>--uocr-doc-page-window</code></td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><code>base</code></td>
      <td style={{padding: "9px 12px"}}>One global view per page; the model card's multi-page setting.</td>
      <td style={{padding: "9px 12px"}}><code>--uocr-doc-page-window</code></td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><code>large</code></td>
      <td style={{padding: "9px 12px"}}>Higher-resolution global view.</td>
      <td style={{padding: "9px 12px"}}><code>--uocr-doc-page-window</code></td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px"}}><code>gundam</code></td>
      <td style={{padding: "9px 12px"}}>Default. Global view plus dynamic tiles for dense pages.</td>
      <td style={{padding: "9px 12px"}}><code>--uocr-doc-page-window</code></td>
    </tr>
  </tbody>
</table>
