# Diffusion cookbook authoring contract

Use this contract for every model page under
`docs/cookbook/diffusion/<Vendor>/<Model>.mdx`.

## Opening structure

Import and render the shared tag widget immediately after frontmatter:

```mdx
import { DiffusionModelTags } from '/src/snippets/diffusion/model-tags.jsx';

<DiffusionModelTags tags={["video + audio", "text-to-video", "reference control", "4–15 seconds"]} />
```

Use 4–6 tags, ordered from broad to specific:

1. output modality or product class;
2. primary request modes;
3. the capability that differentiates the model;
4. an important scale, latency, duration, resolution, or architecture fact.

Do not spend tags on generic claims such as `native`, `fast`, `high quality`, or
`SGLang`. Attention backends, Cache-DiT, and online quantization are feature overlays,
not model identity, unless a published checkpoint is intrinsically tied to that format.

The page must start with `## 1. Model Introduction` followed by 2–3 short paragraphs:

- paragraph 1: what the model does and where it is genuinely strong;
- paragraph 2: when to choose it and the most important limitation or tradeoff;
- optional paragraph/table: checkpoint or mode routing that a user must understand before serving.

Keep the lead prose between roughly 45 and 180 words. Replace marketing superlatives with
concrete capabilities. A reader should learn more than “this is a powerful image/video
model,” but should not have to read the architecture section to choose the right model.

## Deployment picker versus feature overlays

The command picker emits a verified **base deployment recipe**. Keep an axis in the picker
only when it changes one of these:

- hardware topology or required GPU count;
- checkpoint/weight partition;
- pipeline or request mode needed to produce the matching command and cURL;
- a placement policy with a separately validated capacity/performance cell.

Do not multiply the matrix for independent feature knobs. Put them in an `Optional feature
overlays` section after the base recipe:

| Goal | Overlay | Quality contract | Verified scope |
| --- | --- | --- | --- |
| Force an attention kernel | `--attention-backend ...` | Native precision or approximate/quantized | Exact hardware and task tested |
| Quantize weights | `--quantization ...` | Approximate; list protected FP32 layers | Exact checkpoint and hardware tested |
| Reuse computation | Cache-DiT sampling config | Approximate block reuse | Exact task profile tested |

For every overlay, document the default, the exact flag/config, whether it changes numerical
or perceptual output, incompatible combinations, and the hardware/task actually validated.
Sampling controls stay with request examples; they do not belong in the deployment matrix.

## Review checklist

- tags render before section 1 and describe the model rather than the runtime;
- the first screen explains capability, strength, and boundary without marketing filler;
- checkpoint variants and request modes are unambiguous;
- the picker remains a small base-recipe selector;
- attention, quantization, caching, compile, and similar orthogonal features have explicit
  quality contracts and verified scopes;
- unverified hardware or performance claims are absent;
- `node docs/scripts/check_cookbook_configs.mjs` and Mintlify validation pass.
