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

Do not multiply the matrix for independent feature knobs. Use the shared
`DiffusionFeatureGuide` after the base recipe and separate controls by lifecycle:

- **Serve-time overlays** are startup flags that change kernels, precision, placement, or
  execution policy without creating a new base topology. Examples include attention backend,
  online quantization, encoder scheduling, and graph execution.
- **Request-time features** are sampling fields that may change between requests on the same
  resident server. Examples include an audited Cache-DiT quality level or multiple outputs.

If a verified best setting depends on the selected topology, emit it explicitly in the base
command instead of asking the reader to remember an overlay. Encoder placement is a common
example: the picker may emit `auto` for a single host and `replicate` across nodes. Mark that
feature as handled by the picker; reserve copied fragments for deliberate overrides.

Each feature entry must expose its default, exact incremental flag/config, quality contract,
incompatible combinations, and verified hardware/task. Use the quality labels to distinguish
native or lossless execution from approximate and experimental paths. Keep the collapsed
summary decision-complete; put installation steps, full commands, and benchmark methodology in
the detailed reference below the widget.

Do not make the guide another stateful command builder. Its copy action emits only the overlay
fragment, and its choices do not join the deployment matrix or URL state. Sampling controls stay
with request examples even when the guide summarizes them.

## Review checklist

- tags render before section 1 and describe the model rather than the runtime;
- the first screen explains capability, strength, and boundary without marketing filler;
- checkpoint variants and request modes are unambiguous;
- the picker remains a small base-recipe selector;
- topology-dependent recommended defaults are explicit in the generated command;
- the feature guide separates serve-time startup flags from request-time sampling fields;
- attention, quantization, caching, compile, and similar orthogonal features have explicit
  quality contracts and verified scopes;
- unverified hardware or performance claims are absent;
- `node docs/scripts/check_cookbook_configs.mjs` and Mintlify validation pass.
