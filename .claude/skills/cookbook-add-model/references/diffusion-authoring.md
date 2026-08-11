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

## Scoped command builder

New diffusion pages use `templates/diffusion-config.jsx.tmpl` and opt into the shared
`commandBuilder` renderer. Base, Server, and Request choices share one semantic selection and
one command composer; do not create a second command engine or assemble fragments in MDX.

Classify every dimension by lifecycle:

- `scope: "base"`: hardware-independent required decisions such as checkpoint weights and
  request mode. Hardware and Nodes × GPUs/node are supplied by the shared builder.
- `scope: "serve"`: startup flags such as placement, attention, precision, encoder scheduling,
  and graph execution. They modify only the complete Serve command.
- `scope: "request"`: sampling fields such as quality and outputs. They modify only the complete
  Request command.

Keep the topology registry small and honest. `verifiedRecipes` contains only exact end-to-end
runs on that hardware and resource shape. `autoTopology(selection)` may construct a legal custom
shape, but `resolveDeployment` must mark it `unverified` unless it exactly matches a recipe.
`validateTopology` returns static errors for impossible world-size, head, partition, or placement
combinations; errors disable both Copy actions. Never silently label a nearby GPU or topology as
verified.

Expose topology-dependent best values explicitly. A Server row may remain `Auto`, but its summary
and generated flag must show the effective policy—for example, encoder `auto` resolving to
`replicate` across nodes. Disable hardware-gated options rather than generating commands outside
their validated capability boundary. Keep `torch.compile` and similarly narrow experiments in
the detailed prose until they have a broadly compatible recipe.

Each dimension should provide a concise `description`, an optional quality label, and a
`learnMore` anchor. Each option should provide at most two lines of decision-relevant explanation,
its exact `flags`/`stripPrefixes`/`env`/`hints`, and a `disabled` or `verifiedWhen` predicate when
support is conditional. The builder stores all semantic choices in the URL hash; active scope,
expanded state, head address, and node rank stay local.

Legacy configs without `commandBuilder` continue to use the old matrix renderer. Do not migrate
an existing page opportunistically; use the new schema for new diffusion models and deliberate
model-by-model migrations.

## Review checklist

- tags render before section 1 and describe the model rather than the runtime;
- the first screen explains capability, strength, and boundary without marketing filler;
- checkpoint variants and request modes are unambiguous;
- the builder separates Base, Server, and Request inputs while keeping both output commands visible;
- topology-dependent recommended defaults are explicit in the setting summary and generated command;
- legal custom topologies are Unverified and copyable; statically illegal combinations block Copy;
- attention, quantization, caching, compile, and similar orthogonal features have explicit
  quality contracts and verified scopes;
- unverified hardware or performance claims are absent;
- `node docs/scripts/check_cookbook_configs.mjs` and Mintlify validation pass.
