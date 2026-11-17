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

The page must start with `## 1. Quick start`: keep installation to one runnable
command, add at most one short orientation paragraph, and render the scoped command
builder before detailed capability tables or deployment commentary. A reader should
reach a generated Serve command without scrolling through model background first.

Follow it with `## 2. Model capabilities` and 2–3 short paragraphs:

- paragraph 1: what the model does and where it is genuinely strong;
- paragraph 2: when to choose it and the most important limitation or tradeoff;
- optional paragraph/table: checkpoint or mode routing that a user must understand before serving.

Keep the capability lead between roughly 45 and 180 words. Replace marketing superlatives with
concrete capabilities. A reader should learn more than “this is a powerful image/video
model,” but should not have to read the architecture section to choose the right model.

## Scoped command builder

New diffusion pages use `templates/diffusion-config.jsx.tmpl` and opt into the shared
`commandBuilder` renderer. Setup, Server, and Request choices share one semantic selection and
one command composer; do not create a second command engine or assemble fragments in MDX.

Classify every dimension by lifecycle:

- `scope: "base"`: the visible Setup tab—hardware-independent required decisions such as
  checkpoint weights and request mode. Hardware, Nodes × GPUs/node, and the recommended
  verified deployment are supplied by the shared builder.
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
`replicate` across nodes. Reserve `disabled` for combinations that cannot work (a kernel the
platform does not ship, a mode the checkpoint cannot serve); an option that runs but has not
been through a verification round should declare `soft` instead, which keeps it selectable and
labels the pick as unverified. Keep `torch.compile` and similarly narrow experiments in the
detailed prose until they have a broadly compatible recipe.

Each dimension should provide a concise `description`, an optional quality label, and a
`learnMore` anchor. Each option should provide at most two lines of decision-relevant explanation,
its exact `flags`/`stripPrefixes`/`env`/`hints`, and a `disabled` (with `disableReason`), `soft`
(with `softReason`), or `verifiedWhen` predicate when support is conditional — the reason strings
are user-facing: blocked options flash theirs under the row on tap and expose it as a tooltip.
The builder stores all semantic choices in the URL hash; active scope, expanded state, head
address, and node rank stay local.

Legacy configs without `commandBuilder` continue to use the old matrix renderer. Do not migrate
an existing page opportunistically; use the new schema for new diffusion models and deliberate
model-by-model migrations.

## Review checklist

- tags render before section 1 and describe the model rather than the runtime;
- Quick start puts installation and the generated commands before detailed model background;
- the capability section explains strength and boundary without marketing filler;
- checkpoint variants and request modes are unambiguous;
- Setup keeps both commands visible as a deployment overview; Server and Request each show only
  the command controlled by that scope;
- command bodies grow naturally up to their collapsed limit and do not reserve empty height;
- topology-dependent recommended defaults are explicit in the setting summary and generated command;
- legal custom topologies are Unverified and copyable; statically illegal combinations block Copy;
- attention, quantization, caching, compile, and similar orthogonal features have explicit
  quality contracts and verified scopes;
- unverified hardware or performance claims are absent;
- `node docs/scripts/check_cookbook_configs.mjs` and Mintlify validation pass.
