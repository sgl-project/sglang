# MDX authoring rules (Mintlify) + invocation-example patterns

Loaded on demand by the `cookbook-add-model` skill (Phase 5, writing the page prose).
These are model-agnostic Mintlify hygiene rules — the most common review findings.
The cookbook is **Mintlify**, not Docusaurus.

## Mintlify syntax

**Allowed components**: `<Card>`, `<CardGroup>`, `<Note>`, `<Tip>`, `<Warning>`,
`<Info>`, `<Accordion>`, `<AccordionGroup>`, `<Steps>`, `<Step>`, `<Tabs>`, `<Tab>`,
`<CodeGroup>`, `<Frame>`, `<Icon>`.

**Forbidden** (flag every occurrence):
- Docusaurus admonitions (`:::note` / `:::warning` / …) — use `<Note>` / `<Warning>`.
- `@site/...` / `@theme/...` imports — use absolute `/src/snippets/...`.
- GitHub alert blocks (`> [!NOTE]`, `> [!WARNING]`).
- **Markdown pipe tables** on new pages — use JSX `<table>` (see below).
- Inline `<details>` / `<summary>` — use `<Accordion>`.
- Unknown / non-Mintlify components.
- `<CardGroup>` / `<Card>` on individual model pages — those are for category `intro.mdx` only.

**Code fences**: always labeled — ` ```python Example `, ` ```bash Command `,
` ```shell Command `, ` ```text Output `. When nesting a fenced block inside another,
the **outer** fence uses four backticks.

**Internal links**: root-relative, no extension (`/cookbook/<category>/<Vendor>/<Model>`);
`docs.sglang.io` is canonical. Flag `.md`/`.mdx` extensions and `../`-relative page links
in body prose. (Existing cookbook pages do use `../../../docs/...` for cross-links into
the non-cookbook docs tree — that's the established exception; don't introduce new ones.)

## JSX tables (required for all tables on new pages)

```jsx
<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700}}>Col</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style={{padding: "9px 12px"}}>cell</td></tr>
  </tbody>
</table>
```

Alternate column background colors (`rgba(255,255,255,0.02)` / `0.05`) for readability;
adjust `<colgroup>` widths for 3- or 5-column tables. The DeepSeek-V4 page §1 variants
table is a live reference.

## Invocation-example patterns (§3 Advanced Usage)

- **Reasoning-parser output shape must match the example**:
  - *Separate-field* parsers (most qwen/glm, `kimi_k2`, `deepseek-v4`): thinking lands
    in `message.reasoning_content`, answer in `message.content` — print both.
  - *Inline-tag* parsers (e.g. `minimax-append-think`): thinking is wrapped in
    `<think>...</think>` **inside** `message.content` — the client parses the tags; for
    streaming, buffer and split on the markers.
  Pick the pattern from the model card / SGLang docs for that specific parser.
- **Hybrid reasoning models**: show both thinking-on (default) and thinking-off
  (`extra_body={"chat_template_kwargs": {"thinking": False}}` or `enable_thinking: False`).
- **Tool-call follow-up on thinking models**: the final assistant turn may put text in
  `reasoning_content` instead of (or with) `content` — print both so the output isn't a
  misleading `None`.
- **Every runnable block** is immediately followed by `**Output Example:**` + a
  ` ```text Output ` block with **real** server output (verbatim, not paraphrased).
  `Pending update...` is acceptable only with the user's explicit acknowledgement.
- **Do not hardcode sampling params** (`temperature`, `top_p`) in sample code — SGLang
  uses `generation_config.json` defaults. Listing "Recommended generation" in §1 is fine.
- Format raw API objects (`ChatCompletionMessage(...)`) into readable Reasoning /
  Content / Tool Calls blocks.

## Commands & ports

- **Deploy/launch** commands use `sglang serve --model-path …` — never
  `python -m sglang.launch_server` / `python3 -m sglang.launch_server` (deprecated).
- **Benchmark workload** commands use `python3 -m sglang.bench_serving …` (never bare
  `python -m`); built-in accuracy scripts use `python3 benchmark/...`.
- Port **30000** everywhere on a page — launch, curl, client `base_url`, and bench must
  agree. Keep one canonical deploy command (the Deploy widget) and don't re-paste launch
  commands across sections; the documented command must match the widget's output for the
  same selection (doc ↔ config parity).

## Factual hygiene

- License must match the actual HuggingFace license (don't copy from another model).
- HF URLs resolve to a real model; Docker images from `lmsysorg/sglang`.
- No Google-Drive image links (they don't render); host images in the repo.
- Shell placeholders are `export VAR=<value>`, not `export VAR=${VAR}` (a bash no-op).
- `tag: NEW` is sparing — at most one per `<category>/<Vendor>/` dir (the newest); strip
  it from siblings when adding a new NEW page.
