# SGLang Documentation

The official documentation and cookbook for [SGLang](https://github.com/sgl-project/sglang) — a high-performance serving framework for large language models and vision-language models.

- **Docs**: Getting started guides, installation, and reference
- **Cookbook**: Battle-tested recipes for deploying specific models (Qwen, DeepSeek, Llama, GLM, etc.) on various hardware


## Project structure

```
.
├── docs.json              # Site configuration (navigation, theme, metadata)
├── index.mdx              # Homepage
├── docs/                  # Documentation pages
│   └── get-started/
│       └── install.mdx    # Installation guide
├── cookbook/              # Model deployment recipes (one .mdx page per model)
│   ├── intro.mdx          # Cookbook overview and recipe index
│   └── autoregressive/    # Autoregressive recipes (also: diffusion/, omni/, …)
│       └── DeepSeek/
│           └── DeepSeek-V4.mdx
└── src/snippets/          # Config-driven cookbook engine
    ├── _deployment.jsx    # Shared deploy-matrix engine (no model-specific code)
    ├── _playground.jsx    # Shared override-playground engine
    └── configs/
        └── deepseek-ai/   # Per-model config + benchmarks (HF-org folder)
            └── deepseek-v4.jsx
```

Pages are `.mdx` files with YAML frontmatter. Navigation is defined in `docs.json`.

## Local development

### Prerequisites

- Node.js >= 20

### Setup

```bash
# Install the CLI
npm i -g mint

# From docs_new/ (where docs.json lives), start the dev server (hot reload)
mint dev
```

Preview at `http://localhost:3000`.

### Useful commands

```bash
mint dev            # Start local preview server
mint broken-links   # Check for broken links
mint update         # Update the CLI
```

## Contributing

We welcome contributions! Whether you want to add a recipe for a new model, improve existing docs, or fix a typo — PRs are appreciated.

### Quick edit (GitHub)

1. Navigate to the file you want to edit on GitHub
2. Click the pencil icon to edit
3. Submit a pull request

### Local development workflow

```bash
# 1. Fork sgl-project/sglang and clone your fork
git clone https://github.com/<YOUR_USERNAME>/sglang.git
cd sglang/docs_new

# 2. Create a branch
git checkout -b my-changes

# 3. Start the dev server and make your changes
mint dev

# 4. Verify links aren't broken
mint broken-links

# 5. Commit and push
git add <files>
git commit -m "docs: describe your change"
git push origin my-changes

# 6. Open a pull request on GitHub
```

### Adding a new cookbook recipe

The autoregressive cookbook is **config-driven**: two shared engines —
`src/snippets/_deployment.jsx` (the deploy matrix) and `src/snippets/_playground.jsx`
(the override playground) — contain **no** model-specific code. Adding a model means adding
*data*: a per-model config (plus optional benchmarks) that both engines consume, and an
`.mdx` page that imports them. Copy [`DeepSeek-V4`](cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx)
as the reference instance.

**Recommended — use the Claude Code skill `/cookbook-add-model`.** It walks the whole flow
interactively: collect the model card + verified `sglang serve` recipes → instantiate the
template → wire up the nav/card → validate → fill in measured benchmarks. Related skills:
`/cookbook-migrate-model` (port an existing legacy-template page) and `/cookbook-review-pr`
(review a cookbook PR against the checklist).

**The files it creates / edits — using DeepSeek-V4 as the example:**

| File | Purpose |
|---|---|
| `src/snippets/configs/deepseek-ai/deepseek-v4.jsx` | Per-model config: `supportedHardware`, `variants`, `quantizations`, `strategies`, the `cells[]` deploy matrix (verified env + flags per `hw × variant × quant × strategy × nodes`), and `playgroundFeatures`. |
| `src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx` | One entry per cell with measured speed/accuracy + the `sglang_version` it ran on. Optional — skip until you have numbers. |
| `cookbook/autoregressive/DeepSeek/DeepSeek-V4.mdx` | The page: imports the engines + config, renders `<Deployment>` / `<Playground>`, and adds prose (intro + specs + license, config tips, advanced usage). |
| `docs.json` | Nav entry under Cookbook → category → vendor. |
| `cookbook/autoregressive/intro.mdx` | Vendor `<Card>` on the category homepage. |

Note the two folder conventions: under `configs/` the folder is the **HuggingFace org**
(`deepseek-ai`); under `cookbook/` it's the **display vendor** (`DeepSeek`). The page wires
everything together with data only — no engine edits:

```mdx
import { Deployment } from "/src/snippets/_deployment.jsx";
import { Playground } from "/src/snippets/_playground.jsx";
import { config }     from "/src/snippets/configs/deepseek-ai/deepseek-v4.jsx";
import { benchmarks } from "/src/snippets/configs/deepseek-ai/deepseek-v4-benchmarks.jsx";

<Deployment config={config} benchmarks={benchmarks} />
<Playground config={config} />
```

Keep `tag: NEW` on the new page and strip it from same-vendor siblings (≤1 per vendor), then
validate from `docs_new/`: `mint validate`, `mint broken-links`, and `mint dev` for a visual
smoke test.

> Diffusion / omni / specbundle pages follow their own category structure — don't force the
> autoregressive config-driven template on them.

### Writing guidelines

- Use active voice: "Run the command" not "The command should be run"
- Address the reader as "you"
- Keep sentences concise — one idea per sentence
- Lead with the goal, then the steps
- Use consistent terminology
- Include concrete examples and code snippets

## Acknowledgements

Thank you to all the authors who contributed to the original documentation in `sglang/docs/` and the original cookbook in [`sgl-cookbook`](https://github.com/sgl-project/sgl-cookbook). The migration to the new Mintlify-based documentation was led by the following [ACM-VIT](https://github.com/ACM-VIT) students:

[@Adhyan Jain](https://github.com/Adhyan-Jain), [@Maitri-shah29](https://github.com/Maitri-shah29), [@architnigam](https://github.com/architnigam), [@Nakul-Sinha](https://github.com/Nakul-Sinha), [@divyamagrawal06](https://github.com/divyamagrawal06), [@A-Taman](https://github.com/A-Taman), [@nimeshas](https://github.com/nimeshas), [@IshhanKheria](https://github.com/IshhanKheria), [@Krishang-Zinzuwadia](https://github.com/Krishang-Zinzuwadia), [@pokymono](https://github.com/pokymono), [@Ishitajoshii](https://github.com/Ishitajoshii), [@AdityaVKochar](https://github.com/AdityaVKochar)

Advised by [@adarshxs](https://github.com/adarshxs) (ACM-VIT) and [@wisclmy0611](https://github.com/wisclmy0611), [@Richardczl98](https://github.com/Richardczl98) (LMSYS).

## Community

- [GitHub](https://github.com/sgl-project/sglang)
- [Slack](https://slack.sglang.io/)
- [Discord](https://discord.gg/4ugb2t6YY2)
- [X / Twitter](https://x.com/lmsysorg)
- [LinkedIn](https://www.linkedin.com/company/sgl-project/)

## License

Apache License 2.0 — see the [LICENSE](LICENSE) for details.
