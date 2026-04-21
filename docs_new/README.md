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
└── cookbook/               # Model deployment recipes
    ├── intro.mdx           # Cookbook overview and recipe index
    └── autoregressive/     # Autoregressive model recipes
        └── Qwen/
            └── Qwen3.5.mdx
```

Pages are `.mdx` files with YAML frontmatter. Navigation is defined in `docs.json`.

## Local development

### Prerequisites

- Node.js >= 20

### Setup

```bash
# Install the CLI
npm i -g mint

# Start the dev server (with hot reload)
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
# 1. Fork and clone the repo
git clone https://github.com/<YOUR_USERNAME>/sgl-docs.git
cd sgl-docs

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

1. Create a new `.mdx` file under `cookbook/` following the existing directory structure (e.g., `cookbook/llm/<Vendor>/<Model>.mdx` or `cookbook/vlm/<Vendor>/<Model>.mdx`)
2. Use an existing recipe like `cookbook/llm/Qwen/Qwen3.5.mdx` as a template
3. Add your page to the navigation in `docs.json`
4. Each recipe should include:
   - Model introduction and key specs
   - Installation / environment setup
   - Deployment configuration (with hardware recommendations)
   - Usage examples (basic + advanced)
   - Benchmarks (if available)

### Writing guidelines

- Use active voice: "Run the command" not "The command should be run"
- Address the reader as "you"
- Keep sentences concise — one idea per sentence
- Lead with the goal, then the steps
- Use consistent terminology
- Include concrete examples and code snippets

## Acknowledgements

Thank you to all the authors who contributed to the original documentation in [`sglang/docs/`](https://github.com/sgl-project/sglang/tree/main/docs) and the original cookbook in [`sgl-cookbook`](https://github.com/sgl-project/sgl-cookbook). The migration to the new Mintlify-based documentation was led by the following [ACM-VIT](https://github.com/ACM-VIT) students:

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
