---
name: cookbook-review-pr
description: Review a pull request against the SGLang Cookbook (docs_new/, Mintlify) contribution checklist. Run with /cookbook-review-pr <PR number>.
---

# Cookbook Review PR

Fetch the diff, run the checklist, report what you find. The cookbook lives in `docs_new/` (Mintlify) — this checklist targets that layout.

## Usage

```
/cookbook-review-pr <PR number>
```

## Steps

1. `gh pr view <N> --json title,body,files,author,baseRefName,headRefName,commits,reviews`
2. `gh pr diff <N>`
3. `gh pr list --state open --search "<model name>"` (duplicate check)
4. Run every checklist item against the diff
5. Output per-file verdicts and overall recommendation

## Checklist

### 1. File hygiene
- No stray files (`package-lock.json`, `.claude/settings.local.json`, unrelated `.gitignore` changes, IDE configs)
- Cookbook PRs should only touch:
  - `docs_new/cookbook/**/*.mdx` — new or edited pages
  - `docs_new/src/snippets/**/*.jsx` — deployment widgets
  - `docs_new/docs.json` — navigation entry
  - `docs_new/cookbook/autoregressive/intro.mdx` (or similar `intro.mdx`) — vendor card
  - `docs_new/cards/logos/<vendor>.png` — only when adding a brand-new vendor
- Pages must be `.mdx`, not `.md` — Mintlify uses MDX. Flag any `.md` additions under `docs_new/cookbook/`.
- Files must end with a trailing newline — flag `\ No newline at end of file`.
- Check commit history for unrelated commits (e.g., "Create settings.local.json") that may have been included accidentally.

### 2. Frontmatter
- Every new MDX page needs `title:` and `metatags.description:` — flag missing fields.
- `tag: NEW` should only be added when this is genuinely a new launch. If the PR adds a `NEW` tag, check whether older `tag: NEW` entries should be removed in the same PR (`grep -RlE "^tag: NEW" docs_new/cookbook/`). The site shouldn't accumulate stale NEW tags.
- `description` must be a real one-line value prop — not a copy-paste from another vendor.

### 3. Launch command
- Must use `sglang serve` — flag `python -m sglang.launch_server` (deprecated, issue #33).
- Applies to docs AND the snippet's `generateCommand`.
- New PRs should fix pre-existing deprecated commands in files they touch, not pile on more hardware.
- Also flag `python3 -m sglang.launch_server` (same issue, just python3 variant).

### 4. Hardware specs

| GPU    | Memory |
|--------|--------|
| A100   | 80GB   |
| H100   | 80GB   |
| H200   | 141GB  |
| B200   | 180GB  |
| B300   | 275GB  |
| GB300  | 275GB  |
| MI300X | 192GB  |
| MI325X | 256GB  |
| MI350X | 288GB  |
| MI355X | 288GB  |

- TP must make sense: `model_weight_GB / (tp * gpu_mem)` should fit with ~20-30% headroom.
- BF16 ≈ params * 2 GB, FP8 ≈ params * 1 GB, FP4 ≈ params * 0.5 GB.
- For MoE models, use total weight size (all experts), not active params.
- Multi-node configs: verify node count × GPUs × memory is sufficient.
- GB300 single-node host is typically 4 GPUs → TP=4 is the practical ceiling.

### 5. Deployment snippet quality
- Snippet file is a single `.jsx` under `docs_new/src/snippets/autoregressive/<modelname>-deployment.jsx` (kebab-case, no folder, no `index.js`).
- Single named export `<ModelName>Deployment` (PascalCase ending in `Deployment`) — must match the import in the MDX page exactly.
- `generateCommand` output must match what the docs describe.
- GPU names uppercase in labels: `MI300X` not `MI300x`.
- Hardware selection must actually change the generated command.
- `modelConfigs` needs an entry for every (variant × hardware) combo the UI allows (missing = runtime crash).
- Model path in `generateCommand` must match the right model (e.g., don't use Scout path in a Maverick generator).
- Option IDs must be unique — two items with `id: 'enabled'` in the same group is a bug.
- Each radio option group needs exactly one `default: true`.
- Watch for JS syntax errors (mismatched braces, trailing commas inside JSX).
- If a quantization/weight option exists in the UI, it must do something in `generateCommand` — dead options are misleading.
- No dead code: unused functions, unreachable `commandRule` properties superseded by `generateCommand`, or `getDynamicItems` that return static arrays.
- No duplicate conditional blocks with identical conditions — consolidate them.
- Platform-required flags (e.g., AMD Triton attention) must be unconditional, not gated behind optional checkboxes.
- Silently ignoring user selections (e.g., DP checkbox does nothing for a platform) is a UX bug — show a message or disable the option.
- Don't import shared base classes or components — snippets in `docs_new/` are intentionally self-contained. Each snippet inlines its rendering boilerplate.

### 6. Port consistency
- Use `--port 30000` everywhere (not 8000).
- Applies to docs, generated commands, curl examples, benchmark commands, and client code (`base_url`).
- All examples on the same page must use the same port — launch command port must match client/curl port.
- For multi-node configs, shell variable placeholders like `${PORT}` are acceptable but document the expected value.

### 7. Quantization rules
- FP4 is Blackwell-only (B200/B300/GB300) — never AMD.
- BF16 and FP8 work on both NVIDIA and AMD.
- AMD FP4 options must be `disabled: true` in the UI.
- FP8 configs that add `--kv-cache-dtype fp8_e4m3` should note potential accuracy trade-offs.

### 8. Duplicate PRs
- Another open PR for the same model? Flag it.
- Compare which is more complete (MDX content, snippet correctness, benchmarks, launch commands).
- Note merge conflict risk if they touch the same files (especially `docs.json` and the vendor `intro.mdx` card list).
- Same author with overlapping PRs: flag the older one as potentially superseded.

### 9. Navigation and homepage
- New page → `docs_new/docs.json` must be updated. The page path goes under the right vendor group inside `navigation` → Cookbook → Autoregressive Models, in root-relative form **without `.mdx` extension**: `cookbook/autoregressive/<Vendor>/<Model>`.
- Deleted/renamed page → both `docs.json` and any `<Card href="...">` referencing it must follow.
- New flagship for an existing vendor → the `<Card href>` in `docs_new/cookbook/autoregressive/intro.mdx` should point to the new page.
- New vendor → a new `<Card>` in `docs_new/cookbook/autoregressive/intro.mdx` AND a logo asset under `docs_new/cards/logos/<vendor>.png`.
- Don't change `docs_new/cookbook/intro.mdx` for individual model adds — it only lists top-level categories.

### 10. Links and factual claims
- HuggingFace URLs point to the right model — verify the model actually exists.
- No `sgl-project-dev` references (use `sgl-project`).
- Docker images should come from `lmsysorg/sglang` — flag alternatives like `rocm/sgl-dev`.
- Internal links use root-relative paths without extension (Mintlify convention): `/cookbook/autoregressive/<Vendor>/<Model>`. Flag `.md` / `.mdx` extensions in internal links and `../`-style relative paths to other pages.
- External docs links: `docs.sglang.io` is canonical (`.ai` redirects there).
- Markdown links well-formed: `[text](url)` not `[text] (url)`.
- Google Drive sharing links will NOT render as images in markdown — must use direct image URLs or host in the repo (`docs_new/images/` or `docs_new/cards/`).
- License claims must match the actual HuggingFace model license (common error: saying "Community License" when model is Apache 2.0).
- Shell placeholders like `export VAR=${VAR}` are bash no-ops — use `export VAR=<placeholder>` or actual example values.

### 11. Scope
- Do the changes match what the PR title says?
- Flag global changes hiding behind a platform-specific title (e.g., "H200 FP8" PR that adds `--kv-cache-dtype` to every platform).
- Check conditionals carefully: `if (quantization === 'fp8')` without a hardware guard affects ALL platforms, not just the one in the title.
- Unmentioned side-fixes (bug fixes, flag renames, casing corrections) should be documented in the PR body.

### 12. Benchmarks
- Benchmarks use `python3 -m sglang.bench_serving`, not `sglang serve` with benchmark flags.
- Deploy and benchmark are separate steps.
- Benchmark port must match the deployment port in the same section.
- Benchmark "Test Environment" quantization must match a quantization actually listed in Section 1 — `(BF16)` on a model that only released INT4 is a factual bug.

### 13. Build / validate
```bash
cd docs_new
mint validate
mint broken-links
```
Optional: `mint dev` for a visual smoke test at `localhost:3000`.

### 14. Doc ↔ snippet parity
- For each hardware option in the snippet, the launch command shown in the doc (benchmark section, tips, deployment example) must equal the snippet's emitted command — same flags, same ordering. This is the single most common review finding.
- If a flag is platform-required (not user-toggleable), the snippet owns it and the doc must mirror it.

### 15. Grammar and spelling (docs PRs)
- Check all added/changed lines for typos, misspellings, and grammar errors.
- Common issues: "recommend" vs "recommended", subject-verb agreement, misspelled technical terms.
- Flag each error with the exact wrong text and correction.

### 16. Reviewer feedback
- Check existing review comments from `gh api repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/pulls/<N>/comments` — have prior reviewer requests been addressed?
- Unresolved requested changes from collaborators should be flagged.
- If a reviewer requested something specific (e.g., accuracy warning), verify it was added in the latest diff.

## Output

Per file:
- ✅ PASS
- ⚠️ ISSUE: <what>
- 🔴 BLOCK: <what>

Overall: **APPROVE** / **REQUEST CHANGES** / **BLOCKED**
