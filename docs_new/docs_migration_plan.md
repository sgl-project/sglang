# SGLang Documentation Migration Plan

## Background

Migrate the new Mintlify-based documentation (currently in the standalone `sgl-docs` repo) into the sglang main repo under `docs_new/`, and point `staging.docs.sglang.io` to it.

### Current State

| Item | Location | Stack | Domain |
|------|----------|-------|--------|
| Old docs | `sglang/docs/` | Sphinx + GitHub Pages | `docs.sglang.io` |
| New docs + cookbook | `sgl-project/sgl-docs` repo | Mintlify | `lmsysorg.mintlify.app` (temp preview) |

- Cookbook is already inside `sgl-docs/cookbook/`, no separate repo needed.
- Old docs CI (`execute-notebook.yml`, `lint.yml`) only watches `docs/**`, will not be triggered by `docs_new/**`.

---

## Phase 1: Git Subtree Merge (Local Experiment)

> Goal: Merge `sgl-docs` into `sglang` repo's `docs_new/` directory, preserving full commit history and authorship.

```bash
# 1. Create a new branch (sglang remote is NOT affected)
cd /path/to/sglang
git checkout -b docs-new-migration

# 2. Add sgl-docs as a remote (sgl-docs repo is NOT affected, read-only fetch)
git remote add sgl-docs git@github.com:sgl-project/sgl-docs.git
git fetch sgl-docs

# 3. Subtree merge — all sgl-docs content goes into docs_new/, full history preserved
git subtree add --prefix=docs_new sgl-docs main

# 4. Keep the remote for ongoing sync during migration period
#    (remove only after sgl-docs is officially archived)
```

### Safety Guarantees

- `sgl-docs` original repo: **unaffected** (fetch only, no push)
- `sglang` remote: **unaffected** (local branch, no push until ready)
- Rollback: `git checkout main && git branch -D docs-new-migration`

### Side Effect: Contributors

`git subtree add` (without `--squash`) imports all original commits. Authors from `sgl-docs` will appear in `sglang`'s git history and GitHub Contributors list. This is intentional — it gives proper credit.

---

## Phase 2: Configure Mintlify for `docs_new/` (on branch)

> Goal: Make Mintlify read from `sglang` repo's `docs_new/` subdirectory instead of the standalone `sgl-docs` repo. **No need to merge to main first** — Mintlify can point to a specific branch for validation.

1. Log in to [Mintlify Dashboard](https://dashboard.mintlify.com)
2. Change the project's **GitHub repository** from `sgl-project/sgl-docs` to `sgl-project/sglang`
3. Set **Branch** to `docs-new-migration` (temporarily, for validation)
4. Set **Documentation directory** to `docs_new` (Mintlify supports monorepo subdirectory)
5. `docs.json` (Mintlify config) will be at `docs_new/docs.json` after the subtree merge — paths inside it (e.g., `cookbook/llm/Qwen/Qwen3`) are relative to `docs_new/`, so no changes needed
6. Verify the preview build succeeds on Mintlify

---

## Phase 3: DNS & Custom Domain for `staging.docs.sglang.io`

> Goal: Make `staging.docs.sglang.io` serve the new Mintlify docs.

1. **DNS**: Add a CNAME record for `staging.docs.sglang.io` pointing to Mintlify's endpoint (typically `cname.mintlify.dev`)
2. **Mintlify Dashboard**: Settings > Custom Domain > add `staging.docs.sglang.io`
3. Mintlify handles SSL certificate automatically
4. Verify `staging.docs.sglang.io` loads correctly

---

## Phase 4: Ongoing Sync During Migration Period

> During the transition, `sgl-docs` may still receive updates. Sync them into `docs_new/` as needed.

```bash
# Pull latest changes from sgl-docs into docs_new/
git subtree pull --prefix=docs_new sgl-docs main
```

Once `sgl-docs` is frozen, this step is no longer needed.

---

## Phase 5: CI/CD (Optional, Post-Migration)

Current `docs/**` CI workflows will **NOT** trigger for `docs_new/**` changes. This is fine initially since Mintlify has its own GitHub integration for auto-deployment on push to main.

Optional additions later:
- Link checking (lychee) for `docs_new/**/*.mdx`
- Mintlify broken-link or build validation on PR

---

## Phase 6: Final Cutover

> Goal: Promote staging to production.

| Stage | `docs.sglang.io` | `staging.docs.sglang.io` |
|-------|-------------------|--------------------------|
| After Phase 3 | Sphinx (old docs) | Mintlify (new docs) |
| After cutover | Mintlify (new docs) | Keep or remove |

Cutover steps:
1. Confirm `staging.docs.sglang.io` is stable and content-complete
2. Update DNS: point `docs.sglang.io` CNAME from GitHub Pages to Mintlify (`cname.mintlify.dev`)
3. Update Mintlify Dashboard custom domain to `docs.sglang.io`
4. Remove or archive old resources:
   - Delete `sglang/docs/` (old Sphinx docs)
   - Delete `.github/workflows/release-docs.yml` and `.github/workflows/execute-notebook.yml`
   - Archive `sgl-project/sgl-docs` repo on GitHub
   - Remove the `sgl-docs` git remote: `git remote remove sgl-docs`
   - Optionally archive `sgl-project/sgl-project.github.io` repo

---

## Execution Order

> Mintlify supports pointing to a specific branch, so we can validate on `docs-new-migration` **before** merging to main.

| Step | Action | Who | Dependency |
|------|--------|-----|------------|
| 1 | Phase 1: subtree merge on local branch | Dev | — |
| 2 | Push branch to `sgl-project/sglang` | Dev | Step 1 |
| 3 | Phase 2: configure Mintlify Dashboard to read from `sgl-project/sglang` branch `docs-new-migration` `docs_new/` | Admin (Mintlify access) | Step 2 |
| 4 | Phase 3: DNS CNAME + Mintlify custom domain for `staging.docs.sglang.io` | Admin (DNS access) | Step 3 |
| 5 | Verify staging site | Team | Step 4 |
| 6 | Merge PR to main, switch Mintlify branch back to `main` | Dev + Admin | Step 5 confirmed OK |
| 7 | Phase 4: sync any remaining sgl-docs updates | Dev | As needed |
| 8 | Phase 6: final cutover when ready | Admin | Step 6 done |
