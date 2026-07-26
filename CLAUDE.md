# sglang-kimi

## Git workflow

- `k3-track` (tracking branch): commit and push directly, no PR.
- `kimi-k3` (main dev branch): PR by default — make changes on a feature branch in a separate worktree and land them through a PR, never push directly to `kimi-k3`.

## Secrets — never commit

Any key / token / credential (HF token `hf_*`, GitHub token `ghp_*` / `github_pat_*`, cloud key, SSH private key, kubeconfig credential, API key, password) is **forbidden** from:

- any branch of this repo (code branches, `k3-track`'s README / journals / repro.md), commit messages, PRs
- shared storage (`/cluster-storage`, scripts and configs written to a shared devbox) and any file that gets uploaded or synced

Requirements:

- Secrets travel through transient channels only: interactive login (e.g. `hf auth login`), runtime env vars (never written to a file), local untracked config
- When docs or journals need to mention a token, always use a placeholder (`hf_xxxxxxxx...`) and state how to obtain the real one
- Before `git add`, grep the change for token patterns (`hf_[A-Za-z0-9]{20,}`, `ghp_`, `github_pat_`, `AKIA`, `BEGIN.*PRIVATE KEY`, `sk-`); stop on any hit
- Once a secret has been committed or pushed, treat it as **leaked**: report it immediately and rotate. Deleting the file or rewriting history is not a fix.

## K3 Track — plans & progress tracking

Kimi K3's plans, progress, and experiment records do not live on the code branches. They live on this repo's separate orphan branch `k3-track`:

- Browse online: https://github.com/DarkSharpness/sglang-kimi/tree/k3-track
- Local worktree (**required**): `git worktree add ../sglang-kimi-k3-track k3-track` (if it already exists, just cd into it). All tracking reads and writes happen in that worktree only — do not check out `k3-track` in a code worktree, and do not write tracking files into a code branch.

### Journals — operation log (so agents can retrieve context)

Any meaningful operation (an experiment finished, a debugging conclusion, a settled design decision, a PR opened or merged, a status change) can get a new file under `journals/` on the `k3-track` branch, committed and pushed directly, plainly describing what was done:

- Filename: `<date>-<time>-<author>-<commit>-<descript>.md`, e.g. `2026-07-18-1954-lsyin-1ccf020d-kda-fused-decode.md` (commit is the short sha of the related code commit)
- No structured template required — just make it clear what was done
- Only add new files, never edit an old journal; one operation per file (naturally free of merge conflicts)
- Journals are a lightweight channel and are not subject to the review workflow below; an agent may write them directly

### Update workflow (AI must follow)

1. **AI detects and reminds**: while working alongside a human in a code worktree, if you notice that tracking may need an update (an experiment produced results, a todo completed, a new direction settled, a status changed), remind the human proactively, but do not update it yourself.
2. **Human provides the prompt**: only once the human states explicitly what to update does the AI touch the k3-track worktree.
3. **AI updates, human reviews**: the AI writes it following the routing and format in the `CLAUDE.md` at the root of k3-track, then hands it to the human for review; commit and push only after approval.

Benchmark numbers must be bound to a code commit (`data@YYYY-MM-DD-<sha8>` round). Numbers without that binding are not recorded.
