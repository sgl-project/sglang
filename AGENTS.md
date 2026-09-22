# Agent instructions

Repo-wide guidance for coding agents lives under `.claude/` and is symlinked
at `.agents/` so non-Claude agents can use it too.

- `.agents/rules/` -- coding rules. Each file's `paths:` frontmatter lists the
  globs it governs; `modify-component-must-read.md` has no scope and always
  applies. Before editing a file, read every rule whose globs match it.
- `.agents/skills/<name>/SKILL.md` -- task playbooks (adding kernels, writing
  tests, CI triage, profiling, naming conventions, ...). Each frontmatter
  `description` says when the skill applies; read the full `SKILL.md` before
  starting such a task.

Claude Code loads `.claude/rules/` and `.claude/skills/` natively. This file
exists for Codex and other agents that read `AGENTS.md`.
