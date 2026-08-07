# SGLang Gemini Review Guidelines

When reviewing pull requests in this repository, read
`.claude/rules/modify-component-must-read.md` and check whether the change
follows it.

Use that file as the source of truth. When applying these checks:

- If the pull request modifies a component listed in
  `.claude/rules/modify-component-must-read.md`, verify that the implementation
  follows the corresponding required guidance.
- Treat violations of those component-specific requirements as review findings,
  not as optional style suggestions.
- Prefer actionable comments that identify the changed component, the required
  skill guidance, and the specific code path that appears inconsistent with it.
- Avoid broad comments about files outside the pull request diff unless they are
  needed to explain the violation.
