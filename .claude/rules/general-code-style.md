---
paths:
  - "**/*.py"
---

# General Code Style

Default conventions for new and modified Python code. Prefer these unless there is a concrete reason not to; call out deviations in review.

- **Prefer stateless.** Favor pure functions over methods that mutate instance state; pass inputs in, return outputs out.
- **Prefer immutable.** Default to immutable data (frozen structs, tuples, read-only values); mutate only when there is a clear need.
- **Functions stay small.** Keep each function under ~100 LOC; split larger ones into named helpers.
- **Files stay small.** Keep each file under ~2k LOC; split larger modules along cohesive boundaries.
- **Core functions read like pseudocode.** The main / orchestration function of a unit should be short and read like algorithm pseudocode — push detail into well-named helpers so the top-level flow is obvious.
- **Avoid mixins.** Don't add behavior via mixin classes; prefer explicit composition (hold a collaborator and call it) or plain functions.
- **Prefer protected over public.** Default methods to protected (`_name`); expose only what callers actually use.
- **Prefer keyword arguments.** Call functions of 2+ args by keyword, and design APIs to be called that way.
- **Pass what you need, not the god object.** Give a callee the specific values it uses (by keyword), not a whole large object (`ModelRunner`, `Scheduler`); reserve passing the whole object for a leaf whose contract genuinely requires it. Even then, keep it read-only — read fields off it and return results for the caller to assign, rather than writing fields back through it.
