---
id: preserve-user-visible-behavior-as-a-hard-rule
type: maxim
title: Preserve user-visible behavior as a hard rule
status: active
created: 2026-02-11
updated: 2026-02-11
tags: [pensieve, maxim]
---

# Preserve user-visible behavior as a hard rule

## One-line Conclusion
> Any unexpected user-visible behavior change is treated as a bug.

## Quote
"We do not break user-visible behavior."

## Guidance
- Keep outputs, contracts, and UX stable unless change is explicitly approved.
- Treat behavior regressions as priority defects.
- Add tests for behavior that users already rely on.

## Boundaries
- Explicitly approved behavior changes are allowed when documented and reviewed.

## Context Links (recommended)
- Based on: [[knowledge/taste-review/content]]
- Related: [[maxims/eliminate-special-cases-by-redesigning-data-flow]]
- Related: [[maxims/prefer-pragmatic-solutions-over-theoretical-completeness]]
