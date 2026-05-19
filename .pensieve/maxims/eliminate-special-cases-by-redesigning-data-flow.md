---
id: eliminate-special-cases-by-redesigning-data-flow
type: maxim
title: Eliminate special cases by redesigning data flow
status: active
created: 2026-02-11
updated: 2026-02-11
tags: [pensieve, maxim]
---

# Eliminate special cases by redesigning data flow

## One-line Conclusion
> Prefer changing data shape or flow so edge cases become the default path.

## Quote
"Sometimes you can see a problem in a different way and rewrite it so that the special case goes away and becomes the normal case."

## Guidance
- Remove branches that only exist to patch one-off inputs.
- Fix upstream contracts before adding downstream conditionals.
- Refactor data structures first, then simplify code paths.

## Boundaries
- If removing a branch breaks user-visible behavior, treat it as a migration and review first.

## Context Links (recommended)
- Based on: [[knowledge/taste-review/content]]
- Related: [[maxims/preserve-user-visible-behavior-as-a-hard-rule]]
- Related: [[maxims/reduce-complexity-before-adding-branches]]
