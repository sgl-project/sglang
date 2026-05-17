---
id: reduce-complexity-before-adding-branches
type: maxim
title: Reduce complexity before adding branches
status: active
created: 2026-02-11
updated: 2026-02-11
tags: [pensieve, maxim]
---

# Reduce complexity before adding branches

## One-line Conclusion
> When logic grows hard to read, simplify structure first and branch later only if necessary.

## Quote
"If you need more than 3 levels of indentation, you're screwed anyway; fix your program."

## Guidance
- Split large functions by responsibility.
- Keep control flow shallow and explicit.
- Prefer clear naming over explanatory comments.

## Boundaries
- Small, local branches are acceptable when they improve clarity.

## Context Links (recommended)
- Based on: [[knowledge/taste-review/content]]
- Related: [[maxims/prefer-pragmatic-solutions-over-theoretical-completeness]]
- Related: [[maxims/eliminate-special-cases-by-redesigning-data-flow]]
