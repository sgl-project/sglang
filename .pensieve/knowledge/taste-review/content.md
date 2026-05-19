---
id: taste-review-content
type: knowledge
title: Code Taste Review Knowledge Base
status: active
created: 2026-02-28
updated: 2026-02-28
tags: [pensieve, knowledge, review, taste]
---

# Code Taste Review Knowledge Base

Core philosophy, warning signals, and classic examples for code review.

## Sources

- Linus Torvalds public talks & Linux Kernel style
- John Ousterhout "A Philosophy of Software Design"
- Google Engineering Practices (Code Review guidelines)

## Supporting Materials

The `source/` directory can hold project-specific references. Language style guides are best pulled from their official repositories:

- Google Style Guides: https://github.com/google/styleguide

Example (project uses Python + TypeScript):

```bash
mkdir -p source/google-style-guides
curl -o source/google-style-guides/pyguide.md https://raw.githubusercontent.com/google/styleguide/gh-pages/pyguide.md
curl -o source/google-style-guides/tsguide.html https://raw.githubusercontent.com/google/styleguide/gh-pages/tsguide.html
```

## Summary

This knowledge base weaves three threads together:

1. Linus: Eliminate special branches through data structures and rewrites
2. Ousterhout: Control complexity through module depth and abstraction
3. Google: Review by priority with code health as the goal

## Applicable Scenarios

- Need theoretical backing for review comments
- Need to identify implementations that "work but are unmaintainable"
- Need to align the team on complexity and code health standards

---

## Core Principles

### 1) Linus: Good Taste

Key ideas:
- Eliminate special cases through refactoring, not by piling on conditional branches
- Think about data structures first, then write control flow
- Control nesting depth and function length
- User-visible behavior is a hard boundary (do not change it casually)

### 2) Ousterhout: Complexity Management

Three symptoms of complexity:

1. **Change amplification**: a small change ripples across the codebase
2. **Cognitive load**: too much context required before making a change
3. **Unknown unknowns**: unclear what else needs to change

Design principles:
- Interfaces should be simple; modules should be "deep"
- Push complexity down to lower layers as much as possible
- Design it twice and compare alternatives

### 3) Google: Code Health First

Recommended review order:

`Design -> Functionality -> Complexity -> Tests -> Naming -> Comments -> Style -> Docs`

Practical focus:
- Small, self-contained changes are easier to review at high quality
- Changes that improve overall code health should not be blocked by pursuit of perfection

---

## Warning Signal Checklist

### Structural Warnings

| Signal | Threshold | Severity |
|---|---|---|
| Nesting depth | > 3 levels | CRITICAL |
| Function length | > 100 lines | CRITICAL |
| Local variable count | > 10 | WARNING |
| Resource cleanup paths | Multiple exits with scattered cleanup | WARNING |

### Error Handling Warnings

| Signal | Description | Severity |
|---|---|---|
| Defensive default value overuse | e.g., `?? 0` / `|| default` everywhere | WARNING |
| Exception handling overshadows main logic | try/catch outnumbers business code | CRITICAL |
| Fallback masks upstream issues | Problems never surface | WARNING |

### Module & Interface Warnings

| Signal | Description | Severity |
|---|---|---|
| Shallow module | Interface complexity rivals implementation complexity | CRITICAL |
| Information leakage | Internal decisions exposed externally | CRITICAL |
| Naming difficulty | Hard to name, hard to explain | WARNING |

---

## Classic Quotes (Original)

### Linus Torvalds

- "Bad programmers worry about the code. Good programmers worry about data structures."
- "If you need more than 3 levels of indentation, you're screwed anyway."
- "Sometimes you can see a problem in a different way and rewrite it so that the special case goes away."

### John Ousterhout

- "Shallow modules don't help much in the battle against complexity."
- "Design it twice. You'll end up with a much better result."

### Google Code Review

- "A CL that improves the overall code health of the system should not be delayed for perfection."

---

## Classic Examples

### 1) Linked List Deletion: Eliminating Special Branches

**Bad taste (special branch for head node)**:

```c
void remove_list_entry(List *list, Entry *entry) {
    Entry *prev = NULL;
    Entry *walk = list->head;
    while (walk != entry) {
        prev = walk;
        walk = walk->next;
    }
    if (prev == NULL) {
        list->head = entry->next;
    } else {
        prev->next = entry->next;
    }
}
```

**Good taste (unified path)**:

```c
void remove_list_entry(List *list, Entry *entry) {
    Entry **indirect = &list->head;
    while (*indirect != entry)
        indirect = &(*indirect)->next;
    *indirect = entry->next;
}
```

Key point: With an indirect pointer, "deleting the head node" and "deleting a middle node" become the same operation.

### 2) Defensive Defaults vs. Fail Fast

**Not recommended**:

```typescript
function processUser(user: User | null) {
    const name = user?.name ?? "Unknown";
    const email = user?.email ?? "";
    sendEmail(email, `Hello ${name}`);
}
```

**Recommended**:

```typescript
function processUser(user: User) {
    sendEmail(user.email, `Hello ${user.name}`);
}
```

Key point: Do not swallow upstream errors; let the type system and tests surface problems early.

---

## Review Execution Recommendations

- Look at design and complexity first, then style details
- Catch regression-causing issues first, then handle optional optimizations
- Tie all review comments to verifiable evidence (logs, tests, behavior)
- Promptly write settled conclusions into `decision` or `maxim`
