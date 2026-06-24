# Splitting a mechanical change into prep + move

A "move a method/function from one place to another" change is really **two
operations with different correctness criteria**:

| Operation | What it does | How you check it |
|---|---|---|
| **Semantic reshape** | method → free function or method; `self.X` → a parameter, or `self` retyped to the target class; signature / typing change | behavior unchanged: lint + tests pass |
| **Physical move** | cut from the source, paste into the target, fix imports | the moved body is byte-identical, line for line, and the only other change is an import |

Put both in one commit and the two criteria contaminate each other: a single hunk then
contains the reshape **and** an indentation shift **and** a cross-file relocation, so
neither a human nor a tool can mechanically confirm "the body that landed is the body
that left" — you have to re-read the logic to be sure.

**Rule:** split each mechanical move into two consecutive commits, each carrying one
kind of operation with its own check. The **move** commit is then certifiable by
`verify_move_commit` (see `SKILL.md`, verify mode); the **prep** commit is small and
covered by tests.

The shape of the **prep** commit depends on where the code is going — to a
module-level function (**Case 1**) or onto a class (**Case 2**). The **move** commit is
the same idea in both: a pure relocation whose body is byte-identical.

## Case 1: method → free function (in a module)

### Commit 1 — prep: de-self into a module-level free function, in place

Reshape the method into a **module-level free function in the same file**, so it no
longer depends on `self` and the caller refers to it by a bare name:

- `self.X` (read) → pass `X` in as a parameter.
- `self.X = v` (write) → `return v`; the caller assigns. (Or pass an explicit mutable
  object.)
- `self.other_method(...)` → prep that method in the same commit, or inject it as a
  `Callable` argument.
- define `def foo(...)` at **module level** in the source file (not a `@staticmethod`
  on the class) and remove the method from the class.
- call site: `self.foo(args)` → `foo(args)` (bare name; `foo` is now module-level in the
  same file).

Making it a free function — rather than a `@staticmethod` left on the class — is what
lets the next commit be certified. The caller already uses the bare name `foo(...)`,
which does **not** change when `foo` later moves to another module; only the import
does. A `@staticmethod` would instead force the move to also rewrite `TheClass.foo` →
`foo` and drop the decorator, and those are non-import changes the move verifier
rejects.

**Check:** lint + tests pass; the diff is the body reshape plus the bare-name call site.

### Commit 2 — move: relocate to the module

Cut the free function, paste it into the target module, and do only the minimal sealing:

- move `def foo(...)` to the target module; the body is **unchanged, line for line**.
- source file: add `from target import foo`, and remove any now-unused imports.
- the call site `foo(args)` is **untouched** — only the import changed.

**Check:** the only non-relocated change is the import, so `verify_move_commit <commit>`
reports `CLEAN MOVE`; cross-check with
`git show <commit> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change`,
which marks the whole block as moved.

## Case 2: method → method on a class

Use when the goal is to pull **several methods and the fields they read/write** out
into a new (or existing) class. Here prep does **not** convert every `self.X` into a
parameter — instead it builds the class and switches the **type** of `self`, leaving
the body untouched.

### Commit 1 — prep: build the class, retype `self`

1. Create the target class with the fields the moved methods touch (a frozen dataclass
   is simplest; drop `frozen` only if the methods mutate fields).
2. Wire an instance into the call path — by composition (`self.component = Target(...)`
   in the source ctor), by constructing it at the call site, or by temporarily holding
   both.
3. Retype each moved method as a `@staticmethod` whose parameter is still named `self`
   but **typed** as the target class — body unchanged:

   ```python
   class Source:
       component: Target

       @staticmethod
       def foo(self: Target) -> None:
           ...  # body still reads self.field_a / self.field_b
   ```

4. caller: `self.foo(...)` → `Source.foo(self.component, ...)`.

The trick is keeping the parameter named `self` and changing only its type. `self` is
an ordinary parameter name in Python, so every `self.X` in the body resolves against
the target class both statically and at runtime (the argument *is* a target-class
instance). Renaming the parameter would force rewriting every `self.X` and destroy the
"body unchanged across both commits" invariant.

**Prep stays minimal.** Do only what is needed to relocate the methods onto the class.
Fancier reshapes — signature redesign, extracting helpers, parameter objects,
mutate→return, method renames, splitting a method, dead-branch removal — belong in
**later, non-mechanical follow-up commits**, never in prep.

**Runtime-mutable state → inject a `Callable` getter (still in prep).** If a moved
method reads state on the source object that changes every step (counters, the current
batch, running stats), inject a `Callable[[], T]` getter into the target ctor and
rewrite the body `self.X` → `self.get_X()`. Do **not** thread it as a per-call keyword
argument, and do **not** reach back into the source object.

```python
class Target:
    def __init__(self, *, static_field, get_running_state: "Callable[[], State]"):
        self.static_field = static_field
        self.get_running_state = get_running_state

    @staticmethod
    def check(self: "Target") -> None:
        running = self.get_running_state()   # was self.running_state
        ...
```

```python
# source ctor
self.component = Target(
    static_field=...,
    get_running_state=lambda: self.running_state,
)
```

Per-call keyword arguments are rejected because they make every call site noisy, make
the component API non-self-contained, and force the caller to remember to thread the
state.

**Check:** lint + tests pass; the body is unchanged; types check (`self: Target`
matches the instance the caller passes).

### Commit 2 — move: relocate into the class

Cut `foo` into the target class and drop `@staticmethod` (it becomes a normal instance
method); the body is **unchanged, line for line**:

- header `def foo(self: Target)` → `def foo(self)` (inside the class the type can be
  omitted).
- caller: `Source.foo(self.component, ...)` → `self.component.foo(...)`.

**Check:** the body is byte-identical, but this move also drops `@staticmethod` and
changes the caller's qualifier (`Source.foo` → `self.component.foo`) — both non-import
changes — so `verify_move_commit` reports `NEEDS REVIEW` and lists exactly those lines.
A class-to-class method move cannot be import-residual-only, so it is not certified by
the move verifier; verify it with reproduce mode (a transform script, see
`reproduce-mode.md`) or read those few lines by hand. The split still pays off: because
prep left the body untouched, the move is a clean cut/paste and the only lines to read
are the decorator and the caller.

## Anti-pattern: prep adds the body, move deletes it

If the prep commit **adds** a large block to the target file and the move commit
**deletes** the same block from the source, you have reversed the order. In the correct
order, prep leaves the body in the source (it only builds the target skeleton, retypes
the header, and qualifies the caller); the move does the cut/paste. The body should
appear and disappear exactly once — on the move side. Fix a reversed pair by pushing
the "add the body" work out of prep and into the move as a cut/paste.

## When NOT to split (single commit)

- Moving an **already** module-level free function → single move-only commit.
- Pure file rename / whole-file move → single commit.
- Trivial field deletion, or `getattr(obj, "x", ...)` → direct attribute access → single commit.
- A class-internal helper relocated next to another helper in the same module → single commit.

## Which actions are mechanical vs not

Boundary: everything needed to build the component correctly the first time is
mechanical; reshaping it *after* it exists is not.

| Action | Bucket |
|---|---|
| target class skeleton + ctor + fields | mechanical (prep) |
| `@dataclass(frozen=True, slots=True, kw_only=True)` decoration | mechanical (prep) |
| composition wiring (`self.component = Target(...)`) | mechanical (prep) |
| `Callable` getter injection for runtime-mutable state | mechanical (prep) |
| platform conditionals carried along with the body | mechanical (prep / move) |
| cross-file import path rewrites | mechanical (move) |
| field-ownership migration into the component ctor | mechanical (a single pre-step) |
| inlining an `init_*` method body into a ctor | mechanical (a single pre-step) |
| privacy flip (`_x` ↔ `x`) | mechanical (a single rename) |
| signature redesign (new kwargs, changed defaults, positional → kw-only) | **not** mechanical |
| body simplification / dead-branch removal / logic rewrite | **not** mechanical |
| semantic method rename | **not** mechanical |

The smaller prep is, the easier "behavior unchanged" is to confirm. More commits, each
small and independently reviewable, beats one big prep that mixes ten flavors of
semantic change. Review order follows commit order: prep → move → then any
non-mechanical reshapes as separate follow-up commits.

## Long constructor calls → `init_X` wrapper methods

See `.claude/skills/large-class-style/SKILL.md`.

## Naming

A move that is split uses two consecutive commits with reserved suffixes:

```
<id>-prep    # commit 1: in-place reshape (Case 1 → free function, or Case 2 retype-self)
<id>-move    # commit 2: pure relocation
```

Use a short kebab identifier for `<id>`.
