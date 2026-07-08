# Split a mechanical change: prepare, move, postpare

A "move a method/function from one place to another" change is really **two
operations with different correctness criteria**:

| Operation | What it does | How you check it |
|---|---|---|
| **Semantic reshape** | method → free function or method; `self.X` → a parameter, or `self` retyped to the target class; signature / typing change | behavior unchanged: lint + tests pass |
| **Physical move** | cut from the source, paste into the target, fix imports | the moved body is byte-identical, line for line, and the only other changes are move artifacts (imports, a dropped `@staticmethod`, requalified call sites) |

Put both in one commit and the two criteria contaminate each other: a single hunk then
contains the reshape **and** an indentation shift **and** a cross-file relocation, so
neither a human nor a tool can mechanically confirm "the body that landed is the body
that left" — you have to re-read the logic to be sure.

**Rule:** a behaviour-preserving relocation is up to **three commits**, in this order:

- an **(optional) prepare** commit — a **minimal** in-place reshape that the relocation needs
  (de-self a method, retype `self`). A human reviews it, so it must be small and contain **no
  cross-file def relocation and no body relocation**: the code stays where it is.
- a **move** commit — the pure relocation, certified by the reproduce proof
  (`guide-construct-proof.md`; the property it certifies is `spec-reproduction-utils.md`).
  This carries the **bulk**.
- an **(optional) postpare** commit — a **minimal** tail fixup the relocation cannot do
  mechanically: a module path inside a **string literal**, a doc reference. A human reviews it.

Both ends are optional, minimal, and covered by tests; neither prepare nor postpare ever
relocates a def across files or moves a body. The move-artifact whitelist is exactly what a
relocation *forces*; it is **not** a licence to fold reshape work into the move. ("prep"
below is the prepare phase.)

**A large semantic refactor is not one of these phases.** Consolidating bookkeeping,
deduplicating logic, restructuring control flow, or redesigning an API is its **own commit**,
reviewed for **equivalence** (tests or a written argument) — never folded into a prepare under
the "small reshape" label. Prepare is for the *minimal* reshape a relocation forces.

**Prep is human-reviewed, so it stays small and relocates nothing.** The code keeps its
place — prep only changes its *shape* (de-self a method, retype `self`) so a human can
eyeball the whole diff. The **move** does all the relocating and is machine-certified;
the proof forgives only the artifacts a relocation forces (imports, a dropped
`@staticmethod`, requalifying the moved symbol's call sites). Never fold reshape work
into the move to make the proof pass — if the move's diff has anything outside those
artifacts, the reshape leaked in and belongs back in prep.

The shape of the **prep** commit depends on where the code is going — to a
module-level function (**Case 1**) or onto a class (**Case 2**). The **move** commit is
the same idea in both: a pure relocation whose body is byte-identical.

## Case 1: method → free function (in a module)

### Commit 1 — prep: de-self in place (no relocation)

Reshape the method **in its original file and position** so it no longer depends on
`self`. The body stays exactly where it is — this is a small, human-reviewed diff:

- `self.X` (read) → pass `X` in as a parameter.
- `self.X = v` (write) → `return v`; the caller assigns. (Or pass an explicit mutable
  object.)
- `self.other_method(...)` → prep that method in the same commit, or inject it as a
  `Callable` argument.
- once `self` is gone, mark it `@staticmethod`; the body **does not move**.
- call site: `self.foo(args)` → `TheClass.foo(args)` (class-qualified).

Qualifying the call site reflects the real fact that `foo` no longer needs an instance.
The decorator and this qualifier are the only relocation artifacts the next commit will
carry, and the whitelist (`spec-reproduction-utils.md` §1.1) forgives exactly those — so prep can stay this small.

**Check:** lint + tests pass; the diff is just the body reshape plus the call-site
qualifier — and nothing has moved.

### Commit 2 — move: relocate to the module

Cut the `@staticmethod` block, paste it into the target module, and do only the minimal
sealing:

- drop `@staticmethod`, dedent to module level; the body is **unchanged, line for line**.
- source file: add the import of the moved symbol, and remove any now-unused imports.
- call site: `TheClass.foo(args)` → `foo(args)` (qualifier removed; args untouched).

**Check:** the body is byte-identical and the only other changes are move artifacts —
the dropped decorator, the import, and the requalified call site — so the reproduce proof
(`mechanical_refactor_proof_generator.py <commit>`) reports `PASS`. Cross-check with
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

**Check:** the body is byte-identical, and the dropped `@staticmethod` and the
`def foo(self: Target)` → `def foo(self)` annotation drop are move artifacts. The caller
changes from `Source.foo(self.component, ...)` to `self.component.foo(...)` — the receiver
moves out of the argument list — which the reproduce proof replays with its `lower_call_sites`
primitive, so the whole commit reports `PASS` (`mechanical_refactor_proof_generator.py
<commit>`). The split still pays off: because prep left the body untouched, the move is a
clean cut/paste.

## Extracting to a new module: the move gathers scattered defs under an authored header

When the destination module does not exist yet, the **move gathers the defs straight from
wherever they sit** in the source — no prep is needed to stage them at the tail first. The
reproduce proof replays this with `extract_symbols_to_new_module`: each def/class is cut from
the source **verbatim** (the byte diff certifies the body), and the new file is assembled under
an **authored header** — the module-level imports, a `logger`, platform constants, an
`if TYPE_CHECKING:` block — reproduced from the target. The header is small authored
boilerplate, the same harmless category as an import; the defs are the proven relocation. A
module-level constant that moved into the header (e.g. `_is_hip = is_hip()`) is dropped from
the source too. So a pure new-module extraction is **one move commit, no prep** (the proof
reports `PASS`). The only thing that lands outside it is a non-mechanical reference the move
cannot derive — e.g. a module path inside a string literal — which is a one-line **postpare**.

If a symbol is **not top-level** in the source (a method still inside a class), prepare must
de-self it out first (Case 1); the proof reports `UNSUPPORTED` until then.

## Extract-function: the bulk goes in the move

Turning an inline block into a new function is an extraction, so its **bulk — the relocated
body — belongs in a certified move**, not buried in a prep. The `extract_function` primitive
cuts the inline block **verbatim** into the new def (the byte diff certifies the body) and
authors only the small interface: the `def` signature, an optional `return`, and the `call`
that replaces the block.

This is faithful **only when the body moves unchanged**. If the extraction also de-selfs
(`self.x` → a parameter), restructures control flow (an `if/elif/else` chain becoming early
`return`s), or folds in a bookkeeping change, those are **semantic** and must be a separate
commit reviewed for equivalence **first** — then the move relocates the now-unchanged body. An
extraction that rewrites the body *as* it extracts (the two entangled) is a semantic commit,
not a certifiable move; do not dress it up as one.

## A move never renames

The moved symbol keeps the **same name on both sides**. A rename — even a privacy flip
(`_foo` → `foo`) — is a separate single-purpose commit *before* the move (rename in place,
update call sites), so the move stays a same-named relocation. A move commit that also
renames cannot be machine-certified and must be split: rename first, then move.

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

## Naming

A split relocation uses consecutive commits with reserved suffixes:

```
<id>-prepare: <subject>    # optional: minimal in-place reshape (de-self, or retype-self)
<id>-move: <subject>       # pure relocation, certified by the reproduce proof
<id>-postpare: <subject>   # optional: minimal tail fixup (e.g. a string-literal path)
```

The `<phase>:` form is what the range command's `--match -move:` regex keys on.

Both ends are optional and minimal. A large semantic refactor is a separate commit, not one of
these phases. Use a short kebab identifier for `<id>`.
