# Split a mechanical change: prepare, move, postpare

## 1. Why split

A "move a method/function" change is really **two operations with different correctness
criteria**:

| Operation | What it does | How you check it |
|---|---|---|
| **Semantic reshape** | method → free function or method; `self.X` → a parameter, or `self` retyped to the target class; signature / typing change | behavior unchanged: lint + tests pass |
| **Physical move** | cut from the source, paste into the target, fix imports | the moved body is byte-identical, line for line; the only other changes are move artifacts |

Put both in one commit and the criteria contaminate each other:

- one hunk then holds the reshape **and** an indentation shift **and** a cross-file
  relocation;
- neither a human nor a tool can mechanically confirm "the body that landed is the body
  that left" — you must re-read the logic.

## 2. The rule — up to three commits, in this order

- **prepare (optional)** — a **minimal** in-place reshape the relocation needs (de-self a
  method, retype `self`). Human-reviewed, so: small, **no cross-file def relocation, no
  body relocation** — the code stays where it is.
- **move** — the pure relocation; carries the **bulk**; certified by the reproduce proof
  (`guide-construct-proof.md`; property: `spec-reproduction-utils.md`).
- **postpare (optional)** — a **minimal** tail fixup the move cannot do mechanically (a
  module path inside a string literal, a doc reference). Human-reviewed.

Hard lines ("prep" below = the prepare phase):

- Both ends are optional, minimal, and covered by tests; neither ever relocates a def
  across files or moves a body.
- The move-artifact whitelist is what a relocation *forces* — **not** a licence to fold
  reshape work into the move. Anything outside the artifacts in the move's diff = the
  reshape leaked; push it back into prep.
- **A large semantic refactor is not a phase.** Consolidating bookkeeping, deduplicating
  logic, restructuring control flow, redesigning an API → its **own commit**, reviewed for
  **equivalence** (tests or a written argument). Never smuggled into prep as a "small
  reshape".

The prep's shape depends on the destination — a module-level function (§3) or a class
(§4). The move is the same idea in both: a pure relocation, body byte-identical.

## 3. Case 1: method → free function

### 3.1 Commit 1 — prep: de-self in place (no relocation)

Reshape the method **in its original file and position** so it no longer needs `self`:

- `self.X` (read) → pass `X` in as a parameter.
- `self.X = v` (write) → `return v`; the caller assigns. (Or pass an explicit mutable
  object.)
- `self.other_method(...)` → prep that method in the same commit, or inject it as a
  `Callable` argument.
- Once `self` is gone → mark `@staticmethod`; the body **does not move**.
- Call site: `self.foo(args)` → `TheClass.foo(args)`.

The decorator and the qualifier are the only artifacts the move will carry — exactly what
the whitelist (`spec-reproduction-utils.md` §2.1) forgives.

**Check:** lint + tests pass; the diff is the body reshape plus the call-site qualifier;
nothing moved.

### 3.2 Commit 2 — move: relocate to the module

- Cut the `@staticmethod` block; paste into the target module.
- Drop `@staticmethod`, dedent to module level — body **unchanged, line for line**.
- Source file: import the moved symbol; drop now-unused imports.
- Call site: `TheClass.foo(args)` → `foo(args)` (args untouched).

**Check:** `mechanical_refactor_proof_generator.py <commit>` reports `PASS`. Cross-check:
`git show <commit> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change`
marks the whole block as moved.

## 4. Case 2: method → method on a class

For pulling **several methods and the fields they touch** into a new (or existing) class.
Prep does **not** de-self — it builds the class and retypes `self`, body untouched.

### 4.1 Commit 1 — prep: build the class, retype `self`

1. Create the target class with the fields the moved methods touch (a frozen dataclass is
   simplest; drop `frozen` only if they mutate).
2. Wire an instance into the call path — composition (`self.component = Target(...)` in
   the source ctor), construction at the call site, or temporarily both.
3. Retype each moved method as a `@staticmethod` whose parameter is still **named** `self`
   but **typed** as the target class — body unchanged:

   ```python
   class Source:
       component: Target

       @staticmethod
       def foo(self: Target) -> None:
           ...  # body still reads self.field_a / self.field_b
   ```

4. Caller: `self.foo(...)` → `Source.foo(self.component, ...)`.

Why keep the name `self`: it is an ordinary parameter name, so every `self.X` resolves
against the target class statically and at runtime (the argument *is* a target-class
instance). Renaming it would rewrite every `self.X` and destroy the "body unchanged across
both commits" invariant.

Boundaries:

- **Prep stays minimal.** Signature redesign, helper extraction, parameter objects,
  mutate→return, renames, method splits, dead-branch removal → later non-mechanical
  commits, never prep.
- **Runtime-mutable state → inject a `Callable` getter (still prep).** State that changes
  every step (counters, the current batch, running stats): inject `Callable[[], T]` into
  the target ctor; rewrite `self.X` → `self.get_X()`. Do **not** thread it per call and do
  **not** reach back into the source object — per-call kwargs make every call site noisy,
  the API non-self-contained, and the threading a caller chore.

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

**Check:** lint + tests pass; body unchanged; types check (`self: Target` matches the
instance the caller passes).

### 4.2 Commit 2 — move: relocate into the class

- Cut `foo` into the target class; drop `@staticmethod` — body **unchanged, line for
  line**.
- Header: `def foo(self: Target)` → `def foo(self)` (type redundant inside the class).
- Caller: `Source.foo(self.component, ...)` → `self.component.foo(...)` — the receiver
  moves out of the argument list (replayed by `lower_call_sites`).

**Check:** `mechanical_refactor_proof_generator.py <commit>` reports `PASS`. The split
paid off: prep left the body untouched, so the move is a clean cut/paste.

## 5. Extracting to a new module: one move commit, no prep

- The move gathers the defs **from wherever they sit** — no prep staging at the source
  tail. Replayed by `extract_symbols_to_new_module`.
- Each def/class is cut **verbatim** (the byte diff certifies the bodies); the new file's
  small header (imports, a logger, constants, a `TYPE_CHECKING` block) is authored from
  the target and audited (`spec-reproduction-utils.md` §2.1).
- A module-level constant that moved into the header (e.g. `_is_hip = is_hip()`) is
  dropped from the source too.
- The only work outside the move: a non-mechanical reference the move cannot derive (a
  string-literal module path) — a one-line **postpare**.
- A symbol **not top-level** in the source (a method still in a class): prepare de-selfs
  it out first (§3); the proof reports `UNSUPPORTED` until then.

## 6. Extract-function: the bulk goes in the move

- The relocated body belongs in a certified move, not buried in a prep: the
  `extract_function` primitive cuts the inline block **verbatim** and authors only the
  interface (signature, optional `return`, the replacing `call`).
- Faithful **only when the body moves unchanged.** De-self, control-flow restructure, or a
  bookkeeping change folded in → do that as a separate semantic commit (reviewed for
  equivalence) **first**, then move the now-unchanged body.
- An extraction that rewrites the body *as* it extracts is a semantic commit, not a
  certifiable move — do not dress it up as one.

## 7. A move never renames

- The moved symbol keeps the **same name on both sides**.
- A rename — even a privacy flip `_foo` → `foo` — is its own single-purpose commit
  *before* the move (rename in place, update call sites).
- A move that also renames cannot be machine-certified: split it — rename first, then
  move.

## 8. Anti-pattern: prep adds the body, move deletes it

- Symptom: prep **adds** a large block to the target; the move **deletes** the same block
  from the source. The order is reversed.
- Correct order: prep leaves the body in the source (target skeleton, header retype,
  caller qualification only); the move does the cut/paste.
- The body appears and disappears exactly once — on the move side. Fix by pushing the
  "add the body" work out of prep into the move.

## 9. When NOT to split (single commit)

- Moving an **already** module-level free function.
- Pure file rename / whole-file move.
- Trivial field deletion, or `getattr(obj, "x", ...)` → direct attribute access.
- A class-internal helper relocated next to another helper in the same module.

## 10. Which actions are mechanical vs not

Boundary: building the component correctly the first time is mechanical; reshaping it
*after* it exists is not.

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

The smaller the prep, the easier "behavior unchanged" is to confirm. Many small,
independently reviewable commits beat one big prep mixing ten flavors of change. Review
order = commit order: prep → move → non-mechanical follow-ups.

## 11. Naming

Consecutive commits with reserved suffixes; short kebab `<id>`:

```
<id>-prepare: <subject>    # optional: minimal in-place reshape (de-self, or retype-self)
<id>-move: <subject>       # pure relocation, certified by the reproduce proof
<id>-postpare: <subject>   # optional: minimal tail fixup (e.g. a string-literal path)
```

The `<phase>:` form is what the range command's `--match -move:` regex keys on.
