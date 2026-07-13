# Split a mechanical refactor

- Two levels of splitting, one chapter each: §1 splits the **PR/branch** into small
  classified pieces (the chain contract); §2 splits **one piece** into prepare + move +
  postpare so its move is provable.

## 1. Split the PR into small verifiable pieces

### 1.1 The chain contract — what a compliant branch satisfies

When asked to make (or fix) a refactor branch so it "satisfies this skill", ALL of the
following must hold over `base..branch`; run the chain verifier
(`guide-verify-proof.md` §1) to check the machine-checkable part in one command.

1. **Every commit is classified, in the required subject format:**

   ```text
   <group-id>(<commit-id>,<kind>): <message>
   ```

   with `<kind>` exactly `mechanical_provable` or `non_mechanical_provable`, and
   `<group-id>` / `<commit-id>` kebab-case (contiguous same-`<group-id>` commits form one
   future PR). The verifier machine-checks the standalone-word rule
   (`spec-reproduction-cli.md` §2.1); the full format is required on top of it so the
   chain can be grouped into PRs.
2. **Classification is correct — mechanical work is labeled mechanical.** Every operation
   expressible as the whitelisted relocations (an extract-function, a bulk move, a file
   split, an import repoint, …) is its own `mechanical_provable` commit. Hiding provable
   content inside a `non_mechanical_provable` commit — dodging the verifier — is
   forbidden (§2.2 maximality); catching it is the reviewer's duty
   (`guide-verify-proof.md` §1). How to split so this holds: §2.
3. **Every `mechanical_provable` commit has a proof that PASSes.** Produce the proofs
   with the generator (`guide-construct-proof.md` §1); the chain verifier re-runs every
   one of them against the proof folder.
4. **Every `non_mechanical_provable` commit is correctness-reviewed by eyes.** Its diff
   must be confirmed to do exactly what its message claims: no lost logic, no hidden bug,
   no unintended behavior change riding along (`guide-verify-proof.md` §1). When the
   commit claims to be behavior-preserving that means checking equivalence — but a chain
   need not be a pure refactor, and a commit that intentionally changes behavior is
   reviewed for the correctness of that change instead. The machine never certifies
   these — that is exactly why they must stay minimal (item 2).

### 1.2 Commit naming and classification

- The subject format is exactly §1.1 item 1 — no reserved phase suffixes are required.
  The `<commit-id>` is free (naming it after the phase, e.g. `foo-prepare` / `foo-move`,
  is fine but optional).
- The phases map onto the classification word directly: **move** commits declare
  `mechanical_provable`; **prepare**, **postpare**, and standalone semantic commits
  declare `non_mechanical_provable`.
- The generator's range command selects the provable commits by the word itself:
  `--match '(?<!_)mechanical_provable'` (the lookbehind keeps `non_mechanical_provable`
  from matching).

## 2. Split one piece into prepare + move + postpare

### 2.1 Why split

- A "move a method/function" change is really **two operations with different
  correctness criteria**:

| Operation | What it does | How you check it |
|---|---|---|
| **Semantic reshape** | method → free function or method; `self.X` → a parameter, or `self` retyped to the target class; signature / typing change | behavior unchanged: lint + tests pass |
| **Physical move** | cut from the source, paste into the target, fix imports | the moved body is byte-identical, line for line; the only other changes are move artifacts |

- Put both in one commit and the criteria contaminate each other:
    - one hunk then holds the reshape **and** an indentation shift **and** a cross-file
      relocation;
    - neither a human nor a tool can mechanically confirm "the body that landed is the
      body that left" — you must re-read the logic.

### 2.2 The rule — up to three commits, in this order

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
- **Provable content never hides in a non-provable commit.** The dual of the previous
  two rules: a commit declared `non_mechanical_provable` must be the **minimal residue**
  the relocation primitives cannot express. Any part reproducible as whitelisted
  relocations (`spec-reproduction-utils.md` §2.1) — a def moved across files, a scattered
  extract, an import repoint riding along — is split into its own `mechanical_provable`
  commit with a proof, never folded into a semantic commit where the verifier cannot see
  it. Declaring provable work non-provable to dodge the verifier violates the chain
  property (`spec-reproduction-cli.md` §2.1); the reviewer is instructed to hunt for
  exactly this (`guide-verify-proof.md` §1).
- **"Semantic" is not banned from prepare — oversized or hidden semantics are.** prepare's
  own edits *are* meaning-carrying (de-self, retype-`self`, co-locating bookkeeping);
  "minimal" caps their **size**, it does not forbid semantics. The two bans are narrower:
  (1) no semantic change inside the **move** commit — the move is a pure relocation; and
  (2) don't pass a **large** reshape off as a trivial "small reshape" to dodge the
  equivalence review. A large but honestly-labeled, equivalence-reviewed reshape placed
  *before* the move is legitimate — that is exactly what "its own commit" means, and it
  may serve as the prepare.

- The prep's shape depends on the destination: a module-level function (§2.3) or a class
  (§2.4).
- The move is the same idea in both: a pure relocation, body byte-identical.

### 2.3 Case 1: method → free function

#### 2.3.1 Commit 1 — prep: de-self in place (no relocation)

Reshape the method **in its original file and position** so it no longer needs `self`.
The body stays put:

- `self.X` (read) → pass `X` in as a parameter.
- `self.X = v` (write) → `return v`; the caller assigns. (Or pass an explicit mutable
  object.)
- `self.other_method(...)` → prep that method in the same commit, or inject it as a
  `Callable` argument.
- Once `self` is gone → mark `@staticmethod`; the body **does not move**.
- Call site: `self.foo(args)` → `TheClass.foo(args)`.
- **Seed the destination's module-level scaffolding here too**, if the target module
  lacks what the moved body needs — a `logger = logging.getLogger(__name__)`, a
  module-level constant the body reads (`_is_hip = is_hip()`), and the `import` each
  requires. This is destination groundwork (like a class skeleton, §2.7.4), **not** the
  body: adding it in prep keeps the move a pure cut+paste. Folding it into the move
  instead bundles a non-relocation edit and breaks the byte proof (the move would both
  paste the body **and** author a new `logger`, which the whitelist does not forgive).
  A move into a **new** module is the exception — there the whole header, logger
  included, is authored in the move itself (§2.5).

- The decorator and the qualifier are the only artifacts the move will carry — exactly
  what the whitelist (`spec-reproduction-utils.md` §2.1) forgives.

**Check:** lint + tests pass; the diff is the body reshape, the call-site qualifier, and
any destination scaffolding seeded above; nothing moved.

#### 2.3.2 Commit 2 — move: relocate to the module

- Cut the `@staticmethod` block; paste into the target module.
- Drop `@staticmethod`, dedent to module level — body **unchanged, line for line**.
- Source file: import the moved symbol; drop now-unused imports.
- Call site: `TheClass.foo(args)` → `foo(args)` (args untouched).

**Check:** `mechanical_refactor_proof_generator.py <commit>` reports `PASS`. Cross-check:
`git show <commit> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change`
marks the whole block as moved.

### 2.4 Case 2: method → method on a class

- For pulling **several methods and the fields they touch** into a new (or existing)
  class.
- Prep does **not** de-self — it builds the class and retypes `self`, body untouched.

#### 2.4.1 Commit 1 — prep: build the class, retype `self`

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

Why keep the name `self`:

- it is an ordinary parameter name, so every `self.X` resolves against the target class
  statically and at runtime (the argument *is* a target-class instance);
- renaming it would rewrite every `self.X` and destroy the "body unchanged across both
  commits" invariant.

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

#### 2.4.2 Commit 2 — move: relocate into the class

- Cut `foo` into the target class; drop `@staticmethod` — body **unchanged, line for
  line**.
- Header: `def foo(self: Target)` → `def foo(self)` (type redundant inside the class).
- Caller: `Source.foo(self.component, ...)` → `self.component.foo(...)` — the receiver
  moves out of the argument list (replayed by `lower_call_sites`).

**Check:** `mechanical_refactor_proof_generator.py <commit>` reports `PASS`. The split
paid off: prep left the body untouched, so the move is a clean cut/paste.

### 2.5 Case 3: extract to a new module — one move commit, no prep

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
  it out first (§2.3); the proof reports `UNSUPPORTED` until then.

### 2.6 Case 4: extract-function — the bulk goes in the move

- The relocated body belongs in a certified move, not buried in a prep: the
  `extract_function` primitive cuts the inline block **verbatim** and authors only the
  interface (signature, optional `return`, the replacing `call`).
- Faithful **only when the body moves unchanged.** De-self, control-flow restructure, or a
  bookkeeping change folded in → do that as a separate semantic commit (reviewed for
  equivalence) **first**, then move the now-unchanged body.
- An extraction that rewrites the body *as* it extracts is a semantic commit, not a
  certifiable move — do not dress it up as one.

### 2.7 Remarks

#### 2.7.1 A move never renames

- The moved symbol keeps the **same name on both sides**.
- A rename — even a privacy flip `_foo` → `foo` — is its own single-purpose commit
  *before* the move (rename in place, update call sites).
- A move that also renames cannot be machine-certified: split it — rename first, then
  move.

#### 2.7.2 Anti-pattern: prep adds the body, move deletes it

- Symptom: prep **adds** a large block to the target; the move **deletes** the same block
  from the source. The order is reversed.
- Correct order: prep leaves the body in the source (target skeleton, header retype,
  caller qualification only); the move does the cut/paste.
- The body appears and disappears exactly once — on the move side. Fix by pushing the
  "add the body" work out of prep into the move.

#### 2.7.3 Anti-pattern: the giant prep (relocating inside the source to stage the move)

- Symptom: the prep's diff is **hundreds of lines** for a single function — because it
  also moved the function to the source file's tail, rewrote it as a free function next
  to a staged import/constant block, or reordered its neighbors so the move can cut one
  contiguous block.
- All of that staging is unnecessary: `extract_symbols_to_new_module` gathers symbols
  **from wherever they sit** (§2.5) — the move needs no contiguity and no tail parking.
- A prep's legitimate diff is the handful of lines the primitives cannot derive: the
  `@staticmethod` decorator, the kwargs signature, `self.x` → parameter reads, an added
  `return`, the class-qualified call site. For one function that is **tens of lines, not
  hundreds** — a prep in the hundreds is the signal the relocation leaked into it.
- Why it matters: every relocated-but-not-certified line in a prep is a line the machine
  never checks and a reviewer must eyeball; parking blocks mid-file also leaves broken or
  duplicated intermediate states (a staged import for a module that does not exist yet).
- Fix: strip the prep back to the interface edits above, leave the body **in place**,
  and let the certified move do all relocation.

#### 2.7.4 When NOT to split (single commit)

- Moving an **already** module-level free function.
- Pure file rename / whole-file move.
- Trivial field deletion, or `getattr(obj, "x", ...)` → direct attribute access.
- A class-internal helper relocated next to another helper in the same module.

#### 2.7.5 Which actions are mechanical vs not

- Boundary: building the component correctly the first time is mechanical; reshaping it
  *after* it exists is not.

| Action | Bucket |
|---|---|
| target class skeleton + ctor + fields | mechanical (prep) |
| destination module scaffolding (a `logger`, a module-level constant) the moved symbol needs, when moving into an **existing** module | mechanical (prep) |
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

- The smaller the prep, the easier "behavior unchanged" is to confirm.
- Many small, independently reviewable commits beat one big prep mixing ten flavors of
  change.
- Review order = commit order: prep → move → non-mechanical follow-ups.

#### 2.7.6 Anti-pattern: the non-mechanical label as an escape hatch

- Symptom: a commit whose body is a **pure relocation** (a cut+paste move, a module-level
  constant move, a verbatim inline-block extract) is labelled `non_mechanical_provable` and
  ships with no proof — because the generator reported `UNSUPPORTED` or a primitive could
  not express the exact insertion point, so the author reached for the softer label instead
  of a proof.
- Real example from this repo's history: `kvc-move-mamba-ratio-constants` relocated
  module-level constants unchanged into `kv_cache_configurator.py` but was labelled
  `non_mechanical_provable`, because the constants had to land *above* an
  `if TYPE_CHECKING:` guard and `move_symbol`/`move_assign` only anchored with `before=`,
  which overshot past the guard. The relocation was fully mechanical; only the tool's
  insertion-anchor was missing.
- The rule, in order:
    1. A pure relocation **must** be `mechanical_provable` and carry a proof. The label is
       a claim about the change, not about how easy the tooling made it.
    2. Generator says `UNSUPPORTED` but the change *is* a relocation → **hand-write the
       `Repro`** from the same primitives (guide-construct-proof.md §2.3). Inference falling
       short is not a licence to drop the proof.
    3. A primitive genuinely cannot express the faithful edit (the missing `after=` anchor
       above) → **enhance the primitive first**, then prove it. The fix for a tooling gap is
       to close the gap, not to relabel the commit as unprovable.
- Only a change that is genuinely *not* a relocation (a signature redesign, a logic rewrite,
  a de-self restructure) earns `non_mechanical_provable`. If you cannot say which
  non-relocation edit justifies the label, the label is wrong.
