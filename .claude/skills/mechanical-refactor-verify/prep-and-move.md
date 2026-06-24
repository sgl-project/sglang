# Splitting a mechanical change into prep + move

A "move a method/function from one place to another" change is really **two
operations with different correctness criteria**:

| Operation | What it does | How you check it |
|---|---|---|
| **Semantic reshape** | method → free function, `self.X` → parameter, return-vs-mutate, signature change | behavior unchanged: lint + tests pass |
| **Physical move** | cut from the source, paste into the target, fix imports + call sites | the moved body is byte-identical, line for line |

Put both in one commit and the two criteria contaminate each other: a single hunk
then contains `self.X → X` **and** an indentation shift **and** a cross-file
relocation, so neither a human nor a tool can mechanically confirm "the body that
landed is the body that left" — you have to re-read the logic to be sure.

**Rule:** split each mechanical move into two consecutive commits, each carrying one
kind of operation with its own check. The **move** commit is then certifiable by
`verify_move_commit` (see `SKILL.md`, verify mode); the **prep** commit is small and
covered by tests.

## Commit 1 — prep (reshape in place, detach from `self`)

Reshape the function in its **original file and position** so it no longer depends on
`self`, and qualify the call site:

- `self.X` (read) → pass `X` in as a parameter.
- `self.X = v` (write) → `return v`; the caller assigns. (Or pass an explicit mutable
  object.)
- `self.other_method(...)` → prep that method in the same commit, or inject it as a
  `Callable` argument.
- once `self` is gone, mark it `@staticmethod`; the body **stays where it is**.
- call site: `self.foo(args)` → `TheClass.foo(args)` (class-qualified).

Qualifying the call site now reflects the real fact that `foo` no longer needs an
instance (rather than relying on "a staticmethod happens to be callable on an
instance"), and it makes the move commit's call-site change a pure prefix deletion
(`TheClass.foo` → `foo`) that a tool can verify.

**Check:** lint + tests pass; the diff is limited to the body reshape plus the
call-site prefix rewrite.

## Commit 2 — move (pure relocation)

Cut the `@staticmethod` block, paste it into the target module, and do only the
minimal sealing:

- drop `@staticmethod`, dedent to module level; the body is **unchanged, line for
  line**.
- source file: remove the now-unused imports.
- target file: add the import of the moved symbol.
- call site: `TheClass.foo(args)` → `foo(args)` (prefix deletion; args untouched).

**Check:** the moved hunk is byte-identical.
`git show <commit> --color-moved=dimmed-zebra --color-moved-ws=allow-indentation-change`
marks the whole block as moved, and `verify_move_commit <commit>` certifies it (only
imports / call sites remain as residual).

### Anti-pattern: prep adds the body, move deletes it

If the prep commit **adds** a large block to the target file and the move commit
**deletes** the same block from the source, you have reversed the order. In the
correct order, prep leaves the body in the source (it only retypes the header, adds
`@staticmethod`, and qualifies the caller); the move does the cut/paste. The body
should appear and disappear exactly once — on the move side. Fix a reversed pair by
pushing the "add the body" work out of prep and into the move as a cut/paste.

## When NOT to split (single commit)

- Moving an **already** module-level free function → single move-only commit.
- Pure file rename / whole-file move → single commit.
- Trivial field deletion, or `getattr(obj, "x", ...)` → direct attribute access → single commit.
- A class-internal helper relocated next to another helper in the same module → single commit.

## Variant: extracting methods + their fields into a new class

When the goal is to pull several methods **and the fields they read/write** out into a
new class, prep does not have to convert every `self.X` into a parameter. Instead,
build the target class and switch the **type** of `self` from the source class to the
target class, leaving the body untouched.

Prep:

1. Create the target class with the fields the moved methods touch (a frozen dataclass
   is simplest; drop `frozen` only if the methods mutate fields).
2. Wire an instance into the call path — by composition (`self.component =
   Target(...)` in the source ctor), by constructing it at the call site, or by
   temporarily holding both.
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

This is a technique, not a template: prep is the semantic layer, so changing the
signature is legal — `self: Target` is just the support that keeps the body
byte-identical across prep + move when you want that.

**Prep stays minimal.** Do only what is needed to detach from `self`. Fancier
reshapes — signature redesign, extracting helpers, parameter objects, mutate→return,
method renames, splitting a method, dead-branch removal — belong in **later,
non-mechanical follow-up commits**, never in prep.

Move: cut `foo` into the target class, drop `@staticmethod` (it becomes a normal
instance method), body unchanged; caller `Source.foo(self.component, ...)` →
`self.component.foo(...)`.

### Runtime-mutable state: inject a `Callable` getter (in prep)

If a moved method reads state on the source object that changes every step (counters,
the current batch, running stats), prep injects a `Callable[[], T]` getter into the
target ctor and rewrites the body `self.X` → `self.get_X()`. Do **not** thread it as a
per-call keyword argument, and do **not** reach back into the source object.

```python
class Component:
    def __init__(self, *, static_field, get_running_state: "Callable[[], State]"):
        self.static_field = static_field
        self.get_running_state = get_running_state

    def check(self) -> None:
        running = self.get_running_state()   # was self.running_state
        ...
```

```python
# source ctor
self.component = Component(
    static_field=...,
    get_running_state=lambda: self.running_state,
)
```

Per-call keyword arguments are rejected because they make every call site noisy, make
the component API non-self-contained, and force the caller to remember to thread the
state.

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
semantic change. Review order follows commit order: prep (detach only) → move (pure
relocation) → then any non-mechanical reshapes as separate follow-up commits.

## Long constructor calls → `init_X` wrapper methods (single commit)

A multi-line constructor assignment in `__init__` (a kwarg call wrapped over two or
more lines) should be wrapped in an `init_<field>()` method:

```python
def __init__(self, ...):
    ...
    self.init_configurator()

def init_configurator(self):
    self.configurator = Configurator(
        option_a=...,
        option_b=...,
    )
```

This keeps `__init__` thin and readable, makes each init step a named method that can
be reviewed / unit-tested / overridden on its own, and narrows the diff of a later
mechanical change to "edit one method body." It is "extract inline → method," not a
prep + move pair, so it is a **single commit**; several such wrappers can share one
extract commit.

## Naming

A move that is split uses two consecutive commits with reserved suffixes:

```
<id>-prep    # commit 1: in-place de-self
<id>-move    # commit 2: pure relocation
```

Use a short kebab identifier for `<id>`.
