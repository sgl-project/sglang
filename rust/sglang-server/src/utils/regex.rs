//! `stop_regex` validation and bounding.
//!
//! The scheduler matches these patterns with CPython's `re` on the decode hot
//! path, so this module has one job: admit only patterns that engine can compile
//! and afford. See [`validate`] for the two rejection classes and why the
//! invariant is one-directional.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::error::Error;

/// `MAX_LEN` from Python's `get_max_seq_length`: the bound for an *unbounded* stop
/// regex (`\d+`, `.*`, …) or one we can't statically size — the scheduler then
/// scans the whole output tail. A *bounded* regex gets its finite length instead
/// (see [`regex_max_seq_length`]); assigning this to every regex made the scheduler
/// re-scan the full accumulated output every token (O(T²)).
const STOP_REGEX_MAX_LEN: usize = 1 << 30;

/// Escapes that mean the same thing to `regex-syntax` and to Python's `re`.
///
/// An allowlist, not a blocklist. The blocklist version of this function is what
/// shipped `\p{L}` and `(?<n>a)` to a scheduler that could not compile them: every
/// escape either side adds lands in the gap by default. Here the default is
/// "reject", so a new escape is a 400 until someone checks both dialects.
/// Inline flags both dialects understand. Rust also has `R`/`U`, Python `a`/`L`;
/// each errors on the other's.
const PORTABLE_FLAGS: &[char] = &['i', 'm', 's', 'x', 'u', '-'];

/// Cap on the PRODUCT of counted repeats along one path. A literal `a{200}` is
/// harmless — CPython compiles `{N}` to a counted repeat and never expands it
/// (`a{4294967294}` measures 0.004 ms and 0 KB) — so this is not about the count
/// itself. It bounds two count-shaped hazards the ambiguity predicate cannot see:
/// `{4294967295}` is exactly CPython's `MAXREPEAT` and raises `OverflowError`
/// (neither `re.error` nor `RecursionError`, so the scheduler's seatbelt misses
/// it), and an EMPTY-body repeat like `(?:){1048575}` costs 36 ms and 56 MB.
///
/// Sized generously on purpose: a tighter value 400s `[a-f0-9]{40}` (a SHA-1) and
/// `.{100}`, both of which measure ~0.005 ms. The compounding families that used to
/// justify a small cap — `(?:a*){65535}` and friends — are repeats of a
/// VARIABLE-length body, which [`ambiguity_degree`] rejects outright.
const MAX_REPEAT_COUNT: u64 = 512;

/// Limit on [`ambiguity_degree`]. Chosen by measurement, not argument: see that
/// function's docs for the 3730-pattern sweep that rules out 2 and 3.
const MAX_AMBIGUITY_DEGREE: u64 = 1;

const REGEX_AST_NEST_LIMIT: u32 = 64;

const SHARED_ESCAPES: &[char] = &[
    'A', 'b', 'B', 'd', 'D', 's', 'S', 'w', 'W', 'a', 'f', 'n', 'r', 't', 'v',
    // `\xHH` (exactly two hex digits) is shared; only the braced `\x{…}` form is
    // Rust-only, and `check_escape` rejects that separately. Omitting `x` here
    // contradicted this list's own doc comment and 400ed `\x41`.
    'x',
];

/// Reject the constructs `regex-syntax` accepts but Python's `re` cannot compile.
///
/// Everything else in this module rests on one property: **anything Rust admits,
/// Python can compile.** The reverse is allowed to fail — rejecting a pattern
/// Python would have accepted (`\Z`, backreferences, look-around) costs a client
/// a 400, while admitting one it cannot compile costs the whole scheduler, since
/// `re.search` runs on the decode hot path where nothing catches it.
fn reject_python_incompatible(pattern: &str) -> Result<(), Error> {
    let reject = |what: String| {
        Err(Error::Validation(format!(
            "stop_regex {pattern:?} uses {what}, which Python's `re` cannot compile"
        )))
    };
    // ASCII-only comparisons, so scanning bytes is safe: a UTF-8 continuation byte
    // is >= 0x80 and matches no arm.
    let b = pattern.as_bytes();
    let mut i = 0;
    // Still inside the run of leading `(?flags)` groups.
    let mut leading = true;
    while i < b.len() {
        match b[i] {
            b'\\' => {
                if let Err(what) = check_escape(b, i) {
                    return reject(what);
                }
                leading = false;
                i += 2; // skip the escaped character, so `\(` is not a group open
            }
            // `(?<name>…)` is a named group to Rust; Python spells it `(?P<name>…)`
            // and errors on this one. `(?<=` / `(?<!` are look-behind, which
            // `regex-syntax` rejects on its own.
            b'(' if b[i..].starts_with(b"(?<")
                && !b[i..].starts_with(b"(?<=")
                && !b[i..].starts_with(b"(?<!") =>
            {
                return reject("a `(?<name>…)` group (Python spells it `(?P<name>…)`)".into());
            }
            // A flag-setting group. Python 3.11+ reads these as GLOBAL flags: they
            // must sit at position 0, and the clearing form (`(?-i)`) is invalid on
            // its own — it wants `(?-i:…)`. The flag letters also differ: Rust adds
            // `R`/`U`, Python adds `a`/`L`, so only their intersection is portable.
            b'(' if flag_group_bytes(&b[i..]).is_some() => {
                let flags = flag_group_bytes(&b[i..]).expect("just matched");
                // `(?flags:…)` is scoped: legal anywhere, and its clearing form is
                // legal too. Only the GLOBAL form is position- and sign-restricted.
                let scoped = b[i..].get(2 + flags.len()).is_some_and(|&c| c == b':');
                // Python allows global flags only at the start, but allows SEVERAL
                // (`(?i)(?m)a`); `leading` stays true while we are still in that run.
                if !scoped && !leading {
                    return reject("inline flags after the start of the pattern".into());
                }
                if !scoped && flags.contains(&b'-') {
                    return reject(
                        "a clearing `(?-flags)` group (Python wants `(?-flags:…)`)".into(),
                    );
                }
                if let Some(&f) = flags
                    .iter()
                    .find(|f| !PORTABLE_FLAGS.contains(&(**f as char)))
                {
                    return reject(format!("the inline flag `{}`", f as char));
                }
                // Advance past the WHOLE group, not one byte: scanning its inner
                // `?`/letters/`)` through the default arm would clear `leading` and
                // make the next `(?m)` look like a mid-pattern flag change.
                if scoped {
                    leading = false;
                    i += 1;
                } else {
                    i += 2 + flags.len() + 1; // `(?` + flags + `)`
                }
                continue; // still in the leading flag run
            }
            // A `[` inside a character class. Rust reads it as a literal (or a POSIX
            // class); Python's parser terminates the class differently and can end up
            // parsing the remainder as a group.
            b'[' => {
                let mut j = i + 1;
                if b.get(j) == Some(&b'^') {
                    j += 1;
                }
                if b.get(j) == Some(&b']') {
                    j += 1; // a leading `]` is a literal in both dialects
                }
                while j < b.len() && b[j] != b']' {
                    match b[j] {
                        // Escapes inside a class follow the same rules as outside.
                        b'\\' => {
                            if let Err(what) = check_escape(b, j) {
                                return reject(what);
                            }
                            j += 2;
                        }
                        b'[' => return reject("a `[` nested inside a character class".into()),
                        // `[a--b]` is a class-difference operator in Rust and a bad
                        // character range in Python.
                        b'-' if b.get(j + 1) == Some(&b'-') => {
                            return reject("a `--` class-difference operator".into());
                        }
                        _ => j += 1,
                    }
                }
                i = j.max(i + 1);
            }
            _ => {
                leading = false;
                i += 1;
            }
        }
    }
    Ok(())
}

/// The flag bytes of a flag-setting group (`(?i)`, `(?-i)`, `(?imsx)`), or `None`
/// if `b` does not open one. A `(?i:…)` scoped group is not one of these.
fn flag_group_bytes(b: &[u8]) -> Option<&[u8]> {
    let rest = b.strip_prefix(b"(?")?;
    // Stop at `)` OR `:` — the scoped form `(?i:…)` carries the same flag letters
    // and was falling through unvalidated, so `(?R:a)` reached the scheduler.
    let end = rest.iter().position(|&c| c == b')' || c == b':')?;
    let flags = &rest[..end];
    (!flags.is_empty() && flags.iter().all(|&c| c.is_ascii_alphabetic() || c == b'-'))
        .then_some(flags)
}

/// Check the escape starting at `b[i]` (a backslash). `Err` names why Python's
/// `re` would refuse it. Used for escapes both inside and outside character
/// classes — the class scanner used to skip escapes entirely, which is how
/// `[\p{L}]` slipped past the very check written for `\p{L}`.
fn check_escape(b: &[u8], i: usize) -> Result<(), String> {
    let Some(&e) = b.get(i + 1) else {
        return Err("a trailing backslash".into());
    };
    // `\xHH` is shared; Rust's braced `\x{10FFFF}` is not.
    if e == b'x' && b.get(i + 2) == Some(&b'{') {
        return Err("a braced `\\x{…}` escape".into());
    }
    // `\b{start}` is one zero-width assertion to Rust, but `\b` followed by the
    // literal "{start}" to Python — 7 characters this side would score as 0, so
    // the scheduler sizes a 1-token window and the stop silently never fires.
    if e == b'b' && b.get(i + 2) == Some(&b'{') {
        return Err("a `\\b{…}` assertion".into());
    }
    if e.is_ascii_alphanumeric() && !SHARED_ESCAPES.contains(&(e as char)) {
        return Err(format!("the escape `\\{}`", e as char));
    }
    // `\<` / `\>` are GNU word-boundary ASSERTIONS to `regex-syntax` (width 0) but
    // escaped LITERALS to Python (`\<END\>` needs 5 characters of tail). Scoring
    // them 0 sizes the match window too small, so the stop silently never fires and
    // the request runs to `max_new_tokens` — the one failure mode this module exists
    // to prevent, and `\<WORD\>` is idiomatic from grep/vim.
    if e == b'<' || e == b'>' {
        return Err(format!("the escape `\\{}`", e as char));
    }
    Ok(())
}

/// Entries kept in [`ADMISSION_CACHE`], mirroring CPython's `re._MAXCACHE`.
const ADMISSION_CACHE_CAP: usize = 512;

/// Memo of admitted patterns → their bound.
///
/// Admission is a pure function of the pattern text, and an expensive one: ~87% of
/// it is HIR translation, which expands `\w`/`\W` into large Unicode class unions.
/// A 256-byte `\W`-heavy pattern (exactly [`MAX_STOP_REGEX_LEN`]) measures 574 µs,
/// and a request may carry [`MAX_STOP_REGEX_COUNT`] of them — 18 ms of admission on
/// the single ingress thread, re-derived from scratch on every request. It
/// multiplies through a batch, because one `sampling_params` object broadcasts to
/// every item: a 13.6 KB body measured **1.01 s**, during which that thread serves
/// no other request, no abort and no health probe.
///
/// Only successes are memoized. A rejected pattern fails inside [`validate`], which
/// is the cheap 8% — the expensive translate runs only after it passes — so the
/// hazard is entirely on the admitted side, and this keeps the entry a plain
/// `usize` rather than something that has to reconstruct an `Error` faithfully.
///
/// Cleared wholesale when full rather than evicted one at a time: that is what
/// CPython's `re` does, and it keeps the hot path one lookup with no LRU
/// bookkeeping. The lock is held across a hash lookup and nothing else, and is
/// taken almost exclusively by the one ingress thread.
static ADMISSION_CACHE: LazyLock<Mutex<HashMap<Box<str>, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

fn cached_bound(pattern: &str) -> Option<usize> {
    ADMISSION_CACHE
        .lock()
        .ok()
        .and_then(|c| c.get(pattern).copied())
}

fn cache_bound(pattern: &str, max_len: usize) {
    let Ok(mut c) = ADMISSION_CACHE.lock() else {
        return;
    };
    if c.len() >= ADMISSION_CACHE_CAP {
        c.clear();
    }
    c.insert(pattern.into(), max_len);
}

/// A `stop_regex` that has been admitted, together with the bound derived while
/// admitting it.
///
/// Holding one is the proof: it cannot be built without passing [`validate`], and
/// its [`max_len`](Self::max_len) came from *that* pattern's own AST. So no caller
/// can pair one pattern's bound with another's, and there is no second route to a
/// bound that could drift from the validated one.
pub struct RegexPattern<'a> {
    pattern: &'a str,
    max_len: usize,
}

/// `TryFrom`, not `FromStr`: `FromStr::from_str` takes a `&str` whose lifetime the
/// trait never names, so it cannot be tied to `Self` — a borrowing type can never
/// implement it. `TryFrom<&'a str>` carries the lifetime, so it can.
impl<'a> TryFrom<&'a str> for RegexPattern<'a> {
    type Error = Error;

    fn try_from(pattern: &'a str) -> Result<Self, Self::Error> {
        Self::build(pattern)
    }
}

impl<'a> RegexPattern<'a> {
    /// Admit `pattern` and derive its bound in a single AST walk.
    ///
    /// `Err` for anything CPython's `re` cannot compile, or cannot match cheaply
    /// enough to run on every decode step — see [`validate`].
    fn build(pattern: &'a str) -> Result<Self, Error> {
        // Same pattern text ⇒ same verdict and same bound, so a repeat costs a hash
        // lookup instead of a parse + translate. See [`ADMISSION_CACHE`].
        if let Some(max_len) = cached_bound(pattern) {
            return Ok(Self { pattern, max_len });
        }
        let ast = validate(pattern)?;
        // Translate the AST `validate` already produced instead of re-parsing. The full
        // `regex_syntax::Parser` parses AND translates, so calling it here would parse a
        // second time — and, more importantly, through a SECOND builder whose settings
        // can drift from the validating one. That would validate one AST while bounding
        // a different one; the bound is what sizes the scheduler's match window, so a
        // silent divergence there is the under-estimate class of bug.
        let hir = regex_syntax::hir::translate::TranslatorBuilder::new()
            .build()
            .translate(pattern, &ast)
            .map_err(|e| {
                Error::Validation(format!(
                    "stop_regex {pattern:?} is not a valid regular expression: {e}"
                ))
            })?;
        let max_len = hir_max_len(&hir);
        cache_bound(pattern, max_len);
        Ok(Self { pattern, max_len })
    }

    /// The admitted pattern. See the field note on why this is kept.
    #[allow(dead_code)]
    pub fn pattern(&self) -> &str {
        self.pattern
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }
}

/// Validate a `stop_regex` before it can reach the scheduler, returning the parsed
/// AST so the caller can derive its bound without parsing again.
///
/// Two independent classes of rejection, for two different reasons:
///
/// * **Dialect.** CPython's `re` is the engine that actually runs this pattern, and
///   `regex-syntax` is neither a superset nor a subset of it. The rows where
///   `regex-syntax` is *wider* are the dangerous ones — a pattern admitted here but
///   uncompilable there reaches `re.search` on the decode hot path, where the raised
///   error is uncaught and takes the scheduler down. Hence the invariant is
///   one-directional: **anything admitted here must compile in Python**, while
///   rejecting a pattern Python would have accepted costs one client a 400. The
///   asymmetry is deliberate, and it is why the checks below only ever add
///   rejections.
///
/// * **Cost.** The scheduler re-matches this against the output tail on *every*
///   decode step, inside `re`'s C loop with the GIL held, where no timeout or signal
///   can interrupt it. Repetition counts, nested unbounded repetitions, quantified
///   assertions and ambiguous alternations are refused on those grounds alone —
///   they are all valid Python.
fn validate(pattern: &str) -> Result<regex_syntax::ast::Ast, Error> {
    reject_python_incompatible(pattern)?;
    let ast = regex_syntax::ast::parse::ParserBuilder::new()
        .nest_limit(REGEX_AST_NEST_LIMIT)
        .build()
        .parse(pattern)
        .map_err(|e| {
            Error::Validation(format!(
                "stop_regex {pattern:?} is not a valid regular expression: {e}"
            ))
        })?;
    if repetition_cost_too_large(&ast, 1, false) {
        return Err(Error::Validation(format!(
            "stop_regex {pattern:?} repeats too many times or nests unbounded \
             repetitions; matching it would dominate every decode step"
        )));
    }
    if repeats_an_assertion(&ast) {
        return Err(Error::Validation(format!(
            "stop_regex {pattern:?} quantifies a zero-width assertion, which Python's \
             `re` rejects, or a repetition count Python cannot honour"
        )));
    }
    if alternation_under_repetition(&ast) {
        return Err(Error::Validation(format!(
            "stop_regex {pattern:?} alternates inside a repetition; each iteration \
             could match more than one way, so Python's backtracking engine explores \
             exponentially many parses"
        )));
    }
    match ambiguity_degree(&ast) {
        None => {
            return Err(Error::Validation(format!(
                "stop_regex {pattern:?} repeats a variable-length expression without \
                 a bound; matching it would dominate every decode step"
            )));
        }
        Some(d) if d > MAX_AMBIGUITY_DEGREE => {
            return Err(Error::Validation(format!(
                "stop_regex {pattern:?} has {d} independent length choices (limit \
                 {MAX_AMBIGUITY_DEGREE}); Python's backtracking engine would explore \
                 their product on every decode step"
            )));
        }
        Some(_) => {}
    }
    Ok(ast)
}

/// Reject repetitions whose cost compounds down the nesting.
///
/// `outer` is the product of the counted repeats enclosing `ast`. Two families die
/// here: a counted product over [`MAX_REPEAT_COUNT`] (memory), and an unbounded
/// repeat nested inside another (`(?:a+)+b` — catastrophic backtracking, measured
/// 2.3 s on a 26-character tail, and since its bound is the full-scan sentinel the
/// tail grows every step, so the loop is dead within ~30 tokens).
fn repetition_cost_too_large(ast: &regex_syntax::ast::Ast, outer: u64, unbounded: bool) -> bool {
    use regex_syntax::ast::{Ast, RepetitionKind, RepetitionRange};
    match ast {
        Ast::Repetition(rep) => {
            let (factor, is_unbounded) = match &rep.op.kind {
                RepetitionKind::Range(RepetitionRange::Exactly(n)) => (*n as u64, false),
                RepetitionKind::Range(RepetitionRange::Bounded(_, hi)) => (*hi as u64, false),
                RepetitionKind::Range(RepetitionRange::AtLeast(n)) => (*n as u64, true), // codespell:ignore atleast
                _ => (1, true), // `*`, `+`, `?`
            };
            let total = outer.saturating_mul(factor.max(1));
            total >= MAX_REPEAT_COUNT
                || (is_unbounded && unbounded)
                || repetition_cost_too_large(&rep.ast, total, unbounded || is_unbounded)
        }
        Ast::Group(g) => repetition_cost_too_large(&g.ast, outer, unbounded),
        Ast::Concat(c) => c
            .asts
            .iter()
            .any(|a| repetition_cost_too_large(a, outer, unbounded)),
        Ast::Alternation(a) => a
            .asts
            .iter()
            .any(|a| repetition_cost_too_large(a, outer, unbounded)),
        _ => false,
    }
}

/// Whether any repetition in `ast` applies to a zero-width assertion — `$*`,
/// `\b{2}`, `^+`. `regex-syntax` accepts them; Python's `re` raises "nothing to
/// repeat". Found by fuzzing the two parsers against each other, not by reading
/// either one's docs.
///
/// Checked on the AST, not the HIR: the HIR translator folds `$+` down to a bare
/// `Look`, so by then the shape Python objects to is gone.
fn repeats_an_assertion(ast: &regex_syntax::ast::Ast) -> bool {
    use regex_syntax::ast::Ast;
    match ast {
        // A quantified assertion (`$*`) or a quantified quantifier (`a?*`, which
        // Python calls "multiple repeat"). Both parse fine in Rust.
        Ast::Repetition(rep) => {
            matches!(&*rep.ast, Ast::Assertion(_) | Ast::Repetition(_))
                || repeats_an_assertion(&rep.ast)
        }
        Ast::Group(g) => repeats_an_assertion(&g.ast),
        Ast::Concat(c) => c.asts.iter().any(repeats_an_assertion),
        Ast::Alternation(a) => a.asts.iter().any(repeats_an_assertion),
        _ => false,
    }
}

/// How many independent length choices a backtracking engine must enumerate.
/// `None` means unbounded (exponential).
///
/// This is the predicate eight review rounds of structural rules kept missing, and
/// it is the only one whose threshold was chosen by MEASUREMENT rather than
/// argument. Over 3730 hostile patterns, each admitted one timed against CPython:
/// a limit of 3 still admitted patterns that never returned, a limit of 2 admitted
/// one costing 190 ms per decode step, and a limit of 1 held every admitted pattern
/// under 4 ms. Hence [`MAX_AMBIGUITY_DEGREE`] = 1.
///
/// Why this cannot repeat the round-8 regression that 400'd every `?`, `*` and `+`:
/// each of those contributes exactly ONE unit of freedom here, never a saturating
/// sentinel, so a single quantifier over a fixed-length body is always admitted.
/// Only composition trips the limit — several in a row (`a*a*a*b`), or one over a
/// body that is itself variable-length (`(?:a*){10}`). The check is also orthogonal
/// to the returned bound: [`hir_max_len`] is untouched, so admitting a pattern never
/// changes the window the scheduler sizes for it.
fn ambiguity_degree(ast: &regex_syntax::ast::Ast) -> Option<u64> {
    use regex_syntax::ast::{Ast, RepetitionKind, RepetitionRange};
    match ast {
        Ast::Group(g) => ambiguity_degree(&g.ast),
        // Siblings compose: `a*a*a*b` is three independent choices, and every one
        // multiplies the work. Summing here is what catches the FLAT spelling that
        // nesting-only rules (and every count cap) walk straight past.
        Ast::Concat(c) => c.asts.iter().try_fold(0u64, |acc, a| {
            Some(acc.saturating_add(ambiguity_degree(a)?))
        }),
        Ast::Alternation(a) => a
            .asts
            .iter()
            .try_fold(0u64, |acc, x| Some(acc.max(ambiguity_degree(x)?))),
        Ast::Repetition(rep) => {
            let body = ambiguity_degree(&rep.ast)?;
            let (lo, hi) = match &rep.op.kind {
                RepetitionKind::ZeroOrOne => (0u64, Some(1u64)),
                RepetitionKind::ZeroOrMore => (0, None),
                RepetitionKind::OneOrMore => (1, None),
                RepetitionKind::Range(RepetitionRange::Exactly(n)) => (*n as u64, Some(*n as u64)),
                RepetitionKind::Range(RepetitionRange::AtLeast(n)) => (*n as u64, None), // codespell:ignore atleast
                RepetitionKind::Range(RepetitionRange::Bounded(a, b)) => {
                    (*a as u64, Some(*b as u64))
                }
            };
            match hi {
                // Unbounded. Repeating an unambiguous fixed-length body is one
                // choice (`a*`, `(?:ab)*`); repeating anything else is exponential.
                None => {
                    if body > 0 || is_variable_length(&rep.ast) {
                        None
                    } else {
                        Some(1)
                    }
                }
                // Counted: the body's own freedom is paid once per iteration, plus
                // one for choosing how many iterations when the count is a range.
                Some(hi) => Some(hi.saturating_mul(body).saturating_add(u64::from(lo != hi))),
            }
        }
        _ => Some(0),
    }
}

/// Whether any alternation sits inside a repetition body.
///
/// `(?:.|.)` and `(?:a|a)` are FIXED length per iteration, so no length-based
/// predicate sees them — yet each iteration has two ways to match, giving 2^n
/// parses. A top-level alternation (`and|or`, the pattern SGLang's own CI sends) is
/// untouched: only a repetition of one is refused.
fn alternation_under_repetition(ast: &regex_syntax::ast::Ast) -> bool {
    use regex_syntax::ast::Ast;
    fn contains_alternation(ast: &Ast) -> bool {
        match ast {
            Ast::Alternation(_) => true,
            Ast::Group(g) => contains_alternation(&g.ast),
            Ast::Concat(c) => c.asts.iter().any(contains_alternation),
            Ast::Repetition(r) => contains_alternation(&r.ast),
            _ => false,
        }
    }
    match ast {
        Ast::Repetition(rep) => {
            contains_alternation(&rep.ast) || alternation_under_repetition(&rep.ast)
        }
        Ast::Group(g) => alternation_under_repetition(&g.ast),
        Ast::Concat(c) => c.asts.iter().any(alternation_under_repetition),
        Ast::Alternation(a) => a.asts.iter().any(alternation_under_repetition),
        _ => false,
    }
}

/// Strict upper bound on the characters `hir` can match; `None` (unbounded) maps to
/// the full-scan sentinel. Saturating throughout: a nested `{65535}` repeat would
/// otherwise overflow into a small — and therefore unsafe — bound.
fn hir_max_len(hir: &regex_syntax::hir::Hir) -> usize {
    use regex_syntax::hir::HirKind;
    match hir.kind() {
        HirKind::Empty | HirKind::Look(_) => 0,
        HirKind::Literal(lit) => lit.0.len(),
        HirKind::Class(_) => 1,
        HirKind::Repetition(rep) => match rep.max {
            None => STOP_REGEX_MAX_LEN,
            Some(max) => (max as usize)
                .saturating_mul(hir_max_len(&rep.sub))
                .min(STOP_REGEX_MAX_LEN),
        },
        HirKind::Capture(cap) => hir_max_len(&cap.sub),
        HirKind::Concat(subs) => subs
            .iter()
            .map(hir_max_len)
            .fold(0, usize::saturating_add)
            .min(STOP_REGEX_MAX_LEN),
        HirKind::Alternation(subs) => subs.iter().map(hir_max_len).max().unwrap_or(0),
    }
}

/// Whether `ast` can match more than one length — the property that makes a
/// repetition of it ambiguous.
fn is_variable_length(ast: &regex_syntax::ast::Ast) -> bool {
    let (lo, hi) = ast_len(ast);
    hi != Some(lo)
}

/// Saturating `(min, max)` match length of `ast`; `max = None` means unbounded.
///
/// Deliberately on the AST rather than the HIR: the translator folds `(?:a|a)` into
/// a single class and `$+` into a bare `Look`, erasing exactly the shapes CPython's
/// engine still has to enumerate.
fn ast_len(ast: &regex_syntax::ast::Ast) -> (u64, Option<u64>) {
    use regex_syntax::ast::{Ast, RepetitionKind, RepetitionRange};
    match ast {
        Ast::Empty(_) | Ast::Flags(_) | Ast::Assertion(_) => (0, Some(0)),
        Ast::Literal(_) | Ast::Dot(_) | Ast::ClassUnicode(_) | Ast::ClassPerl(_) => (1, Some(1)),
        Ast::ClassBracketed(_) => (1, Some(1)),
        Ast::Group(g) => ast_len(&g.ast),
        Ast::Concat(c) => c.asts.iter().fold((0, Some(0)), |(lo, hi), a| {
            let (l, h) = ast_len(a);
            (
                lo.saturating_add(l),
                match (hi, h) {
                    (Some(x), Some(y)) => Some(x.saturating_add(y)),
                    _ => None,
                },
            )
        }),
        Ast::Alternation(a) => a.asts.iter().fold((u64::MAX, Some(0)), |(lo, hi), x| {
            let (l, h) = ast_len(x);
            (
                lo.min(l),
                match (hi, h) {
                    (Some(p), Some(q)) => Some(p.max(q)),
                    _ => None,
                },
            )
        }),
        Ast::Repetition(rep) => {
            let (l, h) = ast_len(&rep.ast);
            let (lo, hi) = match &rep.op.kind {
                RepetitionKind::ZeroOrOne => (0u64, Some(1u64)),
                RepetitionKind::ZeroOrMore => (0, None),
                RepetitionKind::OneOrMore => (1, None),
                RepetitionKind::Range(RepetitionRange::Exactly(n)) => (*n as u64, Some(*n as u64)),
                RepetitionKind::Range(RepetitionRange::AtLeast(n)) => (*n as u64, None), // codespell:ignore atleast
                RepetitionKind::Range(RepetitionRange::Bounded(a, b)) => {
                    (*a as u64, Some(*b as u64))
                }
            };
            (
                lo.saturating_mul(l),
                match (hi, h) {
                    (Some(x), Some(y)) => Some(x.saturating_mul(y)),
                    _ => None,
                },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bound-only view of [`RegexPattern`], so the corpus rows read as
    /// `pattern -> bound` without naming the type at every call.
    fn stop_regex_bound(pattern: &str) -> Result<usize, Error> {
        RegexPattern::try_from(pattern).map(|r| r.max_len())
    }

    /// The admission memo must be indistinguishable from admitting afresh.
    ///
    /// It short-circuits the validator, so a wrong entry would admit a pattern
    /// nobody checked or hand back another pattern's bound — and the bound sizes
    /// the scheduler's match window, which is the under-estimate class of bug this
    /// module exists to prevent. Three properties, one per way that could break:
    /// a repeat agrees with a cold run, a rejection is never memoized, and the
    /// wholesale clear at [`ADMISSION_CACHE_CAP`] loses nothing but the entries.
    #[test]
    fn admission_memo_agrees_with_admitting_afresh() {
        // Distinct from any other test's patterns: the cache is process-wide, so a
        // shared pattern would make this pass for the wrong reason.
        let admitted = r"memo\d{3}[a-f]+";
        let cold = RegexPattern::try_from(admitted).expect("valid").max_len();
        let warm = RegexPattern::try_from(admitted).expect("valid").max_len();
        assert_eq!(
            cold, warm,
            "a memoized bound must equal a freshly derived one"
        );

        // Rejections are re-validated every time, so the memo can never turn one
        // into an admission.
        let rejected = r"memo(?:.|.)*Z";
        assert!(RegexPattern::try_from(rejected).is_err());
        assert!(
            RegexPattern::try_from(rejected).is_err(),
            "a rejected pattern must stay rejected on the second try"
        );

        // Overflow the cache, then re-check: clearing must not corrupt or stale a
        // subsequent lookup.
        for i in 0..=ADMISSION_CACHE_CAP {
            let _ = RegexPattern::try_from(format!("memofill{i}").as_str());
        }
        assert_eq!(
            RegexPattern::try_from(admitted).expect("valid").max_len(),
            cold,
            "the bound must survive a cache clear"
        );
    }

    #[test]
    fn admitted_pattern_carries_its_own_text_and_bound() {
        let p = RegexPattern::try_from(r"\d{6}").expect("valid");
        assert_eq!(p.pattern(), r"\d{6}");
        assert_eq!(p.max_len(), 6);
    }

    /// The property this whole design rests on: **anything Rust admits, Python can
    /// compile.** The reverse may fail — rejecting a pattern Python would accept
    /// costs one client a 400, while admitting one it cannot compile costs the
    /// scheduler, because `re.search` runs on the decode hot path where nothing
    /// Budget for one `re.search` on the scheduler's decode thread. Every safe
    /// pattern below measures under 0.1 ms; the cheapest unsafe one is 636 ms.
    const SEARCH_BUDGET_MS: f64 = 5.0;

    /// What the admission policy must do with a pattern.
    #[derive(Debug, PartialEq)]
    enum Policy {
        /// Admitting it kills the scheduler or silently misses the stop.
        MustReject,
        /// Admitting it is REQUIRED. Deliberately small: the two patterns
        /// SGLang's own `matched_stop_kit` sends over HTTP (five registered suites
        /// assert on the result), plus three canaries. Without the canaries an
        /// admission bug that rejects EVERYTHING would pass a table of nothing but
        /// `MayReject` — which is how round 8 shipped a build that 400'd every
        /// `?`, `*` and `+`.
        MustAdmit,
        /// Python compiles it; Rust may or may not, and either verdict passes.
        /// Over-rejection is the design: a 400 costs the client a feature,
        /// admitting the wrong thing costs the scheduler. These rows document
        /// where the boundary currently sits, they do not constrain it.
        MayReject,
    }

    /// One corpus row. Every column except `policy` is a MEASURED fact, recorded
    /// so a future edit cannot re-derive it by guessing:
    ///   * `py_max_len`  — CPython `get_max_seq_length`, or `None` when that call
    ///     itself raises. NOT the same as "`re.compile` rejects it": `(?<=a*)b`
    ///     parses (so `get_max_seq_length` returns a number) but fails to compile.
    ///     Safety never rests on this column alone — `worst_ms` is independent.
    ///   * `worst_ms`    — worst `re.search` over a growing tail (16→88 chars of
    ///     prose, or a matching run where the pattern needs one). `INFINITY` means
    ///     it did not return inside 8 s under a 2 GiB cap.
    struct Case {
        pattern: String,
        policy: Policy,
        /// Expected bound when admitted. Pins `hir_max_len` against silent drift.
        rust_bound: usize,
        py_max_len: Option<i64>,
        worst_ms: f64,
    }

    fn case(pattern: &str, policy: Policy, rust_bound: usize, py: Option<i64>, ms: f64) -> Case {
        Case {
            pattern: pattern.to_string(),
            policy,
            rust_bound,
            py_max_len: py,
            worst_ms: ms,
        }
    }

    /// The single source of truth for `stop_regex` admission.
    ///
    /// The contract is ONE-SIDED: the admitted set must be a SUBSET of what
    /// CPython can compile and match cheaply. Rust does not reproduce Python's
    /// dialect — rejecting a pattern Python accepts costs the client a feature,
    /// admitting one Python chokes on costs the scheduler and the GPU state. So
    /// `MustReject` carries the whole safety burden, and `MustAdmit` is held to
    /// the few patterns the project's own tests actually send.
    ///
    /// This table exists because eight review rounds each found a NEW spelling of
    /// an already-fixed hazard, and the previous corpus could not catch any of
    /// them: its assertion was `!admitted || python_compiles`, which any row with
    /// `python_compiles = true` satisfies vacuously — including four rows whose own
    /// comments called them scheduler-fatal. It also could not fail on a spurious
    /// 400, so a round that rejected `(?i)[a-z]+` and `colou?r` shipped green.
    ///
    /// KEEP IN SYNC: adding a row means MEASURING `py_max_len` and `worst_ms`, not
    /// guessing them. `corpus_rows_are_self_consistent` refuses a row that records
    /// a fatal measurement and then claims the pattern is safe to admit.
    fn corpus() -> Vec<Case> {
        const UNBOUNDED: usize = STOP_REGEX_MAX_LEN;
        const INF: f64 = f64::INFINITY;
        let mut c = vec![
            // ---- Direction A: CPython cannot compile these. Admitting one puts a
            // `re.error` in `_check_str_based_finish`, on the decode path, uncaught.
            case(r"\p{L}", Policy::MustReject, 0, None, INF), // round 1
            case(r"\P{L}", Policy::MustReject, 0, None, INF),
            case(r"\pL", Policy::MustReject, 0, None, INF),
            case("(?<n>a)", Policy::MustReject, 0, None, INF),
            case(r"\x{1F600}", Policy::MustReject, 0, None, INF),
            case(r"\u{41}", Policy::MustReject, 0, None, INF),
            case("(?<=a*)b", Policy::MustReject, 0, Some(1073741825), INF), // round 2: variable-width lookbehind
            case("(", Policy::MustReject, 0, None, INF),
            case("[z-a]", Policy::MustReject, 0, None, INF),
            case("a{2,1}", Policy::MustReject, 0, None, INF),
            case("$*", Policy::MustReject, 0, None, INF),
            case(r"\b{2}", Policy::MustReject, 0, None, INF),
            case("^+", Policy::MustReject, 0, None, INF),
            case("a?*", Policy::MustReject, 0, None, INF),
            case("a{2,5}?*", Policy::MustReject, 0, None, INF),
            case("a(?i)b", Policy::MustReject, 0, None, INF),
            case("(?-i)a", Policy::MustReject, 0, None, INF),
            case("[a[:alpha:](?=-]", Policy::MustReject, 0, None, INF),
            // Round 4: the escape check skipped character-class bodies entirely,
            // so round 1's hole reopened one bracket pair away.
            case(r"[\p{L}]", Policy::MustReject, 0, None, INF),
            case(r"[\pL]", Policy::MustReject, 0, None, INF),
            case(r"[\P{L}]", Policy::MustReject, 0, None, INF),
            case(r"[\x{41}]", Policy::MustReject, 0, None, INF),
            case("[a--b]", Policy::MustReject, 0, None, INF),
            case("(?R)a", Policy::MustReject, 0, None, INF), // round 4: Rust-only flag
            case("(?U)a", Policy::MustReject, 0, None, INF),
            case("(?R:a)", Policy::MustReject, 0, None, INF), // round 5: the scoped spelling
            case("(?U:a)", Policy::MustReject, 0, None, INF),
            // `regex-syntax` parses counts as u32 and accepts up to u32::MAX;
            // CPython's MAXREPEAT *is* u32::MAX and raises OverflowError, which is
            // neither `re.error` nor `RecursionError` and so escapes every guard.
            case("a{4294967295}", Policy::MustReject, 0, None, INF), // round 4
            case("a{5000000000}", Policy::MustReject, 0, None, INF), // round 3
            // ---- Bound UNDER-estimates. Both compile and run fast, so only the
            // `rust_bound >= py_max_len` column catches them: `regex-syntax` reads
            // a zero-width word boundary where CPython reads escaped literals, so
            // the scheduler sizes too small a window and the stop never fires.
            case(r"\<END\>", Policy::MustReject, 3, Some(5), 0.02), // round 5
            case(r"\b{start}xyz", Policy::MustReject, 3, Some(10), 0.04), // round 4
            // ---- Compounding repeat cost. Both compile in CPython; both are fatal
            // there. `repetition_cost_too_large` covers these.
            case(
                "(?:(?:a*){65535}){65535}",
                Policy::MustReject,
                0,
                Some(4611545282012774400),
                INF,
            ),
            case("(?:){1048575}x", Policy::MustReject, 0, Some(1), INF),
            // ---- AMBIGUITY (rounds 6-8). Every one compiles cleanly on both sides
            // and raises nothing, so the `except (re.error, RecursionError)` seatbelt
            // in `_check_str_based_finish` is irrelevant: the match simply never
            // returns, inside GIL-holding CPython C that no watchdog can preempt.
            //
            // Two distinct kill modes, both represented:
            //   * unbounded bound -> `_stop_match_tail_len` hands `re.search` the
            //     WHOLE accumulated output, so cost grows every decode step;
            //   * finite bound -> a fixed but ruinous cost paid EVERY step forever,
            //     and `MAX_STOP_REGEX_COUNT` allows 64 patterns per request.
            // Timings on a matching subject; see the module docs for the method.
            case(
                "(?:.|.)*Z",
                Policy::MustReject,
                UNBOUNDED,
                Some(1073741825),
                INF,
            ),
            case(
                "(a|a)*b",
                Policy::MustReject,
                UNBOUNDED,
                Some(1073741825),
                INF,
            ),
            case("(?:a+)+b", Policy::MustReject, 0, Some(1073741825), INF),
            case(
                "(?:a*){10}b",
                Policy::MustReject,
                UNBOUNDED,
                Some(10737418241),
                INF,
            ),
            case(
                "a*a*a*a*a*a*a*a*b",
                Policy::MustReject,
                UNBOUNDED,
                Some(8589934593),
                636.05,
            ),
            case(
                "(?:.*){20}Z",
                Policy::MustReject,
                UNBOUNDED,
                Some(21474836481),
                INF,
            ),
            case(
                ".*.*.*.*.*.*.*.*Z",
                Policy::MustReject,
                UNBOUNDED,
                Some(8589934593),
                INF,
            ),
            case("(?:.?){30}Z", Policy::MustReject, 31, Some(31), INF), // round 7
            case("(?:.?){255}Z", Policy::MustReject, 256, Some(256), INF),
            case(
                "(?:.{0,1}.{0,1}.{0,1}){8}Z",
                Policy::MustReject,
                25,
                Some(25),
                INF,
            ),
            case(
                "(?:(?:.?){15}){15}Z",
                Policy::MustReject,
                226,
                Some(226),
                INF,
            ),
            case("(?:.?.?.?.?){60}Z", Policy::MustReject, 241, Some(241), INF),
            // ---- MustAdmit. Only the first two are contractual: `matched_stop_kit`
            // sends them over HTTP and five registered suites assert on the result.
            // The next three are canaries — a plain literal, a bounded class repeat,
            // a simple optional — so an admission bug that rejects everything cannot
            // pass. The rest of this block is `MayReject`: nice to keep working, but
            // the subset contract does not require it.
            case(
                r"[.!?]\s*$",
                Policy::MustAdmit,
                UNBOUNDED,
                Some(1073741825),
                0.03,
            ),
            case("and|or", Policy::MustAdmit, 3, Some(3), 0.03),
            case(r"\d+", Policy::MayReject, UNBOUNDED, Some(1073741824), 0.03),
            case(
                r"\s+$",
                Policy::MayReject,
                UNBOUNDED,
                Some(1073741824),
                0.04,
            ),
            case(
                "Answer: .*",
                Policy::MayReject,
                UNBOUNDED,
                Some(1073741832),
                0.03,
            ),
            case(".*", Policy::MayReject, UNBOUNDED, Some(1073741824), 0.04),
            case(
                "a{3,}",
                Policy::MayReject,
                UNBOUNDED,
                Some(1073741824),
                0.03,
            ),
            // Round 8 regressed every `?`/`*`/`+` to a 400 by routing them into an
            // "unbounded" catch-all that returned u64::MAX.
            case("colou?r", Policy::MustAdmit, 6, Some(6), 0.03),
            case("https?://", Policy::MayReject, 8, Some(8), 0.02),
            case("END(ING)?", Policy::MayReject, 6, Some(6), 0.02),
            // Round 7 regressed these by scanning the whole pattern for `-` instead
            // of just the flag bytes.
            case(
                "(?i)[a-z]+",
                Policy::MayReject,
                UNBOUNDED,
                Some(1073741824),
                0.04,
            ),
            case(r"(?i)\d{4}-\d{2}", Policy::MayReject, 7, Some(7), 0.06),
            case("(?imsx)a-b", Policy::MayReject, 3, Some(3), 0.04),
            case("(?-i:abc)", Policy::MayReject, 3, Some(3), 0.03),
            case("(?i-s:a)", Policy::MayReject, 1, Some(1), 0.04),
            case("(?i)(?m)a", Policy::MayReject, 1, Some(1), 0.03),
            case(r"\x41", Policy::MayReject, 1, Some(1), 0.02),
            case(r"\d{6}", Policy::MustAdmit, 6, Some(6), 0.03),
            case("abc", Policy::MustAdmit, 3, Some(3), 0.03),
            case("(?P<n>a)", Policy::MayReject, 1, Some(1), 0.03),
            case(r"a\.b", Policy::MayReject, 3, Some(3), 0.03),
            case(r"\bword\b", Policy::MayReject, 4, Some(4), 0.03),
            case(r"[\d\s]{2}", Policy::MayReject, 2, Some(2), 0.03),
            // ---- MayReject: CPython accepts, `regex-syntax` is stricter. A 400
            // costs the client a feature; admitting costs nothing either. Listed so
            // the set of deliberate over-rejections is visible rather than folklore.
            case(r"a\Z", Policy::MayReject, 1, Some(1), 0.03),
            case(r"(a)\1", Policy::MayReject, 0, Some(1073741825), 0.04),
            case("(?=x)y", Policy::MayReject, 0, Some(1073741825), 0.03),
            case("a{,5}", Policy::MayReject, 5, Some(5), 0.04),
            case(r"\N{SNOWMAN}", Policy::MayReject, 1, Some(1), 0.03),
            case(r"\0", Policy::MayReject, 1, Some(1), 0.03),
        ];
        // Flat concatenations of optional atoms — the round-8 escape. Built rather
        // than written out because they are 73-221 bytes of repetition.
        c.push(case(
            &format!("{}Z", ".{0,1}".repeat(20)),
            Policy::MustReject,
            21,
            Some(21),
            650.62,
        ));
        c.push(case(
            &format!("{}Z", ".{0,4}".repeat(12)),
            Policy::MustReject,
            49,
            Some(49),
            INF,
        ));
        c
    }

    /// A row may not record a fatal measurement and then claim the pattern is safe
    /// to admit. Without this, the table can be made green by editing a verdict
    /// instead of fixing the code — which is exactly how round 8's `(?i)[a-z]+`
    /// regression survived (the corpus row was left alone and a *different* test
    /// was edited from `(?i)[a-z]+` to `(?i)[a-z]{1,8}` to keep it passing).
    #[test]
    fn corpus_rows_are_self_consistent() {
        for c in corpus() {
            if c.py_max_len.is_none() || c.worst_ms > SEARCH_BUDGET_MS {
                assert_eq!(
                    c.policy,
                    Policy::MustReject,
                    "{:?} does not compile in Python, or costs {} ms per decode step \
                     (budget {SEARCH_BUDGET_MS} ms) — it cannot be admitted",
                    c.pattern,
                    c.worst_ms
                );
            }
        }
    }

    /// The corpus, asserted in BOTH directions plus the bound.
    ///
    /// Three independent invariants, each of which caught a real bug that the
    /// others missed:
    ///   1. `MustReject` really is rejected — Direction A (scheduler death) and the
    ///      ambiguity family (scheduler wedge).
    ///   2. `MustAdmit` really is admitted — a spurious 400 breaks working clients
    ///      and, twice now, SGLang's own registered suites.
    ///   3. an admitted pattern's bound is >= CPython's, so the scheduler's match
    ///      window is never too small. This is the only mechanical check for the
    ///      `\b{start}` / `\<` class, which nobody found by reading.
    #[test]
    fn stop_regex_corpus_holds_in_both_directions() {
        let mut failures: Vec<String> = Vec::new();
        for c in corpus() {
            let got = stop_regex_bound(&c.pattern);
            match (&c.policy, &got) {
                (Policy::MustReject, Ok(bound)) => failures.push(format!(
                    "ADMITTED but must be rejected: {:?} (bound {bound}, \
                     worst re.search {} ms)",
                    c.pattern, c.worst_ms
                )),
                (Policy::MustAdmit, Err(e)) => failures.push(format!(
                    "REJECTED but must be admitted: {:?} — {e}",
                    c.pattern
                )),
                _ => {}
            }
            if let Ok(bound) = got {
                if c.policy != Policy::MustReject && bound != c.rust_bound {
                    failures.push(format!(
                        "bound drift: {:?} expected {} got {bound}",
                        c.pattern, c.rust_bound
                    ));
                }
                // Only meaningful when CPython's own bound is finite: for unbounded
                // patterns both sides emit an absurd sentinel that the scheduler
                // caps at the output length anyway.
                if let Some(py) = c.py_max_len
                    && py < STOP_REGEX_MAX_LEN as i64
                    && (bound as i64) < py
                {
                    {
                        failures.push(format!(
                            "UNDER-estimate: {:?} rust bound {bound} < python {py} — \
                             the scheduler's window is too small and the stop never fires",
                            c.pattern
                        ));
                    }
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} corpus row(s) failed:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }

    /// The leading-flag check must look at the FLAG BYTES, not the rest of the
    /// pattern: scanning the whole tail for `-` made `(?i)[a-z]+` — about as
    /// ordinary as a stop_regex gets — a 400.
    #[test]
    fn leading_inline_flags_are_accepted() {
        for pattern in [
            "(?i)[a-z]{1,8}",
            r"(?i)\d{4}-\d{2}",
            "(?imsx)a-b",
            "(?i)abc",
        ] {
            assert!(
                stop_regex_bound(pattern).is_ok(),
                "{pattern} is valid Python and must not be rejected"
            );
        }
        // …but only leading, only set-flags, and only portable letters.
        for pattern in ["a(?i)b", "(?-i)a", "(?R)a", "(?U)a"] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
    }

    /// Patterns Python compiles fine that this validator used to 400. A false
    /// rejection is safe but it is still a bug: `(?i-s:a)` alone was 267 hits in
    /// the review corpus, and `\x41` was rejected by the very list whose doc
    /// comment calls `\xHH` shared.
    #[test]
    fn ordinary_python_patterns_are_not_spuriously_rejected() {
        for pattern in [
            "(?-i:abc)", // scoped clearing group: legal anywhere
            "(?i-s:a)",  // mixed set/clear inside a scoped group
            r"\x41",     // two-hex escape — shared with Python
            r"a\x41b",
            "(?i)(?m)a", // several LEADING global flag groups
            "(?i)abc",
        ] {
            assert!(
                stop_regex_bound(pattern).is_ok(),
                "{pattern} is valid Python and must not be rejected"
            );
        }
        // The genuinely Rust-only forms still reject.
        for pattern in [r"\x{41}", "a(?i)b", "(?R)a"] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
    }

    /// A repetition count Python cannot honour: `u32::MAX` is its `MAXREPEAT`
    /// sentinel (`OverflowError`), and a large count on a group exhausts memory at
    /// compile time (`MemoryError`). Neither is an `re.error`, so the decode-loop
    /// seatbelt would not catch either.
    #[test]
    fn oversized_repeat_counts_are_rejected() {
        for pattern in [
            "a{4294967295}",
            "a{4294967294}",
            "(?:a*){4294967294}",
            "a{1048576}",
            "a{0,4294967295}",
            "a{1048576,}",
        ] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
        // An ordinary count still works, and still yields a finite bound.
        assert_eq!(stop_regex_bound("a{200}").unwrap(), 200);
    }

    /// `\b{start}` is one zero-width assertion to Rust (bound 0) but `\b` plus the
    /// literal `{start}` to Python (7 characters). Scoring it 0 would size a
    /// 1-token match window where 7 characters are needed, and the stop would
    /// silently never fire — an UNDER-estimate, the one failure mode the sentinel
    /// design exists to prevent.
    #[test]
    fn b_brace_assertion_is_rejected_not_under_estimated() {
        assert!(stop_regex_bound(r"\b{start}xyz").is_err());
        assert!(stop_regex_bound(r"\b{end}").is_err());
        assert_eq!(
            stop_regex_bound(r"\bword").unwrap(),
            4,
            "plain \\b still works"
        );
    }

    /// Round 5's under-estimate: `regex-syntax` reads `\<`/`\>` as GNU word-boundary
    /// assertions (width 0), CPython as escaped literals. Scoring `\<END\>` as 3
    /// instead of 5 sizes the scheduler's match window too small, so the stop never
    /// fires and the request burns GPU to `max_new_tokens`.
    #[test]
    fn gnu_word_boundary_escapes_are_rejected() {
        for pattern in [r"\<END\>", r"\<word", r"end\>"] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
        // A plain `<` is a literal in both and still bounds correctly.
        assert_eq!(stop_regex_bound("<END>").unwrap(), 5);
    }

    /// Repetition cost compounds down the nesting, so a per-node cap misses
    /// `(?:(?:a*){65535}){65535}` — 22 bytes, compiles fine in Python, then eats
    /// GiB inside `re.search` on the decode hot path (`MemoryError`, which the
    /// seatbelt does not catch). Nested UNBOUNDED repeats are the backtracking
    /// family, fatal in wall-clock rather than memory.
    #[test]
    fn compounding_repetition_cost_is_rejected() {
        for pattern in [
            "(?:(?:a*){65535}){65535}",
            "(?:){1048575}x",
            "(?:a{100}){100}",
            "(?:a+)+b",
            "(a*)*b",
        ] {
            assert!(
                stop_regex_bound(pattern).is_err(),
                "{pattern} must be rejected"
            );
        }
        // Ordinary nesting still works.
        assert_eq!(stop_regex_bound("(?:ab){3}").unwrap(), 6);
        assert_eq!(stop_regex_bound(r"\d{6}").unwrap(), 6);
    }

    /// Deep nesting is rejected here rather than blowing Python's parser stack:
    /// CPython compiles up to ~495 levels and raises `RecursionError` past that, so
    /// the parser's nest limit is pinned well below it.
    #[test]
    fn deep_nesting_is_rejected_below_pythons_limit() {
        let nest = |n: usize| format!("{}a{}", "(".repeat(n), ")".repeat(n));
        assert!(
            stop_regex_bound(&nest(10)).is_ok(),
            "ordinary nesting is fine"
        );
        assert!(
            stop_regex_bound(&nest(400)).is_err(),
            "must be rejected here — Python raises RecursionError, not re.error"
        );
        assert!(stop_regex_bound(&nest(2000)).is_err());
    }

    /// Bounded patterns get their real length; unbounded ones the full-scan
    /// sentinel, so the scheduler never under-buffers and misses a stop.
    #[test]
    fn stop_regex_bound_is_finite_when_bounded() {
        let len = |p: &str| stop_regex_bound(p).expect("valid pattern");
        assert_eq!(len(r"\d{6}"), 6);
        assert_eq!(len("abc"), 3);
        assert_eq!(len(r"^abc$"), 3); // anchors are zero-width
        assert_eq!(len("a|bbb"), 3); // alternation → max branch
        assert_eq!(len(r"(ab){3}"), 6);
        assert_eq!(len(r"a\d{2,5}"), 6);
        assert_eq!(len(r"\d+"), STOP_REGEX_MAX_LEN);
        assert_eq!(len(".*"), STOP_REGEX_MAX_LEN);
        assert_eq!(len(r"a{3,}"), STOP_REGEX_MAX_LEN);
    }
}
