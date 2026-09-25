//! Shared request primitives.

use serde::{Deserialize, Serialize};

/// A flat token-id buffer — one request's `array("q")` cell on the Python side.
/// Wrapped in [`OneOrMany`] on the wire, where a bare list is one prompt's ids
/// (or a broadcast) and a list of lists is per-prompt.
pub type TokenIds = Vec<i32>;

/// A field taking a bare `T` **or** `[T,…]` (`text: "hi"` or `text: ["a","b"]`).
/// `untagged` takes the first variant that matches, so a `T` that itself accepts
/// a sequence would make `Many` unreachable — hence the [`OneOrManyItem`] gate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OneOrMany<T: OneOrManyItem> {
    One(T),
    Many(Vec<T>),
}

/// Types vetted for [`OneOrMany`]. Sealed, so adding one is a deliberate act in
/// this file.
///
/// **Never implement this for a self-describing type — `serde_json::Value`,
/// `rmpv::Value`, or anything else that deserializes from *any* shape.** Such a
/// `T` matches `[1,2]` as `One(Value::Array(…))`, so `Many` is never selected and
/// a batch silently arrives as a single request. Those types need a
/// `deserialize_any` dispatch instead (see the server's `SamplingParamsBatch`).
///
/// [`TokenIds`] and `Vec<String>` are the members that do accept a sequence, and
/// that ambiguity is the intended semantics: flat `[1,2]` is one prompt's ids
/// (or a broadcast), `[[1],[2]]` is per-prompt — the shapes Python's
/// `_normalize_batch` distinguishes, and `mm_hashes`'s
/// `Union[List[str], List[List[str]]]` reads the same way. `String` / `bool` /
/// `i64` never match a list, so both forms round-trip.
pub trait OneOrManyItem: sealed::SealedItem {}

impl<T: sealed::SealedItem> OneOrManyItem for T {}

mod sealed {
    /// Supertrait no downstream module can implement, sealing [`super::OneOrManyItem`].
    pub trait SealedItem {}

    impl SealedItem for bool {}
    impl SealedItem for i64 {}
    impl SealedItem for String {}
    impl SealedItem for super::TokenIds {}
    /// `mm_hashes`: a flat list is one request's hashes, nested is per-request.
    impl SealedItem for Vec<String> {}
    // Nullable elements for the PD bootstrap fields (`List[Optional[...]]` in
    // Python — the PD router sends `bootstrap_port: [null, …]` when deferring to
    // the scheduler's default port). A bare `null` never reaches `One(None)`: the
    // outer `Option<OneOrMany<…>>` field consumes it first.
    impl SealedItem for Option<i64> {}
    impl SealedItem for Option<String> {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pins `untagged`'s first-match-wins variant selection for the vetted
    /// [`OneOrManyItem`] types: the `TokenIds` rows are the shapes
    /// `GenerateBody::into_requests` relies on (flat = one prompt / broadcast, nested =
    /// per-prompt), and `String` is the unambiguous case.
    #[test]
    fn untagged_selects_the_first_matching_variant() {
        let one_of = |json: &str| -> bool {
            matches!(
                serde_json::from_str::<OneOrMany<TokenIds>>(json).unwrap(),
                OneOrMany::One(_)
            )
        };
        assert!(one_of("[1,2]"), "a flat id list is one prompt's ids");
        assert!(!one_of("[[1],[2]]"), "a nested list is per-prompt");

        // A string can never match a sequence, so both forms stay unambiguous.
        assert!(matches!(
            serde_json::from_str::<OneOrMany<String>>(r#""hi""#).unwrap(),
            OneOrMany::One(_)
        ));
        assert!(matches!(
            serde_json::from_str::<OneOrMany<String>>(r#"["a","b"]"#).unwrap(),
            OneOrMany::Many(v) if v.len() == 2
        ));

        // The hazard case is no longer expressible: `OneOrMany<serde_json::Value>`
        // fails to compile because `Value` is not an `OneOrManyItem`, so `Many`
        // can never be silently unreachable. (Verified by construction — adding
        // that instantiation anywhere is a compile error.)
    }
}
