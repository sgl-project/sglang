# dynamo-renderer vendor note

This directory vendors `dynamo-renderer` 5.1.2 from the crates.io archive with
SHA-256 `3ae2eaa139651c535aeaad8856c5546709608931ccd4d24d3a529f6c5233c4b6`.
The archive records upstream `ai-dynamo/frontend-crates` revision
`92cc8bc5f3960a76cd6b92dd348db48a48c83b1f` (`renderer`).

SGLang retains the crate locally for two functional changes. The direct-server
message adapter accesses the renderer's existing content-array classification
through the accessor seam in `src/lib.rs` and `src/template/oai.rs`. In
`src/template/tokcfg.rs`, the compact `tojson` formatter uses Serde's
shortest-roundtrip digits with Python's notation thresholds and exponent
spelling for finite `f64` values. The regression corpus covers notation
boundaries and decimal ties against a pinned Python reference; it does not
establish exhaustive equivalence for every binary64 value. The `indent` branch
continues to use Serde's `PrettyFormatter`; numeric parity for that mode is not
claimed. This package also carries modification notices and comment-only
spelling/codespell metadata.

To update or retire this override, first verify that a compatible upstream
release or a reviewed local replacement preserves both the classification
accessor and the compact numeric output behavior. Then remove the
`[patch.crates-io]` path override, update `rust/Cargo.lock`, remove this vendor
directory, and rerun the affected renderer, message, numeric prompt/token and
repository checks. Accessor availability alone is not sufficient to remove the
override. Do not replace the vendored crate solely to satisfy formatting or
spelling checks.
