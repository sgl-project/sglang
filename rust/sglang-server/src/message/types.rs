//! Scheduler wire-shape types.

pub use sglang_renderer::{OneOrMany, OneOrManyItem, TokenIds};

/// A msgspec `tag=True` struct: element 0 of its array is the Python class name
/// the scheduler decodes by. Declared explicitly rather than taken from
/// `type_name`, which is unspecified and path-qualified.
pub(super) trait Tagged {
    const TAG: &'static str;
}

/// Declare msgspec `array_like=True` wire structs by their *own* fields, in wire
/// order; the inherited `BaseReq` preamble (`tag`, `rid`, `http_worker_ipc`) and
/// the [`Tagged`] impl are generated. **The struct name is the Python class
/// name** (via `stringify!`), so name it exactly as `io_struct.py` does — the
/// per-message tests assert the tag, so a rename fails loudly. Field attributes
/// pass through. The http_worker_ipc is just a placeholder for the scheduler's
/// `BaseReq` and is always `()`, without it the scheduler's `BaseReq` would be
/// misaligned on the wire.
///
/// Two forms, and **one invocation may use only one** (an arm matches the whole
/// invocation): `Name<'a>` borrows its rid — zero-copy, for the hot path;
/// `Name` owns it, for a message held by the owned `Request` it would borrow.
macro_rules! wire_struct {
    ($(
        $(#[$meta:meta])*
        $vis:vis $name:ident<$lt:lifetime> {
            $($(#[$field_meta:meta])* $field:ident: $ty:ty,)*
        }
    )+) => {$(
        $(#[$meta])*
        #[derive(Debug)]
        $vis struct $name<$lt> {
            rid: &$lt str,
            $($(#[$field_meta])* $field: $ty,)*
        }

        /// Hand-written so the `BaseReq` preamble is SYNTHESIZED rather than stored.
        ///
        /// `tag` and `http_worker_ipc` are the same two values for every message, so
        /// carrying them as fields meant every constructor restated them — and a
        /// `Default`-based shortcut is the wrong fix twice over: the borrowed form
        /// holds `&SamplingParams`, which has no `Default` at all, and
        /// `..Default::default()` would let a field added here but missed in a
        /// constructor ship a silent default on a POSITIONAL wire. Emitting them
        /// here instead means the tag comes from [`Tagged::TAG`] and cannot be
        /// forgotten, mistyped, or paired with the wrong struct.
        ///
        /// `serialize_struct` (not `serialize_seq`) keeps `rmp_serde` on exactly the
        /// code path the derive used, so the bytes are unchanged — which
        /// `to_header_msgpack_is_positionally_aligned` asserts index by index.
        impl<$lt> serde::Serialize for $name<$lt> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeStruct; // codespell:ignore ser
                // Annotated so the zero-field case (a bare `BaseReq`) still infers.
                let own = <[&'static str]>::len(&[$(stringify!($field)),*]);
                let mut st = serializer.serialize_struct(stringify!($name), 3 + own)?;
                st.serialize_field("tag", <Self as Tagged>::TAG)?;
                st.serialize_field("rid", &self.rid)?;
                st.serialize_field("http_worker_ipc", &())?;
                $(st.serialize_field(stringify!($field), &self.$field)?;)*
                st.end()
            }
        }

        impl<$lt> $name<$lt> {
            pub fn encode(&self) -> Result<Bytes, Error> {
                rmp_serde::to_vec(self)
                    .map(Bytes::from)
                    .map_err(|e| Error::Codec(e.to_string()))
            }
        }

        impl<$lt> Tagged for $name<$lt> {
            const TAG: &'static str = stringify!($name);
        }
    )+};
    ($(
        $(#[$meta:meta])*
        $vis:vis $name:ident {
            $($(#[$field_meta:meta])* $field:ident: $ty:ty,)*
        }
    )+) => {$(
        $(#[$meta])*
        #[derive(Debug)]
        $vis struct $name {
            rid: String,
            $($(#[$field_meta])* $field: $ty,)*
        }

        /// Hand-written so the `BaseReq` preamble is SYNTHESIZED rather than stored.
        ///
        /// `tag` and `http_worker_ipc` are the same two values for every message, so
        /// carrying them as fields meant every constructor restated them — and a
        /// `Default`-based shortcut is the wrong fix twice over: the borrowed form
        /// holds `&SamplingParams`, which has no `Default` at all, and
        /// `..Default::default()` would let a field added here but missed in a
        /// constructor ship a silent default on a POSITIONAL wire. Emitting them
        /// here instead means the tag comes from [`Tagged::TAG`] and cannot be
        /// forgotten, mistyped, or paired with the wrong struct.
        ///
        /// `serialize_struct` (not `serialize_seq`) keeps `rmp_serde` on exactly the
        /// code path the derive used, so the bytes are unchanged — which
        /// `to_header_msgpack_is_positionally_aligned` asserts index by index.
        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                use serde::ser::SerializeStruct; // codespell:ignore ser
                // Annotated so the zero-field case (a bare `BaseReq`) still infers.
                let own = <[&'static str]>::len(&[$(stringify!($field)),*]);
                let mut st = serializer.serialize_struct(stringify!($name), 3 + own)?;
                st.serialize_field("tag", <Self as Tagged>::TAG)?;
                st.serialize_field("rid", &self.rid)?;
                st.serialize_field("http_worker_ipc", &())?;
                $(st.serialize_field(stringify!($field), &self.$field)?;)*
                st.end()
            }
        }

        impl $name {
            pub fn get_rid(&self) -> &str {
                &self.rid
            }

            pub fn encode(&self) -> Result<Bytes, Error> {
                rmp_serde::to_vec(self)
                    .map(Bytes::from)
                    .map_err(|e| Error::Codec(e.to_string()))
            }
        }

        impl Tagged for $name {
            const TAG: &'static str = stringify!($name);
        }
    )+};
}

/// Declare every owned-rid control message *and* the `ControlRequest` enum that
/// carries them. The enum, its variants and its delegating methods are generated
/// from this one list, so adding a message is a single declaration instead of an
/// edit in three places (enum + each method).
macro_rules! control_messages {
    ($(
        $(#[$meta:meta])*
        $name:ident {
            $($(#[$field_meta:meta])* $field:ident: $ty:ty,)*
        }
    )+) => {
        wire_struct! {$(
            $(#[$meta])*
            pub(crate) $name {
                $($(#[$field_meta])* $field: $ty,)*
            }
        )+}

        /// Which control message a request carries, as the wire struct itself.
        /// These own their rid (`String`), so the enum needs no lifetime — a
        /// borrowed rid would point back at the `Request` that owns this value.
        #[derive(Debug)]
        pub enum ControlRequest {
            $($name($name),)+
        }

        impl ControlRequest {
            /// The rid this message carries; `submit` reuses it as the request's
            /// rid, so the two cannot disagree.
            pub(crate) fn rid(&self) -> &str {
                match self {
                    $(Self::$name(m) => m.get_rid(),)+
                }
            }

            /// Encode as the msgspec tagged array. The variant selects the wire
            /// struct, so an unknown tag is not representable.
            pub(crate) fn encode(&self) -> Result<Bytes, Error> {
                match self {
                    $(Self::$name(m) => m.encode(),)+
                }
            }
        }
    };
}

// `macro_rules!` is scoped by declaration order; name it so siblings can `use` it.
pub(super) use {control_messages, wire_struct};

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
