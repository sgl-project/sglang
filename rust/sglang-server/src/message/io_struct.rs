//! The scheduler wire structs — the Rust mirror of the Python `io_struct`
//! messages this server sends (`python/sglang/srt/managers/io_struct.py`).
//! Each is a msgspec `array_like=True` struct, so **field order is wire order**
//! and `rmp_serde`'s default struct-as-array encoding reproduces it.

use bytes::Bytes;
use serde::Serialize;
use serde::ser::{SerializeMap, SerializeStruct}; // codespell:ignore ser

use super::types::{Tagged, control_messages, wire_struct};
use super::{GenerateRequest, SamplingParams, TokenIds};
use crate::error::Error;

wire_struct! {
    /// The scheduler's `TokenizedGenerateReqInput`. Keep in lockstep with the
    /// Python declaration: inserting a field anywhere but the end shifts every
    /// later field on the wire.
    pub(super) TokenizedGenerateReqInput<'a> {
        input_text: Option<&'a str>,
        /// Always nil: the ids ride the ring's columnar buffer, not msgpack.
        input_ids: (),
        input_embeds: (),
        mm_inputs: (),
        token_type_ids: (),
        sampling_params: SamplingParamsMap<'a>,
        return_logprob: bool,
        logprob_start_len: i64,
        top_logprobs_num: i64,
        token_ids_logprob: Option<&'a TokenIds>,
        stream: bool,
        /// Not exposed by this server yet; the scheduler needs the slot filled.
        return_sampling_mask: bool,
        return_hidden_states: bool,
    }
}

// Owned-rid messages: these are held by a [`ControlRequest`] inside an owned
// `Request`, so they cannot borrow the rid that request owns. `pub(crate)`
// because that enum is crate-visible — their fields stay private, so only the
// constructors below can build one.
control_messages! {
    /// The scheduler's `AbortReq`: stop generating for one rid.
    AbortReq {
        /// This server never aborts the whole queue — only the one rid.
        abort_all: bool,
        finished_reason: (),
        abort_message: (),
    }

    /// `/server_info`'s control request: a bare `BaseReq` with no extra fields.
    GetInternalStateReq {}
}

/// `SamplingParams` in its header slot, serialized as a msgspec **map**.
///
/// Python's `SamplingParams` is a plain msgspec Struct (not `array_like`), so it
/// must be a map inside the otherwise-positional header — but `rmp_serde` writes
/// every struct positionally. Carrying that as a NEWTYPE rather than a
/// `#[serde(serialize_with = ...)]` attribute keeps the requirement in the type,
/// where the wire generator can see it: the macro emits the `BaseReq` preamble by
/// hand, and a derive-only field attribute would be silently ignored there.
///
/// It used to route through `serde_json::Value`, whose `Serialize` emits a map.
/// That kept the field list in one place but cost 897 ns and 22 allocations per
/// request — 84% of the whole header encode — to build a tree walked once and
/// dropped, and it deep-cloned `custom_params` (arbitrary client JSON) on top of
/// the clone the batch fan-out already makes. [`StructAsMap`] gets the same result
/// by intercepting the one call the derive makes.
#[derive(Debug)]
pub(super) struct SamplingParamsMap<'a>(pub(super) &'a SamplingParams);

impl Serialize for SamplingParamsMap<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(StructAsMap(serializer))
    }
}

/// A `Serializer` that writes a struct as a MAP and forwards everything else
/// unchanged.
///
/// `rmp_serde`'s struct encoding is positional, which is right for the header's
/// outer struct and wrong for the one field that has to match a Python msgspec
/// `Struct`. Only the top-level `serialize_struct` is ever reached — the derive
/// hands each field VALUE to the wrapped serializer directly — so every other
/// method delegates, and the compiler enforces that none was forgotten.
struct StructAsMap<S>(S);

impl<S: serde::Serializer> serde::Serializer for StructAsMap<S> {
    type Ok = S::Ok;
    type Error = S::Error;
    type SerializeSeq = S::SerializeSeq;
    type SerializeTuple = S::SerializeTuple;
    type SerializeTupleStruct = S::SerializeTupleStruct;
    type SerializeTupleVariant = S::SerializeTupleVariant;
    type SerializeMap = S::SerializeMap;
    /// The interception: a struct is serialized through the map writer, so the
    /// derive's `serialize_field` calls land as `key: value` pairs.
    type SerializeStruct = MapAsStruct<S::SerializeMap>;
    type SerializeStructVariant = S::SerializeStructVariant;

    fn serialize_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        self.0.serialize_map(Some(len)).map(MapAsStruct)
    }

    fn serialize_bool(self, v: bool) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_bool(v)
    }
    fn serialize_i8(self, v: i8) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i8(v)
    }
    fn serialize_i16(self, v: i16) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i16(v)
    }
    fn serialize_i32(self, v: i32) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i32(v)
    }
    fn serialize_i64(self, v: i64) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_i64(v)
    }
    fn serialize_u8(self, v: u8) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u8(v)
    }
    fn serialize_u16(self, v: u16) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u16(v)
    }
    fn serialize_u32(self, v: u32) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u32(v)
    }
    fn serialize_u64(self, v: u64) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_u64(v)
    }
    fn serialize_f32(self, v: f32) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_f32(v)
    }
    fn serialize_f64(self, v: f64) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_f64(v)
    }
    fn serialize_char(self, v: char) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_char(v)
    }
    fn serialize_str(self, v: &str) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_str(v)
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_bytes(v)
    }
    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_none()
    }
    fn serialize_some<T: ?Sized + Serialize>(self, v: &T) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_some(v)
    }
    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_unit()
    }
    fn serialize_unit_struct(self, n: &'static str) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_unit_struct(n)
    }
    fn serialize_unit_variant(
        self,
        n: &'static str,
        i: u32,
        v: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_unit_variant(n, i, v)
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        n: &'static str,
        v: &T,
    ) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_newtype_struct(n, v)
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        n: &'static str,
        i: u32,
        var: &'static str,
        v: &T,
    ) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_newtype_variant(n, i, var, v)
    }
    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        self.0.serialize_seq(len)
    }
    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        self.0.serialize_tuple(len)
    }
    fn serialize_tuple_struct(
        self,
        n: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        self.0.serialize_tuple_struct(n, len)
    }
    fn serialize_tuple_variant(
        self,
        n: &'static str,
        i: u32,
        v: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        self.0.serialize_tuple_variant(n, i, v, len)
    }
    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        self.0.serialize_map(len)
    }
    fn serialize_struct_variant(
        self,
        n: &'static str,
        i: u32,
        v: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        self.0.serialize_struct_variant(n, i, v, len)
    }
}

/// Adapts a map writer to the `SerializeStruct` interface, so the derive's
/// `serialize_field(name, value)` becomes `serialize_entry(name, value)`.
struct MapAsStruct<M>(M);

impl<M: SerializeMap> SerializeStruct for MapAsStruct<M> {
    type Ok = M::Ok;
    type Error = M::Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), Self::Error> {
        self.0.serialize_entry(key, value)
    }

    /// `skip_serializing_if` reaches this instead of `serialize_field`; a map has
    /// no fixed arity, so an omitted field simply is not written.
    fn skip_field(&mut self, _key: &'static str) -> Result<(), Self::Error> {
        Ok(())
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}

/// Borrow a request as its wire struct, resolving `Option` scalars to the wire
/// defaults Python's own fields carry. Borrowed, not owned: every field is a
/// reference into `req`, so an owning `From` would return references to a
/// dropped local.
///
/// The rid comes from [`GenerateRequest::rid`] — the same value `submit` copied
/// into the owning `Request`, so the scheduler, the detok registration and
/// `meta_info.id` cannot disagree.
impl<'a> From<&'a GenerateRequest> for TokenizedGenerateReqInput<'a> {
    fn from(req: &'a GenerateRequest) -> Self {
        Self {
            rid: &req.rid,
            input_text: req.text.as_deref(),
            input_ids: (),
            input_embeds: (),
            mm_inputs: (),
            token_type_ids: (),
            sampling_params: SamplingParamsMap(&req.sampling_params),
            return_logprob: req.return_logprob.unwrap_or(false),
            logprob_start_len: req.logprob_start_len.unwrap_or(-1),
            top_logprobs_num: req.top_logprobs_num.unwrap_or(0),
            token_ids_logprob: req.token_ids_logprob.as_ref(),
            stream: req.stream,
            return_sampling_mask: false,
            return_hidden_states: req.return_hidden_states.unwrap_or(false),
        }
    }
}

impl GetInternalStateReq {
    pub fn new(rid: String) -> Self {
        Self { rid }
    }
}

impl AbortReq {
    pub fn new(rid: String, abort_all: bool) -> Self {
        Self {
            rid,
            abort_all,
            finished_reason: (),
            abort_message: (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abort_req_msgpack_shape() {
        let b = AbortReq::new("12345".into(), false).encode().unwrap();
        let val = rmpv::decode::read_value(&mut &b[..]).unwrap();
        let arr = val.as_array().expect("array");
        assert_eq!(
            arr.len(),
            6,
            "AbortReq = [tag, rid, http_ipc, abort_all, finished_reason, abort_message]"
        );
        assert_eq!(arr[0].as_str(), Some("AbortReq"));
        assert_eq!(arr[1].as_str(), Some("12345"));
        assert!(arr[2].is_nil());
        assert_eq!(arr[3].as_bool(), Some(false));
        assert!(arr[4].is_nil());
        assert!(arr[5].is_nil());
    }

    /// The header must be positionally aligned: `input_embeds` (idx 5) /
    /// `token_type_ids` (idx 7) present as nil so `sampling_params` lands at idx 8 and
    /// the array reaches msgspec's min length. Regression guard for that decode failure.
    #[test]
    fn to_header_msgpack_is_positionally_aligned() {
        let req = GenerateRequest {
            rid: "r1".into(),
            text: Some("hi".into()),
            input_ids: Some(vec![1, 2, 3]),
            sampling_params: SamplingParams {
                max_new_tokens: Some(5),
                ..Default::default()
            },
            return_logprob: Some(true),
            logprob_start_len: Some(-1),
            top_logprobs_num: Some(3),
            return_hidden_states: Some(true),
            stream: true,
            ..Default::default()
        };
        let bytes = TokenizedGenerateReqInput::from(&req).encode().unwrap();
        let val = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
        let arr = val.as_array().expect("array");
        // msgspec requires >= 14 (through `stream`); we emit 16.
        assert!(
            arr.len() >= 14,
            "header must have >=14 elements, got {}",
            arr.len()
        );
        assert_eq!(arr[0].as_str(), Some("TokenizedGenerateReqInput"));
        assert_eq!(arr[1].as_str(), Some("r1"));
        assert!(arr[5].is_nil(), "idx 5 must be input_embeds (nil)");
        assert!(arr[7].is_nil(), "idx 7 must be token_type_ids (nil)");
        assert!(arr[8].is_map(), "sampling_params must land at idx 8");
        assert_eq!(arr[9].as_bool(), Some(true), "return_logprob at idx 9");
        assert_eq!(arr[11].as_u64(), Some(3), "top_logprobs_num at idx 11");
        assert_eq!(arr[13].as_bool(), Some(true), "stream at idx 13");
        // idx 14 is `return_sampling_mask` (never client-set); a shift here would
        // silently flip the wrong scheduler field.
        assert_eq!(
            arr[14].as_bool(),
            Some(false),
            "return_sampling_mask at idx 14"
        );
        assert_eq!(
            arr[15].as_bool(),
            Some(true),
            "return_hidden_states at idx 15"
        );
    }
}
