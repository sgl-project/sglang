//! The native (pure-Rust, GIL-free) MM request driver.
//!
//! Stages mirror the Python `mm_processor` pipeline: fetch/decode
//! (`sglang_mm::common`), model preprocess (`VisionProcessor`), tokenize +
//! placeholder expansion (`common::tokens`), image-only M-RoPE — then the
//! result is parked in the [`NativeSidecar`] for the drain-time adapter.
//!
//! Scope: image-only requests with raw sources (str/bytes). Anything else —
//! video/audio, precomputed dicts, placeholder-count mismatches — returns
//! [`Outcome::Fallback`] and takes the Python path unchanged.

use rayon::prelude::*;
use sglang_mm::common::{self, fetch, tokens};
use sglang_mm::registry::{MropeItem, ProcessedImage};

use super::{NativeContext, NativeMmResult};
use crate::message::MmRequest;

pub enum Outcome {
    /// Processed natively; sidecar populated; these are the expanded ids.
    Done(Vec<i32>),
    /// Not natively processable — drive the Python handler instead.
    Fallback(String),
    /// A real per-request failure → reject as 400 (same as the Python path).
    Failed(String),
}

#[derive(Debug)]
enum NativeErr {
    Fallback(String),
    Failed(String),
}

impl From<NativeErr> for Outcome {
    fn from(e: NativeErr) -> Self {
        match e {
            NativeErr::Fallback(s) => Outcome::Fallback(s),
            NativeErr::Failed(s) => Outcome::Failed(s),
        }
    }
}

pub fn process(ctx: &NativeContext, req: &MmRequest) -> Outcome {
    match try_process(ctx, req) {
        Ok(ids) => Outcome::Done(ids),
        Err(e) => e.into(),
    }
}

fn try_process(ctx: &NativeContext, req: &MmRequest) -> Result<Vec<i32>, NativeErr> {
    let payload = parse_payload(&req.payload)?;

    // Stage 1+2+3: fetch → decode → preprocess, parallel across images.
    let processed: Vec<ProcessedImage> = common::pool().install(|| {
        payload
            .images
            .par_iter()
            .map(|src| {
                let bytes = match src {
                    ImgSrc::Str(s) => fetch::fetch_bytes(s).map_err(|e| match e {
                        fetch::FetchError::Unsupported(m) => NativeErr::Fallback(m),
                        fetch::FetchError::Failed(m) => NativeErr::Failed(m),
                    })?,
                    ImgSrc::Bytes(b) => b.clone(),
                };
                let (rgb, h, w) = common::decode_rgb(&bytes).map_err(NativeErr::Failed)?;
                ctx.pipeline
                    .processor
                    .process_image(&rgb, h, w)
                    .map_err(NativeErr::Failed)
            })
            .collect::<Result<Vec<_>, NativeErr>>()
    })?;

    // Stage 4a: the prompt ids (client-supplied, or tokenized here).
    let ids = match &payload.input_ids {
        Some(ids) if !ids.is_empty() => ids.clone(),
        _ => {
            let text = payload.text.as_deref().ok_or_else(|| {
                NativeErr::Failed("multimodal request without text or input_ids".into())
            })?;
            let tokenizer = ctx.tokenizer.as_ref().ok_or_else(|| {
                NativeErr::Failed(
                    "skip_tokenizer_init is set: multimodal text prompts require input_ids".into(),
                )
            })?;
            tokenizer
                .encode(text)
                .map_err(|e| NativeErr::Failed(e.to_string()))?
        }
    };

    // Stage 4b: expand each placeholder to its image's token count. A count
    // mismatch (exotic prompt style) falls back to the Python processor.
    let counts: Vec<usize> = processed
        .iter()
        .map(|p| ctx.pipeline.processor.tokens_per_image(&p.grid_thw))
        .collect();
    let expanded = tokens::expand_placeholders(&ids, ctx.pipeline.image_token_id, &counts)
        .map_err(NativeErr::Fallback)?;

    // Stage 4c: image-only M-RoPE from offsets + grids.
    let items: Vec<MropeItem> = expanded
        .offsets
        .iter()
        .zip(&processed)
        .map(|(&(start, end), p)| MropeItem {
            start,
            end,
            grid: p.grid_thw,
        })
        .collect();
    let (mrope, mrope_delta) = ctx
        .pipeline
        .processor
        .mrope_image_only(expanded.input_ids.len(), &items)
        .map_err(NativeErr::Failed)?;

    // Stage 5: park the buffers for the drain-time adapter — strictly before
    // `MmEncoded`, so a drained request always finds its entry.
    let mut features = Vec::with_capacity(processed.iter().map(|p| p.pixel_values.len()).sum());
    let mut grids = Vec::with_capacity(processed.len());
    for p in processed {
        features.extend_from_slice(&p.pixel_values);
        grids.push(p.grid_thw);
    }
    ctx.sidecar.lock().unwrap().insert(
        req.rid.clone(),
        NativeMmResult {
            features,
            grids,
            offsets: expanded.offsets,
            mrope,
            mrope_delta,
        },
    );
    Ok(expanded.input_ids)
}

/// One string- or bytes-typed image source (any other shape → fallback).
enum ImgSrc {
    Str(String),
    Bytes(Vec<u8>),
}

struct Payload {
    text: Option<String>,
    input_ids: Option<Vec<i32>>,
    images: Vec<ImgSrc>,
}

/// Decode the `[text, input_ids, image_data, video_data, audio_data]` msgpack
/// payload (`GenerateRequest::to_mm_payload_msgpack`).
fn parse_payload(payload: &[u8]) -> Result<Payload, NativeErr> {
    use rmpv::Value;

    let value = rmpv::decode::read_value(&mut &payload[..])
        .map_err(|e| NativeErr::Failed(format!("mm payload decode: {e}")))?;
    let Value::Array(fields) = value else {
        return Err(NativeErr::Failed("mm payload is not an array".into()));
    };
    if fields.len() != 5 {
        return Err(NativeErr::Failed("mm payload arity mismatch".into()));
    }

    if value_present(&fields[3]) || value_present(&fields[4]) {
        return Err(NativeErr::Fallback("video/audio input".into()));
    }

    let text = match &fields[0] {
        Value::Nil => None,
        Value::String(s) => s.as_str().map(str::to_owned),
        _ => return Err(NativeErr::Failed("mm payload: non-string text".into())),
    };
    let input_ids = match &fields[1] {
        Value::Nil => None,
        Value::Array(a) => Some(
            a.iter()
                .map(|v| {
                    v.as_i64()
                        .map(|i| i as i32)
                        .ok_or_else(|| NativeErr::Failed("mm payload: non-int input id".into()))
                })
                .collect::<Result<Vec<i32>, _>>()?,
        ),
        _ => return Err(NativeErr::Failed("mm payload: bad input_ids".into())),
    };

    let mut images = Vec::new();
    collect_images(&fields[2], &mut images)?;
    if images.is_empty() {
        return Err(NativeErr::Fallback("no raw image sources".into()));
    }
    Ok(Payload {
        text,
        input_ids,
        images,
    })
}

/// Flatten `image_data` (a source or a list of sources) into [`ImgSrc`]s.
/// Dicts (precomputed features) and deeper nesting fall back to Python.
fn collect_images(v: &rmpv::Value, out: &mut Vec<ImgSrc>) -> Result<(), NativeErr> {
    use rmpv::Value;
    match v {
        Value::Nil => Ok(()),
        Value::String(s) => {
            let s = s
                .as_str()
                .ok_or_else(|| NativeErr::Failed("non-utf8 image source".into()))?;
            out.push(ImgSrc::Str(s.to_owned()));
            Ok(())
        }
        Value::Binary(b) => {
            out.push(ImgSrc::Bytes(b.clone()));
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                match item {
                    Value::String(_) | Value::Binary(_) | Value::Nil => collect_images(item, out)?,
                    _ => {
                        return Err(NativeErr::Fallback("nested/typed image_data shape".into()));
                    }
                }
            }
            Ok(())
        }
        _ => Err(NativeErr::Fallback("unsupported image_data shape".into())),
    }
}

/// Python `has_valid_data` mirror: nil and (recursively) empty lists don't count.
fn value_present(v: &rmpv::Value) -> bool {
    use rmpv::Value;
    match v {
        Value::Nil => false,
        Value::Array(a) => a.iter().any(value_present),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(text: &str, image: rmpv::Value) -> Vec<u8> {
        let arr = rmpv::Value::Array(vec![
            rmpv::Value::from(text),
            rmpv::Value::Nil,
            image,
            rmpv::Value::Nil,
            rmpv::Value::Nil,
        ]);
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &arr).unwrap();
        buf
    }

    #[test]
    fn video_falls_back() {
        let arr = rmpv::Value::Array(vec![
            rmpv::Value::from("t"),
            rmpv::Value::Nil,
            rmpv::Value::Nil,
            rmpv::Value::from("http://v/v.mp4"),
            rmpv::Value::Nil,
        ]);
        let mut buf = Vec::new();
        rmpv::encode::write_value(&mut buf, &arr).unwrap();
        assert!(matches!(parse_payload(&buf), Err(NativeErr::Fallback(_))));
    }

    #[test]
    fn dict_image_falls_back() {
        let map = rmpv::Value::Map(vec![(rmpv::Value::from("format"), rmpv::Value::from("x"))]);
        let buf = payload("t", rmpv::Value::Array(vec![map]));
        assert!(matches!(parse_payload(&buf), Err(NativeErr::Fallback(_))));
    }

    #[test]
    fn string_and_list_images_parse() {
        let buf = payload("t", rmpv::Value::from("data:image/png;base64,xxxx"));
        assert_eq!(parse_payload(&buf).unwrap().images.len(), 1);
        let buf = payload(
            "t",
            rmpv::Value::Array(vec![rmpv::Value::from("a"), rmpv::Value::from("b")]),
        );
        assert_eq!(parse_payload(&buf).unwrap().images.len(), 2);
    }
}
