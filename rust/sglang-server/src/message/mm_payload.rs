//! Convert a parked request's [`MmWorkItem`] into the typed [`MmInput`] the
//! `sglang-mm` driver consumes — an in-process handoff, nothing serialized.
//!
//! Every `Err` rejects the request back to the client; the message says whether
//! the input is malformed or merely outside the pipeline's scope (video/audio,
//! precomputed features, …).

use bytes::Bytes;
use rmpv::Value;
use sglang_mm::driver::{ImageSource, MmInput};

use super::request::MmWorkItem;

/// Fully resolved media payload for an external native MM component. All I/O
/// sources were prefetched on the async API layer; data URLs and bare base64
/// are decoded here on the MM worker.
pub struct ResolvedMediaWork {
    pub text: Option<String>,
    pub input_ids: Option<Vec<i32>>,
    pub images: Vec<Vec<u8>>,
    pub videos: Vec<Vec<u8>>,
    pub audios: Vec<Vec<u8>>,
    pub multimodal_placeholders: Option<Value>,
}

/// Resolve image/video/audio values in the prefetch contract's fixed modality
/// order. This is the generic loader seam used by external processors.
pub fn resolve_media_work(work: MmWorkItem) -> Result<ResolvedMediaWork, String> {
    let mut prefetched = work.prefetched.iter();
    let images = collect_media(work.image_data.as_ref(), &mut prefetched, "image_data")?;
    let videos = collect_media(work.video_data.as_ref(), &mut prefetched, "video_data")?;
    let audios = collect_media(work.audio_data.as_ref(), &mut prefetched, "audio_data")?;
    if prefetched.next().is_some() {
        return Err("media prefetch produced more payloads than the request consumes".into());
    }
    Ok(ResolvedMediaWork {
        text: work.text,
        input_ids: work.input_ids,
        images,
        videos,
        audios,
        multimodal_placeholders: work.multimodal_placeholders,
    })
}

fn collect_media(
    value: Option<&Value>,
    prefetched: &mut std::slice::Iter<Bytes>,
    field: &str,
) -> Result<Vec<Vec<u8>>, String> {
    fn one(
        value: &Value,
        prefetched: &mut std::slice::Iter<Bytes>,
        field: &str,
        out: &mut Vec<Vec<u8>>,
    ) -> Result<(), String> {
        match value {
            Value::Nil => Ok(()),
            Value::String(value) => {
                let source = value
                    .as_str()
                    .ok_or_else(|| format!("non-utf8 {field} source"))?;
                if is_io_source(source) {
                    out.push(
                        prefetched
                            .next()
                            .ok_or_else(|| format!("I/O-backed {field} source was not prefetched"))?
                            .to_vec(),
                    );
                } else {
                    out.push(sglang_mm::common::fetch::fetch_bytes(source)?);
                }
                Ok(())
            }
            Value::Binary(bytes) => {
                out.push(bytes.clone());
                Ok(())
            }
            Value::Array(values) => {
                for value in values {
                    one(value, prefetched, field, out)?;
                }
                Ok(())
            }
            Value::Map(entries) => {
                let source = entries
                    .iter()
                    .find(|(key, _)| key.as_str() == Some("url"))
                    .map(|(_, value)| value)
                    .ok_or_else(|| format!("{field} object requires a url field"))?;
                one(source, prefetched, field, out)
            }
            _ => Err(format!("unsupported {field} shape")),
        }
    }

    let mut out = Vec::new();
    if let Some(value) = value {
        one(value, prefetched, field, &mut out)?;
    }
    Ok(out)
}

/// True for sources the API layer must resolve before MM dispatch: I/O — network
/// *or* disk, since a network mount can hang past any HTTP timeout — never runs
/// on the fixed MM worker pool (see `api_server::prefetch`). `data:` and bare
/// base64 are pure CPU and stay on the worker. Lives next to `collect_images` so
/// the prefetch walk and the parse walk cannot drift.
pub fn is_io_source(src: &str) -> bool {
    src.starts_with("http://")
        || src.starts_with("https://")
        || src.starts_with("file://")
        || src.starts_with('/')
}

/// The I/O-backed sources of an `image_data` value, in `collect_images` order.
pub fn io_sources(value: &Value) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(value: &Value, out: &mut Vec<String>) {
        match value {
            Value::String(value) => {
                if let Some(src) = value.as_str().filter(|s| is_io_source(s)) {
                    out.push(src.to_owned());
                }
            }
            Value::Array(values) => values.iter().for_each(|value| walk(value, out)),
            Value::Map(entries) => {
                if let Some((_, source)) =
                    entries.iter().find(|(key, _)| key.as_str() == Some("url"))
                {
                    walk(source, out);
                }
            }
            _ => {}
        }
    }
    walk(value, &mut out);
    out
}

/// How many media items an `image_data` value contributes, walked the way
/// [`collect_images`] walks it, so the item budget can reject before fetching.
pub fn item_count(value: &Value) -> usize {
    match value {
        Value::Nil => 0,
        Value::Array(values) => values.iter().map(item_count).sum(),
        _ => 1,
    }
}

/// I/O-backed sources are swapped for their `work.prefetched` bytes (in
/// [`io_sources`] order); one left without an entry is an internal error here,
/// never a fetch.
pub fn to_mm_input(work: MmWorkItem) -> Result<MmInput, String> {
    let present = |v: &Option<Value>| v.as_ref().is_some_and(value_present);
    if present(&work.video_data) || present(&work.audio_data) {
        return Err("unsupported modality: video/audio input".into());
    }
    if present(&work.multimodal_placeholders) {
        return Err("unsupported multimodal_placeholders contract for this processor".into());
    }
    let mut images = Vec::new();
    if let Some(image_data) = &work.image_data {
        collect_images(image_data, &mut work.prefetched.iter(), &mut images)?;
    }
    if images.is_empty() {
        return Err("no raw image sources in mm input".into());
    }
    Ok(MmInput {
        text: work.text,
        input_ids: work.input_ids,
        images,
    })
}

fn collect_images(
    value: &Value,
    prefetched: &mut std::slice::Iter<Bytes>,
    out: &mut Vec<ImageSource>,
) -> Result<(), String> {
    match value {
        Value::Nil => Ok(()),
        Value::String(value) => {
            let value = value
                .as_str()
                .ok_or_else(|| "non-utf8 image source".to_string())?;
            if is_io_source(value) {
                let bytes = prefetched
                    .next()
                    .ok_or_else(|| "I/O-backed image source was not prefetched".to_string())?;
                out.push(ImageSource::Bytes(bytes.to_vec()));
            } else {
                out.push(ImageSource::String(value.to_owned()));
            }
            Ok(())
        }
        Value::Binary(value) => {
            out.push(ImageSource::Bytes(value.clone()));
            Ok(())
        }
        Value::Array(values) => {
            for value in values {
                match value {
                    Value::String(_) | Value::Binary(_) | Value::Nil => {
                        collect_images(value, prefetched, out)?
                    }
                    _ => {
                        return Err("unsupported image_data shape: nested/typed item".into());
                    }
                }
            }
            Ok(())
        }
        _ => Err("unsupported image_data shape".into()),
    }
}

/// Rust mirror of Python `has_valid_data`: `nil` and (recursively) empty or
/// all-nil lists don't count as multimodal input. Shared with the ingress
/// `has_multimodal` check so routing and parsing cannot drift.
pub fn value_present(value: &Value) -> bool {
    match value {
        Value::Nil => false,
        Value::Array(values) => values.iter().any(value_present),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_work(image: Value) -> MmWorkItem {
        MmWorkItem {
            text: Some("prompt".into()),
            image_data: Some(image),
            ..Default::default()
        }
    }

    #[test]
    fn converts_string_and_list_images() {
        let one = to_mm_input(image_work(Value::from("data:image/png;base64,x"))).unwrap();
        assert_eq!(one.images.len(), 1);
        let many = to_mm_input(image_work(Value::Array(vec![
            Value::from("a"),
            Value::from("b"),
        ])))
        .unwrap();
        assert_eq!(many.images.len(), 2);
    }

    #[test]
    fn unsupported_modalities_and_shapes_rejected() {
        let video = MmWorkItem {
            video_data: Some(Value::from("video.mp4")),
            ..Default::default()
        };
        assert!(to_mm_input(video).err().unwrap().contains("video/audio"));

        let dict = Value::Map(vec![(Value::from("format"), Value::from("x"))]);
        assert!(
            to_mm_input(image_work(Value::Array(vec![dict])))
                .err()
                .unwrap()
                .contains("image_data shape")
        );
    }

    #[test]
    fn empty_video_audio_lists_are_not_modalities() {
        // Mirrors Python `has_valid_data`: nil / empty lists don't count.
        let work = MmWorkItem {
            input_ids: Some(vec![1]),
            image_data: Some(Value::from("a")),
            video_data: Some(Value::Array(vec![])),
            audio_data: Some(Value::Array(vec![Value::Array(vec![])])),
            ..Default::default()
        };
        assert_eq!(to_mm_input(work).unwrap().images.len(), 1);
    }

    /// I/O-backed sources (URLs, file paths) take their prefetched bytes in walk
    /// order; one left unfetched errors, so no I/O can reach an MM worker.
    #[test]
    fn io_sources_use_prefetched_bytes() {
        let image = Value::Array(vec![
            Value::from("http://a/x.png"),
            Value::from("data:image/png;base64,x"),
            Value::from("/mnt/nfs/y.png"),
        ]);
        assert_eq!(io_sources(&image), vec!["http://a/x.png", "/mnt/nfs/y.png"]);

        let mut work = image_work(image.clone());
        work.prefetched = vec![Bytes::from_static(b"aa"), Bytes::from_static(b"bb")];
        let input = to_mm_input(work).unwrap();
        let as_bytes = |i: usize| match &input.images[i] {
            ImageSource::Bytes(b) => b.as_slice(),
            other => panic!("expected bytes, got {other:?}"),
        };
        assert_eq!(as_bytes(0), b"aa");
        assert_eq!(as_bytes(2), b"bb");
        assert!(matches!(&input.images[1], ImageSource::String(_)));

        let err = to_mm_input(image_work(image)).err().unwrap();
        assert!(err.contains("not prefetched"), "{err}");
    }

    #[test]
    fn generic_media_accepts_structured_url_items() {
        let structured = |url: &str| {
            Value::Map(vec![
                (Value::from("url"), Value::from(url)),
                (Value::from("max_dynamic_patch"), Value::from(16)),
            ])
        };
        let work = MmWorkItem {
            image_data: Some(structured("https://example.test/image.png")),
            video_data: Some(structured("file:///tmp/video.mp4")),
            audio_data: Some(Value::from("data:audio/wav;base64,YQ==")),
            prefetched: vec![Bytes::from_static(b"image"), Bytes::from_static(b"video")],
            ..Default::default()
        };

        assert_eq!(
            io_sources(work.image_data.as_ref().unwrap()),
            vec!["https://example.test/image.png"]
        );
        let resolved = resolve_media_work(work).unwrap();
        assert_eq!(resolved.images, vec![b"image".to_vec()]);
        assert_eq!(resolved.videos, vec![b"video".to_vec()]);
        assert_eq!(resolved.audios, vec![b"a".to_vec()]);
    }

    #[test]
    fn image_free_work_rejected() {
        assert!(
            to_mm_input(image_work(Value::Nil))
                .err()
                .unwrap()
                .contains("no raw image sources")
        );
        assert!(
            to_mm_input(MmWorkItem::default())
                .err()
                .unwrap()
                .contains("no raw image sources")
        );
    }
}
