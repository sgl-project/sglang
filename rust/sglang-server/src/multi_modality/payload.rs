//! Convert a parked request's [`MmWorkItem`] into the typed [`MmInput`] the
//! `sglang-mm` driver consumes — an in-process handoff, nothing serialized.
//!
//! Every `Err` rejects the request back to the client; the message says whether
//! the input is malformed or merely outside the pipeline's scope (video/audio,
//! precomputed features, …).

use bytes::Bytes;
use sglang_mm::common::fetch::{ByteBudget, fetch_bytes_budgeted};
use sglang_mm::driver::{ImageSource, MmInput};

use crate::message::multimodal::MmItem;
use crate::message::request::{MmWorkItem, ProcessorExtensions};

/// Fully resolved media for a multimodal processor. I/O sources were
/// prefetched on the async API layer; data URLs and bare base64 are decoded on
/// the MM worker.
pub struct ResolvedMediaWork {
    pub text: Option<String>,
    pub input_ids: Option<Vec<i32>>,
    pub images: Vec<Bytes>,
    pub videos: Vec<Bytes>,
    pub audios: Vec<Bytes>,
    /// Request fields owned by the selected processor rather than this shared
    /// payload layer.
    pub processor_extensions: ProcessorExtensions,
}

/// Resolve all modality fields in the fixed image/video/audio prefetch order.
pub fn resolve_media_work(work: MmWorkItem) -> Result<ResolvedMediaWork, String> {
    resolve_media_work_with_budget(work, sglang_mm::driver::MAX_REQUEST_BYTES)
}

fn resolve_media_work_with_budget(
    work: MmWorkItem,
    max_request_bytes: u64,
) -> Result<ResolvedMediaWork, String> {
    let MmWorkItem {
        text,
        input_ids,
        image_data,
        video_data,
        audio_data,
        processor_extensions,
        prefetched,
        mm_hashes: _,
    } = work;
    let mut prefetched = prefetched.into_iter();
    let budget = ByteBudget::new(max_request_bytes);
    let images = collect_media(image_data, &mut prefetched, "image_data", &budget)?;
    let videos = collect_media(video_data, &mut prefetched, "video_data", &budget)?;
    let audios = collect_media(audio_data, &mut prefetched, "audio_data", &budget)?;
    if prefetched.next().is_some() {
        return Err("media prefetch produced more payloads than the request consumes".into());
    }
    Ok(ResolvedMediaWork {
        text,
        input_ids,
        images,
        videos,
        audios,
        processor_extensions,
    })
}

fn collect_media(
    items: Vec<MmItem>,
    prefetched: &mut std::vec::IntoIter<Bytes>,
    field: &str,
    budget: &ByteBudget,
) -> Result<Vec<Bytes>, String> {
    items
        .into_iter()
        .map(|item| match item {
            MmItem::Source(source) | MmItem::Ref { url: source } => {
                if is_io_source(&source) {
                    let bytes = prefetched
                        .next()
                        .ok_or_else(|| format!("I/O-backed {field} source was not prefetched"))?;
                    budget.charge_existing(bytes.len(), field)?;
                    Ok(bytes)
                } else {
                    fetch_bytes_budgeted(&source, budget).map(Bytes::from)
                }
            }
            MmItem::Preprocessed { format } => Err(format!(
                "unsupported {field} item: preprocessed `{format}` input"
            )),
        })
        .collect()
}

/// True for sources the API layer must resolve before MM dispatch: I/O — network
/// *or* disk, since a network mount can hang past any HTTP timeout — never runs
/// on the fixed MM worker pool (see `api_server::prefetch`). `data:` and bare
/// base64 are pure CPU and stay on the worker. Lives next to [`image_source`]
/// so the prefetch walk and the parse walk cannot drift.
pub fn is_io_source(src: &str) -> bool {
    src.starts_with("http://")
        || src.starts_with("https://")
        || src.starts_with("file://")
        || src.starts_with('/')
}

/// The I/O-backed sources of one modality's items, in item order.
pub fn io_sources(items: &[MmItem]) -> Vec<String> {
    items
        .iter()
        .filter_map(MmItem::source)
        .filter(|src| is_io_source(src))
        .map(str::to_owned)
        .collect()
}

/// I/O-backed sources are swapped for their `work.prefetched` bytes (in
/// [`io_sources`] order); one left without an entry is an internal error here,
/// never a fetch.
pub fn to_mm_input(work: MmWorkItem) -> Result<MmInput, String> {
    let MmWorkItem {
        text,
        input_ids,
        image_data,
        video_data,
        audio_data,
        processor_extensions,
        prefetched,
        mm_hashes: _,
    } = work;
    if !video_data.is_empty() || !audio_data.is_empty() {
        return Err("unsupported modality: video/audio input".into());
    }
    if !processor_extensions.is_empty() {
        return Err("unsupported generate extensions for this processor".into());
    }
    let mut prefetched = prefetched.iter();
    let images = image_data
        .into_iter()
        .map(|item| image_source(item, &mut prefetched))
        .collect::<Result<Vec<_>, _>>()?;
    if images.is_empty() {
        return Err("no raw image sources in mm input".into());
    }
    Ok(MmInput {
        text,
        input_ids,
        images,
    })
}

fn image_source(
    item: MmItem,
    prefetched: &mut std::slice::Iter<Bytes>,
) -> Result<ImageSource, String> {
    match item {
        MmItem::Source(source) | MmItem::Ref { url: source } => {
            if !is_io_source(&source) {
                return Ok(ImageSource::String(source));
            }
            prefetched
                .next()
                .map(|bytes| ImageSource::Bytes(bytes.to_vec()))
                .ok_or_else(|| "I/O-backed image source was not prefetched".to_string())
        }
        MmItem::Preprocessed { format } => Err(format!(
            "unsupported image_data item: preprocessed `{format}` input"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(s: &str) -> MmItem {
        MmItem::Source(s.to_owned())
    }

    fn image_work(image_data: Vec<MmItem>) -> MmWorkItem {
        MmWorkItem {
            text: Some("prompt".into()),
            image_data,
            ..Default::default()
        }
    }

    #[test]
    fn converts_source_and_ref_images() {
        let one = to_mm_input(image_work(vec![src("data:image/png;base64,x")])).unwrap();
        assert_eq!(one.images.len(), 1);
        let many =
            to_mm_input(image_work(vec![src("a"), MmItem::Ref { url: "b".into() }])).unwrap();
        assert_eq!(many.images.len(), 2);
        assert!(matches!(&many.images[1], ImageSource::String(s) if s == "b"));
    }

    #[test]
    fn unsupported_modalities_and_items_rejected() {
        let video = MmWorkItem {
            video_data: vec![src("video.mp4")],
            ..Default::default()
        };
        assert!(to_mm_input(video).err().unwrap().contains("video/audio"));

        let err = to_mm_input(image_work(vec![MmItem::Preprocessed {
            format: "processor_output".into(),
        }]))
        .err()
        .unwrap();
        assert!(err.contains("preprocessed `processor_output`"), "{err}");

        let extension = MmWorkItem {
            processor_extensions: std::iter::once((
                "multimodal_custom".to_owned(),
                rmpv::Value::Boolean(true),
            ))
            .collect(),
            ..Default::default()
        };
        assert!(
            to_mm_input(extension)
                .err()
                .unwrap()
                .contains("unsupported generate extensions")
        );
    }

    /// I/O-backed sources (URLs, file paths) take their prefetched bytes in walk
    /// order; one left unfetched errors, so no I/O can reach an MM worker.
    #[test]
    fn io_sources_use_prefetched_bytes() {
        let image = vec![
            src("http://a/x.png"),
            src("data:image/png;base64,x"),
            MmItem::Ref {
                url: "/mnt/nfs/y.png".into(),
            },
        ];
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
    fn image_free_work_rejected() {
        assert!(
            to_mm_input(MmWorkItem::default())
                .err()
                .unwrap()
                .contains("no raw image sources")
        );
    }

    #[test]
    fn resolved_media_shares_one_byte_budget_across_source_forms() {
        let work = MmWorkItem {
            image_data: vec![src("YWJjZA=="), src("ZWZnaA==")],
            ..Default::default()
        };
        let err = resolve_media_work_with_budget(work, 7).err().unwrap();
        assert!(err.contains("request media byte budget"), "{err}");

        let work = MmWorkItem {
            image_data: vec![src("YWJjZA==")],
            video_data: vec![src("ZWZnaA==")],
            ..Default::default()
        };
        let err = resolve_media_work_with_budget(work, 7).err().unwrap();
        assert!(err.contains("request media byte budget"), "{err}");
    }

    #[test]
    fn resolved_media_preserves_order_and_prefetched_allocations() {
        let image = Bytes::from(vec![1, 2]);
        let video = Bytes::from(vec![3, 4]);
        let image_ptr = image.as_ptr();
        let video_ptr = video.as_ptr();
        let work = MmWorkItem {
            image_data: vec![src("/image"), src("BQY=")],
            video_data: vec![src("https://example.test/video")],
            audio_data: vec![src("Bwg=")],
            prefetched: vec![image, video],
            ..Default::default()
        };
        let resolved = resolve_media_work_with_budget(work, 8).unwrap();
        assert_eq!(resolved.images[0].as_ptr(), image_ptr);
        assert_eq!(resolved.videos[0].as_ptr(), video_ptr);
        assert_eq!(resolved.images[1].as_ref(), [5, 6]);
        assert_eq!(resolved.audios[0].as_ref(), [7, 8]);

        for work in [
            image_work(vec![src("/missing")]),
            MmWorkItem {
                prefetched: vec![Bytes::from_static(b"extra")],
                ..Default::default()
            },
        ] {
            assert!(resolve_media_work(work).is_err());
        }
        let work = MmWorkItem {
            image_data: vec![src("/image")],
            audio_data: vec![src("Bwg=")],
            prefetched: vec![Bytes::from_static(b"1234")],
            ..Default::default()
        };
        assert!(resolve_media_work_with_budget(work, 5).is_err());
    }
}
