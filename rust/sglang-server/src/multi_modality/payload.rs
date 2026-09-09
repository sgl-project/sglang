//! Convert a parked request's [`MmWorkItem`] into the typed [`MmInput`] the
//! `sglang-mm` driver consumes — an in-process handoff, nothing serialized.
//!
//! Every `Err` rejects the request back to the client; the message says whether
//! the input is malformed or merely outside the pipeline's scope (video/audio,
//! precomputed features, …).

use bytes::Bytes;
use sglang_mm::driver::{ImageSource, MmInput};

use crate::message::multimodal::MmItem;
use crate::message::request::MmWorkItem;

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
        prefetched,
        mm_hashes: _,
    } = work;
    if !video_data.is_empty() || !audio_data.is_empty() {
        return Err("unsupported modality: video/audio input".into());
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
}
