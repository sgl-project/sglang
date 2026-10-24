//! Resolve a request's [`MmData`] into the encoded media bytes a multimodal
//! processor consumes. Every `Err` rejects the request back to the client.

use bytes::Bytes;

use super::fetch::{ByteBudget, MAX_REQUEST_BYTES, fetch_bytes_budgeted};

use crate::message::multimodal::{MediaHints, MmItem};
use crate::message::request::MmData;
use crate::message::types::TokenIds;

/// Fully resolved media for a multimodal processor. I/O sources were
/// prefetched on the async API layer; data URLs and bare base64 are decoded on
/// the MM worker.
pub struct ResolvedMediaWork {
    /// The prompt ids with placeholders unexpanded -- always present, the
    /// tokenizer pool having run first for a text prompt.
    pub input_ids: TokenIds,
    pub images: Vec<Bytes>,
    pub videos: Vec<Bytes>,
    pub audios: Vec<Bytes>,
    /// Each item's hints, parallel to `images` / `videos` / `audios`.
    pub image_hints: Vec<MediaHints>,
    pub video_hints: Vec<MediaHints>,
    pub audio_hints: Vec<MediaHints>,
}

/// Resolve all modality fields in the fixed image/video/audio prefetch order.
pub fn resolve_media_work(input_ids: TokenIds, mm: MmData) -> Result<ResolvedMediaWork, String> {
    resolve_media_work_with_budget(input_ids, mm, MAX_REQUEST_BYTES)
}

fn resolve_media_work_with_budget(
    input_ids: TokenIds,
    mm: MmData,
    max_request_bytes: u64,
) -> Result<ResolvedMediaWork, String> {
    let MmData {
        image_data,
        video_data,
        audio_data,
        prefetched,
        mm_hashes: _,
    } = mm;
    let mut prefetched = prefetched.into_iter();
    let budget = ByteBudget::new(max_request_bytes);
    let (images, image_hints) = collect_media(image_data, &mut prefetched, "image_data", &budget)?;
    let (videos, video_hints) = collect_media(video_data, &mut prefetched, "video_data", &budget)?;
    let (audios, audio_hints) = collect_media(audio_data, &mut prefetched, "audio_data", &budget)?;
    if prefetched.next().is_some() {
        return Err("media prefetch produced more payloads than the request consumes".into());
    }
    Ok(ResolvedMediaWork {
        input_ids,
        images,
        videos,
        audios,
        image_hints,
        video_hints,
        audio_hints,
    })
}

fn collect_media(
    items: Vec<MmItem>,
    prefetched: &mut std::vec::IntoIter<Bytes>,
    field: &str,
    budget: &ByteBudget,
) -> Result<(Vec<Bytes>, Vec<MediaHints>), String> {
    let resolved = items
        .into_iter()
        .map(|item| {
            let (source, hints) = match item {
                MmItem::Source(source) => (source, MediaHints::default()),
                MmItem::Ref { url, hints } => (url, hints),
            };
            let bytes = if is_io_source(&source) {
                let bytes = prefetched
                    .next()
                    .ok_or_else(|| format!("I/O-backed {field} source was not prefetched"))?;
                budget.charge_existing(bytes.len(), field)?;
                bytes
            } else {
                fetch_bytes_budgeted(&source, budget).map(Bytes::from)?
            };
            Ok((bytes, hints))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(resolved.into_iter().unzip())
}

/// True for sources the API layer must resolve before MM dispatch: I/O — network
/// *or* disk, since a network mount can hang past any HTTP timeout — never runs
/// on the fixed MM worker pool (see `core::prefetch`). `data:` and bare
/// base64 are pure CPU and stay on the worker. Lives next to [`collect_media`]
/// so the prefetch walk and the resolve walk cannot drift.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn src(s: &str) -> MmItem {
        MmItem::Source(s.to_owned())
    }

    const IDS: [i64; 3] = [7, 1, 8];

    fn image_work(image_data: Vec<MmItem>) -> MmData {
        MmData {
            image_data,
            ..Default::default()
        }
    }

    /// I/O-backed sources (URLs, file paths) take their prefetched bytes in walk
    /// order; one left unfetched errors, so no I/O can reach an MM worker.
    #[test]
    fn io_sources_use_prefetched_bytes() {
        let image = vec![
            src("http://a/x.png"),
            src("data:image/png;base64,eA=="),
            MmItem::Ref {
                url: "/mnt/nfs/y.png".into(),
                hints: MediaHints::default(),
            },
        ];
        assert_eq!(io_sources(&image), vec!["http://a/x.png", "/mnt/nfs/y.png"]);

        let mut work = image_work(image.clone());
        work.prefetched = vec![Bytes::from_static(b"aa"), Bytes::from_static(b"bb")];
        let resolved = resolve_media_work(IDS.to_vec(), work).unwrap();
        let images: Vec<&[u8]> = resolved.images.iter().map(|b| b.as_ref()).collect();
        assert_eq!(images, [b"aa".as_slice(), b"x", b"bb"]);

        let err = resolve_media_work(IDS.to_vec(), image_work(image))
            .err()
            .unwrap();
        assert!(err.contains("not prefetched"), "{err}");
    }

    #[test]
    fn resolved_media_shares_one_byte_budget_across_source_forms() {
        let work = MmData {
            image_data: vec![src("YWJjZA=="), src("ZWZnaA==")],
            ..Default::default()
        };
        let err = resolve_media_work_with_budget(IDS.to_vec(), work, 7)
            .err()
            .unwrap();
        assert!(err.contains("request media byte budget"), "{err}");

        let work = MmData {
            image_data: vec![src("YWJjZA==")],
            video_data: vec![src("ZWZnaA==")],
            ..Default::default()
        };
        let err = resolve_media_work_with_budget(IDS.to_vec(), work, 7)
            .err()
            .unwrap();
        assert!(err.contains("request media byte budget"), "{err}");
    }

    #[test]
    fn resolved_media_preserves_order_and_prefetched_allocations() {
        let image = Bytes::from(vec![1, 2]);
        let video = Bytes::from(vec![3, 4]);
        let image_ptr = image.as_ptr();
        let video_ptr = video.as_ptr();
        let work = MmData {
            image_data: vec![src("/image"), src("BQY=")],
            video_data: vec![MmItem::Ref {
                url: "https://example.test/video".into(),
                hints: MediaHints { fps: Some(2.0) },
            }],
            audio_data: vec![src("Bwg=")],
            prefetched: vec![image, video],
            ..Default::default()
        };
        let resolved = resolve_media_work_with_budget(IDS.to_vec(), work, 8).unwrap();
        assert_eq!(resolved.images[0].as_ptr(), image_ptr);
        assert_eq!(resolved.videos[0].as_ptr(), video_ptr);
        assert_eq!(resolved.images[1].as_ref(), [5, 6]);
        assert_eq!(resolved.audios[0].as_ref(), [7, 8]);
        // Hints stay parallel to their items; a bare source carries none.
        assert_eq!(resolved.image_hints, vec![MediaHints::default(); 2]);
        assert_eq!(resolved.video_hints[0].fps, Some(2.0));
        assert_eq!(resolved.audio_hints, vec![MediaHints::default()]);

        for work in [
            image_work(vec![src("/missing")]),
            MmData {
                prefetched: vec![Bytes::from_static(b"extra")],
                ..Default::default()
            },
        ] {
            assert!(resolve_media_work(IDS.to_vec(), work).is_err());
        }
        let work = MmData {
            image_data: vec![src("/image")],
            audio_data: vec![src("Bwg=")],
            prefetched: vec![Bytes::from_static(b"1234")],
            ..Default::default()
        };
        assert!(resolve_media_work_with_budget(IDS.to_vec(), work, 5).is_err());
    }
}
