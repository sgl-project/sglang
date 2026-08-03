//! Shared multimodal request driver for the server (pure-Rust) pipeline.
//!
//! Owns the request control flow — parallel fan-out, layout application,
//! failure semantics — while every model decision lives behind
//! [`MmFamilyProcessor`] (see `pipeline.rs`). Families produce data; they
//! cannot alter orchestration.

use crate::common::{self, fetch, par, token_layout};
use crate::pipeline::{DecodedMedia, MmFamilyProcessor, PositionOutput, ProcessedItem};

/// Per-request bounds: together with [`fetch::MAX_FETCH_BYTES`] they cap what
/// one request can make the pipeline buffer.
pub const MAX_ITEMS_PER_REQUEST: usize = 64;
pub const MAX_REQUEST_BYTES: u64 = 256 << 20;

/// One raw image source from the request.
#[derive(Debug)]
pub enum ImageSource {
    /// `data:`/base64/file/http — resolved by [`fetch::fetch_bytes`].
    String(String),
    /// Already-raw encoded image bytes.
    Bytes(Vec<u8>),
}

/// Typed multimodal request input. The server's message layer owns the wire
/// format and parses its payload into this before calling [`process`].
pub struct MmInput {
    pub text: Option<String>,
    pub input_ids: Option<Vec<i32>>,
    pub images: Vec<ImageSource>,
}

/// One processed media item at the request boundary.
pub struct OutputItem {
    pub feature: crate::pipeline::Tensor,
    pub aux: crate::pipeline::NamedTensors,
    /// [`common::content_hash_u64`] of the raw encoded source bytes — the same
    /// identity role as the Python path's `hash_feature`, but a different
    /// algorithm, so hashes are consistent within the server pipeline and
    /// never comparable across the two paths.
    pub hash: u64,
}

/// The per-request result parked for the scheduler drain.
pub struct Output {
    pub input_ids: Vec<i32>,
    /// In prompt order; `offsets[i]` is `items[i]`'s inclusive token range.
    pub items: Vec<OutputItem>,
    pub offsets: Vec<(u32, u32)>,
    pub positions: PositionOutput,
}

/// Resolve one image source to raw encoded bytes, borrowing when the request
/// already carries them.
fn resolve(source: &ImageSource) -> Result<std::borrow::Cow<'_, [u8]>, String> {
    match source {
        ImageSource::String(source) => Ok(fetch::fetch_bytes(source)?.into()),
        ImageSource::Bytes(bytes) => Ok(bytes.as_slice().into()),
    }
}

/// Run one request through the pipeline. Any `Err` rejects the request back
/// to the client — including inputs merely outside the pipeline's scope
/// (video/audio, precomputed features, undecodable images), since there is
/// no Python fallback path.
pub fn process(
    family: &dyn MmFamilyProcessor,
    input: MmInput,
    tokenize: impl FnOnce(&str) -> Result<Vec<i32>, String>,
) -> Result<Output, String> {
    if input.images.is_empty() {
        return Err("multimodal request without image sources".into());
    }
    if input.images.len() > MAX_ITEMS_PER_REQUEST {
        return Err(format!(
            "multimodal request exceeds {MAX_ITEMS_PER_REQUEST} media items"
        ));
    }
    // Stage 1 (fetch) is blocking I/O and runs inline, sequentially — never on
    // the CPU pool, where a slow URL would starve decode/resize for other
    // requests. Contract: callers on a fixed worker pool (the server) must
    // resolve network sources on their own I/O layer and pass `Bytes`.
    let mut fetched: Vec<std::borrow::Cow<'_, [u8]>> = Vec::with_capacity(input.images.len());
    let mut total: u64 = 0;
    for source in &input.images {
        let bytes = resolve(source)?;
        total += bytes.len() as u64;
        if total > MAX_REQUEST_BYTES {
            return Err(format!(
                "multimodal request exceeds {MAX_REQUEST_BYTES} total media bytes"
            ));
        }
        fetched.push(bytes);
    }
    let processed: Vec<(ProcessedItem, u64)> =
        par::try_map(&fetched, |bytes| -> Result<(ProcessedItem, u64), String> {
            let hash = common::content_hash_u64(bytes);
            // Inputs PIL accepts but decode_rgb refuses (e.g. 16-bit PNG)
            // error here and reject the request.
            let (rgb, height, width) = common::decode_rgb(bytes)?;
            let item = family.process_item(&DecodedMedia::Image { rgb, height, width })?;
            Ok((item, hash))
        })?;

    let input_ids = match input.input_ids {
        Some(input_ids) if !input_ids.is_empty() => input_ids,
        _ => {
            let text = input
                .text
                .as_deref()
                .ok_or("multimodal request without text or input_ids")?;
            tokenize(text)?
        }
    };
    let geometries = processed
        .iter()
        .map(|(item, _)| item.geometry.clone())
        .collect::<Vec<_>>();
    let layout = family.layout(&input_ids, &geometries)?;
    let expanded = token_layout::apply_layout(&input_ids, &layout, processed.len())?;
    let positions = family.positions(expanded.input_ids.len(), &expanded.offsets, &geometries)?;

    Ok(Output {
        input_ids: expanded.input_ids,
        items: processed
            .into_iter()
            .map(|(item, hash)| OutputItem {
                feature: item.feature,
                aux: item.aux,
                hash,
            })
            .collect(),
        offsets: expanded.offsets,
        positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::pipeline_from_spec;

    const SPEC: &str = r#"{"family":"qwen_vl","image_token_id":1,"patch_size":2,
        "merge_size":2,"temporal_patch_size":2,"min_pixels":4,
        "max_pixels":1073741824,"image_mean":[0.0,0.0,0.0],"image_std":[1.0,1.0,1.0]}"#;

    fn png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb([x as u8, y as u8, 7]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    #[test]
    fn processes_typed_image_request() {
        let family = pipeline_from_spec(SPEC).unwrap();
        let input = MmInput {
            text: None,
            input_ids: Some(vec![7, 1, 8]),
            images: vec![ImageSource::Bytes(png(8, 8))],
        };
        let out = process(family.as_ref(), input, |_| Err("no tokenizer".into())).unwrap();
        // 8x8, factor 4 → grid [1, 4, 4] → 16 patches / merge² = 4 tokens.
        assert_eq!(out.input_ids, vec![7, 1, 1, 1, 1, 8]);
        assert_eq!(out.offsets, vec![(1, 4)]);
        let item = &out.items[0];
        assert_eq!(item.feature.shape, [16, 3 * 2 * 2 * 2]);
        assert_eq!(item.aux[0].0, "image_grid_thw");
        let crate::pipeline::PositionOutput::MRope { positions, .. } = &out.positions else {
            panic!("qwen emits mrope")
        };
        assert_eq!(positions.len(), 3 * out.input_ids.len());
    }

    /// String sources — HTTP URL, `file://`, and bare path — all resolve to
    /// the same bytes and flow through the full pipeline.
    #[test]
    fn fetches_url_and_file_sources() {
        let png = png(8, 8);
        let dir = std::env::temp_dir().join(format!("sglang-mm-driver-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("img.png");
        std::fs::write(&path, &png).unwrap();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body = png.clone();
        let server = std::thread::spawn(move || {
            use std::io::{BufRead, Write};
            let (stream, _) = listener.accept().unwrap();
            let mut reader = std::io::BufReader::new(stream);
            let mut line = String::new();
            while reader.read_line(&mut line).unwrap() > 2 {
                line.clear(); // headers until the blank line
            }
            let mut stream = reader.into_inner();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(&body).unwrap();
        });

        let family = pipeline_from_spec(SPEC).unwrap();
        let input = MmInput {
            text: None,
            input_ids: Some(vec![7, 1, 1, 1, 8]),
            images: vec![
                ImageSource::String(format!("http://{addr}/img.png")),
                ImageSource::String(format!("file://{}", path.display())),
                ImageSource::String(path.display().to_string()),
            ],
        };
        let out = process(family.as_ref(), input, |_| unreachable!()).unwrap();
        server.join().unwrap();
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(out.items.len(), 3);
        // Identical source bytes → identical content hashes.
        assert_eq!(out.items[0].hash, out.items[1].hash);
        assert_eq!(out.items[1].hash, out.items[2].hash);
        assert_eq!(out.input_ids.len(), 5 + 3 * 3); // each placeholder → 4 tokens
    }

    #[test]
    fn per_request_caps_enforced() {
        let family = pipeline_from_spec(SPEC).unwrap();
        let too_many = MmInput {
            text: None,
            input_ids: Some(vec![1]),
            images: (0..=MAX_ITEMS_PER_REQUEST)
                .map(|_| ImageSource::Bytes(vec![]))
                .collect(),
        };
        let err = process(family.as_ref(), too_many, |_| unreachable!())
            .err()
            .unwrap();
        assert!(err.contains("media items"), "{err}");

        let chunk = (MAX_REQUEST_BYTES / 2 + 1) as usize;
        let too_big = MmInput {
            text: None,
            input_ids: Some(vec![1, 1]),
            images: vec![
                ImageSource::Bytes(vec![0; chunk]),
                ImageSource::Bytes(vec![0; chunk]),
            ],
        };
        let err = process(family.as_ref(), too_big, |_| unreachable!())
            .err()
            .unwrap();
        assert!(err.contains("total media bytes"), "{err}");
    }

    #[test]
    fn image_free_and_mismatched_requests_rejected() {
        let family = pipeline_from_spec(SPEC).unwrap();
        let no_images = MmInput {
            text: None,
            input_ids: Some(vec![7, 1]),
            images: vec![],
        };
        let err = process(family.as_ref(), no_images, |_| unreachable!())
            .err()
            .unwrap();
        assert!(err.contains("image sources"));
        let no_placeholder = MmInput {
            text: None,
            input_ids: Some(vec![7, 8]),
            images: vec![ImageSource::Bytes(png(8, 8))],
        };
        let err = process(family.as_ref(), no_placeholder, |_| unreachable!())
            .err()
            .unwrap();
        assert!(err.contains("placeholder"));
    }
}
