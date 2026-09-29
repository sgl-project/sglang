//! InternVL family server-pipeline image processor.
//!
//! Matches the image path of the Python `InternVLProcessor`
//! (`python/sglang/srt/multimodal/processors/internvl.py`): choose a tile grid
//! from the aspect ratio, resize and split into tiles, append an optional
//! thumbnail, normalize each tile, and expand each image slot into
//! `<img> + <IMG_CONTEXT>*N + </img>`.
//!
//! The Python processor normalizes the float tensor before its bicubic resize;
//! this version resizes the u8 buffer first and normalizes the tiles after, so
//! bit-exact parity would need a float resize path. `aux` is empty until the
//! sglang-server MM worker is wired for InternVL.

use crate::common::resize;
use crate::pipeline::{
    DecodedMedia, Geometry, MmFamilyProcessor, ProcessedItem, Segment, Tensor, TensorData,
    TokenLayout, TokenPattern,
};

/// Resolved processor params, deserialized from the Python-side spec JSON.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct InternVlSpec {
    /// Placeholder token that marks one image slot in the tokenized prompt.
    pub image_token_id: i64,
    /// `<IMG_CONTEXT>`: the vision token repeated between `<img>` and `</img>`.
    pub img_context_token_id: i64,
    pub img_start_token_id: i64,
    pub img_end_token_id: i64,
    /// Square tile edge in pixels (e.g. 448).
    pub image_size: u32,
    /// Vision tokens per tile: `(image_size // patch_size)^2 * downsample_ratio^2`.
    pub num_image_token: u32,
    /// Tile budget for the aspect-ratio search before the thumbnail.
    pub max_num: u32,
    /// Add a downscaled whole-image thumbnail when more than one tile is used.
    #[serde(default = "default_true")]
    pub use_thumbnail: bool,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

fn default_true() -> bool {
    true
}

/// One normalized tile in CHW order: `3 * image_size * image_size` floats.
type Tile = Vec<f32>;

pub struct InternVlProcessor {
    spec: InternVlSpec,
}

impl InternVlProcessor {
    pub fn new(spec: InternVlSpec) -> Result<Self, String> {
        if spec.image_size == 0 || spec.num_image_token == 0 || spec.max_num == 0 {
            return Err("intern_vl spec: sizes must be positive".into());
        }
        for (i, &s) in spec.image_std.iter().enumerate() {
            if s == 0.0 {
                return Err(format!("intern_vl spec: image_std[{i}] is zero"));
            }
        }
        Ok(Self { spec })
    }

    pub fn from_spec_json(json: &str) -> Result<Self, String> {
        let spec: InternVlSpec =
            serde_json::from_str(json).map_err(|e| format!("intern_vl spec: {e}"))?;
        Self::new(spec)
    }

    /// All candidate `(cols, rows)` tile ratios with `cols * rows <= max_num`,
    /// sorted by tile count ascending — the Python `dynamic_preprocess`
    /// `target_ratios`. Tie order among equal areas is irrelevant: the matching
    /// loop below breaks equal-diff ties by the larger tile count, and two
    /// equally reduced ratios with the same count cannot both exist.
    fn target_ratios(max_num: u32) -> Vec<(u32, u32)> {
        let mut ratios = Vec::new();
        for i in 1..=max_num {
            for j in 1..=max_num {
                if i * j <= max_num {
                    ratios.push((i, j));
                }
            }
        }
        ratios.sort_unstable_by_key(|&(i, j)| i * j);
        ratios
    }

    /// Closest aspect-ratio tile split for a `W/H` aspect ratio, mirroring the
    /// Python loop: strictly smaller diff wins; equal diffs go to more tiles.
    fn closest_ratio(aspect: f64, ratios: &[(u32, u32)]) -> (u32, u32) {
        let mut best = (1u32, 1u32);
        let mut best_diff = f64::INFINITY;
        for &(x, y) in ratios {
            let diff = (aspect - x as f64 / y as f64).abs();
            let blocks = x * y;
            let best_blocks = best.0 * best.1;
            if diff < best_diff || (diff == best_diff && blocks > best_blocks) {
                best_diff = diff;
                best = (x, y);
            }
        }
        best
    }

    /// Normalize one HWC u8 tile into a CHW f32 tile: `(v/255 - mean)/std`.
    fn normalize_tile(&self, tile: &[u8]) -> Tile {
        let size = self.spec.image_size as usize;
        let mut out = vec![0.0f32; 3 * size * size];
        let inv255 = 1.0f32 / 255.0;
        for p in 0..size * size {
            for c in 0..3 {
                let raw = tile[p * 3 + c] as f32 * inv255;
                out[c * size * size + p] =
                    (raw - self.spec.image_mean[c]) / self.spec.image_std[c];
            }
        }
        out
    }

    /// Port of the Python `dynamic_preprocess` tile pipeline, working on the
    /// decoded HWC u8 image. Returns one normalized CHW tile per patch; the
    /// thumbnail, when used, is appended last.
    fn dynamic_preprocess(&self, rgb: &[u8], h: usize, w: usize) -> Result<Vec<Tile>, String> {
        if h == 0 || w == 0 {
            return Err("intern_vl: empty image".into());
        }
        let size = self.spec.image_size as usize;
        let ratios = Self::target_ratios(self.spec.max_num);
        let (cols, rows) = Self::closest_ratio(w as f64 / h as f64, &ratios);
        let target_h = rows as usize * size;
        let target_w = cols as usize * size;
        let blocks = (rows * cols) as usize;

        let resized = resize::resize_rgb(
            rgb,
            h,
            w,
            target_h,
            target_w,
            resize::Resample::Pil(resize::Filter::Bicubic),
        );
        let mut tiles = Vec::with_capacity(blocks + usize::from(self.spec.use_thumbnail));
        for b in 0..blocks {
            let x0 = (b % cols as usize) * size;
            let y0 = (b / cols as usize) * size;
            let mut tile = vec![0u8; 3 * size * size];
            for ty in 0..size {
                let src = ((y0 + ty) * target_w + x0) * 3;
                let dst = ty * size * 3;
                tile[dst..dst + size * 3].copy_from_slice(&resized[src..src + size * 3]);
            }
            tiles.push(self.normalize_tile(&tile));
        }
        if self.spec.use_thumbnail && blocks > 1 {
            let thumb = resize::resize_rgb(
                rgb,
                h,
                w,
                size,
                size,
                resize::Resample::Pil(resize::Filter::Bicubic),
            );
            tiles.push(self.normalize_tile(&thumb));
        }
        Ok(tiles)
    }
}

impl MmFamilyProcessor for InternVlProcessor {
    fn process_item(&self, media: &DecodedMedia) -> Result<ProcessedItem, String> {
        let DecodedMedia::Image { rgb, height, width } = media;
        let tiles = self.dynamic_preprocess(rgb, *height, *width)?;
        let size = self.spec.image_size as usize;
        let per = 3 * size * size;
        let mut flat = Vec::with_capacity(tiles.len() * per);
        for tile in &tiles {
            debug_assert_eq!(tile.len(), per);
            flat.extend_from_slice(tile);
        }
        Ok(ProcessedItem {
            feature: Tensor {
                shape: vec![tiles.len(), 3, size, size],
                data: TensorData::F32(flat),
            },
            aux: Vec::new(),
            geometry: Geometry::Tiles(tiles.len() as u32),
        })
    }

    fn layout(&self, input_ids: &[i64], items: &[Geometry]) -> Result<TokenLayout, String> {
        let mut segments: Vec<Segment> = Vec::new();
        let mut text_start = 0usize;
        let mut item = 0usize;
        for (pos, &id) in input_ids.iter().enumerate() {
            if id != self.spec.image_token_id {
                continue;
            }
            let num_tiles = match items.get(item) {
                Some(Geometry::Tiles(n)) => *n,
                Some(_) => {
                    return Err(format!("intern_vl: media item {item} has non-tiles geometry"))
                }
                None => {
                    return Err(format!(
                        "intern_vl: prompt has more image placeholders than media items"
                    ))
                }
            };
            let repeat = self.spec.num_image_token as usize * num_tiles as usize;
            let mut expanded = Vec::with_capacity(repeat + 2);
            expanded.push(self.spec.img_start_token_id);
            expanded.extend(std::iter::repeat(self.spec.img_context_token_id).take(repeat));
            expanded.push(self.spec.img_end_token_id);
            segments.push(Segment::Text(text_start..pos));
            segments.push(Segment::Media {
                item,
                pattern: TokenPattern::Explicit(expanded),
            });
            text_start = pos + 1;
            item += 1;
        }
        if item != items.len() {
            return Err(format!(
                "intern_vl: prompt has {item} image placeholder(s) but {} media item(s)",
                items.len()
            ));
        }
        segments.push(Segment::Text(text_start..input_ids.len()));
        Ok(TokenLayout { segments })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::token_layout;

    fn spec() -> InternVlSpec {
        InternVlSpec {
            image_token_id: 1,
            img_context_token_id: 2,
            img_start_token_id: 3,
            img_end_token_id: 4,
            image_size: 448,
            num_image_token: 256,
            max_num: 12,
            use_thumbnail: true,
            image_mean: [0.485, 0.456, 0.406],
            image_std: [0.229, 0.224, 0.225],
        }
    }

    #[test]
    fn closest_ratio_matches_python_for_4x3() {
        let ratios = InternVlProcessor::target_ratios(12);
        // 640x480 is exactly 4:3; with max_num=12 the smallest-area 4:3 is (4,3).
        assert_eq!(
            InternVlProcessor::closest_ratio(640.0 / 480.0, &ratios),
            (4, 3)
        );
    }

    #[test]
    fn closest_ratio_picks_largest_square_for_square_image() {
        let ratios = InternVlProcessor::target_ratios(12);
        // A square image ties (diff 0) on every square ratio; more tiles wins.
        assert_eq!(InternVlProcessor::closest_ratio(1.0, &ratios), (3, 3));
    }

    #[test]
    fn dynamic_preprocess_makes_tiles() {
        // 448x896 is exactly 2:1, so under max_num=2 the split is (2, 1).
        // Thumbnail is off here to isolate the tile split.
        let mut s = spec();
        s.max_num = 2;
        s.use_thumbnail = false;
        let proc = InternVlProcessor::new(s).unwrap();
        let rgb = vec![128u8; 448 * 896 * 3];
        let item = proc
            .process_item(&DecodedMedia::Image {
                rgb,
                height: 448,
                width: 896,
            })
            .unwrap();
        assert!(matches!(item.geometry, Geometry::Tiles(2)));
        assert_eq!(item.feature.shape, vec![2, 3, 448, 448]);
        let TensorData::F32(data) = &item.feature.data else {
            panic!("intern_vl: expected f32 feature");
        };
        assert_eq!(data.len(), 2 * 3 * 448 * 448);
    }

    #[test]
    fn dynamic_preprocess_appends_thumbnail() {
        // Two tiles plus the whole-image thumbnail == three tiles total.
        let mut s = spec();
        s.max_num = 2;
        let proc = InternVlProcessor::new(s).unwrap();
        let rgb = vec![128u8; 448 * 896 * 3];
        let item = proc
            .process_item(&DecodedMedia::Image {
                rgb,
                height: 448,
                width: 896,
            })
            .unwrap();
        assert!(matches!(item.geometry, Geometry::Tiles(3)));
        assert_eq!(item.feature.shape, vec![3, 3, 448, 448]);
        let TensorData::F32(data) = &item.feature.data else {
            panic!("intern_vl: expected f32 feature");
        };
        assert_eq!(data.len(), 3 * 3 * 448 * 448);
    }

    #[test]
    fn layout_expansion_matches_reference() {
        let proc = InternVlProcessor::new(spec()).unwrap();
        // [T, <image>, T] with one 2-tile image expands to
        // [T, <img>, <IMG_CONTEXT>*512, </img>, T].
        let input_ids = vec![7, 1, 9];
        let items = vec![Geometry::Tiles(2)];
        let layout = proc.layout(&input_ids, &items).unwrap();
        let expanded = token_layout::apply_layout(&input_ids, &layout, 1).unwrap();

        let mut expected = vec![7, 3];
        expected.extend(std::iter::repeat(2).take(512));
        expected.push(4);
        expected.push(9);
        assert_eq!(expanded.input_ids, expected);
        assert_eq!(expanded.offsets, vec![(1, 514)]);
    }
}