//! Qwen-VL smart_resize port. Matches HF transformers Qwen2VLImageProcessor.smart_resize.
//!
//! The critical bit: Python's `round()` is banker's rounding (half-to-even). Rust's
//! default `f64::round` is half-away-from-zero. Mismatch by 1 in either dim corrupts
//! `image_grid_thw` and downstream token count.

/// Round half-to-even (banker's rounding), matching Python `round(x)` semantics.
fn round_half_to_even(x: f64) -> f64 {
    let floor = x.floor();
    let diff = x - floor;
    if diff < 0.5 {
        floor
    } else if diff > 0.5 {
        floor + 1.0
    } else {
        // exactly .5 — go to nearest even
        if (floor as i64) % 2 == 0 {
            floor
        } else {
            floor + 1.0
        }
    }
}

/// Round x up to nearest multiple of factor.
fn round_up(x: f64, factor: u32) -> u32 {
    let f = factor as f64;
    (x / f).ceil() as u32 * factor
}

/// Round x down to nearest multiple of factor.
fn round_down(x: f64, factor: u32) -> u32 {
    let f = factor as f64;
    (x / f).floor() as u32 * factor
}

/// Round x to nearest multiple of factor using banker's rounding.
fn round_to_multiple(x: f64, factor: u32) -> u32 {
    let f = factor as f64;
    (round_half_to_even(x / f) as u32) * factor
}

#[derive(Debug, Clone, Copy)]
pub struct SmartResizeCfg {
    pub factor: u32,      // patch_size * merge_size (e.g. 14*2 = 28)
    pub min_pixels: u32,  // e.g. 56*56*4 = 12544
    pub max_pixels: u32,  // e.g. 28*28*1280 = 1003520
    pub max_ratio: f64,   // e.g. 200.0
}

impl Default for SmartResizeCfg {
    fn default() -> Self {
        // Qwen2-VL defaults
        Self {
            factor: 28,
            min_pixels: 56 * 56 * 4,
            max_pixels: 28 * 28 * 1280,
            max_ratio: 200.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Target {
    pub h: u32,
    pub w: u32,
}

impl Target {
    pub fn num_patches(&self, patch_size: u32) -> u32 {
        (self.h / patch_size) * (self.w / patch_size)
    }
    pub fn grid_h(&self, patch_size: u32) -> u32 {
        self.h / patch_size
    }
    pub fn grid_w(&self, patch_size: u32) -> u32 {
        self.w / patch_size
    }
}

pub fn smart_resize(h: u32, w: u32, cfg: SmartResizeCfg) -> anyhow::Result<Target> {
    if h == 0 || w == 0 {
        anyhow::bail!("zero dimension");
    }
    let (hf, wf) = (h as f64, w as f64);
    let ratio = hf.max(wf) / hf.min(wf);
    if ratio > cfg.max_ratio {
        anyhow::bail!("aspect ratio {ratio} exceeds {}", cfg.max_ratio);
    }

    let mut h_bar = round_to_multiple(hf, cfg.factor).max(cfg.factor);
    let mut w_bar = round_to_multiple(wf, cfg.factor).max(cfg.factor);

    let area = (h_bar as u64) * (w_bar as u64);

    if area > cfg.max_pixels as u64 {
        let beta = ((hf * wf) / cfg.max_pixels as f64).sqrt();
        h_bar = round_down(hf / beta, cfg.factor).max(cfg.factor);
        w_bar = round_down(wf / beta, cfg.factor).max(cfg.factor);
    } else if area < cfg.min_pixels as u64 {
        let beta = (cfg.min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = round_up(hf * beta, cfg.factor);
        w_bar = round_up(wf * beta, cfg.factor);
    }

    Ok(Target { h: h_bar, w: w_bar })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banker_rounding() {
        assert_eq!(round_half_to_even(0.5), 0.0);
        assert_eq!(round_half_to_even(1.5), 2.0);
        assert_eq!(round_half_to_even(2.5), 2.0);
        assert_eq!(round_half_to_even(3.5), 4.0);
        assert_eq!(round_half_to_even(0.49999), 0.0);
        assert_eq!(round_half_to_even(0.50001), 1.0);
    }

    #[test]
    fn smart_resize_typical() {
        // 1024x768 - well within bounds, just snap to factor=28
        let t = smart_resize(768, 1024, SmartResizeCfg::default()).unwrap();
        // 768 / 28 = 27.43 → 27 → 756; 1024 / 28 = 36.57 → 37 → 1036
        assert_eq!(t.h, 756);
        assert_eq!(t.w, 1036);
        assert!((t.h as u64) * (t.w as u64) <= 28 * 28 * 1280);
    }

    #[test]
    fn smart_resize_too_big() {
        // 4096x3072 - over max_pixels, must downscale
        let t = smart_resize(3072, 4096, SmartResizeCfg::default()).unwrap();
        assert!((t.h as u64) * (t.w as u64) <= (28u64 * 28 * 1280));
        assert_eq!(t.h % 28, 0);
        assert_eq!(t.w % 28, 0);
    }
}
