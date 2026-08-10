use super::par;

/// PIL's `PRECISION_BITS` for 8-bit images: weights quantized to i32.
const PIL_PRECISION_BITS: u32 = 32 - 8 - 2;

/// Resampling filters, bit-exact clones of PIL's kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// support 3.0 — PIL `LANCZOS`.
    Lanczos,
    /// support 2.0, a = -0.5 — PIL `BICUBIC`.
    Bicubic,
}

/// A resampler reproduced bit-exactly. Both share PIL's geometry, kernels and
/// per-pass u8 rounding, and differ only in how the weights are quantized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resample {
    /// PIL `Image.resize`, i32 weights.
    Pil(Filter),
    /// ATen's uint8 antialias bicubic — torchvision `resize(antialias=True)` on
    /// a uint8 tensor. i16 weights, so it rounds unlike `Pil(Bicubic)`.
    AtenU8,
}

impl Resample {
    fn filter(self) -> Filter {
        match self {
            Resample::Pil(filter) => filter,
            Resample::AtenU8 => Filter::Bicubic,
        }
    }

    /// Fixed-point precision for one axis's already-normalized weights. ATen
    /// (`_compute_weights_precision`) takes the widest that stays inside i16.
    fn precision(self, weights: &[f64]) -> u32 {
        match self {
            Resample::Pil(_) => PIL_PRECISION_BITS,
            Resample::AtenU8 => {
                let wmax = weights.iter().fold(0.0f64, |m, w| m.max(w.abs()));
                (1..PIL_PRECISION_BITS)
                    .take_while(|&p| (0.5 + wmax * (1u64 << p) as f64) < (1 << 15) as f64)
                    .last()
                    .unwrap_or(1)
            }
        }
    }
}

impl Filter {
    fn support(self) -> f64 {
        match self {
            Filter::Lanczos => 3.0,
            Filter::Bicubic => 2.0,
        }
    }

    fn eval(self, x: f64) -> f64 {
        match self {
            Filter::Lanczos => lanczos(x),
            Filter::Bicubic => bicubic(x),
        }
    }
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let x = x * std::f64::consts::PI;
    x.sin() / x
}

fn lanczos(x: f64) -> f64 {
    if (-3.0..3.0).contains(&x) {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

struct Coeffs {
    bounds: Vec<(usize, usize)>,
    kk: Vec<i32>,
    ksize: usize,
    prec: u32,
}

fn precompute_coeffs(in_size: usize, out_size: usize, resample: Resample) -> Coeffs {
    let filter = resample.filter();
    let scale = in_size as f64 / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = filter.support() * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let ss = 1.0 / filterscale;

    let mut kkf = vec![0.0f64; out_size * ksize];
    let mut bounds = vec![(0usize, 0usize); out_size];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5) as i32;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i32;
        if xmax > in_size as i32 {
            xmax = in_size as i32;
        }
        let count = (xmax - xmin) as usize;
        let k = &mut kkf[xx * ksize..(xx + 1) * ksize];
        let mut ww = 0.0f64;
        for (x, kv) in k[..count].iter_mut().enumerate() {
            let w = filter.eval((x as f64 + xmin as f64 - center + 0.5) * ss);
            *kv = w;
            ww += w;
        }
        if ww != 0.0 {
            for kv in k[..count].iter_mut() {
                *kv /= ww;
            }
        }
        bounds[xx] = (xmin as usize, count);
    }

    let prec = resample.precision(&kkf);
    let factor = (1i64 << prec) as f64;
    let kk = kkf
        .iter()
        .map(|&v| {
            if v < 0.0 {
                (-0.5 + v * factor) as i32
            } else {
                (0.5 + v * factor) as i32
            }
        })
        .collect();
    Coeffs {
        bounds,
        kk,
        ksize,
        prec,
    }
}

#[inline]
fn clip8(v: i32, prec: u32) -> u8 {
    if v >= 1 << (prec + 8) {
        255
    } else if v <= 0 {
        0
    } else {
        (v >> prec) as u8
    }
}

fn resample_horizontal(src: &[u8], h: usize, w: usize, out_w: usize, c: &Coeffs) -> Vec<u8> {
    let mut out = vec![0u8; h * out_w * 3];
    par::for_chunks_mut(&mut out, out_w * 3, |y, row| {
        let src_row = &src[y * w * 3..(y + 1) * w * 3];
        for xx in 0..out_w {
            let (xmin, count) = c.bounds[xx];
            let k = &c.kk[xx * c.ksize..xx * c.ksize + count];
            let mut s = [1i32 << (c.prec - 1); 3];
            for (x, &coef) in k.iter().enumerate() {
                let p = (xmin + x) * 3;
                s[0] += src_row[p] as i32 * coef;
                s[1] += src_row[p + 1] as i32 * coef;
                s[2] += src_row[p + 2] as i32 * coef;
            }
            let o = xx * 3;
            row[o] = clip8(s[0], c.prec);
            row[o + 1] = clip8(s[1], c.prec);
            row[o + 2] = clip8(s[2], c.prec);
        }
    });
    out
}

fn resample_vertical(src: &[u8], w: usize, out_h: usize, c: &Coeffs) -> Vec<u8> {
    let mut out = vec![0u8; out_h * w * 3];
    par::for_chunks_mut(&mut out, w * 3, |yy, row| {
        let (ymin, count) = c.bounds[yy];
        let k = &c.kk[yy * c.ksize..yy * c.ksize + count];
        for x in 0..w {
            let mut s = [1i32 << (c.prec - 1); 3];
            for (y, &coef) in k.iter().enumerate() {
                let p = ((ymin + y) * w + x) * 3;
                s[0] += src[p] as i32 * coef;
                s[1] += src[p + 1] as i32 * coef;
                s[2] += src[p + 2] as i32 * coef;
            }
            let o = x * 3;
            row[o] = clip8(s[0], c.prec);
            row[o + 1] = clip8(s[1], c.prec);
            row[o + 2] = clip8(s[2], c.prec);
        }
    });
    out
}

/// Separable resize of a flat HWC RGB buffer, bit-exact against `resample`.
///
/// Enters the fan-out pool once for both passes; the per-row `for_chunks_mut`
/// calls inside then reuse that entry rather than injecting a job per pass.
pub fn resize_rgb(
    src: &[u8],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    resample: Resample,
) -> Vec<u8> {
    par::in_pool(move || resize_passes(src, h, w, out_h, out_w, resample))
}

fn resize_passes(
    src: &[u8],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    resample: Resample,
) -> Vec<u8> {
    // Per-axis coefficients — and, under `AtenU8`, a per-axis precision.
    let coeffs = |in_size, out_size| precompute_coeffs(in_size, out_size, resample);
    match (out_w != w, out_h != h) {
        (true, true) => {
            let tmp = resample_horizontal(src, h, w, out_w, &coeffs(w, out_w));
            resample_vertical(&tmp, out_w, out_h, &coeffs(h, out_h))
        }
        (true, false) => resample_horizontal(src, h, w, out_w, &coeffs(w, out_w)),
        (false, true) => resample_vertical(src, w, out_h, &coeffs(h, out_h)),
        (false, false) => src.to_vec(),
    }
}

pub fn resize_lanczos_rgb(src: &[u8], h: usize, w: usize, out_h: usize, out_w: usize) -> Vec<u8> {
    resize_rgb(src, h, w, out_h, out_w, Resample::Pil(Filter::Lanczos))
}

pub fn scaled_dims(w: usize, h: usize, frac: Option<f64>, cap: Option<i64>) -> (usize, usize) {
    let Some(frac) = frac else {
        return (w, h);
    };
    let long_edge = w.max(h);
    if long_edge == 0 {
        return (w, h);
    }
    let mut target = long_edge as f64 * frac;
    if let Some(cap) = cap {
        let effective_cap = cap.max(long_edge as i64);
        target = target.min(effective_cap as f64);
    }
    let ratio = target / long_edge as f64;
    if ratio == 1.0 {
        return (w, h);
    }
    let scale = |v: usize| ((v as f64 * ratio + 0.5).floor() as i64).max(1) as usize;
    (scale(w), scale(h))
}
