use super::par;

const PRECISION_BITS: i32 = 32 - 8 - 2;

/// Resampling filters, bit-exact clones of PIL's kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    /// support 3.0 — PIL `LANCZOS`.
    Lanczos,
    /// support 2.0, a = -0.5 — PIL `BICUBIC` (≈ torchvision antialiased
    /// bicubic, which the HF "fast" image processors use).
    Bicubic,
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
}

fn precompute_coeffs(in_size: usize, out_size: usize, filter: Filter) -> Coeffs {
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

    let factor = (1i64 << PRECISION_BITS) as f64;
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
    Coeffs { bounds, kk, ksize }
}

#[inline]
fn clip8(v: i32) -> u8 {
    if v >= 1 << (PRECISION_BITS + 8) {
        255
    } else if v <= 0 {
        0
    } else {
        (v >> PRECISION_BITS) as u8
    }
}

fn resample_horizontal(src: &[u8], h: usize, w: usize, out_w: usize, c: &Coeffs) -> Vec<u8> {
    let mut out = vec![0u8; h * out_w * 3];
    par::for_chunks_mut(&mut out, out_w * 3, |y, row| {
        let src_row = &src[y * w * 3..(y + 1) * w * 3];
        for xx in 0..out_w {
            let (xmin, count) = c.bounds[xx];
            let k = &c.kk[xx * c.ksize..xx * c.ksize + count];
            let mut s = [1i32 << (PRECISION_BITS - 1); 3];
            for (x, &coef) in k.iter().enumerate() {
                let p = (xmin + x) * 3;
                s[0] += src_row[p] as i32 * coef;
                s[1] += src_row[p + 1] as i32 * coef;
                s[2] += src_row[p + 2] as i32 * coef;
            }
            let o = xx * 3;
            row[o] = clip8(s[0]);
            row[o + 1] = clip8(s[1]);
            row[o + 2] = clip8(s[2]);
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
            let mut s = [1i32 << (PRECISION_BITS - 1); 3];
            for (y, &coef) in k.iter().enumerate() {
                let p = ((ymin + y) * w + x) * 3;
                s[0] += src[p] as i32 * coef;
                s[1] += src[p + 1] as i32 * coef;
                s[2] += src[p + 2] as i32 * coef;
            }
            let o = x * 3;
            row[o] = clip8(s[0]);
            row[o + 1] = clip8(s[1]);
            row[o + 2] = clip8(s[2]);
        }
    });
    out
}

/// PIL-exact separable resize of a flat HWC RGB buffer with the given filter.
///
/// Enters the fan-out pool once for both passes; the per-row `for_chunks_mut`
/// calls inside then reuse that entry rather than injecting a job per pass.
pub fn resize_rgb_filter(
    src: &[u8],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    filter: Filter,
) -> Vec<u8> {
    par::in_pool(move || resize_passes(src, h, w, out_h, out_w, filter))
}

fn resize_passes(
    src: &[u8],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
    filter: Filter,
) -> Vec<u8> {
    let need_h = out_w != w;
    let need_v = out_h != h;
    if need_h && need_v {
        let ch = precompute_coeffs(w, out_w, filter);
        let tmp = resample_horizontal(src, h, w, out_w, &ch);
        let cv = precompute_coeffs(h, out_h, filter);
        resample_vertical(&tmp, out_w, out_h, &cv)
    } else if need_h {
        let ch = precompute_coeffs(w, out_w, filter);
        resample_horizontal(src, h, w, out_w, &ch)
    } else if need_v {
        let cv = precompute_coeffs(h, out_h, filter);
        resample_vertical(src, w, out_h, &cv)
    } else {
        src.to_vec()
    }
}

pub fn resize_lanczos_rgb(src: &[u8], h: usize, w: usize, out_h: usize, out_w: usize) -> Vec<u8> {
    resize_rgb_filter(src, h, w, out_h, out_w, Filter::Lanczos)
}

/// One output coordinate's source neighbors along an axis for the torch
/// bilinear kernel: `v = (1-lambda)*src[i0] + lambda*src[i1]`.
struct BilinearTap {
    i0: usize,
    i1: usize,
    lambda: f32,
}

/// `(scale * (out_idx + 0.5) - 0.5).max(0)` per output index — torch's
/// `area_pixel_compute_source_index` (align_corners=False), in f32 like the
/// CPU kernel for float inputs.
fn bilinear_taps(in_size: usize, out_size: usize) -> Vec<BilinearTap> {
    let scale = in_size as f32 / out_size as f32;
    (0..out_size)
        .map(|o| {
            let src = (scale * (o as f32 + 0.5) - 0.5).max(0.0);
            let i0 = (src as usize).min(in_size - 1);
            BilinearTap {
                i0,
                i1: (i0 + 1).min(in_size - 1),
                lambda: src - i0 as f32,
            }
        })
        .collect()
}

/// `torch.nn.functional.interpolate(mode="bilinear", align_corners=False)`
/// equivalent on a flat HWC u8 RGB buffer, producing f32 in the source pixel
/// scale. Unlike the PIL kernels above this never antialiases on downscale —
/// torch without `antialias=True` point-samples one 2x2 neighborhood per
/// output pixel — and it never quantizes back to u8.
pub fn resize_rgb_f32_torch_bilinear(
    src: &[u8],
    h: usize,
    w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    if (out_h, out_w) == (h, w) {
        // Identity scale maps every output index onto its source index
        // (lambda 0), so torch degenerates to the plain float cast.
        return src.iter().map(|&v| v as f32).collect();
    }
    let ys = bilinear_taps(h, out_h);
    let xs = bilinear_taps(w, out_w);
    let mut out = vec![0.0f32; out_h * out_w * 3];
    par::for_chunks_mut(&mut out, out_w * 3, |yy, row| {
        let y = &ys[yy];
        let (r0, r1) = (y.i0 * w * 3, y.i1 * w * 3);
        for (xx, x) in xs.iter().enumerate() {
            for c in 0..3 {
                let v00 = src[r0 + x.i0 * 3 + c] as f32;
                let v01 = src[r0 + x.i1 * 3 + c] as f32;
                let v10 = src[r1 + x.i0 * 3 + c] as f32;
                let v11 = src[r1 + x.i1 * 3 + c] as f32;
                // Torch's accumulation order: lerp horizontally per row,
                // then vertically, all in f32.
                row[xx * 3 + c] = (1.0 - y.lambda) * ((1.0 - x.lambda) * v00 + x.lambda * v01)
                    + y.lambda * ((1.0 - x.lambda) * v10 + x.lambda * v11);
            }
        }
    });
    out
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Values from `torch.nn.functional.interpolate(mode="bilinear",
    /// align_corners=False)` run offline on a 4x6 ramp image (pixel value =
    /// flat index), down- and up-scaled. Pins the align_corners=False source
    /// mapping and the no-antialias 2x2 sampling — an antialiased or
    /// align_corners=True rewrite produces different values on both.
    #[test]
    fn torch_bilinear_matches_reference() {
        let (h, w) = (4usize, 6usize);
        let src: Vec<u8> = (0..(h * w * 3) as u8).collect();

        let down = resize_rgb_f32_torch_bilinear(&src, h, w, 2, 4);
        let ch = |out: &[f32], c: usize| -> Vec<f32> {
            out.iter().skip(c).step_by(3).copied().collect()
        };
        assert_eq!(
            ch(&down, 0),
            [9.75, 14.25, 18.75, 23.25, 45.75, 50.25, 54.75, 59.25]
        );
        assert_eq!(
            ch(&down, 2),
            [11.75, 16.25, 20.75, 25.25, 47.75, 52.25, 56.75, 61.25]
        );

        let up = resize_rgb_f32_torch_bilinear(&src, h, w, 5, 7);
        let row = |c: usize, y: usize| -> Vec<f32> {
            up[(y * 7) * 3..(y + 1) * 7 * 3]
                .iter()
                .skip(c)
                .step_by(3)
                .copied()
                .collect()
        };
        let expect_r0 = [
            0.0,
            2.357142925262451,
            4.928571701049805,
            7.5,
            10.071428298950195,
            12.642857551574707,
            15.0,
        ];
        let expect_c1_r3 = [
            42.39999771118164,
            44.75714111328125,
            47.328575134277344,
            49.89999771118164,
            52.4714241027832,
            55.0428581237793,
            57.39999771118164,
        ];
        for (got, want) in row(0, 0).iter().zip(expect_r0) {
            assert!((got - want).abs() < 1e-4, "{got} != {want}");
        }
        for (got, want) in row(1, 3).iter().zip(expect_c1_r3) {
            assert!((got - want).abs() < 1e-4, "{got} != {want}");
        }
    }

    /// Identity size must be the plain float cast (torch's identity mapping),
    /// not a resampled copy.
    #[test]
    fn torch_bilinear_identity_is_exact() {
        let src: Vec<u8> = (0..48).collect();
        let out = resize_rgb_f32_torch_bilinear(&src, 4, 4, 4, 4);
        assert_eq!(out, src.iter().map(|&v| v as f32).collect::<Vec<_>>());
    }
}
