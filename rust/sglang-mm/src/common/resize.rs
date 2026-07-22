use rayon::prelude::*;

const PRECISION_BITS: i32 = 32 - 8 - 2;

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

struct Coeffs {
    bounds: Vec<(usize, usize)>,
    kk: Vec<i32>,
    ksize: usize,
}

fn precompute_coeffs(in_size: usize, out_size: usize) -> Coeffs {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = 3.0 * filterscale;
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
        for x in 0..count {
            let w = lanczos((x as f64 + xmin as f64 - center + 0.5) * ss);
            k[x] = w;
            ww += w;
        }
        if ww != 0.0 {
            for x in 0..count {
                k[x] /= ww;
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
    out.par_chunks_mut(out_w * 3)
        .enumerate()
        .for_each(|(y, row)| {
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
    out.par_chunks_mut(w * 3).enumerate().for_each(|(yy, row)| {
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

pub fn resize_lanczos_rgb(src: &[u8], h: usize, w: usize, out_h: usize, out_w: usize) -> Vec<u8> {
    let need_h = out_w != w;
    let need_v = out_h != h;
    if need_h && need_v {
        let ch = precompute_coeffs(w, out_w);
        let tmp = resample_horizontal(src, h, w, out_w, &ch);
        let cv = precompute_coeffs(h, out_h);
        resample_vertical(&tmp, out_w, out_h, &cv)
    } else if need_h {
        let ch = precompute_coeffs(w, out_w);
        resample_horizontal(src, h, w, out_w, &ch)
    } else if need_v {
        let cv = precompute_coeffs(h, out_h);
        resample_vertical(src, w, out_h, &cv)
    } else {
        src.to_vec()
    }
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
