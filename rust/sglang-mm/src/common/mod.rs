pub mod resize;
pub mod transforms;

use std::cell::RefCell;
use std::sync::OnceLock;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::env::var("SGL_MM_RS_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(8, |c| c.get().min(8)));
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("sgl-mm-{i}"))
            .build()
            .expect("failed to build rayon pool")
    })
}

// ---- Thread-local buffer pools ----

thread_local! {
    static RGB_POOL: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(1024 * 1024));
}

#[cfg(feature = "turbojpeg-decode")]
thread_local! {
    static JPEG_DECOMP: RefCell<Option<turbojpeg::Decompressor>> = const { RefCell::new(None) };
}

// ---- Format sniffing ----

#[cfg(feature = "turbojpeg-decode")]
#[inline]
fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.len() >= 3 && bytes[0] == 0xFF && bytes[1] == 0xD8 && bytes[2] == 0xFF
}

#[inline]
fn is_png(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[..8] == b"\x89PNG\r\n\x1a\n"
}

// ---- JPEG decode via libjpeg-turbo (with scaled iDCT) ----

#[cfg(feature = "turbojpeg-decode")]
fn with_jpeg_decompressor<R>(
    f: impl FnOnce(&mut turbojpeg::Decompressor) -> Result<R, String>,
) -> Result<R, String> {
    JPEG_DECOMP.with(|d| {
        let mut slot = d.borrow_mut();
        if slot.is_none() {
            *slot =
                Some(turbojpeg::Decompressor::new().map_err(|e| format!("turbojpeg init: {e}"))?);
        }
        f(slot.as_mut().unwrap())
    })
}

#[cfg(feature = "turbojpeg-decode")]
fn jpeg_header_dims(bytes: &[u8]) -> Result<(usize, usize), String> {
    with_jpeg_decompressor(|decomp| {
        let header = decomp
            .read_header(bytes)
            .map_err(|e| format!("jpeg header: {e}"))?;
        Ok((header.height, header.width))
    })
}

#[cfg(feature = "turbojpeg-decode")]
fn decode_jpeg_into(
    bytes: &[u8],
    target: Option<(usize, usize)>,
    buf: &mut Vec<u8>,
) -> Result<(usize, usize), String> {
    with_jpeg_decompressor(|decomp| {
        let header = decomp
            .read_header(bytes)
            .map_err(|e| format!("jpeg header: {e}"))?;

        let (scaled_h, scaled_w) = if let Some((th, tw)) = target {
            let factors = turbojpeg::Decompressor::supported_scaling_factors();
            let mut chosen = turbojpeg::ScalingFactor::new(1, 1);
            for sf in factors.into_iter().rev() {
                if sf.num() > sf.denom() {
                    continue;
                }
                if sf.scale(header.height) >= th && sf.scale(header.width) >= tw {
                    chosen = sf;
                    break;
                }
            }
            decomp
                .set_scaling_factor(chosen)
                .map_err(|e| format!("jpeg set_scaling_factor: {e}"))?;
            (chosen.scale(header.height), chosen.scale(header.width))
        } else {
            decomp
                .set_scaling_factor(turbojpeg::ScalingFactor::new(1, 1))
                .map_err(|e| format!("jpeg set_scaling_factor: {e}"))?;
            (header.height, header.width)
        };

        let needed = scaled_h * scaled_w * 3;
        buf.clear();
        buf.resize(needed, 0u8);

        let image = turbojpeg::Image {
            pixels: &mut buf[..needed],
            width: scaled_w,
            pitch: scaled_w * 3,
            height: scaled_h,
            format: turbojpeg::PixelFormat::RGB,
        };
        decomp
            .decompress(bytes, image)
            .map_err(|e| format!("jpeg decompress: {e}"))?;
        Ok((scaled_h, scaled_w))
    })
}

// ---- PNG streaming decode ----

fn decode_png_into(bytes: &[u8], buf: &mut Vec<u8>) -> Result<(usize, usize), String> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().map_err(|e| format!("png: {e}"))?;
    let w = reader.info().width as usize;
    let h = reader.info().height as usize;
    let (output_color, _) = reader.output_color_type();

    let needed = w * h * 3;
    buf.clear();
    buf.resize(needed, 0u8);

    let mut out_off = 0;
    while let Some(row) = reader.next_row().map_err(|e| format!("png: {e}"))? {
        let r = row.data();
        match output_color {
            png::ColorType::Rgb => {
                buf[out_off..out_off + w * 3].copy_from_slice(&r[..w * 3]);
                out_off += w * 3;
            }
            png::ColorType::Rgba => {
                for i in 0..w {
                    buf[out_off + i * 3] = r[i * 4];
                    buf[out_off + i * 3 + 1] = r[i * 4 + 1];
                    buf[out_off + i * 3 + 2] = r[i * 4 + 2];
                }
                out_off += w * 3;
            }
            png::ColorType::Grayscale => {
                for i in 0..w {
                    let v = r[i];
                    buf[out_off + i * 3] = v;
                    buf[out_off + i * 3 + 1] = v;
                    buf[out_off + i * 3 + 2] = v;
                }
                out_off += w * 3;
            }
            png::ColorType::GrayscaleAlpha => {
                for i in 0..w {
                    let v = r[i * 2];
                    buf[out_off + i * 3] = v;
                    buf[out_off + i * 3 + 1] = v;
                    buf[out_off + i * 3 + 2] = v;
                }
                out_off += w * 3;
            }
            _ => return decode_fallback_into(bytes, buf),
        }
    }
    Ok((h, w))
}

// ---- Fallback via image crate ----

fn decode_fallback_into(data: &[u8], buf: &mut Vec<u8>) -> Result<(usize, usize), String> {
    let img = image::load_from_memory(data).map_err(|e| format!("image decode: {e}"))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    let raw = rgb.into_raw();
    buf.clear();
    buf.extend_from_slice(&raw);
    Ok((h as usize, w as usize))
}

// ---- Public decode API ----

pub fn decode_rgb_into(
    data: &[u8],
    #[allow(unused_variables)] target: Option<(usize, usize)>,
    buf: &mut Vec<u8>,
) -> Result<(usize, usize), String> {
    #[cfg(feature = "turbojpeg-decode")]
    if is_jpeg(data) {
        return decode_jpeg_into(data, target, buf);
    }

    if is_png(data) {
        return decode_png_into(data, buf);
    }

    decode_fallback_into(data, buf)
}

pub fn sha256_u64(data: &[u8]) -> u64 {
    let digest = blake3::hash(data);
    u64::from_be_bytes(digest.as_bytes()[..8].try_into().unwrap())
}

pub fn decode_rgb(data: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    RGB_POOL.with(|pool| {
        let mut buf = pool.borrow_mut();
        let (h, w) = decode_rgb_into(data, None, &mut buf)?;
        let needed = h * w * 3;
        Ok((buf[..needed].to_vec(), h, w))
    })
}

pub fn decode_rescale(
    data: &[u8],
    rescale_frac: Option<f64>,
    rescale_cap: Option<i64>,
) -> Result<(Vec<u8>, usize, usize), String> {
    RGB_POOL.with(|pool| {
        let mut buf = pool.borrow_mut();

        #[cfg(feature = "turbojpeg-decode")]
        if is_jpeg(data) {
            if let Some(frac) = rescale_frac {
                let (orig_h, orig_w) = jpeg_header_dims(data)?;
                let (tw, th) = resize::scaled_dims(orig_w, orig_h, Some(frac), rescale_cap);
                if (tw, th) == (orig_w, orig_h) {
                    let (h, w) = decode_jpeg_into(data, None, &mut buf)?;
                    return Ok((buf[..h * w * 3].to_vec(), h, w));
                }
                let (dh, dw) = decode_jpeg_into(data, Some((th, tw)), &mut buf)?;
                if (dw, dh) == (tw, th) {
                    return Ok((buf[..th * tw * 3].to_vec(), th, tw));
                }
                let resized = resize::resize_lanczos_rgb(&buf[..dh * dw * 3], dh, dw, th, tw);
                return Ok((resized, th, tw));
            }
            let (h, w) = decode_jpeg_into(data, None, &mut buf)?;
            return Ok((buf[..h * w * 3].to_vec(), h, w));
        }

        let (h, w) = decode_rgb_into(data, None, &mut buf)?;
        let (tw, th) = resize::scaled_dims(w, h, rescale_frac, rescale_cap);
        if (tw, th) == (w, h) {
            return Ok((buf[..h * w * 3].to_vec(), h, w));
        }
        let resized = resize::resize_lanczos_rgb(&buf[..h * w * 3], h, w, th, tw);
        Ok((resized, th, tw))
    })
}

// --- Python-exposed functions ---

#[pyfunction]
pub fn resize_rgb<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray3<'py, u8>,
    out_w: usize,
    out_h: usize,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    if out_w == 0 || out_h == 0 {
        return Err(PyValueError::new_err("output size must be positive"));
    }
    let shape = arr.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if c != 3 {
        return Err(PyValueError::new_err(format!(
            "expected HWC RGB array with 3 channels, got {c}"
        )));
    }
    let data = arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))?
        .to_vec();
    let out =
        py.detach(move || pool().install(|| resize::resize_lanczos_rgb(&data, h, w, out_h, out_w)));
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (w, h, rescale_frac=None, rescale_cap=None))]
pub fn scaled_dims(
    w: usize,
    h: usize,
    rescale_frac: Option<f64>,
    rescale_cap: Option<i64>,
) -> (usize, usize) {
    resize::scaled_dims(w, h, rescale_frac, rescale_cap)
}

#[pyfunction]
pub fn image_decode_rgb<'py>(
    py: Python<'py>,
    data: Vec<u8>,
) -> PyResult<(usize, usize, Bound<'py, PyArray1<u8>>)> {
    let (rgb, h, w) = py
        .detach(move || decode_rgb(&data))
        .map_err(PyValueError::new_err)?;
    Ok((h, w, rgb.into_pyarray(py)))
}

#[pyfunction]
pub fn data_hash(py: Python<'_>, data: Vec<u8>) -> u64 {
    py.detach(move || {
        let digest = blake3::hash(&data);
        u64::from_be_bytes(digest.as_bytes()[..8].try_into().unwrap())
    })
}

#[pyfunction]
pub fn base64_decode<'py>(
    py: Python<'py>,
    encoded: &str,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    use base64::Engine;
    let decoded = py
        .detach(|| {
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|e| format!("base64 decode error: {e}"))
        })
        .map_err(PyValueError::new_err)?;
    Ok(pyo3::types::PyBytes::new(py, &decoded))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "common")?;
    m.add_function(wrap_pyfunction!(resize_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(scaled_dims, &m)?)?;
    m.add_function(wrap_pyfunction!(image_decode_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(data_hash, &m)?)?;
    m.add_function(wrap_pyfunction!(base64_decode, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
