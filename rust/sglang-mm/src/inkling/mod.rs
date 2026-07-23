use std::sync::OnceLock;

use half::bf16;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::common;

const MEAN: [f32; 3] = [
    0.48145466f64 as f32,
    0.4578275f64 as f32,
    0.40821073f64 as f32,
];
const STD: [f32; 3] = [
    0.26862954f64 as f32,
    0.2613026f64 as f32,
    0.2757771f64 as f32,
];
const INV255: f32 = (1.0f64 / 255.0f64) as f32;
const PAD_RAW: f32 = (-1.0f64 / 255.0f64) as f32;

#[inline]
fn pad_bits() -> [u16; 3] {
    core::array::from_fn(|c| bf16::from_f32((PAD_RAW - MEAN[c]) / STD[c]).to_bits())
}

fn luts() -> &'static [[u16; 256]; 3] {
    static LUTS: OnceLock<[[u16; 256]; 3]> = OnceLock::new();
    LUTS.get_or_init(|| {
        core::array::from_fn(|c| {
            core::array::from_fn(|v| {
                let raw = v as u8 as f32 * INV255;
                bf16::from_f32((raw - MEAN[c]) / STD[c]).to_bits()
            })
        })
    })
}

#[inline]
pub fn grid(h: usize, w: usize, ps: usize) -> (usize, usize) {
    ((h + ps - 1) / ps, w / ps + 1)
}

fn patchify_into(arr: &[u8], h: usize, w: usize, ps: usize, out: &mut [u16]) {
    let (_nph, npw) = grid(h, w, ps);
    let pad = pad_bits();
    let lut = luts();
    let patch_elems = ps * ps * 3;
    let row_elems = npw * patch_elems;

    let body = |(i, row): (usize, &mut [u16])| {
        let y_base = i * ps;
        for j in 0..npw {
            let x_base = j * ps;
            let chunk = &mut row[j * patch_elems..(j + 1) * patch_elems];
            for y in 0..ps {
                let iy = y_base + y;
                if iy >= h {
                    for x in 0..ps {
                        let o = (y * ps + x) * 3;
                        chunk[o..o + 3].copy_from_slice(&pad);
                    }
                    continue;
                }
                let n_real = if x_base < w { (w - x_base).min(ps) } else { 0 };
                let src = (iy * w + x_base) * 3;
                for x in 0..n_real {
                    let o = (y * ps + x) * 3;
                    let p = src + x * 3;
                    chunk[o] = lut[0][arr[p] as usize];
                    chunk[o + 1] = lut[1][arr[p + 1] as usize];
                    chunk[o + 2] = lut[2][arr[p + 2] as usize];
                }
                for x in n_real..ps {
                    let o = (y * ps + x) * 3;
                    chunk[o..o + 3].copy_from_slice(&pad);
                }
            }
        }
    };

    common::pool().install(|| {
        out.par_chunks_mut(row_elems).enumerate().for_each(body);
    });
}

fn patchify_alloc(arr: &[u8], h: usize, w: usize, ps: usize) -> Vec<u16> {
    let (nph, npw) = grid(h, w, ps);
    let mut out = vec![0u16; nph * npw * ps * ps * 3];
    patchify_into(arr, h, w, ps, &mut out);
    out
}

fn check_ps(ps: usize) -> PyResult<()> {
    if ps == 0 {
        return Err(PyValueError::new_err("patch_size must be greater than zero"));
    }
    Ok(())
}

#[pyfunction]
fn patchify_rgb<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray3<'py, u8>,
    patch_size: usize,
) -> PyResult<Bound<'py, PyArray1<u16>>> {
    check_ps(patch_size)?;
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
    let out = py.allow_threads(move || patchify_alloc(&data, h, w, patch_size));
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (data, patch_size, rescale_frac=None, rescale_cap=None))]
fn decode_patchify<'py>(
    py: Python<'py>,
    data: Vec<u8>,
    patch_size: usize,
    rescale_frac: Option<f64>,
    rescale_cap: Option<i64>,
) -> PyResult<(usize, usize, Bound<'py, PyArray1<u16>>)> {
    check_ps(patch_size)?;
    let (h, w, out) = py
        .allow_threads(move || {
            common::pool().install(|| {
                let (rgb, h, w) = common::decode_rescale(&data, rescale_frac, rescale_cap)?;
                Ok::<_, String>((h, w, patchify_alloc(&rgb, h, w, patch_size)))
            })
        })
        .map_err(PyValueError::new_err)?;
    Ok((h, w, out.into_pyarray_bound(py)))
}

#[pyfunction]
#[pyo3(signature = (datas, patch_size, rescale_frac=None, rescale_cap=None))]
fn decode_patchify_batch<'py>(
    py: Python<'py>,
    datas: Vec<Vec<u8>>,
    patch_size: usize,
    rescale_frac: Option<f64>,
    rescale_cap: Option<i64>,
) -> PyResult<Vec<(usize, usize, Bound<'py, PyArray1<u16>>)>> {
    check_ps(patch_size)?;
    let results: Vec<Result<(usize, usize, Vec<u16>), String>> =
        py.allow_threads(move || {
            common::pool().install(|| {
                datas
                    .par_iter()
                    .map(|data| {
                        let (rgb, h, w) = common::decode_rescale(data, rescale_frac, rescale_cap)?;
                        Ok((h, w, patchify_alloc(&rgb, h, w, patch_size)))
                    })
                    .collect()
            })
        });
    results
        .into_iter()
        .map(|r| {
            let (h, w, v) = r.map_err(PyValueError::new_err)?;
            Ok((h, w, v.into_pyarray_bound(py)))
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (datas, patch_size, rescale_frac=None, rescale_cap=None))]
fn preprocess_images<'py>(
    py: Python<'py>,
    datas: Vec<Vec<u8>>,
    patch_size: usize,
    rescale_frac: Option<f64>,
    rescale_cap: Option<i64>,
) -> PyResult<Vec<(usize, usize, Bound<'py, PyArray1<u16>>, u64)>> {
    check_ps(patch_size)?;
    let results: Vec<Result<(usize, usize, Vec<u16>, u64), String>> =
        py.allow_threads(move || {
            common::pool().install(|| {
                datas
                    .par_iter()
                    .map(|data| {
                        let hash = common::sha256_u64(data);
                        let (rgb, h, w) = common::decode_rescale(data, rescale_frac, rescale_cap)?;
                        Ok((h, w, patchify_alloc(&rgb, h, w, patch_size), hash))
                    })
                    .collect()
            })
        });
    results
        .into_iter()
        .map(|r| {
            let (h, w, v, hash) = r.map_err(PyValueError::new_err)?;
            Ok((h, w, v.into_pyarray_bound(py), hash))
        })
        .collect()
}

/// Struct implementing ImageProcessorSpec for Inkling.
pub struct InklingProcessor;

impl crate::registry::ImageProcessorSpec for InklingProcessor {
    fn name(&self) -> &'static str {
        "inkling"
    }

    fn preprocess_batch(
        &self,
        datas: &[Vec<u8>],
        patch_size: usize,
        rescale_frac: Option<f64>,
        rescale_cap: Option<i64>,
    ) -> Result<Vec<(usize, usize, Vec<u16>, u64)>, String> {
        if patch_size == 0 {
            return Err("patch_size must be greater than zero".into());
        }
        common::pool().install(|| {
            datas
                .par_iter()
                .map(|data| {
                    let hash = common::sha256_u64(data);
                    let (rgb, h, w) = common::decode_rescale(data, rescale_frac, rescale_cap)?;
                    Ok((h, w, patchify_alloc(&rgb, h, w, patch_size), hash))
                })
                .collect()
        })
    }
}


#[pyfunction]
#[pyo3(signature = (arr, raw_bytes, patch_size, rescale_frac=None, rescale_cap=None))]
fn rescale_patchify_hash<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray3<'py, u8>,
    raw_bytes: &[u8],
    patch_size: usize,
    rescale_frac: Option<f64>,
    rescale_cap: Option<i64>,
) -> PyResult<(usize, usize, Bound<'py, PyArray1<u16>>, u64)> {
    check_ps(patch_size)?;
    let shape = arr.shape();
    let (h, w, c) = (shape[0], shape[1], shape[2]);
    if c != 3 {
        return Err(PyValueError::new_err(format!(
            "expected HWC RGB array with 3 channels, got {c}"
        )));
    }
    let hash = common::sha256_u64(raw_bytes);
    let rgb = arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("array must be C-contiguous"))?
        .to_vec();
    let (oh, ow, out) = py.allow_threads(move || {
        common::pool().install(|| {
            let (tw, th) = common::resize::scaled_dims(w, h, rescale_frac, rescale_cap);
            let (rgb, h, w) = if (tw, th) != (w, h) {
                (common::resize::resize_lanczos_rgb(&rgb, h, w, th, tw), th, tw)
            } else {
                (rgb, h, w)
            };
            (h, w, patchify_alloc(&rgb, h, w, patch_size))
        })
    });
    Ok((oh, ow, out.into_pyarray_bound(py), hash))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "inkling")?;
    m.add_function(wrap_pyfunction!(patchify_rgb, &m)?)?;
    m.add_function(wrap_pyfunction!(decode_patchify, &m)?)?;
    m.add_function(wrap_pyfunction!(decode_patchify_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(preprocess_images, &m)?)?;
    m.add_function(wrap_pyfunction!(rescale_patchify_hash, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
