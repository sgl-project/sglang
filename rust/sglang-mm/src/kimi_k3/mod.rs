//! Kimi-K3 image preprocessing, bit-exact against the checkpoint's PIL/numpy
//! reference (`kimi_k3_vision_processing.py`, fill stage `"after_resize"`):
//! PIL BICUBIC resize (RGBA keeps alpha) -> f32 background composite with
//! `astype(u8)` truncation -> zero-pad -> normalize (numpy's f64-through
//! in-place ops, as per-channel LUTs) -> NaViT patchify to
//! `(n_patches, 3, ps, ps)` f32. Sizing and prompt handling stay in Python.

use crate::common::par;
use crate::common::resize::{self, Filter, Resample};

/// `media_utils.TransparentBgConfig` — what to composite RGBA images onto.
#[derive(Clone, Copy, Debug)]
pub enum Background {
    White,
    Black,
    Gray,
    Chessboard {
        square_size: usize,
        /// Historical name: `True` keeps the *white* square at the top-left.
        square_on_top_left: bool,
        white_value: u8,
        gray_value: u8,
    },
}

impl Background {
    fn value_at(&self, y: usize, x: usize) -> u8 {
        match *self {
            Background::White => 255,
            Background::Black => 0,
            Background::Gray => 128,
            Background::Chessboard {
                square_size,
                square_on_top_left,
                white_value,
                gray_value,
            } => {
                let gray_parity = usize::from(square_on_top_left);
                if (y / square_size + x / square_size) % 2 == gray_parity {
                    gray_value
                } else {
                    white_value
                }
            }
        }
    }
}

/// Composite a flat HWC RGBA buffer onto `bg`: f32 alpha blend, then
/// `astype(np.uint8)` truncation. `bg == None` = `Image.convert("RGB")`,
/// which just drops the alpha channel.
pub fn composite_rgba(rgba: &[u8], h: usize, w: usize, bg: Option<Background>) -> Vec<u8> {
    let mut out = vec![0u8; h * w * 3];
    par::for_chunks_mut(&mut out, w * 3, |y, row| {
        for x in 0..w {
            let p = (y * w + x) * 4;
            let o = x * 3;
            match bg {
                None => row[o..o + 3].copy_from_slice(&rgba[p..p + 3]),
                Some(bg) => {
                    let alpha = rgba[p + 3] as f32 / 255.0;
                    let bg_v = bg.value_at(y, x) as f32;
                    for ch in 0..3 {
                        let v = alpha * rgba[p + ch] as f32 + (1.0 - alpha) * bg_v;
                        row[o + ch] = v as u8;
                    }
                }
            }
        }
    });
    out
}

/// u8 -> normalized f32 table for `media_utils.normalize`: numpy runs the
/// in-place f32 (x) f64-scalar ops through f64 and casts back each step.
fn norm_lut(mean: f64, std: f64) -> [f32; 256] {
    let std_inv = 1.0 / std;
    core::array::from_fn(|v| {
        let a = (v as f64 / 255.0) as f32;
        let b = ((a as f64) - mean) as f32;
        ((b as f64) * std_inv) as f32
    })
}

pub struct PreprocessOutput {
    /// `(n_patches, 3, ps, ps)` f32, C-order, flattened.
    pub patches: Vec<f32>,
    /// `grid_thw = (1, padded_h / ps, padded_w / ps)`.
    pub grid_thw: [i64; 3],
}

/// The full post-decode pipeline for one u8 image of `channels` 3 (RGB) or
/// 4 (RGBA, composited onto `bg` after the resize).
#[allow(clippy::too_many_arguments)]
pub fn preprocess_image(
    src: &[u8],
    h: usize,
    w: usize,
    channels: usize,
    new_w: usize,
    new_h: usize,
    pad_w: usize,
    pad_h: usize,
    patch_size: usize,
    mean: [f64; 3],
    std: [f64; 3],
    bg: Option<Background>,
) -> Result<PreprocessOutput, String> {
    if src.len() != h * w * channels {
        return Err(format!(
            "source buffer holds {} bytes, expected {h}x{w}x{channels}",
            src.len()
        ));
    }
    if new_w == 0 || new_h == 0 || patch_size == 0 {
        return Err("resized dims and patch size must be positive".into());
    }
    let (out_h, out_w) = (new_h + pad_h, new_w + pad_w);
    if out_h % patch_size != 0 || out_w % patch_size != 0 {
        return Err(format!(
            "padded dims {out_h}x{out_w} not divisible by patch size {patch_size}"
        ));
    }

    par::in_pool(move || {
        let bicubic = Resample::Pil(Filter::Bicubic);
        let rgb = match channels {
            3 => resize::resize_rgb(src, h, w, new_h, new_w, bicubic),
            4 => {
                let rgba = resize::resize_rgba(src, h, w, new_h, new_w, bicubic);
                composite_rgba(&rgba, new_h, new_w, bg)
            }
            other => return Err(format!("expected 3 or 4 channels, got {other}")),
        };

        let luts: [[f32; 256]; 3] = core::array::from_fn(|c| norm_lut(mean[c], std[c]));
        let (nph, npw) = (out_h / patch_size, out_w / patch_size);
        let patch_elems = 3 * patch_size * patch_size;
        let mut patches = vec![0f32; nph * npw * patch_elems];
        // One chunk per patch row: patches land in row-major (i * npw + j)
        // order, matching navit_patchify's reshape/transpose.
        par::for_chunks_mut(&mut patches, npw * patch_elems, |i, row| {
            for j in 0..npw {
                let patch = &mut row[j * patch_elems..(j + 1) * patch_elems];
                for (ch, lut) in luts.iter().enumerate() {
                    for py in 0..patch_size {
                        let y = i * patch_size + py;
                        for px in 0..patch_size {
                            let x = j * patch_size + px;
                            let v = if y < new_h && x < new_w {
                                rgb[(y * new_w + x) * 3 + ch]
                            } else {
                                0 // np.pad zeros run through the same normalize
                            };
                            patch[(ch * patch_size + py) * patch_size + px] = lut[v as usize];
                        }
                    }
                }
            }
        });

        Ok(PreprocessOutput {
            patches,
            grid_thw: [1, nph as i64, npw as i64],
        })
    })
}

// --- Python bindings (feature-gated: absent from the pure-Rust rlib) ---

#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use super::{Background, composite_rgba, preprocess_image};
    use crate::common::resize::{self, Filter, Resample};

    /// `(pattern, square_size, square_on_top_left, white_value, gray_value)`
    /// mirroring `TransparentBgConfig`; only chessboard reads the last four.
    type PyBg = (String, usize, bool, u8, u8);

    /// Flattened patch buffer plus its `(t, h, w)` grid.
    type PyPatches<'py> = (Bound<'py, PyArray1<f32>>, (i64, i64, i64));

    fn parse_bg(bg: Option<PyBg>) -> PyResult<Option<Background>> {
        let Some((pattern, square_size, square_on_top_left, white_value, gray_value)) = bg else {
            return Ok(None);
        };
        Ok(Some(match pattern.as_str() {
            "white" => Background::White,
            "black" => Background::Black,
            "gray" => Background::Gray,
            "chessboard" => {
                if square_size == 0 {
                    return Err(PyValueError::new_err("chessboard square size must be > 0"));
                }
                Background::Chessboard {
                    square_size,
                    square_on_top_left,
                    white_value,
                    gray_value,
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "invalid background pattern {other:?}"
                )));
            }
        }))
    }

    fn contiguous(arr: &PyReadonlyArray3<'_, u8>) -> PyResult<(Vec<u8>, usize, usize, usize)> {
        let shape = arr.shape();
        let (h, w, c) = (shape[0], shape[1], shape[2]);
        let data = arr
            .as_slice()
            .map_err(|_| PyValueError::new_err("array must be C-contiguous"))?
            .to_vec();
        Ok((data, h, w, c))
    }

    /// PIL-bit-exact `Image.resize(..., BICUBIC)` of an HWC u8 array with 3
    /// (RGB) or 4 (RGBA) channels. Exposed for the stage-parity tests.
    #[pyfunction]
    pub fn resize_bicubic<'py>(
        py: Python<'py>,
        arr: PyReadonlyArray3<'py, u8>,
        out_w: usize,
        out_h: usize,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        if out_w == 0 || out_h == 0 {
            return Err(PyValueError::new_err("output size must be positive"));
        }
        let (data, h, w, c) = contiguous(&arr)?;
        let bicubic = Resample::Pil(Filter::Bicubic);
        let out = py.detach(move || match c {
            3 => Ok(resize::resize_rgb(&data, h, w, out_h, out_w, bicubic)),
            4 => Ok(resize::resize_rgba(&data, h, w, out_h, out_w, bicubic)),
            other => Err(format!("expected 3 or 4 channels, got {other}")),
        });
        Ok(out.map_err(PyValueError::new_err)?.into_pyarray(py))
    }

    /// `fill_transparent_bg_with` on an HWC RGBA u8 array. Exposed for the
    /// stage-parity tests.
    #[pyfunction]
    #[pyo3(signature = (arr, bg=None))]
    pub fn fill_transparent_bg<'py>(
        py: Python<'py>,
        arr: PyReadonlyArray3<'py, u8>,
        bg: Option<PyBg>,
    ) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let bg = parse_bg(bg)?;
        let (data, h, w, c) = contiguous(&arr)?;
        if c != 4 {
            return Err(PyValueError::new_err(format!(
                "expected HWC RGBA array with 4 channels, got {c}"
            )));
        }
        Ok(py
            .detach(move || composite_rgba(&data, h, w, bg))
            .into_pyarray(py))
    }

    /// The full post-decode pipeline for one image; returns the flattened
    /// `(n_patches, 3, patch_size, patch_size)` f32 buffer and its
    /// `(t, h, w)` grid.
    #[pyfunction]
    #[pyo3(signature = (arr, new_w, new_h, pad_w, pad_h, patch_size, mean, std, bg=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn preprocess<'py>(
        py: Python<'py>,
        arr: PyReadonlyArray3<'py, u8>,
        new_w: usize,
        new_h: usize,
        pad_w: usize,
        pad_h: usize,
        patch_size: usize,
        mean: (f64, f64, f64),
        std: (f64, f64, f64),
        bg: Option<PyBg>,
    ) -> PyResult<PyPatches<'py>> {
        let bg = parse_bg(bg)?;
        let (data, h, w, c) = contiguous(&arr)?;
        let out = py
            .detach(move || {
                preprocess_image(
                    &data,
                    h,
                    w,
                    c,
                    new_w,
                    new_h,
                    pad_w,
                    pad_h,
                    patch_size,
                    [mean.0, mean.1, mean.2],
                    [std.0, std.1, std.2],
                    bg,
                )
            })
            .map_err(PyValueError::new_err)?;
        let [t, gh, gw] = out.grid_thw;
        Ok((out.patches.into_pyarray(py), (t, gh, gw)))
    }

    pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new(parent.py(), "kimi_k3")?;
        m.add_function(wrap_pyfunction!(resize_bicubic, &m)?)?;
        m.add_function(wrap_pyfunction!(fill_transparent_bg, &m)?)?;
        m.add_function(wrap_pyfunction!(preprocess, &m)?)?;
        parent.add_submodule(&m)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python::register;
