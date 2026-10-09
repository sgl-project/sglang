//! V4.1 PIL bicubic resize, centered padding and CHW patch packing.
use crate::common::{par, resize};
use half::bf16;

pub struct ImagePlan {
    pub out_h: usize,
    pub out_w: usize,
    pub resize_h: usize,
    pub resize_w: usize,
    pub top: usize,
    pub left: usize,
    pub patch_size: usize,
}

pub fn resize_patchify(
    rgb: &[u8],
    h: usize,
    w: usize,
    plan: ImagePlan,
) -> Result<Vec<u16>, String> {
    let ImagePlan {
        out_h,
        out_w,
        resize_h,
        resize_w,
        top,
        left,
        patch_size: ps,
    } = plan;
    if ps == 0
        || h == 0
        || w == 0
        || resize_h == 0
        || resize_w == 0
        || out_h == 0
        || out_w == 0
        || !out_h.is_multiple_of(ps)
        || !out_w.is_multiple_of(ps)
        || resize_h > out_h
        || resize_w > out_w
        || top > out_h - resize_h
        || left > out_w - resize_w
        || h.checked_mul(w).and_then(|n| n.checked_mul(3)) != Some(rgb.len())
    {
        return Err("invalid V4.1 image geometry".into());
    }
    let len = out_h
        .checked_mul(out_w)
        .and_then(|n| n.checked_mul(3))
        .ok_or("V4.1 output size overflow")?;
    let resized = resize::resize_rgb(
        rgb,
        h,
        w,
        resize_h,
        resize_w,
        resize::Resample::Pil(resize::Filter::Bicubic),
    );
    let lut: [u16; 256] =
        core::array::from_fn(|i| bf16::from_f32(((i as f32 / 255.0) - 0.5) / 0.5).to_bits());
    let mut out = vec![0u16; len];
    let patch_len = 3 * ps * ps;
    let grid_w = out_w / ps;
    par::for_chunks_mut(&mut out, patch_len, |index, patch| {
        let py = index / grid_w * ps;
        let px = index % grid_w * ps;
        for c in 0..3 {
            for y in 0..ps {
                for x in 0..ps {
                    let iy = py + y;
                    let ix = px + x;
                    let value =
                        if iy >= top && iy < top + resize_h && ix >= left && ix < left + resize_w {
                            resized[((iy - top) * resize_w + ix - left) * 3 + c]
                        } else {
                            127
                        };
                    patch[(c * ps + y) * ps + x] = lut[value as usize];
                }
            }
        }
    });
    Ok(out)
}

#[cfg(feature = "python")]
mod python {
    use numpy::{IntoPyArray, PyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
    use pyo3::{exceptions::PyValueError, prelude::*};

    #[pyfunction]
    fn resize_patchify<'py>(
        py: Python<'py>,
        arr: PyReadonlyArray3<'py, u8>,
        output_size: (usize, usize),
        resize_size: (usize, usize),
        padding_start: (usize, usize),
        patch_size: usize,
    ) -> PyResult<Bound<'py, PyArray1<u16>>> {
        let (out_h, out_w) = output_size;
        let (resize_h, resize_w) = resize_size;
        let (top, left) = padding_start;
        let shape = arr.shape();
        let (h, w) = (shape[0], shape[1]);
        if shape[2] != 3 {
            return Err(PyValueError::new_err("expected HWC RGB"));
        }
        let data = arr
            .as_slice()
            .map_err(|_| PyValueError::new_err("expected contiguous RGB"))?
            .to_vec();
        let bits = py
            .detach(move || {
                super::resize_patchify(
                    &data,
                    h,
                    w,
                    super::ImagePlan {
                        out_h,
                        out_w,
                        resize_h,
                        resize_w,
                        top,
                        left,
                        patch_size,
                    },
                )
            })
            .map_err(PyValueError::new_err)?;
        Ok(bits.into_pyarray(py))
    }
    pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
        let m = PyModule::new(parent.py(), "dsv41")?;
        m.add_function(wrap_pyfunction!(resize_patchify, &m)?)?;
        parent.add_submodule(&m)
    }
}
#[cfg(feature = "python")]
pub use python::register;
