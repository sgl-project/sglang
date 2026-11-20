/// Typed tensor payload. Grows a variant per dtype actually produced by a
/// family — not speculatively.
pub enum TensorData {
    F32(Vec<f32>),
    I64(Vec<i64>),
    /// Raw BF16 bits, exposed to numpy as u16 without copying the allocation.
    Bf16(Vec<u16>),
}

impl TensorData {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(data) => data.len(),
            Self::I64(data) => data.len(),
            Self::Bf16(data) => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: TensorData,
}
