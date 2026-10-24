//! The feature tensors every multimodal processor hands the worker, built-in
//! and external alike (`MmProcessor`), and their conversion from Dynamo's.

use sglang_processor::dynamo_multimodal::processor as dynamo;

/// Typed tensor payload. Grows a variant per dtype a processor produces.
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

impl TryFrom<dynamo::Tensor> for Tensor {
    type Error = String;

    fn try_from(tensor: dynamo::Tensor) -> Result<Self, String> {
        let data = match tensor.data {
            dynamo::TensorData::F32(values) => TensorData::F32(values),
            dynamo::TensorData::I64(values) => TensorData::I64(values),
            _ => return Err("mm: unsupported tensor dtype from dynamo-multimodal".into()),
        };
        Ok(Self {
            shape: tensor.shape,
            data,
        })
    }
}
