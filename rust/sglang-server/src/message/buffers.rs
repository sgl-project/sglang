//! The one Rust → scheduler data plane. Every non-scalar payload a request
//! carries — `input_ids`, `token_ids_logprob`, the multimodal feature tensors
//! and their metadata sidecar — is a named, shaped [`Buffer`] riding the ring
//! beside the msgpack scalar header, and crosses the pyo3 boundary the same
//! way: an inline buffer hands its very allocation to numpy (no copy); a shm
//! buffer hands over only its segment name, for the receiver to map after the
//! TP broadcast.
//!
//! The storage choice is the producer's, from a switch Python sets at start
//! (`MmSpec::feature_shm`); the transport itself has no policy.

use crate::utils::shm::ShmSegment;

/// Element type, named as the numpy dtype Python views the bytes with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    I64,
    F32,
    U32,
    U64,
    /// Raw BF16 bits; Python reinterprets them without a copy.
    U16,
    /// Opaque bytes — a msgpack sidecar for structured scalars.
    U8,
}

impl DType {
    pub fn numpy(self) -> &'static str {
        match self {
            DType::I64 => "int64",
            DType::F32 => "float32",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::U16 => "uint16",
            DType::U8 => "uint8",
        }
    }
}

/// Typed element storage: the inline path moves the `Vec` itself into a numpy
/// array, so the elements must keep their native type end to end.
#[derive(Debug, PartialEq)]
pub enum BufferData {
    I64(Vec<i64>),
    F32(Vec<f32>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    U16(Vec<u16>),
    U8(Vec<u8>),
}

impl BufferData {
    pub fn dtype(&self) -> DType {
        match self {
            BufferData::I64(_) => DType::I64,
            BufferData::F32(_) => DType::F32,
            BufferData::U32(_) => DType::U32,
            BufferData::U64(_) => DType::U64,
            BufferData::U16(_) => DType::U16,
            BufferData::U8(_) => DType::U8,
        }
    }

    /// Element count.
    pub fn len(&self) -> usize {
        match self {
            BufferData::I64(v) => v.len(),
            BufferData::F32(v) => v.len(),
            BufferData::U32(v) => v.len(),
            BufferData::U64(v) => v.len(),
            BufferData::U16(v) => v.len(),
            BufferData::U8(v) => v.len(),
        }
    }

    /// The elements as native-endian bytes (what a shm segment holds and what
    /// numpy views on the other side).
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            BufferData::I64(v) => bytemuck::cast_slice(v),
            BufferData::F32(v) => bytemuck::cast_slice(v),
            BufferData::U32(v) => bytemuck::cast_slice(v),
            BufferData::U64(v) => bytemuck::cast_slice(v),
            BufferData::U16(v) => bytemuck::cast_slice(v),
            BufferData::U8(v) => v,
        }
    }
}

impl From<Vec<i64>> for BufferData {
    fn from(v: Vec<i64>) -> Self {
        BufferData::I64(v)
    }
}
impl From<Vec<f32>> for BufferData {
    fn from(v: Vec<f32>) -> Self {
        BufferData::F32(v)
    }
}
impl From<Vec<u32>> for BufferData {
    fn from(v: Vec<u32>) -> Self {
        BufferData::U32(v)
    }
}
impl From<Vec<u64>> for BufferData {
    fn from(v: Vec<u64>) -> Self {
        BufferData::U64(v)
    }
}
impl From<Vec<u16>> for BufferData {
    fn from(v: Vec<u16>) -> Self {
        BufferData::U16(v)
    }
}
impl From<Vec<u8>> for BufferData {
    fn from(v: Vec<u8>) -> Self {
        BufferData::U8(v)
    }
}

/// Where a buffer's bytes live between the producer and the scheduler drain.
#[derive(Debug)]
pub enum BufferStore {
    /// In-process: the `Vec` moves through the ring and numpy takes ownership
    /// of it at the boundary. Single-rank serving, or anything small.
    Inline(BufferData),
    /// A POSIX shared-memory segment (see [`ShmSegment`]); only the name
    /// crosses to Python, which maps it on every TP rank after the broadcast.
    /// Dropped unconsumed — request aborted while parked, late result — the
    /// segment is unlinked with it.
    Shm { segment: ShmSegment, dtype: DType },
}

/// One named payload of a scheduler request.
#[derive(Debug)]
pub struct Buffer {
    /// The key the Python drain attaches it by (`input_ids`, `mm.feature.0`, …).
    pub name: String,
    /// Logical shape; numpy views the elements with it. `[len]` for a flat
    /// buffer.
    pub shape: Vec<usize>,
    pub store: BufferStore,
}

fn check_shape(shape: &[usize], len: usize) -> Result<(), String> {
    let elements = shape
        .iter()
        .try_fold(1usize, |size, &dim| size.checked_mul(dim));
    if elements != Some(len) {
        return Err(format!(
            "buffer shape {shape:?} does not hold {len} elements"
        ));
    }
    Ok(())
}

impl Buffer {
    /// A flat inline buffer.
    pub fn inline(name: impl Into<String>, data: impl Into<BufferData>) -> Self {
        let data = data.into();
        Self {
            name: name.into(),
            shape: vec![data.len()],
            store: BufferStore::Inline(data),
        }
    }

    /// An inline buffer with a logical shape (`Err` if it does not match).
    pub fn inline_shaped(
        name: impl Into<String>,
        shape: Vec<usize>,
        data: impl Into<BufferData>,
    ) -> Result<Self, String> {
        let data = data.into();
        check_shape(&shape, data.len())?;
        Ok(Self {
            name: name.into(),
            shape,
            store: BufferStore::Inline(data),
        })
    }

    /// Park `data` in a fresh POSIX segment named `segment`. `Err` leaves the
    /// caller free to fall back to inline; nothing is left behind on failure.
    pub fn shm(
        name: impl Into<String>,
        segment: String,
        shape: Vec<usize>,
        data: &BufferData,
    ) -> Result<Self, String> {
        check_shape(&shape, data.len())?;
        let segment = ShmSegment::create(segment, data.as_bytes())?;
        Ok(Self {
            name: name.into(),
            shape,
            store: BufferStore::Shm {
                segment,
                dtype: data.dtype(),
            },
        })
    }

    #[cfg(test)]
    pub fn dtype(&self) -> DType {
        match &self.store {
            BufferStore::Inline(data) => data.dtype(),
            BufferStore::Shm { dtype, .. } => *dtype,
        }
    }
}

/// Test helper: find a buffer by name.
#[cfg(test)]
pub fn find<'a>(buffers: &'a [Buffer], name: &str) -> Option<&'a Buffer> {
    buffers.iter().find(|b| b.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::shm::{shm_path, unique_name};

    /// An inline buffer keeps its typed vector; a shm buffer holds the same
    /// bytes in a segment that lives exactly as long as the buffer.
    #[test]
    fn inline_and_shm_stores_agree_on_shape_and_bytes() {
        let data = BufferData::from(vec![1.5f32, -2.0, 3.25, 4.0]);
        let inline = Buffer::inline_shaped("x", vec![2, 2], vec![1.5f32, -2.0, 3.25, 4.0]).unwrap();
        assert_eq!((inline.dtype(), &inline.shape), (DType::F32, &vec![2, 2]));
        assert_eq!(Buffer::inline("y", vec![1u8, 2, 3]).shape, [3]);

        let segment = unique_name("test");
        let shm = Buffer::shm("x", segment.clone(), vec![2, 2], &data).unwrap();
        assert_eq!((shm.dtype(), &shm.shape), (DType::F32, &vec![2, 2]));
        assert_eq!(std::fs::read(shm_path(&segment)).unwrap(), data.as_bytes());
        drop(shm);
        assert!(!shm_path(&segment).exists(), "dropping the buffer unlinks");
    }

    /// A shape that does not hold the data is rejected before anything is
    /// created, inline or shm.
    #[test]
    fn mismatched_shapes_are_rejected() {
        assert!(Buffer::inline_shaped("x", vec![3], vec![1i64, 2]).is_err());
        assert!(Buffer::inline_shaped("x", vec![usize::MAX, 2], vec![1i64, 2]).is_err());
        let segment = unique_name("test");
        assert!(Buffer::shm("x", segment.clone(), vec![3], &BufferData::I64(vec![1, 2])).is_err());
        assert!(!shm_path(&segment).exists());
    }

    #[test]
    fn dtype_names_are_numpy_dtypes() {
        assert_eq!(BufferData::from(vec![1i64]).dtype().numpy(), "int64");
        assert_eq!(BufferData::from(vec![1u32]).dtype().numpy(), "uint32");
        assert_eq!(BufferData::from(vec![1u64]).dtype().numpy(), "uint64");
        assert_eq!(BufferData::from(vec![1u16]).dtype().numpy(), "uint16");
        assert_eq!(BufferData::from(vec![1u8]).dtype().numpy(), "uint8");
        assert_eq!(BufferData::from(vec![1.0f32]).as_bytes().len(), 4);
    }
}
