//! Opaque values stored on radix-tree nodes.

use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

/// Backward-compatible value type when callers omit `V`.
#[cfg(feature = "torch")]
pub type DefaultRadixValue = tch::Tensor;

/// Torch-free default used by native consumers that omit `V`.
#[cfg(not(feature = "torch"))]
pub type DefaultRadixValue = PageValue<usize>;

/// Value operations required by the radix-tree mechanism.
///
/// Implementations should make [`Self::shallow_clone`] cheap. Values are index
/// descriptors rather than mutable KV data, so sharing immutable storage is
/// safe for CPU simulation backends.
pub trait RadixValue: Debug + Sized + 'static {
    /// Number of logical radix atoms represented by this value.
    fn len(&self) -> usize;

    /// Whether the value contains no atoms.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cheap handle clone used while collecting a matched path.
    fn shallow_clone(&self) -> Self;

    /// Ownership-independent value used when the tree adopts external indices.
    /// Immutable backends may continue sharing their backing storage.
    fn copy_for_adoption(&self) -> Self;

    /// View or copy a contiguous logical range.
    fn slice(&self, start: usize, len: usize) -> Self;

    /// Split an owned value into non-overlapping logical ranges.
    fn split_owned(self, at: usize) -> (Self, Self);

    /// Concatenate values in path order.
    fn concat(values: &[Self]) -> Self;

    /// Empty host-side value. Device-specific empty values are supplied to the
    /// tree constructor by the backend.
    fn empty() -> Self;

    /// Convert indices to the representation expected by the SWA host pool.
    fn to_swa_host_indices(&self) -> Self;

    /// Convert a request slot to the representation expected by the Mamba pool.
    fn to_mamba_device_indices(&self) -> Self;

    /// Materialize integer values for optional inspection/debug APIs.
    fn to_i64_vec(&self) -> Vec<i64> {
        panic!("this value backend does not support integer inspection")
    }
}

/// Immutable, cheaply sliced list suitable for simulated KV page identifiers.
#[derive(Clone, Debug)]
pub struct PageValue<T> {
    storage: Arc<[T]>,
    range: Range<usize>,
}

// Equality is over the visible range only; two values may share or differ in
// backing storage and still represent the same pages.
impl<T: PartialEq> PartialEq for PageValue<T> {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Eq> Eq for PageValue<T> {}

impl<T> PageValue<T> {
    pub fn from_vec(values: Vec<T>) -> Self {
        let len = values.len();
        Self {
            storage: values.into(),
            range: 0..len,
        }
    }

    pub fn as_slice(&self) -> &[T] {
        &self.storage[self.range.clone()]
    }

    pub fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        self.as_slice().to_vec()
    }
}

impl<T> Default for PageValue<T> {
    fn default() -> Self {
        Self {
            storage: Arc::from([]),
            range: 0..0,
        }
    }
}

impl<T> From<Vec<T>> for PageValue<T> {
    fn from(values: Vec<T>) -> Self {
        Self::from_vec(values)
    }
}

impl<T> RadixValue for PageValue<T>
where
    T: Clone + Debug + 'static,
{
    fn len(&self) -> usize {
        self.range.len()
    }

    fn shallow_clone(&self) -> Self {
        self.clone()
    }

    fn copy_for_adoption(&self) -> Self {
        self.shallow_clone()
    }

    fn slice(&self, start: usize, len: usize) -> Self {
        assert!(start <= self.len(), "slice start exceeds value length");
        assert!(
            len <= self.len() - start,
            "slice length exceeds value length"
        );
        let absolute_start = self.range.start + start;
        Self {
            storage: Arc::clone(&self.storage),
            range: absolute_start..absolute_start + len,
        }
    }

    fn split_owned(self, at: usize) -> (Self, Self) {
        assert!(at <= self.len(), "split point exceeds value length");
        let middle = self.range.start + at;
        let head = Self {
            storage: Arc::clone(&self.storage),
            range: self.range.start..middle,
        };
        let tail = Self {
            storage: self.storage,
            range: middle..self.range.end,
        };
        (head, tail)
    }

    fn concat(values: &[Self]) -> Self {
        let Some(first) = values.first() else {
            return Self::default();
        };
        let mut end = first.range.end;
        if values.iter().skip(1).all(|value| {
            let is_contiguous =
                Arc::ptr_eq(&first.storage, &value.storage) && value.range.start == end;
            end = value.range.end;
            is_contiguous
        }) {
            return Self {
                storage: Arc::clone(&first.storage),
                range: first.range.start..end,
            };
        }

        let len = values.iter().map(Self::len).sum();
        let mut joined = Vec::with_capacity(len);
        for value in values {
            joined.extend_from_slice(value.as_slice());
        }
        Self::from_vec(joined)
    }

    fn empty() -> Self {
        Self::default()
    }

    fn to_swa_host_indices(&self) -> Self {
        self.shallow_clone()
    }

    fn to_mamba_device_indices(&self) -> Self {
        self.shallow_clone()
    }
}

#[cfg(feature = "torch")]
impl RadixValue for tch::Tensor {
    fn len(&self) -> usize {
        self.size()[0] as usize
    }

    fn shallow_clone(&self) -> Self {
        tch::Tensor::shallow_clone(self)
    }

    fn copy_for_adoption(&self) -> Self {
        self.copy()
    }

    fn slice(&self, start: usize, len: usize) -> Self {
        self.narrow(0, start as i64, len as i64)
    }

    fn split_owned(self, at: usize) -> (Self, Self) {
        let len = RadixValue::len(&self);
        assert!(0 < at && at < len, "split point must be internal");
        (
            self.narrow(0, 0, at as i64).copy(),
            self.narrow(0, at as i64, (len - at) as i64).copy(),
        )
    }

    fn concat(values: &[Self]) -> Self {
        tch::Tensor::cat(values, 0)
    }

    fn empty() -> Self {
        tch::Tensor::empty([0], (tch::Kind::Int64, tch::Device::Cpu))
    }

    fn to_swa_host_indices(&self) -> Self {
        self.to_kind(tch::Kind::Int64)
    }

    fn to_mamba_device_indices(&self) -> Self {
        self.unsqueeze(0)
    }

    fn to_i64_vec(&self) -> Vec<i64> {
        Vec::<i64>::try_from(&self.to(tch::Device::Cpu)).expect("failed to copy radix value to CPU")
    }
}

#[cfg(test)]
mod tests {
    use super::{PageValue, RadixValue};

    #[test]
    fn page_value_slices_split_and_concatenates_without_changing_order() {
        let value = PageValue::from_vec(vec![10_u64, 11, 12, 13]);
        assert_eq!(value.slice(1, 2).as_slice(), &[11, 12]);

        let value_ptr = value.as_slice().as_ptr();
        let (head, tail) = value.split_owned(2);
        assert_eq!(head.as_slice(), &[10, 11]);
        assert_eq!(tail.as_slice(), &[12, 13]);
        let joined = PageValue::concat(&[head, tail]);
        assert_eq!(joined.as_slice(), &[10, 11, 12, 13]);
        assert_eq!(joined.as_slice().as_ptr(), value_ptr);
    }

    #[test]
    fn page_value_equality_ignores_backing_storage() {
        let shared = PageValue::from_vec(vec![1_u32, 2, 3]);
        assert_eq!(shared.slice(0, 2), PageValue::from_vec(vec![1, 2]));
        assert_ne!(shared.slice(0, 2), shared.slice(1, 2));
    }
}
