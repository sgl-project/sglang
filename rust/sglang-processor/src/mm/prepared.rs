//! Transport-independent structural contract for prepared multimodal tensors.
//!
//! [`PreparedInput::check`] validates encoding, shapes, sizes, names, and span
//! geometry only. Passing that check does **not** admit a profile, modality, or
//! tensor name for serving, and does **not** perform model-semantic validation,
//! layout, positions, hashing, or scheduler construction.
//!
//! [`crate::MmProcessor::finish_prepared`] is the generic completion bridge that
//! turns already-validated canonical
//! [`ProcessedItem`](dynamo_multimodal::processor::ProcessedItem) values plus a
//! checked view into [`crate::MmOutput`]. Profile-owned semantic import remains
//! outside this module; neither API grants automatic prepared serving.
//!
//! [`CheckedPreparedInput::decode_tensors`] converts one item's already-checked
//! little-endian payloads into native [`Tensor`] values. It never picks a "main"
//! feature by hardcoded name.

use dynamo_multimodal::processor::{Modality, NamedTensors, Tensor, TensorData};

use crate::ProcessorError;

/// Dtypes this boundary can decode into native [`TensorData`].
///
/// These variants mirror current `TensorData` support (`F32`, `I64`), not a
/// universal ML dtype catalog.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDtype {
    Float32,
    Int64,
}

impl TensorDtype {
    fn width(self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Int64 => 8,
        }
    }
}

/// One named tensor viewed as little-endian contiguous bytes.
///
/// The byte slice is borrowed and treated as immutable for the lifetime of
/// validation and conversion. This type does not describe protobuf, shared
/// memory, or any wire mapping. Borrowing does not establish cross-process
/// immutability or a shared-memory access/lifetime protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedTensor<'a> {
    pub name: &'a str,
    pub dtype: TensorDtype,
    pub shape: &'a [u64],
    pub data: &'a [u8],
}

/// Half-open `[start, end)` feature positions declared by the caller.
///
/// This describes declared spans only. It is not a guarantee that the span
/// length equals pixel rows, patch counts, or model feature-token counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenSpan {
    pub start: u32,
    pub end: u32,
}

/// One media item: modality, named tensors (including auxiliaries), and spans.
///
/// Tensor names are not arbitrary model kwargs. Structural uniqueness is
/// checked per item; processor-level allowlisting remains a later semantic
/// step. The same name may appear on different items.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedItem<'a> {
    pub modality: Modality,
    pub tensors: &'a [PreparedTensor<'a>],
    pub spans: &'a [TokenSpan],
}

/// A prepared multimodal request view before structural admission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedInput<'a> {
    pub profile: &'a str,
    pub processor_fingerprint: &'a str,
    pub input_ids: &'a [i32],
    pub items: &'a [PreparedItem<'a>],
}

/// Caller-supplied structural budgets. There is no unbounded default.
///
/// Satisfying these limits does not grant support for a profile or modality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedLimits {
    pub max_tensor_bytes: usize,
    pub max_items: usize,
    pub max_tensors_per_item: usize,
    pub max_rank: usize,
    pub max_spans_per_item: usize,
    pub max_input_tokens: usize,
}

/// Opaque handle produced by [`PreparedInput::check`].
///
/// **Warning:** existence of this value means only that structural checks
/// passed (encoding sizes, ranks, names, span geometry, and the supplied
/// budgets). It does **not** grant semantic safety, processor admission, or
/// serving support for any family. [`crate::MmProcessor::finish_prepared`]
/// still requires the profile caller to finish model-semantic validation
/// first; this handle is not a universal external-tensor admission proof.
#[derive(Clone, Debug)]
pub struct CheckedPreparedInput<'a> {
    input: PreparedInput<'a>,
    total_tensor_bytes: usize,
}

impl<'a> PreparedInput<'a> {
    /// Structurally validate this borrowed view against `expected_*` and
    /// `limits`.
    ///
    /// The expected strings must come from trusted processor/service
    /// configuration, not from this request. Equality is a compatibility guard,
    /// not proof of preprocessing provenance or semantic correctness.
    ///
    /// On success, every request byte budget has been checked and no tensor
    /// payload has been decoded yet. On failure, no [`CheckedPreparedInput`] is
    /// returned.
    pub fn check(
        self,
        expected_profile: &str,
        expected_fingerprint: &str,
        limits: &PreparedLimits,
    ) -> Result<CheckedPreparedInput<'a>, ProcessorError> {
        if expected_profile.is_empty() {
            return Err("expected prepared profile must be nonempty".into());
        }
        if expected_fingerprint.is_empty() {
            return Err("expected prepared processor fingerprint must be nonempty".into());
        }
        if self.profile != expected_profile {
            return Err(format!(
                "prepared profile mismatch: got {:?}, expected {:?}",
                self.profile, expected_profile
            )
            .into());
        }
        if self.processor_fingerprint != expected_fingerprint {
            return Err(format!(
                "prepared processor fingerprint mismatch: got {:?}, expected {:?}",
                self.processor_fingerprint, expected_fingerprint
            )
            .into());
        }

        if self.input_ids.is_empty() {
            return Err("prepared input_ids must be nonempty".into());
        }
        if self.input_ids.len() > limits.max_input_tokens {
            return Err(format!(
                "prepared input_ids length {} exceeds max_input_tokens {}",
                self.input_ids.len(),
                limits.max_input_tokens
            )
            .into());
        }
        for (index, &id) in self.input_ids.iter().enumerate() {
            if id < 0 {
                return Err(format!("prepared input_ids[{index}] is negative ({id})").into());
            }
        }

        if self.items.is_empty() {
            return Err("prepared input must contain at least one item".into());
        }
        if self.items.len() > limits.max_items {
            return Err(format!(
                "prepared item count {} exceeds max_items {}",
                self.items.len(),
                limits.max_items
            )
            .into());
        }

        let input_len = u32::try_from(self.input_ids.len()).map_err(|_| {
            "prepared input_ids length does not fit in u32 for span bounds".to_owned()
        })?;
        let mut total_tensor_bytes = 0usize;
        let mut total_spans = 0usize;

        for (item_index, item) in self.items.iter().enumerate() {
            if item.tensors.is_empty() {
                return Err(format!("prepared item {item_index} has no tensors").into());
            }
            if item.tensors.len() > limits.max_tensors_per_item {
                return Err(format!(
                    "prepared item {item_index} tensor count {} exceeds max_tensors_per_item {}",
                    item.tensors.len(),
                    limits.max_tensors_per_item
                )
                .into());
            }

            let mut seen_names: Vec<&str> = Vec::with_capacity(item.tensors.len());
            for (tensor_index, tensor) in item.tensors.iter().enumerate() {
                if tensor.name.is_empty() {
                    return Err(format!(
                        "prepared item {item_index} tensor {tensor_index} has an empty name"
                    )
                    .into());
                }
                if seen_names.contains(&tensor.name) {
                    return Err(format!(
                        "prepared item {item_index} has duplicate tensor name {:?}",
                        tensor.name
                    )
                    .into());
                }
                seen_names.push(tensor.name);

                if tensor.shape.len() > limits.max_rank {
                    return Err(format!(
                        "prepared item {item_index} tensor {:?} rank {} exceeds max_rank {}",
                        tensor.name,
                        tensor.shape.len(),
                        limits.max_rank
                    )
                    .into());
                }

                let elements = element_count(tensor.shape)?;
                let expected_bytes =
                    elements.checked_mul(tensor.dtype.width()).ok_or_else(|| {
                        format!(
                            "prepared item {item_index} tensor {:?} byte length overflow",
                            tensor.name
                        )
                    })?;
                if tensor.data.len() != expected_bytes {
                    return Err(format!(
                        "prepared item {item_index} tensor {:?} byte length {} does not match shape/dtype size {}",
                        tensor.name,
                        tensor.data.len(),
                        expected_bytes
                    )
                    .into());
                }
                total_tensor_bytes = total_tensor_bytes.checked_add(expected_bytes).ok_or_else(|| {
                    format!(
                        "prepared total tensor byte sum overflow at item {item_index} tensor {:?}",
                        tensor.name
                    )
                })?;
                if total_tensor_bytes > limits.max_tensor_bytes {
                    return Err(format!(
                        "prepared total tensor bytes {total_tensor_bytes} exceeds max_tensor_bytes {}",
                        limits.max_tensor_bytes
                    )
                    .into());
                }
            }

            if item.spans.is_empty() {
                return Err(format!("prepared item {item_index} has no spans").into());
            }
            if item.spans.len() > limits.max_spans_per_item {
                return Err(format!(
                    "prepared item {item_index} span count {} exceeds max_spans_per_item {}",
                    item.spans.len(),
                    limits.max_spans_per_item
                )
                .into());
            }
            total_spans = total_spans
                .checked_add(item.spans.len())
                .ok_or_else(|| format!("prepared span count overflow at item {item_index}"))?;

            for (span_index, span) in item.spans.iter().enumerate() {
                if span.start >= span.end {
                    return Err(format!(
                        "prepared item {item_index} span {span_index} is empty or reversed ({}..{})",
                        span.start, span.end
                    )
                    .into());
                }
                if span.end > input_len {
                    return Err(format!(
                        "prepared item {item_index} span {span_index} end {} exceeds input_ids length {input_len}",
                        span.end
                    )
                    .into());
                }
            }
            for pair in item.spans.windows(2) {
                if pair[0].start >= pair[1].start {
                    return Err(format!(
                        "prepared item {item_index} spans are not in increasing start order"
                    )
                    .into());
                }
            }
        }

        for pair in self.items.windows(2) {
            if pair[0].spans[0].start >= pair[1].spans[0].start {
                return Err("prepared items are not ordered by increasing first-span start".into());
            }
        }

        // Capacity is bounded by already-checked per-item span counts.
        let mut ordered: Vec<TokenSpan> = Vec::new();
        ordered.try_reserve(total_spans).map_err(|_| {
            "prepared span scratch allocation failed within checked span limits".to_owned()
        })?;
        for item in self.items {
            ordered.extend_from_slice(item.spans);
        }
        ordered.sort_by_key(|span| span.start);
        for pair in ordered.windows(2) {
            if pair[0].end > pair[1].start {
                return Err(format!(
                    "prepared spans overlap: {}..{} and {}..{}",
                    pair[0].start, pair[0].end, pair[1].start, pair[1].end
                )
                .into());
            }
        }

        Ok(CheckedPreparedInput {
            input: self,
            total_tensor_bytes,
        })
    }
}

impl<'a> CheckedPreparedInput<'a> {
    /// The structurally checked borrowed input view.
    pub fn input(&self) -> &PreparedInput<'a> {
        &self.input
    }

    /// Sum of all tensor payload sizes that passed structural checks.
    pub fn total_tensor_bytes(&self) -> usize {
        self.total_tensor_bytes
    }

    /// Decode one item's tensors into native [`NamedTensors`].
    ///
    /// Returns every named tensor for the item. Callers must not treat any
    /// hardcoded name as the main feature here; feature selection stays with
    /// later processor semantics.
    pub fn decode_tensors(&self, item_index: usize) -> Result<NamedTensors, ProcessorError> {
        let item = self.input.items.get(item_index).ok_or_else(|| {
            format!(
                "prepared item index {item_index} is out of range for {} items",
                self.input.items.len()
            )
        })?;
        let mut out = Vec::with_capacity(item.tensors.len());
        for tensor in item.tensors {
            out.push((tensor.name.to_owned(), decode_tensor(tensor)));
        }
        Ok(out)
    }
}

fn element_count(shape: &[u64]) -> Result<usize, ProcessorError> {
    // Check every dimension even for an empty tensor: the native shape must be
    // representable. A zero dimension makes the product zero regardless of its
    // position, so do not overflow on an earlier nonzero prefix.
    for &dim in shape {
        usize::try_from(dim)
            .map_err(|_| format!("prepared tensor shape dimension {dim} does not fit in usize"))?;
    }
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim as usize)
            .ok_or_else(|| "prepared tensor element count overflows usize".into())
    })
}

fn decode_tensor(tensor: &PreparedTensor<'_>) -> Tensor {
    // Only callable through CheckedPreparedInput; all dimensions and payload
    // lengths were checked before allocating or reading any tensor payload.
    let shape = tensor.shape.iter().map(|&dim| dim as usize).collect();
    let data = match tensor.dtype {
        TensorDtype::Float32 => TensorData::F32(
            tensor
                .data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        ),
        TensorDtype::Int64 => TensorData::I64(
            tensor
                .data
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect(),
        ),
    };
    Tensor { shape, data }
}
