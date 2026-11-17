//! Transport-independent layout validation of decoded CPU RGB pixels.
//!
//! Layout checks for the opt-in api.v1 shared-memory source,
//! not a shared-memory opener or a new model adapter. The ingress must first
//! authorize the object, fence its engine instance and acquire its byte/lifetime
//! leases. This module never opens caller-selected paths or maps mutable files.

use std::ops::Range;

/// Upright HWC u8 RGB; model preprocessing remains with existing processors.
#[derive(Clone, Copy, Debug)]
pub struct RgbImageLayout {
    pub width: u32,
    pub height: u32,
    pub row_stride_bytes: u64,
}

impl RgbImageLayout {
    fn sizes(self) -> Result<(u64, u64, u64), String> {
        if self.width == 0 || self.height == 0 {
            return Err("decoded RGB dimensions must be positive".into());
        }
        let row_bytes = u64::from(self.width) * 3;
        if self.row_stride_bytes < row_bytes {
            return Err("decoded RGB row stride is smaller than width * 3".into());
        }
        let required = self
            .row_stride_bytes
            .checked_mul(u64::from(self.height) - 1)
            .and_then(|n| n.checked_add(row_bytes))
            .ok_or("decoded RGB byte range overflow")?;
        let packed = row_bytes
            .checked_mul(u64::from(self.height))
            .ok_or("decoded RGB packed size overflow")?;
        Ok((row_bytes, required, packed))
    }

    /// Check the referenced region before reading or allocating pixels.
    /// `max_bytes` is an ingress-selected bound, not a new global default.
    /// The ingress must also reserve the aggregate in-flight allowance.
    pub fn validate_region(
        self,
        storage_len: u64,
        offset: u64,
        length: u64,
        max_bytes: u64,
    ) -> Result<Range<usize>, String> {
        let (_, required, packed) = self.sizes()?;
        if length < required {
            return Err("decoded RGB region is shorter than its layout".into());
        }
        if length > max_bytes || packed > max_bytes {
            return Err("decoded RGB region exceeds its byte allowance".into());
        }
        let end = offset
            .checked_add(length)
            .ok_or("decoded RGB region end overflow")?;
        if end > storage_len {
            return Err("decoded RGB region exceeds its storage object".into());
        }
        let start = usize::try_from(offset).map_err(|_| "decoded RGB offset exceeds usize")?;
        let end = usize::try_from(end).map_err(|_| "decoded RGB end exceeds usize")?;
        Ok(start..end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> RgbImageLayout {
        RgbImageLayout {
            width: 1,
            height: 2,
            row_stride_bytes: 4,
        }
    }

    #[test]
    fn rejects_short_out_of_bounds_and_over_budget_regions() {
        assert!(layout().validate_region(7, 0, 6, 7).is_err());
        assert!(layout().validate_region(7, 1, 7, 7).is_err());
        assert!(layout().validate_region(8, 0, 8, 7).is_err());
        assert!(layout().validate_region(u64::MAX, u64::MAX, 7, 7).is_err());
    }

    #[test]
    fn rejects_invalid_dimensions_stride_and_overflow() {
        for bad in [
            RgbImageLayout {
                width: 0,
                ..layout()
            },
            RgbImageLayout {
                height: 0,
                ..layout()
            },
            RgbImageLayout {
                row_stride_bytes: 2,
                ..layout()
            },
            RgbImageLayout {
                row_stride_bytes: u64::MAX,
                ..layout()
            },
        ] {
            assert!(bad.validate_region(u64::MAX, 0, 7, u64::MAX).is_err());
        }
    }
}
