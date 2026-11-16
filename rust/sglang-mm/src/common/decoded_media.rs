//! Transport-independent import of already-decoded CPU RGB pixels.
//!
//! This is the processor seam for the proposed api.v1 shared-memory source,
//! not a shared-memory opener or a new model adapter. The ingress must first
//! authorize the object, fence its engine instance and acquire its byte/lifetime
//! leases. This module never opens caller-selected paths or maps mutable files.

use std::ops::Range;

use crate::pipeline::DecodedMedia;

/// Upright HWC u8 RGB; model preprocessing remains with MmFamilyProcessor.
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

/// A validated, owned snapshot accepted by the existing processor interface.
///
/// An owned copy deliberately ends the dependency on a caller-owned mapping
/// before asynchronous preprocessing. This draft does not claim zero-copy;
/// mapping-backed processor views need a separate ownership/lifetime review.
pub struct DecodedRgbImage {
    media: DecodedMedia,
}

impl DecodedRgbImage {
    /// Import an already-authorized buffer, compacting row padding if present.
    /// This does not decode BMP/JPEG, resize, normalize, or expand token spans.
    pub fn from_region(
        layout: RgbImageLayout,
        storage: &[u8],
        offset: u64,
        length: u64,
        max_bytes: u64,
    ) -> Result<Self, String> {
        let range = layout.validate_region(storage.len() as u64, offset, length, max_bytes)?;
        let (row_bytes, _, packed) = layout.sizes()?;
        let row_bytes = usize::try_from(row_bytes).map_err(|_| "decoded RGB row exceeds usize")?;
        let stride = usize::try_from(layout.row_stride_bytes)
            .map_err(|_| "decoded RGB stride exceeds usize")?;
        let packed = usize::try_from(packed).map_err(|_| "decoded RGB size exceeds usize")?;
        let pixels = &storage[range];
        let mut rgb = Vec::new();
        rgb.try_reserve_exact(packed)
            .map_err(|_| "decoded RGB allocation failed")?;
        for row in 0..layout.height as usize {
            let start = row * stride;
            rgb.extend_from_slice(&pixels[start..start + row_bytes]);
        }
        Ok(Self {
            media: DecodedMedia::Image {
                rgb,
                height: layout.height as usize,
                width: layout.width as usize,
            },
        })
    }

    /// Hand the same common input to whichever processor the engine selected.
    pub fn into_media(self) -> DecodedMedia {
        self.media
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
    fn imports_rgb_rows_without_padding_or_redecoding() {
        let storage = [99, 1, 2, 3, 0, 4, 5, 6, 99];
        let image = DecodedRgbImage::from_region(layout(), &storage, 1, 7, 7).unwrap();
        let DecodedMedia::Image { rgb, height, width } = image.into_media();
        assert_eq!((height, width), (2, 1));
        assert_eq!(rgb, [1, 2, 3, 4, 5, 6]);
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
