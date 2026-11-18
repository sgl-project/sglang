//! Transport-independent input to the existing native image processors.
//!
//! The host authorizes storage and owns byte/lifetime budgets before handing
//! over an immutable snapshot. No paths, shared-memory handles or model-specific
//! metadata cross this seam.

use std::ops::Range;

use dynamo_multimodal::{MediaDtype, content_hash_canonical_image};

use crate::ProcessorError;

/// Owned upright HWC u8 RGB, without row padding or an encoded-image header.
/// Resize, normalization, patches and positions remain the family's work.
pub struct DecodedRgbImage {
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl std::fmt::Debug for DecodedRgbImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodedRgbImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes", &self.rgb.len())
            .finish()
    }
}

impl DecodedRgbImage {
    pub fn new(rgb: Vec<u8>, width: u32, height: u32) -> Result<Self, ProcessorError> {
        let image = Self { rgb, width, height };
        image.validate()?;
        Ok(image)
    }

    // Public fields also permit host construction; revalidate at processing.
    pub(super) fn validate(&self) -> Result<(), ProcessorError> {
        let (_, _, packed) = RgbImageLayout {
            width: self.width,
            height: self.height,
            row_stride_bytes: u64::from(self.width) * 3,
        }
        .sizes()?;
        if self.rgb.len() as u64 != packed {
            return Err("decoded RGB byte count does not match its dimensions".into());
        }
        Ok(())
    }

    /// The existing canonical decoded-image identity: effective RGB + HWC shape.
    /// Object names, row padding and snapshot transport never affect identity.
    pub(super) fn content_hash(&self) -> u64 {
        content_hash_canonical_image(
            &[self.height as usize, self.width as usize, 3],
            MediaDtype::U8,
            &self.rgb,
        )
    }
}

/// A decoded RGB source layout, independent of its storage transport.
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
        // Match MmProcessor's encoded-image decode limits before allocating or
        // invoking a family. The layout's byte allowance is an additional cap.
        let pixels = u64::from(self.width) * u64::from(self.height);
        if pixels > super::PIL_DECOMPRESSION_BOMB_PIXELS as u64 {
            return Err("decoded RGB image exceeds decode pixel limit".into());
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
        let packed = pixels * 3;
        usize::try_from(packed).map_err(|_| "decoded RGB packed size exceeds usize")?;
        Ok((row_bytes, required, packed))
    }

    /// Check a referenced region before reading or allocating pixels.
    /// The host must also reserve an aggregate in-flight byte allowance.
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

    /// Copy only effective RGB rows from a snapshot into packed owned storage.
    pub fn pack(self, bytes: &[u8]) -> Result<DecodedRgbImage, ProcessorError> {
        let (row, required, packed) = self.sizes()?;
        if (bytes.len() as u64) < required {
            return Err("decoded RGB snapshot is shorter than its layout".into());
        }
        let row = row as usize;
        let stride = usize::try_from(self.row_stride_bytes)
            .map_err(|_| "decoded RGB row stride exceeds usize")?;
        let mut rgb = Vec::with_capacity(packed as usize);
        for y in 0..self.height as usize {
            rgb.extend_from_slice(&bytes[y * stride..y * stride + row]);
        }
        DecodedRgbImage::new(rgb, self.width, self.height)
    }

    /// Reuse the snapshot allocation, removing padding in place when needed.
    /// Already packed pixels are moved through without an additional copy.
    pub fn pack_owned(self, mut bytes: Vec<u8>) -> Result<DecodedRgbImage, ProcessorError> {
        let (row, required, packed) = self.sizes()?;
        if (bytes.len() as u64) < required {
            return Err("decoded RGB snapshot is shorter than its layout".into());
        }
        let row = row as usize;
        let stride = usize::try_from(self.row_stride_bytes)
            .map_err(|_| "decoded RGB row stride exceeds usize")?;
        if stride != row {
            for y in 1..self.height as usize {
                bytes.copy_within(y * stride..y * stride + row, y * row);
            }
        }
        bytes.truncate(packed as usize);
        DecodedRgbImage::new(bytes, self.width, self.height)
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
    fn rejects_invalid_dimensions_stride_overflow_and_pixel_bombs() {
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
            RgbImageLayout {
                width: 20_000,
                height: 20_000,
                row_stride_bytes: 60_000,
            },
        ] {
            assert!(
                bad.validate_region(u64::MAX, 0, u64::MAX, u64::MAX)
                    .is_err()
            );
        }
        assert!(DecodedRgbImage::new(vec![1, 2], 1, 1).is_err());
        assert!(DecodedRgbImage::new(vec![1, 2, 3, 4], 1, 1).is_err());
        assert!(layout().pack_owned(vec![0; 6]).is_err());
    }

    #[test]
    fn owned_rows_are_compacted_without_reallocating_or_hashing_padding() {
        let bytes = vec![1, 2, 3, 99, 4, 5, 6, 88, 77];
        let allocation = bytes.as_ptr();
        let packed = layout().pack_owned(bytes).unwrap();
        assert_eq!(packed.rgb.as_ptr(), allocation);
        assert_eq!(packed.rgb, [1, 2, 3, 4, 5, 6]);
        let different_padding = layout().pack(&[1, 2, 3, 44, 4, 5, 6]).unwrap();
        assert_eq!(packed.content_hash(), different_padding.content_hash());
        let different_shape = DecodedRgbImage::new(packed.rgb.clone(), 2, 1).unwrap();
        assert_ne!(packed.content_hash(), different_shape.content_hash());
        let different_pixels = DecodedRgbImage::new(vec![1, 2, 3, 4, 5, 7], 1, 2).unwrap();
        assert_ne!(packed.content_hash(), different_pixels.content_hash());
    }

    #[test]
    fn packed_owned_input_keeps_its_allocation() {
        let bytes = vec![1, 2, 3, 4, 5, 6];
        let allocation = bytes.as_ptr();
        let packed = RgbImageLayout {
            row_stride_bytes: 3,
            ..layout()
        }
        .pack_owned(bytes)
        .unwrap();
        assert_eq!(packed.rgb.as_ptr(), allocation);
    }
}
