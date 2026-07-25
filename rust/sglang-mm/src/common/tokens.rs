//! Placeholder-token expansion for the native MM path.
//!
//! Id-level equivalent of the HF processor's string expansion (e.g. Qwen's
//! `<|image_pad|>` → N copies): expanding the already-tokenized prompt means
//! non-media tokens can never drift from a retokenize (the
//! `SGLANG_MM_AVOID_RETOKENIZE` idea, unconditional here).

/// The expanded prompt plus, per media item (in prompt order), the inclusive
/// `(start, end)` token range it occupies — the Python `get_mm_items_offset`
/// convention.
pub struct ExpandedPrompt {
    pub input_ids: Vec<i32>,
    pub offsets: Vec<(u32, u32)>,
}

/// Expand each occurrence of `placeholder_id` in `ids` to `counts[i]` copies
/// (i-th occurrence ↔ i-th media item). Errs when the occurrence count and
/// `counts` disagree — the caller decides whether that falls back or rejects.
pub fn expand_placeholders(
    ids: &[i32],
    placeholder_id: i32,
    counts: &[usize],
) -> Result<ExpandedPrompt, String> {
    let found = ids.iter().filter(|&&id| id == placeholder_id).count();
    if found != counts.len() {
        return Err(format!(
            "prompt has {found} media placeholder(s) but {} media item(s)",
            counts.len()
        ));
    }
    let total: usize = counts.iter().sum();
    let mut out = Vec::with_capacity(ids.len() - found + total);
    let mut offsets = Vec::with_capacity(counts.len());
    let mut item = 0;
    for &id in ids {
        if id == placeholder_id {
            let n = counts[item];
            item += 1;
            let start = out.len() as u32;
            out.resize(out.len() + n, placeholder_id);
            offsets.push((start, start + n as u32 - 1));
        } else {
            out.push(id);
        }
    }
    Ok(ExpandedPrompt {
        input_ids: out,
        offsets,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_in_order_with_inclusive_offsets() {
        // [7, PAD, 8, PAD, 9] with counts [2, 3]
        let e = expand_placeholders(&[7, 1, 8, 1, 9], 1, &[2, 3]).unwrap();
        assert_eq!(e.input_ids, vec![7, 1, 1, 8, 1, 1, 1, 9]);
        assert_eq!(e.offsets, vec![(1, 2), (4, 6)]);
    }

    #[test]
    fn count_mismatch_errs() {
        assert!(expand_placeholders(&[7, 1, 9], 1, &[2, 3]).is_err());
        assert!(expand_placeholders(&[7, 1, 1, 9], 1, &[2]).is_err());
    }

    #[test]
    fn no_placeholders_no_items_ok() {
        let e = expand_placeholders(&[7, 8], 1, &[]).unwrap();
        assert_eq!(e.input_ids, vec![7, 8]);
        assert!(e.offsets.is_empty());
    }
}
