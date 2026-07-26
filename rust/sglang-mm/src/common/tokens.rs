//! Token-layout mechanics for the server MM pipeline.
//!
//! Families describe their prompt geometry as a [`TokenLayout`] value
//! (`pipeline.rs`); [`apply_layout`] applies it mechanically. Expanding the
//! already-tokenized prompt means non-media tokens can never drift from a
//! retokenize (the `SGLANG_MM_AVOID_RETOKENIZE` idea, unconditional here).

use crate::pipeline::{Segment, TokenLayout, TokenPattern};

/// The expanded prompt plus, per media item (indexed as in the layout), the
/// inclusive `(start, end)` token range it occupies — the Python
/// `get_mm_items_offset` convention.
pub struct ExpandedPrompt {
    pub input_ids: Vec<i32>,
    pub offsets: Vec<(u32, u32)>,
}

/// Apply a family's [`TokenLayout`] to the original prompt. Validates that
/// text ranges are in bounds, that every one of the `n_items` media items is
/// placed exactly once, and that no item expands to zero tokens (which would
/// have no representable offset).
pub fn apply_layout(
    src: &[i32],
    layout: &TokenLayout,
    n_items: usize,
) -> Result<ExpandedPrompt, String> {
    let mut out = Vec::new();
    let mut offsets: Vec<Option<(u32, u32)>> = vec![None; n_items];
    for segment in &layout.segments {
        match segment {
            Segment::Text(range) => {
                let text = src
                    .get(range.clone())
                    .ok_or_else(|| format!("layout: text range {range:?} out of bounds"))?;
                out.extend_from_slice(text);
            }
            Segment::Media { item, pattern } => {
                let start = out.len() as u32;
                let n = match pattern {
                    TokenPattern::Repeat { id, n } => {
                        out.resize(out.len() + n, *id);
                        *n
                    }
                    TokenPattern::Explicit(ids) => {
                        out.extend_from_slice(ids);
                        ids.len()
                    }
                };
                if n == 0 {
                    return Err(format!("layout: media item {item} expands to zero tokens"));
                }
                let slot = offsets
                    .get_mut(*item)
                    .ok_or_else(|| format!("layout: media item {item} out of range"))?;
                if slot.replace((start, start + n as u32 - 1)).is_some() {
                    return Err(format!("layout: media item {item} placed twice"));
                }
            }
        }
    }
    let offsets = offsets
        .into_iter()
        .enumerate()
        .map(|(i, slot)| slot.ok_or_else(|| format!("layout: media item {i} not placed")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ExpandedPrompt {
        input_ids: out,
        offsets,
    })
}

/// Build the simplest layout: each occurrence of `placeholder_id` in `ids`
/// becomes `counts[i]` copies (i-th occurrence ↔ i-th media item). Errs when
/// the occurrence count and `counts` disagree.
pub fn layout_by_placeholder(
    ids: &[i32],
    placeholder_id: i32,
    counts: &[usize],
) -> Result<TokenLayout, String> {
    let found = ids.iter().filter(|&&id| id == placeholder_id).count();
    if found != counts.len() {
        return Err(format!(
            "prompt has {found} media placeholder(s) but {} media item(s)",
            counts.len()
        ));
    }
    let mut segments = Vec::new();
    let mut text_start = 0;
    let mut item = 0;
    for (pos, &id) in ids.iter().enumerate() {
        if id == placeholder_id {
            if text_start < pos {
                segments.push(Segment::Text(text_start..pos));
            }
            segments.push(Segment::Media {
                item,
                pattern: TokenPattern::Repeat {
                    id: placeholder_id,
                    n: counts[item],
                },
            });
            item += 1;
            text_start = pos + 1;
        }
    }
    if text_start < ids.len() {
        segments.push(Segment::Text(text_start..ids.len()));
    }
    Ok(TokenLayout { segments })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expand(ids: &[i32], placeholder: i32, counts: &[usize]) -> Result<ExpandedPrompt, String> {
        apply_layout(
            ids,
            &layout_by_placeholder(ids, placeholder, counts)?,
            counts.len(),
        )
    }

    #[test]
    fn expands_in_order_with_inclusive_offsets() {
        // [7, PAD, 8, PAD, 9] with counts [2, 3]
        let e = expand(&[7, 1, 8, 1, 9], 1, &[2, 3]).unwrap();
        assert_eq!(e.input_ids, vec![7, 1, 1, 8, 1, 1, 1, 9]);
        assert_eq!(e.offsets, vec![(1, 2), (4, 6)]);
    }

    #[test]
    fn count_mismatch_errs() {
        assert!(expand(&[7, 1, 9], 1, &[2, 3]).is_err());
        assert!(expand(&[7, 1, 1, 9], 1, &[2]).is_err());
    }

    #[test]
    fn zero_count_errs() {
        assert!(expand(&[7, 1, 9], 1, &[0]).is_err());
    }

    #[test]
    fn no_placeholders_no_items_ok() {
        let e = expand(&[7, 8], 1, &[]).unwrap();
        assert_eq!(e.input_ids, vec![7, 8]);
        assert!(e.offsets.is_empty());
    }

    #[test]
    fn explicit_patterns_and_placement_validation() {
        // Structured expansion: marker tokens around the item span.
        let layout = TokenLayout {
            segments: vec![
                Segment::Text(0..1),
                Segment::Media {
                    item: 0,
                    pattern: TokenPattern::Explicit(vec![90, 5, 5, 91]),
                },
                Segment::Text(2..3),
            ],
        };
        let e = apply_layout(&[7, 1, 9], &layout, 1).unwrap();
        assert_eq!(e.input_ids, vec![7, 90, 5, 5, 91, 9]);
        assert_eq!(e.offsets, vec![(1, 4)]);

        // Every item must be placed exactly once; ranges must be in bounds.
        let missing = TokenLayout {
            segments: vec![Segment::Text(0..3)],
        };
        assert!(apply_layout(&[7, 1, 9], &missing, 1).is_err());
        let out_of_bounds = TokenLayout {
            segments: vec![Segment::Text(0..4)],
        };
        assert!(apply_layout(&[7, 1, 9], &out_of_bounds, 0).is_err());
    }
}
