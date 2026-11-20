use crate::error::{MmError, Result};

pub fn expand_token_placeholders(
    input_ids: &[i64],
    expansion_spec: &[(i64, Vec<Vec<i64>>)],
    start_len: usize,
) -> Result<Vec<i64>> {
    if start_len > input_ids.len() {
        return Err(MmError::invalid(
            "mm_token_expansion_start_len must be in [0, len(input_ids)].",
        ));
    }
    let mut placements: Vec<(usize, &[i64])> = Vec::new();
    for (index, (token_id, expansions)) in expansion_spec.iter().enumerate() {
        if expansion_spec[..index].iter().any(|(id, _)| id == token_id) {
            return Err(MmError::internal(format!(
                "placeholder token {token_id} appears twice in the expansion spec"
            )));
        }
        let positions = token_positions(input_ids, *token_id, start_len);
        let count = positions.len();
        if count > expansions.len() || (start_len == 0 && count != expansions.len()) {
            return Err(MmError::invalid(format!(
                "Found {count} placeholders for token {token_id}, \
                 but received {} media expansions",
                expansions.len()
            )));
        }
        let trailing = &expansions[expansions.len() - count..];
        placements.extend(
            positions
                .into_iter()
                .zip(trailing.iter().map(Vec::as_slice)),
        );
    }
    placements.sort_by_key(|(position, _)| *position);

    let inserted: usize = placements
        .iter()
        .map(|(_, expansion)| expansion.len())
        .sum();
    let mut expanded = Vec::with_capacity(input_ids.len() - placements.len() + inserted);
    let mut end = 0;
    for (position, expansion) in placements {
        expanded.extend_from_slice(&input_ids[end..position]);
        expanded.extend_from_slice(expansion);
        end = position + 1;
    }
    expanded.extend_from_slice(&input_ids[end..]);
    Ok(expanded)
}

pub fn locate_media(
    input_ids: &[i64],
    token_id: i64,
    expansions: &[Vec<i64>],
) -> Result<Vec<Vec<(usize, usize)>>> {
    let counts: Vec<usize> = expansions
        .iter()
        .map(|expansion| expansion.iter().filter(|id| **id == token_id).count())
        .collect();
    let positions = token_positions(input_ids, token_id, 0);
    let expected: usize = counts.iter().sum();
    if positions.len() != expected {
        return Err(MmError::invalid(format!(
            "{} placeholder tokens in input_ids for {} media expanding to {expected}; \
             a partial expansion needs every media of its prefix",
            positions.len(),
            expansions.len()
        )));
    }
    let mut spans = Vec::with_capacity(counts.len());
    let mut offset = 0;
    for count in counts {
        spans.push(contiguous_spans(&positions[offset..offset + count]));
        offset += count;
    }
    Ok(spans)
}

fn token_positions(input_ids: &[i64], token_id: i64, start: usize) -> Vec<usize> {
    input_ids[start..]
        .iter()
        .enumerate()
        .filter(|(_, id)| **id == token_id)
        .map(|(index, _)| start + index)
        .collect()
}

fn contiguous_spans(positions: &[usize]) -> Vec<(usize, usize)> {
    let mut spans: Vec<(usize, usize)> = Vec::new();
    for &position in positions {
        match spans.last_mut() {
            Some((_, end)) if *end == position => *end = position + 1,
            _ => spans.push((position, position + 1)),
        }
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE: i64 = 100;
    const VIDEO: i64 = 101;

    #[test]
    fn expands_each_placeholder_in_prompt_order() {
        let spec = vec![
            (IMAGE, vec![vec![IMAGE, IMAGE], vec![9, IMAGE, 9]]),
            (VIDEO, vec![vec![7, VIDEO, 7, VIDEO]]),
        ];
        let expanded = expand_token_placeholders(&[1, IMAGE, 2, VIDEO, IMAGE, 3], &spec, 0);
        assert_eq!(
            expanded.unwrap(),
            vec![1, IMAGE, IMAGE, 2, 7, VIDEO, 7, VIDEO, 9, IMAGE, 9, 3]
        );
    }

    #[test]
    fn partial_expansion_keeps_the_prefix_and_takes_trailing_expansions() {
        let spec = vec![(IMAGE, vec![vec![IMAGE, IMAGE], vec![8, IMAGE]])];
        let expanded = expand_token_placeholders(&[1, IMAGE, IMAGE, 2, IMAGE], &spec, 4);
        assert_eq!(expanded.unwrap(), vec![1, IMAGE, IMAGE, 2, 8, IMAGE]);
    }

    #[test]
    fn rejects_mismatched_counts_and_start() {
        let one = vec![(IMAGE, vec![vec![IMAGE]])];
        let invalid = |result: Result<Vec<i64>>| matches!(result, Err(MmError::InvalidRequest(_)));
        assert!(invalid(expand_token_placeholders(&[IMAGE, IMAGE], &one, 0)));
        assert!(invalid(expand_token_placeholders(&[1, 2], &one, 0)));
        assert!(invalid(expand_token_placeholders(
            &[1, IMAGE, IMAGE],
            &one,
            1
        )));
        assert!(invalid(expand_token_placeholders(&[1], &one, 2)));
        assert_eq!(
            expand_token_placeholders(&[IMAGE, 1], &one, 1).unwrap(),
            vec![IMAGE, 1]
        );

        let duplicate = vec![(IMAGE, vec![vec![IMAGE]]), (IMAGE, vec![vec![IMAGE]])];
        assert!(matches!(
            expand_token_placeholders(&[IMAGE], &duplicate, 0),
            Err(MmError::Internal(_))
        ));
    }

    #[test]
    fn locates_each_media_as_contiguous_spans() {
        let expansions = vec![vec![IMAGE, IMAGE], vec![7, IMAGE, 7, IMAGE, IMAGE], vec![7]];
        let ids = [1, IMAGE, IMAGE, 2, 7, IMAGE, 7, IMAGE, IMAGE, 7, 3];
        assert_eq!(
            locate_media(&ids, IMAGE, &expansions).unwrap(),
            vec![vec![(1, 3)], vec![(5, 6), (7, 9)], vec![]]
        );
        assert!(matches!(
            locate_media(&ids[..8], IMAGE, &expansions),
            Err(MmError::InvalidRequest(_))
        ));
    }
}
