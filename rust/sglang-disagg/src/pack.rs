//! Binary packing helpers — `common/utils.py` `pack_list_of_buffers`,
//! `unpack_list_of_buffers`, `pack_int_lists`, `unpack_int_lists`, and the
//! ASCII integer frames (`int(str)`).

use crate::Result;
use crate::error::Error;

/// Strict ASCII `int(s)`: optional surrounding whitespace, optional sign,
/// decimal digits only. Python's real `int()` also accepts PEP 515
/// underscores and unicode digits — none ever appear on this wire and
/// accepting them would hide corruption, so they are rejected here.
pub fn py_int(s: &[u8]) -> Result<i64> {
    let t = std::str::from_utf8(s)
        .map_err(|_| Error::Codec("int frame is not utf-8".into()))?
        .trim_matches(|c: char| c.is_ascii_whitespace());
    let (neg, digits) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Error::Codec(format!("not an int frame: {:?}", t)));
    }
    let v: i64 = digits
        .parse()
        .map_err(|_| Error::Codec(format!("int frame out of range: {:?}", t)))?;
    Ok(if neg { -v } else { v })
}

/// `pack_list_of_buffers`: empty -> b""; else `<{n+1}I` header (n, *lens) +
/// concatenated payloads.
pub fn pack_list_of_buffers(buffers: &[&[u8]]) -> Vec<u8> {
    if buffers.is_empty() {
        return Vec::new();
    }
    let n = buffers.len() as u32;
    let mut out = Vec::with_capacity(
        4 * (buffers.len() + 1) + buffers.iter().map(|b| b.len()).sum::<usize>(),
    );
    out.extend_from_slice(&n.to_le_bytes());
    for b in buffers {
        out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    }
    for b in buffers {
        out.extend_from_slice(b);
    }
    out
}

/// `unpack_list_of_buffers`: b"" -> []; short header -> Codec error
/// (Python's struct.unpack raises struct.error); payload slices clamp like
/// Python slicing, trailing bytes ignored.
pub fn unpack_list_of_buffers(buf: &[u8]) -> Result<Vec<Vec<u8>>> {
    if buf.is_empty() {
        return Ok(Vec::new());
    }
    let bad = || Error::Codec("unpack_list_of_buffers: truncated header".to_string());
    let n = u32::from_le_bytes(buf.get(0..4).ok_or_else(bad)?.try_into().unwrap()) as usize;
    let lens_end = 4 + 4 * n;
    let lens_buf = buf.get(4..lens_end).ok_or_else(bad)?;
    let mut out = Vec::with_capacity(n);
    let mut offset = lens_end;
    for i in 0..n {
        let len = u32::from_le_bytes(lens_buf[4 * i..4 * i + 4].try_into().unwrap()) as usize;
        let end = (offset + len).min(buf.len());
        out.push(buf[offset.min(buf.len())..end].to_vec());
        offset = end;
    }
    Ok(out)
}

pub fn pack_ints_u64(list: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 * list.len());
    for v in list {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn pack_ints_u32(list: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 * list.len());
    for v in list {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

pub fn pack_ints_i32(list: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 * list.len());
    for v in list {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// `pack_int_lists(lists, "Q")`.
pub fn pack_u64_lists(lists: &[Vec<u64>]) -> Vec<u8> {
    let bufs: Vec<Vec<u8>> = lists.iter().map(|l| pack_ints_u64(l)).collect();
    let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
    pack_list_of_buffers(&refs)
}

/// `pack_int_lists(lists, "I")`.
pub fn pack_u32_lists(lists: &[Vec<u32>]) -> Vec<u8> {
    let bufs: Vec<Vec<u8>> = lists.iter().map(|l| pack_ints_u32(l)).collect();
    let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
    pack_list_of_buffers(&refs)
}

/// `pack_int_lists(lists, "i")`.
pub fn pack_i32_lists(lists: &[Vec<i32>]) -> Vec<u8> {
    let bufs: Vec<Vec<u8>> = lists.iter().map(|l| pack_ints_i32(l)).collect();
    let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
    pack_list_of_buffers(&refs)
}

fn unpack_ints<const W: usize>(buf: &[u8], read: fn([u8; W]) -> i64) -> Result<Vec<i64>> {
    if !buf.len().is_multiple_of(W) {
        return Err(Error::Codec("unpack_int_lists: bad length".into()));
    }
    Ok(buf
        .chunks_exact(W)
        .map(|c| read(c.try_into().unwrap()))
        .collect())
}

/// `unpack_int_lists(buf, "Q")` — errors on len%8 (struct.unpack would).
pub fn unpack_u64_lists(buf: &[u8]) -> Result<Vec<Vec<u64>>> {
    let bufs = unpack_list_of_buffers(buf)?;
    let mut out = Vec::with_capacity(bufs.len());
    for b in &bufs {
        out.push(
            unpack_ints::<8>(b, |c| u64::from_le_bytes(c) as i64)?
                .into_iter()
                .map(|v| v as u64)
                .collect(),
        );
    }
    Ok(out)
}

/// `unpack_int_lists(buf, "I")`.
pub fn unpack_u32_lists(buf: &[u8]) -> Result<Vec<Vec<u32>>> {
    let bufs = unpack_list_of_buffers(buf)?;
    let mut out = Vec::with_capacity(bufs.len());
    for b in &bufs {
        out.push(
            unpack_ints::<4>(b, |c| u32::from_le_bytes(c) as i64)?
                .into_iter()
                .map(|v| v as u32)
                .collect(),
        );
    }
    Ok(out)
}

/// `unpack_int_lists(buf, "i")`.
pub fn unpack_i32_lists(buf: &[u8]) -> Result<Vec<Vec<i32>>> {
    let bufs = unpack_list_of_buffers(buf)?;
    let mut out = Vec::with_capacity(bufs.len());
    for b in &bufs {
        out.push(
            unpack_ints::<4>(b, |c| i32::from_le_bytes(c) as i64)?
                .into_iter()
                .map(|v| v as i32)
                .collect(),
        );
    }
    Ok(out)
}
