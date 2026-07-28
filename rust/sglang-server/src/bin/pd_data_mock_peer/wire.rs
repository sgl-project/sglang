use super::*;

pub(super) fn encode_prepare(shape: &RoomShape) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(18 + shape.destination_pages.len() * 4);
    bytes.extend_from_slice(&shape.room_number.to_be_bytes());
    bytes.extend_from_slice(&shape.valid_tokens.to_be_bytes());
    bytes.push(u8::from(shape.prefill_first));
    bytes.push(u8::try_from(shape.destination_pages.len()).expect("at most 64 pages"));
    for page in &shape.destination_pages {
        bytes.extend_from_slice(&page.to_be_bytes());
    }
    bytes
}

pub(super) fn decode_prepare(bytes: &[u8]) -> HarnessResult<RoomShape> {
    let mut reader = BytesReader::new(bytes);
    let room_number = reader.u64()?;
    let valid_tokens = reader.u32()?;
    let prefill_first = match reader.u8()? {
        0 => false,
        1 => true,
        _ => return Err("Prepare arrival-order flag was invalid".into()),
    };
    let count = usize::from(reader.u8()?);
    if count == 0 || count > 64 {
        return Err("Prepare page count was invalid".into());
    }
    let mut destination_pages = Vec::with_capacity(count);
    for _ in 0..count {
        destination_pages.push(reader.u32()?);
    }
    reader.finish()?;
    Ok(RoomShape {
        room_number,
        valid_tokens,
        destination_pages,
        prefill_first,
    })
}

pub(super) fn encode_accepted(source_pages: &[u32], digest: &[u8; 32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(33 + source_pages.len() * 4);
    bytes.push(u8::try_from(source_pages.len()).expect("at most 64 pages"));
    for page in source_pages {
        bytes.extend_from_slice(&page.to_be_bytes());
    }
    bytes.extend_from_slice(digest);
    bytes
}

pub(super) fn decode_accepted(bytes: &[u8]) -> HarnessResult<(Vec<u32>, [u8; 32])> {
    let mut reader = BytesReader::new(bytes);
    let count = usize::from(reader.u8()?);
    if count == 0 || count > 64 {
        return Err("PrepareAccepted page count was invalid".into());
    }
    let mut source_pages = Vec::with_capacity(count);
    for _ in 0..count {
        source_pages.push(reader.u32()?);
    }
    let digest = reader.array::<32>()?;
    reader.finish()?;
    Ok((source_pages, digest))
}

pub(super) fn verify_terminal_digest(frame: &Frame, plan: &TransferPlan) -> HarnessResult<()> {
    if frame.payload.as_slice() != plan.digest().as_bytes() {
        return Err("terminal data-plane message used a stale digest".into());
    }
    Ok(())
}

pub(super) fn send_frame(stream: &mut TcpStream, kind: u8, payload: &[u8]) -> HarnessResult<()> {
    if payload.len() > FRAME_LIMIT {
        return Err("data harness frame exceeded its fixed bound".into());
    }
    stream
        .write_all(&[kind])
        .and_then(|()| {
            stream.write_all(
                &u32::try_from(payload.len())
                    .expect("frame limit fits u32")
                    .to_be_bytes(),
            )
        })
        .and_then(|()| stream.write_all(payload))
        .and_then(|()| stream.flush())
        .map_err(|error| format!("data frame write failed: {error}"))
}

pub(super) fn receive_frame(stream: &mut TcpStream) -> HarnessResult<Frame> {
    let mut header = [0_u8; 5];
    stream
        .read_exact(&mut header)
        .map_err(|error| format!("data frame header read failed: {error}"))?;
    let length = usize::try_from(u32::from_be_bytes(
        header[1..]
            .try_into()
            .map_err(|_| "invalid data frame header")?,
    ))
    .map_err(display)?;
    if length > FRAME_LIMIT {
        return Err("data frame length exceeded its fixed bound".into());
    }
    let mut payload = vec![0; length];
    stream
        .read_exact(&mut payload)
        .map_err(|error| format!("data frame payload read failed: {error}"))?;
    Ok(Frame {
        kind: header[0],
        payload,
    })
}

pub(super) fn connect_with_retry(address: &str) -> HarnessResult<TcpStream> {
    let mut last_error = None;
    for _ in 0..200 {
        match TcpStream::connect(address) {
            Ok(stream) => return Ok(stream),
            Err(error) => {
                last_error = Some(error);
                thread::sleep(Duration::from_millis(10));
            }
        }
    }
    Err(format!(
        "data connect failed: {}",
        last_error
            .map(|error| error.to_string())
            .unwrap_or_else(|| "no attempts".into())
    ))
}

pub(super) struct BytesReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> BytesReader<'a> {
    pub(super) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    pub(super) fn u8(&mut self) -> HarnessResult<u8> {
        Ok(self.array::<1>()?[0])
    }

    pub(super) fn u16(&mut self) -> HarnessResult<u16> {
        Ok(u16::from_be_bytes(self.array()?))
    }

    pub(super) fn u32(&mut self) -> HarnessResult<u32> {
        Ok(u32::from_be_bytes(self.array()?))
    }

    pub(super) fn u64(&mut self) -> HarnessResult<u64> {
        Ok(u64::from_be_bytes(self.array()?))
    }

    pub(super) fn array<const N: usize>(&mut self) -> HarnessResult<[u8; N]> {
        self.bytes(N)?
            .try_into()
            .map_err(|_| "data frame field had the wrong width".into())
    }

    pub(super) fn bytes(&mut self, length: usize) -> HarnessResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| "data frame offset overflowed".to_string())?;
        let bytes = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| "data frame was truncated".to_string())?;
        self.offset = end;
        Ok(bytes)
    }

    pub(super) fn finish(self) -> HarnessResult<()> {
        if self.offset == self.bytes.len() {
            Ok(())
        } else {
            Err("data frame contained trailing bytes".into())
        }
    }
}

pub(super) fn data_byte(region_id: u16, source_page: u32, offset: usize) -> u8 {
    u8::try_from(
        (usize::from(region_id) * 17
            + usize::try_from(source_page).unwrap_or(0) * 13
            + offset % 251)
            % 256,
    )
    .expect("value is reduced modulo 256")
}

pub(super) fn display(error: impl std::fmt::Display) -> String {
    error.to_string()
}
