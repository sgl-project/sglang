use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use zeroize::Zeroize;

use crate::pd::protocol::codec::{HEADER_BYTES, MAX_PAYLOAD_BYTES, TAG_BYTES};
use crate::pd::protocol::{
    ControlPayload, DecodedFrame, Direction, FrameCodec, FrameError, SessionKeys,
};

pub struct DirectionalSession {
    send_direction: Direction,
    receive_direction: Direction,
    send_key: [u8; 32],
    receive_key: [u8; 32],
    next_send_sequence: u64,
    next_receive_sequence: u64,
}

impl DirectionalSession {
    pub fn decode_side(keys: &SessionKeys) -> Self {
        Self {
            send_direction: Direction::DecodeToPrefill,
            receive_direction: Direction::PrefillToDecode,
            send_key: keys.decode_to_prefill,
            receive_key: keys.prefill_to_decode,
            next_send_sequence: 2,
            next_receive_sequence: 2,
        }
    }

    pub fn prefill_side(keys: &SessionKeys) -> Self {
        Self {
            send_direction: Direction::PrefillToDecode,
            receive_direction: Direction::DecodeToPrefill,
            send_key: keys.prefill_to_decode,
            receive_key: keys.decode_to_prefill,
            next_send_sequence: 2,
            next_receive_sequence: 2,
        }
    }

    pub fn encode(
        &mut self,
        payload: &ControlPayload,
        deadline_unix_ms: u64,
    ) -> Result<Vec<u8>, FrameError> {
        let sequence = self.next_send_sequence;
        let frame = FrameCodec::encode(
            payload.kind(),
            self.send_direction,
            sequence,
            deadline_unix_ms,
            payload,
            &self.send_key,
        )?;
        self.next_send_sequence = sequence.checked_add(1).ok_or(FrameError::Sequence)?;
        Ok(frame)
    }

    pub fn decode(&mut self, frame: &[u8], now_unix_ms: u64) -> Result<DecodedFrame, FrameError> {
        let decoded = FrameCodec::decode(
            frame,
            self.receive_direction,
            self.next_receive_sequence,
            now_unix_ms,
            &self.receive_key,
        )?;
        self.next_receive_sequence = self
            .next_receive_sequence
            .checked_add(1)
            .ok_or(FrameError::Sequence)?;
        Ok(decoded)
    }

    pub const fn next_send_sequence(&self) -> u64 {
        self.next_send_sequence
    }

    pub const fn next_receive_sequence(&self) -> u64 {
        self.next_receive_sequence
    }
}

impl Drop for DirectionalSession {
    fn drop(&mut self) {
        self.send_key.zeroize();
        self.receive_key.zeroize();
    }
}

pub async fn read_raw_frame<R>(reader: &mut R) -> Result<Vec<u8>, SessionError>
where
    R: AsyncRead + Unpin,
{
    let mut header = [0_u8; HEADER_BYTES];
    reader
        .read_exact(&mut header)
        .await
        .map_err(|_| SessionError::Read)?;
    let payload_len = u32::from_be_bytes(
        header[12..16]
            .try_into()
            .map_err(|_| SessionError::Header)?,
    ) as usize;
    if payload_len > MAX_PAYLOAD_BYTES {
        return Err(SessionError::PayloadTooLarge);
    }
    let tail_len = payload_len
        .checked_add(TAG_BYTES)
        .ok_or(SessionError::PayloadTooLarge)?;
    let mut frame = Vec::with_capacity(HEADER_BYTES + tail_len);
    frame.extend_from_slice(&header);
    frame.resize(HEADER_BYTES + tail_len, 0);
    reader
        .read_exact(&mut frame[HEADER_BYTES..])
        .await
        .map_err(|_| SessionError::Read)?;
    Ok(frame)
}

pub async fn write_raw_frame<W>(writer: &mut W, frame: &[u8]) -> Result<(), SessionError>
where
    W: AsyncWrite + Unpin,
{
    writer
        .write_all(frame)
        .await
        .map_err(|_| SessionError::Write)?;
    writer.flush().await.map_err(|_| SessionError::Write)
}

#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("failed to read a bounded PD control frame")]
    Read,
    #[error("failed to write a PD control frame")]
    Write,
    #[error("PD control frame header is malformed")]
    Header,
    #[error("PD control payload exceeds 65536 bytes")]
    PayloadTooLarge,
}
