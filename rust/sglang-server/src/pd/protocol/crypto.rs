use std::fs::OpenOptions;
use std::io::Read;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;

use hkdf::Hkdf;
use sha2::{Digest, Sha256};
use thiserror::Error;
use zeroize::Zeroize;

use crate::pd::protocol::types::FixedBytes;

const HKDF_SALT_DOMAIN: &[u8] = b"SGLANG-PD-HKDF-SALT-V1";
const HKDF_INFO_DOMAIN: &[u8] = b"SGLANG-PD-CONTROL-V1";

pub struct Psk([u8; 32]);

impl Psk {
    pub fn load(path: &Path) -> Result<Self, CryptoError> {
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(path)
            .map_err(|_| CryptoError::PskOpen)?;
        let metadata = file.metadata().map_err(|_| CryptoError::PskMetadata)?;
        if !metadata.file_type().is_file() {
            return Err(CryptoError::PskNotRegular);
        }
        if metadata.mode() & 0o7777 != 0o400 {
            return Err(CryptoError::PskMode);
        }
        if metadata.len() != 32 {
            return Err(CryptoError::PskLength);
        }

        let mut bytes = [0_u8; 32];
        if file.read_exact(&mut bytes).is_err() {
            bytes.zeroize();
            return Err(CryptoError::PskRead);
        }
        let mut extra = [0_u8; 1];
        if file.read(&mut extra).map_err(|_| CryptoError::PskRead)? != 0 {
            bytes.zeroize();
            return Err(CryptoError::PskLength);
        }
        Ok(Self(bytes))
    }

    pub fn id(&self) -> [u8; 8] {
        let digest = Sha256::digest(self.0.as_slice());
        digest[..8]
            .try_into()
            .expect("SHA-256 has at least 8 bytes")
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Drop for Psk {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

pub struct SessionKeys {
    pub decode_to_prefill: [u8; 32],
    pub prefill_to_decode: [u8; 32],
}

impl Drop for SessionKeys {
    fn drop(&mut self) {
        self.decode_to_prefill.zeroize();
        self.prefill_to_decode.zeroize();
    }
}

pub fn random_nonce() -> Result<FixedBytes<32>, CryptoError> {
    let mut bytes = [0_u8; 32];
    getrandom::fill(&mut bytes).map_err(|_| CryptoError::Random)?;
    Ok(FixedBytes::new(bytes))
}

pub fn frame_hash(frame: &[u8]) -> [u8; 32] {
    Sha256::digest(frame).into()
}

pub fn transcript_hash(client_hello: &[u8], server_hello: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(client_hello);
    hasher.update(server_hello);
    hasher.finalize().into()
}

pub fn derive_session_keys(
    psk: &Psk,
    decode_nonce: FixedBytes<32>,
    prefill_nonce: FixedBytes<32>,
    transcript: FixedBytes<32>,
    decode_process_epoch: FixedBytes<16>,
    prefill_process_epoch: FixedBytes<16>,
) -> Result<SessionKeys, CryptoError> {
    let mut salt_hasher = Sha256::new();
    salt_hasher.update(HKDF_SALT_DOMAIN);
    salt_hasher.update(decode_nonce.as_bytes());
    salt_hasher.update(prefill_nonce.as_bytes());
    let salt: [u8; 32] = salt_hasher.finalize().into();

    let mut info =
        Vec::with_capacity(HKDF_INFO_DOMAIN.len() + transcript.as_bytes().len() + 16 + 16);
    info.extend_from_slice(HKDF_INFO_DOMAIN);
    info.extend_from_slice(transcript.as_bytes());
    info.extend_from_slice(decode_process_epoch.as_bytes());
    info.extend_from_slice(prefill_process_epoch.as_bytes());

    let hkdf = Hkdf::<Sha256>::new(Some(&salt), psk.as_bytes());
    let mut output = [0_u8; 64];
    hkdf.expand(&info, &mut output)
        .map_err(|_| CryptoError::Hkdf)?;
    let keys = SessionKeys {
        decode_to_prefill: output[..32]
            .try_into()
            .expect("HKDF output has a decode-to-prefill key"),
        prefill_to_decode: output[32..]
            .try_into()
            .expect("HKDF output has a prefill-to-decode key"),
    };
    output.zeroize();
    Ok(keys)
}

#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("could not securely open the PD control PSK")]
    PskOpen,
    #[error("could not inspect the opened PD control PSK")]
    PskMetadata,
    #[error("PD control PSK must be a regular file")]
    PskNotRegular,
    #[error("PD control PSK must have exact mode 0400")]
    PskMode,
    #[error("PD control PSK must contain exactly 32 raw bytes")]
    PskLength,
    #[error("could not read the PD control PSK")]
    PskRead,
    #[error("operating-system CSPRNG failed")]
    Random,
    #[error("HKDF-SHA256 expansion failed")]
    Hkdf,
}
