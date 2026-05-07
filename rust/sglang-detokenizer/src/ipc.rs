use zmq::{Context, Socket, SocketType};

pub const MSGPACK_MAGIC: &[u8] = b"0xSG02";
pub const PICKLE_MAGIC: &[u8] = b"0xSG01";

pub struct IpcChannels {
    pub recv_from_scheduler: Socket,
    pub send_to_tokenizer: Socket,
}

impl IpcChannels {
    pub fn new(ctx: &Context, detokenizer_ipc: &str, tokenizer_ipc: &str) -> Self {
        let recv = ctx.socket(zmq::PULL).expect("Failed to create PULL socket");
        configure_socket(&recv, zmq::PULL);
        recv.bind(detokenizer_ipc)
            .unwrap_or_else(|e| panic!("Failed to bind PULL socket to {detokenizer_ipc}: {e}"));

        let send = ctx.socket(zmq::PUSH).expect("Failed to create PUSH socket");
        configure_socket(&send, zmq::PUSH);
        send.connect(tokenizer_ipc)
            .unwrap_or_else(|e| panic!("Failed to connect PUSH socket to {tokenizer_ipc}: {e}"));

        IpcChannels {
            recv_from_scheduler: recv,
            send_to_tokenizer: send,
        }
    }

    /// Receive a msgpack-encoded message. Returns the raw msgpack bytes.
    /// Panics if a pickle message is received (unsupported path).
    pub fn recv(&self) -> Vec<u8> {
        let parts = self
            .recv_from_scheduler
            .recv_multipart(0)
            .expect("Failed to receive message");

        if parts.len() == 1 {
            // Legacy single-frame pickle message
            panic!("Received pickle message (single frame). Only msgpack is supported. Set SGLANG_IPC_USE_MSGPACK=1.");
        }

        let magic = &parts[0];
        if magic == PICKLE_MAGIC {
            panic!("Received pickle message. Only msgpack is supported. Set SGLANG_IPC_USE_MSGPACK=1.");
        }
        assert_eq!(
            magic, MSGPACK_MAGIC,
            "Unknown magic number: {magic:?}"
        );
        parts[1].clone()
    }

    /// Send msgpack bytes with the magic number prefix.
    pub fn send_msgpack(&self, data: &[u8]) {
        self.send_to_tokenizer
            .send_multipart(&[MSGPACK_MAGIC, data], 0)
            .expect("Failed to send message");
    }
}

fn configure_socket(socket: &Socket, socket_type: SocketType) {
    match socket_type {
        zmq::PUSH | zmq::PULL => {
            socket.set_sndhwm(0).ok();
            socket.set_rcvhwm(0).ok();
        }
        _ => {}
    }
}
