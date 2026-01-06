use std::{
    pin::Pin,
    task::{Context, Poll},
};

use axum::body::Body;
use bytes::Bytes;
use http_body::Frame;

pub struct AttachedBody<T: Send + 'static> {
    inner: Body,
    _attached: T,
}

impl<T: Send + 'static> AttachedBody<T> {
    pub fn new(inner: Body, attached: T) -> Self {
        Self {
            inner,
            _attached: attached,
        }
    }
}

impl<T: Send + 'static> http_body::Body for AttachedBody<T> {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

