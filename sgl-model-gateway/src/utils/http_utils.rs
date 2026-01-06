use std::{
    pin::Pin,
    task::{Context, Poll},
};

use axum::body::Body;
use bytes::Bytes;
use http_body::Frame;

pub struct AttachedBody<T> {
    inner: Body,
    _attached: T,
}

impl<T> AttachedBody<T> {
    pub fn new(inner: Body, attached: T) -> Self {
        Self {
            inner,
            _attached: attached,
        }
    }
}

impl<T: Send + Unpin + 'static> AttachedBody<T> {
    pub fn wrap_response(
        response: axum::response::Response,
        attached: T,
    ) -> axum::response::Response {
        let (parts, body) = response.into_parts();
        axum::response::Response::from_parts(parts, Body::new(Self::new(body, attached)))
    }
}

impl<T: Send + Unpin + 'static> http_body::Body for AttachedBody<T> {
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
