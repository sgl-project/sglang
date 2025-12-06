from unittest.mock import patch

import pytest
from PIL import Image

from sglang.srt.managers.io_struct import GenerateReqInput


def fake_img():
    """Create a tiny PIL image for testing."""
    return Image.new("RGB", (10, 10), color="white")


@pytest.fixture
def req_1():
    r = GenerateReqInput()
    r.text = ["<image>\nFree OCR."]
    r.image_data = None
    return r


def test_normalize_pdf_data_string_input(req_1):
    req_1.pdf_data = "https://example.com/file1.pdf"  # input is a single url string.

    with patch("sglang.srt.multimodal.mm_utils.load_pdf_as_images") as mock_loader:
        mock_loader.return_value = [fake_img(), fake_img(), fake_img()]
        req_1._normalize_pdf_data()

    assert req_1.pdf_data == ["https://example.com/file1.pdf"]
    assert len(req_1.image_data) == 3
    assert all(isinstance(img, Image.Image) for img in req_1.image_data)

    assert req_1.text == [
        "<image>\nFree OCR.",
        "<image>\nFree OCR.",
        "<image>\nFree OCR.",
    ]


@pytest.fixture
def req_2():
    r = GenerateReqInput()
    r.text = ["<image>\nFree OCR.", "<image>\nFree OCR."]
    r.image_data = None
    return r


def test_normalize_pdf_data_list_of_strings_input(req_2):
    req_2.pdf_data = ["https://example.com/file1.pdf", "https://example.com/file2.pdf"]

    with patch("sglang.srt.multimodal.mm_utils.load_pdf_as_images") as mock_loader:
        # Each PDF returns 2 images
        mock_loader.side_effect = lambda pdf_path: [fake_img(), fake_img()]
        req_2._normalize_pdf_data()

    assert req_2.pdf_data == [
        "https://example.com/file1.pdf",
        "https://example.com/file2.pdf",
    ]
    assert len(req_2.image_data) == 4  # 2 images per PDF
    assert all(isinstance(img, Image.Image) for img in req_2.image_data)

    assert req_2.text == [
        "<image>\nFree OCR.",
        "<image>\nFree OCR.",
        "<image>\nFree OCR.",
        "<image>\nFree OCR.",
    ]


def test_extract_pdf_data(req_1):
    req_1.pdf_data = ["https://example.com/file1.pdf"]

    with patch("sglang.srt.multimodal.mm_utils.load_pdf_as_images") as mock_loader:
        mock_loader.return_value = [fake_img(), fake_img()]

        req_1._extract_pdf_data()

    assert len(req_1.image_data) == 2
    assert all(isinstance(img, Image.Image) for img in req_1.image_data)
    assert req_1.text == ["<image>\nFree OCR.", "<image>\nFree OCR."]


def test_extract_pdf_data_error(req_1):
    req_1.pdf_data = ["https://example.com/file1.pdf"]

    # Mock load_pdf_as_images to raise an exception
    with patch("sglang.srt.multimodal.mm_utils.load_pdf_as_images") as mock_loader:
        mock_loader.side_effect = Exception("PDF load failed")

        # Check that _extract_pdf_data raises ValueError
        with pytest.raises(ValueError) as exc_info:
            req_1._extract_pdf_data()

    assert "Failed to extract images from PDF" in str(exc_info.value)
