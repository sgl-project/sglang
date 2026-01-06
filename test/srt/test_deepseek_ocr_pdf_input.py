import unittest
from unittest.mock import patch

from PIL import Image

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.test_utils import CustomTestCase


def fake_img():
    """Create a tiny PIL image for testing."""
    return Image.new("RGB", (10, 10), color="white")


def req_1():
    r = GenerateReqInput()
    r.text = ["<image>\nFree OCR."]
    r.image_data = None
    return r


def req_2():
    r = GenerateReqInput()
    r.text = ["<image>\nFree OCR.", "<image>\nFree OCR."]
    r.image_data = None
    return r


class TestPdfInput(CustomTestCase):
    def test_normalize_pdf_data_string_input(self):
        r = req_1()
        r.pdf_data = "https://example.com/file1.pdf"  # input is a single url string.

        with patch("sglang.srt.multimodal.mm_utils.load_pdf_as_images") as mock_loader:
            mock_loader.return_value = [fake_img(), fake_img(), fake_img()]
            r._normalize_pdf_data()

        self.assertEqual(r.pdf_data, ["https://example.com/file1.pdf"])
        self.assertEqual(len(r.image_data), 3)
        self.assertTrue(all(isinstance(img, Image.Image) for img in r.image_data))

        self.assertEqual(
            r.text, ["<image>\nFree OCR.", "<image>\nFree OCR.", "<image>\nFree OCR."]
        )

    def test_normalize_pdf_data_list_of_strings_input(self):
        r = req_2()
        r.pdf_data = ["https://example.com/file1.pdf", "https://example.com/file2.pdf"]

        with patch("sglang.srt.multimodal.mm_utils.load_pdf_as_images") as mock_loader:
            # Each PDF returns 2 images
            mock_loader.side_effect = lambda pdf_path: [fake_img(), fake_img()]
            r._normalize_pdf_data()

        self.assertEqual(
            r.pdf_data,
            ["https://example.com/file1.pdf", "https://example.com/file2.pdf"],
        )
        self.assertEqual(len(r.image_data), 4)  # 2 images per PDF
        self.assertTrue(all(isinstance(img, Image.Image) for img in r.image_data))

        self.assertEqual(
            r.text,
            [
                "<image>\nFree OCR.",
                "<image>\nFree OCR.",
                "<image>\nFree OCR.",
                "<image>\nFree OCR.",
            ],
        )

    def test_extract_pdf_data(self):
        r = req_1()
        r.pdf_data = ["https://example.com/file1.pdf"]

        with patch("sglang.srt.multimodal.mm_utils.load_pdf_as_images") as mock_loader:
            mock_loader.return_value = [fake_img(), fake_img()]

            r._extract_pdf_data()

        self.assertEqual(len(r.image_data), 2)
        self.assertTrue(all(isinstance(img, Image.Image) for img in r.image_data))
        self.assertEqual(r.text, ["<image>\nFree OCR.", "<image>\nFree OCR."])

    def test_extract_pdf_data_error(self):
        r = req_1()
        r.pdf_data = ["https://example.com/file1.pdf"]

        # Mock load_pdf_as_images to raise an exception
        with patch("sglang.srt.multimodal.mm_utils.load_pdf_as_images") as mock_loader:
            mock_loader.side_effect = Exception("PDF load failed")

            # Check that _extract_pdf_data raises ValueError
            with self.assertRaises(ValueError) as context:
                r._extract_pdf_data()

        self.assertIn("Failed to extract images from PDF", str(context.exception))


if __name__ == "__main__":
    unittest.main()
