"""Unit tests for document parser."""
import pytest
import tempfile
from pathlib import Path
from PyPDF2 import PdfWriter
from rag.document_parser import parse_document, DocumentParsingError, ParsedPage


def test_parse_txt_file(tmp_path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello world! This is a test document.\nSecond line.", encoding="utf-8")

    pages = parse_document(txt_file)
    assert len(pages) == 1
    assert pages[0].page_number is None
    assert "Hello world!" in pages[0].text
    assert "Second line." in pages[0].text


def test_parse_empty_txt_file(tmp_path):
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n  ", encoding="utf-8")

    with pytest.raises(DocumentParsingError, match="empty"):
        parse_document(empty_file)


def test_parse_unsupported_format(tmp_path):
    invalid_file = tmp_path / "image.png"
    invalid_file.write_bytes(b"fakepngbytes")

    with pytest.raises(DocumentParsingError, match="Unsupported file format"):
        parse_document(invalid_file)


def test_parse_missing_file():
    missing = Path("non_existent_file_12345.pdf")
    with pytest.raises(DocumentParsingError, match="File not found"):
        parse_document(missing)


def test_parse_corrupted_pdf(tmp_path):
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"%PDF-1.4 completely corrupted binary content")

    with pytest.raises(DocumentParsingError, match="Corrupted or invalid PDF"):
        parse_document(bad_pdf)
