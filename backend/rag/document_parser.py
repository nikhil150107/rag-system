"""Document parsing module for PDF and TXT files with page tracking."""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from PyPDF2 import PdfReader


class DocumentParsingError(Exception):
    """Raised when document parsing fails."""
    pass


@dataclass
class ParsedPage:
    """Represents a single page or section of text from a document."""
    page_number: Optional[int]
    text: str


def parse_document(file_path: Path, filename: Optional[str] = None) -> List[ParsedPage]:
    """
    Parse a PDF or TXT document and extract text page-by-page.

    Args:
        file_path: Absolute or relative Path to the file.
        filename: Optional explicit filename for extension checks.

    Returns:
        List of ParsedPage objects.

    Raises:
        DocumentParsingError: If file is unsupported, empty, corrupted, or missing.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise DocumentParsingError(f"File not found: {file_path}")

    target_name = filename if filename else file_path.name
    ext = Path(target_name).suffix.lower()

    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext == ".txt":
        return _parse_txt(file_path)
    else:
        raise DocumentParsingError(
            f"Unsupported file format '{ext}'. Only .pdf and .txt files are supported."
        )


def _parse_pdf(file_path: Path) -> List[ParsedPage]:
    """Extract text page-by-page from a PDF."""
    try:
        reader = PdfReader(str(file_path))
    except Exception as e:
        raise DocumentParsingError(f"Corrupted or invalid PDF file: {str(e)}") from e

    if not reader.pages:
        raise DocumentParsingError("PDF document has no pages.")

    parsed_pages: List[ParsedPage] = []
    for idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as e:
            # Handle extraction failure gracefully on specific pages
            page_text = ""

        cleaned_text = page_text.strip()
        if cleaned_text:
            parsed_pages.append(ParsedPage(page_number=idx, text=cleaned_text))

    if not parsed_pages:
        raise DocumentParsingError("No extractable text found in PDF document.")

    return parsed_pages


def _parse_txt(file_path: Path) -> List[ParsedPage]:
    """Extract text from a TXT file as a single logical document."""
    try:
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="latin-1", errors="replace")
    except Exception as e:
        raise DocumentParsingError(f"Failed to read TXT file: {str(e)}") from e

    cleaned_content = content.strip()
    if not cleaned_content:
        raise DocumentParsingError("Text file is empty.")

    return [ParsedPage(page_number=None, text=cleaned_content)]
