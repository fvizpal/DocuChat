# loader.py
import fitz  # pymupdf
import trafilatura
from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    """A loaded document with its text content and metadata."""
    text: str
    metadata: dict


def load_pdf(file_path: str) -> list[Document]:
    """
    Extract text from a PDF file, one Document per page.
    Preserves page numbers as metadata for citations later.
    """
    documents = []

    pdf = fitz.open(file_path)

    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = page.get_text()

        # Skip pages with no extractable text
        # (e.g. scanned/image-only pages)
        if not text.strip():
            continue

        documents.append(Document(
            text=text,
            metadata={
                "source": file_path,
                "page": page_num + 1,       # human-readable (1-indexed)
                "total_pages": len(pdf),
                "type": "pdf"
            }
        ))

    pdf.close()

    if not documents:
        raise ValueError(
            "No extractable text found in this PDF. "
            "It may be a scanned document — OCR support "
            "can be added later."
        )

    return documents


def load_url(url: str) -> list[Document]:
    """
    Fetch and extract clean article text from a URL.
    Trafilatura strips navigation, ads, footers automatically.
    """
    # fetch=True tells trafilatura to download the page itself
    downloaded = trafilatura.fetch_url(url)

    if not downloaded:
        raise ValueError(
            f"Could not fetch content from URL: {url}\n"
            "Check that the URL is accessible and try again."
        )

    text = trafilatura.extract(
        downloaded,
        include_comments=False,   # skip comment sections
        include_tables=True,      # keep tables — they often have key facts
        no_fallback=False         # use fallback extractor if primary fails
    )

    if not text or not text.strip():
        raise ValueError(
            "Could not extract article text from this URL. "
            "The page may require JavaScript or a login."
        )

    # For URLs we get one Document (the whole article)
    # We'll chunk it in the next step just like a PDF page
    return [Document(
        text=text,
        metadata={
            "source": url,
            "page": 1,
            "type": "url"
        }
    )]


def load_document(source: str, is_url: bool = False) -> list[Document]:
    """
    Unified entry point — routes to the right loader
    based on whether source is a file path or URL.
    """
    if is_url:
        return load_url(source)
    else:
        return load_pdf(source)