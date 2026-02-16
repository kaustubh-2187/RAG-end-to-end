"""
Tests for individual document loaders.
Tests PDF, DOCX, and TXT loading independently.
"""

import pytest
from pathlib import Path
from multi_doc_chat.utils.document_ops import load_documents


@pytest.fixture
def sample_txt_file(tmp_path):
    """Create a temporary TXT file."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("This is a test text document.", encoding="utf-8")
    return txt_file


@pytest.fixture
def sample_pdf_file(tmp_path):
    """Create a temporary PDF file using pypdf."""
    from pypdf import PdfWriter
    
    pdf_file = tmp_path / "test.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    
    # Add some text (simple approach)
    with open(pdf_file, "wb") as f:
        writer.write(f)
    
    return pdf_file


@pytest.fixture
def sample_docx_file(tmp_path):
    """Create a temporary DOCX file."""
    from docx import Document as DocxDocument
    
    docx_file = tmp_path / "test.docx"
    doc = DocxDocument()
    doc.add_paragraph("This is a test Word document.")
    doc.save(str(docx_file))
    
    return docx_file


def test_txt_loader(sample_txt_file):
    """Test TextLoader loads .txt files correctly."""
    docs = load_documents([sample_txt_file])
    
    assert len(docs) == 1
    assert "test text document" in docs[0].page_content
    assert docs[0].metadata["source"] == str(sample_txt_file)


def test_pdf_loader(sample_pdf_file):
    """Test PyPDFLoader loads .pdf files without errors."""
    docs = load_documents([sample_pdf_file])
    
    assert len(docs) >= 1  # At least one page
    assert docs[0].metadata["source"] == str(sample_pdf_file)
    # PDF might be blank, just ensure no crash


def test_docx_loader(sample_docx_file):
    """Test Docx2txtLoader loads .docx files correctly."""
    docs = load_documents([sample_docx_file])
    
    assert len(docs) == 1
    assert "test Word document" in docs[0].page_content
    assert docs[0].metadata["source"] == str(sample_docx_file)


def test_multiple_documents(sample_txt_file, sample_pdf_file):
    """Test loading multiple documents at once."""
    docs = load_documents([sample_txt_file, sample_pdf_file])
    
    assert len(docs) >= 2  # At least 1 TXT + 1 PDF page
    sources = [doc.metadata.get("source") for doc in docs]
    assert str(sample_txt_file) in sources
    assert str(sample_pdf_file) in sources


def test_unsupported_extension_skipped(tmp_path):
    """Test that unsupported file extensions are skipped."""
    unsupported = tmp_path / "test.xyz"
    unsupported.write_text("test", encoding="utf-8")
    
    docs = load_documents([unsupported])
    
    assert len(docs) == 0  # Should skip unsupported files


def test_empty_path_list():
    """Test handling of empty path list."""
    docs = load_documents([])
    assert len(docs) == 0


def test_nonexistent_file_raises_error(tmp_path):
    """Test that loading nonexistent file raises error."""
    fake_file = tmp_path / "nonexistent.txt"
    
    with pytest.raises(Exception):  # Should raise DocumentPortalException or similar
        load_documents([fake_file])
