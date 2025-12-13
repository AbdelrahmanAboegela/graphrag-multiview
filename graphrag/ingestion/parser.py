"""Document Ingestion Pipeline - Document Parser."""

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from graphrag.core.models import DocumentMetadata, DocumentType


class DocumentParser:
    """Parse documents into structured hierarchical format.

    Supports: PDF, DOCX, HTML, TXT
    """

    def __init__(self):
        """Initialize parser."""
        pass

    def compute_hash(self, content: bytes) -> str:
        """Compute SHA256 hash of content.

        Args:
            content: Raw document bytes.

        Returns:
            Hex digest.
        """
        return hashlib.sha256(content).hexdigest()

    async def parse(
        self,
        file_path: str | Path,
        metadata: DocumentMetadata | None = None,
    ) -> dict[str, Any]:
        """Parse a document file.

        Args:
            file_path: Path to document.
            metadata: Optional metadata.

        Returns:
            Parsed document structure.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        # Read file
        with open(path, "rb") as f:
            content = f.read()

        # Compute hash for deduplication
        content_hash = self.compute_hash(content)

        # Parse based on file type
        if suffix == ".pdf":
            parsed = await self._parse_pdf(path)
        elif suffix in (".docx", ".doc"):
            parsed = await self._parse_docx(path)
        elif suffix in (".html", ".htm"):
            parsed = await self._parse_html(path)
        elif suffix == ".txt":
            parsed = await self._parse_txt(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        # Build document structure
        doc_id = str(uuid4())

        return {
            "id": doc_id,
            "title": metadata.title if metadata else path.stem,
            "doc_type": metadata.doc_type.value if metadata else DocumentType.SOP.value,
            "version": metadata.version if metadata else "1.0",
            "classification": metadata.classification.value if metadata else "internal",
            "source_hash": content_hash,
            "source_path": str(path),
            "sections": parsed["sections"],
            "tables": parsed.get("tables", []),
            "images": parsed.get("images", []),
            "metadata": metadata.model_dump() if metadata else {},
        }

    async def _parse_pdf(self, path: Path) -> dict[str, Any]:
        """Parse PDF document.

        Args:
            path: Path to PDF.

        Returns:
            Parsed content.
        """
        try:
            from unstructured.partition.pdf import partition_pdf

            elements = partition_pdf(
                filename=str(path),
                strategy="hi_res",  # High resolution for tables
                extract_images_in_pdf=True,
                include_page_breaks=True,
            )

            return self._structure_elements(elements)

        except ImportError:
            # Fallback to pypdf
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            sections = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                sections.append({
                    "id": str(uuid4()),
                    "title": f"Page {i + 1}",
                    "level": 1,
                    "sequence": i,
                    "page_start": i + 1,
                    "page_end": i + 1,
                    "content": text,
                })

            return {"sections": sections, "tables": [], "images": []}

    async def _parse_docx(self, path: Path) -> dict[str, Any]:
        """Parse DOCX document.

        Args:
            path: Path to DOCX.

        Returns:
            Parsed content.
        """
        try:
            from unstructured.partition.docx import partition_docx

            elements = partition_docx(filename=str(path))
            return self._structure_elements(elements)

        except ImportError:
            from docx import Document

            doc = Document(str(path))
            sections = []
            current_section = None

            for i, para in enumerate(doc.paragraphs):
                # Check for headings
                if para.style.name.startswith("Heading"):
                    level = int(para.style.name[-1]) if para.style.name[-1].isdigit() else 1
                    current_section = {
                        "id": str(uuid4()),
                        "title": para.text,
                        "level": level,
                        "sequence": len(sections),
                        "content": "",
                    }
                    sections.append(current_section)
                elif current_section:
                    current_section["content"] += para.text + "\n"
                else:
                    sections.append({
                        "id": str(uuid4()),
                        "title": "Introduction",
                        "level": 1,
                        "sequence": 0,
                        "content": para.text + "\n",
                    })
                    current_section = sections[-1]

            return {"sections": sections, "tables": [], "images": []}

    async def _parse_html(self, path: Path) -> dict[str, Any]:
        """Parse HTML document.

        Args:
            path: Path to HTML.

        Returns:
            Parsed content.
        """
        try:
            from unstructured.partition.html import partition_html

            elements = partition_html(filename=str(path))
            return self._structure_elements(elements)

        except ImportError:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            return {
                "sections": [{
                    "id": str(uuid4()),
                    "title": "Content",
                    "level": 1,
                    "sequence": 0,
                    "content": content,
                }],
                "tables": [],
                "images": [],
            }

    async def _parse_txt(self, path: Path) -> dict[str, Any]:
        """Parse plain text document.

        Args:
            path: Path to TXT.

        Returns:
            Parsed content.
        """
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by double newlines as sections
        paragraphs = content.split("\n\n")
        sections = []

        for i, para in enumerate(paragraphs):
            if para.strip():
                sections.append({
                    "id": str(uuid4()),
                    "title": f"Section {i + 1}",
                    "level": 1,
                    "sequence": i,
                    "content": para.strip(),
                })

        return {"sections": sections, "tables": [], "images": []}

    def _structure_elements(self, elements: list) -> dict[str, Any]:
        """Structure unstructured elements into sections.

        Args:
            elements: Unstructured elements.

        Returns:
            Structured content.
        """
        sections = []
        tables = []
        images = []

        current_section = None

        for elem in elements:
            elem_type = type(elem).__name__

            if elem_type == "Title":
                current_section = {
                    "id": str(uuid4()),
                    "title": str(elem),
                    "level": 1,
                    "sequence": len(sections),
                    "content": "",
                }
                sections.append(current_section)

            elif elem_type in ("NarrativeText", "Text", "ListItem"):
                if current_section:
                    current_section["content"] += str(elem) + "\n"
                else:
                    current_section = {
                        "id": str(uuid4()),
                        "title": "Content",
                        "level": 1,
                        "sequence": 0,
                        "content": str(elem) + "\n",
                    }
                    sections.append(current_section)

            elif elem_type == "Table":
                tables.append({
                    "id": str(uuid4()),
                    "content": str(elem),
                    "section_id": current_section["id"] if current_section else None,
                })

            elif elem_type == "Image":
                images.append({
                    "id": str(uuid4()),
                    "path": getattr(elem, "metadata", {}).get("filename", ""),
                    "section_id": current_section["id"] if current_section else None,
                })

        return {"sections": sections, "tables": tables, "images": images}


# Singleton
def get_document_parser() -> DocumentParser:
    """Get document parser instance."""
    return DocumentParser()
