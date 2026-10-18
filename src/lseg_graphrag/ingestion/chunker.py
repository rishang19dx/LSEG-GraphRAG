"""Section-aware document chunker preserving hierarchical provenance metadata."""

import re
from typing import List, Optional
from lseg_graphrag.schema.provenance import DocumentMetadata, ChunkMetadata


class SectionAwareChunker:
    """Chunks text documents while preserving document, section, and company provenance."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, respect_paragraphs: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_paragraphs = respect_paragraphs

    def chunk_section(
        self,
        doc_meta: DocumentMetadata,
        section_key: str,
        section_title: str,
        text: str,
    ) -> List[ChunkMetadata]:
        """Split a document section into structured chunks with provenance."""
        cleaned_text = self._clean_text(text)
        if not cleaned_text:
            return []

        if self.respect_paragraphs:
            raw_chunks = self._chunk_by_paragraphs(cleaned_text)
        else:
            raw_chunks = self._chunk_by_characters(cleaned_text)

        chunks: List[ChunkMetadata] = []
        for idx, chunk_text in enumerate(raw_chunks):
            # Estimate token count (~4 characters per token heuristic)
            token_est = max(1, len(chunk_text) // 4)
            chunk_id = f"{doc_meta.doc_id}_{section_key}_c{idx:03d}"

            chunk = ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_meta.doc_id,
                company_symbol=doc_meta.company_symbol,
                section=section_key,
                section_title=section_title,
                chunk_index=idx,
                text=chunk_text,
                token_count=token_est,
                metadata={
                    "cik": doc_meta.cik,
                    "filing_date": doc_meta.filing_date,
                    "form_type": doc_meta.form_type,
                },
            )
            chunks.append(chunk)

        return chunks

    def _clean_text(self, text: str) -> str:
        """Normalize whitespaces and remove control characters."""
        text = re.sub(r"\r\n?", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk_by_paragraphs(self, text: str) -> List[str]:
        """Chunk text respecting paragraph boundaries while adhering to chunk_size and overlap."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return [text]

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for para in paragraphs:
            para_len = len(para)

            # If a single paragraph exceeds chunk size, split it by sentences or characters
            if para_len > self.chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                sub_chunks = self._chunk_by_characters(para)
                chunks.extend(sub_chunks)
                continue

            if current_length + para_len + (2 if current_chunk else 0) <= self.chunk_size:
                current_chunk.append(para)
                current_length += para_len + (2 if len(current_chunk) > 1 else 0)
            else:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))

                # Handle overlap: take trailing paragraphs if possible
                overlap_chunk: List[str] = []
                overlap_len = 0
                for prev_para in reversed(current_chunk):
                    if overlap_len + len(prev_para) <= self.chunk_overlap:
                        overlap_chunk.insert(0, prev_para)
                        overlap_len += len(prev_para)
                    else:
                        break

                current_chunk = overlap_chunk + [para]
                current_length = sum(len(p) for p in current_chunk) + 2 * (len(current_chunk) - 1)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def _chunk_by_characters(self, text: str) -> List[str]:
        """Fallback character-level sliding window chunker."""
        chunks: List[str] = []
        start = 0
        text_len = len(text)
        step = max(1, self.chunk_size - self.chunk_overlap)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == text_len:
                break
            start += step

        return chunks
