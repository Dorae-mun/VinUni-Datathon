from __future__ import annotations

import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".docx"}


def _read_text_file(file_path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1258", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _read_docx_file(file_path: Path) -> str:
    try:
        with zipfile.ZipFile(file_path) as zf:
            xml_bytes = zf.read("word/document.xml")
    except Exception:
        return ""

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    text_nodes = root.findall(".//w:t", namespace)
    return "\n".join(node.text for node in text_nodes if node.text)


def _chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start = max(0, end - overlap)
    return chunks


def _collect_document_chunks(
    raw_docs_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[str], List[Dict[str, str]]]:
    texts: List[str] = []
    metadatas: List[Dict[str, str]] = []

    for file_path in sorted(raw_docs_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if file_path.suffix.lower() == ".docx":
            content = _read_docx_file(file_path).strip()
        else:
            content = _read_text_file(file_path).strip()
        if not content:
            continue

        for i, chunk in enumerate(
            _chunk_text(content, chunk_size=chunk_size, overlap=chunk_overlap),
            start=1,
        ):
            texts.append(chunk)
            metadatas.append(
                {
                    "source": str(file_path),
                    "chunk_id": str(i),
                }
            )
    return texts, metadatas


def ingest_documents(
    raw_docs_dir: Path,
    rag_dir: Path,
    *,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
    max_features: int = 12000,
) -> Dict[str, int]:
    raw_docs_dir = Path(raw_docs_dir)
    rag_dir = Path(rag_dir)
    rag_dir.mkdir(parents=True, exist_ok=True)

    texts, metadatas = _collect_document_chunks(
        raw_docs_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not texts:
        payload = {"metadatas": [], "texts": []}
        with open(rag_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {"documents": 0, "chunks": 0}

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words=None,
    )
    matrix = vectorizer.fit_transform(texts)

    joblib.dump(vectorizer, rag_dir / "vectorizer.joblib")
    joblib.dump(matrix, rag_dir / "matrix.joblib")
    with open(rag_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"metadatas": metadatas, "texts": texts}, f, ensure_ascii=False, indent=2)

    unique_docs = len({m["source"] for m in metadatas})
    return {"documents": unique_docs, "chunks": len(texts)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build local RAG index from data/raw_docs")
    parser.add_argument("--raw-docs", type=str, default="data/raw_docs")
    parser.add_argument("--rag-dir", type=str, default="data/rag")
    args = parser.parse_args()

    stats = ingest_documents(Path(args.raw_docs), Path(args.rag_dir))
    print(f"Indexed {stats['documents']} docs / {stats['chunks']} chunks -> {args.rag_dir}")

