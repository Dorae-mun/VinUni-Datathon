from __future__ import annotations

from typing import Dict, List


def build_augmented_user_message(user_question: str, retrieved_chunks: List[Dict[str, str]]) -> str:
    if not retrieved_chunks:
        return user_question

    context_blocks: List[str] = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "unknown")
        chunk_id = chunk.get("chunk_id", "?")
        score = chunk.get("score", "0.0000")
        content = chunk.get("content", "")
        context_blocks.append(
            f"[Doc {i}] source={source} | chunk={chunk_id} | score={score}\n{content}"
        )

    context = "\n\n".join(context_blocks)
    return (
        "Use the document context below to answer the user question. "
        "If context is insufficient, say what is missing.\n\n"
        f"DOCUMENT_CONTEXT:\n{context}\n\n"
        f"USER_QUESTION:\n{user_question}"
    )

