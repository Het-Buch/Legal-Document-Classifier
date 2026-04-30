# src/preprocessing/chunker.py

from typing import List, Dict, Any


def chunk_tokens(
    token_ids: List[int],
    max_len: int = 512,
    stride: int = 128
) -> List[Dict[str, Any]]:
    """
    Sliding-window chunking over token IDs.

    Returns a list of chunk metadata dictionaries:
    [
        {
            "chunk_id": int,
            "start": int,
            "end": int
        },
        ...
    ]

    This format is REQUIRED by:
    - MTL inference
    - Streamlit chunk highlighting
    - Document-level aggregation
    """

    chunks = []
    start = 0
    chunk_id = 0
    n_tokens = len(token_ids)

    while start < n_tokens:
        end = min(start + max_len, n_tokens)

        chunks.append({
            "chunk_id": chunk_id,
            "start": start,
            "end": end
        })

        if end >= n_tokens:
            break

        start += max_len - stride
        chunk_id += 1

    return chunks
