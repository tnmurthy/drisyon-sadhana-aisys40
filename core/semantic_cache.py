"""
Drisyon Sadhana · AI System Design Starter Kit
Module 3: Multi-Tier Semantic Cache
"""

import math
from typing import List, Dict, Optional, Any


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if len(v1) != len(v2):
        raise ValueError("Vector dimensions must match.")

    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


class SemanticCache:
    """In-memory semantic cache using vector similarity."""

    def __init__(self, similarity_threshold: float = 0.92):
        self.threshold = similarity_threshold
        self.entries: List[Dict[str, Any]] = []

    def set(self, query: str, embedding: List[float], response: str) -> None:
        """Stores a query, its embedding vector, and corresponding response."""
        self.entries.append({
            "query": query,
            "embedding": embedding,
            "response": response
        })

    def get(self, query_embedding: List[float]) -> Optional[Dict[str, Any]]:
        """
        Retrieves cached response if cosine similarity >= threshold.
        Returns dict with response and similarity score, or None.
        """
        best_match = None
        highest_sim = -1.0

        for entry in self.entries:
            sim = cosine_similarity(query_embedding, entry["embedding"])
            if sim > highest_sim:
                highest_sim = sim
                best_match = entry

        if highest_sim >= self.threshold and best_match is not None:
            return {
                "response": best_match["response"],
                "similarity": round(highest_sim, 4),
                "hit": True
            }

        return None
