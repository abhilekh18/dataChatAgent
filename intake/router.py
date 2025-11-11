"""Simple keyword-based intent router for dataset selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .catalog import DatasetCatalog, DatasetMetadata, tokenize


@dataclass(frozen=True)
class RouterResult:
    dataset: DatasetMetadata
    score: float
    matched_keywords: Tuple[str, ...]
    semantic_matches: Tuple[str, ...] = ()


class QuestionRouter:
    """Score datasets based on keyword overlap with the user question."""

    def __init__(self, catalog: DatasetCatalog, *, minimum_score: float = 0.1) -> None:
        self.catalog = catalog
        self.minimum_score = minimum_score

    def rank(self, question: str, *, limit: int = 3) -> List[RouterResult]:
        """Return ranked datasets for the provided question."""
        question_tokens = set(tokenize(question))
        if not question_tokens:
            return []

        results: List[RouterResult] = []
        for metadata in self.catalog:
            if not metadata.keywords:
                continue
            overlap = question_tokens.intersection(metadata.keywords)
            if not overlap:
                continue
            precision = len(overlap) / len(metadata.keywords)
            recall = len(overlap) / len(question_tokens)
            semantic_overlap: Tuple[str, ...] = ()
            if metadata.semantic:
                semantic_tokens = set(tokenize(" ".join(metadata.semantic.all_synonyms())))
                semantic_overlap = tuple(sorted(question_tokens.intersection(semantic_tokens)))
            semantic_bonus = len(semantic_overlap) / len(question_tokens) if question_tokens else 0.0
            score = (precision + recall + semantic_bonus) / 3.0
            if score < self.minimum_score:
                continue
            results.append(
                RouterResult(
                    dataset=metadata,
                    score=score,
                    matched_keywords=tuple(sorted(overlap)),
                    semantic_matches=semantic_overlap,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]


