"""Dataset catalog for discovering CSV files in the local data directory."""

from __future__ import annotations

import csv
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, Mapping, Optional, Tuple

from semantic import (
    DatasetSemantic,
    SemanticConfigError,
    SemanticModel,
    load_semantic_model,
)
from semantic.loader import DEFAULT_SEMANTIC_PATH


logger = logging.getLogger(__name__)

TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
DATE_COLUMN_HINTS = ("date", "time", "day", "month", "year", "signup", "join", "registration")
MONTH_KEYWORDS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
)
SIGNUP_KEYWORDS = ("signup", "signups", "signed", "signing", "sign", "up")


def tokenize(value: str) -> Tuple[str, ...]:
    """Tokenize a string into lowercase alphanumeric keywords."""
    return tuple(sorted({token for token in TOKEN_SPLIT_RE.split(value.lower()) if token}))


def _semantic_name_candidates(dataset_name: str) -> Tuple[str, ...]:
    """Return possible semantic keys for a dataset name with suffixes.

    Examples
    --------
    >>> _semantic_name_candidates("subscriptions_20250101")
    ('subscriptions_20250101', 'subscriptions')
    """
    segments = dataset_name.split("_")
    candidates: list[str] = []
    for length in range(len(segments), 0, -1):
        candidate = "_".join(segments[:length])
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata describing a CSV dataset discovered by the catalog."""

    name: str
    path: Path
    columns: Tuple[str, ...] = field(default_factory=tuple)
    row_count: int = 0
    keywords: Tuple[str, ...] = field(default_factory=tuple)
    semantic: Optional[DatasetSemantic] = None

    @property
    def display_name(self) -> str:
        if self.semantic:
            return self.semantic.display_name
        return self.name.replace("_", " ").title()


class DatasetCatalog:
    """Index CSV datasets stored in the project's ``data/`` directory."""

    def __init__(
        self,
        root: Path | str = Path("data"),
        *,
        auto_load: bool = True,
        semantic_path: Path | str | None = None,
        semantic_model: Optional[SemanticModel] = None,
        semantic_enabled: bool = True,
    ) -> None:
        self.root = Path(root)
        self._datasets: Dict[str, DatasetMetadata] = {}
        self._semantic_path = Path(semantic_path) if semantic_path else DEFAULT_SEMANTIC_PATH
        self._semantic_model = semantic_model
        self._semantic_model_provided = semantic_model is not None
        self._semantic_enabled = semantic_enabled
        if auto_load:
            self.refresh()

    def refresh(self) -> None:
        """Rebuild the catalog by scanning for CSV files."""
        self._datasets.clear()
        semantic_model = self._load_semantic_model()
        if not self.root.exists():
            logger.info("Catalog root %s does not exist yet; skipping refresh.", self.root)
            return

        for csv_path in sorted(self.root.glob("*.csv")):
            metadata = self._build_metadata(csv_path, semantic_model)
            if metadata is None:
                continue
            self._datasets[metadata.name] = metadata

    def _load_semantic_model(self) -> Optional[SemanticModel]:
        if not self._semantic_enabled:
            self._semantic_model = None
            return None

        if self._semantic_model_provided and self._semantic_model is not None:
            return self._semantic_model

        try:
            model = load_semantic_model(self._semantic_path)
        except SemanticConfigError as exc:
            logger.warning("Failed to load semantic layer from %s: %s", self._semantic_path, exc)
            model = SemanticModel(datasets={})
        self._semantic_model = model
        return model

    def _build_metadata(self, csv_path: Path, semantic_model: Optional[SemanticModel]) -> Optional[DatasetMetadata]:
        """Extract metadata for a single CSV file."""
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                columns = tuple(reader.fieldnames or ())
                row_count = sum(1 for _ in reader)
        except FileNotFoundError:
            logger.warning("CSV file %s disappeared during catalog build.", csv_path)
            return None
        except csv.Error as exc:
            logger.warning("Failed to parse CSV %s: %s", csv_path, exc)
            return None
        except OSError as exc:
            logger.warning("Failed to read CSV %s: %s", csv_path, exc)
            return None

        base_tokens = tokenize(csv_path.stem)
        column_tokens = [token for column in columns for token in tokenize(column)]
        column_names_lower = [column.lower() for column in columns]

        semantic = None
        if semantic_model is not None:
            for candidate in _semantic_name_candidates(csv_path.stem):
                semantic = semantic_model.get(candidate)
                if semantic is not None:
                    break
        semantic_text_parts = []
        if semantic is not None:
            semantic_text_parts.append(semantic.display_name)
            semantic_text_parts.extend(semantic.synonyms)
            semantic_text_parts.extend(semantic.all_synonyms())
            for column in semantic.columns.values():
                semantic_text_parts.append(column.name)
                if column.description:
                    semantic_text_parts.append(column.description)
                semantic_text_parts.extend(column.synonyms)
            for metric in semantic.metrics.values():
                semantic_text_parts.append(metric.name)
                if metric.description:
                    semantic_text_parts.append(metric.description)
                semantic_text_parts.extend(metric.synonyms)
        semantic_tokens = tokenize(" ".join(semantic_text_parts))

        keywords_set = set(base_tokens).union(column_tokens).union(semantic_tokens)

        if any(any(hint in column for hint in DATE_COLUMN_HINTS) for column in column_names_lower):
            keywords_set.update(MONTH_KEYWORDS)
            keywords_set.update({"between", "range"})

        if any("signup" in column for column in column_names_lower):
            keywords_set.update(SIGNUP_KEYWORDS)

        keywords = tuple(sorted(keywords_set))

        metadata = DatasetMetadata(
            name=csv_path.stem,
            path=csv_path,
            columns=columns,
            row_count=row_count,
            keywords=keywords,
            semantic=semantic,
        )
        logger.debug("Catalogued dataset %s with keywords %s", metadata.name, keywords)
        return metadata

    def __len__(self) -> int:
        return len(self._datasets)

    def __iter__(self) -> Iterator[DatasetMetadata]:
        return iter(self._datasets.values())

    def get(self, name: str) -> Optional[DatasetMetadata]:
        return self._datasets.get(name)

    def as_mapping(self) -> Mapping[str, DatasetMetadata]:
        return dict(self._datasets)

    def list(self) -> Iterable[DatasetMetadata]:
        return list(self._datasets.values())

    @property
    def semantic_model(self) -> Optional[SemanticModel]:
        """Return the currently loaded semantic model, if any."""
        return self._semantic_model


