"""Stub analysis engine that prepares datasets for future reasoning."""

from __future__ import annotations

import csv
import logging
import os
import statistics
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import uuid4

from intake import DatasetMetadata, tokenize


def _infer_currency_unit(*texts: Optional[str]) -> Optional[str]:
    combined = " ".join(text for text in texts if text).lower()
    if not combined:
        return None
    if any(token in combined for token in ("usd", "us dollar", "us dollars", "dollar", "dollars", "$")):
        return "USD"
    if any(token in combined for token in ("revenue", "sales")):
        return "USD"
    return None

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from .pandasai_client import (
    PandasAIClient,
    PandasAIExecutionError,
    PandasAISetupError,
    SQLQueryLogEntry,
)


@dataclass(frozen=True)
class NumericSummary:
    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None


@dataclass(frozen=True)
class DatasetSummary:
    name: str
    display_name: str
    row_count: int
    columns: Tuple[str, ...]
    preview: Tuple[Dict[str, str], ...] = field(default_factory=tuple)
    numeric_summary: Dict[str, NumericSummary] = field(default_factory=dict)
    column_units: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None
    semantic_matches: Tuple[str, ...] = field(default_factory=tuple)
    column_highlights: Tuple[str, ...] = field(default_factory=tuple)
    metric_highlights: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class QuestionAnalysis:
    question: str
    datasets: Tuple[DatasetSummary, ...]
    notes: str | None = None
    insight: str | None = None
    insight_error: str | None = None
    insight_sql_queries: Tuple[SQLQueryLogEntry, ...] = field(default_factory=tuple)
    chart_paths: Tuple[str, ...] = field(default_factory=tuple)


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _load_preview(
    metadata: DatasetMetadata, *, max_rows: int = 5
) -> Tuple[List[Dict[str, str]], int, Dict[str, NumericSummary]]:
    preview: List[Dict[str, str]] = []
    try:
        with metadata.path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = tuple(reader.fieldnames or ())
            row_count = 0
            numeric_buckets: Dict[str, List[float]] = {column: [] for column in columns}
            for row in reader:
                row_count += 1
                if len(preview) < max_rows:
                    preview.append({key: value or "" for key, value in row.items()})
                for column, value in row.items():
                    if value is None or not _is_number(value):
                        continue
                    numeric_buckets[column].append(float(value))
    except FileNotFoundError:
        return [], 0, {}

    numeric_summary: Dict[str, NumericSummary] = {}
    for column, values in numeric_buckets.items():
        if not values:
            continue
        numeric_summary[column] = NumericSummary(
            count=len(values),
            minimum=min(values),
            maximum=max(values),
            mean=statistics.fmean(values) if values else None,
        )

    return preview, row_count, numeric_summary


class AnalysisEngine:
    """Simple engine that prepares dataset context for reasoning."""

    def __init__(
        self,
        *,
        preview_rows: int = 5,
        row_limit: int = 20000,
        timeout_seconds: int = 30,
        model_name: str | None = None,
        enable_pandasai: bool = True,
        chart_dir: Path | str = Path("exports/charts"),
    ) -> None:
        self.preview_rows = preview_rows
        self.row_limit = row_limit
        self.timeout_seconds = timeout_seconds
        self.model_name = model_name or os.getenv("PANDASAI_MODEL", "gpt-4o-mini")
        self.enable_pandasai = enable_pandasai
        self._pandasai_client: Optional[PandasAIClient] = None
        self._chart_dir = Path(chart_dir)

    def analyze(self, question: str, datasets: Sequence[DatasetMetadata]) -> QuestionAnalysis:
        summaries: List[DatasetSummary] = []
        dataframes: List[Tuple[DatasetMetadata, "pd.DataFrame"]] = []
        question_tokens = set(tokenize(question))
        for metadata in datasets:
            preview, row_count, numeric_summary = _load_preview(metadata, max_rows=self.preview_rows)
            dataframe = self._load_dataframe(metadata, limit=self.row_limit)
            if dataframe is not None:
                dataframes.append((metadata, dataframe))

            description: Optional[str] = None
            semantic_matches: Tuple[str, ...] = ()
            column_highlights: List[str] = []
            metric_highlights: List[str] = []
            column_units: Dict[str, str] = {}
            if metadata.semantic:
                semantic_keywords = set(tokenize(" ".join(metadata.semantic.all_synonyms())))
                semantic_matches = tuple(sorted(question_tokens.intersection(semantic_keywords)))
                description = metadata.semantic.description

                for column in metadata.semantic.columns.values():
                    column_tokens = set(tokenize(" ".join([column.name, *column.synonyms])))
                    if question_tokens.intersection(column_tokens):
                        label = column.description or column.name.replace("_", " ").title()
                        column_highlights.append(label)
                    unit = _infer_currency_unit(
                        column.name,
                        column.description,
                        " ".join(column.synonyms),
                        metadata.semantic.display_name,
                        metadata.semantic.description,
                    )
                    if unit is None:
                        unit = _infer_currency_unit(column.name)
                    if unit:
                        column_units[column.name] = unit

                for metric in metadata.semantic.metrics.values():
                    metric_tokens = set(tokenize(" ".join([metric.name, *metric.synonyms])))
                    if question_tokens.intersection(metric_tokens):
                        label = metric.description or metric.name.replace("_", " ").title()
                        metric_highlights.append(label)
                    unit = _infer_currency_unit(
                        metric.name,
                        metric.description,
                        " ".join(metric.synonyms),
                        metadata.semantic.display_name,
                        metadata.semantic.description,
                    )
                    if unit:
                        column_units.setdefault(metric.name, unit)

            column_highlights = tuple(dict.fromkeys(column_highlights))
            metric_highlights = tuple(dict.fromkeys(metric_highlights))

            for column_name in metadata.columns:
                if column_name not in column_units:
                    unit = _infer_currency_unit(column_name, metadata.display_name)
                    if unit:
                        column_units[column_name] = unit

            summaries.append(
                DatasetSummary(
                    name=metadata.name,
                    display_name=metadata.display_name,
                    row_count=row_count or metadata.row_count,
                    columns=metadata.columns,
                    preview=tuple(preview),
                    numeric_summary=numeric_summary,
                    column_units=column_units,
                    description=description,
                    semantic_matches=semantic_matches,
                    column_highlights=column_highlights,
                    metric_highlights=metric_highlights,
                )
            )

        notes = None
        if not summaries:
            notes = "No datasets were selected for analysis."

        insight = None
        insight_error = None
        insight_sql_queries: Tuple[SQLQueryLogEntry, ...] = tuple()
        chart_paths: Tuple[str, ...] = tuple()
        if self.enable_pandasai and dataframes:
            insight, insight_error, insight_sql_queries, chart_paths = self._generate_insight(
                question, dataframes
            )

        return QuestionAnalysis(
            question=question,
            datasets=tuple(summaries),
            notes=notes,
            insight=insight,
            insight_error=insight_error,
            insight_sql_queries=insight_sql_queries,
            chart_paths=chart_paths,
        )

    def _load_dataframe(
        self, metadata: DatasetMetadata, *, limit: Optional[int] = None
    ) -> Optional["pd.DataFrame"]:
        if pd is None:  # pragma: no cover - optional dependency
            return None
        try:
            df = pd.read_csv(metadata.path)
        except Exception as exc:  # pragma: no cover - log unexpected IO errors
            logger.warning("Failed to load DataFrame for %s: %s", metadata.name, exc)
            return None

        if limit and len(df.index) > limit:
            df = df.head(limit)
        return df

    @contextmanager
    def _capture_charts(self, captured: List[str]):
        try:  # pragma: no cover - optional dependency
            import matplotlib
            matplotlib.use("Agg", force=True)  # type: ignore[attr-defined]
            import matplotlib.pyplot as plt
        except ImportError:
            yield
            return

        original_show = getattr(plt, "show", None)

        def _patched_show(*args, **kwargs):  # type: ignore[no-untyped-def]
            self._chart_dir.mkdir(parents=True, exist_ok=True)
            filename = f"chart_{uuid4().hex}.png"
            destination = self._chart_dir / filename
            try:
                plt.savefig(destination, bbox_inches="tight")
                captured.append(str(destination))
            finally:
                plt.close("all")

        plt.show = _patched_show  # type: ignore[assignment]
        try:
            yield
        finally:
            if original_show is not None:
                plt.show = original_show  # type: ignore[assignment]

    def _normalize_insight_output(
        self,
        answer: Optional[str],
        captured_charts: Sequence[str],
    ) -> Tuple[Optional[str], Tuple[str, ...]]:
        chart_paths: List[str] = list(dict.fromkeys(map(str, captured_charts)))
        additional_paths: List[str] = []

        if answer is None:
            return None, tuple(chart_paths)

        stripped = answer.strip()
        if not stripped:
            return None, tuple(chart_paths)

        lines = stripped.splitlines()
        kept_lines: List[str] = []

        for line in lines:
            raw_candidate = line.strip().strip("'\"")
            if self._is_chart_path_reference(raw_candidate, chart_paths):
                continue
            if self._is_chart_path_reference(raw_candidate, ()):
                resolved = self._resolve_existing_path(raw_candidate)
                if resolved:
                    additional_paths.append(resolved)
                    continue
            kept_lines.append(line)

        normalized_chart_paths = self._merge_chart_paths(chart_paths, additional_paths)
        cleaned = "\n".join(kept_lines).strip()

        if cleaned:
            return cleaned, normalized_chart_paths
        if normalized_chart_paths:
            return None, normalized_chart_paths
        return stripped, normalized_chart_paths

    def _is_chart_path_reference(
        self,
        candidate: str,
        chart_paths: Sequence[str],
    ) -> bool:
        if not candidate:
            return False

        normalized_candidate = candidate.replace("\\", "/")
        suffix = Path(normalized_candidate).suffix.lower()

        for chart_path in chart_paths:
            if not chart_path:
                continue
            normalized_chart = chart_path.replace("\\", "/")
            chart_name = Path(chart_path).name
            chart_suffix = Path(normalized_chart).suffix.lower()
            if (
                normalized_candidate == normalized_chart
                or normalized_candidate == chart_name
                or normalized_chart in normalized_candidate
            ):
                return True
            if chart_suffix and normalized_candidate.endswith(chart_suffix):
                if chart_name in normalized_candidate:
                    return True
                candidate_path = Path(normalized_candidate)
                candidate_parts = [part for part in candidate_path.parts if part not in {".", ""}]
                chart_parts = [
                    part for part in Path(normalized_chart).parts if part not in {".", ""}
                ]
                if chart_parts and candidate_parts[-len(chart_parts) :] == chart_parts:
                    return True

        if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
            resolved = self._resolve_existing_path(candidate)
            if resolved:
                for chart_path in chart_paths:
                    try:
                        if Path(resolved).resolve() == Path(chart_path).resolve():
                            return True
                    except Exception:
                        continue
                if not chart_paths:
                    return True
        return False

    def _resolve_existing_path(self, candidate: str) -> Optional[str]:
        if not candidate:
            return None
        path_candidate = Path(candidate)
        try:
            if path_candidate.exists():
                return str(path_candidate.resolve())
        except Exception:
            return None
        return None

    def _merge_chart_paths(
        self,
        existing_paths: Sequence[str],
        additional_paths: Sequence[str],
    ) -> Tuple[str, ...]:
        normalized: Dict[str, str] = {}
        for path in list(existing_paths) + list(additional_paths):
            if not path:
                continue
            try:
                key = str(Path(path).resolve())
            except Exception:
                key = str(Path(path))
            normalized.setdefault(key, str(path))
        return tuple(normalized.values())

    def _generate_insight(
        self,
        question: str,
        datasets: Sequence[Tuple[DatasetMetadata, "pd.DataFrame"]],
    ) -> Tuple[
        Optional[str],
        Optional[str],
        Tuple[SQLQueryLogEntry, ...],
        Tuple[str, ...],
    ]:
        """Run the PandasAI agent to answer a question using the selected datasets."""
        try:
            if self._pandasai_client is None:
                self._pandasai_client = PandasAIClient.from_environment(
                    default_model=self.model_name
                )
            captured_charts: List[str] = []
            with self._capture_charts(captured_charts):
                answer = self._pandasai_client.run(question, datasets)
            normalized_answer, normalized_charts = self._normalize_insight_output(
                answer, captured_charts
            )
            return (
                normalized_answer,
                None,
                self._pandasai_client.last_sql_queries,
                normalized_charts,
            )
        except PandasAISetupError as exc:
            return None, str(exc), tuple(), tuple()
        except PandasAIExecutionError as exc:  # pragma: no cover - runtime guardrail
            logger.error(
                "PandasAI agent failed to answer question '%s': %s", question, exc, exc_info=True
            )
            return (
                None,
                "I couldn't analyse the data automatically. Please try narrowing your question.",
                tuple(),
                tuple(),
            )



