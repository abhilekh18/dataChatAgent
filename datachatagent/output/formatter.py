"""Utilities for turning analysis results into Slack-friendly text."""

from __future__ import annotations

from typing import Iterable, List

from datachatagent.engine.analysis import DatasetSummary, NumericSummary, QuestionAnalysis


def _format_numeric_summary(summary: NumericSummary) -> str:
    parts = []
    if summary.minimum is not None and summary.maximum is not None:
        parts.append(f"{summary.minimum:.2f}–{summary.maximum:.2f}")
    if summary.mean is not None:
        parts.append(f"avg {summary.mean:.2f}")
    return ", ".join(parts) if parts else f"{summary.count} values"


def _format_preview_rows(summary: DatasetSummary) -> str:
    if not summary.preview:
        return "No preview available."
    rows: List[str] = []
    for row in summary.preview:
        formatted = ", ".join(f"{key}: {value}" for key, value in row.items())
        rows.append(f"• {formatted}")
    return "\n".join(rows)


def format_dataset_section(summary: DatasetSummary) -> str:
    lines = [
        f"*{summary.display_name}* — {summary.row_count} rows, {len(summary.columns)} columns",
    ]
    if summary.description:
        lines.append(summary.description)
    if summary.semantic_matches:
        lines.append(f"Matched terms: {', '.join(summary.semantic_matches)}")
    if summary.column_highlights:
        lines.append(f"Relevant columns: {', '.join(summary.column_highlights)}")
    if summary.metric_highlights:
        lines.append(f"Relevant metrics: {', '.join(summary.metric_highlights)}")
    if summary.numeric_summary:
        stats_parts: List[str] = []
        for column, column_summary in summary.numeric_summary.items():
            label = column
            unit = summary.column_units.get(column)
            if unit:
                label = f"{label} ({unit})"
            stats_parts.append(f"{label}: {_format_numeric_summary(column_summary)}")
        lines.append(f"Numeric snapshot: {', '.join(stats_parts)}")
    lines.append("Sample rows:")
    lines.append(_format_preview_rows(summary))
    return "\n".join(lines)


def format_analysis_response(analysis: QuestionAnalysis, *, include_candidates: bool = True) -> str:
    lines = [f"*Question:* {analysis.question}"]

    if analysis.insight:
        lines.append(f"*Insight:* {analysis.insight}")
    elif analysis.insight_error:
        lines.append(f"_Insight unavailable:_ {analysis.insight_error}")

    if analysis.insight_sql_queries:
        lines.append("*SQL queries:*")
        for entry in analysis.insight_sql_queries:
            details = []
            if entry.params:
                details.append(f"params={entry.params}")
            if entry.row_count is not None:
                details.append(f"rows={entry.row_count}")
            detail_suffix = f" | {' | '.join(details)}" if details else ""
            lines.append(f"• {entry.query}{detail_suffix}")

    if analysis.chart_paths:
        chart_count = len(analysis.chart_paths)
        descriptor = "chart" if chart_count == 1 else "charts"
        lines.append(f"*Charts:* {chart_count} {descriptor} attached.")

    if not analysis.datasets:
        fallback = analysis.notes or "No datasets matched your question—try clarifying your intent."
        lines.append(f"_Result:_ {fallback}")
        return "\n".join(lines)

    if include_candidates:
        sections = [format_dataset_section(summary) for summary in analysis.datasets]
        lines.append("*Candidate datasets:*")
        lines.append("\n\n".join(sections))
    if analysis.insight_error and analysis.insight:
        lines.append(f"_Additional note:_ {analysis.insight_error}")
    return "\n".join(lines)


