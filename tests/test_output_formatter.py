"""Tests for Slack formatting helpers."""

from __future__ import annotations

from engine.analysis import DatasetSummary, NumericSummary, QuestionAnalysis
from engine.pandasai_client import SQLQueryLogEntry
from output.formatter import format_analysis_response, format_dataset_section


def _sample_summary() -> DatasetSummary:
    return DatasetSummary(
        name="revenue",
        display_name="Revenue",
        row_count=10,
        columns=("product", "revenue"),
        preview=(
            {"product": "Widget", "revenue": "100"},
            {"product": "Gadget", "revenue": "200"},
        ),
        numeric_summary={
            "revenue": NumericSummary(count=2, minimum=100.0, maximum=200.0, mean=150.0),
        },
        column_units={"revenue": "USD"},
    )


def test_format_dataset_section_includes_preview() -> None:
    section = format_dataset_section(_sample_summary())
    assert "Sample rows" in section
    assert "Widget" in section
    assert "Numeric snapshot" in section


def test_format_analysis_response_handles_no_datasets() -> None:
    analysis = QuestionAnalysis(
        question="What is revenue?",
        datasets=tuple(),
        notes="No match",
        insight_error="error",
    )
    response = format_analysis_response(analysis)
    assert "No match" in response
    assert "error" in response


def test_format_analysis_response_includes_sections() -> None:
    analysis = QuestionAnalysis(
        question="What is revenue?",
        datasets=(_sample_summary(),),
        insight="Revenue increased by 10% MoM.",
        insight_sql_queries=(
            SQLQueryLogEntry(query="SELECT 1", params=None, row_count=1),
        ),
    )
    response = format_analysis_response(analysis)
    assert "*Question:*" in response
    assert "Revenue" in response
    assert "Insight" in response
    assert "*SQL queries:*" in response
    assert "rows=1" in response


def test_format_analysis_response_can_hide_candidates() -> None:
    analysis = QuestionAnalysis(
        question="What is revenue?",
        datasets=(_sample_summary(),),
        insight="Revenue increased by 10% MoM.",
    )
    response = format_analysis_response(analysis, include_candidates=False)
    assert "*Candidate datasets:*" not in response
    assert "*Revenue* —" not in response
    assert "(USD)" not in response


def test_format_analysis_response_mentions_charts() -> None:
    analysis = QuestionAnalysis(
        question="Show chart",
        datasets=(_sample_summary(),),
        chart_paths=("exports/charts/chart.png",),
    )
    response = format_analysis_response(analysis)
    assert "*Charts:* 1 chart attached." in response


def test_dataset_section_includes_currency_unit() -> None:
    section = format_dataset_section(_sample_summary())
    assert "(USD)" in section


