"""Tests for the stub analysis engine."""

from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

from datachatagent.engine.analysis import AnalysisEngine, QuestionAnalysis
from datachatagent.engine.pandasai_client import PandasAISetupError
from datachatagent.intake.catalog import DatasetCatalog, DatasetMetadata


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_analysis_engine_produces_summaries(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "revenue.csv",
        header=["product", "revenue"],
        rows=[["Widget", "100"], ["Gadget", "200"]],
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True)
    metadata = catalog.get("revenue")
    assert metadata is not None

    engine = AnalysisEngine(preview_rows=1, enable_pandasai=False)
    analysis = engine.analyze("Show me revenue", [metadata])

    assert isinstance(analysis, QuestionAnalysis)
    assert analysis.datasets
    summary = analysis.datasets[0]
    assert summary.name == "revenue"
    assert summary.row_count == metadata.row_count
    assert len(summary.preview) == 1
    assert "revenue" in summary.numeric_summary


def test_analysis_engine_includes_semantic_context(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "revenue.csv",
        header=["product", "revenue"],
        rows=[["Widget", "100"]],
    )

    semantic_path = tmp_path / "semantic.yaml"
    semantic_path.write_text(
        dedent(
            """
            datasets:
              revenue:
                description: Revenue across all products.
                columns:
                  revenue:
                    description: Amount billed.
                    synonyms:
                      - sales amount
                metrics:
                  total_revenue:
                    description: Total revenue figure.
                    synonyms:
                      - sales total
            """
        ).strip(),
        encoding="utf-8",
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True, semantic_path=semantic_path)
    metadata = catalog.get("revenue")
    assert metadata is not None

    engine = AnalysisEngine(preview_rows=1, enable_pandasai=False)
    analysis = engine.analyze("How are sales totals trending?", [metadata])
    summary = analysis.datasets[0]

    assert summary.description == "Revenue across all products."
    assert "Amount billed." in summary.column_highlights
    assert "Total revenue figure." in summary.metric_highlights
    assert "sales" in summary.semantic_matches


def test_analysis_engine_supports_semantic_suffix(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "subscriptions_20250101.csv",
        header=["subscription_id", "user_id", "status"],
        rows=[["1", "1", "active"]],
    )

    semantic_path = tmp_path / "semantic.yaml"
    semantic_path.write_text(
        dedent(
            """
            datasets:
              subscriptions:
                description: Subscription records with statuses.
                metrics:
                  active_subscriptions:
                    description: Active subscription count.
                    synonyms:
                      - active subscriptions
            """
        ).strip(),
        encoding="utf-8",
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True, semantic_path=semantic_path)
    metadata = catalog.get("subscriptions_20250101")
    assert metadata is not None

    engine = AnalysisEngine(preview_rows=1, enable_pandasai=False)
    analysis = engine.analyze("Count of subscriptions currently marked active", [metadata])
    summary = analysis.datasets[0]

    assert summary.description == "Subscription records with statuses."
    assert "subscriptions" in summary.semantic_matches
    assert any("Active subscription count." in metric for metric in summary.metric_highlights)


def test_analysis_engine_reports_pandasai_setup_error(monkeypatch) -> None:
    engine = AnalysisEngine(enable_pandasai=True)
    dataset = DatasetMetadata(name="table", path=Path("table.csv"))
    df = pd.DataFrame({"value": [1]})

    class FailingClient:
        @classmethod
        def from_environment(cls, *, default_model: str):  # type: ignore[no-untyped-def]
            raise PandasAISetupError("Missing API key.")

    monkeypatch.setattr("datachatagent.engine.analysis.PandasAIClient", FailingClient)

    insight, error, queries, charts = engine._generate_insight("question", [(dataset, df)])
    assert insight is None
    assert error == "Missing API key."
    assert queries == tuple()
    assert charts == tuple()


def test_analysis_engine_handles_no_datasets() -> None:
    engine = AnalysisEngine(enable_pandasai=False)
    analysis = engine.analyze("No data question", [])
    assert not analysis.datasets
    assert analysis.notes is not None


def test_analysis_engine_removes_chart_path_insight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "revenue.csv",
        header=["product", "revenue"],
        rows=[["Widget", "100"], ["Gadget", "200"]],
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True)
    metadata = catalog.get("revenue")
    assert metadata is not None

    chart_path = tmp_path / "charts" / "chart.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.touch()

    class StubClient:
        def __init__(self) -> None:
            self._last_sql_queries: tuple = tuple()

        @classmethod
        def from_environment(cls, *, default_model: str):  # type: ignore[no-untyped-def]
            return cls()

        @property
        def last_sql_queries(self):  # type: ignore[no-untyped-def]
            return self._last_sql_queries

        def run(self, question, datasets):  # type: ignore[no-untyped-def]
            return chart_path.as_posix()

    engine = AnalysisEngine(enable_pandasai=True, chart_dir=chart_path.parent)

    def fake_load_dataframe(self, metadata, limit=None):  # type: ignore[no-untyped-def]
        return pd.DataFrame({"value": [1, 2]})

    @contextmanager
    def fake_capture_charts(self, captured):  # type: ignore[no-untyped-def]
        captured.append(chart_path.as_posix())
        yield

    monkeypatch.setattr("datachatagent.engine.analysis.PandasAIClient", StubClient)
    monkeypatch.setattr(AnalysisEngine, "_load_dataframe", fake_load_dataframe)
    monkeypatch.setattr(AnalysisEngine, "_capture_charts", fake_capture_charts)

    analysis = engine.analyze("Plot revenue", [metadata])

    assert analysis.insight is None
    assert analysis.chart_paths == (chart_path.as_posix(),)
    assert analysis.insight_error is None


def test_analysis_engine_adds_missing_chart_path_from_insight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "revenue.csv",
        header=["product", "revenue"],
        rows=[["Widget", "100"], ["Gadget", "200"]],
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True)
    metadata = catalog.get("revenue")
    assert metadata is not None

    chart_path = tmp_path / "charts" / "chart.png"
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.touch()

    relative_reference = f"exports\\charts\\{chart_path.name}"

    class StubClient:
        def __init__(self) -> None:
            self._last_sql_queries: tuple = tuple()

        @classmethod
        def from_environment(cls, *, default_model: str):  # type: ignore[no-untyped-def]
            return cls()

        @property
        def last_sql_queries(self):  # type: ignore[no-untyped-def]
            return self._last_sql_queries

        def run(self, question, datasets):  # type: ignore[no-untyped-def]
            return relative_reference

    engine = AnalysisEngine(enable_pandasai=True, chart_dir=chart_path.parent)

    def fake_load_dataframe(self, metadata, limit=None):  # type: ignore[no-untyped-def]
        return pd.DataFrame({"value": [1, 2]})

    @contextmanager
    def fake_capture_charts(self, captured):  # type: ignore[no-untyped-def]
        captured.append(chart_path.as_posix())
        yield

    monkeypatch.setattr("datachatagent.engine.analysis.PandasAIClient", StubClient)
    monkeypatch.setattr(AnalysisEngine, "_load_dataframe", fake_load_dataframe)
    monkeypatch.setattr(AnalysisEngine, "_capture_charts", fake_capture_charts)

    analysis = engine.analyze("Plot revenue", [metadata])

    assert analysis.insight is None
    assert analysis.chart_paths == (chart_path.as_posix(),)
    assert analysis.insight_error is None
    # No chart text should remain because the path is treated as an attachment
    assert analysis.insight is None


def test_analysis_engine_uses_pandasai(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "revenue.csv",
        header=["product", "revenue"],
        rows=[["Widget", "100"], ["Gadget", "200"]],
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True)
    metadata = catalog.get("revenue")
    assert metadata is not None

    engine = AnalysisEngine(preview_rows=1, enable_pandasai=True)

    def fake_load_dataframe(self, metadata, limit=None):  # type: ignore[no-untyped-def]
        return object()

    def fake_generate_insight(self, question, datasets):  # type: ignore[no-untyped-def]
        return "Insight answer", None, tuple(), ("path/to/chart.png",)

    monkeypatch.setattr(AnalysisEngine, "_load_dataframe", fake_load_dataframe)
    monkeypatch.setattr(AnalysisEngine, "_generate_insight", fake_generate_insight)

    analysis = engine.analyze("Question?", [metadata])

    assert analysis.insight == "Insight answer"
    assert analysis.insight_error is None
    assert analysis.insight_sql_queries == tuple()
    assert analysis.chart_paths == ("path/to/chart.png",)


def test_analysis_engine_infers_currency_units(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "revenue.csv",
        header=["product", "revenue"],
        rows=[["Widget", "100"], ["Gadget", "200"]],
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True)
    metadata = catalog.get("revenue")
    assert metadata is not None

    engine = AnalysisEngine(preview_rows=1, enable_pandasai=False)
    analysis = engine.analyze("How much revenue did we make?", [metadata])
    summary = analysis.datasets[0]

    assert summary.column_units.get("revenue") == "USD"


def test_capture_charts_saves_plot(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("matplotlib.pyplot")

    chart_dir = tmp_path / "charts"
    engine = AnalysisEngine(enable_pandasai=True, chart_dir=chart_dir)

    captured: list[str] = []
    with engine._capture_charts(captured):
        import matplotlib.pyplot as plt

        plt.plot([1, 2, 3], [1, 4, 9])
        plt.show()

    assert captured
    saved_path = Path(captured[0])
    assert saved_path.exists()


