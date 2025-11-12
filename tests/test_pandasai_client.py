"""Tests for the PandasAI client helper."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from engine.pandasai_client import (
    PandasAIClient,
    PandasAIExecutionError,
    PandasAISettings,
    PandasAISetupError,
    SQLQueryLogEntry,
)
import engine.pandasai_client as pandasai_client_module


@dataclass(frozen=True)
class _MetadataStub:
    name: str
    display_name: str


def _make_settings(**overrides) -> PandasAISettings:
    defaults = dict(
        provider="fake",
        model="test-model",
        temperature=0.0,
        max_retries=1,
        verbose=False,
        save_logs=False,
        memory_size=5,
        api_key=None,
        base_url=None,
    )
    defaults.update(overrides)
    return PandasAISettings(**defaults)


def test_pandasai_client_uses_agent(monkeypatch) -> None:
    captured = {}

    class DummyAgent:
        def __init__(self, dfs, config, memory_size, description) -> None:  # type: ignore[no-untyped-def]
            captured["dfs"] = dfs
            captured["memory_size"] = memory_size
            captured["description"] = description

        def chat(self, question: str) -> str:
            captured["question"] = question
            return "agent answer"

    monkeypatch.setattr("engine.pandasai_client.Agent", DummyAgent)
    client = PandasAIClient(_make_settings())

    df = pd.DataFrame({"value": [1, 2]})
    metadata = _MetadataStub(name="table", display_name="Test Table")

    result = client.run("What is the total?", [(metadata, df)])

    assert result == "agent answer"
    assert captured["memory_size"] == 5
    assert "Test Table" in captured["description"]
    assert captured["question"] == "What is the total?"
    assert hasattr(captured["dfs"][0], "schema")
    assert captured["dfs"][0].schema.name == "table"
    assert client.last_sql_queries == tuple()


def test_pandasai_client_records_sql_queries(monkeypatch) -> None:
    class QueryAgent:
        def __init__(self, dfs, config, memory_size, description) -> None:  # type: ignore[no-untyped-def]
            self._dfs = dfs

        def chat(self, question: str) -> str:
            manager = pandasai_client_module.duckdb_manager_module.DuckDBConnectionManager()
            dataset = self._dfs[0]
            manager.register(dataset.schema.name, dataset)
            relation = manager.sql(
                f"SELECT COUNT(*) AS cnt FROM {dataset.schema.name}",
                params=None,
            )
            count = relation.df().iloc[0, 0]
            return f"{question}: {int(count)}"

    monkeypatch.setattr("engine.pandasai_client.Agent", QueryAgent)
    client = PandasAIClient(_make_settings())

    df = pd.DataFrame({"value": [1, 2, 3]})
    metadata = _MetadataStub(name="events", display_name="Events")

    result = client.run("How many rows?", [(metadata, df)])

    assert result == "How many rows?: 3"
    assert client.last_sql_queries, "Expected DuckDB queries to be recorded"
    for entry in client.last_sql_queries:
        assert isinstance(entry, SQLQueryLogEntry)
        assert entry.row_count == 1
    assert any("FROM events" in entry.query for entry in client.last_sql_queries)


def test_pandasai_client_rewrites_strftime(monkeypatch) -> None:
    class StrftimeAgent:
        def __init__(self, dfs, config, memory_size, description) -> None:  # type: ignore[no-untyped-def]
            self._dfs = dfs

        def chat(self, question: str) -> str:
            manager = pandasai_client_module.duckdb_manager_module.DuckDBConnectionManager()
            dataset = self._dfs[0]
            manager.register(dataset.schema.name, dataset)
            relation = manager.sql(
                f"""
                SELECT strftime('%Y', CAST(start_date AS DATE)) AS year_value
                FROM {dataset.schema.name}
                ORDER BY start_date
                """,
                params=None,
            )
            frame = relation.df()
            return ", ".join(frame["year_value"].astype(str))

    monkeypatch.setattr("engine.pandasai_client.Agent", StrftimeAgent)
    client = PandasAIClient(_make_settings())

    df = pd.DataFrame({"start_date": ["2024-01-01", "2025-02-15"]})
    metadata = _MetadataStub(name="subscriptions", display_name="Subscriptions")

    result = client.run("Which years?", [(metadata, df)])

    assert result == "2024, 2025"
    assert client.last_sql_queries, "Expected query log entries"
    rewritten_queries = [entry.query.lower() for entry in client.last_sql_queries]
    assert any("strftime(cast(start_date as date)" in q for q in rewritten_queries)
    assert any(("'%y'" in q) or ('"%y"' in q) for q in rewritten_queries)


def test_pandasai_client_rewrites_year(monkeypatch) -> None:
    class YearAgent:
        def __init__(self, dfs, config, memory_size, description) -> None:  # type: ignore[no-untyped-def]
            self._dfs = dfs

        def chat(self, question: str) -> str:
            manager = pandasai_client_module.duckdb_manager_module.DuckDBConnectionManager()
            dataset = self._dfs[0]
            manager.register(dataset.schema.name, dataset)
            relation = manager.sql(
                f"""
                SELECT year(start_date) AS year_value
                FROM {dataset.schema.name}
                ORDER BY start_date
                """,
                params=None,
            )
            frame = relation.df()
            return ", ".join(frame["year_value"].astype(str))

    monkeypatch.setattr("engine.pandasai_client.Agent", YearAgent)
    client = PandasAIClient(_make_settings())

    df = pd.DataFrame({"start_date": ["2024-01-01", "2025-02-15"]})
    metadata = _MetadataStub(name="subscriptions", display_name="Subscriptions")

    result = client.run("Which years?", [(metadata, df)])

    assert result == "2024, 2025"
    assert client.last_sql_queries, "Expected query log entries"
    normalized_queries = [entry.query.lower().replace(" ", "") for entry in client.last_sql_queries]
    assert any("year(cast(start_dateasdate))" in q for q in normalized_queries)

def test_pandasai_client_requires_key_for_openai() -> None:
    with pytest.raises(PandasAISetupError):
        PandasAIClient(_make_settings(provider="openai"))


def test_pandasai_client_wraps_agent_errors(monkeypatch) -> None:
    class FailingAgent:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            pass

        def chat(self, question: str) -> str:
            raise ValueError("boom")

    monkeypatch.setattr("engine.pandasai_client.Agent", FailingAgent)
    client = PandasAIClient(_make_settings())

    df = pd.DataFrame({"value": [1]})
    metadata = _MetadataStub(name="table", display_name="Test Table")

    with pytest.raises(PandasAIExecutionError):
        client.run("question", [(metadata, df)])

