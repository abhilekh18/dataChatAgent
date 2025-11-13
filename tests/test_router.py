"""Tests for the dataset intent router."""

from __future__ import annotations

import csv
from pathlib import Path
from textwrap import dedent

from datachatagent.intake.catalog import DatasetCatalog
from datachatagent.intake.router import QuestionRouter


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def test_router_ranks_relevant_datasets(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "revenue_by_product.csv",
        header=["product", "revenue"],
        rows=[["Widget", "100"]],
    )
    _write_csv(
        data_dir / "support_tickets.csv",
        header=["ticket_id", "status"],
        rows=[["1", "open"]],
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True)
    router = QuestionRouter(catalog)

    results = router.rank("What's our product revenue this month?", limit=5)
    assert results
    top = results[0]
    assert top.dataset.name == "revenue_by_product"
    assert "revenue" in top.matched_keywords
    assert 0 < top.score <= 1


def test_router_handles_no_matches(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "support_tickets.csv",
        header=["ticket_id", "status"],
        rows=[["1", "open"]],
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True)
    router = QuestionRouter(catalog)

    results = router.rank("Tell me about revenue", limit=5)
    assert not results


def test_router_uses_semantic_synonyms(tmp_path: Path) -> None:
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
                synonyms:
                  - memberships
                metrics:
                  active_subscriptions:
                    synonyms:
                      - active subscriptions
            """
        ).strip(),
        encoding="utf-8",
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True, semantic_path=semantic_path)
    router = QuestionRouter(catalog)

    results = router.rank("What is the count of subscriptions currently marked active?", limit=3)
    assert results
    top = results[0]
    assert top.dataset.name == "subscriptions_20250101"
    assert "subscriptions" in top.semantic_matches
    assert top.score > 0.0


def test_router_matches_country_question(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "users_20250101.csv",
        header=["user_id", "signup_date", "country"],
        rows=[
            ["1", "2024-01-01", "US"],
            ["2", "2024-01-02", "GB"],
        ],
    )

    semantic_path = tmp_path / "semantic.yaml"
    semantic_path.write_text(
        dedent(
            """
            datasets:
              users:
                display_name: Users
                description: Registered users and their signup profile.
                synonyms:
                  - customers
                columns:
                  country:
                    description: Reported country for the user.
                    synonyms:
                      - locale
            """
        ).strip(),
        encoding="utf-8",
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True, semantic_path=semantic_path)
    router = QuestionRouter(catalog)

    results = router.rank("What are the top 10 reported countries for the users?", limit=3)
    assert results
    top = results[0]
    assert top.dataset.name == "users_20250101"
    assert "reported" in top.matched_keywords or "users" in top.matched_keywords
    assert top.score > 0.0


def test_router_matches_signup_between_dates(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_csv(
        data_dir / "users.csv",
        header=["user_id", "signup_date"],
        rows=[["1", "2025-03-01"]],
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True)
    router = QuestionRouter(catalog)

    results = router.rank("How many users signed up between March 2025 and June 2025?")
    assert results
    top = results[0]
    assert top.dataset.name == "users"
    assert "users" in top.matched_keywords or "between" in top.matched_keywords
    assert top.score > 0.0