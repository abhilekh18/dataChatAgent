"""Tests for the dataset catalog module."""

from __future__ import annotations

import csv
from pathlib import Path
from textwrap import dedent

from datachatagent.intake.catalog import DatasetCatalog


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_catalog_handles_missing_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    catalog = DatasetCatalog(root=data_dir, auto_load=True)
    assert len(catalog) == 0


def test_catalog_reads_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_csv = data_dir / "revenue_by_product.csv"
    with sample_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["product", "revenue"])
        writer.writerow(["Widget", "100"])
        writer.writerow(["Gadget", "200"])

    catalog = DatasetCatalog(root=data_dir, auto_load=True)
    assert len(catalog) == 1

    metadata = catalog.get("revenue_by_product")
    assert metadata is not None
    assert metadata.columns == ("product", "revenue")
    assert metadata.row_count == 2
    assert "revenue" in metadata.keywords
    assert "product" in metadata.keywords


def test_catalog_merges_semantic_metadata(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_csv = data_dir / "revenue_by_product.csv"
    with sample_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["product", "revenue"])
        writer.writerow(["Widget", "100"])

    semantic_path = tmp_path / "semantic.yaml"
    _write_text(
        semantic_path,
        dedent(
            """
            datasets:
              revenue_by_product:
                display_name: Product Revenue
                description: Revenue tracked per product.
                synonyms:
                  - sales
                columns:
                  revenue:
                    description: Total revenue per product.
                    synonyms:
                      - sales_amount
                metrics:
                  total_revenue:
                    description: Overall revenue.
                    synonyms:
                      - revenue_total
            """
        ).strip(),
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True, semantic_path=semantic_path)
    metadata = catalog.get("revenue_by_product")
    assert metadata is not None
    assert metadata.display_name == "Product Revenue"
    assert metadata.semantic is not None
    assert metadata.semantic.description == "Revenue tracked per product."
    # Synonym tokens should be integrated into the keyword index.
    assert "sales" in metadata.keywords


def test_catalog_semantic_lookup_with_suffix(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_csv = data_dir / "subscriptions_20250101.csv"
    with sample_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subscription_id", "user_id", "status"])
        writer.writerow(["1", "1", "active"])

    semantic_path = tmp_path / "semantic.yaml"
    _write_text(
        semantic_path,
        dedent(
            """
            datasets:
              subscriptions:
                display_name: Subscriptions
                description: Subscription records with current status.
                synonyms:
                  - memberships
            """
        ).strip(),
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True, semantic_path=semantic_path)

    metadata = catalog.get("subscriptions_20250101")
    assert metadata is not None
    assert metadata.semantic is not None
    assert metadata.display_name == "Subscriptions"
    assert metadata.semantic.description == "Subscription records with current status."
    assert "memberships" in metadata.keywords


def test_catalog_loads_semantic_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "users.csv").write_text("user_id\n1\n", encoding="utf-8")
    (data_dir / "payments.csv").write_text("payment_id,amount\np1,10\n", encoding="utf-8")

    semantic_dir = tmp_path / "semantic"
    _write_text(
        semantic_dir / "users.yaml",
        dedent(
            """
            datasets:
              users:
                display_name: Users
            """
        ).strip(),
    )
    _write_text(
        semantic_dir / "payments.yaml",
        dedent(
            """
            datasets:
              payments:
                display_name: Payments
            """
        ).strip(),
    )

    catalog = DatasetCatalog(root=data_dir, auto_load=True, semantic_path=semantic_dir)

    users_metadata = catalog.get("users")
    payments_metadata = catalog.get("payments")

    assert users_metadata is not None and users_metadata.display_name == "Users"
    assert payments_metadata is not None and payments_metadata.display_name == "Payments"
