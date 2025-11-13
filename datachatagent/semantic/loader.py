"""Utilities for loading semantic metadata from YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import yaml


class SemanticConfigError(ValueError):
    """Raised when the semantic configuration file contains invalid data."""


@dataclass(frozen=True)
class ColumnSemantic:
    """Semantic metadata for a single dataset column."""

    name: str
    description: Optional[str]
    synonyms: Tuple[str, ...]


@dataclass(frozen=True)
class MetricSemantic:
    """Simple metric definition for a dataset."""

    name: str
    description: Optional[str]
    expression: Optional[str]
    synonyms: Tuple[str, ...]


@dataclass(frozen=True)
class DatasetSemantic:
    """Semantic metadata describing a dataset."""

    name: str
    display_name: str
    description: Optional[str]
    synonyms: Tuple[str, ...]
    columns: Mapping[str, ColumnSemantic]
    metrics: Mapping[str, MetricSemantic]

    def all_synonyms(self) -> Tuple[str, ...]:
        """Return the dataset-level synonyms combined with column/metric aliases."""
        column_aliases = (alias for column in self.columns.values() for alias in column.synonyms)
        column_names = (column.name for column in self.columns.values())
        metric_aliases = (alias for metric in self.metrics.values() for alias in metric.synonyms)
        metric_names = (metric.name for metric in self.metrics.values())
        combined = set(self.synonyms)
        combined.add(self.name)
        combined.add(self.display_name)
        combined.update(column_names)
        combined.update(column_aliases)
        combined.update(metric_names)
        combined.update(metric_aliases)
        return tuple(sorted(combined))


@dataclass(frozen=True)
class SemanticModel:
    """Collection of semantic metadata keyed by dataset name."""

    datasets: Mapping[str, DatasetSemantic]

    def get(self, name: str) -> Optional[DatasetSemantic]:
        """Retrieve semantics for a dataset if present."""
        return self.datasets.get(name)

    def __iter__(self) -> Iterable[DatasetSemantic]:
        return iter(self.datasets.values())


DEFAULT_SEMANTIC_PATH = Path(__file__).parent


def load_semantic_model(path: Path | str = DEFAULT_SEMANTIC_PATH) -> SemanticModel:
    """Load semantic YAML files into a structured model.

    Parameters
    ----------
    path:
        Location of the semantic configuration. Can point to a directory of
        YAML files or a single YAML file. Defaults to the ``semantic`` package
        directory.

    Returns
    -------
    SemanticModel
        The parsed semantic configuration. Missing files result in an empty model.

    Raises
    ------
    SemanticConfigError
        If the YAML files contain invalid structure or types.
    """

    semantic_path = Path(path)
    if not semantic_path.exists():
        return SemanticModel(datasets={})

    yaml_files = _collect_yaml_files(semantic_path)
    if not yaml_files:
        return SemanticModel(datasets={})

    datasets: Dict[str, DatasetSemantic] = {}
    for yaml_file in yaml_files:
        dataset_name, dataset_semantic = _load_single_dataset(yaml_file)
        if dataset_name in datasets:
            raise SemanticConfigError(
                f"Duplicate dataset definition '{dataset_name}' found in {yaml_file}."
            )
        datasets[dataset_name] = dataset_semantic

    return SemanticModel(datasets=datasets)


def _collect_yaml_files(path: Path) -> Sequence[Path]:
    if path.is_dir():
        files = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in {".yaml", ".yml"}
        )
        return files
    if path.is_file():
        return (path,)
    return ()


def _load_single_dataset(yaml_file: Path) -> Tuple[str, DatasetSemantic]:
    try:
        raw = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - YAML parsing errors
        raise SemanticConfigError(f"Failed to parse semantic YAML {yaml_file}: {exc}") from exc
    except OSError as exc:
        raise SemanticConfigError(f"Failed to read semantic YAML {yaml_file}: {exc}") from exc

    datasets_section = raw.get("datasets") if isinstance(raw, dict) else None
    if datasets_section is None:
        raise SemanticConfigError(
            f"Semantic YAML {yaml_file} must contain a top-level 'datasets' mapping."
        )
    if not isinstance(datasets_section, dict):
        raise SemanticConfigError(
            f"'datasets' in {yaml_file} must be a mapping of dataset names to definitions."
        )

    entries = list(datasets_section.items())
    if not entries:
        raise SemanticConfigError(f"{yaml_file} does not define any datasets.")
    if len(entries) != 1:
        raise SemanticConfigError(f"{yaml_file} must define exactly one dataset.")

    dataset_name, dataset_payload = entries[0]
    if not isinstance(dataset_payload, dict):
        raise SemanticConfigError(f"Dataset '{dataset_name}' in {yaml_file} must be a mapping.")

    dataset = _build_dataset_semantic(dataset_name, dataset_payload)
    return dataset_name, dataset


def _build_dataset_semantic(dataset_name: str, dataset_payload: Mapping[str, object]) -> DatasetSemantic:
    display_name = str(dataset_payload.get("display_name") or dataset_name.replace("_", " ").title())
    description = _optional_str(dataset_payload.get("description"))
    synonyms = _normalize_synonyms(dataset_payload.get("synonyms"))

    columns_payload = dataset_payload.get("columns", {})
    columns = _parse_columns(dataset_name, columns_payload)

    metrics_payload = dataset_payload.get("metrics", {})
    metrics = _parse_metrics(dataset_name, metrics_payload)

    return DatasetSemantic(
        name=dataset_name,
        display_name=display_name,
        description=description,
        synonyms=synonyms,
        columns=columns,
        metrics=metrics,
    )


def _parse_columns(dataset_name: str, payload: object) -> Mapping[str, ColumnSemantic]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise SemanticConfigError(f"'columns' for dataset '{dataset_name}' must be a mapping.")

    columns: Dict[str, ColumnSemantic] = {}
    for column_name, column_payload in payload.items():
        if not isinstance(column_payload, dict):
            raise SemanticConfigError(f"Column '{column_name}' in dataset '{dataset_name}' must be a mapping.")
        description = _optional_str(column_payload.get("description"))
        synonyms = _normalize_synonyms(column_payload.get("synonyms"))
        columns[column_name] = ColumnSemantic(
            name=column_name,
            description=description,
            synonyms=synonyms,
        )
    return columns


def _parse_metrics(dataset_name: str, payload: object) -> Mapping[str, MetricSemantic]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise SemanticConfigError(f"'metrics' for dataset '{dataset_name}' must be a mapping.")

    metrics: Dict[str, MetricSemantic] = {}
    for metric_name, metric_payload in payload.items():
        if not isinstance(metric_payload, dict):
            raise SemanticConfigError(f"Metric '{metric_name}' in dataset '{dataset_name}' must be a mapping.")
        description = _optional_str(metric_payload.get("description"))
        expression = _optional_str(metric_payload.get("expression"))
        synonyms = _normalize_synonyms(metric_payload.get("synonyms"))
        metrics[metric_name] = MetricSemantic(
            name=metric_name,
            description=description,
            expression=expression,
            synonyms=synonyms,
        )
    return metrics


def _normalize_synonyms(value: object) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        synonyms = tuple(str(item).strip() for item in value if str(item).strip())
        return tuple(sorted(set(synonyms)))
    raise SemanticConfigError("Synonyms must be provided as a string or list of strings.")


def _optional_str(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

