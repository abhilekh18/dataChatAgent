"""PandasAI agent integration for generating insights."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pandasai.agent import Agent
    from pandasai.config import Config
    from pandasai.llm.base import LLM
    from pandasai.llm.fake import FakeLLM
except ImportError:  # pragma: no cover
    Agent = None  # type: ignore[assignment]
    Config = None  # type: ignore[assignment]
    LLM = None  # type: ignore[assignment]
    FakeLLM = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from openai import OpenAI as OpenAIClient
except ImportError:  # pragma: no cover
    OpenAIClient = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pandasai.core.prompts.base import BasePrompt
except ImportError:  # pragma: no cover
    BasePrompt = Any  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pandasai import DataFrame as PandasAIDataFrame
    from pandasai.data_loader import duck_db_connection_manager as duckdb_manager_module
    from pandasai.helpers.sql_sanitizer import sanitize_sql_table_name_lowercase
    from pandasai.query_builders.sql_parser import SQLParser
except ImportError:  # pragma: no cover
    PandasAIDataFrame = None  # type: ignore[assignment]
    duckdb_manager_module = None  # type: ignore[assignment]
    sanitize_sql_table_name_lowercase = None  # type: ignore[assignment]
    SQLParser = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

def _rewrite_strftime_argument_order(query: str) -> str:
    """Rewrite strftime calls with swapped arguments to DuckDB order."""

    def _split_arguments(arguments: str) -> tuple[str, str] | None:
        depth = 0
        in_quote: str | None = None
        for index, char in enumerate(arguments):
            if in_quote:
                if char == in_quote and (index == 0 or arguments[index - 1] != "\\"):
                    in_quote = None
                continue
            if char in "\"'":
                in_quote = char
                continue
            if char == "(":
                depth += 1
                continue
            if char == ")":
                depth = max(depth - 1, 0)
                continue
            if char == "," and depth == 0:
                first = arguments[:index]
                second = arguments[index + 1 :]
                return first, second
        return None

    pattern = "strftime"
    lower_query = query.lower()
    cursor = 0
    rewritten: list[str] = []

    while True:
        position = lower_query.find(pattern, cursor)
        if position == -1:
            rewritten.append(query[cursor:])
            break

        rewritten.append(query[cursor:position])
        current = position + len(pattern)
        length = len(query)

        while current < length and query[current].isspace():
            current += 1

        if current >= length or query[current] != "(":
            rewritten.append(query[position:current])
            cursor = current
            continue

        start_arguments = current + 1
        depth = 1
        index = start_arguments
        in_quote: str | None = None

        while index < length and depth > 0:
            char = query[index]
            if in_quote:
                if char == in_quote and query[index - 1] != "\\":
                    in_quote = None
            else:
                if char in "\"'":
                    in_quote = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
            index += 1

        if depth != 0:
            rewritten.append(query[position:index])
            cursor = index
            continue

        end_arguments = index - 1
        arguments = query[start_arguments:end_arguments]
        split = _split_arguments(arguments)

        if not split:
            rewritten.append(query[position:index])
            cursor = index
            continue

        raw_first, raw_second = split
        first = raw_first.strip()
        second = raw_second.strip()

        if not first:
            rewritten.append(query[position:index])
            cursor = index
            continue

        if first.startswith("'") and first.endswith("'") or first.startswith('"') and first.endswith('"'):
            new_call = f"strftime({second}, {first})"
            rewritten.append(new_call)
        else:
            rewritten.append(query[position:index])

        cursor = index

    return "".join(rewritten)


def _rewrite_year_argument_cast(query: str) -> str:
    """Ensure YEAR() operates on temporal data by casting string inputs to DATE."""

    pattern = "year"
    lower_query = query.lower()
    cursor = 0
    rewritten: list[str] = []

    while True:
        position = lower_query.find(pattern, cursor)
        if position == -1:
            rewritten.append(query[cursor:])
            break

        rewritten.append(query[cursor:position])
        current = position + len(pattern)
        length = len(query)

        while current < length and query[current].isspace():
            current += 1

        if current >= length or query[current] != "(":
            rewritten.append(query[position:current])
            cursor = current
            continue

        start_argument = current + 1
        depth = 1
        index = start_argument
        in_quote: str | None = None

        while index < length and depth > 0:
            char = query[index]
            if in_quote:
                if char == in_quote and query[index - 1] != "\\":
                    in_quote = None
            else:
                if char in "\"'":
                    in_quote = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
            index += 1

        if depth != 0:
            rewritten.append(query[position:index])
            cursor = index
            continue

        end_argument = index - 1
        argument = query[start_argument:end_argument].strip()
        argument_lower = argument.lower()
        normalized_argument = " ".join(argument_lower.split())

        if not argument:
            rewritten.append(query[position:index])
            cursor = index
            continue

        casts_to_date = normalized_argument.startswith("cast(") and " as date" in normalized_argument
        casts_to_timestamp = normalized_argument.startswith("cast(") and " as timestamp" in normalized_argument
        has_double_colon = "::date" in normalized_argument or "::timestamp" in normalized_argument
        already_temporal = (
            argument_lower.startswith("date(")
            or argument_lower.startswith("timestamp(")
            or argument_lower.startswith("datetime(")
        )

        if casts_to_date or has_double_colon or already_temporal:
            rewritten.append(query[position:index])
            cursor = index
            continue

        if casts_to_timestamp or argument_lower.startswith("cast("):
            new_call = f"year(CAST({argument} AS DATE))"
            rewritten.append(new_call)
            cursor = index
            continue

        new_call = f"year(CAST({argument} AS DATE))"
        rewritten.append(new_call)
        cursor = index

    return "".join(rewritten)


class PandasAISetupError(RuntimeError):
    """Raised when the PandasAI agent cannot be initialised."""


class PandasAIExecutionError(RuntimeError):
    """Raised when the PandasAI agent fails during execution."""


@dataclass(frozen=True)
class SQLQueryLogEntry:
    """Captured DuckDB SQL query."""

    query: str
    params: Tuple[Any, ...] | None = None
    row_count: Optional[int] = None


def _parse_bool(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_float(value: Optional[str], *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float value '%s'; falling back to %s.", value, default)
        return default


def _parse_int(value: Optional[str], *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer value '%s'; falling back to %s.", value, default)
        return default


@dataclass(frozen=True)
class PandasAISettings:
    """Configuration derived from environment variables."""

    provider: str
    model: str
    temperature: float
    max_retries: int
    verbose: bool
    save_logs: bool
    memory_size: int
    api_key: Optional[str]
    base_url: Optional[str]

    @classmethod
    def from_environment(cls, *, default_model: str) -> "PandasAISettings":
        provider = os.getenv("PANDASAI_LLM_PROVIDER", "openai").strip().lower()
        model = os.getenv("PANDASAI_LLM_MODEL", default_model).strip()
        temperature = _parse_float(os.getenv("PANDASAI_LLM_TEMPERATURE"), default=0.0)
        max_retries = _parse_int(os.getenv("PANDASAI_MAX_RETRIES"), default=2)
        verbose = _parse_bool(os.getenv("PANDASAI_VERBOSE"), default=False)
        save_logs = _parse_bool(os.getenv("PANDASAI_SAVE_LOGS"), default=False)
        memory_size = _parse_int(os.getenv("PANDASAI_MEMORY_SIZE"), default=10)
        api_key = os.getenv("PANDASAI_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("PANDASAI_LLM_BASE_URL")
        return cls(
            provider=provider,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            verbose=verbose,
            save_logs=save_logs,
            memory_size=memory_size,
            api_key=api_key,
            base_url=base_url,
        )


class OpenAIChatLLM(LLM):
    """Minimal OpenAI chat completion adapter for PandasAI."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float,
        base_url: Optional[str] = None,
    ) -> None:
        if OpenAIClient is None:  # pragma: no cover - optional dependency
            raise PandasAISetupError(
                "openai python package is required for PANDASAI_LLM_PROVIDER=openai."
            )
        if not api_key:
            raise PandasAISetupError(
                "An API key is required for PANDASAI_LLM_PROVIDER=openai. "
                "Set OPENAI_API_KEY or PANDASAI_LLM_API_KEY."
            )

        super().__init__(api_key=api_key)
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = OpenAIClient(**client_kwargs)
        self._model = model
        self._temperature = temperature

    @property
    def type(self) -> str:
        return "openai"

    def call(self, instruction: BasePrompt, context=None) -> str:  # type: ignore[override]
        prompt = instruction.to_string()
        memory = getattr(context, "memory", None)
        messages = []
        if memory:
            system_prompt = self.get_system_prompt(memory)
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )
        except Exception as exc:  # pragma: no cover - upstream SDK errors
            raise PandasAIExecutionError(f"OpenAI request failed: {exc}") from exc

        try:
            choice = response.choices[0]
            content = getattr(choice.message, "content", None)
        except (AttributeError, IndexError) as exc:  # pragma: no cover
            raise PandasAIExecutionError("OpenAI response did not include any content.") from exc

        if not content:
            raise PandasAIExecutionError("OpenAI response was empty.")
        return content


class PandasAIClient:
    """Thin wrapper around the PandasAI Agent API."""

    def __init__(self, settings: PandasAISettings) -> None:
        if Agent is None or Config is None or LLM is None:
            raise PandasAISetupError("pandasai is not installed.")
        if pd is None:
            raise PandasAISetupError("pandas is required for PandasAI insights.")
        if duckdb is None:
            raise PandasAISetupError("duckdb is required for SQL inspection support.")
        if (
            PandasAIDataFrame is None
            or duckdb_manager_module is None
            or sanitize_sql_table_name_lowercase is None
            or SQLParser is None
        ):
            raise PandasAISetupError(
                "PandasAI duckdb integration is unavailable. Verify pandasai v3 is installed."
            )

        self.settings = settings
        self._llm = self._build_llm(settings)
        self._config = Config(
            llm=self._llm,
            save_logs=settings.save_logs,
            verbose=settings.verbose,
            max_retries=settings.max_retries,
        )
        self._last_sql_queries: Tuple[SQLQueryLogEntry, ...] = tuple()

    @classmethod
    def from_environment(cls, *, default_model: str) -> "PandasAIClient":
        settings = PandasAISettings.from_environment(default_model=default_model)
        return cls(settings)

    @property
    def last_sql_queries(self) -> Tuple[SQLQueryLogEntry, ...]:
        """Return the SQL queries executed during the most recent run."""
        return self._last_sql_queries

    def _build_llm(self, settings: PandasAISettings) -> LLM:  # type: ignore[override]
        if settings.provider == "fake":
            if FakeLLM is None:
                raise PandasAISetupError("pandasai.llm.fake.FakeLLM is unavailable.")
            return FakeLLM()  # type: ignore[return-value]
        if settings.provider == "openai":
            return OpenAIChatLLM(
                api_key=settings.api_key or "",
                model=settings.model,
                temperature=settings.temperature,
                base_url=settings.base_url,
            )
        raise PandasAISetupError(
            f"Unsupported PANDASAI_LLM_PROVIDER '{settings.provider}'. "
            "Use 'openai' or 'fake'."
        )

    def _table_name_for_metadata(self, metadata: "DatasetMetadataLike") -> str:
        raw_name = getattr(metadata, "name", "") or "dataset"
        sanitized = raw_name.lower()
        if sanitize_sql_table_name_lowercase is not None:
            try:
                sanitized = sanitize_sql_table_name_lowercase(raw_name)
            except Exception:
                fallback = f"dataset_{abs(hash(raw_name))}"
                try:
                    sanitized = sanitize_sql_table_name_lowercase(fallback)
                except Exception:
                    sanitized = fallback
        return sanitized

    def _prepare_agent_inputs(
        self,
        datasets: Sequence[Tuple["DatasetMetadataLike", pd.DataFrame]],
    ) -> List[PandasAIDataFrame]:
        prepared: List[PandasAIDataFrame] = []
        for metadata, dataframe in datasets:
            prepared.append(self._convert_to_agent_dataframe(metadata, dataframe))
        return prepared

    def _convert_to_agent_dataframe(
        self, metadata: "DatasetMetadataLike", dataframe: pd.DataFrame
    ) -> PandasAIDataFrame:
        table_name = self._table_name_for_metadata(metadata)
        agent_df = PandasAIDataFrame(dataframe, _table_name=table_name)

        semantic = getattr(metadata, "semantic", None)
        description = getattr(semantic, "description", None)
        if description:
            agent_df.schema.description = description

        return agent_df

    @contextmanager
    def _capture_sql_queries(self, log: List[SQLQueryLogEntry]):
        if duckdb_manager_module is None:
            raise PandasAISetupError("Unable to patch DuckDB manager; pandasai not installed.")
        original_manager = duckdb_manager_module.DuckDBConnectionManager
        connection = duckdb.connect()
        registered_tables: set[str] = set()

        class LoggingDuckDBConnectionManager:
            def __init__(self) -> None:
                self.connection = connection

            def register(self, name: str, df: Any) -> None:
                try:
                    connection.unregister(name)
                except Exception:
                    pass
                connection.register(name, df)
                registered_tables.add(name)

            def unregister(self, name: str) -> None:
                if name in registered_tables:
                    try:
                        connection.unregister(name)
                    except Exception:
                        pass
                    registered_tables.discard(name)

            def sql(self, query: str, params: Optional[list] = None):
                fixed_query = _rewrite_strftime_argument_order(query)
                fixed_query = _rewrite_year_argument_cast(fixed_query)
                normalized = fixed_query
                if SQLParser is not None:
                    normalized = SQLParser.transpile_sql_dialect(
                        fixed_query, to_dialect="duckdb"
                    )
                normalized = _rewrite_strftime_argument_order(normalized)
                normalized = _rewrite_year_argument_cast(normalized)
                params_tuple = tuple(params) if params is not None else None
                relation = connection.sql(normalized, params=params)
                row_count: Optional[int] = None
                try:
                    aggregate_relation = relation.aggregate("count(*)")
                    row_count_result = aggregate_relation.fetchone()
                    if row_count_result is not None:
                        row_count = int(row_count_result[0])
                except Exception:
                    row_count = None
                log.append(
                    SQLQueryLogEntry(
                        query=str(normalized),
                        params=params_tuple,
                        row_count=row_count,
                    )
                )
                return relation

            def close(self) -> None:
                # Connection lifecycle handled by the enclosing context.
                pass

            def __del__(self) -> None:  # pragma: no cover - defensive cleanup
                self.close()

        duckdb_manager_module.DuckDBConnectionManager = LoggingDuckDBConnectionManager
        try:
            yield
        finally:
            duckdb_manager_module.DuckDBConnectionManager = original_manager
            for table in list(registered_tables):
                try:
                    connection.unregister(table)
                except Exception:
                    pass
            connection.close()

    def run(
        self,
        question: str,
        datasets: Sequence[Tuple["DatasetMetadataLike", pd.DataFrame]],
    ) -> str:
        """Execute a question against the PandasAI Agent."""
        if not datasets:
            raise PandasAIExecutionError("No datasets available for PandasAI.")

        agent_inputs = self._prepare_agent_inputs(datasets)
        description = _build_agent_description(datasets)
        query_log: List[SQLQueryLogEntry] = []

        try:
            with self._capture_sql_queries(query_log):
                agent = Agent(
                    agent_inputs,
                    config=self._config,
                    memory_size=self.settings.memory_size,
                    description=description,
                )
                result = agent.chat(question)
        except ValueError as exc:
            raise PandasAIExecutionError(str(exc)) from exc
        except PandasAIExecutionError:
            raise
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.error("PandasAI agent failed: %s", exc, exc_info=True)
            raise PandasAIExecutionError(
                "PandasAI agent failed to generate an answer."
            ) from exc

        self._last_sql_queries = tuple(query_log)

        if isinstance(result, str):
            return result.strip()
        return str(result)


def _build_agent_description(
    datasets: Sequence[Tuple["DatasetMetadataLike", pd.DataFrame]]
) -> str:
    parts: list[str] = []
    for metadata, df in datasets:
        columns = ", ".join(map(str, df.columns))
        parts.append(f"{metadata.display_name} ({columns})")
    return " | ".join(parts)


class DatasetMetadataLike:  # pragma: no cover - typing convenience
    name: str
    display_name: str

