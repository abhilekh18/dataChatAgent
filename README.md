# Data Chat Agent

Slack-first agent that helps data teams explore CSV datasets with lightweight semantic metadata, PandasAI insights, and auto-generated chart attachments.

## Architecture Overview

The project is organised around four collaborating layers:

```text
Slack (Socket Mode)
        │
        ▼
Intake (`datachatagent/intake/`)
  • `slack_app.py`  – receives messages, routes commands, persists questions
  • `catalog.py`    – discovers CSV datasets and enriches keywords/metadata
  • `router.py`     – ranks datasets against the user question
        │
        ▼
Engine (`datachatagent/engine/`)
  • `analysis.py`        – builds dataset previews, numeric summaries, charts
  • `pandasai_client.py` – wraps PandasAI v3 for SQL generation & insights
        │
        ▼
Output (`datachatagent/output/formatter.py`)
  • Formats analysis results and SQL traces for Slack-friendly text
        │
        ▼
Semantic Layer (`datachatagent/semantic/`)
  • YAML-driven dataset semantics, synonyms, and metric definitions
```

Generated charts are saved under `exports/charts/` so they can be uploaded as Slack attachments.

## Getting Started

### Prerequisites

- Python 3.11+
- [Poetry](https://python-poetry.org/) for dependency management
- Slack workspace with a Socket Mode app (Bot Token, App Token, Signing Secret)
- Optional: OpenAI-compatible credentials for PandasAI insights (`OPENAI_API_KEY` or `PANDASAI_LLM_API_KEY`)

### Install Dependencies

```bash
poetry install
```

### Configure Environment

Create a `.env` file (see `.env.example` if provided) with at least:

- `SLACK_BOT_TOKEN`
- `SLACK_APP_TOKEN`
- `SLACK_SIGNING_SECRET`

Optional variables:

- `SLACK_DEFAULT_CHANNEL` – send a startup message on launch
- `INSIGHTSAI_ENV_FILE` – alternate env file path
- `INSIGHTSAI_DATA_DIR` – override dataset location (defaults to `data/`)
- `INSIGHTSAI_METADATA_DIR` – override semantic YAML directory
- `INSIGHTSAI_LOG_LEVEL` – logging verbosity (`DEBUG`, `INFO`, etc.)
- PandasAI tuning (`PANDASAI_LLM_PROVIDER`, `PANDASAI_LLM_MODEL`, `PANDASAI_VERBOSE`, ...)

Environment variables already present take precedence over `.env` values.

### Prepare Data & Semantics

1. Place CSV datasets in `data/`. Filenames and column names become searchable keywords.
2. Optionally describe datasets in `datachatagent/semantic/semantic.yaml` (display names, synonyms, metric definitions, etc.). The loader gracefully handles empty or missing files.

Run `refresh catalog` in Slack to rescan without restarting the app.

### Run the Slack Bot

```bash
poetry run python -m datachatagent.main
```

Flags:

- `--show-candidates` – include candidate dataset sections in Slack responses
- `--no-show-candidates` (default) – suppress candidate sections

Press `Ctrl+C` to stop the Socket Mode handler safely.

### Example Workflow

1. Upload or sync `users.csv`, `subscriptions.csv`, and `payments.csv` into `data/`.
2. Start the bot and message it in Slack:
   - “What were monthly subscription signups this year?”
3. The bot:
   - Logs the question to `questions.txt`
   - Ranks datasets using keyword overlap and semantic hints
   - Builds previews and statistics via `AnalysisEngine`
   - Runs PandasAI (if enabled) to attempt an insight and chart
   - Replies with summaries, SQL traces, and attaches charts when available

## Development Workflow

- Format and lint: Ruff (configured via `pyproject.toml`)
- Tests: `poetry run pytest -q`
- The codebase adheres to PEP 8 and NumPy-style docstrings
- PandasAI v3 integration expects optional dependencies (`pandas`, `pandasai`, `duckdb`, `openai`)

## Project Layout

```
datachatagent/
├── main.py                 # CLI entry point (env loading, Slack startup)
├── intake/
│   ├── catalog.py          # CSV discovery & keyword enrichment
│   ├── router.py           # Question-to-dataset ranking
│   └── slack_app.py        # Slack Bolt handlers and Socket Mode bootstrap
├── engine/
│   ├── analysis.py         # Dataset summaries and PandasAI orchestration
│   └── pandasai_client.py  # PandasAI Agent wrapper and DuckDB logging
├── output/
│   └── formatter.py        # Assemble Slack-friendly responses
└── semantic/
    ├── loader.py           # YAML parsing & semantic model
    └── semantic.yaml       # Dataset semantics (customise as needed)
```

Supporting assets:

- `data/` – sample CSVs or your project datasets (ignored by git if large)
- `exports/charts/` – auto-generated chart artifacts
- `questions.txt` – running log of user questions
- `tests/` – pytest coverage for intake, engine, and output modules

## Troubleshooting

- “`slack_bolt` is not installed” – add optional Slack dependencies: `poetry add slack-bolt`
- PandasAI setup errors – ensure `pandas`, `pandasai==3.*`, `duckdb`, and OpenAI credentials are available
- Missing datasets – confirm `INSIGHTSAI_DATA_DIR` points to the directory containing your CSV files
- YAML parsing errors – validate `semantic/semantic.yaml` structure; see `SemanticConfigError` messages for details