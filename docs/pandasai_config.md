# PandasAI Configuration

The integration now targets the PandasAI v3 `Agent` API and is driven entirely through environment variables so deployments can swap LLMs without touching the code.

## Provider & Model
- `PANDASAI_LLM_PROVIDER`: defaults to `openai`. Set to `fake` to run without an external LLM (useful for tests).
- `PANDASAI_LLM_MODEL`: overrides the provider’s model. When unset the engine default (`gpt-4o-mini`) is used.
- `PANDASAI_LLM_TEMPERATURE`: optional float (default `0.0`).
- `PANDASAI_MAX_RETRIES`: optional integer (default `2`).
- `PANDASAI_MEMORY_SIZE`: optional integer (default `10`) controlling conversation history depth.
- `PANDASAI_VERBOSE` / `PANDASAI_SAVE_LOGS`: enable additional agent logging when troubleshooting.

## Credentials
- `PANDASAI_LLM_API_KEY` is preferred; we fall back to `OPENAI_API_KEY` for compatibility.
- `PANDASAI_LLM_BASE_URL` allows targeting Azure/OpenAI-compatible endpoints.

## Behaviour
- A shared `PandasAIClient` wraps `pandasai.agent.Agent` and reuses configuration across questions.
- DataFrames are trimmed earlier in the pipeline (20,000-row cap) before reaching the agent to keep prompts efficient.
- For `openai`, we use a thin adapter around the official SDK. Additional providers can be added by extending the client.
- Setting `PANDASAI_LLM_PROVIDER=fake` swaps in PandasAI’s deterministic `FakeLLM`, keeping CI fast and offline.
- All PandasAI executions run against a managed DuckDB in-memory database. Each dataset is registered as a table so the generated SQL can be replayed or inspected after a run. The executed statements are exposed via `QuestionAnalysis.insight_sql_queries` and surfaced in Slack responses.

## Failure Handling
- Missing dependencies or credentials surface actionable messages (for example, “Set OPENAI_API_KEY or PANDASAI_LLM_API_KEY to enable PandasAI insights.”).
- Runtime errors are logged with stack traces, and the user receives the existing fallback response: “I couldn't analyse the data automatically. Please try narrowing your question.”

These defaults keep the agent integration reproducible while leaving room for custom providers or on-prem deployments.

