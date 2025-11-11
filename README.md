# Insights AI Capstone Project - Futureproof DS - Andres Vourakis

## Slack Bot Quickstart

1. Ensure the `.env` file contains valid values for `SLACK_BOT_TOKEN`,
   `SLACK_APP_TOKEN`, and `SLACK_SIGNING_SECRET`. Use `.env.example` as a
   template.
2. Install dependencies with Poetry:
   ```bash
   poetry install
   ```
3. Start the bot locally:
   ```bash
   poetry run python -m insightsai.main
   ```
4. Drop CSV files into `insightsai/csv_files/` (or point `INSIGHTSAI_DATA_DIR`
   to another directory). Filenames such as `revenue_by_product.csv` will be
   tokenized so the intake router can match keywords like “revenue” or “product”.
5. (Optional) Describe each dataset in `insightsai/semantic_layer/*.yaml`. You
   can add field descriptions or synonyms; set `INSIGHTSAI_METADATA_DIR` if you
   keep YAML files elsewhere.
6. Send a message in a Slack channel or DM where the app is installed. The bot
   logs incoming messages, applies the intake parser/router, and replies with
   the datasets it believes are most relevant.
7. Press `Ctrl+C` in the terminal to stop the Socket Mode handler; the app now
   shuts down cleanly.
8. Candidate dataset sections are hidden by default. Pass `--show-candidates`
   (e.g. `poetry run python -m insightsai.main --show-candidates`) to include them.