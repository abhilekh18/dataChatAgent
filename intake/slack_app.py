"""Slack intake layer for the data chat agent.

This module is responsible for setting up a minimal Slack Bolt application
that listens for user messages and responds with an acknowledgement. The
implementation keeps the surface area small so we can later plug in the full
question understanding and analysis pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


try:
    from slack_bolt import App  # type: ignore[import-not-found]
    from slack_bolt.adapter.socket_mode import SocketModeHandler  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised when dependency missing
    App = None  # type: ignore[assignment]
    SocketModeHandler = None  # type: ignore[assignment]

from engine.analysis import AnalysisEngine, QuestionAnalysis
from output.formatter import format_analysis_response

from .catalog import DatasetCatalog
from .router import QuestionRouter, RouterResult


logger = logging.getLogger(__name__)

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "questions.txt"


@dataclass(frozen=True)
class SlackAppConfig:
    """Configuration values required to initialise the Slack Bolt app."""

    bot_token: str
    app_token: str
    signing_secret: str
    # Optional channel to send a startup message; used during manual verification.
    default_channel: Optional[str] = None
    token_verification_enabled: bool = True
    show_candidate_datasets: bool = False


class SlackDependencyError(ImportError):
    """Raised when optional Slack dependencies are not installed."""


def _ensure_dependencies() -> None:
    if App is None or SocketModeHandler is None:  # pragma: no cover - see above
        raise SlackDependencyError(
            "slack_bolt is not installed. Install the optional Slack dependencies "
            "with `poetry add slack-bolt` or equivalent."
        )


def _build_analysis(
    question: str,
    matches: Sequence[RouterResult],
    analysis_engine: AnalysisEngine,
    *,
    show_candidate_datasets: bool = False,
) -> QuestionAnalysis:
    metadata = [match.dataset for match in matches]
    analysis = analysis_engine.analyze(question, metadata)
    if not matches and not analysis.notes:
        analysis = QuestionAnalysis(
            question=analysis.question,
            datasets=analysis.datasets,
            notes="No datasets matched your question.",
        )
    return analysis


def build_slack_app(
    config: SlackAppConfig,
    *,
    catalog: Optional[DatasetCatalog] = None,
    router: Optional[QuestionRouter] = None,
    analysis_engine: Optional[AnalysisEngine] = None,
) -> App:
    """Create and return a Slack Bolt app instance.

    The handler simply echoes incoming user messages so we can validate the flow
    of events before introducing more complex reasoning layers.
    """
    _ensure_dependencies()

    app = App(
        token=config.bot_token,
        signing_secret=config.signing_secret,
        token_verification_enabled=config.token_verification_enabled,
    )

    dataset_catalog = catalog or DatasetCatalog()
    question_router = router or QuestionRouter(dataset_catalog)
    engine = analysis_engine or AnalysisEngine()

    @app.event("message")
    def handle_message(event: dict, say, logger: logging.Logger) -> None:  # type: ignore[no-untyped-def]
        """Echo user messages while skipping bot messages and edits."""
        if event.get("bot_id") or event.get("subtype") in {"message_changed", "message_deleted"}:
            return

        user = event.get("user", "unknown user")
        text = event.get("text", "")
        logger.info("Received message from %s: %s", user, text)

        if text.strip().lower() == "refresh catalog":
            dataset_catalog.refresh()
            refreshed_count = len(dataset_catalog)
            say(f"Catalog refreshed. Found {refreshed_count} dataset(s).")
            return

        if not text:
            say("Thanks for checking in! I didn't spot a question though—try again?")
            return

        _persist_question(text)
        matches = question_router.rank(text)
        analysis = _build_analysis(
            text,
            matches,
            engine,
            show_candidate_datasets=config.show_candidate_datasets,
        )
        response = format_analysis_response(
            analysis,
            include_candidates=config.show_candidate_datasets,
        )
        say(response)
        chart_paths = getattr(analysis, "chart_paths", tuple())
        if chart_paths:
            channel = event.get("channel")
            thread_ts = event.get("thread_ts")
            if channel:
                for chart_path in chart_paths:
                    path_obj = Path(chart_path)
                    try:
                        data = path_obj.read_bytes()
                    except Exception:  # pragma: no cover - unexpected filesystem errors
                        logger.warning("Failed to read chart bytes from %s", chart_path, exc_info=True)
                        continue
                    try:
                        app.client.files_upload_v2(  # type: ignore[no-untyped-call]
                            channel=channel,
                            thread_ts=thread_ts,
                            file=data,
                            filename=path_obj.name or "chart.png",
                            title=f"Chart for '{text}'",
                            initial_comment="",
                        )
                    except Exception:  # pragma: no cover - Slack upload failures
                        logger.warning("Failed to upload chart %s", chart_path, exc_info=True)

    app._analysis_handler = handle_message  # type: ignore[attr-defined]

    return app


def _persist_question(question: str) -> None:
    try:
        QUESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with QUESTIONS_FILE.open("a", encoding="utf-8") as file:
            file.write(question.strip() + "\n")
    except Exception:  # pragma: no cover - unexpected filesystem errors
        logger.warning("Failed to persist question to %s", QUESTIONS_FILE, exc_info=True)


def start_socket_mode_app(config: SlackAppConfig) -> SocketModeHandler:
    """Initialise and return a SocketModeHandler ready to start."""
    _ensure_dependencies()
    app = build_slack_app(config)
    handler = SocketModeHandler(app, config.app_token)
    return handler


