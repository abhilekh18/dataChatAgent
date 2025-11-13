"""Tests for the Slack intake layer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pytest

try:  # pragma: no cover - executed during test collection only
    import slack_bolt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - executed during test collection only
    slack_bolt = None  # type: ignore[assignment]

from datachatagent.engine.analysis import QuestionAnalysis
from datachatagent.intake.catalog import DatasetMetadata
from datachatagent.intake.router import RouterResult
from datachatagent.intake.slack_app import (  # noqa: E402
    SlackAppConfig,
    _build_analysis,
    build_slack_app,
    start_socket_mode_app,
)
from datachatagent.main import _shutdown_handler  # noqa: E402


slack_required = pytest.mark.skipif(
    slack_bolt is None, reason="slack_bolt dependency not available"
)


@pytest.fixture()
def slack_config() -> SlackAppConfig:
    return SlackAppConfig(
        bot_token="xoxb-test",
        app_token="xapp-test",
        signing_secret="shhh",
        token_verification_enabled=False,
    )


@slack_required
def test_build_slack_app_returns_app(slack_config: SlackAppConfig) -> None:
    app = build_slack_app(slack_config)
    assert app is not None
    assert app.client.token == slack_config.bot_token


@slack_required
def test_start_socket_mode_app_returns_handler(slack_config: SlackAppConfig) -> None:
    handler = start_socket_mode_app(slack_config)
    assert handler is not None
    assert handler.app_token == slack_config.app_token


class StubAnalysisEngine:
    def __init__(self, response: QuestionAnalysis) -> None:
        self.response = response
        self.calls = []

    def analyze(self, question, datasets):  # type: ignore[no-untyped-def]
        self.calls.append((question, datasets))
        return self.response


def test_build_analysis_with_matches() -> None:
    metadata = DatasetMetadata(
        name="revenue",
        path=Path("revenue.csv"),
        columns=("product",),
        row_count=10,
        keywords=("revenue", "product"),
    )
    match = RouterResult(dataset=metadata, score=1.0, matched_keywords=("revenue",))
    engine = StubAnalysisEngine(
        QuestionAnalysis(question="q", datasets=tuple(), notes="analysis")
    )

    analysis = _build_analysis("q", [match], engine)

    assert engine.calls
    question, datasets = engine.calls[0]
    assert question == "q"
    assert datasets[0] is metadata
    assert analysis.notes == "analysis"


def test_build_analysis_without_matches() -> None:
    engine = StubAnalysisEngine(
        QuestionAnalysis(question="q", datasets=tuple(), notes="analysis")
    )
    analysis = _build_analysis("q", [], engine)
    assert engine.calls == [("q", [])]
    assert analysis.notes == "analysis"


@slack_required
def test_handle_message_uploads_charts(
    monkeypatch: pytest.MonkeyPatch, slack_config: SlackAppConfig
) -> None:
    metadata = DatasetMetadata(
        name="users",
        path=Path("users.csv"),
        columns=("user_id",),
        row_count=1,
        keywords=("users",),
    )
    router_result = RouterResult(dataset=metadata, score=1.0, matched_keywords=("users",))

    class StubRouter:
        def rank(self, question: str):  # type: ignore[no-untyped-def]
            return [router_result]

    analysis = QuestionAnalysis(
        question="plot",
        datasets=tuple(),
        chart_paths=("exports/charts/chart.png",),
    )
    chart_file = Path("exports/charts/chart.png")
    chart_file.parent.mkdir(parents=True, exist_ok=True)
    chart_file.write_bytes(b"fake-bytes")

    class StubAnalysisEngine:
        def __init__(self) -> None:
            self.calls: list = []

        def analyze(self, question, datasets):  # type: ignore[no-untyped-def]
            self.calls.append((question, datasets))
            return analysis

    uploads: list = []
    messages: list = []

    class StubCatalog:
        def refresh(self) -> None:
            pass

    app = build_slack_app(
        slack_config,
        catalog=StubCatalog(),  # type: ignore[arg-type]
        router=StubRouter(),  # type: ignore[arg-type]
        analysis_engine=StubAnalysisEngine(),
    )

    monkeypatch.setattr(app.client, "files_upload_v2", lambda **kwargs: uploads.append(kwargs))

    def say(text: str) -> None:
        messages.append(text)

    event = {"text": "plot", "channel": "C123"}
    app._analysis_handler(event, say, app.logger)  # type: ignore[attr-defined]

    assert messages, "Expected a Slack message to be sent"
    assert uploads, "Expected chart upload via files_upload_v2"
    assert uploads[0]["channel"] == "C123"
    assert isinstance(uploads[0]["file"], bytes)
    assert uploads[0]["file"] == b"fake-bytes"


@slack_required
def test_handle_message_persists_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slack_config: SlackAppConfig
) -> None:
    questions_file = tmp_path / "questions.txt"

    class StubRouter:
        def rank(self, question: str):  # type: ignore[no-untyped-def]
            return []

    class StubAnalysisEngine:
        def analyze(self, question, datasets):  # type: ignore[no-untyped-def]
            return QuestionAnalysis(question=question, datasets=tuple(), notes="analysis")

    class StubCatalog:
        def refresh(self) -> None:
            pass

    monkeypatch.setattr("datachatagent.intake.slack_app.QUESTIONS_FILE", questions_file)

    app = build_slack_app(
        slack_config,
        catalog=StubCatalog(),  # type: ignore[arg-type]
        router=StubRouter(),  # type: ignore[arg-type]
        analysis_engine=StubAnalysisEngine(),
    )

    messages: list[str] = []

    def say(text: str) -> None:
        messages.append(text)

    event = {"text": "How many users are active?"}
    app._analysis_handler(event, say, app.logger)  # type: ignore[attr-defined]

    assert messages, "Expected the handler to respond to the user"
    assert questions_file.exists()
    content = questions_file.read_text(encoding="utf-8")
    assert content == "How many users are active?\n"


def test_startup_message_sent_when_default_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    posted = {}

    class DummyApp:
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            self.client = type(
                "ClientStub",
                (),
                {
                    "chat_postMessage": lambda self, **call_kwargs: posted.update(call_kwargs)
                },
            )()
            self.logger = logging.getLogger("dummy")
            self.client.proxy = None

        def event(self, _name):  # type: ignore[no-untyped-def]
            def decorator(func):
                return func

            return decorator

    monkeypatch.setattr("datachatagent.intake.slack_app.App", DummyApp)

    config = SlackAppConfig(
        bot_token="xoxb-test",
        app_token="xapp-test",
        signing_secret="secret",
        default_channel="C123",
        token_verification_enabled=False,
    )

    start_socket_mode_app(config)

    assert posted["channel"] == "C123"
    assert "ready" in posted["text"].lower()

def test_shutdown_handler_invokes_stop_join_close() -> None:
    calls: List[Tuple[str, Tuple]] = []

    class StubHandler:
        def stop(self) -> None:
            calls.append(("stop", ()))

        def join(self, timeout: int | None = None) -> None:
            calls.append(("join", (timeout,)))

        def close(self) -> None:
            calls.append(("close", ()))

    handler = StubHandler()

    _shutdown_handler(handler)

    assert ("stop", ()) in calls
    assert ("join", (5,)) in calls or ("join", (None,)) in calls
    assert ("close", ()) in calls

