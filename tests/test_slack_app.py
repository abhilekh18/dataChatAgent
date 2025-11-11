"""Tests for the Slack intake layer."""

from __future__ import annotations

from pathlib import Path

import pytest

slack_bolt = pytest.importorskip("slack_bolt", reason="slack_bolt dependency not available")

from engine.analysis import QuestionAnalysis
from intake.catalog import DatasetMetadata
from intake.router import RouterResult
from intake.slack_app import SlackAppConfig, build_slack_app, start_socket_mode_app, _build_analysis  # noqa: E402


@pytest.fixture()
def slack_config() -> SlackAppConfig:
    return SlackAppConfig(
        bot_token="xoxb-test",
        app_token="xapp-test",
        signing_secret="shhh",
        token_verification_enabled=False,
    )


def test_build_slack_app_returns_app(slack_config: SlackAppConfig) -> None:
    app = build_slack_app(slack_config)
    assert app is not None
    assert app.client.token == slack_config.bot_token


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


def test_handle_message_uploads_charts(monkeypatch: pytest.MonkeyPatch, slack_config: SlackAppConfig) -> None:
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

    monkeypatch.setattr("intake.slack_app.QUESTIONS_FILE", questions_file)

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


