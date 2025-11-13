"""Tests for the main entrypoint utilities."""

from __future__ import annotations

import os

import pytest

import datachatagent.main as main_module
from datachatagent.main import _load_env_file, build_config_from_env


def test_build_config_from_env_reads_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "bot")
    monkeypatch.setenv("SLACK_APP_TOKEN", "app")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "secret")
    monkeypatch.delenv("SLACK_DEFAULT_CHANNEL", raising=False)

    config = build_config_from_env()

    assert config.bot_token == "bot"
    assert config.app_token == "app"
    assert config.signing_secret == "secret"
    assert config.default_channel is None
    assert config.show_candidate_datasets is False


def test_build_config_from_env_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)

    with pytest.raises(RuntimeError):
        build_config_from_env()


def test_load_env_file_sets_missing_values(tmp_path: pytest.PathLike[str], monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "SLACK_BOT_TOKEN=fromfile\n"
        "SLACK_APP_TOKEN=fromfile\n"
        "SLACK_SIGNING_SECRET=fromfile\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_SIGNING_SECRET", raising=False)

    _load_env_file(".env")

    assert os.getenv("SLACK_BOT_TOKEN") == "fromfile"
    assert os.getenv("SLACK_APP_TOKEN") == "fromfile"
    assert os.getenv("SLACK_SIGNING_SECRET") == "fromfile"


def test_main_handles_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "bot")
    monkeypatch.setenv("SLACK_APP_TOKEN", "app")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "secret")

    class DummyHandler:
        def __init__(self) -> None:
            self.closed = False

        def start(self) -> None:
            raise KeyboardInterrupt

        def close(self) -> None:
            self.closed = True

    dummy_handler = DummyHandler()
    captured = {}

    def fake_start_socket_mode_app(config):  # type: ignore[no-untyped-def]
        captured["config"] = config
        return dummy_handler

    monkeypatch.setattr("datachatagent.main.start_socket_mode_app", fake_start_socket_mode_app)

    caplog.set_level("INFO")
    main_module.main([])

    assert dummy_handler.closed is True
    assert captured["config"].show_candidate_datasets is False
    assert any("Shutdown requested by user." in record.getMessage() for record in caplog.records)


def test_main_respects_no_show_candidates_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "bot")
    monkeypatch.setenv("SLACK_APP_TOKEN", "app")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "secret")

    captured = {}

    class DummyHandler:
        def __init__(self) -> None:
            self.closed = False

        def start(self) -> None:
            raise KeyboardInterrupt

        def close(self) -> None:
            self.closed = True

    dummy_handler = DummyHandler()

    def fake_start_socket_mode_app(config):  # type: ignore[no-untyped-def]
        captured["config"] = config
        return dummy_handler

    monkeypatch.setattr("datachatagent.main.start_socket_mode_app", fake_start_socket_mode_app)

    main_module.main(["--no-show-candidates"])

    assert captured["config"].show_candidate_datasets is False
    assert dummy_handler.closed is True


def test_main_respects_show_candidates_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLACK_BOT_TOKEN", "bot")
    monkeypatch.setenv("SLACK_APP_TOKEN", "app")
    monkeypatch.setenv("SLACK_SIGNING_SECRET", "secret")

    captured = {}

    class DummyHandler:
        def __init__(self) -> None:
            self.closed = False

        def start(self) -> None:
            raise KeyboardInterrupt

        def close(self) -> None:
            self.closed = True

    dummy_handler = DummyHandler()

    def fake_start_socket_mode_app(config):  # type: ignore[no-untyped-def]
        captured["config"] = config
        return dummy_handler

    monkeypatch.setattr("datachatagent.main.start_socket_mode_app", fake_start_socket_mode_app)

    main_module.main(["--show-candidates"])

    assert captured["config"].show_candidate_datasets is True
    assert dummy_handler.closed is True


