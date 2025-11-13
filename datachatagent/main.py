"""CLI entrypoint for the data chat agent Slack bot."""

from __future__ import annotations

import logging
import os
import argparse
from pathlib import Path
from typing import Optional, Sequence

from datachatagent.intake.slack_app import (
    SlackAppConfig,
    SlackDependencyError,
    start_socket_mode_app,
)

LOGGER = logging.getLogger(__name__)


def _load_env_file(path: Optional[str] = None) -> None:
    """Load environment variables from a `.env` file if present.

    Values already present in the environment take precedence.
    """
    candidate = path or os.getenv("INSIGHTSAI_ENV_FILE", ".env")
    env_path = Path(candidate).expanduser()
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:  # pragma: no cover - unexpected IO errors
        LOGGER.warning("Failed to read environment file %s: %s", env_path, exc)


def _get_env_var(name: str, *, optional: bool = False) -> Optional[str]:
    value = os.getenv(name)
    if value:
        return value
    if optional:
        return None
    raise RuntimeError(
        f"Environment variable `{name}` is required but was not provided. "
        "Set it in your environment or a local `.env` file."
    )


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("INSIGHTSAI_LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def build_config_from_env(*, show_candidate_datasets: bool = False) -> SlackAppConfig:
    """Create a SlackAppConfig from environment variables."""
    bot_token = _get_env_var("SLACK_BOT_TOKEN")
    app_token = _get_env_var("SLACK_APP_TOKEN")
    signing_secret = _get_env_var("SLACK_SIGNING_SECRET")
    default_channel = _get_env_var("SLACK_DEFAULT_CHANNEL", optional=True)
    return SlackAppConfig(
        bot_token=bot_token,
        app_token=app_token,
        signing_secret=signing_secret,
        default_channel=default_channel,
        show_candidate_datasets=show_candidate_datasets,
    )


def _shutdown_handler(handler) -> None:  # type: ignore[no-untyped-def]
    """Attempt to stop, join, and close the SocketModeHandler safely."""
    if handler is None:
        return

    for method_name in ("stop",):
        method = getattr(handler, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:  # pragma: no cover - defensive cleanup
                LOGGER.debug(
                    "SocketModeHandler %s() raised an exception during shutdown",
                    method_name,
                    exc_info=True,
                )

    join_method = getattr(handler, "join", None)
    if callable(join_method):
        try:
            join_method(timeout=5)
        except TypeError:
            try:
                join_method()  # type: ignore[misc]
            except Exception:  # pragma: no cover - defensive cleanup
                LOGGER.debug(
                    "SocketModeHandler join() raised an exception during shutdown",
                    exc_info=True,
                )
        except Exception:  # pragma: no cover - defensive cleanup
            LOGGER.debug(
                "SocketModeHandler join(timeout=5) raised an exception during shutdown",
                exc_info=True,
            )

    for method_name in ("close",):
        method = getattr(handler, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:  # pragma: no cover - defensive cleanup
                LOGGER.debug(
                    "SocketModeHandler %s() raised an exception during shutdown",
                    method_name,
                    exc_info=True,
                )


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the data chat agent Slack bot.")
    parser.add_argument(
        "--show-candidates",
        dest="show_candidates",
        action="store_true",
        default=False,
        help="Include candidate dataset sections in Slack responses.",
    )
    parser.add_argument(
        "--no-show-candidates",
        dest="show_candidates",
        action="store_false",
        help="Hide candidate dataset sections from Slack responses (default).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Start the Slack Socket Mode handler."""
    args = _parse_args(argv)
    _configure_logging()
    _load_env_file()
    try:
        config = build_config_from_env(show_candidate_datasets=args.show_candidates)
        handler = start_socket_mode_app(config)
    except SlackDependencyError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc

    LOGGER.info("Starting Slack Socket Mode handler. Press Ctrl+C to exit.")
    try:
        handler.start()
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested by user.")
    finally:
        _shutdown_handler(handler)


if __name__ == "__main__":
    main()


