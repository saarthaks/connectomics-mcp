"""Shared fixtures for live-API integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run live-API integration tests (requires credentials).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--integration"):
        return
    skip_integration = pytest.mark.skip(reason="need --integration flag to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(autouse=True)
def _load_env() -> None:
    """Load .env file so credentials are available."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    """Point artifact output to a temp dir and return it."""
    os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
    return tmp_path
