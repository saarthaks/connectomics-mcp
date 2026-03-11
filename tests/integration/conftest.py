"""Shared fixtures for live-API integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True)
def _load_env() -> None:
    """Load .env file so credentials are available."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@pytest.fixture()
def artifact_dir(tmp_path: Path) -> Path:
    """Point artifact output to a temp dir and return it."""
    os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
    return tmp_path


# ── Session-scoped client fixtures for schema audit ────────────────


@pytest.fixture(scope="session")
def _session_env() -> None:
    """Load .env once per session."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


@pytest.fixture(scope="session")
def cave_client_minnie65(_session_env):
    """Session-scoped CAVEclient for minnie65_public."""
    try:
        import caveclient

        c = caveclient.CAVEclient("minnie65_public", write_server_cache=False)
        token = os.environ.get("CAVE_CLIENT_TOKEN")
        if token:
            c.auth.token = token
        return c
    except Exception as e:
        pytest.skip(f"Cannot create minnie65 CAVEclient: {e}")


@pytest.fixture(scope="session")
def cave_client_flywire(_session_env):
    """Session-scoped CAVEclient for flywire_fafb_public."""
    try:
        import caveclient

        c = caveclient.CAVEclient("flywire_fafb_public", write_server_cache=False)
        token = os.environ.get("CAVE_CLIENT_TOKEN")
        if token:
            c.auth.token = token
        return c
    except Exception as e:
        pytest.skip(f"Cannot create flywire CAVEclient: {e}")



@pytest.fixture(scope="session")
def neuprint_client(_session_env):
    """Session-scoped neuPrint client for hemibrain:v1.2.1."""
    try:
        from neuprint import Client

        token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS", "")
        return Client("neuprint.janelia.org", dataset="hemibrain:v1.2.1", token=token)
    except Exception as e:
        pytest.skip(f"Cannot create neuPrint client: {e}")
