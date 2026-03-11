"""Tests for artifact writer: save, cache, and manifest correctness."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from connectomics_mcp.artifacts.writer import save_artifact
from connectomics_mcp.output_contracts.schemas import ArtifactManifest


@pytest.fixture
def artifact_dir(tmp_path: Path):
    """Use a temporary directory for artifact output."""
    with patch.dict(os.environ, {"CONNECTOMICS_MCP_ARTIFACT_DIR": str(tmp_path)}):
        yield tmp_path


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "partner_id": [100, 200, 300],
        "direction": ["upstream", "upstream", "downstream"],
        "n_synapses": [50, 30, 20],
        "weight_normalized": [0.5, 0.3, 0.2],
    })


class TestSaveArtifact:
    def test_writes_parquet(self, artifact_dir: Path):
        df = _sample_df()
        manifest = save_artifact(
            df, tool="connectivity", dataset="minnie65",
            neuron_id=123, materialization_version=943,
        )

        assert isinstance(manifest, ArtifactManifest)
        assert manifest.n_rows == 3
        assert manifest.cache_hit is False
        assert manifest.artifact_format == "parquet"

        # File exists and is readable
        path = Path(manifest.artifact_path)
        assert path.exists()
        loaded = pd.read_parquet(path)
        assert len(loaded) == 3
        assert list(loaded.columns) == list(df.columns)

    def test_manifest_columns(self, artifact_dir: Path):
        df = _sample_df()
        manifest = save_artifact(
            df, tool="connectivity", dataset="minnie65",
            neuron_id=123, materialization_version=943,
        )
        assert manifest.columns == ["partner_id", "direction", "n_synapses", "weight_normalized"]
        assert "partner_id" in manifest.schema_description

    def test_filename_convention(self, artifact_dir: Path):
        df = _sample_df()
        manifest = save_artifact(
            df, tool="connectivity", dataset="minnie65",
            neuron_id=720575940621039145, materialization_version=943,
        )
        path = Path(manifest.artifact_path)
        assert path.name.startswith("minnie65_connectivity_720575940621039145_v943_")
        assert path.suffix == ".parquet"

    def test_cache_hit(self, artifact_dir: Path):
        df = _sample_df()
        m1 = save_artifact(
            df, tool="connectivity", dataset="minnie65",
            neuron_id=123, materialization_version=943,
        )
        assert m1.cache_hit is False

        m2 = save_artifact(
            df, tool="connectivity", dataset="minnie65",
            neuron_id=123, materialization_version=943,
        )
        assert m2.cache_hit is True
        assert m2.artifact_path == m1.artifact_path
        assert m2.n_rows == 3

    def test_cache_expiry(self, artifact_dir: Path):
        df = _sample_df()
        m1 = save_artifact(
            df, tool="connectivity", dataset="minnie65",
            neuron_id=123, materialization_version=943,
        )

        # Make the cached file appear old
        old_path = Path(m1.artifact_path)
        old_time = time.time() - 7200  # 2 hours ago
        os.utime(old_path, (old_time, old_time))

        m2 = save_artifact(
            df, tool="connectivity", dataset="minnie65",
            neuron_id=123, materialization_version=943,
        )
        assert m2.cache_hit is False
        # A new file was written (may have same or different name depending on timing)
        assert Path(m2.artifact_path).exists()
        assert m2.n_rows == 3

    def test_no_neuron_id(self, artifact_dir: Path):
        df = pd.DataFrame({"region": ["A", "B"], "count": [10, 20]})
        manifest = save_artifact(
            df, tool="region_connectivity", dataset="hemibrain",
        )
        assert manifest.n_rows == 2
        path = Path(manifest.artifact_path)
        assert "hemibrain_region_connectivity_vNone_" in path.name
