"""Tests for artifact writer: save, cache, and manifest correctness."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from connectomics_mcp.artifacts.writer import (
    load_cached_artifact,
    save_artifact,
    _CACHE_MAX_AGE_SECONDS,
)
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
        """Labeled segments (nid-, xk-) prevent cache key collisions."""
        df = _sample_df()
        manifest = save_artifact(
            df, tool="connectivity", dataset="minnie65",
            neuron_id=720575940621039145, materialization_version=943,
        )
        path = Path(manifest.artifact_path)
        assert path.name.startswith(
            "minnie65_connectivity_nid-720575940621039145_xk-None_v943_"
        )
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

    def test_versioned_cache_never_expires(self, artifact_dir: Path):
        """Versioned queries (mat_version set) never expire."""
        df = _sample_df()
        m1 = save_artifact(
            df, tool="connectivity", dataset="minnie65",
            neuron_id=123, materialization_version=943,
        )

        # Make the cached file appear very old (30 days)
        old_path = Path(m1.artifact_path)
        old_time = time.time() - 30 * 86400
        os.utime(old_path, (old_time, old_time))

        m2 = save_artifact(
            df, tool="connectivity", dataset="minnie65",
            neuron_id=123, materialization_version=943,
        )
        assert m2.cache_hit is True
        assert m2.artifact_path == m1.artifact_path

    def test_unversioned_cache_expires(self, artifact_dir: Path):
        """Unversioned queries (mat_version=None) expire after 7 days."""
        df = _sample_df()
        m1 = save_artifact(
            df, tool="connectivity", dataset="hemibrain",
            neuron_id=123, materialization_version=None,
        )

        # Make the cached file appear older than 7 days
        old_path = Path(m1.artifact_path)
        old_time = time.time() - _CACHE_MAX_AGE_SECONDS - 3600
        os.utime(old_path, (old_time, old_time))

        m2 = save_artifact(
            df, tool="connectivity", dataset="hemibrain",
            neuron_id=123, materialization_version=None,
        )
        assert m2.cache_hit is False
        assert Path(m2.artifact_path).exists()
        assert m2.n_rows == 3

    def test_no_neuron_id(self, artifact_dir: Path):
        df = pd.DataFrame({"region": ["A", "B"], "count": [10, 20]})
        manifest = save_artifact(
            df, tool="region_connectivity", dataset="hemibrain",
        )
        assert manifest.n_rows == 2
        path = Path(manifest.artifact_path)
        assert "hemibrain_region_connectivity_nid-None_xk-None_vNone_" in path.name

    def test_no_collision_neuron_id_vs_extra_key(self, artifact_dir: Path):
        """neuron_id=12345 and extra_key='12345' must not collide."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [2]})

        m1 = save_artifact(
            df1, tool="test", dataset="minnie65",
            neuron_id=12345, extra_key=None, materialization_version=100,
        )
        m2 = save_artifact(
            df2, tool="test", dataset="minnie65",
            neuron_id=None, extra_key="12345", materialization_version=100,
        )

        # Must be different files — no collision
        assert m1.artifact_path != m2.artifact_path
        assert m1.columns == ["a"]
        assert m2.columns == ["b"]


class TestLoadCachedArtifact:
    def test_returns_none_on_miss(self, artifact_dir: Path):
        result = load_cached_artifact(
            tool="connectivity", dataset="minnie65",
            neuron_id=999, materialization_version=943,
        )
        assert result is None

    def test_returns_df_and_manifest_on_hit(self, artifact_dir: Path):
        df = _sample_df()
        save_artifact(
            df, tool="connectivity", dataset="minnie65",
            neuron_id=123, materialization_version=943,
        )

        result = load_cached_artifact(
            tool="connectivity", dataset="minnie65",
            neuron_id=123, materialization_version=943,
        )
        assert result is not None
        cached_df, manifest = result
        assert len(cached_df) == 3
        assert manifest.cache_hit is True
        assert manifest.n_rows == 3

    def test_respects_extra_key(self, artifact_dir: Path):
        df = _sample_df()
        save_artifact(
            df, tool="bulk", dataset="minnie65",
            neuron_id=None, extra_key="abc123",
            materialization_version=100,
        )

        # Same extra_key → hit
        assert load_cached_artifact(
            tool="bulk", dataset="minnie65",
            neuron_id=None, extra_key="abc123",
            materialization_version=100,
        ) is not None

        # Different extra_key → miss
        assert load_cached_artifact(
            tool="bulk", dataset="minnie65",
            neuron_id=None, extra_key="def456",
            materialization_version=100,
        ) is None
