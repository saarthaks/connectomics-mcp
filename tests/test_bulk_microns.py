"""Tests for bulk MICrONS-specific tools."""

from __future__ import annotations

import pandas as pd
import pytest

from connectomics_mcp.exceptions import DatasetNotSupported
from connectomics_mcp.tools.cave_specific import (
    get_bulk_coregistration,
    get_bulk_functional_area,
    get_bulk_functional_properties,
    get_bulk_synapse_targets,
)

IDS = [864691135000000, 864691135000001, 864691135000002]


class TestBulkCoregistration:

    def test_normal_response(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_coregistration(IDS, "minnie65")
        assert result["dataset"] == "minnie65"
        assert result["n_root_ids"] == 3
        assert result["n_units"] > 0
        assert result["cached"] is False
        assert result["artifact_manifest"] is not None

    def test_artifact_readable(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_coregistration(IDS[:2], "minnie65")
        path = result["artifact_manifest"]["artifact_path"]
        df = pd.read_parquet(path)
        assert not df.empty
        assert "score" in df.columns
        assert "session" in df.columns

    def test_score_distribution(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_coregistration(IDS, "minnie65")
        dist = result["score_distribution"]
        assert "mean" in dist
        assert "median" in dist
        assert "max" in dist

    def test_sessions_populated(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_coregistration(IDS, "minnie65")
        assert len(result["sessions"]) > 0

    def test_empty_ids(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_coregistration([], "minnie65")
        assert result["n_root_ids"] == 0
        assert result["n_units"] == 0

    def test_stale_raises(self, mock_cave_backend_stale, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="Stale root IDs"):
            get_bulk_coregistration(IDS, "minnie65")

    def test_hemibrain_raises(self, mock_neuprint_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(DatasetNotSupported):
            get_bulk_coregistration(IDS, "hemibrain")


class TestBulkFunctionalProperties:

    def test_normal_response(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_functional_properties(IDS, "minnie65")
        assert result["dataset"] == "minnie65"
        assert result["n_root_ids"] == 3
        assert result["n_units"] > 0
        assert result["coregistration_source"] == "auto_phase3"

    def test_artifact_has_osi(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_functional_properties(IDS[:2], "minnie65")
        path = result["artifact_manifest"]["artifact_path"]
        df = pd.read_parquet(path)
        assert "OSI" in df.columns
        assert "DSI" in df.columns

    def test_ori_distribution(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_functional_properties(IDS, "minnie65")
        assert "mean" in result["ori_selectivity_distribution"]

    def test_alt_source(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_functional_properties(
            IDS, "minnie65", coregistration_source="coreg_v4"
        )
        assert result["coregistration_source"] == "coreg_v4"

    def test_empty_ids(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_functional_properties([], "minnie65")
        assert result["n_units"] == 0

    def test_stale_raises(self, mock_cave_backend_stale, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="Stale root IDs"):
            get_bulk_functional_properties(IDS, "minnie65")

    def test_hemibrain_raises(self, mock_neuprint_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(DatasetNotSupported):
            get_bulk_functional_properties(IDS, "hemibrain")


class TestBulkSynapseTargets:

    def test_normal_response(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_synapse_targets(IDS, "minnie65")
        assert result["dataset"] == "minnie65"
        assert result["n_root_ids"] == 3
        assert result["n_synapses"] > 0
        assert result["direction"] == "post"

    def test_target_distribution(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_synapse_targets(IDS, "minnie65")
        dist = result["target_distribution"]
        assert "spine" in dist
        assert "shaft" in dist

    def test_artifact_has_tag(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_synapse_targets(IDS[:2], "minnie65")
        path = result["artifact_manifest"]["artifact_path"]
        df = pd.read_parquet(path)
        assert "tag" in df.columns

    def test_direction_pre(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_synapse_targets(IDS, "minnie65", direction="pre")
        assert result["direction"] == "pre"

    def test_empty_ids(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_synapse_targets([], "minnie65")
        assert result["n_synapses"] == 0

    def test_stale_raises(self, mock_cave_backend_stale, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="Stale root IDs"):
            get_bulk_synapse_targets(IDS, "minnie65")

    def test_hemibrain_raises(self, mock_neuprint_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(DatasetNotSupported):
            get_bulk_synapse_targets(IDS, "hemibrain")


class TestBulkFunctionalArea:

    def test_normal_response(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_functional_area(IDS, "minnie65")
        assert result["dataset"] == "minnie65"
        assert result["n_root_ids"] == 3
        assert result["n_total"] > 0

    def test_area_distribution(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_functional_area(IDS, "minnie65")
        assert "V1" in result["area_distribution"]

    def test_artifact_has_tag_and_value(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_functional_area(IDS[:2], "minnie65")
        path = result["artifact_manifest"]["artifact_path"]
        df = pd.read_parquet(path)
        assert "tag" in df.columns
        assert "value" in df.columns

    def test_empty_ids(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_functional_area([], "minnie65")
        assert result["n_total"] == 0

    def test_stale_raises(self, mock_cave_backend_stale, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="Stale root IDs"):
            get_bulk_functional_area(IDS, "minnie65")

    def test_hemibrain_raises(self, mock_neuprint_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(DatasetNotSupported):
            get_bulk_functional_area(IDS, "hemibrain")
