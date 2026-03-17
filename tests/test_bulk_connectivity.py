"""Tests for the get_bulk_connectivity tool."""

from __future__ import annotations

import pandas as pd
import pytest

from connectomics_mcp.tools.universal import get_bulk_connectivity


class TestBulkConnectivityCAVE:
    """Tests using the mock CAVE (minnie65) backend."""

    def test_normal_response(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        ids = [864691135000000, 864691135000001, 864691135000002]
        result = get_bulk_connectivity(ids, "minnie65")

        assert result["dataset"] == "minnie65"
        assert result["n_root_ids"] == 3
        assert result["direction"] == "both"
        assert result["n_edges"] > 0
        assert result["total_synapses"] > 0
        assert result["cached"] is False
        assert result["artifact_manifest"] is not None

    def test_artifact_readable(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        ids = [864691135000000, 864691135000001]
        result = get_bulk_connectivity(ids, "minnie65")

        path = result["artifact_manifest"]["artifact_path"]
        df = pd.read_parquet(path)
        assert not df.empty
        assert list(df.columns) == [
            "pre_root_id", "post_root_id", "syn_count", "neuropil"
        ]

    def test_direction_pre(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        ids = [864691135000000, 864691135000001]
        result = get_bulk_connectivity(ids, "minnie65", direction="pre")
        assert result["direction"] == "pre"
        assert result["n_edges"] > 0

    def test_empty_ids(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_bulk_connectivity([], "minnie65")
        assert result["n_root_ids"] == 0
        assert result["n_edges"] == 0
        assert result["total_synapses"] == 0

    def test_stale_raises_valueerror(
        self, mock_cave_backend_stale, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="Stale root IDs"):
            get_bulk_connectivity(
                [864691135737064068], "minnie65"
            )


class TestBulkConnectivityNeuPrint:
    """Tests using the mock neuPrint (hemibrain) backend."""

    def test_normal_response(self, mock_neuprint_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        ids = [5813105172, 5813105173]
        result = get_bulk_connectivity(ids, "hemibrain")

        assert result["dataset"] == "hemibrain"
        assert result["n_root_ids"] == 2
        assert result["n_edges"] > 0

    def test_neuropil_populated(self, mock_neuprint_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        ids = [5813105172, 5813105173]
        result = get_bulk_connectivity(ids, "hemibrain")

        path = result["artifact_manifest"]["artifact_path"]
        df = pd.read_parquet(path)
        assert "neuropil" in df.columns
        # neuPrint mock has neuropil values
        assert df["neuropil"].notna().any()


class TestBulkConnectivityUnsupported:
    """Tests for unsupported datasets."""

    def test_unsupported_dataset(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        from connectomics_mcp.exceptions import DatasetNotSupported

        with pytest.raises(DatasetNotSupported):
            get_bulk_connectivity([1, 2], "unknown_dataset")
