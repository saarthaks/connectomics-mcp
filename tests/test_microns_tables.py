"""Tests for MICrONS-specific annotation table tools."""

from __future__ import annotations

import pandas as pd
import pytest

from connectomics_mcp.exceptions import DatasetNotSupported, StaleRootIdError
from connectomics_mcp.tools import cave_specific


MOCK_ROOT_ID = 864691135571546917
MOCK_NUCLEUS_ID = 264824


# ---------------------------------------------------------------------------
# get_coregistration
# ---------------------------------------------------------------------------


class TestGetCoregistration:
    def test_normal_by_root_id(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_coregistration(MOCK_ROOT_ID, "minnie65")
        assert result["n_units"] == 3
        assert result["query_by"] == "root_id"
        assert 4 in result["sessions"]
        assert 9 in result["sessions"]
        assert result["artifact_manifest"] is not None
        df = pd.read_parquet(result["artifact_manifest"]["artifact_path"])
        assert len(df) == 3

    def test_by_nucleus_id(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_coregistration(
            MOCK_NUCLEUS_ID, "minnie65", by="nucleus_id"
        )
        assert result["query_by"] == "nucleus_id"
        assert result["n_units"] == 3

    def test_score_distribution(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_coregistration(MOCK_ROOT_ID, "minnie65")
        dist = result["score_distribution"]
        assert "mean" in dist
        assert "median" in dist
        assert "max" in dist

    def test_minnie65_only(self, mock_cave_backend):
        with pytest.raises(DatasetNotSupported):
            cave_specific.get_coregistration(MOCK_ROOT_ID, "flywire")

    def test_stale_raises(self, mock_cave_backend_stale, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(StaleRootIdError):
            cave_specific.get_coregistration(MOCK_ROOT_ID, "minnie65")


# ---------------------------------------------------------------------------
# get_functional_properties
# ---------------------------------------------------------------------------


class TestGetFunctionalProperties:
    def test_normal(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_functional_properties(
            MOCK_ROOT_ID, "minnie65"
        )
        assert result["n_units"] == 2
        assert result["coregistration_source"] == "auto_phase3"
        assert result["artifact_manifest"] is not None
        df = pd.read_parquet(result["artifact_manifest"]["artifact_path"])
        assert "OSI" in df.columns
        assert "DSI" in df.columns

    def test_alt_coregistration_source(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_functional_properties(
            MOCK_ROOT_ID, "minnie65", coregistration_source="coreg_v4"
        )
        assert result["coregistration_source"] == "coreg_v4"

    def test_osi_distribution(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_functional_properties(
            MOCK_ROOT_ID, "minnie65"
        )
        assert "mean" in result["ori_selectivity_distribution"]

    def test_minnie65_only(self, mock_cave_backend):
        with pytest.raises(DatasetNotSupported):
            cave_specific.get_functional_properties(MOCK_ROOT_ID, "flywire")

    def test_stale_raises(self, mock_cave_backend_stale, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(StaleRootIdError):
            cave_specific.get_functional_properties(MOCK_ROOT_ID, "minnie65")


# ---------------------------------------------------------------------------
# get_synapse_targets
# ---------------------------------------------------------------------------


class TestGetSynapseTargets:
    def test_normal(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_synapse_targets(MOCK_ROOT_ID, "minnie65")
        assert result["n_synapses"] == 6
        assert result["direction"] == "post"
        dist = result["target_distribution"]
        assert dist["spine"] == 3
        assert dist["shaft"] == 2
        assert dist["soma"] == 1

    def test_direction_pre(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_synapse_targets(
            MOCK_ROOT_ID, "minnie65", direction="pre"
        )
        assert result["direction"] == "pre"

    def test_artifact_readable(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_synapse_targets(MOCK_ROOT_ID, "minnie65")
        df = pd.read_parquet(result["artifact_manifest"]["artifact_path"])
        assert len(df) == 6
        assert "tag" in df.columns

    def test_minnie65_only(self, mock_cave_backend):
        with pytest.raises(DatasetNotSupported):
            cave_specific.get_synapse_targets(MOCK_ROOT_ID, "flywire")

    def test_stale_raises(self, mock_cave_backend_stale, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(StaleRootIdError):
            cave_specific.get_synapse_targets(MOCK_ROOT_ID, "minnie65")


# ---------------------------------------------------------------------------
# get_multi_input_spines
# ---------------------------------------------------------------------------


class TestGetMultiInputSpines:
    def test_normal(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_multi_input_spines(MOCK_ROOT_ID, "minnie65")
        assert result["n_synapses"] == 5
        assert result["n_spine_groups"] == 2
        assert result["direction"] == "post"

    def test_deprecation_warning(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_multi_input_spines(MOCK_ROOT_ID, "minnie65")
        assert any("deprecated" in w.lower() for w in result["warnings"])

    def test_artifact_has_group_id(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_multi_input_spines(MOCK_ROOT_ID, "minnie65")
        df = pd.read_parquet(result["artifact_manifest"]["artifact_path"])
        assert "group_id" in df.columns

    def test_minnie65_only(self, mock_cave_backend):
        with pytest.raises(DatasetNotSupported):
            cave_specific.get_multi_input_spines(MOCK_ROOT_ID, "flywire")

    def test_stale_raises(self, mock_cave_backend_stale, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(StaleRootIdError):
            cave_specific.get_multi_input_spines(MOCK_ROOT_ID, "minnie65")


# ---------------------------------------------------------------------------
# get_cell_mtypes
# ---------------------------------------------------------------------------


class TestGetCellMtypes:
    def test_population_query(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_cell_mtypes("minnie65")
        assert result["n_total"] == 5
        assert "excitatory" in result["classification_system_distribution"]
        assert "inhibitory" in result["classification_system_distribution"]

    def test_filter_by_type(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_cell_mtypes(
            "minnie65", cell_type="L2a"
        )
        assert result["n_total"] == 2
        assert result["query_cell_type"] == "L2a"

    def test_single_neuron(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_cell_mtypes(
            "minnie65", neuron_id=MOCK_ROOT_ID
        )
        assert result["n_total"] == 1
        df = pd.read_parquet(result["artifact_manifest"]["artifact_path"])
        assert df.iloc[0]["cell_type"] == "L2a"

    def test_by_nucleus_id(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_cell_mtypes(
            "minnie65", neuron_id=MOCK_NUCLEUS_ID, by="nucleus_id"
        )
        assert result["n_total"] == 1
        assert result["query_by"] == "nucleus_id"

    def test_minnie65_only(self, mock_cave_backend):
        with pytest.raises(DatasetNotSupported):
            cave_specific.get_cell_mtypes("flywire")

    def test_stale_raises(self, mock_cave_backend_stale, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(StaleRootIdError):
            cave_specific.get_cell_mtypes(
                "minnie65", neuron_id=MOCK_ROOT_ID
            )


# ---------------------------------------------------------------------------
# get_functional_area
# ---------------------------------------------------------------------------


class TestGetFunctionalArea:
    def test_population_query(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_functional_area("minnie65")
        assert result["n_total"] == 4
        assert "V1" in result["area_distribution"]
        assert result["area_distribution"]["V1"] == 2

    def test_filter_by_area(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_functional_area("minnie65", area="V1")
        assert result["n_total"] == 2
        assert result["query_area"] == "V1"

    def test_single_neuron_by_nucleus(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_functional_area(
            "minnie65", neuron_id=MOCK_NUCLEUS_ID, by="nucleus_id"
        )
        assert result["n_total"] == 1
        df = pd.read_parquet(result["artifact_manifest"]["artifact_path"])
        assert df.iloc[0]["tag"] == "V1"

    def test_artifact_has_value_column(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = cave_specific.get_functional_area("minnie65")
        df = pd.read_parquet(result["artifact_manifest"]["artifact_path"])
        assert "value" in df.columns
        assert "tag" in df.columns

    def test_minnie65_only(self, mock_cave_backend):
        with pytest.raises(DatasetNotSupported):
            cave_specific.get_functional_area("flywire")

    def test_stale_raises(self, mock_cave_backend_stale, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        with pytest.raises(StaleRootIdError):
            cave_specific.get_functional_area(
                "minnie65", neuron_id=MOCK_ROOT_ID
            )


# ---------------------------------------------------------------------------
# Artifact cache bug fix verification
# ---------------------------------------------------------------------------


class TestArtifactCacheBugFix:
    """Verify that different annotation table queries produce distinct artifacts."""

    def test_different_tables_distinct_artifacts(
        self, mock_cave_backend, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))

        result_a = cave_specific.query_annotation_table(
            "minnie65", "table_a"
        )
        result_b = cave_specific.query_annotation_table(
            "minnie65", "table_b"
        )

        path_a = result_a["artifact_manifest"]["artifact_path"]
        path_b = result_b["artifact_manifest"]["artifact_path"]
        assert path_a != path_b
