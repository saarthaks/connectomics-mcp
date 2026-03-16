"""Integration tests verifying cross-tool workflows and server wiring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from connectomics_mcp.exceptions import DatasetNotSupported, StaleRootIdError
from connectomics_mcp.output_contracts.schemas import (
    ConnectivityResponse,
    CypherQueryResponse,
    NeuronInfoResponse,
    NeuronsByTypeResponse,
    RootIdValidationResponse,
    SynapseCompartmentResponse,
)
from connectomics_mcp.tools import cave_specific, neuprint_specific, universal


class TestServerSetup:
    def test_mcp_server_has_correct_name(self):
        from connectomics_mcp.server import mcp

        assert mcp.name == "connectomics-mcp"

    def test_all_tools_registered(self):
        from connectomics_mcp.server import mcp

        expected_tools = {
            "get_neuron_info",
            "get_connectivity",
            "validate_root_ids",
            "get_proofreading_status",
            "build_neuroglancer_url",
            "get_cell_type_taxonomy",
            "search_cell_types",
            "get_neurons_by_type",
            "get_region_connectivity",
            "resolve_nucleus_ids",
            "query_annotation_table",
            "get_edit_history",
            "fetch_cypher",
            "get_synapse_compartments",
        }
        # FastMCP stores tools internally; access via _tool_manager
        registered = set(mcp._tool_manager._tools.keys())
        assert expected_tools.issubset(registered), (
            f"Missing tools: {expected_tools - registered}"
        )


class TestCrossToolWorkflows:
    def test_validate_then_get_neuron_info(
        self, mock_cave_backend, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))

        # Step 1: validate root IDs
        val_result = universal.validate_root_ids(
            [720575940621039145], "minnie65"
        )
        val_resp = RootIdValidationResponse(**val_result)
        assert val_resp.results[0].is_current is True

        # Step 2: use validated ID to get neuron info
        neuron_id = val_resp.results[0].root_id
        info_result = universal.get_neuron_info(neuron_id, "minnie65")
        info_resp = NeuronInfoResponse(**info_result)
        assert info_resp.neuron_id == neuron_id
        assert info_resp.cell_type is not None

    def test_get_neuron_info_then_connectivity(
        self, mock_cave_backend, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))

        # Step 1: get neuron info
        info_result = universal.get_neuron_info(
            720575940621039145, "minnie65"
        )
        info_resp = NeuronInfoResponse(**info_result)

        # Step 2: use same neuron_id for connectivity
        conn_result = universal.get_connectivity(
            info_resp.neuron_id, "minnie65"
        )
        conn_resp = ConnectivityResponse(**conn_result)
        assert conn_resp.neuron_id == info_resp.neuron_id

        # Verify artifact exists and is readable
        artifact_path = conn_resp.artifact_manifest.artifact_path
        assert Path(artifact_path).exists()
        df = pd.read_parquet(artifact_path)
        assert len(df) == conn_resp.artifact_manifest.n_rows

    def test_get_neurons_by_type_then_neuron_info(
        self, mock_cave_backend, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))

        # Step 1: get all neurons of a type
        type_result = universal.get_neurons_by_type("L2/3 IT", "minnie65")
        type_resp = NeuronsByTypeResponse(**type_result)
        assert type_resp.n_total > 0

        # Step 2: load artifact and pick a neuron
        df = pd.read_parquet(type_resp.artifact_manifest.artifact_path)
        first_neuron_id = int(df.iloc[0]["neuron_id"])

        # Step 3: get info for that neuron
        info_result = universal.get_neuron_info(first_neuron_id, "minnie65")
        info_resp = NeuronInfoResponse(**info_result)
        assert info_resp.neuron_id == first_neuron_id

    def test_neuprint_workflow(
        self, mock_neuprint_backend, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))

        # Step 1: get neuron info
        info_result = universal.get_neuron_info(12345, "hemibrain")
        info_resp = NeuronInfoResponse(**info_result)
        assert info_resp.dataset == "hemibrain"

        # Step 2: get synapse compartments for the same neuron
        comp_result = neuprint_specific.get_synapse_compartments(
            info_resp.neuron_id, "hemibrain"
        )
        comp_resp = SynapseCompartmentResponse(**comp_result)
        assert comp_resp.neuron_id == info_resp.neuron_id
        assert len(comp_resp.compartments) > 0

        # Step 3: run a Cypher query
        cypher_result = neuprint_specific.fetch_cypher(
            "MATCH (n:Neuron) RETURN n.bodyId LIMIT 5", "hemibrain"
        )
        cypher_resp = CypherQueryResponse(**cypher_result)
        assert cypher_resp.n_rows > 0
        assert Path(cypher_resp.artifact_manifest.artifact_path).exists()


class TestErrorPropagation:
    def test_stale_root_id_across_tools(
        self, mock_cave_backend_stale, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        stale_id = 720575940621039145

        with pytest.raises(StaleRootIdError):
            universal.get_neuron_info(stale_id, "minnie65")

        with pytest.raises(StaleRootIdError):
            universal.get_connectivity(stale_id, "minnie65")

        with pytest.raises(StaleRootIdError):
            cave_specific.get_edit_history(stale_id, "minnie65")

    def test_dataset_not_supported_across_tiers(
        self, mock_cave_backend, mock_neuprint_backend, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))

        # CAVE tool on neuPrint dataset
        with pytest.raises(DatasetNotSupported):
            cave_specific.get_proofreading_status(12345, "hemibrain")

        # neuPrint tool on CAVE dataset
        with pytest.raises(DatasetNotSupported):
            neuprint_specific.fetch_cypher("MATCH (n) RETURN n", "minnie65")

        with pytest.raises(DatasetNotSupported):
            neuprint_specific.get_synapse_compartments(
                720575940621039145, "minnie65"
            )


class TestArtifactConsistency:
    def test_connectivity_artifact_round_trip(
        self, mock_cave_backend, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))

        result = universal.get_connectivity(
            720575940621039145, "minnie65"
        )
        resp = ConnectivityResponse(**result)

        df = pd.read_parquet(resp.artifact_manifest.artifact_path)

        # Verify expected columns are present
        expected_cols = {"partner_id", "direction", "n_synapses"}
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing columns: {expected_cols - set(df.columns)}"
        )

        # Row count matches manifest
        assert len(df) == resp.artifact_manifest.n_rows

        # Total counts match actual data
        n_upstream = len(df[df["direction"] == "upstream"])
        n_downstream = len(df[df["direction"] == "downstream"])
        assert resp.n_upstream_total == n_upstream
        assert resp.n_downstream_total == n_downstream

    def test_multiple_artifact_tools_share_cache_dir(
        self, mock_cave_backend, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))

        # Call two different artifact-producing tools
        conn_result = universal.get_connectivity(
            720575940621039145, "minnie65"
        )
        type_result = universal.get_neurons_by_type("L2/3 IT", "minnie65")

        conn_resp = ConnectivityResponse(**conn_result)
        type_resp = NeuronsByTypeResponse(**type_result)

        # Both artifacts should be in the same directory
        conn_dir = Path(conn_resp.artifact_manifest.artifact_path).parent
        type_dir = Path(type_resp.artifact_manifest.artifact_path).parent
        assert conn_dir == type_dir == tmp_path

        # Both should be valid Parquet files
        conn_df = pd.read_parquet(conn_resp.artifact_manifest.artifact_path)
        type_df = pd.read_parquet(type_resp.artifact_manifest.artifact_path)
        assert len(conn_df) > 0
        assert len(type_df) > 0
