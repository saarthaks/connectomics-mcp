"""Tests for neuPrint-specific tools (Tier 3)."""

from __future__ import annotations

import pandas as pd
import pytest

from connectomics_mcp.exceptions import DatasetNotSupported
from connectomics_mcp.output_contracts.schemas import (
    CypherQueryResponse,
    SynapseCompartmentResponse,
)
from connectomics_mcp.tools.neuprint_specific import (
    fetch_cypher,
    get_synapse_compartments,
)


class TestFetchCypher:
    def test_normal_response(self, mock_neuprint_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        query = "MATCH (n:Neuron) RETURN n.bodyId, n.type, n.pre, n.post LIMIT 10"
        result = fetch_cypher(query, "hemibrain")

        resp = CypherQueryResponse(**result)
        assert resp.dataset == "hemibrain"
        assert resp.n_rows == 4
        assert len(resp.columns) > 0
        assert resp.warnings == []

    def test_artifact_written_and_readable(self, mock_neuprint_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        query = "MATCH (n:Neuron) RETURN n.bodyId, n.type LIMIT 5"
        result = fetch_cypher(query, "hemibrain")

        resp = CypherQueryResponse(**result)
        artifact_path = resp.artifact_manifest.artifact_path
        df = pd.read_parquet(artifact_path)
        assert len(df) == resp.n_rows
        assert list(df.columns) == resp.columns

    def test_n_rows_matches_artifact(self, mock_neuprint_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = fetch_cypher("MATCH (n) RETURN n", "hemibrain")

        resp = CypherQueryResponse(**result)
        df = pd.read_parquet(resp.artifact_manifest.artifact_path)
        assert resp.n_rows == len(df)
        assert resp.artifact_manifest.n_rows == len(df)

    def test_query_echoed(self, mock_neuprint_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        query = "MATCH (n:Neuron {type:'KC-ab'}) RETURN n.bodyId"
        result = fetch_cypher(query, "hemibrain")

        resp = CypherQueryResponse(**result)
        assert resp.query == query

    def test_cave_dataset_raises(self, mock_cave_backend):
        with pytest.raises(DatasetNotSupported):
            fetch_cypher("MATCH (n) RETURN n", "minnie65")

    def test_unsupported_dataset_raises(self):
        with pytest.raises(DatasetNotSupported):
            fetch_cypher("MATCH (n) RETURN n", "nonexistent_dataset")

    def test_response_validates_against_schema(self, mock_neuprint_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = fetch_cypher("MATCH (n) RETURN n", "hemibrain")
        resp = CypherQueryResponse.model_validate(result)
        json_str = resp.model_dump_json()
        assert isinstance(json_str, str)

    def test_different_queries_get_different_artifacts(self, mock_neuprint_backend, tmp_path, monkeypatch):
        """Regression: different Cypher queries must not collide in cache."""
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        query_a = "MATCH (n:Neuron) RETURN n.bodyId LIMIT 10"
        query_b = "MATCH (n:Neuron)-[c:ConnectsTo]->(m) RETURN n.bodyId, m.bodyId"

        result_a = fetch_cypher(query_a, "hemibrain")
        result_b = fetch_cypher(query_b, "hemibrain")

        resp_a = CypherQueryResponse(**result_a)
        resp_b = CypherQueryResponse(**result_b)

        # Different queries should produce different artifact files
        assert resp_a.artifact_manifest.artifact_path != resp_b.artifact_manifest.artifact_path


class TestGetSynapseCompartments:
    def test_normal_input_response(self, mock_neuprint_backend):
        result = get_synapse_compartments(12345, "hemibrain", direction="input")

        resp = SynapseCompartmentResponse(**result)
        assert resp.neuron_id == "12345"
        assert resp.dataset == "hemibrain"
        assert resp.direction == "input"
        assert len(resp.compartments) > 0
        assert resp.n_total_synapses > 0
        assert resp.warnings == []

    def test_output_direction(self, mock_neuprint_backend):
        result_input = get_synapse_compartments(12345, "hemibrain", direction="input")
        result_output = get_synapse_compartments(12345, "hemibrain", direction="output")

        resp_in = SynapseCompartmentResponse(**result_input)
        resp_out = SynapseCompartmentResponse(**result_output)

        # Output (pre) should have different counts than input (post)
        assert resp_out.direction == "output"
        assert resp_out.n_total_synapses != resp_in.n_total_synapses

    def test_fractions_sum_to_one(self, mock_neuprint_backend):
        result = get_synapse_compartments(12345, "hemibrain", direction="input")

        resp = SynapseCompartmentResponse(**result)
        total_fraction = sum(c.fraction for c in resp.compartments)
        assert abs(total_fraction - 1.0) < 0.01

    def test_compartment_fields_valid(self, mock_neuprint_backend):
        result = get_synapse_compartments(12345, "hemibrain", direction="input")

        resp = SynapseCompartmentResponse(**result)
        for comp in resp.compartments:
            assert isinstance(comp.compartment, str)
            assert len(comp.compartment) > 0
            assert comp.n_synapses > 0
            assert 0.0 < comp.fraction <= 1.0

    def test_cave_dataset_raises(self, mock_cave_backend):
        with pytest.raises(DatasetNotSupported):
            get_synapse_compartments(720575940621039145, "minnie65")

    def test_unsupported_dataset_raises(self):
        with pytest.raises(DatasetNotSupported):
            get_synapse_compartments(12345, "nonexistent_dataset")

    def test_response_validates_against_schema(self, mock_neuprint_backend):
        result = get_synapse_compartments(12345, "hemibrain")
        resp = SynapseCompartmentResponse.model_validate(result)
        json_str = resp.model_dump_json()
        assert isinstance(json_str, str)
