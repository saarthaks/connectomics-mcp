"""Tests for FlyWire-specific annotation support.

Tests hierarchy classification, neurotransmitter predictions,
and backward compatibility with minnie65/hemibrain.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from connectomics_mcp.output_contracts.schemas import (
    ConnectivityResponse,
    NeuronInfoResponse,
    NeuronsByTypeResponse,
)
from connectomics_mcp.tools import universal


# ── FlyWire neuron info ──────────────────────────────────────────────


class TestFlyWireNeuronInfo:
    """get_neuron_info returns hierarchy and NT for FlyWire."""

    ROOT_ID = 720575940621039145

    def test_returns_classification_hierarchy(
        self, mock_flywire_backend, tmp_path: Path
    ) -> None:
        os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
        result = universal.get_neuron_info(self.ROOT_ID, "flywire")
        resp = NeuronInfoResponse(**result)

        assert resp.classification_hierarchy is not None
        assert "cell_type" in resp.classification_hierarchy
        assert resp.classification_hierarchy["cell_type"] == "DA1_lPN"
        assert "super_class" in resp.classification_hierarchy
        assert "cell_class" in resp.classification_hierarchy

    def test_returns_neurotransmitter_type(
        self, mock_flywire_backend, tmp_path: Path
    ) -> None:
        os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
        result = universal.get_neuron_info(self.ROOT_ID, "flywire")
        resp = NeuronInfoResponse(**result)

        assert resp.neurotransmitter_type is not None
        assert resp.neurotransmitter_type == "acetylcholine"

    def test_cell_type_from_hierarchy(
        self, mock_flywire_backend, tmp_path: Path
    ) -> None:
        os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
        result = universal.get_neuron_info(self.ROOT_ID, "flywire")
        resp = NeuronInfoResponse(**result)

        # cell_type should come from finest hierarchy level
        assert resp.cell_type == "DA1_lPN"

    def test_synapse_counts_populated(
        self, mock_flywire_backend, tmp_path: Path
    ) -> None:
        os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
        result = universal.get_neuron_info(self.ROOT_ID, "flywire")
        resp = NeuronInfoResponse(**result)

        assert resp.n_pre_synapses is not None
        assert resp.n_post_synapses is not None
        assert resp.n_pre_synapses > 0
        assert resp.n_post_synapses > 0


# ── FlyWire connectivity ─────────────────────────────────────────────


class TestFlyWireConnectivity:
    """get_connectivity returns NT-enriched artifacts for FlyWire."""

    ROOT_ID = 720575940621039145

    def test_artifact_has_nt_columns(
        self, mock_flywire_backend, tmp_path: Path
    ) -> None:
        os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
        result = universal.get_connectivity(self.ROOT_ID, "flywire")
        resp = ConnectivityResponse(**result)

        # Check artifact has NT columns
        manifest = resp.artifact_manifest
        assert manifest is not None
        df = pd.read_parquet(manifest.artifact_path)
        assert "partner_nt_type" in df.columns
        assert "partner_nt_confidence" in df.columns

        # Check values are populated
        assert df["partner_nt_type"].notna().any()
        assert df["partner_nt_confidence"].notna().any()

    def test_nt_distribution_in_response(
        self, mock_flywire_backend, tmp_path: Path
    ) -> None:
        os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
        result = universal.get_connectivity(self.ROOT_ID, "flywire")
        resp = ConnectivityResponse(**result)

        assert resp.neurotransmitter_distribution is not None
        assert isinstance(resp.neurotransmitter_distribution, dict)
        assert len(resp.neurotransmitter_distribution) > 0
        # acetylcholine should be most common in mock data
        assert "acetylcholine" in resp.neurotransmitter_distribution


# ── FlyWire neurons by type ──────────────────────────────────────────


class TestFlyWireNeuronsByType:
    """get_neurons_by_type uses hierarchy cache for FlyWire."""

    def test_returns_neurons_from_hierarchy(
        self, mock_flywire_backend, tmp_path: Path
    ) -> None:
        os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
        result = universal.get_neurons_by_type("DA1_lPN", "flywire")
        resp = NeuronsByTypeResponse(**result)

        assert resp.n_total > 0
        assert resp.query_cell_type == "DA1_lPN"
        assert resp.artifact_manifest is not None

        df = pd.read_parquet(resp.artifact_manifest.artifact_path)
        assert len(df) > 0
        assert "neuron_id" in df.columns
        assert "cell_type" in df.columns
        assert all(df["cell_type"] == "DA1_lPN")

    def test_no_results_for_unknown_type(
        self, mock_flywire_backend, tmp_path: Path
    ) -> None:
        os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
        result = universal.get_neurons_by_type("NONEXISTENT_TYPE", "flywire")
        resp = NeuronsByTypeResponse(**result)

        assert resp.n_total == 0


# ── Backward compatibility ───────────────────────────────────────────


class TestBackwardCompatibility:
    """minnie65 and hemibrain responses don't gain FlyWire-specific fields."""

    def test_minnie65_no_nt_type(
        self, mock_cave_backend, tmp_path: Path
    ) -> None:
        os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
        result = universal.get_neuron_info(864691135000000, "minnie65")
        resp = NeuronInfoResponse(**result)

        assert resp.neurotransmitter_type is None
        assert resp.classification_hierarchy is None

    def test_hemibrain_no_nt_type(
        self, mock_neuprint_backend, tmp_path: Path
    ) -> None:
        os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
        result = universal.get_neuron_info(5813105172, "hemibrain")
        resp = NeuronInfoResponse(**result)

        assert resp.neurotransmitter_type is None
        assert resp.classification_hierarchy is None

    def test_minnie65_connectivity_no_nt_dist(
        self, mock_cave_backend, tmp_path: Path
    ) -> None:
        os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
        result = universal.get_connectivity(864691135000000, "minnie65")
        resp = ConnectivityResponse(**result)

        assert resp.neurotransmitter_distribution is None

    def test_hemibrain_connectivity_no_nt_dist(
        self, mock_neuprint_backend, tmp_path: Path
    ) -> None:
        os.environ["CONNECTOMICS_MCP_ARTIFACT_DIR"] = str(tmp_path)
        result = universal.get_connectivity(5813105172, "hemibrain")
        resp = ConnectivityResponse(**result)

        assert resp.neurotransmitter_distribution is None
