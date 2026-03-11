"""Tests for universal tools (Tier 1)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from connectomics_mcp.exceptions import DatasetNotSupported, StaleRootIdError
from connectomics_mcp.output_contracts.schemas import (
    ConnectivityResponse,
    NeuroglancerUrlResponse,
    NeuronInfoResponse,
    NeuronsByTypeResponse,
    RegionConnectivityResponse,
    RootIdValidationResponse,
)
from connectomics_mcp.tools.universal import (
    build_neuroglancer_url_tool,
    get_connectivity,
    get_neuron_info,
    get_neurons_by_type,
    get_region_connectivity,
    validate_root_ids,
)


class TestGetNeuronInfo:
    def test_cave_normal_response(self, mock_cave_backend):
        result = get_neuron_info(720575940621039145, "minnie65")

        resp = NeuronInfoResponse(**result)
        assert resp.neuron_id == 720575940621039145
        assert resp.dataset == "minnie65"
        assert resp.cell_type == "L2/3 IT"
        assert resp.cell_class == "excitatory"
        assert resp.soma_position_nm == (200000.0, 300000.0, 400000.0)
        assert resp.n_pre_synapses == 1500
        assert resp.n_post_synapses == 3200
        assert resp.materialization_version == 943
        assert resp.neuroglancer_url != ""
        assert resp.warnings == []

    def test_neuprint_normal_response(self, mock_neuprint_backend):
        result = get_neuron_info(12345, "hemibrain")

        resp = NeuronInfoResponse(**result)
        assert resp.neuron_id == 12345
        assert resp.dataset == "hemibrain"
        assert resp.cell_type == "MBON14"
        assert resp.neuroglancer_url != ""
        assert resp.proofread is None

    def test_stale_root_id_raises(self, mock_cave_backend_stale):
        with pytest.raises(StaleRootIdError) as exc_info:
            get_neuron_info(720575940621039145, "minnie65")
        assert "720575940621039145" in str(exc_info.value)
        assert "validate_root_ids" in str(exc_info.value)

    def test_unsupported_dataset_raises(self):
        with pytest.raises(DatasetNotSupported):
            get_neuron_info(123, "nonexistent_dataset")

    def test_response_validates_against_schema(self, mock_cave_backend):
        result = get_neuron_info(720575940621039145, "minnie65")
        resp = NeuronInfoResponse.model_validate(result)
        data = resp.model_dump()
        assert isinstance(data, dict)
        json_str = resp.model_dump_json()
        assert isinstance(json_str, str)

    def test_neuroglancer_url_present(self, mock_cave_backend):
        result = get_neuron_info(720575940621039145, "minnie65")
        assert result["neuroglancer_url"].startswith("https://")

    def test_nucleus_id_resolved(self, mock_cave_backend):
        """Provide nucleus_id for minnie65; should resolve and return info."""
        result = get_neuron_info(0, "minnie65", nucleus_id=100001)
        resp = NeuronInfoResponse(**result)
        # Should have used the resolved pt_root_id 864691135000001
        assert resp.neuron_id == 864691135000001
        assert resp.warnings == []

    def test_nucleus_id_merge_conflict_warning(self, mock_cave_backend):
        """Merge conflict should proceed with warning."""
        result = get_neuron_info(0, "minnie65", nucleus_id=100002)
        resp = NeuronInfoResponse(**result)
        assert resp.neuron_id == 864691135000099
        assert any("merge error" in w for w in resp.warnings)
        assert any("100003" in w for w in resp.warnings)

    def test_nucleus_id_no_segment_raises(self, mock_cave_backend):
        """No segment should raise ValueError."""
        with pytest.raises(ValueError, match="no associated segment"):
            get_neuron_info(0, "minnie65", nucleus_id=999999)

    def test_nucleus_id_non_minnie65_raises(self, mock_neuprint_backend):
        """nucleus_id on non-minnie65 should raise DatasetNotSupported."""
        with pytest.raises(DatasetNotSupported):
            get_neuron_info(12345, "hemibrain", nucleus_id=100001)


class TestGetConnectivity:
    @pytest.fixture(autouse=True)
    def _artifact_dir(self, tmp_path: Path):
        """Route artifacts to a temp dir for all connectivity tests."""
        with patch.dict(os.environ, {"CONNECTOMICS_MCP_ARTIFACT_DIR": str(tmp_path)}):
            self._tmp = tmp_path
            yield

    def test_cave_normal_response(self, mock_cave_backend):
        result = get_connectivity(720575940621039145, "minnie65")

        resp = ConnectivityResponse(**result)
        assert resp.neuron_id == 720575940621039145
        assert resp.dataset == "minnie65"
        assert resp.n_upstream_total == 5
        assert resp.n_downstream_total == 4
        assert resp.neuroglancer_url != ""
        assert resp.warnings == []

    def test_neuprint_normal_response(self, mock_neuprint_backend):
        result = get_connectivity(12345, "hemibrain")

        resp = ConnectivityResponse(**result)
        assert resp.neuron_id == 12345
        assert resp.dataset == "hemibrain"
        assert resp.n_upstream_total == 6
        assert resp.n_downstream_total == 3

    def test_artifact_written_and_readable(self, mock_cave_backend):
        result = get_connectivity(720575940621039145, "minnie65")

        manifest = result["artifact_manifest"]
        assert manifest is not None
        path = Path(manifest["artifact_path"])
        assert path.exists()

        df = pd.read_parquet(path)
        assert len(df) == manifest["n_rows"]
        # Total rows should match upstream + downstream
        assert len(df) == 9  # 5 upstream + 4 downstream

    def test_artifact_row_count_matches_totals(self, mock_cave_backend):
        result = get_connectivity(720575940621039145, "minnie65")

        manifest = result["artifact_manifest"]
        df = pd.read_parquet(manifest["artifact_path"])
        upstream_rows = df[df["direction"] == "upstream"]
        downstream_rows = df[df["direction"] == "downstream"]
        assert len(upstream_rows) == result["n_upstream_total"]
        assert len(downstream_rows) == result["n_downstream_total"]

    def test_upstream_sample_has_3_entries(self, mock_cave_backend):
        result = get_connectivity(720575940621039145, "minnie65")
        assert len(result["upstream_sample"]) == 3
        # They should be the top-3 by synapse count
        synapse_counts = [s["n_synapses"] for s in result["upstream_sample"]]
        assert synapse_counts == sorted(synapse_counts, reverse=True)

    def test_downstream_sample_has_3_entries(self, mock_cave_backend):
        result = get_connectivity(720575940621039145, "minnie65")
        assert len(result["downstream_sample"]) == 3

    def test_sample_note_present(self, mock_cave_backend):
        result = get_connectivity(720575940621039145, "minnie65")
        assert "sample_note" in result
        assert "orientation only" in result["sample_note"]
        assert "artifact_manifest" in result["sample_note"]

    def test_weight_distribution_keys(self, mock_cave_backend):
        result = get_connectivity(720575940621039145, "minnie65")
        for dist_key in ["upstream_weight_distribution", "downstream_weight_distribution"]:
            dist = result[dist_key]
            assert "mean" in dist
            assert "median" in dist
            assert "max" in dist
            assert "p90" in dist

    def test_stale_root_id_raises(self, mock_cave_backend_stale):
        with pytest.raises(StaleRootIdError) as exc_info:
            get_connectivity(720575940621039145, "minnie65")
        assert "720575940621039145" in str(exc_info.value)

    def test_unsupported_dataset_raises(self):
        with pytest.raises(DatasetNotSupported):
            get_connectivity(123, "nonexistent_dataset")

    def test_direction_upstream_only(self, mock_cave_backend):
        result = get_connectivity(720575940621039145, "minnie65", direction="upstream")
        assert result["n_upstream_total"] == 5
        assert result["n_downstream_total"] == 0

    def test_direction_downstream_only(self, mock_cave_backend):
        result = get_connectivity(720575940621039145, "minnie65", direction="downstream")
        assert result["n_upstream_total"] == 0
        assert result["n_downstream_total"] == 4

    def test_response_validates_against_schema(self, mock_cave_backend):
        result = get_connectivity(720575940621039145, "minnie65")
        resp = ConnectivityResponse.model_validate(result)
        json_str = resp.model_dump_json()
        assert isinstance(json_str, str)

    def test_per_partner_neuroglancer_urls(self, mock_cave_backend):
        result = get_connectivity(720575940621039145, "minnie65")
        manifest = result["artifact_manifest"]
        df = pd.read_parquet(manifest["artifact_path"])
        assert "neuroglancer_url" in df.columns
        # All URLs should be non-empty
        assert all(url.startswith("https://") for url in df["neuroglancer_url"])

    def test_minnie65_artifact_has_nucleus_columns(self, mock_cave_backend):
        """MICrONS connectivity artifact should include nucleus enrichment."""
        result = get_connectivity(720575940621039145, "minnie65")
        manifest = result["artifact_manifest"]
        df = pd.read_parquet(manifest["artifact_path"])

        assert "partner_nucleus_id" in df.columns
        assert "partner_nucleus_conflict" in df.columns

        # At least some partners should have nucleus IDs
        has_nuc = df["partner_nucleus_id"].notna().sum()
        assert has_nuc > 0

        # At least one conflict should exist (partner 3 upstream)
        has_conflict = df["partner_nucleus_conflict"].sum()
        assert has_conflict >= 1


class TestValidateRootIds:
    def test_cave_all_current(self, mock_cave_backend):
        result = validate_root_ids([100, 200, 300], "minnie65")

        resp = RootIdValidationResponse(**result)
        assert resp.dataset == "minnie65"
        assert resp.materialization_version == 943
        assert len(resp.results) == 3
        assert resp.n_stale == 0
        assert all(r.is_current for r in resp.results)

    def test_cave_stale_ids(self, mock_cave_backend_stale):
        result = validate_root_ids([100, 200], "minnie65")

        resp = RootIdValidationResponse(**result)
        assert resp.n_stale == 2
        assert all(not r.is_current for r in resp.results)
        # Stale IDs should have suggested replacements
        for r in resp.results:
            assert r.suggested_current_id is not None
            assert r.last_edit_timestamp is not None

    def test_neuprint_always_current(self, mock_neuprint_backend):
        result = validate_root_ids([12345, 67890], "hemibrain")

        resp = RootIdValidationResponse(**result)
        assert resp.n_stale == 0
        assert len(resp.results) == 2
        assert all(r.is_current for r in resp.results)
        # neuPrint has no timestamps or replacements
        for r in resp.results:
            assert r.suggested_current_id is None
            assert r.last_edit_timestamp is None

    def test_unsupported_dataset_raises(self):
        with pytest.raises(DatasetNotSupported):
            validate_root_ids([123], "nonexistent_dataset")

    def test_response_validates_against_schema(self, mock_cave_backend):
        result = validate_root_ids([100, 200], "minnie65")
        resp = RootIdValidationResponse.model_validate(result)
        json_str = resp.model_dump_json()
        assert isinstance(json_str, str)

    def test_n_stale_matches_actual(self, mock_cave_backend_stale):
        result = validate_root_ids([100, 200, 300], "minnie65")
        resp = RootIdValidationResponse(**result)
        actual_stale = sum(1 for r in resp.results if not r.is_current)
        assert resp.n_stale == actual_stale


class TestBuildNeuroglancerUrlTool:
    def test_normal_url_generation(self, mock_cave_backend):
        result = build_neuroglancer_url_tool([720575940621039145], "minnie65")

        resp = NeuroglancerUrlResponse(**result)
        assert resp.url.startswith("https://")
        assert resp.dataset == "minnie65"
        assert resp.n_segments == 1

    def test_layers_included(self, mock_cave_backend):
        result = build_neuroglancer_url_tool([123, 456], "minnie65")
        assert result["layers_included"] == ["em", "segmentation"]

    def test_coordinate_space(self, mock_cave_backend):
        result = build_neuroglancer_url_tool([123], "minnie65")
        assert result["coordinate_space"] == "nm"

    def test_unsupported_dataset_raises(self):
        with pytest.raises(DatasetNotSupported):
            build_neuroglancer_url_tool([123], "nonexistent_dataset")

    def test_response_validates_against_schema(self, mock_cave_backend):
        result = build_neuroglancer_url_tool([123, 456, 789], "minnie65")
        resp = NeuroglancerUrlResponse.model_validate(result)
        json_str = resp.model_dump_json()
        assert isinstance(json_str, str)
        assert resp.n_segments == 3


class TestGetNeuronsByType:
    @pytest.fixture(autouse=True)
    def _artifact_dir(self, tmp_path: Path):
        """Route artifacts to a temp dir for all neurons_by_type tests."""
        with patch.dict(os.environ, {"CONNECTOMICS_MCP_ARTIFACT_DIR": str(tmp_path)}):
            self._tmp = tmp_path
            yield

    def test_cave_normal_response(self, mock_cave_backend):
        result = get_neurons_by_type("L2/3 IT", "minnie65")

        resp = NeuronsByTypeResponse(**result)
        assert resp.dataset == "minnie65"
        assert resp.query_cell_type == "L2/3 IT"
        assert resp.n_total > 0
        assert len(resp.type_distribution) > 0
        assert len(resp.region_distribution) > 0

    def test_neuprint_normal_response(self, mock_neuprint_backend):
        result = get_neurons_by_type("KC-ab", "hemibrain")

        resp = NeuronsByTypeResponse(**result)
        assert resp.dataset == "hemibrain"
        assert resp.query_cell_type == "KC-ab"
        assert resp.n_total == 6

    def test_artifact_written_and_readable(self, mock_cave_backend):
        result = get_neurons_by_type("L2/3 IT", "minnie65")

        manifest = result["artifact_manifest"]
        assert manifest is not None
        path = Path(manifest["artifact_path"])
        assert path.exists()

        df = pd.read_parquet(path)
        assert len(df) == manifest["n_rows"]
        # Verify expected columns
        for col in ["neuron_id", "cell_type", "cell_class", "region",
                     "n_pre_synapses", "n_post_synapses", "proofread"]:
            assert col in df.columns

    def test_artifact_row_count_matches_n_total(self, mock_cave_backend):
        result = get_neurons_by_type("L2/3 IT", "minnie65")

        manifest = result["artifact_manifest"]
        df = pd.read_parquet(manifest["artifact_path"])
        assert len(df) == result["n_total"]

    def test_type_distribution_sums_to_n_total(self, mock_cave_backend):
        result = get_neurons_by_type("L2/3 IT", "minnie65")
        total = sum(result["type_distribution"].values())
        assert total == result["n_total"]

    def test_region_filter(self, mock_cave_backend):
        all_result = get_neurons_by_type("L2/3 IT", "minnie65")
        v1_result = get_neurons_by_type("L2/3 IT", "minnie65", region="V1")

        assert v1_result["n_total"] < all_result["n_total"]
        assert v1_result["n_total"] > 0

    def test_unsupported_dataset_raises(self):
        with pytest.raises(DatasetNotSupported):
            get_neurons_by_type("L2/3 IT", "nonexistent_dataset")

    def test_response_validates_against_schema(self, mock_cave_backend):
        result = get_neurons_by_type("L2/3 IT", "minnie65")
        resp = NeuronsByTypeResponse.model_validate(result)
        json_str = resp.model_dump_json()
        assert isinstance(json_str, str)


class TestGetRegionConnectivity:
    @pytest.fixture(autouse=True)
    def _artifact_dir(self, tmp_path: Path):
        """Route artifacts to a temp dir for all region connectivity tests."""
        with patch.dict(os.environ, {"CONNECTOMICS_MCP_ARTIFACT_DIR": str(tmp_path)}):
            self._tmp = tmp_path
            yield

    def test_cave_normal_response(self, mock_cave_backend):
        result = get_region_connectivity("minnie65")

        resp = RegionConnectivityResponse(**result)
        assert resp.dataset == "minnie65"
        assert resp.n_regions > 0
        assert resp.total_synapses > 0
        assert len(resp.top_5_connections) > 0
        assert resp.warnings == []

    def test_neuprint_normal_response(self, mock_neuprint_backend):
        result = get_region_connectivity("hemibrain")

        resp = RegionConnectivityResponse(**result)
        assert resp.dataset == "hemibrain"
        assert resp.n_regions > 0
        assert resp.total_synapses > 0

    def test_artifact_written_and_readable(self, mock_cave_backend):
        result = get_region_connectivity("minnie65")

        manifest = result["artifact_manifest"]
        assert manifest is not None
        path = Path(manifest["artifact_path"])
        assert path.exists()

        df = pd.read_parquet(path)
        assert len(df) == manifest["n_rows"]
        for col in ["source_region", "target_region", "n_synapses",
                     "n_neurons_pre", "n_neurons_post"]:
            assert col in df.columns

    def test_top_5_connections_sorted_desc(self, mock_cave_backend):
        result = get_region_connectivity("minnie65")
        top_5 = result["top_5_connections"]
        assert len(top_5) <= 5
        synapse_counts = [c["n_synapses"] for c in top_5]
        assert synapse_counts == sorted(synapse_counts, reverse=True)

    def test_n_regions_matches_artifact(self, mock_cave_backend):
        result = get_region_connectivity("minnie65")
        manifest = result["artifact_manifest"]
        df = pd.read_parquet(manifest["artifact_path"])
        unique_regions = set(df["source_region"]) | set(df["target_region"])
        assert result["n_regions"] == len(unique_regions)

    def test_total_synapses_matches_artifact(self, mock_cave_backend):
        result = get_region_connectivity("minnie65")
        manifest = result["artifact_manifest"]
        df = pd.read_parquet(manifest["artifact_path"])
        assert result["total_synapses"] == int(df["n_synapses"].sum())

    def test_source_region_filter(self, mock_cave_backend):
        v1_result = get_region_connectivity("minnie65", source_region="V1")

        # V1 as source has 4 rows in mock data (V1→V1, V1→LM, V1→AL, V1→RL)
        assert v1_result["artifact_manifest"]["n_rows"] == 4
        df = pd.read_parquet(v1_result["artifact_manifest"]["artifact_path"])
        assert all(df["source_region"] == "V1")
        assert v1_result["total_synapses"] < 120500  # less than full total

    def test_target_region_filter(self, mock_cave_backend):
        lm_result = get_region_connectivity("minnie65", target_region="LM")

        # LM as target has 4 rows in mock data (V1→LM, LM→LM, AL→LM, RL→LM)
        assert lm_result["artifact_manifest"]["n_rows"] == 4
        df = pd.read_parquet(lm_result["artifact_manifest"]["artifact_path"])
        assert all(df["target_region"] == "LM")

    def test_unsupported_dataset_raises(self):
        with pytest.raises(DatasetNotSupported):
            get_region_connectivity("nonexistent_dataset")

    def test_response_validates_against_schema(self, mock_cave_backend):
        result = get_region_connectivity("minnie65")
        resp = RegionConnectivityResponse.model_validate(result)
        json_str = resp.model_dump_json()
        assert isinstance(json_str, str)
