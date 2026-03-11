"""Tests for CAVE-specific tools (Tier 2)."""

from __future__ import annotations

import pandas as pd
import pytest

from connectomics_mcp.exceptions import DatasetNotSupported, StaleRootIdError
from connectomics_mcp.output_contracts.schemas import (
    AnnotationTableResponse,
    EditHistoryResponse,
    NucleusResolutionResult,
    NucleusResolutionStatus,
    ProofreadingStatusResponse,
)
from connectomics_mcp.tools.cave_specific import (
    get_edit_history,
    get_proofreading_status,
    query_annotation_table,
    resolve_nucleus_ids,
)


class TestGetProofreadingStatus:
    def test_normal_proofread_response(self, mock_cave_backend):
        result = get_proofreading_status(720575940621039145, "minnie65")

        resp = ProofreadingStatusResponse(**result)
        assert resp.neuron_id == 720575940621039145
        assert resp.dataset == "minnie65"
        assert resp.axon_proofread is True
        assert resp.dendrite_proofread is True
        assert resp.strategy_axon == "axon_fully_extended"
        assert resp.strategy_dendrite == "dendrite_fully_extended"
        assert resp.n_edits == 42
        assert resp.last_edit_timestamp == "2026-02-20T14:15:00"
        assert resp.warnings == []

    def test_stale_root_id_raises(self, mock_cave_backend_stale):
        with pytest.raises(StaleRootIdError) as exc_info:
            get_proofreading_status(720575940621039145, "minnie65")
        assert "720575940621039145" in str(exc_info.value)
        assert "validate_root_ids" in str(exc_info.value)

    def test_neuprint_dataset_raises(self, mock_neuprint_backend):
        with pytest.raises(DatasetNotSupported):
            get_proofreading_status(12345, "hemibrain")

    def test_unsupported_dataset_raises(self):
        with pytest.raises(DatasetNotSupported):
            get_proofreading_status(123, "nonexistent_dataset")

    def test_response_validates_against_schema(self, mock_cave_backend):
        result = get_proofreading_status(720575940621039145, "minnie65")
        resp = ProofreadingStatusResponse.model_validate(result)
        json_str = resp.model_dump_json()
        assert isinstance(json_str, str)


class TestResolveNucleusIds:
    def test_resolved_nucleus(self, mock_cave_backend):
        result = resolve_nucleus_ids([100001], "minnie65")

        resp = NucleusResolutionResult(**result)
        assert resp.dataset == "minnie65"
        assert resp.materialization_version == 943
        assert resp.n_resolved == 1
        assert resp.n_merge_conflicts == 0
        assert resp.n_no_segment == 0
        assert len(resp.resolutions) == 1

        res = resp.resolutions[0]
        assert res.nucleus_id == 100001
        assert res.pt_root_id == 864691135000001
        assert res.resolution_status == NucleusResolutionStatus.RESOLVED
        assert res.conflicting_nucleus_ids == []

    def test_merge_conflict(self, mock_cave_backend):
        result = resolve_nucleus_ids([100002, 100003], "minnie65")

        resp = NucleusResolutionResult(**result)
        assert resp.n_resolved == 0
        assert resp.n_merge_conflicts == 2
        assert resp.n_no_segment == 0

        for res in resp.resolutions:
            assert res.resolution_status == NucleusResolutionStatus.MERGE_CONFLICT
            assert res.pt_root_id == 864691135000099
            assert len(res.conflicting_nucleus_ids) == 1

        # Check cross-references
        assert resp.resolutions[0].conflicting_nucleus_ids == [100003]
        assert resp.resolutions[1].conflicting_nucleus_ids == [100002]

    def test_no_segment(self, mock_cave_backend):
        result = resolve_nucleus_ids([999999], "minnie65")

        resp = NucleusResolutionResult(**result)
        assert resp.n_resolved == 0
        assert resp.n_merge_conflicts == 0
        assert resp.n_no_segment == 1

        res = resp.resolutions[0]
        assert res.nucleus_id == 999999
        assert res.pt_root_id is None
        assert res.resolution_status == NucleusResolutionStatus.NO_SEGMENT

    def test_mixed_statuses(self, mock_cave_backend):
        result = resolve_nucleus_ids([100001, 100002, 100003, 999999], "minnie65")

        resp = NucleusResolutionResult(**result)
        assert resp.n_resolved == 1
        assert resp.n_merge_conflicts == 2
        assert resp.n_no_segment == 1
        assert len(resp.resolutions) == 4

    def test_minnie65_only(self, mock_cave_backend):
        """flywire and fanc should raise DatasetNotSupported."""
        with pytest.raises(DatasetNotSupported):
            resolve_nucleus_ids([100001], "flywire")

    def test_hemibrain_raises(self, mock_neuprint_backend):
        with pytest.raises(DatasetNotSupported):
            resolve_nucleus_ids([100001], "hemibrain")

    def test_response_validates_against_schema(self, mock_cave_backend):
        result = resolve_nucleus_ids([100001, 100002, 100003, 999999], "minnie65")
        resp = NucleusResolutionResult.model_validate(result)
        json_str = resp.model_dump_json()
        assert isinstance(json_str, str)


class TestQueryAnnotationTable:
    def test_normal_response(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = query_annotation_table("minnie65", "aibs_metamodel_celltypes_v661")

        resp = AnnotationTableResponse(**result)
        assert resp.dataset == "minnie65"
        assert resp.table_name == "aibs_metamodel_celltypes_v661"
        assert resp.n_total == 5
        assert resp.schema_description != ""
        assert resp.warnings == []

    def test_artifact_written_and_readable(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = query_annotation_table("minnie65", "aibs_metamodel_celltypes_v661")

        resp = AnnotationTableResponse(**result)
        artifact_path = resp.artifact_manifest.artifact_path
        df = pd.read_parquet(artifact_path)
        assert len(df) == resp.n_total
        assert "id" in df.columns
        assert "pt_root_id" in df.columns
        assert "cell_type" in df.columns

    def test_n_total_matches_artifact(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = query_annotation_table("minnie65", "test_table")

        resp = AnnotationTableResponse(**result)
        df = pd.read_parquet(resp.artifact_manifest.artifact_path)
        assert resp.n_total == len(df)
        assert resp.artifact_manifest.n_rows == len(df)

    def test_filter_equal_dict(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = query_annotation_table(
            "minnie65", "test_table",
            filter_equal_dict={"cell_type": "L2/3 IT"},
        )

        resp = AnnotationTableResponse(**result)
        assert resp.n_total == 3  # 3 L2/3 IT rows in mock data

    def test_filter_in_dict(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = query_annotation_table(
            "minnie65", "test_table",
            filter_in_dict={"tag": ["V1"]},
        )

        resp = AnnotationTableResponse(**result)
        assert resp.n_total == 3  # 3 rows with tag=V1

    def test_neuprint_raises(self, mock_neuprint_backend):
        with pytest.raises(DatasetNotSupported):
            query_annotation_table("hemibrain", "some_table")

    def test_response_validates_against_schema(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = query_annotation_table("minnie65", "test_table")
        resp = AnnotationTableResponse.model_validate(result)
        json_str = resp.model_dump_json()
        assert isinstance(json_str, str)


class TestGetEditHistory:
    def test_normal_response(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_edit_history(720575940621039145, "minnie65")

        resp = EditHistoryResponse(**result)
        assert resp.neuron_id == 720575940621039145
        assert resp.dataset == "minnie65"
        assert resp.n_edits_total == 6
        assert resp.first_edit_timestamp is not None
        assert resp.last_edit_timestamp is not None
        assert resp.warnings == []

    def test_artifact_written_and_readable(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_edit_history(720575940621039145, "minnie65")

        resp = EditHistoryResponse(**result)
        artifact_path = resp.artifact_manifest.artifact_path
        df = pd.read_parquet(artifact_path)
        assert len(df) == resp.n_edits_total
        expected_cols = {"operation_id", "timestamp", "operation_type", "user_id"}
        assert expected_cols == set(df.columns)

    def test_n_edits_matches_artifact(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_edit_history(720575940621039145, "minnie65")

        resp = EditHistoryResponse(**result)
        df = pd.read_parquet(resp.artifact_manifest.artifact_path)
        assert resp.n_edits_total == len(df)
        assert resp.artifact_manifest.n_rows == len(df)

    def test_timestamps_match_artifact(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_edit_history(720575940621039145, "minnie65")

        resp = EditHistoryResponse(**result)
        df = pd.read_parquet(resp.artifact_manifest.artifact_path)
        assert resp.first_edit_timestamp == str(df["timestamp"].min())
        assert resp.last_edit_timestamp == str(df["timestamp"].max())

    def test_operation_types_valid(self, mock_cave_backend, tmp_path, monkeypatch):
        monkeypatch.setenv("CONNECTOMICS_MCP_ARTIFACT_DIR", str(tmp_path))
        result = get_edit_history(720575940621039145, "minnie65")

        resp = EditHistoryResponse(**result)
        df = pd.read_parquet(resp.artifact_manifest.artifact_path)
        valid_types = {"merge", "split"}
        assert set(df["operation_type"].unique()).issubset(valid_types)

    def test_stale_root_id_raises(self, mock_cave_backend_stale):
        with pytest.raises(StaleRootIdError) as exc_info:
            get_edit_history(720575940621039145, "minnie65")
        assert "720575940621039145" in str(exc_info.value)

    def test_neuprint_raises(self, mock_neuprint_backend):
        with pytest.raises(DatasetNotSupported):
            get_edit_history(12345, "hemibrain")
