"""Tests for CAVE-specific tools (Tier 2)."""

from __future__ import annotations

import pytest

from connectomics_mcp.exceptions import DatasetNotSupported, StaleRootIdError
from connectomics_mcp.output_contracts.schemas import ProofreadingStatusResponse
from connectomics_mcp.tools.cave_specific import get_proofreading_status


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
