"""Live-API integration tests — one per backend.

Run with:  pytest tests/integration/ --integration -v -s

Requires env vars: CAVE_CLIENT_TOKEN, NEUPRINT_APPLICATION_CREDENTIALS
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from connectomics_mcp.output_contracts.schemas import (
    ConnectivityResponse,
    NeuronInfoResponse,
    NucleusResolutionResult,
    RootIdValidationResponse,
    NeuronsByTypeResponse,
)
from connectomics_mcp.registry import _backend_cache
from connectomics_mcp.tools import cave_specific, universal

# All tests require live credentials and the --integration flag.
pytestmark = pytest.mark.integration


# ── helpers ──────────────────────────────────────────────────────────


def _clear_backend_cache() -> None:
    """Ensure we get fresh real backends, not leftover mocks."""
    _backend_cache.clear()


# ── minnie65 (CAVE) ─────────────────────────────────────────────────


class TestMinnie65:
    """Live tests against the MICrONS minnie65_public datastack."""

    NUCLEUS_ID = 264824  # resolved → 864691135571546917

    @pytest.fixture(autouse=True)
    def _setup(self, artifact_dir: Path) -> None:
        _clear_backend_cache()
        self.artifact_dir = artifact_dir

    def test_resolve_nucleus_and_get_connectivity(self) -> None:
        # Step 1: resolve nucleus ID to current root ID
        res_result = cave_specific.resolve_nucleus_ids(
            [self.NUCLEUS_ID], "minnie65"
        )
        res_resp = NucleusResolutionResult(**res_result)

        assert res_resp.dataset == "minnie65"
        assert len(res_resp.resolutions) == 1
        resolution = res_resp.resolutions[0]
        assert resolution.nucleus_id == self.NUCLEUS_ID
        print(f"\n  nucleus {self.NUCLEUS_ID} -> status={resolution.resolution_status}, pt_root_id={resolution.pt_root_id}")

        # If no_segment, the nucleus doesn't map — skip connectivity
        if resolution.resolution_status.value == "no_segment":
            pytest.skip(f"Nucleus {self.NUCLEUS_ID} has no segment; skipping connectivity")

        root_id = resolution.pt_root_id
        assert root_id is not None

        # Step 2: validate root ID is current
        val_result = universal.validate_root_ids([root_id], "minnie65")
        val_resp = RootIdValidationResponse(**val_result)
        validation = val_resp.results[0]
        print(f"  root_id {root_id} is_current={validation.is_current}")

        # If stale, use the suggested replacement
        if not validation.is_current and validation.suggested_current_id:
            root_id = validation.suggested_current_id
            print(f"  -> superseded, using suggested_current_id={root_id}")

        # Step 3: get connectivity
        conn_result = universal.get_connectivity(root_id, "minnie65")
        conn_resp = ConnectivityResponse(**conn_result)

        assert conn_resp.neuron_id == root_id
        assert conn_resp.dataset == "minnie65"
        assert conn_resp.n_upstream_total + conn_resp.n_downstream_total > 0

        # Artifact assertions
        manifest = conn_resp.artifact_manifest
        assert manifest is not None
        artifact_path = Path(manifest.artifact_path)
        assert artifact_path.exists(), f"Artifact not found: {artifact_path}"
        df = pd.read_parquet(artifact_path)
        assert len(df) == manifest.n_rows
        assert len(df) > 0
        assert {"partner_id", "direction", "n_synapses"}.issubset(set(df.columns))

        print(f"  connectivity: {conn_resp.n_upstream_total} upstream, {conn_resp.n_downstream_total} downstream")
        print(f"  artifact: {artifact_path}  ({manifest.n_rows} rows)")


# ── flywire (CAVE) ──────────────────────────────────────────────────


class TestFlyWire:
    """Live tests against the FlyWire FAFB production datastack.

    Note: FlyWire requires dataset-level access permissions.
    If the token lacks ``view`` permission for ``fafb``, the test
    is skipped rather than failed.
    """

    ROOT_ID = 720575940621039145  # DA1 PN

    @pytest.fixture(autouse=True)
    def _setup(self, artifact_dir: Path) -> None:
        _clear_backend_cache()
        self.artifact_dir = artifact_dir

    def _get_active_id(self):
        """Validate root ID and return the active (possibly superseded) ID."""
        val_result = universal.validate_root_ids([self.ROOT_ID], "flywire")
        val_resp = RootIdValidationResponse(**val_result)

        assert val_resp.dataset == "flywire"
        assert len(val_resp.results) == 1
        validation = val_resp.results[0]

        # If we got a permission error, skip
        for w in val_resp.warnings:
            if "missing_permission" in w or "FORBIDDEN" in w:
                pytest.skip(f"FlyWire access denied: {w}")

        active_id = self.ROOT_ID
        if not validation.is_current and validation.suggested_current_id:
            active_id = validation.suggested_current_id
            print(f"\n  root_id {self.ROOT_ID} superseded -> {active_id}")
        else:
            print(f"\n  root_id {self.ROOT_ID} is_current={validation.is_current}")

        return active_id

    def test_validate_and_get_neuron_info(self) -> None:
        active_id = self._get_active_id()

        info_result = universal.get_neuron_info(active_id, "flywire")
        info_resp = NeuronInfoResponse(**info_result)

        assert info_resp.neuron_id == active_id
        assert info_resp.dataset == "flywire"

        # FlyWire-specific enrichment: hierarchy + NT
        print(f"  cell_type={info_resp.cell_type}")
        print(f"  classification_hierarchy={info_resp.classification_hierarchy}")
        print(f"  neurotransmitter_type={info_resp.neurotransmitter_type}")
        print(f"  pre={info_resp.n_pre_synapses}, post={info_resp.n_post_synapses}")

        # Synapse counts should now be populated (fixed select_columns bug)
        assert info_resp.n_pre_synapses is not None, "n_pre_synapses should not be None"
        assert info_resp.n_post_synapses is not None, "n_post_synapses should not be None"

        # Hierarchy should be populated from hierarchical_neuron_annotations
        assert info_resp.classification_hierarchy is not None, (
            "classification_hierarchy should be populated for FlyWire"
        )

    def test_connectivity_with_nt(self) -> None:
        active_id = self._get_active_id()

        conn_result = universal.get_connectivity(active_id, "flywire")
        conn_resp = ConnectivityResponse(**conn_result)

        assert conn_resp.neuron_id == active_id
        assert conn_resp.dataset == "flywire"
        assert conn_resp.n_upstream_total + conn_resp.n_downstream_total > 0

        # Artifact assertions
        manifest = conn_resp.artifact_manifest
        assert manifest is not None
        artifact_path = Path(manifest.artifact_path)
        assert artifact_path.exists()
        df = pd.read_parquet(artifact_path)
        assert len(df) == manifest.n_rows
        assert len(df) > 0

        # NT enrichment columns should be present
        assert "partner_nt_type" in df.columns, (
            "FlyWire connectivity artifact should have partner_nt_type column"
        )
        assert df["partner_nt_type"].notna().any(), (
            "At least some partners should have NT predictions"
        )

        print(f"  connectivity: {conn_resp.n_upstream_total} up, {conn_resp.n_downstream_total} down")
        print(f"  artifact: {artifact_path} ({manifest.n_rows} rows)")
        print(f"  nt_distribution: {conn_resp.neurotransmitter_distribution}")


# ── hemibrain (neuPrint) ─────────────────────────────────────────────


class TestHemibrain:
    """Live tests against neuPrint hemibrain:v1.2.1."""

    BODY_ID = 5813105172  # DA1 adPN

    @pytest.fixture(autouse=True)
    def _setup(self, artifact_dir: Path) -> None:
        _clear_backend_cache()
        self.artifact_dir = artifact_dir

    def test_get_connectivity(self) -> None:
        # neuPrint body IDs are immutable — no staleness check needed
        conn_result = universal.get_connectivity(self.BODY_ID, "hemibrain")
        conn_resp = ConnectivityResponse(**conn_result)

        assert conn_resp.neuron_id == self.BODY_ID
        assert conn_resp.dataset == "hemibrain"
        assert conn_resp.n_upstream_total + conn_resp.n_downstream_total > 0

        # Artifact assertions
        manifest = conn_resp.artifact_manifest
        assert manifest is not None
        artifact_path = Path(manifest.artifact_path)
        assert artifact_path.exists(), f"Artifact not found: {artifact_path}"
        df = pd.read_parquet(artifact_path)
        assert len(df) == manifest.n_rows
        assert len(df) > 0
        assert {"partner_id", "direction", "n_synapses"}.issubset(set(df.columns))

        print(f"\n  connectivity: {conn_resp.n_upstream_total} upstream, {conn_resp.n_downstream_total} downstream")
        print(f"  artifact: {artifact_path}  ({manifest.n_rows} rows)")
