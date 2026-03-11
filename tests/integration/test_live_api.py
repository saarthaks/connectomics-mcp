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

    NUCLEUS_ID = 271171

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
    """Live tests against the FlyWire FAFB production datastack."""

    ROOT_ID = 720575940621039145  # DA1 PN

    @pytest.fixture(autouse=True)
    def _setup(self, artifact_dir: Path) -> None:
        _clear_backend_cache()
        self.artifact_dir = artifact_dir

    def test_validate_and_get_neuron_info(self) -> None:
        # Step 1: validate root ID
        val_result = universal.validate_root_ids([self.ROOT_ID], "flywire")
        val_resp = RootIdValidationResponse(**val_result)

        assert val_resp.dataset == "flywire"
        assert len(val_resp.results) == 1
        validation = val_resp.results[0]
        print(f"\n  root_id {self.ROOT_ID} is_current={validation.is_current}")

        active_id = self.ROOT_ID
        if not validation.is_current and validation.suggested_current_id:
            active_id = validation.suggested_current_id
            print(f"  -> superseded, using suggested_current_id={active_id}")

        # Step 2: get neuron info
        info_result = universal.get_neuron_info(active_id, "flywire")
        info_resp = NeuronInfoResponse(**info_result)

        assert info_resp.neuron_id == active_id
        assert info_resp.dataset == "flywire"
        # Cell type may be None if not annotated, but response must be valid
        print(f"  neuron_info: type={info_resp.cell_type}, class={info_resp.cell_class}")
        print(f"  pre={info_resp.n_pre_synapses}, post={info_resp.n_post_synapses}")
        print(f"  warnings={info_resp.warnings}")


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
