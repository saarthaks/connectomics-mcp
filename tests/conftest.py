"""Shared test fixtures with mock backends."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from connectomics_mcp.backends.base import ConnectomeBackend


def _make_cave_connectivity_df(is_current: bool = True) -> pd.DataFrame:
    """Build a realistic mock connectivity DataFrame for CAVE."""
    rows = []
    # 5 upstream partners
    for i in range(5):
        rows.append({
            "partner_id": 864691135000000 + i,
            "direction": "upstream",
            "partner_type": f"L{i + 1} IT" if i < 3 else None,
            "partner_class": "excitatory" if i < 3 else None,
            "n_synapses": 50 - i * 8,
            "weight_normalized": (50 - i * 8) / 170.0,
            "partner_region": None,
            "neuroglancer_url": "",
        })
    # 4 downstream partners
    for i in range(4):
        rows.append({
            "partner_id": 864691135100000 + i,
            "direction": "downstream",
            "partner_type": f"L{i + 2}/3 IT" if i < 2 else None,
            "partner_class": "excitatory" if i < 2 else None,
            "n_synapses": 40 - i * 7,
            "weight_normalized": (40 - i * 7) / 106.0,
            "partner_region": None,
            "neuroglancer_url": "",
        })
    return pd.DataFrame(rows)


def _make_neuprint_connectivity_df() -> pd.DataFrame:
    """Build a realistic mock connectivity DataFrame for neuPrint."""
    rows = []
    # 6 upstream
    for i in range(6):
        rows.append({
            "partner_id": 200000 + i,
            "direction": "upstream",
            "partner_type": f"KC-ab" if i < 4 else "MBIN",
            "partner_class": None,
            "n_synapses": 30 - i * 4,
            "weight_normalized": (30 - i * 4) / 108.0,
            "partner_region": None,
            "neuroglancer_url": "",
        })
    # 3 downstream
    for i in range(3):
        rows.append({
            "partner_id": 300000 + i,
            "direction": "downstream",
            "partner_type": "DAN" if i == 0 else None,
            "partner_class": None,
            "n_synapses": 25 - i * 10,
            "weight_normalized": (25 - i * 10) / 40.0,
            "partner_region": None,
            "neuroglancer_url": "",
        })
    return pd.DataFrame(rows)


def _make_cave_neurons_by_type_df(
    cell_type: str, region: str | None = None
) -> pd.DataFrame:
    """Build a realistic mock neurons-by-type DataFrame for CAVE."""
    rows = [
        {"neuron_id": 864691135000100, "cell_type": "L2/3 IT", "cell_class": "excitatory", "region": "V1", "n_pre_synapses": 1500, "n_post_synapses": 3200, "proofread": True},
        {"neuron_id": 864691135000101, "cell_type": "L2/3 IT", "cell_class": "excitatory", "region": "V1", "n_pre_synapses": 1200, "n_post_synapses": 2800, "proofread": True},
        {"neuron_id": 864691135000102, "cell_type": "L2/3 IT", "cell_class": "excitatory", "region": "LM", "n_pre_synapses": 1400, "n_post_synapses": 3100, "proofread": False},
        {"neuron_id": 864691135000103, "cell_type": "L2/3 IT", "cell_class": "excitatory", "region": "LM", "n_pre_synapses": 1100, "n_post_synapses": 2500, "proofread": False},
        {"neuron_id": 864691135000104, "cell_type": "L2/3 IT", "cell_class": "excitatory", "region": "V1", "n_pre_synapses": 1600, "n_post_synapses": 3400, "proofread": True},
        {"neuron_id": 864691135000105, "cell_type": "L4 IT", "cell_class": "excitatory", "region": "V1", "n_pre_synapses": 900, "n_post_synapses": 2100, "proofread": True},
        {"neuron_id": 864691135000106, "cell_type": "L4 IT", "cell_class": "excitatory", "region": "LM", "n_pre_synapses": 800, "n_post_synapses": 1900, "proofread": False},
        {"neuron_id": 864691135000107, "cell_type": "L2/3 IT", "cell_class": "excitatory", "region": "AL", "n_pre_synapses": 1300, "n_post_synapses": 2900, "proofread": True},
    ]
    df = pd.DataFrame(rows)
    # Filter to matching cell type (mock queries matching the requested type)
    df = df[df["cell_type"].str.contains(cell_type, case=False, na=False)]
    if region:
        df = df[df["region"].str.contains(region, case=False, na=False)]
    return df.reset_index(drop=True)


def _make_neuprint_neurons_by_type_df(
    cell_type: str, region: str | None = None
) -> pd.DataFrame:
    """Build a realistic mock neurons-by-type DataFrame for neuPrint."""
    rows = [
        {"neuron_id": 500001, "cell_type": "KC-ab", "cell_class": "KC-ab(1)", "region": "MB(+)", "n_pre_synapses": 200, "n_post_synapses": 800, "proofread": None},
        {"neuron_id": 500002, "cell_type": "KC-ab", "cell_class": "KC-ab(2)", "region": "MB(+)", "n_pre_synapses": 180, "n_post_synapses": 750, "proofread": None},
        {"neuron_id": 500003, "cell_type": "KC-ab", "cell_class": "KC-ab(3)", "region": "MB(+)", "n_pre_synapses": 210, "n_post_synapses": 820, "proofread": None},
        {"neuron_id": 500004, "cell_type": "KC-ab", "cell_class": "KC-ab(4)", "region": "CA(R)", "n_pre_synapses": 190, "n_post_synapses": 770, "proofread": None},
        {"neuron_id": 500005, "cell_type": "KC-ab", "cell_class": "KC-ab(5)", "region": "CA(R)", "n_pre_synapses": 220, "n_post_synapses": 850, "proofread": None},
        {"neuron_id": 500006, "cell_type": "KC-ab", "cell_class": "KC-ab(6)", "region": "MB(+)", "n_pre_synapses": 170, "n_post_synapses": 710, "proofread": None},
    ]
    df = pd.DataFrame(rows)
    if region:
        df = df[df["region"].str.contains(region, case=False, na=False)]
    return df.reset_index(drop=True)


def _make_cave_region_connectivity_df(
    source_region: str | None = None, target_region: str | None = None
) -> pd.DataFrame:
    """Build a realistic mock region connectivity DataFrame for CAVE."""
    rows = [
        {"source_region": "V1", "target_region": "V1", "n_synapses": 50000, "n_neurons_pre": 1200, "n_neurons_post": 1300},
        {"source_region": "V1", "target_region": "LM", "n_synapses": 12000, "n_neurons_pre": 800, "n_neurons_post": 400},
        {"source_region": "V1", "target_region": "AL", "n_synapses": 8000, "n_neurons_pre": 600, "n_neurons_post": 300},
        {"source_region": "LM", "target_region": "V1", "n_synapses": 9500, "n_neurons_pre": 350, "n_neurons_post": 900},
        {"source_region": "LM", "target_region": "LM", "n_synapses": 15000, "n_neurons_pre": 400, "n_neurons_post": 420},
        {"source_region": "LM", "target_region": "AL", "n_synapses": 3000, "n_neurons_pre": 200, "n_neurons_post": 150},
        {"source_region": "AL", "target_region": "V1", "n_synapses": 4000, "n_neurons_pre": 180, "n_neurons_post": 500},
        {"source_region": "AL", "target_region": "LM", "n_synapses": 2500, "n_neurons_pre": 150, "n_neurons_post": 250},
        {"source_region": "AL", "target_region": "AL", "n_synapses": 6000, "n_neurons_pre": 280, "n_neurons_post": 290},
        {"source_region": "V1", "target_region": "RL", "n_synapses": 5500, "n_neurons_pre": 450, "n_neurons_post": 200},
        {"source_region": "RL", "target_region": "V1", "n_synapses": 3200, "n_neurons_pre": 180, "n_neurons_post": 600},
        {"source_region": "RL", "target_region": "LM", "n_synapses": 1800, "n_neurons_pre": 120, "n_neurons_post": 180},
    ]
    df = pd.DataFrame(rows)
    if source_region:
        df = df[df["source_region"].str.contains(source_region, case=False, na=False)]
    if target_region:
        df = df[df["target_region"].str.contains(target_region, case=False, na=False)]
    return df.reset_index(drop=True)


def _make_neuprint_region_connectivity_df(
    source_region: str | None = None, target_region: str | None = None
) -> pd.DataFrame:
    """Build a realistic mock region connectivity DataFrame for neuPrint."""
    rows = [
        {"source_region": "MB(+)", "target_region": "MB(+)", "n_synapses": 80000, "n_neurons_pre": 2000, "n_neurons_post": 2100},
        {"source_region": "MB(+)", "target_region": "CA(R)", "n_synapses": 25000, "n_neurons_pre": 1500, "n_neurons_post": 800},
        {"source_region": "CA(R)", "target_region": "MB(+)", "n_synapses": 18000, "n_neurons_pre": 700, "n_neurons_post": 1800},
        {"source_region": "CA(R)", "target_region": "CA(R)", "n_synapses": 12000, "n_neurons_pre": 600, "n_neurons_post": 650},
        {"source_region": "MB(+)", "target_region": "LH(R)", "n_synapses": 9000, "n_neurons_pre": 900, "n_neurons_post": 400},
        {"source_region": "LH(R)", "target_region": "MB(+)", "n_synapses": 7000, "n_neurons_pre": 350, "n_neurons_post": 1000},
        {"source_region": "LH(R)", "target_region": "LH(R)", "n_synapses": 15000, "n_neurons_pre": 500, "n_neurons_post": 520},
        {"source_region": "LH(R)", "target_region": "CA(R)", "n_synapses": 4500, "n_neurons_pre": 250, "n_neurons_post": 300},
    ]
    df = pd.DataFrame(rows)
    if source_region:
        df = df[df["source_region"].str.contains(source_region, case=False, na=False)]
    if target_region:
        df = df[df["target_region"].str.contains(target_region, case=False, na=False)]
    return df.reset_index(drop=True)


class MockCAVEBackend(ConnectomeBackend):
    """Mock CAVE backend for testing."""

    def __init__(self, is_current: bool = True) -> None:
        self.is_current = is_current
        self.dataset_name = "minnie65"

    def get_neuron_info(self, neuron_id: int | str) -> dict[str, Any]:
        root_id = int(neuron_id)
        warnings = []
        if not self.is_current:
            warnings.append(
                f"Root ID {root_id} is outdated. "
                f"Use `validate_root_ids()` to get current IDs."
            )
        return {
            "neuron_id": root_id,
            "dataset": "minnie65",
            "cell_type": "L2/3 IT",
            "cell_class": "excitatory",
            "soma_position_nm": (200000.0, 300000.0, 400000.0),
            "n_pre_synapses": 1500,
            "n_post_synapses": 3200,
            "is_current": self.is_current,
            "materialization_version": 943,
            "warnings": warnings,
        }

    def get_connectivity(
        self, neuron_id: int | str, direction: str = "both"
    ) -> dict[str, Any]:
        root_id = int(neuron_id)
        warnings = []
        if not self.is_current:
            warnings.append(
                f"Root ID {root_id} is outdated. "
                f"Use `validate_root_ids()` to get current IDs."
            )
        df = _make_cave_connectivity_df(self.is_current)
        if direction == "upstream":
            df = df[df["direction"] == "upstream"]
        elif direction == "downstream":
            df = df[df["direction"] == "downstream"]
        return {
            "neuron_id": root_id,
            "dataset": "minnie65",
            "is_current": self.is_current,
            "materialization_version": 943,
            "warnings": warnings,
            "partners_df": df,
        }

    def validate_root_ids(self, root_ids: list[int]) -> dict[str, Any]:
        results = []
        for rid in root_ids:
            is_current = self.is_current
            result: dict[str, Any] = {
                "root_id": int(rid),
                "is_current": is_current,
                "last_edit_timestamp": None,
                "suggested_current_id": None,
            }
            if not is_current:
                result["suggested_current_id"] = int(rid) + 1
                result["last_edit_timestamp"] = "2026-01-15T10:30:00"
            results.append(result)
        return {
            "dataset": "minnie65",
            "materialization_version": 943,
            "results": results,
            "warnings": [],
        }

    def get_proofreading_status(self, neuron_id: int) -> dict[str, Any]:
        root_id = int(neuron_id)
        warnings: list[str] = []
        if not self.is_current:
            warnings.append(
                f"Root ID {root_id} is outdated. "
                f"Use `validate_root_ids()` to get current IDs."
            )
        return {
            "neuron_id": root_id,
            "dataset": "minnie65",
            "is_current": self.is_current,
            "axon_proofread": True,
            "dendrite_proofread": True,
            "strategy_axon": "axon_fully_extended",
            "strategy_dendrite": "dendrite_fully_extended",
            "n_edits": 42,
            "last_edit_timestamp": "2026-02-20T14:15:00",
            "warnings": warnings,
        }

    def get_region_connectivity(
        self,
        source_region: str | None = None,
        target_region: str | None = None,
    ) -> dict[str, Any]:
        region_df = _make_cave_region_connectivity_df(source_region, target_region)
        return {
            "dataset": "minnie65",
            "materialization_version": 943,
            "warnings": [],
            "region_df": region_df,
        }

    def get_neurons_by_type(
        self, cell_type: str, region: str | None = None
    ) -> dict[str, Any]:
        neurons_df = _make_cave_neurons_by_type_df(cell_type, region)
        return {
            "dataset": "minnie65",
            "query_cell_type": cell_type,
            "query_region": region,
            "materialization_version": 943,
            "warnings": [],
            "neurons_df": neurons_df,
        }


class MockNeuPrintBackend(ConnectomeBackend):
    """Mock neuPrint backend for testing."""

    def __init__(self) -> None:
        self.dataset_name = "hemibrain"

    def get_neuron_info(self, neuron_id: int | str) -> dict[str, Any]:
        body_id = int(neuron_id)
        return {
            "neuron_id": body_id,
            "dataset": "hemibrain",
            "cell_type": "MBON14",
            "cell_class": "MBON14(a3)",
            "region": "MB(+)",
            "soma_position_nm": (15000.0, 20000.0, 18000.0),
            "n_pre_synapses": 800,
            "n_post_synapses": 2100,
            "warnings": [],
        }

    def get_connectivity(
        self, neuron_id: int | str, direction: str = "both"
    ) -> dict[str, Any]:
        body_id = int(neuron_id)
        df = _make_neuprint_connectivity_df()
        if direction == "upstream":
            df = df[df["direction"] == "upstream"]
        elif direction == "downstream":
            df = df[df["direction"] == "downstream"]
        return {
            "neuron_id": body_id,
            "dataset": "hemibrain",
            "warnings": [],
            "partners_df": df,
        }

    def validate_root_ids(self, root_ids: list[int]) -> dict[str, Any]:
        results = [
            {
                "root_id": int(rid),
                "is_current": True,
                "last_edit_timestamp": None,
                "suggested_current_id": None,
            }
            for rid in root_ids
        ]
        return {
            "dataset": "hemibrain",
            "materialization_version": None,
            "results": results,
            "warnings": [],
        }

    def get_proofreading_status(self, neuron_id: int) -> dict[str, Any]:
        from connectomics_mcp.exceptions import DatasetNotSupported

        raise DatasetNotSupported("hemibrain", "cave")

    def get_region_connectivity(
        self,
        source_region: str | None = None,
        target_region: str | None = None,
    ) -> dict[str, Any]:
        region_df = _make_neuprint_region_connectivity_df(source_region, target_region)
        return {
            "dataset": "hemibrain",
            "materialization_version": None,
            "warnings": [],
            "region_df": region_df,
        }

    def get_neurons_by_type(
        self, cell_type: str, region: str | None = None
    ) -> dict[str, Any]:
        neurons_df = _make_neuprint_neurons_by_type_df(cell_type, region)
        return {
            "dataset": "hemibrain",
            "query_cell_type": cell_type,
            "query_region": region,
            "materialization_version": None,
            "warnings": [],
            "neurons_df": neurons_df,
        }


@pytest.fixture
def mock_cave_backend():
    """Provide a MockCAVEBackend and patch registry to return it."""
    backend = MockCAVEBackend(is_current=True)
    with patch("connectomics_mcp.registry._backend_cache", {"minnie65": backend}):
        yield backend


@pytest.fixture
def mock_cave_backend_stale():
    """Provide a MockCAVEBackend that reports stale root IDs."""
    backend = MockCAVEBackend(is_current=False)
    with patch("connectomics_mcp.registry._backend_cache", {"minnie65": backend}):
        yield backend


@pytest.fixture
def mock_neuprint_backend():
    """Provide a MockNeuPrintBackend and patch registry to return it."""
    backend = MockNeuPrintBackend()
    with patch("connectomics_mcp.registry._backend_cache", {"hemibrain": backend}):
        yield backend
