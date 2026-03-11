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
            # Nucleus enrichment: partner 0-2 have clean 1:1, partner 3 is conflict, partner 4 has no nucleus
            "partner_nucleus_id": [100000 + i if i < 3 else None][0],
            "partner_nucleus_conflict": i == 3,
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
            "partner_nucleus_id": 200000 + i if i < 3 else None,
            "partner_nucleus_conflict": False,
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


def _make_cave_annotation_table_df(
    table_name: str,
    filter_equal_dict: dict[str, Any] | None = None,
    filter_in_dict: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Build a realistic mock annotation table DataFrame for CAVE."""
    rows = [
        {"id": 1, "pt_root_id": 864691135000100, "cell_type": "L2/3 IT", "tag": "V1"},
        {"id": 2, "pt_root_id": 864691135000101, "cell_type": "L2/3 IT", "tag": "V1"},
        {"id": 3, "pt_root_id": 864691135000102, "cell_type": "L4 IT", "tag": "LM"},
        {"id": 4, "pt_root_id": 864691135000103, "cell_type": "L5 PT", "tag": "V1"},
        {"id": 5, "pt_root_id": 864691135000104, "cell_type": "L2/3 IT", "tag": "AL"},
    ]
    df = pd.DataFrame(rows)

    if filter_equal_dict:
        for col, val in filter_equal_dict.items():
            if col in df.columns:
                df = df[df[col] == val]

    if filter_in_dict:
        for col, vals in filter_in_dict.items():
            if col in df.columns:
                df = df[df[col].isin(vals)]

    return df.reset_index(drop=True)


def _make_cave_edit_history_df() -> pd.DataFrame:
    """Build a realistic mock edit history DataFrame for CAVE."""
    rows = [
        {"operation_id": 0, "timestamp": "2025-06-01T10:00:00", "operation_type": "merge", "user_id": "user_a"},
        {"operation_id": 1, "timestamp": "2025-06-15T14:30:00", "operation_type": "split", "user_id": "user_b"},
        {"operation_id": 2, "timestamp": "2025-07-02T09:15:00", "operation_type": "merge", "user_id": "user_a"},
        {"operation_id": 3, "timestamp": "2025-08-10T16:45:00", "operation_type": "merge", "user_id": "user_c"},
        {"operation_id": 4, "timestamp": "2025-09-20T11:00:00", "operation_type": "split", "user_id": "user_b"},
        {"operation_id": 5, "timestamp": "2025-10-05T08:30:00", "operation_type": "merge", "user_id": "user_a"},
    ]
    return pd.DataFrame(rows)


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

    def resolve_nucleus_ids(
        self,
        nucleus_ids: list[int],
        materialization_version: int | None = None,
    ) -> dict[str, Any]:
        """Mock nucleus resolution covering all three statuses.

        Mock data:
        - nucleus 100001 → pt_root_id 864691135000001, resolved
        - nucleus 100002 → pt_root_id 864691135000099 (shared), merge_conflict
        - nucleus 100003 → pt_root_id 864691135000099 (shared), merge_conflict
        - nucleus 999999 → no segment
        """
        # Simulated nucleus_detection_v0 lookup
        nuc_to_root: dict[int, int | None] = {
            100001: 864691135000001,
            100002: 864691135000099,
            100003: 864691135000099,
            # 999999 is intentionally absent → no_segment
        }

        # The real backend queries ALL nucleus entries for a given pt_root_id,
        # so merge conflicts are detected even when only one nucleus is queried.
        # Simulate this by always knowing the full table's root→nucleus mapping.
        all_root_to_nucs: dict[int, list[int]] = {}
        for nid, rid in nuc_to_root.items():
            if rid is not None:
                all_root_to_nucs.setdefault(rid, []).append(nid)

        merge_conflict_nucs: set[int] = set()
        for rid, nids in all_root_to_nucs.items():
            if len(nids) > 1:
                merge_conflict_nucs.update(nids)

        # root_to_nucs scoped to queried IDs (for conflicting_nucleus_ids)
        root_to_nucs = all_root_to_nucs

        resolutions = []
        n_resolved = 0
        n_merge_conflicts = 0
        n_no_segment = 0

        for nid in nucleus_ids:
            rid = nuc_to_root.get(nid)
            if rid is None:
                resolutions.append({
                    "nucleus_id": nid,
                    "pt_root_id": None,
                    "resolution_status": "no_segment",
                    "conflicting_nucleus_ids": [],
                    "materialization_version": 943,
                })
                n_no_segment += 1
            elif nid in merge_conflict_nucs:
                conflicting = [o for o in root_to_nucs[rid] if o != nid]
                resolutions.append({
                    "nucleus_id": nid,
                    "pt_root_id": rid,
                    "resolution_status": "merge_conflict",
                    "conflicting_nucleus_ids": conflicting,
                    "materialization_version": 943,
                })
                n_merge_conflicts += 1
            else:
                resolutions.append({
                    "nucleus_id": nid,
                    "pt_root_id": rid,
                    "resolution_status": "resolved",
                    "conflicting_nucleus_ids": [],
                    "materialization_version": 943,
                })
                n_resolved += 1

        return {
            "dataset": "minnie65",
            "materialization_version": 943,
            "resolutions": resolutions,
            "n_resolved": n_resolved,
            "n_merge_conflicts": n_merge_conflicts,
            "n_no_segment": n_no_segment,
            "warnings": [],
        }

    def query_annotation_table(
        self,
        table_name: str,
        filter_equal_dict: dict[str, Any] | None = None,
        filter_in_dict: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        table_df = _make_cave_annotation_table_df(
            table_name, filter_equal_dict, filter_in_dict
        )
        col_descs = [f"{col}: {dtype}" for col, dtype in table_df.dtypes.items()]
        schema_description = "; ".join(col_descs)
        return {
            "dataset": "minnie65",
            "table_name": table_name,
            "materialization_version": 943,
            "warnings": [],
            "table_df": table_df,
            "schema_description": schema_description,
        }

    def get_edit_history(self, neuron_id: int) -> dict[str, Any]:
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
            "materialization_version": 943,
            "warnings": warnings,
            "edits_df": _make_cave_edit_history_df(),
        }

    def fetch_cypher(self, query: str) -> dict[str, Any]:
        from connectomics_mcp.exceptions import DatasetNotSupported

        raise DatasetNotSupported("minnie65", "neuprint")

    def get_synapse_compartments(
        self, neuron_id: int | str, direction: str = "input"
    ) -> dict[str, Any]:
        from connectomics_mcp.exceptions import DatasetNotSupported

        raise DatasetNotSupported("minnie65", "neuprint")

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


def _make_neuprint_cypher_result_df() -> pd.DataFrame:
    """Build a realistic mock Cypher query result DataFrame."""
    rows = [
        {"bodyId": 500001, "type": "KC-ab", "pre": 200, "post": 800},
        {"bodyId": 500002, "type": "KC-ab", "pre": 180, "post": 750},
        {"bodyId": 500003, "type": "MBON14", "pre": 350, "post": 1200},
        {"bodyId": 500004, "type": "DAN-d1", "pre": 420, "post": 900},
    ]
    return pd.DataFrame(rows)


def _make_neuprint_compartment_data(
    direction: str = "input",
) -> tuple[list[dict[str, Any]], int]:
    """Build mock per-ROI compartment data for neuPrint.

    Returns
    -------
    tuple[list[dict], int]
        (compartments list, n_total_synapses)
    """
    # ROI data with pre and post counts
    roi_data = [
        {"roi": "MB(+)", "pre": 300, "post": 900},
        {"roi": "CA(R)", "pre": 150, "post": 500},
        {"roi": "LH(R)", "pre": 200, "post": 400},
        {"roi": "SMP(R)", "pre": 50, "post": 100},
    ]

    count_col = "post" if direction == "input" else "pre"
    compartments = []
    n_total = 0
    for roi in roi_data:
        count = roi[count_col]
        if count > 0:
            compartments.append({
                "compartment": roi["roi"],
                "n_synapses": count,
            })
            n_total += count

    for comp in compartments:
        comp["fraction"] = round(comp["n_synapses"] / n_total, 4) if n_total > 0 else 0.0

    compartments.sort(key=lambda c: c["n_synapses"], reverse=True)

    return compartments, n_total


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

    def query_annotation_table(
        self,
        table_name: str,
        filter_equal_dict: dict[str, Any] | None = None,
        filter_in_dict: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        from connectomics_mcp.exceptions import DatasetNotSupported

        raise DatasetNotSupported("hemibrain", "cave")

    def get_edit_history(self, neuron_id: int) -> dict[str, Any]:
        from connectomics_mcp.exceptions import DatasetNotSupported

        raise DatasetNotSupported("hemibrain", "cave")

    def fetch_cypher(self, query: str) -> dict[str, Any]:
        result_df = _make_neuprint_cypher_result_df()
        return {
            "dataset": "hemibrain",
            "query": query,
            "materialization_version": None,
            "warnings": [],
            "result_df": result_df,
        }

    def get_synapse_compartments(
        self, neuron_id: int | str, direction: str = "input"
    ) -> dict[str, Any]:
        body_id = int(neuron_id)
        compartments, n_total = _make_neuprint_compartment_data(direction)
        return {
            "neuron_id": body_id,
            "dataset": "hemibrain",
            "direction": direction,
            "compartments": compartments,
            "n_total_synapses": n_total,
            "warnings": [],
        }

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
