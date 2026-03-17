"""Tier 2: CAVE-specific tools (proofreading, edit history, annotations)."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from connectomics_mcp.exceptions import DatasetNotSupported, StaleRootIdError
from connectomics_mcp.output_contracts.formatters import (
    format_annotation_table,
    format_bulk_coregistration,
    format_bulk_functional_area,
    format_bulk_functional_properties,
    format_bulk_synapse_targets,
    format_cell_mtypes,
    format_coregistration,
    format_edit_history,
    format_functional_area,
    format_functional_properties,
    format_multi_input_spines,
    format_nucleus_resolution,
    format_proofreading_status,
    format_synapse_targets,
)
from connectomics_mcp.registry import DATASETS, check_capability, get_backend

logger = logging.getLogger(__name__)


def get_proofreading_status(neuron_id: int, dataset: str) -> dict[str, Any]:
    """Get proofreading status for a CAVE neuron.

    Returns axon/dendrite proofreading flags, proofreading strategy,
    edit count, and last edit timestamp.

    Parameters
    ----------
    neuron_id : int
        Root ID of the neuron.
    dataset : str
        Dataset to query. Must be a CAVE dataset: "minnie65",
        "flywire".

    Returns
    -------
    dict
        ProofreadingStatusResponse as a dict with proofreading flags,
        strategy strings, edit count, and last edit timestamp.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or not a CAVE dataset.
    StaleRootIdError
        If the root ID is outdated.
    """
    check_capability(dataset, "cave")

    backend = get_backend(dataset)
    raw = backend.get_proofreading_status(neuron_id)

    # Raise if the root ID is stale
    if not raw.get("is_current", True):
        raise StaleRootIdError(int(neuron_id))

    response = format_proofreading_status(raw, dataset)
    return response.model_dump()


def query_annotation_table(
    dataset: str,
    table_name: str,
    filter_equal_dict: dict[str, Any] | None = None,
    filter_in_dict: dict[str, list] | None = None,
) -> dict[str, Any]:
    """Query a CAVE annotation table.

    Returns a summary with row count and schema description. The
    complete query result is saved as a Parquet artifact — load it
    with ``pd.read_parquet(artifact_path)`` for full analysis.

    Parameters
    ----------
    dataset : str
        Dataset to query. Must be a CAVE dataset.
    table_name : str
        Name of the annotation table to query.
    filter_equal_dict : dict, optional
        Equality filters (column → value).
    filter_in_dict : dict, optional
        Membership filters (column → list of values).

    Returns
    -------
    dict
        AnnotationTableResponse as a dict with artifact_manifest
        pointing to the full table on disk.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or not a CAVE dataset.
    """
    check_capability(dataset, "cave")

    backend = get_backend(dataset)
    raw = backend.query_annotation_table(
        table_name,
        filter_equal_dict=filter_equal_dict,
        filter_in_dict=filter_in_dict,
    )

    response = format_annotation_table(raw, dataset)
    return response.model_dump()


def get_edit_history(neuron_id: int, dataset: str) -> dict[str, Any]:
    """Get edit history for a CAVE neuron.

    Returns a summary with edit count and timestamp range. The
    complete edit log is saved as a Parquet artifact — load it
    with ``pd.read_parquet(artifact_path)`` for full analysis.

    Parameters
    ----------
    neuron_id : int
        Root ID of the neuron.
    dataset : str
        Dataset to query. Must be a CAVE dataset.

    Returns
    -------
    dict
        EditHistoryResponse as a dict with artifact_manifest
        pointing to the full edit log on disk.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or not a CAVE dataset.
    StaleRootIdError
        If the root ID is outdated.
    """
    check_capability(dataset, "cave")

    backend = get_backend(dataset)
    raw = backend.get_edit_history(neuron_id)

    # Raise if the root ID is stale
    if not raw.get("is_current", True):
        raise StaleRootIdError(int(neuron_id))

    response = format_edit_history(raw, dataset)
    return response.model_dump()


def resolve_nucleus_ids(
    nucleus_ids: list[int], dataset: str
) -> dict[str, Any]:
    """Resolve MICrONS nucleus IDs to current pt_root_ids.

    Nucleus IDs are the stable cross-version cell identifiers in
    MICrONS (minnie65). Unlike pt_root_ids, which change with
    proofreading, nucleus IDs are assigned by the nucleus detection
    model and never change.

    A ``merge_conflict`` result means the segment at this nucleus's
    location contains multiple detected nuclei — the segmentation
    likely has an unresolved merge error. Treat these with caution in
    downstream analysis.

    A ``no_segment`` result means no segment was found at this
    nucleus's position — the cell may be in an under-segmented region.

    Parameters
    ----------
    nucleus_ids : list[int]
        Nucleus IDs to resolve.
    dataset : str
        Dataset to query. Must be "minnie65" — nucleus ID resolution
        is MICrONS-specific.

    Returns
    -------
    dict
        NucleusResolutionResult as a dict with per-nucleus resolution
        status, pt_root_ids, and conflict information.

    Raises
    ------
    DatasetNotSupported
        If the dataset is not "minnie65".
    """
    check_capability(dataset, "cave")

    if dataset != "minnie65":
        raise DatasetNotSupported(
            dataset,
            "nucleus_resolution (nucleus IDs are MICrONS-specific)",
        )

    backend = get_backend(dataset)
    raw = backend.resolve_nucleus_ids(nucleus_ids)

    response = format_nucleus_resolution(raw, dataset)
    return response.model_dump()


# ---------------------------------------------------------------------------
# MICrONS-specific table tools
# ---------------------------------------------------------------------------


def _check_minnie65(dataset: str, tool_name: str) -> None:
    """Validate that the dataset is minnie65 (MICrONS-only tools)."""
    check_capability(dataset, "cave")
    if dataset != "minnie65":
        raise DatasetNotSupported(
            dataset, f"{tool_name} (MICrONS-specific)",
        )


def get_coregistration(
    neuron_id: int, dataset: str, by: str = "root_id"
) -> dict[str, Any]:
    """Get EM-to-functional imaging coregistration for a neuron.

    Maps EM neurons to 2-photon functional imaging units (session,
    scan, unit). Complete results saved as Parquet artifact.

    Parameters
    ----------
    neuron_id : int
        Root ID or nucleus ID.
    dataset : str
        Must be ``"minnie65"``.
    by : str
        ``"root_id"`` or ``"nucleus_id"``.

    Returns
    -------
    dict
        CoregistrationResponse with artifact_manifest.

    Raises
    ------
    DatasetNotSupported
        If not minnie65.
    StaleRootIdError
        If root ID is stale (when ``by="root_id"``).
    """
    _check_minnie65(dataset, "coregistration")

    backend = get_backend(dataset)
    raw = backend.query_coregistration(neuron_id, by=by)

    if not raw.get("is_current", True):
        raise StaleRootIdError(int(neuron_id))

    response = format_coregistration(raw, dataset)
    return response.model_dump()


def get_functional_properties(
    neuron_id: int,
    dataset: str,
    by: str = "root_id",
    coregistration_source: str = "auto_phase3",
) -> dict[str, Any]:
    """Get digital twin functional properties for a neuron.

    Returns orientation/direction selectivity, receptive field centers,
    and model performance metrics. Complete results saved as Parquet
    artifact.

    Parameters
    ----------
    neuron_id : int
        Root ID or nucleus ID.
    dataset : str
        Must be ``"minnie65"``.
    by : str
        ``"root_id"`` or ``"nucleus_id"``.
    coregistration_source : str
        Coregistration table variant: ``"coreg_v4"`` (default),
        ``"auto_phase3"``, or ``"apl_vess"``.

    Returns
    -------
    dict
        FunctionalPropertiesResponse with artifact_manifest.

    Raises
    ------
    DatasetNotSupported
        If not minnie65.
    StaleRootIdError
        If root ID is stale (when ``by="root_id"``).
    """
    _check_minnie65(dataset, "functional_properties")

    backend = get_backend(dataset)
    raw = backend.query_functional_properties(
        neuron_id, by=by, coregistration_source=coregistration_source
    )

    if not raw.get("is_current", True):
        raise StaleRootIdError(int(neuron_id))

    response = format_functional_properties(raw, dataset)
    return response.model_dump()


def get_synapse_targets(
    root_id: int, dataset: str, direction: str = "post"
) -> dict[str, Any]:
    """Get per-synapse structural target predictions for a neuron.

    Classifies each synapse as targeting spine, shaft, or soma.
    Complete results saved as Parquet artifact.

    Parameters
    ----------
    root_id : int
        Root ID of the neuron.
    dataset : str
        Must be ``"minnie65"``.
    direction : str
        ``"post"`` for synapses onto this neuron (default),
        ``"pre"`` for synapses from this neuron.

    Returns
    -------
    dict
        SynapseTargetsResponse with artifact_manifest.

    Raises
    ------
    DatasetNotSupported
        If not minnie65.
    StaleRootIdError
        If root ID is stale.
    """
    _check_minnie65(dataset, "synapse_targets")

    backend = get_backend(dataset)
    raw = backend.query_synapse_targets(root_id, direction=direction)

    if not raw.get("is_current", True):
        raise StaleRootIdError(int(root_id))

    response = format_synapse_targets(raw, dataset)
    return response.model_dump()


def get_multi_input_spines(
    root_id: int, dataset: str, direction: str = "post"
) -> dict[str, Any]:
    """Get multi-input spine predictions for a neuron.

    Deprecated: prefer ``get_synapse_targets`` for general synapse
    target queries. This table identifies spines receiving >1 input
    synapse, grouped by shared postsynaptic compartment.

    Parameters
    ----------
    root_id : int
        Root ID of the neuron.
    dataset : str
        Must be ``"minnie65"``.
    direction : str
        ``"post"`` for spines on this neuron (default),
        ``"pre"`` for spines from this neuron.

    Returns
    -------
    dict
        MultiInputSpinesResponse with artifact_manifest.

    Raises
    ------
    DatasetNotSupported
        If not minnie65.
    StaleRootIdError
        If root ID is stale.
    """
    _check_minnie65(dataset, "multi_input_spines")

    backend = get_backend(dataset)
    raw = backend.query_multi_input_spines(root_id, direction=direction)

    if not raw.get("is_current", True):
        raise StaleRootIdError(int(root_id))

    response = format_multi_input_spines(raw, dataset)
    return response.model_dump()


def get_cell_mtypes(
    dataset: str,
    neuron_id: int | None = None,
    by: str = "root_id",
    cell_type: str | None = None,
) -> dict[str, Any]:
    """Get morphological cell type (mtype) classifications.

    24 types based on dendritic features and connectivity motifs.
    Excitatory: L2a, L2b, L3a, L3b, L3c, L4a, L4b, L4c, L5a, L5b,
    L5ET, L5NP, L6a, L6b, L6c, L6CT, L6wm.
    Inhibitory: PTC, DTC, STC, ITC.
    Complete results saved as Parquet artifact.

    Parameters
    ----------
    dataset : str
        Must be ``"minnie65"``.
    neuron_id : int, optional
        Root ID or nucleus ID for single-neuron lookup.
    by : str
        ``"root_id"`` or ``"nucleus_id"``.
    cell_type : str, optional
        Filter by mtype (e.g. ``"L2a"``, ``"DTC"``).

    Returns
    -------
    dict
        CellMtypesResponse with artifact_manifest.

    Raises
    ------
    DatasetNotSupported
        If not minnie65.
    StaleRootIdError
        If root ID is stale (when querying by root_id).
    """
    _check_minnie65(dataset, "cell_mtypes")

    backend = get_backend(dataset)
    raw = backend.query_cell_mtypes(
        neuron_id=neuron_id, by=by, cell_type=cell_type
    )

    if not raw.get("is_current", True):
        raise StaleRootIdError(int(neuron_id))

    response = format_cell_mtypes(raw, dataset)
    return response.model_dump()


def get_functional_area(
    dataset: str,
    neuron_id: int | None = None,
    by: str = "root_id",
    area: str | None = None,
) -> dict[str, Any]:
    """Get functional brain area assignments for MICrONS neurons.

    Areas: V1, AL, RL, LM — inferred from 2-photon imaging
    boundaries projected into EM space. The ``value`` column is
    distance to nearest area boundary in micrometers (higher =
    more confident). Complete results saved as Parquet artifact.

    Parameters
    ----------
    dataset : str
        Must be ``"minnie65"``.
    neuron_id : int, optional
        Root ID or nucleus ID for single-neuron lookup.
    by : str
        ``"root_id"`` or ``"nucleus_id"``.
    area : str, optional
        Filter by area label (``"V1"``, ``"AL"``, ``"RL"``, ``"LM"``).

    Returns
    -------
    dict
        FunctionalAreaResponse with artifact_manifest.

    Raises
    ------
    DatasetNotSupported
        If not minnie65.
    StaleRootIdError
        If root ID is stale (when querying by root_id).
    """
    _check_minnie65(dataset, "functional_area")

    backend = get_backend(dataset)
    raw = backend.query_functional_area(
        neuron_id=neuron_id, by=by, area=area
    )

    if not raw.get("is_current", True):
        raise StaleRootIdError(int(neuron_id))

    response = format_functional_area(raw, dataset)
    return response.model_dump()


# ---------------------------------------------------------------------------
# Bulk MICrONS tools
# ---------------------------------------------------------------------------


def _bulk_staleness_gate(
    root_ids: list[int], dataset: str
) -> None:
    """Validate all root IDs are current, raising ValueError if any are stale."""
    backend = get_backend(dataset)
    raw_validation = backend.validate_root_ids(root_ids)
    stale = [
        r["root_id"]
        for r in raw_validation["results"]
        if not r["is_current"]
    ]
    if stale:
        raise ValueError(
            f"Stale root IDs: {stale}. Use validate_root_ids() to get "
            f"current IDs before calling bulk tools."
        )


def _bulk_cache_key(root_ids: list[int], *extra_parts: str) -> str:
    """Compute content-addressable cache key for bulk queries."""
    sorted_ids = sorted(root_ids)
    hash_input = ",".join(str(i) for i in sorted_ids)
    if extra_parts:
        hash_input += ":" + ":".join(extra_parts)
    return hashlib.sha256(hash_input.encode()).hexdigest()[:8]


def _check_bulk_cache(
    tool: str, dataset: str, extra_key: str, mat_version: int | None
) -> dict[str, Any] | None:
    """Check for a cached bulk artifact. Returns response dict or None."""
    from connectomics_mcp.artifacts.writer import _find_cached

    cached_path = _find_cached(
        tool=tool,
        dataset=dataset,
        neuron_id=None,
        materialization_version=mat_version,
        extra_key=extra_key,
    )
    if cached_path is None:
        return None

    import pandas as pd
    from datetime import datetime, timezone

    from connectomics_mcp.artifacts.writer import _describe_columns
    from connectomics_mcp.output_contracts.schemas import ArtifactManifest

    cached_df = pd.read_parquet(cached_path)
    manifest = ArtifactManifest(
        artifact_path=str(cached_path),
        n_rows=len(cached_df),
        columns=list(cached_df.columns),
        schema_description=_describe_columns(cached_df),
        dataset=dataset,
        query_timestamp=datetime.fromtimestamp(
            cached_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        materialization_version=mat_version,
        cache_hit=True,
    )
    return {"cached_df": cached_df, "manifest": manifest}


def get_bulk_coregistration(
    root_ids: list[int], dataset: str
) -> dict[str, Any]:
    """Get EM-to-functional coregistration for multiple neurons in bulk.

    Fetches coregistration data for all given root IDs and saves the
    complete result as a single Parquet artifact.

    Parameters
    ----------
    root_ids : list[int]
        Root IDs to query. All must be current.
    dataset : str
        Must be ``"minnie65"``.

    Returns
    -------
    dict
        BulkCoregistrationResponse with artifact_manifest.

    Raises
    ------
    DatasetNotSupported
        If not minnie65.
    ValueError
        If any root ID is stale.
    """
    _check_minnie65(dataset, "bulk_coregistration")
    backend = get_backend(dataset)

    extra_key = _bulk_cache_key(root_ids)
    mat_version = None
    try:
        mat_version = backend.client.materialize.version
    except Exception:
        pass

    cached = _check_bulk_cache(
        "bulk_coregistration", dataset, extra_key, mat_version
    )
    if cached is not None:
        from connectomics_mcp.output_contracts.schemas import (
            BulkCoregistrationResponse,
        )

        df = cached["cached_df"]
        score_dist = {}
        sessions: list[int] = []
        if not df.empty:
            import numpy as np

            if "score" in df.columns:
                score_dist = {
                    "mean": round(float(df["score"].mean()), 2),
                    "median": round(float(df["score"].median()), 2),
                    "max": int(df["score"].max()),
                    "p90": round(float(np.percentile(df["score"], 90)), 2),
                }
            if "session" in df.columns:
                sessions = sorted(int(s) for s in df["session"].dropna().unique())
        return BulkCoregistrationResponse(
            dataset=dataset,
            n_root_ids=len(root_ids),
            n_units=len(df),
            score_distribution=score_dist,
            sessions=sessions,
            cached=True,
            artifact_manifest=cached["manifest"],
            warnings=[],
        ).model_dump()

    _bulk_staleness_gate(root_ids, dataset)

    if not root_ids:
        import pandas as pd

        raw = {
            "table_df": pd.DataFrame(),
            "table_name": "coregistration_auto_phase3_fwd_apl_vess_combined_v2",
            "materialization_version": mat_version,
            "warnings": [],
            "n_root_ids": 0,
        }
        return format_bulk_coregistration(raw, dataset, extra_key).model_dump()

    raw = backend.bulk_query_coregistration(sorted(root_ids))
    raw["n_root_ids"] = len(root_ids)
    return format_bulk_coregistration(raw, dataset, extra_key).model_dump()


def get_bulk_functional_properties(
    root_ids: list[int],
    dataset: str,
    coregistration_source: str = "auto_phase3",
) -> dict[str, Any]:
    """Get digital twin functional properties for multiple neurons in bulk.

    Fetches orientation/direction selectivity and model performance
    for all given root IDs. Complete results saved as Parquet artifact.

    Parameters
    ----------
    root_ids : list[int]
        Root IDs to query. All must be current.
    dataset : str
        Must be ``"minnie65"``.
    coregistration_source : str
        Table variant: ``"auto_phase3"`` (default), ``"coreg_v4"``,
        or ``"apl_vess"``.

    Returns
    -------
    dict
        BulkFunctionalPropertiesResponse with artifact_manifest.

    Raises
    ------
    DatasetNotSupported
        If not minnie65.
    ValueError
        If any root ID is stale.
    """
    _check_minnie65(dataset, "bulk_functional_properties")
    backend = get_backend(dataset)

    extra_key = _bulk_cache_key(root_ids, coregistration_source)
    mat_version = None
    try:
        mat_version = backend.client.materialize.version
    except Exception:
        pass

    cached = _check_bulk_cache(
        "bulk_functional_properties", dataset, extra_key, mat_version
    )
    if cached is not None:
        from connectomics_mcp.output_contracts.schemas import (
            BulkFunctionalPropertiesResponse,
        )

        df = cached["cached_df"]
        ori_dist = {}
        dir_dist = {}
        if not df.empty:
            import numpy as np

            if "OSI" in df.columns:
                ori_dist = {
                    "mean": round(float(df["OSI"].mean()), 2),
                    "median": round(float(df["OSI"].median()), 2),
                    "max": int(df["OSI"].max()),
                    "p90": round(float(np.percentile(df["OSI"], 90)), 2),
                }
            if "DSI" in df.columns:
                dir_dist = {
                    "mean": round(float(df["DSI"].mean()), 2),
                    "median": round(float(df["DSI"].median()), 2),
                    "max": int(df["DSI"].max()),
                    "p90": round(float(np.percentile(df["DSI"], 90)), 2),
                }
        return BulkFunctionalPropertiesResponse(
            dataset=dataset,
            n_root_ids=len(root_ids),
            coregistration_source=coregistration_source,
            n_units=len(df),
            ori_selectivity_distribution=ori_dist,
            dir_selectivity_distribution=dir_dist,
            cached=True,
            artifact_manifest=cached["manifest"],
            warnings=[],
        ).model_dump()

    _bulk_staleness_gate(root_ids, dataset)

    if not root_ids:
        import pandas as pd

        raw = {
            "table_df": pd.DataFrame(),
            "table_name": "",
            "materialization_version": mat_version,
            "warnings": [],
            "n_root_ids": 0,
            "coregistration_source": coregistration_source,
        }
        return format_bulk_functional_properties(
            raw, dataset, extra_key
        ).model_dump()

    raw = backend.bulk_query_functional_properties(
        sorted(root_ids), coregistration_source=coregistration_source
    )
    raw["n_root_ids"] = len(root_ids)
    return format_bulk_functional_properties(
        raw, dataset, extra_key
    ).model_dump()


def get_bulk_synapse_targets(
    root_ids: list[int], dataset: str, direction: str = "post"
) -> dict[str, Any]:
    """Get per-synapse structural target predictions for multiple neurons.

    Classifies each synapse as targeting spine, shaft, or soma for
    all given root IDs. Complete results saved as Parquet artifact.

    Parameters
    ----------
    root_ids : list[int]
        Root IDs to query. All must be current.
    dataset : str
        Must be ``"minnie65"``.
    direction : str
        ``"post"`` for synapses onto these neurons (default),
        ``"pre"`` for synapses from these neurons.

    Returns
    -------
    dict
        BulkSynapseTargetsResponse with artifact_manifest.

    Raises
    ------
    DatasetNotSupported
        If not minnie65.
    ValueError
        If any root ID is stale.
    """
    _check_minnie65(dataset, "bulk_synapse_targets")
    backend = get_backend(dataset)

    extra_key = _bulk_cache_key(root_ids, direction)
    mat_version = None
    try:
        mat_version = backend.client.materialize.version
    except Exception:
        pass

    cached = _check_bulk_cache(
        "bulk_synapse_targets", dataset, extra_key, mat_version
    )
    if cached is not None:
        from connectomics_mcp.output_contracts.schemas import (
            BulkSynapseTargetsResponse,
        )

        df = cached["cached_df"]
        tag_dist = {}
        if not df.empty and "tag" in df.columns:
            counts = df["tag"].dropna().value_counts()
            tag_dist = {str(k): int(v) for k, v in counts.items()}
        return BulkSynapseTargetsResponse(
            dataset=dataset,
            n_root_ids=len(root_ids),
            direction=direction,
            n_synapses=len(df),
            target_distribution=tag_dist,
            cached=True,
            artifact_manifest=cached["manifest"],
            warnings=[],
        ).model_dump()

    _bulk_staleness_gate(root_ids, dataset)

    if not root_ids:
        import pandas as pd

        raw = {
            "table_df": pd.DataFrame(),
            "table_name": "synapse_target_predictions_ssa_v2",
            "materialization_version": mat_version,
            "warnings": [],
            "n_root_ids": 0,
            "direction": direction,
        }
        return format_bulk_synapse_targets(
            raw, dataset, extra_key
        ).model_dump()

    raw = backend.bulk_query_synapse_targets(
        sorted(root_ids), direction=direction
    )
    raw["n_root_ids"] = len(root_ids)
    return format_bulk_synapse_targets(raw, dataset, extra_key).model_dump()


def get_bulk_functional_area(
    root_ids: list[int], dataset: str
) -> dict[str, Any]:
    """Get functional brain area assignments for multiple neurons in bulk.

    Returns area labels (V1, AL, RL, LM) with boundary distances for
    all given root IDs. Complete results saved as Parquet artifact.

    Parameters
    ----------
    root_ids : list[int]
        Root IDs to query. All must be current.
    dataset : str
        Must be ``"minnie65"``.

    Returns
    -------
    dict
        BulkFunctionalAreaResponse with artifact_manifest.

    Raises
    ------
    DatasetNotSupported
        If not minnie65.
    ValueError
        If any root ID is stale.
    """
    _check_minnie65(dataset, "bulk_functional_area")
    backend = get_backend(dataset)

    extra_key = _bulk_cache_key(root_ids)
    mat_version = None
    try:
        mat_version = backend.client.materialize.version
    except Exception:
        pass

    cached = _check_bulk_cache(
        "bulk_functional_area", dataset, extra_key, mat_version
    )
    if cached is not None:
        from connectomics_mcp.output_contracts.schemas import (
            BulkFunctionalAreaResponse,
        )

        df = cached["cached_df"]
        area_dist = {}
        if not df.empty and "tag" in df.columns:
            counts = df["tag"].dropna().value_counts()
            area_dist = {str(k): int(v) for k, v in counts.items()}
        return BulkFunctionalAreaResponse(
            dataset=dataset,
            n_root_ids=len(root_ids),
            n_total=len(df),
            area_distribution=area_dist,
            cached=True,
            artifact_manifest=cached["manifest"],
            warnings=[],
        ).model_dump()

    _bulk_staleness_gate(root_ids, dataset)

    if not root_ids:
        import pandas as pd

        raw = {
            "table_df": pd.DataFrame(),
            "table_name": "nucleus_functional_area_assignment",
            "materialization_version": mat_version,
            "warnings": [],
            "n_root_ids": 0,
        }
        return format_bulk_functional_area(
            raw, dataset, extra_key
        ).model_dump()

    raw = backend.bulk_query_functional_area(sorted(root_ids))
    raw["n_root_ids"] = len(root_ids)
    return format_bulk_functional_area(raw, dataset, extra_key).model_dump()
