"""Tier 2: CAVE-specific tools (proofreading, edit history, annotations)."""

from __future__ import annotations

import logging
from typing import Any

from connectomics_mcp.exceptions import DatasetNotSupported, StaleRootIdError
from connectomics_mcp.output_contracts.formatters import (
    format_annotation_table,
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
