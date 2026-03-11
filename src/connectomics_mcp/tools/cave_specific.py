"""Tier 2: CAVE-specific tools (proofreading, edit history, annotations)."""

from __future__ import annotations

import logging
from typing import Any

from connectomics_mcp.exceptions import DatasetNotSupported, StaleRootIdError
from connectomics_mcp.output_contracts.formatters import (
    format_annotation_table,
    format_edit_history,
    format_nucleus_resolution,
    format_proofreading_status,
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
        "flywire", or "fanc".

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
