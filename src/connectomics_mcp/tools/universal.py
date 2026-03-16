"""Tier 1: Universal tools that work across all datasets."""

from __future__ import annotations

import logging
from typing import Any

from connectomics_mcp.exceptions import DatasetNotSupported, StaleRootIdError
from connectomics_mcp.neuroglancer.url_builder import (
    NEUROGLANCER_CONFIGS,
    build_neuroglancer_url as _build_ngl_url,
)
from connectomics_mcp.output_contracts.formatters import (
    format_cell_type_search,
    format_cell_type_taxonomy,
    format_connectivity,
    format_neuron_info,
    format_neuroglancer_url,
    format_neurons_by_type,
    format_region_connectivity,
    format_validate_root_ids,
)
from connectomics_mcp.registry import DATASETS, check_capability, get_backend

logger = logging.getLogger(__name__)


def get_neuron_info(
    neuron_id: int | str, dataset: str, nucleus_id: int | None = None
) -> dict[str, Any]:
    """Get basic information about a neuron.

    Returns cell type, soma position, synapse counts, and a
    Neuroglancer URL for the given neuron in the specified dataset.

    Parameters
    ----------
    neuron_id : int | str
        The neuron identifier — a root ID for CAVE datasets
        (minnie65, flywire) or a body ID for neuPrint
        datasets (hemibrain).
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "hemibrain".
    nucleus_id : int, optional
        MICrONS nucleus ID (minnie65 only). If provided, resolves
        to the current pt_root_id before querying.

    Returns
    -------
    dict
        NeuronInfoResponse as a dict with keys: neuron_id, dataset,
        cell_type, cell_class, region, soma_position_nm,
        n_pre_synapses, n_post_synapses, proofread,
        materialization_version, neuroglancer_url, warnings.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or does not support universal tools.
    StaleRootIdError
        If the root ID is outdated (CAVE datasets only).
    ValueError
        If nucleus_id is provided but has no associated segment.
    """
    check_capability(dataset, "universal")

    nucleus_warnings: list[str] = []

    # Resolve nucleus ID to pt_root_id (MICrONS only)
    if nucleus_id is not None:
        if dataset != "minnie65":
            raise DatasetNotSupported(
                dataset,
                "nucleus_resolution (nucleus IDs are MICrONS-specific)",
            )

        backend = get_backend(dataset)
        resolution_raw = backend.resolve_nucleus_ids([nucleus_id])
        resolutions = resolution_raw.get("resolutions", [])

        if not resolutions:
            raise ValueError(
                f"Nucleus ID {nucleus_id} could not be resolved in {dataset}."
            )

        res = resolutions[0]
        status = res["resolution_status"]

        if status == "no_segment":
            raise ValueError(
                f"Nucleus ID {nucleus_id} has no associated segment in "
                f"{dataset}. The cell may be in an under-segmented region."
            )
        elif status == "merge_conflict":
            conflicting = res.get("conflicting_nucleus_ids", [])
            nucleus_warnings.append(
                f"Nucleus {nucleus_id} shares pt_root_id {res['pt_root_id']} "
                f"with nuclei {conflicting}. This segment likely contains a "
                f"merge error."
            )

        neuron_id = res["pt_root_id"]

    backend = get_backend(dataset)
    raw = backend.get_neuron_info(neuron_id)

    # CAVE datasets: raise if the root ID is stale
    if DATASETS[dataset]["backend"] == "cave" and not raw.get("is_current", True):
        raise StaleRootIdError(int(neuron_id))

    # Append nucleus resolution warnings
    if nucleus_warnings:
        raw.setdefault("warnings", []).extend(nucleus_warnings)

    response = format_neuron_info(raw, dataset)
    return response.model_dump()


def get_connectivity(
    neuron_id: int | str, dataset: str, direction: str = "both"
) -> dict[str, Any]:
    """Get synaptic connectivity partners for a neuron.

    Returns a summary with weight distributions and 3 orientation
    examples per direction. The complete partner table is saved as a
    Parquet artifact — load it with ``pd.read_parquet(artifact_path)``
    for full analysis.

    Parameters
    ----------
    neuron_id : int | str
        The neuron identifier — a root ID for CAVE datasets
        (minnie65, flywire) or a body ID for neuPrint
        datasets (hemibrain).
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "hemibrain".
    direction : str
        Which partners to return: "upstream", "downstream", or "both"
        (default "both").

    Returns
    -------
    dict
        ConnectivityResponse as a dict containing n_upstream_total,
        n_downstream_total, weight distributions, 3-item orientation
        samples, artifact_manifest with the path to the full Parquet
        file, and a Neuroglancer URL.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or does not support universal tools.
    StaleRootIdError
        If the root ID is outdated (CAVE datasets only).
    """
    check_capability(dataset, "universal")

    backend = get_backend(dataset)
    raw = backend.get_connectivity(neuron_id, direction=direction)

    # CAVE datasets: raise if the root ID is stale
    if DATASETS[dataset]["backend"] == "cave" and not raw.get("is_current", True):
        raise StaleRootIdError(int(neuron_id))

    response = format_connectivity(raw, dataset)
    return response.model_dump()


def validate_root_ids(
    root_ids: list[int], dataset: str
) -> dict[str, Any]:
    """Check whether neuron IDs are current.

    For CAVE datasets, root IDs can become stale after proofreading
    edits. This tool checks currency and suggests current replacements
    for stale IDs. For neuPrint datasets, body IDs are immutable and
    always current.

    Parameters
    ----------
    root_ids : list[int]
        The neuron identifiers to validate.
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "hemibrain".

    Returns
    -------
    dict
        RootIdValidationResponse as a dict with per-ID results,
        n_stale count, and suggested replacements.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or does not support universal tools.
    """
    check_capability(dataset, "universal")

    backend = get_backend(dataset)
    raw = backend.validate_root_ids(root_ids)

    response = format_validate_root_ids(raw, dataset)
    return response.model_dump()


def build_neuroglancer_url_tool(
    segment_ids: list[int | str],
    dataset: str,
    annotations: list[dict] | None = None,
) -> dict[str, Any]:
    """Build a Neuroglancer URL for the given segments.

    Parameters
    ----------
    segment_ids : list[int | str]
        Segment IDs to highlight in the viewer.
    dataset : str
        Dataset to use. Supported: "minnie65", "flywire",
        "hemibrain".
    annotations : list[dict], optional
        Point annotations to overlay.

    Returns
    -------
    dict
        NeuroglancerUrlResponse as a dict with keys: url, dataset,
        n_segments, layers_included, coordinate_space.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or does not support universal tools.
    """
    check_capability(dataset, "universal")

    url = _build_ngl_url(segment_ids, dataset, annotations)

    layers = ["em", "segmentation"]
    if annotations:
        layers.append("annotations")

    config = NEUROGLANCER_CONFIGS[dataset]
    coordinate_space = config.get("coordinate_space", "nm")

    response = format_neuroglancer_url(
        url=url,
        dataset=dataset,
        segment_ids=segment_ids,
        layers=layers,
        coordinate_space=coordinate_space,
    )
    return response.model_dump()


def get_cell_type_taxonomy(dataset: str) -> dict[str, Any]:
    """Get the cell type taxonomy/hierarchy for a dataset.

    Returns the classification levels, top values at each level
    with neuron counts, and example lineages showing the full
    classification path for representative neurons.

    For FlyWire, shows: super_class → cell_class → cell_sub_class →
    cell_type with top values and example lineages. For hemibrain,
    shows the flat type namespace. For MICrONS, shows cell types.

    Use this to understand how a dataset organizes its cell types
    before querying with ``search_cell_types`` or ``get_neurons_by_type``.

    Parameters
    ----------
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "hemibrain".

    Returns
    -------
    dict
        CellTypeTaxonomyResponse with levels, example_lineages,
        and n_total_neurons.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or does not support universal tools.
    """
    check_capability(dataset, "universal")

    backend = get_backend(dataset)
    raw = backend.get_cell_type_taxonomy()

    response = format_cell_type_taxonomy(raw, dataset)
    return response.model_dump()


def search_cell_types(
    query: str, dataset: str
) -> dict[str, Any]:
    """Search for cell types matching a query string.

    Performs case-insensitive substring matching across all available
    annotation levels in the specified dataset. Use this to discover
    cell type naming conventions before calling ``get_neurons_by_type``.

    Parameters
    ----------
    query : str
        Search string (e.g. "EPG", "ring", "compass", "KC").
    dataset : str
        Dataset to search. Supported: "minnie65", "flywire",
        "hemibrain".

    Returns
    -------
    dict
        CellTypeSearchResponse as a dict with keys: dataset, query,
        n_matches, matches (list with cell_type, classification_level,
        n_neurons), warnings.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or does not support universal tools.
    """
    check_capability(dataset, "universal")

    backend = get_backend(dataset)
    raw = backend.search_cell_types(query)

    response = format_cell_type_search(raw, dataset)
    return response.model_dump()


def get_neurons_by_type(
    cell_type: str, dataset: str, region: str | None = None
) -> dict[str, Any]:
    """Get all neurons matching a cell type annotation.

    Returns a summary with type and region distributions. The complete
    neuron list is saved as a Parquet artifact — load it with
    ``pd.read_parquet(artifact_path)`` for full analysis.

    Parameters
    ----------
    cell_type : str
        Cell type annotation to search for (e.g. "L2/3 IT").
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "hemibrain".
    region : str, optional
        Brain region filter.

    Returns
    -------
    dict
        NeuronsByTypeResponse as a dict containing n_total,
        type_distribution, region_distribution, and
        artifact_manifest with path to the full Parquet file.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or does not support universal tools.
    """
    check_capability(dataset, "universal")

    backend = get_backend(dataset)
    raw = backend.get_neurons_by_type(cell_type, region=region)

    response = format_neurons_by_type(raw, dataset)
    return response.model_dump()


def get_region_connectivity(
    dataset: str,
    source_region: str | None = None,
    target_region: str | None = None,
) -> dict[str, Any]:
    """Get region-to-region synaptic connectivity.

    Returns a summary with the top 5 connections and total synapse
    count. The complete region-pair table is saved as a Parquet
    artifact in long format — load it with
    ``pd.read_parquet(artifact_path)`` for full analysis. The long
    format can be converted to a matrix with ``pivot_table``.

    Parameters
    ----------
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "hemibrain".
    source_region : str, optional
        Filter to connections from this region.
    target_region : str, optional
        Filter to connections to this region.

    Returns
    -------
    dict
        RegionConnectivityResponse as a dict containing n_regions,
        top_5_connections, total_synapses, and artifact_manifest
        with the path to the full Parquet file.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or does not support universal tools.
    """
    check_capability(dataset, "universal")

    backend = get_backend(dataset)
    raw = backend.get_region_connectivity(
        source_region=source_region, target_region=target_region
    )

    response = format_region_connectivity(raw, dataset)
    return response.model_dump()
