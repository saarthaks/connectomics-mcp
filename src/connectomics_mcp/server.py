"""FastMCP server entry point for Connectomics MCP."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from connectomics_mcp.tools import cave_specific, universal

logger = logging.getLogger(__name__)

mcp = FastMCP("connectomics-mcp")


@mcp.tool()
def get_neuron_info(neuron_id: int | str, dataset: str) -> dict:
    """Get basic information about a neuron.

    Returns cell type, soma position, synapse counts, and a
    Neuroglancer URL for the given neuron in the specified dataset.

    Parameters
    ----------
    neuron_id : int | str
        The neuron identifier — a root ID for CAVE datasets
        (minnie65, flywire, fanc) or a body ID for neuPrint
        datasets (hemibrain).
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "fanc", "hemibrain".

    Returns
    -------
    dict
        NeuronInfoResponse with keys: neuron_id, dataset, cell_type,
        cell_class, region, soma_position_nm, n_pre_synapses,
        n_post_synapses, proofread, materialization_version,
        neuroglancer_url, warnings.
    """
    return universal.get_neuron_info(neuron_id, dataset)


@mcp.tool()
def get_connectivity(
    neuron_id: int | str, dataset: str, direction: str = "both"
) -> dict:
    """Get synaptic connectivity partners for a neuron.

    Returns a summary with weight distributions and 3 orientation
    examples per direction. The complete partner table is saved as a
    Parquet artifact — load it with ``pd.read_parquet(artifact_path)``
    for full analysis.

    Parameters
    ----------
    neuron_id : int | str
        The neuron identifier — a root ID for CAVE datasets
        (minnie65, flywire, fanc) or a body ID for neuPrint
        datasets (hemibrain).
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "fanc", "hemibrain".
    direction : str
        Which partners to return: "upstream", "downstream", or "both"
        (default "both").

    Returns
    -------
    dict
        ConnectivityResponse with artifact_manifest pointing to the
        full partner table on disk, plus summary statistics and
        3-item orientation samples.
    """
    return universal.get_connectivity(neuron_id, dataset, direction)


@mcp.tool()
def validate_root_ids(root_ids: list[int], dataset: str) -> dict:
    """Check whether neuron IDs are current.

    For CAVE datasets, root IDs can become stale after proofreading
    edits. This tool checks currency and suggests current replacements
    for any stale IDs. For neuPrint datasets, body IDs are immutable
    and always reported as current.

    Parameters
    ----------
    root_ids : list[int]
        The neuron identifiers to validate.
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "fanc", "hemibrain".

    Returns
    -------
    dict
        RootIdValidationResponse with per-ID results including
        is_current, suggested_current_id, and n_stale count.
    """
    return universal.validate_root_ids(root_ids, dataset)


@mcp.tool()
def get_proofreading_status(neuron_id: int, dataset: str) -> dict:
    """Get proofreading status for a CAVE neuron.

    Returns whether axon and dendrite have been proofread, the
    proofreading strategy used, edit count, and last edit timestamp.
    Only available for CAVE datasets (minnie65, flywire, fanc).

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
        ProofreadingStatusResponse with proofreading flags,
        strategy strings, edit count, and last edit timestamp.
    """
    return cave_specific.get_proofreading_status(neuron_id, dataset)


@mcp.tool()
def build_neuroglancer_url(
    segment_ids: list[int | str],
    dataset: str,
    annotations: list[dict] | None = None,
) -> dict:
    """Build a Neuroglancer URL for visualizing neuron segments.

    Constructs a fully encoded Neuroglancer URL with EM and
    segmentation layers, with the specified segments selected.
    Optionally includes point annotations.

    Parameters
    ----------
    segment_ids : list[int | str]
        Segment IDs to highlight in the viewer.
    dataset : str
        Dataset to use. Supported: "minnie65", "flywire",
        "fanc", "hemibrain".
    annotations : list[dict], optional
        Point annotations to overlay as an annotation layer.

    Returns
    -------
    dict
        NeuroglancerUrlResponse with keys: url, dataset,
        n_segments, layers_included, coordinate_space.
    """
    return universal.build_neuroglancer_url_tool(segment_ids, dataset, annotations)


@mcp.tool()
def get_neurons_by_type(
    cell_type: str, dataset: str, region: str | None = None
) -> dict:
    """Get all neurons matching a cell type annotation.

    Returns a summary with type and region distributions. The complete
    neuron list is saved as a Parquet artifact — load it with
    ``pd.read_parquet(artifact_path)`` for full analysis.

    Parameters
    ----------
    cell_type : str
        Cell type annotation to search for (e.g. "L2/3 IT",
        "MBON14", "KC-ab").
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "fanc", "hemibrain".
    region : str, optional
        Brain region filter. Only neurons in this region are
        included.

    Returns
    -------
    dict
        NeuronsByTypeResponse with artifact_manifest pointing to the
        full neuron table on disk, plus n_total, type_distribution,
        and region_distribution summaries.
    """
    return universal.get_neurons_by_type(cell_type, dataset, region)


@mcp.tool()
def get_region_connectivity(
    dataset: str,
    source_region: str | None = None,
    target_region: str | None = None,
) -> dict:
    """Get region-to-region synaptic connectivity.

    Returns a summary with the top 5 connections by synapse count.
    The complete region-pair table is saved as a Parquet artifact in
    long format — load it with ``pd.read_parquet(artifact_path)``
    for full analysis. Convert to matrix with ``pivot_table``.

    Parameters
    ----------
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "fanc", "hemibrain".
    source_region : str, optional
        Filter to connections from this region.
    target_region : str, optional
        Filter to connections to this region.

    Returns
    -------
    dict
        RegionConnectivityResponse with artifact_manifest pointing to
        the full region-pair table on disk, plus n_regions,
        top_5_connections, and total_synapses.
    """
    return universal.get_region_connectivity(dataset, source_region, target_region)


def main() -> None:
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
