"""FastMCP server entry point for Connectomics MCP."""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from connectomics_mcp.tools import cave_specific, neuprint_specific, universal

logger = logging.getLogger(__name__)

mcp = FastMCP("connectomics-mcp")


@mcp.tool()
def resolve_nucleus_ids(nucleus_ids: list[int], dataset: str) -> dict:
    """Resolve MICrONS nucleus IDs to current pt_root_ids.

    Nucleus IDs are the stable cross-version cell identifiers in
    MICrONS (minnie65). Unlike pt_root_ids, which change with
    proofreading, nucleus IDs never change. Use this tool to convert
    nucleus IDs to current pt_root_ids before passing them to other
    tools.

    A ``merge_conflict`` result means the segment at this nucleus's
    location contains multiple detected nuclei — the segmentation
    likely has an unresolved merge error.

    A ``no_segment`` result means no segment was found at this
    nucleus's position.

    Parameters
    ----------
    nucleus_ids : list[int]
        Nucleus IDs to resolve.
    dataset : str
        Dataset to query. Must be "minnie65".

    Returns
    -------
    dict
        NucleusResolutionResult with per-nucleus resolution status,
        pt_root_ids, and conflict information.
    """
    return cave_specific.resolve_nucleus_ids(nucleus_ids, dataset)


@mcp.tool()
def get_neuron_info(
    neuron_id: int | str, dataset: str, nucleus_id: int | None = None
) -> dict:
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
    nucleus_id : int, optional
        MICrONS nucleus ID (minnie65 only). If provided, resolves
        to the current pt_root_id before querying. Nucleus IDs are
        stable cross-version identifiers.

    Returns
    -------
    dict
        NeuronInfoResponse with keys: neuron_id, dataset, cell_type,
        cell_class, region, soma_position_nm, n_pre_synapses,
        n_post_synapses, proofread, materialization_version,
        neuroglancer_url, warnings.
    """
    return universal.get_neuron_info(neuron_id, dataset, nucleus_id=nucleus_id)


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
def query_annotation_table(
    dataset: str,
    table_name: str,
    filter_equal_dict: dict | None = None,
    filter_in_dict: dict | None = None,
) -> dict:
    """Query a CAVE annotation table.

    Returns a summary with row count and schema description. The
    complete query result is saved as a Parquet artifact — load it
    with ``pd.read_parquet(artifact_path)`` for full analysis.

    Only available for CAVE datasets (minnie65, flywire, fanc).

    Parameters
    ----------
    dataset : str
        Dataset to query. Must be a CAVE dataset.
    table_name : str
        Name of the annotation table (e.g.
        "aibs_metamodel_celltypes_v661", "nucleus_detection_v0").
    filter_equal_dict : dict, optional
        Equality filters (column → value).
    filter_in_dict : dict, optional
        Membership filters (column → list of values).

    Returns
    -------
    dict
        AnnotationTableResponse with artifact_manifest pointing to
        the full table on disk, plus n_total and schema_description.
    """
    return cave_specific.query_annotation_table(
        dataset, table_name, filter_equal_dict, filter_in_dict
    )


@mcp.tool()
def get_edit_history(neuron_id: int, dataset: str) -> dict:
    """Get edit history for a CAVE neuron.

    Returns a summary with edit count and timestamp range. The
    complete edit log is saved as a Parquet artifact — load it
    with ``pd.read_parquet(artifact_path)`` for full analysis.

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
        EditHistoryResponse with artifact_manifest pointing to
        the full edit log on disk, plus n_edits_total,
        first_edit_timestamp, and last_edit_timestamp.
    """
    return cave_specific.get_edit_history(neuron_id, dataset)


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


@mcp.tool()
def fetch_cypher(query: str, dataset: str) -> dict:
    """Execute a Cypher query against a neuPrint dataset.

    Returns a summary with row count and column names. The complete
    query result is saved as a Parquet artifact — load it with
    ``pd.read_parquet(artifact_path)`` for full analysis.

    Only available for neuPrint datasets (hemibrain).

    Parameters
    ----------
    query : str
        Cypher query string.
    dataset : str
        Dataset to query. Must be a neuPrint dataset: "hemibrain".

    Returns
    -------
    dict
        CypherQueryResponse with artifact_manifest pointing to
        the full query result on disk, plus n_rows and columns.
    """
    return neuprint_specific.fetch_cypher(query, dataset)


@mcp.tool()
def get_synapse_compartments(
    neuron_id: int | str, dataset: str, direction: str = "input"
) -> dict:
    """Get synapse distribution across ROI compartments for a neuron.

    Returns per-ROI synapse counts and fractions for the specified
    direction. Response is inherently small (one entry per ROI).

    Only available for neuPrint datasets (hemibrain).

    Parameters
    ----------
    neuron_id : int | str
        Body ID of the neuron.
    dataset : str
        Dataset to query. Must be a neuPrint dataset: "hemibrain".
    direction : str
        "input" for post-synaptic or "output" for pre-synaptic
        (default "input").

    Returns
    -------
    dict
        SynapseCompartmentResponse with per-ROI compartment stats
        and total synapse count.
    """
    return neuprint_specific.get_synapse_compartments(
        neuron_id, dataset, direction
    )


def main() -> None:
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
