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
        (minnie65, flywire) or a body ID for neuPrint
        datasets (hemibrain).
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "hemibrain".
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
        ConnectivityResponse with artifact_manifest pointing to the
        full partner table on disk, plus summary statistics and
        3-item orientation samples.
    """
    return universal.get_connectivity(neuron_id, dataset, direction)


@mcp.tool()
def get_bulk_connectivity(
    root_ids: list[int], dataset: str, direction: str = "both"
) -> dict:
    """Get synaptic connectivity for multiple neurons in bulk.

    Fetches all connections involving the given root IDs and saves
    the complete edge table as a single Parquet artifact. Much faster
    than calling ``get_connectivity`` per neuron for circuit analysis.

    The artifact has columns: ``pre_root_id``, ``post_root_id``,
    ``syn_count``, ``neuropil``. Load with
    ``pd.read_parquet(artifact_path)`` for full analysis.

    Parameters
    ----------
    root_ids : list[int]
        Neuron identifiers to query.
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "hemibrain".
    direction : str
        "pre" (outgoing), "post" (incoming), or "both" (default).

    Returns
    -------
    dict
        BulkConnectivityResponse with artifact_manifest pointing to
        the full edge table on disk, plus n_edges, total_synapses,
        and cached flag.
    """
    return universal.get_bulk_connectivity(root_ids, dataset, direction)


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
        "hemibrain".

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
    Only available for CAVE datasets (minnie65, flywire).

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
        "hemibrain".
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
def get_cell_type_taxonomy(dataset: str) -> dict:
    """Get the cell type taxonomy/hierarchy for a dataset.

    Returns the full classification structure showing how cell types
    are organized. Use this FIRST when working with an unfamiliar
    dataset to understand its naming conventions before searching
    for specific cell types.

    For FlyWire, the hierarchy has 4 levels:
      super_class → cell_class → cell_sub_class → cell_type
    For hemibrain, types are a flat namespace.
    For MICrONS, cell types are a flat list.

    The response includes example lineages showing the full path
    from broadest to finest level for representative neurons.

    Parameters
    ----------
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "hemibrain".

    Returns
    -------
    dict
        CellTypeTaxonomyResponse with levels (each with top values
        and neuron counts), example_lineages, and n_total_neurons.
    """
    return universal.get_cell_type_taxonomy(dataset)


@mcp.tool()
def search_cell_types(query: str, dataset: str) -> dict:
    """Search for cell types matching a query string.

    Performs case-insensitive substring matching across all available
    annotation levels. Use this tool FIRST to discover what cell type
    names exist in a dataset before calling ``get_neurons_by_type``.

    For FlyWire, searches across all hierarchy levels (super_class,
    cell_class, cell_sub_class, cell_type). For hemibrain, searches
    neuron type annotations. For MICrONS, searches the cell type table.

    Examples: search for "EPG" to find EPG neurons, "ring" to find
    ring neurons, "compass" or "head_direction" for compass/HD neurons.

    Parameters
    ----------
    query : str
        Search string — case-insensitive substring match.
    dataset : str
        Dataset to search. Supported: "minnie65", "flywire",
        "hemibrain".

    Returns
    -------
    dict
        CellTypeSearchResponse with matches (cell_type name,
        classification_level, n_neurons) sorted by relevance.
    """
    return universal.search_cell_types(query, dataset)


@mcp.tool()
def get_neurons_by_type(
    cell_type: str, dataset: str, region: str | None = None
) -> dict:
    """Get all neurons matching a cell type annotation.

    Returns a summary with type and region distributions. The complete
    neuron list is saved as a Parquet artifact — load it with
    ``pd.read_parquet(artifact_path)`` for full analysis.

    For FlyWire, uses progressive matching: exact match at cell_type
    level → exact at any hierarchy level → case-insensitive →
    substring. If no match is found, the response suggests using
    ``search_cell_types()`` for discovery.

    Tip: if you're unsure of the exact cell type name, call
    ``search_cell_types()`` first to discover available names.

    Parameters
    ----------
    cell_type : str
        Cell type annotation to search for (e.g. "L2/3 IT",
        "MBON14", "KC-ab", "EPG").
    dataset : str
        Dataset to query. Supported: "minnie65", "flywire",
        "hemibrain".
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

    Only available for CAVE datasets (minnie65, flywire).

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

    Only available for CAVE datasets (minnie65, flywire).

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
        "hemibrain".
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


@mcp.tool()
def get_coregistration(
    neuron_id: int, dataset: str, by: str = "root_id"
) -> dict:
    """Get EM-to-functional imaging coregistration for a neuron.

    Maps EM neurons to 2-photon functional imaging units (session,
    scan, unit). Complete results saved as Parquet artifact — load
    with ``pd.read_parquet(artifact_path)``.

    Only available for MICrONS (minnie65).

    Parameters
    ----------
    neuron_id : int
        Root ID or nucleus ID.
    dataset : str
        Must be "minnie65".
    by : str
        "root_id" or "nucleus_id" (default "root_id").

    Returns
    -------
    dict
        CoregistrationResponse with artifact_manifest, n_units,
        score_distribution, and sessions list.
    """
    return cave_specific.get_coregistration(neuron_id, dataset, by)


@mcp.tool()
def get_functional_properties(
    neuron_id: int,
    dataset: str,
    by: str = "root_id",
    coregistration_source: str = "auto_phase3",
) -> dict:
    """Get digital twin functional properties for a neuron.

    Returns orientation/direction selectivity, receptive field
    centers, and model performance metrics. Complete results saved
    as Parquet artifact.

    Only available for MICrONS (minnie65).

    Parameters
    ----------
    neuron_id : int
        Root ID or nucleus ID.
    dataset : str
        Must be "minnie65".
    by : str
        "root_id" or "nucleus_id" (default "root_id").
    coregistration_source : str
        Table variant: "auto_phase3" (default, largest coverage),
        "coreg_v4" (manual), or "apl_vess".

    Returns
    -------
    dict
        FunctionalPropertiesResponse with artifact_manifest,
        ori_selectivity_distribution, dir_selectivity_distribution.
    """
    return cave_specific.get_functional_properties(
        neuron_id, dataset, by, coregistration_source
    )


@mcp.tool()
def get_synapse_targets(
    root_id: int, dataset: str, direction: str = "post"
) -> dict:
    """Get per-synapse structural target predictions for a neuron.

    Classifies each synapse as targeting spine, shaft, or soma.
    Complete results saved as Parquet artifact.

    Only available for MICrONS (minnie65).

    Parameters
    ----------
    root_id : int
        Root ID of the neuron.
    dataset : str
        Must be "minnie65".
    direction : str
        "post" for synapses onto this neuron (default),
        "pre" for synapses from this neuron.

    Returns
    -------
    dict
        SynapseTargetsResponse with artifact_manifest, n_synapses,
        target_distribution (spine/shaft/soma counts).
    """
    return cave_specific.get_synapse_targets(root_id, dataset, direction)


@mcp.tool()
def get_multi_input_spines(
    root_id: int, dataset: str, direction: str = "post"
) -> dict:
    """Get multi-input spine predictions for a neuron (deprecated).

    Deprecated: prefer ``get_synapse_targets`` for general use.
    Identifies spines receiving >1 input synapse, grouped by
    shared postsynaptic compartment. Complete results saved as
    Parquet artifact.

    Only available for MICrONS (minnie65).

    Parameters
    ----------
    root_id : int
        Root ID of the neuron.
    dataset : str
        Must be "minnie65".
    direction : str
        "post" for spines on this neuron (default),
        "pre" for spines from this neuron.

    Returns
    -------
    dict
        MultiInputSpinesResponse with artifact_manifest, n_synapses,
        n_spine_groups, target_distribution.
    """
    return cave_specific.get_multi_input_spines(root_id, dataset, direction)


@mcp.tool()
def get_cell_mtypes(
    dataset: str,
    neuron_id: int | None = None,
    by: str = "root_id",
    cell_type: str | None = None,
) -> dict:
    """Get morphological cell type (mtype) classifications.

    24 types based on dendritic features and connectivity motifs.
    Excitatory: L2a-L6wm. Inhibitory: PTC, DTC, STC, ITC.
    Complete results saved as Parquet artifact.

    Only available for MICrONS (minnie65).

    Parameters
    ----------
    dataset : str
        Must be "minnie65".
    neuron_id : int, optional
        Root ID or nucleus ID for single-neuron lookup.
    by : str
        "root_id" or "nucleus_id" (default "root_id").
    cell_type : str, optional
        Filter by mtype (e.g. "L2a", "DTC").

    Returns
    -------
    dict
        CellMtypesResponse with artifact_manifest, n_total,
        classification_system_distribution, cell_type_distribution.
    """
    return cave_specific.get_cell_mtypes(dataset, neuron_id, by, cell_type)


@mcp.tool()
def get_functional_area(
    dataset: str,
    neuron_id: int | None = None,
    by: str = "root_id",
    area: str | None = None,
) -> dict:
    """Get functional brain area assignments for MICrONS neurons.

    Areas: V1, AL, RL, LM — inferred from 2-photon imaging
    boundaries. The ``value`` column is distance to nearest area
    boundary in micrometers (higher = more confident). Complete
    results saved as Parquet artifact.

    Only available for MICrONS (minnie65).

    Parameters
    ----------
    dataset : str
        Must be "minnie65".
    neuron_id : int, optional
        Root ID or nucleus ID for single-neuron lookup.
    by : str
        "root_id" or "nucleus_id" (default "root_id").
    area : str, optional
        Filter by area label: "V1", "AL", "RL", or "LM".

    Returns
    -------
    dict
        FunctionalAreaResponse with artifact_manifest, n_total,
        area_distribution.
    """
    return cave_specific.get_functional_area(dataset, neuron_id, by, area)


@mcp.tool()
def get_bulk_coregistration(root_ids: list[int], dataset: str) -> dict:
    """Get EM-to-functional coregistration for multiple neurons in bulk.

    Fetches coregistration data for all given root IDs and saves the
    complete result as a single Parquet artifact. Load with
    ``pd.read_parquet(artifact_path)`` for full analysis.

    Only available for MICrONS (minnie65).

    Parameters
    ----------
    root_ids : list[int]
        Root IDs to query. All must be current.
    dataset : str
        Must be "minnie65".

    Returns
    -------
    dict
        BulkCoregistrationResponse with artifact_manifest, n_units,
        score_distribution, and sessions list.
    """
    return cave_specific.get_bulk_coregistration(root_ids, dataset)


@mcp.tool()
def get_bulk_functional_properties(
    root_ids: list[int],
    dataset: str,
    coregistration_source: str = "auto_phase3",
) -> dict:
    """Get digital twin functional properties for multiple neurons in bulk.

    Fetches orientation/direction selectivity and model performance
    metrics for all given root IDs. Complete results saved as
    Parquet artifact.

    Only available for MICrONS (minnie65).

    Parameters
    ----------
    root_ids : list[int]
        Root IDs to query. All must be current.
    dataset : str
        Must be "minnie65".
    coregistration_source : str
        Table variant: "auto_phase3" (default), "coreg_v4",
        or "apl_vess".

    Returns
    -------
    dict
        BulkFunctionalPropertiesResponse with artifact_manifest,
        ori_selectivity_distribution, dir_selectivity_distribution.
    """
    return cave_specific.get_bulk_functional_properties(
        root_ids, dataset, coregistration_source
    )


@mcp.tool()
def get_bulk_synapse_targets(
    root_ids: list[int], dataset: str, direction: str = "post"
) -> dict:
    """Get per-synapse structural target predictions for multiple neurons.

    Classifies each synapse as targeting spine, shaft, or soma for
    all given root IDs. Complete results saved as Parquet artifact.

    Only available for MICrONS (minnie65).

    Parameters
    ----------
    root_ids : list[int]
        Root IDs to query. All must be current.
    dataset : str
        Must be "minnie65".
    direction : str
        "post" for synapses onto these neurons (default),
        "pre" for synapses from these neurons.

    Returns
    -------
    dict
        BulkSynapseTargetsResponse with artifact_manifest,
        n_synapses, target_distribution.
    """
    return cave_specific.get_bulk_synapse_targets(root_ids, dataset, direction)


@mcp.tool()
def get_bulk_functional_area(root_ids: list[int], dataset: str) -> dict:
    """Get functional brain area assignments for multiple neurons in bulk.

    Returns area labels (V1, AL, RL, LM) with boundary distances for
    all given root IDs. Complete results saved as Parquet artifact.

    Only available for MICrONS (minnie65).

    Parameters
    ----------
    root_ids : list[int]
        Root IDs to query. All must be current.
    dataset : str
        Must be "minnie65".

    Returns
    -------
    dict
        BulkFunctionalAreaResponse with artifact_manifest, n_total,
        area_distribution.
    """
    return cave_specific.get_bulk_functional_area(root_ids, dataset)


def main() -> None:
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO)
    mcp.run()


if __name__ == "__main__":
    main()
