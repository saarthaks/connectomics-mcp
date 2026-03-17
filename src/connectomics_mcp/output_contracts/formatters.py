"""Format raw backend responses into Pydantic schema instances."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from connectomics_mcp.artifacts.writer import save_artifact
from connectomics_mcp.neuroglancer.url_builder import build_neuroglancer_url
from connectomics_mcp.output_contracts.schemas import (
    AnnotationTableResponse,
    BulkCoregistrationResponse,
    BulkConnectivityResponse,
    BulkFunctionalAreaResponse,
    BulkFunctionalPropertiesResponse,
    BulkSynapseTargetsResponse,
    CellMtypesResponse,
    CellTypeMatch,
    CellTypeSearchResponse,
    CellTypeTaxonomyResponse,
    TaxonomyLevel,
    CompartmentStats,
    ConnectivityResponse,
    CoregistrationResponse,
    CypherQueryResponse,
    EditHistoryResponse,
    FunctionalAreaResponse,
    FunctionalPropertiesResponse,
    MultiInputSpinesResponse,
    NeuronInfoResponse,
    NeuroglancerUrlResponse,
    NeuronsByTypeResponse,
    NucleusResolution,
    NucleusResolutionResult,
    NucleusResolutionStatus,
    ProofreadingStatusResponse,
    RegionConnectivityResponse,
    RootIdValidationResponse,
    RootIdValidationResult,
    SynapseCompartmentResponse,
    SynapseTargetsResponse,
    SynapticPartnerSample,
)

logger = logging.getLogger(__name__)


def format_neuron_info(raw: dict[str, Any], dataset: str) -> NeuronInfoResponse:
    """Convert a raw backend neuron info dict to NeuronInfoResponse.

    Parameters
    ----------
    raw : dict
        Raw dict returned by a backend's get_neuron_info method.
    dataset : str
        Dataset name for the Neuroglancer URL.

    Returns
    -------
    NeuronInfoResponse
        Validated, serializable response.
    """
    neuron_id = raw["neuron_id"]

    # Build Neuroglancer URL for this neuron
    try:
        ngl_url = build_neuroglancer_url([neuron_id], dataset)
    except KeyError:
        ngl_url = ""
        logger.warning("No Neuroglancer config for dataset %s", dataset)

    return NeuronInfoResponse(
        neuron_id=neuron_id,
        dataset=dataset,
        cell_type=raw.get("cell_type"),
        cell_class=raw.get("cell_class"),
        region=raw.get("region"),
        soma_position_nm=raw.get("soma_position_nm"),
        n_pre_synapses=raw.get("n_pre_synapses"),
        n_post_synapses=raw.get("n_post_synapses"),
        proofread=raw.get("proofread"),
        materialization_version=raw.get("materialization_version"),
        neuroglancer_url=ngl_url,
        neurotransmitter_type=raw.get("neurotransmitter_type"),
        classification_hierarchy=raw.get("classification_hierarchy"),
        warnings=raw.get("warnings", []),
    )


def _weight_distribution(series: pd.Series) -> dict:
    """Compute summary statistics for a weight series."""
    if series.empty:
        return {"mean": 0.0, "median": 0.0, "max": 0, "p90": 0.0}
    return {
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "max": int(series.max()),
        "p90": round(float(np.percentile(series, 90)), 2),
    }


def _top_samples(
    df: pd.DataFrame, n: int = 3
) -> list[SynapticPartnerSample]:
    """Extract the top-N partners by synapse count as sample objects."""
    if df.empty:
        return []
    top = df.nlargest(min(n, len(df)), "n_synapses")
    samples = []
    for _, row in top.iterrows():
        samples.append(
            SynapticPartnerSample(
                partner_id=int(row["partner_id"]),
                partner_type=(
                    str(row["partner_type"])
                    if "partner_type" in row.index and pd.notna(row["partner_type"])
                    else None
                ),
                n_synapses=int(row["n_synapses"]),
                weight_normalized=(
                    float(row["weight_normalized"])
                    if pd.notna(row.get("weight_normalized"))
                    else None
                ),
            )
        )
    return samples


def format_connectivity(
    raw: dict[str, Any], dataset: str
) -> ConnectivityResponse:
    """Convert raw backend connectivity data to ConnectivityResponse.

    Saves the complete partner DataFrame as a Parquet artifact and
    returns a lightweight summary with 3-item orientation samples.

    Parameters
    ----------
    raw : dict
        Raw dict from backend with key ``partners_df`` (a pd.DataFrame).
    dataset : str
        Dataset name.

    Returns
    -------
    ConnectivityResponse
        Summary + artifact manifest for the full partner table.
    """
    neuron_id = raw["neuron_id"]
    partners_df: pd.DataFrame = raw["partners_df"]
    mat_version = raw.get("materialization_version")

    # Build per-partner Neuroglancer URLs
    if not partners_df.empty and "neuroglancer_url" in partners_df.columns:
        urls = []
        for _, row in partners_df.iterrows():
            try:
                url = build_neuroglancer_url(
                    [neuron_id, int(row["partner_id"])], dataset
                )
            except KeyError:
                url = ""
            urls.append(url)
        partners_df = partners_df.copy()
        partners_df["neuroglancer_url"] = urls

    # Split by direction
    upstream_df = partners_df[partners_df["direction"] == "upstream"] if not partners_df.empty else pd.DataFrame()
    downstream_df = partners_df[partners_df["direction"] == "downstream"] if not partners_df.empty else pd.DataFrame()

    # Compute distributions over synapse counts (not normalized weights)
    upstream_dist = _weight_distribution(upstream_df["n_synapses"]) if not upstream_df.empty else _weight_distribution(pd.Series(dtype=float))
    downstream_dist = _weight_distribution(downstream_df["n_synapses"]) if not downstream_df.empty else _weight_distribution(pd.Series(dtype=float))

    # 3-item orientation samples
    upstream_sample = _top_samples(upstream_df, 3)
    downstream_sample = _top_samples(downstream_df, 3)

    # Save artifact
    manifest = save_artifact(
        df=partners_df,
        tool="connectivity",
        dataset=dataset,
        neuron_id=neuron_id,
        materialization_version=mat_version,
    )

    # Overall Neuroglancer URL
    try:
        ngl_url = build_neuroglancer_url([neuron_id], dataset)
    except KeyError:
        ngl_url = ""

    # Compute neurotransmitter distribution if NT columns present
    nt_distribution = None
    nt_cols = ["gaba", "ach", "glut", "oct", "ser", "da"]
    if not partners_df.empty and "partner_nt_type" in partners_df.columns:
        nt_counts = partners_df["partner_nt_type"].dropna().value_counts().to_dict()
        if nt_counts:
            nt_distribution = {str(k): int(v) for k, v in nt_counts.items()}

    return ConnectivityResponse(
        neuron_id=neuron_id,
        dataset=dataset,
        n_upstream_total=len(upstream_df),
        n_downstream_total=len(downstream_df),
        upstream_weight_distribution=upstream_dist,
        downstream_weight_distribution=downstream_dist,
        upstream_sample=upstream_sample,
        downstream_sample=downstream_sample,
        neuroglancer_url=ngl_url,
        neurotransmitter_distribution=nt_distribution,
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_bulk_connectivity(
    raw: dict[str, Any], dataset: str, extra_key: str
) -> BulkConnectivityResponse:
    """Convert raw bulk connectivity data to BulkConnectivityResponse.

    Saves the complete edge DataFrame as a Parquet artifact and
    returns a lightweight summary.

    Parameters
    ----------
    raw : dict
        Raw dict with key ``edges_df`` (a pd.DataFrame).
    dataset : str
        Dataset name.
    extra_key : str
        Content-addressable hash for artifact caching.

    Returns
    -------
    BulkConnectivityResponse
        Summary + artifact manifest for the full edge table.
    """
    edges_df: pd.DataFrame = raw["edges_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=edges_df,
        tool="bulk_connectivity",
        dataset=dataset,
        neuron_id=None,
        materialization_version=mat_version,
        extra_key=extra_key,
    )

    total_synapses = int(edges_df["syn_count"].sum()) if not edges_df.empty else 0

    return BulkConnectivityResponse(
        dataset=dataset,
        n_root_ids=raw.get("n_root_ids", 0),
        direction=raw.get("direction", "both"),
        n_edges=len(edges_df),
        total_synapses=total_synapses,
        cached=False,
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_validate_root_ids(
    raw: dict[str, Any], dataset: str
) -> RootIdValidationResponse:
    """Convert raw backend validation results to RootIdValidationResponse.

    Parameters
    ----------
    raw : dict
        Raw dict from backend's validate_root_ids method.
    dataset : str
        Dataset name.

    Returns
    -------
    RootIdValidationResponse
        Validated, serializable response.
    """
    results = [
        RootIdValidationResult(
            root_id=r["root_id"],
            is_current=r["is_current"],
            last_edit_timestamp=r.get("last_edit_timestamp"),
            suggested_current_id=r.get("suggested_current_id"),
        )
        for r in raw.get("results", [])
    ]

    n_stale = sum(1 for r in results if not r.is_current)

    return RootIdValidationResponse(
        dataset=dataset,
        materialization_version=raw.get("materialization_version") or 0,
        results=results,
        n_stale=n_stale,
        warnings=raw.get("warnings", []),
    )


def format_proofreading_status(
    raw: dict[str, Any], dataset: str
) -> ProofreadingStatusResponse:
    """Convert raw backend proofreading data to ProofreadingStatusResponse.

    Parameters
    ----------
    raw : dict
        Raw dict from backend's get_proofreading_status method.
    dataset : str
        Dataset name.

    Returns
    -------
    ProofreadingStatusResponse
        Validated, serializable response.
    """
    return ProofreadingStatusResponse(
        neuron_id=raw["neuron_id"],
        dataset=dataset,
        axon_proofread=raw.get("axon_proofread"),
        dendrite_proofread=raw.get("dendrite_proofread"),
        strategy_axon=raw.get("strategy_axon"),
        strategy_dendrite=raw.get("strategy_dendrite"),
        n_edits=raw.get("n_edits"),
        last_edit_timestamp=raw.get("last_edit_timestamp"),
        warnings=raw.get("warnings", []),
    )


def format_neuroglancer_url(
    url: str,
    dataset: str,
    segment_ids: list[int | str],
    layers: list[str],
    coordinate_space: str,
) -> NeuroglancerUrlResponse:
    """Format a Neuroglancer URL into NeuroglancerUrlResponse.

    Parameters
    ----------
    url : str
        The fully encoded Neuroglancer URL.
    dataset : str
        Dataset name.
    segment_ids : list[int | str]
        Segment IDs included in the URL.
    layers : list[str]
        Layer names included in the state.
    coordinate_space : str
        Coordinate space (e.g. "nm").

    Returns
    -------
    NeuroglancerUrlResponse
    """
    return NeuroglancerUrlResponse(
        url=url,
        dataset=dataset,
        n_segments=len(segment_ids),
        layers_included=layers,
        coordinate_space=coordinate_space,
    )


def format_region_connectivity(
    raw: dict[str, Any], dataset: str
) -> RegionConnectivityResponse:
    """Convert raw backend region connectivity data to RegionConnectivityResponse.

    Saves the complete region-pair DataFrame as a Parquet artifact and
    returns a lightweight summary with the top 5 connections.

    Parameters
    ----------
    raw : dict
        Raw dict from backend with key ``region_df`` (a pd.DataFrame).
    dataset : str
        Dataset name.

    Returns
    -------
    RegionConnectivityResponse
        Summary + artifact manifest for the full region connectivity table.
    """
    region_df: pd.DataFrame = raw["region_df"]
    mat_version = raw.get("materialization_version")

    # Save artifact
    manifest = save_artifact(
        df=region_df,
        tool="region_connectivity",
        dataset=dataset,
        neuron_id=None,
        materialization_version=mat_version,
    )

    # Compute summary stats
    n_regions = 0
    total_synapses = 0
    top_5: list[dict] = []

    if not region_df.empty:
        all_regions = set(region_df["source_region"].unique()) | set(
            region_df["target_region"].unique()
        )
        n_regions = len(all_regions)
        total_synapses = int(region_df["n_synapses"].sum())

        top_rows = region_df.nlargest(min(5, len(region_df)), "n_synapses")
        for _, row in top_rows.iterrows():
            top_5.append({
                "source_region": row["source_region"],
                "target_region": row["target_region"],
                "n_synapses": int(row["n_synapses"]),
            })

    return RegionConnectivityResponse(
        dataset=dataset,
        n_regions=n_regions,
        top_5_connections=top_5,
        total_synapses=total_synapses,
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_cell_type_search(
    raw: dict[str, Any], dataset: str
) -> CellTypeSearchResponse:
    """Convert raw backend cell type search results to CellTypeSearchResponse.

    Parameters
    ----------
    raw : dict
        Raw dict from backend's search_cell_types method.
    dataset : str
        Dataset name.

    Returns
    -------
    CellTypeSearchResponse
        Validated, serializable response.
    """
    matches = [
        CellTypeMatch(
            cell_type=m["cell_type"],
            classification_level=m.get("classification_level"),
            n_neurons=m.get("n_neurons", 0),
        )
        for m in raw.get("matches", [])
    ]

    return CellTypeSearchResponse(
        dataset=dataset,
        query=raw["query"],
        n_matches=len(matches),
        matches=matches,
        taxonomy_hints=raw.get("taxonomy_hints", []),
        warnings=raw.get("warnings", []),
    )


def format_cell_type_taxonomy(
    raw: dict[str, Any], dataset: str
) -> CellTypeTaxonomyResponse:
    """Convert raw backend taxonomy data to CellTypeTaxonomyResponse."""
    levels = [
        TaxonomyLevel(
            level_name=lv["level_name"],
            values=lv.get("values", []),
        )
        for lv in raw.get("levels", [])
    ]

    return CellTypeTaxonomyResponse(
        dataset=dataset,
        n_total_neurons=raw.get("n_total_neurons", 0),
        levels=levels,
        example_lineages=raw.get("example_lineages", []),
        warnings=raw.get("warnings", []),
    )


def format_neurons_by_type(
    raw: dict[str, Any], dataset: str
) -> NeuronsByTypeResponse:
    """Convert raw backend neurons-by-type data to NeuronsByTypeResponse.

    Saves the complete neuron DataFrame as a Parquet artifact and
    returns a lightweight summary with type and region distributions.

    Parameters
    ----------
    raw : dict
        Raw dict from backend with key ``neurons_df`` (a pd.DataFrame).
    dataset : str
        Dataset name.

    Returns
    -------
    NeuronsByTypeResponse
        Summary + artifact manifest for the full neuron table.
    """
    neurons_df: pd.DataFrame = raw["neurons_df"]
    mat_version = raw.get("materialization_version")

    # Save artifact
    manifest = save_artifact(
        df=neurons_df,
        tool="neurons_by_type",
        dataset=dataset,
        neuron_id=None,
        materialization_version=mat_version,
    )

    # Compute distributions
    type_dist: dict = {}
    if not neurons_df.empty and "cell_type" in neurons_df.columns:
        type_dist = neurons_df["cell_type"].dropna().value_counts().to_dict()

    region_dist: dict = {}
    if not neurons_df.empty and "region" in neurons_df.columns:
        region_dist = neurons_df["region"].dropna().value_counts().to_dict()

    return NeuronsByTypeResponse(
        dataset=dataset,
        query_cell_type=raw["query_cell_type"],
        query_region=raw.get("query_region"),
        n_total=len(neurons_df),
        type_distribution=type_dist,
        region_distribution=region_dist,
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_annotation_table(
    raw: dict[str, Any], dataset: str
) -> AnnotationTableResponse:
    """Convert raw backend annotation table data to AnnotationTableResponse.

    Saves the complete table DataFrame as a Parquet artifact and
    returns a lightweight summary.

    Parameters
    ----------
    raw : dict
        Raw dict from backend with key ``table_df`` (a pd.DataFrame).
    dataset : str
        Dataset name.

    Returns
    -------
    AnnotationTableResponse
        Summary + artifact manifest for the full annotation table.
    """
    table_df: pd.DataFrame = raw["table_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=table_df,
        tool="annotation_table",
        dataset=dataset,
        neuron_id=None,
        materialization_version=mat_version,
        extra_key=raw["table_name"],
    )

    return AnnotationTableResponse(
        dataset=dataset,
        table_name=raw["table_name"],
        n_total=len(table_df),
        schema_description=raw.get("schema_description", ""),
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_edit_history(
    raw: dict[str, Any], dataset: str
) -> EditHistoryResponse:
    """Convert raw backend edit history data to EditHistoryResponse.

    Saves the complete edit log DataFrame as a Parquet artifact and
    returns a lightweight summary.

    Parameters
    ----------
    raw : dict
        Raw dict from backend with key ``edits_df`` (a pd.DataFrame).
    dataset : str
        Dataset name.

    Returns
    -------
    EditHistoryResponse
        Summary + artifact manifest for the full edit history.
    """
    edits_df: pd.DataFrame = raw["edits_df"]
    mat_version = raw.get("materialization_version")
    neuron_id = raw["neuron_id"]

    manifest = save_artifact(
        df=edits_df,
        tool="edit_history",
        dataset=dataset,
        neuron_id=neuron_id,
        materialization_version=mat_version,
    )

    first_edit = None
    last_edit = None
    if not edits_df.empty and "timestamp" in edits_df.columns:
        first_edit = str(edits_df["timestamp"].min())
        last_edit = str(edits_df["timestamp"].max())

    return EditHistoryResponse(
        neuron_id=neuron_id,
        dataset=dataset,
        n_edits_total=len(edits_df),
        first_edit_timestamp=first_edit,
        last_edit_timestamp=last_edit,
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_nucleus_resolution(
    raw: dict[str, Any], dataset: str
) -> NucleusResolutionResult:
    """Convert raw backend nucleus resolution data to NucleusResolutionResult.

    Parameters
    ----------
    raw : dict
        Raw dict from backend's resolve_nucleus_ids method.
    dataset : str
        Dataset name.

    Returns
    -------
    NucleusResolutionResult
        Validated, serializable response.
    """
    resolutions = [
        NucleusResolution(
            nucleus_id=r["nucleus_id"],
            pt_root_id=r.get("pt_root_id"),
            resolution_status=NucleusResolutionStatus(r["resolution_status"]),
            conflicting_nucleus_ids=r.get("conflicting_nucleus_ids", []),
            materialization_version=r["materialization_version"],
        )
        for r in raw.get("resolutions", [])
    ]

    return NucleusResolutionResult(
        dataset=dataset,
        materialization_version=raw.get("materialization_version", 0),
        resolutions=resolutions,
        n_resolved=raw.get("n_resolved", 0),
        n_merge_conflicts=raw.get("n_merge_conflicts", 0),
        n_no_segment=raw.get("n_no_segment", 0),
        warnings=raw.get("warnings", []),
    )


def format_cypher_query(
    raw: dict[str, Any], dataset: str
) -> CypherQueryResponse:
    """Convert raw backend Cypher query result to CypherQueryResponse.

    Saves the complete result DataFrame as a Parquet artifact and
    returns a lightweight summary.

    Parameters
    ----------
    raw : dict
        Raw dict from backend with key ``result_df`` (a pd.DataFrame).
    dataset : str
        Dataset name.

    Returns
    -------
    CypherQueryResponse
        Summary + artifact manifest for the full query result.
    """
    result_df: pd.DataFrame = raw["result_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=result_df,
        tool="cypher",
        dataset=dataset,
        neuron_id=None,
        materialization_version=mat_version,
    )

    return CypherQueryResponse(
        dataset=dataset,
        query=raw["query"],
        n_rows=len(result_df),
        columns=list(result_df.columns) if not result_df.empty else [],
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_synapse_compartments(
    raw: dict[str, Any], dataset: str
) -> SynapseCompartmentResponse:
    """Convert raw backend compartment data to SynapseCompartmentResponse.

    Parameters
    ----------
    raw : dict
        Raw dict from backend's get_synapse_compartments method.
    dataset : str
        Dataset name.

    Returns
    -------
    SynapseCompartmentResponse
        Validated, serializable response.
    """
    compartments = [
        CompartmentStats(
            compartment=c["compartment"],
            n_synapses=c["n_synapses"],
            fraction=c["fraction"],
        )
        for c in raw.get("compartments", [])
    ]

    return SynapseCompartmentResponse(
        neuron_id=raw["neuron_id"],
        dataset=dataset,
        direction=raw.get("direction", "input"),
        compartments=compartments,
        n_total_synapses=raw.get("n_total_synapses", 0),
        warnings=raw.get("warnings", []),
    )


# ---------------------------------------------------------------------------
# MICrONS-specific formatters
# ---------------------------------------------------------------------------


def format_coregistration(
    raw: dict[str, Any], dataset: str
) -> CoregistrationResponse:
    """Convert raw coregistration data to CoregistrationResponse."""
    table_df: pd.DataFrame = raw["table_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=table_df,
        tool="coregistration",
        dataset=dataset,
        neuron_id=raw["neuron_id"],
        materialization_version=mat_version,
        extra_key=raw["table_name"],
    )

    score_dist = (
        _weight_distribution(table_df["score"])
        if not table_df.empty and "score" in table_df.columns
        else {}
    )
    sessions = (
        sorted(int(s) for s in table_df["session"].dropna().unique())
        if not table_df.empty and "session" in table_df.columns
        else []
    )

    return CoregistrationResponse(
        neuron_id=raw["neuron_id"],
        query_by=raw["by"],
        dataset=dataset,
        n_units=len(table_df),
        score_distribution=score_dist,
        sessions=sessions,
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_functional_properties(
    raw: dict[str, Any], dataset: str
) -> FunctionalPropertiesResponse:
    """Convert raw functional properties data to FunctionalPropertiesResponse."""
    table_df: pd.DataFrame = raw["table_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=table_df,
        tool="functional_properties",
        dataset=dataset,
        neuron_id=raw["neuron_id"],
        materialization_version=mat_version,
        extra_key=raw["table_name"],
    )

    ori_dist = (
        _weight_distribution(table_df["OSI"])
        if not table_df.empty and "OSI" in table_df.columns
        else {}
    )
    dir_dist = (
        _weight_distribution(table_df["DSI"])
        if not table_df.empty and "DSI" in table_df.columns
        else {}
    )

    return FunctionalPropertiesResponse(
        neuron_id=raw["neuron_id"],
        query_by=raw["by"],
        dataset=dataset,
        coregistration_source=raw["coregistration_source"],
        n_units=len(table_df),
        ori_selectivity_distribution=ori_dist,
        dir_selectivity_distribution=dir_dist,
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def _tag_distribution(df: pd.DataFrame, col: str) -> dict[str, int]:
    """Compute value counts for a string column, returning {value: count}."""
    if df.empty or col not in df.columns:
        return {}
    counts = df[col].dropna().value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def format_synapse_targets(
    raw: dict[str, Any], dataset: str
) -> SynapseTargetsResponse:
    """Convert raw synapse target data to SynapseTargetsResponse."""
    table_df: pd.DataFrame = raw["table_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=table_df,
        tool="synapse_targets",
        dataset=dataset,
        neuron_id=raw["neuron_id"],
        materialization_version=mat_version,
        extra_key=raw["table_name"],
    )

    return SynapseTargetsResponse(
        neuron_id=raw["neuron_id"],
        dataset=dataset,
        direction=raw["direction"],
        n_synapses=len(table_df),
        target_distribution=_tag_distribution(table_df, "tag"),
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_multi_input_spines(
    raw: dict[str, Any], dataset: str
) -> MultiInputSpinesResponse:
    """Convert raw multi-input spine data to MultiInputSpinesResponse."""
    table_df: pd.DataFrame = raw["table_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=table_df,
        tool="multi_input_spines",
        dataset=dataset,
        neuron_id=raw["neuron_id"],
        materialization_version=mat_version,
        extra_key=raw["table_name"],
    )

    n_spine_groups = (
        int(table_df["group_id"].nunique())
        if not table_df.empty and "group_id" in table_df.columns
        else 0
    )

    return MultiInputSpinesResponse(
        neuron_id=raw["neuron_id"],
        dataset=dataset,
        direction=raw["direction"],
        n_synapses=len(table_df),
        n_spine_groups=n_spine_groups,
        target_distribution=_tag_distribution(table_df, "tag"),
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_cell_mtypes(
    raw: dict[str, Any], dataset: str
) -> CellMtypesResponse:
    """Convert raw cell mtype data to CellMtypesResponse."""
    table_df: pd.DataFrame = raw["table_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=table_df,
        tool="cell_mtypes",
        dataset=dataset,
        neuron_id=raw.get("neuron_id"),
        materialization_version=mat_version,
        extra_key=raw["table_name"],
    )

    # Try both possible column names for the classification system
    cls_col = (
        "classification_system"
        if "classification_system" in table_df.columns
        else "classification-system"
        if "classification-system" in table_df.columns
        else None
    )

    return CellMtypesResponse(
        dataset=dataset,
        query_neuron_id=raw.get("neuron_id"),
        query_by=raw.get("by"),
        query_cell_type=raw.get("cell_type"),
        n_total=len(table_df),
        classification_system_distribution=(
            _tag_distribution(table_df, cls_col) if cls_col else {}
        ),
        cell_type_distribution=_tag_distribution(table_df, "cell_type"),
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_functional_area(
    raw: dict[str, Any], dataset: str
) -> FunctionalAreaResponse:
    """Convert raw functional area data to FunctionalAreaResponse."""
    table_df: pd.DataFrame = raw["table_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=table_df,
        tool="functional_area",
        dataset=dataset,
        neuron_id=raw.get("neuron_id"),
        materialization_version=mat_version,
        extra_key=raw["table_name"],
    )

    return FunctionalAreaResponse(
        dataset=dataset,
        query_neuron_id=raw.get("neuron_id"),
        query_by=raw.get("by"),
        query_area=raw.get("area"),
        n_total=len(table_df),
        area_distribution=_tag_distribution(table_df, "tag"),
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


# ---------------------------------------------------------------------------
# Bulk MICrONS formatters
# ---------------------------------------------------------------------------


def format_bulk_coregistration(
    raw: dict[str, Any], dataset: str, extra_key: str
) -> BulkCoregistrationResponse:
    """Format bulk coregistration data into BulkCoregistrationResponse."""
    table_df: pd.DataFrame = raw["table_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=table_df,
        tool="bulk_coregistration",
        dataset=dataset,
        neuron_id=None,
        materialization_version=mat_version,
        extra_key=extra_key,
    )

    score_dist = (
        _weight_distribution(table_df["score"])
        if not table_df.empty and "score" in table_df.columns
        else {}
    )
    sessions = (
        sorted(int(s) for s in table_df["session"].dropna().unique())
        if not table_df.empty and "session" in table_df.columns
        else []
    )

    return BulkCoregistrationResponse(
        dataset=dataset,
        n_root_ids=raw.get("n_root_ids", 0),
        n_units=len(table_df),
        score_distribution=score_dist,
        sessions=sessions,
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_bulk_functional_properties(
    raw: dict[str, Any], dataset: str, extra_key: str
) -> BulkFunctionalPropertiesResponse:
    """Format bulk functional properties into BulkFunctionalPropertiesResponse."""
    table_df: pd.DataFrame = raw["table_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=table_df,
        tool="bulk_functional_properties",
        dataset=dataset,
        neuron_id=None,
        materialization_version=mat_version,
        extra_key=extra_key,
    )

    ori_dist = (
        _weight_distribution(table_df["OSI"])
        if not table_df.empty and "OSI" in table_df.columns
        else {}
    )
    dir_dist = (
        _weight_distribution(table_df["DSI"])
        if not table_df.empty and "DSI" in table_df.columns
        else {}
    )

    return BulkFunctionalPropertiesResponse(
        dataset=dataset,
        n_root_ids=raw.get("n_root_ids", 0),
        coregistration_source=raw.get("coregistration_source", "auto_phase3"),
        n_units=len(table_df),
        ori_selectivity_distribution=ori_dist,
        dir_selectivity_distribution=dir_dist,
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_bulk_synapse_targets(
    raw: dict[str, Any], dataset: str, extra_key: str
) -> BulkSynapseTargetsResponse:
    """Format bulk synapse targets into BulkSynapseTargetsResponse."""
    table_df: pd.DataFrame = raw["table_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=table_df,
        tool="bulk_synapse_targets",
        dataset=dataset,
        neuron_id=None,
        materialization_version=mat_version,
        extra_key=extra_key,
    )

    return BulkSynapseTargetsResponse(
        dataset=dataset,
        n_root_ids=raw.get("n_root_ids", 0),
        direction=raw.get("direction", "post"),
        n_synapses=len(table_df),
        target_distribution=_tag_distribution(table_df, "tag"),
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )


def format_bulk_functional_area(
    raw: dict[str, Any], dataset: str, extra_key: str
) -> BulkFunctionalAreaResponse:
    """Format bulk functional area into BulkFunctionalAreaResponse."""
    table_df: pd.DataFrame = raw["table_df"]
    mat_version = raw.get("materialization_version")

    manifest = save_artifact(
        df=table_df,
        tool="bulk_functional_area",
        dataset=dataset,
        neuron_id=None,
        materialization_version=mat_version,
        extra_key=extra_key,
    )

    return BulkFunctionalAreaResponse(
        dataset=dataset,
        n_root_ids=raw.get("n_root_ids", 0),
        n_total=len(table_df),
        area_distribution=_tag_distribution(table_df, "tag"),
        artifact_manifest=manifest,
        warnings=raw.get("warnings", []),
    )
