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
    CompartmentStats,
    ConnectivityResponse,
    CypherQueryResponse,
    EditHistoryResponse,
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
                partner_type=row.get("partner_type"),
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
