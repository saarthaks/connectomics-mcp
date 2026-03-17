"""Pydantic models for all tool output contracts.

Every tool returns one of these models. See OUTPUT_CONTRACTS.md for
the design rationale behind each schema.

Key rule: tabular tools save complete Parquet artifacts to disk and
return an ``ArtifactManifest`` in context.  Scalar-only tools return
their full response directly.  The words "truncated", "n_shown",
"limit", and "cap" do not exist in this codebase.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


class ArtifactManifest(BaseModel):
    artifact_path: str
    artifact_format: str = "parquet"
    n_rows: int
    columns: list[str]
    schema_description: str
    dataset: str
    query_timestamp: str
    materialization_version: int | None = None
    cache_hit: bool = False


# ---------------------------------------------------------------------------
# Universal schemas
# ---------------------------------------------------------------------------


class NeuronInfoResponse(BaseModel):
    """Scalar-only — no artifact needed."""

    neuron_id: int | str
    dataset: str
    cell_type: str | None = None
    cell_class: str | None = None
    region: str | None = None
    soma_position_nm: tuple[float, float, float] | None = None
    n_pre_synapses: int | None = None
    n_post_synapses: int | None = None
    proofread: bool | None = None
    materialization_version: int | None = None
    neuroglancer_url: str = ""
    neurotransmitter_type: str | None = None
    classification_hierarchy: dict | None = None
    warnings: list[str] = []


class SynapticPartnerSample(BaseModel):
    """Lightweight partner for the 3-item orientation sample."""

    partner_id: int | str
    partner_type: str | None = None
    n_synapses: int = 0
    weight_normalized: float | None = None


class ConnectivityResponse(BaseModel):
    """Artifact-producing — full partner table saved to Parquet."""

    neuron_id: int | str
    dataset: str
    n_upstream_total: int = 0
    n_downstream_total: int = 0
    upstream_weight_distribution: dict = {}
    downstream_weight_distribution: dict = {}
    upstream_sample: list[SynapticPartnerSample] = []
    downstream_sample: list[SynapticPartnerSample] = []
    sample_note: str = (
        "upstream_sample and downstream_sample show the 3 highest-weight partners "
        "for orientation only. Load artifact_manifest.artifact_path for the complete dataset."
    )
    neuroglancer_url: str = ""
    neurotransmitter_distribution: dict | None = None
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class CellTypeMatch(BaseModel):
    """A single cell type match from a search query."""

    cell_type: str
    classification_level: str | None = None
    n_neurons: int = 0


class CellTypeSearchResponse(BaseModel):
    """Scalar-only — cell type discovery tool."""

    dataset: str
    query: str
    n_matches: int = 0
    matches: list[CellTypeMatch] = []
    taxonomy_hints: list[str] = []
    warnings: list[str] = []


class TaxonomyLevel(BaseModel):
    """A single level in the cell type taxonomy."""

    level_name: str
    values: list[dict] = []


class CellTypeTaxonomyResponse(BaseModel):
    """Scalar-only — cell type taxonomy/hierarchy for a dataset."""

    dataset: str
    n_total_neurons: int = 0
    levels: list[TaxonomyLevel] = []
    example_lineages: list[dict] = []
    warnings: list[str] = []


class NeuronsByTypeResponse(BaseModel):
    """Artifact-producing — full neuron list saved to Parquet."""

    dataset: str
    query_cell_type: str
    query_region: str | None = None
    n_total: int = 0
    type_distribution: dict = {}
    region_distribution: dict = {}
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class BulkConnectivityResponse(BaseModel):
    """Artifact-producing — complete edge table saved to Parquet."""

    dataset: str
    n_root_ids: int = 0
    direction: str = "both"
    n_edges: int = 0
    total_synapses: int = 0
    cached: bool = False
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class RegionConnectivityResponse(BaseModel):
    """Artifact-producing — long-format region pairs saved to Parquet."""

    dataset: str
    n_regions: int = 0
    top_5_connections: list[dict] = []
    total_synapses: int = 0
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class NeuroglancerUrlResponse(BaseModel):
    """Scalar-only — no artifact needed."""

    url: str
    dataset: str
    n_segments: int = 0
    layers_included: list[str] = []
    coordinate_space: str = "nm"


class RootIdValidationResult(BaseModel):
    root_id: int
    is_current: bool = True
    last_edit_timestamp: str | None = None
    suggested_current_id: int | None = None


class RootIdValidationResponse(BaseModel):
    """Scalar-only — no artifact needed."""

    dataset: str
    materialization_version: int = 0
    results: list[RootIdValidationResult] = []
    n_stale: int = 0
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# CAVE-specific schemas
# ---------------------------------------------------------------------------


class ProofreadingStatusResponse(BaseModel):
    """Scalar-only — no artifact needed."""

    neuron_id: int
    dataset: str
    axon_proofread: bool | None = None
    dendrite_proofread: bool | None = None
    strategy_axon: str | None = None
    strategy_dendrite: str | None = None
    n_edits: int | None = None
    last_edit_timestamp: str | None = None
    warnings: list[str] = []


class NucleusResolutionStatus(str, Enum):
    """Resolution status for a nucleus ID → pt_root_id mapping."""

    RESOLVED = "resolved"
    MERGE_CONFLICT = "merge_conflict"
    NO_SEGMENT = "no_segment"


class NucleusResolution(BaseModel):
    """Resolution result for a single nucleus ID."""

    nucleus_id: int
    pt_root_id: int | None = None
    resolution_status: NucleusResolutionStatus
    conflicting_nucleus_ids: list[int] = []
    materialization_version: int


class NucleusResolutionResult(BaseModel):
    """Scalar-only — bounded by input list size."""

    dataset: str
    materialization_version: int
    resolutions: list[NucleusResolution]
    n_resolved: int
    n_merge_conflicts: int
    n_no_segment: int
    warnings: list[str] = []


class AnnotationTableResponse(BaseModel):
    """Artifact-producing — raw query result saved to Parquet."""

    dataset: str
    table_name: str
    n_total: int = 0
    schema_description: str = ""
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# MICrONS-specific schemas
# ---------------------------------------------------------------------------


class CoregistrationResponse(BaseModel):
    """Artifact-producing — EM-to-functional imaging unit mappings."""

    neuron_id: int
    query_by: str
    dataset: str
    n_units: int = 0
    score_distribution: dict = {}
    sessions: list[int] = []
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class FunctionalPropertiesResponse(BaseModel):
    """Artifact-producing — digital twin tuning properties."""

    neuron_id: int
    query_by: str
    dataset: str
    coregistration_source: str
    n_units: int = 0
    ori_selectivity_distribution: dict = {}
    dir_selectivity_distribution: dict = {}
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class SynapseTargetsResponse(BaseModel):
    """Artifact-producing — per-synapse structural target classification."""

    neuron_id: int
    dataset: str
    direction: str
    n_synapses: int = 0
    target_distribution: dict = {}
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class MultiInputSpinesResponse(BaseModel):
    """Artifact-producing — multi-input spine predictions (deprecated)."""

    neuron_id: int
    dataset: str
    direction: str
    n_synapses: int = 0
    n_spine_groups: int = 0
    target_distribution: dict = {}
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class CellMtypesResponse(BaseModel):
    """Artifact-producing — morphological cell type classifications."""

    dataset: str
    query_neuron_id: int | None = None
    query_by: str | None = None
    query_cell_type: str | None = None
    n_total: int = 0
    classification_system_distribution: dict = {}
    cell_type_distribution: dict = {}
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class FunctionalAreaResponse(BaseModel):
    """Artifact-producing — brain area assignments per nucleus."""

    dataset: str
    query_neuron_id: int | None = None
    query_by: str | None = None
    query_area: str | None = None
    n_total: int = 0
    area_distribution: dict = {}
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class BulkCoregistrationResponse(BaseModel):
    """Artifact-producing — bulk coregistration for multiple neurons."""

    dataset: str
    n_root_ids: int = 0
    n_units: int = 0
    score_distribution: dict = {}
    sessions: list[int] = []
    cached: bool = False
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class BulkFunctionalPropertiesResponse(BaseModel):
    """Artifact-producing — bulk functional properties for multiple neurons."""

    dataset: str
    n_root_ids: int = 0
    coregistration_source: str = "auto_phase3"
    n_units: int = 0
    ori_selectivity_distribution: dict = {}
    dir_selectivity_distribution: dict = {}
    cached: bool = False
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class BulkSynapseTargetsResponse(BaseModel):
    """Artifact-producing — bulk synapse targets for multiple neurons."""

    dataset: str
    n_root_ids: int = 0
    direction: str = "post"
    n_synapses: int = 0
    target_distribution: dict = {}
    cached: bool = False
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class BulkFunctionalAreaResponse(BaseModel):
    """Artifact-producing — bulk functional area for multiple neurons."""

    dataset: str
    n_root_ids: int = 0
    n_total: int = 0
    area_distribution: dict = {}
    cached: bool = False
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class EditHistoryResponse(BaseModel):
    """Artifact-producing — edit log saved to Parquet."""

    neuron_id: int
    dataset: str
    n_edits_total: int = 0
    first_edit_timestamp: str | None = None
    last_edit_timestamp: str | None = None
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


# ---------------------------------------------------------------------------
# neuPrint-specific schemas
# ---------------------------------------------------------------------------


class CypherQueryResponse(BaseModel):
    """Artifact-producing — full query result saved to Parquet."""

    dataset: str
    query: str
    n_rows: int = 0
    columns: list[str] = []
    artifact_manifest: ArtifactManifest | None = None
    warnings: list[str] = []


class CompartmentStats(BaseModel):
    compartment: str
    n_synapses: int = 0
    fraction: float = 0.0


class SynapseCompartmentResponse(BaseModel):
    """Scalar-only — always <= 5 compartment rows."""

    neuron_id: int | str
    dataset: str
    direction: str = "input"
    compartments: list[CompartmentStats] = []
    n_total_synapses: int = 0
    warnings: list[str] = []
