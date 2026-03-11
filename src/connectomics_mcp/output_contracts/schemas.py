"""Pydantic models for all tool output contracts.

Every tool returns one of these models. See OUTPUT_CONTRACTS.md for
the design rationale behind each schema.

Key rule: tabular tools save complete Parquet artifacts to disk and
return an ``ArtifactManifest`` in context.  Scalar-only tools return
their full response directly.  The words "truncated", "n_shown",
"limit", and "cap" do not exist in this codebase.
"""

from __future__ import annotations

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
    artifact_manifest: ArtifactManifest | None = None
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


class AnnotationTableResponse(BaseModel):
    """Artifact-producing — raw query result saved to Parquet."""

    dataset: str
    table_name: str
    n_total: int = 0
    schema_description: str = ""
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
