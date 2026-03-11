# TOOL_REGISTRY.md — Connectomics MCP Tool Registry

This file is the source of truth for what's built, what's in progress, and what's planned. Updated at the end of every Claude Code session.

## Status Legend
- ✅ Complete — implemented, tested, schema registered
- 🔧 In Progress — implementation started
- 📋 Planned — designed but not started
- ❌ Blocked — waiting on dependency or decision

---

## Tier 1: Universal Tools (all datasets)

| Tool | Status | Datasets Tested | Output Schema | Notes |
|------|--------|-----------------|---------------|-------|
| `get_neuron_info` | ✅ 2026-03-10 | minnie65, hemibrain (mocked) | `NeuronInfoResponse` | Scalar-only, no artifact. Accepts optional `nucleus_id` for MICrONS. |
| `get_connectivity` | ✅ 2026-03-10 | minnie65, hemibrain (mocked) | `ConnectivityResponse` + `ArtifactManifest` | Artifact-first: full partner Parquet, 3-item samples, weight distributions. MICrONS artifact includes `partner_nucleus_id` + `partner_nucleus_conflict` columns. |
| `get_neurons_by_type` | ✅ 2026-03-10 | minnie65, hemibrain (mocked) | `NeuronsByTypeResponse` + `ArtifactManifest` | Artifact-first: full neuron list Parquet, type/region distributions |
| `get_region_connectivity` | ✅ 2026-03-10 | minnie65, hemibrain (mocked) | `RegionConnectivityResponse` + `ArtifactManifest` | Artifact-first: long-format region pairs, optional source/target region filters |
| `build_neuroglancer_url` | ✅ 2026-03-10 | minnie65, hemibrain (mocked) | `NeuroglancerUrlResponse` | Scalar-only, wraps existing URL builder as MCP tool |
| `validate_root_ids` | ✅ 2026-03-10 | minnie65, hemibrain (mocked) | `RootIdValidationResponse` | Scalar-only; CAVE checks currency + suggests replacements; neuPrint returns all current |

---

## Tier 2: CAVE-Specific Tools

| Tool | Status | Datasets | Output Schema | Notes |
|------|--------|----------|---------------|-------|
| `resolve_nucleus_ids` | ✅ 2026-03-10 | minnie65 (mocked) | `NucleusResolutionResult` | Scalar-only; minnie65-only; resolves nucleus IDs → current pt_root_ids via `nucleus_detection_v0` |
| `get_proofreading_status` | ✅ 2026-03-10 | minnie65 (mocked) | `ProofreadingStatusResponse` | Scalar-only; CAVE-only; queries proofreading table + edit changelog |
| `get_neuron_at_timepoint` | 📋 | minnie65, flywire | `NeuronInfoResponse` | Scalar-only |
| `query_annotation_table` | ✅ 2026-03-10 | minnie65 (mocked) | `AnnotationTableResponse` + `ArtifactManifest` | Artifact-first: raw CAVE table query, all columns preserved, schema_description auto-generated from dtypes |
| `get_edit_history` | ✅ 2026-03-10 | minnie65 (mocked) | `EditHistoryResponse` + `ArtifactManifest` | Artifact-first: edit changelog with operation_id, timestamp, operation_type, user_id. Staleness gate raises StaleRootIdError. |

---

## Tier 3: neuPrint-Specific Tools

| Tool | Status | Datasets | Output Schema | Notes |
|------|--------|----------|---------------|-------|
| `fetch_cypher` | ✅ 2026-03-10 | hemibrain (mocked) | `CypherQueryResponse` + `ArtifactManifest` | Artifact-first: wraps `client.fetch_custom()`, echoes query, saves full result as Parquet |
| `get_synapse_compartments` | ✅ 2026-03-10 | hemibrain (mocked) | `SynapseCompartmentResponse` | Scalar-only: per-ROI synapse distribution via `fetch_neurons` roi_df |

---

## Infrastructure

| Component | Status | Notes |
|-----------|--------|-------|
| `artifacts/writer.py` | ✅ 2026-03-10 | `save_artifact()` with Parquet + 1hr cache |
| `ArtifactManifest` schema | ✅ 2026-03-10 | Embedded in all artifact-producing responses |

---

## Output Schemas Registered (schemas.py)

| Schema | Used By | Artifact? |
|--------|---------|-----------|
| `ArtifactManifest` | All artifact-producing tools | Shared type |
| `NeuronInfoResponse` | `get_neuron_info`, `get_neuron_at_timepoint` | No |
| `SynapticPartnerSample` | `ConnectivityResponse` (3-item sample) | No |
| `ConnectivityResponse` | `get_connectivity` | Yes |
| `NeuronsByTypeResponse` | `get_neurons_by_type` | Yes |
| `RegionConnectivityResponse` | `get_region_connectivity` | Yes |
| `NeuroglancerUrlResponse` | `build_neuroglancer_url` | No |
| `RootIdValidationResponse` | `validate_root_ids` | No |
| `NucleusResolutionResult` | `resolve_nucleus_ids` | No |
| `NucleusResolution` | `NucleusResolutionResult` (per-nucleus entry) | No |
| `NucleusResolutionStatus` | `NucleusResolution` (enum) | No |
| `ProofreadingStatusResponse` | `get_proofreading_status` | No |
| `AnnotationTableResponse` | `query_annotation_table` | Yes |
| `EditHistoryResponse` | `get_edit_history` | Yes |
| `CypherQueryResponse` | `fetch_cypher` | Yes |
| `SynapseCompartmentResponse` | `get_synapse_compartments` | No |

---

## Error Types

| Error | Raised When |
|-------|------------|
| `DatasetNotSupported` | Tool called against incompatible dataset |
| `StaleRootIdError` | CAVE root ID fails currency check |
| `BackendConnectionError` | Cannot reach API server |

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| — | Skip H01 for V1 | No live query API; bulk Avro files require local DB setup |
| — | CAVE is reference implementation | Covers both MICrONS and FlyWire; neuPrint is second adapter |
| — | Neuroglancer URL builder is pure code | Deterministic transformation; no LLM reasoning needed |
| 2026-03-10 | Neuroglancer URLs use zlib + base64url encoding | Standard Neuroglancer fragment encoding; tested round-trip decode |
| 2026-03-10 | FlyWire datastack set to `flywire_fafb_production` | Matches current CAVE production datastack name |
| 2026-03-10 | All Pydantic schemas implemented upfront | Enables schema validation tests before any tool is built |
| 2026-03-10 | Artifact-first pattern adopted | Tabular tools save complete Parquet to disk, return lightweight manifest + 3-item sample in context. No truncation, no row limits. |
| 2026-03-10 | Artifact cache window = 1 hour | Same (tool, dataset, neuron_id, mat_version) reuses cached Parquet within 1hr |
| 2026-03-10 | Weight distribution in ConnectivityResponse uses synapse counts | mean/median/max/p90 computed over n_synapses column, not normalized weights |
| 2026-03-10 | Removed `truncated`, `n_shown`, `limit`, `top_n` from all schemas | Per updated OUTPUT_CONTRACTS.md — these concepts don't exist in the artifact-first pattern |
| 2026-03-10 | Fixed `base.py` abstract signatures | `get_connectivity` now uses `direction` (not `top_n`); `get_neurons_by_type` no longer has `limit` param |
| 2026-03-10 | `get_synapse_compartments` uses ROIs as compartments | Hemibrain doesn't expose morphological compartment labels (axon/dendrite) via standard neuPrint API; per-ROI synapse distribution is the closest equivalent |
| 2026-03-10 | `fetch_cypher` and `get_synapse_compartments` are neuPrint-only | CAVE datasets raise `DatasetNotSupported`; consistent with Tier 3 designation |
| 2026-03-10 | `validate_root_ids` does NOT raise `StaleRootIdError` | This tool IS the resolution mechanism for stale IDs; it reports status without blocking |
| 2026-03-10 | `get_proofreading_status` uses dataset-specific proofreading tables | minnie65: `proofreading_status_public_release`, flywire: `proofreading_status_table`, fanc: `proofreading_status` |
| 2026-03-10 | Nucleus IDs are MICrONS-only | No equivalent stable cell identifier exists in FlyWire or hemibrain in this version |
