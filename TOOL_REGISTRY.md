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
| `get_neuron_info` | ✅ 2026-03-10 | minnie65, hemibrain, flywire (mocked) | `NeuronInfoResponse` | Scalar-only, no artifact. Accepts optional `nucleus_id` for MICrONS. FlyWire: returns `classification_hierarchy` (5-level) + `neurotransmitter_type` from output synapse NT predictions. |
| `get_connectivity` | ✅ 2026-03-10 | minnie65, hemibrain, flywire (mocked) | `ConnectivityResponse` + `ArtifactManifest` | Artifact-first: full partner Parquet, 3-item samples, weight distributions. MICrONS: `partner_nucleus_id` + `partner_nucleus_conflict`. FlyWire: `partner_nt_type` + `partner_nt_confidence` + `neurotransmitter_distribution` in response. |
| `get_cell_type_taxonomy` | ✅ 2026-03-16 | minnie65, hemibrain, flywire (mocked) | `CellTypeTaxonomyResponse` | Scalar-only. Returns the full classification hierarchy with top values/counts at each level and example lineages. FlyWire: 4-level hierarchy (super_class → cell_class → cell_sub_class → cell_type). Use FIRST to understand naming conventions. |
| `search_cell_types` | ✅ 2026-03-16 | minnie65, hemibrain, flywire (mocked) | `CellTypeSearchResponse` | Scalar-only. Case-insensitive substring search across all annotation levels. FlyWire: searches all hierarchy levels. On 0 results: includes taxonomy_hints with available categories and pointer to `get_cell_type_taxonomy()`. |
| `get_neurons_by_type` | ✅ 2026-03-10 | minnie65, hemibrain, flywire (mocked) | `NeuronsByTypeResponse` + `ArtifactManifest` | Artifact-first: full neuron list Parquet, type/region distributions. FlyWire: progressive matching (exact → cross-level → case-insensitive → substring) across all hierarchy levels. Suggests `search_cell_types()` when 0 results. |
| `get_region_connectivity` | ✅ 2026-03-10 | minnie65, hemibrain (mocked) | `RegionConnectivityResponse` + `ArtifactManifest` | Artifact-first: long-format region pairs, optional source/target region filters |
| `build_neuroglancer_url` | ✅ 2026-03-10 | minnie65, hemibrain (mocked) | `NeuroglancerUrlResponse` | Scalar-only, wraps existing URL builder as MCP tool |
| `get_bulk_connectivity` | ✅ 2026-03-16 | minnie65, hemibrain (mocked) | `BulkConnectivityResponse` + `ArtifactManifest` | Artifact-first: bulk edge table for multiple neurons. Batched (200 IDs, 0.5s sleep). Content-addressable cache via sha256(sorted_ids:direction). CAVE: staleness gate raises ValueError listing all stale IDs. Columns: pre_root_id, post_root_id, syn_count, neuropil. |
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
| `get_coregistration` | ✅ 2026-03-16 | minnie65 (mocked + live) | `CoregistrationResponse` + `ArtifactManifest` | MICrONS-only. Maps EM neurons → 2-photon functional imaging units. Table: `coregistration_auto_phase3_fwd_apl_vess_combined_v2`. Uses content-aware API for root_id, `target_id` for nucleus_id. |
| `get_functional_properties` | ✅ 2026-03-16 | minnie65 (mocked + live) | `FunctionalPropertiesResponse` + `ArtifactManifest` | MICrONS-only. Digital twin tuning: OSI, DSI, pref_ori, pref_dir, cc_abs/max/norm. Default source: `auto_phase3` (largest coverage). 3 table variants via `coregistration_source` param. |
| `get_synapse_targets` | ✅ 2026-03-16 | minnie65 (mocked + live) | `SynapseTargetsResponse` + `ArtifactManifest` | MICrONS-only. Per-synapse spine/shaft/soma classification. Table: `synapse_target_predictions_ssa_v2`. Content-aware API required. |
| `get_multi_input_spines` | ✅ 2026-03-16 | minnie65 (mocked) | `MultiInputSpinesResponse` + `ArtifactManifest` | MICrONS-only. DEPRECATED — prefer `get_synapse_targets`. Spines with >1 input, grouped by `group_id`. |
| `get_cell_mtypes` | ✅ 2026-03-16 | minnie65 (mocked + live) | `CellMtypesResponse` + `ArtifactManifest` | MICrONS-only. 24 morphological types (L2a-L6wm, PTC/DTC/STC/ITC). Table: `aibs_metamodel_mtypes_v661_v2`. `classification_system` values: `excitatory_neuron`/`inhibitory_neuron`. |
| `get_functional_area` | ✅ 2026-03-16 | minnie65 (mocked + live) | `FunctionalAreaResponse` + `ArtifactManifest` | MICrONS-only. Brain area labels: V1, AL, RL, LM. `value` = distance to boundary (μm). Table: `nucleus_functional_area_assignment`. |
| `get_bulk_coregistration` | ✅ 2026-03-16 | minnie65 (mocked) | `BulkCoregistrationResponse` + `ArtifactManifest` | MICrONS-only. Bulk version of `get_coregistration`. Batched content-aware API (200 IDs, 0.5s sleep). Content-addressable cache. Staleness gate. |
| `get_bulk_functional_properties` | ✅ 2026-03-16 | minnie65 (mocked) | `BulkFunctionalPropertiesResponse` + `ArtifactManifest` | MICrONS-only. Bulk version of `get_functional_properties`. Supports `coregistration_source` param. |
| `get_bulk_synapse_targets` | ✅ 2026-03-16 | minnie65 (mocked) | `BulkSynapseTargetsResponse` + `ArtifactManifest` | MICrONS-only. Bulk version of `get_synapse_targets`. Supports `direction` param. |
| `get_bulk_functional_area` | ✅ 2026-03-16 | minnie65 (mocked) | `BulkFunctionalAreaResponse` + `ArtifactManifest` | MICrONS-only. Bulk version of `get_functional_area`. |

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
| CAVE backend subclasses | ✅ 2026-03-10 | `CAVEBackend` (concrete base) → `MICrONSBackend`, `FlyWireBackend`. Template-method hooks: `_enrich_neuron_info`, `_enrich_connectivity`, `_interpret_proofreading_row`. No `if self.dataset_name` conditionals; no module-level mapping dicts. |

---

## Integration Tests (Live API)

Run with: `pytest tests/integration/ --integration -v -s`

| Dataset | Status | Test Cell | Tool Tested | Notes |
|---------|--------|-----------|-------------|-------|
| minnie65 | ✅ 2026-03-10 | nucleus 264824 → root 864691135571546917 | `resolve_nucleus_ids` → `validate_root_ids` → `get_connectivity` | 8840 partners, artifact verified |
| flywire | ✅ 2026-03-10 | root 720575940621039145 → superseded → 720575940605214636 | `validate_root_ids` → `get_neuron_info` | Staleness detected; cell_type returns "AMMC-AMMC PN" after schema fix |
| hemibrain | ✅ 2026-03-10 | bodyId 5813105172 (DA1 adPN) | `get_connectivity` | 4986 partners, artifact verified |

## Schema Audit (Live API)

Run with: `pytest tests/integration/test_schema_audit.py --integration -v -s`

See `SCHEMA_REFERENCE.md` for verified column names per table per dataset.

| Dataset | Status | Tests | Notes |
|---------|--------|-------|-------|
| minnie65 | ✅ 2026-03-10 | 5 pass | All tables verified; proofreading table renamed to `proofreading_status_and_strategy` |
| flywire | ✅ 2026-03-10 | 5 pass | Tables renamed: `neuron_information_v2`, `synapses_nt_v1`, `proofread_neurons`; cell type column is `tag` |
| hemibrain | ✅ 2026-03-10 | 3 pass | `somaLocation` confirmed as `list`; `fetch_roi_connectivity` doesn't exist; ROI connectivity uses Cypher |

---

## Output Schemas Registered (schemas.py)

| Schema | Used By | Artifact? |
|--------|---------|-----------|
| `ArtifactManifest` | All artifact-producing tools | Shared type |
| `NeuronInfoResponse` | `get_neuron_info`, `get_neuron_at_timepoint` | No |
| `SynapticPartnerSample` | `ConnectivityResponse` (3-item sample) | No |
| `BulkConnectivityResponse` | `get_bulk_connectivity` | Yes |
| `ConnectivityResponse` | `get_connectivity` | Yes |
| `CellTypeTaxonomyResponse` | `get_cell_type_taxonomy` | No |
| `TaxonomyLevel` | `CellTypeTaxonomyResponse` (per-level entry) | No |
| `CellTypeSearchResponse` | `search_cell_types` | No |
| `CellTypeMatch` | `CellTypeSearchResponse` (per-match entry) | No |
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
| `CoregistrationResponse` | `get_coregistration` | Yes |
| `FunctionalPropertiesResponse` | `get_functional_properties` | Yes |
| `SynapseTargetsResponse` | `get_synapse_targets` | Yes |
| `MultiInputSpinesResponse` | `get_multi_input_spines` | Yes |
| `CellMtypesResponse` | `get_cell_mtypes` | Yes |
| `FunctionalAreaResponse` | `get_functional_area` | Yes |
| `BulkCoregistrationResponse` | `get_bulk_coregistration` | Yes |
| `BulkFunctionalPropertiesResponse` | `get_bulk_functional_properties` | Yes |
| `BulkSynapseTargetsResponse` | `get_bulk_synapse_targets` | Yes |
| `BulkFunctionalAreaResponse` | `get_bulk_functional_area` | Yes |
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
| 2026-03-10 | `get_proofreading_status` uses dataset-specific proofreading tables | minnie65: `proofreading_status_and_strategy`, flywire: `proofread_neurons` |
| 2026-03-10 | Nucleus IDs are MICrONS-only | No equivalent stable cell identifier exists in FlyWire or hemibrain in this version |
| 2026-03-10 | neuprint `fetch_adjacencies` returns `(neuron_df, conn_df)` | Fixed reversed variable names; conn_df has columns bodyId_pre, bodyId_post, roi, weight |
| 2026-03-10 | neuPrint backend methods must access `self.client` before module-level functions | neuprint module-level functions like `fetch_adjacencies` use default client set by `Client()` constructor |
| 2026-03-10 | neuprint-python 0.6 has Python 3.11 SyntaxError | Backslash in f-string in `connectivity.py`; patched locally in two places |
| 2026-03-10 | Integration tests gated behind `--integration` pytest flag | Tests in `tests/integration/`; skipped by default |
| 2026-03-10 | FlyWire datastack corrected to `flywire_fafb_public` | Was `flywire_fafb_production`; confirmed via `c.info.get_datastacks()` |
| 2026-03-10 | Schema audit: fixed 3 pre-confirmed bugs + 4 table name bugs | See `SCHEMA_REFERENCE.md` for verified column names |
| 2026-03-10 | `get_tabular_changelog` → `get_tabular_change_log([root_id])[root_id]` | Method name was wrong; returns `dict[int, DataFrame]` not `DataFrame` |
| 2026-03-10 | neuPrint `somaLocation` is `list`, not `dict` | Confirmed via live API; removed `.get("coordinates")` call |
| 2026-03-10 | neuPrint `fetch_roi_connectivity` doesn't exist | Replaced with Cypher query in `get_region_connectivity` |
| 2026-03-10 | FlyWire table names differ from assumed | `neuron_information_v2` (not `classification`), `synapses_nt_v1` (not `synapses`), `proofread_neurons` (not `proofreading_status_table`) |
| 2026-03-10 | FlyWire uses `tag` column for cell type | Not `cell_type`; added `CELL_TYPE_COLUMN` mapping dict |
| 2026-03-10 | minnie65 proofreading table is `proofreading_status_and_strategy` | Not `proofreading_status_public_release`; `status_axon`/`status_dendrite` are proper booleans |
| 2026-03-10 | FlyWire synapse count returns None | Server bug: `select_columns=["id"]` on `synapses_nt_v1` causes `'BigInteger' object has no attribute 'id'` |
| 2026-03-10 | FlyWire synapse count fix | Use `select_columns=["pre_pt_root_id"]`/`["post_pt_root_id"]` instead of `["id"]` for all datasets |
| 2026-03-10 | FlyWire hierarchical classification | `hierarchical_neuron_annotations` cached in-memory (10min TTL); cannot filter by `pt_root_id` server-side |
| 2026-03-10 | FlyWire neurotransmitter prediction | Mean of per-synapse NT probabilities (gaba/ach/glut/oct/ser/da) → argmax for dominant NT type |
| 2026-03-10 | FlyWire connectivity NT enrichment | Partner-level NT type via per-synapse probabilities; `partner_nt_type` + `partner_nt_confidence` columns in artifact |
| 2026-03-10 | FlyWire `get_neurons_by_type` uses hierarchy | Queries cached hierarchy table at `cell_type` classification level instead of `neuron_information_v2` |
| 2026-03-10 | CAVE backend refactored into subclasses | `CAVEBackend` (concrete base) + `MICrONSBackend`, `FlyWireBackend`. Removed 5 module-level mapping dicts and all `if self.dataset_name` conditionals. Dataset config is class attributes; dataset-specific behaviour uses template-method hooks. |
| 2026-03-16 | Fixed artifact cache bug: `extra_key` param in `save_artifact` | Cache key now includes table name to disambiguate different annotation table queries for the same dataset |
| 2026-03-16 | CAVE reference tables use content-aware API for root_id filtering | `pt_root_id` is a bound spatial point column that fails with `filter_equal_dict` on many CAVE reference tables; `_query_reference_table()` helper uses content-aware API (`client.materialize.tables.<name>(pt_root_id=val).query()`) |
| 2026-03-16 | MICrONS nucleus ID FK column is `target_id` | NOT `id` (which is the annotation row ID). Confirmed by live API for all reference tables. |
| 2026-03-16 | `classification_system` values are `excitatory_neuron`/`inhibitory_neuron` | NOT plain `excitatory`/`inhibitory` as documentation suggests. Verified via live API against `aibs_metamodel_mtypes_v661_v2`. |
| 2026-03-16 | Default `coregistration_source` is `auto_phase3` | Automatically-coregistered cells (larger coverage) vs `coreg_v4` (manual). Per user requirement. |
