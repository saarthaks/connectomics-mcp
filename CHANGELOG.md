# Changelog

## 2026-03-10 — Session 9

### Added
- `tests/test_integration.py`: 10 integration tests covering server setup, cross-tool workflows, error propagation, and artifact consistency
- Enhanced `README.md`: expanded from 46 → ~170 lines with artifact-first explanation, full tool reference, example workflow, configuration guide, development instructions
- `pyproject.toml` metadata: license (MIT), keywords, classifiers, readme reference, `[dev]` optional dependencies
- Test count: 130 total (all passing)

### Notes
- Integration tests verify: MCP server name, all 12 tools registered, validate→info→connectivity workflow chains, neuPrint multi-tool workflow, StaleRootIdError consistency across tools, DatasetNotSupported across tiers, artifact round-trip correctness, shared artifact directory
- `get_neuron_at_timepoint` remains 📋 Planned — not part of the KICKSTART_PROMPT.md session order
- All tests use mocked backends — no live API calls

---

## 2026-03-10 — Session 8

### Added
- `fetch_cypher` tool (neuPrint-specific): executes arbitrary Cypher queries via `client.fetch_custom()`, saves complete result as Parquet artifact, returns `CypherQueryResponse` with query echo, row count, and column names
- `get_synapse_compartments` tool (neuPrint-specific): fetches per-ROI synapse distribution via `fetch_neurons` roi_df, returns `SynapseCompartmentResponse` with compartment names, synapse counts, and fractions
- `format_cypher_query` and `format_synapse_compartments` formatters
- `fetch_cypher` and `get_synapse_compartments` backend methods on `NeuPrintBackend`
- Abstract methods `fetch_cypher` and `get_synapse_compartments` on `ConnectomeBackend` base class
- CAVE stubs for both tools (raise `DatasetNotSupported`)
- `tools/neuprint_specific.py`: new module for neuPrint-specific tools (Tier 3)
- 14 new tests: 7 for `fetch_cypher`, 7 for `get_synapse_compartments`
- Test count: 120 total (all passing)

### Notes
- `fetch_cypher` has no staleness check (arbitrary query, not single neuron)
- `get_synapse_compartments` uses ROIs as compartments — hemibrain doesn't expose morphological compartment labels (axon/dendrite) via standard neuPrint API
- `get_synapse_compartments` direction: "input" uses post column, "output" uses pre column from roi_df
- Both tools are neuPrint-only (Tier 3); CAVE datasets raise `DatasetNotSupported`
- `fetch_cypher` artifact cache key uses `neuron_id=None` (arbitrary query)
- All Tier 1, 2, and 3 tools are now complete

---

## 2026-03-10 — Session 7

### Added
- `query_annotation_table` tool (CAVE-specific): queries arbitrary CAVE annotation tables with optional `filter_equal_dict` and `filter_in_dict`, saves complete result as Parquet artifact, returns `AnnotationTableResponse` with row count and auto-generated `schema_description` from DataFrame dtypes
- `get_edit_history` tool (CAVE-specific): fetches neuron edit changelog via `chunkedgraph.get_tabular_changelog()`, saves as Parquet artifact with columns `operation_id`, `timestamp`, `operation_type` ("merge"/"split"), `user_id`; returns `EditHistoryResponse` with edit count and timestamp range
- `format_annotation_table` and `format_edit_history` formatters
- `query_annotation_table` and `get_edit_history` backend methods on `CAVEBackend`
- Abstract methods `query_annotation_table` and `get_edit_history` on `ConnectomeBackend` base class
- neuPrint stubs for both tools (raise `DatasetNotSupported`)
- 14 new tests: 7 for `query_annotation_table`, 7 for `get_edit_history`
- Test count: 106 total (all passing)

### Notes
- `query_annotation_table` has no staleness check (table-level query, not single neuron)
- `get_edit_history` has standard staleness gate (raises `StaleRootIdError` for stale root IDs)
- `query_annotation_table` passes `filter_equal_dict` and `filter_in_dict` directly to `client.materialize.query_table()` — no filter reinvention
- `schema_description` is auto-generated from `df.dtypes` since CAVE has no table-level docstrings
- Artifact cache key uses `neuron_id=None` for `query_annotation_table` (table-level query)
- `get_edit_history` maps CAVE changelog's `is_merge` bool to `operation_type` ("merge"/"split")

---

## 2026-03-10 — Session 6

### Added
- `resolve_nucleus_ids` tool (CAVE-specific, minnie65-only): resolves MICrONS nucleus IDs to current pt_root_ids via `nucleus_detection_v0`, with merge conflict and no-segment detection
- `NucleusResolutionResult`, `NucleusResolution`, `NucleusResolutionStatus` schemas
- `format_nucleus_resolution` formatter
- `resolve_nucleus_ids` backend method on `CAVEBackend`
- Optional `nucleus_id` parameter on `get_neuron_info` (minnie65 only): resolves nucleus → pt_root_id internally, warns on merge conflicts, raises on no-segment
- MICrONS nucleus enrichment in `get_connectivity` artifact: `partner_nucleus_id` and `partner_nucleus_conflict` columns added to Parquet for minnie65 queries
- 12 new tests: 7 for `resolve_nucleus_ids`, 4 for `get_neuron_info` nucleus_id, 1 for connectivity nucleus columns
- Test count: 92 total (all passing)

### Notes
- Nucleus IDs are MICrONS-specific — no equivalent stable cell identifier in FlyWire or hemibrain
- `resolve_nucleus_ids` tool rejects non-minnie65 datasets with `DatasetNotSupported`
- Merge conflicts detected by grouping nucleus entries by pt_root_id (real backend queries full table)
- Connectivity nucleus enrichment is a post-query join in the CAVE backend, not the formatter
- `partner_nucleus_id` is null when partner has multiple nuclei (merge conflict) or no nucleus entry
- `partner_nucleus_conflict` is True only for the merge conflict case

---

## 2026-03-10 — Session 5

### Added
- `get_region_connectivity` tool (universal): queries region-to-region synapse counts in long format, saves complete region-pair table as Parquet artifact, returns top 5 connections + total synapse count + `ArtifactManifest`
- `format_region_connectivity` formatter
- CAVE backend `get_region_connectivity`: queries synapse table, joins cell type annotations for regions, groups by (source_region, target_region)
- neuPrint backend `get_region_connectivity`: uses `fetch_roi_connectivity()`, maps to artifact schema
- Abstract method `get_region_connectivity` added to `ConnectomeBackend` base class
- Mock backends with 12 CAVE region pairs and 8 neuPrint ROI pairs
- 10 new tests: normal response (CAVE + neuPrint), artifact correctness, top_5 sorting, n_regions/total_synapses consistency, source/target region filters, unsupported dataset, schema validation
- Test count: 80 total (all passing)

### Notes
- Region filters use case-insensitive substring matching (consistent with `get_neurons_by_type`)
- No staleness check (region-level query, not single neuron)
- Artifact cache key is (tool, dataset, neuron_id=None, mat_version) — filtered queries within the same test get separate artifacts via per-test tmp_path
- Long format artifact can be converted to matrix with `pd.pivot_table(df, index='source_region', columns='target_region', values='n_synapses')`

---

## 2026-03-10 — Session 4

### Added
- `build_neuroglancer_url` standalone MCP tool (universal): wraps existing URL builder as a tool returning `NeuroglancerUrlResponse` with url, n_segments, layers_included, coordinate_space
- `get_neurons_by_type` tool (universal): queries neurons by cell type annotation, saves complete neuron list as Parquet artifact, returns type/region distributions + `ArtifactManifest`
- `format_neuroglancer_url` and `format_neurons_by_type` formatters
- CAVE backend `get_neurons_by_type`: queries cell type table, filters by region
- neuPrint backend `get_neurons_by_type`: uses `fetch_neurons(NC(type=...))` with ROI mapping
- Mock backends with realistic neuron populations (~8 CAVE, ~6 neuPrint neurons)
- 13 new tests: 5 for `build_neuroglancer_url`, 8 for `get_neurons_by_type`
- Test count: 70 total (all passing)

### Notes
- `build_neuroglancer_url` tool function named `build_neuroglancer_url_tool` internally to avoid import collision with the URL builder function
- `get_neurons_by_type` does not perform staleness checks (population query, not single root ID)
- Region filter uses case-insensitive substring matching in CAVE backend
- neuPrint primary region determined by ROI with highest post-synaptic count

---

## 2026-03-10 — Session 3

### Fixed
- **`base.py` abstract signatures**: `get_connectivity` now uses `direction: str = "both"` (was `top_n: int = 10`); `get_neurons_by_type` no longer has `limit` parameter. Both mismatches were from Session 1 pre-artifact-first design.

### Added
- `validate_root_ids` tool (universal): checks root ID currency for CAVE datasets, suggests current replacements for stale IDs; neuPrint body IDs always reported as current (immutable)
- `get_proofreading_status` tool (CAVE-specific): queries dataset-specific proofreading tables, returns axon/dendrite proofread flags, strategy strings, edit count, and last edit timestamp
- `tools/cave_specific.py`: new module for CAVE-specific tools (Tier 2)
- `PROOFREADING_TABLES` dict in `cave_backend.py` mapping dataset → proofreading table name
- 11 new tests: 6 for `validate_root_ids`, 5 for `get_proofreading_status`
- Test count: 57 total (all passing)

### Notes
- `validate_root_ids` does NOT raise `StaleRootIdError` — it IS the resolution tool for stale IDs
- `get_proofreading_status` raises `StaleRootIdError` for stale root IDs (standard staleness gate)
- `get_proofreading_status` on neuPrint raises `DatasetNotSupported` (no proofreading concept)
- Both tools are scalar-only (no artifacts needed)

---

## 2026-03-10 — Session 2

### Changed (breaking)
- **Artifact-first architecture**: all tabular tools now save complete Parquet artifacts to disk and return a lightweight `ArtifactManifest` + 3-item orientation sample in context. No data is truncated or hidden.
- Rewrote `schemas.py`: removed `truncated`, `n_shown`, `limit`, `top_n` fields. Added `ArtifactManifest`, `SynapticPartnerSample`. Renamed `NeuronListResponse` → `NeuronsByTypeResponse`, `RegionMatrixResponse` → `RegionConnectivityResponse`.
- Removed `SynapticPartner`, `NeuronSummary`, `AnnotationRow` models (replaced by artifact pattern).

### Added
- `artifacts/writer.py`: `save_artifact()` saves DataFrames as Parquet with snappy compression, 1-hour cache, returns `ArtifactManifest`
- `get_connectivity` tool (CAVE + neuPrint backends): queries all partners, computes weight distributions (mean/median/max/p90), saves full partner table to Parquet, returns 3 highest-weight samples per direction
- `server.py`: registered `get_connectivity` as MCP tool
- 20 new tests: 6 artifact writer tests, 14 connectivity tests (artifact readability, row count verification, sample sizes, weight distributions, directional filtering, stale ID handling, per-partner Neuroglancer URLs)
- Test count: 46 total (all passing)

### Notes
- Backends return raw DataFrames for connectivity; formatting/artifact-saving happens in `formatters.py`
- Per-partner Neuroglancer URLs highlight the queried neuron + that specific partner
- `direction` parameter supports "upstream", "downstream", or "both" (default)

## 2026-03-10 — Session 1

### Added
- Full project scaffold: `pyproject.toml`, src layout, test directory
- All Pydantic output contract schemas in `output_contracts/schemas.py`
- `exceptions.py`: `DatasetNotSupported`, `StaleRootIdError`, `BackendConnectionError`
- `registry.py`: dataset → backend routing with capability checking (flywire, minnie65, fanc, hemibrain)
- `backends/base.py`: abstract `ConnectomeBackend` with Tier 1 method stubs
- `backends/cave_backend.py`: `get_neuron_info` via CAVEclient (cell type, nucleus, synapse counts, staleness check)
- `backends/neuprint_backend.py`: `get_neuron_info` via neuprint-python (fetch_neurons)
- `output_contracts/formatters.py`: `format_neuron_info()` raw dict → `NeuronInfoResponse`
- `neuroglancer/url_builder.py`: deterministic URL builder with zlib+base64 encoding for all 4 datasets
- `tools/universal.py`: `get_neuron_info` with capability check, backend routing, stale ID error
- `server.py`: FastMCP entry point with `get_neuron_info` registered
- 25 tests: 13 schema tests, 7 neuroglancer tests, 6 universal tool tests (all mocked)

### Notes
- FlyWire datastack set to `flywire_fafb_production` (current production name)
- Backends are lazily initialized on first `get_backend()` call
- No live API calls in tests — all backends mocked via `conftest.py`
