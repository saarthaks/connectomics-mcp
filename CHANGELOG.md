# Changelog

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
