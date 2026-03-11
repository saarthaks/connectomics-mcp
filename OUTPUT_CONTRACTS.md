# OUTPUT_CONTRACTS.md — Tool Response Design Spec

This document specifies the exact output contract for each tool. These contracts drive `schemas.py`.

## Core Principle: Artifact-First

This MCP is a scientific tool. Data is never truncated, hidden, or summarized in ways that lose information.

**Tools that return tabular data** save a complete, untruncated Parquet file to disk and return an `ArtifactManifest` alongside a lightweight summary. The agent loads the artifact with `pd.read_parquet(artifact_path)` to do analysis. All filtering and thresholding happens in agent code — not in the MCP.

**Tools that return only scalar data** (`get_neuron_info`, `validate_root_ids`, `get_proofreading_status`, `build_neuroglancer_url`) return their full response directly into context — no artifact needed, because the response is inherently small.

**The words `truncated`, `n_shown`, `limit`, and `cap` do not exist in this codebase.**

---

## Shared Types

### `ArtifactManifest`

Included in the response of every tool that writes tabular data.

```python
class ArtifactManifest(BaseModel):
    artifact_path: str              # absolute path to the saved Parquet file
    artifact_format: str = "parquet"
    n_rows: int                     # total rows in the artifact — the complete result
    columns: list[str]              # column names
    schema_description: str         # human-readable description of each column
    dataset: str
    query_timestamp: str            # ISO format
    materialization_version: int | None   # CAVE datasets only
    cache_hit: bool                 # True if artifact was returned from cache
```

### Artifact naming convention (enforced by `artifacts/writer.py`)

```
{dataset}_{tool}_{neuron_id}_{mat_version}_{iso_timestamp}.parquet
# e.g. flywire_connectivity_720575940621039145_v1078_2025-03-10T14:32:00.parquet
```

### Cache policy (enforced by `artifacts/writer.py`)

If an artifact exists for the same (neuron_id, tool, dataset, materialization_version) and is less than 1 hour old, return the cached path without re-querying the API. Set `cache_hit: True` in the manifest.

---

## Universal Tool Contracts

### `get_neuron_info` — scalar only, no artifact

```python
class NeuronInfoResponse(BaseModel):
    neuron_id: int | str            # root_id (CAVE) or bodyId (neuPrint)
    dataset: str
    cell_type: str | None           # e.g. "L2/3 IT", "MBON14"
    cell_class: str | None          # e.g. "excitatory", "inhibitory"
    region: str | None              # primary brain region / ROI
    soma_position_nm: tuple[float, float, float] | None
    n_pre_synapses: int | None      # total output synapse count
    n_post_synapses: int | None     # total input synapse count
    proofread: bool | None          # CAVE datasets only
    materialization_version: int | None
    neuroglancer_url: str           # always included
    warnings: list[str] = []
```

---

### `get_connectivity` — artifact required

The artifact contains every synaptic partner, one row per partner. The context response contains summary statistics and 3 orientation examples — explicitly labeled as non-exhaustive.

```python
class SynapticPartnerSample(BaseModel):
    partner_id: int | str
    partner_type: str | None
    n_synapses: int
    weight_normalized: float | None     # fraction of total input or output

class ConnectivityResponse(BaseModel):
    neuron_id: int | str
    dataset: str

    # Complete counts — always the true total, never a shown count
    n_upstream_total: int
    n_downstream_total: int

    # Weight distribution summaries (computed over the full result set)
    upstream_weight_distribution: dict  # {"mean": 4.1, "median": 2, "max": 94, "p90": 11}
    downstream_weight_distribution: dict

    # Orientation sample — NOT for analysis, labeled explicitly
    # These are the 3 highest-weight partners in each direction only
    upstream_sample: list[SynapticPartnerSample]    # len == 3
    downstream_sample: list[SynapticPartnerSample]  # len == 3
    sample_note: str = (
        "upstream_sample and downstream_sample show the 3 highest-weight partners "
        "for orientation only. Load artifact_manifest.artifact_path for the complete dataset."
    )

    neuroglancer_url: str
    artifact_manifest: ArtifactManifest     # full partner table on disk
    warnings: list[str] = []
```

**Artifact schema** (`artifact_path` Parquet columns):

| Column | Type | Description |
|--------|------|-------------|
| `partner_id` | int64 | root_id or bodyId of partner |
| `direction` | str | "upstream" or "downstream" |
| `partner_type` | str | cell type annotation, if available |
| `partner_class` | str | excitatory / inhibitory, if available |
| `n_synapses` | int32 | synapse count for this connection |
| `weight_normalized` | float32 | fraction of total input or output synapses |
| `partner_region` | str | primary ROI of partner soma |
| `neuroglancer_url` | str | URL highlighting this specific connection |
| `partner_nucleus_id` | int64 \| null | MICrONS only: nucleus ID of partner (null if merge conflict or no nucleus entry) |
| `partner_nucleus_conflict` | bool | MICrONS only: True if partner pt_root_id maps to multiple nucleus IDs (merge error) |

---

### `get_neurons_by_type` — artifact required

```python
class NeuronsByTypeResponse(BaseModel):
    dataset: str
    query_cell_type: str
    query_region: str | None
    n_total: int                    # complete count matching the query
    type_distribution: dict         # subtype breakdown if available, e.g. {"L2/3 IT": 412, ...}
    region_distribution: dict       # {"V1": 301, "LM": 89, ...}
    artifact_manifest: ArtifactManifest
    warnings: list[str] = []
```

**Artifact schema**:

| Column | Type | Description |
|--------|------|-------------|
| `neuron_id` | int64 | root_id or bodyId |
| `cell_type` | str | cell type annotation |
| `cell_class` | str | excitatory / inhibitory |
| `region` | str | primary brain region |
| `n_pre_synapses` | int32 | output synapse count |
| `n_post_synapses` | int32 | input synapse count |
| `proofread` | bool | CAVE only |

---

### `get_region_connectivity` — artifact required

```python
class RegionConnectivityResponse(BaseModel):
    dataset: str
    n_regions: int                  # total number of regions in the result
    top_5_connections: list[dict]   # top 5 region pairs by synapse count, for orientation
    total_synapses: int             # grand total across all region pairs
    artifact_manifest: ArtifactManifest
    warnings: list[str] = []
```

**Artifact schema** (long format, not matrix — matrices can be reconstructed with `pivot_table`):

| Column | Type | Description |
|--------|------|-------------|
| `source_region` | str | presynaptic ROI |
| `target_region` | str | postsynaptic ROI |
| `n_synapses` | int32 | synapse count |
| `n_neurons_pre` | int32 | number of presynaptic neurons |
| `n_neurons_post` | int32 | number of postsynaptic neurons |

---

### `build_neuroglancer_url` — scalar only, no artifact

```python
class NeuroglancerUrlResponse(BaseModel):
    url: str
    dataset: str
    n_segments: int
    layers_included: list[str]      # e.g. ["em", "segmentation", "synapses"]
    coordinate_space: str           # e.g. "nm"
```

---

### `validate_root_ids` — scalar only, no artifact

```python
class RootIdValidationResult(BaseModel):
    root_id: int
    is_current: bool
    last_edit_timestamp: str | None
    suggested_current_id: int | None    # populated if not current

class RootIdValidationResponse(BaseModel):
    dataset: str
    materialization_version: int
    results: list[RootIdValidationResult]
    n_stale: int
    warnings: list[str] = []
```

---

## CAVE-Specific Contracts

### `get_proofreading_status` — scalar only, no artifact

```python
class ProofreadingStatusResponse(BaseModel):
    neuron_id: int
    dataset: str
    axon_proofread: bool | None
    dendrite_proofread: bool | None
    strategy_axon: str | None       # e.g. "axon_fully_extended"
    strategy_dendrite: str | None
    n_edits: int | None
    last_edit_timestamp: str | None
    warnings: list[str] = []
```

---

### `resolve_nucleus_ids` — scalar only, no artifact (minnie65 only)

Resolves MICrONS nucleus IDs to current pt_root_ids via `nucleus_detection_v0`.

```python
class NucleusResolutionStatus(str, Enum):
    RESOLVED = "resolved"           # 1:1 mapping, safe to use
    MERGE_CONFLICT = "merge_conflict"  # this nucleus shares a pt_root_id with others
    NO_SEGMENT = "no_segment"       # nucleus has no associated pt_root_id

class NucleusResolution(BaseModel):
    nucleus_id: int
    pt_root_id: int | None
    resolution_status: NucleusResolutionStatus
    conflicting_nucleus_ids: list[int] = []
    materialization_version: int

class NucleusResolutionResult(BaseModel):
    dataset: str
    materialization_version: int
    resolutions: list[NucleusResolution]
    n_resolved: int
    n_merge_conflicts: int
    n_no_segment: int
    warnings: list[str] = []
```

---

### `query_annotation_table` — artifact required

```python
class AnnotationTableResponse(BaseModel):
    dataset: str
    table_name: str
    n_total: int                    # complete row count matching the query
    schema_description: str         # description of the table's columns and their meaning
    artifact_manifest: ArtifactManifest
    warnings: list[str] = []
```

**Artifact schema**: columns vary by table. The artifact contains the raw query result as returned by CAVEclient, with all columns preserved. `schema_description` in the manifest documents the columns for the specific table queried.

---

### `get_edit_history` — artifact required

```python
class EditHistoryResponse(BaseModel):
    neuron_id: int
    dataset: str
    n_edits_total: int
    first_edit_timestamp: str | None
    last_edit_timestamp: str | None
    artifact_manifest: ArtifactManifest
    warnings: list[str] = []
```

**Artifact schema**:

| Column | Type | Description |
|--------|------|-------------|
| `operation_id` | int64 | edit operation ID |
| `timestamp` | str | ISO timestamp of edit |
| `operation_type` | str | "merge" or "split" |
| `user_id` | str | anonymized editor ID |

---

## neuPrint-Specific Contracts

### `fetch_cypher` — artifact required

```python
class CypherQueryResponse(BaseModel):
    dataset: str
    query: str                      # echo the query for traceability
    n_rows: int                     # complete row count
    columns: list[str]
    artifact_manifest: ArtifactManifest
    warnings: list[str] = []
```

**Artifact schema**: columns are whatever the Cypher query returns. All rows preserved.

---

### `get_synapse_compartments` — scalar only, no artifact

Response is inherently small (one row per compartment type — always ≤ 5 rows).

```python
class CompartmentStats(BaseModel):
    compartment: str                # "axon", "dendrite", "spine", "soma"
    n_synapses: int
    fraction: float                 # fraction of total for this direction

class SynapseCompartmentResponse(BaseModel):
    neuron_id: int | str
    dataset: str
    direction: str                  # "input" or "output"
    compartments: list[CompartmentStats]
    n_total_synapses: int
    warnings: list[str] = []
```

---

## Summary: Artifact vs. Scalar by Tool

| Tool | Returns Artifact | Rationale |
|------|-----------------|-----------|
| `get_neuron_info` | No | Scalar properties only |
| `get_connectivity` | **Yes** | Unbounded partner list |
| `get_neurons_by_type` | **Yes** | Unbounded neuron list |
| `get_region_connectivity` | **Yes** | NxN matrix, N unbounded |
| `build_neuroglancer_url` | No | Single URL string |
| `validate_root_ids` | No | One result per input ID |
| `get_proofreading_status` | No | ~8 scalar fields |
| `resolve_nucleus_ids` | No | One result per input nucleus ID |
| `query_annotation_table` | **Yes** | Unbounded annotation rows |
| `get_edit_history` | **Yes** | Unbounded edit log |
| `fetch_cypher` | **Yes** | Unbounded query result |
| `get_synapse_compartments` | No | ≤5 compartment rows |
