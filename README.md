# Connectomics MCP

An MCP (Model Context Protocol) server that gives LLMs structured, context-window-safe access to connectomic datasets. Wraps [CAVE](https://caveclient.readthedocs.io/) (MICrONS, FlyWire) and [neuPrint](https://neuprint.janelia.org/) (hemibrain) behind a unified semantic API.

## How It Works

Tools return **lightweight summaries** into the LLM context window (counts, distributions, 3 orientation examples) while saving **complete, untruncated results** as Parquet artifacts on disk. The agent reads the summary to understand what was fetched, then writes code to analyze the full artifact. No data is hidden. No context window is flooded.

```
Agent calls get_connectivity(neuron_id, dataset)
  -> MCP queries all synaptic partners from the backend
  -> Saves complete partner table to ~/.connectomics_mcp/artifacts/minnie65_connectivity_864..._943_2026-03-10T....parquet
  -> Returns to context: {n_upstream: 142, n_downstream: 87, top 3 partners, artifact_path, neuroglancer_url}

Agent then: df = pd.read_parquet(artifact_path)  # full analysis on complete data
```

## Quick Start

```bash
pip install -e .
```

### Authentication

```bash
# CAVE datasets (MICrONS minnie65, FlyWire)
export CAVE_CLIENT_TOKEN="your-cave-token"

# neuPrint datasets (hemibrain)
export NEUPRINT_APPLICATION_CREDENTIALS="your-neuprint-token"
```

- **CAVE token**: Obtain from [https://global.daf-apis.com/auth/api/v1/create_token](https://global.daf-apis.com/auth/api/v1/create_token)
- **neuPrint token**: Obtain from [https://neuprint.janelia.org](https://neuprint.janelia.org) (Account > Auth Token)

### Run the Server

```bash
connectomics-mcp
```

Or directly:

```bash
python -m connectomics_mcp.server
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `CAVE_CLIENT_TOKEN` | *(required for CAVE)* | CAVE API authentication token |
| `NEUPRINT_APPLICATION_CREDENTIALS` | *(required for neuPrint)* | neuPrint API token |
| `CONNECTOMICS_MCP_ARTIFACT_DIR` | `~/.connectomics_mcp/artifacts/` | Directory for Parquet artifact output |

## Supported Datasets

| Dataset | Backend | Datastack / Server | Capabilities |
|---------|---------|-------------------|-------------|
| `minnie65` | CAVE | `minnie65_public` | universal, cave |
| `flywire` | CAVE | `flywire_fafb_production` | universal, cave |
| `hemibrain` | neuPrint | `neuprint.janelia.org` | universal, neuprint |

## Tools

### Tier 1: Universal (all datasets)

| Tool | Description | Artifact? |
|------|-------------|-----------|
| `get_neuron_info` | Cell type, soma position, synapse counts, Neuroglancer URL | No |
| `get_connectivity` | All synaptic partners with weight distributions | Yes |
| `get_neurons_by_type` | All neurons matching a cell type annotation | Yes |
| `get_region_connectivity` | Region-to-region synapse counts (long format) | Yes |
| `build_neuroglancer_url` | Construct a Neuroglancer visualization URL | No |
| `validate_root_ids` | Check root ID currency, suggest replacements for stale IDs | No |

### Tier 2: CAVE-Specific (minnie65, flywire)

| Tool | Description | Artifact? |
|------|-------------|-----------|
| `get_proofreading_status` | Axon/dendrite proofread flags, edit count | No |
| `resolve_nucleus_ids` | Resolve MICrONS nucleus IDs to current root IDs (minnie65 only) | No |
| `query_annotation_table` | Query arbitrary CAVE annotation tables | Yes |
| `get_edit_history` | Neuron edit changelog (merge/split operations) | Yes |

### Tier 3: neuPrint-Specific (hemibrain)

| Tool | Description | Artifact? |
|------|-------------|-----------|
| `fetch_cypher` | Execute arbitrary Cypher queries | Yes |
| `get_synapse_compartments` | Per-ROI synapse distribution | No |

## Artifact-First Pattern

Tools that return tabular data follow the **artifact-first pattern**:

1. The complete query result is saved as a Parquet file (snappy compression) to `CONNECTOMICS_MCP_ARTIFACT_DIR`
2. The context-window response contains only: counts, distributions, a 3-item orientation sample, the artifact path, and a Neuroglancer URL
3. The orientation sample is explicitly labeled as non-exhaustive — it must never be used for analysis
4. The agent loads the artifact with `pd.read_parquet(artifact_path)` for full analysis

Artifacts are cached for 1 hour. If an identical query (same tool, dataset, neuron ID, materialization version) has a cached artifact less than 1 hour old, the cached path is returned without re-querying.

**Artifact naming**: `{dataset}_{tool}_{neuron_id}_{materialization_version}_{iso_timestamp}.parquet`

## Example Workflow

```python
# 1. Get basic neuron info
info = get_neuron_info(720575940621039145, "minnie65")
# Returns: cell_type="L2/3 IT", n_pre=1500, n_post=3200, neuroglancer_url=...

# 2. Get all connectivity partners (artifact-producing)
conn = get_connectivity(720575940621039145, "minnie65")
# Returns: n_upstream=142, n_downstream=87, artifact_path=..., top 3 samples

# 3. Load and analyze the full artifact
import pandas as pd
df = pd.read_parquet(conn["artifact_manifest"]["artifact_path"])
# df has ALL 229 partners with columns: partner_id, direction, n_synapses, ...

# 4. Filter and analyze in code
top_inputs = df[df["direction"] == "upstream"].nlargest(10, "n_synapses")
```

## Root ID Staleness (CAVE Datasets)

CAVE root IDs are invalidated by proofreading edits. All tools that accept a root ID for a CAVE dataset check currency first. If the ID is stale, a `StaleRootIdError` is raised with a message directing the user to `validate_root_ids()` for current replacements.

Use `validate_root_ids([root_id], dataset)` to check currency and get suggested replacements before querying.

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests (all mocked, no live API calls needed)
pytest tests/ -v
```

Tests use mock backends defined in `tests/conftest.py`. No authentication tokens or network access required.

## Architecture

See `CLAUDE.md` for full architectural details. Key files:

```
src/connectomics_mcp/
├── server.py              # FastMCP entry point (12 tools registered)
├── tools/                 # Tool functions (universal, cave_specific, neuprint_specific)
├── backends/              # API adapters (CAVE, neuPrint)
├── output_contracts/      # Pydantic schemas + formatters
├── artifacts/             # Parquet save/cache logic
├── neuroglancer/          # URL builder
└── registry.py            # Dataset -> backend routing
```

See `TOOL_REGISTRY.md` for tool implementation status and `OUTPUT_CONTRACTS.md` for response schema specifications.
