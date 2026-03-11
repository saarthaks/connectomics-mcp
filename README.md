# Connectomics MCP

A [Model Context Protocol](https://modelcontextprotocol.io/) server that gives LLMs structured access to connectomic datasets. It wraps [CAVE](https://caveclient.readthedocs.io/) (MICrONS minnie65, FlyWire) and [neuPrint](https://neuprint.janelia.org/) (hemibrain) behind a unified API, returning lightweight summaries into the context window while saving complete, untruncated query results as Parquet artifacts on disk. The agent reads the summary to understand what was fetched, then writes code to analyze the full artifact. No data is hidden. No context window is flooded.

## Prerequisites

- **Python 3.11+**
- **CAVE token** for MICrONS and FlyWire datasets -- obtain from <https://global.daf-apis.com/auth/api/v1/create_token>
- **neuPrint token** for hemibrain -- obtain from <https://neuprint.janelia.org> (Account > Auth Token)

## Quickstart

### Install

```bash
git clone https://github.com/saarthaks/connectomics-mcp.git
cd connectomics-mcp
pip install -e .
```

### Configure credentials

Create a `.env` file or export directly:

```bash
export CAVE_CLIENT_TOKEN="your-cave-token"
export NEUPRINT_APPLICATION_CREDENTIALS="your-neuprint-token"
```

Optionally set a custom artifact output directory (default: `~/.connectomics_mcp/artifacts/`):

```bash
export CONNECTOMICS_MCP_ARTIFACT_DIR="/path/to/artifacts"
```

### Register with Claude Code

Add to your Claude Code MCP config (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "connectomics": {
      "command": "connectomics-mcp"
    }
  }
}
```

Or run the server directly:

```bash
connectomics-mcp
```

## Example prompts

**1. Identify a neuron and explore its connectivity**

> "What cell type is MICrONS neuron 864691135571546917? Show me its top input partners."

The agent calls `get_neuron_info` to retrieve cell type, synapse counts, and a Neuroglancer link, then calls `get_connectivity` to save the full partner table as a Parquet artifact. It loads the artifact and filters for the strongest upstream partners.

**2. Compare neurotransmitter usage across a cell type in FlyWire**

> "Find all DA1_lPN neurons in FlyWire and summarize their neurotransmitter predictions."

The agent calls `get_neurons_by_type("DA1_lPN", "flywire")` to get every matching neuron, then iterates through them with `get_neuron_info` to collect NT predictions from output synapse profiles. FlyWire responses include hierarchical classification and per-neuron neurotransmitter type.

**3. Run a custom Cypher query on hemibrain**

> "How many Kenyon cells in hemibrain have more than 500 postsynaptic sites? Save the results."

The agent calls `fetch_cypher` with a Cypher query like `MATCH (n:Neuron) WHERE n.type =~ 'KC.*' AND n.post > 500 RETURN n.bodyId, n.type, n.post`. The complete result is saved as a Parquet artifact; the context window receives the row count and column names.

## Tools reference

### Tier 1: Universal (minnie65, flywire, hemibrain)

| Tool | Description | Returns |
|------|-------------|---------|
| `get_neuron_info` | Cell type, soma position, synapse counts, Neuroglancer URL | Scalar |
| `get_connectivity` | All synaptic partners with weight distributions | Artifact |
| `get_neurons_by_type` | All neurons matching a cell type annotation | Artifact |
| `get_region_connectivity` | Region-to-region synapse counts in long format | Artifact |
| `build_neuroglancer_url` | Construct a Neuroglancer visualization URL | Scalar |
| `validate_root_ids` | Check root ID currency, suggest replacements for stale IDs | Scalar |

### Tier 2: CAVE-specific (minnie65, flywire)

| Tool | Description | Returns |
|------|-------------|---------|
| `get_proofreading_status` | Axon/dendrite proofread flags, edit count | Scalar |
| `resolve_nucleus_ids` | Resolve MICrONS nucleus IDs to current root IDs (minnie65 only) | Scalar |
| `query_annotation_table` | Query arbitrary CAVE annotation tables | Artifact |
| `get_edit_history` | Neuron edit changelog (merge/split operations) | Artifact |

### Tier 3: neuPrint-specific (hemibrain)

| Tool | Description | Returns |
|------|-------------|---------|
| `fetch_cypher` | Execute arbitrary Cypher queries | Artifact |
| `get_synapse_compartments` | Per-ROI synapse distribution | Scalar |

**Artifact** tools save the complete query result as a Parquet file and return a lightweight manifest (counts, distributions, 3 orientation examples, file path, Neuroglancer URL) into the context window. **Scalar** tools return their full response directly -- no artifact needed.

## How artifacts work

```
Agent calls get_connectivity(neuron_id, dataset)
  -> MCP queries all synaptic partners from the backend API
  -> Saves complete table to ~/.connectomics_mcp/artifacts/{dataset}_connectivity_{id}_{version}_{timestamp}.parquet
  -> Returns to context: {n_upstream: 142, n_downstream: 87, top 3 samples, artifact_path, neuroglancer_url}

Agent then: df = pd.read_parquet(artifact_path)  # full analysis on complete data
```

Artifacts are cached for 1 hour. Identical queries (same tool, dataset, neuron ID, materialization version) reuse the cached file.

## Dataset context

| Dataset | Organism | Region | Neurons | Source |
|---------|----------|--------|---------|--------|
| `minnie65` | Mouse | Visual cortex (V1/LM) | ~75,000 proofread | [MICrONS Explorer](https://www.microns-explorer.org/) |
| `flywire` | *Drosophila* | Whole brain | ~130,000 proofread | [FlyWire Codex](https://codex.flywire.ai/) |
| `hemibrain` | *Drosophila* | Half brain | ~25,000 traced | [neuPrint](https://neuprint.janelia.org/) |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v                                          # unit tests (mocked, no credentials needed)
pytest tests/integration/ --integration -v -s             # live API tests (requires tokens)
```

## Citations

If you use this tool in your research, please cite the underlying datasets:

**MICrONS (minnie65)**
> MICrONS Consortium et al. "Functional connectomics spanning multiple areas of mouse visual cortex." *Nature* (2025). <https://doi.org/10.1038/s41586-025-08790-w>

**FlyWire**
> Dorkenwald, S. et al. "Neuronal wiring diagram of an adult brain." *Nature* 634, 124--138 (2024). <https://doi.org/10.1038/s41586-024-07558-y>

**hemibrain**
> Scheffer, L.K. et al. "A connectome and analysis of the adult *Drosophila* central brain." *eLife* 9:e57443 (2020). <https://doi.org/10.7554/eLife.57443>

## License

MIT
