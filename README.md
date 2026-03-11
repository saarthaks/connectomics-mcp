# Connectomics MCP

An MCP (Model Context Protocol) server that gives LLMs structured, context-window-safe access to connectomic datasets (MICrONS, FlyWire, hemibrain).

## Installation

```bash
pip install -e .
```

## Authentication

Set the following environment variables before running the server:

```bash
# For CAVE datasets (MICrONS minnie65, FlyWire, FANC)
export CAVE_CLIENT_TOKEN="your-cave-token"

# For neuPrint datasets (hemibrain)
export NEUPRINT_APPLICATION_CREDENTIALS="your-neuprint-token"
```

- **CAVE token**: Obtain from [https://global.daf-apis.com/auth/api/v1/create_token](https://global.daf-apis.com/auth/api/v1/create_token)
- **neuPrint token**: Obtain from [https://neuprint.janelia.org](https://neuprint.janelia.org) (Account > Auth Token)

## Running

```bash
connectomics-mcp
```

Or run directly:

```bash
python -m connectomics_mcp.server
```

## Supported Datasets

| Dataset | Backend | Capabilities |
|---------|---------|-------------|
| `minnie65` | CAVE | universal, cave |
| `flywire` | CAVE | universal, cave |
| `fanc` | CAVE | universal, cave |
| `hemibrain` | neuPrint | universal, neuprint |
