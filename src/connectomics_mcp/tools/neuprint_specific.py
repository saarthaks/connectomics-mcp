"""Tier 3: neuPrint-specific tools (Cypher queries, synapse compartments)."""

from __future__ import annotations

import logging
from typing import Any

from connectomics_mcp.output_contracts.formatters import (
    format_cypher_query,
    format_synapse_compartments,
)
from connectomics_mcp.registry import check_capability, get_backend

logger = logging.getLogger(__name__)


def fetch_cypher(query: str, dataset: str) -> dict[str, Any]:
    """Execute a Cypher query against a neuPrint dataset.

    Returns a summary with row count and column names. The complete
    query result is saved as a Parquet artifact — load it with
    ``pd.read_parquet(artifact_path)`` for full analysis.

    Parameters
    ----------
    query : str
        Cypher query string.
    dataset : str
        Dataset to query. Must be a neuPrint dataset (e.g. "hemibrain").

    Returns
    -------
    dict
        CypherQueryResponse as a dict with artifact_manifest pointing
        to the full query result on disk.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or not a neuPrint dataset.
    """
    check_capability(dataset, "neuprint")

    backend = get_backend(dataset)
    raw = backend.fetch_cypher(query)

    response = format_cypher_query(raw, dataset)
    return response.model_dump()


def get_synapse_compartments(
    neuron_id: int | str, dataset: str, direction: str = "input"
) -> dict[str, Any]:
    """Get synapse distribution across ROI compartments for a neuron.

    Returns per-ROI synapse counts and fractions for the specified
    direction. Response is inherently small (one entry per ROI).

    Parameters
    ----------
    neuron_id : int | str
        Body ID of the neuron.
    dataset : str
        Dataset to query. Must be a neuPrint dataset (e.g. "hemibrain").
    direction : str
        "input" for post-synaptic or "output" for pre-synaptic
        (default "input").

    Returns
    -------
    dict
        SynapseCompartmentResponse as a dict with per-ROI compartment
        stats and total synapse count.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or not a neuPrint dataset.
    """
    check_capability(dataset, "neuprint")

    backend = get_backend(dataset)
    raw = backend.get_synapse_compartments(neuron_id, direction=direction)

    response = format_synapse_compartments(raw, dataset)
    return response.model_dump()
