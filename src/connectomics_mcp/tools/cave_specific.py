"""Tier 2: CAVE-specific tools (proofreading, edit history, annotations)."""

from __future__ import annotations

import logging
from typing import Any

from connectomics_mcp.exceptions import StaleRootIdError
from connectomics_mcp.output_contracts.formatters import format_proofreading_status
from connectomics_mcp.registry import DATASETS, check_capability, get_backend

logger = logging.getLogger(__name__)


def get_proofreading_status(neuron_id: int, dataset: str) -> dict[str, Any]:
    """Get proofreading status for a CAVE neuron.

    Returns axon/dendrite proofreading flags, proofreading strategy,
    edit count, and last edit timestamp.

    Parameters
    ----------
    neuron_id : int
        Root ID of the neuron.
    dataset : str
        Dataset to query. Must be a CAVE dataset: "minnie65",
        "flywire", or "fanc".

    Returns
    -------
    dict
        ProofreadingStatusResponse as a dict with proofreading flags,
        strategy strings, edit count, and last edit timestamp.

    Raises
    ------
    DatasetNotSupported
        If the dataset is unknown or not a CAVE dataset.
    StaleRootIdError
        If the root ID is outdated.
    """
    check_capability(dataset, "cave")

    backend = get_backend(dataset)
    raw = backend.get_proofreading_status(neuron_id)

    # Raise if the root ID is stale
    if not raw.get("is_current", True):
        raise StaleRootIdError(int(neuron_id))

    response = format_proofreading_status(raw, dataset)
    return response.model_dump()
